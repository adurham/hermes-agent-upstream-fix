"""An attempt the OWNER interrupted was never read: it must cost the card nothing.

Measured 2026-11-24 on the ps5 lane's board: 13 of 241 runs in one 4 h window were booked
``crashed`` with ``exit code 130 [owner admission settled interrupted]``, the worker's whole
output the session id. Their owner (the gateway) had restarted: boot recovery fenced the in-flight
kanban admissions ``unknown`` and settled the unattended ones ``interrupted``
(``settle_recovered_unattended``). Nothing in the card ran, yet every one of them charged
``consecutive_failures`` and reached the lane's supervisor worded "crashed (worker exited)".

These tests pin the corrected disposition:
  * the run is booked ``interrupted`` (outcome ``reclaimed``) and NOT one attempt is charged;
  * the run's error names the owner's interrupt, so the lane can tell "never started" from
    "crashed";
  * an owner-recorded ``worker_lost`` still makes it the worker's own crash (A3 stands).
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import kanban_db_dispatch as kbd
from tests.hermes_cli.test_kanban_managed_worker_loss import board  # the isolated board fixture


def _interrupted_130(board, tid="t_owner_stop", **kw):
    """A lost managed worker whose own owner settled its admission interrupted, exit 130."""
    run_id = board.running(tid, **kw)
    board.event(tid, run_id, "worker_result",
                {"pid": board.state(tid)["worker_pid"], "claim_lock": board.state(tid)["claim_lock"],
                 "exit_code": 130, "last_output": ""}, board.state(tid)["started_at"] + 60)
    board.conn.commit()
    return run_id


def test_an_owner_interrupt_charges_no_attempt_and_is_not_a_crash(board):
    run_id = _interrupted_130(board)
    result = board.tick(failure_limit=2)
    row, run = board.state("t_owner_stop"), board.run(run_id)

    # Disposed of, back on the queue, nothing held.
    assert row["status"] == "ready"
    assert (row["claim_lock"], row["worker_pid"], row["current_run_id"]) == (None, None, None)

    # The retry accounting: the card's attempt count does not move.
    assert row["consecutive_failures"] == 0

    # The booking says "the owner stopped it", not "it crashed".
    assert run["outcome"] == "reclaimed"
    assert run["outcome"] != "crashed"
    assert "owner interrupted this attempt at admission" in run["error"]
    assert "not a card failure" in run["error"]
    assert row["last_failure_error"].startswith("pid ")
    assert "owner interrupted this attempt at admission" in row["last_failure_error"]

    meta = json.loads(run["metadata"])
    assert meta["owner_interrupted"] is True and meta["attempt_charged"] is False
    assert meta["reason"] == "owner admission settled interrupted"

    kinds = [r[0] for r in board.conn.execute(
        "SELECT kind FROM task_events WHERE task_id='t_owner_stop' AND run_id=? ORDER BY id", (run_id,))]
    assert "interrupted" in kinds and "crashed" not in kinds
    assert "gave_up" not in kinds

    # The dispatcher's own accounting agrees: requeued, not crashed.
    assert "t_owner_stop" in result.owner_interrupted
    assert "t_owner_stop" not in result.crashed
    assert result.auto_blocked == []


def test_owner_interrupts_never_exhaust_the_retry_budget(board):
    """Three in a row is still not a card defect — the breaker must stay shut."""
    for n in range(3):
        run_id = _interrupted_130(board, tid=f"t_owner_stop_{n}")
        board.tick(failure_limit=2)
        row = board.state(f"t_owner_stop_{n}")
        assert row["status"] == "ready", n
        assert row["consecutive_failures"] == 0, n
        assert board.run(run_id)["outcome"] == "reclaimed"


def test_an_owner_interrupt_is_still_bounded_by_a_lapsed_claim(board):
    """The card must never sit ``running`` after its claim lapsed, interrupt or not."""
    _interrupted_130(board, tid="t_lapsed_owner", claim_expires_in=-1800, heartbeat_age=None)
    board.tick(failure_limit=2)
    assert board.state("t_lapsed_owner")["status"] == "ready"


def test_a_worker_loss_the_owner_recorded_is_still_the_workers_own_crash(board):
    """A3 stands: the owner saw THIS worker die (cgroup evidence), so the attempt is charged.

    The same ``terminal/interrupted`` admission is present — ``_settle_lost`` settles a lost
    managed execution that way — so the discriminator is the owner's ``worker_lost`` record,
    never the settlement alone.
    """
    run_id = _interrupted_130(board, tid="t_own_loss", worker_lost="managed_worker_failed",
                             loss_extra={"memory": {"events": {"oom_kill": 1}}})
    board.tick(failure_limit=2)
    row, run = board.state("t_own_loss"), board.run(run_id)
    assert row["status"] == "ready"
    assert row["consecutive_failures"] == 1
    assert run["outcome"] == "crashed"
    assert "managed worker lost: managed_worker_failed" in run["error"]
    assert "owner interrupted this attempt" not in run["error"]


def test_five_interrupted_starts_in_one_tick_all_land_not_crashed(board):
    """The card's acceptance: five dispatched workers, five interrupted admissions, zero charges.

    Five ``ready`` cards are dispatched in one sweep and each one's owner settles its admission
    ``interrupted`` before the card is read. Every run must book ``reclaimed``/``interrupted``
    — never ``crashed`` — and every card's retry budget must come out untouched, so the lane
    can never read "the owner restarted" as "the card gave up".
    """
    ids = [f"t_five_{n}" for n in range(5)]
    runs = {tid: _interrupted_130(board, tid=tid) for tid in ids}
    result = board.tick(failure_limit=2)

    for tid in ids:
        row, run = board.state(tid), board.run(runs[tid])
        assert row["status"] == "ready", tid
        assert row["consecutive_failures"] == 0, tid
        assert run["outcome"] == "reclaimed", tid
        assert run["outcome"] != "crashed", tid
        assert "owner interrupted this attempt at admission" in run["error"], tid
        kinds = [r[0] for r in board.conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? AND run_id=? ORDER BY id", (tid, runs[tid]))]
        assert "crashed" not in kinds and "gave_up" not in kinds, tid
    assert sorted(result.owner_interrupted) == sorted(ids)
    assert result.crashed == [] and result.auto_blocked == []


@pytest.mark.parametrize("admission", [("terminal", "completed"), ("started", None), None])
def test_only_a_terminal_interrupted_settlement_is_an_owner_interrupt(board, admission):
    """Quota, an in-flight admission and no receipt are not the owner saying "I stopped it"."""
    run_id = _interrupted_130(board, tid="t_not_interrupt", admission=admission)
    board.tick(failure_limit=5)
    run = board.run(run_id)
    assert run["outcome"] == "crashed"
    assert board.state("t_not_interrupt")["consecutive_failures"] == 1
