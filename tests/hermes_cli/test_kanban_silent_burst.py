"""A SILENT, cross-lane crash burst is the HOST's, not a shared card bug (kanban t_43970201).

Measured 2026-11-25 on the ps5 board: 22 blocked cards, each `gave_up` sticky at
``effective_limit: 1`` because eleven workers died inside one 40 s window sharing one
``managed_worker_failed`` fingerprint. The host was in a memory storm (PSI 20-32%, 5.3 GiB
available, 12 workers exiting 1), the workers printed nothing at all (107-byte logs,
``worker_result.last_output == ""``), and two cards from that wave — one on ps5-pm, one on
nest-pm — each completed on their first allowed run once unblocked by hand.

The rule these tests pin:

  * silent + >= 3 identical fingerprints in one tick + >= 2 distinct assignees -> infrastructure
    (strike-free, bounded per card by ``INFRA_STRIKE_FREE_LIMIT``);
  * ONE assignee keeps today's behaviour (that lane's own model/config is equally likely);
  * a crash that quotes real worker output keeps today's behaviour (a shared textual bug);
  * the bound holds: a card in a REPEATING silent burst is charged after the free retries and
    still blocks at ``failure_limit``;
  * the residue an older classifier left behind is re-openable, at <= 3/tick, with the burst
    recomputed from ``task_events`` and recorded on the ``infrastructure_recovered`` event.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

#: The measured shape: our head, the managed-loss record, and no worker output at all.
SILENT = (
    "pid 506084 exited with code 1 [managed worker lost: managed_worker_failed "
    "[oom_kill=0; cgroup peak 1.10 GiB of 6.00 GiB cap]]"
)
#: Silence, but the CLI's session banner survived into the captured output.
SILENT_BANNER = SILENT + kbd.WORKER_OUTPUT_SEP + "'Session: local-580e26a8'"
#: The card's own words: a shared textual bug, not the host.
LOUD = SILENT + kbd.WORKER_OUTPUT_SEP + (
    "'Traceback (most recent call last): ImportError: no module named frobnicate'"
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _failures(conn, tid):
    return int(conn.execute("SELECT consecutive_failures FROM tasks WHERE id = ?", (tid,)).fetchone()[0])


def _status(conn, tid):
    return conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()[0]


def _events(conn, tid, kind):
    return conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id", (tid, kind)
    ).fetchall()


def _payloads(conn, tid, kind):
    return [kb._json_dict(row[0]) for row in _events(conn, tid, kind)]


def _dead(text):
    return kbd._DeadWorker(
        "nonzero_exit", 1, text, "crashed", {"pid": 999999, "claimer": "test"},
    )


def _crash(conn, tid, text):
    """The reclaim sweep's shape: stamp the verdict, write the `crashed` event, then charge it."""
    dead = _dead(text)
    kbd._stamp_infrastructure(dead)
    run_id = kb._current_run_id(conn, tid)
    with kb.write_txn(conn):
        kb._append_event(conn, tid, dead.event_kind, dead.event_payload, run_id=run_id)
        if run_id is not None:
            conn.execute("UPDATE task_runs SET error = ? WHERE id = ?", (text, run_id))
    return (tid, 999999, "test", dead)


def _running(conn, title, assignee):
    tid = kb.create_task(conn, title=title, assignee=assignee)
    assert kb.claim_task(conn, tid) is not None
    return tid


# ---------------------------------------------------------------------------
# 1. Silent + >= 3 identical fingerprints + >= 2 assignees -> infrastructure
# ---------------------------------------------------------------------------

def test_a_silent_cross_lane_burst_books_infrastructure(conn):
    tids = [_running(conn, f"silent {n}", who)
            for n, who in enumerate(("ps5-pm", "nest-pm", "ps5-pm"))]

    kbd._account_crashes(conn, [_crash(conn, tid, SILENT) for tid in tids])

    for tid in tids:
        assert _failures(conn, tid) == 0, "a host burst must not charge any card a strike"
        assert _status(conn, tid) != "blocked", "the systemic branch must stay unreachable"
        assert not _events(conn, tid, "gave_up")
        crashed = _payloads(conn, tid, "crashed")
        assert crashed[-1].get("infrastructure") is True
        assert crashed[-1].get("silent_burst") is True
        assert crashed[-1].get("burst_assignees") == ["nest-pm", "ps5-pm"]
        assert kb._has_sticky_block(conn, tid) is False


def test_the_session_banner_alone_is_still_silence(conn):
    """`Session: <id>` names the session, not the work: two lanes of that are the host's."""
    assert kbd._silent_crash(SILENT_BANNER) is True
    assert kbd._silent_crash(SILENT) is True
    assert kbd._silent_crash(LOUD) is False

    tids = [_running(conn, f"banner {n}", who) for n, who in enumerate(("ps5-pm", "nest-pm", "ps5-pm"))]
    kbd._account_crashes(conn, [_crash(conn, tid, SILENT_BANNER) for tid in tids])

    for tid in tids:
        assert _failures(conn, tid) == 0
        assert _payloads(conn, tid, "crashed")[-1].get("infrastructure") is True


def test_several_banners_are_still_silence():
    """A worker that re-ran in its own log leaves one banner per attempt — still no output.

    Measured on the live 12:26:13 wave: `Worker's last output: 'Session: … Session: … Session: …'`
    (``_worker_final_output`` joins the log's tail with spaces).
    """
    three = SILENT + kbd.WORKER_OUTPUT_SEP + (
        "'Session: local-aaaa Session: local-bbbb Session: local-cccc'"
    )
    assert kbd._silent_crash(three) is True
    # A banner next to real output is NOT silence.
    assert kbd._silent_crash(
        SILENT + kbd.WORKER_OUTPUT_SEP + "'Session: local-aaaa Traceback (most recent call last):'"
    ) is False


# ---------------------------------------------------------------------------
# 2. ONE assignee keeps today's behaviour (that lane's own fault is equally likely)
# ---------------------------------------------------------------------------

def test_a_single_lane_silent_burst_stays_sticky(conn):
    tids = [_running(conn, f"one lane {n}", "ps5-pm") for n in range(3)]

    kbd._account_crashes(conn, [_crash(conn, tid, SILENT) for tid in tids])

    for tid in tids:
        assert _failures(conn, tid) == 1
        assert _status(conn, tid) == "blocked"
        gave_up = _payloads(conn, tid, "gave_up")
        assert gave_up and gave_up[-1].get("sticky") is True
        assert gave_up[-1].get("effective_limit") == 1
        assert kb._has_sticky_block(conn, tid) is True


# ---------------------------------------------------------------------------
# 3. A crash that quotes real worker output is a card-level symptom
# ---------------------------------------------------------------------------

def test_a_loud_cross_lane_burst_stays_sticky(conn):
    tids = [_running(conn, f"loud {n}", who)
            for n, who in enumerate(("ps5-pm", "nest-pm", "ps5-pm"))]

    kbd._account_crashes(conn, [_crash(conn, tid, LOUD) for tid in tids])

    for tid in tids:
        assert _failures(conn, tid) == 1, "shared text is evidence about the code, not the host"
        assert _status(conn, tid) == "blocked"
        assert _payloads(conn, tid, "gave_up")[-1].get("sticky") is True


# ---------------------------------------------------------------------------
# 4. The bound: a repeating silent burst still blocks a card at failure_limit
# ---------------------------------------------------------------------------

def test_a_repeating_silent_burst_is_charged_after_the_free_retries(conn):
    """A card cannot ride a host-shaped rule forever: 3 free, then charged like its own."""
    subject = _running(conn, "retry sink", "ps5-pm")

    def burst_tick():
        others = [_running(conn, f"fresh {time.time_ns()} {n}", "nest-pm") for n in range(2)]
        kbd._account_crashes(conn, [_crash(conn, tid, SILENT) for tid in [subject, *others]])

    for _ in range(kbd.INFRA_STRIKE_FREE_LIMIT):
        burst_tick()
        assert _failures(conn, subject) == 0, "the first runs are the host's"
        assert _status(conn, subject) != "blocked"

    burst_tick()
    assert _failures(conn, subject) == 1, "past the cap the failure counts like the card's own"
    assert _status(conn, subject) != "blocked", "one charge is below failure_limit"

    # Consecutive-streak semantics: the window restarts, so the same shape repeats until the
    # second charge lands. Bounded, and it ends at the breaker instead of looping.
    for _ in range(kbd.INFRA_STRIKE_FREE_LIMIT + 1):
        if _status(conn, subject) == "blocked":
            break
        burst_tick()
    assert _status(conn, subject) == "blocked"
    assert _failures(conn, subject) == kbd.DEFAULT_FAILURE_LIMIT
    assert _payloads(conn, subject, "gave_up")
    # Bounded: two charged failures over the whole run, not one per burst.
    assert len([p for p in _payloads(conn, subject, "crashed") if not p.get("infrastructure")]) == 2


# ---------------------------------------------------------------------------
# 5. The residue: sticky give-ups an older classifier wrote, recomputed from task_events
# ---------------------------------------------------------------------------

BURST_AT = 1_795_631_865  # the 12:37:45 wave


def _crashed_card(conn, assignee, error, at, *, title="residue"):
    tid = kb.create_task(conn, title=f"{title} {at} {time.time_ns()}", assignee=assignee)
    run_id = int(conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, error) "
        "VALUES (?, ?, 'crashed', ?, ?, 'crashed', ?)",
        (tid, assignee, at - 60, at, error),
    ).lastrowid)
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?, ?, ?, ?, ?)",
            (tid, run_id, "crashed", json.dumps({"pid": 1, "exit_code": 1}), at),
        )
    return tid


def _sticky_gave_up(conn, tid, *, at=None, error=SILENT):
    at = int(time.time()) if at is None else at
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'blocked', consecutive_failures = 1, last_failure_error = ? "
            "WHERE id = ?", (error, tid),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?, NULL, ?, ?, ?)",
            (tid, "gave_up", json.dumps({
                "failures": 1, "effective_limit": 1, "limit_source": "dispatcher",
                "sticky": True, "trigger_outcome": "crashed", "error": error,
            }), at),
        )
    return tid


def _residue_burst(conn, lanes=("ps5-pm", "nest-pm", "ps5-pm"), error=SILENT, at=BURST_AT):
    ids = []
    for n, who in enumerate(lanes):
        tid = _crashed_card(conn, who, error, at + n)
        ids.append(_sticky_gave_up(conn, tid))
    return ids


def test_the_sticky_residue_is_listed_by_a_dry_run_without_touching_it(conn):
    ids = _residue_burst(conn)

    # The min-age gate is shared with the live path (a card that gave up seconds ago may still be
    # in a broken host); a preview of the 3-hour-old residue reads it from the same clock.
    listed = kbd.reclassify_silent_burst_gave_ups(
        conn, dry_run=True, now=int(time.time()) + kbd.INFRA_REOPEN_MIN_AGE_S + 1,
    )

    assert sorted(listed) == sorted(ids)
    for tid in ids:
        assert _status(conn, tid) == "blocked", "a dry run writes nothing"
        assert not _events(conn, tid, "infrastructure_recovered")


def test_the_sticky_residue_reopens_with_the_burst_on_the_event(conn):
    _residue_burst(conn)

    reopened = kbd.reclassify_silent_burst_gave_ups(
        conn, now=int(time.time()) + kbd.INFRA_REOPEN_MIN_AGE_S + 1,
    )

    assert len(reopened) == kbd.INFRA_REOPEN_PER_TICK, "<= 3 per tick, never a mass unblock"
    for tid in reopened:
        assert _status(conn, tid) == "ready"
        assert _failures(conn, tid) == 0
        recovered = _payloads(conn, tid, "infrastructure_recovered")
        assert recovered
        assert recovered[-1]["reclassified"] is True
        assert recovered[-1]["burst_at"] == BURST_AT + 2
        assert recovered[-1]["burst_cards"] == 3
        assert recovered[-1]["burst_assignees"] == ["nest-pm", "ps5-pm"]


def test_a_single_lane_sticky_give_up_is_not_reclassified(conn):
    ids = _residue_burst(conn, lanes=("ps5-pm", "ps5-pm", "ps5-pm"))

    assert kbd.reclassify_silent_burst_gave_ups(
        conn, now=int(time.time()) + kbd.INFRA_REOPEN_MIN_AGE_S + 1,
    ) == []
    for tid in ids:
        assert _status(conn, tid) == "blocked"


def test_lanes_are_counted_over_the_whole_burst_not_only_its_silent_cards(conn):
    """The live 12:26:13 shape: silent ps5-pm cards, loud nest-pm partners in the same wave.

    `_account_crashes` counts the fingerprint's assignees over EVERY crash in the tick, so the
    recompute must read the chain the same way — the loud partner names the lane the host reached.
    """
    ids = []
    for n in range(3):
        tid = _crashed_card(conn, "ps5-pm", SILENT, BURST_AT + n)
        ids.append(_sticky_gave_up(conn, tid))
    _crashed_card(conn, "nest-pm", LOUD, BURST_AT + 1)      # same fingerprint, second lane

    reopened = kbd.reclassify_silent_burst_gave_ups(
        conn, now=int(time.time()) + kbd.INFRA_REOPEN_MIN_AGE_S + 1,
    )

    assert sorted(reopened) == sorted(ids)
    recovered = _payloads(conn, ids[0], "infrastructure_recovered")[-1]
    assert recovered["burst_assignees"] == ["nest-pm", "ps5-pm"]
    assert recovered["burst_cards"] == 4


def test_a_loud_sticky_give_up_is_not_reclassified(conn):
    """The burst's own crashes must be silent: a shared traceback is not the host's."""
    ids = _residue_burst(conn, error=LOUD)

    assert kbd.reclassify_silent_burst_gave_ups(
        conn, now=int(time.time()) + kbd.INFRA_REOPEN_MIN_AGE_S + 1,
    ) == []
    for tid in ids:
        assert _status(conn, tid) == "blocked"


def test_a_card_that_blocked_itself_again_is_not_reclassified(conn):
    ids = _residue_burst(conn)
    with kb.write_txn(conn):
        kb._append_event(conn, ids[0], "blocked", {"reason": "goal unreachable by design"})

    reopened = kbd.reclassify_silent_burst_gave_ups(
        conn, now=int(time.time()) + kbd.INFRA_REOPEN_MIN_AGE_S + 1,
    )

    assert ids[0] not in reopened, "a later block is a decision we must not overrule"
    assert _status(conn, ids[0]) == "blocked"
    assert reopened, "the rest of the burst is still reclassifiable"


def test_a_gave_up_that_already_says_infrastructure_is_left_to_the_probe_recovery(conn):
    tid = _crashed_card(conn, "ps5-pm", SILENT, BURST_AT)
    _crashed_card(conn, "nest-pm", SILENT, BURST_AT + 1)
    _crashed_card(conn, "ps5-pm", SILENT, BURST_AT + 2)
    _sticky_gave_up(conn, tid)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_events SET payload = ? WHERE task_id = ? AND kind = 'gave_up'",
            (json.dumps({"failures": 1, "sticky": True, "infrastructure": True,
                         "error": SILENT}), tid),
        )

    assert kbd.reclassify_silent_burst_gave_ups(
        conn, now=int(time.time()) + kbd.INFRA_REOPEN_MIN_AGE_S + 1,
    ) == []
    assert _status(conn, tid) == "blocked"
