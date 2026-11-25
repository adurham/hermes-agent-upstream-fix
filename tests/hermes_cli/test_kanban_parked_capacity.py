"""A capacity deferral is its own state, not a block.

Regression for the 2026-11-24 mint-cap flood: the backlog guard parked eleven
cards with ``kind='transient'`` and the text "unblock to resume". Each park read
as a repeated card-level failure, the unblock-loop breaker moved them to
``triage``, and every one of them cost an orchestration turn to diagnose — while
the actual cause was "the lane's ready queue is at its ceiling".

The contract pinned here:

* ``parked_capacity`` is a VALID block kind that carries the lane ``ceiling``
  (``--ceiling``) it parked against;
* it NEVER touches ``block_recurrences`` and NEVER routes to ``triage``, however
  many park/resume cycles it goes through (the failure counter is unchanged);
* it is sticky against ``recompute_ready`` (a park must not be promoted back by
  the parent-completion pass, which knows nothing about the ceiling);
* ``resume_capacity_parks`` returns parked cards to ``ready`` in CREATION order
  as the lane drains below the recorded ceiling, with no human, no unblock and
  no supervisor — and the dispatcher tick calls it on its own;
* the park fires ``kanban_task_parked_capacity``, never ``kanban_task_blocked``,
  so block-oriented observers do not page on capacity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _ready(conn, title="card", assignee="ps5-pm") -> str:
    """A ``ready`` card (create_task lands there unless it has open parents)."""
    tid = kb.create_task(conn, title=title, assignee=assignee)
    assert kb.get_task(conn, tid).status == "ready"
    return tid


def _park(conn, tid, ceiling, reason="lane ready queue at ceiling"):
    assert kb.block_task(
        conn, tid, reason=reason, kind="parked_capacity", ceiling=ceiling,
    )
    return kb.get_task(conn, tid)


# ---------------------------------------------------------------------------
# (a) the park carries parked_capacity + its ceiling
# ---------------------------------------------------------------------------


def test_park_records_kind_and_ceiling(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _ready(conn)
        parked = _park(conn, tid, ceiling=2)
        event = kb.list_events(conn, tid)[-1]
    assert parked.status == "blocked"
    assert parked.block_kind == "parked_capacity"
    assert parked.park_ceiling == 2
    assert event.kind == "parked_capacity"
    assert (event.payload or {}).get("ceiling") == 2


def test_park_requires_a_positive_ceiling(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _ready(conn)
        with pytest.raises(ValueError):
            kb.block_task(conn, tid, kind="parked_capacity")
        with pytest.raises(ValueError):
            kb.block_task(conn, tid, kind="parked_capacity", ceiling=0)
        assert kb.get_task(conn, tid).status == "ready"


def test_ceiling_flag_reaches_the_kernel(kanban_home: Path, capsys) -> None:
    """``hermes kanban block <id> --kind parked_capacity --ceiling N``."""
    import argparse

    with kbc.connect_closing() as conn:
        tid = _ready(conn)
    args = argparse.Namespace(
        task_id=tid, ids=None, reason=["lane", "full"], kind="parked_capacity", ceiling=3,
    )
    assert kanban_cli._cmd_block(args) == 0
    out = capsys.readouterr().out
    assert "Parked" in out and "not a failure" in out
    with kbc.connect_closing() as conn:
        parked = kb.get_task(conn, tid)
    assert (parked.block_kind, parked.park_ceiling) == ("parked_capacity", 3)


# ---------------------------------------------------------------------------
# (b) the failure / escalation counter never sees a capacity park
# ---------------------------------------------------------------------------


def test_five_park_resume_cycles_never_escalate(kanban_home: Path) -> None:
    """Far past BLOCK_RECURRENCE_LIMIT: still blocked-with-kind, never triage."""
    with kbc.connect_closing() as conn:
        tid = _ready(conn)
        for cycle in range(kb.BLOCK_RECURRENCE_LIMIT + 3):
            parked = _park(conn, tid, ceiling=4)
            assert parked.status == "blocked", f"cycle {cycle} escalated to {parked.status}"
            assert parked.block_kind == "parked_capacity"
            # The park is undone by capacity resume, never by a human unblock.
            assert kb.resume_capacity_parks(conn) == [tid]
            after = kb.get_task(conn, tid)
            assert after.status == "ready"
            assert after.block_kind is None and after.park_ceiling is None
            assert after.block_recurrences == 0
            assert after.consecutive_failures == 0
        kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert "block_loop_detected" not in kinds


def test_capacity_park_fires_its_own_hook_only(kanban_home: Path, monkeypatch) -> None:
    fired: list[str] = []
    monkeypatch.setattr(kb, "_fire_task_hook", lambda event, *a, **k: fired.append(event))
    with kbc.connect_closing() as conn:
        tid = _ready(conn)
        kb.block_task(conn, tid, reason="full", kind="parked_capacity", ceiling=1)
        _park_control = _ready(conn, title="real block")
        kb.block_task(conn, _park_control, reason="needs an answer", kind="needs_input")
    assert fired == ["kanban_task_parked_capacity", "kanban_task_blocked"]


# ---------------------------------------------------------------------------
# (c) automatic resume, in creation order, with no external action
# ---------------------------------------------------------------------------


def test_resume_never_runs_ahead_of_the_ceiling(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        keep = [_ready(conn, title=f"keep-{i}") for i in range(2)]
        parked = [_ready(conn, title=f"park-{i}") for i in range(3)]
        for tid in parked:
            _park(conn, tid, ceiling=2)
        # ready == ceiling (2): nothing may come back.
        assert kb.resume_capacity_parks(conn) == []
        assert all(kb.get_task(conn, t).status == "blocked" for t in parked)

        # Drain one slot: exactly one card resumes, the OLDEST parked one.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (keep[0],))
        assert kb.resume_capacity_parks(conn) == [parked[0]]
        assert kb.get_task(conn, parked[0]).status == "ready"
        assert kb.get_task(conn, parked[1]).status == "blocked"

        # Drain again: the next oldest, not the newest.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parked[0],))
        assert kb.resume_capacity_parks(conn) == [parked[1]]
        assert kb.get_task(conn, parked[2]).status == "blocked"


def test_resume_event_is_not_an_unblock(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _ready(conn)
        _park(conn, tid, ceiling=1)
        assert kb.resume_capacity_parks(conn) == [tid]
        kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert "capacity_resumed" in kinds
    assert "unblocked" not in kinds


def test_recompute_ready_does_not_promote_a_capacity_park(kanban_home: Path) -> None:
    """Parent-completion promotion knows nothing about the lane ceiling."""
    with kbc.connect_closing() as conn:
        for i in range(2):
            _ready(conn, title=f"filler-{i}")
        tid = _ready(conn, title="parked")
        _park(conn, tid, ceiling=1)
        # ready (2) still above the ceiling: recompute must leave it alone.
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"


def test_dispatch_tick_resumes_with_no_external_action(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    spawns: list[str] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 4242

    with kbc.connect() as conn:
        tid = _ready(conn)
        _park(conn, tid, ceiling=1)
        # ready == 0 < ceiling: the very next tick returns and claims it.
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert res.capacity_resumed == [tid]
        assert kb.get_task(conn, tid).status == "running"
    assert spawns == [tid]


# ---------------------------------------------------------------------------
# Back-compat: a human block is still a block
# ---------------------------------------------------------------------------


def test_transient_still_counts_and_escalates(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _ready(conn)
        for _ in range(kb.BLOCK_RECURRENCE_LIMIT - 1):
            kb.block_task(conn, tid, reason="flaky", kind="transient")
            kb.unblock_task(conn, tid)
        kb.block_task(conn, tid, reason="flaky", kind="transient")
        landed = kb.get_task(conn, tid)
        assert landed.status == "triage"
        assert landed.block_recurrences == kb.BLOCK_RECURRENCE_LIMIT
