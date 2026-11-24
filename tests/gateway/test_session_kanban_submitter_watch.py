"""The reap must END a worker that outlived its run, not just stop looking at it.

``gateway.session_kanban._watch_submitter`` is a managed worker's life tie: every second the bound
interpreter reads its run row's ``worker_pid`` and ``os._exit(1)``s once that worker is dead. So
``hermes_cli.kanban_db_dispatch.reap_terminal_workers`` records its work in ``worker_reaped_at`` and
RETAINS the pid — clearing the column is the watcher's "the dispatcher has not recorded a pid yet"
state, and a watched-NULL run is blind, not dead. Measured on the live board: 12 of 587 ended runs
outlived their run past the reaper's own 120 s grace (worst 2347 s), and the incident interpreter
published into the shared tree 772 s after its run was recorded done.

Arms (fixture board; the only processes signalled are sleepers this module spawned):
  * the blind control — NULL is "not recorded yet", never dead;
  * the tie — a retained pid that is gone ends the interpreter within one watcher tick;
  * the guard — an attempt that is still ``running`` is never touched;
  * end-to-end through the real reaper and the real watcher (the acceptance shape).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as conn:
        yield conn, db_path


@pytest.fixture
def watch(monkeypatch):
    """Start the REAL ``_watch_submitter``, with only the process-ending syscall intercepted.

    Same SELECT, same ``_worker_alive`` predicate, same one-second tick as the managed interpreter;
    ``os._exit`` is recorded instead of ending pytest, so the decision is observable.
    """
    started: list[threading.Event] = []

    def start(db_path, run_id):
        done = threading.Event()
        exits: list[int] = []

        def fake_exit(code):
            exits.append(code)
            done.set()

        monkeypatch.setattr(session_kanban, "_TURNS_DONE", done)
        monkeypatch.setattr(os, "_exit", fake_exit)
        session_kanban._watch_submitter(db_path, run_id)
        started.append(done)
        return done, exits

    yield start
    for done in started:
        done.set()


def _sleeper():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.2)
    return proc


def _terminal_run(conn, proc, *, ended_ago: int = 600) -> tuple[str, int]:
    """A card this dispatcher spawned ``proc`` for, closed ``ended_ago`` seconds ago."""
    tid = kb.create_task(conn, title="finished", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    assert run_id is not None
    kbd._set_worker_pid(conn, tid, proc.pid)
    assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
    conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?", (ended_ago, run_id))
    return tid, run_id


def _wait_for(predicate, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_a_cleared_pid_is_blind_not_dead(board, watch):
    """The defect this fix removes, kept as the control: a reap that wiped the pid left the
    interpreter with nothing to notice, so it ran on with the tree-write path open."""
    conn, db_path = board
    proc = _sleeper()
    try:
        _tid, run_id = _terminal_run(conn, proc)
        conn.execute("UPDATE task_runs SET worker_pid = NULL, worker_started_at = NULL WHERE id=?", (run_id,))
        _done, exits = watch(db_path, run_id)
        assert not _wait_for(lambda: bool(exits), timeout=2.5), "a NULL pid must not be read as dead"
    finally:
        proc.kill()
        proc.wait()


def test_a_dead_retained_pid_ends_the_bound_interpreter(board, watch):
    conn, db_path = board
    proc = _sleeper()
    try:
        _tid, run_id = _terminal_run(conn, proc)
        proc.kill()
        proc.wait()
        _done, exits = watch(db_path, run_id)
        assert _wait_for(lambda: bool(exits), timeout=3.0), "a retained dead pid ends the interpreter"
        assert exits == [1]  # the watcher's own exit code
        row = conn.execute(
            "SELECT worker_pid, worker_started_at FROM task_runs WHERE id=?", (run_id,)
        ).fetchone()
        assert row["worker_pid"] == proc.pid and row["worker_started_at"] is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_a_running_attempt_is_never_touched(board, watch):
    conn, db_path = board
    proc = _sleeper()
    try:
        tid = kb.create_task(conn, title="running", assignee="coder")
        kb.claim_task(conn, tid, claimer=kb._claimer_id())
        run_id = kb._current_run_id(conn, tid)
        kbd._set_worker_pid(conn, tid, proc.pid)
        _done, exits = watch(db_path, run_id)
        assert not _wait_for(lambda: bool(exits), timeout=2.5)
        row = conn.execute(
            "SELECT ended_at, worker_reaped_at FROM task_runs WHERE id=?", (run_id,)
        ).fetchone()
        assert row["ended_at"] is None and row["worker_reaped_at"] is None
    finally:
        proc.kill()
        proc.wait()


def test_reap_ends_the_survivor_through_the_watcher(board, watch):
    """The acceptance shape: a run recorded done, a bound interpreter already watching it, and
    the dispatcher's reap. The interpreter must be gone within one watcher tick of the kill, and
    the evidence it watched must still be on the row."""
    conn, db_path = board
    proc = _sleeper()
    try:
        tid, run_id = _terminal_run(conn, proc)
        _done, exits = watch(db_path, run_id)

        assert kbd.reap_terminal_workers(conn) == [tid]
        assert proc.wait(timeout=10) is not None
        assert _wait_for(lambda: bool(exits), timeout=3.0), (
            "the reap must end the bound interpreter, not blind it"
        )
        row = conn.execute(
            "SELECT worker_pid, worker_started_at, worker_reaped_at FROM task_runs WHERE id=?", (run_id,)
        ).fetchone()
        assert row["worker_pid"] == proc.pid and row["worker_started_at"] is not None
        assert row["worker_reaped_at"] is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
