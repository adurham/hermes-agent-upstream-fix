"""A fenced (``unknown``) owner receipt must not libel a card that ran, and must say where it looked.

Measured 2026-11-24: 31 of ~90 dispatcher runs in one day crashed with the identical sentence
"Kanban owner refused: Kanban execution unknown; retained by owner" — across ps5-pm, nest-pm and
default, ``oom_kill=0`` in every row. Every inspected attempt had ALREADY written its own
``worker_result`` task event (exit 1, "Hermes couldn't save this conversation") before the owner
fenced the receipt: the turn ran, the settlement write could not commit. The refusal named no
board, no DB and no claim, so the board showed a dead card and nothing to diagnose.

These tests pin the client half:
  * an attempt's own ``worker_result`` outranks the fence (its reply IS the answer);
  * ``unknown`` is not final — the owner's drain settles an unattended card admission, so the
    client waits (bounded) for that or for the attempt's result;
  * a refusal is recorded ON THE CARD with resolved_board / card_board / db_path / claim_id.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from hermes_cli import gateway_client, kanban_db_connect as kbc, kanban_worker_client as kwc

CLAIM = "this-host:424242"
TASK = "t_refused"
RUN = 5
# Captured before any test patches ``kwc.asyncio`` (the module object IS ``asyncio``'s).
_REAL_SLEEP = asyncio.sleep


class _Board:
    """A scratch board holding one ``running`` card and its attempt."""

    def __init__(self, root):
        self.home = root / "hermes"
        (self.home / "kanban").mkdir(parents=True)
        self.path = self.home / "kanban.db"
        kbc.init_db(self.path)
        self.conn = kbc.connect(self.path)
        self.conn.execute(
            "INSERT INTO tasks (id, title, assignee, status, created_at, claim_lock, claim_expires, "
            "worker_pid, current_run_id, started_at, priority) "
            "VALUES (?, ?, 'default', 'running', ?, ?, ?, 999999, ?, ?, 5)",
            (TASK, TASK, 1, CLAIM, 10**12, RUN, 1))
        self.conn.execute(
            "INSERT INTO task_runs (id, task_id, profile, status, claim_lock, claim_expires, worker_pid, started_at) "
            "VALUES (?, ?, 'default', 'running', ?, ?, 999999, ?)", (RUN, TASK, CLAIM, 10**12, 1))
        self.conn.commit()

    @property
    def params(self):
        return {"board": "default", "task_id": TASK, "run_id": RUN, "claim_lock": CLAIM}

    def event(self, kind: str, payload: dict, run_id=RUN):
        self.conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?, ?, ?, ?, ?)",
            (TASK, run_id, kind, json.dumps(payload), 2))
        self.conn.commit()

    def result(self, exit_code=1, last_output="the turn's own last words", claim_lock=CLAIM):
        self.event("worker_result", {"pid": 999999, "claim_lock": claim_lock,
                                     "exit_code": exit_code, "last_output": last_output})

    def events(self, kind: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
            (TASK, kind)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def close(self):
        self.conn.close()


@pytest.fixture()
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_UNKNOWN_GRACE_SECONDS", raising=False)
    scratch = _Board(tmp_path)
    yield scratch
    scratch.close()


# --- the attempt's own result outranks the fence ---------------------------------------------

def test_an_unknown_receipt_whose_attempt_recorded_its_result_is_not_a_refusal(board, capsys):
    """The card RAN: its ``worker_result`` is the answer, so the worker exits with that code."""
    board.result(exit_code=1, last_output="⚠️ No reply: Hermes couldn't save this conversation")
    assert kwc._exit_code({"status": "unknown"}, board.params, str(board.path)) == 1
    assert "couldn't save this conversation" in capsys.readouterr().out
    assert board.events("dispatch_refused") == []


def test_a_neighbour_attempts_result_does_not_answer_for_this_one(board):
    """``worker_result`` is exact: another claim's reply must not be read as this attempt's."""
    board.result(exit_code=0, claim_lock="someone-else:1")
    assert kwc._attempt_result(str(board.path), board.params) == {}


def test_a_terminal_receipt_still_prefers_the_attempts_own_result(board):
    board.result(exit_code=3, last_output="done, mostly")
    assert kwc._exit_code({"status": "terminal"}, board.params, str(board.path)) == 3


# --- a refusal names its reason and lands on the card -----------------------------------------

def test_a_refusal_records_resolved_board_card_board_db_and_claim(board):
    with pytest.raises(gateway_client.GatewayClientError) as caught:
        kwc._exit_code({"status": "unknown"}, board.params, str(board.path))

    sentence = str(caught.value)
    assert sentence.startswith("Kanban execution unknown; retained by owner")
    assert "resolved_board=default" in sentence
    assert "card_board=default" in sentence
    assert f"db_path={board.path.resolve()}" in sentence
    assert f"claim_id={CLAIM}" in sentence

    [record] = board.events("dispatch_refused")
    assert record["resolved_board"] == "default"
    assert record["card_board"] == ["default"]
    assert record["card_board_path"] == str(board.path.resolve())
    assert record["db_path"] == str(board.path.resolve())
    assert record["claim_id"] == CLAIM
    assert record["run_id"] == RUN and record["task_id"] == TASK
    assert record["session_id"] is None and record["admission_id"] is None


def test_a_refusal_names_the_board_that_actually_holds_a_cross_lane_card(board):
    """The resolved board may not be the card's: a refusal that hides that is undiagnosable."""
    kbc.init_db(board.home / "kanban" / "boards" / "hub" / "kanban.db")
    board.conn.execute("DELETE FROM tasks WHERE id = ?", (TASK,))
    board.conn.execute("DELETE FROM task_runs WHERE id = ?", (RUN,))
    board.conn.commit()
    hub = kbc.connect(board.home / "kanban" / "boards" / "hub" / "kanban.db")
    hub.execute("INSERT INTO tasks (id, title, assignee, status, created_at, priority) "
                "VALUES (?, ?, 'ps5-pm', 'ready', 1, 5)", (TASK, TASK))
    hub.commit()
    hub.close()

    record = kwc._refusal_record(board.params, str(board.path), None)
    assert record["resolved_board"] == "default"
    assert record["card_board"] == ["hub"]
    assert record["card_board_path"] == str((board.home / "kanban" / "boards" / "hub" / "kanban.db").resolve())

    # The record lands where the CARD is, not where the worker resolved its board: an event for a
    # task id the resolved file never heard of is a row nobody can join.
    kwc._record_refusal(record)
    hub = kbc.connect(board.home / "kanban" / "boards" / "hub" / "kanban.db")
    try:
        assert hub.execute("SELECT count(*) FROM task_events WHERE task_id = ? AND kind = 'dispatch_refused'",
                           (TASK,)).fetchone()[0] == 1
    finally:
        hub.close()
    assert board.events("dispatch_refused") == []


# --- unknown is not final: wait for the attempt's result or the owner's drain -----------------

class _Client:
    """``prompt.receipt`` answers from a script; the last answer repeats."""

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    async def rpc(self, method, **params):
        assert method == "prompt.receipt"
        self.calls += 1
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return {"status": status}


def _claim():
    return kwc._Claim(answered=0.0, session_id="local-1", admission_id="admission-1")


@pytest.fixture()
def _no_sleep(monkeypatch):
    monkeypatch.setattr(kwc.asyncio, "sleep", lambda *_: _REAL_SLEEP(0))


@pytest.mark.asyncio
async def test_unknown_then_terminal_is_waited_out_not_refused(board, _no_sleep):
    """The owner's drain settles an unattended card admission right after fencing it."""
    client = _Client(["unknown", "unknown", "terminal"])
    receipt = await kwc._await_receipt(client, board.params, str(board.path), _claim())
    assert receipt["status"] == "terminal"


@pytest.mark.asyncio
async def test_an_attempt_result_arriving_under_the_fence_ends_the_wait(board, monkeypatch):
    def _result_then_yield(*_):
        board.result(exit_code=0)
        return _REAL_SLEEP(0)

    monkeypatch.setattr(kwc.asyncio, "sleep", _result_then_yield)
    receipt = await kwc._await_receipt(_Client(["unknown"]), board.params, str(board.path), _claim())
    assert receipt["status"] == "unknown"
    assert kwc._exit_code(receipt, board.params, str(board.path)) == 0


@pytest.mark.asyncio
async def test_the_grace_is_bounded_and_the_refusal_follows(board, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_UNKNOWN_GRACE_SECONDS", "0")
    monkeypatch.setattr(kwc.asyncio, "sleep", lambda *_: _REAL_SLEEP(0))
    receipt = await kwc._await_receipt(_Client(["unknown"]), board.params, str(board.path), _claim())
    assert receipt["status"] == "unknown"
    with pytest.raises(gateway_client.GatewayClientError):
        kwc._exit_code(receipt, board.params, str(board.path), claim=_claim())
    assert board.events("dispatch_refused")[0]["admission_id"] == "admission-1"


def test_the_grace_knob_is_operator_settable(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_UNKNOWN_GRACE_SECONDS", raising=False)
    assert kwc._unknown_grace_s() == kwc.UNKNOWN_RESOLUTION_GRACE_S
    monkeypatch.setenv("HERMES_KANBAN_UNKNOWN_GRACE_SECONDS", "5")
    assert kwc._unknown_grace_s() == 5.0
    monkeypatch.setenv("HERMES_KANBAN_UNKNOWN_GRACE_SECONDS", "nonsense")
    assert kwc._unknown_grace_s() == kwc.UNKNOWN_RESOLUTION_GRACE_S
