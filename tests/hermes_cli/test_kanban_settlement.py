"""Settlement tests: a deferral recorded as a completion must be machine-readable.

``status='done'`` used to carry two incompatible meanings — verified landing vs
deferred/armed with nothing written — distinguished only by the free text of
``task_runs.summary``. ``t_d898441d`` measured 82 of 1061 greens in 14 days
asserting non-landing in prose only, so no query could tell a settle from a
deferral. These tests pin the contract:

* a handoff whose own words say the subject is not in the tree is recorded
  ``deferred`` (task + run + event), never a plain green settle;
* an explicit ``settlement='landed'`` naming a present subject is verified and
  recorded, while an absent subject is downgraded to ``deferred``;
* a sha mismatch refuses the completion with nothing written;
* ``unsettled_greens`` answers the one query the board could not: which greens
  are deferrals (including legacy rows, classified from prose read-only).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _complete(conn, tid, **kw):
    kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    return kb.complete_task(conn, tid, expected_run_id=run_id, **kw)


def _settle_row(conn, tid):
    return conn.execute(
        "SELECT status, settlement, subject_path, subject_sha256, subject_state "
        "FROM tasks WHERE id = ?", (tid,),
    ).fetchone()


def _run_settlement(conn, tid):
    row = conn.execute(
        "SELECT settlement FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    return row["settlement"] if row else None


def _completed_payload(conn, tid):
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'completed' "
        "ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    return json.loads(row["payload"]) if row and row["payload"] else {}


def test_prose_asserting_non_landing_records_deferred(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="recover the bytes", assignee="worker")
        assert _complete(
            conn, tid,
            summary="Bytes recovered and verified but NOT yet in the tree",
        )
        row = _settle_row(conn, tid)
        assert row["status"] == "done"
        assert row["settlement"] == "deferred"
        assert _run_settlement(conn, tid) == "deferred"
        payload = _completed_payload(conn, tid)
        assert payload["settlement"] == "deferred"
        assert payload["settlement_source"] == "prose"


def test_explicit_landed_with_present_subject_verifies(kanban_home: Path, tmp_path: Path) -> None:
    target = tmp_path / "landed.txt"
    target.write_text("the bytes\n", encoding="utf-8")
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="land a file", assignee="worker")
        assert _complete(
            conn, tid, summary="Wrote the file and measured it",
            settlement="landed", subject=str(target), subject_sha256=sha[:16],
        )
        row = _settle_row(conn, tid)
        assert row["settlement"] == "landed"
        assert row["subject_path"] == str(target)
        assert row["subject_sha256"] == sha[:16]
        assert row["subject_state"] == "present"
        assert _completed_payload(conn, tid)["settlement_source"] == "declared"


def test_landed_claim_with_absent_subject_is_recorded_deferred(
    kanban_home: Path, tmp_path: Path,
) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="claim a landing", assignee="worker")
        assert _complete(
            conn, tid, summary="Everything is in place",
            settlement="landed", subject=str(tmp_path / "never-written.py"),
        )
        row = _settle_row(conn, tid)
        assert row["status"] == "done"
        assert row["settlement"] == "deferred"
        assert row["subject_state"] == "absent"
        assert _completed_payload(conn, tid)["settlement_source"] == "subject_absent"


def test_sha_mismatch_refuses_with_nothing_written(kanban_home: Path, tmp_path: Path) -> None:
    target = tmp_path / "different.txt"
    target.write_text("actual bytes\n", encoding="utf-8")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="mismatched sha", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        with pytest.raises(kb.SettlementError) as excinfo:
            kb.complete_task(
                conn, tid, expected_run_id=run_id, summary="landed, honest",
                settlement="landed", subject=str(target),
                subject_sha256="deadbeefdeadbeef",
            )
        assert "does not verify" in str(excinfo.value)
        row = _settle_row(conn, tid)
        assert row["status"] == "running"          # nothing was mutated
        assert row["settlement"] is None
        # No completed run was synthesized for the refused handoff.
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_runs WHERE task_id=? AND outcome='completed'",
            (tid,),
        ).fetchone()["n"] == 0


def test_request_review_records_deferred_and_approval_inherits(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="armed, nothing written", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        assert kb.request_review(
            conn, tid,
            summary="Fix is armed but NOT LANDED; nothing was written",
            reviewer="reviewer", expected_run_id=run_id,
        ) is True
        assert _settle_row(conn, tid)["settlement"] == "deferred"
        assert _run_settlement(conn, tid) == "deferred"

        # The reviewer's bare approval carries no claim of its own: it must not
        # launder the implementer's deferral into a landing.
        assert kb.complete_task(
            conn, tid, summary="Review approved: looks implementable",
        ) is True
        row = _settle_row(conn, tid)
        assert row["status"] == "done"
        assert row["settlement"] == "deferred"
        assert _completed_payload(conn, tid)["settlement_source"] == "review_inherited"


def test_unsettled_greens_returns_deferred_and_legacy_only(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        landed = kb.create_task(conn, title="plain landing", assignee="worker")
        assert _complete(conn, landed, summary="Wrote and verified the row")

        deferred = kb.create_task(conn, title="explicit deferral", assignee="worker")
        assert _complete(
            conn, deferred, summary="Staged for the next window",
            settlement="deferred", subject="state/payload.json",
        )

        legacy = kb.create_task(conn, title="legacy green", assignee="worker")
        assert _complete(conn, legacy, summary="Landed the fix, bytes on disk")
        # Simulate a pre-settlement row: the column did not exist when it closed.
        conn.execute("UPDATE tasks SET settlement = NULL WHERE id = ?", (legacy,))
        conn.execute(
            "UPDATE task_runs SET settlement = NULL, summary = ? WHERE task_id = ?",
            ("Publish queued; nothing was written in this window", legacy),
        )
        conn.commit()

        rows = {r["id"]: r for r in kb.unsettled_greens(conn, days=14, check_disk=True)}
        assert deferred in rows and rows[deferred]["source"] == "recorded"
        assert legacy in rows and rows[legacy]["source"] == "prose"
        assert landed not in rows


def test_settlement_from_prose_is_narrow() -> None:
    assert kb.settlement_from_prose("NOT yet in the tree") == "deferred"
    assert kb.settlement_from_prose("nothing was written, re-fired, or re-filed") == "deferred"
    assert kb.settlement_from_prose("the unit is still armed") == "deferred"
    assert kb.settlement_from_prose("the publish has NOT landed") == "deferred"
    # Words that merely contain the lexicon are not assertions of non-landing.
    assert kb.settlement_from_prose("landed after the deferred press-free window") is None
    assert kb.settlement_from_prose("") is None
