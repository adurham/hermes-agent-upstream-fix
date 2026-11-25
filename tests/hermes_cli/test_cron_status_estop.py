"""`hermes cron status` must name an engaged ESTOP instead of printing GREEN (kanban t_75dcf7b7).

Measured 2026-11-24 22:14: ``~/.hermes/profiles/ps5-pm/ESTOP`` (reason "safe restart of
hermes-gateway") stood through 40 consecutive tick cycles.  ``cron/scheduler_tick.py`` returned 0
at the ESTOP gate BEFORE the due scan on every one, and this status path still printed
"✓ Gateway is running — cron jobs will fire automatically": the heartbeat it trusts is written
after every cycle OUTSIDE that gate, and the last-success marker is bumped by a refused tick too
(a refusal is not a failure).  So the sentinel is asked FIRST, above the heartbeat/stale/failing
verdicts, and it names the file, the stamp and the reason it read.

The control case matters as much as the engaged one: remove the read and
``test_an_engaged_estop_is_named_and_never_reads_green`` fails on the green line it never prints.
The paused state is proved against a THROWAWAY home, never by pausing the live profile -- an
engaged ESTOP holds every new gateway turn, so a test that paused the host would be a defect.
"""

from __future__ import annotations

import json
import time

import pytest

from hermes_cli.cron import _print_ticker_health

REASON = "safe restart of hermes-gateway"
ENGAGED_AT = "2026-11-24T21:42:39-06:00"


@pytest.fixture()
def world(tmp_path, monkeypatch):
    """A throwaway home whose cron markers read GREEN, plus its own ESTOP path.

    ``agent/estop.py`` binds its two resolvers at import, so the names IT holds are the seam:
    patching them keeps a live operator sentinel out of this pin (and this pin out of the live
    store).
    """
    cron = tmp_path / "cron"
    cron.mkdir()
    now = time.time()
    (cron / "ticker_heartbeat").write_text(f"{now} {12345}\n", encoding="utf-8")
    (cron / "ticker_last_success").write_text(f"{now}\n", encoding="utf-8")
    monkeypatch.setattr("cron.jobs.CRON_DIR", cron)
    monkeypatch.setattr("cron.jobs.JOBS_FILE", cron / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", cron / "output")
    monkeypatch.setattr("agent.estop._hermes_home", lambda: tmp_path)
    monkeypatch.setattr("agent.estop._canonical_root", lambda: tmp_path)
    return tmp_path


def _engage(world, body=None):
    """Write the sentinel the way ``hermes pause`` does (``body`` overrides the exact bytes)."""
    path = world / "ESTOP"
    path.write_text(body if body is not None else json.dumps(
        {"reason": REASON, "engaged_at": ENGAGED_AT}), encoding="utf-8")
    return path


def test_a_dispatching_store_still_reads_green(world, capsys):
    _print_ticker_health([4242])
    out = capsys.readouterr().out
    assert "cron jobs will fire automatically" in out
    assert "EMERGENCY STOP" not in out


def test_an_engaged_estop_is_named_and_never_reads_green(world, capsys):
    path = _engage(world)
    _print_ticker_health([4242])
    out = capsys.readouterr().out
    assert "cron jobs will fire automatically" not in out, (
        "the green 'jobs will fire' line survived an engaged ESTOP: the heartbeat it reads is "
        "written outside the ESTOP gate and says nothing about dispatch")
    assert "EMERGENCY STOP ENGAGED" in out
    assert str(path) in out, "the notice must name the sentinel it read"
    assert REASON in out
    assert ENGAGED_AT in out
    assert "hermes resume" in out
    assert "4242" in out, "the live gateway PID stays visible beside the pause"


def test_a_torn_sentinel_still_pauses_and_says_so(world, capsys):
    path = _engage(world, body="")
    _print_ticker_health([])
    out = capsys.readouterr().out
    assert "cron jobs will fire automatically" not in out
    assert str(path) in out
    assert "no reason recorded" in out
    assert "unstated" in out


def test_lifting_it_restores_the_verdict(world, capsys):
    path = _engage(world)
    _print_ticker_health([])
    assert "EMERGENCY STOP" in capsys.readouterr().out
    path.unlink()
    _print_ticker_health([])
    assert "cron jobs will fire automatically" in capsys.readouterr().out
