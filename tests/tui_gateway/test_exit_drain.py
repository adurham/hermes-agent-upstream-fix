"""Graceful-exit drain: a stop/update waits for in-flight turns instead of killing them.

The backend's SIGTERM path used to interrupt every running turn at once. It now waits up to
``desktop.exit_drain_seconds`` for them to finish (a restart carries the work), and only then
falls back to the interrupt.
"""
import logging
import threading
import time

import pytest

from tui_gateway import session_reaper


@pytest.fixture
def sessions(monkeypatch):
    state = {}
    monkeypatch.setattr(session_reaper, "_sessions", state, raising=False)
    monkeypatch.setattr(session_reaper, "_sessions_lock", threading.Lock(), raising=False)
    monkeypatch.setattr(session_reaper, "logger", logging.getLogger("test.exit_drain"), raising=False)
    return state


def test_no_running_turns_returns_immediately(sessions):
    started = time.monotonic()
    assert session_reaper._await_running_turns(5.0) is True
    assert time.monotonic() - started < 1.0


def test_drain_waits_for_a_turn_to_finish(sessions):
    sessions["s1"] = {"running": True}

    def finish():
        time.sleep(0.3)
        sessions["s1"]["running"] = False

    threading.Thread(target=finish, daemon=True).start()
    started = time.monotonic()
    assert session_reaper._await_running_turns(10.0) is True
    elapsed = time.monotonic() - started
    assert 0.25 <= elapsed < 5.0


def test_drain_gives_up_at_the_budget(sessions):
    sessions["s1"] = {"running": True}
    started = time.monotonic()
    assert session_reaper._await_running_turns(0.3) is False
    assert 0.25 <= time.monotonic() - started < 3.0


def test_budget_of_zero_disables_the_drain(sessions):
    sessions["s1"] = {"running": True}
    assert session_reaper._await_running_turns(0.0) is False


def test_budget_comes_from_env_then_config(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_EXIT_DRAIN_S", "42")
    assert session_reaper._exit_drain_budget() == 42.0
    monkeypatch.setenv("HERMES_TUI_EXIT_DRAIN_S", "nonsense")
    monkeypatch.setattr(session_reaper, "_EXIT_DRAIN_DEFAULT_S", 7.0, raising=False)
    assert session_reaper._exit_drain_budget() in (7.0, 900.0)  # config read may supply its own value


def test_stop_grace_outlasts_the_drain(monkeypatch):
    from hermes_cli import dashboard_procs

    monkeypatch.setattr(session_reaper, "_exit_drain_budget", lambda: 900.0)
    assert dashboard_procs._stop_grace_seconds() >= 930.0
