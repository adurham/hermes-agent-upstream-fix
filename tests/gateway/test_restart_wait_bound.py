"""The restart wait is BOUNDED: a user-initiated restart converges even when a holder never goes idle.

Measured failure this pins shut (2026-11-23): a no_agent cron script run inside the gateway cgroup
held ``_awaitable_work_count`` forever, ``await_idle`` warned every ``restart_after_turn_timeout``
and never ended, and the dispatched kanban lanes stayed parked for the whole wait.
"""
import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.restart_when_idle as idle_mod
from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_WAIT_TIMEOUT,
    DEFAULT_GATEWAY_UPDATE_RESTART_WAIT_TIMEOUT,
    parse_restart_wait_timeout,
    parse_update_restart_wait_timeout,
)
from gateway.restart_when_idle import (
    arm_external_restart_backstop,
    external_restart_argv,
    force_restart,
    request_restart,
    request_service_restart,
)
from tests.gateway.restart_test_helpers import make_restart_runner


def _runner(monkeypatch):
    """A restart runner whose stop() and owner notice are observable, and whose external reaper is
    never actually spawned (the argv its reaper would run has its own test)."""
    runner, _adapter = make_restart_runner()
    runner._scale_to_zero_status = MagicMock()
    runner.stop = AsyncMock()
    runner._notify_home_channels_restart_forced = AsyncMock(return_value=True)
    monkeypatch.setattr(idle_mod, "arm_external_restart_backstop", MagicMock(return_value=True))
    return runner


def _hold(monkeypatch, unit: dict) -> None:
    """One holder that never goes away — today's ``switch-crash-alert`` no_agent cron run."""
    monkeypatch.setattr(idle_mod, "background_units", lambda _runner: [dict(unit)])


@pytest.mark.asyncio
async def test_bound_expiry_stops_live_work_reports_it_and_restarts(monkeypatch, caplog):
    runner = _runner(monkeypatch)
    runner._restart_wait_timeout = 1.2
    _hold(monkeypatch, {"kind": "cron", "job_id": "13723b9c8575", "pid": 2786254})

    caplog.set_level(logging.WARNING, logger="gateway.run")
    started = time.monotonic()
    assert request_restart(runner, detached=False, via_service=True) is True
    await asyncio.wait_for(runner._restart_task, timeout=5.0)
    elapsed = time.monotonic() - started

    # The restart happened, through the existing force escalation: no drain, exit for the service.
    runner.stop.assert_awaited_once_with(restart=True, detached_restart=False, service_restart=True)
    assert runner._restart_drain_timeout == 0.0
    # Bounded, and not before the bound.
    assert 1.2 <= elapsed < 4.0

    # Visible, never silent: a WARNING naming the unit, and one line to the owner's home channels.
    assert "Restart wait bound 1s" in caplog.text
    assert "13723b9c8575" in caplog.text
    runner._notify_home_channels_restart_forced.assert_awaited_once()
    notice = runner._notify_home_channels_restart_forced.await_args.args[0]
    assert "13723b9c8575" in notice and "after 1s" in notice
    # The outside-the-process restart was armed before the in-process stop could wedge.
    assert idle_mod.arm_external_restart_backstop.called


@pytest.mark.asyncio
async def test_a_turn_finishing_inside_the_bound_drains_cleanly(monkeypatch):
    """The honest slow case: nothing is interrupted, and the normal drain budget survives."""
    runner = _runner(monkeypatch)
    runner._restart_wait_timeout = 5.0
    runner._restart_drain_timeout = 3.0
    runner._restart_after_turn_timeout = 0.05
    holder = [{"kind": "chat", "session": "s1"}]
    monkeypatch.setattr(idle_mod, "background_units", lambda _runner: list(holder))

    assert request_restart(runner, detached=False, via_service=True) is True
    asyncio.get_running_loop().call_later(0.2, holder.clear)
    await asyncio.wait_for(runner._restart_task, timeout=5.0)

    runner.stop.assert_awaited_once_with(restart=True, detached_restart=False, service_restart=True)
    # Not forced: the configured drain budget is untouched and nobody was told work was killed.
    assert runner._restart_drain_timeout == 3.0
    runner._notify_home_channels_restart_forced.assert_not_awaited()
    assert not idle_mod.arm_external_restart_backstop.called


@pytest.mark.asyncio
async def test_bound_zero_keeps_the_old_unbounded_wait(monkeypatch):
    """``agent.restart_wait_timeout: 0`` is the opt-out: warn forever, never force."""
    runner = _runner(monkeypatch)
    runner._restart_wait_timeout = 0.0
    _hold(monkeypatch, {"kind": "cron", "job_id": "j1"})

    assert request_restart(runner, detached=False, via_service=True) is True
    assert runner._restart_wait_bound == 0.0
    await asyncio.sleep(0.4)
    runner.stop.assert_not_awaited()
    assert not runner._restart_task.done()
    runner._restart_task.cancel()


@pytest.mark.asyncio
async def test_update_pause_keeps_its_own_longer_bound(monkeypatch):
    """An update must not be forced mid-write: adopting the wait re-bounds it to the update bound."""
    runner = _runner(monkeypatch)
    runner._restart_wait_timeout = 0.1
    runner._update_restart_wait_timeout = 30.0
    _hold(monkeypatch, {"kind": "cron", "job_id": "j1"})

    assert request_restart(runner, detached=False, via_service=True) is True
    assert runner._restart_wait_bound == 0.1
    assert request_service_restart(runner) is True  # the pause adopts the waiting restart
    assert runner._restart_wait_bound == 30.0
    await asyncio.sleep(0.4)  # well past the restart bound, far inside the update bound
    runner.stop.assert_not_awaited()
    assert not runner._restart_task.done()
    runner._restart_task.cancel()


@pytest.mark.asyncio
async def test_an_explicit_force_still_stops_at_once(monkeypatch):
    runner = _runner(monkeypatch)
    runner._restart_wait_timeout = 600.0
    _hold(monkeypatch, {"kind": "chat", "session": "s1"})

    assert request_restart(runner, detached=False, via_service=True) is True
    assert force_restart(runner) is True
    await asyncio.wait_for(runner._restart_task, timeout=2.0)
    runner.stop.assert_awaited_once_with(restart=True, detached_restart=False, service_restart=True)
    assert runner._restart_drain_timeout == 0.0


def test_external_backstop_bounces_the_unit_only_while_this_pid_lives():
    argv = external_restart_argv(4242, "hermes-gateway.service", 30)
    assert argv[:2] == ["bash", "-lc"]
    script = argv[2]
    assert "sleep 30" in script
    assert "kill -0 4242" in script
    assert "systemctl --user restart hermes-gateway.service" in script


def test_backstop_is_not_armed_for_a_detached_restart(monkeypatch):
    runner, _adapter = make_restart_runner()
    runner._restart_via_service = False
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    assert arm_external_restart_backstop(runner) is False


def test_backstop_restarts_the_unit_through_systemd(monkeypatch):
    runner, _adapter = make_restart_runner()
    runner._restart_via_service = True
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    monkeypatch.setattr(idle_mod.shutil, "which", lambda _name: "/usr/bin/systemctl")
    monkeypatch.setattr("gateway.shutdown_forensics.systemd_unit_name", lambda: "hermes-gateway.service")
    popen = MagicMock()
    monkeypatch.setattr(idle_mod.subprocess, "Popen", popen)

    assert arm_external_restart_backstop(runner) is True
    argv = popen.call_args.args[0]
    assert "systemctl --user restart hermes-gateway.service" in argv[2]
    assert runner._restart_backstop_armed is True
    # Armed once per process: a second expiry must not stack reapers.
    assert arm_external_restart_backstop(runner) is False


def test_restart_wait_timeout_defaults_and_parses():
    assert parse_restart_wait_timeout(None) == 120.0 == DEFAULT_GATEWAY_RESTART_WAIT_TIMEOUT
    assert parse_restart_wait_timeout("") == DEFAULT_GATEWAY_RESTART_WAIT_TIMEOUT
    assert parse_restart_wait_timeout("bogus") == DEFAULT_GATEWAY_RESTART_WAIT_TIMEOUT
    assert parse_restart_wait_timeout(0) == 0.0
    assert parse_restart_wait_timeout("-5") == 0.0
    assert parse_restart_wait_timeout("45") == 45.0
    assert parse_update_restart_wait_timeout(None) == 1800.0 == DEFAULT_GATEWAY_UPDATE_RESTART_WAIT_TIMEOUT
    assert parse_update_restart_wait_timeout(0) == 0.0


def test_loader_reads_config_then_env_and_keeps_zero(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    for key in ("HERMES_RESTART_WAIT_TIMEOUT", "HERMES_UPDATE_RESTART_WAIT_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "config.yaml").write_text(
        "agent:\n  restart_wait_timeout: 33\n  update_restart_wait_timeout: 900\n", encoding="utf-8"
    )
    assert gateway_run.GatewayRunner._load_restart_wait_timeout() == 33.0
    assert gateway_run.GatewayRunner._load_update_restart_wait_timeout() == 900.0

    monkeypatch.setenv("HERMES_RESTART_WAIT_TIMEOUT", "0")
    assert gateway_run.GatewayRunner._load_restart_wait_timeout() == 0.0


def test_forced_notice_reaches_every_served_home_channel():
    """The escalation line is delivered per served home channel; a failure costs the line, not the wait."""
    import asyncio
    from unittest.mock import MagicMock

    from gateway.config import Platform
    from gateway.run_notifications import GatewayNotificationsMixin

    runner = MagicMock()
    sent: list[tuple[str, str]] = []
    runner._served_home_channel_transports = lambda: [
        (None, Platform.DISCORD, MagicMock(), MagicMock(chat_id="dm-1"), MagicMock()),
        (None, Platform.DISCORD, MagicMock(), MagicMock(chat_id="dm-2"), MagicMock()),
    ]

    async def _send(_platform, home, _transport, message, _fmt):
        sent.append((home.chat_id, message))
        return True

    runner._send_home_channel_message = _send

    delivered = asyncio.run(
        GatewayNotificationsMixin._notify_home_channels_restart_forced(runner, "bound expired"))

    assert delivered is True
    assert [chat for chat, _ in sent] == ["dm-1", "dm-2"]
    assert all("bound expired" in message for _, message in sent)
