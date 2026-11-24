"""The gateway arms a stall reporter, so a stalled loop always has a duration and a stack."""

from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import MagicMock, patch

from gateway.run_startup_liveness import start_loop_liveness_guards


def _stub_runner(loop_watchdog: bool = True):
    runner = MagicMock()
    runner.config = MagicMock()
    runner.config.loop_watchdog = loop_watchdog
    runner.config.log_dir = None
    runner.config.loop_watchdog_probe_interval_s = 30.0
    runner.config.loop_watchdog_probe_timeout_s = 10.0
    runner.config.loop_watchdog_max_strikes = 3
    return runner


def test_stall_reporter_is_armed_on_the_gateway_loop(tmp_path):
    """The gateway had no duration measurement: a stall only showed as a victim's 10 s write timeout."""
    async def scenario():
        loop = asyncio.get_running_loop()
        runner = _stub_runner()
        with (
            patch("hermes_state_loop_guard.install_from_config"),
            patch("gateway.shutdown_watchdog._arm_loop_floor_timer", return_value=None),
            patch("gateway.shutdown_watchdog.start_loop_liveness_watchdog", return_value=None),
        ):
            start_loop_liveness_guards(runner, loop)
        watchdog = runner._loop_stall_watchdog
        assert watchdog is not None, "the reporter must be armed"
        await asyncio.sleep(0.3)
        watchdog.stop()
        return watchdog

    watchdog = asyncio.run(scenario())
    assert watchdog.stalls() == [] or all("drift_s" in s for s in watchdog.stalls())


def test_stall_reporter_survives_the_loop_watchdog_opt_out(tmp_path):
    """``gateway.loop_watchdog: false`` disarms the hard-exit backstop, not the diagnostic."""
    async def scenario():
        loop = asyncio.get_running_loop()
        runner = _stub_runner(loop_watchdog=False)
        with (
            patch("hermes_state_loop_guard.install_from_config"),
            patch("gateway.shutdown_watchdog._arm_loop_floor_timer", return_value=None),
            patch("gateway.shutdown_watchdog.start_loop_liveness_watchdog", return_value=None) as start,
        ):
            start_loop_liveness_guards(runner, loop)
        started = start.called
        watchdog = runner._loop_stall_watchdog
        watchdog.stop() if watchdog is not None else None
        return watchdog, started

    watchdog, started = asyncio.run(scenario())
    assert watchdog is not None
    assert started is False, "the hang backstop stays gated by config"


def test_a_stall_with_a_collection_in_flight_is_attributed(tmp_path):
    """The report names the collection and its elapsed time, which is what makes a 25 s stall actionable."""
    from hermes_cli.loop_watchdog import install

    log = logging.getLogger("gateway.run")
    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _H()
    log.addHandler(handler)
    try:
        async def scenario():
            loop = asyncio.get_running_loop()
            watchdog = install(
                loop, name="gateway", interval_s=0.05, stall_threshold_s=0.05,
                stack_dump_after_s=None, log=log,
            )
            await asyncio.sleep(0.12)
            watchdog._gc._callback("start", {"generation": 2})
            time.sleep(0.2)  # noqa: ASYNC251 -- the stall under test
            await asyncio.sleep(0.15)
            watchdog.stop()
            return watchdog

        asyncio.run(scenario())
    finally:
        log.removeHandler(handler)

    stalled = [m for m in records if m.startswith("event loop stalled")]
    assert stalled
    assert "attribution=gc gen2 in flight" in stalled[0]
