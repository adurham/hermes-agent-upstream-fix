"""The loop watchdog: every stall reported with a duration, an attribution and (mid-stall) stacks."""

from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import patch

import pytest

from hermes_cli.loop_watchdog import (
    MAX_RECORDED_STALLS,
    GcActivity,
    LoopWatchdog,
    install,
)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self) -> list[str]:
        return [r.getMessage() for r in self.records]


@pytest.fixture
def cap():
    logger = logging.getLogger("test.loop_watchdog")
    handler = _Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


def _run(coro):
    return asyncio.run(coro)


def test_blocking_call_on_the_loop_is_reported_with_its_duration(cap):
    """A synchronous sleep on the loop is the second cause: it must be timed, not just noticed."""

    async def scenario():
        loop = asyncio.get_running_loop()
        watchdog = install(
            loop, name="test", interval_s=0.05, stall_threshold_s=0.15, stack_dump_after_s=None,
            log=logging.getLogger("test.loop_watchdog"),
        )
        await asyncio.sleep(0.2)
        time.sleep(0.45)  # noqa: ASYNC251 -- the stall under test: no callback can run
        await asyncio.sleep(0.3)
        watchdog.stop()
        return watchdog

    watchdog = _run(scenario())
    stalls = watchdog.stalls()
    assert stalls, "a 0.45 s block must be recorded"
    assert stalls[0]["drift_s"] >= 0.15
    messages = cap.messages()
    stalled = [m for m in messages if m.startswith("event loop stalled")]
    assert stalled, messages
    # iso-certify.py and the ops greps key on this exact prefix: a rewrite would silently
    # stop the stall counters.
    assert stalled[0].startswith("event loop stalled 0.")
    assert "(GIL pressure suspected)" in stalled[0]


def test_gc_idle_attribution_rules_the_collection_out(cap):
    async def scenario():
        loop = asyncio.get_running_loop()
        watchdog = install(
            loop, interval_s=0.05, stall_threshold_s=0.05, stack_dump_after_s=None,
            log=logging.getLogger("test.loop_watchdog"),
        )
        await asyncio.sleep(0.12)
        time.sleep(0.2)  # noqa: ASYNC251 -- forced stall so the attribution runs on a real tick
        await asyncio.sleep(0.15)
        watchdog.stop()
        return watchdog

    _run(scenario())
    stalled = [m for m in cap.messages() if m.startswith("event loop stalled")]
    assert stalled
    assert "attribution=gc idle" in stalled[0]
    assert "threads=" in stalled[0]


def test_a_collection_in_flight_is_named_with_its_generation_and_elapsed(cap):
    """gc=gen2 in flight 25.2s is the attribution that explains a real 25 s stall."""

    async def scenario():
        loop = asyncio.get_running_loop()
        watchdog = LoopWatchdog(
            loop, interval_s=0.05, stall_threshold_s=0.05, stack_dump_after_s=None,
            log=logging.getLogger("test.loop_watchdog"),
        ).start()
        try:
            # Exactly what gc.callbacks delivers while a collection runs (the callback is what the
            # watchdog times; the collector itself refuses to be made slow on demand).
            watchdog._gc._callback("start", {"generation": 2})
            attribution = watchdog._attribution(25.2)
        finally:
            watchdog.stop()
        return attribution

    attribution = _run(scenario())
    assert "attribution=gc gen2 in flight" in attribution
    assert "holds the GIL for its whole length" in attribution


def test_gc_activity_records_generation_duration_and_collected():
    activity = GcActivity()
    activity.install()
    try:
        activity._callback("start", {"generation": 1})
        assert activity.in_flight() is not None
        activity._callback("stop", {"generation": 1, "collected": 7})
        assert activity.in_flight() is None
        recent = activity.recent()
        assert recent[-1]["generation"] == 1
        assert recent[-1]["collected"] == 7
        assert recent[-1]["duration_s"] >= 0.0
    finally:
        activity.uninstall()


def test_no_stack_dump_when_disabled():
    async def scenario():
        loop = asyncio.get_running_loop()
        watchdog = install(
            loop, interval_s=0.05, stall_threshold_s=0.05, stack_dump_after_s=None,
            log=logging.getLogger("test.loop_watchdog"),
        )
        with patch("hermes_cli.loop_watchdog.faulthandler.dump_traceback_later") as dump:
            await asyncio.sleep(0.12)
        watchdog.stop()
        return dump

    dump = _run(scenario())
    assert not dump.called


def test_stacks_are_armed_mid_stall_and_cancelled_each_tick(tmp_path):
    """faulthandler's C timer is what captures the stack WHILE the GIL is held."""
    dump_path = tmp_path / "loop_stall_stacks.log"

    async def scenario():
        loop = asyncio.get_running_loop()
        watchdog = install(
            loop, interval_s=0.05, stall_threshold_s=0.05, stack_dump_after_s=0.2,
            log=logging.getLogger("test.loop_watchdog"), dump_path=str(dump_path),
        )
        with patch(
            "hermes_cli.loop_watchdog.faulthandler.dump_traceback_later"
        ) as dump, patch(
            "hermes_cli.loop_watchdog.faulthandler.cancel_dump_traceback_later"
        ) as cancel:
            await asyncio.sleep(0.16)
        watchdog.stop()
        return dump, cancel

    dump, cancel = _run(scenario())
    assert dump.called, "the C watchdog must be re-armed on every tick"
    delays = [call.args[0] for call in dump.call_args_list]
    assert set(delays) == {0.2}
    assert cancel.call_count >= len(delays) - 1
    assert dump_path.exists(), "the dump file is opened once and stays open for the C thread"


def test_install_is_idempotent_per_loop():
    async def scenario():
        loop = asyncio.get_running_loop()
        first = install(
            loop, interval_s=0.05, stack_dump_after_s=None,
            log=logging.getLogger("test.loop_watchdog"),
        )
        second = install(
            loop, interval_s=0.05, stack_dump_after_s=None,
            log=logging.getLogger("test.loop_watchdog"),
        )
        first.stop()
        return first, second

    first, second = _run(scenario())
    assert first is second


def test_stall_ledger_is_bounded():
    async def scenario():
        loop = asyncio.get_running_loop()
        watchdog = install(
            loop, interval_s=0.02, stall_threshold_s=0.0, stack_dump_after_s=None,
            log=logging.getLogger("test.loop_watchdog"),
        )
        await asyncio.sleep(0.3)
        watchdog.stop()
        return watchdog

    watchdog = _run(scenario())
    assert 0 < len(watchdog.stalls()) <= MAX_RECORDED_STALLS


def test_stop_disarms_the_watchdog():
    async def scenario():
        loop = asyncio.get_running_loop()
        watchdog = install(
            loop, interval_s=0.02, stall_threshold_s=0.0, stack_dump_after_s=None,
            log=logging.getLogger("test.loop_watchdog"),
        )
        await asyncio.sleep(0.1)
        watchdog.stop()
        count = len(watchdog.stalls())
        await asyncio.sleep(0.15)
        return watchdog, count

    watchdog, count = _run(scenario())
    assert len(watchdog.stalls()) == count
