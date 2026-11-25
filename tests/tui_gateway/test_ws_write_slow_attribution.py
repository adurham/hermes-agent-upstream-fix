"""``ws write slow (loop stalled >10.0s)`` must say what was actually measured.

The timeout is a pool thread waiting for the loop to run a send that was scheduled on it, and
``_safe_send_many`` holds the connection's ``_send_lock`` across ``await send_text(...)``. So the
same line fires in two very different worlds:

* the loop really did stop running (the world the wording claims), or
* the loop is healthy and THIS peer stopped draining its socket, parking the lock in front of every
  later frame for that peer (kernel backpressure, up to ``_WS_SEND_DEADLINE_S``).

130 of these a day were read as proof the gateway was blind. The loop watchdog measures the loop
directly, so the line now reports the measurement instead of asserting a cause.
"""

from __future__ import annotations

import asyncio
import logging

from tui_gateway.ws import WSTransport

from hermes_cli import loop_watchdog


class _ParkedWS:
    """A peer that never drains: ``send_text`` parks (kernel backpressure), the loop is untouched."""

    def __init__(self) -> None:
        self.sends = 0

    async def send_text(self, line: str) -> None:
        self.sends += 1
        await asyncio.Event().wait()

    async def close(self, code: int = 1011) -> None:
        return None


def _run_parked_write(timeout_s: float, *, watchdog_interval: float | None):
    """Drive one ``write()`` from a pool thread against a parked peer; return the log lines."""

    async def scenario(logs: list[str]):
        loop = asyncio.get_running_loop()
        watchdog = None
        if watchdog_interval is not None:
            watchdog = loop_watchdog.install(
                loop, name="test", interval_s=watchdog_interval, stall_threshold_s=60.0,
                stack_dump_after_s=None, summary_every_s=None,
            )
        transport = WSTransport(_ParkedWS(), loop, peer="127.0.0.1:9")
        try:
            await loop.run_in_executor(None, transport.write, {"id": 1, "result": "x"})
        finally:
            if watchdog is not None:
                watchdog.stop()
        return list(logs)

    logs: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                logs.append(record.getMessage())

    handler = _Capture()
    logger = logging.getLogger("tui_gateway.ws")
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        asyncio.run(scenario(logs))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    return logs


def test_parked_peer_on_a_ticking_loop_is_not_reported_as_a_loop_stall(monkeypatch):
    """The regression: a healthy loop plus one non-draining peer must not read as a stall."""
    monkeypatch.setattr("tui_gateway.ws._WS_WRITE_TIMEOUT_S", 0.3, raising=False)

    logs = _run_parked_write(0.3, watchdog_interval=0.02)

    assert logs, "the write budget expired: the transport must say so"
    assert any("not a loop stall" in line for line in logs), logs
    assert any("loop ticked" in line for line in logs), logs
    assert not any("loop stalled" in line for line in logs), (
        f"the loop was ticking every 20 ms; it must not be accused of stalling: {logs}")


def test_a_blind_loop_is_reported_with_the_measured_silence(monkeypatch):
    """When the watchdog does say the loop is gone, the line carries its number."""
    monkeypatch.setattr("tui_gateway.ws._WS_WRITE_TIMEOUT_S", 0.05, raising=False)
    monkeypatch.setattr("tui_gateway.ws._loop_silence_s", lambda loop: 12.5)

    logs = _run_parked_write(0.05, watchdog_interval=None)

    assert logs, "the write budget expired: the transport must say so"
    assert any("loop stalled >0.05s, loop silent 12.5s" in line for line in logs), logs
    assert not any("not a loop stall" in line for line in logs), logs


def test_without_a_watchdog_the_line_says_unmeasured_rather_than_guessing(monkeypatch):
    """A process with no watchdog (a test harness, a bare server) keeps the legacy prefix and admits it."""
    monkeypatch.setattr("tui_gateway.ws._WS_WRITE_TIMEOUT_S", 0.05, raising=False)

    logs = _run_parked_write(0.05, watchdog_interval=None)

    assert any("loop stalled >0.05s (unmeasured)" in line for line in logs), logs
