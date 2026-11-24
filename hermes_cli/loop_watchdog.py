"""Measure event-loop stalls and name what the process was doing during them.

A stalled loop is not a slow socket: it means no callback ran at all, so every RPC the process
serves (worker persists, prompt receipts, kanban admissions, desktop streams) waits. Two things can
stop a loop: a synchronous call on the loop thread, or another thread holding the GIL — a full
``gc.collect()`` is the one a long-lived multi-GB gateway produces on its own (it walks the whole
live graph with the GIL held; the wall time then inflates roughly with the number of runnable
threads competing for that GIL).

The bare drift number is not actionable, so each report carries an attribution:
``gc=gen2 in flight 25.2s`` names a collection that was running when the loop stopped, while
``gc=idle`` rules it out and points at a blocking call instead. For the second case the stacks
matter, and by the time the loop breathes again they are gone — so the heartbeat also re-arms
``faulthandler``'s C watchdog on every tick. That dumps all thread stacks from a C timer thread,
with the GIL still held by whatever is blocking, i.e. mid-stall rather than after it.
"""
from __future__ import annotations

import asyncio
import faulthandler
import gc
import logging
import os
import threading
import time
from typing import Any, Optional

_log = logging.getLogger(__name__)

#: Heartbeat period. Any callback drift beyond it is what ``drift`` measures; 2 s matches the
#: desktop backend's long-standing tick so both processes report on the same grid.
DEFAULT_INTERVAL_S = 2.0

#: Drift above this is reported. Below it the loop is merely busy: a healthy gateway under load
#: drifts a few hundred ms per tick, and logging that would bury the real events.
DEFAULT_STALL_THRESHOLD_S = 5.0

#: Drift above which the stacks are captured as well. Kept above the threshold so a merely slow
#: tick does not cost a full-process stack dump.
DEFAULT_STACK_DUMP_AFTER_S = 8.0

#: Bounded: a stall storm must not grow the process's heap to record itself.
MAX_RECORDED_STALLS = 200


class GcActivity:
    """Track collections via ``gc.callbacks`` so a stall can name the one in flight.

    The callback runs on the collecting thread, once at start and once at stop, and must stay
    trivial: it is inside the collection it is timing.
    """

    def __init__(self, keep: int = 8) -> None:
        self._lock = threading.Lock()
        self._started: Optional[dict[str, Any]] = None
        self._recent: list[dict[str, Any]] = []
        self._keep = keep
        self._installed = False

    def install(self) -> None:
        if not self._installed:
            gc.callbacks.append(self._callback)
            self._installed = True

    def uninstall(self) -> None:
        if self._installed:
            try:
                gc.callbacks.remove(self._callback)
            except ValueError:
                pass
            self._installed = False

    def _callback(self, phase: str, info: dict[str, Any]) -> None:
        # Cheap by construction: no formatting, no logging (the log call would run under the GIL
        # during the collection and can itself block on a handler lock).
        now = time.monotonic()
        if phase == "start":
            with self._lock:
                self._started = {"generation": info.get("generation"), "at": now}
        else:
            with self._lock:
                started = self._started
                self._started = None
                if started is not None:
                    self._recent.append({
                        "generation": started["generation"],
                        "started_at": started["at"],
                        "duration_s": now - started["at"],
                        "collected": info.get("collected"),
                    })
                    del self._recent[:-self._keep]

    def in_flight(self) -> Optional[tuple[Any, float]]:
        """``(generation, elapsed_s)`` for the collection running right now, else ``None``."""
        with self._lock:
            started = self._started
        if started is None:
            return None
        return started["generation"], time.monotonic() - started["at"]

    def recent(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._recent)


def _generation_name(generation: Any) -> str:
    return {0: "gen0", 1: "gen1", 2: "gen2"}.get(generation, f"gen{generation}")


class LoopWatchdog:
    """Report every loop stall with its duration and an attribution; optionally capture stacks."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        name: str = "",
        interval_s: float = DEFAULT_INTERVAL_S,
        stall_threshold_s: float = DEFAULT_STALL_THRESHOLD_S,
        stack_dump_after_s: Optional[float] = DEFAULT_STACK_DUMP_AFTER_S,
        log: Optional[logging.Logger] = None,
        dump_path: Optional[str] = None,
    ) -> None:
        self._loop = loop
        self._name = name
        self._interval = max(0.05, float(interval_s))
        self._threshold = max(0.0, float(stall_threshold_s))
        self._stack_after = stack_dump_after_s
        self._log = log or _log
        self._dump_path = dump_path
        self._gc = GcActivity()
        self._stalls: list[dict[str, Any]] = []
        self._handle: Optional[asyncio.TimerHandle] = None
        self._dump_file: Any = None
        self._started = False
        self._armed_dump = False

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> LoopWatchdog:
        if self._started:
            return self
        self._started = True
        self._gc.install()
        now = self._loop.time()
        self._handle = self._loop.call_later(self._interval, self._tick, now + self._interval)
        self._log.info(
            "loop watchdog armed%s: tick=%.1fs stall_threshold=%.1fs stack_dump=%s",
            f" ({self._name})" if self._name else "", self._interval, self._threshold,
            f"after {self._stack_after:.1f}s" if self._stack_after else "off",
        )
        return self

    def stop(self) -> None:
        self._started = False
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        self._disarm_stack_dump()
        self._gc.uninstall()

    def stalls(self) -> list[dict[str, Any]]:
        """Recorded stalls, oldest first (bounded)."""
        return list(self._stalls)

    # -- heartbeat ---------------------------------------------------------------
    def _tick(self, expected: float) -> None:
        if not self._started:
            return
        now = self._loop.time()
        drift = now - expected
        if drift >= self._threshold:
            self._report(drift)
        # Re-arm the C-level dump: if this loop never reaches the next tick, faulthandler's own
        # timer fires from its C thread and the stacks are captured WHILE whatever holds the GIL
        # still holds it. Cancelling first keeps exactly one timer in flight.
        self._arm_stack_dump()
        self._handle = self._loop.call_later(self._interval, self._tick, now + self._interval)

    def _arm_stack_dump(self) -> None:
        if not self._stack_after:
            return
        try:
            if self._dump_file is None:
                self._dump_file = _open_dump_file(self._dump_path)
                if self._dump_file is None:
                    self._stack_after = None  # nothing to write to: do not retry every tick
                    return
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(
                self._stack_after, repeat=False, file=self._dump_file)
            self._armed_dump = True
        except (AttributeError, RuntimeError, ValueError, OSError):
            # faulthandler unavailable (or the fd was closed under us): the duration report is
            # still worth having, so degrade instead of failing the heartbeat.
            self._stack_after = None

    def _disarm_stack_dump(self) -> None:
        try:
            faulthandler.cancel_dump_traceback_later()
        except (AttributeError, RuntimeError, ValueError):
            pass
        self._armed_dump = False

    # -- reporting ---------------------------------------------------------------
    def _attribution(self, drift: float) -> str:
        in_flight = self._gc.in_flight()
        if in_flight is not None:
            generation, elapsed = in_flight
            return (
                f"attribution=gc {_generation_name(generation)} in flight for {elapsed:.1f}s"
                " (a full collection holds the GIL for its whole length)"
            )
        parts = ["attribution=gc idle"]
        recent = self._gc.recent()
        if recent:
            last = recent[-1]
            age = max(0.0, time.monotonic() - last["started_at"] - last["duration_s"])
            parts.append(
                f"last {_generation_name(last['generation'])} {last['duration_s']:.1f}s {age:.0f}s ago")
        parts.append(f"threads={threading.active_count()}")
        parts.append(
            "a blocking call on the loop is the other cause: see the stack dump"
            if self._stack_after else "no stack dump armed")
        return "; ".join(parts)

    def _report(self, drift: float) -> None:
        record = {
            "at": time.time(),
            "drift_s": drift,
            "attribution": self._attribution(drift),
            "name": self._name,
        }
        self._stalls.append(record)
        del self._stalls[:-MAX_RECORDED_STALLS]
        # Keep the long-standing wording: iso-certify.py and the ops greps key on it.
        self._log.warning(
            "event loop stalled %.1fs (GIL pressure suspected); %s%s",
            drift, record["attribution"],
            f"; stacks: {self._dump_path}" if self._dump_path and self._stack_after else "",
        )


def _open_dump_file(path: Optional[str]) -> Any:
    if not path:
        return None
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return open(path, "a", encoding="utf-8")  # windows-footgun: ok (append-only stack log)
    except OSError:
        return None


def default_dump_path(log_dir: Optional[str] = None) -> str:
    """Where mid-stall stacks go: beside the process's other logs when the caller knows them."""
    if not log_dir:
        try:
            from hermes_constants import get_default_hermes_root
            log_dir = os.path.join(str(get_default_hermes_root()), "logs")
        except Exception:
            log_dir = "."
    return os.path.join(log_dir, "loop_stall_stacks.log")


def install(
    loop: asyncio.AbstractEventLoop,
    *,
    name: str = "",
    interval_s: float = DEFAULT_INTERVAL_S,
    stall_threshold_s: float = DEFAULT_STALL_THRESHOLD_S,
    stack_dump_after_s: Optional[float] = DEFAULT_STACK_DUMP_AFTER_S,
    log: Optional[logging.Logger] = None,
    dump_path: Optional[str] = None,
) -> LoopWatchdog:
    """Install a watchdog on ``loop``. Idempotent per loop: a loop already carrying one keeps it."""
    existing = getattr(loop, "_hermes_loop_watchdog", None)
    if isinstance(existing, LoopWatchdog) and existing._started:
        return existing
    watchdog = LoopWatchdog(
        loop, name=name, interval_s=interval_s, stall_threshold_s=stall_threshold_s,
        stack_dump_after_s=stack_dump_after_s, log=log, dump_path=dump_path,
    ).start()
    try:
        loop._hermes_loop_watchdog = watchdog  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass
    return watchdog
