"""Startup loop guards, separate from the gateway startup coordinator."""
import asyncio
import logging

from gateway.run_shutdown import _log_suppressed
from gateway.shutdown_watchdog import (
    DEFAULT_LOOP_WATCHDOG_INTERVAL_S, DEFAULT_LOOP_WATCHDOG_MAX_STRIKES,
    DEFAULT_LOOP_WATCHDOG_TIMEOUT_S,
)

logger = logging.getLogger("gateway.run")


def _arm_stall_reporter(self, loop: asyncio.AbstractEventLoop) -> None:
    """Report every loop stall with its duration, an attribution and a mid-stall stack dump.

    The gateway had no duration measurement at all: a stall only surfaced as a worker's
    ``ws write slow (loop stalled >10.0s)`` — a victim's 10 s write deadline, with no number for
    how long the loop was actually gone and no way to say what it was doing.
    """
    from hermes_cli.loop_watchdog import default_dump_path, install

    existing = getattr(self, "_loop_stall_watchdog", None)
    if existing is not None:
        return
    try:
        log_dir = getattr(getattr(self, "config", None), "log_dir", None)
        self._loop_stall_watchdog = install(
            loop, name="gateway", log=logger,
            dump_path=default_dump_path(log_dir),
        )
    except Exception:
        logger.debug("Failed to arm the gateway loop stall reporter", exc_info=True)


def start_loop_liveness_guards(self, loop: asyncio.AbstractEventLoop) -> None:
    """Arm the selector floor, the stall reporter and the out-of-loop hang watchdog before adapters.

    The stall reporter is armed unconditionally: it is the diagnostic (every stall with its duration,
    an attribution and a mid-stall stack dump), while ``gateway.loop_watchdog`` governs only the
    hard-exit backstop.

    See #69089.
    """
    from gateway.shutdown_watchdog import _arm_loop_floor_timer, start_loop_liveness_watchdog
    from gateway.run import _load_gateway_config
    from hermes_state_loop_guard import install_from_config
    install_from_config(_load_gateway_config())  # gateway.sessiondb_loop_guard (debug, default off)
    _arm_stall_reporter(self, loop)
    config = getattr(self, "config", None)
    if config is not None and not getattr(config, "loop_watchdog", True):
        return
    if getattr(self, "_loop_floor_timer_handle", None) is None:
        with _log_suppressed(logging.DEBUG, "Failed to arm gateway loop floor timer", exc_info=True):
            self._loop_floor_timer_handle = _arm_loop_floor_timer(loop)
    watchdog = getattr(self, "_loop_liveness_watchdog", None)
    if watchdog is None or not watchdog.is_alive():
        try:
            # getattr defaults cover config=None test paths; loaded values are already clamped.
            self._loop_liveness_watchdog = start_loop_liveness_watchdog(
                loop,
                probe_interval=float(getattr(
                    config, "loop_watchdog_probe_interval_s", DEFAULT_LOOP_WATCHDOG_INTERVAL_S
                )),
                probe_timeout=float(getattr(
                    config, "loop_watchdog_probe_timeout_s", DEFAULT_LOOP_WATCHDOG_TIMEOUT_S
                )),
                max_strikes=int(getattr(
                    config, "loop_watchdog_max_strikes", DEFAULT_LOOP_WATCHDOG_MAX_STRIKES
                )),
            )
        except Exception:
            logger.debug("Failed to start gateway loop liveness watchdog", exc_info=True)
