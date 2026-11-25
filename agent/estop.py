"""Global emergency stop (ESTOP) — a resumable pause for NEW work only.

``hermes pause`` writes a sentinel at ``$HERMES_HOME/ESTOP``; ``hermes resume``
removes it. While it exists the cron scheduler, kanban dispatcher and new gateway
turns skip work; in-flight work is never killed. The check is one or two uncached
``os.stat`` calls (process home + fleet root when they differ). The body is optional
JSON ``{"reason", "engaged_at"}``; a corrupt/empty file still counts as engaged
(fail safe, e.g. ``touch ~/.hermes/ESTOP``). Ported from gastownhall/gastown estop.go (MIT).

Two kinds of sentinel share the file, and only one of them is the boot's to lift:

* an **operator** pause (``hermes pause``, no ``restart_token``) belongs to the person who
  wrote it and only ``hermes resume`` ends it — never a boot, never a restart;
* a **restart** pause (``hermes pause --restart-token …``, written by
  ``hermes_safe_restart.sh``) means "start no new work until this gateway has restarted".
  Its own writer cannot always lift it: a restart of the unit the script runs inside
  SIGKILLs the script, so its ``trap cleanup EXIT`` never runs (measured: a
  ``safe restart of hermes-gateway`` sentinel stood 40 min past the boot it belonged to,
  t_d4e7d48c). The boot IS the restart, so :func:`clear_completed_restart` lifts it —
  only with proof of provenance (a token, and an ``engaged_at`` older than this boot).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import suppress
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Optional

# Same profile-aware / fleet-root resolvers the file-safety guards use (fail-open to ~/.hermes).
from agent.file_safety import _hermes_home_path as _hermes_home, _hermes_root_path as _canonical_root

SENTINEL_NAME = "ESTOP"

# Body key that marks a sentinel as written BY a restart (``hermes pause --restart-token``).
# Its presence — plus a parseable ``engaged_at`` older than this boot — is the ONLY thing that
# lets the boot path remove a sentinel. Nothing else in the body can authorize a delete.
RESTART_TOKEN_KEY = "restart_token"

# Per-component "logged already for this engagement" flags: log once per engagement, not per tick.
_log_lock = threading.Lock()
_logged_components: set[str] = set()


def sentinel_path() -> Path:
    """Path of the ESTOP sentinel this process would write on `hermes pause`."""
    return _hermes_home() / SENTINEL_NAME


def _candidate_sentinel_paths() -> list:
    """Profile home first, then the fleet root if it is a different directory: a profile
    gateway (HERMES_HOME=~/.hermes/profiles/<n>) must still honor an operator's ~/.hermes/ESTOP."""
    primary = sentinel_path()
    try:
        root = _canonical_root() / SENTINEL_NAME
    except Exception:
        return [primary]
    try:
        distinct = root.resolve() != primary.resolve()
    except Exception:
        # Non-Path test doubles fail .resolve(); plain equality still dedupes.
        distinct = root != primary
    return [primary, root] if distinct else [primary]


def is_engaged() -> bool:
    """True if ANY candidate sentinel exists; fail SAFE (True) on stat errors."""
    saw_stat_error = False
    for path in _candidate_sentinel_paths():
        try:
            if path.exists():
                return True
        except OSError:
            saw_stat_error = True
    return saw_stat_error


def engage(reason: Optional[str] = None, *, restart_token: Optional[str] = None) -> Path:
    """Create the ESTOP sentinel. Idempotent; re-engaging updates the file.

    *restart_token* marks the pause as written BY a restart: the boot that completes that
    restart lifts it (:func:`clear_completed_restart`). Omit it for an operator pause — an
    untagged sentinel is never lifted by anything but ``hermes resume``.
    """
    path = sentinel_path()
    payload = {"engaged_at": datetime.now(UTC).isoformat(), "reason": reason or None}
    if restart_token:
        payload[RESTART_TOKEN_KEY] = str(restart_token)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        with suppress(OSError):  # Best effort: an empty/partial sentinel still pauses (fail safe).
            path.touch(exist_ok=True)
    return path


def disengage() -> bool:
    """Remove every visible sentinel (process-local and fleet-root)."""
    lifted = False
    for path in _candidate_sentinel_paths():
        try:
            path.unlink()
            lifted = True
        except (OSError, AttributeError):
            continue
    return lifted


def engaged_sentinel() -> Optional[tuple[Path, dict]]:
    """``(path, {"reason", "engaged_at"})`` for the visible sentinel, or None when dispatching.

    The path is kept because a status surface must name the FILE it read: a profile sentinel
    (``<profile>/ESTOP``) and a fleet-root one (``~/.hermes/ESTOP``) pause different scopes, and
    "an ESTOP is engaged somewhere" is not actionable after the fact. An unreadable/corrupt body
    still reports engaged (fail safe, with both fields None), exactly as ``get_state`` does.
    """
    for path in _candidate_sentinel_paths():
        try:
            if not path.exists():
                continue
        except OSError:
            return path, {"reason": None, "engaged_at": None, "restart_token": None}
        except AttributeError:
            continue
        state = {"reason": None, "engaged_at": None, "restart_token": None}
        with suppress(OSError, ValueError, AttributeError):
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(raw, dict):
                state = {"reason": raw.get("reason") or None, "engaged_at": raw.get("engaged_at") or None,
                         RESTART_TOKEN_KEY: raw.get(RESTART_TOKEN_KEY) or None}
        return path, state
    return None


def get_state() -> Optional[dict]:
    """Return ``{"reason", "engaged_at"}`` or None when not engaged; an unreadable/corrupt
    body still reports engaged with both fields None."""
    found = engaged_sentinel()
    return None if found is None else found[1]


def paused_reply() -> Optional[str]:
    """Short user-facing notice for new gateway turns, or None if not paused."""
    state = get_state()
    if state is None:
        return None
    tag = f" ({state['reason']})" if state.get("reason") else ""
    return f"⏸️ Hermes is paused{tag}. New work is on hold; run `hermes resume` to pick things back up."


def check_paused(component: str, logger: logging.Logger) -> bool:
    """Return True when engaged, logging once per engagement per component (re-armed after a resume)."""
    if not is_engaged():
        with _log_lock:
            _logged_components.discard(component)
        return False
    with _log_lock:
        first = component not in _logged_components
        _logged_components.add(component)
    if first:
        reason = (get_state() or {}).get("reason")
        suffix = f" (reason: {reason})" if reason else ""
        logger.info(
            "%s dispatch paused by global emergency stop%s — remove with `hermes resume` (%s)",
            component, suffix, sentinel_path(),
        )
    return True


# ── lifting a sentinel a RESTART wrote (never an operator's) ──────────────────────────────
#
# ``hermes_safe_restart.sh`` pauses new work, watches the in-flight set drain, restarts the
# unit, then resumes. When it runs from a process inside that unit's cgroup (an agent turn, a
# kanban worker) the restart SIGKILLs the script — so its ``trap cleanup EXIT`` never runs and
# the ``hermes resume`` half of write→restart→remove never happens. The sentinel then outlives
# the restart it belongs to and silently freezes the lane (measured, t_d4e7d48c): 40 minutes of
# no cron dispatch, six frozen clocks, a refused deploy gate, 13 interrupted admissions.
#
# The boot that the restart ends in IS the completion of the intent "start no new work until
# this gateway has restarted", so the boot lifts the sentinel. The rule is deliberately one-way
# and needs proof: a body that parses, a ``restart_token``, and an ``engaged_at`` older than
# this boot. An operator's ``hermes pause`` (no token) is untouched, a torn/empty/hand-touched
# body proves nothing and is untouched, and a sentinel written AFTER this boot belongs to a
# later restart and is untouched.

def _sentinel_homes() -> list:
    """Every home whose sentinel a boot here can own: this process's, the fleet root, and each
    profile home (one gateway multiplexes every profile's cron store, so a profile's sentinel has
    no boot of its own — the host gateway's boot is the only boot it will ever get).

    The fleet is included only when THIS home is the fleet: the root itself or one of its
    ``profiles/<name>`` homes. A gateway booted in a scratch/dev home (any ``HERMES_HOME`` that
    is neither) owns only that home — otherwise a throwaway boot could lift a live restart pause
    another process is still draining. Removal is the one direction where widening the scope is
    unsafe, so an unproven home owns nothing else.
    """
    from agent.file_safety import _hermes_dirs, _profile_homes

    dirs = list(_hermes_dirs())
    if not dirs:
        return []
    active, root = dirs[0], dirs[-1]
    owns_fleet = active == root or (active.parent.name == "profiles" and active.parent.parent == root)
    homes = [active]
    if owns_fleet:
        homes.append(root)
        with suppress(Exception):
            homes += [p for p in _profile_homes() if not p.name.startswith(".")]
    seen: set = set()
    unique = []
    for home in homes:
        key = str(home)
        if key not in seen:
            seen.add(key)
            unique.append(home)
    return unique


def _body_at(path: Path) -> Optional[dict]:
    """STRICT body read: ``None`` for missing, unreadable, unparseable or non-dict bodies.

    Deliberately not the fail-safe read ``engaged_sentinel`` uses. Reading fail-safe is right for
    "is it engaged"; removal needs proof of provenance, and a torn body proves nothing.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return raw if isinstance(raw, dict) else None


def _epoch_of(value) -> Optional[float]:
    """Epoch seconds for an ISO ``engaged_at``, or None when it is absent/unparseable.

    A naive timestamp is read as UTC (every writer here stamps ``datetime.now(UTC)``) rather
    than as local time, which would shift it by hours and could clear a sentinel too early.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def clear_completed_restart(*, boot_started_at: Optional[float] = None, homes=None) -> list:
    """Lift the sentinels a restart wrote, because THIS boot completed those restarts.

    Returns one record per removed sentinel (``path``, ``reason``, ``engaged_at``,
    ``restart_token``); removal is best-effort and a path that cannot be unlinked is skipped, so
    a read-only home can never make a boot fail. ``homes`` defaults to :func:`_sentinel_homes`
    and exists so a test can point the sweep at throwaway directories instead of real ones.
    """
    started = time.time() if boot_started_at is None else float(boot_started_at)
    cleared = []
    for home in (_sentinel_homes() if homes is None else list(homes)):
        path = Path(home) / SENTINEL_NAME
        body = _body_at(path)
        if not body:
            continue
        token = body.get(RESTART_TOKEN_KEY)
        if not token:
            continue
        engaged_at = body.get("engaged_at")
        engaged_epoch = _epoch_of(engaged_at)
        if engaged_epoch is None or engaged_epoch > started:
            continue
        try:
            path.unlink()
        except OSError:
            continue
        cleared.append({"path": str(path), "reason": body.get("reason") or None,
                        "engaged_at": engaged_at, "restart_token": str(token)})
    return cleared


def release_restart(restart_token: str) -> Optional[Path]:
    """Lift the sentinel carrying exactly *restart_token*; ``None`` when none does.

    This is the half of ``hermes resume`` a restart script may safely run: it removes only the
    sentinel that script wrote. A later operator ``hermes pause`` that overwrote the body loses
    the token, so a stale cleanup can never lift a pause a person asked for.
    """
    if not restart_token:
        return None
    wanted = str(restart_token)
    for path in _candidate_sentinel_paths():
        body = _body_at(path)
        if not body or body.get(RESTART_TOKEN_KEY) != wanted:
            continue
        try:
            path.unlink()
        except (OSError, AttributeError):
            continue
        return path
    return None
