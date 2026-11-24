"""Dispatcher subprocess: submit one claim to its profile owner; never run an agent."""
# Spawned as a bare ``sys.executable -m``: in a PM install that interpreter carries no dependencies
# until hermes_bootstrap selects the committed environment, so it must be the first import.
try:
    import hermes_bootstrap
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap" and not "hermes_bootstrap".startswith(f"{exc.name}."):
        raise

import asyncio
from contextlib import closing
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Optional

# A worker that loses its owner (a dropped socket, a restart, `hermes update` draining it or a new
# owner still starting) waits for it instead of exiting: the condition is the host's, not the card's,
# and a failed exit here spends the card's failure budget and lets the owner end the turn. Asking
# again is safe: ``kanban.run`` is idempotent per attempt and ``prompt.receipt`` is a read.
# ``OWNER_WAIT_S`` is the longest the owner may stay silent between answers before the worker
# gives up (the first ``kanban.run`` answer has its own wait, ``admission_answer_budget``).
OWNER_WAIT_S = 180.0
# A PLANNED restart (a ``hermes update`` drain, ``hermes gateway restart``) is not a silent owner,
# so it is not failed on ``OWNER_WAIT_S``. The gateway's after-turn wait counts the card's managed
# turn as active work and defers ``stop()`` for it (measured 2026-11-17: "Restart requested with 7
# active work unit(s)" naming seven ``kind: managed`` units), but while it waits it refuses every
# client read, so this worker cannot see its receipt until the NEXT owner answers. Giving up at
# OWNER_WAIT_S is what amputated every in-flight card attempt that day — 13 attempts in three
# drains, each dying 181-188 s after the cut — and the exit is what ends the very turn the drain
# was deferring ``stop()`` for (``_watch_submitter``). A restart this worker can SEE
# (``draining``/``restart_wait``) is therefore waited out on the restart's own timescale: the
# after-turn cap ``agent.restart_after_turn_timeout`` (default 1800 s, the bound ``hermes update``
# itself waits) plus the replacement owner's stop/start/admission (~60 s measured).
RESTART_WAIT_BUDGET_S = 2400.0
_TRANSIENT_OWNER_STATES = frozenset({'draining', 'starting'})
# In-band owner refusals that leave the attempt intact: a draining owner admits nothing new, and a
# retired authority's successor holds the admission the next connection reads.
_TRANSIENT_RPC_REASONS = frozenset({'runtime_draining', 'runtime_retired'})
# Control-socket answers a restarting owner gives (an ``inaccessible`` owner's reason, from
# discovery or the ticket fetch): a reset or half-written answer, or a replaced owner refusing a
# stale instance. Unsafe nodes or peers, a refused authorization, a bad pointer or a failed start
# stay final.
_RESTART_CONTROL_REASONS = frozenset({'invalid_control_peer', 'incomplete_control_response',
                                      'invalid_control_response'})
# Refusals a planned restart cannot explain away: they name this profile's own configuration or
# trust, which the replacement owner will give the same answer to. Everything else an
# ``inaccessible`` owner says *while a restart it announced is in flight* is retried on the
# restart's budget instead of ending the attempt (measured 2026-11-17:
# ``Gateway inaccessible: service_identity_unverified`` killed this card's first attempt 40 s into
# an update's down window, while the unit was being rewritten).
_FINAL_OWNER_REASONS = frozenset({'profile_mismatch', 'authorization', 'unsafe_control_permissions',
                                  'runtime_protocol', 'session_authority_unavailable',
                                  'invalid_runtime_state', 'invalid_runtime_identity',
                                  'invalid_api_origin'})
# Past the board lock wait: the owner's session setup and receipt before and after the write.
ADMISSION_ANSWER_MARGIN_S = 60.0

# An owner that answers ``unknown`` has FENCED the receipt: its settlement write could not commit
# (a held SQLite writer, a full disk), so the attempt can never be replayed. That is the owner's
# answer about the ADMISSION, not about the card: an unattended card attempt is settled
# ``interrupted`` by the owner's own drain right afterwards (``settle_unattended_unknown``), and
# the attempt's managed interpreter normally records its ``worker_result`` before any of this.
# Measured 2026-11-24: 31 of 90 runs crashed with this refusal, every inspected one after its
# ``worker_result`` was already on the board — the card had run; only the receipt was unknown.
# So an ``unknown`` receipt is not final: the attempt's own result (which ends the question), else
# a bounded wait for the drain's resolution, decides first. 120 s covers a stalled owner loop
# (measured 10 s+ event-loop stalls) without holding a card's slot for a claim TTL.
UNKNOWN_RESOLUTION_GRACE_S = 120.0


def _unknown_grace_s() -> float:
    """``HERMES_KANBAN_UNKNOWN_GRACE_SECONDS`` (operator knob, 0 = never wait), else the default."""
    raw = (os.environ.get('HERMES_KANBAN_UNKNOWN_GRACE_SECONDS') or '').strip()
    if not raw:
        return UNKNOWN_RESOLUTION_GRACE_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return UNKNOWN_RESOLUTION_GRACE_S


def _attempt_result(board_db, params):
    """This attempt's own ``worker_result`` event, or ``{}``.

    The managed interpreter's finalizer writes it BEFORE the owner settles the admission, so it is
    the strongest statement the board holds that this exact attempt's turn ended — under this run
    id and this claim lock, never a neighbour attempt's (``worker_result`` enforces both). When the
    owner then fails to commit its settlement and fences the receipt ``unknown``, refusing the card
    on that fence alone libels an attempt that ran: its reply is right here.
    """
    try:
        from gateway.session_kanban import worker_result
        return worker_result(board_db, params) or {}
    except (OSError, ValueError, sqlite3.Error):
        return {}


def _task_boards(task_id: str, db_path: str, board: Optional[str]) -> list[tuple[str, str]]:
    """``[(slug, path)]`` of every board whose OWN ``kanban.db`` holds this card.

    A refusal that names only ``resolved_board`` cannot be diagnosed when the card lives in
    another board's file — the exact cross-lane shape that killed t_a6bccd8c / t_a43541ef. The
    resolved board is probed first (the normal case: one hit), then every sibling board, then the
    root ``default`` file, all read-only.
    """
    try:
        from hermes_cli import kanban_db as kb
        root, default_db = kb.boards_root(), kb.kanban_home() / 'kanban.db'
    except Exception:
        root, default_db = None, None
    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    def probe(slug: str, path) -> None:
        try:
            db_file = str(Path(path).resolve())
        except OSError:
            return
        if db_file in seen or not Path(db_file).is_file():
            return
        seen.add(db_file)
        try:
            with closing(sqlite3.connect(Path(db_file).as_uri() + '?mode=ro', uri=True)) as conn:
                if conn.execute('SELECT 1 FROM tasks WHERE id = ?', (task_id,)).fetchone():
                    found.append((slug, db_file))
        except sqlite3.Error:
            return  # unreadable is not "holds the card"

    probe(board or 'default', db_path)
    if root is not None:
        try:
            for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                if child.is_dir() and (child / 'kanban.db').is_file():
                    probe(child.name, child / 'kanban.db')
        except OSError:
            pass
    if default_db is not None:
        probe('default', default_db)
    return found


def _refusal_record(params, board_db, claim) -> dict:
    """Everything a reader needs to place a refused dispatch: boards, DB, claim, admission."""
    try:
        db_path = str(Path(board_db).resolve())
    except OSError:
        db_path = str(board_db)
    boards = _task_boards(params['task_id'], db_path, params.get('board'))
    return {
        'reason': 'owner answered unknown: the attempt settlement could not be committed',
        'resolved_board': params.get('board'), 'card_board': [slug for slug, _ in boards] or None,
        'card_board_path': boards[0][1] if boards else None, 'db_path': db_path,
        'claim_id': params.get('claim_lock'), 'run_id': params.get('run_id'),
        'task_id': params.get('task_id'), 'session_id': getattr(claim, 'session_id', None),
        'admission_id': getattr(claim, 'admission_id', None), 'pid': os.getpid(),
    }


def _refusal_sentence(record: dict) -> str:
    """The one line the worker prints, keeping the old prefix a grep already knows."""
    card = ','.join(record['card_board'] or ['unknown'])
    return ('Kanban execution unknown; retained by owner '
            f"[resolved_board={record['resolved_board']}; card_board={card}; db_path={record['db_path']}; "
            f"claim_id={record['claim_id']}; run_id={record['run_id']}; session_id={record['session_id']}; "
            f"admission_id={record['admission_id']}]")


def _record_refusal(record: dict) -> None:
    """Write the refusal onto the card itself, so it outlives this process's stderr.

    The dispatcher folds only the worker's LAST OUTPUT into the crashed event; the refusal used to
    name no board, DB or claim, which made this whole class guesswork. Written to the file that
    actually HOLDS the card (``card_board_path``) when the resolved board is a different one —
    an event for a task id the resolved DB has never heard of would be a row nobody can join.
    Best-effort: a refusal must still exit when the board cannot be written (that is often why we
    are here).
    """
    try:
        from hermes_cli.kanban_db_connect import connect_closing
        from hermes_cli import kanban_db as kb
        db_path = record.get('card_board_path') or record['db_path']
        with connect_closing(Path(db_path)) as conn, kb.write_txn(conn):
            kb._append_event(conn, record['task_id'], 'dispatch_refused', record, run_id=record['run_id'])
    except Exception:
        pass


def admission_answer_budget():
    """Seconds to wait for ``kanban.run``'s answer: the longest the owner's admission write can
    wait on the board lock (``write_txn`` retries ``BEGIN IMMEDIATE`` ``_BUSY_MAX_RETRIES`` times,
    each under the busy timeout) plus ``ADMISSION_ANSWER_MARGIN_S``."""
    from hermes_cli import kanban_db_connect
    lock_wait = (kanban_db_connect._BUSY_MAX_RETRIES + 1) * kanban_db_connect._resolve_busy_timeout_ms() / 1000
    return lock_wait + ADMISSION_ANSWER_MARGIN_S


@dataclass
class _Claim:
    """What the owner has said about this attempt, carried across connections."""
    answered: float  # monotonic time of the owner's last answer
    session_id: Optional[str] = None
    admission_id: Optional[str] = None

    def silent_for(self):
        return time.monotonic() - self.answered


async def run(params, board_db):
    from agent.retry_utils import jittered_backoff
    from hermes_cli.gateway_client import connect_gateway, GatewayClientError, outside_restart_wait
    from hermes_cli.gateway_owner_restart import restartable
    from websockets.exceptions import WebSocketException
    if os.environ.get('HERMES_TUI_GATEWAY_URL'):
        raise GatewayClientError('Kanban requires the assigned local profile owner')
    # A lost socket is not a lost attempt, and exiting here would let ``_watch_submitter`` end the
    # owner's in-flight turn, so the next connection resumes whatever the last one reached. The
    # wait is bounded from the owner's last answer: an owner that keeps answering keeps the
    # worker, and one that stays silent for ``OWNER_WAIT_S`` fails the worker honestly — unless it
    # said it is RESTARTING, which is the host's business and gets ``RESTART_WAIT_BUDGET_S``.
    claim = _Claim(answered=time.monotonic())
    restart_at = None
    failures = 0
    while True:
        seen = claim.answered
        try:
            # Never into a restart's after-turn wait: it cut this socket so the turn's submitter
            # waits for the NEXT owner, and a worker polling its receipt inside the wait keeps the
            # turn it waits on alive, holding `hermes update` until restart_after_turn_timeout.
            with outside_restart_wait():
                async with connect_gateway() as client:
                    # kanban.run subscribes this socket to the card's session, but only receipts
                    # matter here: discard the turn's events, or they fill the client's bounded
                    # queue and its reader stops, with the next receipt behind it.
                    drain = asyncio.create_task(_discard(client.events))
                    try:
                        receipt = await _await_receipt(client, params, board_db, claim)
                    finally:
                        drain.cancel()
        # A restart also surfaces as a refused or reset dial, a failed handshake, or an answer
        # that never came.
        except (GatewayClientError, WebSocketException, OSError, TimeoutError) as exc:
            if claim.answered != seen:
                # The owner answered during that connection, so a restart this worker had seen is
                # over: a silence that follows an answer is measured from the answer again.
                restart_at = None
            if restart_at is None and (_planned_restart(exc) or _planned_restart_in_flight()):
                restart_at = time.monotonic()
            if not _restartable(exc, planned_restart=restart_at is not None):
                raise
            budget = RESTART_WAIT_BUDGET_S if restart_at is not None else OWNER_WAIT_S
            waited = time.monotonic() - (claim.answered if restart_at is None else restart_at)
            if waited >= budget:
                # WebSocket errors can embed remote bodies; only our own messages are quoted.
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                detail = exc if isinstance(exc, GatewayClientError) else type(exc).__name__ + (
                    f' (HTTP {status})' if status else '')
                restarted = ' after a planned gateway restart' if restart_at is not None else ''
                raise GatewayClientError(
                    f'Kanban owner silent for {claim.silent_for():.0f}s: {detail}{restarted}') from exc
        else:
            return _exit_code(receipt, params, board_db, planned_restart=restart_at is not None, claim=claim)
        failures = 1 if claim.answered != seen else failures + 1
        # Jittered (1-2 s first, up to 16 s): workers of one dead owner do not all redial, and so
        # start a replacement, in the same instant.
        await asyncio.sleep(jittered_backoff(failures, base_delay=1.0, max_delay=8.0, jitter_ratio=1.0))


def _planned_restart(exc):
    """True when *exc* is the owner's own "I am restarting" answer, not a silence.

    ``connect_gateway`` inside ``outside_restart_wait()`` refuses an owner in its restart's
    after-turn wait with exactly this verdict (``draining`` + ``restart_wait``); it is the one
    signal that tells a planned restart a worker must wait out from a wedge it must fail on.
    """
    from hermes_cli.gateway_client import GatewayUnavailableError
    from gateway.drain_control import RESTART_WAIT_REASON
    return (isinstance(exc, GatewayUnavailableError) and exc.state == 'draining'
            and exc.reason_code == RESTART_WAIT_REASON)


def _planned_restart_in_flight() -> bool:
    """True while a gateway restart was REQUESTED recently, from the marker the host stamps.

    Not every dial into a planned restart is answered with ``restart_wait``: once ``stop()`` has
    begun the old owner is gone, and discovery refuses with ``service_identity_unverified`` while
    the unit is rewritten and the replacement fails its identity check — a verdict indistinguishable
    from a broken install, which is how an attempt of this very card died 40 s into an update's
    down window (2026-11-17, t_39446095). ``.restart_pending.json`` is stamped when the restart is
    ACCEPTED (``request_restart``), before the after-turn wait, and consumed by the next boot, so
    a fresh marker is exactly "a planned restart is in flight".

    Both homes are read because the marker lives where the SERVING gateway runs: a profile served
    by the host multiplexer keeps it in the host root, a standalone profile gateway in its own home
    (the process home, which is what ``planned_restart_marker_path()`` returns here).
    """
    import time
    from gateway.restart_notice import planned_restart_marker_path
    from hermes_constants import get_default_hermes_root
    now = time.time()
    for home in (None, get_default_hermes_root()):
        try:
            payload = json.loads(planned_restart_marker_path(home).read_text(encoding='utf-8-sig'))
            requested_at = float(payload.get('requested_at') or 0)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        if 0 <= now - requested_at <= RESTART_WAIT_BUDGET_S:
            return True
    return False


def _restartable(exc, *, planned_restart=False):
    """False when *exc* is the owner's own final answer; True for a restarting owner or a lost transport.

    With *planned_restart* (a restart this worker already saw announced) an ``inaccessible``
    owner's refusal is the restart's too, unless it names this profile's own configuration:
    between ``stop()`` and the replacement owner's first answer the old unit is being rewritten, so
    identity/handshake refusals are expected and must not end the attempt.
    """
    from hermes_cli.gateway_client import GatewayRPCError, GatewayUnavailableError
    if isinstance(exc, GatewayRPCError):
        return str(exc) in _TRANSIENT_RPC_REASONS
    if isinstance(exc, GatewayUnavailableError):
        if exc.state in _TRANSIENT_OWNER_STATES or exc.reason_code in _RESTART_CONTROL_REASONS:
            return True
        return planned_restart and exc.reason_code not in _FINAL_OWNER_REASONS
    return True


async def _discard(events):
    while True:
        await events.get()


async def _await_receipt(client, params, board_db, claim):
    """Submit the claim (or, on a later connection, re-read the admission it got) and return its
    settled receipt, recording each owner answer on *claim*."""
    if claim.admission_id is None:
        db = str(await asyncio.to_thread(Path(board_db).resolve))
        # The owner answers only after it checks the claim and writes its ``owner_admitted`` marker
        # in one board write transaction. The board (the cross-profile dispatch bus) can make that
        # wait out its busy timeout, so the 30 s default left the owner admitting an attempt whose
        # submitter had exited (the later-bound worker's submitter-liveness watcher ended it, and the
        # admission was left ``unknown``). The wait is still finite: once the marker commits,
        # ``enforce_max_runtime`` and TTL reclaim skip the run (``owner_reclaim_paused``), so an owner
        # that stalls after admitting is bounded by this wait and ``OWNER_WAIT_S``.
        accepted = await client.rpc('kanban.run', _timeout=admission_answer_budget(), **dict(params, db=db))
        receipt = accepted['receipt']
        claim.session_id, claim.admission_id = accepted['session_id'], receipt['admission_id']
        print(f'Session: {claim.session_id}', file=sys.stderr)
    else:
        receipt = await _receipt(client, claim)
    claim.answered = time.monotonic()
    # ``unknown`` is the owner's fence around an admission it could not settle — never a replay, but
    # also not yet an answer about the CARD: its drain settles an unattended (kanban) admission
    # interrupted right afterwards, and the attempt's own ``worker_result`` outranks both. Wait for
    # whichever comes first, bounded by ``_unknown_grace_s()``.
    grace = _unknown_grace_s()
    fenced_at = None
    while True:
        if receipt['status'] == 'terminal':
            return receipt
        if receipt['status'] == 'unknown':
            if _attempt_result(board_db, params):
                return receipt
            fenced_at = fenced_at or time.monotonic()
            if grace <= 0 or time.monotonic() - fenced_at >= grace:
                return receipt
        await asyncio.sleep(.2)
        receipt = await _receipt(client, claim)
        claim.answered = time.monotonic()


def _exit_code(receipt, params, board_db, *, planned_restart=False, claim=None):
    from hermes_cli.gateway_client import GatewayClientError
    result = _attempt_result(board_db, params)
    if receipt['status'] == 'unknown' and not result:
        # Nothing on the board says this attempt's turn ended, and the owner fenced the receipt:
        # refuse, but say WHERE we looked and WHAT we held. Best-effort board record first, so the
        # refusal survives this process's stderr.
        record = _refusal_record(params, board_db, claim)
        _record_refusal(record)
        raise GatewayClientError(_refusal_sentence(record))
    # The attempt's reply ends the log, as a one-shot worker's did: crash diagnostics quote it.
    if result.get('last_output'):
        print(result['last_output'])
    code = result.get('exit_code', 1)
    if planned_restart and not result:
        # No ``worker_result`` event for this run: the owner never settled the turn, which is what
        # a stop() that outran the drain looks like. Say so, in the worker's last output, so the
        # board's host-failure verdict (and the next reader) does not charge the card for the
        # host's restart. A turn that settled — even with a nonzero code of its own — keeps its
        # own result and is NOT named here: a restart that merely overlapped a card's real failure
        # must not excuse it.
        print('Kanban attempt ended by a planned gateway restart: the owner drained for a restart '
              'and the replacement owner did not resume the turn.')
    return code


async def _receipt(client, claim):
    return await client.rpc('prompt.receipt', session_id=claim.session_id, admission_id=claim.admission_id)


def main():
    # The dead-worker sweep books this process's exit from the trailer in its log (it never
    # reaps us), so every exit path writes it — the same contract as a quiet one-shot worker.
    from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER
    rc = 1
    try:
        rc = _submit()
    finally:
        print(f'\n{KANBAN_WORKER_EXIT_TRAILER}{int(rc)}', file=sys.stderr, flush=True)
    return rc


def _submit():
    try:
        params = {'board': os.environ['HERMES_KANBAN_BOARD'], 'task_id': os.environ['HERMES_KANBAN_TASK'],
            'run_id': int(os.environ['HERMES_KANBAN_RUN_ID']), 'claim_lock': os.environ['HERMES_KANBAN_CLAIM_LOCK']}
        board_db = os.environ['HERMES_KANBAN_DB']
        # Owner startup must never inherit this worker's task/tool/goal identity.
        for key in list(os.environ):
            # The busy timeout and the unknown-receipt grace are host tuning knobs, not task
            # identity: the answer budget and ``_unknown_grace_s`` read them from this env.
            if key.startswith('HERMES_KANBAN_') and key not in (
                    'HERMES_KANBAN_HOME', 'HERMES_KANBAN_BUSY_TIMEOUT_MS',
                    'HERMES_KANBAN_UNKNOWN_GRACE_SECONDS'):
                os.environ.pop(key)
        for key in ('HERMES_SESSION_SOURCE', 'HERMES_TENANT', 'TERMINAL_CWD'):
            os.environ.pop(key, None)
        return asyncio.run(run(params, board_db))
    except (KeyError, ValueError, OSError) as exc:
        print(f'Kanban owner refused: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
