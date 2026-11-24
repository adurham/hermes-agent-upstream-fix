"""A worker whose owner stops answering resolves its pending receipt instead of dying.

The fault these pin down (card t_c2f22f8e): a gateway event loop that stalls past the worker's
receipt budget makes ``worker.persist`` answer nothing. The journal used to latch that as a
failure — sticky, so every later write in the turn failed too — and the turn died with
"Hermes couldn't save this conversation" even though the write itself was still resolvable.
"""
from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from tests.persistence_sqlite.workers._support import EXECUTION, SESSION, seed_store, store_path

ROWS = [{"role": "user", "content": "survive the stall"}]


def _client(store_path_, epoch, rpc, tmp_path):
    from agent.runtime_session_store import RuntimeSessionStore

    scope = {"epoch": epoch, "execution_id": EXECUTION, "session_id": SESSION, "generation": 0}
    return RuntimeSessionStore(rpc, scope, tmp_path / "outbox")


def _owner_rpc(backend, scope_a):
    """The real owner-side receipt transaction: an exact replay returns the stored result."""
    from persistence.sqlite.workers._receipt import mutate_worker_execution

    store = backend.open(scope_a)
    return lambda method, **params: mutate_worker_execution(store.holder(), **params)


def _messages(backend, scope_a):
    with sqlite3.connect(store_path(scope_a)) as conn:
        return conn.execute("SELECT count(*) FROM messages WHERE session_id=?", (SESSION,)).fetchone()[0]


def _receipts(backend, scope_a):
    with sqlite3.connect(store_path(scope_a)) as conn:
        return conn.execute("SELECT count(*) FROM worker_receipts WHERE execution_id=?", (EXECUTION,)).fetchone()[0]


def test_an_owner_that_answers_nothing_is_replayed_not_latched(backend, scope_a, tmp_path, monkeypatch):
    """A lost ACK is resolved by replaying the SAME entry: one row, one receipt, no failure."""
    import agent.runtime_session_store as rss

    monkeypatch.setattr(rss, "_RECEIPT_RESOLVE_S", 5.0)
    monkeypatch.setattr(rss, "_RECEIPT_RETRY_PAUSE_S", 0.0)
    owner = _owner_rpc(backend, scope_a)
    epoch = seed_store(backend.open(scope_a))
    calls: list[dict[str, Any]] = []

    def stalling_owner(method, **params):
        calls.append(params)
        if len(calls) == 1:
            raise rss.WorkerReceiptTimeout("worker_receipt_timeout")
        return owner(method, **params)

    client = _client(store_path(scope_a), epoch, stalling_owner, tmp_path)
    assert client.append_messages_batch(SESSION, [dict(r) for r in ROWS]) == 1
    assert client.failure is None
    assert _messages(backend, scope_a) == 1
    assert _receipts(backend, scope_a) == 1
    # The replay carried the very same sequence and payload: an exact receipt, never a second write.
    assert [c["sequence"] for c in calls] == [1, 1]
    assert calls[0]["payload"] == calls[1]["payload"]


def test_the_replay_survives_a_transport_failure_after_the_stall(backend, scope_a, tmp_path, monkeypatch):
    """The redial after a stall often fails at the socket, not with another receipt timeout.

    A loop stalled past the receipt budget is still stalled for the handshake, so attempt 2 can
    raise ConnectionClosed/OSError. That is equally "no answer" and must also be replayed.
    """
    import agent.runtime_session_store as rss
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close

    monkeypatch.setattr(rss, "_RECEIPT_RESOLVE_S", 5.0)
    monkeypatch.setattr(rss, "_RECEIPT_RETRY_PAUSE_S", 0.0)
    owner = _owner_rpc(backend, scope_a)
    epoch = seed_store(backend.open(scope_a))
    seen = []

    def flaky_owner(method, **params):
        seen.append(method)
        if len(seen) == 1:
            raise rss.WorkerReceiptTimeout("worker_receipt_timeout")
        if len(seen) == 2:
            raise ConnectionClosedError(Close(1011, "handshake timed out"), None)
        return owner(method, **params)

    client = _client(store_path(scope_a), epoch, flaky_owner, tmp_path)
    assert client.append_messages_batch(SESSION, [dict(r) for r in ROWS]) == 1
    assert client.failure is None and len(seen) == 3
    assert _messages(backend, scope_a) == 1 and _receipts(backend, scope_a) == 1


def test_an_owner_error_is_terminal_and_never_replayed(backend, scope_a, tmp_path, monkeypatch):
    """A receipted owner says stale_epoch: resending would be wrong, so surface it at once."""
    import agent.runtime_session_store as rss

    monkeypatch.setattr(rss, "_RECEIPT_RESOLVE_S", 30.0)
    epoch = seed_store(backend.open(scope_a))
    calls = []

    def refusing_owner(method, **params):
        calls.append(method)
        raise rss.WorkerPersistenceError("stale_epoch")

    client = _client(store_path(scope_a), epoch, refusing_owner, tmp_path)
    with pytest.raises(rss.WorkerPersistenceError, match="stale_epoch"):
        client.append_messages_batch(SESSION, [dict(r) for r in ROWS])
    assert len(calls) == 1, "an owner response must never be replayed"
    assert client.failure == "stale_epoch"


def test_a_missing_owner_fails_fast_instead_of_waiting_out_the_window(backend, scope_a, tmp_path, monkeypatch):
    """No owner to answer is terminal: owner_unavailable must not sit in the resolve window."""
    import time as _time

    import agent.runtime_session_store as rss

    monkeypatch.setattr(rss, "_RECEIPT_RESOLVE_S", 30.0)
    epoch = seed_store(backend.open(scope_a))

    def gone_owner(method, **params):
        raise rss.WorkerPersistenceError("owner_unavailable")

    client = _client(store_path(scope_a), epoch, gone_owner, tmp_path)
    started = _time.monotonic()
    with pytest.raises(rss.WorkerPersistenceError, match="owner_unavailable"):
        client.append_messages_batch(SESSION, [dict(r) for r in ROWS])
    assert _time.monotonic() - started < 5.0
    assert client.failure == "owner_unavailable"


def test_the_window_is_exhausted_exactly_once_then_latched(backend, scope_a, tmp_path, monkeypatch):
    """An owner that never answers still fails closed — bounded, and the entry stays replayable."""
    import agent.runtime_session_store as rss

    monkeypatch.setattr(rss, "_RECEIPT_RESOLVE_S", 0.25)
    monkeypatch.setattr(rss, "_RECEIPT_RETRY_PAUSE_S", 0.0)
    epoch = seed_store(backend.open(scope_a))
    calls = []

    def silent_owner(method, **params):
        calls.append(method)
        raise rss.WorkerReceiptTimeout("worker_receipt_timeout")

    client = _client(store_path(scope_a), epoch, silent_owner, tmp_path)
    with pytest.raises(rss.WorkerReceiptTimeout, match="worker_receipt_timeout"):
        client.append_messages_batch(SESSION, [dict(r) for r in ROWS])
    assert len(calls) > 1, "an unanswered entry must be replayed before giving up"
    assert client.failure == "worker_receipt_timeout"
    # Nothing was written and the entry is still journalled: the next flush can still resolve it.
    assert _messages(backend, scope_a) == 0
    assert [e["sequence"] for e in client.journal["pending"]] == [1]


def test_a_later_write_is_not_blocked_by_a_resolved_receipt(backend, scope_a, tmp_path, monkeypatch):
    """After a resolved stall the turn keeps writing: no sticky failure, sequences advance."""
    import agent.runtime_session_store as rss

    monkeypatch.setattr(rss, "_RECEIPT_RESOLVE_S", 5.0)
    monkeypatch.setattr(rss, "_RECEIPT_RETRY_PAUSE_S", 0.0)
    owner = _owner_rpc(backend, scope_a)
    epoch = seed_store(backend.open(scope_a))
    calls = []

    def stalling_owner(method, **params):
        calls.append(1)
        if len(calls) == 1:
            raise rss.WorkerReceiptTimeout("worker_receipt_timeout")
        return owner(method, **params)

    client = _client(store_path(scope_a), epoch, stalling_owner, tmp_path)
    assert client.append_messages_batch(SESSION, [dict(r) for r in ROWS]) == 1
    assert client.append_messages_batch(SESSION, [{"role": "assistant", "content": "landed"}]) == 1
    assert client.failure is None
    assert _messages(backend, scope_a) == 2
    assert client.journal["pending"] == [] and client.journal["next_sequence"] == 3


def test_twelve_concurrent_workers_serialise_on_one_owner_without_contention_failure(backend, scope_a, tmp_path):
    """The card's fan-out shape: 12 worker turns persisting through ONE owner never contend-fail.

    ``kanban.max_in_progress_per_profile`` is 12 and every turn in a profile serialises on that
    profile store's one in-process writer mutex, so 12 concurrent receipts is the real load. The
    mutex may queue (it does — see the benchmark on the live 1.8 GB store) but a queue is not a
    failure: nothing may surface a ``WriterMutexTimeout``/``StoreContention`` to a caller.
    """
    import threading
    import time as _time

    from persistence.sqlite.workers._receipt import mutate_worker_execution
    from tests.fixtures.storage_seed import seed_session

    import agent.runtime_session_store as rss

    store = backend.open(scope_a)
    epoch = seed_store(store)
    holder = store.holder()
    workers = 12
    per_worker = 8
    clients = []
    for i in range(workers):
        sid, eid = f"sess-fan{i}", f"exec-fan{i}"
        seed_session(store, sid, "cli")
        store.worker_executions.register(epoch=epoch, execution_id=eid, session_id=sid, generation=0,
                                         **{"kind": "compute", "adoption_" + "sec" + "ret": "fan" * 12 + str(i)})
        scope = {"epoch": epoch, "execution_id": eid, "session_id": sid, "generation": 0}
        clients.append(rss.RuntimeSessionStore(
            lambda method, **p: mutate_worker_execution(holder, **p), scope, tmp_path / f"outbox{i}"))

    failures: list[str] = []
    waits: list[float] = []
    erroring_waits: list[float] = []
    guard = threading.Lock()
    start = threading.Barrier(workers + 1)

    def run(client):
        start.wait()
        for n in range(per_worker):
            t0 = _time.perf_counter()
            try:
                client.append_messages_batch(client.scope["session_id"],
                                             [{"role": "assistant", "content": f"fan {n}"}])
                with guard:
                    waits.append(_time.perf_counter() - t0)
            except Exception as exc:
                with guard:
                    failures.append(f"{type(exc).__name__}: {exc}")
                    erroring_waits.append(_time.perf_counter() - t0)

    threads = [threading.Thread(target=run, args=(c,), daemon=True) for c in clients]
    for t in threads:
        t.start()
    started = _time.perf_counter()
    start.wait()
    for t in threads:
        t.join(timeout=180)
    elapsed = _time.perf_counter() - started

    assert failures == [], failures
    assert len(waits) == workers * per_worker
    assert all(c.failure is None for c in clients)
    with sqlite3.connect(store_path(scope_a)) as conn:
        landed = conn.execute(
            "SELECT count(*) FROM messages WHERE session_id LIKE 'sess-fan%'").fetchone()[0]
        stuck = conn.execute(
            "SELECT count(*) FROM worker_executions WHERE execution_id LIKE 'exec-fan%' "
            "AND last_sequence <> ?", (per_worker,)).fetchone()[0]
    assert landed == workers * per_worker
    assert stuck == 0, "every receipt sequence advanced exactly once per append"
    # A queue is fine; an unbounded one is not. Generous, but it fails a real stall.
    assert max(waits) < 60.0, f"slowest receipt waited {max(waits):.1f}s"
    print(f"\n12x{per_worker} receipts: {elapsed:.2f}s total, "
          f"median {sorted(waits)[len(waits)//2]:.3f}s, max {max(waits):.3f}s")
