"""A full garbage collection must never be the thing that freezes the gateway.

2026-11-25, production: ``event loop stalled 58.3s ... attribution=gc idle; last gen2 58.3s 1s
ago`` — the loop was gone for the entire collection. These pin the two levers that fix it.
"""
import gc

import gateway.run_startup_liveness as liveness


def _capture(monkeypatch, *, collect=None, freeze=None, threshold=None):
    calls = []
    monkeypatch.setattr(liveness, "_GC_TUNED", False)
    monkeypatch.setattr(gc, "freeze", freeze if freeze is not None else (lambda: calls.append("freeze")),
                        raising=False)
    monkeypatch.setattr(gc, "collect",
                        collect or (lambda generation=2: calls.append(("collect", generation))))
    monkeypatch.setattr(gc, "set_threshold",
                        threshold or (lambda *args: calls.append(("threshold", args))))
    return calls


def test_freezes_the_startup_heap_after_settling_it(monkeypatch):
    calls = _capture(monkeypatch)

    liveness.tune_gc_for_long_lived_process()

    assert calls == [("collect", 2), "freeze", ("threshold", (5000, 20, 20))]


def test_tuning_is_idempotent(monkeypatch):
    calls = _capture(monkeypatch)

    liveness.tune_gc_for_long_lived_process()
    liveness.tune_gc_for_long_lived_process()

    assert calls.count("freeze") == 1


def test_missing_freeze_still_tunes_thresholds(monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.setattr(gc, "freeze", None, raising=False)

    liveness.tune_gc_for_long_lived_process()  # must not raise

    assert calls == [("collect", 2), ("threshold", (5000, 20, 20))]


def test_a_failing_collection_never_breaks_startup(monkeypatch):
    def boom(*_args, **_kwargs):
        raise MemoryError("no room")

    _capture(monkeypatch, collect=boom)

    liveness.tune_gc_for_long_lived_process()  # swallowed on purpose
