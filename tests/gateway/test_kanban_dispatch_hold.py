"""Dispatch hold: a restart or external drain must stop NEW kanban spawns while
in-flight workers finish, or the dispatcher refills every freed slot and the
idle wait never converges (observed: a 40-minute restart wait that never ended).
"""
from types import SimpleNamespace

import pytest

from gateway.kanban_watchers_dispatcher import _DispatcherSettings, _KanbanDispatcher


def _settings(max_spawn=None) -> _DispatcherSettings:
    return _DispatcherSettings(
        interval=60.0,
        max_spawn=max_spawn,
        max_in_progress=24,
        failure_limit=2,
        stale_timeout_seconds=0,
        reconcile_orphans=True,
        default_assignee=None,
        max_in_progress_per_profile=12,
    )


def _dispatcher(settings=None) -> _KanbanDispatcher:
    return _KanbanDispatcher(SimpleNamespace(), settings or _settings())


def test_hold_caps_spawns_without_dropping_the_tick():
    kwargs = _dispatcher().tick_kwargs(hold=True)
    # 0 is a live concurrency cap in dispatch_once: running_count >= 0 always holds,
    # so bookkeeping (reaping, stale timeouts, reclaims) still runs but nothing spawns.
    assert kwargs["max_spawn"] == 0
    assert kwargs["max_in_progress"] == 24


def test_no_hold_keeps_the_configured_cap():
    settings = _settings(max_spawn=8)
    kwargs = _dispatcher(settings).tick_kwargs(hold=False)
    assert kwargs["max_spawn"] == 8


def test_tick_once_passes_the_hold_to_every_board():
    d = _dispatcher()
    seen = []

    def fake_tick(slug, hold=False):
        seen.append((slug, hold))
        return "result"

    d._board_slugs = lambda: ["ps5-hv", "shark"]
    d.tick_once_for_board = fake_tick

    assert d.tick_once(True) == [("ps5-hv", "result"), ("shark", "result")]
    assert seen == [("ps5-hv", True), ("shark", True)]


@pytest.mark.asyncio
async def test_restart_request_engages_the_hold():
    from gateway import restart_when_idle

    runner = SimpleNamespace(
        _restart_task_started=False,
        _restart_requested=False,
        _restart_detached=False,
        _restart_via_service=False,
        _restart_force_event=None,
        _restart_task=None,
    )
    try:
        assert restart_when_idle.request_restart(runner, detached=False, via_service=False) is True
        assert runner._dispatch_hold["reason"] == "restart requested"
    finally:
        task = getattr(runner, "_restart_task", None)
        if task is not None:
            task.cancel()
