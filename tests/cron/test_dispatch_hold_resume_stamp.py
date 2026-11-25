"""`dispatch_held.json` is a LIVE state: a hold that ended must read as ended.

Measured 2026-11-24 on the ps5 lane: the marker held `{since: 21:44:52, last: 21:44:52}` for 34
minutes while the ticker heartbeated every minute and every ps5-pm job fired on its normal
interval. The lane read the frozen `last` as "cron dispatch is held" and reddened six pins and one
cut. The marker used to be written only while a hold lasted and never touched again, so a RESUMED
hold was indistinguishable from a stuck one.

The rule now: the first dispatching tick stamps `resumed`, and every reader asks `hold_active()`
instead of inferring a hold from the file's presence or the age of `last`.
"""

import json
import time

import pytest

from cron import dispatch_hold
from cron.dispatch_hold import TICK_SLACK_SECONDS, grace_starts_at, hold_active, note_dispatching, note_held, read

STORE = None


@pytest.fixture(autouse=True)
def cron_store(tmp_path, monkeypatch):
    """The active cron store the hold functions resolve (same shape as the drain-hold tests)."""
    global STORE
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    (tmp_path / "cron").mkdir()
    dispatch_hold._hold_since.clear()
    STORE = tmp_path / "cron"
    yield STORE
    dispatch_hold._hold_since.clear()


def test_a_resumed_hold_says_so_instead_of_freezing_at_its_last_tick(cron_store):
    t0 = time.time()
    note_held(now=t0 - 300)
    note_held(now=t0 - 240)
    assert hold_active(cron_store, now=t0 - 200) is True

    note_dispatching(now=t0)
    held = read(cron_store)
    assert held["since"] == pytest.approx(t0 - 300)   # the hold's real start is preserved
    assert held["last"] == pytest.approx(t0 - 240)    # last HELD tick
    assert held["resumed"] == pytest.approx(t0)       # and when dispatch came back
    assert hold_active(cron_store, now=t0) is False
    assert hold_active(cron_store, now=t0 + 6 * 3600) is False


def test_the_marker_is_stamped_once_and_not_rewritten_every_tick(cron_store):
    t0 = time.time()
    note_held(now=t0 - 60)
    note_dispatching(now=t0)
    first = (cron_store / "dispatch_held.json").read_text()
    note_dispatching(now=t0 + 60)
    note_dispatching(now=t0 + 120)
    assert (cron_store / "dispatch_held.json").read_text() == first


def test_the_ps5_artefact_a_frozen_marker_is_not_a_live_hold(cron_store):
    """The exact shape the lane read: `last` 34 min old, no `resumed` (written before this fix)."""
    t0 = time.time()
    (cron_store / "dispatch_held.json").write_text(
        json.dumps({"since": t0 - 2000, "last": t0 - 1999}), encoding="utf-8")
    assert read(cron_store)["resumed"] is None
    assert hold_active(cron_store, now=t0) is False          # stale: the ticker came back
    note_dispatching(now=t0)                                  # ... and the next tick stamps it
    assert read(cron_store)["resumed"] == pytest.approx(t0)


def test_a_new_hold_after_a_resume_does_not_inherit_the_old_since(cron_store):
    t0 = time.time()
    note_held(now=t0 - 600)
    note_dispatching(now=t0 - 500)
    note_held(now=t0 - 30)
    held = read(cron_store)
    assert held["since"] == pytest.approx(t0 - 30)
    assert held["resumed"] is None
    assert hold_active(cron_store, now=t0) is True


def test_grace_still_counts_from_the_hold_the_one_shot_came_due_in(cron_store):
    """Stamping `resumed` must not lose the #93526 extension for a one-shot due mid-drain."""
    t0 = time.time()
    note_held(now=t0 - 100)
    note_dispatching(now=t0)
    assert grace_starts_at(cron_store, t0 - 90, 120) == pytest.approx(t0 - 100 + TICK_SLACK_SECONDS)
