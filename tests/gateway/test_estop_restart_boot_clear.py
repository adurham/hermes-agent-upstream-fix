"""A sentinel a RESTART wrote must not outlive the restart: the boot that completes it lifts it.

The measured defect (t_d4e7d48c, t_75dcf7b7): ``hermes_safe_restart.sh`` wrote
``~/.hermes/profiles/ps5-pm/ESTOP`` (reason ``safe restart of hermes-gateway``), ran the restart,
and was SIGKILLed 5 s after the boot — it was inside the unit's own cgroup, so ``trap cleanup
EXIT`` never ran and the ``hermes resume`` half never happened. The sentinel then stood ~40 min
past the restart it belonged to, freezing cron dispatch, six clocks, a deploy gate and 13 kanban
admissions.

The fix has to be narrow in BOTH directions, so each direction is pinned here:

* the boot lifts a sentinel that carries a ``restart_token`` and an ``engaged_at`` older than the
  boot (``test_the_boot_lifts_a_restart_sentinel_*``), including one in a PROFILE home, which is
  where the measured orphan lived and which has no boot of its own;
* the boot never lifts an operator's ``hermes pause`` (no token), a torn/empty body, a body whose
  ``engaged_at`` will not parse, or a sentinel written after this boot was already running;
* the clear is seated ONLY on the boot (``boot_lift_restart_estop``, called by
  ``start_gateway`` after it claims the gateway pid file): merely constructing a GatewayRunner —
  which the test suite and the desktop's web server do — lifts nothing, and a boot in a scratch
  home owns no other home's sentinels.

Nothing here ever reads or writes a real sentinel: every arm passes explicit ``homes`` or
monkeypatches ``estop._sentinel_homes``, so a run with a live operator pause engaged cannot be
turned into a false green.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, UTC

import pytest

from agent import estop
from gateway.run_runtime_init import GatewayRuntimeInitMixin, boot_lift_restart_estop
from hermes_cli.subcommands import pause as pause_cli


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Two throwaway homes (a fleet root + one profile) and no reachable real home."""
    root = tmp_path / "fleet"
    profile = root / "profiles" / "ps5-pm"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    estop._logged_components.clear()
    return [root, profile], root, profile


def _write_sentinel(path, body) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body if isinstance(body, str) else json.dumps(body), encoding="utf-8")


def _iso(seconds_from_now: float) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds_from_now)).isoformat()


# ── the clearing direction: a restart's own sentinel is lifted ───────────────────────────────

def test_the_boot_lifts_a_restart_sentinel_that_predates_it(homes):
    dirs, root, _profile = homes
    sentinel = root / "ESTOP"
    estop.engage(reason="safe restart of hermes-gateway", restart_token="safe-restart/1/2")
    assert sentinel.exists()

    # The boot starts one second after the pause was written — the restart it was waiting for.
    cleared = estop.clear_completed_restart(boot_started_at=_now_s() + 1.0, homes=dirs)

    assert [record["path"] for record in cleared] == [str(sentinel)]
    assert cleared[0]["restart_token"] == "safe-restart/1/2"
    assert cleared[0]["reason"] == "safe restart of hermes-gateway"
    assert not sentinel.exists(), "the restart's sentinel outlived the boot that completed it"


def test_the_boot_lifts_a_restart_sentinel_in_a_profile_home(homes):
    """The measured orphan was a PROFILE sentinel — and a profile has no boot of its own."""
    dirs, root, profile = homes
    sentinel = profile / "ESTOP"
    _write_sentinel(sentinel, {"engaged_at": _iso(-120), "reason": "safe restart of hermes-gateway",
                               "restart_token": "safe-restart/21/22"})

    cleared = estop.clear_completed_restart(boot_started_at=_now_s(), homes=dirs)

    assert [record["path"] for record in cleared] == [str(sentinel)]
    assert not sentinel.exists()
    assert not (root / "ESTOP").exists()


def test_the_boot_seat_lifts_the_restart_sentinel(homes, monkeypatch):
    """The seat is the boot helper ``start_gateway`` calls once it owns the gateway pid file."""
    dirs, root, _profile = homes
    sentinel = root / "ESTOP"
    _write_sentinel(sentinel, {"engaged_at": _iso(-30), "reason": "safe restart of hermes-gateway",
                               "restart_token": "safe-restart/31/32"})
    monkeypatch.setattr(estop, "_sentinel_homes", lambda: list(dirs))

    cleared = boot_lift_restart_estop(boot_started_at=_now_s())

    assert [record["path"] for record in cleared] == [str(sentinel)]
    assert not sentinel.exists(), "the boot seat left the restart's pause engaged"


def test_constructing_a_gateway_runner_never_lifts_a_restart_sentinel(homes, monkeypatch):
    """A GatewayRunner is built by the test suite and the desktop too — that is not a boot.

    The first version of this fix cleared from ``_init_registries_and_clocks``, so any pytest run
    would have lifted a live restart pause that another process was still draining. This arm pins
    that construction alone never clears; only the post-claim boot call does.
    """
    dirs, root, _profile = homes
    sentinel = root / "ESTOP"
    _write_sentinel(sentinel, {"engaged_at": _iso(-30), "reason": "safe restart of hermes-gateway",
                               "restart_token": "safe-restart/33/34"})
    monkeypatch.setattr(estop, "_sentinel_homes", lambda: list(dirs))

    class _Runner(GatewayRuntimeInitMixin):
        def __getattr__(self, name):  # unrelated collaborators the init path touches
            return lambda *a, **k: {}

    GatewayRuntimeInitMixin._init_registries_and_clocks(_Runner())

    assert sentinel.exists(), "a non-boot GatewayRunner construction lifted a live restart pause"


def test_start_gateway_calls_the_seat_after_it_claims_the_gateway():
    """Wiring pin: the clear must sit in the boot runtime AFTER the pid-file claim, not before."""
    import inspect

    from gateway import run_bootstrap

    src = inspect.getsource(run_bootstrap._start_gateway_runtime)
    claim = src.index("_start_gateway_claim_pid_file")
    seat = src.index("boot_lift_restart_estop(boot_started_at=_boot_started_at)")
    assert claim < seat, "the clear runs before this process is proven to own the gateway"


# ── the never-clearing direction: an operator's pause, a torn body, a later write ─────────────

def test_the_boot_never_lifts_an_operator_pause(homes):
    dirs, root, _profile = homes
    sentinel = root / "ESTOP"
    estop.engage(reason="runaway cron fan-out")  # no restart token: a person's decision

    cleared = estop.clear_completed_restart(boot_started_at=_now_s() + 3600.0, homes=dirs)

    assert cleared == []
    assert sentinel.exists(), "the boot silently un-paused an operator's `hermes pause`"
    assert estop.is_engaged() is True


def test_constructing_a_runner_never_lifts_an_operator_pause(homes, monkeypatch):
    _dirs, root, _profile = homes
    sentinel = root / "ESTOP"
    estop.engage(reason="operator: stop the lane")
    monkeypatch.setattr(estop, "_sentinel_homes", lambda: [root])

    class _Runner(GatewayRuntimeInitMixin):
        def __getattr__(self, name):
            return lambda *a, **k: {}

    GatewayRuntimeInitMixin._init_registries_and_clocks(_Runner())

    assert sentinel.exists()


@pytest.mark.parametrize("body", ["", "   ", "not json at all", "[]", "null", "{}",
                                  json.dumps({"reason": "safe restart of hermes-gateway"})])
def test_the_boot_never_lifts_a_body_that_cannot_prove_provenance(homes, body):
    """A torn/empty/untagged body still pauses (fail safe) but proves nothing about who wrote it."""
    dirs, root, _profile = homes
    sentinel = root / "ESTOP"
    _write_sentinel(sentinel, body)

    cleared = estop.clear_completed_restart(boot_started_at=_now_s() + 3600.0, homes=dirs)

    assert cleared == []
    assert sentinel.exists()


def test_the_boot_never_lifts_a_token_without_a_usable_engaged_at(homes):
    dirs, root, _profile = homes
    sentinel = root / "ESTOP"
    _write_sentinel(sentinel, {"reason": "safe restart", "restart_token": "safe-restart/41/42"})

    cleared = estop.clear_completed_restart(boot_started_at=_now_s() + 3600.0, homes=dirs)

    assert cleared == []
    assert sentinel.exists()


def test_the_boot_never_lifts_a_sentinel_written_after_this_boot(homes):
    """Two restarts in flight: the second one's pause belongs to a boot that has not happened."""
    dirs, root, _profile = homes
    sentinel = root / "ESTOP"
    _write_sentinel(sentinel, {"engaged_at": _iso(+60), "reason": "safe restart of hermes-gateway",
                               "restart_token": "safe-restart/51/52"})

    cleared = estop.clear_completed_restart(boot_started_at=_now_s(), homes=dirs)

    assert cleared == []
    assert sentinel.exists()


def test_the_boot_lifts_only_the_tagged_sentinel_when_both_homes_hold_one(homes):
    dirs, root, profile = homes
    operator = profile / "ESTOP"
    restart = root / "ESTOP"
    _write_sentinel(operator, {"engaged_at": _iso(-600), "reason": "operator: hardware swap"})
    _write_sentinel(restart, {"engaged_at": _iso(-60), "reason": "safe restart of hermes-gateway",
                              "restart_token": "safe-restart/61/62"})

    cleared = estop.clear_completed_restart(boot_started_at=_now_s(), homes=dirs)

    assert [record["path"] for record in cleared] == [str(restart)]
    assert not restart.exists()
    assert operator.exists(), "the boot lifted an operator pause in another home"


# ── the writer's own narrow cleanup (the half a killed script would have run) ────────────────

def test_release_restart_lifts_only_its_own_token(homes):
    _dirs, root, _profile = homes
    sentinel = root / "ESTOP"
    estop.engage(reason="safe restart of hermes-gateway", restart_token="safe-restart/71/72")

    assert estop.release_restart("some-other-restart") is None
    assert sentinel.exists()

    assert estop.release_restart("safe-restart/71/72") == sentinel
    assert not sentinel.exists()
    assert estop.release_restart("safe-restart/71/72") is None


def test_release_restart_cannot_lift_an_operator_pause_that_overwrote_it(homes):
    """An operator pausing mid-restart rewrites the body; a stale cleanup must not lift it."""
    _dirs, root, _profile = homes
    sentinel = root / "ESTOP"
    estop.engage(reason="safe restart of hermes-gateway", restart_token="safe-restart/81/82")
    estop.engage(reason="operator holds the lane")  # same path, token gone

    assert estop.release_restart("safe-restart/81/82") is None
    assert sentinel.exists()
    state = estop.get_state() or {}
    assert state.get("reason") == "operator holds the lane"


# ── the CLI the script actually calls ────────────────────────────────────────────────────────

def test_pause_cli_writes_the_restart_tag_and_resume_cli_lifts_only_it(homes, capsys):
    _dirs, root, _profile = homes
    sentinel = root / "ESTOP"

    rc = pause_cli.cmd_pause(argparse.Namespace(reason="safe restart of hermes-gateway",
                                               restart_token="safe-restart/91/92"))
    assert rc == 0
    body = json.loads(sentinel.read_text(encoding="utf-8"))
    assert body["restart_token"] == "safe-restart/91/92"
    assert "lifts it on its own" in capsys.readouterr().out

    rc = pause_cli.cmd_resume(argparse.Namespace(restart_token="safe-restart/91/92"))
    assert rc == 0 and not sentinel.exists()

    # An operator pause is not lifted by a restart's token.
    pause_cli.cmd_pause(argparse.Namespace(reason="operator", restart_token=None))
    rc = pause_cli.cmd_resume(argparse.Namespace(restart_token="safe-restart/91/92"))
    assert rc == 0 and sentinel.exists()


def test_pause_without_a_token_stays_untagged(homes):
    _dirs, root, _profile = homes
    pause_cli.cmd_pause(argparse.Namespace(reason="operator", restart_token=None))
    body = json.loads((root / "ESTOP").read_text(encoding="utf-8"))
    assert estop.RESTART_TOKEN_KEY not in body
    found = estop.engaged_sentinel() or (None, {})
    assert found[1].get("restart_token") is None


# ── which homes a boot owns (the sweep's composition, not an injected list) ─────────────────

def _fleet(tmp_path, monkeypatch, *, home):
    fleet = tmp_path / "fleet"
    (fleet / "profiles" / "ps5-pm").mkdir(parents=True, exist_ok=True)
    (fleet / "profiles" / "other-pm").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("agent.file_safety._hermes_home_path", lambda: fleet / home)
    monkeypatch.setattr("agent.file_safety._hermes_root_path", lambda: fleet)
    return fleet.resolve()


def test_sentinel_homes_owns_the_fleet_from_a_profile_home(tmp_path, monkeypatch):
    """The host gateway boots in a profile scope too; it still owns every profile's sentinel."""
    fleet = _fleet(tmp_path, monkeypatch, home="profiles/ps5-pm")
    homes = [p.resolve() for p in estop._sentinel_homes()]
    assert homes[0] == fleet / "profiles" / "ps5-pm"
    assert fleet in homes
    assert {fleet / "profiles" / "ps5-pm", fleet / "profiles" / "other-pm"} <= set(homes)


def test_sentinel_homes_owns_the_fleet_from_the_root(tmp_path, monkeypatch):
    fleet = _fleet(tmp_path, monkeypatch, home=".")
    homes = [p.resolve() for p in estop._sentinel_homes()]
    assert homes[0] == fleet
    assert fleet / "profiles" / "ps5-pm" in homes


def test_a_scratch_boot_owns_only_its_own_home(tmp_path, monkeypatch):
    """A dev/throwaway HERMES_HOME under the fleet tree must not inherit the fleet's sentinels."""
    _fleet(tmp_path, monkeypatch, home="profiles/ps5-pm/cache/dev-home")
    (tmp_path / "fleet" / "profiles" / "ps5-pm" / "cache" / "dev-home").mkdir(parents=True)
    homes = [p.resolve() for p in estop._sentinel_homes()]
    assert homes == [(tmp_path / "fleet" / "profiles" / "ps5-pm" / "cache" / "dev-home").resolve()]


def test_a_fleet_boot_lifts_the_profile_sentinel_with_no_homes_argument(tmp_path, monkeypatch):
    """The measured orphan lived in a PROFILE home: prove the production sweep reaches it."""
    fleet = _fleet(tmp_path, monkeypatch, home=".")
    sentinel = fleet / "profiles" / "ps5-pm" / "ESTOP"
    _write_sentinel(sentinel, {"engaged_at": _iso(-1800), "reason": "safe restart of hermes-gateway",
                               "restart_token": "safe-restart/101/102"})

    cleared = estop.clear_completed_restart(boot_started_at=_now_s())

    assert [record["path"] for record in cleared] == [str(sentinel)]
    assert not sentinel.exists()


def test_a_scratch_boot_cannot_lift_the_fleets_restart_sentinel(tmp_path, monkeypatch):
    """Widening the sweep is the unsafe direction: an unrelated boot proves no ownership."""
    fleet = _fleet(tmp_path, monkeypatch, home="profiles/ps5-pm/cache/dev-home")
    (fleet / "profiles" / "ps5-pm" / "cache" / "dev-home").mkdir(parents=True)
    for path in (fleet / "ESTOP", fleet / "profiles" / "ps5-pm" / "ESTOP"):
        _write_sentinel(path, {"engaged_at": _iso(-600), "reason": "safe restart of hermes-gateway",
                               "restart_token": "safe-restart/111/112"})

    cleared = estop.clear_completed_restart(boot_started_at=_now_s())

    assert cleared == []
    assert (fleet / "ESTOP").exists()
    assert (fleet / "profiles" / "ps5-pm" / "ESTOP").exists()


def _now_s() -> float:
    return datetime.now(UTC).timestamp()

