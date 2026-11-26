"""A lane's PM bot may never restart the host gateway.

Measured 2026-11-25: a ps5-pm agent committed to the checkout at 18:29:11 and requested a gateway
restart ten seconds later. The 1800 s drain bound expired at 18:59 and force-stopped 12 managed
workers — with the owner away from home and no idea anything had asked for it. A lane PM runs
unattended over a shared gateway, so a restart it asks for is every other lane's outage.
"""
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_identity import RoutingIdentity
from gateway.slash_commands import gateway_restart_denied_profile
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _event(command: str = "/restart", profile: str | None = None) -> MessageEvent:
    source = make_restart_source()
    if profile is not None:
        source.profile = profile
    return MessageEvent(
        text=command,
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
        platform_update_id=1,
    )


@pytest.mark.parametrize("profile", ["ps5-pm", "comma-pm", "hub-pm", "nest-pm", "shark-pm", "switch-pm"])
def test_every_lane_profile_is_denied(profile):
    assert gateway_restart_denied_profile(make_restart_source_with(profile)) == profile


def make_restart_source_with(profile):
    source = make_restart_source()
    source.profile = profile
    return source


@pytest.mark.parametrize("profile", ["default", "work", "ops"])
def test_the_owners_own_profiles_are_not_denied(profile, monkeypatch):
    """The allowlist defaults to `default` alone; a named owner profile may be added in config."""
    from gateway import slash_commands
    monkeypatch.setattr(slash_commands, "restart_allowed_profiles",
                        lambda: frozenset({"default", "work", "ops"}))
    assert gateway_restart_denied_profile(make_restart_source_with(profile)) is None


def test_a_profile_nobody_allowlisted_is_denied_even_without_a_pm_suffix(monkeypatch):
    """A renamed or newly added lane must be denied by default, not by its name."""
    from gateway import slash_commands
    monkeypatch.setattr(slash_commands, "restart_allowed_profiles", lambda: frozenset({"default"}))
    assert gateway_restart_denied_profile(make_restart_source_with("newlane")) == "newlane"


def test_default_is_always_allowed_even_when_config_is_silent():
    from gateway.slash_commands import restart_allowed_profiles
    assert "default" in restart_allowed_profiles()


def test_an_unresolved_profile_fails_open():
    """The owner's own transports can reach a command with no profile; never block those."""
    assert gateway_restart_denied_profile(make_restart_source()) is None


def test_the_pinned_identity_outranks_the_source_attribute():
    """A profile_routes entry re-homes the event: the EXECUTING profile is the one that counts."""
    source = make_restart_source()
    object.__setattr__(
        source,
        "_identity",
        RoutingIdentity(
            transport_profile="default",
            runtime_profile="ps5-pm",
            authorization_home=gateway_run._hermes_home,
            runtime_home=gateway_run._hermes_home,
        ),
    )
    assert gateway_restart_denied_profile(source) == "ps5-pm"


@pytest.mark.asyncio
async def test_restart_command_from_a_lane_refuses_before_any_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock()

    reply = await runner._handle_restart_command(_event("/restart", profile="ps5-pm"))

    assert "may never restart the host gateway" in reply
    runner.request_restart.assert_not_called()
    assert not (tmp_path / ".restart_notify.json").exists()


@pytest.mark.asyncio
async def test_update_command_from_a_lane_refuses_before_spawning_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock()

    reply = await runner._handle_update_command(_event("/update", profile="comma-pm"))

    assert "may never restart the host gateway" in reply
    runner.request_restart.assert_not_called()
    assert not (tmp_path / ".update_pending.json").exists()
