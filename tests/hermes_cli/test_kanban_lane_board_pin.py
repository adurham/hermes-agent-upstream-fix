"""Lane board pin: a profile declares its board, so a successor mint cannot inherit another lane's.

Measured failure (2026-11-24): ``<root>/kanban/current`` held ``hub`` (hub-pm's board, switched at
17:22 the night before). Every ps5-pm card minted with no explicit board resolved that pointer, and
a ps5-pm card on ``hub`` cannot be executed — the owner answers "Kanban execution unknown; retained
by owner" and the worker exits 1. Two successors died that way in 30 minutes, and the second was
minted by the first one's own worker (its board came from the ``HERMES_KANBAN_BOARD`` the dispatcher
handed it), so one wrong mint replicates down the whole successor chain.

These tests pin the two halves of the fix:
  * ``kanban.board`` resolves between the env pin and the shared ``current`` pointer, per profile;
  * a create for a pinned lane on a board outside its ``kanban.cross_board`` list is refused with a
    one-line reason instead of filing a card that cannot run.
"""
import sqlite3

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture()
def lane(tmp_path, monkeypatch):
    """A kanban home with one extra board per slug, plus an isolated profile config dir."""
    home = tmp_path / "hermes"
    (home / "kanban" / "boards" / "hub").mkdir(parents=True)
    (home / "kanban" / "boards" / "ps5-hv").mkdir(parents=True)
    (home / "kanban" / "boards" / "ps5").mkdir(parents=True)
    for slug in ("hub", "ps5-hv", "ps5"):
        (home / "kanban" / "boards" / slug / "board.json").write_text(f'{{"slug": "{slug}"}}')
    (home / "kanban" / "current").write_text("hub\n")
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr(kb, "lane_board_pin", lambda profile=None: None)
    monkeypatch.setattr(kb, "lane_cross_boards", lambda profile=None: frozenset())
    return home


def _pin(monkeypatch, pin, cross=()):
    monkeypatch.setattr(kb, "lane_board_pin", lambda profile=None: pin)
    monkeypatch.setattr(kb, "lane_cross_boards", lambda profile=None: frozenset(cross))


def test_current_pointer_still_wins_without_a_pin(lane):
    """No pin = no behaviour change: every other lane keeps resolving the shared pointer."""
    assert kb.get_current_board() == "hub"


def test_lane_pin_beats_the_shared_current_pointer(lane, monkeypatch):
    """The whole bug: `current` says hub, the ps5 lane is pinned to default."""
    _pin(monkeypatch, "default")
    assert kb.get_current_board() == "default"


def test_env_pin_beats_the_lane_pin(lane, monkeypatch):
    """A dispatcher-spawned worker carries its own card's board and must keep resolving it."""
    _pin(monkeypatch, "default")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "ps5-hv")
    assert kb.get_current_board() == "ps5-hv"


def test_stale_lane_pin_falls_through_to_current(lane, monkeypatch):
    """A hand-edited pin naming a board that does not exist must not crash or resurrect it."""
    _pin(monkeypatch, "no-such-board")
    assert kb.get_current_board() == "hub"


def test_lane_pin_resolves_from_a_profile_config_file(tmp_path, monkeypatch):
    """The pin is read off <profile>/config.yaml, so a creator in another profile can read it."""
    from hermes_cli import profiles

    prof = tmp_path / "profiles" / "ps5-pm"
    prof.mkdir(parents=True)
    (prof / "config.yaml").write_text("kanban:\n  board: default\n  cross_board:\n    - ps5-hv\n")
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "ps5-pm")
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: prof)
    assert kb.lane_board_pin("ps5-pm") == "default"
    assert kb.lane_cross_boards("ps5-pm") == frozenset({"ps5-hv"})


def test_no_pin_means_no_violation(lane):
    """Unpinned lanes are untouched: filing on any board still works."""
    assert kb.lane_board_violation(assignee="nest-pm", board="hub") is None


def test_misroute_to_another_lane_is_refused(lane, monkeypatch):
    """Acceptance: a pinned lane's card on a board it cannot run on is refused, by name."""
    _pin(monkeypatch, "default", cross=["ps5-hv"])
    why = kb.lane_board_violation(assignee="ps5-pm", board="hub")
    assert why and "refused" in why and "'hub'" in why and "'default'" in why


def test_cross_board_list_is_honoured(lane, monkeypatch):
    """The lane really files hypervisor cards on ps5-hv; a pin alone must not forbid that."""
    _pin(monkeypatch, "default", cross=["ps5-hv"])
    assert kb.lane_board_violation(assignee="ps5-pm", board="ps5-hv") is None


def test_unset_board_is_judged_after_resolution(lane, monkeypatch):
    """board=None must be judged on where it WOULD land — a stale session snapshot is the bug."""
    _pin(monkeypatch, "default")
    # No env snapshot: resolution lands on the pin, so there is nothing to refuse.
    assert kb.lane_board_violation(assignee="ps5-pm", board=None) is None
    # A session whose HERMES_KANBAN_BOARD snapshotted hub at boot would land on hub: refuse.
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "hub")
    assert "refused" in (kb.lane_board_violation(assignee="ps5-pm", board=None) or "")


def test_create_task_refuses_a_misrouted_card(lane, monkeypatch):
    """The guard sits in create_task, so tool and CLI creates are both refused before the row."""
    _pin(monkeypatch, "default", cross=["ps5-hv"])
    import hermes_cli.kanban_db_connect as kbc

    conn = kbc.connect(board="hub")
    try:
        with pytest.raises(ValueError, match="refused:"):
            kb.create_task(conn, title="dead successor", assignee="ps5-pm", board="hub",
                           created_by="ps5-pm")
        # The allowlisted cross-board target still files normally.
        tid = kb.create_task(conn, title="hypervisor card", assignee="ps5-pm", board="ps5-hv",
                             created_by="ps5-pm")
        assert kb.get_task(conn, tid) is not None
    finally:
        conn.close()
