"""Tests for hermes_cli.uninstall.remove_node_symlinks.

Regression for #34536: the POSIX installer drops node/npm/npx symlinks in
~/.local/bin pointing into $HERMES_HOME/node and prepends ~/.local/bin to
PATH, shadowing an existing nvm. Uninstall must remove those symlinks, but
only when they still resolve into the Hermes-managed node dir.
"""

from pathlib import Path

import pytest

from hermes_cli import uninstall


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect Path.home() at the home both the installer-symlink target and
    the ~/.local/bin links live under the same temp dir."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
    (home / ".local" / "bin").mkdir(parents=True)
    return home


def _make_hermes_node(hermes_home: Path) -> Path:
    """Create a fake $HERMES_HOME/node/bin/{node,npm,npx} tree."""
    node_bin = hermes_home / "node" / "bin"
    node_bin.mkdir(parents=True)
    for name in ("node", "npm", "npx"):
        (node_bin / name).write_text("#!/bin/sh\n", encoding="utf-8")
        (node_bin / name).chmod(0o755)
    return node_bin




@pytest.mark.parametrize("owned", [False, True])
def test_node_link_cleanup_respects_resolved_owner(fake_home, monkeypatch, owned):
    home = fake_home / ".hermes"
    link = fake_home / ".local" / "bin" / "node"
    target = home / "node" / "bin" / "node" if owned else fake_home / "other" / "node"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("retained binary", encoding="utf-8")
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"native symlink creation unavailable: {exc}")
    monkeypatch.setattr(uninstall, "_node_symlink_candidate_dirs", lambda home=None: [link.parent])
    removed = uninstall.remove_node_symlinks(home)
    assert removed == ([link] if owned else [])
    assert link.exists() is (not owned)
    assert target.read_text(encoding="utf-8") == "retained binary"


@pytest.mark.require_symlinks
def test_leaves_unrelated_symlinks_untouched(fake_home):
    """A node symlink the user repointed at nvm must survive uninstall."""
    hermes_home = fake_home / ".hermes"
    _make_hermes_node(hermes_home)
    local_bin = fake_home / ".local" / "bin"

    # Simulate nvm's node living elsewhere; user's ~/.local/bin/node -> nvm.
    nvm_bin = fake_home / ".nvm" / "versions" / "node" / "v20.0.0" / "bin"
    nvm_bin.mkdir(parents=True)
    (nvm_bin / "node").write_text("#!/bin/sh\n")
    (local_bin / "node").symlink_to(nvm_bin / "node")

    removed = uninstall.remove_node_symlinks(hermes_home)

    assert removed == []
    assert (local_bin / "node").is_symlink()
    assert (local_bin / "node").resolve() == (nvm_bin / "node").resolve()










@pytest.mark.require_symlinks
def test_removes_fhs_symlinks_in_usr_local_bin(fake_home, tmp_path, monkeypatch):
    """Root FHS installs place node symlinks in /usr/local/bin.

    We monkeypatch _node_symlink_candidate_dirs to return a temp dir standing
    in for /usr/local/bin so the test doesn't need real root privileges.
    """
    hermes_home = fake_home / ".hermes"
    node_bin = _make_hermes_node(hermes_home)

    # Fake /usr/local/bin as a temp dir with our symlinks.
    fhs_bin = tmp_path / "usr_local_bin"
    fhs_bin.mkdir()
    for name in ("node", "npm", "npx"):
        (fhs_bin / name).symlink_to(node_bin / name)

    # Ensure ~/.local/bin has NO symlinks (simulate pure FHS install).
    local_bin = fake_home / ".local" / "bin"
    for name in ("node", "npm", "npx"):
        p = local_bin / name
        if p.exists() or p.is_symlink():
            p.unlink()

    # Return only our fake FHS dir as a candidate.
    monkeypatch.setattr(
        uninstall, "_node_symlink_candidate_dirs", lambda home=None: [fhs_bin]
    )

    removed = uninstall.remove_node_symlinks(hermes_home)

    assert sorted(p.name for p in removed) == ["node", "npm", "npx"]
    for name in ("node", "npm", "npx"):
        assert not (fhs_bin / name).is_symlink()


@pytest.mark.require_symlinks
def test_candidate_dirs_follow_the_hermes_home_not_the_process_home(fake_home, tmp_path):
    """A redirected HERMES_HOME must never point uninstall at the real ~/.local/bin.

    Regression for the pytest sandbox leaking into the operator's real
    ``~/.local/bin``: the shim directory is derived from the home being
    uninstalled, so a temp install cannot unlink the user's own shims.
    """
    sandbox = tmp_path / "sandbox" / ".hermes"
    sandbox.mkdir(parents=True)
    dirs = uninstall._node_symlink_candidate_dirs(sandbox)
    assert dirs[0] == tmp_path / "sandbox" / ".local" / "bin"
    assert fake_home / ".local" / "bin" not in dirs


@pytest.mark.require_symlinks
def test_profile_home_still_resolves_to_the_owner_home(fake_home):
    """A profile home (~/.hermes/profiles/x) maps back to its owner's ~/.local/bin."""
    profile = fake_home / ".hermes" / "profiles" / "ps5-pm"
    profile.mkdir(parents=True)
    assert uninstall._node_symlink_candidate_dirs(profile)[0] == fake_home / ".local" / "bin"
