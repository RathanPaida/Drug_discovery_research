"""Vina discovery and per-cycle docking paths.

Both of these were module-level constants in `scripts/dock_40_molecules_tnk1.py`, behind
an `import rdkit` / `import meeko`, so neither could be tested on a machine without a
chemistry stack -- which is precisely where you want to confirm that the binary path
resolves before spending a queue slot. Each test below pins a defect that shipped.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from tnk1.docking import paths as DP


# --------------------------------------------------------------------- per-cycle paths

def test_cycle_paths_follow_the_cycle_number(tmp_path):
    """REGRESSION. `main` parsed --cycle and then used constants pinned to cycle_01, so
    `--cycle 3` silently re-docked Cycle-01 and overwrote its published poses."""
    (tmp_path / "runs").mkdir()
    p3 = DP.cycle_paths(3, root=tmp_path)
    assert p3["runs"].name == "cycle_03"
    for key in ("contract", "contract_fallback", "poses", "results", "summary"):
        assert "cycle_03" in str(p3[key]), f"{key} did not follow --cycle"
        assert "cycle_01" not in str(p3[key])


def test_cycle_paths_zero_pad(tmp_path):
    (tmp_path / "runs").mkdir()
    assert DP.cycle_paths(1, root=tmp_path)["runs"].name == "cycle_01"
    assert DP.cycle_paths(12, root=tmp_path)["runs"].name == "cycle_12"


def test_distinct_cycles_never_share_a_path(tmp_path):
    (tmp_path / "runs").mkdir()
    a = DP.cycle_paths(1, root=tmp_path)
    b = DP.cycle_paths(2, root=tmp_path)
    assert set(a.values()).isdisjoint(set(b.values()))


@pytest.mark.parametrize("bad", [0, -1])
def test_cycle_must_be_positive(bad, tmp_path):
    with pytest.raises(ValueError, match="cycle must be"):
        DP.cycle_paths(bad, root=tmp_path)


def test_pipeline_root_prefers_a_real_runs_dir(tmp_path):
    (tmp_path / "runs").mkdir()
    assert DP.pipeline_root(tmp_path) == tmp_path


def test_pipeline_root_keeps_the_legacy_nested_fallback(tmp_path):
    """Older checkouts nested the pipeline; dropping the fallback silently would break
    them, so it is preserved deliberately."""
    assert DP.pipeline_root(tmp_path) == tmp_path / "tnk1-pipeline" / "tnk1-pipeline"


def test_repo_root_contains_this_package():
    assert (DP.repo_root() / "tnk1" / "docking" / "paths.py").is_file()


# ------------------------------------------------------------------- Vina resolution

def _make_exe(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_explicit_path_wins(tmp_path):
    exe = _make_exe(tmp_path / "mine" / "vina")
    assert DP.resolve_vina(str(exe), root=tmp_path, env={}) == exe


def test_env_var_is_honoured(tmp_path):
    exe = _make_exe(tmp_path / "env" / "vina")
    got = DP.resolve_vina(None, root=tmp_path, env={"TNK1_VINA": str(exe)})
    assert got == exe


def test_explicit_beats_env(tmp_path):
    a = _make_exe(tmp_path / "a" / "vina")
    b = _make_exe(tmp_path / "b" / "vina")
    assert DP.resolve_vina(str(a), root=tmp_path, env={"TNK1_VINA": str(b)}) == a


def test_tools_dir_is_found_without_an_exe_suffix(tmp_path):
    """REGRESSION. The constant was `tools/vina.exe`. A Linux build is named `vina`, so
    the only name the old code accepted was the one that cannot run on the HPC."""
    exe = _make_exe(tmp_path / "tools" / "vina")
    assert DP.resolve_vina(None, root=tmp_path, env={}) == exe


def test_versioned_linux_release_name_is_accepted(tmp_path):
    exe = _make_exe(tmp_path / "tools" / "vina_1.2.7_linux_x86_64")
    assert DP.resolve_vina(None, root=tmp_path, env={}) == exe


def test_windows_exe_still_works(tmp_path):
    """The Cycle-01 campaign ran with vina.exe; that must keep resolving."""
    exe = _make_exe(tmp_path / "tools" / "vina.exe")
    assert DP.resolve_vina(None, root=tmp_path, env={}) == exe


def test_a_directory_is_not_mistaken_for_the_binary(tmp_path):
    (tmp_path / "tools" / "vina").mkdir(parents=True)
    with pytest.raises(SystemExit):
        DP.resolve_vina(None, root=tmp_path, env={"PATH": ""})


def test_missing_binary_raises_systemexit_not_filenotfound(tmp_path, monkeypatch):
    """REGRESSION. Previously the absence surfaced as a bare FileNotFoundError from
    subprocess, inside a worker thread, after the receptor prep had already run."""
    monkeypatch.setattr(DP.shutil, "which", lambda *_a, **_k: None)
    with pytest.raises(SystemExit) as e:
        DP.resolve_vina(None, root=tmp_path, env={})
    msg = str(e.value)
    assert "not found" in msg
    assert "conda install" in msg, "the error must name a way to fix it"
    assert "--vina" in msg
    assert "Looked in" in msg


def test_error_lists_every_path_tried(tmp_path, monkeypatch):
    monkeypatch.setattr(DP.shutil, "which", lambda *_a, **_k: None)
    with pytest.raises(SystemExit) as e:
        DP.resolve_vina("/nope/vina", root=tmp_path, env={"TNK1_VINA": "/also/nope"})
    msg = str(e.value)
    assert "/nope/vina" in msg and "/also/nope" in msg
    assert str(tmp_path / "tools" / "vina") in msg


def test_resolution_does_not_mutate_the_real_environment(tmp_path):
    exe = _make_exe(tmp_path / "tools" / "vina")
    before = dict(os.environ)
    DP.resolve_vina(None, root=tmp_path, env={})
    assert dict(os.environ) == before
    assert exe.is_file()


def test_tilde_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = _make_exe(tmp_path / "bin" / "vina")
    assert DP.resolve_vina("~/bin/vina", root=tmp_path, env={}) == exe
