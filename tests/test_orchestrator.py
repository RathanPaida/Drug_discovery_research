"""Tests for scripts/run_pipeline.py -- the orchestrator's own correctness.

The runner exists to stop a stage running against the wrong inputs, so a bug in IT is the
most expensive kind. Two are pinned here; both occurred while the file was being written.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("run_pipeline", ROOT / "scripts" / "run_pipeline.py")
rp = importlib.util.module_from_spec(_spec)
# Register before exec: @dataclass resolves its annotations via
# sys.modules[cls.__module__].__dict__, which is None for a module loaded by spec alone.
sys.modules["run_pipeline"] = rp
_spec.loader.exec_module(rp)


def test_config_is_propagated_to_every_stage():
    """THE BUG: `--config` was accepted and passed to nothing, so a run launched with a
    scratch config executed every stage against configs/default.yaml -- i.e. against the
    real data/processed tree. Propagation is via $TNK1_CONFIG in the child environment,
    which also reaches the `python -m rl_min.*` stages."""
    src = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    assert '"TNK1_CONFIG": str(cfg["_config_path"])' in src, (
        "the child environment no longer carries TNK1_CONFIG; stages would silently "
        "fall back to the default config")
    # and the env dict must actually be handed to the subprocess
    assert "env=env" in src


def test_every_stage_declares_a_note_and_resolvable_templates():
    plan = rp.build_plan("v9", "cpu", 3, True)
    assert plan, "empty plan"
    ctx = {"proc": Path("/p"), "runs": Path("/r"), "ver": "v9", "cyc": 3}
    for s in plan:
        assert s.note, f"stage {s.name} has no description"
        for tmpl in (*s.needs, *s.makes):
            rp.expand(tmpl, ctx)          # must not raise KeyError on an unknown field


def test_stage_names_are_unique():
    names = [s.name for s in rp.build_plan("v1", "cpu", 1, True)]
    assert len(names) == len(set(names))


def test_an_empty_directory_is_not_a_satisfied_output(tmp_path):
    """THE OTHER BUG, in a different file: `Path.glob()` is a generator and always truthy,
    so an empty `ensemble/` satisfied an existence check and Stage 7 started anyway."""
    d = tmp_path / "ensemble"
    d.mkdir()
    assert not rp.artifact_ok(d), "an empty directory must not count as produced"
    (d / "member_00.pt").write_bytes(b"x")
    assert rp.artifact_ok(d)


def test_an_empty_file_is_not_a_satisfied_output(tmp_path):
    f = tmp_path / "out.csv"
    f.touch()
    assert not rp.artifact_ok(f)
    f.write_text("a,b\n1,2\n")
    assert rp.artifact_ok(f)


def test_missing_artifact_is_not_ok(tmp_path):
    assert not rp.artifact_ok(tmp_path / "nope")


def test_stage_is_current_requires_matching_version_and_dataset():
    state = {"ens": {"version": "v3", "dataset_sha256": "abc123"}}
    ok, _ = rp.stage_is_current(state, "ens", "v3", "abc123")
    assert ok
    ok, why = rp.stage_is_current(state, "ens", "v4", "abc123")
    assert not ok and "version" in why
    ok, why = rp.stage_is_current(state, "ens", "v3", "deadbeef")
    assert not ok and "dataset" in why
    ok, why = rp.stage_is_current({}, "ens", "v3", "abc123")
    assert not ok and "no provenance" in why


def test_version_independent_stages_survive_a_dataset_rebuild():
    """`prior` reads the 788k corpus and never constructs a Ctx, so rebuilding the Stage-1
    table must not mark it stale and trigger hours of needless retraining."""
    state = {"prior": {"version": "v1", "dataset_sha256": "old"}}
    ok, why = rp.stage_is_current(state, "prior", "v3", "new", version_independent=True)
    assert ok and "does not depend" in why
    # ... but a version-DEPENDENT stage with the same record is stale
    ok, _ = rp.stage_is_current(state, "prior", "v3", "new", version_independent=False)
    assert not ok


def test_the_two_version_independent_stages_are_the_expected_ones():
    plan = {s.name: s for s in rp.build_plan("v1", "cpu", 1, True)}
    assert plan["prior"].version_independent
    assert plan["store"].version_independent
    assert not plan["ens"].version_independent, (
        "the ensemble depends on the dataset; marking it independent would let a v3 run "
        "silently reuse a model trained on v1")
