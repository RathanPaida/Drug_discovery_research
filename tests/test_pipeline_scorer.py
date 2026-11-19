"""Regressions for the RL scoring adapter and for repo-root path resolution.

Each test here pins a defect that was silent: nothing raised, nothing logged, and the
run produced a plausible-looking number computed from the wrong thing.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

rdkit = pytest.importorskip("rdkit", reason="PipelineScorer parses SMILES with RDKit")

from rl_min.scorer import PipelineScorer  # noqa: E402

BENZENE = "c1ccccc1"
TOLUENE = "Cc1ccccc1"
PHENOL = "Oc1ccccc1"


class FakeScorer:
    """Stands in for `tnk1.rl.scorer.Scorer`.

    Reward depends on the step, exactly as the real one does through
    `beta_schedule(self.step)` feeding `r_pot`. That is what makes a cross-step cache
    detectable at all.
    """

    def __init__(self):
        self.step = 0
        self.calls: list[list[str]] = []
        self.advances = 0

    def score(self, smiles, advance=True):
        self.calls.append(list(smiles))
        if advance:
            self.step += 1
        # reward falls as the "beta schedule" tightens, so a stale cache reads high
        return {"R": np.full(len(smiles), 1.0 / self.step if self.step else 1.0)}

    def advance_step(self):
        self.step += 1
        self.advances += 1
        return self.step


def test_step_cache_dedupes_within_a_batch_but_not_across_steps():
    """Default mode must re-score a molecule on a later step, at the later step's reward."""
    fake = FakeScorer()
    ps = PipelineScorer(fake, cache="step")

    r1 = ps([BENZENE, TOLUENE])["score"]
    r2 = ps([BENZENE, TOLUENE])["score"]

    assert fake.step == 2, "each call must advance the schedule exactly once"
    # step 1 -> 1.0, step 2 -> 0.5. A run-cache would have returned 1.0 again.
    assert np.allclose(r1, 1.0)
    assert np.allclose(r2, 0.5)
    assert r2[0] != r1[0], "reward was frozen across steps: beta staleness is back"


def test_duplicates_within_one_batch_are_scored_once():
    fake = FakeScorer()
    ps = PipelineScorer(fake, cache="step")
    out = ps([BENZENE, BENZENE, BENZENE, TOLUENE])
    assert len(fake.calls) == 1
    assert fake.calls[0] == [BENZENE, TOLUENE], "repeats were sent through the surrogate"
    assert np.allclose(out["score"], 1.0), "every occurrence still gets its reward"
    assert out["score"].shape == (4,)


def test_schedule_advances_even_when_every_molecule_is_cached():
    """An all-cached batch is still an RL step; beta must not lag behind the step count."""
    fake = FakeScorer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ps = PipelineScorer(fake, cache="run")
    ps([BENZENE])
    assert fake.step == 1
    ps([BENZENE])                      # nothing new to score
    assert fake.step == 2, "beta schedule stalled on an all-cached batch"
    assert fake.advances == 1


def test_run_cache_warns_because_it_freezes_beta():
    with pytest.warns(RuntimeWarning, match="freezes each molecule"):
        PipelineScorer(FakeScorer(), cache="run")


def test_legacy_cache_true_maps_to_run_mode():
    """`cache=True` was the old default; keep it working, but it must warn now."""
    with pytest.warns(RuntimeWarning):
        ps = PipelineScorer(FakeScorer(), cache=True)
    assert ps.mode == "run"


def test_empty_input_does_not_raise_with_caching_off():
    """The old `out["R"]` branch raised NameError here: `out` was never assigned."""
    ps = PipelineScorer(FakeScorer(), cache=False)
    out = ps([])
    assert out["score"].shape == (0,)
    assert out["valid"].shape == (0,)


def test_unparseable_smiles_scores_zero_and_is_flagged_invalid():
    """Absence of a prediction must never inherit a neighbour's reward."""
    fake = FakeScorer()
    ps = PipelineScorer(fake, cache="step")
    out = ps([BENZENE, "not_a_molecule", PHENOL])
    assert out["valid"].tolist() == [True, False, True]
    # the junk string is still handed to the surrogate by this adapter, but if the real
    # Scorer returns 0 for it the adapter must not substitute anything else
    assert out["score"].shape == (3,)


def test_bad_cache_mode_is_rejected():
    with pytest.raises(ValueError, match="cache must be one of"):
        PipelineScorer(FakeScorer(), cache="forever")


def test_scorer_returning_wrong_length_is_caught():
    class Broken(FakeScorer):
        def score(self, smiles, advance=True):
            return {"R": np.ones(len(smiles) + 1)}

    ps = PipelineScorer(Broken(), cache="step")
    with pytest.raises(ValueError, match="rewards for"):
        ps([BENZENE, TOLUENE])


# --------------------------------------------------------------- path resolution
from tnk1.utils.common import load_config          # noqa: E402
from tnk1.utils.paths import repo_root, resolve_path, default_config  # noqa: E402


def test_config_paths_are_absolute_and_cwd_independent(tmp_path, monkeypatch):
    """The defect: every paths.* entry was relative, so running a stage from another
    directory silently created and used a second data/processed tree."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    for key in ("raw", "processed", "runs"):
        p = Path(cfg["paths"][key])
        assert p.is_absolute(), f"paths.{key} is still relative: {p}"
        assert repo_root() in p.parents or p == repo_root(), f"paths.{key} escaped the repo: {p}"
    assert not (tmp_path / "data").exists(), "loading the config created a stray data/ tree"


def test_resolve_path_leaves_absolute_paths_alone(tmp_path):
    assert resolve_path(tmp_path) == tmp_path


def test_resolve_path_anchors_relative_to_repo_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert resolve_path("data/processed") == (repo_root() / "data" / "processed").resolve()


def test_default_config_honours_env_override(monkeypatch, tmp_path):
    fake = tmp_path / "other.yaml"
    fake.write_text("paths: {}\n")
    monkeypatch.setenv("TNK1_CONFIG", str(fake))
    assert default_config() == fake.resolve()


def test_load_config_reports_which_file_it_used():
    cfg = load_config()
    assert Path(cfg["_config_path"]).name == "default.yaml"


def test_missing_config_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="config not found"):
        load_config(tmp_path / "nope.yaml")


# ------------------------------------------- rl_min hyperparameters come from the config
def test_rl_hyperparameters_are_not_inert_argparse_defaults():
    """THE BUG: `--steps/--batch-size/--sigma/--lr` had literal argparse defaults
    (300/64/120.0/1e-4), so rl.steps etc. in the config were ignored. A smoke config asking
    for steps=8, batch_size=16 ran 300 x 64 and took 19.4 minutes instead of seconds, and
    configs/hpc.yaml's steps=1000 would silently have run 300. The argparse default must
    stay None so the config can supply the value."""
    import re
    src = (ROOT / "rl_min" / "run_rl.py").read_text(encoding="utf-8")
    for flag in ("--steps", "--batch-size", "--sigma", "--lr"):
        m = re.search(rf'add_argument\("{re.escape(flag)}".*?\)', src, re.S)
        assert m, f"{flag} not found"
        assert "default=None" in m.group(0), (
            f"{flag} has a literal argparse default again, which makes its config key inert:"
            f"\n  {m.group(0)}")
    # and the resolver must consult the rl section
    assert 'pick(a.steps, "steps"' in src
    assert 'pick(a.batch_size, "batch_size"' in src
    assert 'pick(a.sigma, "sigma"' in src
    assert 'pick(a.lr, "lr"' in src


def test_every_rl_and_rl_min_config_key_is_referenced_in_code():
    """A knob that nothing reads is indistinguishable from a typo. Guards the whole
    rl / rl_min config surface, not just the four flags above."""
    import itertools
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    src = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in itertools.chain((ROOT / "tnk1").rglob("*.py"),
                                 (ROOT / "rl_min").rglob("*.py"),
                                 (ROOT / "scripts").rglob("*.py"))
        if "__pycache__" not in str(p))

    def leaves(d, prefix=""):
        for k, v in (d or {}).items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                yield from leaves(v, p)
            else:
                yield p

    unused = []
    for section in ("rl", "rl_min"):
        for path in leaves({section: cfg.get(section)}):
            leaf = path.split(".")[-1]
            if f'"{leaf}"' not in src and f"'{leaf}'" not in src:
                unused.append(path)
    assert not unused, f"config keys declared but never read: {unused}"


def test_no_stale_argparse_references_after_the_config_resolver():
    """`log_every=max(1, a.steps // 30)` survived the switch to default=None and raised
    `TypeError: unsupported operand type(s) for //: 'NoneType' and 'int'` AFTER the prior
    had been sampled and the reward calibrated -- i.e. it wasted the expensive setup
    before failing. Every post-resolver use must read the resolved DAPConfig."""
    src = (ROOT / "rl_min" / "run_rl.py").read_text(encoding="utf-8")
    body = src.split("dap_kw = dict(", 1)[1]
    # the dict literal itself legitimately mentions a.steps etc.; skip to after it
    body = body.split("cfg = DAPConfig(", 1)[1]
    # strip `#` comments -- the explanatory note about this very bug mentions `a.steps`,
    # and a comment cannot raise a TypeError
    body = "\n".join(line.split("#", 1)[0] for line in body.splitlines())
    for bad in ("a.steps", "a.batch_size", "a.sigma", "a.lr"):
        assert bad not in body, (
            f"{bad} is used after the config resolver; it is None when the value came "
            f"from the config. Read it off the DAPConfig instead.")
