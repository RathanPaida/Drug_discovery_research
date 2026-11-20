"""Regressions for defect 4.1 (RUN_REPORT_V3_HPC.md) — components computed and discarded.

`Scorer.score` built an eight-key `comps` dict and then scattered it with
`for c in self.components`, where `self.components` under the v2 reward is
`COMPONENTS_V2 = [pot, sel, hinge, ad, sa, safety]`. `adme` and `qed` are not in that
list, so both were computed on the line above and dropped: the returned arrays kept the
`np.zeros(n)` initialisation from the top of the method.

Nothing raised and nothing logged. `scripts/run_stage8_cycle.py` feeds `P.adme` as one of
six Pareto objectives, and a constant column cannot affect dominance — every candidate
pair ties on it — so ADME never influenced candidate selection, and `qed` was written into
the candidate CSV as zeros.

These tests are hermetic: the RDKit med-chem profiler and the torch ADMET predictor are
stubbed through `sys.modules` (both are imported lazily inside `score`), and the ensemble,
applicability domain and calibrator are fakes. What is exercised is the real
`Scorer.score` body — the scatter, the reward aggregation and the hard gate.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tnk1.rl import reward as R  # noqa: E402
from tnk1.rl.scorer import Scorer  # noqa: E402
from tnk1.utils.common import load_config  # noqa: E402

SMIS = ["CCO", "CCN", "CCC"]

# Deliberately NOT the values that a zeroed array would produce. p(hERG)=p(AMES)=0 is a
# flawless safety profile, and logS=logPapp=0 the best possible ADME, so a stub of zeros
# would make a discarded component indistinguishable from a computed one.
ENDPOINTS = {"solubility": -4.2, "caco2": -5.0, "microsome": 12.0,
             "herg": 0.03, "ames": 0.07, "cyp3a4": 0.04, "cyp2d6": 0.004}

PROFILE = {"qed": 0.61, "sa": 2.8, "mw": 342.0, "n_alerts": 0, "alerts": "",
           "hinge": True, "n_rot": 4}


class FakeEnsemble:
    """`predict_members(smiles, panel) -> (mu, log_var)`, each (members, n, targets)."""

    def __init__(self, members=3, rng=None):
        self.members = members
        self.rng = rng or np.random.default_rng(0)

    def predict_members(self, smiles, panel):
        shape = (self.members, len(smiles), len(panel))
        mu = 7.0 + self.rng.normal(0.0, 0.15, shape)
        mu[..., 1:] -= 1.2                      # off-targets weaker -> a real selectivity margin
        return mu, np.full(shape, -2.0)

    def latent(self, smiles):
        return self.rng.normal(size=(len(smiles), 8))


class FakeAD:
    d_thr_lat = 1.0

    def distances(self, Z, FP):
        return np.full(len(Z), 0.5), np.full(len(Z), 0.5)


def _install_stubs(monkeypatch, profile_fn=None):
    """Stub the two lazily-imported heavy modules: RDKit profiling and torch ADMET."""
    filters = types.ModuleType("tnk1.admet.filters")
    filters.medchem_profile = profile_fn or (lambda s: dict(PROFILE))

    admet = types.ModuleType("tnk1.admet.admet")
    # bundles map endpoint name -> bundle; here the "bundle" is the constant to return.
    admet.predict_endpoint = lambda bundle, smis, ds, dev: (np.full(len(smis), bundle), None)

    monkeypatch.setitem(sys.modules, "tnk1.admet.filters", filters)
    monkeypatch.setitem(sys.modules, "tnk1.admet.admet", admet)


def _scorer(monkeypatch, use_v2=True):
    _install_stubs(monkeypatch)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["rl"]["use_v2"] = use_v2
    cfg["rl"]["calibrate_at_runtime"] = False
    ens = FakeEnsemble()
    return Scorer(cfg, ens, calibrator=None, ad=FakeAD(),
                  ad_ref_fn=lambda s, e: (e.latent(s), np.ones((len(s), 4))),
                  admet_bundles=dict(ENDPOINTS), desc_scaler=None, device="cpu")


# --------------------------------------------------------------------------- the defect

def test_adme_and_qed_are_reported_under_the_v2_reward(monkeypatch):
    """The regression. Both are computed by `score`; neither is in COMPONENTS_V2.

    Before the fix these two assertions failed with exact zeros, while every other
    component came back populated — which is why the defect survived a full HPC run.
    """
    sc = _scorer(monkeypatch, use_v2=True)
    out = sc.score(SMIS, advance=False)

    assert "adme" not in sc.components, "premise: adme is outside COMPONENTS_V2"
    assert "qed" not in sc.components, "premise: qed is outside COMPONENTS_V2"

    assert np.all(out["adme"] > 0.0), f"adme reported as zeros: {out['adme']}"
    assert np.allclose(out["qed"], PROFILE["qed"]), f"qed not reported: {out['qed']}"


def test_every_computed_component_reaches_the_output(monkeypatch):
    """The general guard: no component may be computed and then dropped.

    Pins the class of defect rather than the instance, so a component added to `comps`
    later cannot silently go missing the way `adme` and `qed` did.
    """
    sc = _scorer(monkeypatch, use_v2=True)
    out = sc.score(SMIS, advance=False)

    for c in set(R.COMPONENTS) | set(R.COMPONENTS_V2) | {"hinge"}:
        assert c in out, f"component {c!r} absent from the scorer output"
        assert not np.allclose(out[c], 0.0), (
            f"component {c!r} is all-zeros; it was computed and discarded, or the stub "
            f"inputs make it genuinely zero (which would make this test blind)")


def test_adme_is_not_constant_across_molecules_with_different_endpoints(monkeypatch):
    """A constant column is the thing Stage 8's Pareto front cannot see.

    Scoring two batches whose ADME endpoints differ must move `adme`. Equality here would
    mean the value is still coming from somewhere other than the computation.
    """
    sc = _scorer(monkeypatch, use_v2=True)
    good = sc.score(SMIS, advance=False)["adme"]

    permeable = dict(ENDPOINTS, caco2=-4.0)        # above the -5.15 threshold
    sc2 = _scorer(monkeypatch, use_v2=True)
    sc2.admet = permeable
    better = sc2.score(SMIS, advance=False)["adme"]

    assert np.all(better > good), (
        f"improving Caco-2 permeability did not raise adme: {good} -> {better}")


def test_v1_component_set_still_reports_every_component(monkeypatch):
    """The Cycle-01 reward must be unaffected: `hinge` is outside COMPONENTS, and under the
    fix it is now reported there too (it was already computed)."""
    sc = _scorer(monkeypatch, use_v2=False)
    out = sc.score(SMIS, advance=False)

    assert np.all(out["adme"] > 0.0)
    assert np.allclose(out["qed"], PROFILE["qed"])
    assert np.all(out["hinge"] > 0.0), "hinge computed but discarded under the v1 set"


def test_invalid_rows_stay_zero(monkeypatch):
    """The scatter writes only at the valid indices. A row RDKit cannot parse must keep its
    zeros rather than inheriting a neighbour's score."""
    _install_stubs(monkeypatch,
                   lambda s: None if s == "not-a-smiles" else dict(PROFILE))

    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["rl"]["use_v2"] = True
    cfg["rl"]["calibrate_at_runtime"] = False
    sc = Scorer(cfg, FakeEnsemble(), None, FakeAD(),
                lambda s, e: (e.latent(s), np.ones((len(s), 4))),
                dict(ENDPOINTS), None, "cpu")

    out = sc.score(["CCO", "not-a-smiles", "CCC"], advance=False)
    assert out["adme"][1] == 0.0 and out["qed"][1] == 0.0
    assert out["adme"][0] > 0.0 and out["adme"][2] > 0.0


def test_all_invalid_batch_returns_every_key(monkeypatch):
    """The early return must carry the same keys as the normal path, or a caller that reads
    `out["adme"]` crashes only on the rare all-invalid batch."""
    _install_stubs(monkeypatch, lambda s: None)

    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["rl"]["use_v2"] = True
    cfg["rl"]["calibrate_at_runtime"] = False
    sc = Scorer(cfg, FakeEnsemble(), None, FakeAD(),
                lambda s, e: (e.latent(s), np.ones((len(s), 4))),
                dict(ENDPOINTS), None, "cpu")

    out = sc.score(SMIS, advance=False)
    for c in set(R.COMPONENTS) | set(R.COMPONENTS_V2):
        assert c in out and np.allclose(out[c], 0.0)
