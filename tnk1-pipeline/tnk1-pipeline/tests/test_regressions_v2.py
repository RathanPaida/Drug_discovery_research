"""Regression tests for the four v2 defects.

Each of these shipped in Cycle 01 and each was silent -- none raised, none logged, and the
notebook reported results as though the affected stage had worked. They are pinned here
because a silent defect that produced plausible numbers once will do so again.

  #1 write_rl_toml emitted unparseable TOML on Windows, so Stage 7 never started.
  #2 ApplicabilityDomain divided by ~0 on dead latent dimensions, so r_ad -- a nominal
     weight-1.5 regulariser -- became the largest single driver of the RL reward.
  #3 Reward sigmoid centres sat outside the population's range, so r_sel spent 19% of the
     weight budget as a near-constant multiplier carrying no gradient.
  #4 Missing ADMET endpoints defaulted to 0.0, which reads as PERFECT safety.
"""
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from tnk1.rl import reinvent_io as RI
from tnk1.rl import reward as RW
from tnk1.uq.applicability import ApplicabilityDomain

# ─────────────────────────────────────────── #1 TOML generation on Windows paths

WINDOWS_PATHS = [
    r"D:\ML_Research\ml_research\tnk1-pipeline\data\processed\prior\kinase.prior",
    r"C:\Users\x\miniconda3\envs\tnk1-reinvent\python.exe",
    r"D:\ML_Research\tnk1\rl\reinvent_client.py",
]


def _cfg():
    return {"rl": {"steps": 1000, "batch_size": 128, "sigma": 120, "lr": 1e-4,
                   "strategy": "dap", "diversity": {"bucket_size": 25, "min_score": 0.55}}}


def test_rl_toml_parses_when_paths_are_windows_native(tmp_path):
    """`data\\processed` contains \\p, an invalid TOML escape; `\\tnk1\\rl` silently became
    TAB + CR. tomllib (what REINVENT 4 uses) rejects the first and corrupts the second."""
    out = tmp_path / "rl.toml"
    RI.write_rl_toml(out, _cfg(), prior=WINDOWS_PATHS[0], agent=WINDOWS_PATHS[0],
                     client_python=WINDOWS_PATHS[1], client_script=WINDOWS_PATHS[2],
                     server_url="http://127.0.0.1:8765")
    doc = tomllib.loads(out.read_text(encoding="utf-8"))          # must not raise
    assert doc["parameters"]["prior_file"].endswith("kinase.prior")
    assert "\\" not in doc["parameters"]["prior_file"]


def test_rl_toml_introduces_no_tab_or_carriage_return(tmp_path):
    out = tmp_path / "rl.toml"
    RI.write_rl_toml(out, _cfg(), prior=WINDOWS_PATHS[0], agent=WINDOWS_PATHS[0],
                     client_python=WINDOWS_PATHS[1], client_script=WINDOWS_PATHS[2],
                     server_url="http://127.0.0.1:8765")
    text = out.read_text(encoding="utf-8")
    assert "\t" not in text, "an invalid \\t escape survived into the generated TOML"
    assert "\r" not in text.replace("\r\n", "\n")


def test_transfer_and_sampling_tomls_also_parse(tmp_path):
    RI.write_transfer_learning_toml(tmp_path / "tl.toml", WINDOWS_PATHS[0],
                                    r"D:\x\train.smi", r"D:\x\val.smi", r"D:\x\out.prior")
    RI.write_sampling_toml(tmp_path / "s.toml", WINDOWS_PATHS[0], r"D:\x\sampled.csv")
    for name in ("tl.toml", "s.toml"):
        tomllib.loads((tmp_path / name).read_text(encoding="utf-8"))


def test_toml_path_rejects_a_quote_it_cannot_escape():
    """Literal strings have no escape mechanism, so a single quote must be refused loudly
    rather than producing a file that parses into the wrong path."""
    with pytest.raises(ValueError):
        RI._toml_path("/tmp/it's/here")


# ─────────────────────────────── #2 applicability domain on degenerate latents

def _degenerate_latent(n=304, d=300, seed=0):
    """304 molecules in a 300-d latent -- the real Stage 4 geometry. 160 units never fire
    on the training set, so their std is exactly 0 and the shipped code divided by 1e-6."""
    rng = np.random.RandomState(seed)
    Z = np.zeros((n, d))
    Z[:, :40] = rng.randn(n, 40)
    Z[:, 40:140] = rng.randn(n, 100) * 1e-4
    FP = (rng.rand(n, 2048) < 0.05).astype(float)
    return Z, FP, rng


def test_dead_latent_dimensions_do_not_explode_d_lat():
    Z, FP, rng = _degenerate_latent()
    Zn = np.zeros((500, Z.shape[1]))
    Zn[:, :40] = rng.randn(500, 40) * 1.5
    Zn[:, 140:] = np.abs(rng.randn(500, 160)) * 0.05        # dead units now activating
    FPn = (rng.rand(500, 2048) < 0.05).astype(float)

    broken = ApplicabilityDomain(sd_floor_frac=0.0, d_lat_clip=0.0).fit(Z, FP)
    fixed = ApplicabilityDomain().fit(Z, FP)
    d_broken, _ = broken.distances(Zn, FPn)
    d_fixed, _ = fixed.distances(Zn, FPn)

    assert d_broken.max() > 1000 * fixed.d_thr_lat, "failed to reproduce the blowup"
    assert d_fixed.max() <= fixed.d_lat_clip * fixed.d_thr_lat
    assert np.isfinite(d_fixed).all()
    assert fixed.n_floored > 0


def test_r_ad_keeps_a_usable_gradient_after_the_fix():
    """exp(-(d_lat - thr)/0.15) underflows to exactly 0.0 for every candidate when d_lat is
    five orders of magnitude above threshold, zeroing the whole geometric-mean reward."""
    Z, FP, rng = _degenerate_latent()
    Zn = np.zeros((500, Z.shape[1]))
    Zn[:, :40] = rng.randn(500, 40) * 1.5
    Zn[:, 140:] = np.abs(rng.randn(500, 160)) * 0.05
    FPn = (rng.rand(500, 2048) < 0.05).astype(float)

    broken = ApplicabilityDomain(sd_floor_frac=0.0, d_lat_clip=0.0).fit(Z, FP)
    fixed = ApplicabilityDomain().fit(Z, FP)
    r_broken = RW.r_ad(broken.distances(Zn, FPn)[0], broken.d_thr_lat)
    r_fixed = RW.r_ad(fixed.distances(Zn, FPn)[0], fixed.d_thr_lat)

    assert (r_broken == 0).all(), "expected total underflow before the fix"
    assert np.log(np.maximum(r_fixed, 1e-4)).std() > 0, "r_ad still carries no gradient"


def test_fp_distance_stays_bounded_regardless():
    """d_fp is a Tanimoto distance, so it is on [0,1] by construction -- which is why the
    reward should prefer it for any hard in/out decision."""
    Z, FP, rng = _degenerate_latent()
    ad = ApplicabilityDomain().fit(Z, FP)
    _, d_fp = ad.distances(Z * 1e6, (rng.rand(50, 2048) < 0.5).astype(float))
    assert ((d_fp >= 0) & (d_fp <= 1)).all()


# ──────────────────────────────────────────── #3 reward centre calibration

def test_calibrate_centre_puts_the_requested_fraction_above_the_midpoint():
    vals = np.random.RandomState(0).normal(-1.24, 0.42, 20000)   # the real mu_sel spread
    c = RW.calibrate_centre(vals, target_frac=0.30)
    above = (RW.sigmoid(vals, c, 2.0) > 0.5).mean()
    assert above == pytest.approx(0.30, abs=0.02)


def test_shipped_sel_centre_crushes_the_attainable_reward_range():
    """sel_c was 1.2 while mu_sel over 12,000 molecules had mean -1.24 and MAX 0.25, so no
    molecule ever reached the midpoint.

    Note what the defect is and is not. In the saturated left tail sigmoid(x) ~ exp(k(x-c)),
    so log r is LINEAR in x and the log-variance is actually at its maximum there -- the
    term does transmit variance into the geometric mean. The damage is to the attainable
    RANGE: every molecule is multiplied by a near-constant heavy discount, R never
    approaches 1, and since DAP uses `prior_logP + sigma * R` the effective sigma collapses
    with it (120 -> ~48 on the shipped pool). A component cannot distinguish "acceptable"
    from "excellent" when the whole population sits on one rail.
    """
    vals = np.random.RandomState(0).normal(-1.24, 0.42, 20000)
    sig = np.zeros_like(vals)
    shipped = RW.r_sel(vals, sig, c=1.2)
    tuned = RW.r_sel(vals, sig, c=RW.calibrate_centre(vals, 0.30))

    assert (shipped < 0.01).mean() > 0.60, "expected the shipped centre to pin to the floor"

    w = 2.0 / 10.5                                   # sel's normalised exponent
    assert (shipped ** w).mean() < 0.5, "expected a heavy near-constant discount"
    assert (tuned ** w).mean() > 0.8, "recalibration should stop crushing R"
    # compare the span over the BULK of the population; the extreme tails of a
    # saturated sigmoid are noise and understate the compression (5x vs 21x).
    span = lambda r: np.percentile(r, 95) - np.percentile(r, 5)
    assert span(tuned) > 10 * span(shipped)


def test_component_signal_flags_a_flat_term():
    """A term whose value barely varies costs weight and returns no gradient; it belongs in
    hard_gate. r_adme and r_qed both measured below 0.02 on the shipped pool."""
    n = 5000
    rng = np.random.RandomState(1)
    comps = {c: rng.uniform(0.2, 0.9, n) for c in RW.COMPONENTS}
    comps["adme"] = np.full(n, 0.55)                    # perfectly flat
    weights = {"pot": 3.0, "sel": 2.0, "ad": 1.5, "sa": 1.0,
               "adme": 1.0, "safety": 1.5, "qed": 0.5}
    sig = RW.component_signal(comps, weights)
    assert sig["adme"] == pytest.approx(0.0, abs=1e-9)
    assert sig["pot"] > sig["adme"]


def test_aggregate_accepts_a_reduced_component_set():
    """Moving adme/qed into the gate requires aggregate to normalise over the survivors."""
    n = 100
    comps = {c: np.full(n, 0.5) for c in RW.COMPONENTS}
    sub = ["pot", "sel", "ad", "sa", "safety"]
    w = {c: 1.0 for c in RW.COMPONENTS}
    out = RW.aggregate(comps, w, np.ones(n), components=sub)
    assert out == pytest.approx(np.full(n, 0.5))


# ───────────────────────────────────────────────── #4 fail-closed ADMET

def test_missing_herg_raises_instead_of_scoring_as_perfect_safety():
    ep = {"solubility": np.zeros(3), "caco2": np.zeros(3), "microsome": np.ones(3),
          "ames": np.zeros(3), "cyp3a4": np.zeros(3)}          # hERG absent
    with pytest.raises(KeyError, match="herg"):
        RW.require_endpoints(ep, 3)


def test_non_finite_endpoint_raises():
    ep = {k: np.zeros(2) for k in RW.REQUIRED_ENDPOINTS}
    ep["herg"] = np.array([0.1, np.nan])
    with pytest.raises(ValueError, match="non-finite"):
        RW.require_endpoints(ep, 2)
