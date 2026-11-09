"""Regression tests for the fail-closed scoring contract and the calibration fixes.

Pure numpy -- no RDKit, no torch, no network -- so these run in the core environment.
"""
import math

import numpy as np
import pytest

from tnk1.rl import reward as R
from tnk1.uq.calibration import Calibrator, conformal_quantile, fit_temperature


# ------------------------------------------------------- fail-closed ADMET requirements

FULL_EP = {
    "solubility": np.array([-4.0]),
    "caco2": np.array([-5.0]),
    "microsome": np.array([10.0]),
    "herg": np.array([0.1]),
    "ames": np.array([0.1]),
    "cyp3a4": np.array([0.2]),
}


def test_require_endpoints_accepts_a_complete_set():
    e = R.require_endpoints(FULL_EP, 1)
    assert set(e) == set(R.REQUIRED_ENDPOINTS)
    assert e["herg"].dtype == float


@pytest.mark.parametrize("missing", R.REQUIRED_ENDPOINTS)
def test_missing_endpoint_raises_instead_of_scoring_perfectly(missing):
    """The old `ep.get(name, np.zeros(n))` turned a missing tox model into perfect safety."""
    ep = {k: v for k, v in FULL_EP.items() if k != missing}
    with pytest.raises(KeyError, match="missing ADMET endpoint"):
        R.require_endpoints(ep, 1)
    with pytest.raises(KeyError):
        R.adme_safety_components(ep, 1)


def test_none_valued_endpoint_is_treated_as_missing():
    ep = dict(FULL_EP, herg=None)
    with pytest.raises(KeyError, match="herg"):
        R.require_endpoints(ep, 1)


def test_wrong_length_and_nonfinite_endpoints_raise():
    with pytest.raises(ValueError, match="expected 3"):
        R.require_endpoints(FULL_EP, 3)
    with pytest.raises(ValueError, match="non-finite"):
        R.require_endpoints(dict(FULL_EP, ames=np.array([np.nan])), 1)


def test_zeros_would_have_scored_as_flawless_safety():
    """Documents the severity of the old default: p=0 across the tox panel is r_safety=1.0."""
    assert R.r_safety(np.array([0.0]), np.array([0.0]), np.array([0.0]))[0] == pytest.approx(1.0)
    assert R.hard_gate([True], [0], [400.0], [0.0], 600.0, p_herg=[0.0])[0] == 1.0


def test_adme_safety_components_match_the_inline_formula():
    out = R.adme_safety_components(FULL_EP, 1)
    expect_adme = R.r_adme(
        R.sigmoid(FULL_EP["solubility"], -5.0, 3.0),
        R.sigmoid(FULL_EP["caco2"], -5.15, 3.0),
        1 - R.sigmoid(np.log10(np.maximum(FULL_EP["microsome"], 1e-3)), np.log10(30.0), 3.0),
    )
    assert out["adme"] == pytest.approx(expect_adme)
    assert out["safety"] == pytest.approx(
        R.r_safety(FULL_EP["herg"], FULL_EP["ames"], FULL_EP["cyp3a4"])
    )


def test_herg_hard_gate_actually_bites():
    """The offline pipeline used to omit p_herg entirely, so this gate never fired there."""
    assert R.hard_gate([True], [0], [400.0], [0.1], 600.0, p_herg=[0.95], p_herg_max=0.85)[0] == 0.0
    assert R.hard_gate([True], [0], [400.0], [0.1], 600.0, p_herg=[0.10], p_herg_max=0.85)[0] == 1.0


# -------------------------------------------------------------------- novelty reward fix

def test_r_novelty_skeleton_check_uses_the_skeleton_set():
    """Passing one set for both arguments silently disabled the 14-char skeletal check."""
    known_full = {"ABCDEFGHIJKLMN-OPQRSTUVWX-Y"}
    skeletons = {"ABCDEFGHIJKLMN"}
    analog = "ABCDEFGHIJKLMN-ZZZZZZZZZZ-Z"          # same skeleton, different stereo/salt layer
    assert R.r_novelty([analog], known_full)[0] == 1.0            # exact check alone: passes
    assert R.r_novelty([analog], known_full, skeletons)[0] == 0.0  # skeletal check: caught


def test_r_novelty_rejects_unparseable_keys():
    assert list(R.r_novelty(["", None], set(), set())) == [0.0, 0.0]


def test_r_novelty_is_not_wired_into_the_aggregate():
    """Guards the documented status quo: novelty is a post-hoc audit, not a reward term."""
    assert "novelty" not in R.COMPONENTS


# ------------------------------------------------------------------------- calibration

def test_calibrator_honours_the_split_argument():
    """`split` used to be accepted and then ignored in favour of a hardcoded len(y)//2."""
    rng = np.random.RandomState(0)
    n = 100
    y = rng.normal(size=n); mu = np.zeros(n); sd = np.ones(n)
    a = Calibrator().fit(y, mu, sd, split=0.5, seed=1)
    b = Calibrator().fit(y, mu, sd, split=0.9, seed=1)
    assert (a.tau, a.q) != (b.tau, b.q)


def test_split_half_is_bit_identical_to_the_legacy_behaviour():
    """Locked Cycle-01 numbers must still reproduce (odd n included)."""
    for n in (45, 100, 7):
        rng = np.random.RandomState(n)
        y, mu, sd = rng.normal(size=n), np.zeros(n), np.ones(n)
        idx = np.random.RandomState(0).permutation(n)
        h = n // 2
        tau = fit_temperature(y[idx[:h]], mu[idx[:h]], sd[idx[:h]])
        q = conformal_quantile(y[idx[h:]], mu[idx[h:]], tau * sd[idx[h:]], 0.10)
        c = Calibrator().fit(y, mu, sd, split=0.5, seed=0)
        assert c.tau == pytest.approx(tau)
        assert c.q == pytest.approx(q)


@pytest.mark.parametrize("split", [0.0, 1.0, -0.1, 1.5])
def test_invalid_split_raises(split):
    y = np.zeros(10)
    with pytest.raises(ValueError, match="split must be"):
        Calibrator().fit(y, y, np.ones(10), split=split)


def test_conformal_quantile_warns_when_n_cannot_support_alpha():
    """At alpha=0.10 the 1-alpha guarantee needs n>=9; below that q_hat is not valid."""
    y = np.array([0.0, 1.0, 2.0]); mu = np.zeros(3); sd = np.ones(3)
    with pytest.warns(RuntimeWarning, match="too small"):
        q = conformal_quantile(y, mu, sd, alpha=0.10)
    assert q == pytest.approx(2.0)                      # clamped to max(s)
    with pytest.raises(ValueError, match="too small"):
        conformal_quantile(y, mu, sd, alpha=0.10, strict=True)


def test_conformal_quantile_is_silent_when_n_suffices():
    rng = np.random.RandomState(3)
    n = 50
    y, mu, sd = rng.normal(size=n), np.zeros(n), np.ones(n)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        q = conformal_quantile(y, mu, sd, alpha=0.10)
    k = math.ceil((n + 1) * 0.9)
    assert q == pytest.approx(np.sort(np.abs(y))[k - 1])


def test_empty_calibration_set_raises():
    with pytest.raises(ValueError, match="non-empty"):
        conformal_quantile(np.array([]), np.array([]), np.array([]))
