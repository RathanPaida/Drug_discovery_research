"""Conformal variants beyond the split/normalised one already in `calibration.py`.

The test that matters most is `test_mondrian_fixes_the_conditional_coverage_split_misses`:
it constructs the exact failure mode this project's reported "93.3% empirical coverage"
can hide -- correct MARGINAL coverage assembled from one over-covered easy group and one
badly under-covered hard group, where the hard group is the novel-scaffold molecules the
pipeline actually makes decisions about.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tnk1.uq import conformal as C  # noqa: E402
from tnk1.uq.calibration import conformal_quantile  # noqa: E402


def heteroscedastic(n=800, seed=1):
    """A difficulty-driven noise scale: easy molecules are predicted well, hard ones badly.

    This is the real geometry. A near-neighbour of the training set is predicted to within
    a fraction of a log unit; a novel scaffold is not. A single interval width cannot serve
    both.
    """
    rng = np.random.default_rng(seed)
    d = rng.uniform(0, 1, n)                      # difficulty proxy (e.g. AD distance)
    y = rng.normal(7.0, 0.25 + 1.75 * d)
    mu = np.full(n, 7.0)
    sigma = np.ones(n)                            # an UNINFORMATIVE sigma, as observed
    return y, mu, sigma, d


# ----------------------------------------------------------------------- coverage

def test_picp_and_mpiw_are_what_they_say():
    y = np.array([1.0, 2.0, 3.0])
    assert C.picp(y, np.array([0.0, 0.0, 10.0]), np.array([2.0, 1.5, 11.0])) == pytest.approx(1 / 3)
    assert C.mpiw(np.zeros(3), np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)


def test_conditional_coverage_bins_and_reports_each_group():
    y, mu, sigma, d = heteroscedastic()
    lo, hi = mu - 2, mu + 2
    tab = C.conditional_coverage(y, lo, hi, d, n_bins=4)
    assert len(tab) == 4
    assert sum(r["n"] for r in tab) == len(y)
    # coverage must fall as difficulty rises, since the width is fixed
    picps = [r["picp"] for r in tab]
    assert picps[0] > picps[-1] + 0.1


def test_mondrian_fixes_the_conditional_coverage_split_misses():
    """THE point of the module.

    Split conformal with an uninformative sigma gives one width to every molecule, so its
    correct marginal coverage is assembled from an over-covered easy group and an
    under-covered hard group. Mondrian calibrates per difficulty bin: narrower where the
    model is reliable, wider where it is not. The hard group's coverage must improve, and
    the worst bin's shortfall must shrink.
    """
    y, mu, sigma, d = heteroscedastic()
    cal, te = slice(0, 400), slice(400, None)

    q = conformal_quantile(y[cal], mu[cal], sigma[cal], 0.10)
    lo_s, hi_s = mu[te] - q * sigma[te], mu[te] + q * sigma[te]

    m = C.MondrianConformal(alpha=0.10, n_bins=4).fit(y[cal], mu[cal], sigma[cal], d[cal])
    lo_m, hi_m = m.interval(mu[te], sigma[te], d[te])

    split = C.conditional_coverage(y[te], lo_s, hi_s, d[te], 4)
    mond = C.conditional_coverage(y[te], lo_m, hi_m, d[te], 4)

    # 1. split conformal has one width everywhere; Mondrian's widths increase
    assert len({round(r["mpiw"], 6) for r in split}) == 1
    assert mond[-1]["mpiw"] > mond[0]["mpiw"] * 1.5

    # 2. the hard bin is better covered
    assert mond[-1]["picp"] > split[-1]["picp"], (
        f"hard-bin coverage {mond[-1]['picp']:.3f} did not improve on "
        f"split conformal's {split[-1]['picp']:.3f}")

    # 3. the worst bin's distance from nominal shrinks
    worst_split = max(abs(r["picp"] - 0.90) for r in split)
    worst_mond = max(abs(r["picp"] - 0.90) for r in mond)
    assert worst_mond < worst_split


def test_mondrian_reduces_bins_and_warns_when_calibration_is_too_small():
    """At alpha=0.10 each bin needs >= 9 points for the required order statistic to exist.
    Asking for more bins than the data supports must say so, not emit a quantile of three
    residuals as if it were calibrated."""
    y, mu, sigma, d = heteroscedastic(n=20, seed=2)
    with pytest.warns(RuntimeWarning, match="cannot support"):
        m = C.MondrianConformal(alpha=0.10, n_bins=5).fit(y, mu, sigma, d)
    assert len(m.q_) < 5


def test_mondrian_falls_back_to_the_global_quantile_for_a_thin_bin():
    """Bins are QUANTILE-based, so a merely skewed difficulty cannot thin them -- each bin
    gets ~n/k points by construction. What does thin them is TIES: a difficulty proxy with
    many identical values (a discretised AD distance, say) makes the quantile edges
    coincide, `np.digitize` sends almost everything to one bin, and the others are left
    with too few residuals to form a quantile. That must fall back to the global quantile
    and SAY so, rather than emit a quantile of two numbers as if it were calibrated."""
    rng = np.random.default_rng(0)
    n = 60
    d = np.concatenate([np.zeros(57), np.ones(3)])     # 95% tied at one value
    y = rng.normal(7, 1, n)
    mu, sigma = np.full(n, 7.0), np.ones(n)
    with pytest.warns(RuntimeWarning, match="using the global quantile"):
        m = C.MondrianConformal(alpha=0.10, n_bins=3, min_per_bin=9).fit(y, mu, sigma, d)
    assert m.q_global_ is not None
    assert any(abs(v - m.q_global_) < 1e-12 for v in m.q_.values()), \
        "no bin fell back to the global quantile"


def test_mondrian_quantile_is_monotone_in_difficulty_for_this_geometry():
    y, mu, sigma, d = heteroscedastic()
    m = C.MondrianConformal(alpha=0.10, n_bins=4).fit(y, mu, sigma, d)
    qs = m.quantile(np.array([0.05, 0.35, 0.65, 0.95]))
    assert (np.diff(qs) > 0).all(), f"quantiles {qs} are not increasing with difficulty"


# ------------------------------------------------------------------------ CQR

def test_cqr_achieves_marginal_coverage_and_adapts_its_width():
    """Romano, Patterson & Candes (NeurIPS 2019). The interval WIDTH is learned from the
    data, which is the property that matters when the model's own sigma is uninformative --
    exactly this project's case."""
    rng = np.random.default_rng(3)
    n = 1200
    x = rng.uniform(0, 1, (n, 1))
    y = rng.normal(7.0, 0.2 + 2.0 * x[:, 0])
    tr, cal, te = slice(0, 600), slice(600, 900), slice(900, None)
    q = C.ConformalizedQuantileRegression(alpha=0.10, random_state=0)
    q.fit(x[tr], y[tr], x[cal], y[cal])
    lo, hi = q.interval(x[te])
    assert 0.82 < C.picp(y[te], lo, hi) < 0.98, "marginal coverage near nominal"
    # width must grow with x, because the noise does
    w = hi - lo
    from scipy import stats
    assert stats.spearmanr(x[te, 0], w).statistic > 0.5, \
        "CQR's width did not adapt to the heteroscedasticity"


def test_cqr_interval_is_ordered_and_predict_returns_the_midpoint():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, (300, 1))
    y = rng.normal(7, 0.5, 300)
    q = C.ConformalizedQuantileRegression(alpha=0.10, random_state=0)
    q.fit(x[:200], y[:200], x[200:], y[200:])
    lo, hi = q.interval(x[:20])
    assert (hi >= lo).all()
    mu, sd = q.predict(x[:20], return_std=True)
    assert np.allclose(mu, (lo + hi) / 2)
    assert (sd > 0).all()


def test_cqr_warns_when_the_calibration_set_cannot_support_alpha():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, (60, 1))
    y = rng.normal(7, 0.5, 60)
    q = C.ConformalizedQuantileRegression(alpha=0.10, random_state=0)
    with pytest.warns(RuntimeWarning, match="no coverage guarantee"):
        q.fit(x[:55], y[:55], x[55:], y[55:])     # 5 calibration points, needs >= 9


def test_cqr_interval_before_fit_raises():
    q = C.ConformalizedQuantileRegression()
    with pytest.raises(RuntimeError):
        q.interval(np.zeros((3, 1)))


# ------------------------------------------------------------------- CV+ / jackknife+

def test_cv_plus_spends_no_data_on_a_calibration_split():
    """Barber, Candes, Ramdas & Tibshirani (Ann. Statist. 2021). At N=172 split conformal
    hands 9+ molecules to calibration and removes them from training; CV+ uses every
    molecule for both. The residual vector must cover every row."""
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(6)
    X = rng.normal(0, 1, (120, 5))
    y = X @ np.array([1.0, -1, 0.5, 0, 0]) + rng.normal(0, 0.3, 120)

    def fit_predict(Xtr, ytr, Xte):
        return Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)

    resid, folds = C.cv_plus_intervals(fit_predict, X, y, n_folds=10)
    assert np.isfinite(resid).all(), "every row must get a leave-fold-out residual"
    assert len(folds) == 10
    assert sum(len(va) for _, va in folds) == len(y)


def test_cv_plus_from_parts_gives_ordered_intervals_with_near_nominal_coverage():
    rng = np.random.default_rng(7)
    n_test, K = 200, 10
    truth = rng.normal(7, 1, n_test)
    mu_per_fold = np.vstack([truth + rng.normal(0, 0.4, n_test) for _ in range(K)])
    resid_per_fold = [np.abs(rng.normal(0, 0.4, 12)) for _ in range(K)]
    lo, hi = C.cv_plus_from_parts(mu_per_fold, resid_per_fold, alpha=0.10)
    assert (hi >= lo).all()
    cov = C.picp(truth, lo, hi)
    assert cov > 0.80, f"CV+ coverage {cov:.3f} is far below the 1-2alpha floor"


def test_cv_plus_from_parts_refuses_an_all_nan_residual_set():
    with pytest.raises(ValueError):
        C.cv_plus_from_parts(np.zeros((2, 5)), [np.array([np.nan]), np.array([np.nan])])


# ------------------------------------------------------------------------- scores

def test_crps_gaussian_is_minimised_at_the_truth():
    y = np.zeros(1)
    best = C.crps_gaussian(y, np.zeros(1), np.full(1, 0.3))
    assert best < C.crps_gaussian(y, np.full(1, 2.0), np.full(1, 0.3))
    assert best < C.crps_gaussian(y, np.zeros(1), np.full(1, 5.0))


def test_interval_score_prices_undercoverage_and_width_together():
    y = np.array([1.0])
    tight_miss = C.interval_score(y, np.array([2.0]), np.array([2.1]), 0.10)
    wide_hit = C.interval_score(y, np.array([0.0]), np.array([2.0]), 0.10)
    assert tight_miss > wide_hit, "a miss must cost more than extra width"
    narrow_hit = C.interval_score(y, np.array([0.9]), np.array([1.1]), 0.10)
    assert narrow_hit < wide_hit, "among hits, narrower must score better"
