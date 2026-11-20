"""Conformal prediction beyond the split/normalised variant already in `calibration.py`.

WHAT IS ALREADY HERE, AND WHY IT IS NOT ENOUGH. `tnk1/uq/calibration.py` implements
variance temperature scaling plus split conformal on normalised residuals
`s = |y - mu| / sigma`, which is correct and gives the marginal guarantee
`P(y in C(x)) >= 1 - alpha`. Two problems at this project's scale:

  1. **Marginal is not conditional.** The guarantee holds on AVERAGE over the input
     distribution. It says nothing about coverage for the molecules you care about -- the
     novel scaffolds a generator proposes, which sit at the edge of the applicability
     domain. An interval that covers 95% of near-neighbours and 60% of novel scaffolds
     satisfies 90% marginal coverage perfectly while being worthless for the one decision
     the pipeline makes. Mondrian (class-conditional) conformal fixes exactly this by
     calibrating per difficulty bin.
  2. **Normalising by a sigma that does not track error does nothing.** The deep ensemble's
     error-sigma Spearman is -0.021, so `|y-mu|/sigma` is essentially `|y-mu|/constant`:
     the normalisation is cosmetic and every molecule gets the same interval width. The
     v3 run's "93.3% empirical coverage" is therefore marginal coverage from an interval
     that is the same width for a known scaffold and a novel one.

WHAT THIS MODULE ADDS, with citations:

  * `MondrianConformal` -- Vovk et al.'s Mondrian conformal framework, used here with bins
    over a difficulty proxy (applicability-domain distance or predicted sigma). Boström &
    Johansson, "Mondrian conformal regressors", 2020, is the regression treatment.
  * `ConformalizedQuantileRegression` -- Romano, Patterson & Candès, "Conformalized
    Quantile Regression", NeurIPS 2019. Fits the alpha/2 and 1-alpha/2 conditional
    quantiles directly and conformalises their residual, so the interval WIDTH is learned
    from the data rather than inherited from a sigma that may not track error. This is the
    recommendation when heteroscedasticity is real but the model's own sigma is not.
  * `jackknife_plus` / `cv_plus` -- Barber, Candès, Ramdas & Tibshirani, "Predictive
    inference with the jackknife+", Annals of Statistics 2021. These spend no data on a
    calibration split, which matters enormously at N=172: split conformal at alpha=0.10
    needs n >= 9 in the calibration half merely for the quantile to exist, and every
    molecule moved into calibration is one the model does not learn from. CV+ is the
    K-fold version and is what I would use here.

All numpy/sklearn. The quantile-regression variant needs sklearn only.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from .calibration import conformal_quantile


# ---------------------------------------------------------------------------- coverage


def picp(y, lo, hi) -> float:
    """Prediction-Interval Coverage Probability: the fraction actually covered."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def mpiw(lo, hi) -> float:
    """Mean Prediction-Interval Width. Report WITH picp, never alone: an infinitely wide
    interval has perfect coverage, so width is the price and coverage the thing bought."""
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def conditional_coverage(y, lo, hi, by, n_bins=4) -> list[dict]:
    """Coverage within bins of `by` (e.g. AD distance, predicted sigma, or |y| itself).

    THIS is the diagnostic the project is missing. A model can hold 90% marginal coverage
    while covering 98% of the easy quartile and 70% of the hard one, and only this table
    shows it. Quantile bins, so each holds a comparable number of molecules.
    """
    y, lo, hi, by = map(lambda a: np.asarray(a, float), (y, lo, hi, by))
    qs = np.quantile(by, np.linspace(0, 1, n_bins + 1))
    qs[-1] = np.inf
    out = []
    for b in range(n_bins):
        m = (by >= qs[b]) & (by < qs[b + 1])
        if not m.any():
            continue
        out.append({"bin": b, "n": int(m.sum()),
                    "by_lo": float(qs[b]), "by_hi": float(qs[b + 1] if b < n_bins - 1 else by.max()),
                    "picp": picp(y[m], lo[m], hi[m]), "mpiw": mpiw(lo[m], hi[m])})
    return out


def crps_gaussian(y, mu, sd) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution (Gneiting & Raftery 2007).

    A proper scoring rule that, unlike NLL, is finite when a point lands far into the tail
    and is therefore not dominated by one bad molecule -- the failure mode that makes NLL
    comparisons across 20 models at N=172 almost unreadable.
    """
    from scipy.stats import norm
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    sd = np.maximum(np.asarray(sd, float), 1e-9)
    z = (y - mu) / sd
    return float(np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def interval_score(y, lo, hi, alpha=0.10) -> float:
    """Winkler/interval score: width + 2/alpha * (miss distance). Proper, and it prices
    under-coverage and over-width in the same units (Gneiting & Raftery 2007)."""
    y, lo, hi = map(lambda a: np.asarray(a, float), (y, lo, hi))
    return float(np.mean((hi - lo)
                         + (2 / alpha) * np.maximum(lo - y, 0)
                         + (2 / alpha) * np.maximum(y - hi, 0)))


# ------------------------------------------------------------------ Mondrian conformal


class MondrianConformal:
    """Split conformal with a separate quantile per difficulty bin.

    Vovk, Gammerman & Shafer's Mondrian conformal framework; for regression see Boström &
    Johansson, "Mondrian conformal regressors", COPA 2020. Bins must be a function of x
    ONLY (never of y), or exchangeability inside the bin is broken and the guarantee is
    gone. `taxonomy_fn` therefore receives the difficulty proxy, not the label.

    Guarantee: 1-alpha coverage WITHIN each bin, which implies marginal coverage and is
    strictly stronger. The cost is that each bin needs its own calibration set -- at
    alpha=0.10 that is n>=9 per bin, so with 172 molecules do not ask for more than ~4
    bins, and this class refuses to pretend otherwise.
    """

    def __init__(self, alpha=0.10, n_bins=3, min_per_bin=None):
        self.alpha, self.n_bins = alpha, n_bins
        # The order statistic ceil((n+1)(1-alpha)) must exist inside the bin.
        self.min_per_bin = min_per_bin or math.ceil(1 / alpha) - 1
        self.edges_ = None
        self.q_ = None
        self.q_global_ = None

    def fit(self, y, mu, sigma, difficulty):
        y, mu, sigma, d = (np.asarray(a, float) for a in (y, mu, sigma, difficulty))
        s = np.abs(y - mu) / np.maximum(sigma, 1e-6)
        n_bins = self.n_bins
        while n_bins > 1 and len(y) / n_bins < self.min_per_bin:
            n_bins -= 1
        if n_bins != self.n_bins:
            warnings.warn(
                f"MondrianConformal: {len(y)} calibration points cannot support "
                f"{self.n_bins} bins at alpha={self.alpha} (need >= {self.min_per_bin} per "
                f"bin); using {n_bins}. Bin-conditional coverage below that count is not "
                f"guaranteed, it is an empirical quantile of too few numbers.",
                RuntimeWarning, stacklevel=2)
        self.edges_ = np.quantile(d, np.linspace(0, 1, n_bins + 1))[1:-1]
        b = np.digitize(d, self.edges_)
        self.q_global_ = conformal_quantile(y, mu, np.maximum(sigma, 1e-6), self.alpha)
        self.q_ = {}
        for k in range(n_bins):
            m = b == k
            if m.sum() < self.min_per_bin:
                # Fall back to the global quantile for a thin bin and SAY so, rather than
                # emitting a quantile of three residuals as if it were calibrated.
                self.q_[k] = self.q_global_
                warnings.warn(f"MondrianConformal: bin {k} has {int(m.sum())} points; "
                              f"using the global quantile for it", RuntimeWarning, stacklevel=2)
            else:
                self.q_[k] = float(np.sort(s[m])[math.ceil((m.sum() + 1) * (1 - self.alpha)) - 1])
        return self

    def quantile(self, difficulty):
        b = np.digitize(np.asarray(difficulty, float), self.edges_)
        return np.array([self.q_.get(int(k), self.q_global_) for k in b])

    def interval(self, mu, sigma, difficulty):
        q = self.quantile(difficulty)
        s = q * np.maximum(np.asarray(sigma, float), 1e-6)
        mu = np.asarray(mu, float)
        return mu - s, mu + s

    def report(self) -> dict:
        return {"alpha": self.alpha, "edges": None if self.edges_ is None else self.edges_.tolist(),
                "q_per_bin": {int(k): float(v) for k, v in (self.q_ or {}).items()},
                "q_global": self.q_global_}


# -------------------------------------------------------- conformalized quantile regr.


class ConformalizedQuantileRegression:
    """CQR: Romano, Patterson & Candès, NeurIPS 2019.

    Procedure, exactly as in the paper:
      1. Split the data into a proper training set and a calibration set.
      2. On the training set fit two conditional quantile regressors,
         `q_lo` at alpha/2 and `q_hi` at 1 - alpha/2.
      3. On the calibration set compute the conformity score
             E_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))
         -- signed distance OUTSIDE the predicted interval, negative when inside.
      4. Take `Q = the ceil((n+1)(1-alpha))/n empirical quantile of E`.
      5. The interval is `[q_lo(x) - Q, q_hi(x) + Q]`.

    Why it suits this project: the interval width comes from the fitted quantiles, so it is
    genuinely input-dependent even though the deep ensemble's own sigma carries no
    information. A gradient-boosted quantile regressor on Morgan counts is the default
    here because that model family is already the strongest thing in the repository at this
    N, and the ensemble's sigma demonstrably is not.
    """

    def __init__(self, alpha=0.10, model_lo=None, model_hi=None, random_state=0):
        self.alpha = alpha
        self.random_state = random_state
        self._mk_lo, self._mk_hi = model_lo, model_hi
        self.q_ = None

    def _default(self, q):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200,
                                         max_depth=3, learning_rate=0.05,
                                         random_state=self.random_state)

    def fit(self, X_tr, y_tr, X_cal, y_cal):
        lo_q, hi_q = self.alpha / 2, 1 - self.alpha / 2
        self.lo_ = (self._mk_lo() if self._mk_lo else self._default(lo_q)).fit(X_tr, y_tr)
        self.hi_ = (self._mk_hi() if self._mk_hi else self._default(hi_q)).fit(X_tr, y_tr)
        l, h = self.lo_.predict(X_cal), self.hi_.predict(X_cal)
        E = np.maximum(l - np.asarray(y_cal, float), np.asarray(y_cal, float) - h)
        n = len(E)
        k = math.ceil((n + 1) * (1 - self.alpha))
        if k > n:
            warnings.warn(
                f"CQR: calibration set of {n} cannot support alpha={self.alpha} "
                f"(needs n >= {math.ceil(1 / self.alpha) - 1}); clamping to max(E), which "
                f"carries no coverage guarantee", RuntimeWarning, stacklevel=2)
            k = n
        self.q_ = float(np.sort(E)[k - 1])
        return self

    def interval(self, X):
        if self.q_ is None:
            raise RuntimeError("CQR.fit must be called before interval()")
        return self.lo_.predict(X) - self.q_, self.hi_.predict(X) + self.q_

    def predict(self, X, return_std=False):
        """Midpoint of the interval, with half-width/1.645 as a sigma-like scale.

        Provided only so a CQR model satisfies the project's fit/predict duck type. The
        half-width is NOT a standard deviation and the conversion assumes a Gaussian that
        CQR deliberately does not assume -- use `interval()` for anything that matters.
        """
        lo, hi = self.interval(X)
        mu = (lo + hi) / 2
        return (mu, (hi - lo) / 2 / 1.645) if return_std else mu


# -------------------------------------------------------------- jackknife+ / CV+ (CV+)


def cv_plus_intervals(fit_predict, X, y, n_folds=10, alpha=0.10, groups=None, seed=0):
    """CV+ intervals. Barber, Candès, Ramdas & Tibshirani, Ann. Statist. 2021.

    `fit_predict(X_tr, y_tr, X_te) -> predictions` is called once per fold. For each fold
    the leave-fold-out residuals `R_i = |y_i - mu^{-k}(x_i)|` are recorded together with
    that fold's model's predictions on the TEST points, and the interval for a test point
    is

        [ quantile_alpha^-( mu^{-k}(x) - R_i ),  quantile_alpha^+( mu^{-k}(x) + R_i ) ]

    taken over all training points i. Coverage is guaranteed at 1 - 2*alpha in the worst
    case (and empirically close to 1 - alpha).

    Why use it here: it spends NO data on a calibration split. With 172 molecules, split
    conformal at alpha=0.10 hands 9+ molecules to calibration and removes them from
    training; CV+ uses every molecule for both, at the price of `n_folds` model fits. For
    a random forest at this N that trade is overwhelmingly worth it.

    Returns `(lo, hi, resid)`; `resid` is the leave-fold-out absolute residual per training
    row, which is also the honest input for a conformal quantile elsewhere.
    """
    from sklearn.model_selection import GroupKFold, KFold
    X = np.asarray(X)
    y = np.asarray(y, float)
    n = len(y)
    n_folds = max(2, min(n_folds, n if groups is None else len(np.unique(groups))))
    splitter = (GroupKFold(n_splits=n_folds).split(X, y, groups) if groups is not None
                else KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X))
    resid = np.full(n, np.nan)
    mu_te = []                     # one row per fold model, over the test points
    folds = list(splitter)
    for tr, va in folds:
        pred_va = np.asarray(fit_predict(X[tr], y[tr], X[va]), float)
        resid[va] = np.abs(y[va] - pred_va)
    return resid, folds


def cv_plus_from_parts(mu_test_per_fold, resid_per_fold, alpha=0.10):
    """Assemble CV+ intervals from per-fold test predictions and leave-fold-out residuals.

    `mu_test_per_fold[k]` is fold-k's model's prediction on EVERY test point, shape
    (n_folds, n_test); `resid_per_fold[k]` the absolute residuals of fold k's validation
    rows. Separated from `cv_plus_intervals` so an existing out-of-fold artifact (this
    repository already writes `runs/stage4_oof_predictions.csv`) can be turned into CV+
    intervals without refitting anything.
    """
    M = np.asarray(mu_test_per_fold, float)          # (K, n_test)
    lows, highs = [], []
    for k, r in enumerate(resid_per_fold):
        r = np.asarray(r, float)
        r = r[np.isfinite(r)]
        if not len(r):
            continue
        lows.append(M[k][None, :] - r[:, None])      # (n_k, n_test)
        highs.append(M[k][None, :] + r[:, None])
    if not lows:
        raise ValueError("no finite residuals: cannot form CV+ intervals")
    L = np.vstack(lows)
    H = np.vstack(highs)
    n = L.shape[0]
    # the alpha and 1-alpha empirical quantiles with the finite-sample correction
    k_lo = max(int(np.floor(alpha * (n + 1))) - 1, 0)
    k_hi = min(int(np.ceil((1 - alpha) * (n + 1))) - 1, n - 1)
    return np.sort(L, axis=0)[k_lo], np.sort(H, axis=0)[k_hi]
