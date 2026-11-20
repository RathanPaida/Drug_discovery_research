"""Combining predictive DISTRIBUTIONS, not just point predictions. Pure numpy/scipy.

Three separate jobs, which the literature says need three different objects, and which this
project currently conflates into one:

  1. **A scalar heteroscedastic scale per molecule**, for the conformal layer and for
     `r_pot`'s uncertainty penalty. That is the MOMENT-MATCHED sigma, `moment_match`
     below, which splits cleanly into a within-member (aleatoric) and a between-member
     (epistemic) term by the law of total variance.
  2. **An honest probabilistic score** (NLL, CRPS) and the objective that fits the
     weights. That is the MIXTURE, `mixture_logpdf` below. A Gaussian approximation to a
     bimodal mixture overstates its own NLL and hides the member disagreement -- which,
     for a project whose central failure was two model classes disagreeing by 1.8 log
     units, is the one thing that must not be hidden.
  3. **The weights themselves**, fitted by maximising the mixture's leave-one-out log
     score on the simplex: Yao, Vehtari, Simpson & Gelman, "Using Stacking to Average
     Bayesian Predictive Distributions", Bayesian Analysis 13(3):917-1007 (2018).

WHY NOT BAYESIAN MODEL AVERAGING. BMA weights are marginal-likelihood posteriors and are
only appropriate in the M-closed case, where one candidate IS the data-generating process.
Yao et al. are explicit that in the M-open and M-complete cases "BMA will asymptotically
select the one single model in the list that is closest in Kullback-Leibler divergence" --
i.e. it degenerates to model selection. None of a GBDT, a GP, an SVR or a D-MPNN is the
true dose-response mechanism of a kinase inhibitor, so this setting is unambiguously
M-open and BMA is the wrong tool. Stacking is the right one.

WHY NOT PRECISION-WEIGHTED AVERAGING. `sum(mu_k/sigma_k^2) / sum(1/sigma_k^2)` minimises
variance only if member errors are unbiased AND uncorrelated. Here they are neither: every
member is fitted on the same 172 molecules with overlapping feature blocks, so the errors
are strongly correlated, and the deep ensemble's sigma is measurably uninformative
(error-sigma correlation -0.021), so its precision is simply wrong. A bias shared across
members cannot be removed by any weighting of them. `precision_weighted` is provided
only so the comparison can be made rather than asserted, and its docstring says so.

ORDER MATTERS: POOL, THEN CALIBRATE. Rahaman & Thiery, "Uncertainty Quantification and
Deep Ensembles", NeurIPS 2021, show that temperature scaling applied to ensemble members
BEFORE averaging is the wrong order, and that pooling first then fitting one scalar can
halve ECE in the low-data regime. `PoolThenCalibrate` implements that order explicitly,
because the natural way to write this code is the wrong way round.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------- moment matching


def moment_match(mu, sigma, w=None) -> dict:
    """Combine member Gaussians into one Gaussian by matching the first two moments.

    `mu`, `sigma`: (K, n) member means and sds. `w`: (K,) simplex weights, uniform if None.

    Returns `mu`, `sigma`, and the two variance terms separately:

        mu*      = sum_k w_k mu_k
        var_ale  = sum_k w_k sigma_k^2                  within-member  (irreducible noise)
        var_epi  = sum_k w_k (mu_k - mu*)^2             between-member (disagreement)
        sigma*   = sqrt(var_ale + var_epi)

    This is algebraically identical to `sum_k w_k (sigma_k^2 + mu_k^2) - mu*^2` but
    numerically stabler (no cancellation of large squared means) and, more importantly,
    separately loggable. LOG THE TWO TERMS APART. The deep ensemble's error-sigma
    correlation of -0.021 is consistent with a total sigma dominated by a mis-learned
    aleatoric head; only the split says whether the between-member spread -- the thing an
    ensemble exists to produce -- carries any signal even when the total does not.

    `var_epi` is also the quantity to feed `rl_min.algos.common.pessimistic_reward`: it is
    model disagreement, which the generator can exploit, as opposed to assay noise, which
    it cannot.
    """
    mu = np.atleast_2d(np.asarray(mu, float))
    sigma = np.atleast_2d(np.asarray(sigma, float))
    K = mu.shape[0]
    w = np.full(K, 1.0 / K) if w is None else np.asarray(w, float)
    if w.sum() <= 0:
        raise ValueError("combination weights sum to zero")
    w = w / w.sum()
    mu_s = (w[:, None] * mu).sum(0)
    var_ale = (w[:, None] * np.maximum(sigma, 0.0) ** 2).sum(0)
    var_epi = (w[:, None] * (mu - mu_s[None, :]) ** 2).sum(0)
    return {"mu": mu_s, "var_ale": var_ale, "var_epi": var_epi,
            "sigma_ale": np.sqrt(var_ale), "sigma_epi": np.sqrt(var_epi),
            "sigma": np.sqrt(var_ale + var_epi), "weights": w}


def precision_weighted(mu, sigma):
    """Inverse-variance weighting. PROVIDED FOR COMPARISON, NOT RECOMMENDED -- see module
    docstring: it is optimal only for unbiased, uncorrelated members, and neither holds
    here."""
    mu = np.atleast_2d(np.asarray(mu, float))
    prec = 1.0 / np.maximum(np.atleast_2d(np.asarray(sigma, float)) ** 2, 1e-12)
    tot = prec.sum(0)
    return {"mu": (mu * prec).sum(0) / tot, "sigma": np.sqrt(1.0 / tot)}


# ----------------------------------------------------------------------- the mixture


def _norm_logpdf(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-9)
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((y - mu) / sigma) ** 2


def mixture_logpdf(y, mu, sigma, w=None):
    """log sum_k w_k N(y_i; mu_ki, sigma_ki) per point, by log-sum-exp.

    The honest predictive density of a weighted ensemble (Lakshminarayanan, Pritzel &
    Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep
    Ensembles", NeurIPS 2017, treat the ensemble as "a uniformly-weighted mixture model";
    the weighted form is the same object with w_k in place of 1/M).
    """
    from scipy.special import logsumexp
    y = np.asarray(y, float)
    mu = np.atleast_2d(np.asarray(mu, float))
    sigma = np.atleast_2d(np.asarray(sigma, float))
    K = mu.shape[0]
    w = np.full(K, 1.0 / K) if w is None else np.asarray(w, float)
    w = np.maximum(w, 0.0)
    w = w / max(w.sum(), 1e-12)
    lp = _norm_logpdf(y[None, :], mu, sigma)                    # (K, n)
    with np.errstate(divide="ignore"):
        lw = np.log(np.maximum(w, 1e-300))[:, None]
    return logsumexp(lp + lw, axis=0)


def mixture_nll(y, mu, sigma, w=None) -> float:
    return float(-np.mean(mixture_logpdf(y, mu, sigma, w)))


def mixture_crps(y, mu, sigma, w=None, n_grid=0) -> float:
    """CRPS of a Gaussian mixture.

    Closed form for the mixture (Grimit et al. 2006): CRPS of a mixture is a double sum of
    pairwise terms. Implemented directly rather than by sampling so the number is
    deterministic -- a stochastic score makes two stackers incomparable at the third
    decimal place, which at this N is the only place they differ.
    """
    from scipy.stats import norm
    y = np.asarray(y, float)
    mu = np.atleast_2d(np.asarray(mu, float))
    sd = np.maximum(np.atleast_2d(np.asarray(sigma, float)), 1e-9)
    K = mu.shape[0]
    w = np.full(K, 1.0 / K) if w is None else np.asarray(w, float)
    w = np.maximum(w, 0.0)
    w = w / max(w.sum(), 1e-12)

    def A(m, s):
        """E|X - 0| for X ~ N(m, s^2) = 2 s phi(m/s) + m (2 Phi(m/s) - 1)."""
        z = m / s
        return 2 * s * norm.pdf(z) + m * (2 * norm.cdf(z) - 1)

    term1 = np.zeros(len(y))
    for k in range(K):
        term1 += w[k] * A(y - mu[k], sd[k])
    term2 = np.zeros(len(y))
    for k in range(K):
        for l in range(K):
            term2 += w[k] * w[l] * A(mu[k] - mu[l], np.sqrt(sd[k] ** 2 + sd[l] ** 2))
    return float(np.mean(term1 - 0.5 * term2))


# ------------------------------------------------- Yao et al. log-score stacking


def pseudo_bma_plus(logpdf_oof, n_boot=1000, seed=0):
    """Pseudo-BMA+ weights (Yao et al. 2018, eq. 2.6). Bayesian-bootstrap regularised.

    `logpdf_oof` is (K, n): member k's out-of-fold log predictive density at point i.

        z_i^k   = log p(y_i | y_-i, M_k)
        alpha_b ~ Dirichlet(1, ..., 1)
        zbar_b^k = sum_i alpha_{i,b} z_i^k
        w_{k,b} = softmax_k( n * zbar_b^k )
        w_k     = mean_b w_{k,b}

    USE THIS RATHER THAN PLAIN PSEUDO-BMA AT THIS N. Yao et al.'s own caveat is the
    operative one: "One potential drawback of LOO is the large variance when the sample
    size is small... when the ratio of relative sample size to effective number of
    parameters is small, the weighting can be unstable." Pseudo-BMA+ exists to regularise
    precisely that, and the Dirichlet bootstrap is what damps it. Also serves as the warm
    start for `stack_log_score` below, as in the paper.
    """
    from scipy.special import softmax
    Z = np.atleast_2d(np.asarray(logpdf_oof, float))
    K, n = Z.shape
    rng = np.random.RandomState(seed)
    W = np.empty((n_boot, K))
    for b in range(n_boot):
        a = rng.dirichlet(np.ones(n))
        W[b] = softmax(n * (Z * a[None, :]).sum(axis=1))
    return W.mean(axis=0)


def stack_log_score(mu_oof, sigma_oof, y, w0=None, max_iter=500):
    """Maximise the MIXTURE's out-of-fold log score on the simplex (Yao et al., eq. 2.2).

        max_w  (1/n) sum_i log sum_k w_k N(y_i; mu_ki, sigma_ki)
        s.t.   w_k >= 0,  sum_k w_k = 1

    The log score, not squared error, and the paper gives the reason: stacking of MEANS is
    the beta=2 energy-score special case, and "the energy score with beta = 2 is not a
    strictly proper scoring rule and can give rise to identification problems". Every
    strictly proper local scoring rule is equivalent to the log score.

    Solved with SLSQP under the simplex constraint, warm-started from Pseudo-BMA+. Returns
    `(w, info)`.
    """
    from scipy.optimize import minimize
    mu = np.atleast_2d(np.asarray(mu_oof, float))
    sd = np.maximum(np.atleast_2d(np.asarray(sigma_oof, float)), 1e-9)
    y = np.asarray(y, float)
    K = mu.shape[0]
    lp = _norm_logpdf(y[None, :], mu, sd)              # (K, n)
    dens = np.exp(lp - lp.max(axis=0, keepdims=True))  # scaled for conditioning
    shift = lp.max(axis=0)

    def neg(w):
        m = np.maximum(dens.T @ w, 1e-300)
        return -float(np.mean(np.log(m) + shift))

    def grad(w):
        m = np.maximum(dens.T @ w, 1e-300)
        return -(dens / m[None, :]).mean(axis=1)

    if w0 is None:
        w0 = pseudo_bma_plus(lp)
    w0 = np.maximum(np.asarray(w0, float), 1e-6)
    w0 = w0 / w0.sum()
    res = minimize(neg, w0, jac=grad, method="SLSQP",
                   bounds=[(0.0, 1.0)] * K,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                 "jac": lambda w: np.ones_like(w)}],
                   options={"maxiter": max_iter, "ftol": 1e-10})
    w = np.maximum(res.x, 0.0)
    w = w / max(w.sum(), 1e-12)
    return w, {"success": bool(res.success), "message": str(res.message),
               "log_score": float(-res.fun), "warm_start": [float(v) for v in w0]}


# ----------------------------------------------------------------- pool, then calibrate


class PoolThenCalibrate:
    """Fit ONE sigma scale, AFTER the members have been combined. Order is the point.

    Rahaman & Thiery (NeurIPS 2021): "calibration methods such as temperature scaling need
    to be slightly tweaked when used with deep-ensembles and, crucially, need to be
    executed after the averaging process". Per-member scaling then averaging is the
    intuitive order and the wrong one.

    `mode="scalar"` fits `sigma_cal = s * sigma`, whose NLL-optimal `s` is the closed-form
    root-mean-square standardised residual -- the same quantity as
    `tnk1.uq.calibration.fit_temperature`, which this project already has; what is new is
    where it is applied. `s > 1` is the expected sign for an overconfident ensemble, and
    Levi, Gispan, Giladi & Fetaya (Sensors 22:5540, 2022) report learned values around
    1.1-1.2.

    `mode="affine"` fits `sigma_cal^2 = a + b * sigma^2`, which can reshape an
    uninformative sigma into a constant (b -> 0) instead of scaling noise. Two parameters
    is the limit at n=172; anything richer fits the calibration set.

    FIT ON DATA THE COMBINER DID NOT SEE. Inside the nested loop of the stacking protocol,
    never on the rows the weights were chosen on.
    """

    def __init__(self, mode="scalar"):
        self.mode = mode
        self.s_ = 1.0
        self.a_ = 0.0
        self.b_ = 1.0

    def fit(self, y, mu, sigma):
        y, mu = np.asarray(y, float), np.asarray(mu, float)
        sigma = np.maximum(np.asarray(sigma, float), 1e-9)
        if self.mode == "scalar":
            self.s_ = float(np.sqrt(max(np.mean(((y - mu) / sigma) ** 2), 1e-12)))
            return self
        if self.mode == "affine":
            from scipy.optimize import minimize
            r2 = (y - mu) ** 2
            v = sigma ** 2

            def neg(p):
                a, b = np.exp(p[0]), np.exp(p[1])
                vv = np.maximum(a + b * v, 1e-12)
                return float(np.mean(0.5 * np.log(2 * np.pi * vv) + 0.5 * r2 / vv))
            res = minimize(neg, np.log([max(r2.mean() * 0.1, 1e-6), 1.0]), method="Nelder-Mead",
                           options={"maxiter": 2000, "fatol": 1e-10})
            self.a_, self.b_ = (float(np.exp(res.x[0])), float(np.exp(res.x[1])))
            return self
        raise ValueError(f"unknown mode {self.mode!r}")

    def transform(self, sigma):
        sigma = np.maximum(np.asarray(sigma, float), 1e-9)
        if self.mode == "scalar":
            return self.s_ * sigma
        return np.sqrt(np.maximum(self.a_ + self.b_ * sigma ** 2, 1e-12))

    def report(self) -> dict:
        return ({"mode": "scalar", "s": self.s_} if self.mode == "scalar"
                else {"mode": "affine", "a": self.a_, "b": self.b_})
