"""Gaussian processes for TNK1. Registers itself on import.

WHY GPs GET THEIR OWN MODULE. A GP is the one model family here that returns a predictive
distribution by construction rather than by a bolted-on wrapper, and the project already
has evidence that this matters: the Tanimoto-kernel GP's error-sigma Spearman is +0.283,
the best in the repository, against -0.021 for the 10-member deep ensemble whose sigma
drives `r_pot`, the conformal interval, the applicability domain and EI/UCB acquisition. At
172 molecules an exact GP is also computationally free -- a 172x172 Cholesky -- so there is
no efficiency argument for approximating anything.

THE KERNEL IS THE MODEL. On molecules the choice of kernel is the choice of what
"similar" means:

  * `MinMaxKernel` (= Tanimoto on count vectors) -- sum(min) / sum(max). Positive definite
    on non-negative vectors, so it is a genuine kernel and not a similarity heuristic
    (Ralaivola, Swamidass, Saigo & Baldi, "Graph kernels for chemical informatics",
    Neural Networks 2005; the molecular-GP treatment is Griffiths et al., "GAUCHE: A
    Library for Gaussian Processes in Chemistry", NeurIPS 2023). It encodes substructure
    OVERLAP directly, which is why it beats an RBF asked to rediscover the same thing from
    2048 sparse counts.
  * `TanimotoKernel` -- the inner-product form <a,b>/(|a|^2+|b|^2-<a,b>), which is what
    `tnk1/baselines/models.py` and `tnk1/uq/applicability.py` already use. Registered
    separately because on count vectors the two forms are NOT identical and the sweep
    should say which is better here rather than inheriting a choice.
  * RBF / Matern on scaled descriptors -- the right kernel for the continuous
    physicochemical block, and a genuinely different inductive bias from substructure
    overlap, which makes it a useful ensemble member rather than a redundant one.

sklearn, NOT gpytorch, is the backend for everything in this module. That is deliberate:
it makes every GP here runnable and testable in the numpy-only environment, where
`gpytorch` and `torch` are absent. `tnk1.baselines.models.TanimotoGP` remains the
gpytorch implementation and stays registered for continuity with the published Stage-2
number.
"""
from __future__ import annotations

import numpy as np

# sklearn is a CORE dependency (requirements-core.txt), so importing it at module level is
# safe in every supported environment -- unlike torch/gpytorch, which are imported lazily
# at the bottom of this file.
#
# These kernels MUST subclass sklearn's `Kernel`. The composition operators check
# `isinstance(other, Kernel)` and, failing that, wrap the operand in a `ConstantKernel`:
# a hand-rolled duck-typed kernel is therefore silently turned into
# `ConstantKernel(<kernel object>)`, whose `theta` calls `np.log` on it and raises
# "loop of ufunc does not support argument 0 of type MinMaxKernel". Duck typing does not
# reach here, and the failure appears only once a GP is actually fitted.
from sklearn.gaussian_process.kernels import Kernel

from .registry import ModelSpec, register


# ------------------------------------------------------------------ molecular kernels


class MinMaxKernel(Kernel):
    """k(a, b) = sum_i min(a_i, b_i) / sum_i max(a_i, b_i), the count-vector Tanimoto.

    Parameter-free, so it carries no hyperparameters to optimise: scale and noise come
    from the `ConstantKernel` and `WhiteKernel` it is multiplied and added to. The
    zero-width gradient returned under `eval_gradient=True` is what tells sklearn's
    optimiser there is nothing here to tune.

    `__call__` loops over the query rows rather than broadcasting into a 3-D array: at
    172x172 either is instant, but scoring a 200k-molecule generated pool against 172
    training rows would allocate 200000*172*2048 floats in the broadcast form and die.
    """

    def __init__(self):
        pass                      # no hyperparameters; sklearn introspects __init__ args

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, float)
        Y = X if Y is None else np.asarray(Y, float)
        K = np.empty((len(X), len(Y)))
        for i in range(len(X)):
            mn = np.minimum(X[i], Y).sum(axis=1)
            mx = np.maximum(X[i], Y).sum(axis=1)
            K[i] = mn / np.maximum(mx, 1e-12)
        if eval_gradient:
            return K, np.empty((len(X), len(Y), 0))
        return K

    def diag(self, X):
        return np.ones(len(X))

    def is_stationary(self):
        return False

    def __repr__(self):
        return "MinMax()"


class TanimotoKernel(MinMaxKernel):
    """k(a, b) = <a,b> / (|a|^2 + |b|^2 - <a,b>).

    The form already used by `tanimoto_sim` and the gpytorch GP. On binary vectors it
    coincides with MinMax; on COUNT vectors it does not, and which is better is an
    empirical question this project has never asked.
    """

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, float)
        Y = X if Y is None else np.asarray(Y, float)
        ab = X @ Y.T
        aa = (X * X).sum(1)[:, None]
        bb = (Y * Y).sum(1)[None, :]
        K = ab / np.maximum(aa + bb - ab, 1e-12)
        if eval_gradient:
            return K, np.empty((len(X), len(Y), 0))
        return K

    def __repr__(self):
        return "Tanimoto()"


# ------------------------------------------------------------------------- the models


class SkGP:
    """sklearn `GaussianProcessRegressor` behind the project's fit/predict contract.

    `normalize_y=True` is on by default and matters more than it looks: a GP's prior mean
    is zero, so without centring the model shrinks predictions toward pIC50 = 0 (an IC50 of
    1 molar) in regions far from the training data. That is a catastrophic prior for a
    generated molecule at the edge of the applicability domain -- exactly the molecules the
    RL critic is asked about -- and it is the kind of silent optimism/pessimism the audit
    of this pipeline was about.
    """

    def __init__(self, kernel="minmax", alpha=1e-6, normalize_y=True, n_restarts=2,
                 length_scale=1.0, nu=2.5, random_state=0):
        self.kernel_name = kernel
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.n_restarts = int(n_restarts)
        self.length_scale = length_scale
        self.nu = nu
        self.random_state = random_state
        self.m_ = None

    def _build_kernel(self):
        from sklearn.gaussian_process.kernels import (ConstantKernel, Matern, RBF,
                                                      WhiteKernel)
        scale = ConstantKernel(1.0, (1e-3, 1e3))
        noise = WhiteKernel(0.1, (1e-6, 1e1))
        if self.kernel_name == "minmax":
            return scale * MinMaxKernel() + noise
        if self.kernel_name == "tanimoto":
            return scale * TanimotoKernel() + noise
        if self.kernel_name == "rbf":
            return scale * RBF(self.length_scale, (1e-2, 1e4)) + noise
        if self.kernel_name == "matern":
            return scale * Matern(self.length_scale, (1e-2, 1e4), nu=self.nu) + noise
        if self.kernel_name == "minmax_rbf":
            # Sum of two kernels = sum of two independent GP contributions: substructure
            # overlap AND descriptor proximity. The natural way to use both blocks of a
            # mixed representation without concatenating incomparable scales.
            return (scale * MinMaxKernel() + ConstantKernel(1.0, (1e-3, 1e3))
                    * RBF(self.length_scale, (1e-2, 1e4)) + noise)
        raise ValueError(f"unknown kernel {self.kernel_name!r}")

    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        self.m_ = GaussianProcessRegressor(
            kernel=self._build_kernel(), alpha=self.alpha,
            normalize_y=self.normalize_y, n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state)
        self.m_.fit(np.asarray(X, float), np.asarray(y, float))
        return self

    def predict(self, X, return_std=False):
        out = self.m_.predict(np.asarray(X, float), return_std=return_std)
        if not return_std:
            return out
        mu, sd = out
        return mu, np.maximum(sd, 1e-6)

    @property
    def noise_sd(self):
        """The learned likelihood noise, i.e. the GP's own aleatoric estimate.

        `tnk1.zoo.critic.ZooCritic(aleatoric="model")` reads this, which is the honest
        route: a GP ESTIMATES the irreducible noise as a hyperparameter instead of having
        it inferred from held-out residuals.
        """
        if self.m_ is None:
            return None
        for p, v in zip(self.m_.kernel_.hyperparameters, self.m_.kernel_.theta):
            if "noise_level" in p.name:
                return float(np.sqrt(np.exp(v)))
        return None

    def log_marginal_likelihood(self):
        """The GP's model-evidence score. A principled, held-out-free way to compare
        kernels on 172 points, where a validation split costs more than it tells you."""
        return None if self.m_ is None else float(self.m_.log_marginal_likelihood_value_)


register(ModelSpec("gp_minmax", lambda p: SkGP(kernel="minmax", **p), rep="ecfp4_count",
                   gives_std=True, family="gp", cost=1.0, needs=("sklearn",),
                   cite="Ralaivola et al. 2005; Griffiths et al. (GAUCHE) 2023",
                   tags=("core", "uq", "seeded")))
register(ModelSpec("gp_tanimoto", lambda p: SkGP(kernel="tanimoto", **p), rep="ecfp4_count",
                   gives_std=True, family="gp", cost=1.0, needs=("sklearn",),
                   cite="this project's Stage-2 baseline 2.4, sklearn backend",
                   tags=("core", "uq", "control", "seeded")))
register(ModelSpec("gp_matern", lambda p: SkGP(kernel="matern", **p), rep="rdkit_desc_all",
                   grid={"nu": [1.5, 2.5], "length_scale": [1.0, 10.0]},
                   gives_std=True, family="gp", cost=1.0, needs=("sklearn",),
                   cite="Matern kernel on scaled descriptors", tags=("core", "uq", "seeded")))
register(ModelSpec("gp_rbf", lambda p: SkGP(kernel="rbf", **p), rep="rdkit_desc_all",
                   grid={"length_scale": [1.0, 10.0, 100.0]},
                   gives_std=True, family="gp", cost=1.0, needs=("sklearn",),
                   cite="RBF on scaled descriptors", tags=("probe", "uq", "seeded")))
register(ModelSpec("gp_minmax_rbf", lambda p: SkGP(kernel="minmax_rbf", **p),
                   rep="ecfp4_desc200",
                   gives_std=True, family="gp", cost=1.5, needs=("sklearn",),
                   cite="additive substructure + descriptor kernel",
                   tags=("core", "uq", "seeded")))


# ------------------------------------------------------------- the gpytorch original


def _gpytorch_tanimoto(p):
    from ..baselines.models import TanimotoGP
    return TanimotoGP(iters=int(p.get("iters", 200)), lr=float(p.get("lr", 0.05)),
                      device=p.get("device", "cpu"))


register(ModelSpec("gp_tanimoto_gpytorch", _gpytorch_tanimoto, rep="ecfp4_count",
                   gives_std=True, family="gp", cost=4.0, needs=("torch", "gpytorch"),
                   cite="tnk1.baselines.models.TanimotoGP -- the published Stage-2 2.4 "
                        "and the Stage-8 audit model",
                   tags=("control", "uq", "slow")))
