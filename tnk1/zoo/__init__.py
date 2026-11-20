"""The model zoo: every base regressor and molecular representation worth testing on TNK1.

WHY THIS PACKAGE EXISTS. The v3 HPC run established that the project's 10-member
heteroscedastic multi-task D-MPNN deep ensemble -- the model the architecture is built
around, and the critic the RL generator is optimised against -- ranks TNK1 compounds WORSE
than a plain Morgan-count random forest (Spearman 0.501 vs 0.686 over 5 paired outer
folds, per-row absolute error favouring RF at p = 8.6e-4), and that its uncertainty is
uninformative (error-sigma correlation -0.021, against +0.258 for the RF and +0.283 for a
Tanimoto GP). At 172 exact TNK1 measurements that is the expected outcome, not an anomaly.

Three modules, one contract:
    features.py    representations   list[str] -> (X, ok)
    regressors.py  point + variance predictors
    gp.py          Gaussian processes and kernel methods
    benchmark.py   leakage-safe nested-CV evaluation of any (model, representation) pair

THE CONTRACT. Every model in this package is duck-typed exactly as
`tnk1/baselines/models.py` already is, because `tnk1/rl/scorer.py` consumes a critic
through two methods and nothing else:

    m.fit(X, y) -> self
    m.predict(X, return_std=False) -> mu  |  (mu, sigma)

Honouring it means any zoo model can be dropped in as the RL critic through
`tnk1.zoo.critic.ZooCritic` without touching the scorer. Nothing here imports torch at
module level, so `import tnk1.zoo` works in the numpy-only environment and the heavy
members report themselves as unavailable rather than failing the import.
"""
from __future__ import annotations

from . import features
from .registry import (
    REGISTRY,
    ModelSpec,
    available,
    build,
    families,
    grid_points,
    register,
    shortlist,
)

# Imported for their REGISTRATION side effect, and in this order: `registry` holds only
# machinery, and these two fill it. Without them `REGISTRY` is empty and `available()`
# cheerfully reports that nothing is missing because nothing is there.
from . import regressors as _regressors   # noqa: E402,F401
from . import gp as _gp                   # noqa: E402,F401

__all__ = ["features", "REGISTRY", "ModelSpec", "available", "build", "families",
           "grid_points", "register", "shortlist"]
