"""Leakage-safe nested cross-validation over (model x representation) pairs.

This generalises `scripts/compare_critic.py`, which does the right thing for three
hardcoded baselines: fits on each outer fold's training keys, predicts that fold's test
rows, and compares against the deep ensemble's own out-of-fold predictions on exactly the
same compounds. The three things it does that matter, and that this module preserves:

  1. **Splits come from the frozen `splits_<version>.json`,** which is keyed by
     `inchikey_block1` and grouped by Bemis-Murcko scaffold. Compounds, not rows, are
     split, so the same molecule measured under IC50 and Ki cannot straddle the boundary,
     and a scaffold cannot appear on both sides.
  2. **Scalers are fitted on the training fold only.** Fitting a `DescriptorScaler` on all
     rows leaks the test fold's location and spread into every fit. At N=172 that is worth
     a visible chunk of apparent skill.
  3. **Hyperparameters are chosen on INNER folds.** Choosing them on the outer test fold
     is the classic optimistic-bias error, and with ~20 models in a sweep it is also a
     multiple-comparison problem: the best of 20 models selected on the test fold beats
     its own honest estimate by roughly the spread of the 20.

WHAT THIS MODULE ADDS

  - An out-of-fold prediction MATRIX, `(n_rows, n_models)`, assembled from the outer folds.
    That matrix is the input a stacker needs (`tnk1.ensembles.stacking`), and building it
    here -- rather than inside the stacker -- is what keeps the stack honest: the
    meta-learner only ever sees predictions made by models that did not see the row.
  - Per-row predictions, so models can be compared by paired per-row absolute error. With
    5 outer folds a Wilcoxon signed-rank test over fold means cannot return p < 0.0625 at
    all; the per-row test is the one with power (`compare_critic.py` says so too).
  - Optional predictive sigma per row, carried through so uncertainty quality can be
    scored with the same folds (`tnk1.uq.metrics`), not a separate split.

STILL THE CALLER'S JOB: deciding what counts as a win. `n=172` with 5 folds means a
Spearman standard error around 0.1, so differences below ~0.15 are noise. Report the
paired per-row test and the fold-level spread, and do not rank 20 models by a third
decimal place.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from .registry import REGISTRY, build, grid_points


# ---------------------------------------------------------------------------- the data


@dataclass
class FoldSpec:
    """One outer fold, as row indices into the TNK1 design matrix."""
    fold: int
    train: np.ndarray
    test: np.ndarray
    inner: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)


def folds_from_splits(splits: dict, keys: np.ndarray) -> list[FoldSpec]:
    """Translate the frozen compound-key splits into row indices.

    `keys` is the `inchikey_block1` column of the TNK1 design matrix, in matrix order. A
    compound with several rows (different assay types) contributes all of them to whichever
    side its key is on, which is the behaviour the splits were built for.
    """
    out = []
    for f in splits["cv"]:
        tr = np.where(np.isin(keys, list(f["train"])))[0]
        te = np.where(np.isin(keys, list(f["test"])))[0]
        inner = [(np.where(np.isin(keys, list(i["train"])))[0],
                  np.where(np.isin(keys, list(i["val"])))[0])
                 for i in f.get("inner", [])]
        out.append(FoldSpec(int(f["outer"]), tr, te, inner))
    return out


def scaled_matrix(X: np.ndarray, train_idx: np.ndarray, dense=None, scaler="robust"):
    """Return a copy of X whose `dense` columns are scaled by a TRAIN-FOLD-ONLY fit.

    `dense` is a boolean column mask (from `tnk1.zoo.features.dense_columns`) or a slice;
    None means "nothing to scale". Count and binary fingerprint columns are deliberately
    left untouched -- see `dense_columns` for why.

    `scaler="cdf"` uses the quantile/CDF transform instead of median+RobustScaler. That is
    what Chemprop does to its 200 global RDKit descriptors (normalised against a 100k-
    compound reference population), and the reason is small-N robustness: a CDF value
    means "the fraction of the reference population below this", which is bounded in
    [0, 1] by construction and cannot be thrown off by one outlier molecule the way a
    z-score or min-max can. At 172 molecules that bound is worth having.
    """
    if dense is None:
        return X
    Xo = X.astype(np.float64, copy=True)
    cols = np.asarray(dense)
    if cols.dtype != bool:                      # accept a slice for backwards compatibility
        idx = np.zeros(X.shape[1], dtype=bool)
        idx[cols] = True
        cols = idx
    if not cols.any():
        return X
    if scaler == "cdf":
        from ..baselines.features import CDFScaler
        sc = CDFScaler().fit(Xo[np.ix_(train_idx, cols)])
    else:
        from ..baselines.features import DescriptorScaler
        sc = DescriptorScaler().fit(Xo[np.ix_(train_idx, cols)])
    Xo[:, cols] = sc.transform(Xo[:, cols])
    return Xo.astype(np.float32)


# ------------------------------------------------------------------------ the inner CV


def select_params(spec, X, y, inner, dense, seed=0, metric="rmse"):
    """Pick hyperparameters by mean inner-fold score. Returns (params, score, n_fits).

    A model whose grid is empty costs no fits at all, which is why `grid_points({})`
    returns one empty dict rather than nothing.
    """
    pts = grid_points(spec.grid)
    if len(pts) == 1:
        return pts[0], float("nan"), 0
    best, best_s, n_fits = pts[0], float("inf"), 0
    for p in pts:
        scores = []
        for tr, va in inner:
            if len(tr) < 5 or not len(va):
                continue
            Xs = scaled_matrix(X, tr, dense)
            try:
                m = build(spec.name, {**p, **_seed_kw(spec, seed)}).fit(Xs[tr], y[tr])
                pred = np.asarray(_point(m, Xs[va]), dtype=float)
            except Exception as e:                     # a bad corner of the grid, not a bug
                warnings.warn(f"{spec.name} {p}: inner fit failed ({e})", RuntimeWarning)
                continue
            n_fits += 1
            scores.append(_score(y[va], pred, metric))
        s = float(np.mean(scores)) if scores else float("inf")
        if s < best_s:
            best, best_s = p, s
    return best, best_s, n_fits


def _seed_kw(spec, seed):
    """Pass a seed only to models that accept one, keyed off the spec rather than guessed."""
    return {"random_state": seed} if "seeded" in spec.tags else {}


def _point(m, X):
    p = m.predict(X)
    return p[0] if isinstance(p, tuple) else p


def _with_std(m, X):
    """-> (mu, sigma|None). Never fabricates a sigma for a model that has none."""
    try:
        out = m.predict(X, return_std=True)
    except TypeError:
        return np.asarray(_point(m, X), float), None
    if isinstance(out, tuple) and len(out) == 2:
        return np.asarray(out[0], float), np.asarray(out[1], float)
    return np.asarray(out, float), None


def _score(y, p, metric):
    if metric == "rmse":
        return float(np.sqrt(np.mean((np.asarray(p) - y) ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(np.asarray(p) - y)))
    if metric == "spearman":                   # minimised, so negate
        from scipy import stats
        r = stats.spearmanr(y, p)[0]
        return -(0.0 if not np.isfinite(r) else float(r))
    raise ValueError(f"unknown inner-CV metric {metric!r}")


# ------------------------------------------------------------------------- the outer CV


def run_model(name: str, X: np.ndarray, y: np.ndarray, folds: list[FoldSpec],
              dense: slice | None = None, seed: int = 42, tune: bool = True,
              inner_metric: str = "rmse"):
    """Evaluate one model over every outer fold.

    Returns `(rows, oof)`:
      rows  one dict per fold: metrics, chosen params, timings
      oof   dict with per-row arrays `mu`, `sigma` (NaN where the model gives none),
            `fold`, and a boolean `filled` mask -- NOT every row is necessarily covered,
            because a fold with too few compounds is skipped rather than fudged.
    """
    from ..uq.metrics import full_report
    spec = REGISTRY[name]
    n = len(y)
    oof = {"mu": np.full(n, np.nan), "sigma": np.full(n, np.nan),
           "fold": np.full(n, -1, dtype=int), "filled": np.zeros(n, dtype=bool)}
    rows = []
    for f in folds:
        if len(f.train) < 10 or len(f.test) < 3:
            rows.append({"model": name, "fold": f.fold, "n": len(f.test),
                         "status": "skipped: too few rows"})
            continue
        t0 = time.time()
        params, inner_s, n_fits = ((select_params(spec, X, y, f.inner, dense, seed, inner_metric))
                                   if (tune and f.inner) else (grid_points(spec.grid)[0],
                                                               float("nan"), 0))
        Xs = scaled_matrix(X, f.train, dense)
        m = build(name, {**params, **_seed_kw(spec, seed)}).fit(Xs[f.train], y[f.train])
        mu, sd = _with_std(m, Xs[f.test])
        oof["mu"][f.test] = mu
        oof["fold"][f.test] = f.fold
        oof["filled"][f.test] = True
        if sd is not None:
            oof["sigma"][f.test] = sd
        rep = full_report(y[f.test], mu, sd) if sd is not None else full_report(y[f.test], mu)
        rows.append({"model": name, "fold": f.fold, "n": int(len(f.test)),
                     "status": "ok", "params": str(params), "inner_score": inner_s,
                     "inner_fits": n_fits, "seconds": time.time() - t0, **rep})
    return rows, oof


def run_sweep(names, X_of, y, folds, seed=42, tune=True, inner_metric="rmse", log=print):
    """Evaluate many models, each on its own representation.

    `X_of(name) -> (X, dense_slice)` is a callable rather than a dict so the caller can
    featurise lazily and cache: `fp_union` is 8192 columns and recomputing it per model
    dominates the sweep's wall clock, while computing every representation up front costs
    memory for ones no selected model uses.

    Returns `(rows, oofs)` where `oofs[name]` is the per-row dict from `run_model`. A model
    that raises is RECORDED as a failure and the sweep continues -- one unavailable
    dependency must not void nineteen valid results -- but the failure appears in the
    output table rather than as a silent absence.
    """
    rows, oofs = [], {}
    for i, name in enumerate(names, 1):
        spec = REGISTRY[name]
        t0 = time.time()
        try:
            X, dense = X_of(name)
            r, o = run_model(name, X, y, folds, dense, seed, tune, inner_metric)
            rows.extend(r)
            oofs[name] = o
            ok = [x for x in r if x.get("status") == "ok"]
            if ok and log:
                log(f"[{i}/{len(names)}] {name:22} rho {np.mean([x['spearman'] for x in ok]):+.3f} "
                    f"rmse {np.mean([x['rmse'] for x in ok]):.3f}  "
                    f"({time.time() - t0:.1f}s, {len(ok)} folds)")
        except Exception as e:
            rows.append({"model": name, "fold": -1, "status": f"FAILED: {type(e).__name__}: {e}"})
            if log:
                log(f"[{i}/{len(names)}] {name:22} FAILED: {type(e).__name__}: {e}")
    return rows, oofs


def oof_matrix(oofs: dict, names=None):
    """Stack per-model OOF predictions into `(n_rows, n_models)` for a stacker.

    Returns `(P, S, used, rows_ok)`:
      P        point predictions, models in `used` order
      S        sigmas, NaN where a model provides none
      used     the model names actually included
      rows_ok  boolean mask of rows EVERY used model predicted

    `rows_ok` is the important one. A stacker fitted on a matrix with NaN holes either
    crashes or, worse, silently imputes -- and an imputed column is a model being credited
    with a prediction it never made. Restricting to the complete-case rows is the honest
    option, and the count is reported so a model that cost half the rows can be dropped.
    """
    used = [n for n in (names or oofs) if n in oofs]
    if not used:
        raise ValueError("no models to stack")
    P = np.column_stack([oofs[n]["mu"] for n in used])
    S = np.column_stack([oofs[n]["sigma"] for n in used])
    filled = np.column_stack([oofs[n]["filled"] for n in used])
    rows_ok = filled.all(axis=1) & np.isfinite(P).all(axis=1)
    return P, S, used, rows_ok
