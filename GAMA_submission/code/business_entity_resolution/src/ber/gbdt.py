"""Gradient-boosted trees with grouped K-fold cross-fitting.

Backend `lightgbm` (cluster default) or `sklearn` HistGradientBoosting (no
extra dependency; used for local smoke tests). Folds come from `fold_gb`,
assigned per S1 in `prepare`, so all candidates of one S1 share a fold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .util import log, n_threads


def _backend(cfg) -> str:
    want = cfg.gbdt.backend
    if want == "lightgbm":
        try:
            import lightgbm  # noqa: F401
            return "lightgbm"
        except ImportError:
            log("WARNING: lightgbm not installed - falling back to sklearn HistGradientBoosting")
            return "sklearn"
    return "sklearn"


class Model:
    """`keep` masks out feature columns that were entirely missing at fit time
    (e.g. dense features in a sparse-only run). `keep=None` (a legacy sklearn
    pickle without its sidecar) re-derives the mask from all-missing columns."""

    def __init__(self, backend: str, obj, features: list[str], keep: np.ndarray | None):
        self.backend, self.obj, self.keep = backend, obj, keep
        self.all_features = list(features)
        self.features = [f for f, k in zip(features, keep) if k] if keep is not None else list(features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        keep = self.keep
        if keep is None:
            keep = np.array([not np.isnan(X[:, j]).all() for j in range(X.shape[1])], dtype=bool)
            want = getattr(self.obj, "n_features_in_", keep.sum())
            if keep.sum() != want:
                raise ValueError(f"legacy model without a .json sidecar expects {want} features, the all-missing "
                                 f"guess gives {int(keep.sum())} on this subset - refit it, or add the sidecar")
        if not keep.all():                         # column selection copies X: only when needed
            X = X[:, keep]
        if self.backend == "lightgbm":
            return self.obj.predict(X, num_iteration=self.obj.best_iteration or None).astype(np.float32)
        return self.obj.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, path: Path) -> None:
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "lightgbm":
            self.obj.save_model(str(path.with_suffix(".txt")))
        else:
            import pickle
            path.with_suffix(".pkl").write_bytes(pickle.dumps(self.obj))
        keep = self.keep if self.keep is not None else np.ones(len(self.all_features), dtype=bool)
        path.with_suffix(".json").write_text(json.dumps(
            {"backend": self.backend, "features": self.all_features, "keep": keep.tolist()}), encoding="utf-8")

    def importance(self) -> dict[str, float]:
        if self.backend == "lightgbm":
            imp = self.obj.feature_importance(importance_type="gain")
            return dict(sorted(zip(self.features, map(float, imp)), key=lambda x: -x[1]))
        return {}


def fit(cfg, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray, features: list[str]) -> Model:
    g = cfg.gbdt
    be = _backend(cfg)
    # column by column: np.isnan(X) on the whole matrix would need another GBs-sized array
    keep = np.array([not np.isnan(X[:, j]).all() for j in range(X.shape[1])], dtype=bool)
    if not keep.all():
        log(f"  dropping all-missing features: {[f for f, k in zip(features, keep) if not k]}")
        X, Xv = X[:, keep], Xv[:, keep]
    names = [f for f, k in zip(features, keep) if k]
    if be == "lightgbm":
        import lightgbm as lgb
        params = dict(objective="binary", learning_rate=g.learning_rate, num_leaves=g.num_leaves,
                      min_data_in_leaf=g.min_data_in_leaf, feature_fraction=g.feature_fraction,
                      bagging_fraction=g.bagging_fraction, bagging_freq=1, lambda_l2=g.lambda_l2,
                      num_threads=n_threads(g.n_threads), verbose=-1, seed=cfg.seed)
        dtr = lgb.Dataset(X, y, feature_name=names, free_raw_data=True)
        dva = lgb.Dataset(Xv, yv, reference=dtr)
        booster = lgb.train(params, dtr, num_boost_round=g.num_boost_round, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(g.early_stopping, verbose=False),
                                       lgb.log_evaluation(200)])
        log(f"  lightgbm best_iteration={booster.best_iteration}")
        return Model(be, booster, features, keep)
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        learning_rate=g.learning_rate, max_iter=min(g.num_boost_round, 500), max_leaf_nodes=min(g.num_leaves, 127),
        min_samples_leaf=g.min_data_in_leaf, l2_regularization=g.lambda_l2, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=min(g.early_stopping, 30), random_state=cfg.seed)
    clf.fit(np.vstack([X, Xv]), np.concatenate([y, yv]))
    return Model(be, clf, features, keep)


def load_model(stem: Path, features: list[str]) -> Model:
    """Model saved by Model.save (stem without suffix). `features` = the column
    list the model was fitted on, used when an older save has no sidecar."""
    import json
    side = stem.with_suffix(".json")
    meta = json.loads(side.read_text(encoding="utf-8")) if side.exists() else None
    if meta is not None:
        features = meta["features"]
    if stem.with_suffix(".txt").exists():
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(stem.with_suffix(".txt")))
        names = set(booster.feature_name())
        keep = (np.array(meta["keep"], dtype=bool) if meta is not None
                else np.array([f in names for f in features], dtype=bool))
        return Model("lightgbm", booster, features, keep)
    if stem.with_suffix(".pkl").exists():
        import pickle
        obj = pickle.loads(stem.with_suffix(".pkl").read_bytes())
        if meta is not None:
            keep = np.array(meta["keep"], dtype=bool)
        elif getattr(obj, "n_features_in_", len(features)) == len(features):
            keep = np.ones(len(features), dtype=bool)
        else:
            keep = None                            # legacy pickle: mask re-derived at predict time
        return Model("sklearn", obj, features, keep)
    raise FileNotFoundError(f"no saved model at {stem}.txt / {stem}.pkl")


def load_fold_models(out_dir: Path, tag: str, K: int, features: list[str]) -> list[Model]:
    return [load_model(out_dir / f"{tag}_fold{k}", features) for k in range(K)]


def predict_by_fold(models: list[Model], X: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Out-of-fold prediction for rows whose fold model is known: row i gets model fold[i]."""
    p = np.full(len(fold), np.nan, dtype=np.float32)
    for k, m in enumerate(models):
        idx = np.flatnonzero(fold == k)
        if idx.size:
            p[idx] = m.predict(X[idx])
    return p


def cross_fit(cfg, frame: pl.LazyFrame, features: list[str], train_s1_frac: float,
              out_dir: Path, tag: str, id_cols: tuple[str, ...] = ("q", "r")) -> tuple[pl.DataFrame, list[Model]]:
    """Grouped K-fold over `fold_gb`: model k trains on folds != k (optionally a
    random S1 subset) and predicts fold k. Returns (id_cols..., p) OOF + the K models."""
    K = int(cfg.data.n_folds_gbdt)
    ids = list(id_cols)
    oof, models = [], []
    for k in range(K):
        tr = frame.filter(pl.col("fold_gb") != k)
        if train_s1_frac < 1.0:
            tr = tr.filter((pl.col("q").hash(cfg.seed) % 10_000) < int(train_s1_frac * 10_000))
        tr = tr.select(["q", "label"] + features).collect()
        # early-stopping split: one more S1-disjoint slice of the training folds
        es = (tr["q"].hash(cfg.seed + 1) % 10).to_numpy() == 0
        X = tr.select(features).to_numpy()
        y = tr["label"].to_numpy()
        del tr
        log(f"{tag} fold {k}: fit on {int((~es).sum()):,} pairs (pos {y[~es].mean():.4f}), "
            f"early-stop on {int(es.sum()):,}")
        Xf, Xe = X[~es], X[es]
        del X                                      # hold one copy of the fold's rows, not three
        m = fit(cfg, Xf, y[~es], Xe, y[es], features)
        del Xf, Xe
        m.save(out_dir / f"{tag}_fold{k}")
        models.append(m)
        te = frame.filter(pl.col("fold_gb") == k).select(ids + features).collect()
        p = m.predict(te.select(features).to_numpy())
        oof.append(te.select(ids).with_columns(pl.Series("p", p)))
    imp = models[0].importance()
    if imp:
        top = list(imp.items())[:15]
        log(f"{tag} top features (gain): " + ", ".join(f"{a}={b:.0f}" for a, b in top))
    return pl.concat(oof), models


def predict_avg(models: list[Model], X: np.ndarray) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0).astype(np.float32)
