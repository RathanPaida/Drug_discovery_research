"""Turning pair probabilities into per-S1 match sets under macro F-beta.

1. calibrate   p' = sigmoid(a * logit(p) + b)       (tuned on out-of-fold data)
2. one-to-one  every S2/S3 record keeps only its best S1 - in the training
               data no record belongs to two S1 clusters
3. select      per S1, among its candidates sorted by p', choose how many to
               predict (0 included: a correct empty set scores 1.0):
     expected_f  maximise E[F_beta] by Monte-Carlo over Bernoulli(p') outcomes
     plugin      maximise (1+b2)*sum_{i<=k} p_i / (b2*sum_i p_i + k); E[F_0]=prod(1-p_i)
     threshold   keep p' >= t
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .metrics import macro_fbeta
from .util import log

MAX_K = 32


def calibrate(p: pl.Expr, a: float, b: float) -> pl.Expr:
    eps = 1e-6
    lo = (p.clip(eps, 1 - eps) / (1 - p.clip(eps, 1 - eps))).log()
    return 1.0 / (1.0 + (-(a * lo + b)).exp())


def one_to_one(pairs: pl.DataFrame) -> pl.DataFrame:
    best = pairs.filter(pl.col("p") == pl.col("p").max().over("r"))
    return best.filter(pl.col("q") == pl.col("q").min().over("r"))


def _group_matrix(pairs: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    g = pairs.sort(["q", "p"], descending=[False, True]).with_columns(
        pl.int_range(pl.len()).over("q").alias("_k"))
    g = g.filter(pl.col("_k") < MAX_K)
    qs = g["q"].unique(maintain_order=True)
    row = g.select(pl.col("q").rank("dense") - 1).to_series().to_numpy()
    P = np.zeros((len(qs), MAX_K), dtype=np.float32)
    P[row, g["_k"].to_numpy()] = g["p"].to_numpy()
    counts = np.bincount(row, minlength=len(qs))
    return g, P, counts


def _choose_k(P: np.ndarray, counts: np.ndarray, beta: float, method: str, samples: int, seed: int) -> np.ndarray:
    b2 = beta * beta
    n, L = P.shape
    best = np.zeros(n, dtype=np.int64)
    if method == "plugin":
        ks = np.arange(1, L + 1, dtype=np.float32)
        step = max(1, 50_000_000 // L)
        for s in range(0, n, step):
            Pb = P[s:s + step]
            tot = Pb.sum(1, keepdims=True)
            ef = np.concatenate([np.prod(1 - Pb, axis=1, keepdims=True),
                                 (1 + b2) * np.cumsum(Pb, 1) / (b2 * tot + ks)], axis=1)
            valid = np.arange(L + 1)[None, :] <= counts[s:s + step, None]
            best[s:s + step] = np.where(valid, ef, -np.inf).argmax(1)
        return best
    # expected_f: Monte-Carlo, one group of S1s per candidate count so each block
    # only carries its own width (most S1s have ~5 candidates, not MAX_K)
    rng = np.random.default_rng(seed)
    for c in np.unique(counts):
        c = int(c)
        rows = np.flatnonzero(counts == c)
        if c == 0:
            continue
        ks = np.arange(1, c + 1, dtype=np.float32)
        step = max(1, 50_000_000 // (samples * c))  # ~50M cells per block
        for s in range(0, len(rows), step):
            ix = rows[s:s + step]
            Pb = P[ix, :c]
            Y = rng.random((samples,) + Pb.shape, dtype=np.float32) < Pb[None]
            T = Y.sum(-1, dtype=np.int32)
            TP = np.cumsum(Y, -1, dtype=np.int32)
            F = (1 + b2) * TP / (b2 * T[..., None] + ks)
            ef = np.concatenate([(T == 0).mean(0)[:, None], F.mean(0)], axis=1)
            best[ix] = ef.argmax(1)
    return best


def select(pairs: pl.DataFrame, params: dict, beta: float, samples: int, seed: int) -> pl.DataFrame:
    """pairs: q, r, p (raw model probability). Returns the chosen (q, r)."""
    df = pairs.select("q", "r", calibrate(pl.col("p"), params["a"], params["b"]).alias("p"))
    if params.get("one_to_one", True):
        df = one_to_one(df)
    if params["method"] == "threshold":
        return df.filter(pl.col("p") >= params["t"]).select("q", "r")
    g, P, counts = _group_matrix(df)
    if not len(counts):
        return df.select("q", "r").head(0)
    k = _choose_k(P, counts, beta, params["method"], samples, seed)
    kq = pl.DataFrame({"q": g["q"].unique(maintain_order=True), "_best": k})
    return g.join(kq, on="q").filter(pl.col("_k") < pl.col("_best")).select("q", "r")


def grid(cfg) -> list[dict]:
    one = bool(cfg.decide.one_to_one)
    out = [{"method": "threshold", "a": 1.0, "b": 0.0, "t": t, "one_to_one": one}
           for t in np.round(np.arange(0.2, 0.96, 0.05), 2).tolist()]
    method = cfg.decide.method
    for a in (0.75, 1.0, 1.25, 1.5):
        for b in (-1.0, -0.5, 0.0, 0.5):
            out.append({"method": "plugin", "a": a, "b": b, "one_to_one": one})
            if method == "expected_f":
                out.append({"method": "expected_f", "a": a, "b": b, "one_to_one": one})
    return out


def tune(cfg, pairs: pl.DataFrame, truth: pl.DataFrame, queries: pl.Series, max_queries: int = 400_000) -> tuple[dict, list]:
    """Grid-search the decision parameters for macro F-beta on out-of-fold data
    (on at most `max_queries` S1s for speed)."""
    return tune_worlds(cfg, [(pairs, queries)], truth, max_queries=max_queries)


def tune_worlds(cfg, worlds: list[tuple[pl.DataFrame, pl.Series]], truth: pl.DataFrame,
                weights: list[float] | None = None, max_queries: int = 400_000) -> tuple[dict, list]:
    """Like `tune`, scoring each parameter set by the weighted mean of macro F-beta
    over several validation worlds (e.g. the full train set and a denser 'stress'
    world that looks like the test set). Each world: (pairs q, r, p; its S1s)."""
    dc = cfg.decide
    weights = weights or [1.0] * len(worlds)
    subs = []
    for pairs, queries in worlds:
        if len(queries) > max_queries:
            queries = queries.sample(max_queries, seed=cfg.seed)
        subs.append((pairs.join(pl.DataFrame({"q": queries}), on="q", how="semi"), queries))
    results = []
    for prm in grid(cfg):
        fs = [macro_fbeta(select(sub, prm, dc.beta, dc.mc_samples, cfg.seed), truth, qs, dc.beta)
              for sub, qs in subs]
        results.append((float(np.average(fs, weights=weights)), prm))
    results.sort(key=lambda x: -x[0])
    for f, prm in results[:5]:
        log(f"  decide grid: F={f:.5f} {prm}")
    return results[0][1], results
