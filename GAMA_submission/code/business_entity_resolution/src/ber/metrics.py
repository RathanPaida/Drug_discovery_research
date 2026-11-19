"""Competition metric (macro F-beta over S1 entities, singletons included) and
candidate-set diagnostics. All frames use integer ids: q = S1 idx, r = record idx.
"""
from __future__ import annotations

import polars as pl


def per_query_fbeta(pred: pl.DataFrame, truth: pl.DataFrame, queries: pl.Series,
                    beta: float = 0.5) -> pl.DataFrame:
    """F-beta per S1: 1.0 for a correctly empty prediction on a singleton,
    0.0 for any prediction on a singleton or an empty prediction on a matched S1."""
    b2 = beta * beta
    qs = pl.DataFrame({"q": queries}).unique()
    pred = pred.select("q", "r").unique().join(qs, on="q", how="semi")
    truth = truth.select("q", "r").unique().join(qs, on="q", how="semi")
    tp = pred.join(truth, on=["q", "r"], how="semi").group_by("q").len("tp")
    df = (qs.join(pred.group_by("q").len("np"), on="q", how="left")
            .join(truth.group_by("q").len("nt"), on="q", how="left")
            .join(tp, on="q", how="left")
            .with_columns(pl.col("np", "nt", "tp").fill_null(0).cast(pl.Float64)))
    p = pl.col("tp") / pl.col("np")
    r = pl.col("tp") / pl.col("nt")
    f = (pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
           .when(pl.col("tp") == 0).then(0.0)
           .otherwise((1 + b2) * p * r / (b2 * p + r)))
    return df.with_columns(f.alias("f"))


def macro_fbeta(pred: pl.DataFrame, truth: pl.DataFrame, queries: pl.Series,
                beta: float = 0.5) -> float:
    return float(per_query_fbeta(pred, truth, queries, beta)["f"].mean())


def candidate_report(cands: pl.DataFrame, truth: pl.DataFrame, queries: pl.Series,
                     beta: float = 0.5) -> dict:
    """Recall ceiling of a candidate set: share of true pairs present, and the
    macro F-beta a perfect matcher restricted to these candidates would get."""
    qs = pl.DataFrame({"q": queries}).unique()
    cands = cands.select("q", "r").unique().join(qs, on="q", how="semi")
    truth = truth.select("q", "r").unique().join(qs, on="q", how="semi")
    found = cands.join(truth, on=["q", "r"], how="semi")
    per_q = cands.group_by("q").len()
    n_q = qs.height
    return {
        "queries": n_q,
        "pairs": cands.height,
        "pairs_per_query_mean": round(cands.height / max(n_q, 1), 2),
        "pairs_per_query_max": int(per_q["len"].max() or 0),
        "true_pairs": truth.height,
        "pair_recall": round(found.height / max(truth.height, 1), 5),
        "oracle_macro_f": round(macro_fbeta(found, truth, queries, beta), 5),
    }
