"""Which of the 491 kinases belong in the pretraining set? Pure numpy.

THE QUESTION NOBODY HAS ASKED HERE. The pan-kinome stage trains on all 491 kinases with
similarity weights (`tnk1.models.weighting.sample_weights`, lambda_sim tuned by
leave-one-kinase-out). That is a WEIGHTING, not a SELECTION: every kinase contributes, and
a kinase whose SAR is unrelated to TNK1's contributes gradient that competes with the 172
TNK1 measurements rather than supporting them. The ablation ladder shows pretraining helps
on average (RMSE 1.148 -> 0.628-0.697), which says the aggregate is positive; it does not
say every task is.

Negative transfer is the standard name for the failure and it is measurable per task. This
module provides the three selection criteria the multi-task literature uses, as pure
functions over arrays, so the choice becomes a sweep rather than an assumption:

  1. `by_similarity`   -- the incumbent criterion, made explicit: rank by binding-site or
                          sequence similarity to the target. Cheap, needs no labels.
  2. `by_activity_correlation` -- rank by how well a kinase's pActivity CORRELATES with
                          the target's over the compounds measured on both. This is the
                          criterion that actually bears on transfer: two kinases with
                          similar pockets but uncorrelated SAR will not help each other,
                          and compounds-in-common is the only direct evidence of shared
                          SAR available without training anything.
  3. `by_leave_one_in`  -- the empirical criterion: train on {target + one donor} and
                          measure the target's validation score. Exact, and the most
                          expensive by a wide margin; `score_fn` is injected so the
                          caller decides what "train" means.

Everything here is numpy so it is testable without torch, RDKit or a GPU, and so the
selection can be inspected and argued about before 100 GPU-minutes are spent on it.
"""
from __future__ import annotations

import numpy as np


def by_similarity(sim_to_target: dict[str, float], k: int | None = None,
                  threshold: float | None = None) -> list[str]:
    """Rank donors by a precomputed similarity to the target. Highest first.

    `sim_to_target` is e.g. `dict(zip(store["ids"], store["S"][target_idx]))` -- the
    kinase store's own similarity matrix, which `sample_weights` already consumes. Pass
    `k` for a top-k cut or `threshold` for an absolute one; both, and the stricter wins.
    """
    items = sorted(((u, float(s)) for u, s in sim_to_target.items()),
                   key=lambda t: -t[1])
    if threshold is not None:
        items = [(u, s) for u, s in items if s >= threshold]
    if k is not None:
        items = items[:k]
    return [u for u, _ in items]


def activity_correlation(df, target: str, min_overlap: int = 10,
                         key_col: str = "inchikey_block1", val_col: str = "pact",
                         uniprot_col: str = "uniprot_id",
                         censored_col: str = "censored") -> dict[str, dict]:
    """Per-donor Spearman correlation with the target, over compounds measured on both.

    Returns `{uniprot: {"rho": float, "n_overlap": int, "p": float}}` for every donor with
    at least `min_overlap` shared compounds. Donors below that are ABSENT from the result
    rather than present with a NaN -- a correlation from 4 compounds is not a weak
    estimate, it is noise, and including it invites ranking on it.

    Censored rows are excluded: they carry a bound, not a value, so correlating them
    against exact measurements mixes two different quantities. This mirrors what every
    evaluation path in this repository already does (`censored == 0`).

    Spearman rather than Pearson because what transfer needs is shared ORDERING of
    compounds -- a donor whose pActivity is a monotone but non-linear function of the
    target's is a perfectly good donor.
    """
    from scipy import stats
    d = df
    if censored_col in d.columns:
        d = d[d[censored_col] == 0]
    tgt = d[d[uniprot_col] == target]
    if not len(tgt):
        raise ValueError(f"target {target} has no exact rows in this table")
    # One value per compound: a compound measured twice on the same kinase (IC50 and Ki)
    # would otherwise contribute twice and inflate the overlap count.
    t_map = tgt.groupby(key_col)[val_col].mean()
    out: dict[str, dict] = {}
    for u, g in d[d[uniprot_col] != target].groupby(uniprot_col):
        g_map = g.groupby(key_col)[val_col].mean()
        common = t_map.index.intersection(g_map.index)
        if len(common) < min_overlap:
            continue
        x = t_map.loc[common].values
        y = g_map.loc[common].values
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            continue
        r = stats.spearmanr(x, y)
        out[str(u)] = {"rho": float(r.statistic), "p": float(r.pvalue),
                       "n_overlap": int(len(common))}
    return out


def by_activity_correlation(corr: dict[str, dict], k: int | None = None,
                            min_rho: float | None = None,
                            min_overlap: int = 10) -> list[str]:
    """Rank donors by activity correlation with the target. Highest rho first."""
    items = [(u, c["rho"]) for u, c in corr.items()
             if c["n_overlap"] >= min_overlap
             and (min_rho is None or c["rho"] >= min_rho)]
    items.sort(key=lambda t: -t[1])
    return [u for u, _ in items[:k]] if k else [u for u, _ in items]


def by_leave_one_in(donors, score_fn, baseline: float | None = None,
                    higher_better: bool = True) -> dict[str, float]:
    """Score each donor by training on {target + that donor} alone.

    `score_fn(donor) -> float` is injected, so this function has no opinion about what
    training means and costs nothing to test. `baseline` is the target-only score; when
    given, the returned value is the DELTA, so a positive number means that donor helped
    and a negative one means it caused negative transfer.

    This is the criterion that actually answers the question, and it is O(n_donors)
    trainings -- so use `by_activity_correlation` to shortlist first and run this on the
    top ~20. The 491-kinase full sweep is not worth its GPU time.
    """
    out = {}
    for u in donors:
        s = float(score_fn(u))
        if baseline is None:
            out[u] = s
        else:
            out[u] = (s - baseline) if higher_better else (baseline - s)
    return out


def detect_negative_transfer(per_task_delta: dict[str, float],
                             tol: float = 0.0) -> dict[str, list[str]]:
    """Split donors into helpful / neutral / harmful by their measured delta.

    `tol` is the band treated as neutral; set it to the cross-validation noise of the
    score (at N=172 a Spearman standard error is around 0.1, so `tol=0.05` is not
    conservative). Returning three groups rather than a ranking is deliberate: the
    actionable output is "drop these", and a ranking invites cutting at an arbitrary k.
    """
    helpful = sorted([u for u, d in per_task_delta.items() if d > tol],
                     key=lambda u: -per_task_delta[u])
    harmful = sorted([u for u, d in per_task_delta.items() if d < -tol],
                     key=lambda u: per_task_delta[u])
    neutral = sorted(set(per_task_delta) - set(helpful) - set(harmful))
    return {"helpful": helpful, "neutral": neutral, "harmful": harmful}


def task_weights_from_correlation(corr: dict[str, dict], floor: float = 0.0,
                                  power: float = 1.0,
                                  shrink_by_overlap: bool = True) -> dict[str, float]:
    """Turn activity correlations into per-task sample weights.

    A drop-in alternative to the similarity weights in `tnk1.models.weighting`, computed
    from measured SAR agreement rather than from pocket geometry. Negative correlations
    clamp to `floor` (default 0, i.e. excluded): a donor whose SAR runs OPPOSITE to the
    target's is worse than useless, because the model is being asked to fit both
    directions with one trunk.

    `shrink_by_overlap` multiplies each weight by `n/(n+10)`, a plain shrinkage toward
    zero for donors whose correlation rests on few shared compounds. Without it a rho of
    0.9 from 11 compounds outranks 0.6 from 400, which is backwards.
    """
    out = {}
    for u, c in corr.items():
        w = max(float(c["rho"]), floor) ** power if c["rho"] > 0 else floor
        if shrink_by_overlap:
            n = int(c["n_overlap"])
            w *= n / (n + 10.0)
        out[u] = float(w)
    return out
