"""Sec 8.1: EI / UCB / LCB + Pareto ranking (numpy/scipy only)."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def expected_improvement(mu, sigma_epi, y_best, xi=0.05):
    s = np.maximum(sigma_epi, 1e-9); z = (mu - y_best - xi) / s
    return (mu - y_best - xi) * norm.cdf(z) + s * norm.pdf(z)


def ucb(mu, sigma_epi, kappa=1.5): return mu + kappa * sigma_epi
def lcb(mu, sigma_epi, k=1.0): return mu - k * sigma_epi


def pareto_rank(objs: np.ndarray, stop_after: int | None = None) -> np.ndarray:
    """Non-dominated sorting (maximise all columns). Returns front index per row (0 = Pareto front).

    `stop_after` stops once at least that many rows have been ranked, and gives every
    remaining row the next front index. Rows left that way are all strictly worse than
    everything already ranked, so any "best N <= stop_after" selection is unchanged --
    which is the only thing `top_by_pareto` asks for.

    It matters because this is O(n^2) per front with a Python-level loop: measured
    0.3 s at n=2,000, 5.7 s at 8,000 and 44 s at 20,000, i.e. about 24 minutes at the
    ~117k candidates Stage 8 now reaches since the pool stopped being starved -- and
    Stage 8 runs once per RL objective. Ranking every one of ~11 fronts to pick the top
    1,500 is work nobody reads.
    """
    n = len(objs); rank = np.full(n, -1); remaining = np.arange(n); f = 0; done = 0
    while len(remaining):
        sub = objs[remaining]; front = []
        for i in range(len(remaining)):
            dom = np.all(sub >= sub[i], 1) & np.any(sub > sub[i], 1)
            if not dom.any(): front.append(remaining[i])
        rank[front] = f; remaining = np.setdiff1d(remaining, front)
        done += len(front); f += 1
        if stop_after is not None and done >= stop_after:
            # `f` has already been incremented, so this is the NEXT front index: the
            # unranked rows sort strictly after everything ranked above. Using `f - 1`
            # here would tie them with the last real front and let the tiebreak pull
            # them into the selection.
            rank[rank < 0] = f
            break
    return rank


def top_by_pareto(objs: np.ndarray, k: int, tiebreak: np.ndarray | None = None) -> np.ndarray:
    r = pareto_rank(objs, stop_after=k)
    tb = -(tiebreak if tiebreak is not None else objs.sum(1))
    return np.lexsort((tb, r))[:k]
