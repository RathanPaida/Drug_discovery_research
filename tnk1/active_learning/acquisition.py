"""Sec 8.1: EI / UCB / LCB + Pareto ranking (numpy/scipy only)."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def expected_improvement(mu, sigma_epi, y_best, xi=0.05):
    s = np.maximum(sigma_epi, 1e-9); z = (mu - y_best - xi) / s
    return (mu - y_best - xi) * norm.cdf(z) + s * norm.pdf(z)


def ucb(mu, sigma_epi, kappa=1.5): return mu + kappa * sigma_epi
def lcb(mu, sigma_epi, k=1.0): return mu - k * sigma_epi


def pareto_rank(objs: np.ndarray) -> np.ndarray:
    """Non-dominated sorting (maximise all columns). Returns front index per row (0 = Pareto front)."""
    n = len(objs); rank = np.full(n, -1); remaining = np.arange(n); f = 0
    while len(remaining):
        sub = objs[remaining]; front = []
        for i in range(len(remaining)):
            dom = np.all(sub >= sub[i], 1) & np.any(sub > sub[i], 1)
            if not dom.any(): front.append(remaining[i])
        rank[front] = f; remaining = np.setdiff1d(remaining, front); f += 1
    return rank


def top_by_pareto(objs: np.ndarray, k: int, tiebreak: np.ndarray | None = None) -> np.ndarray:
    r = pareto_rank(objs); tb = -(tiebreak if tiebreak is not None else objs.sum(1))
    return np.lexsort((tb, r))[:k]
