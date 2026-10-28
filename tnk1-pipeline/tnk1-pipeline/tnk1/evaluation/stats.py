"""Stage 9 statistics (scipy/numpy only)."""
from __future__ import annotations
import numpy as np
from scipy import stats
from scipy.stats import beta


def clopper_pearson(k: int, n: int, alpha=0.05):
    """Exact binomial CI. Returns (lo, hi)."""
    if n == 0: return (np.nan, np.nan)
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def hit_rate_table(counts: dict[str, tuple[int, int]]) -> dict:
    """counts: slice -> (hits, n)."""
    return {s: {"hits": k, "n": n, "rate": k / n if n else np.nan, "ci95": clopper_pearson(k, n)} for s, (k, n) in counts.items()}


def fisher_slice_comparison(a: tuple[int, int], b: tuple[int, int]):
    """a,b = (hits, n); two-sided Fisher exact test, e.g. Exploitation vs Control."""
    return float(stats.fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])[1])


def paired_seed_test(x: np.ndarray, y: np.ndarray) -> dict:
    """Paired Wilcoxon across matched seeds/folds (e.g. A7 vs baseline RMSE)."""
    d = np.asarray(x) - np.asarray(y)
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return {"mean_diff": float(d.mean()), "p_wilcoxon": float(p)}


def bootstrap_ci(values, stat=np.mean, n=5000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed); v = np.asarray(values)
    bs = [stat(v[rng.randint(0, len(v), len(v))]) for _ in range(n)]
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj.tolist()
