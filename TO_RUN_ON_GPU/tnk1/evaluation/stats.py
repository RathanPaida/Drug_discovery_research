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
    """Paired Wilcoxon across matched seeds/folds (e.g. A7 vs baseline RMSE).

    HARD LIMIT, WORTH KNOWING BEFORE READING THE OUTPUT. The exact two-sided signed-rank
    p-value cannot go below 2/2^n: with n = 5 folds the floor is 0.0625, so this test can
    NEVER report p < 0.05 no matter how large the difference. `scripts/compare_critic.py`
    says so in its docstring, and `RUN_REPORT_V3_HPC.md` reports the 5-fold p-values
    accordingly. To obtain a p-value that can clear 0.05, run repeated CV (10 repeats x 5
    folds = 50 scores) and use `corrected_resampled_ttest` below, which is also the test
    designed for scores that share training data.
    """
    d = np.asarray(x) - np.asarray(y)
    n = len(d)
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    out = {"mean_diff": float(d.mean()), "p_wilcoxon": float(p), "n": int(n)}
    if n:
        out["p_floor"] = float(2.0 / (2 ** n))      # the smallest p this n can produce
    return out


def corrected_resampled_ttest(a, b, n_train=None, n_test=None, k_folds=None) -> dict:
    """Nadeau-Bengio corrected resampled t-test for models scored on the SAME folds.

    Nadeau & Bengio, "Inference for the Generalization Error", NIPS 1999 / Machine Learning
    52:239-281 (2003). The ordinary paired t-test over cross-validation scores assumes the
    fold scores are independent. They are not: every pair of folds shares most of its
    training data, so the variance of the mean difference is UNDERSTATED and the Type-I
    error rate is badly inflated (Dietterich, Neural Computation 10:1895, 1998, measures
    this). The correction inflates the variance by a factor that accounts for the overlap:

        t = mean(d) / sqrt( (1/J + rho) * var(d) )

    with `rho = n_test / n_train` for repeated random resampling, or `rho = 1/k` for
    k-fold. Nadeau & Bengio prove no UNBIASED estimator of that variance exists, so this is
    a principled correction rather than an exact one -- which is the honest way to describe
    it in a paper.

    Pass either (`n_train`, `n_test`) or `k_folds`. Degrees of freedom J-1, J = len(a).

    WHY THIS MATTERS FOR THIS PROJECT SPECIFICALLY. Cawley & Talbot, "On Over-fitting in
    Model Selection and Subsequent Selection Bias in Performance Evaluation", JMLR
    11:2079-2107 (2010), find that "the effects of this form of over-fitting are often of
    comparable magnitude to differences in performance between learning algorithms". At
    N=172 with a ~20-model sweep, that is the size of the 0.686-vs-0.501 Spearman gap this
    repository is building conclusions on. Nested CV (which `tnk1.zoo.benchmark` enforces)
    plus this test is what keeps the gap from being an artefact of model selection.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    J = len(d)
    if J < 2:
        return {"mean_diff": float(d.mean()) if J else float("nan"), "t": float("nan"),
                "p": float("nan"), "n": int(J), "rho": float("nan")}
    if k_folds:
        rho = 1.0 / float(k_folds)
    elif n_train and n_test:
        rho = float(n_test) / float(n_train)
    else:
        raise ValueError("pass k_folds, or both n_train and n_test, so rho is defined")
    var = d.var(ddof=1)
    denom = np.sqrt(max((1.0 / J + rho) * var, 1e-300))
    t = float(d.mean() / denom) if denom > 0 else 0.0
    p = float(2 * stats.t.sf(abs(t), df=J - 1))
    return {"mean_diff": float(d.mean()), "t": t, "p": p, "n": int(J), "rho": float(rho),
            "sd_diff": float(np.sqrt(var))}


def bayesian_correlated_ttest(a, b, rope=0.01, rho=0.1, n_samples=20000, seed=0) -> dict:
    """P(a better), P(practically equivalent), P(b better) instead of a p-value.

    Benavoli, Corani, Demsar & Zaffalon, "Time for a Change: a Tutorial for Comparing
    Multiple Classifiers Through Bayesian Analysis", JMLR 18(77):1-36 (2017). The
    correlated Student-t posterior over the mean difference, with the same variance
    inflation as the Nadeau-Bengio correction, integrated over a region of practical
    equivalence (ROPE).

    Preferable to a p-value for this project's question. "Is the RF better than the
    ensemble" has three useful answers -- yes, no, and "the difference is too small to act
    on" -- and only the third is what you want to hear when the candidate decision is
    whether to re-run 2.8 GPU-hours. A ROPE of 0.01 Spearman is the convention used in
    recent molecular-representation benchmarks.

    `rho` is the unknowable fold correlation; sweep it (0.0, 0.05, 0.1, 0.2) and report the
    range rather than picking one, as the benchmarks that use this test do.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    J = len(d)
    if J < 2:
        return {"p_a_better": float("nan"), "p_rope": float("nan"),
                "p_b_better": float("nan"), "n": int(J)}
    mean = d.mean()
    scale = np.sqrt(max((1.0 / J + rho) * d.var(ddof=1), 1e-300))
    rng = np.random.RandomState(seed)
    draws = mean + scale * rng.standard_t(J - 1, size=n_samples)
    return {"p_a_better": float(np.mean(draws > rope)),
             "p_rope": float(np.mean(np.abs(draws) <= rope)),
             "p_b_better": float(np.mean(draws < -rope)),
             "mean_diff": float(mean), "rope": float(rope), "rho": float(rho), "n": int(J)}


def friedman_nemenyi(scores: dict[str, np.ndarray], alpha=0.05) -> dict:
    """Friedman omnibus test + Nemenyi critical difference over many models.

    Demsar, "Statistical Comparisons of Classifiers over Multiple Data Sets", JMLR 7:1-30
    (2006). `scores` maps a model name to its per-fold scores (HIGHER IS BETTER -- negate
    RMSE before passing it). Nemenyi's critical difference already accounts for the number
    of comparisons, so no separate multiplicity correction is applied on top.

    CAVEAT THAT MUST TRAVEL WITH THE RESULT: Demsar's procedure is designed for models
    compared across many DATASETS. This project has one dataset and many folds of it, which
    is a reasonable adaptation but not the published one -- the folds are not independent
    problems. State that where the diagram is used, and prefer
    `corrected_resampled_ttest` against a single pre-nominated incumbent (the random
    forest) for the headline claim: 19 comparisons under Holm beats 190 pairwise ones.
    """
    names = list(scores)
    M = np.vstack([np.asarray(scores[n], float) for n in names])       # (models, folds)
    if M.shape[0] < 3:
        raise ValueError("Friedman needs at least 3 models; use the paired tests instead")
    k, n = M.shape
    # rank within each fold, 1 = best (highest score)
    ranks = np.vstack([stats.rankdata(-M[:, j]) for j in range(n)]).T  # (models, folds)
    mean_ranks = ranks.mean(axis=1)
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(mean_ranks ** 2) - k * (k + 1) ** 2 / 4)
    p = float(stats.chi2.sf(chi2, df=k - 1))
    # Nemenyi critical difference: q_alpha * sqrt(k(k+1)/(6n)); q at alpha=0.05 from the
    # studentised range / sqrt(2), tabulated by Demsar.
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
           9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
           16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}
    q = q05.get(k)
    cd = float(q * np.sqrt(k * (k + 1) / (6.0 * n))) if q else float("nan")
    return {"chi2": float(chi2), "p": p, "k": int(k), "n_folds": int(n),
            "mean_ranks": {nm: float(r) for nm, r in zip(names, mean_ranks)},
            "critical_difference": cd,
            "note": ("Nemenyi CD tabulated to k=20; folds of ONE dataset are not "
                     "independent datasets, so treat this as an adaptation")}


def bootstrap_ci(values, stat=np.mean, n=5000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed); v = np.asarray(values)
    bs = [stat(v[rng.randint(0, len(v), len(v))]) for _ in range(n)]
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj.tolist()
