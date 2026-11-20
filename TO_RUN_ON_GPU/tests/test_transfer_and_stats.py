"""Pretraining-task selection and the comparison statistics. Hermetic numpy/scipy/pandas.

Two things nothing in this repository could do before:

  * Decide WHICH of the 491 donor kinases belong in the pan-kinome pretraining set. The
    existing stage weights them all by pocket similarity and selects none, so a kinase
    whose structure-activity relationship runs opposite to TNK1's still contributes
    gradient that competes with the 172 TNK1 measurements.
  * Produce a paired test that can actually reach p < 0.05. The exact Wilcoxon
    signed-rank test over 5 folds has a p-value FLOOR of 2/2^5 = 0.0625, which
    `compare_critic.py` documents and `RUN_REPORT_V3_HPC.md` reports around. The
    Nadeau-Bengio corrected resampled t-test is the test designed for scores that share
    training data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tnk1.evaluation import stats as S  # noqa: E402
from tnk1.transfer import task_selection as TS  # noqa: E402


# ------------------------------------------------------------------ task selection

def kinome(n_compounds=60, seed=0):
    """A synthetic kinome with four donors whose relationship to the target is KNOWN:

        GOOD   SAR almost identical to the target        -> must be selected
        ANTI   SAR running OPPOSITE to the target        -> must be excluded, not weighted
        NOISE  no relationship                           -> must be excluded
        THIN   identical SAR but only 5 shared compounds -> must be unjudgeable, not ranked
    """
    rng = np.random.default_rng(seed)
    keys = [f"C{i:03d}" for i in range(n_compounds)]
    y = rng.normal(7.0, 1.0, n_compounds)
    rows = [dict(uniprot_id="TARGET", inchikey_block1=k, pact=v, censored=0)
            for k, v in zip(keys, y)]
    for k, v in zip(keys[:50], y[:50]):
        rows.append(dict(uniprot_id="GOOD", inchikey_block1=k,
                         pact=v + rng.normal(0, 0.25), censored=0))
        rows.append(dict(uniprot_id="ANTI", inchikey_block1=k,
                         pact=-v + rng.normal(0, 0.25), censored=0))
        rows.append(dict(uniprot_id="NOISE", inchikey_block1=k,
                         pact=rng.normal(7, 1), censored=0))
    for k, v in zip(keys[:5], y[:5]):
        rows.append(dict(uniprot_id="THIN", inchikey_block1=k, pact=v, censored=0))
    return pd.DataFrame(rows)


def test_activity_correlation_recovers_the_known_relationships():
    corr = TS.activity_correlation(kinome(), "TARGET", min_overlap=10)
    assert corr["GOOD"]["rho"] > 0.85
    assert corr["ANTI"]["rho"] < -0.85
    assert abs(corr["NOISE"]["rho"]) < 0.35
    assert corr["GOOD"]["n_overlap"] == 50


def test_a_donor_with_too_few_shared_compounds_is_absent_not_noisy():
    """A correlation from 5 compounds is not a weak estimate, it is noise. Including it
    with a NaN or a wide confidence interval invites ranking on it."""
    corr = TS.activity_correlation(kinome(), "TARGET", min_overlap=10)
    assert "THIN" not in corr
    assert "THIN" in TS.activity_correlation(kinome(), "TARGET", min_overlap=3)


def test_censored_rows_are_excluded_from_the_correlation():
    """A censored row carries a BOUND, not a value; correlating bounds against exact
    measurements mixes two different quantities. Every evaluation path in this repository
    already filters `censored == 0`."""
    df = kinome()
    bad = df[df.uniprot_id == "GOOD"].copy()
    bad["pact"] = 99.0          # absurd values, flagged as censored
    bad["censored"] = 1
    corr = TS.activity_correlation(pd.concat([df, bad], ignore_index=True),
                                   "TARGET", min_overlap=10)
    clean = TS.activity_correlation(df, "TARGET", min_overlap=10)
    assert corr["GOOD"]["rho"] == pytest.approx(clean["GOOD"]["rho"])


def test_a_compound_measured_twice_on_one_kinase_counts_once():
    """IC50 and Ki rows for the same compound would otherwise double the overlap count and
    inflate every donor's apparent evidence."""
    df = kinome()
    dup = df[df.uniprot_id == "GOOD"].copy()
    both = pd.concat([df, dup], ignore_index=True)
    assert (TS.activity_correlation(both, "TARGET", 10)["GOOD"]["n_overlap"]
            == TS.activity_correlation(df, "TARGET", 10)["GOOD"]["n_overlap"])


def test_by_activity_correlation_ranks_and_thresholds():
    corr = TS.activity_correlation(kinome(), "TARGET", min_overlap=10)
    assert TS.by_activity_correlation(corr)[0] == "GOOD"
    assert TS.by_activity_correlation(corr, min_rho=0.5) == ["GOOD"]
    assert TS.by_activity_correlation(corr, k=2)[0] == "GOOD"


def test_task_weights_clamp_anticorrelated_donors_to_zero():
    """A donor whose SAR runs opposite to the target's is worse than useless: the trunk is
    asked to fit both directions at once. The current pipeline weights it by pocket
    similarity and trains on it."""
    w = TS.task_weights_from_correlation(
        TS.activity_correlation(kinome(), "TARGET", min_overlap=10))
    assert w["ANTI"] == 0.0
    assert w["GOOD"] > 0.5
    assert w["GOOD"] > w["NOISE"]


def test_task_weights_shrink_a_correlation_resting_on_few_compounds():
    """rho 0.9 from 11 compounds must not outrank rho 0.6 from 400."""
    corr = {"few": {"rho": 0.9, "p": 0.01, "n_overlap": 11},
            "many": {"rho": 0.6, "p": 1e-20, "n_overlap": 400}}
    w = TS.task_weights_from_correlation(corr, shrink_by_overlap=True)
    assert w["many"] > w["few"]
    w_no = TS.task_weights_from_correlation(corr, shrink_by_overlap=False)
    assert w_no["few"] > w_no["many"], "without shrinkage the thin donor wins (the bug)"


def test_detect_negative_transfer_splits_into_three_groups():
    """Three groups, not a ranking: the actionable output is 'drop these', and a ranking
    invites cutting at an arbitrary k."""
    g = TS.detect_negative_transfer({"a": 0.12, "b": 0.01, "c": -0.09}, tol=0.05)
    assert g == {"helpful": ["a"], "neutral": ["b"], "harmful": ["c"]}


def test_detect_negative_transfer_tolerance_absorbs_noise():
    """At N=172 a Spearman standard error is around 0.1, so tol=0.05 is not conservative."""
    g = TS.detect_negative_transfer({"a": 0.03, "b": -0.03}, tol=0.05)
    assert g["neutral"] == ["a", "b"] and not g["helpful"] and not g["harmful"]


def test_by_leave_one_in_reports_deltas_against_a_baseline():
    scores = {"x": 0.70, "y": 0.55}
    out = TS.by_leave_one_in(["x", "y"], lambda u: scores[u], baseline=0.60)
    assert out["x"] == pytest.approx(0.10)
    assert out["y"] == pytest.approx(-0.05), "a negative delta is negative transfer"


def test_by_similarity_orders_and_truncates():
    assert TS.by_similarity({"a": 0.9, "b": 0.5, "c": 0.7}, k=2) == ["a", "c"]
    assert TS.by_similarity({"a": 0.9, "b": 0.5}, threshold=0.8) == ["a"]


def test_activity_correlation_raises_when_the_target_is_absent():
    with pytest.raises(ValueError):
        TS.activity_correlation(kinome(), "NOT_A_TARGET")


# ------------------------------------------------------------------- the statistics

def test_wilcoxon_floor_is_reported_so_a_p_value_is_readable():
    """With 5 folds the exact two-sided test cannot return p < 0.0625 no matter how large
    the difference. The floor must travel with the result."""
    a = np.array([0.70, 0.72, 0.68, 0.71, 0.69])
    b = a - 0.30                                   # an enormous, consistent difference
    out = S.paired_seed_test(a, b)
    assert out["p_wilcoxon"] >= out["p_floor"]
    assert out["p_floor"] == pytest.approx(0.0625)
    assert out["mean_diff"] == pytest.approx(0.30)


def test_wilcoxon_floor_shrinks_with_more_folds():
    rng = np.random.default_rng(0)
    a = rng.normal(0.7, 0.02, 50)
    out = S.paired_seed_test(a, a - 0.3)
    assert out["p_floor"] < 1e-12
    assert out["p_wilcoxon"] < 0.05, "50 fold scores CAN reach significance"


def test_corrected_resampled_ttest_inflates_the_variance():
    """Nadeau & Bengio (1999/2003). The correction must make the test STRICTLY more
    conservative than an uncorrected paired t-test on the same numbers."""
    from scipy import stats as sps
    rng = np.random.default_rng(1)
    d = rng.normal(0.03, 0.02, 50)
    corrected = S.corrected_resampled_ttest(d, np.zeros_like(d), k_folds=5)
    plain = sps.ttest_1samp(d, 0.0)
    assert corrected["p"] > plain.pvalue, \
        "the corrected test must be more conservative, not less"
    assert abs(corrected["t"]) < abs(plain.statistic)
    assert corrected["rho"] == pytest.approx(0.2)


def test_corrected_resampled_ttest_accepts_either_rho_parameterisation():
    d = np.random.default_rng(2).normal(0.05, 0.02, 20)
    by_k = S.corrected_resampled_ttest(d, np.zeros_like(d), k_folds=5)
    by_n = S.corrected_resampled_ttest(d, np.zeros_like(d), n_train=100, n_test=20)
    assert by_k["rho"] == pytest.approx(by_n["rho"])
    with pytest.raises(ValueError):
        S.corrected_resampled_ttest(d, np.zeros_like(d))


def test_corrected_resampled_ttest_detects_a_real_difference():
    rng = np.random.default_rng(3)
    a = rng.normal(0.70, 0.03, 50)
    b = rng.normal(0.50, 0.03, 50)
    out = S.corrected_resampled_ttest(a, b, k_folds=5)
    assert out["mean_diff"] > 0.15 and out["p"] < 0.01


def test_corrected_resampled_ttest_is_degenerate_safe():
    out = S.corrected_resampled_ttest([0.5], [0.4], k_folds=5)
    assert np.isnan(out["t"]) and out["n"] == 1


def test_bayesian_correlated_ttest_probabilities_sum_to_one():
    rng = np.random.default_rng(4)
    a = rng.normal(0.70, 0.03, 50)
    b = rng.normal(0.50, 0.03, 50)
    out = S.bayesian_correlated_ttest(a, b, rope=0.01, rho=0.1, seed=0)
    assert out["p_a_better"] + out["p_rope"] + out["p_b_better"] == pytest.approx(1.0, abs=1e-9)
    assert out["p_a_better"] > 0.95


def test_bayesian_correlated_ttest_reports_practical_equivalence():
    """The answer a p-value cannot give, and the one that matters when the decision is
    whether to spend 2.8 GPU-hours on a re-run: 'the difference is too small to act on'."""
    rng = np.random.default_rng(5)
    a = rng.normal(0.700, 0.02, 60)
    b = rng.normal(0.702, 0.02, 60)
    out = S.bayesian_correlated_ttest(a, b, rope=0.02, rho=0.1, seed=0)
    assert out["p_rope"] > 0.5, f"near-identical models should land in the ROPE: {out}"


def test_friedman_nemenyi_ranks_models_and_returns_a_critical_difference():
    """Demsar (JMLR 2006). Higher score is better, so ranks are over negated losses."""
    rng = np.random.default_rng(6)
    scores = {"best": rng.normal(0.75, 0.02, 20),
              "mid": rng.normal(0.65, 0.02, 20),
              "worst": rng.normal(0.50, 0.02, 20)}
    out = S.friedman_nemenyi(scores)
    assert out["p"] < 0.01
    r = out["mean_ranks"]
    assert r["best"] < r["mid"] < r["worst"], "rank 1 = best"
    assert out["critical_difference"] > 0
    assert "not independent datasets" in out["note"], \
        "the adaptation caveat must travel with the result"


def test_friedman_requires_at_least_three_models():
    with pytest.raises(ValueError):
        S.friedman_nemenyi({"a": np.zeros(5), "b": np.ones(5)})


def test_holm_controls_the_family_wise_error_rate_monotonically():
    """Used to test 19 challengers against one pre-nominated incumbent rather than all 190
    pairs -- the best of 20 models beats its own honest estimate by roughly the spread of
    the 20."""
    p = [0.001, 0.01, 0.04, 0.5]
    adj = S.holm(p)
    assert all(a >= b for a, b in zip(adj, p)), "adjusted p must never decrease"
    assert adj == sorted(adj), "Holm's adjusted values are monotone in the sorted order"
    assert adj[0] == pytest.approx(0.004)
    assert all(a <= 1.0 for a in adj)
