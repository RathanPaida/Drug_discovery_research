"""Tests for the selection logic that chose the 40-compound contract: acquisition
functions, Pareto ranking, the scaffold-split leakage guards, selectivity margins,
uncertainty decomposition, the applicability domain and the Stage-9 statistics.

Pure numpy/scipy/pandas -- runs in a core-only install. The RDKit-gated med-chem gate
tests live in test_medchem_filters.py so that a missing RDKit cannot skip these too.
"""
import numpy as np
import pandas as pd
import pytest

from tnk1.active_learning import acquisition as ACQ
from tnk1.data import splits as S
from tnk1.evaluation import stats as ST
from tnk1.selectivity.selectivity import member_margin, selectivity_stats, soft_margin
from tnk1.uq.applicability import ApplicabilityDomain
from tnk1.uq.decomposition import decompose


# ------------------------------------------------------------------------- acquisition

def test_ucb_and_lcb_bracket_the_mean():
    mu, sd = np.array([7.0]), np.array([0.3])
    assert ACQ.lcb(mu, sd, 1.0)[0] < mu[0] < ACQ.ucb(mu, sd, 1.5)[0]


def test_lcb_is_the_pessimistic_exploitation_score():
    """Stage 8 ranks EXPLOIT by LCB, so a more uncertain compound must score lower."""
    mu = np.array([7.5, 7.5])
    sd = np.array([0.05, 0.50])
    scores = ACQ.lcb(mu, sd, k=1.0)
    assert scores[0] > scores[1]


def test_ucb_is_the_optimistic_exploration_score():
    mu = np.array([7.5, 7.5])
    sd = np.array([0.05, 0.50])
    assert ACQ.ucb(mu, sd, kappa=1.5)[1] > ACQ.ucb(mu, sd, kappa=1.5)[0]


def test_expected_improvement_is_nonnegative():
    rng = np.random.RandomState(0)
    ei = ACQ.expected_improvement(rng.normal(7, 1, 200), np.abs(rng.normal(0.3, 0.1, 200)), 7.0)
    assert np.all(ei >= 0)


def test_expected_improvement_rises_with_the_mean():
    sd = np.array([0.3])
    lo = ACQ.expected_improvement(np.array([7.0]), sd, 7.0)[0]
    hi = ACQ.expected_improvement(np.array([8.0]), sd, 7.0)[0]
    assert hi > lo


def test_expected_improvement_rises_with_uncertainty_at_the_incumbent():
    mu = np.array([7.0])
    assert (ACQ.expected_improvement(mu, np.array([0.6]), 7.0)[0]
            > ACQ.expected_improvement(mu, np.array([0.1]), 7.0)[0])


def test_expected_improvement_handles_zero_uncertainty():
    ei = ACQ.expected_improvement(np.array([7.0]), np.array([0.0]), 7.0)
    assert np.isfinite(ei).all()


def test_pareto_rank_front_zero_is_non_dominated():
    # (potency, selectivity) maximised; C is dominated by both A and B
    objs = np.array([[8.0, 1.0],      # A
                     [7.0, 2.0],      # B
                     [6.5, 0.5]])     # C
    r = ACQ.pareto_rank(objs)
    assert r[0] == 0 and r[1] == 0
    assert r[2] > 0


def test_pareto_rank_assigns_every_row_a_front():
    rng = np.random.RandomState(1)
    r = ACQ.pareto_rank(rng.normal(size=(40, 3)))
    assert (r >= 0).all() and len(r) == 40


def test_pareto_rank_on_a_total_order_is_a_chain():
    objs = np.array([[3.0], [2.0], [1.0]])
    assert list(ACQ.pareto_rank(objs)) == [0, 1, 2]


def test_a_dominating_point_is_always_selected_first():
    objs = np.array([[1.0, 1.0], [9.0, 9.0], [2.0, 2.0]])
    assert ACQ.top_by_pareto(objs, 1)[0] == 1


def test_top_by_pareto_respects_k_and_tiebreak():
    objs = np.array([[5.0, 1.0], [1.0, 5.0], [0.0, 0.0]])
    sel = ACQ.top_by_pareto(objs, 2, tiebreak=np.array([0.0, 1.0, 0.0]))
    assert len(sel) == 2
    assert sel[0] == 1                      # higher tiebreak wins inside the front
    assert 2 not in sel                     # dominated point excluded


# ------------------------------------------------------------- scaffold splits / leakage

def _tnk1_frame(n_scaffolds=10, per=3):
    rows = []
    for s in range(n_scaffolds):
        for j in range(per):
            rows.append({"inchikey_block1": f"CPD{s:02d}{j}", "uniprot_id": "Q13470",
                         "scaffold": f"scaf{s}", "censored": 0, "pact": 7.0})
    return pd.DataFrame(rows)


def test_balanced_group_folds_keeps_groups_intact():
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    f = S.balanced_group_folds(groups, 2, seed=0)
    for g in set(groups):
        assert len(set(f[groups == g])) == 1, f"scaffold {g} split across folds"


def test_balanced_group_folds_is_deterministic_for_a_seed():
    groups = np.array(list("aabbccddee"))
    a = S.balanced_group_folds(groups, 3, seed=7)
    b = S.balanced_group_folds(groups, 3, seed=7)
    assert np.array_equal(a, b)


def test_balanced_group_folds_balances_sizes():
    groups = np.repeat(np.arange(20), 5).astype(str)
    f = S.balanced_group_folds(groups, 5, seed=0)
    sizes = np.bincount(f, minlength=5)
    assert sizes.max() - sizes.min() <= 5


def test_scaffold_holdout_selects_roughly_the_requested_fraction():
    groups = np.repeat(np.arange(50), 4).astype(str)
    mask = S.scaffold_holdout(groups, 0.2, seed=0)
    assert 0.1 < mask.mean() < 0.35


def test_build_tnk1_splits_shares_no_compound_between_test_and_dev():
    out = S.build_tnk1_splits(_tnk1_frame(), "Q13470", 0.2, 3, 2, 42)
    assert set(out["test_keys"]).isdisjoint(out["dev_keys"])
    assert out["test_keys"] and out["dev_keys"]


def test_build_tnk1_splits_cv_folds_are_disjoint_within_an_outer_fold():
    out = S.build_tnk1_splits(_tnk1_frame(12, 3), "Q13470", 0.2, 3, 2, 42)
    for fold in out["cv"]:
        assert set(fold["train"]).isdisjoint(fold["test"])
        for inner in fold["inner"]:
            assert set(inner["train"]).isdisjoint(inner["val"])


def test_build_tnk1_splits_cv_never_touches_the_frozen_test_set():
    out = S.build_tnk1_splits(_tnk1_frame(12, 3), "Q13470", 0.2, 3, 2, 42)
    test = set(out["test_keys"])
    for fold in out["cv"]:
        assert test.isdisjoint(fold["train"])
        assert test.isdisjoint(fold["test"])


def test_build_tnk1_splits_is_reproducible():
    a = S.build_tnk1_splits(_tnk1_frame(), "Q13470", 0.2, 3, 2, 42)
    b = S.build_tnk1_splits(_tnk1_frame(), "Q13470", 0.2, 3, 2, 42)
    assert a == b


def test_kinome_train_mask_removes_test_compounds_from_every_kinase():
    """The leakage guard that matters: the same molecule measured on SRC would otherwise
    leak a TNK1 test scaffold into training."""
    df = pd.DataFrame({"inchikey_block1": ["A", "A", "B"],
                       "uniprot_id": ["Q13470", "P12931", "P12931"]})
    m = S.kinome_train_mask(df, exclude_compound_keys={"A"})
    assert list(m) == [False, False, True]


def test_kinome_train_mask_excludes_held_out_kinases_for_loko():
    df = pd.DataFrame({"inchikey_block1": ["A", "B"], "uniprot_id": ["Q13470", "P12931"]})
    m = S.kinome_train_mask(df, (), exclude_uniprots={"P12931"})
    assert list(m) == [True, False]


def test_kinome_train_mask_with_no_exclusions_keeps_everything():
    df = pd.DataFrame({"inchikey_block1": ["A"], "uniprot_id": ["Q13470"]})
    assert S.kinome_train_mask(df).all()


def test_assert_no_leakage_passes_on_clean_splits():
    S.assert_no_leakage(["A"], ["B"], ["s1"], ["s2"])


def test_assert_no_leakage_catches_a_shared_compound():
    with pytest.raises(AssertionError, match="both train and test"):
        S.assert_no_leakage(["A", "B"], ["B"])


def test_assert_no_leakage_catches_a_shared_scaffold():
    with pytest.raises(AssertionError, match="scaffolds shared"):
        S.assert_no_leakage(["A"], ["B"], ["s1"], ["s1"])


def test_loko_splits_counts_held_out_rows():
    df = pd.DataFrame({"uniprot_id": ["P1", "P1", "P2"]})
    out = S.loko_splits(df, ["P1", "P2"])
    assert out["P1"]["heldout_n"] == 2 and out["P2"]["heldout_n"] == 1


def test_save_splits_roundtrips(tmp_path):
    import json
    p = tmp_path / "s.json"
    S.save_splits({"seed": 42, "test_keys": ["A"]}, p)
    assert json.loads(p.read_text())["test_keys"] == ["A"]


# ----------------------------------------------------------------------- selectivity

def test_member_margin_uses_the_worst_offtarget():
    mu_t = np.array([[8.0]])
    mu_o = np.array([[[6.0, 7.5, 5.0]]])
    assert member_margin(mu_t, mu_o)[0, 0] == pytest.approx(0.5)


def test_selectivity_stats_fold_is_ten_to_the_margin():
    mu_t = np.array([[8.0], [8.0]])
    mu_o = np.array([[[6.0]], [[6.0]]])
    st = selectivity_stats(mu_t, mu_o)
    assert st["mu_sel"][0] == pytest.approx(2.0)
    assert st["fold"][0] == pytest.approx(100.0)
    assert st["sigma_sel"][0] == pytest.approx(0.0)


def test_selectivity_sigma_reflects_member_disagreement():
    mu_t = np.array([[8.0], [8.0], [8.0]])
    mu_o = np.array([[[6.0]], [[7.0]], [[5.0]]])
    assert selectivity_stats(mu_t, mu_o)["sigma_sel"][0] > 0


def test_a_pan_kinase_profile_gives_a_nonpositive_margin():
    """Sanity check from the manual: a pan-inhibitor must not look selective."""
    mu_t = np.array([[7.0]])
    mu_o = np.array([[[7.5, 7.2]]])
    assert selectivity_stats(mu_t, mu_o)["mu_sel"][0] <= 0


def test_soft_margin_is_more_conservative_than_the_hard_max():
    """LogSumExp over many comparable off-targets penalises more than the single worst."""
    mu_t = np.array([[8.0]])
    mu_o = np.array([[[6.0, 6.0, 6.0, 6.0]]])
    assert soft_margin(mu_t, mu_o, 1.0)[0, 0] < member_margin(mu_t, mu_o)[0, 0]


# ---------------------------------------------------------------- uncertainty decomposition

def test_decompose_splits_total_variance_into_its_two_parts():
    mu_m = np.array([[7.0], [7.4]])
    lv_m = np.array([[np.log(0.04)], [np.log(0.04)]])
    d = decompose(mu_m, lv_m)
    assert d["mu"][0] == pytest.approx(7.2)
    assert d["var_ale"][0] == pytest.approx(0.04)
    assert d["var_epi"][0] == pytest.approx(0.04)                 # ((+-0.2)^2)
    assert d["sigma_tot"][0] == pytest.approx(np.sqrt(0.08))


def test_identical_members_have_zero_epistemic_variance():
    mu_m = np.full((5, 3), 7.0)
    d = decompose(mu_m, np.zeros((5, 3)))
    assert np.allclose(d["var_epi"], 0.0)
    assert np.allclose(d["sigma_tot"], 1.0)


# --------------------------------------------------------------- applicability domain

def test_applicability_domain_flags_a_distant_molecule():
    rng = np.random.RandomState(0)
    Z = rng.normal(size=(60, 8))
    FP = (rng.rand(60, 32) > 0.5).astype(float)
    ad = ApplicabilityDomain(k=5, pct=95).fit(Z, FP)
    assert ad.in_domain(Z[:5], FP[:5]).all()
    far_z = np.full((1, 8), 50.0)
    assert not ad.in_domain(far_z, FP[:1])[0]


def test_applicability_distances_are_nonnegative_and_finite():
    rng = np.random.RandomState(2)
    Z = rng.normal(size=(30, 4)); FP = (rng.rand(30, 16) > 0.5).astype(float)
    ad = ApplicabilityDomain().fit(Z, FP)
    dl, df = ad.distances(Z, FP)
    assert (dl >= 0).all() and np.isfinite(dl).all()
    assert (df >= 0).all() and (df <= 1).all()


# ------------------------------------------------------------------- Stage 9 statistics

def test_clopper_pearson_brackets_the_point_estimate():
    lo, hi = ST.clopper_pearson(3, 10)
    assert lo < 0.3 < hi


def test_clopper_pearson_edges():
    assert ST.clopper_pearson(0, 10)[0] == 0.0
    assert ST.clopper_pearson(10, 10)[1] == 1.0
    assert np.isnan(ST.clopper_pearson(0, 0)[0])


def test_clopper_pearson_narrows_with_more_data():
    w_small = np.subtract(*reversed(ST.clopper_pearson(3, 10)))
    w_large = np.subtract(*reversed(ST.clopper_pearson(300, 1000)))
    assert w_large < w_small


def test_hit_rate_table_reports_rate_and_ci_per_slice():
    t = ST.hit_rate_table({"EXPLOIT": (8, 16), "CONTROL": (0, 4)})
    assert t["EXPLOIT"]["rate"] == pytest.approx(0.5)
    assert t["CONTROL"]["rate"] == pytest.approx(0.0)
    assert t["EXPLOIT"]["ci95"][0] < 0.5 < t["EXPLOIT"]["ci95"][1]


def test_hit_rate_table_handles_an_empty_slice():
    assert np.isnan(ST.hit_rate_table({"X": (0, 0)})["X"]["rate"])


def test_fisher_detects_a_strong_slice_difference():
    assert ST.fisher_slice_comparison((16, 16), (0, 16)) < 0.01
    assert ST.fisher_slice_comparison((8, 16), (8, 16)) == pytest.approx(1.0)


def test_paired_seed_test_on_identical_inputs_is_not_significant():
    r = ST.paired_seed_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert r["mean_diff"] == pytest.approx(0.0)
    assert r["p_wilcoxon"] == pytest.approx(1.0)


def test_paired_seed_test_detects_a_consistent_improvement():
    base = np.array([0.60, 0.62, 0.61, 0.63, 0.59, 0.64, 0.60, 0.62])
    better = base - 0.05
    r = ST.paired_seed_test(better, base)
    assert r["mean_diff"] < 0 and r["p_wilcoxon"] < 0.05


def test_bootstrap_ci_contains_the_mean_and_is_reproducible():
    rng = np.random.RandomState(0)
    v = rng.normal(5.0, 1.0, 200)
    lo, hi = ST.bootstrap_ci(v, n=400, seed=3)
    assert lo < v.mean() < hi
    assert ST.bootstrap_ci(v, n=400, seed=3) == (lo, hi)


def test_holm_is_monotone_and_bounded():
    adj = ST.holm([0.001, 0.02, 0.04, 0.5])
    assert all(0.0 <= a <= 1.0 for a in adj)
    assert adj[0] >= 0.001 * 4 - 1e-12          # smallest p multiplied by m
    assert adj == sorted(adj)                   # input already ascending


def test_holm_is_more_conservative_than_raw_p():
    raw = [0.01, 0.02, 0.03]
    assert all(a >= r for a, r in zip(ST.holm(raw), raw))


def test_holm_on_a_single_test_is_the_identity():
    assert ST.holm([0.03])[0] == pytest.approx(0.03)
