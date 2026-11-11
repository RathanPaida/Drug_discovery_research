"""Tests for the bioactivity ingest transform (pActivity scale, censoring, replicate
aggregation) and the kinome sample weighting.

Pure numpy/pandas, so this runs in a core-only install -- no torch, no RDKit, no data
store. Writing these surfaced defects #23 and #24 in PIPELINE_AUDIT.md.
"""
import numpy as np
import pandas as pd
import pytest

from tnk1.data import activity
from tnk1.models.weighting import sample_weights


# --------------------------------------------------------------------- pActivity transform

@pytest.mark.parametrize("nM, expected", [
    (1.0, 9.0),        # 1 nM    -> 9
    (10.0, 8.0),       # 10 nM   -> 8
    (100.0, 7.0),      # 100 nM  -> 7
    (1_000.0, 6.0),    # 1 uM    -> 6
    (10_000.0, 5.0),   # 10 uM   -> 5  (the censor threshold)
    (100_000.0, 4.0),  # 100 uM  -> 4
])
def test_to_pactivity_anchors(nM, expected):
    assert activity.to_pactivity(nM) == pytest.approx(expected)


def test_to_pactivity_is_monotone_decreasing():
    v = activity.to_pactivity([1.0, 10.0, 100.0, 1000.0])
    assert np.all(np.diff(v) < 0)


def test_to_pactivity_vectorised():
    assert activity.to_pactivity([100.0, 10_000.0]) == pytest.approx([7.0, 5.0])


# ------------------------------------------------------------------------ prepare_records

def _raw(**over):
    base = dict(inchikey_block1=["AAAAAAAAAAAAAA"], smiles=["CCO"], scaffold=["c1ccccc1"],
                uniprot_id=["Q13470"], standard_type=["IC50"], value_nM=[100.0], source=["chembl"])
    base.update(over)
    return pd.DataFrame(base)


def test_missing_relation_column_is_treated_as_exact():
    """Regression: configs/default.yaml declares the PKIS and KINOMEscan inputs with columns
    `smiles,uniprot_id,value_nM,standard_type` -- no `relation`. `df.get("relation", "=")`
    returns the *scalar* "=" in that case, and the old `.fillna()` on it raised
    AttributeError: 'str' object has no attribute 'fillna'. Ingest therefore crashed on
    exactly the datasets it was written to accept."""
    out = activity.prepare_records(_raw())
    assert len(out) == 1
    assert out.relation.iloc[0] == "="
    assert int(out.censored.iloc[0]) == 0
    assert out.pact.iloc[0] == pytest.approx(7.0)


@pytest.mark.parametrize("blank", [None, np.nan, ""])
def test_blank_relation_is_treated_as_exact(blank):
    out = activity.prepare_records(_raw(relation=[blank]))
    assert int(out.censored.iloc[0]) == 0


@pytest.mark.parametrize("rel", [">", ">=", ">>"])
def test_right_censored_relations_are_flagged(rel):
    out = activity.prepare_records(_raw(relation=[rel]))
    assert int(out.censored.iloc[0]) == 1


@pytest.mark.parametrize("rel", ["<", "<=", "<<"])
def test_left_censored_relations_are_dropped(rel):
    """The manual's loss does not model left-censored (unexpectedly potent) records."""
    assert activity.prepare_records(_raw(relation=[rel])).empty


def test_nonpositive_values_are_dropped():
    df = pd.DataFrame(dict(
        inchikey_block1=list("ABC"), smiles=["C"] * 3, scaffold=["s"] * 3,
        uniprot_id=["Q13470"] * 3, standard_type=["IC50"] * 3, source=["x"] * 3,
        value_nM=[0.0, -5.0, 100.0]))
    out = activity.prepare_records(df)
    assert len(out) == 1 and out.value_nM.iloc[0] == 100.0


def test_censored_rows_keep_their_own_per_row_bound():
    """A >1 uM record bounds pActivity at 6; a >100 uM record at 4. Collapsing both onto a
    single global tau loses real information, so prepare_records must not do it."""
    df = pd.DataFrame(dict(
        inchikey_block1=["A", "B"], smiles=["C", "CC"], scaffold=["s", "s"],
        uniprot_id=["Q13470"] * 2, standard_type=["IC50"] * 2, source=["x"] * 2,
        value_nM=[1_000.0, 100_000.0], relation=[">", ">"]))
    out = activity.prepare_records(df)
    assert sorted(out.pact.round(6)) == [4.0, 6.0]


def test_prepare_records_no_longer_accepts_a_dead_tau_argument():
    """`prepare_records(df, tau)` accepted `tau` and never read it, while
    run_stage1_data.py passed cfg.data.censor_tau expecting it to have an effect."""
    with pytest.raises(TypeError):
        activity.prepare_records(_raw(), 5.0)


def test_prepare_records_does_not_mutate_its_input():
    df = _raw(relation=[">"])
    before = df.copy()
    activity.prepare_records(df)
    pd.testing.assert_frame_equal(df, before)


# --------------------------------------------------------------------- aggregate_replicates

def _rec(pacts, censored=None, keys=None, source="x"):
    n = len(pacts)
    return pd.DataFrame({
        "inchikey_block1": keys or ["A"] * n, "uniprot_id": ["Q13470"] * n,
        "standard_type": ["IC50"] * n, "smiles": ["CCO"] * n, "scaffold": ["s"] * n,
        "pact": pacts, "censored": censored or [0] * n, "source": [source] * n,
    })


def test_replicates_are_median_aggregated():
    out = activity.aggregate_replicates(_rec([7.0, 7.2, 7.4]))
    assert len(out) == 1
    assert out.pact.iloc[0] == pytest.approx(7.2)
    assert int(out.n_rep.iloc[0]) == 3
    assert out.sigma_rep.iloc[0] > 0


def test_single_measurement_has_undefined_sigma():
    out = activity.aggregate_replicates(_rec([7.0]))
    assert int(out.n_rep.iloc[0]) == 1
    assert np.isnan(out.sigma_rep.iloc[0])


def test_discordant_replicates_are_dropped_under_drop_policy():
    """A spread of 2.0 log units exceeds range_max=1.0."""
    assert activity.aggregate_replicates(_rec([6.0, 8.0]), range_max=1.0, policy="drop").empty


def test_discordant_replicates_are_kept_under_keep_median_policy():
    out = activity.aggregate_replicates(_rec([6.0, 8.0]), range_max=1.0, policy="keep_median")
    assert len(out) == 1 and out.pact.iloc[0] == pytest.approx(7.0)


def test_concordant_replicates_survive_the_range_check():
    out = activity.aggregate_replicates(_rec([7.0, 7.3]), range_max=1.0, policy="drop")
    assert len(out) == 1


def test_exact_measurements_take_precedence_over_censored():
    out = activity.aggregate_replicates(_rec([7.0, 5.0], censored=[0, 1]))
    assert int(out.censored.iloc[0]) == 0
    assert out.pact.iloc[0] == pytest.approx(7.0)


def test_censored_only_group_keeps_the_tightest_bound():
    """True pActivity lies below every bound, so the smallest bound is the binding one."""
    out = activity.aggregate_replicates(_rec([6.0, 4.0, 5.0], censored=[1, 1, 1]))
    assert int(out.censored.iloc[0]) == 1
    assert out.pact.iloc[0] == pytest.approx(4.0)


def test_distinct_compounds_are_aggregated_separately():
    out = activity.aggregate_replicates(_rec([7.0, 8.0], keys=["A", "B"]))
    assert len(out) == 2
    assert set(out.inchikey_block1) == {"A", "B"}


def test_sources_are_merged_and_sorted():
    d = _rec([7.0, 7.1])
    d["source"] = ["bindingdb", "chembl"]
    assert activity.aggregate_replicates(d).source.iloc[0] == "bindingdb|chembl"


def test_aggregate_sets_a_default_weight():
    assert activity.aggregate_replicates(_rec([7.0])).weight.iloc[0] == pytest.approx(1.0)


# ------------------------------------------------------------------------- sample_weights

def test_sample_weights_follow_the_documented_formula():
    """w_ik = (1 + lambda*S_k^2) * min(1, sqrt(cap_n / N_k))"""
    w = sample_weights(["P1"], {"P1": 0.5}, {"P1": 2000}, lam=3.0, cap_n=2000)
    assert w[0] == pytest.approx((1 + 3.0 * 0.25) * 1.0)


def test_sample_weights_downweight_over_represented_kinases():
    w = sample_weights(["P1"], {"P1": 0.0}, {"P1": 8000}, lam=0.0, cap_n=2000)
    assert w[0] == pytest.approx(0.5)      # sqrt(2000/8000)


def test_sample_weights_cap_at_one_for_scarce_kinases():
    w = sample_weights(["P1"], {"P1": 0.0}, {"P1": 10}, lam=0.0, cap_n=2000)
    assert w[0] == pytest.approx(1.0)


def test_sample_weights_reward_pocket_similarity():
    low = sample_weights(["P1"], {"P1": 0.1}, {"P1": 2000}, lam=3.0)[0]
    high = sample_weights(["P1"], {"P1": 0.9}, {"P1": 2000}, lam=3.0)[0]
    assert high > low


def test_sample_weights_default_unknown_kinases_to_zero_similarity():
    w = sample_weights(["UNKNOWN"], {}, {}, lam=3.0)
    assert w[0] == pytest.approx(1.0)      # S=0 -> no boost, N=1 -> cap 1.0


def test_sample_weights_never_divide_by_zero_counts():
    assert np.isfinite(sample_weights(["P1"], {"P1": 1.0}, {"P1": 0}, lam=1.0)).all()


def test_sample_weights_returns_float32():
    assert sample_weights(["P1"], {"P1": 0.5}, {"P1": 100}, lam=1.0).dtype == np.float32
