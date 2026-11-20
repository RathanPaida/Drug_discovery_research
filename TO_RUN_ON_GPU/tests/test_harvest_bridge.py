"""The harvester -> Stage 1 bridge.

WHY THESE TESTS EXIST. `scripts/tnk1_harvest.py` writes a 40-column provenance-first
table; Stage 1's ingest wants the 5-column PKIS shape. Three of the five column names
differ, so the harvest output could not reach the pipeline at all -- and the obvious
workaround (routing it through `from_long_csv` under one of the `*_csv` config keys)
would have overwritten every row's `source` with a single literal string, destroying the
provenance the harvester exists to record.

`test_from_long_csv_cannot_read_a_harvest_table` pins the first half of that: it is a
regression test for the defect, and it must keep failing against `from_long_csv` forever.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tnk1.data import activity, extract


# --------------------------------------------------------------------------- fixtures

def _harvest_rows() -> pd.DataFrame:
    """A master table covering every branch the filter has to take.

    Row 0  clean exact IC50, ChEMBL                                    -> KEPT
    Row 1  clean exact Ki, BindingDB                                   -> KEPT
    Row 2  right-censored (>) Kd                                       -> KEPT (censored)
    Row 3  left-censored (<)                                           -> dropped
    Row 4  KIBA score (not a concentration)                            -> dropped
    Row 5  percent activity                                            -> dropped
    Row 6  composite semicolon-joined target                           -> dropped
    Row 7  contamination flag set (tankyrase in assay description)     -> dropped
    Row 8  invalid SMILES                                              -> dropped
    Row 9  off-panel target                                            -> dropped
    Row 10 GtoPdb pre-aggregated median                                -> dropped
    Row 11 qualitative outcome, no value                               -> dropped
    Row 12 value_nM missing                                            -> dropped
    Row 13 Davis competition-binding Kd                                -> KEPT
    """
    base = dict(record_class="quantitative", in_panel=True, structure_status="ok",
                flag_contamination="", measure_semantics="chembl_standardised_activity")
    r = [
        {**base, "std_smiles": "CCO", "uniprot_accession": "Q13470", "activity_type": "IC50",
         "relation": "=", "value_nM": 120.0, "source": "chembl"},
        {**base, "std_smiles": "CCN", "uniprot_accession": "Q13470", "activity_type": "Ki",
         "relation": "=", "value_nM": 45.0, "source": "bindingdb",
         "measure_semantics": "bindingdb_measured_affinity_nM"},
        {**base, "std_smiles": "CCC", "uniprot_accession": "Q13470", "activity_type": "Kd",
         "relation": ">", "value_nM": 10000.0, "source": "chembl"},
        {**base, "std_smiles": "CCCC", "uniprot_accession": "Q13470", "activity_type": "IC50",
         "relation": "<", "value_nM": 1.0, "source": "chembl"},
        {**base, "std_smiles": "CCCCC", "uniprot_accession": "Q13470",
         "activity_type": "KIBA_score", "relation": "=", "value_nM": 11.2, "source": "kiba",
         "record_class": "matrix_score",
         "measure_semantics": "kiba_score_NOT_IC50_Ki_Kd(Tang2014; model-integrated; do not convert)"},
        {**base, "std_smiles": "CCCCCC", "uniprot_accession": "Q13470",
         "activity_type": "PCT_INHIBITION", "relation": "=", "value_nM": np.nan,
         "source": "pubchem", "record_class": "single_point",
         "measure_semantics": "pubchem_assay_column_pct"},
        {**base, "std_smiles": "CCCCCCC", "uniprot_accession": "Q13470;P00519",
         "activity_type": "IC50", "relation": "=", "value_nM": 300.0, "source": "chembl"},
        {**base, "std_smiles": "CCCCCCCC", "uniprot_accession": "Q13470",
         "activity_type": "IC50", "relation": "=", "value_nM": 55.0, "source": "chembl",
         "flag_contamination": "tankyrase_or_PARP_in_assay_description"},
        {**base, "std_smiles": "not_a_smiles", "uniprot_accession": "Q13470",
         "activity_type": "IC50", "relation": "=", "value_nM": 77.0, "source": "chembl",
         "structure_status": "invalid_smiles"},
        {**base, "std_smiles": "CCCCCCCCC", "uniprot_accession": "P99999",
         "activity_type": "IC50", "relation": "=", "value_nM": 88.0, "source": "chembl",
         "in_panel": False},
        {**base, "std_smiles": "CCCCCCCCCC", "uniprot_accession": "Q13470",
         "activity_type": "Ki", "relation": "=", "value_nM": 210.0, "source": "iuphar",
         "measure_semantics": "gtopdb_curated_summary_pAffinity(median of curated experiments; "
                              "NOT a single measurement)"},
        {**base, "std_smiles": "CCCCCCCCCCC", "uniprot_accession": "Q13470",
         "activity_type": "IC50", "relation": "=", "value_nM": np.nan, "source": "pubchem",
         "record_class": "qualitative",
         "measure_semantics": "pubchem_activity_outcome_only"},
        {**base, "std_smiles": "CCCCCCCCCCCC", "uniprot_accession": "Q13470",
         "activity_type": "IC50", "relation": "=", "value_nM": np.nan, "source": "chembl"},
        {**base, "std_smiles": "CCCCCCCCCCCCC", "uniprot_accession": "Q13470",
         "activity_type": "Kd", "relation": "=", "value_nM": 500.0, "source": "kinomescan",
         "measure_semantics": "kd_nM_matrix_Davis2011"},
    ]
    return pd.DataFrame(r)


@pytest.fixture()
def master_csv(tmp_path):
    p = tmp_path / "TNK1_kinase_master.csv"
    _harvest_rows().to_csv(p, index=False)
    return p


# ------------------------------------------------------- the defect this bridge closes

def test_from_long_csv_cannot_read_a_harvest_table(master_csv):
    """REGRESSION. The reason the bridge had to exist.

    `from_long_csv` ends with df[["smiles","uniprot_id","standard_type",...]]. The
    harvester calls those columns std_smiles / uniprot_accession / activity_type, so this
    raises. If someone ever "fixes" this by renaming the harvester's columns, the
    provenance test below is the second line of defence.
    """
    with pytest.raises(KeyError):
        extract.from_long_csv(master_csv, "harvest")


def test_from_long_csv_would_have_destroyed_provenance(tmp_path):
    """Even with matching names, the PKIS path flattens `source` to one literal."""
    p = tmp_path / "renamed.csv"
    pd.DataFrame({"smiles": ["CCO", "CCN"], "uniprot_id": ["Q13470"] * 2,
                  "standard_type": ["IC50"] * 2, "relation": ["="] * 2,
                  "value_nM": [1.0, 2.0], "source": ["chembl", "bindingdb"]}).to_csv(p, index=False)
    out = extract.from_long_csv(p, "harvest")
    assert set(out.source) == {"harvest"}, (
        "from_long_csv overwrites source; this is why from_harvest_csv exists")


# ------------------------------------------------------------------------- the schema

def test_returns_exactly_stage1_columns(master_csv):
    out, _ = extract.from_harvest_csv(master_csv)
    assert list(out.columns) == ["smiles", "uniprot_id", "standard_type",
                                 "relation", "value_nM", "source"]


def test_output_concatenates_with_the_other_extractors(master_csv):
    """The whole point: harvest rows must stack with ChEMBL/PKIS rows without coercion."""
    out, _ = extract.from_harvest_csv(master_csv)
    pkis = pd.DataFrame({"smiles": ["c1ccccc1"], "uniprot_id": ["Q13470"],
                         "standard_type": ["IC50"], "relation": ["="],
                         "value_nM": [9.0], "source": ["pkis"]})
    both = pd.concat([out, pkis], ignore_index=True)
    assert len(both) == len(out) + 1
    assert both.value_nM.notna().all()


def test_per_row_source_is_preserved(master_csv):
    out, rep = extract.from_harvest_csv(master_csv)
    assert set(out.source) == {"chembl", "bindingdb", "kinomescan"}
    assert rep["kept_by_source"]["chembl"] == 2
    assert rep["kept_by_source"]["bindingdb"] == 1
    assert rep["kept_by_source"]["kinomescan"] == 1


# ------------------------------------------------------------------------ the filters

def test_keeps_exactly_the_four_usable_rows(master_csv):
    out, rep = extract.from_harvest_csv(master_csv)
    assert rep["n_input"] == 14
    assert rep["n_kept"] == 4 == len(out)
    assert sorted(out.smiles) == sorted(["CCO", "CCN", "CCC", "CCCCCCCCCCCCC"])


def test_right_censored_rows_are_kept_by_default(master_csv):
    """The harvester's own is_training_ready_exact demands relation == '='. The pipeline
    models censored rows, so discarding them here would throw away usable signal."""
    out, rep = extract.from_harvest_csv(master_csv)
    assert rep["n_censored"] == 1
    assert (out.relation == ">").sum() == 1


def test_censored_rows_can_be_excluded(master_csv):
    out, rep = extract.from_harvest_csv(master_csv, allow_censored=False)
    assert rep["n_censored"] == 0
    assert rep["dropped"]["censored_excluded"] == 1
    assert set(out.relation) == {"="}


def test_left_censored_dropped_and_counted(master_csv):
    _, rep = extract.from_harvest_csv(master_csv)
    assert rep["dropped"]["left_censored"] == 1


@pytest.mark.parametrize("reason,n", [
    ("not_allowed_type", 2),       # KIBA_score, PCT_INHIBITION
    ("not_quantitative", 1),       # the outcome-only PubChem row
    ("bad_value", 1),              # IC50 row with no value_nM
    ("composite_target", 1),
    ("contaminated", 1),
    ("invalid_structure", 1),
    ("not_in_panel", 1),
    ("excluded_semantics", 1),     # the GtoPdb pre-aggregated median
])
def test_each_exclusion_is_counted(master_csv, reason, n):
    _, rep = extract.from_harvest_csv(master_csv)
    assert rep["dropped"][reason] == n, rep["dropped"]


def test_non_concentration_readouts_are_not_blamed_on_a_bad_value(master_csv):
    """A PCT_INHIBITION or KIBA_score row has no nM value by construction.

    Testing value_nM before activity_type attributed those rows to `bad_value`, which
    reads as a unit-corrupted measurement -- the exact symptom of a real defect this
    project has already been bitten by (107,795 PubChem Ki rows not in nM). It would send
    a reader hunting a conversion bug in rows that are simply not concentrations.
    """
    _, rep = extract.from_harvest_csv(master_csv)
    assert rep["dropped"]["bad_value"] == 1, (
        "bad_value must count only rows that claim to be a quantitative concentration "
        "and are not one")
    assert rep["dropped"]["not_allowed_type"] == 2


def test_kiba_score_never_enters_the_main_table(master_csv):
    """A KIBA score fuses Ki/Kd/IC50 through a model. It is not a concentration, so
    `pAct = 9 - log10(v)` is meaningless for it."""
    out, _ = extract.from_harvest_csv(master_csv)
    assert "KIBA_score" not in set(out.standard_type)
    assert 11.2 not in set(out.value_nM)


def test_pre_aggregated_medians_are_excluded(master_csv):
    """GtoPdb summary affinities are already a median over experiments. Passing one into
    aggregate_replicates aggregates it twice and records n_rep=1 for many measurements."""
    out, _ = extract.from_harvest_csv(master_csv)
    assert "iuphar" not in set(out.source)


def test_exclusion_list_is_configurable(master_csv):
    """Admitting GtoPdb must be possible, but only as an explicit choice."""
    out, _ = extract.from_harvest_csv(master_csv, exclude_semantics=())
    assert "iuphar" in set(out.source)


def test_composite_targets_can_be_retained(master_csv):
    out, _ = extract.from_harvest_csv(master_csv, drop_composite=False)
    assert any(";" in u for u in out.uniprot_id)


def test_dropped_counts_reconcile_with_the_total(master_csv):
    """Every input row is either kept or attributed to exactly one reason."""
    _, rep = extract.from_harvest_csv(master_csv)
    assert sum(rep["dropped"].values()) + rep["n_kept"] == rep["n_input"]


# -------------------------------------------------------------- semantics bookkeeping

def test_distinguish_semantics_separates_competition_binding_kd(master_csv):
    """Davis Kd_app and solution Kd are both `Kd`, and aggregate_replicates groups on
    standard_type -- so by default they pool. This flag keeps them apart."""
    plain, _ = extract.from_harvest_csv(master_csv)
    tagged, _ = extract.from_harvest_csv(master_csv, distinguish_semantics=True)
    assert set(plain.standard_type) == {"IC50", "Ki", "Kd"}
    assert "Kd@kd_nM_matrix_Davis2011" in set(tagged.standard_type)
    assert "Kd" not in set(tagged.standard_type)


def test_semantics_counts_are_reported(master_csv):
    _, rep = extract.from_harvest_csv(master_csv)
    assert rep["kept_by_semantics"]["kd_nM_matrix_Davis2011"] == 1
    assert rep["kept_by_semantics"]["chembl_standardised_activity"] == 2


# ------------------------------------------------------------------- robustness

def test_missing_required_column_raises_a_named_error(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"smiles": ["CCO"], "value_nM": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="tnk1_harvest master table"):
        extract.from_harvest_csv(p)


def test_error_names_the_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"std_smiles": ["CCO"], "value_nM": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError) as e:
        extract.from_harvest_csv(p)
    assert "uniprot_accession" in str(e.value) and "activity_type" in str(e.value)


@pytest.mark.parametrize("chunksize", [1, 2, 3, 5, 1000])
def test_chunking_does_not_change_the_result(master_csv, chunksize):
    """The master table is read in chunks because it can be millions of rows. A filter
    that behaved differently per chunk boundary would be invisible at small N."""
    ref, rref = extract.from_harvest_csv(master_csv, chunksize=1000)
    out, rep = extract.from_harvest_csv(master_csv, chunksize=chunksize)
    pd.testing.assert_frame_equal(
        out.sort_values(list(out.columns)).reset_index(drop=True),
        ref.sort_values(list(ref.columns)).reset_index(drop=True))
    assert rep["n_kept"] == rref["n_kept"]
    assert rep["dropped"] == rref["dropped"]


def test_smiles_column_fallback_order(tmp_path):
    """std_smiles preferred; smiles_used next; smiles_raw last."""
    rows = {"uniprot_accession": ["Q13470"], "activity_type": ["IC50"], "relation": ["="],
            "value_nM": [10.0], "source": ["chembl"], "record_class": ["quantitative"],
            "in_panel": [True]}
    p1 = tmp_path / "a.csv"
    pd.DataFrame({**rows, "std_smiles": ["STD"], "smiles_used": ["USED"],
                  "smiles_raw": ["RAW"]}).to_csv(p1, index=False)
    assert extract.from_harvest_csv(p1)[0].smiles.iloc[0] == "STD"

    p2 = tmp_path / "b.csv"
    pd.DataFrame({**rows, "smiles_used": ["USED"], "smiles_raw": ["RAW"]}).to_csv(p2, index=False)
    assert extract.from_harvest_csv(p2)[0].smiles.iloc[0] == "USED"

    p3 = tmp_path / "c.csv"
    pd.DataFrame({**rows, "smiles_raw": ["RAW"]}).to_csv(p3, index=False)
    assert extract.from_harvest_csv(p3)[0].smiles.iloc[0] == "RAW"


def test_blank_relation_is_treated_as_exact(tmp_path):
    p = tmp_path / "blank.csv"
    pd.DataFrame({"std_smiles": ["CCO"], "uniprot_accession": ["Q13470"],
                  "activity_type": ["IC50"], "relation": [""], "value_nM": [10.0],
                  "source": ["chembl"], "record_class": ["quantitative"],
                  "in_panel": [True]}).to_csv(p, index=False)
    out, _ = extract.from_harvest_csv(p)
    assert len(out) == 1 and out.relation.iloc[0] == "="


def test_an_all_rejected_table_returns_an_empty_frame_not_an_error(tmp_path):
    p = tmp_path / "none.csv"
    pd.DataFrame({"std_smiles": ["CCO"], "uniprot_accession": ["Q13470"],
                  "activity_type": ["PCT_INHIBITION"], "relation": ["="],
                  "value_nM": [50.0], "source": ["pubchem"],
                  "record_class": ["single_point"], "in_panel": [True]}).to_csv(p, index=False)
    out, rep = extract.from_harvest_csv(p)
    assert len(out) == 0 and rep["n_kept"] == 0
    assert list(out.columns) == ["smiles", "uniprot_id", "standard_type",
                                 "relation", "value_nM", "source"]


def test_report_formats_without_raising(master_csv):
    _, rep = extract.from_harvest_csv(master_csv)
    txt = extract.format_harvest_report(rep)
    assert "harvest rows read" in txt and "kept" in txt


# --------------------------------------------- integration with the rest of Stage 1

def test_bridge_output_flows_through_prepare_and_aggregate(master_csv):
    """End-to-end on the real downstream functions.

    Stage 1 adds inchikey_block1/scaffold during standardisation; this test supplies them
    so the two activity functions can be exercised without rdkit.
    """
    out, _ = extract.from_harvest_csv(master_csv)
    out = out.assign(inchikey_block1=[f"KEY{i}" for i in range(len(out))],
                     scaffold=["c1ccccc1"] * len(out))
    rec = activity.prepare_records(out)
    assert set(rec.columns) >= {"censored", "pact", "relation"}
    assert rec.censored.sum() == 1

    agg = activity.aggregate_replicates(rec, range_max=1.0, policy="drop")
    assert len(agg) == 4
    assert agg.pact.between(activity.PACT_MIN, activity.PACT_MAX).all()
    assert set(agg.source) >= {"chembl", "bindingdb", "kinomescan"}


def test_censored_row_keeps_its_own_bound(master_csv):
    """A '>10 uM' row must bound pActivity at 5, not inherit a global tau."""
    out, _ = extract.from_harvest_csv(master_csv)
    out = out.assign(inchikey_block1=[f"K{i}" for i in range(len(out))],
                     scaffold=["c1ccccc1"] * len(out))
    rec = activity.prepare_records(out)
    cen = rec[rec.censored == 1]
    assert len(cen) == 1
    assert cen.pact.iloc[0] == pytest.approx(9.0 - np.log10(10000.0))


def test_assay_types_are_not_pooled_across_standard_type(master_csv):
    """IC50/Ki/Kd of the SAME compound must stay separate rows: they are different
    quantities and aggregate_replicates groups on standard_type to keep them so."""
    out, _ = extract.from_harvest_csv(master_csv)
    one = out.iloc[:1]
    same = pd.concat([one.assign(standard_type=t, value_nM=v)
                      for t, v in (("IC50", 100.0), ("Ki", 50.0), ("Kd", 25.0))],
                     ignore_index=True)
    same = same.assign(inchikey_block1="SAME", scaffold="c1ccccc1")
    agg = activity.aggregate_replicates(activity.prepare_records(same))
    assert len(agg) == 3, "assay types were pooled into one row"
    assert set(agg.standard_type) == {"IC50", "Ki", "Kd"}
