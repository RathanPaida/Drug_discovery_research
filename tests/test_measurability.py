"""Regressions for the Stage-1 measurability filter.

The defect these pin was the largest in the pipeline: `run_stage1_data.py` filtered the
kinome store with `keep = np.isfinite(pActivity)` and its comment asserted that
%-inhibition and qualitative rows "are dropped with a printed count". They were not. The
store encodes "no dose-response value" as pActivity **0.0**, and `np.isfinite(0.0)` is
True, so of 4,798,165 rows in `kinome_all_measurements_3.95M.parquet`:

  3,374,532 (70.3%) carried pActivity exactly 0.0 with is_censored=0 -- handed to
                    `task_loss` as an EXACT measurement of IC50 = 1 molar;
    108,450         carried 0 != pActivity < 2.0 (median value 3.2e14 nM = 316,000 M),
                    107,795 of them PubChem `Ki` rows whose value was not in nM;
    211,546         named a semicolon-joined set of up to ~150 UniProt accessions rather
                    than one kinase, and were discarded later with no count.

`kinome_long_v2.parquet` was built before the filter and holds 3,372,521 poisoned rows.
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

from tnk1.data.activity import (  # noqa: E402
    DOSE_RESPONSE_TYPES, PACT_MAX, PACT_MIN, SENTINEL_PACT,
    filter_measurable, format_filter_report,
)


def frame(rows):
    return pd.DataFrame(rows, columns=["activity_type", "pActivity", "uniprot_accession"])


def test_the_zero_sentinel_is_dropped_even_though_it_is_finite():
    """isfinite(0.0) is True -- that is the whole bug."""
    assert np.isfinite(SENTINEL_PACT), "guard: the sentinel really is a finite float"
    d = frame([
        ("IC50", 7.0, "Q13470"),
        ("IC50", 0.0, "Q13470"),       # sentinel on a dose-response row
    ])
    kept, rep = filter_measurable(d, target_col="uniprot_accession")
    assert len(kept) == 1
    assert kept.pActivity.tolist() == [7.0]
    assert rep["dropped"]["pact_sentinel_zero"] == 1


def test_non_dose_response_readouts_are_dropped_with_a_count():
    d = frame([
        ("IC50", 7.0, "Q13470"),
        ("OUTCOME", 0.0, "Q13470"),
        ("PCT_INHIBITION", 0.0, "Q13470"),
        ("kon", 0.0, "Q13470"),
        ("koff", 0.0, "Q13470"),
        ("Residual Activity", 0.0, "Q13470"),
    ])
    kept, rep = filter_measurable(d, target_col="uniprot_accession")
    assert len(kept) == 1
    assert rep["dropped"]["not_dose_response"] == 5
    # the count must be attributable, not just a total
    assert rep["dropped_types"]["OUTCOME"] == 1
    assert rep["dropped_types"]["kon"] == 1


def test_physically_impossible_values_are_dropped():
    """pAct -5.2 == 1.6e14 nM == 158,000 molar. 107,795 PubChem Ki rows looked like this."""
    d = frame([
        ("Ki", 8.0, "P00519"),
        ("Ki", -5.2, "P00519"),        # unit-corrupted
        ("Ki", 1.5, "P00519"),         # 10 mM, below pact_min
        ("Ki", 13.0, "P00519"),        # 0.1 pM, above pact_max
    ])
    kept, rep = filter_measurable(d, target_col="uniprot_accession")
    assert kept.pActivity.tolist() == [8.0]
    assert rep["dropped"]["pact_out_of_window"] == 3
    assert rep["pact_window"] == [PACT_MIN, PACT_MAX]


def test_window_boundaries_are_inclusive():
    d = frame([("IC50", PACT_MIN, "Q13470"), ("IC50", PACT_MAX, "Q13470")])
    kept, _ = filter_measurable(d, target_col="uniprot_accession")
    assert len(kept) == 2, "the documented window is closed, not open"


def test_composite_targets_are_dropped_and_counted():
    d = frame([
        ("IC50", 7.0, "Q13470"),
        ("IC50", 7.5, "Q07912;P00519;P12931"),
        ("IC50", 7.6, "P12931;P00519"),
    ])
    kept, rep = filter_measurable(d, target_col="uniprot_accession")
    assert kept.uniprot_accession.tolist() == ["Q13470"]
    assert rep["dropped"]["composite_target"] == 2
    assert rep["n_composite_target_values"] == 2


def test_composite_targets_can_be_retained_deliberately():
    d = frame([("IC50", 7.5, "Q07912;P00519")])
    kept, rep = filter_measurable(d, target_col="uniprot_accession",
                                  drop_composite_targets=False)
    assert len(kept) == 1
    assert "composite_target" not in rep["dropped"]
    assert "note_composite" in rep


def test_non_finite_values_are_dropped_separately_from_the_sentinel():
    d = frame([
        ("IC50", 7.0, "Q13470"),
        ("IC50", np.nan, "Q13470"),
        ("IC50", np.inf, "Q13470"),
    ])
    kept, rep = filter_measurable(d, target_col="uniprot_accession")
    assert len(kept) == 1
    assert rep["dropped"]["pact_not_finite"] == 2
    assert rep["dropped"]["pact_sentinel_zero"] == 0


def test_counts_are_mutually_exclusive_and_sum_to_the_total():
    """Every input row must be either kept or counted in exactly one drop bucket."""
    d = frame([
        ("IC50", 7.0, "Q13470"),                  # keep
        ("IC50", 0.0, "Q13470"),                  # sentinel
        ("Ki", -5.2, "P00519"),                   # out of window
        ("OUTCOME", 0.0, "Q13470"),               # not dose-response
        ("IC50", 7.5, "Q07912;P00519"),           # composite
        ("IC50", np.nan, "Q13470"),               # not finite
    ])
    kept, rep = filter_measurable(d, target_col="uniprot_accession")
    assert rep["n_kept"] == len(kept) == 1
    assert sum(rep["dropped"].values()) == rep["n_dropped"] == 5
    assert rep["n_kept"] + rep["n_dropped"] == rep["n_input"] == 6


def test_all_configured_dose_types_survive():
    d = frame([(t, 7.0, "Q13470") for t in DOSE_RESPONSE_TYPES])
    kept, rep = filter_measurable(d, target_col="uniprot_accession")
    assert len(kept) == len(DOSE_RESPONSE_TYPES)
    assert rep["dropped"]["not_dose_response"] == 0


def test_custom_types_and_window_are_honoured():
    d = frame([("IC50", 7.0, "Q13470"), ("Kd", 7.0, "Q13470")])
    kept, _ = filter_measurable(d, target_col="uniprot_accession", types=("IC50",))
    assert kept.activity_type.tolist() == ["IC50"]
    kept2, rep2 = filter_measurable(d, target_col="uniprot_accession",
                                    pact_min=7.5, pact_max=9.0)
    assert len(kept2) == 0
    assert rep2["dropped"]["pact_out_of_window"] == 2


def test_missing_type_column_is_reported_not_guessed():
    d = pd.DataFrame({"pActivity": [7.0, 0.0], "uniprot_accession": ["Q13470"] * 2})
    kept, rep = filter_measurable(d, target_col="uniprot_accession")
    assert len(kept) == 1                       # sentinel still removed
    assert "note" in rep and "activity_type" in rep["note"]


def test_empty_frame_does_not_raise():
    d = frame([])
    kept, rep = filter_measurable(d, target_col="uniprot_accession")
    assert len(kept) == 0 and rep["n_input"] == 0


def test_report_formats_without_crashing():
    d = frame([("IC50", 7.0, "Q13470"), ("OUTCOME", 0.0, "Q13470")])
    _, rep = filter_measurable(d, target_col="uniprot_accession")
    text = format_filter_report(rep)
    assert "kept" in text and "not_dose_response" in text


# ------------------------------------------------------- the Ctx contamination guard
def test_ctx_guard_rejects_a_sentinel_contaminated_table(tmp_path, monkeypatch):
    """Ctx must refuse to open a poisoned frozen table rather than train on it."""
    pytest.importorskip("pyarrow")
    from tnk1.utils.common import save_locked

    proc = tmp_path / "processed"
    proc.mkdir()
    # 70% sentinel rows, mirroring kinome_long_v2.parquet
    n = 100
    df = pd.DataFrame({
        "inchikey_block1": [f"K{i:013d}" for i in range(n)],
        "smiles": ["c1ccccc1"] * n,
        "scaffold": ["c1ccccc1"] * n,
        "uniprot_id": ["Q13470"] * n,
        "standard_type": ["IC50"] * n,
        "pact": [0.0] * 70 + [7.0] * 30,
        "censored": [0] * n,
        "n_rep": [1] * n,
        "sigma_rep": [0.0] * n,
        "source": ["test"] * n,
        "weight": [1.0] * n,
    })
    save_locked(df, proc / "kinome_long_vtest.parquet")
    (proc / "splits_vtest.json").write_text('{"test_keys": [], "dev_keys": []}')

    import yaml
    cfg = {"paths": {"raw": str(tmp_path / "raw"), "processed": str(proc),
                     "runs": str(tmp_path / "runs")},
           "targets": {"tnk1": "Q13470"}, "model": {"n_desc": 200}, "data": {}}
    cfg_p = tmp_path / "c.yaml"
    cfg_p.write_text(yaml.safe_dump(cfg))

    from tnk1.context import Ctx
    monkeypatch.delenv("TNK1_ALLOW_SENTINEL_TARGETS", raising=False)
    with pytest.raises(RuntimeError, match="sentinel-contaminated"):
        Ctx(str(cfg_p), version="vtest", device="cpu")

    # the override must downgrade it to a warning, for inspection
    monkeypatch.setenv("TNK1_ALLOW_SENTINEL_TARGETS", "1")
    with pytest.warns(RuntimeWarning, match="sentinel-contaminated"):
        C = Ctx(str(cfg_p), version="vtest", device="cpu")
    assert len(C.df) == n
