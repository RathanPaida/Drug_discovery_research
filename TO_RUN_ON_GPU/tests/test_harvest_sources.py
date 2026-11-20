"""The harvester's source registry, and the offline data-gap audit.

Hermetic: nothing here touches the network. The harvester module imports `requests`, which
is not a core dependency, so the registry tests skip cleanly without it; the audit's own
summarisation is pure pandas and always runs.

WHAT THESE TESTS ARE FOR. Adding a source to this harvester means editing four separate
places -- `CONFIG["SOURCES"]`, `COLLECTORS`, `SOURCE_ORDER_RUN` and `SOURCE_ORDER_MASTER`
-- and missing one fails in a way the 2026-10-26 harvest already demonstrated is easy to
overlook: a source silently does not run, or runs and is left out of the master table, and
the summary line says "8 of 11" without saying which failure it was. These tests make the
four lists agree or fail.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _harvest():
    """Load the harvester as a module. It is a CLI entry point, not a package."""
    pytest.importorskip("requests", reason="the harvester imports requests")
    spec = importlib.util.spec_from_file_location(
        "tnk1_harvest", ROOT / "scripts" / "tnk1_harvest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -------------------------------------------------------------- the source registry

NEW_SOURCES = ("klaeger", "anastassiadis", "karaman", "papyrus")
REPAIRED = ("dtc", "kiba", "metz")


def test_every_configured_source_has_a_collector():
    """The failure this prevents: a source enabled in CONFIG with no function to call."""
    H = _harvest()
    missing = sorted(set(H.CONFIG["SOURCES"]) - set(H.COLLECTORS))
    assert not missing, f"configured with no collector: {missing}"


def test_every_collector_is_in_both_orderings():
    """A collector absent from SOURCE_ORDER_RUN never runs. One absent from
    SOURCE_ORDER_MASTER runs, writes its CSV, and is then left out of the master table --
    the worse of the two, because it looks like it worked."""
    H = _harvest()
    assert not sorted(set(H.COLLECTORS) - set(H.SOURCE_ORDER_RUN)), \
        "a collector is missing from SOURCE_ORDER_RUN and will never execute"
    assert not sorted(set(H.COLLECTORS) - set(H.SOURCE_ORDER_MASTER)), \
        "a collector is missing from SOURCE_ORDER_MASTER and its rows will be dropped"


def test_orderings_contain_no_phantom_sources():
    H = _harvest()
    assert not sorted(set(H.SOURCE_ORDER_RUN) - set(H.COLLECTORS))
    assert not sorted(set(H.SOURCE_ORDER_MASTER) - set(H.COLLECTORS))


def test_the_four_new_panel_sources_are_fully_wired():
    H = _harvest()
    for s in NEW_SOURCES:
        assert s in H.CONFIG["SOURCES"], f"{s} not in CONFIG"
        assert H.CONFIG["SOURCES"][s] is True, f"{s} is registered but disabled"
        assert s in H.COLLECTORS, f"{s} has no collector"
        assert s in H.SOURCE_ORDER_RUN, f"{s} would never run"
        assert s in H.SOURCE_ORDER_MASTER, f"{s} would be dropped from the master table"
        assert callable(H.COLLECTORS[s])


def test_manual_table_sources_run_before_the_network_heavy_ones():
    """They either find their file in seconds or report it absent, and knowing which files
    are still missing is the most useful output of a short run."""
    H = _harvest()
    order = H.SOURCE_ORDER_RUN
    for s in NEW_SOURCES:
        assert order.index(s) < order.index("chembl")
        assert order.index(s) < order.index("pubchem")


def test_chembl_wins_duplicate_provenance_over_the_panels():
    """SOURCE_ORDER_MASTER decides which source labels a cross-source duplicate. A
    percent-activity or Kd_app readout must not outrank a curated ChEMBL IC50."""
    H = _harvest()
    m = H.SOURCE_ORDER_MASTER
    assert m.index("chembl") == 0
    for s in ("klaeger", "anastassiadis", "karaman"):
        assert m.index("chembl") < m.index(s)


def test_deepdta_base_is_a_list_so_both_branch_names_are_tried():
    """KIBA failed on the 2026-10-26 harvest, and the likeliest cause is GitHub's rename
    of default branches: a raw.githubusercontent URL does NOT redirect across it, it 404s.
    Both must be candidates, and the helper must build URLs without string-concatenating a
    list -- which is how this change broke the first time."""
    H = _harvest()
    assert isinstance(H.URLS["deepdta"], list)
    assert len(H.URLS["deepdta"]) >= 2
    urls = H.deepdta_urls("davis", "ligands_can.txt")
    assert len(urls) == len(H.URLS["deepdta"])
    assert all(u.endswith("/davis/ligands_can.txt") for u in urls)
    assert any("/main/" in u for u in urls) and any("/master/" in u for u in urls)


def test_deepdta_urls_tolerates_a_bare_string_base():
    """Backwards compatibility: an older config or a hand-edit may leave it a string."""
    H = _harvest()
    saved = H.URLS["deepdta"]
    try:
        H.URLS["deepdta"] = "https://example.invalid/data"
        assert H.deepdta_urls("kiba", "proteins.txt") == \
            ["https://example.invalid/data/kiba/proteins.txt"]
    finally:
        H.URLS["deepdta"] = saved


def test_dtc_static_is_a_list_of_candidates():
    H = _harvest()
    assert isinstance(H.URLS["dtc_static"], list)
    assert len(H.URLS["dtc_static"]) >= 2
    assert all(u.startswith("https://") for u in H.URLS["dtc_static"])


def test_name_resolution_is_configurable_and_on_by_default():
    H = _harvest()
    assert H.CONFIG.get("RESOLVE_COMPOUND_NAMES") is True
    assert callable(H.resolve_names_to_smiles)
    assert callable(H.name_structure_lookup)


def test_resolve_names_to_smiles_makes_no_request_for_an_empty_input():
    """The guard that keeps a post-processing pass over a source with no names free."""
    H = _harvest()
    class Boom:
        def get_json(self, *a, **k):
            raise AssertionError("no request should be made for an empty name list")
    class Ctx:
        http = Boom()
        cfg = {"MAX_REMOTE_STRUCTURE_LOOKUPS": 10}
    assert H.resolve_names_to_smiles(Ctx(), []) == {}
    assert H.resolve_names_to_smiles(Ctx(), ["", "   ", None]) == {}


def test_every_panel_collector_documents_its_measurement_semantics():
    """A percent-activity readout is not an IC50 and a Kinobeads Kd_app is not a solution
    Kd. Pooling them silently is the failure this project's censored-value and
    measurability handling exists to prevent, so each collector must say what its number
    means."""
    H = _harvest()
    for s in ("klaeger", "anastassiadis", "karaman", "papyrus"):
        doc = (H.COLLECTORS[s].__doc__ or "")
        assert len(doc) > 200, f"{s} has no substantive docstring"
        assert any(k in doc.lower() for k in ("not ", "do not", "⚠", "must not")), \
            f"{s} does not state what its measurement is NOT"


def test_metz_collector_searches_assay_descriptions_too():
    """It returned 0 rows, which is an EMPTY SEARCH rather than an absent dataset -- the
    paper's Ki matrix exists. The widened search is the fix; this pins it."""
    H = _harvest()
    import inspect
    src = inspect.getsource(H.collect_metz)
    assert "assay_terms" in src, "the assay-description search was removed again"
    assert "manual" in src.lower()


def test_kiba_collector_has_a_manual_fallback_like_davis():
    H = _harvest()
    import inspect
    src = inspect.getsource(H.collect_kiba)
    assert "manual_wide" in src, "KIBA still has no manual escape hatch"


# ------------------------------------------------------------------ the gap audit

def _audit():
    spec = importlib.util.spec_from_file_location(
        "audit_data_gaps", ROOT / "scripts" / "audit_data_gaps.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _audit()


def _frame(n_exact=172, n_cens=40, tnk1="Q13470"):
    """A frozen-table-shaped frame: TNK1 rows plus off-target kinome rows."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_exact):
        rows.append(dict(uniprot_id=tnk1, inchikey_block1=f"K{i // 2:04d}",
                         scaffold=f"S{i // 6:04d}", pact=float(rng.normal(7, 1)),
                         censored=0, standard_type="IC50"))
    for i in range(n_cens):
        rows.append(dict(uniprot_id=tnk1, inchikey_block1=f"C{i:04d}",
                         scaffold=f"T{i // 4:04d}", pact=5.0, censored=1,
                         standard_type="IC50"))
    for i in range(500):
        rows.append(dict(uniprot_id=f"P{i % 37:05d}", inchikey_block1=f"O{i:05d}",
                         scaffold=f"U{i:05d}", pact=float(rng.normal(6, 1)),
                         censored=0, standard_type="Ki"))
    return pd.DataFrame(rows)


def test_audit_separates_exact_from_censored():
    """The distinction most of the prose collapses: a censored row trains the hinge term
    but cannot carry a pActivity label, so it is not part of 'N'."""
    f = A.summarise_frozen(_frame(172, 40), "Q13470")
    assert f["tnk1_rows_exact"] == 172
    assert f["tnk1_rows_censored"] == 40
    assert f["tnk1_rows_total"] == 212


def test_audit_separates_rows_from_compounds_from_scaffolds():
    """Three different denominators. The frozen splits are keyed by compound, scaffold
    groups are what a scaffold split actually partitions, and only rows are what people
    quote."""
    f = A.summarise_frozen(_frame(172, 0), "Q13470")
    assert f["tnk1_rows_exact"] == 172
    assert f["tnk1_compounds_exact"] == 86, "two rows per compound in the fixture"
    assert f["tnk1_scaffolds_exact"] == 29
    assert f["tnk1_compounds_exact"] < f["tnk1_rows_exact"]
    assert f["tnk1_scaffolds_exact"] < f["tnk1_compounds_exact"]


def test_audit_counts_replicates_which_give_a_noise_floor():
    f = A.summarise_frozen(_frame(172, 0), "Q13470")
    assert f["rows_per_compound_mean"] == pytest.approx(2.0)
    assert f["compounds_measured_more_than_once"] == 86


def test_audit_reports_the_kinome_scale_separately():
    f = A.summarise_frozen(_frame(), "Q13470")
    assert f["kinome_rows_total"] == 712
    assert f["kinome_targets"] == 38


def test_audit_handles_a_table_with_no_tnk1_rows():
    """Must report zeros rather than raising -- a wrong accession should be visible, not
    fatal."""
    f = A.summarise_frozen(_frame(), "NOT_A_TARGET")
    assert f["tnk1_rows_exact"] == 0 and f["tnk1_compounds_exact"] == 0
    assert "tnk1_pact_median" not in f


def test_audit_handles_a_table_with_no_censored_column():
    d = _frame(50, 0).drop(columns=["censored"])
    f = A.summarise_frozen(d, "Q13470")
    assert f["tnk1_rows_exact"] == 50


def test_audit_reports_the_missing_source_column_as_a_gap():
    """The frozen table carries no per-row source, so the audit cannot say which harvest
    source each TNK1 row came from -- the single most useful fact for choosing a new
    source. The absence must be reported, not silently skipped."""
    f = A.summarise_frozen(_frame(), "Q13470")
    assert "tnk1_by_source" not in f
    d = _frame()
    d["source"] = "CHEMBL"
    f2 = A.summarise_frozen(d, "Q13470")
    assert f2["tnk1_by_source"] == {"CHEMBL": 212}


def test_harvest_summary_degrades_gracefully_with_no_output_dir(tmp_path):
    h = A.summarise_harvest(tmp_path / "nope")
    assert h["available"] is False and "looked_in" in h


def test_harvest_summary_counts_tnk1_rows_per_source(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir(parents=True)
    pd.DataFrame({
        "uniprot_accession": ["Q13470", "Q13470", "P12931"],
        "target_name_normalized": ["TNK1", "TNK1", "SRC"],
        "gene_symbol": ["TNK1", "TNK1", "SRC"],
        "inchikey_standardized": ["AAAAAAAAAAAAAA-X", "BBBBBBBBBBBBBB-X", "CCCCCCCCCCCCCC-X"],
        "activity_type": ["IC50"] * 3, "relation": ["="] * 3,
    }).to_csv(csv_dir / "CHEMBL.csv", index=False)
    pd.DataFrame({
        "uniprot_accession": ["P12931"], "target_name_normalized": ["SRC"],
        "gene_symbol": ["SRC"], "inchikey_standardized": ["DDDDDDDDDDDDDD-X"],
        "activity_type": ["Kd"], "relation": ["="],
    }).to_csv(csv_dir / "KLAEGER.csv", index=False)

    h = A.summarise_harvest(tmp_path)
    assert h["available"] is True
    assert h["sources"]["chembl"]["total_rows"] == 3
    assert h["sources"]["chembl"]["tnk1_rows"] == 2
    assert h["sources"]["chembl"]["tnk1_compounds"] == 2
    assert h["sources"]["klaeger"]["tnk1_rows"] == 0, \
        "a source with no TNK1 rows must be reported as such, not omitted"


def test_candidate_table_states_what_is_unverified():
    """Every candidate's expectations come from a paper's description, not a measurement,
    and the planning document is built on them. At least one must say so."""
    assert set(A.CANDIDATES) >= set(NEW_SOURCES) | set(REPAIRED)
    for name, c in A.CANDIDATES.items():
        for k in ("paper", "what", "expect_tnk1", "expect_rows", "route", "value"):
            assert c.get(k), f"{name} is missing {k}"
    assert "UNKNOWN" in A.CANDIDATES["klaeger"]["expect_tnk1"], \
        "Klaeger's TNK1 coverage is genuinely unknown and must not be stated as present"
