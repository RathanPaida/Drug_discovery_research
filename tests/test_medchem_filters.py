"""Tests for the med-chem gate: QED, Ertl SA score, PAINS/Brenk/NIH alerts and the
hinge-binder SMARTS. These feed `hard_gate`, which requires n_alerts == 0.

RDKit-gated, so this module skips in a core-only install. Writing these surfaced defects
#27 (hinge SMARTS recompiled per molecule) and #28 (sys.path grown per sa_score call).
"""
import pytest

rdkit = pytest.importorskip("rdkit", reason="RDKit required for the med-chem filters")

from rdkit import Chem  # noqa: E402

from tnk1.admet import filters as FL  # noqa: E402


IMATINIB_CORE = "Cc1ccc(Nc2nccc(-c3cccnc3)n2)cc1"      # 2-aminopyrimidine hinge binder


def test_medchem_profile_returns_the_full_record():
    p = FL.medchem_profile("CC(=O)Oc1ccccc1C(=O)O")
    assert set(p) >= {"qed", "sa", "mw", "n_alerts", "alerts", "hinge"}
    assert 0.0 <= p["qed"] <= 1.0
    assert 1.0 <= p["sa"] <= 10.0
    assert p["mw"] == pytest.approx(180.16, abs=0.1)


def test_medchem_profile_returns_none_for_invalid_smiles():
    assert FL.medchem_profile("not_a_molecule") is None


def test_hinge_match_detects_an_aminopyrimidine_core():
    mol = Chem.MolFromSmiles(IMATINIB_CORE)
    assert FL.hinge_match(mol) is True
    assert FL.hinge_matches(mol)


def test_hinge_match_rejects_a_non_kinase_scaffold():
    assert FL.hinge_match(Chem.MolFromSmiles("CCCCCCCC")) is False


def test_hinge_patterns_are_compiled_once_and_all_valid():
    """Regression: hinge_match used to call MolFromSmarts on all six patterns for every
    molecule (300k compilations over a 50k pool) and silently treated an unparseable
    SMARTS as a non-match."""
    a = FL._hinge_patterns()
    b = FL._hinge_patterns()
    assert a is b
    assert set(a) == set(FL.HINGE_SMARTS)


def test_sa_scorer_does_not_grow_sys_path_per_call():
    """Regression: sys.path.append ran on every sa_score call, so scoring a 50k pool added
    50k duplicate entries and slowed every later import."""
    import sys
    mol = Chem.MolFromSmiles("CCO")
    FL.sa_score(mol)
    before = len(sys.path)
    for _ in range(25):
        FL.sa_score(mol)
    assert len(sys.path) == before


def test_alerts_flags_a_known_pains_substructure():
    """Catechol is a classic PAINS/Brenk liability and fires several rules at once."""
    p = FL.medchem_profile("Oc1ccccc1O")
    assert p["n_alerts"] >= 1
    assert "catechol" in p["alerts"]


def test_a_clean_drug_like_molecule_has_no_alerts():
    assert FL.medchem_profile("Cn1cnc2c1c(=O)n(C)c(=O)n2C")["n_alerts"] == 0   # caffeine
    assert FL.medchem_profile(IMATINIB_CORE)["n_alerts"] == 0


def test_the_alert_catalog_is_strict_enough_to_reject_aspirin():
    """Documents how aggressive the combined PAINS A/B/C + Brenk + NIH catalog is: aspirin
    fires Brenk's `phenol_ester`. Because `hard_gate` demands n_alerts == 0, any ester-
    containing candidate is eliminated outright -- intended, but worth knowing when a
    generated series disappears from the pool."""
    p = FL.medchem_profile("CC(=O)Oc1ccccc1C(=O)O")
    assert p["n_alerts"] == 1
    assert p["alerts"] == "phenol_ester"
