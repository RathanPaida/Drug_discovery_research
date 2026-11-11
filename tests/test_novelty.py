"""Unit tests for the NoveltyVerifier module.

Two kinds of test live here:
  * offline/hermetic tests (default) -- no network, no 788k store; they pin the FAIL-CLOSED
    contract: anything we could not actually verify must come back INCONCLUSIVE, never
    "novel". These are the ones that protect the headline novelty claim.
  * `@pytest.mark.network` tests -- real PubChem/UniChem calls, deselected by default
    (`-m "not network"` in pytest.ini). Run them with `pytest -m network`.

The whole module skips cleanly if RDKit is absent, so a core-only environment can still
collect and run the rest of the suite (a hard top-level `from rdkit import Chem` previously
aborted collection of all 21 tests).
"""
import numpy as np
import pytest

pytest.importorskip("rdkit", reason="RDKit is required for novelty verification")

from tnk1.novelty import NoveltyVerifier  # noqa: E402

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CANDIDATE = "COc1cc(Nc2ncc(Cl)c(Nc3c(F)cccc3C#N)n2)ccc1NC1CCNCC1"


@pytest.fixture
def bare(tmp_path):
    """A verifier with a deliberately empty feature store."""
    return NoveltyVerifier(feature_store_dir=tmp_path)


# --------------------------------------------------------------------- offline contract

def test_missing_stores_are_reported_not_swallowed(bare):
    assert bare.local_store_loaded is False
    assert bare.fingerprints_loaded is False
    assert set(bare.store_status) == {"inchikeys", "tnk1_curated", "fingerprints"}
    assert all(v == "absent" for v in bare.store_status.values())
    with pytest.raises(RuntimeError, match="not usable"):
        bare.require_stores()


def test_no_novelty_claim_without_a_reference_store(bare):
    """The core regression: an unavailable reference set must NOT yield 'novel'."""
    df = bare.verify_candidates([CANDIDATE], check_pubchem=False, check_unichem=False)
    row = df.iloc[0]
    assert bool(row["is_globally_novel"]) is False
    assert bool(row["novelty_fully_checked"]) is False
    assert row["novelty_reason"].startswith("INCONCLUSIVE")
    assert "local_store_unavailable" in row["novelty_reason"]


def test_unchecked_registries_are_inconclusive_not_novel(bare):
    df = bare.verify_candidates([CANDIDATE], check_pubchem=False, check_unichem=False)
    row = df.iloc[0]
    assert row["in_pubchem"] is None
    assert row["in_unichem"] is None
    assert "pubchem_not_verified" in row["novelty_reason"]
    assert "unichem_not_verified" in row["novelty_reason"]


def test_similarity_is_nan_not_zero_when_store_absent(bare):
    """0.0 is maximal novelty; an absent fingerprint store must not report it."""
    sims = bare.compute_max_tanimoto([ASPIRIN, CANDIDATE])
    assert sims.shape == (2,)
    assert np.isnan(sims).all()


def test_unparseable_smiles_does_not_raise_and_is_not_novel(bare):
    sims = bare.compute_max_tanimoto(["not_a_molecule", ""])
    assert np.isnan(sims).all()
    df = bare.verify_candidates(["not_a_molecule"], check_pubchem=False, check_unichem=False)
    row = df.iloc[0]
    assert row["inchikey"] is None
    assert bool(row["is_globally_novel"]) is False
    assert "smiles_unparseable" in row["novelty_reason"]


def test_known_molecule_in_local_store_is_rejected(bare):
    """A key present in the local store must be reported KNOWN even offline."""
    ik, ik14 = NoveltyVerifier.smiles_to_inchikey(ASPIRIN)
    bare.known_inchikeys.add(ik)
    bare.known_inchikeys14.add(ik14)
    bare.store_status["inchikeys"] = "ok"
    assert bare.check_local(ik) == {"in_local_exact": True, "in_local_skeleton": True}
    df = bare.verify_candidates([ASPIRIN], check_pubchem=False, check_unichem=False)
    assert bool(df.iloc[0]["is_globally_novel"]) is False
    assert "local_exact" in df.iloc[0]["novelty_reason"]


def test_empty_batches_are_tri_state_not_false():
    assert NoveltyVerifier.check_pubchem_batch([]) == {}
    assert NoveltyVerifier.check_unichem_batch([""]) == {"": (None, [])}


def test_smiles_to_inchikey_roundtrip():
    ik, ik14 = NoveltyVerifier.smiles_to_inchikey(ASPIRIN)
    assert ik is not None and len(ik) == 27
    assert ik14 == ik[:14]
    assert NoveltyVerifier.smiles_to_inchikey("@@bad@@") == (None, None)


# ------------------------------------------------------------------------ network tests

@pytest.mark.network
def test_known_molecule_is_found_in_registries(bare):
    ik, _ = NoveltyVerifier.smiles_to_inchikey(ASPIRIN)
    assert NoveltyVerifier.check_pubchem_batch([ik])[ik] is True
    found, sources = NoveltyVerifier.check_unichem_single(ik)
    assert found is True and sources


@pytest.mark.network
def test_candidate_absence_is_verified_not_assumed(bare):
    """Asserts a *verified* absence (False), which a network failure can no longer fake."""
    ik, _ = NoveltyVerifier.smiles_to_inchikey(CANDIDATE)
    pubchem = NoveltyVerifier.check_pubchem_batch([ik])[ik]
    if pubchem is None:
        pytest.skip("PubChem unreachable; absence cannot be verified")
    assert pubchem is False
