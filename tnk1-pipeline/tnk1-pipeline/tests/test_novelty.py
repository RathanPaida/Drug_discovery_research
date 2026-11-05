"""Unit test for NoveltyVerifier module."""
from tnk1.novelty import NoveltyVerifier


def test_novelty_verifier_initialization():
    verifier = NoveltyVerifier()
    assert len(verifier.known_inchikeys) > 0
    assert len(verifier.known_inchikeys14) > 0


def test_novelty_verifier_known_molecule():
    verifier = NoveltyVerifier()
    # Aspirin
    res = verifier.verify_candidates(["CC(=O)Oc1ccccc1C(=O)O"], check_pubchem=True)
    assert len(res) == 1
    assert bool(res.iloc[0]["in_pubchem"]) is True
    assert bool(res.iloc[0]["is_globally_novel"]) is False


def test_novelty_verifier_novel_lead():
    verifier = NoveltyVerifier()
    # TNK1 de novo candidate
    res = verifier.verify_candidates(["COc1cc(Nc2ncc(Cl)c(Nc3c(F)cccc3C#N)n2)ccc1NC1CCNCC1"], check_pubchem=True)
    assert len(res) == 1
    assert bool(res.iloc[0]["in_pubchem"]) is False
    assert bool(res.iloc[0]["in_local_store_exact"]) is False
    assert bool(res.iloc[0]["is_globally_novel"]) is True
