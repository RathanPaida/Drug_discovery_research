"""Tests for `rl_min`, the in-process DAP policy-gradient generator.

These cover the machinery that distinguishes v2 Stage 7 from v1's: v1 sampled a fixed pool
and filtered it, so no generator parameter ever moved. The decisive test here is
`test_one_dap_step_changes_the_agent_but_not_the_prior` -- if that fails, the stage has
regressed to selection.

torch-gated, so the module skips in a core-only install.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is required for the rl_min tests")
pytest.importorskip("rdkit", reason="rdkit is required for the scorer tests")

from rl_min.dap import DAPConfig, ReplayMemory, ScaffoldFilter, train_dap  # noqa: E402
from rl_min.model import SmilesRNN, encode_batch  # noqa: E402
from rl_min.scorer import StandaloneScorer  # noqa: E402
from rl_min.train_prior import EncodedCorpus  # noqa: E402
from rl_min.vocab import Vocabulary, is_lossless, pad_batch, tokenize  # noqa: E402

SMILES = [
    "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
    "Clc1ccccc1", "c1cc[nH]c1", "C[C@@H](N)C(=O)O",
    "Nc1ncnc2[nH]cnc12", "COc1cc2ncnc(N)c2cc1OC",
]


# ─────────────────────────────────────────────────────────────── tokenisation

@pytest.mark.parametrize("smi", SMILES)
def test_tokenizer_round_trips(smi):
    assert "".join(tokenize(smi)) == smi
    assert is_lossless(smi)


def test_multi_character_tokens_are_not_split():
    """Character-level splitting would make 'Cl' two tokens and '[nH]' four, forcing the
    model to relearn that those characters are inseparable."""
    assert "Cl" in tokenize("Clc1ccccc1")
    assert "[nH]" in tokenize("c1cc[nH]c1")
    assert "Br" in tokenize("Brc1ccccc1")
    assert "%10" in tokenize("C%10CCCCC%10")


# ──────────────────────────────────────────────────────────────── vocabulary

def test_encode_decode_round_trip():
    v = Vocabulary.from_smiles(SMILES)
    for s in SMILES:
        assert v.decode(v.encode(s)) == s


def test_encode_raises_on_an_unknown_token():
    """A silent UNK would let the agent emit something the prior's vocabulary cannot
    represent and have it decode to a different molecule."""
    v = Vocabulary.from_smiles(["CCO"])
    with pytest.raises(KeyError):
        v.encode("c1ccccc1")


def test_vocab_save_load_preserves_indices(tmp_path):
    v = Vocabulary.from_smiles(SMILES)
    v.save(tmp_path / "vocab.txt")
    w = Vocabulary.load(tmp_path / "vocab.txt")
    assert w.itos == v.itos and w.pad == v.pad and w.bos == v.bos and w.eos == v.eos


# ───────────────────────────────────────────────────────────── encoded corpus

def test_encoded_corpus_matches_plain_encoding():
    v = Vocabulary.from_smiles(SMILES)
    enc = EncodedCorpus(v, SMILES)
    assert len(enc) == len(SMILES)
    for i, s in enumerate(SMILES):
        assert list(enc[i]) == v.encode(s)


def test_encoded_corpus_is_smaller_than_list_of_lists():
    """Measured on the full 787,283-molecule store: 445 MB -> 201 MB resident."""
    v = Vocabulary.from_smiles(SMILES)
    big = SMILES * 500
    enc = EncodedCorpus(v, big)
    naive = sum(len(v.encode(s)) for s in big) * 8      # pointers alone, ignoring overhead
    assert enc.nbytes() < naive


def test_pad_batch_accepts_numpy_slices():
    v = Vocabulary.from_smiles(SMILES)
    enc = EncodedCorpus(v, SMILES)
    b = pad_batch([enc[i] for i in range(4)], v.pad)
    assert b.shape[0] == 4 and b.dtype == torch.long


# ────────────────────────────────────────────────────────────────────── model

def _tiny(v):
    return SmilesRNN(len(v), emb=16, hidden=32, layers=1, dropout=0.0, pad_id=v.pad)


def test_nll_is_summed_over_the_sequence_not_averaged():
    """DAP adds `sigma * score` to a whole-sequence log probability. Averaging would make
    the reward term length-dependent and bias the agent toward long SMILES."""
    v = Vocabulary.from_smiles(SMILES)
    m = _tiny(v).eval()
    short = encode_batch(v, ["CCO"])
    long_ = encode_batch(v, ["CCCCCCCCCCCCCCCC"])
    with torch.no_grad():
        assert float(m.nll(long_)[0]) > float(m.nll(short)[0])


def test_padding_does_not_contribute_to_nll():
    v = Vocabulary.from_smiles(SMILES)
    m = _tiny(v).eval()
    with torch.no_grad():
        alone = float(m.nll(encode_batch(v, ["CCO"]))[0])
        batched = float(m.nll(encode_batch(v, ["CCO", "CC(=O)Oc1ccccc1C(=O)O"]))[0])
    assert alone == pytest.approx(batched, abs=1e-4)


def test_sample_with_returns_decodable_sequences():
    v = Vocabulary.from_smiles(SMILES)
    m = _tiny(v)
    seqs, nll = m.sample_with(v, 8, max_len=32, device="cpu")
    assert seqs.shape[0] == 8 and nll.shape == (8,)
    assert (nll >= 0).all(), "NLL must be non-negative"
    for row in seqs:
        v.decode(row)          # must not raise


def test_save_load_round_trip(tmp_path):
    v = Vocabulary.from_smiles(SMILES)
    m = _tiny(v)
    m.save(tmp_path / "m.pt")
    m2 = SmilesRNN.load(tmp_path / "m.pt")
    x = encode_batch(v, ["CCO"])
    m.eval(); m2.eval()
    with torch.no_grad():
        assert float(m.nll(x)[0]) == pytest.approx(float(m2.nll(x)[0]), abs=1e-5)


# ──────────────────────────────────────────────────────────── DAP mechanics

def test_scaffold_filter_zeroes_an_over_represented_scaffold():
    f = ScaffoldFilter(bucket=2, min_score=0.5)
    out = f(["A", "A", "A", "A"], [0.9, 0.9, 0.9, 0.9])
    assert out[0] == 0.9 and out[1] == 0.9
    assert out[2] == 0.0 and out[3] == 0.0


def test_scaffold_filter_ignores_low_scorers_when_counting():
    f = ScaffoldFilter(bucket=1, min_score=0.5)
    assert f(["A"], [0.1])[0] == 0.1        # below min_score, not counted
    assert f(["A"], [0.9])[0] == 0.9        # first real hit still allowed


def test_replay_memory_keeps_the_best_and_dedupes():
    r = ReplayMemory(size=3)
    r.add(["a", "b", "c", "d"], [0.1, 0.9, 0.5, 0.7])
    r.add(["b"], [0.9])                      # duplicate must not be stored twice
    assert len(r.smiles) == 3
    assert set(r.smiles) == {"b", "d", "c"}
    assert r.smiles.count("b") == 1


def test_one_dap_step_changes_the_agent_but_not_the_prior():
    """The decisive property. v1's Stage 7 scored a fixed pool and updated nothing; if this
    fails, the stage has regressed from generation back to selection."""
    import copy
    v = Vocabulary.from_smiles(SMILES)
    prior = _tiny(v)
    agent = copy.deepcopy(prior)
    prior_before = [p.detach().clone() for p in prior.parameters()]
    agent_before = [p.detach().clone() for p in agent.parameters()]

    scorer = lambda smis: {"score": np.random.RandomState(0).rand(len(smis)),
                           "valid": np.ones(len(smis), bool),
                           "scaffold": [None] * len(smis)}
    cfg = DAPConfig(steps=2, batch_size=4, sigma=60.0, lr=1e-2, max_len=24,
                    replay_size=4, replay_batch=0)
    hist, _ = train_dap(prior, agent, v, scorer, cfg, device="cpu")

    assert len(hist) == 2 and "score_mean" in hist[0]
    assert all(torch.equal(a, b) for a, b in zip(prior.parameters(), prior_before)), \
        "the prior is the KL anchor and must stay frozen"
    assert any(not torch.equal(a, b) for a, b in zip(agent.parameters(), agent_before)), \
        "the agent did not learn -- no parameter moved"


def test_dap_runs_with_replay_enabled():
    """Replay splices stored SMILES into the batch; their encodings must pad to the same
    width as the sampled sequences or the concatenation fails."""
    import copy
    v = Vocabulary.from_smiles(SMILES)
    prior = _tiny(v); agent = copy.deepcopy(prior)
    scorer = lambda smis: {"score": np.full(len(smis), 0.8),
                           "valid": np.ones(len(smis), bool),
                           "scaffold": [None] * len(smis)}
    cfg = DAPConfig(steps=3, batch_size=4, sigma=60.0, lr=1e-3, max_len=24,
                    replay_size=8, replay_batch=3)
    hist, replay = train_dap(prior, agent, v, scorer, cfg, device="cpu")
    assert len(hist) == 3


# ──────────────────────────────────────────────────────────────────── scorer

def test_standalone_scorer_scores_zero_for_invalid_smiles():
    sc = StandaloneScorer()
    out = sc(["not_a_molecule", "", "c1ccccc1"])
    assert out["score"][0] == 0.0 and out["valid"][0] == np.False_
    assert out["score"][1] == 0.0
    assert out["valid"][2] == np.True_


def test_standalone_scorer_rewards_the_motif():
    sc = StandaloneScorer(motif_smarts="c1ncnc(N)c1")
    out = sc(["Nc1ncnc2[nH]cnc12", "CCCCCCCC"])
    assert out["motif"][0] == 1.0
    assert out["score"][0] > out["score"][1]
