"""Unit tests (CPU only, no model weights). Run with `python -m pytest tests -q`,
or without pytest: `python tests/test_core.py`."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber import decide, io, lexicon, metrics  # noqa: E402
from ber import normalize as N  # noqa: E402
from ber.config import load_config  # noqa: E402


def _cfg(*ov):
    return load_config(None, list(ov))


# -- config ---------------------------------------------------------------
def test_config_overrides_and_types():
    cfg = _cfg("retriever.epochs=3", "retriever.lr=2e-5", "decide.one_to_one=false")
    assert cfg.retriever.epochs == 3 and abs(cfg.retriever.lr - 2e-5) < 1e-12
    assert cfg.decide.one_to_one is False
    assert Path(cfg.paths.work_dir).is_absolute()
    try:
        _cfg("retriever.epochz=3")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown key must raise")


# -- normalisation --------------------------------------------------------
def _lex(names, addrs=None):
    addrs = addrs or [""] * len(names)
    df = pl.DataFrame({"name": names, "addr": addrs}).with_columns(
        N.base_text("name").alias("t_name"), N.base_text("addr").alias("t_addr"))
    df = df.with_columns(N.split_tokens(N.name_clean(pl.col("t_name"))).alias("tok0"),
                         N.split_tokens(N.addr_clean(pl.col("t_addr"))).alias("atok0"), *N.name_flags())
    df = df.with_columns(N.map_tokens(pl.col("tok0"), lexicon.LEGAL_CANON).alias("n_tok"),
                         N.map_tokens(pl.col("atok0"), lexicon.ADDR_CANON).alias("a_tok"))
    return df.with_columns(*N.name_views())


def test_accents_stripped_only_for_latin():
    df = _lex(["Chordia &-P\u00e1rtners Ltd", "\u0930\u093f\u092f\u0932 \u092e\u0949\u0921\u0930\u094d\u0928 \u092b\u0942\u0921"])
    assert df["t_name"][0] == "chordia &-partners ltd"
    assert df["t_name"][1] == "\u0930\u093f\u092f\u0932 \u092e\u0949\u0921\u0930\u094d\u0928 \u092b\u0942\u0921"          # Devanagari vowel signs survive


def test_name_views():
    df = _lex(["AP AP Hospitality Incorporated", "Nexxylo D.B.A. Nion", "maurewilliamscolombier.com",
               "Chordia + Pagnters - 7306204978"])
    assert df["n_core"][0].to_list() == ["ap", "hospitality"] and df["n_legal"][0].to_list() == ["inc"]
    assert df["n_alt"][1].to_list() == ["nion"] and df["f_dba"][1]
    assert df["n_core"][2].to_list() == ["maurewilliamscolombier"] and df["f_domain"][2]
    assert df["f_phone"][3] and "7306204978" not in df["n_tok"][3].to_list()


def test_address_tokens():
    df = _lex(["x"], ["H.NO 1038 SECTOR 9, 8219 17RD STREET, Door No 183"])
    toks = df["a_tok"][0].to_list()
    assert toks.count("hno") == 2 and "sec" in toks and "st" in toks and "17" in toks


def test_french_addons_stripped_for_france_only():
    from ber.prepare import base_columns
    names = ["Maison Atc EURL & Fils", "\u00c9tablissements Sapeurs SASU & Associ\u00e9s",
             "Noisette SAS Et Fils", "Dupont Holding Participations", "Martin & Fils"]
    df = base_columns(pl.DataFrame({"name": names, "addr": [""] * 5,
                                    "country": ["France"] * 4 + ["US"]}))
    assert df["t_name"].to_list() == ["maison atc eurl", "etablissements sapeurs sasu", "noisette sas",
                                      # distractor markers stay: they separate real businesses
                                      "dupont holding participations",
                                      # other countries are untouched
                                      "martin & fils"]
    assert df["tok0"][0].to_list() == ["maison", "atc", "eurl"]


# -- metric ---------------------------------------------------------------
def test_fbeta_readme_example():
    pred = pl.DataFrame({"q": [1, 1, 1], "r": [47, 193, 812]})
    truth = pl.DataFrame({"q": [1, 1], "r": [47, 812]})
    f = metrics.macro_fbeta(pred, truth, pl.Series([1]))
    assert abs(f - 0.7142857) < 1e-5


def test_fbeta_singletons_and_empty():
    queries = pl.Series([1, 2, 3])
    truth = pl.DataFrame({"q": [3], "r": [9]})
    empty = pl.DataFrame({"q": pl.Series([], dtype=pl.Int64), "r": pl.Series([], dtype=pl.Int64)})
    per = metrics.per_query_fbeta(empty, truth, queries).sort("q")
    assert per["f"].to_list() == [1.0, 1.0, 0.0]            # singletons right, matched S1 missed
    wrong = pl.DataFrame({"q": [1], "r": [5]})
    per = metrics.per_query_fbeta(wrong, truth, queries).sort("q")
    assert per["f"].to_list() == [0.0, 1.0, 0.0]


def test_candidate_report_oracle():
    truth = pl.DataFrame({"q": [1, 1, 2], "r": [10, 11, 20]})
    cands = pl.DataFrame({"q": [1, 1, 2, 3], "r": [10, 99, 20, 30]})
    rep = metrics.candidate_report(cands, truth, pl.Series([1, 2, 3]))
    assert rep["pair_recall"] == round(2 / 3, 5)
    # q1: found 1 of 2 -> F0.5 = 1.25*0.5/(0.25+0.5); q2: 1; q3 (singleton): 1
    assert abs(rep["oracle_macro_f"] - round((1.25 * 0.5 / 0.75 + 2) / 3, 5)) < 1e-4


# -- decisions ------------------------------------------------------------
def test_one_to_one_keeps_best_claimant():
    pairs = pl.DataFrame({"q": [1, 2, 2], "r": [7, 7, 8], "p": [0.9, 0.6, 0.8]})
    out = decide.one_to_one(pairs).sort("q", "r")
    assert out.select("q", "r").rows() == [(1, 7), (2, 8)]


def test_expected_f_prefers_empty_for_weak_and_keeps_strong():
    pairs = pl.DataFrame({"q": [1, 1, 2, 3, 3], "r": [10, 11, 20, 30, 31], "p": [0.97, 0.05, 0.08, 0.9, 0.85]})
    for method in ("plugin", "expected_f"):
        sel = decide.select(pairs, {"method": method, "a": 1.0, "b": 0.0, "one_to_one": True},
                            0.5, 512, 0).sort("q", "r")
        assert sel.rows() == [(1, 10), (3, 30), (3, 31)], (method, sel.rows())


def test_threshold_select_and_calibration_identity():
    pairs = pl.DataFrame({"q": [1, 1], "r": [1, 2], "p": [0.7, 0.4]})
    sel = decide.select(pairs, {"method": "threshold", "a": 1.0, "b": 0.0, "t": 0.5}, 0.5, 16, 0)
    assert sel.rows() == [(1, 1)]
    v = pl.DataFrame({"p": [0.2, 0.5, 0.9]}).select(decide.calibrate(pl.col("p"), 1.0, 0.0))
    assert np.allclose(v.to_series().to_numpy(), [0.2, 0.5, 0.9], atol=1e-5)


# -- retrieval pieces -----------------------------------------------------
def test_scipy_topn_matches_bruteforce():
    import scipy.sparse as sp
    from ber.sparse import _scipy_topn
    rng = np.random.default_rng(0)
    A = sp.random(30, 50, density=0.3, format="csr", random_state=1, dtype=np.float32)
    B = sp.random(50, 80, density=0.3, format="csr", random_state=2, dtype=np.float32)
    C = _scipy_topn(A, B, 5, 0.0, 1).tocsr()
    D = (A @ B).toarray()
    for i in range(30):
        got = sorted(C[i].data.tolist(), reverse=True)
        want = sorted([v for v in D[i] if v > 0], reverse=True)[:5]
        assert np.allclose(got, want, atol=1e-6)
    _ = rng


def test_top_terms_per_row_keeps_largest():
    import scipy.sparse as sp
    from ber.sparse import top_terms_per_row
    M = sp.csr_matrix(np.array([[0.1, 0.5, 0.0, 0.3], [0.9, 0.0, 0.0, 0.0]], dtype=np.float32))
    K = top_terms_per_row(M, 2).toarray()
    assert np.allclose(K, [[0.0, 0.5, 0.0, 0.3], [0.9, 0.0, 0.0, 0.0]])
    assert np.allclose(M.toarray()[0], [0.1, 0.5, 0.0, 0.3])      # input untouched


def test_dense_topk_exact():
    import torch
    from ber.dense import topk_search
    g = torch.Generator().manual_seed(0)
    Q = torch.nn.functional.normalize(torch.randn(37, 16, generator=g), dim=-1).half()
    R = torch.nn.functional.normalize(torch.randn(101, 16, generator=g), dim=-1).half()
    qp, rp, s = topk_search(Q, R, 5, 8, torch.device("cpu"))
    full = (Q.float() @ R.float().T).numpy()
    for i in range(37):
        assert set(rp[qp == i]) == set(np.argsort(-full[i])[:5])


def test_dense_retrieve_ensemble_with_fake_models():
    """Exercises dense.retrieve + nn.encode end to end on CPU with a toy
    tokenizer/model: fold-ensemble top-k and cosines for sparse-only pairs."""
    import torch
    from ber.dense import retrieve

    class Tok:
        def __call__(self, texts, **kw):
            return {"input_ids": torch.tensor([[sum(map(ord, t)) % 97 + 1, len(t) % 13 + 1] for t in texts])}

    class Model(torch.nn.Module):
        def __init__(self, seed):
            super().__init__()
            torch.manual_seed(seed)
            self.config = type("C", (), {"hidden_size": 8})()
            self.emb = torch.nn.Embedding(100, 8)

        def forward(self, input_ids):
            return type("O", (), {"last_hidden_state": self.emb(input_ids)})()

    cfg = _cfg("dense.top_k=2", "dense.encode_batch=3", "dense.query_chunk=2")
    texts = ["acme inc | 1 main st", "acme | 1 main street", "zeta llc | 9 elm rd", "zeta | elm road", "omega co | x"]
    rec = pl.DataFrame({"idx": list(range(5))})
    q_idx, r_idx = np.array([0, 2]), np.array([1, 3, 4])
    sparse_pairs = pl.DataFrame({"q": [0, 2], "r": [4, 4]}, schema={"q": pl.UInt32, "r": pl.UInt32})
    models = [Model(1), Model(2)]
    out = retrieve(models, Tok(), texts, rec, q_idx, r_idx, sparse_pairs, cfg, torch.device("cpu"), "t")
    top = out.filter(pl.col("dense_rank").is_not_null())
    assert top.group_by("q").len()["len"].to_list() == [2, 2]
    # every returned similarity equals the mean cosine of the two toy models
    with torch.no_grad():
        def emb(m, t):
            return torch.nn.functional.normalize(m(**Tok()([t])).last_hidden_state[:, 0], dim=-1)[0]
        for q, r, s in out.select("q", "r", "dense_sim").iter_rows():
            want = np.mean([float(emb(m, texts[q]) @ emb(m, texts[r])) for m in models])
            assert abs(s - want) < 2e-2, (q, r, s, want)
    assert out.height == out.select("q", "r").unique().height


def test_dense_pair_sims_chunked_and_up_to_date(tmp_path):
    """pair_sims bounds each chunk by bytes (the fixed row count once ran out of GPU
    memory on the 2-model test ensemble); _up_to_date lets a re-run keep a finished split."""
    import os
    import torch
    from ber.dense import _up_to_date, pair_sims
    g = torch.Generator().manual_seed(0)
    Q = torch.nn.functional.normalize(torch.randn(50, 32, generator=g), dim=-1).half()
    R = torch.nn.functional.normalize(torch.randn(70, 32, generator=g), dim=-1).half()
    rng = np.random.default_rng(0)
    qp, rp = rng.integers(0, 50, 999), rng.integers(0, 70, 999)
    want = (Q[qp].float() * R[rp].float()).sum(1).numpy()
    got = pair_sims(Q, R, qp, rp, torch.device("cpu"), chunk_bytes=4 * 32 * 7)   # 7 rows per chunk
    assert np.allclose(got, want, atol=1e-6)

    src, out, mdir = tmp_path / "in.parquet", tmp_path / "out.parquet", tmp_path / "model"
    mdir.mkdir()
    src.write_text("x")
    (mdir / "w.bin").write_text("w")
    assert not _up_to_date(out, [src, mdir])                 # no output yet
    out.write_text("y")
    for p in (src, mdir / "w.bin"):
        os.utime(p, (1_000, 1_000))
    assert _up_to_date(out, [src, mdir])                     # inputs (file, folder) older -> keep
    os.utime(mdir / "w.bin", None)                           # model rewritten after the output
    os.utime(out, (2_000, 2_000))
    assert not _up_to_date(out, [src, mdir])                 # -> redo
    assert not _up_to_date(out, [tmp_path / "missing"])      # unknown input -> redo


def test_features_part_edges_keep_s1_together():
    """features splits pairs (sorted by S1) into ~chunk_rows parts at S1 boundaries.
    A bug once put every S1 in part 0: one 185M-pair part and an out-of-memory kill."""
    from ber.features import part_edges
    q = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    e = part_edges(q, 4)
    assert e.tolist() == [0, 5, 9, 10]
    parts = [set(q[a:b]) for a, b in zip(e[:-1], e[1:])]
    assert all(not (p & o) for i, p in enumerate(parts) for o in parts[i + 1:])   # no S1 split across parts
    assert part_edges(q, 100).tolist() == [0, 10]                                 # small data: one part
    assert part_edges(np.array([], dtype=np.uint32), 4).tolist() == [0, 0]
    sizes = np.diff(part_edges(np.repeat(np.arange(1000, dtype=np.uint32), 50), 5000))
    assert len(sizes) == 10 and sizes.max() <= 5000                               # many S1 -> many parts


def test_contrastive_loss_masks_same_owner():
    import torch
    from ber.nn import contrastive_loss
    q = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    d = torch.cat([q.clone(), q[:1].clone()])               # doc 3 is another record of S1 #0
    owners_q = torch.tensor([100, 200, 300])
    owners_d = torch.tensor([100, 200, 300, 100])
    masked = contrastive_loss(q, d, owners_q, owners_d, 0.05)
    unmasked = contrastive_loss(q, d, owners_q, torch.tensor([100, 200, 300, -1]), 0.05)
    assert torch.isfinite(masked) and masked < unmasked      # the duplicate is not punished


def test_epoch_batches_single_country_and_complete():
    from ber.retriever import epoch_batches
    pos = pl.DataFrame({"q": [1, 1, 2, 3, 4, 5, 6, 7], "r": list(range(8)),
                        "country": ["US"] * 5 + ["India"] * 3})
    bs = epoch_batches(pos, 2, 0)
    assert all(b["country"].n_unique() == 1 for b in bs)
    assert sorted(pl.concat(bs)["r"].to_list()) == list(range(8))


def test_write_id_lists_format():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.tsv"
        pairs = pl.DataFrame({"s1_id": ["S1-1", "S1-1", "S1-1"], "rec_id": ["S2-5", "S3-7", "S2-5"]})
        io.write_id_lists(path, pl.Series(["S1-1", "S1-2"]), pairs, io.MATCH_HEADER)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines == ["source1_entity_id\tmatched_entity_ids", "S1-1\tS2-5,S3-7", "S1-2\t"]


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
