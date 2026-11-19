"""Step `sparse`: lexical TF-IDF candidate retrieval (CPU), per split and country.

Passes - each keeps the top-k S2/S3 records per S1:
    name_word   word TF-IDF on the core name (legal forms / stopwords removed)
    addr_word   word TF-IDF on normalised address tokens  (finds renamed records)
    full_word   w * name + (1 - w) * address in one product
    name_char   char n-gram TF-IDF on the core name (typos, domains, concatenations)

Retrieval ignores very frequent tokens (`max_df_*`) so the sparse products stay
small; the exact cosines of *every* candidate pair (dense ones too) are
computed later by `features` from the matrices saved here.

Outputs:
    work/sparse/{split}_sparse.parquet        q, r, rk_<pass> (rank within the S1, null = not retrieved)
    work/sparse/mats/{split}__{country}__*    TF-IDF matrices + row -> record idx
    work/sparse/{split}_report.json           recall per pass (train only)
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from .metrics import candidate_report
from .util import Work, log, n_threads, save_json, timed

MATRICES = ("name_word", "addr_word", "name_char")
PASSES = ("name_word", "addr_word", "full_word", "name_char")


def _safe(country: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", country) or "NA"


def mat_path(W: Work, split: str, country: str, kind: str) -> Path:
    ext = "npy" if kind == "rows" else "npz"
    return W.p("sparse", "mats", f"{split}__{_safe(country)}__{kind}.{ext}")


# -- TF-IDF matrices ------------------------------------------------------
def build_matrices(names: list[str], addrs: list[str], char_ngram: int) -> dict[str, sp.csr_matrix]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    def fit(docs, **kw):
        if not any(docs):
            return sp.csr_matrix((len(docs), 1), dtype=np.float32)
        v = TfidfVectorizer(lowercase=False, sublinear_tf=True, dtype=np.float32, **kw)
        return v.fit_transform(docs).tocsr()

    word = dict(tokenizer=str.split, token_pattern=None)
    return {
        "name_word": fit(names, **word),
        "addr_word": fit(addrs, **word),
        "name_char": fit(names, analyzer="char_wb", ngram_range=(char_ngram, char_ngram)),
    }


def full_matrix(N: sp.csr_matrix, A: sp.csr_matrix, w: float) -> sp.csr_matrix:
    """Rows of N and A are unit-norm, so <full_i, full_j> = w*cos_N + (1-w)*cos_A."""
    return sp.hstack([N * np.float32(np.sqrt(w)), A * np.float32(np.sqrt(1 - w))], format="csr")


def keep_columns(M: sp.csr_matrix, rec_rows: np.ndarray, max_df: int) -> np.ndarray:
    dfreq = np.bincount(M[rec_rows].indices, minlength=M.shape[1])
    return np.flatnonzero((dfreq > 0) & (dfreq <= max_df))


def top_terms_per_row(M: sp.csr_matrix, m: int) -> sp.csr_matrix:
    """Keep each row's m highest-weighted terms (the rarest n-grams under TF-IDF):
    bounds the fan-out of a query to m * max_df records."""
    if m <= 0:
        return M
    M = M.tocsr(copy=True)
    counts = np.diff(M.indptr)
    if counts.max(initial=0) <= m:
        return M
    rows = np.repeat(np.arange(M.shape[0]), counts)
    order = np.lexsort((-M.data, rows))
    rank = np.arange(order.size) - M.indptr[rows[order]]
    drop = order[rank >= m]
    M.data[drop] = 0
    M.eliminate_zeros()
    return M


# -- top-k sparse products ------------------------------------------------
def _scipy_topn(A: sp.csr_matrix, B: sp.csr_matrix, k: int, thr: float, nt: int) -> sp.coo_matrix:
    C = (A @ B).tocsr()
    if thr > 0:
        C.data[C.data <= thr] = 0
        C.eliminate_zeros()
    counts = np.diff(C.indptr)
    rows = np.repeat(np.arange(C.shape[0]), counts)
    order = np.lexsort((-C.data, rows))
    rank = np.arange(order.size) - C.indptr[rows[order]]
    sel = order[rank < k]
    return sp.coo_matrix((C.data[sel], (rows[sel], C.indices[sel])), shape=C.shape)


def topn_backend():
    try:
        from sparse_dot_topn import sp_matmul_topn
        params = inspect.signature(sp_matmul_topn).parameters

        def fn(A, B, k, thr, nt):
            kw = {"top_n": k}
            if "threshold" in params and thr > 0:
                kw["threshold"] = thr
            if "n_threads" in params and nt > 1:
                kw["n_threads"] = nt
            return sp_matmul_topn(A, B, **kw).tocoo()
        return fn, "sparse_dot_topn"
    except ImportError:
        pass
    try:
        from sparse_dot_topn import awesome_cossim_topn

        def fn(A, B, k, thr, nt):
            return awesome_cossim_topn(A, B, k, lower_bound=thr, use_threads=nt > 1, n_jobs=max(nt, 1)).tocoo()
        return fn, "sparse_dot_topn (legacy API)"
    except ImportError:
        return _scipy_topn, "scipy (single-threaded fallback - install sparse_dot_topn on the cluster)"


def topk(Q: sp.csr_matrix, R: sp.csr_matrix, k: int, thr: float, nt: int, chunk: int):
    fn, _ = topn_backend()
    RT = R.T.tocsr()
    qs, rs, ss = [], [], []
    for s in range(0, Q.shape[0], chunk):
        C = fn(Q[s:s + chunk], RT, k, thr, nt)
        qs.append(C.row.astype(np.int64) + s)
        rs.append(C.col.astype(np.int64))
        ss.append(C.data.astype(np.float32))
    if not qs:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    return np.concatenate(qs), np.concatenate(rs), np.concatenate(ss)


# -- step -----------------------------------------------------------------
def retrieve_country(cfg, W: Work, split: str, country: str, df: pl.DataFrame) -> pl.DataFrame:
    sc = cfg.sparse
    nt = n_threads(sc.n_threads)
    rows = df["idx"].to_numpy()
    is_q = (df["src"] == 1).to_numpy()
    q_rows, r_rows = np.flatnonzero(is_q), np.flatnonzero(~is_q)
    if q_rows.size == 0 or r_rows.size == 0:
        return pl.DataFrame()

    with timed(f"sparse {split}/{country}: TF-IDF on {len(rows):,} docs"):
        M = build_matrices(df["n_core_str"].to_list(),
                           df["a_tok"].list.join(" ").to_list(), sc.char_ngram)
    np.save(mat_path(W, split, country, "rows"), rows)
    for kind, m in M.items():
        sp.save_npz(mat_path(W, split, country, kind), m, compressed=False)

    out = []
    for pas in sc.passes:
        if pas == "full_word":
            kn = keep_columns(M["name_word"], r_rows, sc.max_df_word)
            ka = keep_columns(M["addr_word"], r_rows, sc.max_df_word)
            X = full_matrix(M["name_word"][:, kn], M["addr_word"][:, ka], sc.full_name_weight)
        else:
            base = M[pas]
            kc = keep_columns(base, r_rows, sc.max_df_char if pas == "name_char" else sc.max_df_word)
            X = base[:, kc]
        Xq = X[q_rows]
        if pas == "name_char":
            Xq = top_terms_per_row(Xq, int(sc.max_query_terms_char))
        with timed(f"sparse {split}/{country}/{pas}: top-{sc.top_k} over {r_rows.size:,} records"):
            qi, ri, s = topk(Xq, X[r_rows], sc.top_k, sc.min_score, nt, sc.chunk_size)
        f = pl.DataFrame({"q": rows[q_rows[qi]].astype(np.uint32),
                          "r": rows[r_rows[ri]].astype(np.uint32), "s": s})
        f = f.with_columns(pl.col("s").rank("ordinal", descending=True).over("q")
                           .cast(pl.UInt16).alias(f"rk_{pas}")).drop("s")
        out.append(f)
    allp = pl.concat(out, how="diagonal")
    return allp.group_by("q", "r").agg(pl.col(f"rk_{p}").min() for p in sc.passes)


def run(cfg, args) -> None:
    W = Work(cfg)
    fn, name = topn_backend()
    log(f"sparse top-k backend: {name}; threads={n_threads(cfg.sparse.n_threads)}")
    splits = [args.split] if getattr(args, "split", None) else ["train", "test"]
    cols = ["idx", "src", "country", "n_core_str", "a_tok"]
    for split in splits:
        rec = pl.read_parquet(W.records(split), columns=cols)
        parts = []
        for country in sorted(rec["country"].unique().to_list()):
            parts.append(retrieve_country(cfg, W, split, country, rec.filter(pl.col("country") == country)))
        cands = pl.concat([p for p in parts if p.height], how="vertical")
        cands.write_parquet(W.sparse(split))
        log(f"sparse {split}: {cands.height:,} candidate pairs -> {W.sparse(split)}")

        if split == "train" and W.gt().exists():
            gt = pl.read_parquet(W.gt())
            qs = rec.filter(pl.col("src") == 1)["idx"]
            rep = {"union": candidate_report(cands, gt, qs)}
            for p in cfg.sparse.passes:
                rep[p] = candidate_report(cands.filter(pl.col(f"rk_{p}").is_not_null()), gt, qs)
            save_json(rep, W.p("sparse", "train_report.json"))
            for k, v in rep.items():
                log(f"  recall[{k}] pair={v['pair_recall']:.4f} oracleF={v['oracle_macro_f']:.4f} "
                    f"pairs/q={v['pairs_per_query_mean']}")
