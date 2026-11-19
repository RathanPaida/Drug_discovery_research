"""Step `features --split S`: candidate union (sparse | dense) + pairwise features (CPU).

Feature groups (all country-agnostic - `country` is kept only as metadata):
  retrieval   dense cosine/rank, sparse pass ranks, exact TF-IDF cosines
  context     candidates per S1, S1s competing per record, gaps to the best
  name        RapidFuzz scores on the core name, alias after "dba",
              concatenated form (domains), token overlap, extra tokens, legal forms
  address     RapidFuzz scores, token overlap, house-number agreement, state
  record      source (S2/S3), noise flags, name frequencies

Output: work/cands/{split}/part_XXXX.parquet  (all S1 pairs of a part are together)
"""
from __future__ import annotations

import shutil

import numpy as np
import polars as pl
import scipy.sparse as sp

from . import strsim
from .sparse import MATRICES, mat_path
from .util import Work, log, n_threads, timed

META = ("q", "r", "label", "fold_gb", "country")
REC_COLS = ["src", "country", "n_core_str", "n_core", "n_alt", "n_legal", "a_tok", "a_nums", "state",
            "f_domain", "f_dba", "f_phone", "f_indic_name", "f_indic_addr", "f_addr_empty",
            "nf_s1", "nf_rec", "fold_gb", "owner"]


def feature_columns(df: pl.DataFrame | pl.LazyFrame, extra_meta: tuple = ()) -> list[str]:
    cols = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
    return [c for c in cols if c not in META and c not in extra_meta]


def parts_glob(W: Work, split: str) -> str:
    return str(W.p("cands", split, "part_*.parquet"))


def part_edges(q_sorted: np.ndarray, chunk_rows: int) -> np.ndarray:
    """Row offsets of parts of ~chunk_rows rows over rows sorted by q. Each S1 goes to
    the part its first row falls in, so all pairs of one S1 stay together."""
    n = len(q_sorted)
    if n == 0:
        return np.array([0, 0])
    starts = np.flatnonzero(np.r_[True, q_sorted[1:] != q_sorted[:-1]])   # first row of each S1
    part = starts // chunk_rows
    first_of_part = np.r_[True, part[1:] != part[:-1]]
    return np.r_[starts[first_of_part], n]


# -- candidate union + context --------------------------------------------
def load_union(W: Work, split: str) -> pl.DataFrame:
    sp_ = pl.read_parquet(W.sparse(split))
    if W.dense(split).exists():
        de = pl.read_parquet(W.dense(split))
        c = de.join(sp_, on=["q", "r"], how="full", coalesce=True)
    else:
        log(f"WARNING: no dense candidates for {split} - sparse-only features")
        c = sp_.with_columns(pl.lit(None, pl.Float32).alias("dense_sim"),
                             pl.lit(None, pl.UInt16).alias("dense_rank"))
    rk = [x for x in c.columns if x.startswith("rk_")]
    c = c.with_columns(pl.min_horizontal(rk).alias("sparse_best_rank"),
                       pl.sum_horizontal([pl.col(x).is_not_null() for x in rk]).cast(pl.Int8).alias("n_sparse_passes"))
    return c.with_columns(
        pl.len().over("q").cast(pl.UInt16).alias("q_ncand"),
        (pl.col("dense_sim").max().over("q") - pl.col("dense_sim")).alias("dense_gap_q"),
        pl.len().over("r").cast(pl.UInt16).alias("r_nq"),
        pl.col("dense_sim").rank("ordinal", descending=True).over("r").cast(pl.UInt16).alias("dense_rank_r"),
        (pl.col("dense_sim").max().over("r") - pl.col("dense_sim")).alias("dense_gap_r"),
    )


# -- sparse cosines from the matrices saved by `sparse` -------------------
class SparseScorer:
    def __init__(self, W: Work, split: str, countries: list[str], n_records: int):
        self.row = np.full(n_records, -1, dtype=np.int64)
        self.cty = np.full(n_records, -1, dtype=np.int16)
        self.mats: dict[str, list[sp.csr_matrix]] = {k: [] for k in MATRICES}
        for ci, c in enumerate(countries):
            rows_path = mat_path(W, split, c, "rows")
            if not rows_path.exists():
                for k in MATRICES:
                    self.mats[k].append(None)
                continue
            rows = np.load(rows_path)
            self.row[rows] = np.arange(len(rows))
            self.cty[rows] = ci
            for k in MATRICES:
                self.mats[k].append(sp.load_npz(mat_path(W, split, c, k)).tocsr())

    def score(self, kind: str, q: np.ndarray, r: np.ndarray, sub: int = 1_000_000) -> np.ndarray:
        out = np.full(len(q), np.nan, dtype=np.float32)
        cq = self.cty[q]
        for ci in np.unique(cq):
            M = self.mats[kind][ci] if ci >= 0 else None
            if M is None:
                continue
            sel = np.flatnonzero(cq == ci)
            for s in range(0, len(sel), sub):
                ix = sel[s:s + sub]
                A, B = M[self.row[q[ix]]], M[self.row[r[ix]]]
                out[ix] = np.asarray(A.multiply(B).sum(axis=1)).ravel()
        return out


# -- pairwise features for one chunk --------------------------------------
def pair_features(ch: pl.DataFrame, recs: pl.DataFrame, scorer: SparseScorer, cfg, nt: int,
                  labelled: bool) -> pl.DataFrame:
    qi = ch["q"].to_numpy().astype(np.int64)
    ri = ch["r"].to_numpy().astype(np.int64)
    Q = recs[qi].rename(lambda c: f"q_{c}")
    R = recs[ri].rename(lambda c: f"r_{c}")
    df = pl.concat([ch, Q, R], how="horizontal")
    n = df.height

    df = df.with_columns(
        pl.col("q_n_core").list.set_intersection(pl.col("r_n_core")).list.len().alias("n_inter"),
        pl.col("q_n_core").list.set_union(pl.col("r_n_core")).list.len().alias("_n_union"),
        pl.col("q_n_core").list.len().alias("q_n_len"),
        pl.col("r_n_core").list.len().alias("r_n_len"),
        (pl.col("q_n_core").list.first() == pl.col("r_n_core").list.first()).alias("n_first_eq"),
        ((pl.col("q_n_core_str") == pl.col("r_n_core_str")) & (pl.col("q_n_core_str") != "")).alias("n_exact"),
        pl.col("q_n_legal").list.set_intersection(pl.col("r_n_legal")).list.len().alias("legal_inter"),
        (pl.col("q_n_legal").list.join(" ") == pl.col("r_n_legal").list.join(" ")).alias("legal_eq"),
        (pl.col("q_n_legal").list.len() == 0).alias("q_legal_empty"),
        (pl.col("r_n_legal").list.len() == 0).alias("r_legal_empty"),
        pl.col("q_a_tok").list.set_intersection(pl.col("r_a_tok")).list.len().alias("a_inter"),
        pl.col("q_a_tok").list.unique().list.len().alias("q_a_len"),
        pl.col("r_a_tok").list.unique().list.len().alias("r_a_len"),
        pl.col("q_a_nums").list.set_intersection(pl.col("r_a_nums")).list.len().alias("num_inter"),
        pl.col("q_a_nums").list.len().alias("q_num_len"),
        pl.col("r_a_nums").list.len().alias("r_num_len"),
        pl.col("q_a_nums").list.first().alias("_qn1"),
        pl.col("r_a_nums").list.first().alias("_rn1"),
        pl.when(pl.col("q_state").is_null() | pl.col("r_state").is_null()).then(None)
          .otherwise((pl.col("q_state") == pl.col("r_state")).cast(pl.Float32)).alias("state_eq"),
    )
    df = df.with_columns(
        (pl.col("n_inter") / pl.col("_n_union")).fill_nan(0.0).alias("n_jacc"),
        (pl.col("q_n_len") - pl.col("n_inter")).alias("q_n_extra"),
        (pl.col("r_n_len") - pl.col("n_inter")).alias("r_n_extra"),
        (pl.col("a_inter") / (pl.col("q_a_len") + pl.col("r_a_len") - pl.col("a_inter"))).fill_nan(0.0).alias("a_jacc"),
        (pl.col("_qn1") == pl.col("_rn1")).cast(pl.Float32).alias("num1_eq"),
        (pl.col("_qn1").str.contains(pl.col("_rn1"), literal=True)
         | pl.col("_rn1").str.contains(pl.col("_qn1"), literal=True)).cast(pl.Float32).alias("num1_sub"),
        (pl.col("_qn1").cast(pl.Float64, strict=False) - pl.col("_rn1").cast(pl.Float64, strict=False))
          .abs().log1p().cast(pl.Float32).alias("num1_logdiff"),
        pl.col("q_nf_s1").log1p().alias("q_nf_s1"),
        pl.col("q_nf_rec").log1p().alias("q_nf_rec"),
        pl.col("r_nf_s1").log1p().alias("r_nf_s1"),
        pl.col("r_nf_rec").log1p().alias("r_nf_rec"),
    )

    # string similarities (RapidFuzz, multi-threaded)
    feats: dict[str, np.ndarray] = {}
    qn, rn = df["q_n_core_str"], df["r_n_core_str"]
    empty_n = ((qn == "") | (rn == "")).to_numpy()
    a_name, b_name = qn.to_list(), rn.to_list()
    for s in ("ratio", "partial_ratio", "token_set_ratio", "token_sort_ratio", "jaro_winkler"):
        feats[f"nm_{s}"] = strsim.pairwise(s, a_name, b_name, nt, empty_n)
    qc, rc = qn.str.replace_all(" ", ""), rn.str.replace_all(" ", "")
    feats["nm_concat_ratio"] = strsim.pairwise("ratio", qc.to_list(), rc.to_list(), nt, empty_n)
    alt = df.select(pl.col("r_n_alt").list.join(" ")).to_series()
    has_alt = np.flatnonzero((alt != "").to_numpy() & ~empty_n)
    feats["nm_alt_tsr"] = np.zeros(n, dtype=np.float32)
    if has_alt.size:
        feats["nm_alt_tsr"][has_alt] = strsim.pairwise(
            "token_set_ratio", [a_name[i] for i in has_alt], alt.gather(has_alt).to_list(), nt)
    qa = df.select(pl.col("q_a_tok").list.join(" ")).to_series()
    ra = df.select(pl.col("r_a_tok").list.join(" ")).to_series()
    empty_a = ((qa == "") | (ra == "")).to_numpy()
    qa_l, ra_l = qa.to_list(), ra.to_list()
    for s in ("ratio", "token_set_ratio", "partial_ratio"):
        feats[f"ad_{s}"] = strsim.pairwise(s, qa_l, ra_l, nt, empty_a)

    for kind in MATRICES:
        feats[f"cos_{kind}"] = scorer.score(kind, qi, ri)
    w = float(cfg.sparse.full_name_weight)
    feats["cos_full"] = w * feats["cos_name_word"] + (1 - w) * np.nan_to_num(feats["cos_addr_word"])

    df = df.with_columns([pl.Series(k, v) for k, v in feats.items()])
    label = ((pl.col("r_owner") == pl.col("q")).fill_null(False).cast(pl.Int8) if labelled
             else pl.lit(None, pl.Int8))
    df = df.with_columns(label.alias("label"), pl.col("q_fold_gb").alias("fold_gb"),
                         pl.col("q_country").alias("country"))
    drop = [c for c in df.columns if c.startswith("_")] + [
        "q_n_core_str", "r_n_core_str", "q_n_core", "r_n_core", "q_n_alt", "r_n_alt", "q_n_legal", "r_n_legal",
        "q_a_tok", "r_a_tok", "q_a_nums", "r_a_nums", "q_state", "r_state", "q_country", "r_country",
        "q_fold_gb", "r_fold_gb", "q_owner", "r_owner", "q_src",
        "q_f_domain", "q_f_dba", "q_f_phone", "q_f_indic_name", "q_f_indic_addr", "q_f_addr_empty"]
    df = df.drop([c for c in drop if c in df.columns])
    feat_cols = feature_columns(df)
    return df.with_columns([pl.col(c).cast(pl.Float32) for c in feat_cols])


def run(cfg, args) -> None:
    W = Work(cfg)
    nt = n_threads(cfg.features.n_threads)
    log(f"string-similarity backend: {strsim.backend()}, threads={nt}")
    splits = [args.split] if getattr(args, "split", None) else ["train", "test"]
    for split in splits:
        recs = pl.read_parquet(W.records(split), columns=["idx"] + REC_COLS)
        assert (recs["idx"].to_numpy() == np.arange(recs.height)).all(), "records idx must equal row number"
        recs = recs.drop("idx")
        countries = sorted(recs["country"].unique().to_list())
        with timed(f"features {split}: loading candidate union + TF-IDF matrices"):
            union = load_union(W, split)
            scorer = SparseScorer(W, split, countries, recs.height)
        # sorted by S1, then cut into parts of ~chunk_rows pairs at S1 boundaries;
        # slices are zero-copy, so only one part's features are in memory at a time
        union = union.sort("q")
        edges = part_edges(union["q"].to_numpy(), int(cfg.features.chunk_rows))
        n_parts = len(edges) - 1
        out_dir = W.p("cands", split, "x").parent
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        for part in range(n_parts):
            ch = union.slice(int(edges[part]), int(edges[part + 1] - edges[part]))
            if ch.height == 0:
                continue
            with timed(f"features {split}: part {part+1}/{n_parts} ({ch.height:,} pairs)"):
                f = pair_features(ch, recs, scorer, cfg, nt,
                                  labelled=split == "train" and W.gt().exists())
                f.write_parquet(out_dir / f"part_{part:04d}.parquet")
            del ch, f
        if split == "train":
            lab = pl.scan_parquet(parts_glob(W, split)).select(pl.col("label").mean(), pl.len()).collect()
            log(f"features train: {lab['len'][0]:,} pairs, positive rate {lab['label'][0]:.4f}")
