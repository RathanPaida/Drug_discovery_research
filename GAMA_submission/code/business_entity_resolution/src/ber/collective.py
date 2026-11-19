"""Level-2 ("collective") features for the second stacker.

The cross-encoder and the level-1 stacker judge one (S1, record) pair at a
time. Many of the remaining mistakes only show next to the other candidates:

competition  p1 of this pair against the S1's other candidates and against the
             other S1s that want the same record. An S1 that already holds
             several strong records rarely needs one more look-alike; a record
             two S1s both want is close to a coin flip.
drift        house-number drift against the S1. A drifted record next to an
             exact-number record of the same S1 is usually a neighbour, not a copy.
coherence    similarity of the record to the S1's best other record (soft transitivity).
namesake     among S1s sharing a name, which one already wins the most records
             that carry an address - the likely owner of the name-only copies.
twins        record graph over S2/S3 (no labels): copies of one entity share name
             and house number; stray look-alikes rarely do.

Everything is computed inside one "world" (the train set, a stress world or the test set)
from out-of-fold level-1 probabilities.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import strsim

REC_COLS = ["idx", "src", "country", "n_core", "n_core_str", "a_tok", "a_nums", "f_addr_empty"]
STATIC = ["h_rel", "g_twin_name", "g_twin_name_house", "g_s1_support", "g_sib_house"]
DYNAMIC = ["p1_logit",
           "c_r_n", "c_r_sum", "c_r_rank", "c_r_n05", "c_r_n03", "c_r_rival", "c_r_margin", "c_r_share",
           "c_q_n", "c_q_sum", "c_q_rank", "c_q_n05", "c_q_gap", "c_q_rival", "c_q_src_n05", "c_q_src_rank",
           "sib_exact_n", "drift_vs_exact_sib",
           "coh_name", "coh_addr", "coh_house_eq", "coh_p_other",
           "ns_size", "ns_cnt", "ns_gap", "ns_share"]
FEATURES = STATIC + DYNAMIC


# -- per-record keys and the record graph (static, per split) ----------------
def record_keys(recs: pl.DataFrame) -> pl.DataFrame:
    """idx, src, nk (country + sorted core-name tokens), house (first address
    number, leading zeros dropped), name/address strings, empty-address flag."""
    house = pl.col("a_nums").list.first().str.strip_chars_start("0")
    core = pl.col("n_core").list.sort().list.join(" ")
    return recs.select(
        "idx", "src",
        pl.when(core == "").then(None).otherwise(pl.concat_str([pl.col("country"), pl.lit("|"), core])).alias("nk"),
        pl.when(house == "").then(pl.lit("0")).otherwise(house).alias("house"),
        pl.col("n_core_str").fill_null("").alias("name_s"),
        pl.col("a_tok").list.join(" ").fill_null("").alias("addr_s"),
        pl.col("f_addr_empty").alias("addr_empty"),
    )


def graph_counts(keys: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(record twins, S1 support, name-at-house counts) over S2/S3 records only."""
    rec = keys.filter(pl.col("src") != 1)
    n_name = rec.drop_nulls("nk").group_by("nk").len("_nn")
    n_nh = rec.drop_nulls(["nk", "house"]).group_by("nk", "house").len("_nh")
    twins = (rec.select("idx", "nk", "house")
                .join(n_name, on="nk", how="left").join(n_nh, on=["nk", "house"], how="left")
                .select(pl.col("idx").alias("r"),
                        (pl.col("_nn") - 1).cast(pl.Float32).alias("g_twin_name"),
                        (pl.col("_nh") - 1).cast(pl.Float32).alias("g_twin_name_house")))
    support = (keys.filter(pl.col("src") == 1).select("idx", "nk", "house")
                   .join(n_nh, on=["nk", "house"], how="left")
                   .select(pl.col("idx").alias("q"), pl.col("_nh").fill_null(0).cast(pl.Float32).alias("g_s1_support")))
    return twins, support, n_nh


def house_rel(hq: pl.Expr, hr: pl.Expr) -> pl.Expr:
    """0 equal, 1 one is a prefix/suffix of the other (a digit dropped or added),
    2 differ by at most 2, 3 different; null when either side has no number."""
    iq, ir = hq.cast(pl.Int64, strict=False), hr.cast(pl.Int64, strict=False)
    affix = hq.str.starts_with(hr) | hr.str.starts_with(hq) | hq.str.ends_with(hr) | hr.str.ends_with(hq)
    return (pl.when(hq.is_null() | hr.is_null()).then(None)
              .when(hq == hr).then(0)
              .when(affix).then(1)
              .when((iq - ir).abs() <= 2).then(2)
              .otherwise(3)).cast(pl.Float32)


def static_pair_features(pairs: pl.DataFrame, keys: pl.DataFrame,
                         graph: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    """pairs (q, r) -> q, r + STATIC (independent of scores and of which S1s exist)."""
    twins, support, n_nh = graph
    kq = keys.select(pl.col("idx").alias("q"), pl.col("nk").alias("_qnk"), pl.col("house").alias("_qh"))
    kr = keys.select(pl.col("idx").alias("r"), pl.col("house").alias("_rh"))
    df = (pairs.select("q", "r").join(kq, on="q", how="left").join(kr, on="r", how="left")
               .join(twins, on="r", how="left").join(support, on="q", how="left")
               .join(n_nh.rename({"nk": "_qnk", "house": "_rh", "_nh": "g_sib_house"}), on=["_qnk", "_rh"], how="left"))
    return df.select("q", "r", house_rel(pl.col("_qh"), pl.col("_rh")).alias("h_rel"),
                     "g_twin_name", "g_twin_name_house", "g_s1_support",
                     pl.col("g_sib_house").fill_null(0).cast(pl.Float32))


# -- score-dependent features (per world) ------------------------------------
def _rival(df: pl.DataFrame, p: str, by: str, rank: str, name: str) -> pl.DataFrame:
    """Best p among the group's other members (0 when the pair is alone)."""
    top = df.group_by(by).agg(pl.col(p).max().alias("_m1"), pl.col(p).top_k(2).min().alias("_m2"), pl.len().alias("_n"))
    df = df.join(top, on=by, how="left")
    return df.with_columns(
        pl.when(pl.col("_n") == 1).then(0.0).when(pl.col(rank) == 1).then(pl.col("_m2"))
          .otherwise(pl.col("_m1")).cast(pl.Float32).alias(name)).drop("_m1", "_m2", "_n")


def competition(df: pl.DataFrame, p: str = "p1") -> pl.DataFrame:
    P = pl.col(p)
    df = df.with_columns(
        pl.len().over("r").cast(pl.Float32).alias("c_r_n"),
        P.sum().over("r").cast(pl.Float32).alias("c_r_sum"),
        P.rank("ordinal", descending=True).over("r").cast(pl.Float32).alias("c_r_rank"),
        (P > 0.5).sum().over("r").cast(pl.Float32).alias("c_r_n05"),
        (P > 0.3).sum().over("r").cast(pl.Float32).alias("c_r_n03"),
        pl.len().over("q").cast(pl.Float32).alias("c_q_n"),
        P.sum().over("q").cast(pl.Float32).alias("c_q_sum"),
        P.rank("ordinal", descending=True).over("q").cast(pl.Float32).alias("c_q_rank"),
        (P > 0.5).sum().over("q").cast(pl.Float32).alias("c_q_n05"),
        (P.max().over("q") - P).cast(pl.Float32).alias("c_q_gap"),
        (P > 0.5).sum().over("q", "r_src").cast(pl.Float32).alias("c_q_src_n05"),
        P.rank("ordinal", descending=True).over("q", "r_src").cast(pl.Float32).alias("c_q_src_rank"),
    )
    df = _rival(df, p, "r", "c_r_rank", "c_r_rival")
    df = _rival(df, p, "q", "c_q_rank", "c_q_rival")
    return df.with_columns(
        (P - pl.col("c_r_rival")).cast(pl.Float32).alias("c_r_margin"),
        (P / pl.max_horizontal(pl.lit(1.0), pl.col("c_r_sum"))).cast(pl.Float32).alias("c_r_share"),
    )


def drift(df: pl.DataFrame, p: str = "p1") -> pl.DataFrame:
    exact = ((pl.col("h_rel") == 0) & (pl.col(p) > 0.5)).fill_null(False).cast(pl.Int32)
    df = df.with_columns((exact.sum().over("q") - exact).cast(pl.Float32).alias("sib_exact_n"))
    return df.with_columns(((pl.col("h_rel") >= 1) & (pl.col("sib_exact_n") > 0))
                           .cast(pl.Float32).alias("drift_vs_exact_sib"))


def coherence(df: pl.DataFrame, keys: pl.DataFrame, p: str = "p1", nt: int = 1) -> pl.DataFrame:
    """Record vs the S1's top other record (top-1, or top-2 when this is the top-1)."""
    t1 = df.filter(pl.col("c_q_rank") == 1).select("q", pl.col("r").alias("_r1"), pl.col(p).alias("_p1o"))
    t2 = df.filter(pl.col("c_q_rank") == 2).select("q", pl.col("r").alias("_r2"), pl.col(p).alias("_p2o"))
    df = df.join(t1, on="q", how="left").join(t2, on="q", how="left")
    is_top = pl.col("r") == pl.col("_r1")
    df = df.with_columns(pl.when(is_top).then(pl.col("_r2")).otherwise(pl.col("_r1")).alias("_ro"),
                         pl.when(is_top).then(pl.col("_p2o")).otherwise(pl.col("_p1o")).cast(pl.Float32).alias("coh_p_other"))
    k = keys.select("idx", "name_s", "addr_s", "house")
    df = (df.join(k.rename({"idx": "r", "name_s": "_nr", "addr_s": "_ar", "house": "_hr"}), on="r", how="left")
            .join(k.rename({"idx": "_ro", "name_s": "_no", "addr_s": "_ao", "house": "_ho"}), on="_ro", how="left"))
    has = df["_ro"].is_not_null().to_numpy()
    nr, no = df["_nr"].fill_null("").to_list(), df["_no"].fill_null("").to_list()
    ar, ao = df["_ar"].fill_null("").to_list(), df["_ao"].fill_null("").to_list()
    cn = strsim.pairwise("ratio", nr, no, nt)
    ca = strsim.pairwise("token_set_ratio", ar, ao, nt)
    cn[~has], ca[~has] = np.nan, np.nan
    df = df.with_columns(pl.Series("coh_name", cn), pl.Series("coh_addr", ca),
                         pl.when(pl.col("_ro").is_null() | pl.col("_hr").is_null() | pl.col("_ho").is_null()).then(None)
                           .otherwise(pl.col("_hr") == pl.col("_ho")).cast(pl.Float32).alias("coh_house_eq"))
    return df.drop([c for c in df.columns if c.startswith("_")])


def namesake(df: pl.DataFrame, keys: pl.DataFrame, world_s1: pl.Series, p: str = "p1") -> pl.DataFrame:
    """For S1s sharing a name key: how many address-bearing records each one
    already wins (argmax S1 with p > 0.5), and the gap to the best namesake."""
    won = (df.filter(pl.col(p) > 0.5).sort([p, "q"], descending=[True, False])
             .unique("r", keep="first", maintain_order=True).select("q", "r"))
    won = won.join(keys.select(pl.col("idx").alias("r"), "addr_empty"), on="r").filter(~pl.col("addr_empty"))
    cnt = won.group_by("q").len("ns_cnt")
    s1 = (pl.DataFrame({"q": world_s1}).join(keys.select(pl.col("idx").alias("q"), "nk"), on="q", how="left")
            .join(cnt, on="q", how="left").with_columns(pl.col("ns_cnt").fill_null(0).cast(pl.Int64))  # signed: gaps go below 0
            .filter(pl.col("nk").is_not_null()))
    m1 = pl.col("ns_cnt").max().over("nk")
    s1 = s1.with_columns(pl.len().over("nk").alias("ns_size"), pl.col("ns_cnt").sum().over("nk").alias("_tot"),
                         m1.alias("_m1"), (pl.col("ns_cnt") == m1).sum().over("nk").alias("_n1"),
                         pl.when(pl.col("ns_cnt") < m1).then(pl.col("ns_cnt")).otherwise(None).max().over("nk").alias("_m2"))
    other = (pl.when(pl.col("ns_cnt") < pl.col("_m1")).then(pl.col("_m1"))
               .when(pl.col("_n1") >= 2).then(pl.col("_m1")).otherwise(pl.col("_m2").fill_null(0)))
    s1 = s1.select("q", pl.col("ns_size").cast(pl.Float32), pl.col("ns_cnt").cast(pl.Float32),
                   (pl.col("ns_cnt") - other).cast(pl.Float32).alias("ns_gap"),
                   (pl.col("ns_cnt") / pl.max_horizontal(pl.lit(1), pl.col("_tot"))).cast(pl.Float32).alias("ns_share"))
    return df.join(s1, on="q", how="left")


def add_features(df: pl.DataFrame, static: pl.DataFrame, keys: pl.DataFrame, world_s1: pl.Series,
                 p: str = "p1", nt: int = 1) -> pl.DataFrame:
    """df: one world's pairs with q, r, r_src and the level-1 probability `p`.
    Returns df + FEATURES."""
    eps = 1e-6
    df = df.join(static, on=["q", "r"], how="left")
    df = df.with_columns((pl.col(p).clip(eps, 1 - eps) / (1 - pl.col(p).clip(eps, 1 - eps))).log()
                         .cast(pl.Float32).alias("p1_logit"))
    df = competition(df, p)
    df = drift(df, p)
    df = coherence(df, keys, p, nt)
    return namesake(df, keys, world_s1, p)
