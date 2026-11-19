"""Step `prepare`: TSV -> normalised parquet + ground-truth pairs + folds + checks.

Outputs (under work/prepared/):
    {split}_records.parquet   one row per record of S1/S2/S3, `idx` = row number
    train_gt.parquet          (q, r) = (S1 idx, S2/S3 idx) true pairs
    maps/translit.parquet     Indic token -> Latin token, mined from training matches
    maps/state_alias.parquet  (country, alias, canonical state) incl. mined aliases
    report.json               row counts and data-assumption checks
"""
from __future__ import annotations

import polars as pl

from . import io, lexicon
from . import normalize as N
from .util import Work, explode, load_json, log, save_json, stable_bucket, timed

SPLITS = ("train", "test")


# -- loading & sampling ---------------------------------------------------
def load_raw(cfg, split: str) -> pl.DataFrame:
    parts = [io.read_source(cfg.paths.data_dir, split, s) for s in (1, 2, 3)]
    return pl.concat(parts).select("id", "src", "country", "name", "addr")


def _sample(raw: pl.DataFrame, gt: pl.DataFrame | None, n_s1: int) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Deterministic subsample: n_s1 S1 records, their true matches, and the
    same fraction of all other records (keeps the distractor ratio)."""
    s1 = raw.filter(pl.col("src") == 1)
    rate = min(1.0, n_s1 / max(s1.height, 1))
    key = pl.Series(stable_bucket(raw["id"], 1 << 30, salt=11).astype("int64"))
    raw = raw.with_columns(key.alias("_h"))
    keep_s1 = raw.filter(pl.col("src") == 1).sort("_h").head(n_s1).select("id")
    other = raw.filter((pl.col("src") != 1) & (pl.col("_h") < int(rate * (1 << 30))))
    if gt is not None:
        # only truly unmatched records join as distractors, so the sample keeps
        # the real records-per-S1 ratio (matches of unsampled S1s are dropped)
        other = other.join(gt.select(pl.col("rec_id").alias("id")).drop_nulls(), on="id", how="anti")
    ids = [keep_s1, other.select("id")]
    if gt is not None:
        gt = gt.join(keep_s1.rename({"id": "s1_id"}), on="s1_id", how="semi")
        ids.append(gt.select(pl.col("rec_id").alias("id")).drop_nulls())
    keep = pl.concat(ids).unique()
    return raw.join(keep, on="id", how="semi").drop("_h"), gt


# -- normalisation --------------------------------------------------------
def base_columns(df: pl.DataFrame, fr_rules: bool = False) -> pl.DataFrame:
    df = df.with_columns(N.base_text("name").alias("t_name"), N.base_text("addr").alias("t_addr"))
    # before every downstream view, so the neural models and the lexical tokens agree.
    # France has no training rows, so only the test side changes:
    #   always      "& Fils" / "& Associes" add-ons stripped (LB 0.985 -> 0.9873 with the new stack)
    #   fr_rules    also French short forms / number noise / departement -> region in S1
    #               style (normalize.fr_name, fr_addr): LB 0.98707 vs 0.98728 without, so off
    fr = pl.col("country") == "France"
    if fr_rules:
        df = df.with_columns(pl.when(fr).then(N.fr_name(pl.col("t_name"))).otherwise(pl.col("t_name")).alias("t_name"),
                             pl.when(fr).then(N.fr_addr(pl.col("t_addr"))).otherwise(pl.col("t_addr")).alias("t_addr"))
    else:
        df = df.with_columns(pl.when(fr).then(N.strip_fr_addons(pl.col("t_name"))).otherwise(pl.col("t_name")).alias("t_name"))
    return df.with_columns(
        N.split_tokens(N.name_clean(pl.col("t_name"))).alias("tok0"),
        N.split_tokens(N.addr_clean(pl.col("t_addr"))).alias("atok0"),
        N.addr_components().alias("comps"),
        pl.col("t_addr").str.extract_all(r"\d+").alias("a_nums"),
        (pl.col("t_addr") == "").alias("f_addr_empty"),
        pl.col("t_addr").str.contains(N.INDIC_RE).alias("f_indic_addr"),
        *N.name_flags(),
    )


def chunked(df: pl.DataFrame, fn, size: int = 1_500_000) -> pl.DataFrame:
    """Apply a row-wise frame transform in slices - bounds peak memory."""
    if df.height <= size:
        return fn(df)
    return pl.concat([fn(df.slice(i, size)) for i in range(0, df.height, size)], rechunk=True)


def static_state_table() -> pl.DataFrame:
    return pl.DataFrame(lexicon.state_rows(), schema=["country", "comp", "state"], orient="row")


def detect_state(df: pl.DataFrame, states: pl.DataFrame) -> pl.Series:
    """Canonical state = last address component found in the (country, alias) table."""
    ex = (explode(df.select("idx", "country", "comps"), "comps")
            .with_columns(pl.int_range(pl.len()).over("idx").alias("pos"))
            .rename({"comps": "comp"})
            .join(states, on=["country", "comp"], how="inner"))
    last = ex.sort("idx", "pos").group_by("idx").agg(pl.col("state").last())
    return df.select("idx").join(last, on="idx", how="left")["state"]


# -- mining from training matches -----------------------------------------
def mine_translit(df: pl.DataFrame, gt: pl.DataFrame, min_count: int, min_share: float) -> dict[str, str]:
    """Token alignment on (S1 Latin name, Indic record name) pairs with equal
    token counts; majority vote per Indic token."""
    s1 = df.filter(pl.col("src") == 1).select(pl.col("idx").alias("q"), pl.col("tok0").alias("a"))
    rec = df.filter(pl.col("f_indic_name")).select(pl.col("idx").alias("r"), pl.col("tok0").alias("b"))
    pairs = (gt.join(rec, on="r").join(s1, on="q")
               .filter(pl.col("a").list.len() == pl.col("b").list.len())
               .select("a", "b"))
    pairs = explode(pairs, "a", "b").filter(
        pl.col("b").str.contains(N.INDIC_RE) & ~pl.col("a").str.contains(N.INDIC_RE))
    counts = pairs.group_by("b", "a").len()
    best = (counts.with_columns(pl.col("len").sum().over("b").alias("tot"))
                  .sort("len", descending=True).group_by("b").first()
                  .filter((pl.col("len") >= min_count) & (pl.col("len") / pl.col("tot") >= min_share)))
    log(f"translit: {pairs.height:,} aligned token pairs -> {best.height:,} Indic tokens mapped")
    return dict(zip(best["b"].to_list(), best["a"].to_list()))


def mine_state_aliases(df: pl.DataFrame, gt: pl.DataFrame, static: pl.DataFrame,
                       min_count: int, min_share: float) -> pl.DataFrame:
    """Short or native-script address components -> the matched S1's state."""
    s1 = df.filter(pl.col("src") == 1).select(pl.col("idx").alias("q"), pl.col("state").alias("s1_state"))
    rec = (explode(df.filter(pl.col("src") != 1).select(pl.col("idx").alias("r"), "country", "comps"), "comps")
             .rename({"comps": "comp"}).drop_nulls("comp")
             .filter(pl.col("comp").str.contains(N.INDIC_RE) | (pl.col("comp").str.len_chars() <= 3))
             .join(static, on=["country", "comp"], how="anti"))
    votes = (gt.join(s1, on="q").drop_nulls("s1_state").join(rec, on="r")
               .group_by("country", "comp", "s1_state").len())
    best = (votes.with_columns(pl.col("len").sum().over("country", "comp").alias("tot"))
                 .sort("len", descending=True).group_by("country", "comp").first()
                 .filter((pl.col("len") >= min_count) & (pl.col("len") / pl.col("tot") >= min_share))
                 .select("country", "comp", pl.col("s1_state").alias("state")))
    log(f"state aliases mined: {best.height:,}")
    return best


# -- assembling the lexical view ------------------------------------------
def token_maps(df: pl.DataFrame, translit: dict[str, str], states: pl.DataFrame) -> tuple[dict, dict]:
    """(name map, address map): mined transliterations + fallback romanisation
    for the remaining Indic tokens; address map also turns native-script state
    names into their canonical Latin form."""
    ind = df.filter(pl.col("f_indic_name") | pl.col("f_indic_addr"))
    fb = N.fallback_translit_map(pl.concat([ind["tok0"], ind["atok0"]]))
    fb = {k: v for k, v in fb.items() if k not in translit}
    tmap = {**fb, **translit}
    smap = {r["comp"]: r["state"] for r in
            states.filter(pl.col("comp").str.contains(N.INDIC_RE)).iter_rows(named=True)}
    log(f"token maps: {len(translit):,} mined + {len(fb):,} fallback transliterations, "
        f"{len(smap):,} native-script states")
    return tmap, {**tmap, **smap}


def lexical_columns(df: pl.DataFrame, nmap: dict, amap: dict) -> pl.DataFrame:
    df = df.with_columns(
        N.map_tokens(N.map_tokens(pl.col("tok0"), nmap), lexicon.LEGAL_CANON).alias("n_tok"),
        N.split_tokens(N.map_tokens(N.map_tokens(pl.col("atok0"), amap), lexicon.ADDR_CANON)
                       .list.join(" ")).alias("a_tok"),
    ).drop("tok0", "atok0", "comps")
    df = df.with_columns(*N.name_views())
    return df.with_columns(pl.col("n_core").list.join(" ").alias("n_core_str"))


def add_name_frequency(df: pl.DataFrame) -> pl.DataFrame:
    """How many S1 / S2+S3 records share this core name in this country."""
    key = ["country", "n_core_str"]
    f1 = df.filter(pl.col("src") == 1).group_by(key).len().rename({"len": "nf_s1"})
    f2 = df.filter(pl.col("src") != 1).group_by(key).len().rename({"len": "nf_rec"})
    return (df.join(f1, on=key, how="left").join(f2, on=key, how="left")
              .with_columns(pl.col("nf_s1").fill_null(0).cast(pl.UInt32),
                            pl.col("nf_rec").fill_null(0).cast(pl.UInt32)))


def add_folds(df: pl.DataFrame, cfg, owner: pl.DataFrame | None) -> pl.DataFrame:
    fnn = stable_bucket(df["id"], cfg.data.n_folds_nn, salt=1)
    fgb = stable_bucket(df["id"], cfg.data.n_folds_gbdt, salt=2)
    df = df.with_columns(
        pl.when(pl.col("src") == 1).then(pl.Series(fnn)).otherwise(-1).cast(pl.Int8).alias("fold_nn"),
        pl.when(pl.col("src") == 1).then(pl.Series(fgb)).otherwise(-1).cast(pl.Int8).alias("fold_gb"),
    )
    if owner is None:
        return df.with_columns(pl.lit(None, pl.UInt32).alias("owner"))
    df = df.join(owner, left_on="idx", right_on="r", how="left").rename({"q": "owner"})
    own_fold = df.select(pl.col("idx").alias("owner"), pl.col("fold_nn").alias("_of")).filter(pl.col("_of") >= 0)
    df = df.join(own_fold, on="owner", how="left")
    return df.with_columns(pl.when(pl.col("src") == 1).then(pl.col("fold_nn"))
                             .otherwise(pl.col("_of").fill_null(-1)).cast(pl.Int8).alias("fold_nn")).drop("_of")


KEEP_COLS = ["idx", "id", "src", "country", "name", "addr", "t_name", "t_addr",
             "n_tok", "n_core", "n_alt", "n_legal", "n_core_str", "a_tok", "a_nums", "state",
             "f_domain", "f_dba", "f_phone", "f_indic_name", "f_indic_addr", "f_addr_empty",
             "nf_s1", "nf_rec", "fold_nn", "fold_gb", "owner"]


# -- checks ---------------------------------------------------------------
def split_report(df: pl.DataFrame) -> dict:
    by = df.group_by("src", "country").len().sort("src", "country")
    rec = df.filter(pl.col("src") != 1)
    return {
        "rows": df.height,
        "by_src_country": {f"S{r['src']}/{r['country']}": r["len"] for r in by.iter_rows(named=True)},
        "duplicate_ids": df.height - df["id"].n_unique(),
        "empty_names": int((df["t_name"] == "").sum()),
        "empty_addr_by_src": {f"S{s}": int(df.filter(pl.col("src") == s)["f_addr_empty"].sum()) for s in (1, 2, 3)},
        "indic_name_records": int(rec["f_indic_name"].sum()),
        "indic_addr_records": int(rec["f_indic_addr"].sum()),
        "domain_names": int(rec["f_domain"].sum()),
        "dba_names": int(rec["f_dba"].sum()),
        "phone_in_name": int(rec["f_phone"].sum()),
        "state_detected_share": {c: round(float(g["state"].is_not_null().mean()), 4)
                                 for (c,), g in df.group_by("country")},
        "s1_sharing_core_name_share": round(float((df.filter(pl.col("src") == 1)["nf_s1"] > 1).mean()), 4),
    }


def gt_report(gt_ids: pl.DataFrame, gt: pl.DataFrame, df: pl.DataFrame) -> dict:
    s1 = df.filter(pl.col("src") == 1)
    n_links = gt_ids.drop_nulls("rec_id").height
    size = gt_ids.group_by("s1_id").agg(pl.col("rec_id").drop_nulls().len().alias("k"))
    cmp = (gt.join(df.select(pl.col("idx").alias("q"), pl.col("country").alias("c1")), on="q")
             .join(df.select(pl.col("idx").alias("r"), pl.col("country").alias("c2"), "src"), on="r"))
    return {
        "gt_rows": gt_ids["s1_id"].n_unique(),
        "s1_rows": s1.height,
        "s1_missing_from_gt": s1.height - s1.join(gt_ids.select(pl.col("s1_id").alias("id")).unique(), on="id", how="semi").height,
        "gt_s1_not_in_source1": gt_ids.select("s1_id").unique().join(s1.select(pl.col("id").alias("s1_id")), on="s1_id", how="anti").height,
        "links": n_links,
        "links_unresolved": n_links - gt.height,
        "records_in_multiple_clusters": gt.group_by("r").len().filter(pl.col("len") > 1).height,
        "links_country_mismatch": int((cmp["c1"] != cmp["c2"]).sum()),
        "singletons": int((size["k"] == 0).sum()),
        "cluster_size_hist": {str(r["k"]): r["len"] for r in size.group_by("k").len().sort("k").iter_rows(named=True)},
        "links_by_src": {f"S{r['src']}": r["len"] for r in cmp.group_by("src").len().sort("src").iter_rows(named=True)},
        "unmatched_records": df.filter((pl.col("src") != 1) & pl.col("owner").is_null()).height,
    }


# -- step -----------------------------------------------------------------
def run(cfg, args=None) -> None:
    W = Work(cfg)
    dcfg = cfg.data
    report: dict = {}
    translit: dict[str, str] = {}
    states = static_state_table()
    only = getattr(args, "split", None) if args is not None else None
    splits = [only] if only else list(SPLITS)
    if only == "test":
        # test side only (e.g. after a test-side normalisation change): the train
        # records stay as they are and the maps mined from them are reused
        tp = pl.read_parquet(W.p("prepared", "maps", "translit.parquet"))
        translit = dict(zip(tp["indic"].to_list(), tp["latin"].to_list()))
        # + static rows added since (e.g. the French regions): keyed by country, so the
        # train-side detections they would change do not exist
        states = pl.concat([pl.read_parquet(W.p("prepared", "maps", "state_alias.parquet")),
                            static_state_table()]).unique(["country", "comp"], keep="first")
        if W.p("prepared", "report.json").exists():
            report = load_json(W.p("prepared", "report.json"))
    fr_rules = bool(dcfg.get("fr_rules", False))
    if not fr_rules:                               # the French region table belongs to the French rules
        states = states.filter(pl.col("country") != "France")

    for split in splits:
        with timed(f"prepare {split}"):
            raw = load_raw(cfg, split)
            gt_ids = io.read_ground_truth(cfg.paths.data_dir) if split == "train" else None
            if dcfg.sample_s1 and dcfg.sample_s1 > 0:
                raw, gt_ids = _sample(raw, gt_ids, int(dcfg.sample_s1))
                log(f"sampled {split}: {raw.height:,} records")
            df = chunked(raw.with_row_index("idx"), lambda d: base_columns(d, fr_rules))
            del raw

            gt = owner = None
            if gt_ids is not None:
                ids = df.select("idx", "id")
                gt = (gt_ids.drop_nulls("rec_id")
                            .join(ids.rename({"id": "s1_id", "idx": "q"}), on="s1_id")
                            .join(ids.rename({"id": "rec_id", "idx": "r"}), on="rec_id")
                            .select("q", "r"))
                owner = gt.unique("r", keep="first")
                gt.write_parquet(W.gt())

            df = df.with_columns(chunked(df, lambda d: detect_state(d, states).to_frame()).to_series())
            if split == "train":
                translit = mine_translit(df, gt, dcfg.translit_min_count, dcfg.translit_min_share)
                mined = mine_state_aliases(df, gt, states, dcfg.state_alias_min_count, dcfg.state_alias_min_share)
                states = pl.concat([states, mined])
                pl.DataFrame({"indic": list(translit), "latin": list(translit.values())}).write_parquet(
                    W.p("prepared", "maps", "translit.parquet"))
                states.write_parquet(W.p("prepared", "maps", "state_alias.parquet"))
                # re-detect with mined aliases (native-script states etc.)
                df = df.with_columns(chunked(df, lambda d: detect_state(d, states).to_frame()).to_series())

            nmap, amap = token_maps(df, translit, states)
            df = chunked(df, lambda d: lexical_columns(d, nmap, amap))
            df = add_name_frequency(df)
            df = add_folds(df, cfg, owner)
            df = df.select(KEEP_COLS)
            df.write_parquet(W.records(split))
            log(f"wrote {W.records(split)}  ({df.height:,} rows, {df.estimated_size('mb'):,.0f} MB in memory)")

            report[split] = split_report(df)
            if gt is not None:
                report["train_gt"] = gt_report(gt_ids, gt, df)
            del df
    save_json(report, W.p("prepared", "report.json"))
    log(f"report -> {W.p('prepared', 'report.json')}")
