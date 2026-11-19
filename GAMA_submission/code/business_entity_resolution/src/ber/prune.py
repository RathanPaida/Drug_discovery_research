"""Step `prune`: GBDT pruner over the candidate union (CPU).

Keeps, per S1, the `min_keep` best-scored candidates plus any candidate with
p >= tau among the `max_keep` best; tau is the largest threshold whose
out-of-fold recall of retrieved true pairs is still >= `keep_recall`.

The kept set is exactly what the cross-encoder scores, so for the test split
it is written as output/candidate_pairs.tsv.

Outputs: work/prune/{split}_pruned.parquet (q, r, prune_p, prune_rank[, label])
         work/prune/report.json, output/candidate_pairs.tsv
"""
from __future__ import annotations

import glob

import polars as pl

from . import gbdt, io
from .features import feature_columns, parts_glob
from .metrics import candidate_report
from .util import Work, die, load_json, log, only_countries, save_json


def with_rank(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("prune_p").rank("ordinal", descending=True).over("q")
                           .cast(pl.UInt16).alias("prune_rank"))


def apply_rule(df: pl.DataFrame, tau: float, pc) -> pl.DataFrame:
    """df must carry prune_rank (see with_rank)."""
    return df.filter((pl.col("prune_rank") <= pc.min_keep)
                     | ((pl.col("prune_p") >= tau) & (pl.col("prune_rank") <= pc.max_keep)))


def choose_tau(oof: pl.DataFrame, pc) -> float:
    pos = oof.filter(pl.col("label") == 1).select("prune_p", "prune_rank")
    total = max(pos.height, 1)
    lo, hi = 0.0, 1.0
    for _ in range(30):                       # bisection on the monotone recall(tau)
        mid = (lo + hi) / 2
        if apply_rule(pos, mid, pc).height / total >= pc.keep_recall:
            lo = mid
        else:
            hi = mid
    return lo


def run(cfg, args) -> None:
    W = Work(cfg)
    pc = cfg.prune
    train = pl.scan_parquet(parts_glob(W, "train"))
    feats = feature_columns(train)
    if getattr(args, "split", None) == "test":
        # test side only: the saved fold models and tau of the last full run
        rep = load_json(W.p("prune", "report.json"))
        models = gbdt.load_fold_models(W.p("prune", "models", "x").parent, "prune", int(cfg.data.n_folds_gbdt), feats)
        log(f"prune test only: {len(models)} saved models, tau={rep['tau']:.4f}")
        predict_test(cfg, W, models, feats, float(rep["tau"]))
        return
    log(f"prune: {len(feats)} features")
    oof, models = gbdt.cross_fit(cfg, train, feats, pc.train_s1_frac, W.p("prune", "models", "x").parent, "prune")
    lab = train.select("q", "r", "label").collect()
    oof = with_rank(oof.rename({"p": "prune_p"}).join(lab, on=["q", "r"]))
    tau = choose_tau(oof, pc)
    kept = apply_rule(oof, tau, pc)
    kept.write_parquet(W.pruned("train"))

    gt = pl.read_parquet(W.gt())
    rec = pl.read_parquet(W.records("train"), columns=["idx", "src", "country"])
    qs = rec.filter(pl.col("src") == 1)["idx"]
    rep = {"tau": tau, "union": candidate_report(oof, gt, qs), "pruned": candidate_report(kept, gt, qs),
           "retrieved_positives_kept": round(int(kept["label"].sum()) / max(int(oof["label"].sum()), 1), 5)}
    for (c,), g in rec.filter(pl.col("src") == 1).group_by("country"):
        rep[f"pruned[{c}]"] = candidate_report(kept, gt, g["idx"])
    save_json(rep, W.p("prune", "report.json"))
    log(f"prune: tau={tau:.4f}  union recall {rep['union']['pair_recall']:.4f} ({rep['union']['pairs_per_query_mean']}/S1) "
        f"-> pruned recall {rep['pruned']['pair_recall']:.4f} ({rep['pruned']['pairs_per_query_mean']}/S1), "
        f"oracle F {rep['pruned']['oracle_macro_f']:.4f}")

    predict_test(cfg, W, models, feats, tau)


def predict_test(cfg, W: Work, models: list, feats: list[str], tau: float) -> None:
    """Test: average of the K fold models, part by part."""
    pc = cfg.prune
    only = only_countries(cfg, "test")
    outs = []
    for path in sorted(glob.glob(parts_glob(W, "test"))):
        part = pl.read_parquet(path, columns=["q", "r", "country"] + feats)
        if only:                                   # partial rerun: all pairs of one S1 share its country
            part = part.filter(pl.col("country").is_in(only))
        p = gbdt.predict_avg(models, part.select(feats).to_numpy())
        outs.append(part.select("q", "r").with_columns(pl.Series("prune_p", p)))
    test = apply_rule(with_rank(pl.concat(outs)), tau, pc)
    if only:
        if not W.pruned("test").exists():
            die(f"data.only_countries={only} needs the previous {W.pruned('test')} for the other countries")
        ctry = pl.read_parquet(W.records("test"), columns=["idx", "country"]).select(
            pl.col("idx").cast(pl.UInt32).alias("q"), "country")
        old = (pl.read_parquet(W.pruned("test")).join(ctry, on="q", how="left")
                 .filter(~pl.col("country").is_in(only)).drop("country"))
        log(f"prune test: re-pruned {only}, kept {old.height:,} pairs of the other countries")
        test = pl.concat([old.select(test.columns), test.select(test.columns).cast(old.select(test.columns).schema)])
    test.write_parquet(W.pruned("test"))
    log(f"prune test: {test.height:,} pairs kept ({test.height / max(test['q'].n_unique(), 1):.1f}/S1 with candidates)")
    write_candidate_pairs(cfg, W, test)


def write_candidate_pairs(cfg, W: Work, pruned_test: pl.DataFrame) -> None:
    rec = pl.read_parquet(W.records("test"), columns=["idx", "id", "src"])
    ids = rec.select("idx", "id")
    pairs = (pruned_test.join(ids.rename({"idx": "q", "id": "s1_id"}), on="q")
                        .join(ids.rename({"idx": "r", "id": "rec_id"}), on="r")
                        .sort(["q", "prune_p"], descending=[False, True]))
    s1_ids = rec.filter(pl.col("src") == 1)["id"]
    io.write_id_lists(f"{cfg.paths.output_dir}/candidate_pairs.tsv", s1_ids, pairs, io.CAND_HEADER)
