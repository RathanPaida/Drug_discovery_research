"""Step `stack`: final GBDTs + decision tuning + validation + matching_results.tsv (CPU).

Level 1: all candidate features, the pruner score, the cross-encoder logit
(out-of-fold for train) and cross-encoder context (rank / gap within the S1 and
among the S1s competing for the same record). Works without cross-encoder scores
too (baseline run).

Level 2 (`stack.second_stage`): the level-1 probability plus collective features
(collective.py): competition among candidates, house-number drift, coherence with
the S1's best other record, namesake arbitration, record twins.

Worlds. The test set looks like train with ~19% of the S1 owners removed (5.75
S2/S3 records per S1 against 4.68): their records stay behind as look-alike
distractors. A stress world builds that on train - drop `decide.stress_drop_s1` of
the S1s (a salt picks which), keep their records - and recomputes everything that
depends on which S1s exist: record-side retrieval context (r_nq, dense rank/gap
over the record's S1s), S1 name frequencies, the pruner score (saved pruner
models, `stack.reprune`) and the cross-encoder / collective context. Candidate
pairs the full world had pruned away are not re-added (they have no
cross-encoder score). `stack.worlds` lists the worlds the stackers train on
(0 = the full train set, k > 0 = stress world with salt k); `stack.eval_world` is
a stress world held out for validation; `stack.tune_on` picks the world(s) the
decision rule is tuned on.

Test (`stack.per_fold_test`): each cross-encoder fold model's scores go through
the stackers on their own and the probabilities are averaged, so the stackers see
the same single-model logits as in training (not a two-model average).

Validation (all out-of-fold, all train S1s including those with no candidates):
macro F-beta overall / per country / singletons / matched S1s, per world.

Outputs: output/matching_results.tsv, work/stack/{report.json, decision.json,
         oof.parquet, oof_worlds.parquet, test_scores.parquet}
"""
from __future__ import annotations

import fnmatch
import glob
from pathlib import Path

import numpy as np
import polars as pl

from . import collective, decide, gbdt, io
from .features import feature_columns, parts_glob
from .metrics import per_query_fbeta
from .util import Work, as_list, die, load_json, log, n_threads, save_json, stable_bucket, timed

CE_CTX = ["ce_rank_q", "ce_gap_q", "ce_npos_q", "ce_rank_r", "ce_gap_r"]


def sdir(cfg) -> str:
    """Work sub-folder of this stack run (`stack.dir`): variants can run side by side."""
    return str(cfg.stack.get("dir", "stack") or "stack")


# -- inputs ----------------------------------------------------------------
def load_ce(W: Work, split: str, per_fold: bool = False) -> pl.DataFrame | dict[int, pl.DataFrame] | None:
    """Cross-encoder logits. per_fold=False: one score per pair (the out-of-fold
    score on train, the mean over fold models on test); True: {fold: scores}."""
    files = sorted(glob.glob(str(W.p("ce", f"{split}_ce_f*_s*.parquet"))))
    if not files:
        log(f"WARNING: no cross-encoder scores for {split} - stacking without them")
        return None
    by_fold: dict[int, list[pl.DataFrame]] = {}
    for f in files:
        fold = int(f.rsplit("_f", 1)[1].split("_s")[0])
        by_fold.setdefault(fold, []).append(pl.read_parquet(f, columns=["q", "r", "ce_logit"]))
    folds = {k: pl.concat(v).unique(["q", "r"]) for k, v in sorted(by_fold.items())}
    if per_fold:
        return folds
    ce = pl.concat([v.with_columns(pl.lit(k, pl.Int8).alias("_fold")) for k, v in folds.items()])
    return ce.group_by("q", "r").agg(pl.col("ce_logit").mean())


def ce_context(df: pl.DataFrame) -> pl.DataFrame:
    if "ce_logit" not in df.columns:
        return df
    c = pl.col("ce_logit")
    return df.drop([x for x in CE_CTX if x in df.columns]).with_columns(
        c.rank("ordinal", descending=True).over("q").cast(pl.Float32).alias("ce_rank_q"),
        (c.max().over("q") - c).alias("ce_gap_q"),
        (c > 0).sum().over("q").cast(pl.Float32).alias("ce_npos_q"),
        c.rank("ordinal", descending=True).over("r").cast(pl.Float32).alias("ce_rank_r"),
        (c.max().over("r") - c).alias("ce_gap_r"),
    )


def assemble(W: Work, split: str, with_ce: bool = True) -> pl.DataFrame:
    """Pruned pairs + their candidate features (+ the one-per-pair cross-encoder score)."""
    pr = pl.read_parquet(W.pruned(split)).select("q", "r", "prune_p", pl.col("prune_rank").cast(pl.Float32))
    # part by part: each part holds whole S1s, so it only meets its own slice of the
    # pruned pairs, and the full candidate table (~190M rows) never sits in memory
    kept = []
    for path in sorted(glob.glob(parts_glob(W, split))):
        part = pl.read_parquet(path)
        lo, hi = part["q"].min(), part["q"].max()
        kept.append(part.join(pr.filter(pl.col("q").is_between(lo, hi)), on=["q", "r"], how="inner"))
        del part
    if not kept:
        raise FileNotFoundError(f"no candidate parts at {parts_glob(W, split)} - run the features step first")
    feats = pl.concat(kept, how="vertical_relaxed")
    if with_ce:
        ce = load_ce(W, split)
        if ce is not None:
            feats = feats.join(ce.select("q", "r", "ce_logit"), on=["q", "r"], how="left")
        return ce_context(feats)
    return feats


def report_f(sel: pl.DataFrame, gt: pl.DataFrame, s1: pl.DataFrame, beta: float) -> dict:
    per = per_query_fbeta(sel, gt, s1["idx"], beta).join(
        s1.select(pl.col("idx").alias("q"), "country"), on="q")
    out = {"macro_f": round(float(per["f"].mean()), 5),
           "singletons": round(float(per.filter(pl.col("nt") == 0)["f"].mean() or 0), 5),
           "matched_s1": round(float(per.filter(pl.col("nt") > 0)["f"].mean() or 0), 5),
           "n_s1": per.height}
    for (c,), g in per.group_by("country"):
        out[f"country[{c}]"] = round(float(g["f"].mean()), 5)
    return out


# -- worlds ------------------------------------------------------------------
class Worlds:
    """Train pairs as seen in the full world (salt 0) or in stress worlds."""

    def __init__(self, cfg, W: Work, train: pl.DataFrame):
        self.cfg, self.W, self.train = cfg, W, train
        rec = pl.read_parquet(W.records("train"), columns=["idx", "id", "src", "country", "n_core_str"])
        self.rec = rec
        self.s1 = rec.filter(pl.col("src") == 1).select("idx", "id", "country", "n_core_str")
        self._union: pl.DataFrame | None = None
        self._prune: tuple[list, list[str]] | None = None
        self._cache: dict[int, pl.DataFrame] = {}

    def kept_s1(self, salt: int) -> pl.Series:
        if salt == 0:
            return self.s1["idx"]
        share = float(self.cfg.decide.stress_drop_s1)
        drop = stable_bucket(self.s1["id"], 10_000, salt=salt) < int(share * 10_000)
        return self.s1.filter(pl.Series(~drop))["idx"]

    def union(self) -> pl.DataFrame:
        """(q, r, dense_sim) of every retrieved train pair (the features parts)."""
        if self._union is None:
            with timed("stack: loading the train candidate union for world context"):
                cols = ["q", "r", "dense_sim"]
                self._union = pl.concat([pl.read_parquet(p, columns=cols)
                                         for p in sorted(glob.glob(parts_glob(self.W, "train")))])
        return self._union

    def pruner(self) -> tuple[list, list[str]]:
        """The saved pruner fold models + their feature list (the parts' columns, as in prune)."""
        if self._prune is None:
            feats = feature_columns(pl.scan_parquet(parts_glob(self.W, "train")))
            models = gbdt.load_fold_models(self.W.p("prune", "models", "x").parent, "prune",
                                           int(self.cfg.data.n_folds_gbdt), feats)
            self._prune = (models, feats)
        return self._prune

    def get(self, salt: int) -> pl.DataFrame:
        if salt not in self._cache:
            self._cache[salt] = self._build(salt)
        return self._cache[salt]

    def _build(self, salt: int) -> pl.DataFrame:
        if salt == 0:
            return self.train.with_columns(pl.lit(0, pl.Int16).alias("world"))
        kept = self.kept_s1(salt)
        kq = pl.DataFrame({"q": kept})
        df = self.train.join(kq, on="q", how="semi")
        # record-side retrieval context over the S1s that still exist
        u = self.union().join(kq, on="q", how="semi")
        u = u.with_columns(
            pl.len().over("r").cast(pl.Float32).alias("r_nq"),
            pl.col("dense_sim").rank("ordinal", descending=True).over("r").cast(pl.Float32).alias("dense_rank_r"),
            (pl.col("dense_sim").max().over("r") - pl.col("dense_sim")).cast(pl.Float32).alias("dense_gap_r"),
        ).select("q", "r", "r_nq", "dense_rank_r", "dense_gap_r")
        ctx = [c for c in ("r_nq", "dense_rank_r", "dense_gap_r") if c in df.columns]
        df = df.drop(ctx).join(u.select(["q", "r"] + ctx), on=["q", "r"], how="left")
        # S1 name frequencies (log1p, as in features) over the S1s that still exist
        s1k = self.s1.join(kq.rename({"q": "idx"}), on="idx", how="semi")
        cnt = s1k.group_by("country", "n_core_str").len("_c")
        names = self.rec.select("idx", "country", "n_core_str")
        for side in ("q", "r"):
            col = f"{side}_nf_s1"
            if col not in df.columns:
                continue
            f = (df.select(side).unique().join(names.rename({"idx": side}), on=side, how="left")
                   .join(cnt, on=["country", "n_core_str"], how="left")
                   .select(side, pl.col("_c").fill_null(0).cast(pl.Float32).log1p().alias(col)))
            df = df.drop(col).join(f, on=side, how="left")
        # pruner score with the recomputed context (out-of-fold: model k scores fold k)
        if self.cfg.stack.get("reprune", True) and "prune_p" in df.columns:
            try:
                models, pf = self.pruner()
                p = gbdt.predict_by_fold(models, df.select(pf).to_numpy(), df["fold_gb"].to_numpy())
                df = df.with_columns(pl.Series("prune_p", p)).with_columns(
                    pl.col("prune_p").rank("ordinal", descending=True).over("q").cast(pl.Float32).alias("prune_rank"))
            except (FileNotFoundError, ImportError) as e:
                log(f"WARNING: stress world {salt}: pruner scores kept from the full world ({e})")
        df = ce_context(df)
        log(f"stress world {salt}: {kq.height:,} of {self.s1.height:,} S1s, {df.height:,} pairs")
        return df.with_columns(pl.lit(salt, pl.Int16).alias("world"))


# -- level 2 -------------------------------------------------------------------
class Level2:
    """Collective features for one split: record keys, record graph and the
    score-independent pair features are computed once for all its pairs."""

    def __init__(self, W: Work, split: str, pairs: pl.DataFrame, nt: int):
        recs = pl.read_parquet(W.records(split), columns=collective.REC_COLS)
        self.keys = collective.record_keys(recs)
        del recs
        self.static = collective.static_pair_features(pairs.select("q", "r"), self.keys,
                                                      collective.graph_counts(self.keys))
        self.nt = nt

    def features(self, df: pl.DataFrame, world_s1: pl.Series, p: str = "p1") -> pl.DataFrame:
        return collective.add_features(df, self.static, self.keys, world_s1, p, self.nt)


# -- ensemble members ----------------------------------------------------------
def member_feature_names(members: list[str]) -> list[str]:
    return [f"m_{m}_logit" for m in members] + ["m_logit_range", "m_logit_mean"] if members else []


def add_members(df: pl.DataFrame, frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Other stack runs as ensemble members: their final out-of-fold probability for the
    same (q, r) as a logit, plus how far apart all views are (this run's level 1 included).
    Members see the same pairs through different evidence (e.g. no cross-encoder, or
    lexical features only); where they disagree is where a single view is least reliable."""
    if not frames:
        return df
    eps = 1e-6
    cols = []
    for name, f in frames.items():
        c = f"m_{name}_logit"
        p = pl.col("p").clip(eps, 1 - eps)
        df = df.join(f.select("q", "r", (p / (1 - p)).log().cast(pl.Float32).alias(c)), on=["q", "r"], how="left")
        cols.append(c)
    views = ["p1_logit"] + cols
    return df.with_columns((pl.max_horizontal(views) - pl.min_horizontal(views)).cast(pl.Float32).alias("m_logit_range"),
                           pl.mean_horizontal(views).cast(pl.Float32).alias("m_logit_mean"))


def member_frames(W: Work, members: list[str], world: int | None) -> dict[str, pl.DataFrame]:
    """world None: the members' test scores; else their out-of-fold scores in that world."""
    out = {}
    for m in members:
        if world is None:
            path = W.p(m, "test_scores.parquet")
            if not path.exists():
                die(f"ensemble member {m}: {path} missing - run `stack --split test stack.dir={m}` first")
            out[m] = pl.read_parquet(path, columns=["q", "r", "p"])
        else:
            path = W.p(m, "oof_worlds.parquet")
            if not path.exists():
                die(f"ensemble member {m}: {path} missing - run `stack --split train stack.dir={m}` first")
            f = pl.read_parquet(path, columns=["world", "q", "r", "p"]).filter(pl.col("world") == world)
            if f.height == 0:
                die(f"ensemble member {m} has no out-of-fold scores for world {world} - train it with the same stack.worlds")
            out[m] = f.drop("world")
    return out


# -- step --------------------------------------------------------------------
def _p_by_fold(models, df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    return gbdt.predict_by_fold(models, df.select(feats).to_numpy(), df["fold_gb"].to_numpy())


def run(cfg, args) -> None:
    """--split train: fit + validate + save the stackers and the decision rule only;
    --split test: predict the test set with what a previous run saved; no split: both."""
    W = Work(cfg)
    dc, sc = cfg.decide, cfg.stack
    nt = n_threads(cfg.features.n_threads)
    only = getattr(args, "split", None)
    if only == "test":
        dec = load_json(W.p(sdir(cfg), "decision.json"))
        best = dec["params"]
        models1, models2 = load_stackers(cfg, W, level2=dec.get("level") == "level2")
        log(f"stack test only: {len(models1)} level-1 models, "
            f"{len(models2) if models2 else 0} level-2 models, decision {best}")
        write_test(cfg, W, models1, models2, best, nt, dec.get("members", []) if models2 else [])
        return
    train = assemble(W, "train")
    feats1 = feature_columns(train, extra_meta=("world",))
    drop = as_list(sc.get("drop_features"))        # e.g. ce_*,dense_*,prune_* : a model without the neural parts
    if drop:
        feats1 = [f for f in feats1 if not any(fnmatch.fnmatchcase(f, pat) for pat in drop)]
        log(f"stack: dropping features matching {drop}")
    train_worlds = [int(s) for s in sc.get("worlds", [0])]
    eval_world = int(sc.get("eval_world", 5) or 0)
    report_worlds = sorted({0, *train_worlds, *([eval_world] if eval_world else [])})
    worlds = Worlds(cfg, W, train)
    log(f"stack: {train.height:,} train pairs, {len(feats1)} level-1 features, "
        f"train worlds {train_worlds}, validation worlds {report_worlds}")
    models_dir = W.p(sdir(cfg), "models", "x").parent

    # level 1: cross-fit on the training worlds, out-of-fold everywhere else
    frame = pl.concat([worlds.get(s) for s in train_worlds], how="diagonal_relaxed")
    oof, models1 = gbdt.cross_fit(cfg, frame.lazy(), feats1, sc.train_s1_frac, models_dir, "stack",
                                  id_cols=("world", "q", "r"))
    del frame
    p1 = {s: oof.filter(pl.col("world") == s).select("q", "r", pl.col("p").alias("p1")) for s in train_worlds}
    for s in report_worlds:
        if s not in p1:
            w = worlds.get(s)
            p1[s] = w.select("q", "r").with_columns(pl.Series("p1", _p_by_fold(models1, w, feats1)))
    worlds._union = None                           # every world is built: the ~190M-row union can go

    # level 2: collective features from the level-1 probabilities of each world
    final = {s: p1[s].rename({"p1": "p"}) for s in report_worlds}
    models2, feats2 = None, None
    members = as_list(sc.get("members"))
    if members and not sc.get("second_stage", True):
        die("stack.members needs stack.second_stage=true (members feed the level-2 stacker)")
    if sc.get("second_stage", True):
        lvl2 = Level2(W, "train", train, nt)
        feats2 = feats1 + collective.FEATURES + member_feature_names(members)
        if members:
            log(f"stack level 2: ensemble members {members}")

        def f2_of(s: int) -> pl.DataFrame:
            f = lvl2.features(worlds.get(s).join(p1[s], on=["q", "r"]), worlds.kept_s1(s))
            return add_members(f, member_frames(W, members, s))

        with timed("stack level 2: collective features for the training worlds"):
            frame2 = pl.concat([f2_of(s) for s in train_worlds], how="diagonal_relaxed")
        for s in train_worlds:                     # their level-1 frames are not needed any more
            worlds._cache.pop(s, None)
        oof2, models2 = gbdt.cross_fit(cfg, frame2.lazy(), feats2, sc.train_s1_frac, models_dir, "stack2",
                                       id_cols=("world", "q", "r"))
        del frame2
        for s in report_worlds:
            if s in train_worlds:
                final[s] = oof2.filter(pl.col("world") == s).select("q", "r", "p")
            else:
                f2 = f2_of(s)
                final[s] = f2.select("q", "r").with_columns(pl.Series("p", _p_by_fold(models2, f2, feats2)))
                del f2
        del lvl2

    # decision rule, tuned on the chosen world(s) - for each level; level 2 is only
    # used when it beats level 1 there (a second stacker can also overfit)
    gt = pl.read_parquet(W.gt())
    s1 = worlds.s1.select("idx", "id", "country")
    tune_on = str(sc.get("tune_on", "stress"))
    tune_set = {"normal": [0], "stress": [eval_world or 0], "blend": [0] + ([eval_world] if eval_world else [])}[tune_on]
    levels = {"level1": {s: p1[s].rename({"p1": "p"}) for s in report_worlds}}
    if models2 is not None:
        levels["level2"] = final
    tuned = {}
    for name, P in levels.items():
        log(f"decision rule for {name} (tuned on worlds {tune_set}):")
        tuned[name] = decide.tune_worlds(cfg, [(P[s], worlds.kept_s1(s)) for s in tune_set], gt)
    use = "level2" if "level2" in tuned and tuned["level2"][1][0][0] > tuned["level1"][1][0][0] else "level1"
    best, grid = tuned[use]
    if use == "level1":
        final, models2 = levels["level1"], None
    log(f"stack: using {use} (tuned F: " + ", ".join(f"{k} {v[1][0][0]:.5f}" for k, v in tuned.items()) + ")")
    save_json({"params": best, "level": use, "tuned_on_worlds": tune_set,
               "members": members if use == "level2" else [],
               "tuned_f": {k: v[1][0][0] for k, v in tuned.items()},
               "grid": [{"f": f, **p} for f, p in grid]}, W.p(sdir(cfg), "decision.json"))

    # out-of-fold scores per world (p = the level in use, p1 = level 1) + the classic oof.parquet
    lab = train.select("q", "r", "label")
    rows = [final[s].join(p1[s], on=["q", "r"]).join(lab, on=["q", "r"]).with_columns(pl.lit(s, pl.Int16).alias("world"))
            for s in report_worlds]
    oof_all = pl.concat(rows, how="diagonal_relaxed")
    oof_all.write_parquet(W.p(sdir(cfg), "oof_worlds.parquet"))
    oof_all.filter(pl.col("world") == 0).select("q", "r", "p", "label").write_parquet(W.p(sdir(cfg), "oof.parquet"))

    rep: dict = {"decision": best, "level": use, "tuned_on_worlds": tune_set, "train_worlds": train_worlds, "worlds": {}}
    for s in report_worlds:
        qs = s1.join(pl.DataFrame({"idx": worlds.kept_s1(s)}), on="idx", how="semi")
        r = {name: report_f(decide.select(levels[name][s], tuned[name][0], dc.beta, dc.mc_samples, cfg.seed),
                            gt, qs, dc.beta) for name in levels}
        r["final"] = r[use]
        r["dropped_s1_share"] = 0.0 if s == 0 else float(dc.stress_drop_s1)
        rep["worlds"][str(s)] = r
    # the two headline numbers keep their old names
    rep["oof"] = rep["worlds"]["0"]["final"]
    if eval_world:
        rep["stress"] = rep["worlds"][str(eval_world)]["final"]
    save_json(rep, W.p(sdir(cfg), "report.json"))
    for s in report_worlds:
        r = rep["worlds"][str(s)]
        both = "  (" + ", ".join(f"{k} {r[k]['macro_f']:.5f}" for k in levels) + ")" if len(levels) > 1 else ""
        log(f"VALIDATION world {s} ({'full train' if s == 0 else 'stress'}) macro F{dc.beta}: "
            f"{r['final']['macro_f']:.5f}{both} | {r['final']}")
    del train, worlds, oof, oof_all, rows, p1, final
    if only == "train":
        log("stack --split train: stackers + decision saved; `stack --split test` writes the submission")
        return
    write_test(cfg, W, models1, models2, best, nt, members if models2 is not None else [])


def load_stackers(cfg, W: Work, level2: bool) -> tuple[list, list | None]:
    """The fold models a previous `stack` run saved (sidecars carry the feature lists)."""
    d, K = W.p(sdir(cfg), "models", "x").parent, int(cfg.data.n_folds_gbdt)
    hint: list[str] = []
    if not (d / "stack_fold0.json").exists():      # older run: rebuild the level-1 column list
        base = assemble(W, "test", with_ce=False)
        hint = feature_columns(base, extra_meta=("world",))
        if load_ce(W, "test", per_fold=True):
            hint += ["ce_logit"] + CE_CTX
        del base
    models1 = gbdt.load_fold_models(d, "stack", K, hint)
    models2 = None
    if level2:
        if not (d / "stack2_fold0.json").exists():
            die(f"decision.json asks for level 2 but {d} has no level-2 stackers - rerun `stack --split train`")
        models2 = gbdt.load_fold_models(d, "stack2", K, [])
    return models1, models2


def match_unseen_rate(cfg, W: Work, pairs: pl.DataFrame, best: dict) -> pl.DataFrame:
    """Countries without training labels (France here): shift their logits so that the
    decision gives them the same matches per S1 as the training countries get on the
    same test set - a base-rate (label-shift) correction; the models are less confident
    on text they never saw, which otherwise shows up as missing matches. pairs: q, r, p."""
    dc = cfg.decide
    seen = set(pl.read_parquet(W.records("train"), columns=["country"])["country"].unique().to_list())
    rec = pl.read_parquet(W.records("test"), columns=["idx", "src", "country"])
    s1 = rec.filter(pl.col("src") == 1).select(pl.col("idx").cast(pairs.schema["q"]).alias("q"), "country")
    df = pairs.join(s1, on="q", how="left")
    unseen = sorted(set(s1["country"].unique().to_list()) - seen)
    if not unseen:
        return pairs

    def shift(part: pl.DataFrame, delta: float) -> pl.DataFrame:
        p = np.clip(part["p"].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)
        return part.with_columns(pl.Series("p", (1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + delta)))).astype(np.float32)))

    def rate(part: pl.DataFrame, n_s1: int, delta: float = 0.0) -> float:
        return decide.select(shift(part, delta), best, dc.beta, dc.mc_samples, cfg.seed).height / max(n_s1, 1)

    known = df.filter(pl.col("country").is_in(list(seen)))
    target = rate(known.select("q", "r", "p"), s1.filter(pl.col("country").is_in(list(seen))).height)
    out, report = [known.select("q", "r", "p")], {"target_matches_per_s1": round(target, 4), "countries": {}}
    for c in unseen:
        part = df.filter(pl.col("country") == c).select("q", "r", "p")
        n = s1.filter(pl.col("country") == c).height
        lo, hi, before = -4.0, 4.0, rate(part, n)
        for _ in range(16):                        # matches per S1 grow with the shift: bisection
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if rate(part, n, mid) < target else (lo, mid)
        delta = (lo + hi) / 2
        part = shift(part, delta)
        after = rate(part, n)
        report["countries"][c] = {"logit_shift": round(delta, 4), "matches_per_s1_before": round(before, 4),
                                  "matches_per_s1_after": round(after, 4), "s1": n}
        log(f"unseen country {c}: logit shift {delta:+.3f}, matches per S1 {before:.3f} -> {after:.3f} "
            f"(training countries: {target:.3f})")
        out.append(part)
    rest = df.filter(pl.col("country").is_null()).select("q", "r", "p")   # pairs whose S1 has no country
    save_json(report, W.p(sdir(cfg), "unseen_rate.json"))
    return pl.concat(out + [rest])


def write_test(cfg, W: Work, models1, models2, best: dict, nt: int, members: list[str] | None = None) -> None:
    dc = cfg.decide
    test_p = predict_test(cfg, W, models1, models2, nt, members or [])
    test_p.write_parquet(W.p(sdir(cfg), "test_scores.parquet"))
    pairs = test_p.select("q", "r", "p")
    if cfg.stack.get("unseen_rate_match", False):
        pairs = match_unseen_rate(cfg, W, pairs, best)
    sel_t = decide.select(pairs, best, dc.beta, dc.mc_samples, cfg.seed)
    trec = pl.read_parquet(W.records("test"), columns=["idx", "id", "src"])
    ids = trec.select("idx", "id")
    pairs = (sel_t.join(ids.rename({"idx": "q", "id": "s1_id"}), on="q")
                  .join(ids.rename({"idx": "r", "id": "rec_id"}), on="r").sort("q"))
    out = f"{cfg.paths.output_dir}/matching_results.tsv"
    io.write_id_lists(out, trec.filter(pl.col("src") == 1)["id"], pairs, io.MATCH_HEADER)
    log(f"test: {sel_t.height:,} matches for {sel_t['q'].n_unique():,} of {trec.filter(pl.col('src') == 1).height:,} S1")
    if not Path(f"{cfg.paths.output_dir}/candidate_pairs.tsv").exists():
        # a variant writing to its own output folder: the candidate set is the pruned test pairs
        from .prune import write_candidate_pairs
        write_candidate_pairs(cfg, W, pl.read_parquet(W.pruned("test")))

    from .submit import validate
    validate(cfg)


def predict_test(cfg, W: Work, models1, models2, nt: int, members: list[str] | None = None) -> pl.DataFrame:
    """q, r, p1 (level 1), p (final): per cross-encoder fold, then averaged."""
    sc = cfg.stack
    per_fold = bool(sc.get("per_fold_test", True))
    feats1 = models1[0].all_features
    feats2 = models2[0].all_features if models2 else None
    if not W.pruned("test").exists():
        raise FileNotFoundError(f"{W.pruned('test')} missing - run prune first")
    if per_fold:
        base = assemble(W, "test", with_ce=False)
        ce = load_ce(W, "test", per_fold=True)
        variants = list(ce.items()) if ce else [(None, None)]
    else:
        base = assemble(W, "test")
        variants = [(None, None)]
    test_s1 = pl.read_parquet(W.records("test"), columns=["idx", "src"]).filter(pl.col("src") == 1)["idx"]
    lvl2 = Level2(W, "test", base, nt) if models2 is not None else None
    mframes = member_frames(W, members or [], None) if models2 is not None else {}
    outs = []
    for fold, ce_f in variants:
        X = ce_context(base.join(ce_f, on=["q", "r"], how="left")) if ce_f is not None else base
        p1 = gbdt.predict_avg(models1, X.select(feats1).to_numpy())
        res = X.select("q", "r").with_columns(pl.Series("p1", p1))
        if lvl2 is not None:
            Xc = add_members(lvl2.features(X.with_columns(pl.Series("p1", p1)), test_s1), mframes)
            res = res.join(Xc.select("q", "r", pl.Series("p", gbdt.predict_avg(models2, Xc.select(feats2).to_numpy()))),
                           on=["q", "r"])
        else:
            res = res.with_columns(pl.col("p1").alias("p"))
        log(f"test: stacked {res.height:,} pairs" + (f" with cross-encoder fold {fold}" if fold is not None else ""))
        outs.append(res)
    return (pl.concat(outs).group_by("q", "r").agg(pl.col("p1").mean(), pl.col("p").mean())
              .with_columns(pl.col("p1").cast(pl.Float32), pl.col("p").cast(pl.Float32)))
