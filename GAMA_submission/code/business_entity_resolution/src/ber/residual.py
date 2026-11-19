"""Step `residual`: boosted residual ensemble over saved stack runs (CPU).

The current model (`residual.base`, a stack run) stays fixed. A small, strongly
regularised LightGBM starts from its logit (init_score) and only learns corrections,
from what that model does not see:

  views         the other boosted stack runs of the same pairs (`residual.members`,
                e.g. without the cross-encoder, lexical only): level-1 and final logits
  disagreement  each view minus the base, range and mean of all views
  competition   the base probability's standing among the S1's candidates and among
                the S1s that want the same record (collective.competition)

When the members add nothing, the trees stay near zero and the base comes through
unchanged. Cross-fitted on the stacks' S1 folds and training worlds (all inputs are
out-of-fold), validated per world, decision rule re-tuned on the held-out stress
world; used only when it beats the base there by `residual.min_gain`.

    python -m ber residual --split train   reads work/<base|members>/oof_worlds.parquet
    python -m ber residual --split test    reads work/<base|members>/test_scores.parquet
    python -m ber residual                  both

Outputs: work/<residual.dir>/{report.json, decision.json, models/, test_scores.parquet},
         output/matching_results.tsv
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from . import collective, decide, io
from .util import Work, as_list, die, load_json, log, n_threads, save_json

EPS = 1e-6
COMP = ["c_r_n", "c_r_sum", "c_r_rank", "c_r_n05", "c_r_n03", "c_r_rival", "c_r_margin", "c_r_share",
        "c_q_n", "c_q_sum", "c_q_rank", "c_q_n05", "c_q_gap", "c_q_rival", "c_q_src_n05", "c_q_src_rank"]


def _logit(c: str) -> pl.Expr:
    p = pl.col(c).clip(EPS, 1 - EPS)
    return (p / (1 - p)).log().cast(pl.Float32)


def _views(W: Work, base: str, members: list[str], test: bool) -> pl.DataFrame:
    fname = "test_scores.parquet" if test else "oof_worlds.parquet"
    key = ["q", "r"] if test else ["world", "q", "r"]
    path = W.p(base, fname)
    if not path.exists():
        die(f"residual base {base}: {path} missing - run `stack` with stack.dir={base} first")
    df = pl.read_parquet(path, columns=key + ["p", "p1"] + ([] if test else ["label"]))
    for m in members:
        mp = W.p(m, fname)
        if not mp.exists():
            die(f"residual member {m}: {mp} missing - run `stack` with stack.dir={m} first")
        f = pl.read_parquet(mp, columns=key + ["p", "p1"]).rename({"p": f"p_{m}", "p1": f"p1_{m}"})
        df = df.join(f, on=key, how="left")
    return df


def feature_names(members: list[str]) -> list[str]:
    return (["z", "z1"] + [f"z_{m}" for m in members] + [f"z1_{m}" for m in members]
            + [f"dz_{m}" for m in members] + ["v_range", "v_mean"] + COMP)


def _features(df: pl.DataFrame, members: list[str], r_src: pl.DataFrame) -> pl.DataFrame:
    """One world (or the test set): the base logit z, the members' views, competition."""
    df = df.with_columns(_logit("p").alias("z"), _logit("p1").alias("z1"),
                         *[_logit(f"p_{m}").alias(f"z_{m}") for m in members],
                         *[_logit(f"p1_{m}").alias(f"z1_{m}") for m in members])
    views = ["z"] + [f"z_{m}" for m in members]
    df = df.with_columns(*[(pl.col(f"z_{m}") - pl.col("z")).alias(f"dz_{m}") for m in members],
                         (pl.max_horizontal(views) - pl.min_horizontal(views)).alias("v_range"),
                         pl.mean_horizontal(views).alias("v_mean"))
    return collective.competition(df.join(r_src, on="r", how="left"), p="p")


def _fit(cfg, X, y, z0, Xv, yv, zv, names):
    rc = cfg.residual
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None
    if lgb is None or cfg.gbdt.backend != "lightgbm":
        # smoke-test fallback: the base logit is a feature here, not an offset
        from sklearn.ensemble import HistGradientBoostingClassifier
        log("WARNING: lightgbm unavailable - residual fallback without init_score")
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=cfg.seed)
        clf.fit(np.vstack([X, Xv]), np.concatenate([y, yv]))
        return ("sklearn", clf)
    params = dict(objective="binary", learning_rate=rc.learning_rate, num_leaves=rc.num_leaves,
                  min_data_in_leaf=rc.min_data_in_leaf, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=rc.lambda_l2, num_threads=n_threads(cfg.gbdt.n_threads), verbose=-1, seed=cfg.seed)
    dtr = lgb.Dataset(X, y, init_score=z0, feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(Xv, yv, init_score=zv, reference=dtr)
    b = lgb.train(params, dtr, num_boost_round=rc.rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(rc.early_stopping, verbose=False), lgb.log_evaluation(200)])
    log(f"  residual lightgbm best_iteration={b.best_iteration}")
    return ("lightgbm", b)


def _predict(models: list, X: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """Average of the fold models: in logit space for the residual boosters."""
    if models[0][0] == "lightgbm":
        g = np.mean([b.predict(X, raw_score=True, num_iteration=b.best_iteration or None) for _, b in models], axis=0)
        return (1.0 / (1.0 + np.exp(-(z0 + g)))).astype(np.float32)
    return np.mean([m.predict_proba(X)[:, 1] for _, m in models], axis=0).astype(np.float32)


def _save(models: list, d: Path, meta: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for k, (kind, obj) in enumerate(models):
        if kind == "lightgbm":
            obj.save_model(str(d / f"resid_fold{k}.txt"))
        else:
            (d / f"resid_fold{k}.pkl").write_bytes(pickle.dumps(obj))
    (d / "resid.json").write_text(json.dumps(meta), encoding="utf-8")


def _load(d: Path) -> tuple[list, dict]:
    meta = json.loads((d / "resid.json").read_text(encoding="utf-8"))
    models = []
    for k in range(meta["folds"]):
        if (d / f"resid_fold{k}.txt").exists():
            import lightgbm as lgb
            models.append(("lightgbm", lgb.Booster(model_file=str(d / f"resid_fold{k}.txt"))))
        else:
            models.append(("sklearn", pickle.loads((d / f"resid_fold{k}.pkl").read_bytes())))
    return models, meta


def run(cfg, args) -> None:
    W = Work(cfg)
    rc, dc = cfg.residual, cfg.decide
    only = getattr(args, "split", None)
    rdir = str(rc.get("dir", "residual") or "residual")
    if only != "test":
        train(cfg, W, rdir)
    if only != "train":
        predict(cfg, W, rdir)


def train(cfg, W: Work, rdir: str) -> None:
    from .stack import Worlds, report_f
    rc, dc = cfg.residual, cfg.decide
    base, members = str(rc.base), as_list(rc.get("members"))
    K = int(cfg.data.n_folds_gbdt)
    rec = pl.read_parquet(W.records("train"), columns=["idx", "src", "fold_gb"])
    r_src = rec.select(pl.col("idx").alias("r"), pl.col("src").alias("r_src"))
    folds = rec.filter(pl.col("src") == 1).select(pl.col("idx").alias("q"), "fold_gb")
    views = _views(W, base, members, test=False)
    worlds_present = sorted(views["world"].unique().to_list())
    train_worlds = [int(s) for s in cfg.stack.get("worlds", [0]) if int(s) in worlds_present]
    eval_world = int(cfg.stack.get("eval_world", 5) or 0)
    names = feature_names(members)
    log(f"residual: base {base}, members {members}, {len(names)} features, worlds {worlds_present} "
        f"(train {train_worlds}, tune on {eval_world if eval_world in worlds_present else 0})")
    frames = {s: _features(views.filter(pl.col("world") == s), members, r_src).join(folds, on="q", how="left")
              for s in worlds_present}
    tr_all = pl.concat([frames[s] for s in train_worlds])
    models, oof = [], {s: np.full(frames[s].height, np.nan, dtype=np.float32) for s in worlds_present}
    for k in range(K):
        tr = tr_all.filter(pl.col("fold_gb") != k)
        es = ((tr["q"].hash(cfg.seed + 1) % 10) == 0).to_numpy()
        X, y, z = tr.select(names).to_numpy(), tr["label"].to_numpy(), tr["z"].to_numpy()
        log(f"residual fold {k}: fit on {int((~es).sum()):,} pairs, early-stop on {int(es.sum()):,}")
        m = _fit(cfg, X[~es], y[~es], z[~es], X[es], y[es], z[es], names)
        del X, tr
        models.append(m)
        for s in worlds_present:
            idx = np.flatnonzero(frames[s]["fold_gb"].to_numpy() == k)
            if idx.size:
                f = frames[s][idx]
                oof[s][idx] = _predict([m], f.select(names).to_numpy(), f["z"].to_numpy())
    if models[0][0] == "lightgbm":
        imp = dict(zip(models[0][1].feature_name(), models[0][1].feature_importance("gain")))
        log("residual top features (gain): " + ", ".join(f"{a}={b:.0f}" for a, b in sorted(imp.items(), key=lambda x: -x[1])[:12]))

    gt = pl.read_parquet(W.gt())
    wl = Worlds(cfg, W, pl.DataFrame())
    tune_world = eval_world if eval_world in worlds_present else 0
    cand = {"base": {s: frames[s].select("q", "r", "p") for s in worlds_present},
            "residual": {s: frames[s].select("q", "r").with_columns(pl.Series("p", oof[s])) for s in worlds_present}}
    tuned = {n: decide.tune_worlds(cfg, [(P[tune_world], wl.kept_s1(tune_world))], gt) for n, P in cand.items()}
    gain = tuned["residual"][1][0][0] - tuned["base"][1][0][0]
    use = "residual" if gain >= float(rc.get("min_gain", 0.0001)) else "base"
    best = tuned[use][0]
    log(f"residual: tuned F on world {tune_world}: base {tuned['base'][1][0][0]:.5f}, "
        f"residual {tuned['residual'][1][0][0]:.5f} (gain {gain:+.5f}) -> using {use}")
    s1 = wl.s1.select("idx", "id", "country")
    rep = {"use": use, "base": base, "members": members, "tuned_world": tune_world, "gain": round(gain, 6),
           "decision": best, "worlds": {}}
    for s in worlds_present:
        qs = s1.join(pl.DataFrame({"idx": wl.kept_s1(s)}), on="idx", how="semi")
        rep["worlds"][str(s)] = {n: report_f(decide.select(cand[n][s], tuned[n][0], dc.beta, dc.mc_samples, cfg.seed),
                                             gt, qs, dc.beta) for n in cand}
        r = rep["worlds"][str(s)]
        log(f"VALIDATION world {s}: base {r['base']['macro_f']:.5f} | residual {r['residual']['macro_f']:.5f} | "
            f"residual {r['residual']}")
    d = W.p(rdir, "models", "x").parent
    _save(models, d, {"folds": K, "features": names, "base": base, "members": members})
    save_json({"params": best, "use": use, "base": base, "members": members,
               "tuned_f": {n: v[1][0][0] for n, v in tuned.items()}}, W.p(rdir, "decision.json"))
    save_json(rep, W.p(rdir, "report.json"))


def predict(cfg, W: Work, rdir: str) -> None:
    dc = cfg.decide
    dec = load_json(W.p(rdir, "decision.json"))
    base, members = dec["base"], dec["members"]
    rec = pl.read_parquet(W.records("test"), columns=["idx", "id", "src"])
    r_src = rec.select(pl.col("idx").alias("r"), pl.col("src").alias("r_src"))
    f = _features(_views(W, base, members, test=True), members, r_src)
    if dec["use"] == "residual":
        models, meta = _load(W.p(rdir, "models", "x").parent)
        p = _predict(models, f.select(meta["features"]).to_numpy(), f["z"].to_numpy())
        log(f"residual test: {f.height:,} pairs, base {base} + residual over {members}")
    else:
        p = f["p"].to_numpy().astype(np.float32)
        log(f"residual test: the residual model did not beat {base} - its scores are written unchanged")
    scores = f.select("q", "r").with_columns(pl.Series("p", p))
    scores.write_parquet(W.p(rdir, "test_scores.parquet"))
    sel = decide.select(scores, dec["params"], dc.beta, dc.mc_samples, cfg.seed)
    ids = rec.select("idx", "id")
    pairs = (sel.join(ids.rename({"idx": "q", "id": "s1_id"}), on="q")
                .join(ids.rename({"idx": "r", "id": "rec_id"}), on="r").sort("q"))
    io.write_id_lists(f"{cfg.paths.output_dir}/matching_results.tsv", rec.filter(pl.col("src") == 1)["id"],
                      pairs, io.MATCH_HEADER)
    if not Path(f"{cfg.paths.output_dir}/candidate_pairs.tsv").exists():
        from .prune import write_candidate_pairs
        write_candidate_pairs(cfg, W, pl.read_parquet(W.pruned("test")))
    log(f"test: {sel.height:,} matches for {sel['q'].n_unique():,} of {rec.filter(pl.col('src') == 1).height:,} S1")
    from .submit import validate
    validate(cfg)
