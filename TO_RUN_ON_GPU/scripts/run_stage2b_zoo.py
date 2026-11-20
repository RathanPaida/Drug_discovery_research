"""Stage 2b: benchmark the whole model zoo on TNK1, leakage-safe, with honest statistics.

    python scripts/run_stage2b_zoo.py --version v3 --tag core
    python scripts/run_stage2b_zoo.py --version v3 --models rf,svr_rbf,gp_minmax --repeats 10
    python scripts/run_stage2b_zoo.py --list

WHY THIS STAGE EXISTS
---------------------
`scripts/run_stage2_baselines.py` evaluates four hardcoded models on ONE representation
and reports a single held-out fold repeated over 5 seeds. `RUN_REPORT_V3_HPC.md` section
4.3 calls that out itself: "stage2_baselines.csv and stage4_uq_comparison.csv report
n = 45, a single held-out fold repeated over seeds. Only [the paired 5-fold critic
comparison] supports a claim about model ranking."

This stage replaces that with a proper sweep: every registered model x its representation,
nested cross-validation with hyperparameters chosen on INNER folds only, repeated CV for
statistical power, and paired tests against a pre-nominated incumbent.

THE THREE THINGS THAT MAKE THE NUMBER TRUSTWORTHY
-------------------------------------------------
1. **Nested CV.** Cawley & Talbot, "On Over-fitting in Model Selection and Subsequent
   Selection Bias in Performance Evaluation", JMLR 11:2079-2107 (2010), find that
   model-selection overfitting is "often of comparable magnitude to differences in
   performance between learning algorithms". At N=172 over a ~20-model sweep, that is the
   size of the 0.686-vs-0.501 Spearman gap this project is building conclusions on. If
   hyperparameters are chosen anywhere a test fold can see, the sweep manufactures its own
   winner.
2. **Repeated CV.** A single 5-fold partition of 172 molecules has large variance, and the
   exact Wilcoxon signed-rank test over 5 folds has a p-value FLOOR of 2/2^5 = 0.0625 --
   it cannot report p < 0.05 at all. `--repeats 10` gives 50 fold scores, which can.
3. **One pre-nominated incumbent.** With 20 models there are 190 pairwise comparisons and
   the best of 20 beats its own honest estimate by roughly the spread of the 20. This
   stage therefore tests every challenger against ONE incumbent (`--incumbent`, default
   `rf`, the strongest model this project has found) under Holm correction: 19 comparisons,
   not 190.

OUTPUTS
-------
    runs/stage2b_zoo_folds.csv     one row per (model, repeat, fold): metrics + params
    runs/stage2b_zoo_summary.csv   per model: mean +- sd over all folds, paired tests
    runs/stage2b_zoo_oof.csv       per-row out-of-fold predictions, per model
    runs/stage2b_zoo.json          environment, availability, verdict, provenance

`stage2b_zoo_oof.csv` is the input to Stage 4b (stacking), and it is produced here rather
than there so the meta-learner can only ever see predictions made by models that did not
see the row.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tnk1.zoo as Z  # noqa: E402
from tnk1.evaluation.stats import (corrected_resampled_ttest, holm,  # noqa: E402
                                   paired_seed_test)
from tnk1.zoo import benchmark as B, features as FZ  # noqa: E402
from tnk1.utils.common import load_config, read_table, seed_everything, verify_locked  # noqa: E402


def build_folds(splits, keys, repeat, seed):
    """Outer folds for one repeat.

    Repeat 0 uses the FROZEN splits verbatim, so this stage's first repeat is directly
    comparable with `runs/stage4_oof_predictions.csv` and with `compare_critic.py`.
    Later repeats re-draw scaffold-grouped folds with a different seed, which is where the
    statistical power comes from.
    """
    if repeat == 0:
        return B.folds_from_splits(splits, keys)
    from tnk1.data.splits import nested_scaffold_cv
    # Re-split on the same scaffold groups so the grouping property is preserved; only the
    # assignment of groups to folds changes.
    import pandas as _pd
    scaf = _pd.Series(keys).map(_SCAFFOLD_OF).values
    cv = nested_scaffold_cv(scaf, 5, 3, seed=seed + 1000 * repeat)
    out = []
    for f in cv:
        out.append(B.FoldSpec(
            f["outer"], np.asarray(f["train"]), np.asarray(f["test"]),
            [(np.asarray(i["train"]), np.asarray(i["val"])) for i in f["inner"]]))
    return out


_SCAFFOLD_OF: dict = {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--version", default="v3")
    ap.add_argument("--models", default=None,
                    help="comma-separated registry names; default: every model tagged --tag")
    ap.add_argument("--tag", default="core", help="shortlist tag (core|slow|probe|control)")
    ap.add_argument("--rep", default=None,
                    help="override every model's representation (for a representation sweep)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeated CV. 1 reuses the frozen splits only; use 10 for the "
                         "statistics to have power (see the module docstring)")
    ap.add_argument("--incumbent", default="rf",
                    help="the single model every challenger is tested against")
    ap.add_argument("--inner-metric", default="rmse", choices=["rmse", "mae", "spearman"])
    ap.add_argument("--no-tune", action="store_true", help="skip the inner-fold search")
    ap.add_argument("--scaler", default="robust", choices=["robust", "cdf"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="default: <runs>/stage2b_zoo")
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    a = ap.parse_args(argv)

    if a.list:
        print(f"{'model':24} {'family':10} {'rep':20} {'std':5} {'cost':>5}  tags / status")
        for n, s in sorted(Z.REGISTRY.items()):
            st = Z.available([n])[n]
            print(f"{n:24} {s.family:10} {s.rep:20} {str(s.gives_std):5} {s.cost:5.1f}  "
                  f"{','.join(s.tags)}  [{st}]")
        print(f"\nrepresentations ({len(FZ.REGISTRY)}):")
        for n, st in FZ.available().items():
            print(f"  {n:22} {FZ.REGISTRY[n].kind:8} {st}")
        return 0

    cfg = load_config(a.config)
    proc, runs = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["runs"])
    runs.mkdir(parents=True, exist_ok=True)
    stem = Path(a.out) if a.out else runs / "stage2b_zoo"

    # ---- the frozen table and splits, exactly as compare_critic.py uses them ----------
    tbl = proc / f"kinome_long_{a.version}.parquet"
    sha = verify_locked(tbl)
    df = read_table(tbl)
    splits = json.loads((proc / f"splits_{a.version}.json").read_text())
    tnk1 = cfg["targets"]["tnk1"]

    t = df[(df.uniprot_id == tnk1) & (df.censored == 0)].reset_index(drop=True)
    y = t.pact.values.astype(float)
    keys = t.inchikey_block1.values
    smiles = list(t.smiles)
    _SCAFFOLD_OF.update(dict(zip(keys, t.scaffold.values)))
    print(f"[zoo] TNK1 exact rows {len(t)} | unique compounds {len(set(keys))} | "
          f"dataset {sha[:12]}")

    # ---- which models ----------------------------------------------------------------
    names = ([m.strip() for m in a.models.split(",") if m.strip()] if a.models
             else Z.shortlist(a.tag))
    avail = Z.available(names)
    runnable = [n for n in names if avail[n] == "ok"]
    skipped = {n: avail[n] for n in names if avail[n] != "ok"}
    if skipped:
        print(f"[zoo] SKIPPING {len(skipped)} model(s) for missing dependencies:")
        for n, why in skipped.items():
            print(f"         {n:22} {why}")
    if not runnable:
        raise SystemExit("[zoo] no runnable model selected; `--list` shows availability")
    print(f"[zoo] {len(runnable)} model(s): {', '.join(runnable)}")

    # ---- featurise once, cache ------------------------------------------------------
    cache: dict[str, tuple] = {}

    def X_of(name):
        rep = a.rep or Z.REGISTRY[name].rep
        if rep not in cache:
            t0 = time.time()
            X, ok = FZ.featurize(rep, smiles)
            if not ok.all():
                raise SystemExit(
                    f"[zoo] representation {rep!r} failed on {int((~ok).sum())} of "
                    f"{len(ok)} TNK1 SMILES. These rows are in the FROZEN table, so this "
                    f"is a standardisation bug upstream, not a bad input -- fix Stage 1 "
                    f"rather than dropping them here.")
            cache[rep] = (X, FZ.dense_columns(rep))
            print(f"[zoo]   featurised {rep:22} -> {X.shape} in {time.time() - t0:.1f}s")
        return cache[rep]

    # ---- the sweep -------------------------------------------------------------------
    all_rows, oof_by_model = [], {}
    for rep_i in range(max(1, a.repeats)):
        folds = build_folds(splits, keys, rep_i, a.seed)
        tag = f"repeat {rep_i + 1}/{a.repeats}"
        print(f"\n[zoo] === {tag}: {len(folds)} outer folds "
              f"({[len(f.test) for f in folds]} test rows) ===")
        seed_everything(a.seed + rep_i)
        rows, oofs = B.run_sweep(
            runnable, X_of, y, folds, seed=a.seed + rep_i,
            tune=not a.no_tune, inner_metric=a.inner_metric,
            log=lambda s: print(f"[zoo]   {s}"))
        for r in rows:
            r["repeat"] = rep_i
        all_rows.extend(rows)
        # Keep repeat 0's OOF predictions as the ones Stage 4b stacks: they come from the
        # frozen splits, so they are paired with the deep ensemble's own OOF artifact.
        if rep_i == 0:
            oof_by_model = oofs

    F = pd.DataFrame(all_rows)
    F.to_csv(f"{stem}_folds.csv", index=False)
    ok = F[F.get("status") == "ok"].copy()
    if ok.empty:
        raise SystemExit("[zoo] every model failed; see the status column in "
                         f"{stem}_folds.csv")

    # ---- summary ---------------------------------------------------------------------
    metrics = [c for c in ("rmse", "mae", "spearman", "pearson", "nll", "cnll", "ece",
                           "mis_area", "cov90", "sharpness", "err_unc_rho", "sigma_scale")
               if c in ok.columns]
    g = ok.groupby("model")
    S = g[metrics].agg(["mean", "std"])
    S.columns = [f"{m}_{s}" for m, s in S.columns]
    S["n_folds"] = g.size()
    S["seconds_total"] = g["seconds"].sum() if "seconds" in ok.columns else np.nan
    S = S.sort_values("spearman_mean", ascending=False)

    print("\n[zoo] === per-fold mean over "
          f"{ok.repeat.nunique()} repeat(s) x {ok.fold.nunique()} folds ===")
    show = [c for c in ("rmse_mean", "spearman_mean", "err_unc_rho_mean", "cnll_mean",
                        "mis_area_mean", "n_folds") if c in S.columns]
    print(S[show].round(3).to_string())

    # ---- paired tests against ONE incumbent -----------------------------------------
    verdict = {"dataset_sha256": sha, "version": a.version, "n_rows": int(len(t)),
               "n_compounds": int(len(set(keys))), "repeats": int(a.repeats),
               "incumbent": a.incumbent, "inner_metric": a.inner_metric,
               "tuned": not a.no_tune, "scaler": a.scaler,
               "models_requested": names, "models_run": runnable,
               "models_skipped": skipped, "comparisons": {}}

    if a.incumbent in set(ok.model):
        inc = ok[ok.model == a.incumbent].set_index(["repeat", "fold"]).sort_index()
        pvals, order = [], []
        print(f"\n[zoo] === paired against the incumbent `{a.incumbent}` "
              f"(Nadeau-Bengio corrected, k=5) ===")
        for name in sorted(set(ok.model) - {a.incumbent}):
            ch = ok[ok.model == name].set_index(["repeat", "fold"]).sort_index()
            common = inc.index.intersection(ch.index)
            if len(common) < 3:
                continue
            c = {}
            for metric, higher_better in (("spearman", True), ("rmse", False)):
                x = ch.loc[common, metric].values
                z = inc.loc[common, metric].values
                # sign convention: positive = the CHALLENGER is better
                d = (x - z) if higher_better else (z - x)
                c[metric] = {
                    **corrected_resampled_ttest(d, np.zeros_like(d), k_folds=5),
                    "wilcoxon": paired_seed_test(x, z)["p_wilcoxon"],
                }
            verdict["comparisons"][name] = c
            pvals.append(c["spearman"]["p"])
            order.append(name)
            arrow = "CHALLENGER better" if c["spearman"]["mean_diff"] > 0 else "incumbent better"
            print(f"  {name:24} d_rho {c['spearman']['mean_diff']:+.4f}  "
                  f"p={c['spearman']['p']:.4f}  n={c['spearman']['n']:3}  {arrow}")
        if pvals:
            for name, padj in zip(order, holm(pvals)):
                verdict["comparisons"][name]["spearman"]["p_holm"] = float(padj)
            sig = [n for n, p in zip(order, holm(pvals))
                   if p < 0.05 and verdict["comparisons"][n]["spearman"]["mean_diff"] > 0]
            print(f"\n[zoo] beats `{a.incumbent}` on Spearman after Holm correction "
                  f"({len(order)} comparisons): {sig or 'none'}")
            verdict["beats_incumbent_holm"] = sig
    else:
        print(f"\n[zoo] incumbent `{a.incumbent}` was not run; no paired tests. "
              f"Pass --models including it, or --incumbent <one that ran>.")

    if a.repeats < 10:
        print(f"\n[zoo] NOTE repeats={a.repeats}. The exact Wilcoxon floor at "
              f"{ok.fold.nunique()} folds is p >= {2 / 2 ** ok.fold.nunique():.4f}; the "
              f"corrected t-test above is the one with power. Use --repeats 10 for a "
              f"result to quote.")

    # ---- out-of-fold predictions, for Stage 4b --------------------------------------
    if oof_by_model:
        recs = []
        for name, o in oof_by_model.items():
            for i in np.where(o["filled"])[0]:
                recs.append({"model": name, "row": int(i), "key": keys[i],
                             "fold": int(o["fold"][i]), "y": float(y[i]),
                             "mu": float(o["mu"][i]),
                             "sigma": (float(o["sigma"][i])
                                       if np.isfinite(o["sigma"][i]) else np.nan)})
        pd.DataFrame(recs).to_csv(f"{stem}_oof.csv", index=False)

    best = S.index[0]
    verdict["best_by_spearman"] = best
    verdict["best_spearman"] = float(S.loc[best, "spearman_mean"])
    verdict["summary"] = {m: {c: float(S.loc[m, c]) for c in show if c in S.columns}
                          for m in S.index}
    S.to_csv(f"{stem}_summary.csv")
    Path(f"{stem}.json").write_text(json.dumps(verdict, indent=2, default=float))

    print(f"\n[zoo] best by Spearman: {best} ({S.loc[best, 'spearman_mean']:.3f})")
    print(f"[zoo] wrote {stem}_folds.csv")
    print(f"[zoo] wrote {stem}_summary.csv")
    if oof_by_model:
        print(f"[zoo] wrote {stem}_oof.csv   <- Stage 4b stacks this")
    print(f"[zoo] wrote {stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
