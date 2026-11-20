"""Stage 4b: combine the zoo into one predictor, and decide honestly whether to.

    python scripts/run_stage4b_stack.py --version v3
    python scripts/run_stage4b_stack.py --version v3 --metric spearman

WHAT IT DOES
------------
Reads the out-of-fold prediction matrix Stage 2b wrote (`runs/stage2b_zoo_oof.csv`), adds
the deep ensemble's own out-of-fold predictions (`runs/stage4_oof_predictions.csv`) as one
more column, and fits every combiner in `tnk1.ensembles.stacking.STACKERS` on it -- then
asks a DISCRETE SUPER LEARNER whether any of them actually beats the best single model.

THE GUARD IS THE POINT. Phillips, van der Laan, Lee & Gruber, "Practical considerations
for specifying a super learner", International Journal of Epidemiology (2023), step 5:
when an ensemble super learner is used, "we recommend it be evaluated as another candidate
in a dSL, along with the individual learners from which it was constructed... and for the
final SL to be the candidate learner that achieved the lowest CV risk."

This project needs that guard more than most. It shipped a combination -- a 10-member deep
ensemble -- that is worse than a single Morgan-count random forest (Spearman 0.501 against
0.686 over 5 paired outer folds), and used it as the RL critic for a 6-hour HPC campaign
that produced zero candidates. A discrete super learner makes that structurally
impossible: every individual model is itself a candidate, so the selected predictor cannot
be worse than the best single one by more than cross-validation noise. It can also return
"no, do not stack", which is what makes it an experiment rather than a ritual.

WHY THE STACK IS NOT FITTED AND SCORED ON THE SAME ROWS
-------------------------------------------------------
Fitting K simplex weights on 172 rows IS a selection step, so scoring the fitted stack on
those rows is the canonical stacking leak (Wolpert 1992 on why the meta-learner needs
out-of-fold inputs; Varma & Simon, BMC Bioinformatics 7:91 (2006), on how plain CV used
for a selection step gives "a very biased estimate of the true error, not much better than
resubstitution"). Every number this script reports for a combiner comes from
scaffold-GROUPED cross-validation over the OOF matrix, with the weights fitted inside each
fold. Greedy selection with replacement is the most exposed to this -- it explicitly
hill-climbs the score -- which is why the bagged variant is included and preferred.

OUTPUTS
-------
    runs/stage4b_stack.json        chosen candidate, CV risks of ALL candidates, weights
    runs/stage4b_stack_oof.csv     the stacked prediction per row, for inspection
    data/processed/stack.pkl       the fitted combiner, for the RL critic to load
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.ensembles import combine as CB, stacking as ST  # noqa: E402
from tnk1.uq.metrics import full_report  # noqa: E402
from tnk1.utils.common import load_config, read_table, verify_locked  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--version", default="v3")
    ap.add_argument("--zoo-oof", default=None, help="default: <runs>/stage2b_zoo_oof.csv")
    ap.add_argument("--ens-oof", default=None,
                    help="the deep ensemble's OOF predictions; "
                         "default: <runs>/stage4_oof_predictions.csv")
    ap.add_argument("--metric", default="rmse", choices=["rmse", "mae", "spearman"],
                    help="the CV risk the discrete super learner minimises. Use spearman "
                         "if ranking is what the RL reward consumes.")
    ap.add_argument("--min-coverage", type=float, default=0.95,
                    help="drop a model that predicted fewer than this fraction of rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    proc, runs = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["runs"])
    stem = Path(a.out) if a.out else runs / "stage4b_stack"
    zoo_p = Path(a.zoo_oof) if a.zoo_oof else runs / "stage2b_zoo_oof.csv"
    ens_p = Path(a.ens_oof) if a.ens_oof else runs / "stage4_oof_predictions.csv"

    if not zoo_p.exists():
        raise SystemExit(
            f"{zoo_p} not found. Stage 2b produces it:\n"
            f"  python scripts/run_stage2b_zoo.py --version {a.version} --tag core")

    Zo = pd.read_csv(zoo_p)
    print(f"[stack] zoo OOF: {Zo.model.nunique()} models x {Zo.row.nunique()} rows")

    # ---- the scaffold groups, so the meta-level CV is grouped the same way -----------
    tbl = proc / f"kinome_long_{a.version}.parquet"
    sha = verify_locked(tbl)
    df = read_table(tbl)
    tnk1 = cfg["targets"]["tnk1"]
    t = df[(df.uniprot_id == tnk1) & (df.censored == 0)].reset_index(drop=True)
    scaffold_of = dict(zip(t.inchikey_block1.values, t.scaffold.values))

    # ---- wide matrix -----------------------------------------------------------------
    P = Zo.pivot_table(index="row", columns="model", values="mu")
    Sg = Zo.pivot_table(index="row", columns="model", values="sigma")
    yv = Zo.groupby("row").y.first()
    kv = Zo.groupby("row").key.first()

    # ---- add the deep ensemble as one more candidate column -------------------------
    if ens_p.exists():
        E = pd.read_csv(ens_p)
        if "key" in E.columns:
            key_to_row = {k: r for r, k in kv.items()}
            E = E.assign(row=E.key.map(key_to_row)).dropna(subset=["row"])
            E["row"] = E["row"].astype(int)
            P["DeepEnsemble"] = E.set_index("row").mu.reindex(P.index)
            for c in ("sigma_epi", "sigma_tot", "sigma"):
                if c in E.columns:
                    Sg["DeepEnsemble"] = E.set_index("row")[c].reindex(P.index)
                    break
            print(f"[stack] added DeepEnsemble from {ens_p.name} "
                  f"({P['DeepEnsemble'].notna().sum()} rows joined)")
        else:
            print(f"[stack] {ens_p.name} predates the `key` column; the deep ensemble "
                  f"cannot be joined row-wise and is EXCLUDED from the candidate set.")
    else:
        print(f"[stack] {ens_p} absent; the deep ensemble is not a candidate. Run\n"
              f"        python scripts/run_pipeline.py --version {a.version} --only oof")

    # ---- drop thin columns, then complete-case rows ---------------------------------
    cov = P.notna().mean()
    thin = cov[cov < a.min_coverage]
    if len(thin):
        print(f"[stack] dropping {len(thin)} model(s) below {a.min_coverage:.0%} row "
              f"coverage: {dict(thin.round(3))}")
        P = P.drop(columns=list(thin.index))
        Sg = Sg.drop(columns=[c for c in thin.index if c in Sg.columns])
    rows_ok = P.notna().all(axis=1)
    P = P[rows_ok]
    yv = yv[rows_ok]
    kv = kv[rows_ok]
    Sg = Sg.reindex(index=P.index, columns=P.columns)
    names = list(P.columns)
    groups = np.array([scaffold_of.get(k, k) for k in kv.values])
    Pm, Y = P.values.astype(float), yv.values.astype(float)
    print(f"[stack] matrix {Pm.shape} over {len(set(groups))} scaffold groups "
          f"({int(rows_ok.sum())}/{len(rows_ok)} complete-case rows)")
    if Pm.shape[1] < 2:
        raise SystemExit("[stack] fewer than two usable models; nothing to combine")

    # ---- the discrete super learner --------------------------------------------------
    dsl = ST.DiscreteSuperLearner(metric=a.metric, seed=a.seed).fit(
        Pm, Y, names=names, groups=groups)
    rep = dsl.report()
    print(f"\n[stack] === candidate CV risk ({a.metric}, scaffold-grouped) ===")
    for e in rep["ranking"][:15]:
        print(f"  {e['candidate']:34} {e['risk']:.4f}")
    if len(rep["ranking"]) > 15:
        print(f"  ... {len(rep['ranking']) - 15} more in {stem}.json")
    print(f"\n[stack] CHOSEN: {rep['chosen']}  (risk {rep['cv_risk']:.4f})")
    print(f"[stack] best single {rep['best_single_risk']:.4f} | "
          f"best stack {rep['best_stack_risk']:.4f} | "
          f"stacking helps: {rep['stack_beats_best_single']}")
    if not rep["stack_beats_best_single"]:
        print("[stack] NOTE no combiner beat the best single model. That is a legitimate\n"
              "        outcome, not a failure: ship the single model and say so.")
    if "weights" in rep:
        w = {k: round(v, 4) for k, v in sorted(rep["weights"].items(),
                                               key=lambda kv: -kv[1]) if v > 1e-3}
        print(f"[stack] weights: {w}")

    # ---- uncertainty: moment-match, then pool-then-calibrate -------------------------
    unc = {}
    have_sigma = Sg.notna().all(axis=0)
    sig_names = [n for n in names if bool(have_sigma.get(n, False))]
    if len(sig_names) >= 2:
        mu_k = P[sig_names].values.T
        sd_k = Sg[sig_names].values.T
        wts = None
        if "weights" in rep:
            wv = np.array([rep["weights"].get(n, 0.0) for n in sig_names])
            wts = wv if wv.sum() > 0 else None
        mm = CB.moment_match(mu_k, sd_k, wts)
        # Rahaman & Thiery (NeurIPS 2021): the scale must be fitted AFTER pooling.
        pc = CB.PoolThenCalibrate("scalar").fit(Y, mm["mu"], mm["sigma"])
        sd_cal = pc.transform(mm["sigma"])
        w_log, info = CB.stack_log_score(mu_k, sd_k, Y)
        unc = {
            "members_with_sigma": sig_names,
            "moment_matched": {k: float(np.mean(mm[k]))
                               for k in ("sigma_ale", "sigma_epi", "sigma")},
            "pool_then_calibrate": pc.report(),
            "mixture_nll_uniform": CB.mixture_nll(Y, mu_k, sd_k),
            "mixture_nll_log_score_weights": CB.mixture_nll(Y, mu_k, sd_k, w_log),
            "mixture_crps_uniform": CB.mixture_crps(Y, mu_k, sd_k),
            "log_score_weights": {n: float(v) for n, v in zip(sig_names, w_log)},
            "log_score_fit": info,
            "report_moment_matched": full_report(Y, mm["mu"], sd_cal),
        }
        print(f"\n[stack] === uncertainty over {len(sig_names)} members with a sigma ===")
        print(f"  sigma_ale {unc['moment_matched']['sigma_ale']:.3f}  "
              f"sigma_epi {unc['moment_matched']['sigma_epi']:.3f}  "
              f"(epistemic = cross-model disagreement; this is what feeds r_pot)")
        print(f"  pool-then-calibrate scale s = {pc.report()['s']:.3f}")
        print(f"  mixture NLL  uniform {unc['mixture_nll_uniform']:.3f} -> "
              f"log-score weights {unc['mixture_nll_log_score_weights']:.3f}")
        r = unc["report_moment_matched"]
        print(f"  combined: rho {r['spearman']:+.3f}  err-sigma rho {r['err_unc_rho']:+.3f}  "
              f"cov90 {r['cov90']:.3f}  cNLL {r['cnll']:.3f}")
        print(f"  (the deep ensemble's err-sigma rho was -0.021; above +0.20 means the "
              f"combined sigma knows where it is wrong)")
    else:
        print(f"\n[stack] only {len(sig_names)} member(s) provide a sigma; the "
              f"uncertainty combination needs at least 2. Include models tagged `uq` "
              f"(gp_*, ngboost, qrf, rf, bayes_ridge) in Stage 2b.")

    # ---- honest per-row prediction for inspection ------------------------------------
    yhat = dsl.predict(Pm)
    pd.DataFrame({"row": P.index, "key": kv.values, "y": Y, "stacked": yhat,
                  "scaffold": groups}).to_csv(f"{stem}_oof.csv", index=False)

    out = {"dataset_sha256": sha, "version": a.version, "metric": a.metric,
           "n_rows": int(len(Y)), "n_models": int(Pm.shape[1]), "models": names,
           "discrete_super_learner": rep, "uncertainty": unc,
           "in_sample_report_do_not_quote": full_report(Y, yhat),
           "note": ("`in_sample_report_do_not_quote` is the chosen candidate scored on the "
                    "rows its weights were fitted on. It is reported only to catch gross "
                    "errors. The honest number is `discrete_super_learner.cv_risk`.")}
    Path(f"{stem}.json").write_text(json.dumps(out, indent=2, default=float))

    # ---- persist the combiner for the RL critic --------------------------------------
    pkl = proc / "stack.pkl"
    pkl.write_bytes(pickle.dumps({"dsl": dsl, "names": names, "metric": a.metric,
                                  "dataset_sha256": sha}))
    print(f"\n[stack] wrote {stem}.json")
    print(f"[stack] wrote {stem}_oof.csv")
    print(f"[stack] wrote {pkl}   <- the RL critic loads this")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
