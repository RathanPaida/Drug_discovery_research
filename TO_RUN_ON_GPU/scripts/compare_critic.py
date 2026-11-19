"""Is the deep ensemble actually better than the Stage-2 baselines? Paired, across folds.

    python scripts/compare_critic.py --version v3

Writes `runs/critic_comparison.csv` (per fold x model) and `runs/critic_comparison.json`
(paired tests + verdict), and prints the table.

WHY THIS EXISTS
---------------
The deep ensemble is the RL critic: `tnk1/rl/scorer.py` builds `r_pot` from its `mu` and
`sigma_epi`, and `r_sel` from its cross-kinase margins. Those two terms carry 5.5 of the
12.5 weight units in `COMPONENTS_V2`. So if the ensemble ranks compounds worse than a
random forest, the generator is being optimised against the weakest model in the
repository, and nothing downstream of it means what it claims to.

On the v1 data it did exactly that:

    Random Forest     RMSE 0.440   Spearman 0.641
    deep ensemble     RMSE 0.623   Spearman 0.264

That comparison was on a SINGLE held-out split of 45 compounds (76 under the v3 splits).
Spearman's standard error at n=76 is about 0.12, so one split cannot separate models whose
rho differs by less than ~0.2 -- and it certainly cannot support "the ensemble is better",
which is the claim the architecture rests on. `CHANGES_V2.md` records the gap as defect 6,
unfixed, and `PIPELINE_AUDIT.md` leaves it open.

WHAT IS COMPARED
----------------
The ensemble's predictions come from `runs/stage4_oof_predictions.csv`, which the `oof`
stage produces by training, for each of the 5 outer folds, an ensemble that never saw that
fold's compounds (they are also removed from the kinome table, so there is no transfer
leakage either). The baselines are then fitted on the same fold's training keys and asked
for the same fold's test rows.

That makes the two genuinely paired: same folds, same compounds, same exclusions. The
per-fold metrics are compared with a Wilcoxon signed-rank test over the 5 folds
(`tnk1.evaluation.stats.paired_seed_test`), and the per-row absolute errors are compared
as well, which is the more sensitive test when the folds are few.

READ THE RESULT HONESTLY. Five folds is a small sample for a signed-rank test: it cannot
produce a p-value below 0.0625 at all. Treat the per-fold mean difference and its direction
as the finding, and the p-value as weak support.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.baselines import features as F, models as M  # noqa: E402
from tnk1.evaluation.stats import paired_seed_test  # noqa: E402
from tnk1.uq.metrics import full_report  # noqa: E402
from tnk1.utils.common import load_config, read_table, seed_everything, verify_locked  # noqa: E402

# Baselines to fit per fold. kNN and the GP give calibrated sigmas too, but the question
# here is ranking, so RMSE / Spearman on the point prediction is what is reported.
BASELINES = {
    "RF": ("tab", lambda: M.ForestWithStd(max_features=0.3, min_samples_leaf=1, random_state=0)),
    "XGB": ("tab", lambda: M.make_xgb(max_depth=5, learning_rate=0.01, reg_lambda=1, random_state=0)),
    "kNN": ("fp", lambda: M.KNNTanimoto(k=5)),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--version", default="v3")
    ap.add_argument("--oof", default=None, help="default: <runs>/stage4_oof_predictions.csv")
    ap.add_argument("--out", default=None, help="default: <runs>/critic_comparison")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    proc, runs = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["runs"])
    oof_p = Path(a.oof) if a.oof else runs / "stage4_oof_predictions.csv"
    out_stem = Path(a.out) if a.out else runs / "critic_comparison"

    if not oof_p.exists():
        raise SystemExit(
            f"{oof_p} not found. The ensemble's out-of-fold predictions come from the `oof`\n"
            f"stage, which must run first:\n"
            f"  python scripts/run_pipeline.py --version {a.version} --only oof")

    oof = pd.read_csv(oof_p)
    has_key = "key" in oof.columns
    if not has_key:
        print("NOTE: this stage4_oof_predictions.csv predates the `key` column, so the\n"
              "      per-row comparison is unavailable; only per-fold metrics are computed.")

    # --- the same frozen table and splits the ensemble was trained against -------------
    tbl = proc / f"kinome_long_{a.version}.parquet"
    sha = verify_locked(tbl)
    df = read_table(tbl)
    spl = json.loads((proc / f"splits_{a.version}.json").read_text())
    tnk1 = cfg["targets"]["tnk1"]

    t = df[df.uniprot_id == tnk1].reset_index(drop=True)
    # Censored rows carry a bound, not a value; the ensemble's oof stage scores exact rows
    # only (`C.tnk1_rows(te_keys, censored=False)`), so the baselines must match.
    t = t[t.censored == 0].reset_index(drop=True)
    print(f"TNK1 exact rows: {len(t)}   dataset {sha[:12]}")

    fp = F.morgan_count(t.smiles)
    de = F.rdkit_descriptors(t.smiles)
    asy = F.assay_onehot(t.standard_type)
    y = t.pact.values
    key = t.inchikey_block1.values

    def X_for(kind, tr_idx):
        if kind == "fp":
            return fp
        sc = F.DescriptorScaler().fit(de[tr_idx])          # fit on train rows only
        return np.hstack([fp, sc.transform(de), asy])

    rows, per_row = [], []
    for f in spl["cv"]:
        fold = f["outer"]
        tr_keys, te_keys = set(f["train"]), set(f["test"])
        tr_idx = np.where(np.isin(key, list(tr_keys)))[0]
        te_idx = np.where(np.isin(key, list(te_keys)))[0]
        if len(tr_idx) < 10 or len(te_idx) < 3:
            print(f"fold {fold}: too few rows (train {len(tr_idx)}, test {len(te_idx)}) -- skipped")
            continue

        ens_f = oof[oof.fold == fold]
        if len(ens_f) != len(te_idx):
            # Not fatal, but it means the two are not scoring the same compound set, which
            # is exactly the thing this script exists to control for. Say so loudly.
            print(f"WARNING fold {fold}: ensemble scored {len(ens_f)} rows, the baselines "
                  f"see {len(te_idx)}. Per-fold metrics are still comparable only if the "
                  f"compound sets match -- check the splits version.")

        rows.append({"fold": fold, "model": "DeepEnsemble", "n": len(ens_f),
                     **full_report(ens_f.y.values, ens_f.mu.values)})

        for name, (kind, ctor) in BASELINES.items():
            seed_everything(a.seed)
            Xa = X_for(kind, tr_idx)
            m = ctor().fit(Xa[tr_idx], y[tr_idx])
            pred = np.asarray(m.predict(Xa[te_idx]))
            rows.append({"fold": fold, "model": name, "n": len(te_idx),
                         **full_report(y[te_idx], pred)})
            if has_key:
                per_row.append(pd.DataFrame({"fold": fold, "model": name,
                                             "key": key[te_idx], "y": y[te_idx], "pred": pred}))

    if not rows:
        raise SystemExit("no fold produced a comparison; check the splits file")

    R = pd.DataFrame(rows)
    R.to_csv(f"{out_stem}.csv", index=False)

    # --- per-fold summary -------------------------------------------------------------
    summary = R.groupby("model")[["rmse", "mae", "spearman", "pearson"]].agg(["mean", "std"])
    print()
    print("=== per-fold mean +- sd over "
          f"{R.fold.nunique()} outer folds ===")
    print(summary.round(3).to_string())

    # --- paired tests, ensemble vs each baseline --------------------------------------
    verdict = {"dataset_sha256": sha, "version": a.version,
               "n_folds": int(R.fold.nunique()), "comparisons": {}}
    ens = R[R.model == "DeepEnsemble"].set_index("fold").sort_index()
    print()
    print("=== paired across folds (Wilcoxon; 5 folds cannot give p < 0.0625) ===")
    for name in BASELINES:
        b = R[R.model == name].set_index("fold").sort_index()
        common = ens.index.intersection(b.index)
        if len(common) < 3:
            continue
        c = {}
        for metric, better_is_low in (("rmse", True), ("spearman", False)):
            # sign convention: positive mean_diff = the ENSEMBLE is better
            if better_is_low:
                d = paired_seed_test(b.loc[common, metric].values, ens.loc[common, metric].values)
            else:
                d = paired_seed_test(ens.loc[common, metric].values, b.loc[common, metric].values)
            c[metric] = d
            arrow = "ensemble better" if d["mean_diff"] > 0 else "BASELINE better"
            print(f"  vs {name:4}  {metric:9} mean_diff {d['mean_diff']:+.4f}  "
                  f"p={d['p_wilcoxon']:.4f}   {arrow}")
        verdict["comparisons"][name] = c

    # --- per-row paired errors (more sensitive than 5 fold-level points) --------------
    if per_row:
        PR = pd.concat(per_row, ignore_index=True)
        PR.to_csv(f"{out_stem}_per_row.csv", index=False)
        e = oof[["fold", "key", "y", "mu"]].copy()
        e["abs_err_ens"] = (e.y - e.mu).abs()
        print()
        print("=== paired per-row absolute error (same compounds, joined on key) ===")
        for name in BASELINES:
            b = PR[PR.model == name].copy()
            b["abs_err_base"] = (b.y - b.pred).abs()
            j = e.merge(b[["fold", "key", "abs_err_base"]], on=["fold", "key"], how="inner")
            if len(j) < 10:
                print(f"  vs {name:4}  only {len(j)} rows joined -- skipped")
                continue
            d = paired_seed_test(j.abs_err_base.values, j.abs_err_ens.values)
            arrow = "ensemble better" if d["mean_diff"] > 0 else "BASELINE better"
            print(f"  vs {name:4}  n={len(j):4}  mean |err| diff {d['mean_diff']:+.4f}  "
                  f"p={d['p_wilcoxon']:.3e}   {arrow}")
            verdict["comparisons"].setdefault(name, {})["per_row_abs_err"] = {**d, "n": int(len(j))}

    # --- the headline ------------------------------------------------------------------
    best = summary[("spearman", "mean")].idxmax()
    ens_rho = float(summary.loc["DeepEnsemble", ("spearman", "mean")])
    best_rho = float(summary.loc[best, ("spearman", "mean")])
    verdict["best_model_by_spearman"] = best
    verdict["ensemble_spearman"] = ens_rho
    verdict["best_spearman"] = best_rho
    verdict["ensemble_is_best"] = bool(best == "DeepEnsemble")

    print()
    if best == "DeepEnsemble":
        print(f"VERDICT: the deep ensemble ranks best (Spearman {ens_rho:.3f}). It is a\n"
              f"         defensible RL critic.")
    else:
        print(f"VERDICT: {best} ranks better than the deep ensemble "
              f"(Spearman {best_rho:.3f} vs {ens_rho:.3f}).")
        print( "         The ensemble is what drives r_pot and r_sel in the RL reward, so the\n"
               "         generator is being optimised against the weaker model. Options:\n"
               "           - run `ablate` (A0-A7) to see which part of the transfer stack pays\n"
               "           - shrink the trunk (model.hidden 300->128, fusion [512,256]->[128,64],\n"
               "             dropout 0.2->0.35) and/or fine-tune with loss=huber not nll\n"
               "           - set uq.independent_kinome_training correctly, or back the critic\n"
               "             with a stacked RF/GP: tnk1/rl/scorer.py touches it through only\n"
               "             two duck-typed methods (predict_members, latent).")
    Path(f"{out_stem}.json").write_text(json.dumps(verdict, indent=2, default=float))
    print()
    print(f"wrote {out_stem}.csv")
    print(f"wrote {out_stem}.json")
    if per_row:
        print(f"wrote {out_stem}_per_row.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
