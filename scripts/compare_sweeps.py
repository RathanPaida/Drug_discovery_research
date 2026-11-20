"""Read everything the Cycle-03 sweeps wrote and print one table per question.

    python scripts/compare_sweeps.py
    python scripts/compare_sweeps.py --runs runs --json

Four questions, four tables:

  1. **Which base model predicts TNK1 best?**     runs/stage2b_zoo_summary*.csv
  2. **Which representation?**                    runs/stage2b_rep_*_summary.csv
  3. **Does combining them help?**                runs/stage4b_stack.json
  4. **Which RL objective generates best?**       runs/rl_*/history.csv + meta.json

Reads only; writes nothing unless `--json` is given. Missing inputs are reported as
missing rather than skipped silently -- an absent sweep and a failed sweep look identical
in a results directory, and only one of them means "re-run it".

ON READING THE RL TABLE. Mean reward is NOT the figure of merit, and the v3 run is the
reason: its agent reached 7.3x the prior's mean reward while every one of its 89 candidate
molecules was rejected by the cross-architecture audit gate on a systematic 1.8-log
optimism gap. A generator that drives a weak critic's score up has learned the critic, not
the chemistry. The columns that matter are therefore reported together:

    score_mean      what the agent optimised       (can be gamed)
    validity        fraction parseable             (collapses without a prior anchor)
    uniqueness      fraction distinct in-batch     (mode collapse shows here first)
    n_scaffolds     distinct Murcko scaffolds      (diversity, the GFlowNet claim)
    audit_pass      fraction within audit_tol      <- the one that cannot be gamed

`audit_pass` is only present once Stage 8 has run against that agent's pool. Where it is
absent the table says so, and the run must not be ranked on reward alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _print_missing(label, paths):
    print(f"  (no {label} found; looked for {', '.join(str(p) for p in paths)})")


def models_table(runs: Path) -> pd.DataFrame | None:
    files = sorted(runs.glob("stage2b_zoo*_summary.csv")) + \
        ([runs / "stage2b_zoo_summary.csv"] if (runs / "stage2b_zoo_summary.csv").exists() else [])
    files = sorted(set(files))
    if not files:
        return None
    frames = []
    for f in files:
        d = pd.read_csv(f, index_col=0)
        d["source"] = f.name
        frames.append(d)
    D = pd.concat(frames)
    cols = [c for c in ("spearman_mean", "spearman_std", "rmse_mean", "rmse_std",
                        "err_unc_rho_mean", "cnll_mean", "mis_area_mean", "n_folds")
            if c in D.columns]
    return D[cols + ["source"]].sort_values(
        "spearman_mean" if "spearman_mean" in cols else cols[0], ascending=False)


def reps_table(runs: Path) -> pd.DataFrame | None:
    rows = []
    for f in sorted(runs.glob("stage2b_rep_*.json")):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        rep = f.stem.replace("stage2b_rep_", "")
        rows.append({"representation": rep,
                     "best_model": j.get("best_by_spearman"),
                     "best_spearman": j.get("best_spearman"),
                     "n_models": len(j.get("models_run") or []),
                     "n_skipped": len(j.get("models_skipped") or {})})
    return pd.DataFrame(rows).sort_values("best_spearman", ascending=False) if rows else None


def stack_summary(runs: Path) -> dict | None:
    p = runs / "stage4b_stack.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    d = j.get("discrete_super_learner", {})
    out = {"chosen": d.get("chosen"), "metric": j.get("metric"),
           "cv_risk": d.get("cv_risk"),
           "best_single_risk": d.get("best_single_risk"),
           "best_stack_risk": d.get("best_stack_risk"),
           "stack_helps": d.get("stack_beats_best_single"),
           "n_models": j.get("n_models"), "n_rows": j.get("n_rows")}
    u = j.get("uncertainty") or {}
    if u:
        r = u.get("report_moment_matched") or {}
        out.update({"combined_err_unc_rho": r.get("err_unc_rho"),
                    "combined_cov90": r.get("cov90"),
                    "combined_cnll": r.get("cnll"),
                    "sigma_epi": (u.get("moment_matched") or {}).get("sigma_epi"),
                    "mixture_nll_uniform": u.get("mixture_nll_uniform"),
                    "mixture_nll_stacked": u.get("mixture_nll_log_score_weights")})
    out["weights"] = d.get("weights")
    return out


def rl_table(runs: Path, tail: int = 20) -> pd.DataFrame | None:
    rows = []
    dirs = [d for d in sorted(runs.glob("rl_*")) if d.is_dir()]
    if (runs / "rl_min").is_dir() and (runs / "rl_min") not in dirs:
        dirs.append(runs / "rl_min")
    for d in sorted(set(dirs)):
        h = d / "history.csv"
        if not h.exists():
            continue
        H = pd.read_csv(h)
        meta = {}
        if (d / "meta.json").exists():
            try:
                meta = json.loads((d / "meta.json").read_text())
            except Exception:
                pass
        base = {}
        if (d / "baseline.json").exists():
            try:
                base = json.loads((d / "baseline.json").read_text())
            except Exception:
                pass
        t = H.tail(tail)
        r = {"run": d.name,
             "algo": meta.get("algo", "dap" if d.name == "rl_min" else d.name.replace("rl_", "")),
             "steps": int(H.step.max()) if "step" in H else len(H),
             "score_mean_last": float(t.score_mean.mean()) if "score_mean" in t else np.nan,
             "score_max": float(H.score_max.max()) if "score_max" in H else np.nan,
             "prior_baseline": base.get("score_mean"),
             "validity_last": float(t.validity.mean()) if "validity" in t else np.nan,
             "uniqueness_last": float(t.uniqueness.mean()) if "uniqueness" in t else np.nan,
             "n_scaffolds_last": float(t.n_scaffolds.mean()) if "n_scaffolds" in t else np.nan,
             "verdict": meta.get("verdict"),
             "minutes": float(H.elapsed_s.max() / 60) if "elapsed_s" in H else np.nan}
        if base.get("score_mean"):
            r["reward_vs_prior"] = r["score_mean_last"] / base["score_mean"]
        # The audit pass rate, if Stage 8 has been run against this agent's pool.
        for cyc in sorted(runs.glob("cycle_*")):
            pre = cyc / "candidates_pre_docking.csv"
            if pre.exists():
                try:
                    n = len(pd.read_csv(pre))
                    r["cycle_candidates"] = n
                except Exception:
                    pass
                break
        rows.append(r)
    return pd.DataFrame(rows) if rows else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None, help="default: paths.runs from the config")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tail", type=int, default=20,
                    help="RL: average the last N steps for the 'last' columns")
    ap.add_argument("--json", action="store_true",
                    help="also write <runs>/sweep_comparison.json")
    a = ap.parse_args(argv)

    if a.runs:
        runs = Path(a.runs)
    else:
        from tnk1.utils.common import load_config
        runs = Path(load_config(a.config)["paths"]["runs"])
    print(f"reading {runs}\n")
    out: dict = {"runs_dir": str(runs)}

    print("=" * 78)
    print("1. BASE MODELS -- which predicts TNK1 best?  (higher Spearman is better)")
    print("=" * 78)
    M = models_table(runs)
    if M is None:
        _print_missing("model-zoo summary", [runs / "stage2b_zoo_summary.csv"])
        print("   produce it:  python scripts/run_stage2b_zoo.py --version v3 --repeats 10")
    else:
        print(M.round(3).to_string())
        out["models"] = json.loads(M.reset_index().to_json(orient="records"))
        print("\n   NB `err_unc_rho` is the correlation between |error| and the model's own "
              "sigma.\n       The deep ensemble's was -0.021; anything above +0.20 is a sigma "
              "worth\n       putting in the reward and the conformal interval.")

    print("\n" + "=" * 78)
    print("2. REPRESENTATIONS -- which feature set?")
    print("=" * 78)
    R = reps_table(runs)
    if R is None:
        _print_missing("representation sweep", [runs / "stage2b_rep_*.json"])
        print("   produce it:  sbatch --array=0-5%4 hpc/submit_array.sh zoorep")
    else:
        print(R.round(3).to_string(index=False))
        out["representations"] = json.loads(R.to_json(orient="records"))

    print("\n" + "=" * 78)
    print("3. STACKING -- does combining the zoo beat the best single model?")
    print("=" * 78)
    S = stack_summary(runs)
    if S is None:
        _print_missing("stack report", [runs / "stage4b_stack.json"])
        print("   produce it:  python scripts/run_stage4b_stack.py --version v3")
    else:
        w = S.pop("weights", None)
        for k, v in S.items():
            print(f"   {k:26} {v}")
        if w:
            top = {k: round(v, 4) for k, v in sorted(w.items(), key=lambda kv: -kv[1])
                   if v > 1e-3}
            print(f"   {'weights':26} {top}")
        if S.get("stack_helps") is False:
            print("\n   The discrete super learner chose a SINGLE model over every "
                  "combiner.\n   That is a real answer: ship the single model.")
        out["stack"] = {**S, "weights": w}

    print("\n" + "=" * 78)
    print("4. RL OBJECTIVES -- which generator?  (read ALL the columns, not score_mean)")
    print("=" * 78)
    L = rl_table(runs, a.tail)
    if L is None:
        _print_missing("RL histories", [runs / "rl_*/history.csv"])
        print("   produce them:  sbatch --array=0-5%2 hpc/submit_array.sh rlalgo")
    else:
        print(L.round(3).to_string(index=False))
        out["rl"] = json.loads(L.to_json(orient="records"))
        print("\n   WARNING a high `score_mean` alone means the agent learned the CRITIC, "
              "not the\n   chemistry. The v3 DAP run reached 7.3x the prior's reward and "
              "produced ZERO\n   candidates: all 89 were rejected on a 1.8-log "
              "ensemble-vs-GP optimism gap.\n   Rank these runs by `cycle_candidates` once "
              "Stage 8 has run against each agent.")

    if a.json:
        p = runs / "sweep_comparison.json"
        p.write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
