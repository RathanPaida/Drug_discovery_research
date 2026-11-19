"""Stage 9: prospective hit rates with exact Clopper-Pearson CIs per slice, across all cycles.

This stage consumes WET-LAB RETURNS (`runs/cycle_*/results.csv`), not predictions. Until
Team B reports, there is nothing for it to compute -- and that is the normal state, not an
error, so it exits 0 with an explanation rather than a traceback.

    python scripts/run_stage9_report.py
    python scripts/run_stage9_report.py --min-purity 90 --out runs/stage9_hit_rates.csv

What changed and why
--------------------
This file used to be bare module-level code: no `main()`, no argparse, and two CWD-relative
paths (`runs/cycle_*/results.csv` to read, `runs/stage9_hit_rates.csv` to write). Three
consequences, all silent or confusing:

  * run from anywhere but the repo root, the glob matched nothing and `pd.concat([])`
    raised `ValueError: No objects to concatenate` -- which reads like a data problem, not
    a working-directory problem;
  * with no wet-lab results yet (the actual situation for Cycle 01) it raised the same
    error, so "the experiment has not returned" was indistinguishable from "the stage is
    broken";
  * it wrote its output next to wherever the operator was standing.

Paths now come from the config (absolutised against the repo root), the required columns
are checked by name, and every slice comparison states the n it is based on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.evaluation.stats import hit_rate_table, fisher_slice_comparison, holm  # noqa: E402
from tnk1.utils.common import load_config  # noqa: E402

# A wet-lab return must carry these to be interpretable at all.
REQUIRED_COLUMNS = ("batch_slice", "pact", "censored", "purity_pct")

# (label, pActivity threshold). pAct 6.0 == 1 uM, 7.0 == 100 nM.
THRESHOLDS = (("IC50<1uM", 6.0), ("IC50<100nM", 7.0))


def load_cycle_results(runs: Path) -> tuple[pd.DataFrame, list[str]]:
    """Concatenate every `cycle_*/results.csv`, tagging each row with its cycle."""
    paths = sorted(runs.glob("cycle_*/results.csv"))
    frames, used = [], []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            # Fail closed on a malformed return rather than averaging a wrong column away.
            raise ValueError(
                f"{p} is missing required column(s) {missing}; "
                f"found {sorted(df.columns)[:12]}..."
            )
        # cycle_07/results.csv -> "07"
        frames.append(df.assign(cycle=p.parent.name.split("cycle_")[-1]))
        used.append(str(p))
    if not frames:
        return pd.DataFrame(columns=[*REQUIRED_COLUMNS, "cycle"]), used
    return pd.concat(frames, ignore_index=True), used


def hit_rates(R: pd.DataFrame) -> dict:
    out: dict = {}
    for name, thr in THRESHOLDS:
        counts = {
            slice_: (int(((g.pact >= thr) & (g.censored == 0)).sum()), len(g))
            for slice_, g in R.groupby("batch_slice")
        }
        table = hit_rate_table(counts)
        # Holm-adjust the slice-vs-control comparisons together: two uncorrected Fisher
        # tests at alpha=0.05 is a 10% family-wise error rate on the headline claim.
        comps = {
            f"{arm}_vs_CONTROL": fisher_slice_comparison(counts[arm], counts["CONTROL"])
            for arm in ("EXPLOIT", "EXPLORE")
            if arm in counts and "CONTROL" in counts
        }
        table["fisher_p_holm"] = dict(zip(comps, holm(list(comps.values())))) if comps else {}
        if "CONTROL" not in counts:
            table["fisher_note"] = "no CONTROL slice in the returns; no comparison possible"
        out[name] = table
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--min-purity", type=float, default=95.0,
                    help="drop returns below this purity%% (default 95)")
    ap.add_argument("--out", default=None,
                    help="default: <runs>/stage9_hit_rates.csv")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    runs = Path(cfg["paths"]["runs"])          # absolute: load_config resolved it
    out_csv = Path(a.out) if a.out else runs / "stage9_hit_rates.csv"

    R, used = load_cycle_results(runs)
    if R.empty:
        print(f"Stage 9: no wet-lab returns found under {runs}/cycle_*/results.csv")
        print("         This is expected until Team B reports. Nothing to compute.")
        print("         A return needs columns: " + ", ".join(REQUIRED_COLUMNS))
        return 0

    n_before = len(R)
    R = R[R.purity_pct >= a.min_purity]
    print(f"Stage 9: {n_before} returned compounds from {len(used)} cycle file(s); "
          f"{len(R)} pass purity >= {a.min_purity}%")
    if R.empty:
        print(f"         No compound met the purity bar; refusing to report a hit rate.")
        return 1

    res = hit_rates(R)
    print(json.dumps(res, indent=2, default=float))

    table = pd.DataFrame({
        label: {
            s: f"{v['hits']}/{v['n']} ({v['ci95'][0]:.2f}-{v['ci95'][1]:.2f})"
            for s, v in d.items() if s not in ("fisher_p_holm", "fisher_note")
        }
        for label, d in res.items()
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv)
    (out_csv.with_suffix(".json")).write_text(json.dumps(
        {"sources": used, "min_purity": a.min_purity, "n_analysed": len(R), "results": res},
        indent=2, default=float))
    print(f"\nwrote {out_csv}\nwrote {out_csv.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
