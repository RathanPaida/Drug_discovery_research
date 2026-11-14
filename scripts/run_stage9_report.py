"""Stage 9: prospective hit rates with exact Clopper-Pearson CIs per slice across all cycles + slice comparisons."""
import glob, json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.evaluation.stats import hit_rate_table, fisher_slice_comparison, holm

dfs = [pd.read_csv(p).assign(cycle=p.split("cycle_")[1][:2]) for p in sorted(glob.glob("runs/cycle_*/results.csv"))]
R = pd.concat(dfs); R = R[R.purity_pct >= 95]
out = {}
for name, thr in (("IC50<1uM", 6.0), ("IC50<100nM", 7.0)):
    c = {s: (int(((g.pact >= thr) & (g.censored == 0)).sum()), len(g)) for s, g in R.groupby("batch_slice")}
    out[name] = hit_rate_table(c)
    comps = {f"{a}_vs_CONTROL": fisher_slice_comparison(c[a], c["CONTROL"]) for a in ("EXPLOIT", "EXPLORE") if a in c and "CONTROL" in c}
    out[name]["fisher_p"] = dict(zip(comps, holm(list(comps.values())))) if comps else {}
print(json.dumps(out, indent=2, default=float))
pd.DataFrame({k: {s: f"{v['hits']}/{v['n']} ({v['ci95'][0]:.2f}-{v['ci95'][1]:.2f})" for s, v in d.items() if s != 'fisher_p'} for k, d in out.items()}).to_csv("runs/stage9_hit_rates.csv")
