"""Stage 2: baselines on TNK1 using the frozen scaffold splits. Reports mu +- sigma over 5 seeds."""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.utils.common import load_config, seed_everything, read_table, verify_locked
from tnk1.baselines import features as F, models as M
from tnk1.uq.metrics import full_report


def grid(d): return [dict(zip(d, v)) for v in itertools.product(*d.values())]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--version", default="v1"); a = ap.parse_args()
    cfg = load_config(a.config); proc = Path(cfg["paths"]["processed"])
    path = proc / f"kinome_long_{a.version}.parquet"; sha = verify_locked(path)
    df = read_table(path); spl = json.loads((proc / f"splits_{a.version}.json").read_text())
    t = df[(df.uniprot_id == cfg["targets"]["tnk1"]) & (df.censored == 0)].reset_index(drop=True)
    fp, de, asy = F.morgan_count(t.smiles), F.rdkit_descriptors(t.smiles), F.assay_onehot(t.standard_type)
    y = t.pact.values; key = t.inchikey_block1.values
    rows_of = lambda keys: np.where(np.isin(key, keys))[0]
    dev, test = rows_of(spl["dev_keys"]), rows_of(spl["test_keys"])

    def X_for(kind, tr):
        if kind == "fp": return fp
        sc = F.DescriptorScaler().fit(de[tr]); return np.hstack([fp, sc.transform(de), asy])

    specs = {
        "kNN": ("fp", lambda p: M.KNNTanimoto(**p), {"k": [1, 3, 5, 7]}),
        "RF": ("tab", lambda p: M.ForestWithStd(**p), {"max_features": ["sqrt", .3, .5], "min_samples_leaf": [1, 3, 5]}),
        "XGB": ("tab", lambda p: M.make_xgb(**p), {"max_depth": [3, 5, 7], "learning_rate": [.01, .05], "reg_lambda": [1, 10]}),
        "TanimotoGP": ("fp", lambda p: M.TanimotoGP(), {"_": [0]}),
    }
    results = []
    for name, (kind, ctor, g) in specs.items():
        # hyperparameter selection: mean inner-CV RMSE across outer folds' inner splits
        best, best_s = None, 1e9
        for p in grid(g):
            errs = []
            for f in spl["cv"][:2]:
                for inn in f["inner"][:2]:
                    tr, va = rows_of(inn["train"]), rows_of(inn["val"])
                    if len(tr) < 5 or not len(va): continue
                    Xa = X_for(kind, tr); m = ctor({k: v for k, v in p.items() if k != "_"}).fit(Xa[tr], y[tr])
                    errs.append(np.sqrt(np.mean((m.predict(Xa[va]) - y[va]) ** 2)))
            s = np.mean(errs) if errs else 1e9
            if s < best_s: best, best_s = p, s
        per_seed = []
        for sd in cfg["seeds"]:
            seed_everything(sd)
            p = {k: v for k, v in best.items() if k != "_"}
            if name in ("RF", "XGB"): p["random_state"] = sd
            Xa = X_for(kind, dev); m = ctor(p).fit(Xa[dev], y[dev])
            if name in ("RF", "kNN", "TanimotoGP"):
                mu, sig = m.predict(Xa[test], return_std=True); per_seed.append(full_report(y[test], mu, sig))
            else:
                per_seed.append(full_report(y[test], m.predict(Xa[test])))
        df_s = pd.DataFrame(per_seed)
        results.append({"model": name, "best_params": str(best),
                        **{f"{c}_mean": df_s[c].mean() for c in df_s}, **{f"{c}_std": df_s[c].std() for c in df_s}})
        print(name, df_s[["rmse", "spearman"]].mean().round(3).to_dict())
    out = Path(cfg["paths"]["runs"]) / "stage2_baselines.csv"; out.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False); print("wrote", out, "| dataset", sha[:12])


if __name__ == "__main__":
    main()
