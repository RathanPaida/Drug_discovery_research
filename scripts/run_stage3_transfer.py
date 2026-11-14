"""Stage 3: python scripts/run_stage3_transfer.py {store|pretrain|lambda|ablate} [--device cuda]"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.context import Ctx
from tnk1.utils.common import seed_everything, Tracker
from tnk1.models import kinase_features as KF
from tnk1.models.ablations import ABLATIONS
from tnk1.models.pipeline import kinome_stage, finetune_tnk1, loko_tune_lambda, _mk
from tnk1.models.trainer import predict
from tnk1.uq.metrics import full_report


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--version", default="v1"); ap.add_argument("--device", default="cuda"); ap.add_argument("--only", nargs="*"); a = ap.parse_args()
    C = Ctx(a.config, a.version, a.device); cfg = C.cfg; proc = C.proc
    ssl = proc / "ssl_encoder.pt"

    if a.cmd == "store":
        kin = pd.read_csv(cfg["data"]["kinase_table"])
        KF.save_store(KF.build_kinase_store(kin, C.tnk1, a.device, cfg["kinome_train"]["sim_klifs"], cfg["kinome_train"]["sim_esm"]), proc / "kinase_store.npz")
        S = dict(zip(*[C.store[k] for k in ("ids", "S")])); print(sorted(S.items(), key=lambda x: -x[1])[:10])

    elif a.cmd == "pretrain":
        from tnk1.models.pretrain import pretrain
        p = Path(cfg["data"]["chembl_unlabelled_smiles"])
        if not p.exists():
            from tnk1.utils.paths import feature_file
            feat_p = feature_file("unique_molecules_788k.parquet")
            if feat_p.exists():
                print(f"Exporting unlabelled SMILES from {feat_p} to {p}...")
                p.parent.mkdir(parents=True, exist_ok=True)
                df_feat = pd.read_parquet(feat_p, columns=["canonical_smiles"])
                df_feat["canonical_smiles"].dropna().to_csv(p, index=False, header=False)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                C.df["smiles"].drop_duplicates().to_csv(p, index=False, header=False)
        smi = [l.split()[0] for l in open(p)]
        pretrain(cfg, smi, C.desc_scaler(), ssl, a.device)

    elif a.cmd == "lambda":
        store, ds = C.store, C.desc_store()
        related = [u for u in cfg["targets"]["focus_kinases"].values() if u in store["ids"]]
        best, res = loko_tune_lambda(cfg, C.kinome_train_df(), store, ds, ssl if ssl.exists() else None, related, a.device)
        pd.Series(res).to_csv(C.runs / "loko_lambda.csv"); print("best lambda", best)

    elif a.cmd == "ablate":
        store, ds = C.store, C.desc_store(); rows = []
        kin_df = C.kinome_train_df()
        tr_all = C.tnk1_rows(C.dev_keys); te = C.tnk1_rows(C.test_keys, censored=False)
        inner = C.splits["cv"][0]["inner"][0]; ids = list(ABLATIONS) if not a.only else a.only
        for aid in ids:
            spec = ABLATIONS[aid]
            for seed in cfg["seeds"]:
                seed_everything(seed)
                tr = tr_all if spec["l2sp"] else tr_all[tr_all.censored == 0]          # censored loss only in A7
                va_keys = set(inner["val"]) & C.dev_keys; trn, va = tr[~tr.inchikey_block1.isin(va_keys)], tr[tr.inchikey_block1.isin(va_keys)]
                if spec["kinome"] is None:
                    mini = {"ids": [C.tnk1], "esm": np.zeros((1, 1), np.float32), "klifs": np.zeros((1, 1), np.float32)}
                    tmp = dict(cfg); tmp["model"] = {**cfg["model"], "design": "A"}
                    from tnk1.models.model import KinomeModel
                    model = KinomeModel(tmp, mini, design="A", n_kinases=1).to(a.device)
                    if spec["ssl"] and ssl.exists(): model.enc.load_state_dict(torch.load(ssl, map_location=a.device))
                    st = mini
                else:
                    lam = cfg["kinome_train"]["lambda_sim"] if spec["weighting"] else 0.0
                    kcfg = cfg if spec["weighting"] else {**cfg, "kinome_train": {**cfg["kinome_train"], "cap_n": 10**9}}
                    model = kinome_stage(kcfg, kin_df, store, ds, ssl if spec["ssl"] and ssl.exists() else None, spec["kinome"], lam, seed, a.device)
                    st = store
                if spec["finetune"]:
                    model = finetune_tnk1(cfg, model, trn, va, st, ds, a.device, seed, use_l2sp=spec["l2sp"], three_phase=spec["three_phase"],
                                          loss_mode="nll" if spec["l2sp"] else "huber")
                kmap = {u: i for i, u in enumerate(st["ids"])}
                mu, _ = predict(model, _mk(te, kmap, ds), a.device)
                rows.append({"ablation": aid, "seed": seed, **full_report(te.pact.values, mu)}); print(rows[-1])
        res = pd.DataFrame(rows); res.to_csv(C.runs / "stage3_ablation_raw.csv", index=False)
        agg = res.groupby("ablation")[["rmse", "mae", "spearman", "pearson"]].agg(["mean", "std"]); agg.to_csv(C.runs / "stage3_ablation.csv"); print(agg)


if __name__ == "__main__":
    main()
