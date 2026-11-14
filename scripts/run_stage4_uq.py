"""Stage 4: train the 10-member ensemble (A7), calibrate (temperature + conformal), fit AD, compare UQ methods.
usage: run_stage4_uq.py {train|calibrate|compare}"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.context import Ctx
from tnk1.uq.ensemble import train_ensemble, Ensemble
from tnk1.uq.decomposition import decompose, mc_dropout_summary
from tnk1.uq.calibration import Calibrator
from tnk1.uq.applicability import ApplicabilityDomain
from tnk1.uq.metrics import full_report
from tnk1.baselines.features import morgan_count


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--version", default="v1"); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    C = Ctx(a.config, a.version, a.device); cfg = C.cfg; ck = C.proc / "ensemble"; ssl = C.proc / "ssl_encoder.pt"
    store, ds = C.store, C.desc_store()

    if a.cmd == "train":
        inner = C.splits["cv"][0]["inner"][0]
        tr = C.tnk1_rows(C.dev_keys); va_keys = set(inner["val"])
        train_ensemble(cfg, C.kinome_train_df(), tr[~tr.inchikey_block1.isin(va_keys)], tr[tr.inchikey_block1.isin(va_keys)],
                       store, ds, ssl if ssl.exists() else None, ck, a.device)

    elif a.cmd == "oof":
        # Out-of-fold predictions for calibration: for each outer fold, train a (smaller) ensemble that never sees
        # that fold's compounds (also removed from the kinome table), then predict the fold.
        import shutil
        from tnk1.uq.ensemble import train_member
        n_oof = int(cfg["uq"].get("oof_members", 5)); rows = []
        for f in C.splits["cv"]:
            te_keys, tr_keys = set(f["test"]), set(f["train"]); inner = f["inner"][0]; va = set(inner["val"])
            kin_df = C.df[~C.df.inchikey_block1.isin(C.test_keys | te_keys)].reset_index(drop=True)
            tr = C.tnk1_rows(tr_keys); mus, lvs = [], []
            shared = None
            for m in range(n_oof):
                mem = train_member(cfg, kin_df, tr[~tr.inchikey_block1.isin(va)], tr[tr.inchikey_block1.isin(va)], store, ds,
                                   ssl if ssl.exists() else None, 5000 + 10 * f["outer"] + m, a.device, shared)
                d = C.runs / f"oof_tmp_{f['outer']}"; d.mkdir(exist_ok=True); torch.save(mem.state_dict(), d / f"member_{m:02d}.pt")
            e = Ensemble(cfg, store, ds, d, a.device); te = C.tnk1_rows(te_keys, censored=False)
            mu_m, lv_m = e.predict_members(list(te.smiles), [C.tnk1]); dd = decompose(mu_m[..., 0], lv_m[..., 0])
            rows.append(pd.DataFrame({"fold": f["outer"], "y": te.pact.values, "mu": dd["mu"], "sigma_tot": dd["sigma_tot"], "sigma_epi": dd["sigma_epi"]}))
            shutil.rmtree(d)
        pd.concat(rows).to_csv(C.runs / "stage4_oof_predictions.csv", index=False)

    elif a.cmd == "calibrate":
        # Out-of-fold calibration set: the 5 outer-fold test sets, each scored by an ensemble trained WITHOUT it
        # (run `train` per outer fold with --fold and merge; here we calibrate on the supplied OOF file).
        oof = pd.read_csv(C.runs / "stage4_oof_predictions.csv")   # columns: y, mu, sigma_tot (from the 5 outer folds)
        cal = Calibrator(cfg["uq"]["alpha"]).fit(oof.y.values, oof.mu.values, oof.sigma_tot.values)
        pickle.dump(cal, open(C.proc / "calibrator.pkl", "wb")); print(f"tau={cal.tau:.3f}  q_hat={cal.q:.3f}")
        ens = Ensemble(cfg, store, ds, ck, a.device); tr = C.tnk1_rows(C.dev_keys, censored=False)
        Z, FP = ens.latent(list(tr.smiles)), morgan_count(tr.smiles)
        ad = ApplicabilityDomain(cfg["uq"]["ad_k"], cfg["uq"]["ad_percentile"]).fit(Z, FP)
        pickle.dump(ad, open(C.proc / "ad.pkl", "wb")); print("AD thresholds", ad.d_thr_lat, ad.d_thr_fp)

    elif a.cmd == "compare":
        ens, cal = Ensemble(cfg, store, ds, ck, a.device), pickle.load(open(C.proc / "calibrator.pkl", "rb"))
        te = C.tnk1_rows(C.test_keys, censored=False)
        mu_m, lv_m = ens.predict_members(list(te.smiles), [C.tnk1])
        d = decompose(mu_m[..., 0], lv_m[..., 0]); y = te.pact.values
        rep = {"ensemble_raw": full_report(y, d["mu"], d["sigma_tot"]), "ensemble_cal": full_report(y, d["mu"], cal.sigma(d["sigma_tot"]))}
        from tnk1.models.trainer import predict
        from tnk1.models.pipeline import _mk
        kmap = {u: i for i, u in enumerate(store["ids"])}; lab = _mk(te, kmap, ds)
        passes = np.stack([predict(ens.members[0], lab, a.device, mc_dropout=True)[0] for _ in range(cfg["uq"]["mc_dropout_T"])])
        s = mc_dropout_summary(passes); rep["mc_dropout"] = full_report(y, s["mu"], s["sigma_epi"])
        pd.DataFrame(rep).T.to_csv(C.runs / "stage4_uq_comparison.csv"); print(pd.DataFrame(rep).T.round(3))
        # Tanimoto-GP audit scorer is evaluated in Stage 2; evidential head: train with head_type='evidential' via tnk1.models.pipeline


if __name__ == "__main__":
    main()
