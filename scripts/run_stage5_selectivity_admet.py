"""Stage 5: train the 7 ADMET ensembles; selectivity sanity check on pan-kinase inhibitors."""
import argparse, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.context import Ctx
from tnk1.uq.ensemble import Ensemble
from tnk1.selectivity.selectivity import selectivity_stats, sanity_check_pan_kinase


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--version", default="v1"); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    C = Ctx(a.config, a.version, a.device); cfg = C.cfg
    if a.cmd == "admet":
        from tnk1.admet.admet import train_endpoint
        sc = C.desc_scaler(); bundles = {e: train_endpoint(cfg, e, sc, a.device) for e in cfg["admet"]["endpoints"]}
        torch.save(bundles, C.proc / "admet_bundles.pt")
    elif a.cmd == "selectivity":
        ens = Ensemble(cfg, C.store, C.desc_store(), C.proc / "ensemble", a.device)
        ref = pd.read_csv("data/raw/reference_panel.csv")                  # name,smiles  (dasatinib, bosutinib, TP-0903, ...)
        panel = [C.tnk1] + cfg["offtarget_panel"]
        mu_m, _ = ens.predict_members(list(ref.smiles), panel)
        s = selectivity_stats(mu_m[..., 0], mu_m[..., 1:]); ref["mu_sel"], ref["sigma_sel"], ref["fold"] = s["mu_sel"], s["sigma_sel"], s["fold"]
        print(ref); print(sanity_check_pan_kinase(dict(zip(ref.name.str.lower(), ref.mu_sel))))
        ref.to_csv(C.runs / "stage5_selectivity_reference.csv", index=False)


if __name__ == "__main__":
    main()
