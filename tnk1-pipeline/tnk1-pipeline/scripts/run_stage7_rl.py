"""Stage 7: usage
  run_stage7_rl.py serve    # (tnk1-predict env) start the LCB scoring service
  run_stage7_rl.py config   # write the REINVENT 4 staged-learning TOML
then in tnk1-reinvent env:  reinvent runs/rl/rl.toml"""
import argparse, pickle, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.context import Ctx
from tnk1.rl import reinvent_io as RI
from tnk1.rl.scorer import Scorer
from tnk1.rl.server import serve


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--version", default="v1"); ap.add_argument("--device", default="cuda"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--reinvent-python", default="python"); a = ap.parse_args()
    C = Ctx(a.config, a.version, a.device); cfg = C.cfg
    if a.cmd == "config":
        d = Path("runs/rl"); d.mkdir(parents=True, exist_ok=True)
        RI.write_rl_toml(d / "rl.toml", cfg, str(C.proc / "prior/kinase.prior"), str(C.proc / "prior/kinase.prior"),
                         a.reinvent_python, str(Path("tnk1/rl/reinvent_client.py").resolve()), f"http://127.0.0.1:{a.port}")
        print("wrote", d / "rl.toml")
    elif a.cmd == "serve":
        from tnk1.uq.ensemble import Ensemble
        from tnk1.baselines.features import morgan_count
        ens = Ensemble(cfg, C.store, C.desc_store(), C.proc / "ensemble", a.device)
        cal, ad = pickle.load(open(C.proc / "calibrator.pkl", "rb")), pickle.load(open(C.proc / "ad.pkl", "rb"))
        bundles = torch.load(C.proc / "admet_bundles.pt", weights_only=False)
        ref = lambda smi, e: (e.latent(smi), morgan_count(smi))
        serve(Scorer(cfg, ens, cal, ad, ref, bundles, C.desc_scaler(), a.device), port=a.port)


if __name__ == "__main__":
    main()
