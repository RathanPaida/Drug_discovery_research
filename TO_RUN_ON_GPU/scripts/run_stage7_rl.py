"""SUPERSEDED -- the external-REINVENT-4 backend. The pipeline does not use this.

`scripts/run_pipeline.py`'s `rl` stage runs `python -m rl_min.run_rl`, in process, with
eight selectable objectives (`--rl-algo`). This script is the older two-environment
arrangement: it writes a REINVENT 4 staged-learning TOML and serves the surrogate over
HTTP so a separate `reinvent` process can call it.

It is kept because it still works and because the v1 results were produced with it -- not
as the recommended path. Prefer `rl_min` unless you specifically need the REINVENT 4
generator, and read `rl_min/README.md` first: the v1 Stage 7 run it fed **performed
selection, not reinforcement learning** (12,000 molecules sampled from the prior, scored
once, best kept; no generator parameter was ever updated), which is the defect `rl_min`
was written to fix. Nothing in THIS script causes that -- it only writes config and
serves scores -- but a run built on it needs its staged-learning section checked before
any claim of reinforcement learning is made about the output.

Usage:
  run_stage7_rl.py serve    # (tnk1-predict env) start the LCB scoring service
  run_stage7_rl.py config   # write the REINVENT 4 staged-learning TOML
then in tnk1-reinvent env:  reinvent runs/rl/rl.toml
"""
import argparse, pickle, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.context import Ctx
from tnk1.rl import reinvent_io as RI
from tnk1.rl.scorer import Scorer
from tnk1.rl.server import serve


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--config", default=None)   # None -> tnk1.utils.paths.default_config()
    ap.add_argument("--version", default="v1"); ap.add_argument("--device", default="cuda"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--reinvent-python", default="python"); a = ap.parse_args()
    # Say so on every invocation. The filename follows the stage numbering, so someone
    # working through the stages in order lands here naturally and would not otherwise
    # learn that the pipeline took a different route.
    print("NOTE: this is the superseded external-REINVENT-4 backend. The pipeline's `rl` "
          "stage runs `python -m rl_min.run_rl` instead (see rl_min/README.md).",
          file=sys.stderr)
    C = Ctx(a.config, a.version, a.device); cfg = C.cfg
    if a.cmd == "config":
        # C.runs is absolute (load_config resolves paths.runs against the repo root), so the
        # TOML no longer lands in whatever directory the operator happened to launch from.
        d = C.runs / "rl"; d.mkdir(parents=True, exist_ok=True)
        from tnk1.rl.reinvent_client import client_path
        RI.write_rl_toml(d / "rl.toml", cfg, str(C.proc / "prior/kinase.prior"), str(C.proc / "prior/kinase.prior"),
                         a.reinvent_python, str(client_path()), f"http://127.0.0.1:{a.port}")
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
