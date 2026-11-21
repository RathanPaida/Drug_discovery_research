"""SUPERSEDED -- the REINVENT-4 prior path. The pipeline does not use this.

Builds the kinase corpus, writes REINVENT 4 transfer-learning/sampling TOMLs and QCs the
tuned prior. `scripts/run_pipeline.py`'s `prior` stage runs `python -m rl_min.train_prior`
instead, which trains its own SMILES RNN on the 788k store and needs no
`priors/reinvent.prior`, no second conda environment and no TOML.

Kept because it works and produced the v1 prior. See `rl_min/README.md` and
`priors/README.md` for which backend you are actually on.

usage: run_stage6_prior.py {prepare|qc}   (run `reinvent <toml>` in the tnk1-reinvent env between the two)
"""
import argparse, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.context import Ctx
from tnk1.rl import reinvent_io as RI


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--config", default=None)   # None -> tnk1.utils.paths.default_config()
    ap.add_argument("--version", default="v1"); ap.add_argument("--base-prior", default="priors/reinvent.prior"); a = ap.parse_args()
    C = Ctx(a.config, a.version, "cpu"); out = C.proc / "prior"
    if a.cmd == "prepare":
        S = dict(zip(C.store["ids"], C.store["S"]))
        info = RI.make_prior_corpus(C.df, S, C.tnk1, C.test_keys, out); print(info)
        RI.write_transfer_learning_toml(out / "tl.toml", a.base_prior, info["train"], info["val"], str(out / "kinase.prior"))
        RI.write_sampling_toml(out / "sample_prior.toml", str(out / "kinase.prior"), str(out / "prior_samples.csv"))
        print("next:  conda run -n tnk1-reinvent reinvent", out / "tl.toml", "&&  reinvent", out / "sample_prior.toml")
    elif a.cmd == "qc":
        smp = pd.read_csv(out / "prior_samples.csv").iloc[:, 0].tolist()
        tr = [l.strip() for l in open(out / "kinase_unique.smi")]
        qc = RI.prior_qc(smp, tr); print(qc)
        assert qc["validity"] > 0.95 and qc["uniqueness"] > 0.9, "prior fails the Stage 6 gate (validity>95%, uniqueness>90%)"


if __name__ == "__main__":
    main()
