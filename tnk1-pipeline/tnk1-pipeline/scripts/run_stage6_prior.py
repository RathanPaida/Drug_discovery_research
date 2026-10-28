"""Stage 6: build the kinase corpus, write REINVENT 4 transfer-learning/sampling TOMLs, and QC the tuned prior.
usage: run_stage6_prior.py {prepare|qc}   (run `reinvent <toml>` in the tnk1-reinvent env between the two)"""
import argparse, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.context import Ctx
from tnk1.rl import reinvent_io as RI


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--config", default="configs/default.yaml")
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
