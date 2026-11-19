"""Command line: one sub-command per pipeline step.

    python -m ber <step> [--config configs/default.yaml] [--fold F] [--split S]
                         [--shard I --num-shards N] [key=value ...]

Steps, in pipeline order (see README.md):
    preflight        paths, data files, models, packages, GPU
    prepare          TSV -> normalised parquet, ground-truth pairs, folds, checks   (CPU)
    sparse           lexical TF-IDF candidate retrieval                              (CPU)
    train-retriever  fine-tune bge-m3 bi-encoder on one fold  (--fold)               (GPU)
    dense            dense retrieval: OOF for train, fold-ensemble for test           (GPU)
    features         merge candidates + pairwise features  (--split)                 (CPU)
    prune            GBDT pruner -> candidate set fed to the cross-encoder            (CPU)
    train-ce         fine-tune bge-reranker-v2-m3 cross-encoder on one fold (--fold)  (GPU)
    score-ce         cross-encoder scores (--split, --fold, --shard/--num-shards)     (GPU)
    stack            GBDT stacker + decision tuning + validation + output TSVs        (CPU)
    residual         residual LightGBM over saved stack runs (ensemble), output TSVs (CPU)
    frpost           rule-based corrections for test countries without labels (France) (CPU)
    validate         run the official submission validator on output/
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time

from .config import DEFAULT_CONFIG, load_config, to_json
from .util import Work, log

STEPS = {
    "preflight": "ber.preflight",
    "prepare": "ber.prepare",
    "sparse": "ber.sparse",
    "train-retriever": "ber.retriever",
    "dense": "ber.dense",
    "features": "ber.features",
    "prune": "ber.prune",
    "train-ce": "ber.crossenc",
    "score-ce": "ber.crossenc_score",
    "stack": "ber.stack",
    "residual": "ber.residual",
    "frpost": "ber.frpost",
    "validate": "ber.submit",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ber", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=list(STEPS))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--split", choices=["train", "test"], default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. retriever.epochs=2")
    # intermixed: `stack --split train stack.dir=x` works too - plain parse_args on
    # Python < 3.12 rejects key=value overrides that come after an option
    args = ap.parse_intermixed_args(argv)

    cfg = load_config(args.config, args.overrides)
    W = Work(cfg)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (W.p("runs", f"{args.step}_{stamp}.json")).write_text(to_json(cfg), encoding="utf-8")
    log(f"step={args.step} config={args.config} overrides={args.overrides} work={W.root}")

    module = importlib.import_module(STEPS[args.step])
    module.run(cfg, args)
    log(f"step={args.step} finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
