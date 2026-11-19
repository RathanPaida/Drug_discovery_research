"""Step `score-ce --split S [--fold F] [--shard I --num-shards N]`: cross-encoder logits.

train  model F scores the pruned candidates of S1s with fold_nn == F (out-of-fold)
test   model F scores every pruned test pair; `stack` averages the fold logits
Omitting --fold runs all folds in sequence. Shards split the pairs by S1
(q % N == I) so large runs can be spread over a SLURM array.

Output: work/ce/{split}_ce_f{F}_s{I}.parquet  (q, r, ce_logit)
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
import torch

from . import nn
from .crossenc import pair_texts
from .util import Work, die, log, only_countries


@torch.inference_mode()
def score_pairs(model, tok, a_txt, b_txt, q: np.ndarray, r: np.ndarray, cfg, dev) -> np.ndarray:
    cc = cfg.crossenc
    n = len(q)
    out = np.empty(n, dtype=np.float32)
    lens = np.fromiter((len(a_txt[i]) + len(b_txt[j]) for i, j in zip(q, r)), dtype=np.int32, count=n)
    order = np.argsort(-lens, kind="stable")
    bs = int(cc.score_batch)

    def tokenize(s):
        ix = order[s:s + bs]
        return ix, tok([a_txt[i] for i in q[ix]], [b_txt[j] for j in r[ix]], padding=True,
                       truncation="longest_first", max_length=cc.max_len, return_tensors="pt")

    t0, done = time.time(), 0
    for ix, enc in nn.prefetch(tokenize, iter(range(0, n, bs))):
        with nn.autocast(dev):
            logits = model(**nn.to_device(enc, dev)).logits.squeeze(-1)
        out[ix] = logits.float().cpu().numpy()
        done += len(ix)
        if done % (bs * 400) < bs:
            el = max(time.time() - t0, 1e-6)
            log(f"  scored {done:,}/{n:,} ({done/el:,.0f} pairs/s, eta {(n-done)/max(done/el,1)/60:.0f}m)")
    return out


def run(cfg, args) -> None:
    split = args.split or die("score-ce needs --split")
    W = Work(cfg)
    dev = nn.setup(cfg.seed)
    K = int(cfg.data.n_folds_nn)
    folds = [args.fold] if args.fold is not None else list(range(K))
    pairs = pl.read_parquet(W.pruned(split), columns=["q", "r"])
    if args.num_shards > 1:
        pairs = pairs.filter(pl.col("q") % args.num_shards == args.shard)
    rec_fold = pl.read_parquet(W.records(split), columns=["fold_nn"])["fold_nn"].to_numpy()
    only = only_countries(cfg, split)
    rec_cty = pl.read_parquet(W.records(split), columns=["country"])["country"].to_numpy() if only else None
    a_txt, b_txt = pair_texts(W, split)
    from transformers import AutoModelForSequenceClassification

    for f in folds:
        sub = pairs.filter(pl.Series(rec_fold[pairs["q"].to_numpy()] == f)) if split == "train" else pairs
        reuse = None
        if only:
            # partial rerun: pairs of the other countries keep their previous score (their
            # texts did not change); the listed countries and any new pair are scored
            path = W.ce_scores(split, f, args.shard)
            if not path.exists():
                die(f"data.only_countries={only} needs the previous {path} for the other countries")
            other = sub.filter(~pl.Series(np.isin(rec_cty[sub["q"].to_numpy()], only)))
            reuse = other.join(pl.read_parquet(path, columns=["q", "r", "ce_logit"]), on=["q", "r"], how="inner")
            sub = sub.join(reuse.select("q", "r"), on=["q", "r"], how="anti")
            log(f"score-ce {split} fold {f}: reusing {reuse.height:,} scores of countries outside {only}")
        log(f"score-ce {split} fold {f} shard {args.shard}/{args.num_shards}: {sub.height:,} pairs")
        model = nn.from_pretrained(AutoModelForSequenceClassification, W.model(f"ce_f{f}"),
                                   dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32).to(dev).eval()
        tok = nn.load_tokenizer(W.model(f"ce_f{f}"))
        q = sub["q"].to_numpy().astype(np.int64)
        r = sub["r"].to_numpy().astype(np.int64)
        logit = score_pairs(model, tok, a_txt, b_txt, q, r, cfg, dev) if len(q) else np.empty(0, dtype=np.float32)
        out = sub.with_columns(pl.Series("ce_logit", logit))
        if reuse is not None:
            out = pl.concat([reuse.select(out.columns), out])
        out.write_parquet(W.ce_scores(split, f, args.shard))
        log(f"  -> {W.ce_scores(split, f, args.shard)}")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
