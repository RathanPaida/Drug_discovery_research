"""Step `train-ce --fold F`: fine-tune the bge-reranker-v2-m3 cross-encoder.

Same K-fold convention as the retriever: model F trains on S1s with
fold_nn != F and later scores only fold F (and, averaged, the test set).

Training pairs per S1 (from the pruned candidate set = what the model will
see at inference): every true record, the `neg_per_query` hardest non-matches
(by pruner score), and - optionally - true records the retrieval missed.
Pointwise BCE; the logit is later combined with the GBDT features by `stack`.

Output: work/models/ce_f{F}/  (HF format)
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
import torch

from . import nn
from .normalize import model_text
from .util import Work, die, log, save_json

SRC_TAG = {1: "", 2: "[s2] ", 3: "[s3] "}


def pair_texts(W: Work, split: str) -> tuple[list[str], list[str]]:
    """(S1-side text, record-side text) indexed by record idx."""
    rec = pl.read_parquet(W.records(split), columns=["src", "t_name", "t_addr"])
    base = rec.select(model_text(pl.col("t_name"), pl.col("t_addr"))).to_series()
    tag = rec["src"].replace_strict(SRC_TAG, return_dtype=pl.String)
    return base.to_list(), (tag + base).to_list()


def training_pairs(W: Work, cfg, fold: int) -> pl.DataFrame:
    cc = cfg.crossenc
    rec = pl.read_parquet(W.records("train"), columns=["idx", "src", "fold_nn"])
    s1 = rec.filter((pl.col("src") == 1) & (pl.col("fold_nn") != fold)).select(pl.col("idx").alias("q"))
    pr = pl.read_parquet(W.pruned("train")).join(s1, on="q")
    pos = pr.filter(pl.col("label") == 1).select("q", "r", "label")
    neg = (pr.filter(pl.col("label") == 0).sort(["q", "prune_p"], descending=[False, True])
             .group_by("q", maintain_order=True).head(cc.neg_per_query).select("q", "r", "label"))
    parts = [pos, neg]
    if cc.include_missed_positives:
        gt = pl.read_parquet(W.gt()).join(s1, on="q")
        missed = gt.join(pos.select("q", "r"), on=["q", "r"], how="anti").with_columns(pl.lit(1, pl.Int8).alias("label"))
        parts.append(missed.select("q", "r", "label"))
        log(f"  + {missed.height:,} true pairs the retrieval missed")
    return pl.concat([p.with_columns(pl.col("label").cast(pl.Int8)) for p in parts])


def length_batches(pairs: pl.DataFrame, lens: np.ndarray, bs: int, seed: int) -> list[np.ndarray]:
    """Shuffle, then sort by length inside windows of 50 batches -> little padding."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(pairs.height)
    win = bs * 50
    batches = []
    for s in range(0, len(order), win):
        w = order[s:s + win]
        w = w[np.argsort(lens[w], kind="stable")]
        batches += [w[i:i + bs] for i in range(0, len(w), bs)]
    return [batches[i] for i in rng.permutation(len(batches))]


def run(cfg, args) -> None:
    fold = args.fold
    if fold is None:
        die("train-ce needs --fold")
    cc = cfg.crossenc
    W = Work(cfg)
    out_dir = W.model(f"ce_f{fold}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = nn.setup(cfg.seed + 10 + fold)

    pairs = training_pairs(W, cfg, fold)
    log(f"fold {fold}: {pairs.height:,} training pairs, positive rate {pairs['label'].mean():.3f}")
    a_txt, b_txt = pair_texts(W, "train")
    q = pairs["q"].to_numpy().astype(np.int64)
    r = pairs["r"].to_numpy().astype(np.int64)
    y = pairs["label"].to_numpy().astype(np.float32)
    lens = np.fromiter((len(a_txt[i]) + len(b_txt[j]) for i, j in zip(q, r)), dtype=np.int32, count=len(q))

    path = nn.resolve_model(cfg.paths.model_dir, cc.base_model)
    from transformers import AutoModelForSequenceClassification
    tok = nn.load_tokenizer(path)
    model = nn.from_pretrained(AutoModelForSequenceClassification, path, dtype=torch.float32, num_labels=1).to(dev)
    if cc.grad_checkpointing:
        nn.enable_grad_ckpt(model)
    model.train()

    schedule = [b for e in range(int(cc.epochs)) for b in length_batches(pairs, lens, cc.batch_size, cfg.seed + 31 * e + fold)]
    total = len(schedule) if not cc.max_steps else min(len(schedule), cc.max_steps)
    opt = nn.make_optimizer(model, cc.lr, cc.weight_decay)
    sched = nn.linear_schedule(opt, cc.warmup_steps, total)
    resume = out_dir / "resume.pt"
    step0 = nn.load_resume(resume, model, opt, sched) if cc.get("resume", True) else 0
    bce = torch.nn.BCEWithLogitsLoss()
    log(f"training {total:,} steps of {cc.batch_size} pairs, lr={cc.lr}, max_len={cc.max_len}")

    def make_batch(ix: np.ndarray):
        enc = tok([a_txt[i] for i in q[ix]], [b_txt[j] for j in r[ix]], padding=True,
                  truncation="longest_first", max_length=cc.max_len, return_tensors="pt")
        return ix, enc

    t0, run_loss = time.time(), None
    for step, (ix, enc) in enumerate(nn.prefetch(make_batch, iter(schedule[step0:total])), start=step0):
        with nn.autocast(dev):
            logits = model(**nn.to_device(enc, dev)).logits.squeeze(-1)
        loss = bce(logits.float(), torch.from_numpy(y[ix]).to(dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        lv = float(loss.detach())
        run_loss = lv if run_loss is None else 0.98 * run_loss + 0.02 * lv
        if (step + 1) % cc.log_every == 0 or step + 1 == total:
            el = max(time.time() - t0, 1e-6)
            done = step + 1 - step0
            log(f"step {step+1:,}/{total:,} loss {run_loss:.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"{done/el:.2f} it/s eta {el/done*(total-step-1)/60:.0f}m")
        if cc.get("ckpt_every", 0) and (step + 1) % cc.ckpt_every == 0 and step + 1 < total:
            nn.save_resume(resume, model, opt, sched, step + 1)

    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    if resume.exists():
        resume.unlink()
    save_json({"fold": fold, "steps": total, "final_loss": run_loss, "pairs": pairs.height,
               "base_model": str(path)}, out_dir / "train_info.json")
    log(f"saved cross-encoder fold {fold} -> {out_dir}")
