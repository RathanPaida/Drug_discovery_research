"""Step `train-retriever --fold F`: fine-tune the bge-m3 bi-encoder.

K-fold cross-fitting: model F trains on S1 entities with fold_nn != F and is
later applied (step `dense`) only to S1 entities with fold_nn == F, so every
training S1 gets out-of-fold candidates whose scores behave like test ones.

Batches: `batch_queries` unique S1s of one country, each with one true record
and `hard_negatives` non-matching records sampled from its top sparse
candidates (the lookalikes that make this task hard). Records of held-out
S1s (fold_nn == F) never appear, not even as negatives.

Output: work/models/retriever_f{F}/  (HF format: config, weights, tokenizer)
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
import torch

from . import nn
from .normalize import model_text
from .util import Work, die, log, save_json


def training_pairs(W: Work, fold: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    rec = pl.read_parquet(W.records("train"), columns=["idx", "src", "country", "fold_nn", "owner"])
    gt = pl.read_parquet(W.gt())
    s1 = rec.filter((pl.col("src") == 1) & (pl.col("fold_nn") != fold)).select(pl.col("idx").alias("q"), "country")
    pos = gt.join(s1, on="q")
    # hard-negative pool: best-ranked sparse candidates that are not true matches
    # and do not belong to a held-out S1
    ok_rec = rec.filter((pl.col("src") != 1) & (pl.col("fold_nn") != fold)).select(
        pl.col("idx").alias("r"), pl.col("owner"))
    sp = pl.read_parquet(W.sparse("train"))
    rk = [c for c in sp.columns if c.startswith("rk_")]
    pool = (sp.join(s1.select("q"), on="q")
              .join(ok_rec, on="r")
              .filter(pl.col("owner").is_null() | (pl.col("owner") != pl.col("q")))
              .with_columns(pl.min_horizontal(rk).alias("best"))
              .select("q", "r", "best"))
    return pos, pool


def epoch_batches(pos: pl.DataFrame, bq: int, seed: int) -> list[pl.DataFrame]:
    """Single-country batches of positives; an S1's k-th pair lands in 'round' k,
    so the same S1 rarely appears twice in a batch (and is masked if it does)."""
    f = (pos.with_columns(pl.Series("u", np.random.default_rng(seed).random(pos.height)))
            .sort("u").with_columns(pl.int_range(pl.len()).over("q").alias("round"))
            .sort("country", "round", "u"))
    batches = []
    for _, g in f.group_by("country", maintain_order=True):
        cut = [g.slice(i, bq) for i in range(0, g.height, bq)]
        if len(cut) > 1 and cut[-1].height < max(2, bq // 4):   # fold a tiny tail into its neighbour
            tail = cut.pop()
            cut[-1] = pl.concat([cut[-1], tail])
        batches += cut
    order = np.random.default_rng(seed + 1).permutation(len(batches))
    return [batches[i] for i in order]


def run(cfg, args) -> None:
    fold = args.fold
    if fold is None:
        die("train-retriever needs --fold")
    rc = cfg.retriever
    W = Work(cfg)
    out_dir = W.model(f"retriever_f{fold}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = nn.setup(cfg.seed + fold)

    pos, pool = training_pairs(W, fold)
    log(f"fold {fold}: {pos.height:,} positive pairs over {pos['q'].n_unique():,} S1; "
        f"hard-negative pool {pool.height:,} pairs")
    # compact hard-negative pools: per S1, the hard_neg_pool best-ranked candidates
    pool = (pool.sort("q", "best").group_by("q", maintain_order=True)
                .head(rc.hard_neg_pool))
    pq = pool["q"].to_numpy().astype(np.int64)
    pr = pool["r"].to_numpy().astype(np.int64)
    starts = np.searchsorted(pq, np.arange(pq.max() + 2 if pq.size else 1))

    rec = pl.read_parquet(W.records("train"), columns=["idx", "src", "country", "t_name", "t_addr", "owner", "fold_nn"])
    texts = rec.select(model_text(pl.col("t_name"), pl.col("t_addr"))).to_series().to_list()
    owner = rec["owner"].fill_null(-1).cast(pl.Int64).to_numpy()
    # random-negative fallback (S1s without sparse candidates): never a held-out record
    country_recs = {c: g["idx"].to_numpy().astype(np.int64) for (c,), g in
                    rec.filter((pl.col("src") != 1) & (pl.col("fold_nn") != fold)).group_by("country")}
    del rec

    path = nn.resolve_model(cfg.paths.model_dir, rc.base_model)
    from transformers import AutoModel
    tok = nn.load_tokenizer(path)
    model = nn.from_pretrained(AutoModel, path, dtype=torch.float32).to(dev)
    if rc.grad_checkpointing:
        nn.enable_grad_ckpt(model)
    model.train()

    epochs = int(rc.epochs)
    per_epoch = [epoch_batches(pos, rc.batch_queries, cfg.seed + 100 * e + fold) for e in range(epochs)]
    schedule = [b for ep in per_epoch for b in ep]
    total = len(schedule) if not rc.max_steps else min(len(schedule), rc.max_steps)
    opt = nn.make_optimizer(model, rc.lr, rc.weight_decay)
    sched = nn.linear_schedule(opt, rc.warmup_steps, total)
    resume = out_dir / "resume.pt"
    step0 = nn.load_resume(resume, model, opt, sched) if rc.get("resume", True) else 0
    log(f"training {total:,} steps ({epochs} epoch(s), {rc.batch_queries} queries x "
        f"(1 + {rc.hard_negatives}) docs), lr={rc.lr}, T={rc.temperature}")

    rng = np.random.default_rng(cfg.seed + 7 * fold)
    H = int(rc.hard_negatives)

    def make_batch(b: pl.DataFrame):
        q = b["q"].to_numpy().astype(np.int64)
        r = b["r"].to_numpy().astype(np.int64)
        negs = np.empty((len(q), H), dtype=np.int64)
        crec = country_recs[b["country"][0]]
        for i, qq in enumerate(q):
            lo, hi = (starts[qq], starts[qq + 1]) if qq + 1 < len(starts) else (0, 0)
            negs[i] = pr[lo + rng.integers(0, hi - lo, H)] if hi > lo else crec[rng.integers(0, len(crec), H)]
        docs = np.concatenate([r, negs.ravel()])
        enc = tok([texts[i] for i in q] + [texts[i] for i in docs], padding=True, truncation=True,
                  max_length=rc.max_len, return_tensors="pt")
        return q, docs, enc

    t0, run_loss = time.time(), None
    steps = iter(schedule[step0:total])
    for step, (q, docs, enc) in enumerate(nn.prefetch(make_batch, steps), start=step0):
        with nn.autocast(dev):
            h = model(**nn.to_device(enc, dev)).last_hidden_state[:, 0]
        e = torch.nn.functional.normalize(h.float(), dim=-1)
        loss = nn.contrastive_loss(
            e[:len(q)], e[len(q):],
            torch.as_tensor(q, device=dev), torch.as_tensor(owner[docs], device=dev),
            rc.temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        lv = float(loss.detach())
        run_loss = lv if run_loss is None else 0.98 * run_loss + 0.02 * lv
        if (step + 1) % rc.log_every == 0 or step + 1 == total:
            el = max(time.time() - t0, 1e-6)
            done = step + 1 - step0
            eta = el / done * (total - step - 1) / 60
            log(f"step {step+1:,}/{total:,} loss {run_loss:.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"{done/el:.2f} it/s eta {eta:.0f}m")
        if rc.get("ckpt_every", 0) and (step + 1) % rc.ckpt_every == 0 and step + 1 < total:
            nn.save_resume(resume, model, opt, sched, step + 1)

    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    if resume.exists():
        resume.unlink()
    save_json({"fold": fold, "steps": total, "final_loss": run_loss, "base_model": str(path),
               "positives": pos.height}, out_dir / "train_info.json")
    log(f"saved retriever fold {fold} -> {out_dir}")
