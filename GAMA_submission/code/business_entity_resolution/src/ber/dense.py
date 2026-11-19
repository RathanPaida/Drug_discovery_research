"""Step `dense`: bge-m3 retrieval (GPU), exact inner-product top-k per country.

train  model F encodes every train record and the S1s with fold_nn == F
       (out-of-fold: F never trained on them).
test   all K fold models; per-model embeddings are concatenated and scaled by
       1/sqrt(K), so one search ranks by the mean cosine over folds.

Every sparse candidate pair also gets its dense cosine here, so each pair of
the union carries a dense score for the ranking models.

Output: work/dense/{split}_dense.parquet  q, r, dense_sim, dense_rank (null = only sparse found it)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import torch

from . import nn
from .normalize import model_text
from .util import Work, die, log, only_countries, timed


@torch.inference_mode()
def topk_search(Q: torch.Tensor, R: torch.Tensor, k: int, chunk: int, dev: torch.device):
    """Exact top-k of Q @ R.T; Q, R are CPU fp16. Returns numpy (q_pos, r_pos, sim)."""
    k = min(k, R.shape[0])
    dt = torch.float16 if dev.type == "cuda" else torch.float32
    Rg = R.to(dev, dtype=dt)
    qs, rs, ss = [], [], []
    for s in range(0, Q.shape[0], chunk):
        sims = Q[s:s + chunk].to(dev, dtype=dt) @ Rg.T
        v, i = torch.topk(sims, k, dim=1)
        qs.append(np.repeat(np.arange(s, s + v.shape[0]), k))
        rs.append(i.reshape(-1).cpu().numpy())
        ss.append(v.float().reshape(-1).cpu().numpy())
        del sims
    del Rg
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(qs), np.concatenate(rs), np.concatenate(ss)


@torch.inference_mode()
def pair_sims(Q: torch.Tensor, R: torch.Tensor, qpos: np.ndarray, rpos: np.ndarray,
              dev: torch.device, chunk_bytes: int = 2 << 30) -> np.ndarray:
    """<Q[qpos[i]], R[rpos[i]]> for each i. Chunks hold ~chunk_bytes of fp32 per operand,
    whatever the width: the test ensemble has K x 1024 dims, and a fixed row count
    sized for one model ran out of GPU memory there (4M x 2048 fp32 = 30 GB per copy)."""
    out = np.empty(len(qpos), dtype=np.float32)
    chunk = max(1, chunk_bytes // (4 * Q.shape[1]))
    for s in range(0, len(qpos), chunk):
        a = Q[torch.from_numpy(qpos[s:s + chunk])].to(dev).float()
        b = R[torch.from_numpy(rpos[s:s + chunk])].to(dev).float()
        out[s:s + chunk] = (a * b).sum(1).cpu().numpy()
        del a, b
    return out


def _newest(path) -> float:
    """mtime of a file, or of the newest file in a folder; inf if missing."""
    p = Path(path)
    if p.is_dir():
        return max((c.stat().st_mtime for c in p.iterdir() if c.is_file()), default=math.inf)
    return p.stat().st_mtime if p.exists() else math.inf


def _up_to_date(out, inputs) -> bool:
    """`out` exists and is newer than every input: a re-run after a crash can keep it."""
    out = Path(out)
    return out.exists() and all(_newest(i) < out.stat().st_mtime for i in inputs)


def _positions(idx: np.ndarray, size: int) -> np.ndarray:
    pos = np.full(size, -1, dtype=np.int64)
    pos[idx] = np.arange(len(idx))
    return pos


def retrieve(models, tok, texts, rec: pl.DataFrame, q_idx: np.ndarray, r_idx: np.ndarray,
             sparse_pairs: pl.DataFrame, cfg, dev, tag: str) -> pl.DataFrame:
    """Encode (ensemble of `models`), search, and score the sparse pairs of these queries."""
    dc, rc = cfg.dense, cfg.retriever
    K = len(models)
    dim = models[0].config.hidden_size
    # fold ensemble: per-model embeddings side by side in one preallocated array
    # (no concat copy -> peak RAM = final array + one model's block)
    R = torch.empty((len(r_idx), dim * K), dtype=torch.float16)
    Q = torch.empty((len(q_idx), dim * K), dtype=torch.float16)
    r_txt, q_txt = [texts[i] for i in r_idx], [texts[i] for i in q_idx]
    for f, m in enumerate(models):
        R[:, f * dim:(f + 1) * dim] = nn.encode(m, tok, r_txt, rc.max_len, dc.encode_batch, dev, f" {tag} records")
        Q[:, f * dim:(f + 1) * dim] = nn.encode(m, tok, q_txt, rc.max_len, dc.encode_batch, dev, f" {tag} S1")
    del r_txt, q_txt
    if K > 1:                     # scale so <q, r> = mean cosine over the fold models
        R.mul_(1.0 / math.sqrt(K))
        Q.mul_(1.0 / math.sqrt(K))

    with timed(f"dense {tag}: top-{cfg.dense.top_k} for {len(q_idx):,} S1 over {len(r_idx):,} records"):
        qp, rp, s = topk_search(Q, R, cfg.dense.top_k, cfg.dense.query_chunk, dev)
    top = pl.DataFrame({"q": q_idx[qp].astype(np.uint32), "r": r_idx[rp].astype(np.uint32),
                        "dense_sim": s.astype(np.float32)})
    top = top.with_columns(pl.col("dense_sim").rank("ordinal", descending=True).over("q")
                           .cast(pl.UInt16).alias("dense_rank"))

    size = int(rec.height)
    qpos_all, rpos_all = _positions(q_idx, size), _positions(r_idx, size)
    extra = (sparse_pairs.join(pl.DataFrame({"q": q_idx.astype(np.uint32)}), on="q", how="semi")
                         .join(top.select("q", "r"), on=["q", "r"], how="anti"))
    if extra.height:
        qp2 = qpos_all[extra["q"].to_numpy().astype(np.int64)]
        rp2 = rpos_all[extra["r"].to_numpy().astype(np.int64)]
        ok = (qp2 >= 0) & (rp2 >= 0)
        sims = np.full(extra.height, np.nan, dtype=np.float32)
        sims[ok] = pair_sims(Q, R, qp2[ok], rp2[ok], dev)
        extra = extra.with_columns(pl.Series("dense_sim", sims), pl.lit(None, pl.UInt16).alias("dense_rank"))
        top = pl.concat([top, extra.select(top.columns)])
    return top


def run(cfg, args) -> None:
    W = Work(cfg)
    dev = nn.setup(cfg.seed)
    K = int(cfg.data.n_folds_nn)
    splits = [args.split] if getattr(args, "split", None) else ["train", "test"]
    from transformers import AutoModel

    def load(f):
        p = W.model(f"retriever_f{f}")
        m = nn.from_pretrained(AutoModel, p, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32)
        return m.to(dev).eval()

    tok = nn.load_tokenizer(W.model("retriever_f0"))
    for split in splits:
        # a split finished before an interruption is kept (unless asked for by --split);
        # it is redone when the records, sparse pairs or retriever models are newer
        inputs = [W.records(split), W.sparse(split)] + [W.model(f"retriever_f{f}") for f in range(K)]
        if not getattr(args, "split", None) and _up_to_date(W.dense(split), inputs):
            log(f"dense {split}: {W.dense(split)} is newer than its inputs - kept (delete it to recompute)")
            continue
        rec = pl.read_parquet(W.records(split), columns=["idx", "src", "country", "t_name", "t_addr", "fold_nn"])
        texts = rec.select(model_text(pl.col("t_name"), pl.col("t_addr"))).to_series().to_list()
        sparse_pairs = pl.read_parquet(W.sparse(split), columns=["q", "r"])
        out = []
        if split == "train":
            # out-of-fold: model f only ever sees S1s of fold f at inference
            for f in range(K):
                model = load(f)
                for (country,), g in rec.group_by("country", maintain_order=True):
                    q_idx = g.filter((pl.col("src") == 1) & (pl.col("fold_nn") == f))["idx"].to_numpy().astype(np.int64)
                    r_idx = g.filter(pl.col("src") != 1)["idx"].to_numpy().astype(np.int64)
                    if len(q_idx) and len(r_idx):
                        out.append(retrieve([model], tok, texts, rec, q_idx, r_idx, sparse_pairs,
                                            cfg, dev, f"train/{country}/f{f}"))
                del model
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
        else:
            only = only_countries(cfg, split)
            models = [load(f) for f in range(K)]
            for (country,), g in rec.group_by("country", maintain_order=True):
                if only and country not in only:
                    continue
                q_idx = g.filter(pl.col("src") == 1)["idx"].to_numpy().astype(np.int64)
                r_idx = g.filter(pl.col("src") != 1)["idx"].to_numpy().astype(np.int64)
                if len(q_idx) and len(r_idx):
                    out.append(retrieve(models, tok, texts, rec, q_idx, r_idx, sparse_pairs,
                                        cfg, dev, f"test/{country}"))
            del models
            if only:
                if not W.dense(split).exists():
                    die(f"data.only_countries={only} needs the previous {W.dense(split)} for the other countries")
                # the other countries' inputs did not change: keep their rows from the previous run
                ctry = rec.select(pl.col("idx").cast(pl.UInt32).alias("q"), "country")
                old = (pl.read_parquet(W.dense(split)).join(ctry, on="q", how="left")
                         .filter(~pl.col("country").is_in(only)).drop("country"))
                log(f"dense {split}: recomputed {only}, kept {old.height:,} pairs of the other countries")
                out.append(old)
        res = pl.concat([o.select(out[0].columns) for o in out])
        res.write_parquet(W.dense(split))
        log(f"dense {split}: {res.height:,} pairs -> {W.dense(split)}")

        if split == "train" and W.gt().exists():
            from .metrics import candidate_report
            from .util import save_json
            gt = pl.read_parquet(W.gt())
            qs = rec.filter(pl.col("src") == 1)["idx"]
            dense_only = res.filter(pl.col("dense_rank").is_not_null())
            union = pl.concat([dense_only.select("q", "r"), sparse_pairs]).unique()
            rep = {"dense": candidate_report(dense_only, gt, qs), "dense+sparse": candidate_report(union, gt, qs)}
            for k in (5, 10, 20, 50):
                rep[f"dense@{k}"] = candidate_report(dense_only.filter(pl.col("dense_rank") <= k), gt, qs)
            save_json(rep, W.p("dense", "train_report.json"))
            for k, v in rep.items():
                log(f"  recall[{k}] pair={v['pair_recall']:.4f} oracleF={v['oracle_macro_f']:.4f} pairs/q={v['pairs_per_query_mean']}")
