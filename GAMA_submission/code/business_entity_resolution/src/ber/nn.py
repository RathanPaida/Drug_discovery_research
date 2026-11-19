"""Shared torch / transformers helpers for the bi-encoder (bge-m3) and the
cross-encoder (bge-reranker-v2-m3).

Models are loaded strictly from local folders (`local_files_only=True`):
compute nodes are offline, weights are pre-staged under paths.model_dir by
hpc/download_models.sh.
"""
from __future__ import annotations

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .util import log


# -- setup ----------------------------------------------------------------
def setup(seed: int) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # TF32 for fp32 matmuls. torch >= 2.9 replaced allow_tf32 with fp32_precision,
        # and mixing the two APIs is an error there, so use whichever this torch has.
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        dev = torch.device("cuda")
        free, total = torch.cuda.mem_get_info()
        log(f"GPU {torch.cuda.get_device_name(0)}  {free/2**30:.0f}/{total/2**30:.0f} GB free, "
            f"bf16={torch.cuda.is_bf16_supported()}")
        return dev
    log("WARNING: no CUDA device - running on CPU (only sensible for smoke tests)")
    return torch.device("cpu")


def resolve_model(model_dir: str, name_or_path: str) -> Path:
    p = Path(name_or_path)
    if not p.is_absolute():
        p = Path(model_dir) / name_or_path
    if not (p / "config.json").exists():
        raise FileNotFoundError(
            f"no HF model at {p} (expected config.json) - stage it with hpc/download_models.sh "
            f"or point paths.model_dir / BER_MODEL_DIR at the folder that contains it")
    return p


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def from_pretrained(cls, path: Path, dtype: torch.dtype | None = None, **kw):
    """transformers >= 5 renamed `torch_dtype` -> `dtype`: try the new name, fall back to the old one."""
    kw = dict(local_files_only=True, **kw)
    if dtype is None:
        return cls.from_pretrained(str(path), **kw)
    try:
        return cls.from_pretrained(str(path), dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(str(path), torch_dtype=dtype, **kw)


def enable_grad_ckpt(model) -> None:
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()


def autocast(dev: torch.device):
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


# -- batching / tokenisation ----------------------------------------------
def prefetch(fn: Callable, items: Iterator, depth: int = 3) -> Iterator:
    """Run fn(item) in a background thread, `depth` items ahead (tokenisers release the GIL)."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        futs = []
        for it in items:
            futs.append(ex.submit(fn, it))
            if len(futs) > depth:
                yield futs.pop(0).result()
        for f in futs:
            yield f.result()


def to_device(enc: dict, dev: torch.device) -> dict:
    return {k: v.to(dev, non_blocking=True) for k, v in enc.items()}


@torch.inference_mode()
def encode(model, tok, texts: Sequence[str], max_len: int, batch_size: int,
           dev: torch.device, desc: str = "") -> torch.Tensor:
    """L2-normalised CLS embeddings (fp16, CPU), in input order. Longest texts first,
    so padding is minimal and an OOM would surface immediately."""
    n = len(texts)
    dim = model.config.hidden_size
    out = torch.empty((n, dim), dtype=torch.float16)
    if n == 0:
        return out
    order = np.argsort(-np.fromiter((len(t) for t in texts), dtype=np.int64, count=n), kind="stable")
    starts = range(0, n, batch_size)

    def tokenize(s):
        idx = order[s:s + batch_size]
        return idx, tok([texts[i] for i in idx], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")

    t0, done = time.time(), 0
    for idx, enc in prefetch(tokenize, iter(starts)):
        with autocast(dev):
            h = model(**to_device(enc, dev)).last_hidden_state[:, 0]
        out[torch.from_numpy(idx)] = F.normalize(h.float(), dim=-1).half().cpu()
        done += len(idx)
        if done % (batch_size * 500) < batch_size:
            log(f"  encode{desc}: {done:,}/{n:,} ({done / max(time.time() - t0, 1e-6):,.0f}/s)")
    return out


# -- optimisation ---------------------------------------------------------
def make_optimizer(model, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "LayerNorm" in name or "layer_norm" in name else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(groups, lr=lr, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(groups, lr=lr)


def linear_schedule(opt: torch.optim.Optimizer, warmup: int, total: int):
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))
    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def save_resume(path: Path, model, opt, sched, step: int) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": step}, tmp)
    os.replace(tmp, path)
    log(f"  resume checkpoint @ step {step} -> {path}")


def load_resume(path: Path, model, opt, sched) -> int:
    if not path.exists():
        return 0
    st = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    log(f"resumed from {path} at step {st['step']}")
    return int(st["step"])


# -- losses ---------------------------------------------------------------
def contrastive_loss(q: torch.Tensor, d: torch.Tensor, q_owner: torch.Tensor,
                     d_owner: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE with in-batch + hard negatives.

    q        (B, D) normalised query (S1) embeddings
    d        (N, D) normalised docs; d[:B] are the positives (d[i] matches q[i]),
             the rest are hard negatives
    q_owner  (B,)  S1 id of each query
    d_owner  (N,)  S1 id owning each doc (-1 = matches no S1)

    Any doc that truly matches query i other than its designated positive
    (another record of the same S1 in the batch) is masked, never a negative.
    """
    B = q.shape[0]
    tgt = torch.arange(B, device=q.device)
    logits = (q @ d.T) / temperature
    same = d_owner.unsqueeze(0) == q_owner.unsqueeze(1)
    same[tgt, tgt] = False
    loss_q = F.cross_entropy(logits.masked_fill(same, float("-inf")), tgt)
    logits_d = (d[:B] @ q.T) / temperature
    dup = q_owner.unsqueeze(0) == q_owner.unsqueeze(1)
    dup.fill_diagonal_(False)
    loss_d = F.cross_entropy(logits_d.masked_fill(dup, float("-inf")), tgt)
    return (loss_q + loss_d) / 2
