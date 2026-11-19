"""Step `preflight`: checks before burning a GPU allocation. Safe on the login node.

    data files present, config resolves, work/output writable,
    required packages importable, pre-staged models + tokenizers load offline,
    and - when a GPU is visible - device, bf16 and free memory.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

from . import io
from .util import log

REQUIRED = ["polars", "numpy", "scipy", "sklearn", "yaml", "torch", "transformers"]
RECOMMENDED = {"lightgbm": "GBDT backend (falls back to sklearn, slower)",
               "rapidfuzz": "string features (pure-python fallback is ~100x slower)",
               "sparse_dot_topn": "multi-threaded sparse top-k (scipy fallback is single-threaded)",
               "anyascii": "Indic romanisation fallback (unidecode also works)"}


def _version(mod: str) -> str:
    m = importlib.import_module(mod)
    return str(getattr(m, "__version__", "ok"))


def run(cfg, args) -> None:
    problems, warnings = [], []
    log(f"root   : {cfg.paths.root}")
    log(f"data   : {cfg.paths.data_dir}")
    log(f"work   : {cfg.paths.work_dir}")
    log(f"models : {cfg.paths.model_dir}")

    for split in ("train", "test"):
        for s in (1, 2, 3):
            p = io.source_path(cfg.paths.data_dir, split, s)
            if not p.exists():
                problems.append(f"missing data file {p}")
    if not (Path(cfg.paths.data_dir) / "train" / "train_ground_truth.tsv").exists():
        problems.append("missing train_ground_truth.tsv")

    for d in (cfg.paths.work_dir, cfg.paths.output_dir):
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
            probe = Path(d) / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
        except OSError as e:
            problems.append(f"cannot write to {d}: {e}")

    for m in REQUIRED:
        try:
            log(f"  {m:16s} {_version(m)}")
        except Exception as e:
            problems.append(f"required package '{m}' not importable ({e.__class__.__name__})")
    for m, why in RECOMMENDED.items():
        try:
            log(f"  {m:16s} {_version(m)}")
        except Exception:
            warnings.append(f"optional package '{m}' missing - {why}")

    if os.environ.get("HF_HUB_OFFLINE") != "1":
        warnings.append("HF_HUB_OFFLINE != 1 - compute nodes have no network; the hpc/ scripts set it")
    try:
        from .nn import load_tokenizer, resolve_model
        for name in (cfg.retriever.base_model, cfg.crossenc.base_model):
            p = resolve_model(cfg.paths.model_dir, name)
            tok = load_tokenizer(p)
            n = len(tok("sample business | 1 main street, delhi")["input_ids"])
            log(f"  model {name}: {p} (tokenizer ok, {n} tokens for a sample)")
    except Exception as e:
        problems.append(f"model check: {e}")

    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            log(f"  GPU {torch.cuda.get_device_name(0)} {free/2**30:.0f}/{total/2**30:.0f} GB free, "
                f"bf16={torch.cuda.is_bf16_supported()}")
            if not torch.cuda.is_bf16_supported():
                problems.append("GPU without bf16 support")
        else:
            log("  no GPU visible (fine on the login node)")
    except Exception as e:
        problems.append(f"torch/GPU check: {e}")

    for w in warnings:
        log(f"WARN: {w}")
    if problems:
        for p in problems:
            log(f"PROBLEM: {p}")
        raise SystemExit(1)
    log("preflight OK")
