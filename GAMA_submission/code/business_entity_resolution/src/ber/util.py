"""Small shared helpers: logging, timing, JSON, folds, thread counts, work paths."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import polars as pl

_T0 = time.time()

# polars >= 1.43 warns that explode's empty-list default will flip in 2.0;
# pin today's behaviour (empty list -> one null row) where the kwarg exists.
_EXPLODE_KW = ({"empty_as_null": True}
               if "empty_as_null" in inspect.signature(pl.DataFrame.explode).parameters else {})


def explode(df: pl.DataFrame, *cols: str) -> pl.DataFrame:
    return df.explode(list(cols), **_EXPLODE_KW)


def explode_series(s: pl.Series) -> pl.Series:
    return s.to_frame().explode(s.name, **_EXPLODE_KW).to_series()


def log(msg: str) -> None:
    el = time.time() - _T0
    print(f"[ber {time.strftime('%H:%M:%S')} +{el/60:6.1f}m] {msg}", flush=True)


@contextmanager
def timed(what: str) -> Iterator[None]:
    t = time.time()
    log(f"{what} ...")
    yield
    log(f"{what} done in {time.time() - t:.1f}s")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def as_list(x: Any) -> list[str]:
    """Config value -> list of strings: a YAML/JSON list, or 'a,b' / '[a,b]' from the command line."""
    if x is None or x == "":
        return []
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [p.strip().strip("'\"") for p in str(x).strip().strip("[]").split(",") if p.strip()]


def only_countries(cfg, split: str) -> list[str]:
    """Test-side partial rerun: the countries whose pairs are recomputed (empty = all).
    The other countries keep the previous run's results - valid only when nothing
    feeding them changed (e.g. a normalisation change applied to one country)."""
    return as_list(cfg.data.get("only_countries")) if split == "test" else []


def n_threads(requested: int = 0) -> int:
    """Threads to use: explicit value, else the SLURM allocation, else all cores."""
    if requested and requested > 0:
        return int(requested)
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return int(slurm)
    return os.cpu_count() or 1


def stable_bucket(ids: pl.Series, k: int, salt: int = 0) -> np.ndarray:
    """Deterministic bucket in [0, k) per entity id - stable across machines and
    library versions (unlike polars' hash). Ids look like 'S1-925783039'."""
    num = ids.str.replace(r"^S\d-", "").cast(pl.UInt64, strict=False)
    if num.null_count() == 0:
        x = num.to_numpy().astype(np.uint64) + np.uint64(salt)
        with np.errstate(over="ignore"):
            h = (x * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(29)
        return (h % np.uint64(k)).astype(np.int64)
    return np.array([int(hashlib.md5(f"{salt}:{s}".encode()).hexdigest()[:12], 16) % k
                     for s in ids.to_list()], dtype=np.int64)


class Work:
    """Canonical artefact locations under paths.work_dir."""

    def __init__(self, cfg):
        self.root = Path(cfg.paths.work_dir)
        self.cfg = cfg

    def p(self, *parts: str, mkdir: bool = True) -> Path:
        path = self.root.joinpath(*parts)
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # prepared data
    def records(self, split: str) -> Path:
        return self.p("prepared", f"{split}_records.parquet")

    def gt(self) -> Path:
        return self.p("prepared", "train_gt.parquet")

    # retrieval / ranking artefacts
    def sparse(self, split: str) -> Path:
        return self.p("sparse", f"{split}_sparse.parquet")

    def dense(self, split: str) -> Path:
        return self.p("dense", f"{split}_dense.parquet")

    def cands(self, split: str) -> Path:
        return self.p("cands", f"{split}_cands.parquet")

    def pruned(self, split: str) -> Path:
        return self.p("prune", f"{split}_pruned.parquet")

    def ce_scores(self, split: str, fold: int, shard: int = 0) -> Path:
        return self.p("ce", f"{split}_ce_f{fold}_s{shard}.parquet")

    def model(self, name: str) -> Path:
        return self.p("models", name, mkdir=False)


def die(msg: str, code: int = 1) -> None:
    print(f"[ber] ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)
