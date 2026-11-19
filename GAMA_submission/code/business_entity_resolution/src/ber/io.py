"""Reading the competition TSVs and writing the two submission files.

The TSVs contain literal `"` characters (823 lines) and empty addresses, so
quoting is disabled and empty fields stay "" - never NaN/null.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from .util import explode, log

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")


def source_path(data_dir: str | Path, split: str, src: int) -> Path:
    return Path(data_dir) / split / f"{split}_source{src}.tsv"


def count_lines(path: str | Path) -> int:
    """Data rows (excluding header) - used to prove the parser dropped nothing."""
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            n += block.count(b"\n")
    return n - 1


def read_tsv(path: str | Path, expected_cols: list[str]) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                     has_header=True, encoding="utf8")
    cols = [c.strip() for c in df.columns]
    if cols != expected_cols:
        raise ValueError(f"{path}: columns {cols} != expected {expected_cols}")
    df.columns = cols
    df = df.with_columns(pl.all().fill_null(""))
    n_raw = count_lines(path)
    if df.height != n_raw:
        raise ValueError(f"{path}: parsed {df.height} rows but file has {n_raw} data lines")
    return df


def read_source(data_dir: str | Path, split: str, src: int) -> pl.DataFrame:
    path = source_path(data_dir, split, src)
    df = read_tsv(path, SOURCE_COLS)
    log(f"read {path.name}: {df.height:,} rows")
    return df.rename({"entity_id": "id", "business_name": "name",
                      "business_address": "addr"}).with_columns(pl.lit(src, pl.UInt8).alias("src"))


def read_ground_truth(data_dir: str | Path) -> pl.DataFrame:
    """Exploded ground truth: one row per (s1_id, rec_id); singletons have rec_id=null."""
    path = Path(data_dir) / "train" / "train_ground_truth.tsv"
    df = read_tsv(path, list(MATCH_HEADER))
    log(f"read {path.name}: {df.height:,} rows")
    df = (df.rename({"source1_entity_id": "s1_id", "matched_entity_ids": "ids"})
            .with_columns(pl.col("ids").str.strip_chars().str.split(",")))
    return (explode(df, "ids")
              .with_columns(pl.col("ids").str.strip_chars())
              .with_columns(pl.when(pl.col("ids") == "").then(None).otherwise(pl.col("ids")).alias("rec_id"))
              .select("s1_id", "rec_id"))


def write_id_lists(path: str | Path, s1_ids: pl.Series, pairs: pl.DataFrame,
                   header: tuple[str, str]) -> None:
    """One row per S1 id (in `s1_ids` order); `pairs` has columns s1_id, rec_id.
    Writes a TAB-separated, UTF-8 file; ids in a list are unique, comma-joined."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    agg = (pairs.select("s1_id", "rec_id").unique(maintain_order=True)
                .group_by("s1_id", maintain_order=True)
                .agg(pl.col("rec_id").str.join(",").alias("ids")))
    out = (pl.DataFrame({"s1_id": s1_ids}).join(agg, on="s1_id", how="left")
             .with_columns(pl.col("ids").fill_null(""))
             .rename({"s1_id": header[0], "ids": header[1]}))
    out.write_csv(path, separator="\t", quote_style="never", include_header=True)
    n_nonempty = out.filter(pl.col(header[1]) != "").height
    log(f"wrote {path} - {out.height:,} rows ({n_nonempty:,} non-empty)")
