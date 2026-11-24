#!/usr/bin/env python
"""Rebuild the ChEMBL-37 kinome table from the raw union, with the TARGET kept.

WHY THIS SCRIPT EXISTS
----------------------
The second acquisition drop shipped `processed/kinome_measurements_v3.parquet`
(302,463 rows) as its kinome-wide multi-task set. That file has no target column at
all: 460 kinases went in and zero came out. It cannot be used to pretrain a
multi-task model, and it cannot be merged with `kinome_long_v3.parquet`, whose key
is (inchikey_block1, uniprot_id, standard_type).

The cause is one line in that drop's own curation code, `tnk1_acquire/curate.py`:

    key = tuple(k for k in key if k in df.columns)

`aggregate_replicates` defaults to key=("inchikey", "target_gene", "endpoint"). On
the ChEMBL-SQLite path the frame carries `accession`, and `target_gene` is only
mapped on AFTER aggregation -- so the key silently degraded to
("inchikey", "endpoint") and every kinase for a given compound collapsed into one
row. The follow-up `if "accession" in agg.columns` was then also False, because
groupby had already dropped it, so `target_gene` never got set either. No exception,
no warning, no row-count check: the file looks like a result.

Measured consequences, against `interim/kinome_raw_union.parquet`:

  * 0 duplicate (inchikey, endpoint, censored) rows in the shipped file, which is
    what proves the key really was that pair.
  * Aggregating on the correct key yields 383,303 rows; the shipped file has
    302,463. 21% of distinct (compound, kinase, endpoint) cells were merged away.
  * 82,764 compounds had genuinely different potencies across kinases -- mean range
    1.28 log units, 95th percentile 3.15, max 6.60 -- all replaced by one median.
  * 30.6% of compounds are multi-kinase; one spans 406 kinases.

Taking the median across kinases records a compound measured at 9.0 on TNK1 and 5.0
on forty other kinases as roughly 5.2. That destroys precisely the selectivity
signal that `r_sel` and Stage 5 consume. It is the same class of defect as the v2
`pActivity == 0.0` poisoning that the drop's own tier3 docstring boasts of
preventing.

WHAT THIS SCRIPT DOES INSTEAD
-----------------------------
Re-derives the table from `kinome_raw_union.parquet`, which retains `accession` on
all 617,200 rows, using THIS repository's own derivation so the result is directly
comparable with the frozen v3 table:

    tnk1.data.standardize.standardize_smiles   same med-chem filter, same InChIKey
    tnk1.data.activity.prepare_records         same relation -> censored convention
    tnk1.data.activity.aggregate_replicates    key is hardcoded correct here:
                                               (inchikey_block1, uniprot_id, standard_type)

Output columns are the aliases `run_stage1_data.py::_pick` already accepts, so
Stage 1 reads this file with no reader changes.

    python scripts/build_kinome_chembl37.py \
        --raw data/raw/kinome_raw_union.parquet \
        --out data/features/kinome_chembl37.parquet --jobs 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tnk1.data import activity, standardize  # noqa: E402
from tnk1.utils.common import load_config  # noqa: E402
from tnk1.utils.paths import default_config  # noqa: E402

# Units we can convert to nM. `ug.mL-1` needs a molecular weight per record and `/s`
# is a rate constant, not an affinity; both are dropped with a count rather than
# coerced. They are 786 of 617,200 rows.
CONVERTIBLE = set(activity.UNIT_TO_NM)


def _std_one(smi: str, mw_range, hac_range, allowed):
    """Worker: standardize one SMILES. Returns (smi, result-or-None)."""
    try:
        return smi, standardize.standardize_smiles(
            smi, mw_range=tuple(mw_range), hac_range=tuple(hac_range),
            allowed=tuple(allowed))
    except Exception:
        return smi, None


def _standardize_unique(smiles: list[str], d: dict, jobs: int) -> dict:
    """Standardize each unique SMILES once. The tautomer canonicaliser is the
    expensive step (~5 ms/molecule), so this is the only part worth parallelising:
    270k molecules is ~25 min on one core and ~1 min on 32."""
    mw = d.get("mw_range", (150, 900))
    hac = d.get("hac_range", (10, 65))
    allowed = d.get("allowed_elements",
                    ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "H"))
    out: dict[str, dict] = {}
    if jobs <= 1:
        for i, s in enumerate(smiles):
            _, r = _std_one(s, mw, hac, allowed)
            if r:
                out[s] = r
            if i % 25000 == 0:
                print(f"  standardized {i:,}/{len(smiles):,}", flush=True)
        return out

    import multiprocessing as mp
    from functools import partial
    fn = partial(_std_one, mw_range=mw, hac_range=hac, allowed=allowed)
    # chunksize matters here: the default of 1 spends more time pickling than
    # standardizing on a 270k-item list.
    with mp.Pool(jobs) as pool:
        for i, (s, r) in enumerate(pool.imap_unordered(fn, smiles, chunksize=256)):
            if r:
                out[s] = r
            if i % 25000 == 0:
                print(f"  standardized {i:,}/{len(smiles):,}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/kinome_raw_union.parquet",
                    help="interim/kinome_raw_union.parquet from the acquisition drop")
    ap.add_argument("--out", default="data/features/kinome_chembl37.parquet")
    ap.add_argument("--kinases", default="data/raw/kinases.csv",
                    help="uniprot -> gene map, for the gene_symbol column")
    ap.add_argument("--config", default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--source-label", default="CHEMBL37")
    a = ap.parse_args()

    cfg = load_config(a.config or default_config())
    d = cfg["data"]
    raw_path, out_path = Path(a.raw), Path(a.out)
    if not raw_path.exists():
        raise SystemExit(
            f"ERROR: {raw_path} not found.\n"
            f"  It is interim/kinome_raw_union.parquet inside the acquisition drop\n"
            f"  (Drug_discovery_research_data-*.zip). 11.5 MB; extract it to {raw_path}."
        )

    rep: dict[str, int] = {}
    df = pd.read_parquet(raw_path)
    rep["rows_in"] = len(df)
    print(f"[c37] read {len(df):,} rows from {raw_path.name}")

    # ---- drop rows that carry no measurement -------------------------------------
    # Exact duplicate records first: 783 of them, byte-identical across all 21 columns.
    df = df.drop_duplicates()
    rep["dropped_exact_duplicate_rows"] = rep["rows_in"] - len(df)

    need = ["smiles", "accession", "standard_type", "standard_value"]
    before = len(df)
    for c in need:
        df = df[df[c].notna()]
        if df[c].dtype == object:
            df = df[df[c].astype(str).str.strip() != ""]
    rep["dropped_empty_required_field"] = before - len(df)

    before = len(df)
    df = df[pd.to_numeric(df["standard_value"], errors="coerce") > 0]
    rep["dropped_nonpositive_value"] = before - len(df)

    before = len(df)
    df = df[df["standard_units"].isin(CONVERTIBLE)]
    rep["dropped_inconvertible_units"] = before - len(df)
    print(f"[c37] {len(df):,} rows carry a convertible measurement "
          f"({rep['dropped_inconvertible_units']:,} dropped for units like ug.mL-1, /s)")

    df["value_nM"] = (pd.to_numeric(df["standard_value"], errors="coerce")
                      * df["standard_units"].map(activity.UNIT_TO_NM))

    # ---- standardize chemistry ----------------------------------------------------
    uniq = sorted(set(df["smiles"].astype(str)))
    print(f"[c37] standardizing {len(uniq):,} unique SMILES on {a.jobs} core(s)")
    std = _standardize_unique(uniq, d, a.jobs)
    rep["unique_smiles_in"] = len(uniq)
    rep["unique_smiles_kept"] = len(std)
    print(f"[c37] {len(std):,}/{len(uniq):,} passed the med-chem filter "
          f"(MW {d.get('mw_range')}, HAC {d.get('hac_range')})")

    before = len(df)
    keep = df["smiles"].astype(str).map(lambda s: s in std)
    df = df[keep]
    rep["dropped_failed_standardization"] = before - len(df)

    s = df["smiles"].astype(str)
    df["inchikey_block1"] = s.map(lambda x: std[x]["inchikey_block1"])
    df["scaffold"] = s.map(lambda x: std[x]["scaffold"])
    df["smiles"] = s.map(lambda x: std[x]["smiles"])

    # ---- pActivity and censoring, by this repo's convention ----------------------
    df = df.rename(columns={"accession": "uniprot_id",
                            "standard_relation": "relation"})
    df["source"] = a.source_label
    before = len(df)
    recs = activity.prepare_records(
        df[["inchikey_block1", "smiles", "scaffold", "uniprot_id",
            "standard_type", "value_nM", "source", "relation"]])
    # prepare_records drops left-censored ('<', '<=') rows: a "<10 nM" record bounds
    # the true value ABOVE its mark, which the manual's loss does not model.
    rep["dropped_left_censored"] = before - len(recs)
    print(f"[c37] {len(recs):,} records after pActivity derivation "
          f"({rep['dropped_left_censored']:,} left-censored dropped)")

    before = len(recs)
    recs = recs[recs["pact"].between(float(d.get("pact_min", 2.0)),
                                     float(d.get("pact_max", 12.0)))]
    rep["dropped_pact_out_of_range"] = before - len(recs)

    # ---- aggregate replicates ON THE CORRECT KEY ---------------------------------
    # tnk1.data.activity.aggregate_replicates hardcodes
    #   ["inchikey_block1", "uniprot_id", "standard_type"]
    # so the kinase cannot be dropped the way the acquisition drop dropped it.
    print(f"[c37] aggregating replicates on "
          f"(inchikey_block1, uniprot_id, standard_type)")
    agg = activity.aggregate_replicates(
        recs,
        range_max=float(d.get("replicate_range_max", 1.0)),
        policy=str(d.get("replicate_policy", "drop")))
    rep["rows_after_aggregation"] = len(agg)

    if "uniprot_id" not in agg.columns or agg["uniprot_id"].nunique() < 2:
        raise SystemExit(
            "ERROR: the aggregated table has no usable uniprot_id column.\n"
            "  That is the exact defect this script exists to repair -- refusing to\n"
            "  write a kinome table whose target identity has been collapsed."
        )

    # gene symbols, for readability downstream; absent map is not fatal
    kin = Path(a.kinases)
    if kin.exists():
        km = pd.read_csv(kin)
        col = "uniprot" if "uniprot" in km.columns else km.columns[0]
        gene = "gene" if "gene" in km.columns else km.columns[1]
        agg["gene_symbol"] = agg["uniprot_id"].map(
            dict(zip(km[col], km[gene]))).fillna("")
    else:
        agg["gene_symbol"] = ""
        print(f"[c37] WARN {kin} absent -- gene_symbol left blank")

    agg = agg.rename(columns={"pact": "p_activity", "n_rep": "n_replicates"})
    agg["sigma_rep"] = pd.to_numeric(agg["sigma_rep"], errors="coerce").fillna(0.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out_path, index=False)

    print(f"\n[c37] wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    print(f"[c37] {len(agg):,} rows | {agg['inchikey_block1'].nunique():,} compounds "
          f"| {agg['uniprot_id'].nunique()} kinases "
          f"| {int((agg['censored'] == 0).sum()):,} exact "
          f"| {int((agg['censored'] == 1).sum()):,} censored")
    print("\n[c37] attrition:")
    for k, v in rep.items():
        print(f"    {k:<36} {v:>10,}")

    side = out_path.with_suffix(".build.json")
    side.write_text(json.dumps(
        {"source": str(raw_path), "attrition": rep,
         "rows": len(agg),
         "compounds": int(agg["inchikey_block1"].nunique()),
         "kinases": int(agg["uniprot_id"].nunique()),
         "aggregation_key": ["inchikey_block1", "uniprot_id", "standard_type"]},
        indent=2))
    print(f"[c37] attrition record -> {side.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
