"""Tier 4 -- kinase descriptors.

Builds `data/raw/kinases.csv`: one row per kinase with UniProt accession, gene
name, KLIFS group/family, and the 85-residue KLIFS pocket string.

Why this file is load-bearing: it is the only thing that lets a multi-task model
*generalise* across kinases instead of learning 491 independent per-target
offsets. With 172 exact TNK1 points, per-target offsets are exactly what the
model will learn if you give it a target one-hot. The pocket string also defines
the kinome for tier 3 -- `kinase_accessions()` output is what the ChEMBL SQL
extract filters on.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .util import LOG, Http, Layout, Manifest

KLIFS_API = "https://klifs.net/api"
UNIPROT_API = "https://rest.uniprot.org"

#: KLIFS pocket alignment length. Any other length means a parsing error.
POCKET_LEN = 85


def fetch_klifs_kinases(http: Http, species: str = "HUMAN") -> pd.DataFrame:
    """All KLIFS kinases with their aligned pocket sequences."""
    data = http.json(f"{KLIFS_API}/kinase_information",
                     params={"species": species})
    df = pd.DataFrame(data)
    ren = {"kinase_ID": "klifs_id", "name": "klifs_name", "HGNC": "gene",
           "family": "family", "group": "group", "kinase_class": "kinase_class",
           "species": "species", "full_name": "full_name",
           "uniprot": "uniprot", "iuphar": "iuphar", "pocket": "pocket"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    for col in ("gene", "uniprot", "pocket"):
        if col not in df.columns:
            raise RuntimeError(f"KLIFS response lacks `{col}`; got {list(df.columns)}")
    df["pocket"] = df["pocket"].fillna("").astype(str).str.strip().str.upper()
    df["pocket_len"] = df["pocket"].str.len()
    bad = df["pocket_len"].between(1, POCKET_LEN - 1) | (df["pocket_len"] > POCKET_LEN)
    if bad.any():
        LOG.warning("%d kinases have a pocket string that is not %d residues: %s",
                    int(bad.sum()), POCKET_LEN,
                    df.loc[bad, ["gene", "pocket_len"]].head(10).to_dict("records"))
    n_empty = int((df["pocket_len"] == 0).sum())
    LOG.info("KLIFS: %d kinases (%d with a full %d-residue pocket, %d empty)",
             len(df), int((df["pocket_len"] == POCKET_LEN).sum()), POCKET_LEN, n_empty)
    return df


def fetch_uniprot_meta(http: Http, accessions: list[str]) -> pd.DataFrame:
    """Gene name, protein name, length and sequence for each accession."""
    rows: list[dict] = []
    batch = 100
    uniq = sorted({a for a in accessions if a and isinstance(a, str)})
    for i in range(0, len(uniq), batch):
        chunk = uniq[i:i + batch]
        query = " OR ".join(f"accession:{a}" for a in chunk)
        js = http.json(f"{UNIPROT_API}/uniprotkb/search", params={
            "query": query, "format": "json", "size": batch,
            "fields": "accession,id,gene_primary,protein_name,sequence,length,"
                      "cc_catalytic_activity,ft_domain",
        })
        for r in js.get("results", []):
            seq = (r.get("sequence") or {}).get("value")
            genes = r.get("genes") or []
            rows.append({
                "uniprot": r.get("primaryAccession"),
                "uniprot_entry": r.get("uniProtkbId"),
                "gene_uniprot": (genes[0].get("geneName", {}).get("value")
                                 if genes else None),
                "protein_name": ((r.get("proteinDescription") or {})
                                 .get("recommendedName", {})
                                 .get("fullName", {}).get("value")),
                "seq_len": (r.get("sequence") or {}).get("length"),
                "sequence": seq,
            })
        LOG.info("UniProt: %d/%d accessions resolved", len(rows), len(uniq))
    return pd.DataFrame(rows)


def build_kinases_csv(http: Http, layout: Layout, manifest: Manifest, *,
                      species: str = "HUMAN",
                      with_sequences: bool = True) -> Path:
    klifs = fetch_klifs_kinases(http, species=species)
    out = klifs
    if with_sequences:
        meta = fetch_uniprot_meta(http, klifs["uniprot"].dropna().tolist())
        out = klifs.merge(meta, on="uniprot", how="left")
        mism = out[(out["gene"].notna()) & (out["gene_uniprot"].notna())
                   & (out["gene"].str.upper() != out["gene_uniprot"].str.upper())]
        if len(mism):
            LOG.warning("%d gene-symbol mismatches KLIFS vs UniProt (kept both "
                        "columns): %s", len(mism),
                        mism[["gene", "gene_uniprot"]].head(8).to_dict("records"))

    cols = [c for c in ("klifs_id", "gene", "gene_uniprot", "uniprot", "klifs_name",
                        "full_name", "protein_name", "group", "family",
                        "kinase_class", "species", "pocket", "pocket_len",
                        "seq_len", "sequence") if c in out.columns]
    out = out[cols].sort_values("gene", na_position="last")
    path = layout.raw / "kinases.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    manifest.record(path, tier="4", source="klifs+uniprot",
                    url=f"{KLIFS_API}/kinase_information",
                    rows=len(out),
                    notes=f"{int((out['pocket_len'] == POCKET_LEN).sum())} kinases "
                          f"with a full {POCKET_LEN}-residue pocket")
    LOG.info("wrote %s (%d kinases)", path, len(out))
    return path


def kinase_accessions(path: Path, require_pocket: bool = True) -> list[str]:
    """The kinome definition used by the tier 3 extract."""
    df = pd.read_csv(path)
    if require_pocket and "pocket_len" in df.columns:
        df = df[df["pocket_len"] == POCKET_LEN]
    accs = sorted({a for a in df["uniprot"].dropna().astype(str)
                   if a and a.lower() != "nan"})
    LOG.info("kinome definition: %d accessions from %s", len(accs), path.name)
    return accs


def pocket_features(path: Path) -> pd.DataFrame:
    """One-hot / index encoding of the 85-residue pocket, per kinase.

    Returned wide (85 integer columns) so it can be concatenated to a compound
    representation directly, and long-form amino-acid identity is preserved so a
    learned residue embedding can be used instead.
    """
    aa = "ACDEFGHIKLMNPQRSTVWY-"
    idx = {c: i for i, c in enumerate(aa)}
    df = pd.read_csv(path)
    df = df[df.get("pocket_len", 0) == POCKET_LEN].copy()
    mat = [[idx.get(ch, idx["-"]) for ch in p] for p in df["pocket"]]
    feats = pd.DataFrame(mat, columns=[f"pk{i:02d}" for i in range(POCKET_LEN)])
    feats.insert(0, "uniprot", df["uniprot"].values)
    feats.insert(1, "gene", df["gene"].values)
    LOG.info("pocket feature matrix: %s", feats.shape)
    return feats
