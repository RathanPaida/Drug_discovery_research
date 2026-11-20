"""ChEMBL: REST for the small, high-value pulls (tiers 1-2) and the SQLite
release dump for the kinome-wide pull (tier 3) and the unlabelled corpus
(tier 5).

Why both: ChEMBL's REST API caps deep paging, so a multi-million-row kinome
query over REST is both slow and silently truncating. The release SQLite dump
is one ~6 GB download and one SQL statement, and it is reproducible -- the
release version is pinned in the manifest, so the extract can be rebuilt
byte-identically later. That reproducibility is exactly what the missing
`data/features/` rebuild step needs.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from .util import LOG, Http, Layout, Manifest, extract, human, write_parquet

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_FTP = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest"

ACTIVITY_TYPES = ("IC50", "Ki", "Kd", "EC50", "XC50", "AC50", "Potency")

#: Assay confidence: 9 = direct single protein, 8 = homologous single protein.
MIN_CONFIDENCE = 8

#: ChEMBL flags a subset of rows as suspect; these are the ones to keep.
OK_VALIDITY = ("Manually validated",)


# --------------------------------------------------------------------------- #
# release discovery
# --------------------------------------------------------------------------- #
def chembl_release(http: Http) -> str:
    """Return e.g. '35'. Pinned into the manifest for reproducibility."""
    try:
        status = http.json(f"{CHEMBL_API}/status.json")
        ver = str(status.get("chembl_db_version", ""))
        m = re.search(r"(\d+)", ver)
        if m:
            LOG.info("ChEMBL release %s (from API status)", m.group(1))
            return m.group(1)
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("ChEMBL status endpoint failed (%s); sniffing FTP", exc)
    html = http.get(f"{CHEMBL_FTP}/").text
    vers = sorted({int(m) for m in re.findall(r"chembl_(\d+)_sqlite\.tar\.gz", html)})
    if not vers:
        raise RuntimeError("could not determine ChEMBL release from FTP listing")
    LOG.info("ChEMBL release %s (from FTP listing)", vers[-1])
    return str(vers[-1])


# --------------------------------------------------------------------------- #
# tiers 1-2: REST
# --------------------------------------------------------------------------- #
_ACT_FIELDS = (
    "molecule_chembl_id", "canonical_smiles", "target_chembl_id",
    "assay_chembl_id", "assay_description", "assay_type",
    "standard_type", "standard_relation", "standard_value", "standard_units",
    "pchembl_value", "standard_flag", "activity_comment",
    "data_validity_comment", "potential_duplicate", "document_chembl_id",
    "document_year", "src_id", "bao_label", "activity_id",
)


def fetch_activities_rest(http: Http, target_chembl_ids: list[str], *,
                          activity_types: tuple[str, ...] = ACTIVITY_TYPES,
                          page: int = 1000) -> pd.DataFrame:
    """Page through /activity.json for a handful of targets.

    We request each (target, type) pair separately: that keeps every result set
    well under ChEMBL's deep-paging limit, so no page is silently dropped. The
    row count per request is asserted against `page_meta.total_count`.
    """
    rows: list[dict] = []
    for tid in target_chembl_ids:
        for atype in activity_types:
            offset, total = 0, None
            got = 0
            while True:
                payload = http.json(
                    f"{CHEMBL_API}/activity.json",
                    params={
                        "target_chembl_id": tid,
                        "standard_type": atype,
                        "limit": page,
                        "offset": offset,
                    },
                )
                meta = payload.get("page_meta", {})
                total = meta.get("total_count") if total is None else total
                batch = payload.get("activities", [])
                for a in batch:
                    rows.append({k: a.get(k) for k in _ACT_FIELDS})
                got += len(batch)
                offset += page
                if not batch or (total is not None and got >= total):
                    break
            if total:
                LOG.info("  %s / %-8s %5d rows (API total_count=%s)",
                         tid, atype, got, total)
                if got < total:
                    raise RuntimeError(
                        f"truncated pull for {tid}/{atype}: {got} < {total}")
    df = pd.DataFrame(rows)
    if df.empty:
        LOG.warning("REST pull returned no rows for %s", target_chembl_ids)
        return df
    df = df.drop_duplicates(subset=["activity_id"])
    df["smiles"] = df["canonical_smiles"]
    df["source"] = "chembl_rest"
    LOG.info("ChEMBL REST: %d unique activities over %d targets",
             len(df), len(target_chembl_ids))
    return df


def apply_quality_filters(df: pd.DataFrame, *,
                          min_confidence: int | None = None,
                          drop_duplicates_flag: bool = True) -> pd.DataFrame:
    """ChEMBL-specific provenance filters (separate from unit/range filtering)."""
    if df.empty:
        return df
    n0 = len(df)
    keep = pd.Series(True, index=df.index)
    if "data_validity_comment" in df:
        dv = df["data_validity_comment"]
        keep &= dv.isna() | dv.isin(OK_VALIDITY)
    if drop_duplicates_flag and "potential_duplicate" in df:
        pd_flag = pd.to_numeric(df["potential_duplicate"], errors="coerce").fillna(0)
        keep &= pd_flag == 0
    if "activity_comment" in df:
        ac = df["activity_comment"].astype(str).str.lower()
        # 'not active'/'inconclusive' rows carry no usable number.
        keep &= ~ac.isin({"not active", "inconclusive", "not determined",
                          "undetermined", "inactive"})
    if min_confidence is not None and "confidence_score" in df:
        cs = pd.to_numeric(df["confidence_score"], errors="coerce")
        keep &= cs.isna() | (cs >= min_confidence)
    out = df.loc[keep].copy()
    LOG.info("ChEMBL quality filters: %d -> %d rows", n0, len(out))
    return out


def fetch_assay_confidence(http: Http, assay_ids: list[str]) -> pd.DataFrame:
    """Confidence score + assay organism for a list of assay ids (batched)."""
    rows = []
    uniq = sorted({a for a in assay_ids if a})
    for i in range(0, len(uniq), 50):
        chunk = uniq[i:i + 50]
        payload = http.json(f"{CHEMBL_API}/assay.json",
                            params={"assay_chembl_id__in": ",".join(chunk),
                                    "limit": 100})
        for a in payload.get("assays", []):
            rows.append({
                "assay_chembl_id": a.get("assay_chembl_id"),
                "confidence_score": a.get("confidence_score"),
                "assay_organism": a.get("assay_organism"),
                "assay_cell_type": a.get("assay_cell_type"),
                "bao_format": a.get("bao_format"),
            })
    LOG.info("fetched confidence for %d/%d assays", len(rows), len(uniq))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# tier 3 / tier 5: release dumps
# --------------------------------------------------------------------------- #
def download_sqlite(http: Http, layout: Layout, manifest: Manifest,
                    release: str, *, force: bool = False) -> Path:
    """~6 GB compressed, ~25 GB unpacked. One-time cost, pays for tier 3 + 5."""
    d = layout.download_dir(f"chembl_{release}")
    url = f"{CHEMBL_FTP}/chembl_{release}_sqlite.tar.gz"
    tar = http.download(url, d / f"chembl_{release}_sqlite.tar.gz", force=force)
    manifest.record(tar, tier="3", source="chembl_sqlite", url=url,
                    notes=f"ChEMBL release {release} SQLite dump")
    db = next(d.rglob(f"chembl_{release}.db"), None)
    if db is None:
        extract(tar, d / "unpacked")
        db = next((d / "unpacked").rglob(f"chembl_{release}.db"), None)
    if db is None:
        raise RuntimeError(f"chembl_{release}.db not found after extraction in {d}")
    LOG.info("ChEMBL sqlite ready: %s (%s)", db, human(db.stat().st_size))
    return db


_KINOME_SQL = """
SELECT
    md.chembl_id                AS molecule_chembl_id,
    cs.canonical_smiles         AS smiles,
    td.chembl_id                AS target_chembl_id,
    td.pref_name                AS target_pref_name,
    cseq.accession              AS accession,
    act.standard_type           AS standard_type,
    act.standard_relation       AS standard_relation,
    act.standard_value          AS standard_value,
    act.standard_units          AS standard_units,
    act.pchembl_value           AS pchembl_value,
    act.standard_flag           AS standard_flag,
    act.activity_comment        AS activity_comment,
    act.data_validity_comment   AS data_validity_comment,
    act.potential_duplicate     AS potential_duplicate,
    ass.chembl_id               AS assay_chembl_id,
    ass.assay_type              AS assay_type,
    ass.confidence_score        AS confidence_score,
    ass.description             AS assay_description,
    d.year                      AS document_year,
    d.chembl_id                 AS document_chembl_id
FROM activities act
JOIN assays              ass   ON ass.assay_id    = act.assay_id
JOIN target_dictionary   td    ON td.tid          = ass.tid
JOIN target_components   tc    ON tc.tid          = td.tid
JOIN component_sequences cseq  ON cseq.component_id = tc.component_id
JOIN molecule_dictionary md    ON md.molregno     = act.molregno
JOIN compound_structures cs    ON cs.molregno     = act.molregno
LEFT JOIN docs           d     ON d.doc_id        = act.doc_id
WHERE td.target_type   = 'SINGLE PROTEIN'
  AND td.organism      = 'Homo sapiens'
  AND cseq.accession  IN ({placeholders})
  AND act.standard_type IN ({types})
  AND act.standard_value IS NOT NULL
  AND act.standard_value > 0
  AND act.standard_units IS NOT NULL
  AND ass.confidence_score >= {min_conf}
  AND (act.data_validity_comment IS NULL
       OR act.data_validity_comment = 'Manually validated')
  AND (act.potential_duplicate IS NULL OR act.potential_duplicate = 0)
"""


def kinome_from_sqlite(db: Path, accessions: list[str], *,
                       activity_types: tuple[str, ...] = ACTIVITY_TYPES,
                       min_confidence: int = MIN_CONFIDENCE,
                       chunk: int = 400) -> pd.DataFrame:
    """Kinome-wide bioactivity extract, restricted to `accessions`.

    `accessions` should be the kinase accession list built in tier 4 (KLIFS),
    which is what makes the multi-task model generalise across kinases via the
    pocket sequence rather than memorising a per-target offset.

    Note the SQL requires `standard_value > 0` and non-null units, so the v2
    zero-encoding cannot be reproduced here even in principle -- a row with no
    measurement never leaves the database.
    """
    if not accessions:
        raise ValueError("no kinase accessions supplied for tier 3 extract")
    frames = []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        for i in range(0, len(accessions), chunk):
            accs = accessions[i:i + chunk]
            sql = _KINOME_SQL.format(
                placeholders=",".join("?" * len(accs)),
                types=",".join(f"'{t}'" for t in activity_types),
                min_conf=int(min_confidence),
            )
            part = pd.read_sql_query(sql, con, params=accs)
            LOG.info("  kinome chunk %d/%d: %d accessions -> %d rows",
                     i // chunk + 1, (len(accessions) + chunk - 1) // chunk,
                     len(accs), len(part))
            frames.append(part)
    finally:
        con.close()
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df["source"] = "chembl_sqlite"
    LOG.info("tier 3 raw extract: %d rows, %d accessions, %d compounds",
             len(df), df["accession"].nunique() if len(df) else 0,
             df["molecule_chembl_id"].nunique() if len(df) else 0)
    return df


def download_chemreps(http: Http, layout: Layout, manifest: Manifest,
                      release: str, *, force: bool = False) -> Path:
    """chembl_NN_chemreps.txt.gz -- id/SMILES/InChI for every ChEMBL compound.

    This is the honest source for the tier 5 unlabelled corpus: ~2.4 M
    structures, from which the drug-like subset is filtered locally. Downloading
    a pre-filtered .smi from somewhere else means you cannot say what the filter
    was.
    """
    d = layout.download_dir(f"chembl_{release}")
    url = f"{CHEMBL_FTP}/chembl_{release}_chemreps.txt.gz"
    p = http.download(url, d / f"chembl_{release}_chemreps.txt.gz", force=force)
    manifest.record(p, tier="5", source="chembl_chemreps", url=url,
                    notes="all ChEMBL structures (id, SMILES, InChI, InChIKey)")
    return p
