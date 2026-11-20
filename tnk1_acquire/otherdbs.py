"""Non-ChEMBL bioactivity sources, including the three that contributed zero
rows in the previous harvest and were never surfaced: DTC, KIBA and Metz.

Each loader returns a frame in the common schema
(smiles, accession, standard_type, standard_relation, standard_value,
standard_units, source) or an EMPTY frame -- and in the empty case it records a
`HarvestReport` saying *why*. `report_table()` renders those reports, and the
CLI prints it at the end of every run. A source contributing zero rows is now
impossible to miss.
"""
from __future__ import annotations

import io
import json
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .util import LOG, Http, Layout, Manifest, extract, gz_lines, human, try_import

COMMON_COLS = ("smiles", "accession", "standard_type", "standard_relation",
               "standard_value", "standard_units", "source")


# --------------------------------------------------------------------------- #
# harvest reporting
# --------------------------------------------------------------------------- #
@dataclass
class HarvestReport:
    source: str
    rows: int = 0
    compounds: int = 0
    targets: int = 0
    status: str = "ok"            # ok | empty | skipped | failed
    detail: str = ""
    action: str = ""              # what the operator must do, if anything


REPORTS: list[HarvestReport] = []


def _report(r: HarvestReport) -> HarvestReport:
    REPORTS.append(r)
    lvl = LOG.info if r.status == "ok" else LOG.error
    lvl("harvest[%s] %s rows=%d compounds=%d targets=%d %s",
        r.source, r.status.upper(), r.rows, r.compounds, r.targets, r.detail)
    if r.action:
        LOG.error("harvest[%s] ACTION REQUIRED: %s", r.source, r.action)
    return r


def summarize(df: pd.DataFrame, source: str, detail: str = "") -> HarvestReport:
    if df is None or df.empty:
        return _report(HarvestReport(source, status="empty",
                                     detail=detail or "returned 0 rows"))
    return _report(HarvestReport(
        source, rows=len(df),
        compounds=df["smiles"].nunique() if "smiles" in df else 0,
        targets=df["accession"].nunique() if "accession" in df else 0,
        detail=detail))


def report_table() -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in REPORTS])


# --------------------------------------------------------------------------- #
# BindingDB
# --------------------------------------------------------------------------- #
BINDINGDB_SERVICE = ("https://bindingdb.org/axis2/services/BDBService/"
                     "getLigandsByUniprots")
BINDINGDB_DOWNLOADS = "https://www.bindingdb.org/bind/downloads/"


def bindingdb_by_uniprot(http: Http, accessions: list[str]) -> pd.DataFrame:
    """Per-UniProt pull via BindingDB's own web service.

    Far lighter than the ~1 GB full dump and sufficient for tiers 1-2. Affinity
    fields arrive as strings that may carry a relation prefix ('>10000'), which
    is exactly the censored-data signal we must not lose -- it is parsed out
    into standard_relation rather than coerced away.
    """
    rows: list[dict] = []
    for acc in accessions:
        try:
            payload = http.json(BINDINGDB_SERVICE, params={
                "uniprot": acc, "code": 0, "response": "application/json"})
        except Exception as exc:                                   # noqa: BLE001
            LOG.warning("BindingDB service failed for %s: %s", acc, exc)
            continue
        hits = (payload.get("getLigandsByUniprotsResponse", {})
                .get("affinities", []))
        if isinstance(hits, dict):
            hits = [hits]
        for h in hits:
            raw = str(h.get("affinity", "")).strip()
            rel, val = _split_relation(raw)
            rows.append({
                "smiles": h.get("smile") or h.get("smiles"),
                "accession": acc,
                "standard_type": (h.get("affinity_type") or "").strip() or "Ki",
                "standard_relation": rel,
                "standard_value": val,
                "standard_units": "nM",
                "source": "bindingdb_ws",
                "monomerid": h.get("monomerid"),
            })
    df = pd.DataFrame(rows)
    summarize(df, "bindingdb_ws",
              f"{len(accessions)} accessions queried via web service")
    return df


def _split_relation(raw: str) -> tuple[str, float | None]:
    m = re.match(r"^\s*(>=|<=|>>|<<|>|<|~|=)?\s*([0-9.eE+-]+)\s*$", raw or "")
    if not m:
        return "=", None
    rel = m.group(1) or "="
    try:
        return rel, float(m.group(2))
    except ValueError:
        return rel, None


def bindingdb_full(http: Http, layout: Layout, manifest: Manifest, *,
                   url: str | None = None, force: bool = False) -> Path | None:
    """Download the full BindingDB TSV (~1 GB zip) for tier 3 breadth.

    The filename is release-dated, so the latest link is discovered from the
    downloads listing. Pass `url` to pin a specific release.
    """
    if url is None:
        try:
            html = http.get(BINDINGDB_DOWNLOADS).text
            names = re.findall(r"BindingDB_All_(\d{6})_tsv\.zip", html)
            if not names:
                raise RuntimeError("no BindingDB_All_*_tsv.zip in listing")
            latest = max(names)
            url = f"{BINDINGDB_DOWNLOADS}BindingDB_All_{latest}_tsv.zip"
        except Exception as exc:                                   # noqa: BLE001
            _report(HarvestReport(
                "bindingdb_full", status="skipped", detail=str(exc),
                action=f"download the BindingDB_All_*_tsv.zip manually from "
                       f"{BINDINGDB_DOWNLOADS} and re-run with "
                       f"--bindingdb-url file:///path/to/it"))
            return None
    d = layout.download_dir("bindingdb")
    try:
        if url.startswith("file://"):
            zp = Path(url[len("file://"):])
        else:
            zp = http.download(url, d / Path(url).name, force=force)
    except Exception as exc:                                       # noqa: BLE001
        _report(HarvestReport("bindingdb_full", status="failed", detail=str(exc)))
        return None
    manifest.record(zp, tier="3", source="bindingdb_full", url=url,
                    notes="full BindingDB TSV release")
    out = extract(zp, d / "unpacked")
    tsv = next((out if out.is_dir() else out.parent).rglob("BindingDB_All*.tsv"), None)
    if tsv is None:
        _report(HarvestReport("bindingdb_full", status="failed",
                              detail="no BindingDB_All*.tsv after extraction"))
        return None
    LOG.info("BindingDB TSV: %s (%s)", tsv, human(tsv.stat().st_size))
    return tsv


def bindingdb_subset(tsv: Path, accessions: set[str], *,
                     chunksize: int = 200_000) -> pd.DataFrame:
    """Stream the full TSV, keeping only rows for `accessions`."""
    want_cols = {
        "Ligand SMILES": "smiles",
        "UniProt (SwissProt) Primary ID of Target Chain": "accession",
        "Ki (nM)": "Ki", "IC50 (nM)": "IC50", "Kd (nM)": "Kd", "EC50 (nM)": "EC50",
        "PMID": "pmid", "Curation/DataSource": "curation",
    }
    frames: list[pd.DataFrame] = []
    reader = pd.read_csv(tsv, sep="\t", chunksize=chunksize, low_memory=False,
                         on_bad_lines="skip", quoting=3)
    for i, chunk in enumerate(reader):
        cols = {c: want_cols[c] for c in chunk.columns if c in want_cols}
        if "UniProt (SwissProt) Primary ID of Target Chain" not in chunk.columns:
            LOG.warning("BindingDB chunk %d lacks the UniProt column; skipping", i)
            continue
        sub = chunk[list(cols)].rename(columns=cols)
        sub = sub[sub["accession"].isin(accessions)]
        if sub.empty:
            continue
        melted = sub.melt(
            id_vars=[c for c in ("smiles", "accession", "pmid", "curation")
                     if c in sub.columns],
            value_vars=[c for c in ("Ki", "IC50", "Kd", "EC50") if c in sub.columns],
            var_name="standard_type", value_name="raw_value").dropna(subset=["raw_value"])
        if melted.empty:
            continue
        parsed = melted["raw_value"].astype(str).map(_split_relation)
        melted["standard_relation"] = [p[0] for p in parsed]
        melted["standard_value"] = [p[1] for p in parsed]
        melted["standard_units"] = "nM"
        melted["source"] = "bindingdb_full"
        frames.append(melted.drop(columns=["raw_value"]))
        if i % 10 == 0:
            LOG.info("  BindingDB chunk %d -> %d kept so far",
                     i, sum(len(f) for f in frames))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summarize(df, "bindingdb_subset", f"{len(accessions)} accessions of interest")
    return df


# --------------------------------------------------------------------------- #
# PubChem BioAssay
# --------------------------------------------------------------------------- #
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def pubchem_by_accession(http: Http, accessions: list[str], *,
                         max_assays_per_target: int = 400) -> pd.DataFrame:
    """Dose-response points from PubChem BioAssay for each UniProt accession.

    PubChem is where a lot of academic kinase screening lands and never reaches
    ChEMBL. We take only rows with a numeric activity value and a potency-like
    activity name, so single-concentration %-inhibition screens (which are not
    dose-response and would wreck a regression target) stay out.
    """
    rows: list[dict] = []
    for acc in accessions:
        try:
            txt = http.get(f"{PUG}/assay/target/accession/{acc}/aids/TXT").text
        except Exception as exc:                                   # noqa: BLE001
            LOG.warning("PubChem aid lookup failed for %s: %s", acc, exc)
            continue
        aids = [int(x) for x in txt.split() if x.strip().isdigit()]
        LOG.info("PubChem %s: %d assays", acc, len(aids))
        for aid in aids[:max_assays_per_target]:
            try:
                csv = http.get(f"{PUG}/assay/aid/{aid}/concise/CSV").text
            except Exception:                                      # noqa: BLE001
                continue
            try:
                t = pd.read_csv(io.StringIO(csv), low_memory=False)
            except Exception:                                      # noqa: BLE001
                continue
            vcol = next((c for c in t.columns
                         if c.strip().lower() in
                         ("activity value [um]", "activity value (um)",
                          "acvalue", "activity value")), None)
            ncol = next((c for c in t.columns
                         if c.strip().lower() in ("activity name", "acname")), None)
            if vcol is None or "CID" not in t.columns:
                continue
            t = t.dropna(subset=[vcol])
            if ncol is not None:
                t = t[t[ncol].astype(str).str.upper().str.replace(" ", "")
                      .isin({"IC50", "KI", "KD", "EC50", "AC50", "XC50", "POTENCY"})]
            if t.empty:
                continue
            for _, r in t.iterrows():
                rows.append({
                    "cid": int(r["CID"]) if pd.notna(r["CID"]) else None,
                    "accession": acc,
                    "standard_type": (str(r[ncol]).upper().replace(" ", "")
                                      if ncol is not None else "IC50"),
                    "standard_relation": "=",
                    "standard_value": float(r[vcol]),
                    "standard_units": "uM",
                    "source": f"pubchem_aid{aid}",
                    "aid": aid,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = _attach_pubchem_smiles(http, df)
    summarize(df, "pubchem_bioassay", f"{len(accessions)} accessions")
    return df


def _attach_pubchem_smiles(http: Http, df: pd.DataFrame,
                           batch: int = 200) -> pd.DataFrame:
    cids = sorted({int(c) for c in df["cid"].dropna().unique()})
    LOG.info("resolving SMILES for %d PubChem CIDs", len(cids))
    mapping: dict[int, str] = {}
    for i in range(0, len(cids), batch):
        chunk = cids[i:i + batch]
        ids = ",".join(str(c) for c in chunk)
        for prop in ("IsomericSMILES", "CanonicalSMILES", "SMILES"):
            try:
                js = http.json(f"{PUG}/compound/cid/{ids}/property/{prop}/JSON")
                for p in js["PropertyTable"]["Properties"]:
                    s = p.get(prop)
                    if s:
                        mapping[int(p["CID"])] = s
                break
            except Exception:                                      # noqa: BLE001
                continue
    df = df.copy()
    df["smiles"] = df["cid"].map(mapping)
    missing = int(df["smiles"].isna().sum())
    if missing:
        LOG.warning("PubChem: %d rows without a resolvable SMILES, dropped", missing)
    return df.dropna(subset=["smiles"])


# --------------------------------------------------------------------------- #
# Drug Target Commons  (previously: zero rows, "no reachable bulk file")
# --------------------------------------------------------------------------- #
DTC_API_CANDIDATES = (
    "https://drugtargetcommons.fimm.fi/api/data/bioactivity/",
    "http://drugtargetcommons.fimm.fi/api/data/bioactivity/",
)
DTC_BULK_CANDIDATES = (
    "https://drugtargetcommons.fimm.fi/static/Excell_files/DTC_data.csv",
    "https://drugtargetcommons.fimm.fi/static/Excell_files/database.csv",
)


def dtc_by_accession(http: Http, accessions: list[str], *,
                     page_size: int = 1000, max_pages: int = 200) -> pd.DataFrame:
    """DTC via its paged REST API, one UniProt at a time.

    The previous harvest looked for a bulk file, found none reachable, and
    recorded zero rows. The API route works per-target, which is all tiers 1-2
    need; `dtc_bulk()` below handles the kinome-wide case and tells you exactly
    what to download by hand if it is still unreachable.
    """
    rows: list[dict] = []
    working: str | None = None
    for acc in accessions:
        endpoints = (working,) if working else DTC_API_CANDIDATES
        got_any = False
        for base in endpoints:
            if base is None:
                continue
            url: str | None = base
            params: dict | None = {"target_id": acc, "limit": page_size}
            pages = 0
            try:
                while url and pages < max_pages:
                    js = http.json(url, params=params)
                    params = None                 # `next` already carries them
                    recs = js.get("bioactivities") or js.get("results") or []
                    for r in recs:
                        rows.append({
                            "smiles": r.get("compound_smiles") or r.get("smiles"),
                            "inchikey_src": r.get("standard_inchi_key"),
                            "accession": r.get("target_id") or acc,
                            "standard_type": (r.get("standard_type") or "").upper(),
                            "standard_relation": r.get("standard_relation") or "=",
                            "standard_value": r.get("standard_value"),
                            "standard_units": r.get("standard_units"),
                            "source": "dtc_api",
                            "pubmed_id": r.get("pubmed_id"),
                        })
                    got_any = got_any or bool(recs)
                    url = js.get("next")
                    pages += 1
                working = base
                break
            except Exception as exc:                               # noqa: BLE001
                LOG.warning("DTC endpoint %s failed for %s: %s", base, acc, exc)
                continue
        if not got_any:
            LOG.warning("DTC: no rows for %s", acc)
    df = pd.DataFrame(rows)
    if df.empty:
        _report(HarvestReport(
            "dtc_api", status="empty",
            detail="no DTC API endpoint returned rows",
            action="Download the DTC bulk CSV by hand from "
                   "https://drugtargetcommons.fimm.fi/ (Download tab) and re-run "
                   "with --dtc-csv /path/to/DTC_data.csv"))
    else:
        summarize(df, "dtc_api", f"{len(accessions)} accessions")
    return df


def dtc_bulk(http: Http, layout: Layout, manifest: Manifest,
             accessions: set[str] | None = None, *,
             local_csv: Path | None = None, force: bool = False) -> pd.DataFrame:
    """DTC bulk CSV -> common schema. Accepts a hand-downloaded file."""
    path = local_csv
    if path is None:
        d = layout.download_dir("dtc")
        for url in DTC_BULK_CANDIDATES:
            try:
                path = http.download(url, d / Path(url).name, force=force)
                manifest.record(path, tier="3", source="dtc_bulk", url=url)
                break
            except Exception as exc:                               # noqa: BLE001
                LOG.warning("DTC bulk %s unreachable: %s", url, exc)
    if path is None or not Path(path).exists():
        _report(HarvestReport(
            "dtc_bulk", status="skipped",
            detail="no reachable bulk file and no --dtc-csv provided",
            action="Open https://drugtargetcommons.fimm.fi/, use the Download "
                   "tab to fetch the full CSV, then re-run: "
                   "python acquire.py tier3 --dtc-csv /path/to/DTC_data.csv"))
        return pd.DataFrame()
    frames = []
    for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False,
                             on_bad_lines="skip"):
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        acc_col = next((c for c in ("target_id", "uniprot_id", "target")
                        if c in chunk.columns), None)
        smi_col = next((c for c in ("compound_smiles", "smiles") if c in chunk.columns),
                       None)
        if acc_col is None or smi_col is None:
            LOG.warning("DTC bulk: unexpected columns %s", list(chunk.columns)[:12])
            break
        sub = chunk
        if accessions:
            sub = sub[sub[acc_col].isin(accessions)]
        if sub.empty:
            continue
        frames.append(pd.DataFrame({
            "smiles": sub[smi_col],
            "accession": sub[acc_col],
            "standard_type": sub.get("standard_type", pd.Series(index=sub.index))
                             .astype(str).str.upper(),
            "standard_relation": sub.get("standard_relation",
                                         pd.Series("=", index=sub.index)).fillna("="),
            "standard_value": pd.to_numeric(sub.get("standard_value"), errors="coerce"),
            "standard_units": sub.get("standard_units"),
            "source": "dtc_bulk",
        }))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summarize(df, "dtc_bulk", str(path))
    return df


# --------------------------------------------------------------------------- #
# Davis / KIBA / Metz  (previously: KIBA shape mismatch, Metz zero rows)
# --------------------------------------------------------------------------- #
def tdc_dti(name: str, layout: Layout) -> pd.DataFrame:
    """Davis / KIBA via TDC, which ships them already in long format.

    This sidesteps the (2111, 230) vs (2111, 229) failure entirely. That bug is
    a positional-alignment bug: the DeepDTA release stores drugs and targets as
    *dicts*, and the affinity matrix must be indexed by the dict key order, not
    by a positional slice -- one extra or missing key and every column shifts by
    one, which does not raise, it just relabels your whole dataset. See
    `deepdta_matrix()` for the aligned loader if you need the raw files.
    """
    tdc = try_import("tdc")
    if tdc is None:
        _report(HarvestReport(f"tdc_{name.lower()}", status="skipped",
                              detail="PyTDC not installed",
                              action="pip install PyTDC"))
        return pd.DataFrame()
    try:
        from tdc.multi_pred import DTI                              # noqa: PLC0415
        data = DTI(name=name, path=str(layout.download_dir(f"tdc_{name.lower()}")))
        if name.upper() == "DAVIS":
            data.convert_to_log(form="binding")     # Kd(nM) -> pKd
        df = data.get_data()
    except Exception as exc:                                       # noqa: BLE001
        _report(HarvestReport(f"tdc_{name.lower()}", status="failed", detail=str(exc)))
        return pd.DataFrame()
    df = df.rename(columns={"Drug": "smiles", "Target": "target_sequence",
                            "Target_ID": "target_id", "Drug_ID": "compound_id",
                            "Y": "y"})
    df["source"] = f"tdc_{name.lower()}"
    # Davis -> pKd (already -log10 M); KIBA -> KIBA score, NOT a pActivity.
    df["score_kind"] = "pKd" if name.upper() == "DAVIS" else "kiba_score"
    summarize(df.assign(accession=df.get("target_id")), f"tdc_{name.lower()}",
              f"score_kind={df['score_kind'].iat[0] if len(df) else '?'}")
    return df


def deepdta_matrix(folder: Path, source: str = "kiba") -> pd.DataFrame:
    """Aligned loader for the raw DeepDTA release (ligands/proteins/Y).

    The fix for the shape mismatch: read the two JSON dicts, take their key
    order as the authoritative axis labels, and ASSERT that Y.shape equals
    (len(ligands), len(proteins)). If it does not, we say which axis is off by
    how much instead of reshaping until it fits.
    """
    lig_p = next(folder.rglob("ligands_can.txt"), None) or next(
        folder.rglob("ligands_iso.txt"), None)
    prot_p = next(folder.rglob("proteins.txt"), None)
    y_p = next(folder.rglob("Y"), None)
    if not (lig_p and prot_p and y_p):
        _report(HarvestReport(f"deepdta_{source}", status="failed",
                              detail=f"missing ligands/proteins/Y under {folder}"))
        return pd.DataFrame()
    ligands = json.loads(lig_p.read_text())
    proteins = json.loads(prot_p.read_text())
    with y_p.open("rb") as fh:
        Y = np.asarray(pickle.load(fh, encoding="latin1"), dtype="float64")  # noqa: S301
    nL, nP = len(ligands), len(proteins)
    if Y.shape != (nL, nP):
        if Y.shape == (nP, nL):
            LOG.warning("deepdta_%s: Y is transposed (%s); transposing",
                        source, Y.shape)
            Y = Y.T
        else:
            _report(HarvestReport(
                f"deepdta_{source}", status="failed",
                detail=(f"Y.shape={Y.shape} but ligands={nL}, proteins={nP}; "
                        f"axis mismatch of {Y.shape[0]-nL:+d} rows / "
                        f"{Y.shape[1]-nP:+d} cols -- do NOT reshape, the key "
                        f"order would shift and silently relabel every pair"),
                action="re-download the DeepDTA release; if the mismatch "
                       "persists use the TDC copy (tdc_dti) instead"))
            return pd.DataFrame()
    lig_keys, prot_keys = list(ligands), list(proteins)
    rows_i, cols_j = np.where(np.isfinite(Y))
    df = pd.DataFrame({
        "compound_id": [lig_keys[i] for i in rows_i],
        "smiles": [ligands[lig_keys[i]] for i in rows_i],
        "target_id": [prot_keys[j] for j in cols_j],
        "target_sequence": [proteins[prot_keys[j]] for j in cols_j],
        "y": Y[rows_i, cols_j],
        "source": f"deepdta_{source}",
        "score_kind": "kiba_score" if source == "kiba" else "pKd",
    })
    summarize(df, f"deepdta_{source}", f"Y {Y.shape} aligned to dict key order")
    return df


METZ_SOURCES = (
    # Metz et al. 2011 kinase profiling. The supplementary table is the only
    # primary distribution; PubChem AID 1451 mirrors the same screen.
    ("pubchem_aid", "1451"),
)


def metz(http: Http, layout: Layout) -> pd.DataFrame:
    """Metz kinase profiling via its PubChem BioAssay mirror.

    Previous harvest: zero rows, no reason recorded. The underlying issue is
    that Metz has no stable bulk endpoint -- the primary release is a journal
    supplementary table. PubChem AID 1451 carries the same Ki panel and is
    fetchable, so that is what we use.
    """
    rows: list[dict] = []
    for kind, aid in METZ_SOURCES:
        if kind != "pubchem_aid":
            continue
        try:
            csv = http.get(f"{PUG}/assay/aid/{aid}/concise/CSV").text
            t = pd.read_csv(io.StringIO(csv), low_memory=False)
        except Exception as exc:                                   # noqa: BLE001
            _report(HarvestReport(
                "metz", status="failed", detail=f"AID {aid}: {exc}",
                action="download the Metz et al. (Nat Chem Biol 2011) "
                       "supplementary Ki table and pass --metz-csv"))
            return pd.DataFrame()
        vcol = next((c for c in t.columns if "activity value" in c.lower()), None)
        if vcol is None:
            _report(HarvestReport("metz", status="empty",
                                  detail=f"AID {aid} has no activity value column "
                                         f"({list(t.columns)[:8]})"))
            return pd.DataFrame()
        t = t.dropna(subset=[vcol])
        for _, r in t.iterrows():
            rows.append({"cid": r.get("CID"), "standard_type": "Ki",
                         "standard_relation": "=", "standard_value": r[vcol],
                         "standard_units": "uM", "source": f"metz_aid{aid}",
                         "accession": None})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = _attach_pubchem_smiles(http, df)
    summarize(df, "metz", "via PubChem mirror")
    return df


# --------------------------------------------------------------------------- #
# Papyrus -- the single best curated kinase-rich bioactivity set
# --------------------------------------------------------------------------- #
def papyrus(layout: Layout, *, accessions: set[str] | None = None,
            stereo: bool = False) -> pd.DataFrame:
    """Papyrus (Béquignon et al.) via the `papyrus-scripts` package.

    ~60 M standardised bioactivity points with per-target quality flags, heavily
    weighted towards kinases. For a project bottlenecked on 172 exact TNK1
    points this is the highest-yield download in the whole list after ChEMBL.
    """
    mod = try_import("papyrus_scripts")
    if mod is None:
        _report(HarvestReport(
            "papyrus", status="skipped", detail="papyrus-scripts not installed",
            action="pip install papyrus-scripts && "
                   "python -m papyrus_scripts download -V latest"))
        return pd.DataFrame()
    d = layout.download_dir("papyrus")
    try:
        subprocess.run(                                            # noqa: S603
            [sys.executable, "-m", "papyrus_scripts", "download",
             "-V", "latest", "-d", str(d), "--more", "none"],
            check=True, capture_output=True, text=True, timeout=36000)
    except subprocess.CalledProcessError as exc:
        LOG.warning("papyrus download exited %s: %s", exc.returncode,
                    (exc.stderr or "")[-800:])
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("papyrus download failed: %s", exc)
    try:
        from papyrus_scripts import PapyrusDataset                 # noqa: PLC0415
        ds = PapyrusDataset(version="latest", plusplus=True,
                            is3d=stereo, source_path=str(d))
        if accessions:
            ds = ds.proteins(accessions=list(accessions)) if hasattr(ds, "proteins") \
                 else ds
        df = ds.to_dataframe()
    except Exception as exc:                                       # noqa: BLE001
        _report(HarvestReport("papyrus", status="failed", detail=str(exc),
                              action="see papyrus-scripts docs; the bulk files "
                                     "are also on Zenodo"))
        return pd.DataFrame()
    df = df.rename(columns={"SMILES": "smiles", "accession": "accession",
                            "pchembl_value_Mean": "pchembl_value",
                            "type_IC50": "is_ic50"})
    df["source"] = "papyrus"
    if accessions and "accession" in df.columns:
        df = df[df["accession"].isin(accessions)]
    summarize(df, "papyrus", f"accession filter={bool(accessions)}")
    return df
