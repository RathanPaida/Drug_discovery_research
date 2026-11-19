#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
=====================================================================================
 TNK1 / KINOME BIOACTIVITY HARVESTER   (one file - 10 sources - resumable)
=====================================================================================
Goal   : collect measured compound-kinase bioactivity for TNK1 + a broad panel of
         related human kinases, so that a transfer-learning model (GNN etc.) has
         enough data.  Nothing is invented, simulated or duplicated artificially.

Install (once):
    pip install requests pandas numpy rdkit openpyxl

Run:
    python tnk1_harvest.py --check          # 1 min: tests every endpoint from YOUR network
    python tnk1_harvest.py                  # full run (hours; safe to Ctrl+C and re-run)
    python tnk1_harvest.py --panel core     # quick small run (14 kinases) to test everything
    python tnk1_harvest.py --sources davis,kiba,iuphar      # only some sources
    python tnk1_harvest.py --postprocess-only               # just rebuild master + report
    python tnk1_harvest.py --fresh                          # ignore checkpoints, start over

Output folder (default ./tnk1_data):
    csv/CHEMBL.csv  PUBCHEM.csv  BINDINGDB.csv  DTC.csv  IUPHAR.csv
    csv/PKIS.csv    PKIS2.csv    DAVIS.csv      KIBA.csv METZ.csv     <- one CSV per source
    TNK1_kinase_master.csv                    <- all sources merged (+ structure/dup flags)
    counts_by_source_kinase.csv               <- records per source and per kinase
    REPORT.txt                                <- counts, overlap, failures, what to do manually
    manifest.json                             <- checkpoint (do not delete if you want to resume)
    _work/                                    <- raw downloads, partial pages, caches

Sources (all 10 requested):
  1 ChEMBL (REST API, paginated, per-target)        6 PKIS  (via the ChEMBL deposit of Elkins 2016)
  2 PubChem BioAssay (PUG-REST, AID enumeration)    7 PKIS2 (via ChEMBL discovery; manual file ok)
  3 BindingDB (bulk TSV download, API fallback)     8 Davis (DeepDTA GitHub copy of Davis 2011)
  4 Drug Target Commons (bulk CSV / API discovery)  9 KIBA  (DeepDTA GitHub copy)
  5 IUPHAR/BPS Guide to PHARMACOLOGY (bulk + API)  10 Metz  (via ChEMBL deposit of Metz 2011)

Scientific rules enforced
  * TNK1 and TNK2 are always separate targets. No compound is labelled a TNK1 inhibitor
    because it hits TNK2/SRC - every row keeps its own target.
  * Original values and units are kept. Davis/KIBA/PKIS/Metz values are NOT converted
    into IC50/Ki/Kd unless the original definition is that quantity (see measure_semantics).
  * Censored values ('>' / '<') are kept with their relation; Davis 10000 nM is flagged '>'.
  * Non-human targets are labelled and kept in the source CSVs, excluded from the master.
  * Rows without a structure are NOT dropped in source CSVs (structure_status says why).
  * Tankyrase/PARP-type assays (which pollute 'TNK1' name searches) are flagged, not hidden.
  * Cross-source duplicates are FLAGGED (standardised-structure based), never silently
    deleted and never multiplied.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import re
import shutil
import sqlite3
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

VERSION = "1.0"

# =====================================================================================
#  CONFIGURATION  (edit here, or override from the command line)
# =====================================================================================
CONFIG = {
    "OUTPUT_DIR": "tnk1_data",
    # enable / disable individual sources
    "SOURCES": {"chembl": True, "pubchem": True, "bindingdb": True, "dtc": True, "iuphar": True,
                "pkis": True, "pkis2": True, "davis": True, "kiba": True, "metz": True},
    # core | tyrosine | kinome | broad
    #   core     = the core list below (14 kinases)           -> quick test
    #   tyrosine = core + every human tyrosine kinase (UniProt family)
    #   kinome   = core + every human protein kinase (UniProt 'protein kinase superfamily')  [default]
    #   broad    = kinome + every human protein with the UniProt keyword 'Kinase' (lipid/metabolic kinases too)
    "PANEL_MODE": "kinome",
    # TNK1 and TNK2 are ALWAYS separate. Edit / extend freely.
    # (the first 10 = TNK1, TNK2, SRC, ABL1, BTK, YES1, FYN, LCK, HCK, ABL2 ; last 4 = extras you used before)
    "CORE_TARGETS": {"TNK1": "Q13470", "TNK2": "Q07912", "SRC": "P12931", "ABL1": "P00519",
                     "BTK": "Q06187", "YES1": "P07947", "FYN": "P06241", "LCK": "P06239",
                     "HCK": "P08631", "ABL2": "P42684",
                     "TEC": "P42680", "FES": "P07332", "FER": "P16591", "PTK2": "Q05397"},
    "EXTRA_TARGETS": {},                 # add your own {"GENE": "UNIPROT"}
    "RESUME": True,
    "THREADS": 4,                        # moderate concurrency (ChEMBL / PubChem jobs)
    "EMAIL": "",                         # optional: put your e-mail in the User-Agent (polite to the APIs)
    # ---- ChEMBL
    "CHEMBL_PAGE_SIZE": 1000,
    "CHEMBL_ACTIVITY_TYPES": None,       # None = every activity type; or e.g. ["IC50","Ki","Kd","EC50"]
    "CHEMBL_INCLUDE_COMPLEXES": False,   # protein complexes / families as separate labelled rows
    "CHEMBL_MAX_TARGETS": None,          # testing only
    # ---- PubChem
    "PUBCHEM_INCLUDE_QUALITATIVE": True,   # outcome-only rows (active/inactive) go in with record_class='qualitative'
    "PUBCHEM_MAX_AIDS": None,              # None = all assays found for the panel
    "PUBCHEM_MAX_ROWS_PER_AID": None,
    "PUBCHEM_CID_BATCH": 1000,
    "PUBCHEM_FTP_FALLBACK": True,          # bulk zip fallback for AIDs the REST service cannot deliver
    # ---- BindingDB
    "BINDINGDB_API_FALLBACK": True,
    # ---- post-processing
    "STANDARDIZE": True,                   # RDKit standardisation after raw download
    "MAX_REMOTE_STRUCTURE_LOOKUPS": 100000,  # PubChem InChIKey->SMILES for records that have no structure
    "MASTER_DROP_NO_STRUCTURE": False,
    "MASTER_ONLY_PANEL": False,            # True = master contains only panel kinases (Davis/KIBA/PKIS have extra kinases)
    "CHUNK_ROWS": 250_000,
}

SOURCE_ORDER_RUN = ["davis", "kiba", "iuphar", "bindingdb", "dtc", "pkis", "pkis2", "metz", "chembl", "pubchem"]
SOURCE_ORDER_MASTER = ["chembl", "bindingdb", "dtc", "iuphar", "pkis", "pkis2", "metz", "davis", "kiba", "pubchem"]

URLS = {
    "chembl": "https://www.ebi.ac.uk/chembl/api/data",
    "chembl_host": "https://www.ebi.ac.uk",
    "uniprot": "https://rest.uniprot.org/uniprotkb/stream",
    "pug": "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
    "pubchem_ftp": "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data",
    "bdb_page": "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp",
    "bdb_hosts": ["https://www.bindingdb.org", "https://bindingdb.org"],
    "bdb_api": "https://www.bindingdb.org/axis2/services/BDBService/getLigandsByUniprots",
    "dtc_static": ["https://drugtargetcommons.fimm.fi/static/Excell_files/DTC_data.csv"],
    "dtc_api_root": "https://drugtargetcommons.fimm.fi/api/",
    "gtop_data": "https://www.guidetopharmacology.org/DATA",
    "gtop_api": "https://www.guidetopharmacology.org/services",
    "deepdta": "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data",
}

# ---------------------------------------------------------------------------------
#  Scientific constants
# ---------------------------------------------------------------------------------
# heuristic grouping only (replace with a sequence-based measure if you have one)
ACK_FAMILY = {"TNK1", "TNK2"}
CYTO_TK = {"SRC", "YES1", "FYN", "FGR", "LYN", "HCK", "LCK", "BLK", "FRK", "SRMS", "ABL1", "ABL2",
           "TEC", "BTK", "ITK", "BMX", "TXK", "FES", "FER", "PTK2", "PTK2B", "CSK", "MATK",
           "SYK", "ZAP70", "JAK1", "JAK2", "JAK3", "TYK2"}

# DiscoverX / Davis names that differ from HGNC symbols
ALIASES = {"P38-ALPHA": "MAPK14", "P38-BETA": "MAPK11", "P38-DELTA": "MAPK13", "P38-GAMMA": "MAPK12",
           "ERK1": "MAPK3", "ERK2": "MAPK1", "ERK3": "MAPK6", "ERK4": "MAPK4", "ERK5": "MAPK7",
           "ERK8": "MAPK15", "JNK1": "MAPK8", "JNK2": "MAPK9", "JNK3": "MAPK10",
           "PKAC-ALPHA": "PRKACA", "PKAC-BETA": "PRKACB", "AMPK-ALPHA1": "PRKAA1",
           "AMPK-ALPHA2": "PRKAA2", "IKK-ALPHA": "CHUK", "IKK-BETA": "IKBKB", "IKK-EPSILON": "IKBKE",
           "CDK4-CYCLIND1": "CDK4", "CDK4-CYCLIND3": "CDK4", "FAK": "PTK2", "ACK": "TNK2"}
NONHUMAN_HINTS = {"PFALCIPARUM": "Plasmodium falciparum", "MTUBERCULOSIS": "Mycobacterium tuberculosis"}

CONC_TYPES = {"IC50", "KI", "KD", "EC50", "AC50", "XC50", "POTENCY", "GI50", "ED50"}
P_TYPES = {"IC50", "KI", "KD", "EC50", "AC50", "XC50", "POTENCY"}
TYPE_ALIASES = {
    "KI": "Ki", "KD": "Kd", "IC50": "IC50", "EC50": "EC50", "AC50": "AC50", "XC50": "XC50",
    "POTENCY": "Potency", "GI50": "GI50", "ED50": "ED50",
    "INHIBITION": "PCT_INHIBITION", "% INHIBITION": "PCT_INHIBITION", "% OF INHIBITION": "PCT_INHIBITION",
    "PERCENT INHIBITION": "PCT_INHIBITION", "INH": "PCT_INHIBITION", "%INHIBITION": "PCT_INHIBITION",
    "% CONTROL": "PCT_CONTROL", "% OF CONTROL": "PCT_CONTROL", "% CTRL": "PCT_CONTROL",
    "PERCENT CONTROL": "PCT_CONTROL", "%CONTROL": "PCT_CONTROL",
    "% ACTIVITY": "PCT_ACTIVITY", "PERCENT ACTIVITY": "PCT_ACTIVITY",
    "KON": "kon", "K_OFF": "koff", "KOFF": "koff",
}
UNIT_FACTOR_NM = {"nm": 1.0, "nmol/l": 1.0, "nmol l-1": 1.0, "um": 1e3, "umol/l": 1e3, "mm": 1e6,
                  "mmol/l": 1e6, "m": 1e9, "mol/l": 1e9, "pm": 1e-3, "pmol/l": 1e-3, "fm": 1e-6,
                  "nanomolar": 1.0, "micromolar": 1e3, "millimolar": 1e6, "picomolar": 1e-3}
BAD_DESC = re.compile(r"tankyrase|\bPARP|poly\s*\(?ADP", re.I)
KNOWN_BAD_ASSAYS = {"CHEMBL5734558"}   # tankyrase enzyme assay that was mis-mapped to TNK1 (found in your earlier run)
KNOWN_BAD_DOCS = {"CHEMBL5727391"}

# unified schema of every source CSV
COLS = [
    "source", "source_record_id", "source_release",
    "target_name_original", "target_name_normalized", "gene_symbol", "uniprot_accession",
    "target_chembl_id", "target_id_source", "target_organism", "target_type",
    "target_mapping_status", "variant_info",
    "compound_id", "compound_name", "smiles_raw", "inchikey_source", "pubchem_cid", "chembl_molecule_id",
    "activity_type_original", "activity_type", "relation", "value_original", "units_original",
    "value_nM", "p_activity", "pchembl_value", "value_std", "units_std",
    "record_class", "measure_semantics",
    "assay_id", "assay_type", "assay_description", "assay_organism", "assay_cell_type",
    "confidence_score", "doc_id", "pmid", "doi", "pub_year",
    "data_validity_comment", "activity_comment", "flag_contamination", "notes",
]
NUM_COLS = ["value_original", "value_nM", "p_activity", "pchembl_value", "value_std"]

_print_lock = threading.Lock()
T0 = time.time()


def log(msg, src=""):
    el = time.time() - T0
    stamp = f"{int(el // 3600):02d}:{int(el % 3600 // 60):02d}:{int(el % 60):02d}"
    with _print_lock:
        print(f"[{stamp}]{'[' + src + ']' if src else ''} {msg}", flush=True)


# =====================================================================================
#  small utilities
# =====================================================================================
def atomic_write_text(path: Path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Manifest:
    """Checkpoint file. Nested dict, atomic writes, thread safe."""

    def __init__(self, path: Path, fresh=False):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.data = {}
        self._last = 0.0
        if self.path.exists() and not fresh:
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                bak = self.path.with_suffix(".corrupt")
                shutil.copy(self.path, bak)
                log(f"manifest unreadable, backed up to {bak.name}; starting a new one")

    def get(self, *keys, default=None):
        with self.lock:
            d = self.data
            for k in keys:
                if not isinstance(d, dict) or k not in d:
                    return default
                d = d[k]
            return d

    def set(self, *keys_and_value, force=False):
        *keys, value = keys_and_value
        with self.lock:
            d = self.data
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
            self.save(force)

    def save(self, force=False):
        with self.lock:
            now = time.time()
            if not force and now - self._last < 2.0:
                return
            atomic_write_text(self.path, json.dumps(self.data, indent=1, default=str))
            self._last = now


class Http:
    """requests wrapper: timeouts, exponential back-off, per-host throttling, resumable streaming."""

    def __init__(self, user_agent, timeout=(20, 180)):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept": "*/*"})
        ad = HTTPAdapter(pool_connections=16, pool_maxsize=32)
        self.s.mount("https://", ad)
        self.s.mount("http://", ad)
        self.timeout = timeout
        self.gap = {"chembl": 0.08, "pubchem": 0.25, "uniprot": 0.1, "iuphar": 0.15, "bindingdb": 0.5}
        self._next = {}
        self._lk = threading.Lock()
        self.last_error = ""

    def _throttle(self, key):
        gap = self.gap.get(key, 0)
        if not gap:
            return
        with self._lk:
            now = time.monotonic()
            t = max(now, self._next.get(key, 0.0))
            self._next[key] = t + gap
        if t > now:
            time.sleep(t - now)

    def request(self, method, url, *, key="default", params=None, data=None, headers=None,
                stream=False, tries=6, timeout=None):
        resp = None
        for i in range(tries):
            self._throttle(key)
            try:
                resp = self.s.request(method, url, params=params, data=data, headers=headers,
                                      stream=stream, timeout=timeout or self.timeout)
            except requests.exceptions.RequestException as e:
                self.last_error = f"{type(e).__name__}: {e}"
                resp = None
                time.sleep(min(60, 2 ** i + random.random()))
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                ra = resp.headers.get("Retry-After", "")
                wait = float(ra) if ra.isdigit() else min(90, 2 ** i + random.random())
                self.last_error = f"HTTP {resp.status_code}"
                if i < tries - 1:
                    resp.close()
                    time.sleep(wait)
                    continue
            return resp
        return resp

    def get(self, url, **kw):
        return self.request("GET", url, **kw)

    def post(self, url, **kw):
        return self.request("POST", url, **kw)

    def get_json(self, url, **kw):
        r = self.get(url, **kw)
        if r is None or r.status_code != 200:
            return None
        try:
            return r.json()
        except Exception:
            return None

    def download(self, urls, dest: Path, key="default", min_bytes=1000, magic=None, src=""):
        """Resumable streamed download from the first URL that works. Returns Path or None."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size >= min_bytes:
            return dest
        part = dest.with_name(dest.name + ".part")
        for url in urls:
            complete = False
            for attempt in range(8):
                start = part.stat().st_size if part.exists() else 0
                hdr = {"Range": f"bytes={start}-"} if start else {}
                r = self.get(url, key=key, headers=hdr, stream=True, tries=3)
                if r is None:
                    log(f"no response from {url} ({self.last_error})", src)
                    time.sleep(3)
                    continue
                sc = r.status_code
                if sc == 416:
                    r.close()
                    complete = True
                    break
                if sc not in (200, 206):
                    log(f"{url} -> HTTP {sc}", src)
                    r.close()
                    break
                total = None
                if sc == 206:
                    m = re.search(r"/(\d+)$", r.headers.get("Content-Range", ""))
                    total = int(m.group(1)) if m else None
                    mode = "ab"
                else:
                    cl = r.headers.get("Content-Length")
                    total = int(cl) if cl and cl.isdigit() else None
                    mode = "wb"
                    start = 0
                got, t_last, t0 = start, time.time(), time.time()
                try:
                    with open(part, mode) as f:
                        for chunk in r.iter_content(1 << 20):
                            if not chunk:
                                continue
                            f.write(chunk)
                            got += len(chunk)
                            if time.time() - t_last > 10:
                                t_last = time.time()
                                pct = f" ({100 * got / total:.0f}%)" if total else ""
                                log(f"downloading {dest.name}: {got / 1e6:.0f} MB{pct}", src)
                except requests.exceptions.RequestException as e:
                    log(f"connection dropped ({type(e).__name__}); resuming ...", src)
                    time.sleep(3)
                    continue
                finally:
                    r.close()
                if total and part.stat().st_size < total:
                    continue
                complete = True
                break
            if not complete:
                continue
            with open(part, "rb") as f:
                head = f.read(8)
            if part.stat().st_size < min_bytes or head[:1] == b"<" or (magic and not head.startswith(magic)):
                log(f"{url} did not deliver a data file (got HTML/too small) - skipping", src)
                part.unlink(missing_ok=True)
                continue
            os.replace(part, dest)
            return dest
        return None


class Cache:
    """sqlite cache: kv (API lookups), std (standardised structures), lk (structure lookups)."""

    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=120)
        self.lock = threading.RLock()
        with self.lock:
            c = self.conn
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("CREATE TABLE IF NOT EXISTS kv(ns TEXT, k TEXT, v TEXT, PRIMARY KEY(ns,k)) WITHOUT ROWID")
            c.execute("CREATE TABLE IF NOT EXISTS std(smiles TEXT PRIMARY KEY, std TEXT, ik TEXT, status TEXT) WITHOUT ROWID")
            c.execute("CREATE TABLE IF NOT EXISTS lk(kind TEXT, k TEXT, smiles TEXT, PRIMARY KEY(kind,k)) WITHOUT ROWID")
            c.commit()

    def _many(self, sql_tpl, keys, prefix=()):
        out = []
        keys = list(keys)
        with self.lock:
            for i in range(0, len(keys), 500):
                b = keys[i:i + 500]
                q = sql_tpl.format(",".join("?" * len(b)))
                out.extend(self.conn.execute(q, (*prefix, *b)).fetchall())
        return out

    def kv_get(self, ns, keys):
        return {k: v for k, v in self._many("SELECT k,v FROM kv WHERE ns=? AND k IN ({})", keys, (ns,))}

    def kv_put(self, ns, items: dict):
        with self.lock:
            self.conn.executemany("INSERT OR REPLACE INTO kv VALUES(?,?,?)", [(ns, str(k), v) for k, v in items.items()])
            self.conn.commit()

    def std_get(self, smiles):
        return {r[0]: (r[1], r[2], r[3]) for r in self._many("SELECT smiles,std,ik,status FROM std WHERE smiles IN ({})", smiles)}

    def std_known(self):
        with self.lock:
            return {r[0] for r in self.conn.execute("SELECT smiles FROM std")}

    def std_put(self, rows):
        with self.lock:
            self.conn.executemany("INSERT OR REPLACE INTO std VALUES(?,?,?,?)", rows)
            self.conn.commit()

    def lk_put(self, kind, items):
        with self.lock:
            self.conn.executemany("INSERT OR IGNORE INTO lk VALUES(?,?,?)", [(kind, k, s) for k, s in items])
            self.conn.commit()

    def lk_get(self, kind, keys):
        return {r[0]: r[1] for r in self._many("SELECT k,smiles FROM lk WHERE kind=? AND k IN ({})", keys, (kind,))}


# =====================================================================================
#  vectorised normalisation  (used by every source)
# =====================================================================================
REL_RE = re.compile(r"^\s*(>=|<=|>>|<<|>|<|~|=)?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def split_rel_value(series: pd.Series):
    ex = series.astype(str).str.replace(",", "", regex=False).str.extract(REL_RE)
    rel = ex[0].fillna("=").replace({">>": ">", "<<": "<", "": "="})
    return rel, pd.to_numeric(ex[1], errors="coerce")


def norm_type(s: pd.Series) -> pd.Series:
    raw = s.fillna("").astype(str).str.strip()
    key = raw.str.upper().str.replace(r"\s+", " ", regex=True)
    return key.map(TYPE_ALIASES).fillna(raw)


def norm_units(s: pd.Series) -> pd.Series:
    return (s.fillna("").astype(str).str.strip().str.lower()
            .str.replace("µ", "u", regex=False).str.replace("μ", "u", regex=False))


def ensure(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan if c in NUM_COLS else ""
    return df


def derive(df: pd.DataFrame, source: str, release: str = "") -> pd.DataFrame:
    """Fill the derived columns. Source code supplies original type/value/units; this does the rest."""
    df = ensure(df.copy())
    df["source"] = source
    if release:
        df["source_release"] = release
    for c in COLS:
        if c not in NUM_COLS:
            df[c] = df[c].fillna("").astype(str)
    df["activity_type"] = np.where(df["activity_type"].str.len() > 0, df["activity_type"], norm_type(df["activity_type_original"]))
    val = pd.to_numeric(df["value_original"], errors="coerce")
    df["value_original"] = val
    rel = df["relation"].str.strip().replace({">>": ">", "<<": "<", "": "="})
    df["relation"] = rel
    utype = df["activity_type"].str.upper()
    conc = utype.isin(CONC_TYPES)
    fac = norm_units(df["units_original"]).map(UNIT_FACTOR_NM)
    vnm = pd.to_numeric(df["value_nM"], errors="coerce")
    calc = val * fac
    calc = calc.where(conc & (val > 0))
    vnm = vnm.fillna(calc)
    p_in = pd.to_numeric(df["p_activity"], errors="coerce")
    vnm = vnm.fillna(10.0 ** (9.0 - p_in))                       # p-value supplied by source (IUPHAR)
    df["value_nM"] = vnm
    with np.errstate(divide="ignore", invalid="ignore"):
        pcalc = 9.0 - np.log10(vnm.where(vnm > 0))
    df["p_activity"] = p_in.fillna(pcalc.where(utype.isin(P_TYPES)))
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    isp = utype.str.startswith("PCT_")
    df["value_std"] = np.where(vnm.notna(), vnm, val)
    df["units_std"] = np.where(vnm.notna(), "nM", np.where(isp, "%", df["units_original"]))
    empty_cls = df["record_class"] == ""
    cls = np.where(vnm.notna() & conc, "quantitative", np.where(isp & val.notna(), "single_point",
                   np.where(val.notna(), "other_numeric", "qualitative")))
    df["record_class"] = np.where(empty_cls, cls, df["record_class"])
    desc = df["assay_description"].str.contains(BAD_DESC)
    bad = df["assay_id"].isin(KNOWN_BAD_ASSAYS) | df["doc_id"].isin(KNOWN_BAD_DOCS)
    flag = np.where(desc, "tankyrase_or_PARP_in_assay_description", "")
    flag = np.where(bad, "known_contaminated_assay_or_document", flag)
    df["flag_contamination"] = np.where(df["flag_contamination"] == "", flag, df["flag_contamination"])
    return df[COLS]


def append_df(path: Path, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = (not path.exists()) or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=new, index=False)
    return len(df)


def touch_header(path: Path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=COLS).to_csv(path, index=False)


def concat_parts(parts, final: Path):
    """Join part CSVs (identical header) into one CSV, atomically, by streaming bytes."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(final.name + ".tmp")
    wrote_header = False
    n = 0
    with open(tmp, "wb") as out:
        for p in sorted(parts):
            if not Path(p).exists() or Path(p).stat().st_size == 0:
                continue
            with open(p, "rb") as f:
                header = f.readline()
                if not wrote_header:
                    out.write(header)
                    wrote_header = True
                shutil.copyfileobj(f, out, 1 << 22)
    if not wrote_header:
        pd.DataFrame(columns=COLS).to_csv(tmp, index=False)
    os.replace(tmp, final)
    return final


def count_rows(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return max(0, n - 1)   # approximate if fields contain newlines; exact counts come from the manifest


def pick(df, *names, regex=None):
    """first existing column by exact (case-insens.) name or regex"""
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    if regex:
        for c in df.columns:
            if re.search(regex, c, re.I):
                return c
    return None


def col_or_blank(df, name, default=""):
    if name and name in df.columns:
        return df[name].fillna("").astype(str)
    return pd.Series(default, index=df.index, dtype=object)


# =====================================================================================
#  context + kinase panel
# =====================================================================================
class Ctx:
    pass


def relatedness(gene: str, family: str) -> str:
    g = (gene or "").upper()
    if g in ACK_FAMILY:
        return "1_Ack_family(TNK1/TNK2)"
    if g in CYTO_TK:
        return "2_cytoplasmic_tyrosine_kinase"
    if "tyr protein kinase" in (family or "").lower():
        return "3_other_tyrosine_kinase"
    return "4_other_kinase"


def uniprot_query(ctx, query, label):
    params = {"query": query, "format": "tsv", "compressed": "false",
              "fields": "accession,gene_primary,gene_names,protein_name,xref_chembl,xref_geneid,protein_families,length"}
    r = ctx.http.get(URLS["uniprot"], params=params, key="uniprot", tries=4, timeout=(20, 300))
    if r is None or r.status_code != 200:
        log(f"UniProt query '{label}' failed ({r.status_code if r is not None else ctx.http.last_error})", "panel")
        return None
    df = pd.read_csv(io.StringIO(r.text), sep="\t", dtype=str).fillna("")
    if df.shape[1] < 8:
        return None
    df.columns = ["uniprot", "gene", "synonyms", "protein_name", "chembl_targets", "geneid", "family", "length"]
    return df


def build_panel(ctx) -> pd.DataFrame:
    cfg = ctx.cfg
    cache_file = ctx.work / "panel.json"
    key = json.dumps([cfg["PANEL_MODE"], cfg["CORE_TARGETS"], cfg["EXTRA_TARGETS"]], sort_keys=True)
    if cache_file.exists() and cfg["RESUME"]:
        try:
            j = json.loads(cache_file.read_text())
            if j.get("key") == key:
                log(f"panel loaded from cache: {len(j['rows'])} kinases", "panel")
                return pd.DataFrame(j["rows"]).fillna("")
        except Exception:
            pass
    core = dict(cfg["CORE_TARGETS"])
    core.update(cfg["EXTRA_TARGETS"])
    frames = []
    base = "(organism_id:9606) AND (reviewed:true)"
    accq = " OR ".join(f"accession:{a}" for a in core.values())
    f = uniprot_query(ctx, f"({accq})", "core")
    if f is not None:
        frames.append(f)
    mode = cfg["PANEL_MODE"]
    if mode in ("tyrosine", "kinome", "broad"):
        q = f'{base} AND (family:"tyr protein kinase family")' if mode == "tyrosine" else f'{base} AND (family:"protein kinase superfamily")'
        f = uniprot_query(ctx, q, mode)
        if f is not None:
            frames.append(f)
    if mode == "broad":
        f = uniprot_query(ctx, f"{base} AND (keyword:KW-0418)", "broad")
        if f is not None:
            frames.append(f)
    if frames:
        pan = pd.concat(frames).drop_duplicates("uniprot").reset_index(drop=True)
    else:
        log("UniProt unreachable -> panel limited to the CORE list (re-run later for the full panel)", "panel")
        pan = pd.DataFrame({"uniprot": list(core.values()), "gene": list(core.keys()), "synonyms": "", "protein_name": "",
                            "chembl_targets": "", "geneid": "", "family": "Tyr protein kinase family", "length": ""})
    core_acc = set(core.values())
    for g, a in core.items():                       # core targets must exist even if UniProt omitted them
        if a not in set(pan.uniprot):
            pan.loc[len(pan)] = [a, g, "", "", "", "", "", ""]
    # the gene name in the config wins for core targets (TNK1 and TNK2 stay separate)
    inv = {a: g for g, a in core.items()}
    pan["gene"] = [inv.get(a, g) for a, g in zip(pan.uniprot, pan.gene)]
    pan["gene"] = np.where(pan.gene == "", pan.uniprot, pan.gene)
    pan["tier"] = np.where(pan.uniprot.isin(core_acc), "core", "panel")
    pan["relatedness_tier"] = [relatedness(g, f) for g, f in zip(pan.gene, pan.family)]
    pan = pan.sort_values(["tier", "relatedness_tier", "gene"]).reset_index(drop=True)
    atomic_write_text(cache_file, json.dumps({"key": key, "rows": pan.to_dict("records")}))
    log(f"kinase panel ({mode}): {len(pan)} human kinases; core = {int((pan.tier == 'core').sum())}", "panel")
    return pan


def panel_indexes(ctx):
    pan = ctx.panel
    ctx.by_acc = {r.uniprot: r for r in pan.itertuples()}
    ctx.by_gene = {r.gene.upper(): r for r in pan.itertuples()}
    syn = {}
    for r in pan.itertuples():
        for s in re.split(r"[ ;,]+", r.synonyms or ""):
            s = s.strip().upper()
            if s and s != r.gene.upper():
                syn.setdefault(s, set()).add(r.uniprot)
    ctx.by_syn = {s: next(iter(a)) for s, a in syn.items() if len(a) == 1 and s not in ctx.by_gene}


def map_symbol(ctx, name: str):
    """name -> (panel row | None, status). Exact symbol first, alias table, unambiguous synonym."""
    n = (name or "").strip().upper()
    if not n:
        return None, "unmapped_no_name"
    if n in ALIASES:
        n = ALIASES[n]
        r = ctx.by_gene.get(n)
        if r:
            return r, "mapped_alias"
    if n in ctx.by_gene:
        return ctx.by_gene[n], "mapped_gene_symbol"
    base = re.split(r"[(\-]", n)[0]
    if base in ctx.by_gene and base != n:
        return ctx.by_gene[base], "mapped_gene_symbol_variant_stripped"
    if n in ctx.by_syn:
        return ctx.by_acc[ctx.by_syn[n]], "mapped_synonym_unambiguous"
    return None, "unmapped_not_in_panel"


# =====================================================================================
#  ChEMBL helpers (used by ChEMBL, PKIS, PKIS2, Metz)
# =====================================================================================
CHEMBL_ACT_FIELDS = [
    "activity_id", "assay_chembl_id", "assay_type", "assay_description", "assay_organism",
    "assay_variant_accession", "assay_variant_mutation", "bao_label", "canonical_smiles",
    "data_validity_comment", "document_chembl_id", "document_journal", "document_year",
    "molecule_chembl_id", "molecule_pref_name", "pchembl_value", "potential_duplicate",
    "relation", "standard_relation", "standard_text_value", "standard_type", "standard_units",
    "standard_value", "target_chembl_id", "target_organism", "target_pref_name", "type", "units",
    "value", "activity_comment",
]
_CHEMBL_MODE = {"ladder": None}


def chembl_pages(ctx, resource, params, start_offset=0, page=1000, src="chembl"):
    """Generator of (items, next_offset). Handles pagination, repeated-page guard, 'only'/'order_by' fallbacks."""
    url = f"{URLS['chembl']}/{resource}.json"
    offset = start_offset
    prev_first = None
    only = {"activity": CHEMBL_ACT_FIELDS}.get(resource)
    ladder = [(True, True), (False, True), (False, False)] if _CHEMBL_MODE["ladder"] is None else [_CHEMBL_MODE["ladder"]]
    pages_done = 0
    total = None
    while True:
        r = None
        for use_only, use_order in ladder:
            p = dict(params, limit=page, offset=offset)
            if use_only and only:
                p["only"] = ",".join(only)
            if use_order and resource == "activity":
                p["order_by"] = "activity_id"
            r = ctx.http.get(url, params=p, key="chembl", tries=7, timeout=(20, 240))
            if r is not None and r.status_code == 200:
                _CHEMBL_MODE["ladder"] = (use_only, use_order)
                ladder = [(use_only, use_order)]
                break
        if r is None or r.status_code != 200:
            code = r.status_code if r is not None else ctx.http.last_error
            raise RuntimeError(f"ChEMBL {resource} failed at offset {offset}: {code}")
        j = r.json()
        key = next((k for k, v in j.items() if isinstance(v, list)), None)
        items = j.get(key, []) if key else []
        meta = j.get("page_meta", {})
        total = meta.get("total_count", total)
        if not items:
            return
        first = json.dumps(items[0], sort_keys=True)[:300]
        if first == prev_first:
            log(f"{resource}: repeated page at offset {offset} - stopping to avoid an infinite loop", src)
            return
        prev_first = first
        offset += len(items)
        pages_done += 1
        yield items, offset
        if not meta.get("next"):
            return
        if total is not None and offset >= total:
            return


def chembl_target_info(ctx, tids):
    """target_chembl_id -> dict(gene, acc, pref, organism, ttype). Cached."""
    tids = [t for t in set(tids) if t]
    got = ctx.cache.kv_get("chembl_target", tids)
    out = {k: json.loads(v) for k, v in got.items()}
    need = [t for t in tids if t not in out]
    for i in range(0, len(need), 40):
        b = need[i:i + 40]
        j = ctx.http.get_json(f"{URLS['chembl']}/target.json", params={"target_chembl_id__in": ",".join(b), "limit": 100}, key="chembl")
        new = {}
        for t in (j or {}).get("targets", []):
            comps = t.get("target_components") or []
            accs = [c.get("accession") for c in comps if c.get("accession")]
            gene = ""
            for c in comps:
                for s in c.get("target_component_synonyms") or []:
                    if s.get("syn_type") == "GENE_SYMBOL":
                        gene = s.get("component_synonym", "")
                        break
                if gene:
                    break
            info = {"acc": ";".join(accs), "gene": gene, "pref": t.get("pref_name", ""),
                    "organism": t.get("organism", "") or "", "ttype": t.get("target_type", "")}
            new[t["target_chembl_id"]] = info
        for k in b:
            new.setdefault(k, {"acc": "", "gene": "", "pref": "", "organism": "", "ttype": ""})
        ctx.cache.kv_put("chembl_target", {k: json.dumps(v) for k, v in new.items()})
        out.update(new)
    return out


def chembl_df(ctx, items, tinfo, source, release=""):
    df = pd.DataFrame(items)
    for c in CHEMBL_ACT_FIELDS:
        if c not in df.columns:
            df[c] = None
    st = df["standard_type"].fillna(df["type"])
    sv = df["standard_value"].fillna(df["value"])
    su = df["standard_units"].fillna(df["units"])
    sr = df["standard_relation"].fillna(df["relation"])
    tids = df["target_chembl_id"].fillna("")
    accs, genes, status, ttype, tname, torg, tier = [], [], [], [], [], [], []
    for t in tids:
        ti = tinfo.get(t, {})
        acc_list = [a for a in (ti.get("acc") or "").split(";") if a]
        in_panel = [a for a in acc_list if a in ctx.by_acc]
        gene = ";".join(ctx.by_acc[a].gene for a in in_panel) if in_panel else (ti.get("gene") or "")
        multi = len(acc_list) > 1
        accs.append(";".join(acc_list))
        genes.append(gene)
        ttype.append(ti.get("ttype", ""))
        tname.append(ti.get("pref", ""))
        torg.append(ti.get("organism", ""))
        status.append("complex_multi_component" if multi else ("mapped_chembl_target" if in_panel else "unmapped_not_in_panel"))
    comment = df["activity_comment"].fillna("")
    text = df["standard_text_value"].fillna("")
    out = pd.DataFrame({
        "source_record_id": df["activity_id"].astype(str),
        "target_name_original": df["target_pref_name"].fillna(pd.Series(tname, index=df.index)),
        "gene_symbol": genes, "target_name_normalized": [g.upper() for g in genes],
        "uniprot_accession": accs, "target_chembl_id": tids, "target_id_source": "ChEMBL",
        "target_organism": df["target_organism"].fillna(pd.Series(torg, index=df.index)),
        "target_type": ttype, "target_mapping_status": status,
        "variant_info": df["assay_variant_mutation"].fillna(""),
        "compound_id": df["molecule_chembl_id"].fillna(""), "compound_name": df["molecule_pref_name"].fillna(""),
        "smiles_raw": df["canonical_smiles"].fillna(""), "chembl_molecule_id": df["molecule_chembl_id"].fillna(""),
        "activity_type_original": st, "relation": sr, "value_original": sv, "units_original": su,
        "pchembl_value": df["pchembl_value"],
        "assay_id": df["assay_chembl_id"].fillna(""), "assay_type": df["assay_type"].fillna(""),
        "assay_description": df["assay_description"].fillna(""), "assay_organism": df["assay_organism"].fillna(""),
        "doc_id": df["document_chembl_id"].fillna(""), "pub_year": df["document_year"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True),
        "data_validity_comment": df["data_validity_comment"].fillna(""),
        "activity_comment": np.where(comment != "", comment, text),
        "measure_semantics": "chembl_standardised_activity",
        "notes": np.where(df["potential_duplicate"].astype(str).isin(["1", "True", "true"]), "chembl_potential_duplicate", ""),
    })
    keep = (pd.to_numeric(out["value_original"], errors="coerce").notna() | (out["activity_comment"] != ""))
    out = out[keep]
    return derive(out, source, release)


def chembl_enrich_part(ctx, df):
    """fill pmid / doi / confidence / cell type from cached assay + document tables"""
    ids = [a for a in df["assay_id"].unique() if a]
    docs = [d for d in df["doc_id"].unique() if d]
    am = {k: json.loads(v) for k, v in ctx.cache.kv_get("chembl_assay", ids).items()}
    dm = {k: json.loads(v) for k, v in ctx.cache.kv_get("chembl_doc", docs).items()}
    df["confidence_score"] = df["assay_id"].map(lambda a: am.get(a, {}).get("conf", ""))
    df["assay_cell_type"] = df["assay_id"].map(lambda a: am.get(a, {}).get("cell", ""))
    df["pmid"] = df["doc_id"].map(lambda d: dm.get(d, {}).get("pmid", ""))
    df["doi"] = df["doc_id"].map(lambda d: dm.get(d, {}).get("doi", ""))
    yr = df["doc_id"].map(lambda d: dm.get(d, {}).get("year", ""))
    df["pub_year"] = np.where(df["pub_year"].astype(str) == "", yr, df["pub_year"])
    return df


def chembl_fetch_enrichment(ctx, assay_ids, doc_ids, src):
    have_a = ctx.cache.kv_get("chembl_assay", assay_ids)
    have_d = ctx.cache.kv_get("chembl_doc", doc_ids)
    need_a = [a for a in assay_ids if a not in have_a]
    need_d = [d for d in doc_ids if d not in have_d]
    log(f"enrichment: {len(need_a)} assays + {len(need_d)} documents to fetch", src)
    jobs = []
    for i in range(0, len(need_a), 100):
        jobs.append(("assay", need_a[i:i + 100]))
    for i in range(0, len(need_d), 100):
        jobs.append(("document", need_d[i:i + 100]))

    def run(job):
        kind, b = job
        if kind == "assay":
            j = ctx.http.get_json(f"{URLS['chembl']}/assay.json", params={"assay_chembl_id__in": ",".join(b), "limit": 100}, key="chembl")
            res = {}
            for a in (j or {}).get("assays", []):
                res[a["assay_chembl_id"]] = json.dumps({"conf": a.get("confidence_score") if a.get("confidence_score") is not None else "",
                                                       "cell": a.get("assay_cell_type") or ""})
            for k in b:
                res.setdefault(k, json.dumps({"conf": "", "cell": ""}))
            ctx.cache.kv_put("chembl_assay", res)
        else:
            j = ctx.http.get_json(f"{URLS['chembl']}/document.json", params={"document_chembl_id__in": ",".join(b), "limit": 100}, key="chembl")
            res = {}
            for d in (j or {}).get("documents", []):
                res[d["document_chembl_id"]] = json.dumps({"pmid": d.get("pubmed_id") or "", "doi": d.get("doi") or "",
                                                          "year": d.get("year") or ""})
            for k in b:
                res.setdefault(k, json.dumps({"pmid": "", "doi": "", "year": ""}))
            ctx.cache.kv_put("chembl_doc", res)
        return len(b)

    done = 0
    with ThreadPoolExecutor(ctx.cfg["THREADS"]) as ex:
        for f in as_completed([ex.submit(run, j) for j in jobs]):
            done += f.result()
            if done % 1000 < 100:
                log(f"enrichment progress {done}/{len(need_a) + len(need_d)}", src)


def chembl_resolve_targets(ctx):
    cache_file = ctx.work / "chembl_targets.json"
    key = hashlib.md5(",".join(sorted(ctx.panel.uniprot)).encode()).hexdigest() + str(ctx.cfg["CHEMBL_INCLUDE_COMPLEXES"])
    if cache_file.exists() and ctx.cfg["RESUME"]:
        j = json.loads(cache_file.read_text())
        if j.get("key") == key:
            return j["targets"]
    accs = list(ctx.panel.uniprot)
    found = {}

    def add(t):
        comps = t.get("target_components") or []
        a = [c.get("accession") for c in comps if c.get("accession")]
        found[t["target_chembl_id"]] = {"tid": t["target_chembl_id"], "ttype": t.get("target_type", ""),
                                        "pref": t.get("pref_name", ""), "organism": t.get("organism", ""), "accs": a}

    for i in range(0, len(accs), 25):
        b = accs[i:i + 25]
        j = ctx.http.get_json(f"{URLS['chembl']}/target.json", params={"target_components__accession__in": ",".join(b), "limit": 1000}, key="chembl")
        if j and j.get("targets") is not None:
            for t in j["targets"]:
                add(t)
        else:                                  # fallback: one accession per request
            for a in b:
                j = ctx.http.get_json(f"{URLS['chembl']}/target.json", params={"target_components__accession": a, "limit": 100}, key="chembl")
                for t in (j or {}).get("targets", []):
                    add(t)
        if (i // 25) % 5 == 0:
            log(f"resolving ChEMBL targets {min(i + 25, len(accs))}/{len(accs)}", "chembl")
    for r in ctx.panel.itertuples():           # UniProt cross-reference as extra safety
        for tid in re.split(r"[;\s]+", r.chembl_targets or ""):
            if tid.startswith("CHEMBL") and tid not in found:
                found[tid] = {"tid": tid, "ttype": "SINGLE PROTEIN", "pref": "", "organism": "Homo sapiens", "accs": [r.uniprot]}
    keep_types = {"SINGLE PROTEIN"} if not ctx.cfg["CHEMBL_INCLUDE_COMPLEXES"] else {"SINGLE PROTEIN", "PROTEIN COMPLEX", "PROTEIN FAMILY", "PROTEIN-PROTEIN INTERACTION", "CHIMERIC PROTEIN"}
    targets = [t for t in found.values() if t["ttype"] in keep_types and (not t["organism"] or "sapiens" in t["organism"].lower())]
    targets.sort(key=lambda t: (t["accs"][0] if t["accs"] else "", t["tid"]))
    atomic_write_text(cache_file, json.dumps({"key": key, "targets": targets}))
    return targets


# =====================================================================================
#  SOURCE 1  ChEMBL
# =====================================================================================
def collect_chembl(ctx):
    src = "chembl"
    cfg, man = ctx.cfg, ctx.man
    targets = chembl_resolve_targets(ctx)
    if cfg["CHEMBL_MAX_TARGETS"]:
        targets = targets[:cfg["CHEMBL_MAX_TARGETS"]]
    log(f"{len(targets)} ChEMBL targets to download (all activity types, server-side paginated)", src)
    parts = ctx.work / "chembl_parts"
    parts.mkdir(exist_ok=True)
    tinfo = {}
    for t in targets:
        accs = t["accs"]
        tinfo[t["tid"]] = {"acc": ";".join(accs), "gene": "", "pref": t["pref"], "organism": t["organism"], "ttype": t["ttype"]}
    ctx.cache.kv_put("chembl_target", {k: json.dumps(v) for k, v in tinfo.items()})

    t_start = time.time()
    done_rows = [0]

    def job(t):
        tid = t["tid"]
        st = man.get(src, "targets", tid, default={})
        if st.get("status") == "done":
            return st.get("n", 0)
        part = parts / f"{tid}.csv"
        offset, n = st.get("offset", 0), st.get("n", 0)
        if offset > 0 and part.exists():
            with open(part, "r+b") as f:
                f.truncate(st.get("bytes", 0))
        else:
            offset, n = 0, 0
            part.unlink(missing_ok=True)
        params = {"target_chembl_id": tid}
        if cfg["CHEMBL_ACTIVITY_TYPES"]:
            params["standard_type__in"] = ",".join(cfg["CHEMBL_ACTIVITY_TYPES"])
        for items, nxt in chembl_pages(ctx, "activity", params, offset, cfg["CHEMBL_PAGE_SIZE"], src):
            df = chembl_df(ctx, items, tinfo, "CHEMBL")
            n += append_df(part, df)
            man.set(src, "targets", tid, {"status": "partial", "offset": nxt, "n": n, "bytes": part.stat().st_size if part.exists() else 0})
            done_rows[0] += len(df)
        touch_header(part)
        man.set(src, "targets", tid, {"status": "done", "n": n, "gene": ";".join(ctx.by_acc[a].gene for a in t["accs"] if a in ctx.by_acc)}, force=True)
        return n

    results = {}
    with ThreadPoolExecutor(cfg["THREADS"]) as ex:
        futs = {ex.submit(job, t): t for t in targets}
        k = 0
        for f in as_completed(futs):
            t = futs[f]
            k += 1
            try:
                results[t["tid"]] = f.result()
            except Exception as e:
                log(f"target {t['tid']} failed: {e} (will retry on next run)", src)
            if k % 10 == 0 or k == len(targets):
                el = time.time() - t_start
                log(f"targets {k}/{len(targets)} | rows so far {sum(results.values()):,} | {sum(results.values()) / max(el, 1):.0f} rows/s", src)
    failed = [t["tid"] for t in targets if man.get(src, "targets", t["tid"], "status") != "done"]
    if failed:
        raise RuntimeError(f"{len(failed)} targets incomplete (e.g. {failed[:3]}); re-run to resume")
    # enrichment: confidence score, cell type, PMID, DOI, year
    assay_ids, doc_ids = set(), set()
    for p in parts.glob("*.csv"):
        for ch in pd.read_csv(p, usecols=["assay_id", "doc_id"], dtype=str, keep_default_na=False, chunksize=500_000):
            assay_ids.update(ch["assay_id"].unique())
            doc_ids.update(ch["doc_id"].unique())
    chembl_fetch_enrichment(ctx, sorted(a for a in assay_ids if a), sorted(d for d in doc_ids if d), src)
    final = ctx.csv_dir / "CHEMBL.csv"
    tmpdir = ctx.work / "chembl_enriched"
    shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir()
    total = 0
    for p in sorted(parts.glob("*.csv")):
        outp = tmpdir / p.name
        wrote = False
        for ch in pd.read_csv(p, dtype=str, keep_default_na=False, chunksize=200_000):
            ch = chembl_enrich_part(ctx, ch)
            ch.to_csv(outp, mode="a", header=not wrote, index=False)
            wrote = True
            total += len(ch)
        if not wrote:
            touch_header(outp)
    concat_parts(list(tmpdir.glob("*.csv")), final)
    shutil.rmtree(tmpdir, ignore_errors=True)
    return {"rows": total}


def collect_chembl_docs(ctx, source, label, pmids=(), title_terms=(), assay_terms=(), known_docs=(), title_must=None):
    """PKIS / PKIS2 / Metz: published datasets deposited in ChEMBL, found through document metadata."""
    src = source.lower()
    docs = {}
    for d in known_docs:
        docs[d] = "known_from_previous_run"
    for pm in pmids:
        j = ctx.http.get_json(f"{URLS['chembl']}/document.json", params={"pubmed_id": pm, "limit": 20}, key="chembl")
        for d in (j or {}).get("documents", []):
            if title_must is None or re.search(title_must, d.get("title") or "", re.I):
                docs[d["document_chembl_id"]] = f"PMID {pm}: {d.get('title')}"
            else:
                log(f"PMID {pm} -> {d['document_chembl_id']} title '{d.get('title')}' does not match expectation; ignored", src)
    for term in title_terms:
        j = ctx.http.get_json(f"{URLS['chembl']}/document.json", params={"title__icontains": term, "limit": 50}, key="chembl")
        for d in (j or {}).get("documents", []):
            docs.setdefault(d["document_chembl_id"], f"title match: {d.get('title')}")
    for term in assay_terms:
        for items, _ in chembl_pages(ctx, "assay", {"description__icontains": term}, 0, 1000, src):
            for a in items:
                docs.setdefault(a.get("document_chembl_id"), f"assay description contains '{term}'")
    docs.pop(None, None)
    log(f"{label}: ChEMBL documents selected: " + (", ".join(f"{k} ({v[:70]})" for k, v in docs.items()) or "NONE"), src)
    man = ctx.man
    parts = ctx.work / f"{src}_parts"
    parts.mkdir(exist_ok=True)
    total = 0
    for doc in docs:
        st = man.get(src, "docs", doc, default={})
        part = parts / f"{doc}.csv"
        if st.get("status") == "done":
            total += st.get("n", 0)
            continue
        offset, n = st.get("offset", 0), st.get("n", 0)
        if offset > 0 and part.exists():
            with open(part, "r+b") as f:
                f.truncate(st.get("bytes", 0))
        else:
            offset, n = 0, 0
            part.unlink(missing_ok=True)
        for items, nxt in chembl_pages(ctx, "activity", {"document_chembl_id": doc}, offset, ctx.cfg["CHEMBL_PAGE_SIZE"], src):
            tinfo = chembl_target_info(ctx, [i.get("target_chembl_id") for i in items])
            df = chembl_df(ctx, items, tinfo, source.upper())
            df["measure_semantics"] = f"{label}_as_deposited_in_ChEMBL"
            df["notes"] = df["notes"] + np.where(df["notes"] != "", ";", "") + f"selected_via={docs[doc][:60]}"
            n += append_df(part, df)
            man.set(src, "docs", doc, {"status": "partial", "offset": nxt, "n": n, "bytes": part.stat().st_size})
        touch_header(part)
        man.set(src, "docs", doc, {"status": "done", "n": n}, force=True)
        total += n
    plist = list(parts.glob("*.csv"))
    if plist:
        aid = set()
        did = set()
        for p in plist:
            for ch in pd.read_csv(p, usecols=["assay_id", "doc_id"], dtype=str, keep_default_na=False, chunksize=500_000):
                aid.update(ch["assay_id"].unique())
                did.update(ch["doc_id"].unique())
        chembl_fetch_enrichment(ctx, sorted(a for a in aid if a), sorted(d for d in did if d), src)
        outs = []
        tmpdir = ctx.work / f"{src}_enriched"
        shutil.rmtree(tmpdir, ignore_errors=True)
        tmpdir.mkdir()
        for p in plist:
            wrote = False
            for ch in pd.read_csv(p, dtype=str, keep_default_na=False, chunksize=200_000):
                chembl_enrich_part(ctx, ch).to_csv(tmpdir / p.name, mode="a", header=not wrote, index=False)
                wrote = True
            if not wrote:
                touch_header(tmpdir / p.name)
            outs.append(tmpdir / p.name)
        concat_parts(outs, ctx.csv_dir / f"{source.upper()}.csv")
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        touch_header(ctx.csv_dir / f"{source.upper()}.csv")
    return {"rows": total}


def collect_pkis(ctx):
    r = collect_chembl_docs(ctx, "PKIS", "PKIS1_Elkins2016", pmids=["26501955"],
                            title_terms=["Published Kinase Inhibitor Set"], known_docs=["CHEMBL1961873"],
                            title_must=r"kinase inhibitor set")
    if r["rows"] == 0 or True:
        r["rows"] += manual_wide(ctx, "pkis", "PKIS", "PCT_INHIBITION", "%", "pkis1_percent_inhibition_manual", append=True)
    return r


def collect_pkis2(ctx):
    r = collect_chembl_docs(ctx, "PKIS2", "PKIS2_Drewry2017", pmids=["28767711"], title_terms=["PKIS2"],
                            assay_terms=["PKIS2"], title_must=r"chemogenomic|PKIS")
    r["rows"] += manual_wide(ctx, "pkis2", "PKIS2", "PCT_INHIBITION", "%", "pkis2_percent_inhibition_manual", append=True)
    return r


def collect_metz(ctx):
    r = collect_chembl_docs(ctx, "METZ", "Metz2011_Ki", pmids=["21336281"], title_terms=["Navigating the kinome"],
                            title_must=r"kinome")
    r["rows"] += manual_wide(ctx, "metz", "METZ", "Ki", "nM", "metz_ki_manual", append=True)
    return r


# =====================================================================================
#  generic manual 'wide matrix' loader (supplementary tables you download by hand)
# =====================================================================================
def manual_wide(ctx, folder, source, atype, units, semantics, append=False):
    mdir = ctx.manual / folder
    files = [p for p in mdir.glob("*") if p.suffix.lower() in (".csv", ".tsv", ".txt", ".xlsx", ".xls")] if mdir.exists() else []
    if not files:
        return 0
    total = 0
    out = ctx.csv_dir / f"{source}.csv"
    for p in files:
        try:
            if p.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(p, dtype=object)
            else:
                df = pd.read_csv(p, sep=None, engine="python", dtype=object)
        except Exception as e:
            log(f"manual file {p.name} unreadable: {e}", folder)
            continue
        sm = pick(df, "smiles", "canonical_smiles", regex=r"smiles")
        cid = pick(df, "compound", "compound_id", "id", "name", "inhibitor", "molecule", regex=r"compound|molecule|id$")
        kin = {}
        for c in df.columns:
            if c in (sm, cid):
                continue
            row, status = map_symbol(ctx, str(c))
            if row is not None:
                kin[c] = (row, status)
        if not kin:
            log(f"manual file {p.name}: no column header matched a panel kinase symbol - skipped", folder)
            continue
        conc = re.search(r"(\d+(?:\.\d+)?)\s*(um|µm|nm)", p.name, re.I)
        parts = []
        for c, (row, status) in kin.items():
            rel, val = split_rel_value(df[c])
            ok = val.notna()
            if not ok.any():
                continue
            parts.append(pd.DataFrame({
                "source_record_id": f"{p.name}:{c}:" + df.index[ok].astype(str),
                "target_name_original": c, "gene_symbol": row.gene, "target_name_normalized": row.gene.upper(),
                "uniprot_accession": row.uniprot, "target_id_source": "manual_file", "target_organism": "Homo sapiens",
                "target_type": "SINGLE PROTEIN", "target_mapping_status": status,
                "compound_id": df.loc[ok, cid].astype(str) if cid else "", "smiles_raw": df.loc[ok, sm].astype(str) if sm else "",
                "activity_type_original": atype, "relation": rel[ok], "value_original": val[ok], "units_original": units,
                "measure_semantics": semantics, "assay_description": f"manual file {p.name}" + (f"; concentration {conc.group(0)}" if conc else ""),
                "notes": "manual_supplementary_table",
            }))
        if parts:
            d = derive(pd.concat(parts), source.upper())
            total += append_df(out, d)
            log(f"manual file {p.name}: {len(d):,} records for {len(kin)} kinases", folder)
    return total


# =====================================================================================
#  SOURCE 8/9  Davis + KIBA  (DeepDTA GitHub copies of the published matrices)
# =====================================================================================
class _NpUnpickler(pickle.Unpickler):
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
               ("numpy._core.multiarray", "scalar"), ("_codecs", "encode"),
               ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked pickle global {module}.{name}")


def collect_davis(ctx):
    src = "davis"
    d = ctx.raw / "davis"
    base = URLS["deepdta"] + "/davis"
    files = {}
    for f in ("Y", "ligands_can.txt", "proteins.txt"):
        files[f] = ctx.http.download([f"{base}/{f}"], d / f, key="github", min_bytes=100, src=src)
    if not all(files.values()):
        n = manual_wide(ctx, "davis", "DAVIS", "Kd", "nM", "davis_kd_matrix_manual")
        if n:
            return {"rows": n}
        raise RuntimeError("Davis files not downloadable; put a wide Kd matrix (SMILES + kinase columns) in manual/davis/")
    lig = json.load(open(files["ligands_can.txt"]))
    prot = json.load(open(files["proteins.txt"]))
    with open(files["Y"], "rb") as fh:
        Y = np.array(_NpUnpickler(fh, encoding="latin1").load(), dtype=float)
    cids, names = list(lig.keys()), list(prot.keys())
    if Y.shape != (len(cids), len(names)):
        raise RuntimeError(f"Davis matrix shape {Y.shape} != ({len(cids)}, {len(names)})")
    ri, ci = np.where(np.isfinite(Y))
    rows = []
    nm = np.array(names, dtype=object)
    # target mapping per protein name (once)
    tmap = {}
    for name in names:
        up = name.upper()
        nonhuman = next((v for k, v in NONHUMAN_HINTS.items() if k in up), "")
        row, status = (None, "nonhuman_target") if nonhuman else map_symbol(ctx, name)
        variant = ""
        m = re.match(r"^([^()]+?)(?:\((.*)\))?(p)?$", name)
        if m and (m.group(2) or m.group(3)) and name.upper() not in ALIASES:
            variant = (m.group(2) or "") + ("; phosphorylated" if m.group(3) else "")
        tmap[name] = (row, status, nonhuman or "Homo sapiens", variant)
    df = pd.DataFrame({
        "target_name_original": nm[ci],
        "compound_id": ["CID" + cids[i] for i in ri], "pubchem_cid": [cids[i] for i in ri],
        "smiles_raw": [lig[cids[i]] for i in ri], "value_original": Y[ri, ci],
    })
    df["source_record_id"] = ["davis:" + cids[i] + ":" + names[j] for i, j in zip(ri, ci)]
    df["gene_symbol"] = [tmap[n][0].gene if tmap[n][0] is not None else "" for n in df.target_name_original]
    df["target_name_normalized"] = np.where(df.gene_symbol != "", df.gene_symbol.str.upper(), df.target_name_original.str.upper())
    df["uniprot_accession"] = [tmap[n][0].uniprot if tmap[n][0] is not None else "" for n in df.target_name_original]
    df["target_mapping_status"] = [tmap[n][1] for n in df.target_name_original]
    df["target_organism"] = [tmap[n][2] for n in df.target_name_original]
    df["variant_info"] = [tmap[n][3] for n in df.target_name_original]
    df["target_type"] = "SINGLE PROTEIN"
    df["target_id_source"] = "Davis2011_kinase_name"
    df["activity_type_original"] = "Kd"
    df["units_original"] = "nM"
    cens = np.isclose(df.value_original.values, 10000.0)
    df["relation"] = np.where(cens, ">", "=")
    df["measure_semantics"] = "kd_nM_matrix_Davis2011"
    df["notes"] = np.where(cens, "Davis value 10000 nM is the assay ceiling (reported Kd >10 uM, censored)", "")
    df["assay_description"] = "Davis et al. 2011 Nat Biotechnol 29:1046; competition binding (KINOMEscan); DeepDTA repository copy"
    df["pmid"] = "22037378"
    df["record_class"] = ""
    out = derive(df, "DAVIS", "DeepDTA_github")
    path = ctx.csv_dir / "DAVIS.csv"
    path.unlink(missing_ok=True)
    append_df(path, out)
    return {"rows": len(out), "note": f"{int(cens.sum())} censored (>10000 nM) rows flagged"}


def collect_kiba(ctx):
    src = "kiba"
    d = ctx.raw / "kiba"
    base = URLS["deepdta"] + "/kiba"
    files = {f: ctx.http.download([f"{base}/{f}"], d / f, key="github", min_bytes=100, src=src)
             for f in ("kiba_binding_affinity_v2.txt", "ligands_can.txt", "proteins.txt")}
    if not all(files.values()):
        raise RuntimeError("KIBA files not downloadable (GitHub unreachable?)")
    lig = json.load(open(files["ligands_can.txt"]))
    prot = json.load(open(files["proteins.txt"]))
    Y = pd.read_csv(files["kiba_binding_affinity_v2.txt"], sep="\t", header=None).values.astype(float)
    ids, accs = list(lig.keys()), list(prot.keys())
    if Y.shape != (len(ids), len(accs)):
        raise RuntimeError(f"KIBA matrix shape {Y.shape} != ({len(ids)}, {len(accs)})")
    ri, ci = np.where(np.isfinite(Y))          # observed pairs only
    acc_arr = np.array(accs, dtype=object)
    df = pd.DataFrame({
        "uniprot_accession": acc_arr[ci], "compound_id": [ids[i] for i in ri],
        "chembl_molecule_id": [ids[i] if ids[i].startswith("CHEMBL") else "" for i in ri],
        "smiles_raw": [lig[ids[i]] for i in ri], "value_original": Y[ri, ci],
    })
    df["source_record_id"] = ["kiba:" + ids[i] + ":" + accs[j] for i, j in zip(ri, ci)]
    row_of = {a: ctx.by_acc.get(a) for a in accs}
    df["gene_symbol"] = [row_of[a].gene if row_of[a] is not None else "" for a in df.uniprot_accession]
    df["target_name_original"] = df.uniprot_accession
    df["target_name_normalized"] = np.where(df.gene_symbol != "", df.gene_symbol.str.upper(), df.uniprot_accession)
    df["target_mapping_status"] = np.where(df.gene_symbol != "", "mapped_uniprot_exact", "unmapped_not_in_panel")
    df["target_organism"] = "Homo sapiens"
    df["target_type"] = "SINGLE PROTEIN"
    df["target_id_source"] = "UniProt"
    df["activity_type_original"] = "KIBA_score"
    df["activity_type"] = "KIBA_score"
    df["units_original"] = "KIBA score"
    df["relation"] = "="
    df["record_class"] = "matrix_score"
    df["measure_semantics"] = "kiba_score_NOT_IC50_Ki_Kd(Tang2014; model-integrated; do not convert)"
    df["assay_description"] = "KIBA integrated bioactivity score, Tang et al. 2014 J Chem Inf Model; v2 matrix as distributed in DeepDTA repo; verify direction (lower = stronger in the KIBA definition) before using as a label"
    df["notes"] = "observed pairs only; unobserved pairs not generated"
    out = derive(df, "KIBA", "DeepDTA_github_v2")
    path = ctx.csv_dir / "KIBA.csv"
    path.unlink(missing_ok=True)
    append_df(path, out)
    return {"rows": len(out)}


# =====================================================================================
#  SOURCE 3  BindingDB
# =====================================================================================
def bdb_candidate_urls(ctx):
    vers = []
    r = ctx.http.get(URLS["bdb_page"], key="bindingdb", tries=3)
    if r is not None and r.status_code == 200:
        vers = sorted(set(re.findall(r"BindingDB_All_(\d{6})_tsv\.zip", r.text)), reverse=True)
    if not vers:
        y, m = datetime.now().year, datetime.now().month
        for _ in range(8):
            vers.append(f"{y}{m:02d}")
            m -= 1
            if m == 0:
                y, m = y - 1, 12
        log("could not read BindingDB download page; guessing recent release names", "bindingdb")
    urls = []
    for v in vers[:6]:
        path = f"/rwd/bind/downloads/BindingDB_All_{v}_tsv.zip"
        for h in URLS["bdb_hosts"]:
            urls.append(f"{h}/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file={quote(path, safe='')}")
            urls.append(f"{h}{path}")
    return urls, (vers[0] if vers else "")


def bindingdb_parse(ctx, path: Path, release: str):
    src = "bindingdb"
    opener = None
    if str(path).lower().endswith(".zip"):
        zf = zipfile.ZipFile(path)
        member = next(n for n in zf.namelist() if n.lower().endswith(".tsv"))
        fh = zf.open(member)
    else:
        fh = open(path, "rb")
    out = ctx.csv_dir / "BINDINGDB.csv"
    out.unlink(missing_ok=True)
    acc_set = set(ctx.by_acc)
    total_rows = kept = 0
    t0 = time.time()
    reader = pd.read_csv(fh, sep="\t", dtype=str, chunksize=200_000, quoting=csv.QUOTE_NONE,
                         on_bad_lines="skip", encoding="utf-8", encoding_errors="replace", low_memory=False)
    for ch in reader:
        total_rows += len(ch)
        ucols = [c for c in ch.columns if re.search(r"UniProt \(SwissProt\) Primary ID of Target Chain", c, re.I)]
        if not ucols:
            raise RuntimeError("BindingDB TSV: UniProt columns not found - file format changed?")
        U = ch[ucols].fillna("")
        mask = U.isin(acc_set).any(axis=1)
        org = pick(ch, "Target Source Organism According to Curator or DataSource", regex=r"Target Source Organism")
        if org:
            mask &= ch[org].fillna("").str.contains("Homo sapiens", case=False)
        sub = ch[mask]
        U = U[mask]
        if sub.empty:
            continue
        chains = U.apply(lambda r: [x for x in r.tolist() if x], axis=1)
        distinct = chains.map(lambda l: sorted(set(l)))
        multi = distinct.map(len) > 1
        first_hit = chains.map(lambda l: next((x for x in l if x in acc_set), ""))
        gene = first_hit.map(lambda a: ctx.by_acc[a].gene if a in ctx.by_acc else "")
        acc_str = distinct.map(";".join)
        c_smi = pick(sub, "Ligand SMILES")
        c_ik = pick(sub, "Ligand InChI Key", regex=r"InChI Key")
        c_mono = pick(sub, "BindingDB MonomerID")
        c_name = pick(sub, "BindingDB Ligand Name")
        c_rsid = pick(sub, "BindingDB Reactant_set_id", regex=r"Reactant_set_id")
        c_tname = pick(sub, "Target Name")
        c_cid = pick(sub, "PubChem CID")
        c_chembl = pick(sub, "ChEMBL ID of Ligand", regex=r"ChEMBL ID of Ligand")
        c_aid = pick(sub, "PubChem AID")
        c_pmid = pick(sub, "PMID")
        c_doi = pick(sub, "Article DOI")
        c_pat = pick(sub, "Patent Number")
        c_cur = pick(sub, "Curation/DataSource")
        c_ph, c_t = pick(sub, "pH"), pick(sub, "Temp (C)")
        desc = ("pH=" + col_or_blank(sub, c_ph) + ";T(C)=" + col_or_blank(sub, c_t) + ";curation=" + col_or_blank(sub, c_cur))
        frames = []
        for typ in ("Ki", "IC50", "Kd", "EC50"):
            vc = pick(sub, f"{typ} (nM)", regex=rf"^{typ} \(nM\)")
            if not vc:
                continue
            raw = sub[vc].fillna("")
            ok = raw.str.strip() != ""
            if not ok.any():
                continue
            rel, val = split_rel_value(raw[ok])
            okv = val.notna()
            idx = rel.index[okv]
            f = pd.DataFrame({
                "source_record_id": col_or_blank(sub, c_rsid).loc[idx] + ":" + typ,
                "target_name_original": col_or_blank(sub, c_tname).loc[idx],
                "gene_symbol": gene.loc[idx], "target_name_normalized": gene.loc[idx].str.upper(),
                "uniprot_accession": acc_str.loc[idx], "target_id_source": "UniProt(SwissProt) via BindingDB",
                "target_organism": "Homo sapiens", "target_type": np.where(multi.loc[idx], "PROTEIN COMPLEX", "SINGLE PROTEIN"),
                "target_mapping_status": np.where(multi.loc[idx], "complex_multi_chain", "mapped_uniprot_exact"),
                "compound_id": col_or_blank(sub, c_mono).loc[idx], "compound_name": col_or_blank(sub, c_name).loc[idx],
                "smiles_raw": col_or_blank(sub, c_smi).loc[idx], "inchikey_source": col_or_blank(sub, c_ik).loc[idx],
                "pubchem_cid": col_or_blank(sub, c_cid).loc[idx], "chembl_molecule_id": col_or_blank(sub, c_chembl).loc[idx],
                "activity_type_original": typ, "relation": rel.loc[idx], "value_original": val.loc[idx], "units_original": "nM",
                "measure_semantics": "bindingdb_measured_affinity_nM",
                "assay_id": col_or_blank(sub, c_aid).loc[idx], "assay_description": desc.loc[idx],
                "doc_id": col_or_blank(sub, c_pat).loc[idx], "pmid": col_or_blank(sub, c_pmid).loc[idx],
                "doi": col_or_blank(sub, c_doi).loc[idx],
            })
            frames.append(f)
        if frames:
            d = derive(pd.concat(frames), "BINDINGDB", release)
            kept += append_df(out, d)
        el = time.time() - t0
        log(f"scanned {total_rows:,} BindingDB rows | kinase records kept {kept:,} | {total_rows / max(el, 1):,.0f} rows/s", src)
    fh.close()
    if kept == 0:
        touch_header(out)
    return kept


def bindingdb_api(ctx):
    """Fallback: documented REST service getLigandsByUniprots (JSON)."""
    src = "bindingdb"
    out = ctx.csv_dir / "BINDINGDB.csv"
    out.unlink(missing_ok=True)
    kept = 0
    for r in ctx.panel.itertuples():
        resp = ctx.http.get(URLS["bdb_api"], params={"uniprot": r.uniprot, "code": 0, "response": "application/json"}, key="bindingdb", tries=3, timeout=(20, 120))
        if resp is None or resp.status_code != 200:
            continue
        try:
            j = resp.json()
        except Exception:
            continue
        recs = []

        def walk(o):
            if isinstance(o, dict):
                if "smile" in o or "smiles" in o:
                    recs.append(o)
                else:
                    for v in o.values():
                        walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(j)
        if not recs:
            continue
        df = pd.DataFrame(recs)
        rel, val = split_rel_value(df.get("affinity", pd.Series("", index=df.index)))
        d = pd.DataFrame({
            "source_record_id": "bdbapi:" + df.get("monomerid", pd.Series("", index=df.index)).astype(str) + ":" + r.uniprot,
            "target_name_original": r.gene, "gene_symbol": r.gene, "target_name_normalized": r.gene.upper(),
            "uniprot_accession": r.uniprot, "target_organism": "Homo sapiens", "target_type": "SINGLE PROTEIN",
            "target_mapping_status": "mapped_uniprot_exact", "target_id_source": "UniProt",
            "compound_id": df.get("monomerid", "").astype(str), "smiles_raw": df.get("smile", df.get("smiles", "")),
            "activity_type_original": df.get("affinity_type", ""), "relation": rel, "value_original": val, "units_original": "nM",
            "pmid": df.get("pmid", ""), "doi": df.get("doi", ""), "measure_semantics": "bindingdb_api_affinity_nM",
        })
        kept += append_df(out, derive(d[val.notna().values], "BINDINGDB", "api"))
    if kept == 0:
        touch_header(out)
    return kept


def collect_bindingdb(ctx):
    src = "bindingdb"
    manual = None
    mdir = ctx.manual / "bindingdb"
    if mdir.exists():
        manual = next(iter(sorted(mdir.glob("BindingDB_All*"))), None)
    path, release = manual, "manual"
    if path is None:
        urls, release = bdb_candidate_urls(ctx)
        log(f"downloading BindingDB bulk TSV (release {release}); ~570 MB zip, resumable", src)
        path = ctx.http.download(urls, ctx.raw / "bindingdb" / f"BindingDB_All_{release}_tsv.zip", key="bindingdb",
                                 min_bytes=50_000_000, magic=b"PK", src=src)
    if path:
        n = bindingdb_parse(ctx, Path(path), release)
        return {"rows": n, "note": f"bulk file release {release}"}
    if ctx.cfg["BINDINGDB_API_FALLBACK"]:
        log("bulk download failed -> trying documented web service per UniProt accession", src)
        n = bindingdb_api(ctx)
        if n:
            return {"rows": n, "note": "API fallback"}
    raise RuntimeError("BindingDB bulk file not obtainable. Manual: download BindingDB_All_*_tsv.zip from "
                       "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp and put it in "
                       f"{mdir}/ then re-run")


# =====================================================================================
#  SOURCE 4  Drug Target Commons
# =====================================================================================
def dtc_parse(ctx, path: Path, release):
    out = ctx.csv_dir / "DTC.csv"
    out.unlink(missing_ok=True)
    kept = scanned = 0
    for ch in pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=200_000, on_bad_lines="skip", encoding_errors="replace", low_memory=False):
        scanned += len(ch)
        ch.columns = [c.strip().lower() for c in ch.columns]
        tid = pick(ch, "target_id", "uniprot", regex=r"^target_id|uniprot")
        gn = pick(ch, "gene_names", "gene_name", regex=r"gene")
        if tid is None:
            raise RuntimeError(f"DTC file columns not recognised: {list(ch.columns)[:15]}")
        acc = ch[tid].str.strip()
        mask = acc.isin(ctx.by_acc)
        if gn:   # also accept rows where UniProt blank but the gene symbol is an exact panel symbol
            first_gene = ch[gn].str.split(r"[;,| ]", regex=True).str[0].str.upper()
            mask |= (acc == "") & first_gene.isin(ctx.by_gene)
        sub = ch[mask]
        if sub.empty:
            continue
        acc2 = sub[tid].str.strip()
        gene = acc2.map(lambda a: ctx.by_acc[a].gene if a in ctx.by_acc else "")
        if gn:
            fg = sub[gn].str.split(r"[;,| ]", regex=True).str[0].str.upper()
            gene = np.where(gene == "", fg.map(lambda g: ctx.by_gene[g].gene if g in ctx.by_gene else ""), gene)
            gene = pd.Series(gene, index=sub.index)
        acc_final = np.where(acc2 == "", gene.map(lambda g: ctx.by_gene[g.upper()].uniprot if g and g.upper() in ctx.by_gene else ""), acc2)
        rel_c = pick(sub, "standard_relation")
        val_c, unit_c, typ_c = pick(sub, "standard_value"), pick(sub, "standard_units"), pick(sub, "standard_type")
        rel_raw = col_or_blank(sub, rel_c).str.strip().replace({"": "="})
        val = pd.to_numeric(col_or_blank(sub, val_c), errors="coerce")
        cid = col_or_blank(sub, pick(sub, "compound_id"))
        wt = col_or_blank(sub, pick(sub, "wild_type_or_mutant"))
        mut = col_or_blank(sub, pick(sub, "mutation_info"))
        d = pd.DataFrame({
            "source_record_id": "dtc:" + sub.index.astype(str) + ":" + col_or_blank(sub, tid).astype(str),
            "target_name_original": col_or_blank(sub, pick(sub, "target_pref_name")),
            "gene_symbol": gene.values, "target_name_normalized": pd.Series(gene.values, index=sub.index).str.upper().values,
            "uniprot_accession": acc_final, "target_id_source": "UniProt (DTC)",
            "target_organism": "Homo sapiens", "target_type": "SINGLE PROTEIN",
            "target_mapping_status": np.where(acc2 != "", "mapped_uniprot_exact", "mapped_gene_symbol"),
            "variant_info": np.where(wt.str.lower().str.contains("mutant"), mut, ""),
            "compound_id": cid, "compound_name": col_or_blank(sub, pick(sub, "compound_name")),
            "inchikey_source": col_or_blank(sub, pick(sub, "standard_inchi_key", regex=r"inchi_?key")),
            "chembl_molecule_id": np.where(cid.str.startswith("CHEMBL"), cid, ""),
            "activity_type_original": col_or_blank(sub, typ_c), "relation": rel_raw, "value_original": val,
            "units_original": col_or_blank(sub, unit_c),
            "measure_semantics": "dtc_curated_bioactivity",
            "assay_description": col_or_blank(sub, pick(sub, "assay_description")),
            "assay_type": col_or_blank(sub, pick(sub, "assay_type")),
            "assay_cell_type": col_or_blank(sub, pick(sub, "assay_cell_line")),
            "pmid": col_or_blank(sub, pick(sub, "pubmed_id")), "doi": col_or_blank(sub, pick(sub, "doi")),
            "activity_comment": col_or_blank(sub, pick(sub, "activity_comment")),
            "notes": "DTC rows carry InChIKey/ChEMBL id but no SMILES; structures are resolved in post-processing",
        })
        d = d[(val.notna() | (d["activity_comment"] != "")).values]
        d = derive(d, "DTC", release)
        kept += append_df(out, d)
        log(f"scanned {scanned:,} DTC rows | kinase records kept {kept:,}", "dtc")
    if kept == 0:
        touch_header(out)
    return kept


def collect_dtc(ctx):
    src = "dtc"
    mdir = ctx.manual / "dtc"
    path = next(iter(sorted(mdir.glob("*.csv"))), None) if mdir.exists() else None
    release = "manual"
    if path is None:
        path = ctx.http.download(URLS["dtc_static"], ctx.raw / "dtc" / "DTC_data.csv", key="dtc", min_bytes=1_000_000, src=src)
        release = "static_download"
    if path is None:      # documented API discovery (never guess endpoints)
        j = ctx.http.get_json(URLS["dtc_api_root"], key="dtc", tries=2, headers={"Accept": "application/json"})
        eps = []
        if isinstance(j, dict):
            for k, v in j.items():
                if "bioactiv" in k.lower() or (isinstance(v, str) and "bioactiv" in v.lower()):
                    eps.append(v if isinstance(v, str) else k)
        if eps:
            log(f"DTC API advertises: {eps}; paginating", src)
            out_rows = []
            url = eps[0]
            seen = set()
            while url and url not in seen and len(seen) < 200000:
                seen.add(url)
                jj = ctx.http.get_json(url, key="dtc", tries=4)
                if not jj:
                    break
                res = jj.get("results") or jj.get("objects") or []
                out_rows.extend(res)
                url = jj.get("next")
                if len(seen) % 20 == 0:
                    log(f"DTC API pages {len(seen)} rows {len(out_rows):,}", src)
            if out_rows:
                tmp = ctx.raw / "dtc" / "dtc_api.csv"
                tmp.parent.mkdir(exist_ok=True)
                pd.DataFrame(out_rows).to_csv(tmp, index=False)
                path, release = tmp, "api"
    if path is None:
        raise RuntimeError("DTC: no public bulk file/API reachable. Manual: open https://drugtargetcommons.fimm.fi/ -> "
                           f"Download, save the CSV into {mdir}/ and re-run")
    n = dtc_parse(ctx, Path(path), release)
    return {"rows": n, "note": release}


# =====================================================================================
#  SOURCE 5  IUPHAR / BPS Guide to PHARMACOLOGY
# =====================================================================================
def _gtop_read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
    skip = 1 if first.startswith("#") or first.startswith('"#') else 0
    df = pd.read_csv(path, skiprows=skip, dtype=str, keep_default_na=False, encoding_errors="replace", on_bad_lines="skip")
    df.columns = [re.sub(r"[^a-z0-9]", "", c.lower()) for c in df.columns]
    return df


def collect_iuphar(ctx):
    src = "iuphar"
    mdir = ctx.manual / "iuphar"
    ipath = mdir / "interactions.csv" if (mdir / "interactions.csv").exists() else None
    lpath = mdir / "ligands.csv" if (mdir / "ligands.csv").exists() else None
    if ipath is None:
        ipath = ctx.http.download([f"{URLS['gtop_data']}/interactions.csv"], ctx.raw / "iuphar" / "interactions.csv", key="iuphar", min_bytes=10_000, src=src)
    if lpath is None:
        lpath = ctx.http.download([f"{URLS['gtop_data']}/ligands.csv"], ctx.raw / "iuphar" / "ligands.csv", key="iuphar", min_bytes=10_000, src=src)
    out = ctx.csv_dir / "IUPHAR.csv"
    out.unlink(missing_ok=True)
    if ipath is not None:
        I = _gtop_read(ipath)
        L = _gtop_read(lpath) if lpath else pd.DataFrame()
        up = next((c for c in I.columns if c in ("targetuniprotid", "uniprotid")), None) or next((c for c in I.columns if "uniprot" in c and "target" in c), None)
        if up is None:
            raise RuntimeError(f"IUPHAR interactions.csv: UniProt column not found in {list(I.columns)[:20]}")
        sp = next((c for c in I.columns if c == "targetspecies"), None)
        accs = I[up].str.split("|")
        hit = accs.map(lambda l: next((a.strip() for a in l if a.strip() in ctx.by_acc), ""))
        mask = hit != ""
        if sp:
            mask &= I[sp].str.lower().str.contains("human")
        I = I[mask].copy()
        hit = hit[mask]
        smi_map = {}
        if not L.empty:
            lid = next((c for c in L.columns if c == "ligandid"), None)
            ls = next((c for c in L.columns if c == "smiles"), None)
            lik = next((c for c in L.columns if c == "inchikey"), None)
            lcid = next((c for c in L.columns if c == "pubchemcid"), None)
            if lid and ls:
                smi_map = {a: (b, c, d) for a, b, c, d in zip(L[lid], L[ls], L[lik] if lik else [""] * len(L), L[lcid] if lcid else [""] * len(L))}
        c = lambda n: I[n] if n in I.columns else pd.Series("", index=I.index)
        aff_units = c("affinityunits")
        med = pd.to_numeric(c("affinitymedian"), errors="coerce")
        lo = pd.to_numeric(c("affinitylow"), errors="coerce")
        hi = pd.to_numeric(c("affinityhigh"), errors="coerce")
        pval = med.fillna((lo + hi) / 2)
        ligid = c("ligandid")
        d = pd.DataFrame({
            "source_record_id": "gtopdb:" + I.index.astype(str) + ":" + ligid,
            "target_name_original": c("target"), "gene_symbol": hit.map(lambda a: ctx.by_acc[a].gene).values,
            "uniprot_accession": hit.values, "target_id_source": "UniProt (GtoPdb)", "target_organism": "Homo sapiens",
            "target_type": "SINGLE PROTEIN", "target_mapping_status": "mapped_uniprot_exact",
            "compound_id": "GtoPdb:" + ligid, "compound_name": c("ligand"),
            "smiles_raw": ligid.map(lambda x: smi_map.get(x, ("", "", ""))[0]),
            "inchikey_source": ligid.map(lambda x: smi_map.get(x, ("", "", ""))[1]),
            "pubchem_cid": ligid.map(lambda x: smi_map.get(x, ("", "", ""))[2]),
            "activity_type_original": aff_units.str.replace(r"^p", "", regex=True), "relation": "=",
            "value_original": np.nan, "units_original": "", "p_activity": pval,
            "measure_semantics": "gtopdb_curated_summary_pAffinity(median of curated experiments; NOT a single measurement)",
            "assay_description": c("assaydescription"), "pmid": c("pubmedid"),
            "notes": "p-scale supplied by source: low=" + lo.astype(str) + ";high=" + hi.astype(str) + ";action=" + c("action") + ";type=" + c("type"),
        })
        d = d[pval.notna().values & aff_units.str.match(r"^p(IC50|Ki|Kd|EC50)", case=False).values]
        # nM columns, when the file has them, are the original measurements
        nm_c = next((x for x in I.columns if x.startswith("originalaffinitymediannm")), None)
        if nm_c:
            nmv = pd.to_numeric(I.loc[d.index if False else I.index, nm_c], errors="coerce").loc[d.index.map(lambda k: k) if False else I.index]
        out_df = derive(d, "IUPHAR", "bulk_download")
        append_df(out, out_df)
        n = len(out_df)
        if n:
            return {"rows": n}
    # ---- API fallback (documented web services)
    log("bulk files not available/usable -> using the documented web services per target", src)
    kept = 0
    for r in ctx.panel.itertuples():
        j = ctx.http.get_json(f"{URLS['gtop_api']}/targets", params={"geneSymbol": r.gene, "type": "Protein"}, key="iuphar", tries=3)
        tid = None
        for t in (j or []):
            if str(t.get("abbreviation", "")).upper() == r.gene.upper() or (t.get("geneSymbol", "").upper() == r.gene.upper()):
                tid = t.get("targetId")
                break
        if tid is None and j:
            tid = j[0].get("targetId")
        if tid is None:
            continue
        ints = ctx.http.get_json(f"{URLS['gtop_api']}/targets/{tid}/interactions", key="iuphar", tries=3) or []
        rows = []
        for it in ints:
            at = str(it.get("affinityType", ""))
            if not re.match(r"^p(IC50|Ki|Kd|EC50)", at, re.I):
                continue
            try:
                lo_, hi_ = str(it.get("affinity", "")).split("-")[0], str(it.get("affinity", "")).split("-")[-1]
                pv = (float(lo_) + float(hi_)) / 2
            except Exception:
                continue
            lg = it.get("ligandId")
            s = ctx.http.get_json(f"{URLS['gtop_api']}/ligands/{lg}/structure", key="iuphar", tries=2) or {}
            rows.append({"source_record_id": f"gtopdb_api:{tid}:{lg}:{at}", "target_name_original": r.gene, "gene_symbol": r.gene,
                         "uniprot_accession": r.uniprot, "target_organism": "Homo sapiens", "target_type": "SINGLE PROTEIN",
                         "target_mapping_status": "mapped_uniprot_exact", "compound_id": f"GtoPdb:{lg}",
                         "compound_name": it.get("ligandName", ""), "smiles_raw": s.get("smiles", ""), "inchikey_source": s.get("inChIKey", ""),
                         "activity_type_original": re.sub(r"^p", "", at), "relation": "=", "p_activity": pv,
                         "measure_semantics": "gtopdb_curated_summary_pAffinity", "pmid": ";".join(str(x.get("pmid", "")) for x in (it.get("refs") or []))})
        if rows:
            kept += append_df(out, derive(pd.DataFrame(rows), "IUPHAR", "api"))
    if kept == 0:
        touch_header(out)
        raise RuntimeError("IUPHAR: bulk files and API both gave nothing. Manual: download interactions.csv and ligands.csv from "
                           f"https://www.guidetopharmacology.org/download.jsp into {mdir}/ and re-run")
    return {"rows": kept, "note": "API fallback"}


# =====================================================================================
#  SOURCE 2  PubChem BioAssay
# =====================================================================================
OUTCOME = {"1": "inactive", "2": "active", "3": "inconclusive", "4": "unspecified", "5": "probe",
           "inactive": "inactive", "active": "active", "inconclusive": "inconclusive", "unspecified": "unspecified"}
PC_TYPES = {"ic50": "IC50", "ec50": "EC50", "ac50": "AC50", "xc50": "XC50", "gi50": "GI50", "ki": "Ki", "kd": "Kd", "potency": "Potency"}
PC_SKIP = {"hill", "slope", "curve", "class", "sd", "stdev", "error", "std", "fit", "rsq", "r2", "efficacy", "max", "min",
           "pvalue", "count", "flag", "comment", "qualifier", "n", "num", "number", "conc", "concentration"}


def pubchem_detect_cols(columns, units_meta):
    found = {}
    for c in columns:
        if c.upper().startswith("PUBCHEM_"):
            continue
        low = c.lower()
        toks = [t for t in re.split(r"[^a-z0-9%]+", low) if t]
        if not toks or any(t in PC_SKIP for t in toks):
            continue
        key = next((t for t in toks if t in PC_TYPES), None)
        islog = False
        if key is None:
            m = next((t for t in toks if re.fullmatch(r"p(ic50|ec50|ac50|ki|kd)", t)), None)
            if m:
                key, islog = m[1:], True
        if key:
            islog = islog or "log" in toks
            u = (units_meta.get(c) or "").strip()
            if not u:
                m = re.search(r"\(\s*([numµμp]?m)\s*\)", c, re.I) or re.search(r"\b(nm|um|µm|μm|mm|pm)\b", low)
                u = m.group(1) if m else ""
            if not u and key == "potency":
                u = "uM"
            if islog and not u:
                u = "M"
            found[c] = (PC_TYPES[key], u, islog)
            continue
        if ("inhib" in low or "%" in low or "percent" in low) and "curve" not in low:
            t = "PCT_INHIBITION" if "inhib" in low else ("PCT_CONTROL" if "control" in low else "PCT_ACTIVITY")
            found[c] = (t, "%", False)
    return found


def pubchem_aid_frames(path, aid, ginfo, include_qual, max_rows, summary):
    reader = pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=100_000, low_memory=False, on_bad_lines="skip", encoding_errors="replace")
    units_meta, qcols, rows_seen, dropped = {}, None, 0, 0
    for ch in reader:
        if "PUBCHEM_RESULT_TAG" not in ch.columns:
            return
        isnum = ch["PUBCHEM_RESULT_TAG"].str.fullmatch(r"\d+")
        meta, data = ch[~isnum], ch[isnum]
        if len(meta):
            r = meta[meta["PUBCHEM_RESULT_TAG"] == "RESULT_UNIT"]
            if len(r):
                units_meta = {c: str(r.iloc[0][c]) for c in ch.columns}
        if qcols is None:
            qcols = pubchem_detect_cols(ch.columns, units_meta)
        if data.empty:
            continue
        if max_rows and rows_seen >= max_rows:
            return
        rows_seen += len(data)
        cid = data["PUBCHEM_CID"].str.strip() if "PUBCHEM_CID" in data else pd.Series("", index=data.index)
        sid = data["PUBCHEM_SID"].str.strip() if "PUBCHEM_SID" in data else pd.Series("", index=data.index)
        outc = (data["PUBCHEM_ACTIVITY_OUTCOME"].str.strip().str.lower() if "PUBCHEM_ACTIVITY_OUTCOME" in data else pd.Series("", index=data.index))
        outc = outc.map(lambda x: OUTCOME.get(x, x))
        has_cid = cid != ""
        dropped += int((~has_cid).sum())
        has_val = pd.Series(False, index=data.index)
        frames = []
        common = lambda mask, col, typ, units: {
            "source_record_id": f"AID{aid}:SID" + sid[mask] + ":" + col, "compound_id": "CID" + cid[mask], "pubchem_cid": cid[mask],
            "assay_id": f"AID{aid}", "activity_comment": outc[mask], "units_original": units,
            "target_id_source": "PubChem assay target annotation", "assay_description": summary.get("name", ""),
            "doc_id": summary.get("source", ""), "notes": f"column={col}"}
        for col, (typ, units, islog) in (qcols or {}).items():
            rel, val = split_rel_value(data[col])
            ok = val.notna() & has_cid
            has_val |= val.notna()
            if not ok.any():
                continue
            v = val[ok]
            if islog:
                v = 10.0 ** v
            f = pd.DataFrame(common(ok, col, typ, units))
            f["activity_type_original"] = typ
            f["relation"] = rel[ok]
            f["value_original"] = v
            f["record_class"] = "single_point" if typ.startswith("PCT_") else ""
            f["measure_semantics"] = "pubchem_assay_column_" + ("pct" if typ.startswith("PCT_") else "concentration")
            frames.append(f)
        if include_qual:
            q = (~has_val) & has_cid
            if q.any():
                f = pd.DataFrame(common(q, "PUBCHEM_ACTIVITY_OUTCOME", "OUTCOME", ""))
                f["activity_type_original"] = "OUTCOME"
                f["relation"] = ""
                f["value_original"] = np.nan
                f["record_class"] = "qualitative"
                f["measure_semantics"] = "pubchem_activity_outcome_only"
                frames.append(f)
        if frames:
            yield pd.concat(frames), dropped


def collect_pubchem(ctx):
    src = "pubchem"
    cfg, man = ctx.cfg, ctx.man
    PUG = URLS["pug"]
    # ---- step 1: enumerate assay IDs for every panel kinase (GeneID first, symbol as fallback)
    gene_aids = {}
    panel = list(ctx.panel.itertuples())
    cached = ctx.cache.kv_get("pc_gene_aids", [r.uniprot for r in panel])

    def find_aids(r):
        if r.uniprot in cached:
            return r.uniprot, json.loads(cached[r.uniprot])
        aids = None
        for gid in [g for g in re.split(r"[;\s]+", r.geneid or "") if g.isdigit()][:1]:
            resp = ctx.http.get(f"{PUG}/assay/target/geneid/{gid}/aids/JSON", key="pubchem", tries=5)
            if resp is not None and resp.status_code == 200:
                aids = resp.json().get("IdentifierList", {}).get("AID", [])
            elif resp is not None and resp.status_code == 404:
                aids = []
        if aids is None:
            resp = ctx.http.get(f"{PUG}/assay/target/genesymbol/{r.gene}/aids/JSON", key="pubchem", tries=5)
            if resp is not None and resp.status_code == 200:
                aids = resp.json().get("IdentifierList", {}).get("AID", [])
            elif resp is not None and resp.status_code == 404:
                aids = []
        if aids is None:
            return r.uniprot, None
        ctx.cache.kv_put("pc_gene_aids", {r.uniprot: json.dumps(aids)})
        return r.uniprot, aids
    log(f"enumerating PubChem assays for {len(panel)} kinases (PUG-REST, ~4 requests/s)", src)
    with ThreadPoolExecutor(min(3, cfg["THREADS"])) as ex:
        for k, (acc, aids) in enumerate(ex.map(find_aids, panel), 1):
            if aids:
                gene_aids[acc] = aids
            if k % 50 == 0:
                log(f"assay enumeration {k}/{len(panel)} kinases | distinct AIDs so far {len({a for v in gene_aids.values() for a in v}):,}", src)
    aid_genes = {}
    for acc, aids in gene_aids.items():
        for a in aids:
            aid_genes.setdefault(a, []).append(acc)
    all_aids = sorted(aid_genes)
    if cfg["PUBCHEM_MAX_AIDS"]:
        # keep the assays that touch the most important kinases first
        prio = {r.uniprot: i for i, r in enumerate(panel)}
        all_aids = sorted(all_aids, key=lambda a: min(prio[x] for x in aid_genes[a]))[:cfg["PUBCHEM_MAX_AIDS"]]
    log(f"{len(all_aids):,} PubChem assays are annotated to panel kinases", src)
    if not all_aids:
        raise RuntimeError("PubChem returned no assays (service unreachable or throttled); re-run later")
    man.set(src, "n_aids", len(all_aids))
    # ---- step 2: assay names (batch summary), best effort
    need = [str(a) for a in all_aids]
    have = ctx.cache.kv_get("pc_sum", need)
    need = [a for a in need if a not in have]
    for i in range(0, len(need), 100):
        b = need[i:i + 100]
        j = ctx.http.get_json(f"{PUG}/assay/aid/{','.join(b)}/summary/JSON", key="pubchem", tries=3)
        res = {}
        for s in ((j or {}).get("AssaySummaries", {}) or {}).get("AssaySummary", []):
            res[str(s.get("AID"))] = json.dumps({"name": (s.get("Name") or "")[:300], "source": s.get("SourceName", "")})
        for a in b:
            res.setdefault(a, json.dumps({"name": "", "source": ""}))
        ctx.cache.kv_put("pc_sum", res)
    # ---- step 3: assay data, one CSV per AID
    parts = ctx.work / "pubchem_parts"
    tmp = ctx.work / "pubchem_tmp"
    parts.mkdir(exist_ok=True)
    tmp.mkdir(exist_ok=True)
    stats = {"done": 0, "rows": 0, "fail": 0}
    t_start = time.time()
    ftp_zips = {}

    def get_aid_file(aid):
        dest = tmp / f"{aid}.csv"
        r = ctx.http.get(f"{PUG}/assay/aid/{aid}/CSV", key="pubchem", stream=True, tries=5, timeout=(20, 600))
        if r is not None and r.status_code == 200:
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            r.close()
            head = dest.read_bytes()[:200] if dest.stat().st_size < 2000 else b""
            if b"PUGREST" in head or b"Fault" in head:
                dest.unlink(missing_ok=True)
                return None, "pugrest_fault"
            return dest, "rest"
        code = r.status_code if r is not None else ctx.http.last_error
        if r is not None:
            r.close()
        if cfg["PUBCHEM_FTP_FALLBACK"] and code in (400, 414, 504, 500, 503, 405):
            lo = ((aid - 1) // 1000) * 1000 + 1
            name = f"{lo:07d}_{lo + 999:07d}.zip"
            z = ctx.http.download([f"{URLS['pubchem_ftp']}/{name}"], ctx.raw / "pubchem_ftp" / name, key="pubchem", min_bytes=1000, magic=b"PK", src=src)
            if z:
                with zipfile.ZipFile(z) as zf:
                    mem = next((n for n in zf.namelist() if re.search(rf"(^|/)0*{aid}\.csv(\.gz)?$", n)), None)
                    if mem:
                        raw = zf.read(mem)
                        if mem.endswith(".gz"):
                            raw = gzip.decompress(raw)
                        dest.write_bytes(raw)
                        return dest, "ftp"
        return None, f"http_{code}"

    def job(aid):
        st = man.get(src, "aids", str(aid), default={})
        if st.get("status") == "done":
            return aid, st.get("n", 0)
        part = parts / f"{aid}.csv"
        part.unlink(missing_ok=True)
        f, how = get_aid_file(aid)
        if f is None:
            if how in ("http_404",):
                touch_header(part)
                man.set(src, "aids", str(aid), {"status": "done", "n": 0, "note": "no data table"})
                return aid, 0
            man.set(src, "aids", str(aid), {"status": "failed", "why": how})
            return aid, -1
        gl = aid_genes[aid]
        summ = json.loads(ctx.cache.kv_get("pc_sum", [str(aid)]).get(str(aid), '{"name":"","source":""}'))
        rows_ = [ctx.by_acc[a] for a in gl if a in ctx.by_acc]
        multi = len(rows_) > 1
        ginfo = {"gene": ";".join(r.gene for r in rows_), "acc": ";".join(r.uniprot for r in rows_),
                 "status": "assay_multi_target_ambiguous" if multi else "pubchem_assay_target_annotation"}
        n = dropped = 0
        try:
            for df, dropped in pubchem_aid_frames(f, aid, ginfo, cfg["PUBCHEM_INCLUDE_QUALITATIVE"], cfg["PUBCHEM_MAX_ROWS_PER_AID"], summ):
                df["gene_symbol"] = ginfo["gene"]
                df["target_name_original"] = ginfo["gene"]
                df["target_name_normalized"] = ginfo["gene"].upper()
                df["uniprot_accession"] = ginfo["acc"]
                df["target_organism"] = "Homo sapiens"
                df["target_type"] = "SINGLE PROTEIN" if not multi else "ASSAY_MULTI_TARGET"
                df["target_mapping_status"] = ginfo["status"]
                n += append_df(part, derive(df, "PUBCHEM", "pug_rest" if how == "rest" else "ftp"))
        except Exception as e:
            part.unlink(missing_ok=True)
            man.set(src, "aids", str(aid), {"status": "failed", "why": f"parse: {e}"[:200]})
            f.unlink(missing_ok=True)
            return aid, -1
        f.unlink(missing_ok=True)
        touch_header(part)
        man.set(src, "aids", str(aid), {"status": "done", "n": n, "dropped_no_cid": dropped, "via": how})
        return aid, n

    with ThreadPoolExecutor(min(3, cfg["THREADS"])) as ex:
        futs = [ex.submit(job, a) for a in all_aids]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                aid, n = f.result()
            except Exception as e:
                log(f"AID job crashed: {e}", src)
                stats["fail"] += 1
                continue
            if n < 0:
                stats["fail"] += 1
            else:
                stats["rows"] += n
            if k % 25 == 0 or k == len(futs):
                el = time.time() - t_start
                log(f"AIDs {k}/{len(futs)} | rows {stats['rows']:,} | failed {stats['fail']} | {k / max(el, 1):.2f} AIDs/s", src)
    failed = [a for a in all_aids if man.get(src, "aids", str(a), "status") != "done"]
    if failed:
        log(f"{len(failed)} AIDs could not be delivered (examples {failed[:5]}); they are retried on the next run", src)
    # ---- step 4: SMILES for every CID, in batches
    cids = set()
    for p in parts.glob("*.csv"):
        for ch in pd.read_csv(p, usecols=["pubchem_cid"], dtype=str, keep_default_na=False, chunksize=500_000):
            cids.update(ch["pubchem_cid"].unique())
    cids = sorted(int(c) for c in cids if c.isdigit())
    log(f"resolving {len(cids):,} compound structures in batches of {cfg['PUBCHEM_CID_BATCH']}", src)
    pubchem_resolve_cids(ctx, cids, src)
    # ---- step 5: final CSV with structures
    final_parts = ctx.work / "pubchem_final_parts"
    shutil.rmtree(final_parts, ignore_errors=True)
    final_parts.mkdir()
    total = 0
    for p in sorted(parts.glob("*.csv")):
        wrote = False
        for ch in pd.read_csv(p, dtype=str, keep_default_na=False, chunksize=200_000):
            m = ctx.cache.kv_get("cid", list(ch["pubchem_cid"].unique()))
            sm = ch["pubchem_cid"].map(lambda c: m.get(c, "\t").split("\t")[0])
            ik = ch["pubchem_cid"].map(lambda c: m.get(c, "\t").split("\t")[1] if "\t" in m.get(c, "\t") else "")
            ch["smiles_raw"], ch["inchikey_source"] = sm, ik
            ch.to_csv(final_parts / p.name, mode="a", header=not wrote, index=False)
            wrote = True
            total += len(ch)
        if not wrote:
            touch_header(final_parts / p.name)
    concat_parts(list(final_parts.glob("*.csv")), ctx.csv_dir / "PUBCHEM.csv")
    shutil.rmtree(final_parts, ignore_errors=True)
    res = {"rows": total}
    if failed:
        res["note"] = f"{len(failed)} AIDs undeliverable (retry by re-running)"
    return res


def pubchem_resolve_cids(ctx, cids, src="pubchem"):
    PUG = URLS["pug"]
    have = set()
    for i in range(0, len(cids), 5000):
        have.update(ctx.cache.kv_get("cid", [str(c) for c in cids[i:i + 5000]]).keys())
    todo = [c for c in cids if str(c) not in have]
    bs = ctx.cfg["PUBCHEM_CID_BATCH"]
    batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]
    names_ladder = ["SMILES,InChIKey", "CanonicalSMILES,InChIKey"]
    state = {"ladder": 0}

    def run(b):
        for lad in range(state["ladder"], len(names_ladder)):
            r = ctx.http.post(f"{PUG}/compound/cid/property/{names_ladder[lad]}/CSV", data={"cid": ",".join(map(str, b))}, key="pubchem", tries=5, timeout=(20, 300))
            if r is not None and r.status_code == 200:
                state["ladder"] = lad
                df = pd.read_csv(io.StringIO(r.text), dtype=str, keep_default_na=False)
                sc = next((c for c in df.columns if "SMILES" in c.upper()), None)
                res = {}
                for _, row in df.iterrows():
                    res[row["CID"]] = f"{row[sc] if sc else ''}\t{row.get('InChIKey', '')}"
                for c in b:
                    res.setdefault(str(c), "\t")
                ctx.cache.kv_put("cid", res)
                return len(b)
            if r is not None and r.status_code == 400:
                continue
            break
        return 0
    done = 0
    with ThreadPoolExecutor(min(3, ctx.cfg["THREADS"])) as ex:
        for k, n in enumerate(ex.map(run, batches), 1):
            done += n
            if k % 20 == 0 or k == len(batches):
                log(f"structures {done:,}/{len(todo):,}", src)


# =====================================================================================
#  post-processing: structure lookups, standardisation, master, counts, report
# =====================================================================================
def source_csv(ctx, s):
    return ctx.csv_dir / f"{s.upper()}.csv"


def read_chunks(ctx, path, usecols=None):
    return pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=ctx.cfg["CHUNK_ROWS"], usecols=usecols, low_memory=False)


def build_lookup(ctx):
    """collect id->SMILES lookups from all finished sources (for records that arrive without a structure)"""
    log("building structure lookup tables (ChEMBL id / InChIKey / PubChem CID -> SMILES)", "post")
    for s in SOURCE_ORDER_MASTER:
        p = source_csv(ctx, s)
        if not p.exists():
            continue
        for ch in read_chunks(ctx, p, ["chembl_molecule_id", "inchikey_source", "pubchem_cid", "smiles_raw"]):
            ch = ch[ch["smiles_raw"] != ""]
            for kind, col in (("chembl", "chembl_molecule_id"), ("ik", "inchikey_source"), ("cid", "pubchem_cid")):
                x = ch[ch[col] != ""][[col, "smiles_raw"]].drop_duplicates(col)
                if len(x):
                    ctx.cache.lk_put(kind, list(zip(x[col], x["smiles_raw"])))


def attach_smiles(ctx, df):
    sm = df["smiles_raw"].str.strip()
    status = np.where(sm != "", "from_source", "missing")
    miss = sm == ""
    if miss.any():
        for kind, col in (("chembl", "chembl_molecule_id"), ("ik", "inchikey_source"), ("cid", "pubchem_cid")):
            idx = sm.index[sm == ""]
            if len(idx) == 0:
                break
            keys = df.loc[idx, col]
            keys = keys[keys != ""]
            if keys.empty:
                continue
            m = ctx.cache.lk_get(kind, list(keys.unique()))
            got = keys.map(m).dropna()
            if len(got):
                sm.loc[got.index] = got
                status = pd.Series(status, index=df.index)
                status.loc[got.index] = "resolved_from_" + kind
                status = status.values
    return sm, np.asarray(status)


def remote_structure_lookup(ctx):
    """InChIKey -> SMILES for records still without structure (typically DTC). Bounded, batched, cached."""
    cap = ctx.cfg["MAX_REMOTE_STRUCTURE_LOOKUPS"]
    if not cap:
        return
    p = source_csv(ctx, "dtc")
    if not p.exists():
        return
    keys = set()
    for ch in read_chunks(ctx, p, ["smiles_raw", "chembl_molecule_id", "inchikey_source"]):
        ch = ch[(ch.smiles_raw == "") & (ch.inchikey_source != "")]
        keys.update(ch.inchikey_source.unique())
    if not keys:
        return
    known = ctx.cache.lk_get("ik", list(keys))
    done_ns = ctx.cache.kv_get("ik_remote", list(keys))
    todo = sorted(k for k in keys if k not in known and k not in done_ns)[:cap]
    if not todo:
        return
    log(f"{len(todo):,} InChIKeys without structure -> PubChem batch lookup", "post")
    PUG = URLS["pug"]
    for i in range(0, len(todo), 100):
        b = todo[i:i + 100]
        r = ctx.http.post(f"{PUG}/compound/inchikey/property/SMILES,InChIKey/CSV", data={"inchikey": ",".join(b)}, key="pubchem", tries=4, timeout=(20, 200))
        found = {}
        if r is not None and r.status_code == 200:
            try:
                df = pd.read_csv(io.StringIO(r.text), dtype=str, keep_default_na=False)
                sc = next((c for c in df.columns if "SMILES" in c.upper()), None)
                if sc and "InChIKey" in df.columns:
                    found = dict(zip(df["InChIKey"], df[sc]))
            except Exception:
                pass
        if found:
            ctx.cache.lk_put("ik", list(found.items()))
        ctx.cache.kv_put("ik_remote", {k: "1" for k in b})
        if (i // 100) % 20 == 0:
            log(f"remote structure lookup {i + len(b):,}/{len(todo):,}", "post")


# -- standardisation workers (module level so multiprocessing can import them)
_STD = None


def _std_init():
    global _STD
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    _STD = (Chem, rdMolStandardize.LargestFragmentChooser(), rdMolStandardize.Uncharger())


def _std_one(smi):
    try:
        if _STD is None:
            _std_init()
        Chem, lfc, unch = _STD
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return (smi, "", "", "invalid_smiles")
        m = unch.uncharge(lfc.choose(m))
        s = Chem.MolToSmiles(m)
        try:
            from rdkit.Chem import inchi
            ik = inchi.MolToInchiKey(m) or ""
        except Exception:
            ik = ""
        return (smi, s, ik, "ok")
    except Exception:
        return (smi, "", "", "error")


def standardize_all(ctx):
    if not ctx.cfg["STANDARDIZE"]:
        return
    try:
        import rdkit  # noqa: F401
    except ImportError:
        log("RDKit not installed -> skipping standardisation (pip install rdkit). Duplicate flags will use source InChIKeys only.", "post")
        ctx.have_rdkit = False
        return
    ctx.have_rdkit = True
    known = ctx.cache.std_known()
    todo = set()
    for s in SOURCE_ORDER_MASTER:
        p = source_csv(ctx, s)
        if not p.exists():
            continue
        for ch in read_chunks(ctx, p, ["smiles_raw", "chembl_molecule_id", "inchikey_source", "pubchem_cid"]):
            sm, _ = attach_smiles(ctx, ch)
            u = set(sm[sm != ""].unique())
            todo.update(u - known)
    todo = sorted(todo)
    log(f"standardising {len(todo):,} new unique structures (largest fragment, neutralised, canonical + InChIKey)", "post")
    if not todo:
        return
    nproc = max(1, min(8, (os.cpu_count() or 2) - 1))
    t0 = time.time()
    done = 0
    with mp.Pool(nproc, initializer=_std_init) as pool:
        for i in range(0, len(todo), 20000):
            batch = todo[i:i + 20000]
            res = pool.map(_std_one, batch, chunksize=250)
            ctx.cache.std_put(res)
            done += len(batch)
            log(f"standardised {done:,}/{len(todo):,} ({done / max(time.time() - t0, 1):.0f} structures/s)", "post")


MASTER_EXTRA = ["smiles_used", "std_smiles", "inchikey", "inchikey14", "compound_key", "structure_status", "in_panel",
                "relatedness_tier", "is_human_target", "dup_key", "is_cross_source_duplicate",
                "is_within_source_duplicate", "is_training_ready_exact"]


def _h(series: pd.Series) -> np.ndarray:
    return pd.util.hash_pandas_object(series.astype(str), index=False).values


def build_master(ctx):
    cfg = ctx.cfg
    out = ctx.out / "TNK1_kinase_master.csv"
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    pan = ctx.panel
    gene_by_acc = dict(zip(pan.uniprot, pan.gene))
    tier_by_acc = dict(zip(pan.uniprot, pan.relatedness_tier))
    chembl_by_acc = {}
    for r in pan.itertuples():
        for t in re.split(r"[;\s]+", r.chembl_targets or ""):
            if t.startswith("CHEMBL"):
                chembl_by_acc.setdefault(r.uniprot, t)
    tier_by_gene = {g.upper(): t for g, t in zip(pan.gene, pan.relatedness_tier)}
    prev_keys = set()
    arrays = {"meas": {}, "pair": {}, "comp": {}}
    counts, comp_pairs = {}, {}
    stats = {}
    first = True
    for s in SOURCE_ORDER_MASTER:
        p = source_csv(ctx, s)
        if not p.exists():
            continue
        cur = set()
        a_meas, a_pair, a_comp = [], [], []
        nraw = nnonhuman = ndrop = 0
        cp_frames = []
        for ch in read_chunks(ctx, p):
            nraw += len(ch)
            for c in NUM_COLS:
                ch[c] = pd.to_numeric(ch[c], errors="coerce")
            sm, st = attach_smiles(ctx, ch)
            ch["smiles_used"] = sm.values
            ch["structure_status"] = st
            if getattr(ctx, "have_rdkit", False):
                m = ctx.cache.std_get(list(sm[sm != ""].unique()))
                ch["std_smiles"] = sm.map(lambda x: m.get(x, ("", "", ""))[0] if x else "").values
                ch["inchikey"] = sm.map(lambda x: m.get(x, ("", "", ""))[1] if x else "").values
                bad = sm.map(lambda x: bool(x) and m.get(x, ("", "", "x"))[2] != "ok").values
                ch.loc[bad, "structure_status"] = "invalid_smiles"
            else:
                ch["std_smiles"] = sm.values
                ch["inchikey"] = ""
            ch["inchikey"] = np.where(ch["inchikey"] == "", ch["inchikey_source"], ch["inchikey"])
            ch["inchikey14"] = ch["inchikey"].str[:14]
            ch["compound_key"] = np.where(ch["inchikey14"] != "", ch["inchikey14"],
                                          np.where(ch["std_smiles"] != "", "SMI:" + ch["std_smiles"], s.upper() + ":" + ch["compound_id"]))
            # normalise the target using the panel (UniProt is the stable id)
            single = ~ch["uniprot_accession"].str.contains(";")
            acc = ch["uniprot_accession"].where(single, "")
            mapped = acc.map(gene_by_acc)
            ch["gene_symbol"] = np.where(mapped.notna(), mapped, ch["gene_symbol"])
            ch["target_name_normalized"] = np.where(ch["gene_symbol"] != "", ch["gene_symbol"].str.upper(), ch["target_name_normalized"])
            ch["target_chembl_id"] = np.where(ch["target_chembl_id"] == "", acc.map(chembl_by_acc).fillna(""), ch["target_chembl_id"])
            ch["in_panel"] = ch["uniprot_accession"].isin(gene_by_acc) | ch["gene_symbol"].str.upper().isin(tier_by_gene)
            ch["relatedness_tier"] = np.where(acc.map(tier_by_acc).notna(), acc.map(tier_by_acc),
                                              ch["gene_symbol"].str.upper().map(tier_by_gene).fillna(""))
            org = ch["target_organism"].str.lower()
            ch["is_human_target"] = org.str.contains("sapiens|human") | (org == "")
            keep = ch["is_human_target"].copy()
            nnonhuman += int((~keep).sum())
            if cfg["MASTER_ONLY_PANEL"]:
                keep &= ch["in_panel"]
            if cfg["MASTER_DROP_NO_STRUCTURE"]:
                keep &= (ch["smiles_used"] != "")
            ndrop += int((~keep).sum()) - 0
            ch = ch[keep]
            if ch.empty:
                continue
            tkey = np.where(ch["uniprot_accession"] != "", ch["uniprot_accession"], ch["target_name_normalized"])
            vstr = ch["value_std"].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
            dk = (ch["compound_key"] + "|" + pd.Series(tkey, index=ch.index) + "|" + ch["activity_type"].str.upper() + "|" +
                  ch["relation"] + "|" + vstr + "|" + ch["units_std"])
            ch["dup_key"] = _h(dk)
            keys = ch["dup_key"].values
            ch["is_cross_source_duplicate"] = np.fromiter((k in prev_keys for k in keys), bool, len(keys))
            within = np.zeros(len(keys), dtype=bool)
            for i, k in enumerate(keys):
                if k in cur:
                    within[i] = True
                else:
                    cur.add(k)
            ch["is_within_source_duplicate"] = within
            ch["is_training_ready_exact"] = ((ch["record_class"] == "quantitative") & (ch["relation"] == "=") &
                                             ch["activity_type"].isin(["IC50", "Ki", "Kd"]) & (ch["value_nM"] > 0) &
                                             (ch["smiles_used"] != "") & ch["in_panel"] & (ch["structure_status"] != "invalid_smiles") &
                                             (ch["flag_contamination"] == ""))
            a_meas.append(keys)
            a_pair.append(_h(ch["compound_key"] + "|" + pd.Series(tkey, index=ch.index)))
            a_comp.append(_h(ch["compound_key"]))
            g = ch.groupby(ch["gene_symbol"].where(ch["gene_symbol"] != "", ch["target_name_normalized"])).size()
            for gk, n in g.items():
                counts[(s, gk)] = counts.get((s, gk), 0) + int(n)
            cp_frames.append(pd.DataFrame({"g": ch["gene_symbol"].where(ch["gene_symbol"] != "", ch["target_name_normalized"]).values, "c": a_comp[-1]}).drop_duplicates())
            ch["target_chembl_id"] = ch["target_chembl_id"]
            ch[COLS + MASTER_EXTRA].to_csv(tmp, mode="a", header=first, index=False)
            first = False
        prev_keys |= cur
        for nm, arr in (("meas", a_meas), ("pair", a_pair), ("comp", a_comp)):
            arrays[nm][s] = np.unique(np.concatenate(arr)) if arr else np.array([], dtype=np.uint64)
        if cp_frames:
            cp = pd.concat(cp_frames).drop_duplicates()
            for gk, n in cp.groupby("g").size().items():
                comp_pairs[(s, gk)] = int(n)
        stats[s] = {"raw_rows_in_source_csv": nraw, "rows_in_master": len(cur) and None, "nonhuman_excluded": nnonhuman}
        log(f"merged {s.upper()}: {nraw:,} rows read", "master")
    if first:
        raise RuntimeError("no source CSV available to merge")
    os.replace(tmp, out)
    ctx.master_info = {"counts": counts, "comp_pairs": comp_pairs, "arrays": arrays, "stats": stats}
    return out


def write_reports(ctx, results):
    info = ctx.master_info
    counts, comp_pairs, arrays = info["counts"], info["comp_pairs"], info["arrays"]
    pan = ctx.panel
    by_gene = {g: (u, t, r) for g, u, t, r in zip(pan.gene, pan.uniprot, pan.tier, pan.relatedness_tier)}
    genes_all = sorted({g for (_, g) in counts})
    rows = []
    for g in sorted(set(genes_all) | set(pan.gene)):
        u, t, r = by_gene.get(g, ("", "non_panel", ""))
        for s in SOURCE_ORDER_MASTER:
            rows.append({"kinase": g, "uniprot": u, "panel_tier": t, "relatedness_tier": r, "source": s.upper(),
                         "n_records": counts.get((s, g), 0), "n_unique_compounds": comp_pairs.get((s, g), 0)})
    cdf = pd.DataFrame(rows)
    cdf.to_csv(ctx.out / "counts_by_source_kinase.csv", index=False)
    srcs = [s for s in SOURCE_ORDER_MASTER if s in arrays["meas"]]
    lines = []
    ap = lines.append
    ap("TNK1 KINOME HARVEST REPORT  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    ap("=" * 78)
    ap(f"panel mode: {ctx.cfg['PANEL_MODE']} | kinases in panel: {len(pan)} | core targets: {int((pan.tier == 'core').sum())}")
    ap("")
    ap("1) RECORDS PER SOURCE  (master = human targets only; raw = rows in the source CSV)")
    ap(f"{'source':<11}{'status':<10}{'raw rows':>12}{'in master':>12}{'unique meas.':>14}{'unique cpds':>13}")
    tot_master = 0
    for s in SOURCE_ORDER_MASTER:
        r = results.get(s, {})
        n_master = int(sum(v for (ss, _), v in counts.items() if ss == s))
        tot_master += n_master
        ap(f"{s.upper():<11}{r.get('status', 'n/a'):<10}{r.get('rows', 0):>12,}{n_master:>12,}{len(arrays['meas'].get(s, [])):>14,}{len(arrays['comp'].get(s, [])):>13,}")
        if r.get("status") == "failed":
            ap(f"            -> {r.get('why', '')[:300]}")
        elif r.get("note"):
            ap(f"            note: {r['note']}")
    allm = np.unique(np.concatenate([arrays["meas"][s] for s in srcs])) if srcs else []
    allc = np.unique(np.concatenate([arrays["comp"][s] for s in srcs])) if srcs else []
    ap(f"{'TOTAL':<11}{'':<10}{'':>12}{tot_master:>12,}{len(allm):>14,}{len(allc):>13,}")
    ap("   'unique meas.' = distinct (standardised compound, target, type, relation, value, unit) across ALL sources;")
    ap("   a row in the master with is_cross_source_duplicate=True is the same measurement already present in an earlier source.")
    ap(f"   Target of 1,000,000 raw records: {'REACHED' if tot_master >= 1_000_000 else 'NOT reached with currently available public data'} ({tot_master:,} rows).")
    ap("")
    ap("2) TNK1 and its closest relatives (records in master per source)")
    pv = cdf[cdf.kinase.isin(["TNK1", "TNK2", "SRC", "ABL1", "BTK", "YES1", "FYN", "LCK", "HCK", "ABL2"])].pivot_table(
        index="kinase", columns="source", values="n_records", aggfunc="sum", fill_value=0)
    ap(pv.to_string())
    ap("")
    ap("3) TOP 40 KINASES BY TOTAL RECORDS  (full table: counts_by_source_kinase.csv)")
    top = cdf.groupby("kinase").n_records.sum().sort_values(ascending=False).head(40)
    ap(top.to_string())
    ap("")
    ap("4) OVERLAP BETWEEN SOURCES  (number of shared standardised measurements / shared compounds)")
    if srcs:
        mm = pd.DataFrame([[len(np.intersect1d(arrays["meas"][a], arrays["meas"][b], assume_unique=True)) for b in srcs] for a in srcs], index=[s.upper() for s in srcs], columns=[s.upper() for s in srcs])
        cc = pd.DataFrame([[len(np.intersect1d(arrays["comp"][a], arrays["comp"][b], assume_unique=True)) for b in srcs] for a in srcs], index=[s.upper() for s in srcs], columns=[s.upper() for s in srcs])
        ap("shared measurements:")
        ap(mm.to_string())
        ap("shared compounds:")
        ap(cc.to_string())
    ap("")
    ap("5) HOW TO READ THE VALUES (important for the model)")
    ap(" - activity_type IC50/Ki/Kd/EC50 with value_nM and p_activity (=9-log10 nM) are real concentrations.")
    ap(" - PCT_INHIBITION / PCT_CONTROL are single-concentration screens (PKIS, scanMAX): keep as separate label type.")
    ap("   NOTE: '% control' is reversed (lower = stronger) compared with '% inhibition'.")
    ap(" - KIBA_score is a model-integrated score: NOT convertible to IC50/Ki/Kd. Davis Kd is real Kd; value 10000 nM is censored ('>').")
    ap(" - IUPHAR values are curated medians (pAffinity) - not single experiments.")
    ap(" - relation '>' / '<' rows are censored; do not use them as exact regression labels.")
    ap(" - is_training_ready_exact = quantitative IC50/Ki/Kd with relation '=', valid structure, panel kinase, no contamination flag.")
    ap(" - TNK1 vs TNK2 vs tankyrase: rows flagged in flag_contamination come from assays whose text mentions tankyrase/PARP.")
    ap(" - the 'relatedness_tier' column is a heuristic grouping (1 = TNK1/TNK2, 2 = Src/Abl/Tec/Fes/Fak-type cytoplasmic TKs, 3 = other TKs, 4 = rest).")
    ap("")
    ap("6) SOURCES THAT NEED YOU (if any)")
    need = False
    for s, r in results.items():
        if r.get("status") == "failed":
            need = True
            ap(f" - {s.upper()}: {r.get('why', '')}")
    if not need:
        ap(" - none")
    atomic_write_text(ctx.out / "REPORT.txt", "\n".join(lines))
    print("\n".join(lines))


# =====================================================================================
#  --check : endpoint doctor
# =====================================================================================
def doctor(ctx):
    tests = [
        ("UniProt", f"{URLS['uniprot']}?query=accession:Q13470&format=tsv&fields=accession,gene_primary"),
        ("ChEMBL API", f"{URLS['chembl']}/target/CHEMBL5334.json"),
        ("PubChem PUG", f"{URLS['pug']}/assay/target/genesymbol/TNK1/aids/JSON"),
        ("BindingDB page", URLS["bdb_page"]),
        ("DTC bulk file", URLS["dtc_static"][0]),
        ("DTC API root", URLS["dtc_api_root"]),
        ("IUPHAR bulk", f"{URLS['gtop_data']}/interactions.csv"),
        ("IUPHAR API", f"{URLS['gtop_api']}/targets?geneSymbol=TNK1"),
        ("GitHub (Davis/KIBA)", f"{URLS['deepdta']}/davis/ligands_can.txt"),
        ("PubChem FTP zip", f"{URLS['pubchem_ftp']}/0000001_0001000.zip"),
    ]
    print("\nEndpoint check from this computer/network:")
    for name, url in tests:
        t = time.time()
        try:
            r = ctx.http.s.get(url, stream=True, timeout=(10, 30), headers={"Range": "bytes=0-1023"})
            body = next(r.iter_content(512), b"")
            kind = "HTML" if body[:1] == b"<" else "data"
            ok = r.status_code in (200, 206)
            r.close()
            print(f"  {'OK ' if ok else 'FAIL'} {name:<22} HTTP {r.status_code} ({kind}, {time.time() - t:.1f}s)")
        except Exception as e:
            print(f"  FAIL {name:<22} {type(e).__name__}: {str(e)[:90]}")
    print("\nAnything marked FAIL is skipped (or retried) by the main run and explained in REPORT.txt.\n")


# =====================================================================================
#  main
# =====================================================================================
COLLECTORS = {"chembl": collect_chembl, "pubchem": collect_pubchem, "bindingdb": collect_bindingdb, "dtc": collect_dtc,
              "iuphar": collect_iuphar, "pkis": collect_pkis, "pkis2": collect_pkis2, "davis": collect_davis,
              "kiba": collect_kiba, "metz": collect_metz}


def run_source(ctx, s):
    man = ctx.man
    st = man.get("sources", s, "status")
    if st == "done" and ctx.cfg["RESUME"] and source_csv(ctx, s).exists():
        log(f"{s.upper()} already complete ({man.get('sources', s, 'rows'):,} rows) - skipping (use --fresh to redo)")
        return {"status": "done", "rows": man.get("sources", s, "rows", default=0), "note": man.get("sources", s, "note", default="")}
    log(f"===== {s.upper()} =====")
    t = time.time()
    try:
        res = COLLECTORS[s](ctx) or {}
        res.setdefault("rows", 0)
        res["status"] = "done"
        man.set("sources", s, {"status": "done", "rows": res["rows"], "note": res.get("note", ""), "seconds": round(time.time() - t)}, force=True)
        log(f"{s.upper()} finished: {res['rows']:,} records in {time.time() - t:.0f}s", s)
        return res
    except KeyboardInterrupt:
        raise
    except Exception as e:
        why = f"{type(e).__name__}: {e}"
        log(f"{s.upper()} FAILED -> {why}", s)
        log(traceback.format_exc().splitlines()[-3] if traceback.format_exc() else "", s)
        man.set("sources", s, {"status": "failed", "why": why[:600], "rows": 0}, force=True)
        p = source_csv(ctx, s)
        if not p.exists():
            touch_header(p)           # keep the one-file-per-source contract even when nothing could be collected
        return {"status": "failed", "why": why[:600], "rows": 0}


def main():
    ap = argparse.ArgumentParser(description="TNK1 / kinome bioactivity harvester (10 sources, resumable)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sources", default=None, help="comma list, e.g. davis,kiba,chembl")
    ap.add_argument("--panel", choices=["core", "tyrosine", "kinome", "broad"], default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--email", default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoints and caches")
    ap.add_argument("--check", action="store_true", help="test all endpoints and exit")
    ap.add_argument("--postprocess-only", action="store_true")
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--max-chembl-targets", type=int, default=None)
    ap.add_argument("--max-aids", type=int, default=None)
    ap.add_argument("--include-complexes", action="store_true")
    ap.add_argument("--only-panel", action="store_true", help="master keeps only panel kinases")
    a = ap.parse_args()

    cfg = json.loads(json.dumps(CONFIG))
    if a.out:
        cfg["OUTPUT_DIR"] = a.out
    if a.panel:
        cfg["PANEL_MODE"] = a.panel
    if a.threads:
        cfg["THREADS"] = a.threads
    if a.email:
        cfg["EMAIL"] = a.email
    if a.no_standardize:
        cfg["STANDARDIZE"] = False
    if a.max_chembl_targets:
        cfg["CHEMBL_MAX_TARGETS"] = a.max_chembl_targets
    if a.max_aids:
        cfg["PUBCHEM_MAX_AIDS"] = a.max_aids
    if a.include_complexes:
        cfg["CHEMBL_INCLUDE_COMPLEXES"] = True
    if a.only_panel:
        cfg["MASTER_ONLY_PANEL"] = True
    if a.fresh:
        cfg["RESUME"] = False
    if a.sources:
        wanted = {s.strip().lower() for s in a.sources.split(",")}
        bad = wanted - set(COLLECTORS)
        if bad:
            sys.exit(f"unknown source(s): {bad}; choose from {sorted(COLLECTORS)}")
        cfg["SOURCES"] = {s: (s in wanted) for s in COLLECTORS}

    ctx = Ctx()
    ctx.cfg = cfg
    ctx.out = Path(cfg["OUTPUT_DIR"]).resolve()
    ctx.csv_dir = ctx.out / "csv"
    ctx.work = ctx.out / "_work"
    ctx.raw = ctx.work / "raw"
    ctx.manual = ctx.out / "manual"
    for d in (ctx.csv_dir, ctx.work, ctx.raw, ctx.manual):
        d.mkdir(parents=True, exist_ok=True)
    ua = f"TNK1-kinome-harvester/{VERSION} (academic research{'; ' + cfg['EMAIL'] if cfg['EMAIL'] else ''})"
    ctx.http = Http(ua)
    if a.fresh:
        for f in (ctx.out / "manifest.json", ctx.work / "cache.sqlite", ctx.work / "panel.json", ctx.work / "chembl_targets.json"):
            Path(f).unlink(missing_ok=True)
        for d in ("chembl_parts", "pubchem_parts", "pkis_parts", "pkis2_parts", "metz_parts"):
            shutil.rmtree(ctx.work / d, ignore_errors=True)
    ctx.man = Manifest(ctx.out / "manifest.json", fresh=a.fresh)
    ctx.cache = Cache(ctx.work / "cache.sqlite")
    log(f"output folder: {ctx.out}")
    if a.check:
        doctor(ctx)
        return
    ctx.panel = build_panel(ctx)
    panel_indexes(ctx)
    for s in ctx.panel[ctx.panel.gene.isin(["TNK1", "TNK2"])].itertuples():
        log(f"core check: {s.gene} = {s.uniprot} (kept as a separate target)", "panel")

    results = {}
    try:
        if not a.postprocess_only:
            for s in SOURCE_ORDER_RUN:
                if cfg["SOURCES"].get(s):
                    results[s] = run_source(ctx, s)
                    log(f"progress saved to {ctx.out / 'manifest.json'}")
        for s in SOURCE_ORDER_MASTER:
            if s not in results:
                m = ctx.man.get("sources", s, default={})
                results[s] = {"status": m.get("status", "n/a" if not source_csv(ctx, s).exists() else "done"),
                              "rows": m.get("rows", 0), "why": m.get("why", ""), "note": m.get("note", "")}
        log("===== POST-PROCESSING =====")
        build_lookup(ctx)
        remote_structure_lookup(ctx)
        build_lookup(ctx)
        standardize_all(ctx)
        build_master(ctx)
        write_reports(ctx, results)
    except KeyboardInterrupt:
        ctx.man.save(force=True)
        log("interrupted - everything downloaded so far is saved; run the same command again to resume")
        sys.exit(130)
    ctx.man.save(force=True)
    log(f"DONE. Master file: {ctx.out / 'TNK1_kinase_master.csv'}")


if __name__ == "__main__":
    mp.freeze_support()
    main()