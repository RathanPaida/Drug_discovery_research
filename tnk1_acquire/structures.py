"""Tier 7 -- structures.

There is no experimental crystal structure of the TNK1 kinase domain, which is
why 2D predictions and docking correlate at only r ~= 0.19. The acquisition
answer is to stop docking into a single AlphaFold model and build a *conformer
ensemble*:

  * `AF-Q13470-F1`            -- AlphaFold TNK1 model (+ per-residue pLDDT,
                                 which tells you which parts of the pocket the
                                 model is actually confident about)
  * `4EWH`                    -- ACK1/TNK2 kinase domain, ~68% identity to the
                                 TNK1 kinase domain; the best experimental
                                 template available
  * further TNK2/ACK1 entries -- discovered from RCSB by UniProt accession, so
                                 you get the DFG-in/out and activation-loop
                                 variety that a single model cannot give you
  * SRC/ABL1 reference entries-- for the off-target docking comparison

Docking into an ensemble and taking a per-pose-family best score is a different
estimator than docking into one model; it is the cheapest available route to
moving r = 0.19 before any crystallography exists.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .util import LOG, Http, Layout, Manifest, have, try_import

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"
RCSB_FILES = "https://files.rcsb.org/download"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA = "https://data.rcsb.org/rest/v1/core/entry"

#: Experimental template for the TNK1 kinase domain by homology.
ACK1_TEMPLATE = "4EWH"


# --------------------------------------------------------------------------- #
# AlphaFold
# --------------------------------------------------------------------------- #
def fetch_alphafold(http: Http, layout: Layout, manifest: Manifest,
                    accession: str, *, force: bool = False) -> dict[str, Path]:
    """Download the AlphaFold model + PAE/pLDDT metadata for an accession."""
    out: dict[str, Path] = {}
    d = layout.structures / "alphafold"
    d.mkdir(parents=True, exist_ok=True)
    try:
        meta = http.json(f"{ALPHAFOLD_API}/{accession}")
    except Exception as exc:                                       # noqa: BLE001
        LOG.error("AlphaFold API failed for %s: %s", accession, exc)
        return out
    if not meta:
        LOG.error("AlphaFold has no entry for %s", accession)
        return out
    entry = meta[0]
    for key, suffix in (("pdbUrl", ".pdb"), ("cifUrl", ".cif"),
                        ("paeDocUrl", "_pae.json"),
                        ("paeImageUrl", "_pae.png")):
        url = entry.get(key)
        if not url:
            continue
        name = f"{accession.lower()}_alphafold{suffix}"
        try:
            p = http.download(url, d / name, force=force)
        except Exception as exc:                                   # noqa: BLE001
            LOG.warning("  %s failed: %s", key, exc)
            continue
        manifest.record(p, tier="7", source="alphafold", url=url,
                        notes=f"{accession} {entry.get('latestVersion', '')}")
        out[key] = p

    meta_path = d / f"{accession.lower()}_alphafold_meta.json"
    meta_path.write_text(json.dumps(entry, indent=2))
    manifest.record(meta_path, tier="7", source="alphafold",
                    notes="model version, UniProt range, model confidence")

    pdb = out.get("pdbUrl")
    if pdb is not None:
        plddt = plddt_profile(pdb)
        pl_path = d / f"{accession.lower()}_plddt.csv"
        plddt.to_csv(pl_path, index=False)
        manifest.record(pl_path, tier="7", source="derived", rows=len(plddt),
                        notes="per-residue pLDDT from the model B-factor column")
        low = plddt[plddt["plddt"] < 70]
        LOG.info("%s: %d/%d residues below pLDDT 70 -- treat those pocket "
                 "regions as unmodelled, not as geometry",
                 accession, len(low), len(plddt))
    return out


def plddt_profile(pdb: Path) -> pd.DataFrame:
    """AlphaFold writes pLDDT into the B-factor column of CA atoms."""
    rows = []
    for line in pdb.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            rows.append({
                "resi": int(line[22:26]),
                "resn": line[17:20].strip(),
                "chain": line[21].strip(),
                "plddt": float(line[60:66]),
            })
        except ValueError:
            continue
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# RCSB
# --------------------------------------------------------------------------- #
def fetch_pdb(http: Http, layout: Layout, manifest: Manifest, pdb_id: str, *,
              force: bool = False, fmt: str = "pdb") -> Path | None:
    d = layout.structures / "pdb"
    d.mkdir(parents=True, exist_ok=True)
    url = f"{RCSB_FILES}/{pdb_id.upper()}.{fmt}"
    try:
        p = http.download(url, d / f"{pdb_id.lower()}.{fmt}", force=force)
    except Exception as exc:                                       # noqa: BLE001
        LOG.error("PDB %s download failed: %s", pdb_id, exc)
        return None
    manifest.record(p, tier="7", source="rcsb", url=url, notes=f"PDB {pdb_id}")
    return p


def find_pdb_entries(http: Http, accession: str, *, limit: int = 50) -> list[str]:
    """RCSB full-text/attribute search for entries containing an accession."""
    query = {
        "query": {
            "type": "group", "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text",
                 "parameters": {
                     "attribute": "rcsb_polymer_entity_container_identifiers."
                                  "reference_sequence_identifiers.database_accession",
                     "operator": "exact_match", "value": accession}},
                {"type": "terminal", "service": "text",
                 "parameters": {
                     "attribute": "rcsb_polymer_entity_container_identifiers."
                                  "reference_sequence_identifiers.database_name",
                     "operator": "exact_match", "value": "UniProt"}},
            ],
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": limit}},
    }
    try:
        r = http.s.post(RCSB_SEARCH, json=query, timeout=60)
        r.raise_for_status()
        ids = [x["identifier"] for x in r.json().get("result_set", [])]
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("RCSB search failed for %s: %s", accession, exc)
        return []
    LOG.info("RCSB: %d entries for %s", len(ids), accession)
    return ids


def build_structure_set(http: Http, layout: Layout, manifest: Manifest, *,
                        primary_accession: str = "Q13470",
                        template_accessions: tuple[str, ...] = ("Q07912",),
                        reference_accessions: tuple[str, ...] = ("P12931", "P00519"),
                        max_per_accession: int = 25,
                        force: bool = False) -> pd.DataFrame:
    rows: list[dict] = []

    af = fetch_alphafold(http, layout, manifest, primary_accession, force=force)
    if "pdbUrl" in af:
        rows.append({"id": f"AF-{primary_accession}", "kind": "alphafold",
                     "accession": primary_accession, "path": str(af["pdbUrl"]),
                     "role": "primary_model"})

    # The named template first, then whatever else RCSB has for ACK1/TNK2.
    tmpl = fetch_pdb(http, layout, manifest, ACK1_TEMPLATE, force=force)
    if tmpl:
        rows.append({"id": ACK1_TEMPLATE, "kind": "xray", "accession": "Q07912",
                     "path": str(tmpl), "role": "homology_template"})

    for acc in template_accessions:
        for pdb_id in find_pdb_entries(http, acc, limit=max_per_accession):
            if pdb_id.upper() == ACK1_TEMPLATE:
                continue
            p = fetch_pdb(http, layout, manifest, pdb_id, force=force)
            if p:
                rows.append({"id": pdb_id, "kind": "xray", "accession": acc,
                             "path": str(p), "role": "ensemble_member"})

    for acc in reference_accessions:
        for pdb_id in find_pdb_entries(http, acc, limit=5):
            p = fetch_pdb(http, layout, manifest, pdb_id, force=force)
            if p:
                rows.append({"id": pdb_id, "kind": "xray", "accession": acc,
                             "path": str(p), "role": "offtarget_reference"})

    df = pd.DataFrame(rows)
    idx = layout.structures / "structure_index.csv"
    df.to_csv(idx, index=False)
    manifest.record(idx, tier="7", source="derived", rows=len(df),
                    notes="conformer ensemble index for docking")
    LOG.info("structure set: %d files (%s)", len(df),
             df["role"].value_counts().to_dict() if len(df) else {})
    return df


# --------------------------------------------------------------------------- #
# receptor prep
# --------------------------------------------------------------------------- #
def prepare_receptor(pdb: Path, out_pdbqt: Path) -> Path | None:
    """PDB -> PDBQT for AutoDock Vina, via Meeko if present, else Open Babel.

    Meeko is preferred: Open Babel's `-xr` output is accepted by Vina but its
    protonation and atom typing are cruder, and for a pocket you are already
    only modelling at r = 0.19 that difference is not where you want extra
    noise.
    """
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    if try_import("meeko") is not None:
        import subprocess                                          # noqa: PLC0415
        import sys                                                 # noqa: PLC0415
        try:
            subprocess.run(                                        # noqa: S603
                [sys.executable, "-m", "meeko.cli.mk_prepare_receptor",
                 "--read_pdb", str(pdb), "-p", str(out_pdbqt)],
                check=True, capture_output=True, text=True, timeout=600)
            LOG.info("meeko receptor -> %s", out_pdbqt.name)
            return out_pdbqt
        except Exception as exc:                                   # noqa: BLE001
            LOG.warning("meeko receptor prep failed (%s); trying obabel", exc)
    if have("obabel"):
        import subprocess                                          # noqa: PLC0415
        try:
            subprocess.run(                                        # noqa: S603
                ["obabel", str(pdb), "-O", str(out_pdbqt), "-xr", "-p", "7.4"],
                check=True, capture_output=True, text=True, timeout=600)
            LOG.info("obabel receptor -> %s", out_pdbqt.name)
            return out_pdbqt
        except Exception as exc:                                   # noqa: BLE001
            LOG.error("obabel receptor prep failed: %s", exc)
            return None
    LOG.error("neither meeko nor obabel available -- install one: "
              "pip install meeko  OR  conda install -c conda-forge openbabel")
    return None


def prepare_all_receptors(index: pd.DataFrame, layout: Layout,
                          manifest: Manifest) -> list[Path]:
    out: list[Path] = []
    d = layout.structures / "pdbqt"
    for _, r in index.iterrows():
        src = Path(r["path"])
        if src.suffix != ".pdb":
            continue
        dst = d / (src.stem + ".pdbqt")
        p = prepare_receptor(src, dst)
        if p and p.exists():
            manifest.record(p, tier="7", source="derived",
                            notes=f"receptor for docking, from {src.name}")
            out.append(p)
    LOG.info("prepared %d receptors", len(out))
    return out
