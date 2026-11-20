"""Target registry: the primary target, the off-target selectivity panel, and
resolution of UniProt accession -> ChEMBL target id.

Nothing here is hardcoded beyond UniProt accessions and gene symbols (stable
identifiers). ChEMBL target ids are resolved at runtime against the API so the
registry cannot drift out of date silently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .util import LOG, Http

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


@dataclass(frozen=True)
class Target:
    gene: str
    uniprot: str
    role: str            # "primary" | "offtarget"
    aliases: tuple[str, ...] = ()
    note: str = ""


# Tier 1 primary target -------------------------------------------------------
TNK1 = Target(
    gene="TNK1", uniprot="Q13470", role="primary",
    aliases=("KOS1", "tyrosine kinase non-receptor 1"),
    note="No experimental crystal structure of the kinase domain as of build date.",
)

# Tier 2 off-target panel (8 kinases) ----------------------------------------
# TNK2/ACK1 is first on purpose: closest relative to TNK1, and the hardest
# selectivity problem in the project.
OFFTARGETS: tuple[Target, ...] = (
    Target("TNK2", "Q07912", "offtarget", ("ACK1", "ACK-1"),
           "Closest TNK1 relative; primary selectivity counter-screen."),
    Target("SRC",  "P12931", "offtarget", ("c-Src",)),
    Target("ABL1", "P00519", "offtarget", ("ABL",)),
    Target("LCK",  "P06239", "offtarget"),
    Target("PTK2", "Q05397", "offtarget", ("FAK", "FAK1")),
    Target("EGFR", "P00533", "offtarget", ("ERBB1", "HER1")),
    Target("KDR",  "P35968", "offtarget", ("VEGFR2", "FLK1")),
    Target("FLT3", "P36888", "offtarget"),
)

PANEL: tuple[Target, ...] = (TNK1, *OFFTARGETS)

ACCESSION_TO_GENE: dict[str, str] = {t.uniprot: t.gene for t in PANEL}


# ---------------------------------------------------------------------------- #
# ChEMBL resolution
# ---------------------------------------------------------------------------- #
def resolve_chembl_targets(http: Http, accessions: list[str],
                           cache: Path | None = None) -> dict[str, list[dict]]:
    """UniProt accession -> list of ChEMBL targets containing that component.

    We keep every matching target (SINGLE PROTEIN, PROTEIN COMPLEX, mutants)
    and let the caller decide; by default tier 1/2 use SINGLE PROTEIN only, so
    complex/chimera assays don't quietly contaminate a single-target label.
    """
    if cache and cache.exists():
        data = json.loads(cache.read_text())
        if all(a in data for a in accessions):
            return {a: data[a] for a in accessions}

    out: dict[str, list[dict]] = {}
    for acc in accessions:
        payload = http.json(
            f"{CHEMBL_API}/target.json",
            params={"target_components__accession": acc, "limit": 200},
        )
        hits = [
            {
                "target_chembl_id": t["target_chembl_id"],
                "pref_name": t.get("pref_name"),
                "target_type": t.get("target_type"),
                "organism": t.get("organism"),
                "tax_id": t.get("tax_id"),
            }
            for t in payload.get("targets", [])
        ]
        out[acc] = hits
        LOG.info("%s (%s): %d ChEMBL target(s), %d SINGLE PROTEIN",
                 ACCESSION_TO_GENE.get(acc, "?"), acc, len(hits),
                 sum(h["target_type"] == "SINGLE PROTEIN" for h in hits))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, indent=2))
    return out


def single_protein_ids(resolved: dict[str, list[dict]], acc: str,
                       human_only: bool = True) -> list[str]:
    ids = []
    for h in resolved.get(acc, []):
        if h["target_type"] != "SINGLE PROTEIN":
            continue
        if human_only and h.get("tax_id") not in (9606, "9606", None):
            continue
        ids.append(h["target_chembl_id"])
    return sorted(set(ids))
