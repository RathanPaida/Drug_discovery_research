"""Tier 9 -- data for generating *around* the known TNK1 chemotypes.

This is the half of the problem that the eight tiers above do not cover. With
172 exact TNK1 points you cannot learn the SAR de novo, so the generator has to
be anchored: pretrain the prior broadly, then pull it towards chemotypes that
are already known to engage the target or its closest relative.

The chain this module builds the data for:

    788k corpus prior  ->  kinase-inhibitor-like subset  ->  TNK1/TNK2 actives
    (tier 5)               (~100k, from tier 3)             (~hundreds, tier 1/2)

Nothing here is a hardcoded compound list. The anchors are *derived* from the
curated tier 1/2 data at run time, so the focused set cannot drift away from
what the assays actually say. Three expansion routes are built:

1. `nearest_neighbour_expansion` -- PubChem + ChEMBL 2D similarity around each
   anchor. Gives real, catalogued, often purchasable analogues: the honest
   baseline any generative model has to beat.
2. `matched_pairs` -- matched molecular pairs over the kinase bioactivity set,
   restricted to transformations seen on the anchor scaffolds. This is where
   "which R-group swap buys potency without buying hERG" actually lives.
3. `kinase_inhibitor_subset` -- the intermediate fine-tuning corpus: every
   compound in the tier 3 extract with a measured kinase activity, which is a
   far better generator prior than drug-like ChEMBL at large.

Selectivity, which is the project's hardest problem, is handled by
`selectivity_pairs`: compounds with measurements on BOTH TNK1 and TNK2, with
the log-ratio as the label. That frame is small but it is the only data that
directly supervises the thing you are being asked to optimise. Generating
against predicted TNK1 potency alone will walk straight into TNK2.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .util import LOG, Http, Layout, Manifest, write_parquet

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


# --------------------------------------------------------------------------- #
# anchors
# --------------------------------------------------------------------------- #
def select_anchors(curated: pd.DataFrame, *,
                   primary_gene: str = "TNK1",
                   relative_gene: str = "TNK2",
                   p_cut_primary: float = 6.0,
                   p_cut_relative: float = 7.0,
                   max_anchors: int = 250) -> pd.DataFrame:
    """Derive the anchor set from curated activity, not from a hardcoded list.

    Anchors = confirmed TNK1 actives (uncensored, p >= 6) plus potent TNK2/ACK1
    actives. The TNK2 compounds are included deliberately: they are the richest
    available source of chemotypes that bind a TNK-family pocket, and the
    project's selectivity problem means you need to be able to *move away* from
    them in a controlled way, which requires having them in the training data.
    """
    df = curated.copy()
    if "censored" in df:
        df = df[~df["censored"].astype(bool)]
    gene_col = "target_gene" if "target_gene" in df else "gene"
    prim = df[(df[gene_col] == primary_gene) & (df["p_activity"] >= p_cut_primary)]
    rel = df[(df[gene_col] == relative_gene) & (df["p_activity"] >= p_cut_relative)]
    prim = prim.assign(anchor_role="tnk1_active")
    rel = rel.assign(anchor_role="tnk2_active")
    out = (pd.concat([prim, rel], ignore_index=True)
           .sort_values("p_activity", ascending=False)
           .drop_duplicates(subset=["canonical_smiles"])
           .head(max_anchors))
    LOG.info("anchors: %d (%d TNK1 p>=%.1f, %d TNK2 p>=%.1f)",
             len(out), int((out["anchor_role"] == "tnk1_active").sum()),
             p_cut_primary, int((out["anchor_role"] == "tnk2_active").sum()),
             p_cut_relative)
    if out.empty:
        LOG.error("no anchors found -- run tier1/tier2 first, or lower the "
                  "p_activity cutoffs")
    return out


# --------------------------------------------------------------------------- #
# 1. similarity expansion
# --------------------------------------------------------------------------- #
def pubchem_similar(http: Http, smiles: str, *, threshold: int = 85,
                    max_records: int = 500) -> list[int]:
    """PubChem 2D similarity search (POST, so long SMILES are safe)."""
    url = f"{PUG}/compound/fastsimilarity_2d/smiles/cids/JSON"
    try:
        r = http.s.post(url, data={"smiles": smiles,
                                   "Threshold": threshold,
                                   "MaxRecords": max_records}, timeout=180)
        r.raise_for_status()
        return r.json().get("IdentifierList", {}).get("CID", [])
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("PubChem similarity failed: %s", exc)
        return []


def chembl_similar(http: Http, smiles: str, *, threshold: int = 80,
                   limit: int = 200) -> list[dict]:
    """ChEMBL similarity search -- returns compounds already in ChEMBL, so many
    come with their own bioactivity records (free labels, not just structures).
    """
    try:
        js = http.json(f"{CHEMBL_API}/similarity/{smiles}/{threshold}.json",
                       params={"limit": limit})
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("ChEMBL similarity failed: %s", exc)
        return []
    return [{"molecule_chembl_id": m.get("molecule_chembl_id"),
             "smiles": (m.get("molecule_structures") or {}).get("canonical_smiles"),
             "similarity": float(m.get("similarity", 0) or 0)}
            for m in js.get("molecules", [])]


def nearest_neighbour_expansion(http: Http, anchors: pd.DataFrame, *,
                                pubchem_threshold: int = 85,
                                chembl_threshold: int = 80,
                                per_anchor: int = 300,
                                resolve_pubchem_smiles: bool = True
                                ) -> pd.DataFrame:
    rows: list[dict] = []
    smis = anchors["canonical_smiles"].dropna().unique().tolist()
    LOG.info("similarity expansion around %d anchors", len(smis))
    for i, smi in enumerate(smis, 1):
        for hit in chembl_similar(http, smi, threshold=chembl_threshold,
                                  limit=per_anchor):
            if hit["smiles"]:
                rows.append({"smiles": hit["smiles"], "anchor": smi,
                             "similarity": hit["similarity"] / 100.0
                             if hit["similarity"] > 1 else hit["similarity"],
                             "neighbour_source": "chembl",
                             "molecule_chembl_id": hit["molecule_chembl_id"]})
        cids = pubchem_similar(http, smi, threshold=pubchem_threshold,
                               max_records=per_anchor)
        for cid in cids:
            rows.append({"smiles": None, "cid": cid, "anchor": smi,
                         "similarity": None, "neighbour_source": "pubchem"})
        if i % 10 == 0:
            LOG.info("  expanded %d/%d anchors -> %d raw neighbours",
                     i, len(smis), len(rows))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if resolve_pubchem_smiles and "cid" in df.columns:
        from .otherdbs import _attach_pubchem_smiles                # noqa: PLC0415
        need = df[df["smiles"].isna() & df["cid"].notna()].copy()
        if len(need):
            resolved = _attach_pubchem_smiles(http, need)
            df = pd.concat([df[df["smiles"].notna()], resolved], ignore_index=True)
    df = df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles", "anchor"])
    LOG.info("similarity expansion: %d unique neighbour structures",
             df["smiles"].nunique())
    return df


# --------------------------------------------------------------------------- #
# 2. matched molecular pairs
# --------------------------------------------------------------------------- #
def matched_pairs(df: pd.DataFrame, *, smiles_col: str = "canonical_smiles",
                  value_col: str = "p_activity",
                  max_cuts: int = 1, min_core_atoms: int = 8,
                  max_pairs: int = 200_000) -> pd.DataFrame:
    """Single-cut matched molecular pairs with the activity delta.

    One cut only, by default: multi-cut MMPs explode combinatorially and the
    extra pairs are mostly not interpretable as a single R-group change, which
    is the whole point of looking at them.
    """
    from rdkit import Chem                                          # noqa: PLC0415
    from rdkit.Chem import rdMMPA                                   # noqa: PLC0415

    index: dict[str, list[tuple[str, float, str]]] = {}
    for smi, val in zip(df[smiles_col], df[value_col]):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None or not np.isfinite(val or np.nan):
            continue
        try:
            frags = rdMMPA.FragmentMol(mol, maxCuts=max_cuts, resultsAsMols=False)
        except Exception:                                          # noqa: BLE001
            continue
        for core, chains in frags:
            core = core or chains
            if not core:
                continue
            core_mol = Chem.MolFromSmiles(core.replace("[*:1]", "[H]"))
            if core_mol is None or core_mol.GetNumHeavyAtoms() < min_core_atoms:
                continue
            index.setdefault(core, []).append((chains or "", float(val), smi))

    rows: list[dict] = []
    for core, members in index.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                (ri, vi, si), (rj, vj, sj) = members[i], members[j]
                if ri == rj:
                    continue
                rows.append({"core": core, "r_from": ri, "r_to": rj,
                             "smiles_from": si, "smiles_to": sj,
                             "delta": vj - vi})
                if len(rows) >= max_pairs:
                    break
            if len(rows) >= max_pairs:
                break
        if len(rows) >= max_pairs:
            LOG.warning("matched_pairs truncated at %d pairs", max_pairs)
            break
    out = pd.DataFrame(rows)
    if len(out):
        summary = (out.groupby(["r_from", "r_to"])["delta"]
                   .agg(["mean", "std", "count"])
                   .query("count >= 3").sort_values("mean", ascending=False))
        LOG.info("matched pairs: %d pairs, %d transformations seen >=3 times; "
                 "best mean delta %.2f log", len(out), len(summary),
                 summary["mean"].iloc[0] if len(summary) else float("nan"))
    return out


# --------------------------------------------------------------------------- #
# 3. kinase-inhibitor fine-tuning corpus
# --------------------------------------------------------------------------- #
def kinase_inhibitor_subset(kinome: pd.DataFrame, *,
                            p_cut: float = 6.0,
                            min_targets: int = 1) -> pd.DataFrame:
    """Every compound with a measured kinase activity above `p_cut`.

    This is the intermediate fine-tuning stage that makes the 788k -> TNK1
    transfer survivable. Pretraining on drug-like ChEMBL and then fine-tuning
    straight onto a few hundred TNK1 actives is a 4-order-of-magnitude jump in
    data size; the kinase subset sits in between and is the step that was
    missing from the D-MPNN -> multi-task -> fine-tune -> ensemble chain.
    """
    df = kinome[~kinome.get("censored", False).astype(bool)] \
        if "censored" in kinome else kinome
    hot = df[df["p_activity"] >= p_cut]
    per_cmpd = (hot.groupby("canonical_smiles")
                .agg(n_targets=("accession", "nunique"),
                     max_p=("p_activity", "max"),
                     median_p=("p_activity", "median"))
                .reset_index())
    per_cmpd = per_cmpd[per_cmpd["n_targets"] >= min_targets]
    per_cmpd["promiscuity"] = per_cmpd["n_targets"]
    LOG.info("kinase-inhibitor corpus: %d compounds (p>=%.1f on >=%d kinases); "
             "median promiscuity %.1f targets", len(per_cmpd), p_cut, min_targets,
             per_cmpd["n_targets"].median() if len(per_cmpd) else float("nan"))
    return per_cmpd


# --------------------------------------------------------------------------- #
# selectivity supervision
# --------------------------------------------------------------------------- #
def selectivity_pairs(curated: pd.DataFrame, *, primary: str = "TNK1",
                      against: tuple[str, ...] = ("TNK2", "SRC", "ABL1", "LCK",
                                                  "PTK2", "EGFR", "KDR", "FLT3")
                      ) -> pd.DataFrame:
    """One row per compound with a TNK1 measurement, wide over the panel.

    `sel_TNK2 = p_TNK1 - p_TNK2` is the label for the project's hardest problem.
    Censored points are kept as bounds: a compound at TNK1 p=8.0 and TNK2 '<5'
    has selectivity *at least* 3 logs, which is usable supervision, and throwing
    those rows away is how a small selectivity set becomes an empty one.
    """
    gene_col = "target_gene" if "target_gene" in curated else "gene"
    key = "inchikey" if "inchikey" in curated else "canonical_smiles"
    piv = curated.pivot_table(index=[key, "canonical_smiles"], columns=gene_col,
                              values="p_activity", aggfunc="median")
    cen = curated.pivot_table(index=[key, "canonical_smiles"], columns=gene_col,
                              values="censored", aggfunc="max")
    piv = piv.reset_index()
    if primary not in piv.columns:
        LOG.error("no %s column after pivot -- nothing to compute selectivity on",
                  primary)
        return pd.DataFrame()
    out = piv[piv[primary].notna()].copy()
    for g in against:
        if g in out.columns:
            out[f"sel_{g}"] = out[primary] - out[g]
            n = int(out[f"sel_{g}"].notna().sum())
            bounded = int(cen.get(g, pd.Series(dtype=bool)).fillna(False).sum()) \
                if g in cen.columns else 0
            LOG.info("selectivity vs %-5s: %4d compounds with both values "
                     "(%d of the off-target values are censored bounds)",
                     g, n, bounded)
    LOG.info("selectivity frame: %d compounds with a %s measurement", len(out),
             primary)
    return out


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def build_focused(http: Http, layout: Layout, manifest: Manifest, *,
                  curated: pd.DataFrame,
                  kinome: pd.DataFrame | None = None,
                  do_expansion: bool = True,
                  do_mmp: bool = True) -> dict[str, Path]:
    from .curate import Standardizer, murcko_scaffold                # noqa: PLC0415
    out: dict[str, Path] = {}

    anchors = select_anchors(curated)
    if not anchors.empty:
        anchors = anchors.assign(
            murcko_scaffold=anchors["canonical_smiles"].map(murcko_scaffold))
        out["anchors"] = write_parquet(
            anchors, layout.processed / "generative_anchors.parquet", manifest,
            tier="9", source="derived",
            notes="confirmed TNK1 + potent TNK2 actives; the chemotypes the "
                  "generator is steered towards")
        LOG.info("anchor scaffold census: %d unique Murcko scaffolds over %d "
                 "anchors", anchors["murcko_scaffold"].nunique(), len(anchors))

    sel = selectivity_pairs(curated)
    if not sel.empty:
        out["selectivity"] = write_parquet(
            sel, layout.processed / "selectivity_matrix.parquet", manifest,
            tier="9", source="derived",
            notes="p_activity per panel kinase, wide; sel_<GENE> = p_TNK1 - "
                  "p_<GENE>. The only direct supervision for selectivity.")

    if do_expansion and not anchors.empty:
        nn = nearest_neighbour_expansion(http, anchors)
        if not nn.empty:
            std = Standardizer()
            nn = std.apply(nn, smiles_col="smiles")
            nn = nn.drop_duplicates(subset=["canonical_smiles", "anchor"])
            known = set(curated["canonical_smiles"].dropna())
            nn["already_tested"] = nn["canonical_smiles"].isin(known)
            LOG.info("expansion: %d structures, %d already have panel data "
                     "(those are free validation, not new ideas)",
                     nn["canonical_smiles"].nunique(),
                     int(nn["already_tested"].sum()))
            out["neighbours"] = write_parquet(
                nn, layout.processed / "anchor_neighbours.parquet", manifest,
                tier="9", source="chembl+pubchem_similarity",
                notes="catalogued analogues of the anchors; the baseline any "
                      "generated molecule must beat")

    if do_mmp:
        mmp_src = curated[curated.get("target_gene", "") == "TNK1"] \
            if "target_gene" in curated else curated
        if len(mmp_src) >= 20:
            pairs = matched_pairs(mmp_src)
            if not pairs.empty:
                out["mmp"] = write_parquet(
                    pairs, layout.processed / "matched_pairs_tnk1.parquet",
                    manifest, tier="9", source="derived",
                    notes="single-cut MMPs with p_activity delta; R-group "
                          "transformations with their observed effect")
        else:
            LOG.warning("only %d TNK1 rows -- too few for a useful MMP table",
                        len(mmp_src))

    if kinome is not None and not kinome.empty:
        ki = kinase_inhibitor_subset(kinome)
        if not ki.empty:
            out["kinase_corpus"] = write_parquet(
                ki, layout.processed / "kinase_inhibitor_corpus.parquet",
                manifest, tier="9", source="derived",
                notes="intermediate fine-tuning corpus between the 788k prior "
                      "and the TNK1 actives; `promiscuity` is a usable "
                      "selectivity-risk prior")
    return out
