"""Tier orchestration. Each `tierN()` is independently runnable and idempotent.

Dependency order (the CLI enforces it with --all, and each tier checks):

    tier4 (kinases.csv)  ->  tier3 (kinome extract, filtered to those accessions)
    tier1 + tier2        ->  tier9 (anchors, selectivity, MMPs)
    tier3                ->  tier9 (kinase-inhibitor corpus)
    tier1/2/3            ->  features
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import chembl, corpus, descriptors, features as feat, focused, otherdbs, retro
from . import structures as struct
from .admet import build_admet, herg_risk_note
from .curate import (Standardizer, add_scaffold_split, aggregate_replicates,
                     filter_measurable, harmonize_activity)
from .targets import OFFTARGETS, PANEL, TNK1, resolve_chembl_targets, single_protein_ids
from .util import LOG, Http, Layout, Manifest, write_parquet


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read_processed(layout: Layout, name: str) -> pd.DataFrame | None:
    for cand in (layout.processed / name,
                 (layout.processed / name).with_suffix(".csv.gz")):
        if cand.exists():
            return (pd.read_parquet(cand) if cand.suffix == ".parquet"
                    else pd.read_csv(cand))
    return None


def _panel_pull(http: Http, layout: Layout, targets, *,
                include_bindingdb: bool, include_pubchem: bool,
                include_dtc: bool) -> pd.DataFrame:
    """ChEMBL + BindingDB + PubChem + DTC for a set of targets, unified."""
    accs = [t.uniprot for t in targets]
    resolved = resolve_chembl_targets(
        http, accs, cache=layout.interim / "chembl_targets.json")

    frames: list[pd.DataFrame] = []
    for t in targets:
        tids = single_protein_ids(resolved, t.uniprot)
        if not tids:
            LOG.error("%s (%s): no human SINGLE PROTEIN ChEMBL target found",
                      t.gene, t.uniprot)
            continue
        LOG.info("--- ChEMBL pull: %s (%s) via %s", t.gene, t.uniprot, tids)
        df = chembl.fetch_activities_rest(http, tids)
        if df.empty:
            otherdbs.summarize(df, f"chembl_{t.gene}")
            continue
        conf = chembl.fetch_assay_confidence(http, df["assay_chembl_id"].tolist())
        if not conf.empty:
            df = df.merge(conf, on="assay_chembl_id", how="left")
        df = chembl.apply_quality_filters(df, min_confidence=chembl.MIN_CONFIDENCE)
        df["accession"] = t.uniprot
        df["target_gene"] = t.gene
        otherdbs.summarize(df, f"chembl_{t.gene}")
        frames.append(df)

    if include_bindingdb:
        bdb = otherdbs.bindingdb_by_uniprot(http, accs)
        if not bdb.empty:
            frames.append(bdb)
    if include_pubchem:
        pc = otherdbs.pubchem_by_accession(http, accs)
        if not pc.empty:
            frames.append(pc)
    if include_dtc:
        dtc = otherdbs.dtc_by_accession(http, accs)
        if not dtc.empty:
            frames.append(dtc)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    gene_of = {t.uniprot: t.gene for t in PANEL}
    mapped = df["accession"].map(gene_of)
    df["target_gene"] = (df["target_gene"].fillna(mapped)
                         if "target_gene" in df.columns else mapped)
    return df


def _curate_activity(df: pd.DataFrame, *, label: str,
                     drop_censored: bool = False) -> pd.DataFrame:
    std = Standardizer()
    df = std.apply(df, smiles_col="smiles")
    df = harmonize_activity(df)
    df = filter_measurable(df, label=label, drop_censored=drop_censored)
    agg = aggregate_replicates(df)
    return agg


# --------------------------------------------------------------------------- #
# tiers
# --------------------------------------------------------------------------- #
def tier4(http: Http, layout: Layout, manifest: Manifest, **kw) -> Path:
    """Kinase descriptors. Run FIRST -- it defines the kinome for tier 3."""
    LOG.info("=== TIER 4: kinase descriptors (KLIFS + UniProt) ===")
    return descriptors.build_kinases_csv(http, layout, manifest)


def tier1(http: Http, layout: Layout, manifest: Manifest, *,
          include_bindingdb: bool = True, include_pubchem: bool = True,
          include_dtc: bool = True, **kw) -> Path:
    """TNK1 activity -- the binding constraint on the whole project."""
    LOG.info("=== TIER 1: TNK1 bioactivity ===")
    raw = _panel_pull(http, layout, [TNK1],
                      include_bindingdb=include_bindingdb,
                      include_pubchem=include_pubchem,
                      include_dtc=include_dtc)
    if raw.empty:
        raise RuntimeError("tier 1 produced no rows -- this is the project's "
                           "binding constraint, do not continue silently")
    write_parquet(raw, layout.interim / "tnk1_raw.parquet", manifest,
                  tier="1", source="multi", notes="pre-curation union")
    agg = _curate_activity(raw, label="tnk1")
    agg = add_scaffold_split(agg)
    n_exact = int((~agg["censored"]).sum())
    n_cens = int(agg["censored"].sum())
    LOG.info("TIER 1 RESULT: %d points (%d exact, %d censored), %d unique "
             "scaffolds", len(agg), n_exact, n_cens,
             agg["murcko_scaffold"].nunique())
    if n_exact < 400:
        LOG.warning(
            "Only %d exact TNK1 dose-response points. A D-MPNN -> multi-task "
            "transfer -> fine-tune -> 5-member ensemble chain cannot be "
            "supported at this size; that is why the ensemble ranks at "
            "rho=0.264 while a Random Forest on ECFP gets 0.626. Report the RF "
            "as the production model until more assay data lands, and spend the "
            "assay budget on TNK1 dose-response, not on reward engineering.",
            n_exact)
    out = write_parquet(agg, layout.processed / "tnk1_curated.parquet", manifest,
                        tier="1", source="multi",
                        notes=f"{n_exact} exact + {n_cens} censored; "
                              f"scaffold split in `split`")
    return out


def tier2(http: Http, layout: Layout, manifest: Manifest, *,
          include_bindingdb: bool = True, include_pubchem: bool = True,
          include_dtc: bool = True, **kw) -> Path:
    """Off-target panel -- 8 kinases, for selectivity."""
    LOG.info("=== TIER 2: off-target panel (%s) ===",
             ", ".join(t.gene for t in OFFTARGETS))
    raw = _panel_pull(http, layout, list(OFFTARGETS),
                      include_bindingdb=include_bindingdb,
                      include_pubchem=include_pubchem,
                      include_dtc=include_dtc)
    if raw.empty:
        raise RuntimeError("tier 2 produced no rows")
    write_parquet(raw, layout.interim / "offtargets_raw.parquet", manifest,
                  tier="2", source="multi", notes="pre-curation union")
    agg = _curate_activity(raw, label="offtargets")
    per_gene = agg.groupby("target_gene").size().to_dict()
    LOG.info("TIER 2 RESULT: %d points; per target %s", len(agg), per_gene)
    if per_gene.get("TNK2", 0) < 200:
        LOG.warning("TNK2/ACK1 has only %d points. TNK2 selectivity is the "
                    "project's hardest medicinal-chemistry problem and it is "
                    "the off-target you can least afford to model badly.",
                    per_gene.get("TNK2", 0))
    return write_parquet(agg, layout.processed / "offtargets_curated.parquet",
                         manifest, tier="2", source="multi",
                         notes=f"panel of {len(OFFTARGETS)} kinases")


def tier3(http: Http, layout: Layout, manifest: Manifest, *,
          use_sqlite: bool = True, bindingdb_url: str | None = None,
          dtc_csv: Path | None = None, include_papyrus: bool = True,
          force: bool = False, **kw) -> Path:
    """Kinome-wide multi-task data, to pretrain through the TNK1 scarcity.

    Builds `kinome_measurements_v3.parquet`. There is deliberately no v2 path:
    the v2 file encoded "no value" as pActivity == 0.0 in 3,372,521 of 4,798,165
    rows (70.3%), which reads as an exact IC50 of 1 M. `filter_measurable`
    raises `PoisonedDataError` if it ever sees that signature again, so a v2
    file cannot be loaded by accident.
    """
    LOG.info("=== TIER 3: kinome-wide multi-task data ===")
    kin_csv = layout.raw / "kinases.csv"
    if not kin_csv.exists():
        LOG.info("kinases.csv absent -- running tier 4 first")
        kin_csv = tier4(http, layout, manifest)
    accs = descriptors.kinase_accessions(kin_csv)

    frames: list[pd.DataFrame] = []
    if use_sqlite:
        release = chembl.chembl_release(http)
        db = chembl.download_sqlite(http, layout, manifest, release, force=force)
        km = chembl.kinome_from_sqlite(db, accs)
        km = chembl.apply_quality_filters(km)
        otherdbs.summarize(km, "chembl_kinome")
        frames.append(km)

    tsv = otherdbs.bindingdb_full(http, layout, manifest, url=bindingdb_url,
                                  force=force)
    if tsv is not None:
        frames.append(otherdbs.bindingdb_subset(tsv, set(accs)))

    dtc = otherdbs.dtc_bulk(http, layout, manifest, set(accs), local_csv=dtc_csv)
    if not dtc.empty:
        frames.append(dtc)

    if include_papyrus:
        pap = otherdbs.papyrus(layout, accessions=set(accs))
        if not pap.empty:
            frames.append(pap)

    for name in ("DAVIS", "KIBA"):
        d = otherdbs.tdc_dti(name, layout)
        if not d.empty:
            write_parquet(d, layout.interim / f"{name.lower()}_long.parquet",
                          manifest, tier="3", source=f"tdc_{name.lower()}",
                          notes="kept separate: KIBA scores are not pActivity "
                                "and must not be pooled with IC50/Ki")
    mz = otherdbs.metz(http, layout)
    if not mz.empty:
        frames.append(mz)

    if not frames:
        raise RuntimeError("tier 3 produced no rows from any source")
    raw = pd.concat(frames, ignore_index=True)
    write_parquet(raw, layout.interim / "kinome_raw_union.parquet", manifest,
                  tier="3", source="multi", rows=len(raw),
                  notes="pre-curation union of all kinome sources")

    agg = _curate_activity(raw, label="kinome_v3")
    gene_of = pd.read_csv(kin_csv).set_index("uniprot")["gene"].to_dict()
    if "accession" in agg.columns:
        mapped = agg["accession"].map(gene_of)
        agg["target_gene"] = (agg["target_gene"].fillna(mapped)
                              if "target_gene" in agg.columns else mapped)
    LOG.info("TIER 3 RESULT: %d usable measurements over %d kinases, %d compounds",
             len(agg), agg.get("accession", pd.Series(dtype=str)).nunique(),
             agg["canonical_smiles"].nunique())
    return write_parquet(agg, layout.processed / "kinome_measurements_v3.parquet",
                         manifest, tier="3", source="multi",
                         notes="v3 ONLY. v2 had 70.3% rows with pActivity==0.0 "
                               "standing in for 'no value'; filter_measurable "
                               "raises on that signature.")


def tier5(http: Http, layout: Layout, manifest: Manifest, *,
          limit: int | None = None, with_moses: bool = False,
          force: bool = False, **kw) -> Path:
    """Unlabelled corpus for the RL prior + the novelty-scan fingerprints."""
    LOG.info("=== TIER 5: unlabelled corpus ===")
    release = chembl.chembl_release(http)
    chemreps = chembl.download_chemreps(http, layout, manifest, release,
                                        force=force)
    extra = []
    if with_moses:
        m = corpus.download_moses(http, layout, manifest, force=force)
        if m:
            extra.append(m)
    path, df = corpus.build_corpus(chemreps, layout, manifest,
                                   extra_smi=extra, limit=limit)
    sub = df[df["drug_like"]] if "drug_like" in df else df
    LOG.info("TIER 5 RESULT: %d structures total, %d drug-like (prior training "
             "set)", len(df), len(sub))
    corpus.build_fingerprints(df, layout, manifest)
    return path


def tier6(layout: Layout, manifest: Manifest, *, strict: bool = True, **kw) -> Path:
    """ADMET -- seven TDC endpoints. The reward fails closed on any missing one."""
    LOG.info("=== TIER 6: ADMET ===")
    out, cov = build_admet(layout, manifest, strict=strict)
    LOG.info("TIER 6 RESULT: %s", {k: v.get("rows") for k, v in
                                   cov["endpoints"].items()})
    LOG.warning("hERG note: %s", herg_risk_note(layout))
    return out


def tier7(http: Http, layout: Layout, manifest: Manifest, *,
          prepare_receptors: bool = True, force: bool = False, **kw) -> Path:
    """Structures -- AlphaFold TNK1 + the ACK1/TNK2 experimental ensemble."""
    LOG.info("=== TIER 7: structures ===")
    idx = struct.build_structure_set(http, layout, manifest, force=force)
    if prepare_receptors and not idx.empty:
        struct.prepare_all_receptors(idx, layout, manifest)
    LOG.info("TIER 7 RESULT: %d structures. Docking into an ENSEMBLE (not the "
             "single AlphaFold model) is the cheapest route to moving the "
             "r=0.19 2D/docking correlation before any TNK1 crystal exists.",
             len(idx))
    return layout.structures / "structure_index.csv"


def tier8(layout: Layout, manifest: Manifest, *, force: bool = False, **kw):
    """Retrosynthesis models (~1 GB)."""
    LOG.info("=== TIER 8: retrosynthesis ===")
    cfg = retro.download_aizynth(layout, manifest, force=force)
    if cfg is None:
        LOG.error("TIER 8 INCOMPLETE -- Cycle 01 ran no synthesisability screen "
                  "for exactly this reason. The BRICS fallback is a fragment "
                  "check, not a route search; do not report it as "
                  "synthesisability.")
    return cfg


def tier9(http: Http, layout: Layout, manifest: Manifest, **kw) -> dict:
    """Generative anchoring: data for designing around the known chemotypes."""
    LOG.info("=== TIER 9: generative anchoring around known TNK1 chemotypes ===")
    tnk1 = _read_processed(layout, "tnk1_curated.parquet")
    offt = _read_processed(layout, "offtargets_curated.parquet")
    if tnk1 is None:
        raise RuntimeError("tier 9 needs tnk1_curated.parquet -- run tier1 first")
    curated = pd.concat([x for x in (tnk1, offt) if x is not None],
                        ignore_index=True)
    kinome = _read_processed(layout, "kinome_measurements_v3.parquet")
    out = focused.build_focused(http, layout, manifest, curated=curated,
                               kinome=kinome)
    LOG.info("TIER 9 RESULT: %s", {k: Path(v).name for k, v in out.items()})
    return out


def build_features(layout: Layout, manifest: Manifest, **kw) -> dict:
    """The missing data/features/ assembly step."""
    LOG.info("=== FEATURES: data/processed -> data/features ===")
    return feat.build_features(layout, manifest)


def verify(layout: Layout, manifest: Manifest, **kw) -> int:
    LOG.info("=== VERIFY: manifest SHA-256 acceptance test ===")
    return feat.verify_features(layout, manifest)


def write_report(layout: Layout) -> Path:
    """Harvest report -- every source, its row count, and why it was empty."""
    rep = otherdbs.report_table()
    path = layout.logs / "harvest_report.csv"
    if rep.empty:
        LOG.info("no harvest reports collected")
        return path
    rep.to_csv(path, index=False)
    empties = rep[rep["status"] != "ok"]
    LOG.info("\n%s", rep.to_string(index=False))
    if len(empties):
        LOG.error("%d source(s) contributed nothing -- this is surfaced here "
                  "rather than passing silently:", len(empties))
        for _, r in empties.iterrows():
            LOG.error("  %-20s %-8s %s", r["source"], r["status"], r["detail"])
            if r["action"]:
                LOG.error("      -> %s", r["action"])
    return path
