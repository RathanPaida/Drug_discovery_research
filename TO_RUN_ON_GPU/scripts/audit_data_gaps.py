"""What data do we actually have, and what would each candidate source add? Offline.

    python scripts/audit_data_gaps.py --version v3
    python scripts/audit_data_gaps.py --version v3 --harvest data/raw/tnk1_data
    python scripts/audit_data_gaps.py --version v3 --json

MAKES NO NETWORK CALLS AND WRITES NOTHING unless `--json` is passed. It reads the frozen
table and, if present, the harvester's own per-source CSVs.

WHY THIS EXISTS
---------------
Every planning document in this repository states the binding constraint as "172 exact
TNK1 measurements", and that number is the basis for the Spearman standard error, the
Wilcoxon p-value floor, the conformal calibration warnings and the decision to prefer
classical models. Nothing recomputes it. It was quoted from a run in November 2026 and has
been copied forward since.

More to the point, "172" is one number where several are needed to plan data collection:

  * how many are EXACT versus censored -- a ">10 uM" row trains the censored-hinge term
    but cannot carry a pActivity label, and the two are counted together in most of the
    prose;
  * how many distinct COMPOUNDS those rows represent -- the frozen splits are keyed by
    `inchikey_block1`, so a compound measured under IC50 and Ki is one unit of training
    data for splitting purposes and two rows for counting purposes;
  * how many distinct SCAFFOLDS -- which is what actually bounds a scaffold-split
    evaluation, and the number that decides whether a 5-fold split is even meaningful;
  * which SOURCE each row came from -- because that is what tells you whether a candidate
    new source is already represented.

The last one is the planning question. Adding a source that overlaps what ChEMBL already
gave you buys little; adding one that covers chemistry nothing else does buys a lot, and
the only way to know in advance is to look at what is there.

READ THE OVERLAP SECTION CAREFULLY. A panel source contributing 300 TNK1 rows for 300
compounds already in the table is worth much less than its row count suggests -- it is a
second measurement of known chemistry, useful for noise estimation and nearly useless for
extending the applicability domain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.utils.common import load_config, read_table, verify_locked  # noqa: E402


# The candidate sources, with what each is expected to contribute. `rows` and `tnk1_rows`
# are ORDER-OF-MAGNITUDE expectations from the publications' own descriptions, not
# measurements, and the audit prints them as such -- a plan built on an unverified row
# count is how "8 of 11 sources" became a line in a report with no consequence attached.
CANDIDATES = {
    "klaeger": dict(
        paper="Klaeger et al., Science 358:eaan4368 (2017)",
        what="Kinobeads: 243 clinical kinase drugs x expressed kinome, Kd_app",
        expect_tnk1="UNKNOWN -- Kinobeads only detects kinases it pulls down, so TNK1 "
                    "coverage is not guaranteed. Check this first; it decides the value.",
        expect_rows="~10^4-10^5 drug-kinase pairs",
        route="manual: science.org supplementary tables",
        value="HIGHEST if TNK1 is covered: clinical chemistry, not tool compounds",
    ),
    "anastassiadis": dict(
        paper="Anastassiadis et al., Nat Biotechnol 29:1039 (2011)",
        what="178 inhibitors x 300 kinases, percent activity remaining",
        expect_tnk1="likely present (broad recombinant panel)",
        expect_rows="~53,000 pairs",
        route="manual: nature.com supplementary tables",
        value="SELECTIVITY signal, not potency -- a single-concentration percent readout "
              "cannot become an IC50. Targets the component the v3 run failed to move.",
    ),
    "karaman": dict(
        paper="Karaman et al., Nat Biotechnol 26:127 (2008)",
        what="38 inhibitors x 317 kinases, Ambit competition-binding Kd",
        expect_tnk1="likely present",
        expect_rows="~12,000 pairs",
        route="manual: nature.com supplementary tables",
        value="MODERATE: same platform as Davis 2011, which is already harvested, so "
              "expect heavy compound overlap",
    ),
    "papyrus": dict(
        paper="Bequignon et al., J Cheminform 15:3 (2023)",
        what="curated ChEMBL + ExCAPE-DB aggregation, standardised pChEMBL values",
        expect_tnk1="whatever ChEMBL has, re-curated",
        expect_rows="tens of millions overall; panel subset much smaller",
        route="manual: papyrus-scripts export or a release archive",
        value="MODERATE for new TNK1 rows, HIGH for a cleaner pan-kinome table than the "
              "current 14-source merge",
    ),
    "dtc": dict(
        paper="Drug Target Commons",
        what="community-curated bioactivity aggregation",
        expect_tnk1="unknown",
        expect_rows="millions overall",
        route="automatic -- FAILED on the 2026-10-26 harvest; URLs repaired, re-try",
        value="FREE: already coded, already scoped, just never succeeded",
    ),
    "kiba": dict(
        paper="Tang et al. 2014, via the DeepDTA distribution",
        what="integrated KIBA bioactivity scores",
        expect_tnk1="unknown",
        expect_rows="~118,000 pairs",
        route="automatic -- FAILED on the 2026-10-26 harvest; branch-name fix applied",
        value="LOW as labels (a KIBA score is not an IC50 and must not be converted), "
              "useful as pan-kinome structure",
    ),
    "metz": dict(
        paper="Metz et al., Nat Chem Biol 7:200 (2011)",
        what="3,858 compounds x 172 kinases, Ki",
        expect_tnk1="likely present",
        expect_rows="~150,000 pairs",
        route="automatic returned 0 ROWS (an empty search, not an absent dataset); "
              "assay-description search added, manual route documented",
        value="HIGH: real Ki values at scale, and the search that missed it was ours",
    ),
}


def summarise_frozen(df: pd.DataFrame, tnk1: str) -> dict:
    """The numbers every planning document quotes, recomputed from the table itself."""
    t = df[df.uniprot_id == tnk1]
    exact = t[t.censored == 0] if "censored" in t.columns else t
    cens = t[t.censored == 1] if "censored" in t.columns else t.iloc[:0]
    out = {
        "tnk1_rows_total": int(len(t)),
        "tnk1_rows_exact": int(len(exact)),
        "tnk1_rows_censored": int(len(cens)),
        "tnk1_compounds_exact": int(exact.inchikey_block1.nunique()) if len(exact) else 0,
        "tnk1_compounds_total": int(t.inchikey_block1.nunique()) if len(t) else 0,
        "tnk1_scaffolds_exact": (int(exact.scaffold.nunique())
                                 if len(exact) and "scaffold" in exact.columns else None),
        "kinome_rows_total": int(len(df)),
        "kinome_targets": int(df.uniprot_id.nunique()),
    }
    if len(exact):
        out["tnk1_pact_min"] = float(exact.pact.min())
        out["tnk1_pact_max"] = float(exact.pact.max())
        out["tnk1_pact_median"] = float(exact.pact.median())
        # Rows per compound: the gap between the row count and the splitting unit.
        per = exact.groupby("inchikey_block1").size()
        out["rows_per_compound_mean"] = float(per.mean())
        out["compounds_measured_more_than_once"] = int((per > 1).sum())
    if "standard_type" in t.columns:
        out["tnk1_by_assay_type"] = {str(k): int(v) for k, v
                                     in t.standard_type.value_counts().items()}
    for col in ("source", "data_source", "src"):
        if col in t.columns:
            out["tnk1_by_source"] = {str(k): int(v) for k, v
                                     in t[col].value_counts().items()}
            break
    return out


def summarise_harvest(harvest: Path, tnk1_gene="TNK1", tnk1_acc="Q13470") -> dict:
    """Per-source TNK1 and total row counts from the harvester's own CSV directory.

    Reads only the columns needed, in chunks, so a multi-gigabyte source CSV costs memory
    proportional to the chunk rather than the file.
    """
    csv_dir = harvest / "csv"
    if not csv_dir.exists():
        return {"available": False, "looked_in": str(csv_dir)}
    out = {"available": True, "sources": {}}
    for f in sorted(csv_dir.glob("*.csv")):
        src = f.stem.lower()
        total = tnk1_rows = tnk1_cpds = 0
        keys = set()
        err = None
        try:
            for ch in pd.read_csv(f, dtype=object, chunksize=200_000,
                                  usecols=lambda c: c in (
                                      "uniprot_accession", "target_name_normalized",
                                      "gene_symbol", "inchikey_standardized",
                                      "inchikey_source", "activity_type", "relation")):
                total += len(ch)
                m = pd.Series(False, index=ch.index)
                for col, want in (("uniprot_accession", tnk1_acc),
                                  ("target_name_normalized", tnk1_gene),
                                  ("gene_symbol", tnk1_gene)):
                    if col in ch.columns:
                        m |= ch[col].astype(str).str.upper() == want.upper()
                sub = ch[m]
                tnk1_rows += len(sub)
                for col in ("inchikey_standardized", "inchikey_source"):
                    if col in sub.columns:
                        keys.update(x[:14] for x in sub[col].dropna().astype(str)
                                    if len(x) >= 14)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        out["sources"][src] = {"total_rows": total, "tnk1_rows": tnk1_rows,
                               "tnk1_compounds": len(keys), "error": err}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--version", default="v3")
    ap.add_argument("--harvest", default=None,
                    help="the harvester's output dir (default: <raw>/tnk1_data)")
    ap.add_argument("--json", action="store_true",
                    help="also write <runs>/data_gap_audit.json")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    proc = Path(cfg["paths"]["processed"])
    raw = Path(cfg["paths"]["raw"])
    runs = Path(cfg["paths"]["runs"])
    tnk1 = cfg["targets"]["tnk1"]
    report: dict = {"version": a.version, "target": tnk1}

    print("=" * 78)
    print("1. WHAT THE FROZEN TABLE ACTUALLY CONTAINS")
    print("=" * 78)
    tbl = proc / f"kinome_long_{a.version}.parquet"
    if not tbl.exists():
        print(f"   {tbl} is absent -- cannot audit the frozen table.")
        print(f"   Build it:  python scripts/run_stage1_data.py --version {a.version}")
        report["frozen"] = {"available": False, "path": str(tbl)}
    else:
        sha = verify_locked(tbl)
        df = read_table(tbl)
        f = summarise_frozen(df, tnk1)
        report["frozen"] = {"available": True, "path": str(tbl), "sha256": sha, **f}
        print(f"   dataset {sha[:12]}   {tbl.name}")
        print()
        print(f"   TNK1 rows, total                 {f['tnk1_rows_total']:>8,}")
        print(f"        of which EXACT              {f['tnk1_rows_exact']:>8,}"
              f"   <- the number every plan quotes")
        print(f"        of which censored           {f['tnk1_rows_censored']:>8,}"
              f"   (trains the hinge, carries no pActivity label)")
        print(f"   distinct compounds (exact)       {f['tnk1_compounds_exact']:>8,}"
              f"   <- the unit the frozen splits use")
        if f.get("tnk1_scaffolds_exact") is not None:
            print(f"   distinct scaffolds (exact)       {f['tnk1_scaffolds_exact']:>8,}"
                  f"   <- what bounds a scaffold-split evaluation")
        print(f"   pan-kinome rows                  {f['kinome_rows_total']:>8,}"
              f"   over {f['kinome_targets']:,} targets")
        if "rows_per_compound_mean" in f:
            print()
            print(f"   rows per compound, mean          {f['rows_per_compound_mean']:>8.2f}")
            print(f"   compounds measured >1 time       {f['compounds_measured_more_than_once']:>8,}"
                  f"   (replicates: usable for a noise floor)")
        if "tnk1_pact_median" in f:
            print(f"   pActivity range                  "
                  f"{f['tnk1_pact_min']:.2f} .. {f['tnk1_pact_max']:.2f}"
                  f"  (median {f['tnk1_pact_median']:.2f})")
        if f.get("tnk1_by_assay_type"):
            print(f"   by assay type                    {f['tnk1_by_assay_type']}")
        if f.get("tnk1_by_source"):
            print(f"   by source                        {f['tnk1_by_source']}")
        else:
            print()
            print("   NOTE the frozen table carries no per-row source column, so this audit")
            print("        cannot say which harvest source each TNK1 row came from. That is")
            print("        the one thing most worth knowing before choosing a new source.")
            print("        Stage 1 should carry `source` through from the master table.")

        n = f["tnk1_rows_exact"]
        if n:
            print()
            print("   WHAT THIS N IMPLIES, so the planning numbers are not quoted from memory:")
            print(f"     a 5-fold scaffold split gives test folds of about {n // 5} rows")
            print(f"     Spearman standard error at n={n // 5} is roughly "
                  f"{1 / np.sqrt(max(n // 5 - 1, 1)):.3f}")
            print( "     -> differences below about "
                  f"{2 / np.sqrt(max(n // 5 - 1, 1)):.2f} Spearman are not distinguishable")
            print( "     the exact Wilcoxon floor over 5 folds is p >= 0.0625, so 5 folds")
            print( "       cannot reach p < 0.05 at all; 10 repeats x 5 folds can")

    print()
    print("=" * 78)
    print("2. WHAT THE HARVEST COLLECTED, PER SOURCE")
    print("=" * 78)
    hv = Path(a.harvest) if a.harvest else raw / "tnk1_data"
    h = summarise_harvest(hv)
    report["harvest"] = h
    if not h["available"]:
        print(f"   no harvester output at {h['looked_in']}")
        print( "   This audit then cannot tell which source supplied what. Run:")
        print( "     python scripts/tnk1_harvest.py --check      # endpoints + manual files")
        print( "     python scripts/tnk1_harvest.py              # the full harvest")
    else:
        print(f"   {hv}")
        print()
        print(f"   {'source':<16} {'total rows':>12} {'TNK1 rows':>10} {'TNK1 cpds':>10}")
        for src, v in sorted(h["sources"].items(),
                             key=lambda kv: -kv[1]["tnk1_rows"]):
            flag = "  ERROR" if v["error"] else ("  <- no TNK1" if v["tnk1_rows"] == 0 else "")
            print(f"   {src:<16} {v['total_rows']:>12,} {v['tnk1_rows']:>10,} "
                  f"{v['tnk1_compounds']:>10,}{flag}")
            if v["error"]:
                print(f"       {v['error']}")
        zero = [s for s, v in h["sources"].items() if v["total_rows"] == 0]
        if zero:
            print()
            print(f"   {len(zero)} source(s) produced NO rows: {', '.join(sorted(zero))}")
            print( "   An empty source CSV means one of three different things, and the")
            print( "   harvest report distinguishes them: the collector raised, the search")
            print( "   found nothing, or a manual file is missing. Check REPORT.txt.")

    print()
    print("=" * 78)
    print("3. CANDIDATE SOURCES -- what each would add, and how to get it")
    print("=" * 78)
    print("   Row counts below are ORDER-OF-MAGNITUDE EXPECTATIONS from each paper's own")
    print("   description. They are NOT measurements, and nothing in this repository has")
    print("   verified that any of these tables contains a TNK1 column.")
    report["candidates"] = CANDIDATES
    for name, c in CANDIDATES.items():
        have = h.get("sources", {}).get(name, {}) if h.get("available") else {}
        got = (f"HAVE {have['tnk1_rows']:,} TNK1 rows" if have.get("tnk1_rows")
               else ("collected, 0 TNK1 rows" if have.get("total_rows") else "not collected"))
        print()
        print(f"   {name.upper()}   [{got}]")
        print(f"     {c['paper']}")
        print(f"     content : {c['what']}")
        print(f"     TNK1    : {c['expect_tnk1']}")
        print(f"     size    : {c['expect_rows']}")
        print(f"     route   : {c['route']}")
        print(f"     value   : {c['value']}")

    print()
    print("=" * 78)
    print("4. THE THING NO SOURCE CAN SUPPLY")
    print("=" * 78)
    print("   Every source above is a RETROSPECTIVE measurement of chemistry someone else")
    print("   made. None of them tests a molecule this pipeline designed, so none of them")
    print("   can validate it. Stage 9 is a no-op because no wet-lab result exists.")
    print()
    print("   The 40-compound Cycle-01 contract is ready to send. Before commissioning it,")
    print("   two things need settling -- see docs/DATA_COLLECTION_PLAN.md section 4:")
    print("     1. TNK1-NOV-C01-012 sits at p(hERG) = 0.82, above the configured ceiling,")
    print("        and has needed a decision since 2026-11-09.")
    print("     2. The contract was selected BY THE CRITIC THAT LOST the paired")
    print("        comparison. Re-score it against the stacked critic first; that costs")
    print("        minutes and may reorder it.")

    if a.json:
        runs.mkdir(parents=True, exist_ok=True)
        out = runs / "data_gap_audit.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        print()
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
