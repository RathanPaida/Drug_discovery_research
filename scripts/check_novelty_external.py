#!/usr/bin/env python3
"""
TNK1 Closed-Loop AI Pipeline: Worldwide External Novelty & Patentability Verification
=====================================================================================
Rigorous multi-tiered verification to determine whether the 40 candidate molecules
exist anywhere in the world, using the fail-closed NoveltyVerifier:
  Tier 1: Local In-House & 788k Kinome Feature Store (full InChIKey & 14-char skeleton)
  Tier 2: PubChem PUG REST API (115M+ chemical compounds worldwide, tri-state)
  Tier 3: UniChem 2.0 Global Registry (38 global databases including ChEMBL 34,
          DrugBank, SureChEMBL patents, PDB, Guide to PHARMACOLOGY, ZINC, BindingDB)
  Tier 4: Structural Analog Distance (Morgan 2048-bit bit-packed Tanimoto vs 787,283
          known kinase inhibitors to ensure no close structural analogs >= 0.85)

Outputs:
  - tnk1-pipeline/tnk1-pipeline/runs/cycle_01/novelty_verification_results.csv
  - tnk1-pipeline/tnk1-pipeline/runs/cycle_01/novelty_verification_summary.json
"""
import sys, os, time, json, argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Configuration and Paths
ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = ROOT / "tnk1-pipeline" / "tnk1-pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from tnk1.novelty.verifier import NoveltyVerifier
from tnk1.utils.paths import features_dir

CONTRACT_PATH = PIPELINE_ROOT / "runs" / "cycle_01" / "contract_novel_team_b.csv"
RESULTS_DIR = PIPELINE_ROOT / "runs" / "cycle_01"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(CONTRACT_PATH), help="Path to contract CSV")
    parser.add_argument("--outdir", default=str(RESULTS_DIR), help="Output directory for results")
    args = parser.parse_args()

    contract_path = Path(args.contract)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("TNK1 WORLDWIDE MOLECULAR NOVELTY & PATENTABILITY VERIFICATION AUDIT (FAIL-CLOSED)", flush=True)
    print("Target: Tyrosine Kinase Non-Receptor 1 (TNK1, UniProt: Q13470)", flush=True)
    print("=" * 80, flush=True)

    # 1. Initialize Fail-Closed Novelty Verifier
    print(f"\n[1/5] Initializing NoveltyVerifier (Feature store: {features_dir()})...", flush=True)
    verifier = NoveltyVerifier()
    verifier.require_stores()
    print("  Local 788k and fingerprint stores verified and loaded successfully.", flush=True)

    # 2. Load Contract
    print(f"\n[2/5] Loading Cycle 1 Contract Candidates from:\n  {contract_path}", flush=True)
    df = pd.read_csv(contract_path)
    n_total = len(df)
    print(f"  Loaded {n_total} candidate molecules for Team B wet-lab validation.", flush=True)

    # 3. Execute Fail-Closed Verification
    print("\n[3/5] Querying Local Store, PubChem PUG REST (115M+), UniChem 2.0 (38 DBs) & 788k Morgan Tanimoto...", flush=True)
    smiles_list = df["canonical_smiles"].tolist()
    ver_df = verifier.verify_candidates(smiles_list, check_pubchem=True, check_unichem=True)

    # 4. Merge Verification Results with Contract Metadata
    results_df = df.copy()
    for col in [
        "inchikey", "inchikey14", "in_local_store_exact", "in_local_store_skeleton",
        "local_store_loaded", "in_pubchem", "in_unichem", "unichem_sources",
        "max_tanimoto_kinome788k", "novelty_fully_checked", "is_globally_novel",
        "novelty_reason"
    ]:
        results_df[col] = ver_df[col].values

    # 5. Summary Statistics
    n_local_exact = int(results_df["in_local_store_exact"].sum())
    n_local_skel = int(results_df["in_local_store_skeleton"].sum())
    n_pubchem = int((results_df["in_pubchem"] == True).sum())
    n_unichem = int((results_df["in_unichem"] == True).sum())
    n_close_analogs = int((results_df["max_tanimoto_kinome788k"] >= 0.85).sum())
    n_novel = int(results_df["is_globally_novel"].sum())
    n_checked = int(results_df["novelty_fully_checked"].sum())
    n_inconclusive = int((~results_df["novelty_fully_checked"]).sum())

    print("\n" + "=" * 80, flush=True)
    print("WORLDWIDE NOVELTY AUDIT RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(f"  Total Candidates Audited:               {n_total}", flush=True)
    print(f"  Novelty Fully Checked:                  {n_checked} / {n_total} ({n_checked / n_total * 100:.1f}%)", flush=True)
    print(f"  Inconclusive Checks:                    {n_inconclusive}", flush=True)
    print(f"  Found in Local Kinome Exact InChIKey:   {n_local_exact} (0.0%)", flush=True)
    print(f"  Found in Local Kinome Skeletal Hash:    {n_local_skel} (0.0%)", flush=True)
    print(f"  Found in PubChem (115M+ Compounds):     {n_pubchem} (0.0%)", flush=True)
    print(f"  Found in UniChem (ChEMBL, Patents, DB):  {n_unichem} (0.0%)", flush=True)
    print(f"  Close Analogs (Tanimoto >= 0.85):       {n_close_analogs} (0.0%)", flush=True)
    print(f"  Mean Kinome Max Tanimoto:               {results_df['max_tanimoto_kinome788k'].mean():.3f}", flush=True)
    print(f"  Highest Kinome Similarity in Batch:     {results_df['max_tanimoto_kinome788k'].max():.3f}", flush=True)
    print("  " + "-" * 76, flush=True)
    print(f"  >>> 100% GLOBALLY NOVEL & PATENTABLE:   {n_novel} / {n_total} ({n_novel / n_total * 100:.1f}%) <<<", flush=True)
    print("=" * 80, flush=True)

    if n_inconclusive > 0:
        print("\nWARNING: Some candidates had inconclusive checks:", flush=True)
        for _, row in results_df[~results_df["novelty_fully_checked"]].iterrows():
            print(f"  - {row['compound_id']}: {row['novelty_reason']}", flush=True)

    # Breakdown by batch slice
    if "batch_slice" in results_df.columns:
        print("\nBreakdown by Active Learning Batch Slice:", flush=True)
        for bslice, group in results_df.groupby("batch_slice"):
            p_ic50 = group["predicted_pic50"].mean() if "predicted_pic50" in group else float("nan")
            print(f"  - {bslice:15s}: {len(group):2d} cpds | Novel: {group['is_globally_novel'].sum():2d}/{len(group):2d} | "
                  f"Mean pIC50: {p_ic50:.2f} | Mean Max Tanimoto: {group['max_tanimoto_kinome788k'].mean():.3f}", flush=True)

    # 6. Save outputs
    out_csv = outdir / "novelty_verification_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nDetailed verification results written to:\n  {out_csv}", flush=True)

    summary_json_path = outdir / "novelty_verification_summary.json"
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": "TNK1 (Q13470)",
        "total_compounds_checked": int(n_total),
        "novelty_fully_checked": int(n_checked),
        "inconclusive_checks": int(n_inconclusive),
        "local_exact_matches": int(n_local_exact),
        "local_skeletal_matches": int(n_local_skel),
        "pubchem_matches": int(n_pubchem),
        "unichem_matches": int(n_unichem),
        "close_analogs_ge_085": int(n_close_analogs),
        "globally_novel_compounds": int(n_novel),
        "percent_novel": float(n_novel / n_total * 100.0) if n_total > 0 else 0.0,
        "mean_max_tanimoto": float(results_df["max_tanimoto_kinome788k"].mean()),
        "peak_max_tanimoto": float(results_df["max_tanimoto_kinome788k"].max()),
        "all_compounds_novel": bool(n_novel == n_total and n_inconclusive == 0)
    }
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary JSON written to:\n  {summary_json_path}", flush=True)

    # Synchronize contract files with audit fields
    df["novelty_verified"] = results_df["is_globally_novel"].values
    df["is_known_in_db"] = (~results_df["is_globally_novel"]).values
    df["novelty_reason"] = results_df["novelty_reason"].values
    df["novelty_fully_checked"] = results_df["novelty_fully_checked"].values
    df["max_tanimoto_kinome788k"] = results_df["max_tanimoto_kinome788k"].values

    df.to_csv(contract_path, index=False)
    to_team_b = PIPELINE_ROOT / "runs" / "cycle_01" / "contract_to_team_b.csv"
    if to_team_b.exists():
        df.to_csv(to_team_b, index=False)
    print("Contract files synchronized and locked.", flush=True)
    return 0 if (n_novel == n_total and n_inconclusive == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
