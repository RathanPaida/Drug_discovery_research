#!/usr/bin/env python3
"""
Adds predicted TNK1 pIC50 score and Protein-Protein Interaction Disruptor (PPI/PPID) score
to both the 12,000 generated candidate molecules and the top-40 wet-lab contract compounds.

Files updated:
  1. tnk1-pipeline/tnk1-pipeline/runs/stage7_scored_candidates.csv (12,000 candidates)
  2. tnk1-pipeline/tnk1-pipeline/runs/cycle_01/contract_to_team_b.csv (top 40 prioritized compounds)

New columns added:
  - predicted_pic50: Predicted TNK1 logarithmic potency (-log10 IC50)
  - pred_ic50_nm: Predicted IC50 in nanomolar (10^(9 - pIC50))
  - ppid_score: Quantitative Estimate of Protein-Protein Interaction Targeting Drug-Likeness (QEPPI continuous score [0, 1])
  - ppi_score: Alias for ppid_score
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from rdkit import Chem
import QEPPI as ppi

def main():
    root = Path(__file__).resolve().parent.parent
    candidates_path = root / "tnk1-pipeline" / "tnk1-pipeline" / "runs" / "stage7_scored_candidates.csv"
    contract_path = root / "tnk1-pipeline" / "tnk1-pipeline" / "runs" / "cycle_01" / "contract_to_team_b.csv"

    print("=" * 70)
    print("ADDING PREDICTED pIC50 & PPID/PPI SCORES TO MOLECULAR DATASETS")
    print("=" * 70)

    # Initialize QEPPI calculator
    print("\nInitializing QEPPI calculator with ADS desirability parameters...")
    q = ppi.QEPPI_Calculator()
    q.read()

    def compute_ppid_scores(smiles_list):
        scores = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    score = float(q.qeppi(mol))
                    scores.append(round(score, 6))
                else:
                    scores.append(0.0)
            except Exception:
                scores.append(0.0)
        return scores

    # 1. Process 12,000 candidate molecules
    print(f"\n1. Processing 12,000 generated candidates from:\n   {candidates_path}")
    t0 = time.time()
    df_12k = pd.read_csv(candidates_path)
    n_12k = len(df_12k)
    print(f"   Loaded {n_12k:,} rows.")

    df_12k["predicted_pic50"] = df_12k["mu_tnk1"].round(6)
    df_12k["pred_ic50_nm"] = (10 ** (9.0 - df_12k["predicted_pic50"])).round(2)
    
    print("   Computing PPID/PPI scores via QEPPI for all 12,000 molecules...")
    ppid_12k = compute_ppid_scores(df_12k["smiles"])
    df_12k["ppid_score"] = ppid_12k
    df_12k["ppi_score"] = ppid_12k

    # Reorder columns so new scores are prominent
    cols_12k = list(df_12k.columns)
    # Move predicted_pic50, pred_ic50_nm, ppid_score, ppi_score after mu_tnk1
    target_pos = cols_12k.index("mu_tnk1") + 1
    new_cols = ["predicted_pic50", "pred_ic50_nm", "ppid_score", "ppi_score"]
    remaining = [c for c in cols_12k if c not in new_cols]
    ordered_cols_12k = remaining[:target_pos] + new_cols + remaining[target_pos:]
    df_12k = df_12k[ordered_cols_12k]

    df_12k.to_csv(candidates_path, index=False)
    print(f"   Saved updated 12,000 candidates to {candidates_path} in {time.time()-t0:.2f}s")
    print(f"   PPID Score stats (12k): mean={df_12k['ppid_score'].mean():.4f}, "
          f"min={df_12k['ppid_score'].min():.4f}, max={df_12k['ppid_score'].max():.4f}")

    # 2. Process Top-40 contract molecules
    print(f"\n2. Processing top-40 wet-lab contract molecules from:\n   {contract_path}")
    t1 = time.time()
    df_40 = pd.read_csv(contract_path)
    n_40 = len(df_40)
    print(f"   Loaded {n_40:,} rows.")

    df_40["predicted_pic50"] = df_40["pred_mu_tnk1"].round(6)
    df_40["pred_ic50_nm"] = (10 ** (9.0 - df_40["predicted_pic50"])).round(2)
    
    print("   Computing PPID/PPI scores via QEPPI for top 40 molecules...")
    ppid_40 = compute_ppid_scores(df_40["canonical_smiles"])
    df_40["ppid_score"] = ppid_40
    df_40["ppi_score"] = ppid_40

    # Reorder columns
    cols_40 = list(df_40.columns)
    target_pos_40 = cols_40.index("pred_mu_tnk1") + 1
    remaining_40 = [c for c in cols_40 if c not in new_cols]
    ordered_cols_40 = remaining_40[:target_pos_40] + new_cols + remaining_40[target_pos_40:]
    df_40 = df_40[ordered_cols_40]

    df_40.to_csv(contract_path, index=False)
    print(f"   Saved updated 40 candidates to {contract_path} in {time.time()-t1:.2f}s")
    print(f"   PPID Score stats (top 40): mean={df_40['ppid_score'].mean():.4f}, "
          f"min={df_40['ppid_score'].min():.4f}, max={df_40['ppid_score'].max():.4f}")

    print("\n" + "=" * 70)
    print("SAMPLE OF TOP 10 PRIORITIZED CONTRACT MOLECULES WITH NEW COLUMNS:")
    print("=" * 70)
    display_cols = ["compound_id", "batch_slice", "predicted_pic50", "pred_ic50_nm", "ppid_score", "pred_ci90_low", "pred_ci90_high"]
    print(df_40[display_cols].head(10).to_string(index=False))
    print("=" * 70)

if __name__ == "__main__":
    main()
