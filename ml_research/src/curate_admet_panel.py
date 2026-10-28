#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Task ADMET Predictor Suite Data Curation (Stage 5.2 Specification)
Curates 7 core endpoints from Therapeutics Data Commons (TDC):
1. AqSolDB (aqueous solubility, regression)
2. Caco2_Wang (intestinal permeability, regression)
3. Clearance_Microsome_AZ (microsomal clearance, regression)
4. CYP3A4_Veith (DDI liability, binary classification)
5. CYP2D6_Veith (DDI liability, binary classification)
6. hERG_Karim (cardiotoxicity, binary classification)
7. AMES (mutagenicity, binary classification)
Standardizes via Step 1.2 and saves unified multi-task dataset.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from tdc.single_pred import ADME, Tox
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.standardizer import ChemicalStandardizer

ENDPOINTS = [
    ("solubility_aqsoldb", ADME, "aqsol_logS", "regression"),
    ("caco2_wang", ADME, "caco2_logPapp", "regression"),
    ("clearance_microsome_az", ADME, "clearance_microsome_logCLint", "regression"),
    ("cyp3a4_veith", ADME, "cyp3a4_inhibition", "classification"),
    ("cyp2d6_veith", ADME, "cyp2d6_inhibition", "classification"),
    ("herg_karim", Tox, "herg_blocker", "classification"),
    ("ames", Tox, "ames_mutagenic", "classification")
]

def main():
    root_dir = Path(__file__).resolve().parent.parent
    features_dir = root_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading and standardizing 7 TDC ADMET datasets...")
    standardizer = ChemicalStandardizer()
    
    std_cache = {}
    endpoint_data = {}
    
    for tdc_name, cls, col_name, task_type in ENDPOINTS:
        print(f"Fetching {tdc_name} ({col_name})...")
        ds = cls(name=tdc_name)
        df = ds.get_data()
        print(f"  Loaded {len(df)} rows.")
        endpoint_data[col_name] = (df, task_type)
        
        # Cache unique SMILES
        for smi in df['Drug'].dropna().unique():
            if smi not in std_cache:
                std_cache[smi] = None
                
    print(f"\nStandardizing {len(std_cache):,} unique ADMET structures via Step 1.2...")
    t0 = time.time()
    for smi in std_cache:
        std_cache[smi] = standardizer.standardize_smiles(smi)
    print(f"Standardized in {time.time()-t0:.1f}s.")
    
    # Consolidate into multi-task table keyed by InChIKey_block1
    compound_dict = {}
    for col_name, (df, task_type) in endpoint_data.items():
        for _, row in df.iterrows():
            raw_smi = row['Drug']
            val = row['Y']
            if pd.isna(val) or raw_smi not in std_cache:
                continue
            std = std_cache[raw_smi]
            if std["status"] != "success":
                continue
                
            ik_b1 = std["inchikey_block1"]
            if ik_b1 not in compound_dict:
                compound_dict[ik_b1] = {
                    "inchikey_block1": ik_b1,
                    "canonical_smiles": std["canonical_smiles"],
                    "inchikey": std["inchikey"],
                    "mol_weight": std["mol_weight"],
                    "heavy_atom_count": std["heavy_atom_count"],
                    "aqsol_logS": np.nan,
                    "caco2_logPapp": np.nan,
                    "clearance_microsome_logCLint": np.nan,
                    "cyp3a4_inhibition": np.nan,
                    "cyp2d6_inhibition": np.nan,
                    "herg_blocker": np.nan,
                    "ames_mutagenic": np.nan
                }
            # Assign value
            compound_dict[ik_b1][col_name] = float(val)
            
    df_admet = pd.DataFrame(list(compound_dict.values()))
    print(f"\nUnified ADMET multi-task dataset created: {len(df_admet):,} unique molecules.")
    print("\nNon-null counts per endpoint:")
    for _, _, col_name, _ in ENDPOINTS:
        n_valid = df_admet[col_name].notna().sum()
        print(f"  {col_name:30s}: {n_valid:,} compounds")
        
    # Save Parquet and CSV
    parquet_path = features_dir / "admet_7endpoints.parquet"
    csv_path = features_dir / "admet_7endpoints.csv"
    df_admet.to_parquet(parquet_path, index=False)
    df_admet.to_csv(csv_path, index=False)
    print(f"Saved: {parquet_path}")
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()
