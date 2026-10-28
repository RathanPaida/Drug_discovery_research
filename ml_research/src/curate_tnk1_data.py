#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TNK1 Bioactivity Curation, Deduplication & Frozen Scaffold Splits (Stages 1.2, 1.3, 2.0)
Strictly adheres to:
- Step 1.2: RDKit 6-step Standardization
- Step 1.3: Triplet Deduplication (InChIKey_block1, uniprot_id, assay_type)
            Censored weak negative handling (c_ik = 1)
- Step 0.2: Deterministic Seeding (seeds = [42, 101, 202, 303, 404])
- Step 0.3: Parquet & SHA-256 Checksum locking
- Stage 2: 5x5 nested Bemis-Murcko scaffold cross-validation
"""

import os
import sys
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.standardizer import ChemicalStandardizer

SEEDS = [42, 101, 202, 303, 404]

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def extract_and_standardize_tnk1(master_csv_path: Path, output_dir: Path) -> pd.DataFrame:
    print(f"Reading TNK1 records from {master_csv_path}...")
    t0 = time.time()
    standardizer = ChemicalStandardizer()
    
    cols = [
        'source', 'source_record_id', 'gene_symbol', 'uniprot_accession',
        'compound_name', 'smiles_raw', 'smiles_used', 'std_smiles',
        'activity_type', 'relation', 'value_nM', 'p_activity',
        'assay_type', 'flag_contamination', 'is_human_target'
    ]
    
    tnk1_rows = []
    chunk_count = 0
    for chunk in pd.read_csv(master_csv_path, usecols=cols, chunksize=250000, low_memory=False):
        chunk_count += 1
        sub = chunk[chunk['gene_symbol'] == 'TNK1']
        if len(sub) > 0:
            tnk1_rows.append(sub)
            
    df_raw = pd.concat(tnk1_rows, ignore_index=True)
    print(f"Found {len(df_raw)} raw TNK1 rows in {time.time()-t0:.1f}s.")
    
    # Filter out contamination and non-human
    df_clean = df_raw[
        (df_raw['flag_contamination'].isna() | (df_raw['flag_contamination'] == '')) &
        (df_raw['is_human_target'].fillna(True))
    ].copy()
    print(f"Clean (non-contaminated, human) TNK1 rows: {len(df_clean)}")
    
    # Apply Step 1.2 Standardization to each unique SMILES
    unique_smiles = df_clean['smiles_used'].fillna(df_clean['smiles_raw']).unique()
    std_cache = {}
    print(f"Standardizing {len(unique_smiles)} unique structures...")
    
    for smi in unique_smiles:
        if smi and isinstance(smi, str):
            std_cache[smi] = standardizer.standardize_smiles(smi)
            
    valid_records = []
    for idx, row in df_clean.iterrows():
        smi = row['smiles_used'] if pd.notna(row['smiles_used']) and row['smiles_used'] else row['smiles_raw']
        std_res = std_cache.get(smi, {"status": "failed"})
        
        if std_res["status"] != "success":
            continue
            
        act_type = str(row['activity_type']).strip().upper()
        rel = str(row['relation']).strip() if pd.notna(row['relation']) else "="
        if rel in ["==", ""]:
            rel = "="
            
        val_nm = row['value_nM']
        if pd.isna(val_nm) or val_nm <= 0:
            continue
            
        # Determine exact vs censored
        # Exact: relation is '=', activity_type in IC50, Ki, Kd
        is_dose_response = act_type in ["IC50", "KI", "KD"]
        if not is_dose_response:
            continue
            
        p_act = 9.0 - np.log10(val_nm)
        
        is_censored = 1 if rel in [">", ">="] or val_nm >= 10000.0 else 0
        
        # Assay type normalization
        ass_type = str(row['assay_type']).strip().upper() if pd.notna(row['assay_type']) else "B"
        if ass_type not in ["B", "F", "A"]:
            ass_type = "B"
            
        # Murcko Scaffold
        mol = std_res["mol"]
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            scaffold = ""
            
        rec = {
            "source": row["source"],
            "source_record_id": str(row["source_record_id"]),
            "gene_symbol": "TNK1",
            "uniprot_id": "Q13470",
            "compound_name": str(row["compound_name"]) if pd.notna(row["compound_name"]) else "",
            "canonical_smiles": std_res["canonical_smiles"],
            "inchikey": std_res["inchikey"],
            "inchikey_block1": std_res["inchikey_block1"],
            "murcko_scaffold": scaffold,
            "mol_weight": std_res["mol_weight"],
            "heavy_atom_count": std_res["heavy_atom_count"],
            "activity_type": act_type,
            "relation": rel,
            "value_nM": float(val_nm),
            "pActivity_raw": float(p_act),
            "is_censored": int(is_censored),
            "assay_type": ass_type,
        }
        valid_records.append(rec)
        
    df_valid = pd.DataFrame(valid_records)
    print(f"Valid standardized dose-response records: {len(df_valid)}")
    return df_valid


def deduplicate_triplets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1.3: Deduplicate by (InChIKey_block1, uniprot_id, assay_type)
    Compute median pActivity and replicate standard deviation sigma_rep.
    """
    print("Executing Step 1.3 Triplet Deduplication...")
    groups = df.groupby(['inchikey_block1', 'uniprot_id', 'assay_type'])
    
    dedup_rows = []
    for (ik_b1, uid, ass_type), group in groups:
        # Separate exact vs censored
        exact_sub = group[group['is_censored'] == 0]
        
        if len(exact_sub) > 0:
            target_sub = exact_sub
            is_cens = 0
        else:
            target_sub = group
            is_cens = 1
            
        p_acts = target_sub['pActivity_raw'].values
        median_pact = float(np.median(p_acts))
        sigma_rep = float(np.std(p_acts)) if len(p_acts) > 1 else 0.0
        val_range = float(np.ptp(p_acts)) if len(p_acts) > 1 else 0.0
        is_high_variance = 1 if val_range > 1.0 else 0
        
        rep_row = target_sub.iloc[0]
        dedup_rows.append({
            "compound_id": f"TNK1-CPD-{len(dedup_rows)+1:04d}",
            "inchikey_block1": ik_b1,
            "inchikey": rep_row["inchikey"],
            "canonical_smiles": rep_row["canonical_smiles"],
            "murcko_scaffold": rep_row["murcko_scaffold"],
            "mol_weight": rep_row["mol_weight"],
            "heavy_atom_count": rep_row["heavy_atom_count"],
            "uniprot_id": uid,
            "gene_symbol": "TNK1",
            "assay_type": ass_type,
            "activity_type": rep_row["activity_type"],
            "pActivity": median_pact,
            "value_nM": float(10 ** (9.0 - median_pact)),
            "sigma_rep": sigma_rep,
            "replicate_range": val_range,
            "is_high_variance": is_high_variance,
            "n_replicates": len(target_sub),
            "is_censored": is_cens,
            "primary_source": rep_row["source"],
            "compound_name": rep_row["compound_name"]
        })
        
    df_dedup = pd.DataFrame(dedup_rows)
    print(f"Deduplicated to {len(df_dedup)} unique (compound, assay_type) pairs.")
    print(f"Exact records: {len(df_dedup[df_dedup['is_censored'] == 0])}")
    print(f"Censored records: {len(df_dedup[df_dedup['is_censored'] == 1])}")
    print(f"Unique Murcko scaffolds: {df_dedup['murcko_scaffold'].nunique()}")
    return df_dedup


def generate_nested_scaffold_splits(df: pd.DataFrame, n_outer: int = 5, n_inner: int = 5, seed: int = 42) -> Dict[str, Any]:
    """
    Step 2.0: Generate 5x5 nested Bemis-Murcko scaffold cross-validation splits.
    Ensures zero scaffold leakage between train, validation, and test sets.
    """
    rng = np.random.RandomState(seed)
    
    # Group compound indices by Murcko scaffold
    scaffold_to_indices = {}
    for idx, scaf in enumerate(df['murcko_scaffold']):
        if scaf not in scaffold_to_indices:
            scaffold_to_indices[scaf] = []
        scaffold_to_indices[scaf].append(idx)
        
    scaffolds = list(scaffold_to_indices.keys())
    rng.shuffle(scaffolds)
    
    # Partition scaffolds into 5 outer folds approximately equal in compound count
    outer_folds = [[] for _ in range(n_outer)]
    outer_counts = [0] * n_outer
    
    # Sort scaffolds by size descending for greedy balanced partition
    sorted_scaffolds = sorted(scaffolds, key=lambda s: len(scaffold_to_indices[s]), reverse=True)
    for scaf in sorted_scaffolds:
        min_fold = int(np.argmin(outer_counts))
        outer_folds[min_fold].extend(scaffold_to_indices[scaf])
        outer_counts[min_fold] += len(scaffold_to_indices[scaf])
        
    splits = {}
    for outer_idx in range(n_outer):
        test_indices = sorted(outer_folds[outer_idx])
        train_val_indices = sorted([i for f in range(n_outer) if f != outer_idx for i in outer_folds[f]])
        
        # Now partition train_val into n_inner folds
        # Gather scaffolds in train_val
        tv_scaffold_to_indices = {}
        for idx in train_val_indices:
            scaf = df.iloc[idx]['murcko_scaffold']
            if scaf not in tv_scaffold_to_indices:
                tv_scaffold_to_indices[scaf] = []
            tv_scaffold_to_indices[scaf].append(idx)
            
        tv_sorted_scafs = sorted(tv_scaffold_to_indices.keys(), key=lambda s: len(tv_scaffold_to_indices[s]), reverse=True)
        inner_folds = [[] for _ in range(n_inner)]
        inner_counts = [0] * n_inner
        for scaf in tv_sorted_scafs:
            min_inner = int(np.argmin(inner_counts))
            inner_folds[min_inner].extend(tv_scaffold_to_indices[scaf])
            inner_counts[min_inner] += len(tv_scaffold_to_indices[scaf])
            
        inner_splits = []
        for inner_idx in range(n_inner):
            val_indices = sorted(inner_folds[inner_idx])
            train_indices = sorted([i for f in range(n_inner) if f != inner_idx for i in inner_folds[f]])
            inner_splits.append({
                "inner_fold": inner_idx,
                "train_indices": train_indices,
                "val_indices": val_indices
            })
            
        splits[f"outer_fold_{outer_idx}"] = {
            "test_indices": test_indices,
            "train_val_indices": train_val_indices,
            "inner_splits": inner_splits
        }
        
    return splits


def main():
    root_dir = Path(__file__).resolve().parent.parent
    master_csv = root_dir / "tnk1_data" / "TNK1_kinase_master.csv"
    features_dir = root_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract and standardize
    df_valid = extract_and_standardize_tnk1(master_csv, features_dir)
    
    # 2. Triplet Deduplication
    df_curated = deduplicate_triplets(df_valid)
    
    # 3. Save Curated TNK1 Parquet & CSV (Step 0.3 Data Locking)
    parquet_path = features_dir / "tnk1_curated.parquet"
    csv_path = features_dir / "tnk1_curated.csv"
    
    df_curated.to_parquet(parquet_path, index=False)
    df_curated.to_csv(csv_path, index=False)
    
    sha_parquet = compute_sha256(parquet_path)
    sha_csv = compute_sha256(csv_path)
    print(f"Saved: {parquet_path} (SHA-256: {sha_parquet})")
    print(f"Saved: {csv_path} (SHA-256: {sha_csv})")
    
    # 4. Generate 5x5 Nested Scaffold Splits across 5 fixed random seeds
    all_seed_splits = {}
    for seed in SEEDS:
        splits = generate_nested_scaffold_splits(df_curated, n_outer=5, n_inner=5, seed=seed)
        all_seed_splits[f"seed_{seed}"] = splits
        
    splits_json_path = features_dir / "tnk1_nested_scaffold_splits_5x5.json"
    with open(splits_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_seed_splits, f, indent=2)
        
    sha_splits = compute_sha256(splits_json_path)
    print(f"Saved: {splits_json_path} (SHA-256: {sha_splits})")
    
    # 5. Save metadata
    meta = {
        "dataset_name": "TNK1_curated_dose_response",
        "target": "TNK1 (UniProt Q13470)",
        "total_compounds": len(df_curated),
        "exact_compounds": int((df_curated['is_censored'] == 0).sum()),
        "censored_compounds": int((df_curated['is_censored'] == 1).sum()),
        "unique_scaffolds": int(df_curated['murcko_scaffold'].nunique()),
        "seeds": SEEDS,
        "files": {
            "parquet": {"path": str(parquet_path.relative_to(root_dir)), "sha256": sha_parquet},
            "csv": {"path": str(csv_path.relative_to(root_dir)), "sha256": sha_csv},
            "splits_json": {"path": str(splits_json_path.relative_to(root_dir)), "sha256": sha_splits}
        }
    }
    
    meta_path = features_dir / "tnk1_curated_metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_path}")
    print("\nTNK1 curation and 5x5 nested scaffold splitting completed successfully!")

if __name__ == "__main__":
    main()
