#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kinome Multi-Task Dataset Curation (Stage 3.3 Specification)
Curates the ~300,000 compound-kinase pairs across the 491 human kinases
for Multi-Task Pretraining / Transfer Learning (Design A & Design B).
Incorporates:
- Step 1.2 Chemical Standardization
- Step 1.3 Triplet Deduplication (InChIKey_block1, uniprot_id, assay_type)
- Step 3.4 Kinome Pocket Importance Weighting w_ik
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    root_dir = Path(__file__).resolve().parent.parent
    master_csv = root_dir / "tnk1_data" / "TNK1_kinase_master.csv"
    features_dir = root_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Stage 3.4 similarity scores & sample weights
    sim_scores_path = features_dir / "kinome_similarity_scores.parquet"
    if not sim_scores_path.exists():
        print(f"Error: {sim_scores_path} not found. Run generate_kinase_features.py first.")
        sys.exit(1)
        
    df_sim = pd.read_parquet(sim_scores_path)
    weights_lookup = dict(zip(df_sim["uniprot_id"], df_sim["sample_weight_lam3"]))
    sk_lookup = dict(zip(df_sim["uniprot_id"], df_sim["composite_similarity_Sk"]))
    
    print(f"Reading kinome dose-response records from {master_csv}...")
    t0 = time.time()
    
    cols = [
        'source', 'source_record_id', 'gene_symbol', 'uniprot_accession',
        'std_smiles', 'inchikey14', 'activity_type', 'relation',
        'value_nM', 'p_activity', 'assay_type', 'flag_contamination',
        'is_human_target', 'is_training_ready_exact'
    ]
    
    # Select clean quantitative dose-response measurements
    # We aim for ~300,000 representative compound-kinase pairs
    pairs_dict = {}
    chunk_count = 0
    total_processed = 0
    
    for chunk in pd.read_csv(master_csv, usecols=cols, chunksize=500000, low_memory=False):
        chunk_count += 1
        total_processed += len(chunk)
        
        # Filter for clean quantitative human dose-response
        mask = (
            (chunk['flag_contamination'].isna() | (chunk['flag_contamination'] == '')) &
            (chunk['is_human_target'].fillna(True)) &
            (chunk['activity_type'].str.upper().isin(['IC50', 'KI', 'KD'])) &
            (chunk['value_nM'] > 0) &
            (chunk['std_smiles'].notna()) &
            (chunk['std_smiles'] != '') &
            (chunk['inchikey14'].notna()) &
            (chunk['inchikey14'] != '')
        )
        sub = chunk[mask]
        
        for _, r in sub.iterrows():
            ik14 = str(r['inchikey14'])
            uid = str(r['uniprot_accession']).strip().upper()
            if not uid or ";" in uid:  # Keep single protein targets
                continue
                
            ass_type = str(r['assay_type']).strip().upper() if pd.notna(r['assay_type']) else "B"
            if ass_type not in ["B", "F", "A"]:
                ass_type = "B"
                
            triplet_key = (ik14, uid, ass_type)
            val_nm = float(r['value_nM'])
            p_act = 9.0 - np.log10(val_nm)
            rel = str(r['relation']).strip() if pd.notna(r['relation']) else "="
            is_cens = 1 if rel in [">", ">="] or val_nm >= 10000.0 else 0
            
            if triplet_key not in pairs_dict:
                pairs_dict[triplet_key] = {
                    "inchikey14": ik14,
                    "canonical_smiles": str(r['std_smiles']),
                    "uniprot_id": uid,
                    "gene_symbol": str(r['gene_symbol']),
                    "assay_type": ass_type,
                    "p_activities": [p_act],
                    "is_censored": is_cens,
                    "source": str(r['source'])
                }
            else:
                pairs_dict[triplet_key]["p_activities"].append(p_act)
                if is_cens == 0:
                    pairs_dict[triplet_key]["is_censored"] = 0
                    
        # Stop once we have reached ~300,000 to ~350,000 unique compound-kinase pairs
        if len(pairs_dict) >= 320000:
            print(f"Reached {len(pairs_dict):,} unique pairs (target ~300k reached).")
            break
            
    print(f"Processed {total_processed:,} rows in {time.time()-t0:.1f}s. Assembling dataset...")
    
    rows = []
    for (ik14, uid, ass_type), item in pairs_dict.items():
        acts = item["p_activities"]
        med_pact = float(np.median(acts))
        sigma_rep = float(np.std(acts)) if len(acts) > 1 else 0.0
        
        # Look up Stage 3.4 sample importance weight
        sample_w = float(weights_lookup.get(uid, 1.0))
        sk = float(sk_lookup.get(uid, 0.3))
        
        rows.append({
            "inchikey14": ik14,
            "canonical_smiles": item["canonical_smiles"],
            "uniprot_id": uid,
            "gene_symbol": item["gene_symbol"],
            "assay_type": ass_type,
            "pActivity": med_pact,
            "sigma_rep": sigma_rep,
            "is_censored": item["is_censored"],
            "composite_similarity_Sk": sk,
            "sample_weight": sample_w,
            "n_replicates": len(acts),
            "source": item["source"]
        })
        
    df_multitask = pd.DataFrame(rows)
    print(f"\nFinal Stage 3 Multi-Task Kinome dataset: {len(df_multitask):,} pairs.")
    print(f"Unique compounds: {df_multitask['inchikey14'].nunique():,}")
    print(f"Unique kinases: {df_multitask['uniprot_id'].nunique():,}")
    print(f"Exact records: {(df_multitask['is_censored'] == 0).sum():,}")
    print(f"Censored records: {(df_multitask['is_censored'] == 1).sum():,}")
    
    parquet_path = features_dir / "kinome_multitask_300k.parquet"
    csv_path = features_dir / "kinome_multitask_300k.csv"
    
    df_multitask.to_parquet(parquet_path, index=False)
    # Save first 50k rows to CSV as quick preview
    df_multitask.head(50000).to_csv(csv_path, index=False)
    
    print(f"Saved: {parquet_path}")
    print(f"Saved preview: {csv_path}")

if __name__ == "__main__":
    main()
