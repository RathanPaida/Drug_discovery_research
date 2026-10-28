#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Off-Target Selectivity Panel Curation (Stage 5.1 Specification)
Extracts and curates the 8 critical off-target kinases:
O = {TNK2 (ACK1), SRC, ABL1, LCK, PTK2 (FAK), EGFR, KDR (VEGFR2), FLT3}
Applies Step 1.2 standardization and Step 1.3 deduplication.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.standardizer import ChemicalStandardizer

OFFTARGET_GENES = {
    "TNK2": "Q07912",
    "SRC": "P12931",
    "ABL1": "P00519",
    "LCK": "P06239",
    "PTK2": "Q05397",
    "EGFR": "P00533",
    "KDR": "P35968",
    "FLT3": "P36888"
}

def main():
    root_dir = Path(__file__).resolve().parent.parent
    master_csv = root_dir / "tnk1_data" / "TNK1_kinase_master.csv"
    features_dir = root_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading off-target records from {master_csv} for targets: {list(OFFTARGET_GENES.keys())}...")
    t0 = time.time()
    
    cols = [
        'source', 'source_record_id', 'gene_symbol', 'uniprot_accession',
        'compound_name', 'smiles_raw', 'smiles_used', 'std_smiles', 'inchikey14',
        'activity_type', 'relation', 'value_nM', 'p_activity',
        'assay_type', 'flag_contamination', 'is_human_target'
    ]
    
    records_by_gene = {g: [] for g in OFFTARGET_GENES}
    chunk_count = 0
    total_matched = 0
    
    # We read in chunks
    for chunk in pd.read_csv(master_csv, usecols=cols, chunksize=500000, low_memory=False):
        chunk_count += 1
        sub = chunk[chunk['gene_symbol'].isin(OFFTARGET_GENES)]
        if len(sub) > 0:
            total_matched += len(sub)
            for g in OFFTARGET_GENES:
                g_sub = sub[sub['gene_symbol'] == g]
                if len(g_sub) > 0:
                    records_by_gene[g].append(g_sub)
                    
    print(f"Found {total_matched:,} matching off-target records in {time.time()-t0:.1f}s.")
    
    curated_dfs = []
    summary_stats = []
    
    for gene, chunks in records_by_gene.items():
        if not chunks:
            continue
        df_g = pd.concat(chunks, ignore_index=True)
        
        # Filter clean dose-response
        clean_mask = (
            (df_g['flag_contamination'].isna() | (df_g['flag_contamination'] == '')) &
            (df_g['is_human_target'].fillna(True)) &
            (df_g['activity_type'].str.upper().isin(['IC50', 'KI', 'KD'])) &
            (df_g['value_nM'] > 0)
        )
        df_clean = df_g[clean_mask].copy()
        
        # Step 1.3 Activity transformation
        df_clean['pActivity'] = 9.0 - np.log10(df_clean['value_nM'])
        
        # Step 1.3 Triplet Deduplication: (InChIKey_block1, uniprot_id, assay_type)
        df_clean['assay_type_norm'] = df_clean['assay_type'].fillna('B').astype(str).str.upper()
        df_clean['inchikey14_clean'] = df_clean['inchikey14'].fillna('').astype(str)
        df_clean = df_clean[df_clean['inchikey14_clean'] != '']
        
        dedup = df_clean.groupby(['inchikey14_clean', 'assay_type_norm']).agg(
            pActivity_median=('pActivity', 'median'),
            value_nM_median=('value_nM', 'median'),
            pActivity_std=('pActivity', 'std'),
            n_replicates=('pActivity', 'count'),
            canonical_smiles=('std_smiles', 'first'),
            primary_source=('source', 'first'),
            compound_name=('compound_name', 'first')
        ).reset_index()
        
        dedup['gene_symbol'] = gene
        dedup['uniprot_id'] = OFFTARGET_GENES[gene]
        dedup['pActivity_std'] = dedup['pActivity_std'].fillna(0.0)
        
        curated_dfs.append(dedup)
        
        summary_stats.append({
            "gene_symbol": gene,
            "uniprot_id": OFFTARGET_GENES[gene],
            "raw_records": len(df_g),
            "clean_dose_response": len(df_clean),
            "deduplicated_pairs": len(dedup),
            "unique_compounds": dedup['inchikey14_clean'].nunique(),
            "median_pActivity": float(dedup['pActivity_median'].median()),
            "max_pActivity": float(dedup['pActivity_median'].max())
        })
        
    df_all_offtargets = pd.concat(curated_dfs, ignore_index=True)
    df_stats = pd.DataFrame(summary_stats)
    
    parquet_path = features_dir / "offtargets_curated.parquet"
    csv_path = features_dir / "offtargets_curated.csv"
    stats_csv_path = features_dir / "offtargets_summary_stats.csv"
    
    df_all_offtargets.to_parquet(parquet_path, index=False)
    df_all_offtargets.to_csv(csv_path, index=False)
    df_stats.to_csv(stats_csv_path, index=False)
    
    print("\n" + "="*70)
    print("STAGE 5.1 OFF-TARGET SELECTIVITY PANEL SUMMARY:")
    print("="*70)
    print(df_stats.to_string(index=False))
    print(f"\nSaved off-target dataset: {parquet_path} ({len(df_all_offtargets):,} rows)")
    print(f"Saved off-target stats: {stats_csv_path}")

if __name__ == "__main__":
    main()
