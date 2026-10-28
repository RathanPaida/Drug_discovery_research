#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-Scale Kinome Bioactivity Processing Pipeline (All 3.955M Distinct Measurements)
Extracts and standardizes all non-cross-source duplicate bioactivity measurements across the 491 human kinases:
- Attaches Target Kinase metadata and Stage 3.4 pocket similarity S_k + sample weights w_ik
- Transforms affinities into logarithmic potency: pActivity = 9 - log10(Value_nM)
- Flags censored weak negatives (c_ik = 1)
- Flags off-target panel kinases and TNK1
- Streams output into compressed Parquet using PyArrow.
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OFFTARGET_SET = {"TNK2", "SRC", "ABL1", "LCK", "PTK2", "EGFR", "KDR", "FLT3"}

def main():
    root_dir = Path(__file__).resolve().parent.parent
    master_csv = root_dir / "tnk1_data" / "TNK1_kinase_master.csv"
    features_dir = root_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Stage 3.4 similarity scores & sample weights
    sim_scores_path = features_dir / "kinome_similarity_scores.parquet"
    if not sim_scores_path.exists():
        print(f"Error: {sim_scores_path} not found.")
        sys.exit(1)
        
    df_sim = pd.read_parquet(sim_scores_path)
    weights_lookup = dict(zip(df_sim["uniprot_id"], df_sim["sample_weight_lam3"]))
    sk_lookup = dict(zip(df_sim["uniprot_id"], df_sim["composite_similarity_Sk"]))
    
    out_parquet = features_dir / "kinome_all_measurements_3.95M.parquet"
    print(f"Streaming and processing distinct kinome measurements to {out_parquet}...")
    t0 = time.time()
    
    cols = [
        'source', 'source_record_id', 'gene_symbol', 'uniprot_accession',
        'relatedness_tier', 'compound_name', 'std_smiles', 'inchikey', 'inchikey14',
        'activity_type', 'relation', 'value_nM', 'p_activity', 'assay_type',
        'record_class', 'measure_semantics', 'flag_contamination',
        'is_human_target', 'is_cross_source_duplicate', 'is_training_ready_exact'
    ]
    
    writer = None
    total_written = 0
    total_exact_dr = 0
    total_censored_dr = 0
    total_single_point = 0
    total_qualitative = 0
    
    chunk_size = 250000
    for chunk_idx, chunk in enumerate(pd.read_csv(master_csv, usecols=cols, chunksize=chunk_size, low_memory=False)):
        # Filter: human target, non-cross-source duplicate, non-contaminated
        mask = (
            (~chunk['is_cross_source_duplicate'].fillna(False).astype(bool)) &
            (chunk['is_human_target'].fillna(True).astype(bool)) &
            (chunk['flag_contamination'].isna() | (chunk['flag_contamination'] == '')) &
            (chunk['std_smiles'].notna()) & (chunk['std_smiles'] != '')
        )
        sub = chunk[mask].copy()
        if len(sub) == 0:
            continue
            
        # Target accession normalization
        uids = sub['uniprot_accession'].fillna('').astype(str).str.strip().str.upper()
        genes = sub['gene_symbol'].fillna('').astype(str).str.strip().str.upper()
        
        # Attach Stage 3.4 S_k and sample weights w_ik
        sub['composite_similarity_Sk'] = uids.map(sk_lookup).fillna(0.3).astype(np.float32)
        sub['sample_weight_lam3'] = uids.map(weights_lookup).fillna(1.0).astype(np.float32)
        
        # Potency & Censoring
        val_nm = pd.to_numeric(sub['value_nM'], errors='coerce')
        sub['val_nM_clean'] = val_nm
        
        act_types = sub['activity_type'].fillna('').astype(str).str.upper()
        is_dr = act_types.isin(['IC50', 'KI', 'KD', 'EC50']) & (val_nm > 0)
        
        p_act = np.where(is_dr, 9.0 - np.log10(val_nm.clip(lower=1e-6)), sub['p_activity'].fillna(0.0))
        sub['pActivity'] = p_act.astype(np.float32)
        
        rel = sub['relation'].fillna('=').astype(str).str.strip()
        is_cens = np.where(is_dr & (rel.isin(['>', '>=']) | (val_nm >= 10000.0)), 1, 0)
        sub['is_censored'] = is_cens.astype(np.int8)
        
        # Classify measurement
        is_exact = is_dr & (sub['is_censored'] == 0)
        is_sp = act_types.str.contains('PCT|INHIBITION|CONTROL', regex=True)
        is_qual = sub['record_class'].fillna('').str.lower() == 'qualitative'
        
        total_exact_dr += int(is_exact.sum())
        total_censored_dr += int((is_dr & (sub['is_censored'] == 1)).sum())
        total_single_point += int(is_sp.sum())
        total_qualitative += int(is_qual.sum())
        
        # Pipeline flags
        sub['is_tnk1'] = (genes == 'TNK1').astype(bool)
        sub['is_stage5_offtarget'] = genes.isin(OFFTARGET_SET).astype(bool)
        
        # Ensure source_record_id is always string
        sub['source_record_id'] = sub['source_record_id'].fillna('').astype(str)
        
        # Clean columns to save
        save_df = sub[[
            'source', 'source_record_id', 'gene_symbol', 'uniprot_accession',
            'relatedness_tier', 'std_smiles', 'inchikey', 'inchikey14',
            'activity_type', 'relation', 'val_nM_clean', 'pActivity',
            'is_censored', 'assay_type', 'record_class',
            'composite_similarity_Sk', 'sample_weight_lam3',
            'is_tnk1', 'is_stage5_offtarget'
        ]].rename(columns={'val_nM_clean': 'value_nM'})
        
        pa_schema = pa.schema([
            ('source', pa.string()),
            ('source_record_id', pa.string()),
            ('gene_symbol', pa.string()),
            ('uniprot_accession', pa.string()),
            ('relatedness_tier', pa.string()),
            ('std_smiles', pa.string()),
            ('inchikey', pa.string()),
            ('inchikey14', pa.string()),
            ('activity_type', pa.string()),
            ('relation', pa.string()),
            ('value_nM', pa.float64()),
            ('pActivity', pa.float32()),
            ('is_censored', pa.int8()),
            ('assay_type', pa.string()),
            ('record_class', pa.string()),
            ('composite_similarity_Sk', pa.float32()),
            ('sample_weight_lam3', pa.float32()),
            ('is_tnk1', pa.bool_()),
            ('is_stage5_offtarget', pa.bool_())
        ])
        
        table = pa.Table.from_pandas(save_df, schema=pa_schema, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_parquet, pa_schema, compression='snappy')
        writer.write_table(table)
        total_written += len(save_df)
        
        if (chunk_idx + 1) % 4 == 0:
            print(f"  Processed {total_written:,} distinct measurements ({time.time()-t0:.1f}s)...")
            
    if writer:
        writer.close()
        
    print(f"\nCompleted! Written {total_written:,} distinct measurements to {out_parquet} in {time.time()-t0:.1f}s.")
    print(f"  Exact Dose-Response Measurements (IC50/Ki/Kd): {total_exact_dr:,}")
    print(f"  Censored Weak Negatives: {total_censored_dr:,}")
    print(f"  Single-Point Screening Measurements (% Inh / Ctrl): {total_single_point:,}")
    print(f"  Qualitative Outcome Screen Records: {total_qualitative:,}")
    
    summary = {
        "dataset": "kinome_all_measurements_3.95M",
        "total_distinct_measurements": total_written,
        "exact_dose_response_measurements": total_exact_dr,
        "censored_weak_negatives": total_censored_dr,
        "single_point_screen_measurements": total_single_point,
        "qualitative_screen_records": total_qualitative,
        "parquet_size_mb": round(os.path.getsize(out_parquet) / (1024 * 1024), 2)
    }
    with open(features_dir / "kinome_all_measurements_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
