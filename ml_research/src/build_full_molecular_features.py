#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Throughput Parallel Molecular Featurizer for the 7.88 Lakh Unique Kinome Molecules
Computes across 16 parallel CPU workers:
1. 2048-bit Morgan Fingerprints (ECFP4 radius 2) packed into 256 bytes per molecule.
2. 200 continuous RDKit 2D physicochemical descriptors.
3. MedChem structural evaluations: QED, Ertl SA score, PAINS/Brenk/NIH alerts, and Hinge-Binder SMARTS.
4. Bemis-Murcko scaffolds for Stage 7 RL diversity memory tables.
Saves:
- features/unique_molecules_788k.parquet
- features/unique_molecules_fingerprints_2048bit.npz
- features/unique_molecules_descriptors_200d.npz
"""

import os
import sys
import time
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Suppress RDKit C++ warnings
from rdkit import Chem, RDLogger, RDConfig
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# SA score path
sa_dir = os.path.join(RDConfig.RDContribDir, 'SA_Score')
if sa_dir not in sys.path:
    sys.path.append(sa_dir)
import sascorer

# Module-level globals for worker processes
_WORKER_CATALOG = None
_WORKER_HINGE_SMARTS = None
_DESC_LIST = None

def _init_worker():
    global _WORKER_CATALOG, _WORKER_HINGE_SMARTS, _DESC_LIST
    RDLogger.DisableLog("rdApp.*")
    
    # 1. Alert filter catalog
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
    _WORKER_CATALOG = FilterCatalog(params)
    
    # 2. Hinge SMARTS
    _WORKER_HINGE_SMARTS = [
        Chem.MolFromSmarts("c1ncnc(N)c1"),      # 2-aminopyrimidine
        Chem.MolFromSmarts("c1c[nH]c2ncnc12"),  # pyrrolo[2,3-d]pyrimidine
        Chem.MolFromSmarts("c1ccc2ncncc2c1"),   # quinazoline
        Chem.MolFromSmarts("c1ccc2[nH]ncc2c1"),  # indazole
        Chem.MolFromSmarts("c1ncnc2cc[nH]c12"), # pyrazolopyrimidine
        Chem.MolFromSmarts("c1cnnc2ncnc12")     # pyrazolopyrimidine v2
    ]
    
    # 3. 200 standard continuous descriptors
    _DESC_LIST = Descriptors.descList[:200]

def _featurize_single_mol(item: Tuple[str, str, str]) -> Dict[str, Any]:
    """Featurizes a single molecule (ik14, smi, inchikey)."""
    global _WORKER_CATALOG, _WORKER_HINGE_SMARTS, _DESC_LIST
    ik14, smi, full_ik = item
    
    try:
        mol = Chem.MolFromSmiles(smi)
    except Exception:
        mol = None
        
    if mol is None:
        return {
            "ik14": ik14,
            "smi": smi,
            "full_ik": full_ik,
            "murcko_scaffold": "",
            "qed": 0.0,
            "sa_score": 10.0,
            "has_alerts": True,
            "has_hinge": False,
            "mw": 0.0,
            "hac": 0,
            "fp_bytes": bytes(256),
            "desc_200": np.zeros(200, dtype=np.float32)
        }
        
    # 1. 2048-bit Morgan FP packed into 256 bytes
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros(2048, dtype=np.uint8)
    AllChem.DataStructs.ConvertToNumpyArray(bv, arr)
    packed_fp = np.packbits(arr).tobytes()
    
    # 2. 200 Continuous 2D Descriptors
    desc_arr = np.zeros(200, dtype=np.float32)
    for i, (name, fn) in enumerate(_DESC_LIST):
        try:
            val = float(fn(mol))
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            desc_arr[i] = val
        except Exception:
            desc_arr[i] = 0.0
            
    # 3. MedChem structural properties
    try:
        qed_val = float(QED.qed(mol))
    except Exception:
        qed_val = 0.0
        
    try:
        sa_val = float(sascorer.calculateScore(mol))
    except Exception:
        sa_val = 10.0
        
    has_alerts = bool(_WORKER_CATALOG.HasMatch(mol))
    has_hinge = any(mol.HasSubstructMatch(p) for p in _WORKER_HINGE_SMARTS if p is not None)
    
    # 4. Murcko Scaffold
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaf = ""
        
    mw = float(Descriptors.MolWt(mol))
    hac = int(mol.GetNumHeavyAtoms())
    
    return {
        "ik14": ik14,
        "smi": smi,
        "full_ik": full_ik,
        "murcko_scaffold": scaf,
        "qed": qed_val,
        "sa_score": sa_val,
        "has_alerts": has_alerts,
        "has_hinge": has_hinge,
        "mw": mw,
        "hac": hac,
        "fp_bytes": packed_fp,
        "desc_200": desc_arr
    }

def main():
    root_dir = Path(__file__).resolve().parent.parent
    master_csv = root_dir / "tnk1_data" / "TNK1_kinase_master.csv"
    features_dir = root_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting unique molecules from {master_csv}...")
    t0 = time.time()
    
    # Gather unique (inchikey14, std_smiles, inchikey)
    unique_mol_dict = {}
    cols = ['is_cross_source_duplicate', 'is_human_target', 'flag_contamination', 'std_smiles', 'inchikey', 'inchikey14']
    
    for chunk in pd.read_csv(master_csv, usecols=cols, chunksize=500000, low_memory=False):
        mask = (
            (~chunk['is_cross_source_duplicate'].fillna(False).astype(bool)) &
            (chunk['is_human_target'].fillna(True).astype(bool)) &
            (chunk['flag_contamination'].isna() | (chunk['flag_contamination'] == '')) &
            (chunk['std_smiles'].notna()) & (chunk['std_smiles'] != '') &
            (chunk['inchikey14'].notna()) & (chunk['inchikey14'] != '')
        )
        sub = chunk[mask]
        for r in sub.itertuples():
            ik14 = str(r.inchikey14)
            if ik14 not in unique_mol_dict:
                unique_mol_dict[ik14] = (ik14, str(r.std_smiles), str(r.inchikey))
                
    items = list(unique_mol_dict.values())
    n_mols = len(items)
    print(f"Extracted {n_mols:,} unique molecules in {time.time()-t0:.1f}s.")
    
    n_procs = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"Launching parallel featurization across {n_procs} CPU processes...")
    t_start = time.time()
    
    batch_size = 20000
    all_ik14 = []
    all_smi = []
    all_full_ik = []
    all_scaf = []
    all_qed = []
    all_sa = []
    all_alerts = []
    all_hinge = []
    all_mw = []
    all_hac = []
    all_fp_bytes = []
    all_desc = []
    
    out_parquet = features_dir / "unique_molecules_788k.parquet"
    writer = None
    
    with mp.Pool(n_procs, initializer=_init_worker) as pool:
        for b_idx in range(0, n_mols, batch_size):
            batch = items[b_idx:b_idx + batch_size]
            results = pool.map(_featurize_single_mol, batch, chunksize=250)
            
            b_ik14, b_smi, b_full_ik, b_scaf = [], [], [], []
            b_qed, b_sa, b_alerts, b_hinge, b_mw, b_hac = [], [], [], [], [], []
            b_fp, b_desc = [], []
            
            for r in results:
                b_ik14.append(r["ik14"])
                b_smi.append(r["smi"])
                b_full_ik.append(r["full_ik"])
                b_scaf.append(r["murcko_scaffold"])
                b_qed.append(r["qed"])
                b_sa.append(r["sa_score"])
                b_alerts.append(r["has_alerts"])
                b_hinge.append(r["has_hinge"])
                b_mw.append(r["mw"])
                b_hac.append(r["hac"])
                b_fp.append(r["fp_bytes"])
                b_desc.append(r["desc_200"])
                
            all_ik14.extend(b_ik14)
            all_fp_bytes.extend(b_fp)
            all_desc.append(np.vstack(b_desc))
            
            # Write parquet table chunk
            df_chunk = pd.DataFrame({
                "inchikey14": b_ik14,
                "canonical_smiles": b_smi,
                "inchikey": b_full_ik,
                "murcko_scaffold": b_scaf,
                "qed": np.array(b_qed, dtype=np.float32),
                "sa_score": np.array(b_sa, dtype=np.float32),
                "has_alerts": np.array(b_alerts, dtype=bool),
                "has_hinge_motif": np.array(b_hinge, dtype=bool),
                "mol_weight": np.array(b_mw, dtype=np.float32),
                "heavy_atom_count": np.array(b_hac, dtype=np.int16)
            })
            
            table = pa.Table.from_pandas(df_chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_parquet, table.schema, compression='snappy')
            writer.write_table(table)
            
            done = min(b_idx + batch_size, n_mols)
            rate = done / max(time.time() - t_start, 1)
            print(f"  Processed {done:,}/{n_mols:,} molecules ({rate:.0f} mols/s)...")
            
    if writer:
        writer.close()
        
    print(f"\nSaved molecular metadata to {out_parquet} ({os.path.getsize(out_parquet) / 1024**2:.1f} MB)")
    
    # Assemble packed fingerprint array (N, 256) uint8
    print("Packing 2048-bit Morgan Fingerprint matrix (uint8)...")
    packed_fp_matrix = np.frombuffer(b''.join(all_fp_bytes), dtype=np.uint8).reshape((n_mols, 256))
    
    fp_npz_path = features_dir / "unique_molecules_fingerprints_2048bit.npz"
    np.savez_compressed(
        fp_npz_path,
        inchikey14=np.array(all_ik14),
        packed_fps_256bytes=packed_fp_matrix
    )
    print(f"Saved packed fingerprint matrix: {fp_npz_path} ({os.path.getsize(fp_npz_path) / 1024**2:.1f} MB)")
    
    # Assemble continuous 200-dim descriptor matrix (N, 200) float32
    print("Assembling 200-dim continuous RDKit descriptor matrix (float32)...")
    all_desc_matrix = np.vstack(all_desc)
    desc_names = [name for name, _ in Descriptors.descList[:200]]
    
    desc_npz_path = features_dir / "unique_molecules_descriptors_200d.npz"
    np.savez_compressed(
        desc_npz_path,
        inchikey14=np.array(all_ik14),
        descriptors_200d=all_desc_matrix,
        descriptor_names=np.array(desc_names)
    )
    print(f"Saved continuous descriptor matrix: {desc_npz_path} ({os.path.getsize(desc_npz_path) / 1024**2:.1f} MB)")
    print(f"\nAll {n_mols:,} unique kinome molecules featurized successfully in {time.time()-t_start:.1f}s!")

if __name__ == "__main__":
    mp.freeze_support()
    main()
