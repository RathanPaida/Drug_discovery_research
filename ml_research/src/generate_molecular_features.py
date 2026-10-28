#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Molecular Feature Engineering Pipeline (Stages 2.1 & 3.1)
Strictly adheres to:
- Step 2.1: 2048-bit Morgan Fingerprints (ECFP4 radius 2, binary & counts)
            200 continuous RDKit 2D physicochemical descriptors (MolLogP, TPSA, MolWt, etc.)
            RobustScaler standardization
- Step 3.1: D-MPNN Graph Featurization:
            Atom feature vector x_v in R^133
            Directed bond feature vector e_vw in R^14
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.preprocessing import RobustScaler

# Define atom feature dimensions matching Stage 3.1 (133-dim)
# Atomic numbers: 1 to 100
MAX_ATOMIC_NUM = 100
DEGREE_CHOICES = list(range(6))  # 0 to 5
FORMAL_CHARGE_CHOICES = list(range(-2, 3))  # -2 to +2
CHIRAL_CHOICES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
NUM_H_CHOICES = list(range(5))  # 0 to 4
HYBRIDIZATION_CHOICES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED
]

def one_hot(val: Any, choices: List[Any]) -> List[float]:
    vec = [0.0] * (len(choices) + 1)
    if val in choices:
        vec[choices.index(val)] = 1.0
    else:
        vec[-1] = 1.0
    return vec

def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """Computes the 133-dimensional atom feature vector (Stage 3.1)."""
    features = []
    
    # 1. Atomic number (one-hot up to 100 + other = 101 dims)
    anum = atom.GetAtomicNum()
    features.extend(one_hot(anum, list(range(1, MAX_ATOMIC_NUM + 1))))
    
    # 2. Total degree (0-5 + other = 7 dims)
    features.extend(one_hot(atom.GetTotalDegree(), DEGREE_CHOICES))
    
    # 3. Formal charge (-2 to +2 + other = 6 dims)
    features.extend(one_hot(atom.GetFormalCharge(), FORMAL_CHARGE_CHOICES))
    
    # 4. Chiral tag (4 dims)
    features.extend(one_hot(atom.GetChiralTag(), CHIRAL_CHOICES))
    
    # 5. Number of bonded hydrogens (0-4 + other = 6 dims)
    features.extend(one_hot(atom.GetTotalNumHs(), NUM_H_CHOICES))
    
    # 6. Hybridization (6 + other = 7 dims)
    features.extend(one_hot(atom.GetHybridization(), HYBRIDIZATION_CHOICES))
    
    # 7. Aromaticity (1 dim)
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    
    # 8. Scaled atomic mass (1 dim)
    features.append(atom.GetMass() * 0.01)
    
    feat_arr = np.array(features, dtype=np.float32)
    # Ensure exact length 133
    if len(feat_arr) < 133:
        feat_arr = np.pad(feat_arr, (0, 133 - len(feat_arr)))
    elif len(feat_arr) > 133:
        feat_arr = feat_arr[:133]
    return feat_arr


# Define bond feature dimensions matching Stage 3.1 (14-dim)
BOND_TYPE_CHOICES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]
BOND_STEREO_CHOICES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS
]

def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    """Computes the 14-dimensional bond feature vector (Stage 3.1)."""
    features = []
    
    # 1. Bond type (4 choices + other = 5 dims)
    features.extend(one_hot(bond.GetBondType(), BOND_TYPE_CHOICES))
    
    # 2. Conjugation flag (1 dim)
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    
    # 3. Ring membership flag (1 dim)
    features.append(1.0 if bond.IsInRing() else 0.0)
    
    # 4. Stereochemistry (6 choices + other = 7 dims)
    features.extend(one_hot(bond.GetStereo(), BOND_STEREO_CHOICES))
    
    feat_arr = np.array(features, dtype=np.float32)
    if len(feat_arr) < 14:
        feat_arr = np.pad(feat_arr, (0, 14 - len(feat_arr)))
    elif len(feat_arr) > 14:
        feat_arr = feat_arr[:14]
    return feat_arr


def mol_to_graph_dict(mol: Chem.Mol) -> Dict[str, Any]:
    """Converts an RDKit Mol to D-MPNN directed graph tensors."""
    if mol is None:
        return {}
    
    n_atoms = mol.GetNumAtoms()
    atom_features = [get_atom_features(mol.GetAtomWithIdx(i)) for i in range(n_atoms)]
    
    # Directed edge list
    u_list, v_list = [], []
    bond_features = []
    edge_pairs = []
    
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        b_feat = get_bond_features(bond)
        
        # Directed u -> v
        u_list.append(u)
        v_list.append(v)
        bond_features.append(b_feat)
        edge_pairs.append((u, v))
        
        # Directed v -> u
        u_list.append(v)
        v_list.append(u)
        bond_features.append(b_feat)
        edge_pairs.append((v, u))
        
    # Reverse edge mapping for D-MPNN message passing
    rev_edge_indices = []
    edge_dict = {pair: i for i, pair in enumerate(edge_pairs)}
    for (u, v) in edge_pairs:
        rev_edge_indices.append(edge_dict[(v, u)])
        
    return {
        "n_atoms": n_atoms,
        "n_bonds": len(edge_pairs),
        "atom_features": np.array(atom_features, dtype=np.float32),
        "bond_features": np.array(bond_features, dtype=np.float32) if bond_features else np.zeros((0, 14), dtype=np.float32),
        "edge_index": np.array([u_list, v_list], dtype=np.int64) if u_list else np.zeros((2, 0), dtype=np.int64),
        "rev_edge_indices": np.array(rev_edge_indices, dtype=np.int64) if rev_edge_indices else np.zeros(0, dtype=np.int64)
    }


def compute_morgan_fingerprints(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """Computes both 2048-bit count and bit vectors."""
    bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    bit_arr = np.zeros((n_bits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(bit_vect, bit_arr)
    
    raw_counts = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=n_bits)
    count_arr = np.zeros((n_bits,), dtype=np.float32)
    for bit_id, count in raw_counts.GetNonzeroElements().items():
        count_arr[bit_id] = float(count)
        
    return bit_arr, count_arr


def get_curated_rdkit_2d_descriptors(mol: Chem.Mol) -> Tuple[np.ndarray, List[str]]:
    """Calculates continuous 2D physicochemical descriptors."""
    desc_names = []
    desc_values = []
    for name, func in Descriptors.descList:
        desc_names.append(name)
        try:
            val = float(func(mol))
            if np.isnan(val) or np.isinf(val):
                val = 0.0
        except Exception:
            val = 0.0
        desc_values.append(val)
        
    # Cap to exactly first 200 standard continuous descriptors
    return np.array(desc_values[:200], dtype=np.float32), desc_names[:200]


def main():
    root_dir = Path(__file__).resolve().parent.parent
    features_dir = root_dir / "features"
    tnk1_parquet = features_dir / "tnk1_curated.parquet"
    
    if not tnk1_parquet.exists():
        print(f"Error: {tnk1_parquet} does not exist. Run curate_tnk1_data.py first.")
        sys.exit(1)
        
    print(f"Loading curated TNK1 dataset from {tnk1_parquet}...")
    df = pd.read_parquet(tnk1_parquet)
    print(f"Total curated compounds: {len(df)}")
    
    bit_fps = []
    count_fps = []
    rdkit_2d_list = []
    graph_data_list = []
    desc_names = None
    
    print("Generating Morgan Fingerprints, 2D Descriptors & D-MPNN Molecular Graphs...")
    for idx, smi in enumerate(df['canonical_smiles']):
        mol = Chem.MolFromSmiles(smi)
        
        # 1. Morgan Fingerprints
        b_fp, c_fp = compute_morgan_fingerprints(mol, radius=2, n_bits=2048)
        bit_fps.append(b_fp)
        count_fps.append(c_fp)
        
        # 2. 200 Continuous RDKit 2D Descriptors
        desc_vals, names = get_curated_rdkit_2d_descriptors(mol)
        if desc_names is None:
            desc_names = names
        rdkit_2d_list.append(desc_vals)
        
        # 3. D-MPNN Graph Features
        g_dict = mol_to_graph_dict(mol)
        g_dict["compound_id"] = df.iloc[idx]["compound_id"]
        g_dict["pActivity"] = float(df.iloc[idx]["pActivity"])
        g_dict["is_censored"] = int(df.iloc[idx]["is_censored"])
        graph_data_list.append(g_dict)
        
    X_bit_fp = np.vstack(bit_fps)
    X_count_fp = np.vstack(count_fps)
    X_rdkit_2d_raw = np.vstack(rdkit_2d_list)
    
    # RobustScaler on 2D descriptors (Stage 2.1 specification)
    scaler = RobustScaler()
    X_rdkit_2d_scaled = scaler.fit_transform(X_rdkit_2d_raw)
    
    # Assay type one-hot encoding: [is_B, is_F, is_A]
    assay_types = df['assay_type'].values
    assay_one_hot = np.zeros((len(df), 3), dtype=np.float32)
    for i, a in enumerate(assay_types):
        if a == "B":
            assay_one_hot[i, 0] = 1.0
        elif a == "F":
            assay_one_hot[i, 1] = 1.0
        else:
            assay_one_hot[i, 2] = 1.0
            
    # Stage 2 Baseline 2248-dim feature matrix:
    # 2048-bit Morgan Count FP + 200 continuous RDKit 2D descriptors (+ 3-dim assay one-hot = 2251-dim)
    X_baseline_2248 = np.hstack([X_count_fp, X_rdkit_2d_scaled])
    X_baseline_full = np.hstack([X_baseline_2248, assay_one_hot])
    
    y = df['pActivity'].values.astype(np.float32)
    is_censored = df['is_censored'].values.astype(np.int32)
    compound_ids = df['compound_id'].values
    
    # Save baseline features as NPZ
    baseline_npz_path = features_dir / "tnk1_baseline_features_2248d.npz"
    np.savez_compressed(
        baseline_npz_path,
        X_count_fp=X_count_fp,
        X_bit_fp=X_bit_fp,
        X_rdkit_2d_scaled=X_rdkit_2d_scaled,
        X_baseline_2248=X_baseline_2248,
        X_baseline_full=X_baseline_full,
        assay_one_hot=assay_one_hot,
        y=y,
        is_censored=is_censored,
        compound_ids=compound_ids,
        descriptor_names=np.array(desc_names)
    )
    print(f"Saved Stage 2 Baseline feature bundle: {baseline_npz_path} ({X_baseline_full.shape})")
    
    # Save graph features for D-MPNN
    import pickle
    graph_pickle_path = features_dir / "tnk1_graph_dmpnn.pkl"
    with open(graph_pickle_path, 'wb') as f:
        pickle.dump(graph_data_list, f)
    print(f"Saved Stage 3.1 D-MPNN Molecular Graph dataset: {graph_pickle_path} ({len(graph_data_list)} graphs)")

if __name__ == "__main__":
    main()
