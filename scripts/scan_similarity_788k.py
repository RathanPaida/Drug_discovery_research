#!/usr/bin/env python3
"""
Scan the full 787,283 unique kinase molecule feature store to compute the
exact maximum Tanimoto similarity to any known kinase inhibitor.
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Load contract
contract_df = pd.read_csv("tnk1-pipeline/tnk1-pipeline/runs/cycle_01/contract_novel_team_b.csv")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
cand_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in contract_df["canonical_smiles"]]

# Load bit-packed fingerprints
print("Loading 787,283 bit-packed kinase fingerprints...")
fp_npz = np.load("ml_research/features/unique_molecules_fingerprints_2048bit.npz")
packed = fp_npz["packed_fps_256bytes"]  # (787283, 256) uint8

cand_bits = np.array([[int(b) for b in fp.ToBitString()] for fp in cand_fps], dtype=np.uint8)
cand_packed = np.packbits(cand_bits, axis=1)

chunk_size = 50000
n_cpds = len(cand_packed)
n_db = len(packed)
max_sims = np.zeros(n_cpds)

print(f"Scanning {n_db:,} compounds in chunks of {chunk_size:,}...")
for start in range(0, n_db, chunk_size):
    end = min(start + chunk_size, n_db)
    chunk = packed[start:end]
    for i in range(n_cpds):
        c = cand_packed[i:i+1]
        inter = np.bitwise_count(c & chunk).sum(axis=1)
        union = np.bitwise_count(c | chunk).sum(axis=1)
        sim = inter / np.maximum(union, 1)
        m = float(np.max(sim))
        if m > max_sims[i]:
            max_sims[i] = m

print("\n" + "=" * 60)
print("MAXIMUM TANIMOTO SIMILARITY TO 787,283 KNOWN KINASE INHIBITORS")
print("=" * 60)
print(f"Mean max Tanimoto: {np.mean(max_sims):.3f}")
print(f"Range: [{np.min(max_sims):.3f}, {np.max(max_sims):.3f}]")
print(f"Max overall similarity to ANY known molecule: {np.max(max_sims):.3f}")
print(f"Compounds with Tanimoto >= 0.85 (close analogs): {sum(x >= 0.85 for x in max_sims)}")
print(f"Compounds with Tanimoto in [0.70, 0.84]: {sum(0.70 <= x < 0.85 for x in max_sims)}")
print(f"Compounds with Tanimoto in [0.50, 0.69]: {sum(0.50 <= x < 0.70 for x in max_sims)}")
print(f"Compounds with Tanimoto < 0.50: {sum(x < 0.50 for x in max_sims)}")

# Add to dataframe and save
contract_df["max_tanimoto_kinome788k"] = [round(float(x), 3) for x in max_sims]
contract_df.to_csv("tnk1-pipeline/tnk1-pipeline/runs/cycle_01/contract_novel_team_b.csv", index=False)
contract_df.to_csv("tnk1-pipeline/tnk1-pipeline/runs/cycle_01/contract_to_team_b.csv", index=False)
print("Updated contract files with max_tanimoto_kinome788k.")
