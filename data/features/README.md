# TNK1 AI Pipeline: Complete Feature Store & Data Assets

This directory contains the finalized, standardized, and hash-locked datasets and feature matrices built in accordance with the **TNK1 AI Pipeline Technical Manual** (Stages 0 through 5, and Stage 8 Data Contract).

All datasets are locked with SHA-256 checksums recorded in [manifest.json](manifest.json).

**Verify them before quoting any result that depends on them:**

```bash
make verify-features
# or: python -m tnk1.utils.verify_features [filename ...]
# exit 0 only if every manifest entry is present and its sha256 matches
```

These files are gitignored (~280 MB), so a fresh clone reports most of them as `absent`;
that means the stages depending on them cannot run, and is not a pass. If the store lives
outside the repo, point the pipeline at it with `$TNK1_FEATURES_DIR`.

---

## 1. TNK1 Target Dataset & Frozen 5×5 Nested Scaffold Splits (Stages 1 & 2)
- **[tnk1_curated.parquet](tnk1_curated.parquet)** / **[tnk1_curated.csv](tnk1_curated.csv)**:
  - Curated, standardized via RDKit 6-step protocol (Stage 1.2).
  - Triplet deduplicated: `(InChIKey_block1, uniprot_id, assay_type)` $\to$ median $\text{pActivity} = 9 - \log_{10}(\text{Value}_{\text{nM}})$, $\sigma_{\text{rep}}$, and replicate range (Stage 1.3).
  - **380 unique (compound, assay_type) pairs** across **285 unique Bemis-Murcko scaffolds**:
    - **172 exact quantitative measurements** ($IC_{50}, K_i, K_d$ with relation `=`).
    - **208 right-censored weak negatives** ($c_{ik} = 1$, for Stage 3.4 asymmetric censored loss).
  - Tankyrase/PARP assay artifacts (312 records) cleanly excluded.
- **[tnk1_nested_scaffold_splits_5x5.json](tnk1_nested_scaffold_splits_5x5.json)**:
  - 5 outer test folds $\times$ 5 inner validation folds partitioned strictly by Murcko scaffolds (zero scaffold leakage).
  - Executed across the 5 deterministic seeds mandated in Step 0.2 (`[42, 101, 202, 303, 404]`).

---

## 2. Stage 2 Baseline Molecular Features (2248-dim)
- **[tnk1_baseline_features_2248d.npz](tnk1_baseline_features_2248d.npz)**:
  - `X_count_fp`: Shape `(380, 2048)` - 2048-bit Morgan Count Fingerprints ($r=2$, ECFP4) preserving substructure frequency.
  - `X_bit_fp`: Shape `(380, 2048)` - 2048-bit binary Morgan presence fingerprints.
  - `X_rdkit_2d_scaled`: Shape `(380, 200)` - 200 continuous RDKit 2D physicochemical descriptors standardized with `RobustScaler`.
  - `X_baseline_2248`: Shape `(380, 2248)` - `X_count_fp` concatenated with `X_rdkit_2d_scaled` (used by Random Forest & XGBoost).
  - `assay_one_hot`: Shape `(380, 3)` - One-hot encoding of `assay_type` ($IC_{50}, K_i, K_d$).
  - `X_baseline_full`: Shape `(380, 2251)` - 2248d + assay one-hot.
  - `y`: Shape `(380,)` - Logarithmic potency $\text{pActivity}$.
  - `is_censored`: Shape `(380,)` - Binary flag ($1$ for weak negatives, $0$ for exact values).

---

## 3. Stage 3.1 Directed Graph Featurization (D-MPNN)
- **[tnk1_graph_dmpnn.pkl](tnk1_graph_dmpnn.pkl)**:
  - 380 molecular graphs formatted for Chemprop v2 / PyTorch Geometric message passing.
  - **Atom feature vectors** $x_v \in \mathbb{R}^{133}$: One-hot atomic number, total degree (0–5), formal charge ($-2$ to $+2$), chiral tag, bonded hydrogens (0–4), hybridization ($sp, sp^2, sp^3, sp^{3d}, sp^{3d2}$), aromaticity, and scaled atomic mass.
  - **Directed bond feature vectors** $e_{vw} \in \mathbb{R}^{14}$: Bond type, conjugation flag, ring membership, and stereochemistry ($E, Z, cis, trans$).
  - **Reverse edge indices**: Explicit mappings to implement directed message updates $m_{vw}^{t+1} = \sum_{k \in \mathcal{N}(v) \setminus \{w\}} h_{kv}^t$ without reverse message tottering.

---

## 4. Stage 3.3 Kinome Multi-Task Transfer Learning Dataset
- **[kinome_multitask_300k.parquet](kinome_multitask_300k.parquet)** (14.9 MB) / **[kinome_multitask_300k.csv](kinome_multitask_300k.csv)**:
  - **425,906 compound–kinase pairs** curated across **292 human kinases** and **179,914 unique molecules**.
  - 277,713 exact measurements + 148,193 censored weak negatives.
  - Pre-computed Stage 3.4 sample importance weights $w_{ik}$ and composite similarity scores $S_k$.

---

## 5. Stage 3.4 Kinome Pocket Alignment & Importance Weighting
- **[kinome_similarity_scores.parquet](kinome_similarity_scores.parquet)** / **[kinome_similarity_scores.csv](kinome_similarity_scores.csv)**:
  - 491 human kinases aligned against TNK1's canonical 85-residue KLIFS pocket:
    `ELLGSGCFGVVHRVAVKSLDFLREVSVMMNLEPHVLRLHGLQMVMELAPLGSLHARLTAYLGARGLVHRDLATRNLLLVADFGLV`
  - Sequence identity $\text{Sim}_{\text{KLIFS}}(k, \text{TNK1})$ and catalytic domain similarity $\text{Sim}_{\text{domain}}(k, \text{TNK1})$.
  - Composite score: $S_k = 0.7 \cdot \text{Sim}_{\text{KLIFS}} + 0.3 \cdot \text{Sim}_{\text{domain}}$.
    - TNK2 (ACK1) ranks highest ($S_k = 0.704$), followed by ABL2 ($0.554$), SRC ($0.547$), PTK2B ($0.545$), FGR ($0.539$), LCK ($0.539$), ABL1 ($0.538$), PTK2 ($0.529$), and YES1 ($0.522$).
  - Sample importance weights $w_{ik} = (1 + \lambda S_k^2) \cdot \min(1.0, \sqrt{2000 / N_k})$ for $\lambda \in \{0, 1, 3, 10\}$.

---

## 6. Stage 5.1 Off-Target Selectivity Panel ($\mathcal{O}$)
- **[offtargets_curated.parquet](offtargets_curated.parquet)** (4.2 MB) / **[offtargets_curated.csv](offtargets_curated.csv)**:
  - **63,657 deduplicated compound–kinase pairs** across the 8 critical off-target kinases:
    - **TNK2 (ACK1)**: 1,350 pairs (1,296 unique compounds)
    - **SRC**: 6,524 pairs (6,181 unique compounds)
    - **ABL1**: 5,420 pairs (5,298 unique compounds)
    - **LCK**: 4,739 pairs (4,632 unique compounds)
    - **PTK2 (FAK)**: 3,882 pairs (3,824 unique compounds)
    - **EGFR**: 19,085 pairs (17,691 unique compounds)
    - **KDR (VEGFR2)**: 15,756 pairs (15,212 unique compounds)
    - **FLT3**: 6,901 pairs (6,802 unique compounds)
  - Enables simultaneous multi-target prediction and member-level selectivity margin computation:
    $$\Delta S_m(x) = \mu_m(x, \text{TNK1}) - \max_{k \in \mathcal{O}} \mu_m(x, k)$$

---

## 7. Stage 5.2 Multi-Task ADMET Predictor Suite
- **[admet_7endpoints.parquet](admet_7endpoints.parquet)** (2.5 MB) / **[admet_7endpoints.csv](admet_7endpoints.csv)**:
  - **37,452 standardized molecules** across the 7 TDC benchmark endpoints:
    - `aqsol_logS`: 6,856 compounds (regression)
    - `caco2_logPapp`: 776 compounds (regression)
    - `clearance_microsome_logCLint`: 1,086 compounds (regression)
    - `cyp3a4_inhibition`: 11,842 compounds (binary classification)
    - `cyp2d6_inhibition`: 12,608 compounds (binary classification)
    - `herg_blocker`: 10,867 compounds (binary classification)
    - `ames_mutagenic`: 5,243 compounds (binary classification)

---

## 8. Stage 5.3 & Page 10 Pipeline Code Infrastructure
- **[standardizer.py](../src/standardizer.py)**: Full 6-step chemical standardizer (LargestFragmentChooser, Normalizer, Reionizer, Uncharger, TautomerEnumerator, MedChem filter).
- **[medchem_filters.py](../src/medchem_filters.py)**: Evaluates QED, Ertl SA score, PAINS (A/B/C), Brenk, NIH, and Hinge-Binder SMARTS.
- **[data_contract.py](../src/data_contract.py)**: Team A $\leftrightarrow$ Team B interface contract validation and batch generator.

---

## 9. Full Kinome Measurements Dataset (All 39.55 Lakh Clean Assays)
- **[kinome_all_measurements_3.95M.parquet](kinome_all_measurements_3.95M.parquet)** (185.5 MB) / **[kinome_all_measurements_summary.json](kinome_all_measurements_summary.json)**:
  - **4,798,165 total clean bioactivity records** encompassing all cross-source deduplicated, human-target kinase measurements across ChEMBL, BindingDB, PubChem, and DTC:
    - **Exact dose-response**: 946,262 quantitative records ($IC_{50}, K_i, K_d$).
    - **Censored weak negatives**: 408,357 records ($IC_{50} \ge 10\,\mu\text{M}$ or $c_{ik} = 1$).
    - **Single-point percentage screens**: 858,491 records (e.g. $\%$ inhibition at $1\,\mu\text{M}$ or $10\,\mu\text{M}$).
    - **Qualitative outcomes**: 2,365,725 records (active/inactive flags, functional screens).
  - Includes target annotations, relatedness tiers, $S_k$ similarity scores, sample weights $w_{ik}$, and flags for TNK1 and Stage 5 off-targets.

---

## 10. Full Kinome Unique Molecules Feature Store (7.88 Lakh Compounds)
- **[unique_molecules_788k.parquet](unique_molecules_788k.parquet)** (61.4 MB):
  - **787,283 standardized unique chemical structures** extracted from the kinome master database.
  - Columns: `inchikey14`, `canonical_smiles`, `inchikey`, `murcko_scaffold`, `qed`, `sa_score`, `has_alerts`, `has_hinge_motif`, `mol_weight`, `heavy_atom_count`.
- **[unique_molecules_fingerprints_2048bit.npz](unique_molecules_fingerprints_2048bit.npz)** (42.6 MB):
  - `packed_fps_256bytes`: Shape `(787283, 256)` of type `uint8`.
  - Exactly 2048 bits per compound packed bitwise into 256 bytes (`np.packbits`), enabling lightning-fast kinome-wide Tanimoto similarity searching and diversity selection for Stage 7 RL generator experience replay.
- **[unique_molecules_descriptors_200d.npz](unique_molecules_descriptors_200d.npz)** (236.7 MB):
  - `descriptors_200d`: Shape `(787283, 200)` of type `float32`.
  - 200 continuous RDKit 2D physicochemical descriptors computed for every unique kinome molecule.
  - `descriptor_names`: Array of 200 feature names.

---

## 11. Master File Manifest & SHA-256 Checksums
Recorded in [manifest.json](manifest.json):

| Asset File | Size | SHA-256 Checksum | Description |
| :--- | :--- | :--- | :--- |
| `tnk1_curated.parquet` | 57.7 KB | `e1de4cb2445d1f478bb33114b71d6fe94105a1923ed31fe1e8c232b0332c24ec` | 380 curated TNK1 compound-assay pairs |
| `tnk1_nested_scaffold_splits_5x5.json` | 823.5 KB | `7ed825f357dbc9d69ac91aac843984bde6862ce8864c587fbb97fc835781bf6b` | Frozen 5×5 nested scaffold splits (seeds 42, 101, 202, 303, 404) |
| `tnk1_baseline_features_2248d.npz` | 694.2 KB | `275663144ac36479f21bcb0fd563e4d87ecd775f4ad17fd67f83bbf0a5e6dec7` | 2048 Count FP + 200 RDKit continuous + assay one-hot |
| `tnk1_graph_dmpnn.pkl` | 8.32 MB | `65aaed21449ce8e6456e292a1e883144a075e9c4cb5e8bc31fc6337c87451913` | Chemprop D-MPNN molecular graphs (133d atom, 14d bond) |
| `kinome_similarity_scores.parquet` | 65.2 KB | `3b1aef654bbb0b763c215b493f450e469e94b9ee0c5b6b02fcc80b80080852b7` | KLIFS 85-pocket alignment & sample weights for 491 kinases |
| `offtargets_curated.parquet` | 4.08 MB | `54d441ffa41f589b22f42ba0672934a0def09f318bbf947f3605e1530c9bc254` | 63,657 pairs across 8 key off-target kinases |
| `admet_7endpoints.parquet` | 2.47 MB | `5008dea536b424baa7c01a50d3db486314aec9b0d51d2d743fc7f36bb9350a41` | 37,452 compounds across 7 TDC ADMET benchmarks |
| `kinome_multitask_300k.parquet` | 14.25 MB | `55fa58097c2bca58612f62055c1682f3cd9307478a64f4b8aa80229346189610` | 425,906 compound-kinase pairs across 292 kinases |
| `kinome_all_measurements_3.95M.parquet` | 185.52 MB | `16619257e8d7bf4774d2ee59f227323b7f5a7eff3c7a9bf4cde87ddda37549c7` | 4,798,165 full kinome bioactivity measurements |
| `unique_molecules_788k.parquet` | 61.37 MB | `9391e01b8e713157ade34b7218fba28e6d2f2ce8024149a2d4fa0d7aabb13dfd` | 787,283 unique standardized kinome molecules & metadata |
| `unique_molecules_fingerprints_2048bit.npz` | 42.58 MB | `cb71026e1ee7dba45b2c8c13ba267b85139154bb75f4cf4e3b2dd568d9da7e3a` | (787283, 256) uint8 packed 2048-bit Morgan fingerprints |
| `unique_molecules_descriptors_200d.npz` | 236.66 MB | `2f61735fe3cf24e4d48c3e2d9449165c048a6656d2e34257b0a57a66c6eb641b` | (787283, 200) float32 continuous RDKit 2D descriptors |

