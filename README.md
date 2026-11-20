# TNK1 data acquisition

Rebuilds every tier of the TNK1 project's data from primary sources, plus the two
things that were missing from the repo: the `data/features/` assembly step and the
generative-anchoring data for designing around known TNK1/TNK2 chemotypes.

```bash
pip install -r requirements.txt
python acquire.py --list
```

## Run order

`tier4` before `tier3` (tier 4 defines which accessions count as the kinome, and
tier 3 filters on that list). `tier1`+`tier2` before `tier9`. Everything else is
independent.

```bash
# 1. the cheap, highest-value core -- do this first, it takes ~45 min total
python acquire.py tier4 tier1 tier2

# 2. the big pulls, overnight
python acquire.py tier3 tier5

# 3. the rest
python acquire.py tier6 tier7 tier8

# 4. derived data
python acquire.py tier9 features verify
```

Or `python acquire.py --all`. Every stage is idempotent — re-running skips files
already on disk unless you pass `--force`.

## What each tier gets

| stage | content | output |
|---|---|---|
| `tier1` | TNK1 bioactivity: ChEMBL REST + BindingDB web service + PubChem BioAssay + DTC | `processed/tnk1_curated.parquet` |
| `tier2` | 8-kinase off-target panel (TNK2/ACK1, SRC, ABL1, LCK, PTK2, EGFR, KDR, FLT3) | `processed/offtargets_curated.parquet` |
| `tier3` | Kinome-wide multi-task set from the ChEMBL SQLite release + BindingDB full TSV + DTC + Papyrus | `processed/kinome_measurements_v3.parquet` |
| `tier4` | 491-ish kinases with UniProt ID, gene symbol and the 85-residue KLIFS pocket | `raw/kinases.csv` |
| `tier5` | Unlabelled corpus from `chembl_NN_chemreps.txt.gz` + packed 2048-bit Morgan fingerprints | `processed/unique_molecules.parquet`, `features/unique_molecules_fp2048.npy` |
| `tier6` | Seven TDC ADMET endpoints (+ a second hERG set) | `processed/admet_7endpoints.parquet`, `admet_coverage.json` |
| `tier7` | AlphaFold TNK1 model with pLDDT profile, PDB 4EWH, the rest of the ACK1/TNK2 X-ray ensemble, SRC/ABL1 references, PDBQT receptors | `structures/` |
| `tier8` | AiZynthFinder USPTO bundle (~1 GB) | `models/aizynth/` |
| `tier9` | Anchors, selectivity matrix, matched molecular pairs, catalogued analogues, kinase-inhibitor fine-tuning corpus | `processed/generative_*.parquet` etc. |
| `features` | `data/processed` → `data/features` (ECFP4, descriptors, D-MPNN graph tensors, pocket features) | `features/` |
| `verify` | SHA-256 of every artefact against `MANIFEST.json` | exit code |

## The things that were going wrong

**Tier 1 is the binding constraint, and no reward engineering fixes it.** The run
logs the exact-point count and says so explicitly when it is under 400. 172 exact
dose-response points over 285 scaffolds cannot support a D-MPNN → multi-task
transfer → fine-tune → 5-member ensemble chain; that is why the ensemble ranks at
ρ=0.264 while a Random Forest on ECFP gets 0.626. The deep model is fitting
scaffold identity. Until more assay data lands, the RF is the production ranker
and the assay budget goes to TNK1 dose-response.

`tier1` also writes a **scaffold split**, not a random one. With 285 unique
scaffolds over 380 compounds a random split gives almost every test compound a
near-twin in train, which inflates every number you report.

**The v2 poisoning cannot recur.** `curate.filter_measurable` raises
`PoisonedDataError` when more than 0.1% of rows have `p_activity == 0.0` exactly
— the signature of "no value" encoded as a real measurement of 1 M. There is no
code path that reads a v2 file. pActivity is only ever derived from a positive
`standard_value` with a convertible unit; an upstream pChEMBL value is a
cross-check and a units-missing fallback, never a source of invented numbers.
Censored points keep their relation and are aggregated separately from exact
ones.

**Zero-row sources are now loud.** DTC, KIBA and Metz each contributed nothing
last time and nobody found out. Every loader files a `HarvestReport`, and
`logs/harvest_report.csv` plus the end-of-run log lists each source, its row
count, why it was empty, and the exact command to fix it. The three specific
fixes:

- **DTC** — the bulk file is not reliably reachable; the per-UniProt REST API is,
  and that covers tiers 1–2. For kinome-wide coverage, download the bulk CSV by
  hand and pass `--dtc-csv`.
- **KIBA** — the `(2111, 230)` vs `(2111, 229)` mismatch is a positional
  alignment bug. The DeepDTA release stores drugs and targets as *dicts*; the
  affinity matrix has to be indexed by dict key order, and a single extra key
  shifts every column by one without raising. `otherdbs.deepdta_matrix` asserts
  the shape, transposes when that is the actual problem, and otherwise refuses
  and tells you the per-axis delta instead of reshaping until it fits. The TDC
  copy (already long-format) sidesteps it entirely. KIBA scores are **not**
  pActivity and are written to a separate file so they cannot be pooled with
  IC50/Ki.
- **Metz** — has no stable bulk endpoint; the primary release is a journal
  supplementary table. PubChem AID 1451 mirrors the same Ki panel and is
  fetchable.

**`data/features/` now has a builder**, and it is written to be deterministic so
the acceptance test can pass: explicit ordered descriptor tuple (not
`Descriptors.descList`, which is not stable across RDKit versions), explicit
fingerprint radius and size, rows sorted before featurisation, `np.savez` rather
than `savez_compressed` (zlib output varies across builds, which changes the file
hash while the arrays are identical). **Pin RDKit** — `requirements.txt` does. A
checksum mismatch on a feature file after an RDKit upgrade is expected and means
you re-record, not that the data is wrong.

**hERG is probably as much a model problem as a chemistry problem.** 29/40 above
0.20 is a lot. Tier 6 pulls `hERG_Karim` (~13k compounds) for training and the
~650-compound Wang set as an independent check, because a classifier trained on
650 compounds will be confidently wrong on kinase chemotypes. Before redesigning
chemistry against the score, check: which set it was trained on, the training
base rate against your 0.20 threshold (a balanced training set makes 0.20 not a
20% risk), and whether those 29 are even in the applicability domain by max
Tanimoto to training.

**Tier 7 builds an ensemble, not a model.** There is no TNK1 kinase-domain
crystal, which is why 2D predictions and docking correlate at r≈0.19. Docking
into one AlphaFold model is the weakest available estimator. The tier downloads
the AlphaFold model *with its pLDDT profile* (so you can see which pocket
residues the model is actually confident about), PDB 4EWH as the ACK1/TNK2
homology template, and every other TNK2 X-ray entry RCSB has, giving DFG-in/out
and activation-loop variety. Ensemble docking with per-family best scores is a
different and better estimator for the same compute.

**Tier 8's absence is why Cycle 01 had no synthesisability screen.** The BRICS
fallback is kept but labelled: it is a fragment check, not a route search, and
`brics_fragment_check` logs a warning every call saying so. A generator optimised
against a fragment check produces unmakeable molecules that score well.

## Tier 9: generating around the known chemotypes

This is the part the eight tiers did not cover. With 172 exact points you cannot
learn TNK1 SAR de novo, so the generator has to be anchored. The chain:

```
788k corpus prior  →  kinase-inhibitor subset  →  TNK1/TNK2 actives
(tier 5)              (~100k, from tier 3)        (hundreds, tier 1/2)
```

That middle stage is what was missing from the transfer chain — going straight
from drug-like ChEMBL to a few hundred TNK1 actives is a four-order-of-magnitude
jump in dataset size.

Anchors are **derived from the curated data at run time**, never hardcoded, so
they cannot drift from what the assays say: confirmed TNK1 actives (uncensored,
p ≥ 6) plus potent TNK2/ACK1 actives (p ≥ 7). The TNK2 compounds are in
deliberately — they are the richest source of chemotypes that bind a TNK-family
pocket, and you need them in the training data in order to learn to move *away*
from them in a controlled way.

Outputs:

- `generative_anchors.parquet` — the anchor set with Murcko scaffolds.
- `selectivity_matrix.parquet` — compounds wide over the panel, with
  `sel_TNK2 = p_TNK1 − p_TNK2`. **This is the only data that directly supervises
  the thing you are being asked to optimise.** Generating against predicted TNK1
  potency alone walks straight into TNK2. Censored off-target values are kept as
  bounds: TNK1 p=8.0 with TNK2 `<5` is "at least 3 logs selective", which is
  usable supervision, and discarding it is how a small selectivity set becomes
  an empty one.
- `anchor_neighbours.parquet` — catalogued ChEMBL/PubChem analogues of each
  anchor, flagged for whether they already have panel data. These are real,
  often purchasable compounds: the honest baseline any generated molecule has to
  beat, and the ones already tested are free validation.
- `matched_pairs_tnk1.parquet` — single-cut matched molecular pairs with the
  activity delta. This is where "which R-group swap buys potency without buying
  hERG" actually lives.
- `kinase_inhibitor_corpus.parquet` — the intermediate fine-tuning corpus, with a
  `promiscuity` column (number of kinases hit) that doubles as a selectivity-risk
  prior.

## Layout

```
data/
├── acquire.py              CLI
├── tnk1_acquire/           package
├── raw/kinases.csv         tier 4
├── external/<source>/      untrusted downloads, one dir per source
├── interim/                pre-curation unions
├── processed/              curated parquets
├── features/               model-ready arrays + feature_spec.json
├── structures/             pdb / pdbqt / alphafold
├── models/aizynth/         tier 8
├── logs/                   acquire.log, harvest_report.csv
└── MANIFEST.json           sha256 + provenance for every artefact
```

Downloaded archives go in their own directory per source under `external/`, away
from the code, and are extracted with path-traversal checks.

## Acceptance test

```bash
python acquire.py verify
```

Exits non-zero on any checksum mismatch or missing file, and names each one. A
mismatch on a derived feature file means the featuriser stopped being
deterministic — usual causes are an RDKit version change, an edited `DESCRIPTORS`
tuple, `savez_compressed`, or rows not sorted before featurisation.

## Still needs a human

- **TNK1 dose-response data.** Not obtainable by downloading harder. This is the
  one thing the 40-compound contract to Team B exists for, and it is the single
  highest-value acquisition in the project.
- **A TNK1 kinase-domain crystal structure.** Does not exist publicly. Tier 7's
  ensemble is a workaround, not a substitute.
- **DTC bulk CSV**, if the API route is insufficient — manual download.
- **Enamine REAL / other commercial catalogues** for purchasable analogue
  expansion — licensed, cannot be scripted here.
