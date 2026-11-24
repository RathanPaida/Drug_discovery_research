# The second data drop: what was promoted, what was not, and why

The drop arrived as `NEW_DATA/`, 37 GB: eleven zip parts plus three oversized files
Google Drive exported separately. It is a **second, independent acquisition project**
(`tnk1_acquire/`) that set out to rebuild every tier of this project's data from primary
sources, and to close the two gaps `DATA_PROVENANCE.md` records — the `data/features/`
assembly step and the generative-anchoring data.

Nothing is missing from the download. The standalone files are numbered `-003`, `-006`
and `-014`, which are exactly the absent zip part numbers:

| standalone file | is | size |
| :--- | :--- | ---: |
| `kinome_features-003.npz` | `features/kinome_features.npz` | 2.1 GB |
| `chembl_37_sqlite-006.tar.gz` | `external/chembl_37/chembl_37_sqlite.tar.gz` | 5.5 GB |
| `chembl_37-014.db` | that tarball, extracted | 29 GB |

All eleven zips pass `testzip()`; 141 files, no corruption.

---

## 1. The kinome table was unusable as shipped

`processed/kinome_measurements_v3.parquet` (302,463 rows) is the drop's kinome-wide
multi-task set. **It has no target column.** 460 kinases went in and none came out.

One line in the drop's `tnk1_acquire/curate.py` caused it:

```python
key = tuple(k for k in key if k in df.columns)
```

`aggregate_replicates` defaults to `key=("inchikey", "target_gene", "endpoint")`. On the
ChEMBL-SQLite path the frame carries `accession`, and `tier3` only maps
`accession → target_gene` **after** aggregation. So the key silently degraded to
`("inchikey", "endpoint")` and every kinase for a compound collapsed into one median.
The follow-up `if "accession" in agg.columns` was then also false — groupby had dropped
it — so `target_gene` never got set either. No exception, no warning, no row-count check.

What it cost, measured against `interim/kinome_raw_union.parquet`:

| | |
| :--- | ---: |
| duplicate `(inchikey, endpoint, censored)` rows in the shipped file | **0** |
| rows on the correct key | 383,303 |
| rows actually produced | 302,463 |
| distinct `(compound, kinase, endpoint)` cells merged away | **21%** |
| compounds with genuinely different potencies across kinases | 82,764 |
| mean within-compound range across kinases | 1.28 log units |
| 95th percentile / max | 3.15 / 6.60 |
| compounds measured on more than one kinase | 30.6% |
| most kinases for one compound | 406 |

A compound at 9.0 on TNK1 and 5.0 on forty other kinases is recorded as roughly 5.2.
That is precisely the selectivity signal `r_sel` and Stage 5 consume. It is the same
class of defect as the v2 `pActivity == 0.0` poisoning that `tier3`'s own docstring
boasts of preventing — a file that looks like a result.

**Recoverable**, because `interim/kinome_raw_union.parquet` keeps `accession` on all
617,200 rows. `scripts/build_kinome_chembl37.py` re-derives the table from it through
*this* repository's `standardize` / `activity` modules, so the output is directly
comparable with the frozen v3 table. Our `aggregate_replicates` hardcodes the key and
raises on a missing column, so it cannot fail the same way;
`tests/test_kinome_merge.py` pins that.

## 2. The new kinome data is mostly redundant

| | rows | compounds | kinases | sources |
| :--- | ---: | ---: | ---: | :--- |
| frozen v3 | 1,103,498 | 331,620 | 448 | ChEMBL, PubChem, BindingDB, DAVIS, IUPHAR |
| ChEMBL-37 harvest | 302,463 | 253,883 | 0 (lost) | ChEMBL 37 only |

**90.6% of its compounds are already in the 3.95M store.** It adds 23,963; the old store
has 101,700 it lacks. The harvest's other six sources returned nothing —
`logs/harvest_report.csv` records BindingDB, DTC and Papyrus as unreachable and both TDC
loaders as `No module named 'tiledbsoma'`.

It is still worth unioning: ChEMBL 37 is a newer release, and 24k new compounds plus
fresher measurements on overlapping ones is a real if modest gain. It is **not** the
step-change the row counts suggest at first glance.

Because the sources overlap, Stage 1 now dedupes **across** them on
`(inchikey_block1, uniprot_id, standard_type)` — nothing did before, because there was
only ever one source. Resolution: exact beats censored, then more replicates, then
tighter spread. Rows with a blank identity or non-finite `pact` are dropped first, since
a blank key collides with every other blank key.

## 3. The TNK1 data is a regression — do not swap it in

`processed/tnk1_curated.parquet` has **235** TNK1 points against the existing **380**
(172 exact). TNK1 scarcity is this project's binding constraint, so a smaller TNK1 set is
strictly worse. The existing curated set is kept.

## 4. What was promoted

| promoted | why |
| :--- | :--- |
| `interim/kinome_raw_union.parquet` (11.5 MB) | the only file with the kinase intact; input to `build_kinome_chembl37.py` |

That is all that this run needs. Everything else is either redundant, a build input, or
a separable upgrade.

## 5. What was deliberately left off the cluster

| left behind | size | why |
| :--- | ---: | :--- |
| `chembl_37.db` | 29 GB | build input. Nothing on the GPU node reads it; the harvest is already distilled into `kinome_raw_union.parquet` |
| `chembl_37_sqlite.tar.gz` | 5.5 GB | the same database, compressed |
| `kinome_features.npz` | 2.1 GB | keyed on `canonical_smiles` only, and derived from the broken aggregate |
| `models/aizynth/` | 614 MB | retrosynthesis bundle; no stage in this chain calls it |
| `kinome_measurements_v3.parquet` | 10.7 MB | the defective table itself |
| `.git/`, `__pycache__` | — | not data |

The quota mattered: **232 GB of 300 GB** was already in use, leaving 67 GB. The total
added for this run is about **206 MB**.

## 6. The clear next upgrade: the 2.7M-molecule corpus

`processed/unique_molecules.parquet` holds **2,718,348** molecules against the current
`unique_molecules_788k.parquet` — a **3.4×** RL prior corpus, with matching packed
fingerprints in `features/unique_molecules_fp2048.npy` (696 MB).

This is the single largest genuine gain in the drop, and it was **not** taken in this
pass because it is not a drop-in:

- the filename is referenced across Stage 1's scaffold map, Stage 6's prior, the novelty
  scan and `run_stage0_check.py`'s per-stage required-file lists;
- the fingerprints ship as `.npy` plus a gzipped keys file, where the pipeline expects a
  single `.npz` keyed store;
- the column `max_tanimoto_kinome788k` appears in results, figures and the
  `finalize_candidates.py` gate, so the name becomes wrong as well as the data;
- every affected `manifest.json` sha256 changes.

Worth doing as its own change, with its own regression tests, rather than smuggled into a
data merge.

## 7. The v4 result — and what it revealed about v3

```
kinome_long_v4.parquet   sha256 8fba70ebb270a2f1ae65634173732cf2e5186837eef740241f3a3a2400b1fe07
756,714 rows | 358,210 compounds | 467 kinases | 558,314 exact | 198,400 censored
```

| | v3 | v4 |
| :--- | ---: | ---: |
| rows | 1,103,498 | 756,714 |
| **distinct (compound, kinase, endpoint) cells** | **704,375** | **756,714** |
| rows sitting in a duplicated cell | **593,214 (53.8%)** | **0** |
| compounds | 331,620 | 358,210 |
| kinases | 448 | 467 |

Rows per source in v4: CHEMBL 522,416 · **CHEMBL37 106,673** · BINDINGDB 75,764 ·
PUBCHEM 50,035 · DAVIS 1,056 · IUPHAR 770.

**v4 has fewer rows than v3 and strictly more data.** The row count fell because v3 was
double-counting, not because anything was lost:

- **53.8% of v3's rows sat in a duplicated `(compound, kinase, endpoint)` cell.** Only
  704,375 of its 1,103,498 rows were distinct cells. Stage 1 formatted the 3.95M store's
  per-measurement rows straight into the frozen table without aggregating them, so the
  same cell appeared repeatedly.
- In **55,087** of those cells the copies **disagree by more than 0.5 log units** — within-cell
  spread 95th percentile **2.32**, max **8.38** log units. The model was shown the same
  compound-kinase pair with contradictory labels and `task_loss` averaged the
  contradiction.
- Cell-for-cell, v4 **gains 52,345** cells and **loses 2,558**. The losses are deliberate:
  cells whose exact replicates spanned more than `replicate_range_max: 1.0` are dropped
  by `replicate_policy: drop`, which is the configured behaviour and the same rule v3's
  TNK1 rows went through.

This is worth weighing against the v3 run report's two headline negatives — that the
deep ensemble was the worst model in the repository (Spearman 0.501 vs Random Forest's
0.686) and that its uncertainty was uninformative (error–uncertainty correlation
**−0.021**). A table in which half the rows are duplicated cells, tens of thousands of
them carrying mutually inconsistent labels, inflates ensemble variance without that
variance tracking error — which is the exact shape of the reported defect. It does not
make the negative result wrong, and the TNK1 scarcity (172 exact points) is untouched by
any of this. But v3 was not a clean test of the architecture, and v4 is a cleaner one.

Measured with:

```bash
python - <<'EOF'
import pandas as pd
K=["inchikey_block1","uniprot_id","standard_type"]
v3=pd.read_parquet("data/processed/kinome_long_v3.parquet",columns=K+["pact"])
d=v3.duplicated(subset=K,keep=False)
print(f"{int(d.sum()):,} of {len(v3):,} rows are in a duplicated cell")
g=v3[d].groupby(K,observed=True)["pact"]; rng=g.max()-g.min()
print(f"cells disagreeing >0.5 log: {int((rng>0.5).sum()):,}  max spread {rng.max():.2f}")
EOF
```

## 8. Rebuilding

```bash
# on the cluster, as one CPU job (no GPU shards held)
sbatch hpc/build_data_v4.sh

# or by hand
python scripts/build_kinome_chembl37.py \
    --raw data/raw/kinome_raw_union.parquet \
    --out data/features/kinome_chembl37.parquet --jobs 32
python scripts/run_stage1_data.py --version v4
```

`build_kinome_chembl37.py` writes a `.build.json` sidecar recording every attrition step,
so the row count at the end is accountable to the 617,200 it started from.
