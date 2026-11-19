# Data provenance: where every number comes from

This answers one question: **for any artifact in the pipeline, what produced it, from
what, and can it be rebuilt?**

It exists because the answer used to be partly "nobody knows". The feature store was
consumed by Stages 0–8 and produced by no committed code; `PIPELINE_AUDIT.md` OPEN-4 named
a `build_full_kinome_measurements.py` that was not in the repository, and the harvester
that built the raw layer had run from a different machine
(`C:\Users\Abhinav Kumar\OneDrive\Desktop\ml_research\`, visible in
`data/raw/tnk1_data/manifest.json`). Provenance was sha256-verified but the *derivation*
was unreproducible: you could prove the bytes had not changed, and not regenerate them.

---

## The chain

```
  public databases (10 sources, network)
        │
        │  scripts/tnk1_harvest.py          ← RECOVERED, now committed
        ▼
  data/raw/tnk1_data/                       ~10 GB, gitignored
    TNK1_kinase_master.csv   (5.99 GB, all sources merged + dedup flags)
    csv/{CHEMBL,PUBCHEM,BINDINGDB,PKIS,PKIS2,DAVIS,IUPHAR,...}.csv
    REPORT.txt, manifest.json, counts_by_source_kinase.csv
        │
        │  ??? — THE ONE REMAINING GAP (see "Open gap" below)
        ▼
  data/features/            12 files, 584 MB, gitignored, sha256 in manifest.json
        │
        │  scripts/run_stage1_data.py   (+ tnk1.data.activity.filter_measurable)
        ▼
  data/processed/kinome_long_<version>.parquet  + .sha256 sidecar  ← FROZEN
        │
        │  stages 2-8
        ▼
  data/processed/{kinase_store.npz, ensemble/, calibrator.pkl, ad.pkl, admet_bundles.pt}
  runs/**
```

---

## 1. Raw layer — `scripts/tnk1_harvest.py`

A single resumable 2,601-line harvester over ten sources. Recovered from
`ml_research.zip` (which also holds the raw `tnk1_data/` tree) and now committed.

```bash
python scripts/tnk1_harvest.py --check                 # test every endpoint from your network
python scripts/tnk1_harvest.py                         # full run (hours; Ctrl+C safe, resumable)
python scripts/tnk1_harvest.py --panel core            # 14 kinases, quick
python scripts/tnk1_harvest.py --sources davis,kiba    # subset
python scripts/tnk1_harvest.py --postprocess-only      # rebuild master + report only
```

### The harvest that produced the current data was INCOMPLETE — this was never surfaced

From `data/raw/tnk1_data/REPORT.txt` (run 2026-10-26, panel = 491 kinases):

| Source | Status | Raw rows |
| :--- | :--- | ---: |
| CHEMBL | done | 1,891,686 |
| PUBCHEM | done | 2,936,519 (27 AIDs undeliverable) |
| BINDINGDB | done | 670,779 |
| PKIS | done | 165,715 |
| DAVIS | done | 30,056 (20,931 censored flagged) |
| IUPHAR | done | 3,453 |
| PKIS2 | done | 491 |
| **DTC** | **failed** | **0** — no public bulk file/API reachable; needs a manual download |
| **KIBA** | **failed** | **0** — matrix shape (2111, 230) != (2111, 229) |
| **METZ** | returned 0 | **0** |

**8 of 11 sources succeeded.** Totals: 5,698,495 raw rows → **3,955,550 unique
measurements** over 787,987 unique compounds.

`README.md` and the master document describe this as "4.80M bioactivities". The 4.80M
figure is the row count of `kinome_all_measurements_3.95M.parquet` (4,798,165), which is
not the same quantity as unique measurements, and neither number mentions that two sources
contributed nothing. Stage 0 now prints the per-source status at every run so this cannot
stay invisible.

---

## 2. Feature store — `data/features/`, 12 files, 584 MB

Gitignored; only `manifest.json` and the summary JSONs are committed. Verify before
quoting anything that depends on them:

```bash
make verify-features        # exit 0 only if every entry is present and its sha256 matches
```

| File | Size | Consumed by |
| :--- | ---: | :--- |
| `kinome_all_measurements_3.95M.parquet` | 194.5 MB | Stage 1 (`data.full_kinome: true`) |
| `unique_molecules_descriptors_200d.npz` | 248.2 MB | `DescStore` preload |
| `unique_molecules_788k.parquet` | 64.4 MB | RL prior corpus, Stage 3 SSL, novelty scan |
| `unique_molecules_fingerprints_2048bit.npz` | 44.6 MB | novelty Tanimoto scan |
| `kinome_multitask_300k.parquet` | 14.9 MB | Stage 1 (`full_kinome: false`) |
| `tnk1_graph_dmpnn.pkl` | 8.7 MB | Stage 3 featurisation |
| `offtargets_curated.parquet` | 4.3 MB | Stage 5 selectivity |
| `admet_7endpoints.parquet` | 2.6 MB | Stage 5 ADMET (substitutes for the TDC download) |
| `tnk1_nested_scaffold_splits_5x5.json` | 0.8 MB | Stage 2 |
| `tnk1_baseline_features_2248d.npz` | 0.7 MB | Stage 2 |
| `kinome_similarity_scores.parquet` | 65 KB | Stage 3 kinase store, Stage 1 kinase table |
| `tnk1_curated.parquet` | 58 KB | Stage 1 TNK1 rows, novelty audit |

### Open gap: nothing in the repository rebuilds these 12 files

The harvester produces `tnk1_data/`; Stage 1 consumes `data/features/`. The step between
them is still missing. Everything needed to write it is present — the raw master is on
disk and the derivations are implemented in `tnk1.data.standardize`,
`tnk1.data.activity`, `tnk1.baselines.features` and `tnk1.models.graph` — so the work is
assembly, not research.

Acceptance test for whoever writes `scripts/build_feature_store.py`: the regenerated files
must reproduce the sha256 values already in `data/features/manifest.json`. That is the only
check that proves the derivation is the one that actually produced the published store.

Until then, treat `data/features/` as a verified-but-unreproducible input, and do not
delete `ml_research.zip` (9.7 GB) — it is the only copy of the raw tree plus harvester.

---

## 3. Frozen table — `data/processed/kinome_long_<version>.parquet`

Written by `scripts/run_stage1_data.py` via `tnk1.utils.common.save_locked`, which also
writes a `.sha256` sidecar and **refuses to overwrite**. Every downstream artifact is
attributed to this digest; `scripts/run_pipeline.py` compares it against each stage's
recorded provenance and reports `STALE` on a mismatch.

Stage 1 used to `unlink()` both the parquet and the sidecar immediately before calling
`save_locked`, which defeated the freeze completely: a re-run replaced the table and wrote
a new matching digest, so `verify_locked` still passed and nothing recorded that the data
had changed under every published number. Replacing a frozen table now requires `--force`
and prints the old and new digests.

### The three versions are not interchangeable

| Version | Rows | Status |
| :--- | ---: | :--- |
| `v1` | 426,286 | Pre-filter, built from `kinome_multitask_300k.parquet`. Every published Cycle-01 result comes from this. Clean (0 sentinel rows). |
| `v2` | 4,796,129 | **Do not use.** 3,372,521 rows (70.3%) carry `pact == 0.0` with `censored == 0`. `Ctx` refuses to open it. |
| `v3` | 1,103,498 | First table built with the measurability filter. |

**What v2's contamination was.** The kinome store encodes "no dose-response value" as
`pActivity` exactly **0.0**, not NaN. Stage 1 filtered with `keep = np.isfinite(pact)`,
and `np.isfinite(0.0)` is `True`, so the filter removed none of them — while its comment
asserted they were "dropped with a printed count". Of 4,798,165 source rows:

- **3,374,532 (70.3%)** had `pActivity == 0.0`, including *every one* of the 3,338,837
  non-dose-response rows (`OUTCOME` 1.80M, `PCT_INHIBITION` 771k, `kon` 258k, `koff` 258k,
  `Residual Activity` 70k, `Activity` 60k, `PCT_CONTROL` 58k, `PCT_ACTIVITY` 29k). With
  `censored == 0` these reach `task_loss` as *exact* measurements of an IC50 of **1 molar**.
- **108,450** had `0 != pActivity < 2.0`, i.e. `value_nM` from 1.0e7 to 1.3e20 (median
  3.2e14 nM = 316,000 M). 107,795 are PubChem `Ki` rows whose value was not in nM.
- **211,546** named a semicolon-joined set of up to ~150 UniProt accessions (642 distinct
  such strings) rather than one kinase. These were already being discarded silently by
  `kinome_stage`'s `isin(kinase_store ids)`, which is why `uniprot_id.nunique()` reported
  1054 "kinases" when only 491 are real.

`tnk1.data.activity.filter_measurable` removes all three with per-reason counts, governed
by `data.dose_response_types`, `data.pact_min`, `data.pact_max` and
`data.drop_composite_targets`. `Ctx._assert_targets_usable` refuses any table still over
1% contaminated.

---

## 4. Compound identity

`tnk1.data.standardize.standardize_smiles` defines it: a six-step RDKit protocol whose
`inchikey_block1` is the key the frozen splits are stored by, the leakage guard filters on,
replicate aggregation groups by, and the novelty audit matches against.

Tautomer canonicalisation previously stripped sp3 stereochemistry (RDKit's
`TautomerEnumerator` does this unless told not to), collapsing enantiomers to one
InChIKey. **Measured impact: 0 of 40 contract compounds and 0 of 380 curated records
change**, so no published result was affected; it is fixed because the at-risk motifs are
the aminopyrimidine/aminopyrazole hinge binders this project actively selects for.
`$TNK1_LEGACY_TAUTOMER_STEREO=1` restores the old keys.

---

## 5. Reproducing a run

```bash
pip install -r requirements-lock.txt          # the exact environment, not a floating range
make verify-features                          # inputs match their manifest
make plan V=v3                                # what is current, what is stale, and why
make pipeline V=v3                            # fail-closed, resumable
```

`runs/pipeline/last_run.json` records, per stage: status, exit code, wall time, log path,
and any declared output that the command failed to produce. `runs/pipeline/state.json`
records the dataset digest each artifact was built from.
