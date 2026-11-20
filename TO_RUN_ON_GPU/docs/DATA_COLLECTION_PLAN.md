# Data collection plan

**Written:** 2026-11-20
**Status:** plan only. **Nothing in this document has been run.** The code changes it describes are committed and compile; no harvest, no download, no network call has been made.

---

## 0. Why this is the priority

The project's binding constraint is **172 exact TNK1 measurements**, and everything awkward in [`RUN_REPORT_V3_HPC.md`](../RUN_REPORT_V3_HPC.md) follows from it:

| What breaks | Because |
|---|---|
| Model comparison | A 5-fold scaffold split gives ~34-row test folds → Spearman standard error ≈ 0.10, so **differences below ~0.15 are not distinguishable** |
| Significance testing | The exact Wilcoxon floor over 5 folds is **p ≥ 0.0625** — it cannot reach p < 0.05 at all |
| Model selection | Cawley & Talbot (*JMLR* 11:2079, 2010) measure selection overfitting as *"comparable in magnitude to differences in performance between learning algorithms"* — at this N, the size of the 0.686-vs-0.501 gap the project reasons from |
| Conformal intervals | Xu et al. (arXiv:2304.00970, Merck) **excluded their 203-molecule dataset as too small**. 172 is below the floor of the best QSAR conformal study |
| The whole neural family | A D-MPNN loses to a Morgan-count random forest at this scale, which is the expected ordering, not a bug |

**40 new TNK1 measurements is a 23% increase in the training set.** No model in the Cycle-03 zoo can do that. This is where the effort belongs.

### Before planning, measure what you have
```bash
python scripts/audit_data_gaps.py --version v3
```
Offline, reads nothing but the frozen table and the harvester's own CSVs, writes nothing. It recomputes the numbers every planning document quotes — exact vs censored, rows vs compounds vs scaffolds, per-source counts — rather than trusting "172" copied forward from a November run. **It also prints the implied Spearman standard error and fold size, so the planning arithmetic is in front of you instead of in a memory.**

---

## 1. Free first: three sources that are already coded and simply failed

The 2026-10-26 harvest collected **8 of 11 sources**. The three that did not are the cheapest data available, because the collectors exist and only the plumbing broke. Code changes are committed; **they have not been re-run.**

| Source | What happened | What was changed | Expected value |
|---|---|---|---|
| **METZ** | Returned **0 rows** — and 0 rows is a *different failure from an exception*: the collector ran, found no matching ChEMBL document, and reported success with nothing in it. The run log said "8 of 11" without distinguishing "no data exists" from "my search was too narrow". | Added an assay-description search (as `collect_pkis2` already had), widened the title filter, and made the collector return an **explicit note when empty** saying it was an empty search, not an absent dataset. | **HIGH.** 3,858 compounds × 172 kinases of real Ki values. The search that missed it was ours. |
| **DTC** | Download URL failed. | `dtc_static` is now a list of candidate paths the site has used; `Http.download` tries each. **No guessed endpoints** — if all fail, the collector raises with the manual instruction, which is the correct outcome. | Millions of rows overall; TNK1 content unknown. |
| **KIBA** | Files not downloadable from GitHub. | Most likely cause is the branch rename: `raw.githubusercontent.com/.../master/...` **does not redirect** to `main`, it 404s. Both are now tried. KIBA also gained the manual-matrix fallback Davis already had, and a failure message naming the exact files and paths. | **LOW as labels** — a KIBA score is a model-integrated quantity, not an IC50, and must not be converted. Useful as pan-kinome structure. |

**Run order:** `python scripts/tnk1_harvest.py --check` first (one minute, tests every endpoint from your network and now also inventories the manual files), then `--sources metz,dtc,kiba`.

---

## 2. The largest untapped source: published kinome profiling panels

Four new sources are now registered in the harvester. Each contributes **both exact TNK1 rows and pan-kinome rows** for the transfer stage.

All four resolve the same way: try ChEMBL's deposit of the paper by PubMed ID, then fall back to a **manual drop** of the supplementary table. The manual path is the realistic one — the publishers hosting these refuse automated requests (`pubs.acs.org` and `pubs.rsc.org` return 403; `nature.com` redirects to an identity provider), and nothing in the code tries to defeat that. A human downloads the file.

| Source | Content | Where to put it | Honest assessment |
|---|---|---|---|
| **Klaeger 2017** (*Science* 358:eaan4368) | Kinobeads proteomics: 243 **clinical** kinase drugs × expressed kinome, apparent Kd | `<out>/manual/klaeger/` | **Highest potential, biggest unknown.** Kinobeads only detects the kinases it pulls down from the lysates used, so **TNK1 coverage is not guaranteed.** Check this before anything else — it decides the whole source's value. Kd_app is a competition-binding constant, not a solution Kd and not an enzymatic IC50. |
| **Anastassiadis 2011** (*Nat Biotechnol* 29:1039) | 178 inhibitors × 300 kinases, **percent activity remaining** | `<out>/manual/anastassiadis/` | **Selectivity signal, not potency.** A single-concentration percent readout cannot become an IC50 without the Hill slope and assay conditions, and the collector does not try. Its value is that it targets exactly the component the v3 run failed to optimise (`r_sel` pinned at 0.091–0.144). **Put the screening concentration in the filename** — `manual_wide` reads it out of the name, and a percent readout without its concentration is uninterpretable. |
| **Karaman 2008** (*Nat Biotechnol* 26:127) | 38 inhibitors × 317 kinases, Ambit competition-binding Kd | `<out>/manual/karaman/` | **Moderate.** Same platform as Davis 2011, which is already harvested, so expect heavy compound overlap — `build_master` will flag the duplicates, which is why it flags rather than deletes. |
| **Papyrus** (*J Cheminform* 15:3, 2023) | Curated ChEMBL + ExCAPE-DB aggregation, standardised pChEMBL values | `<out>/manual/papyrus/` | **Moderate for new TNK1 rows, high for a cleaner pan-kinome table** than the current 14-source merge — which would reduce the standardisation variance the transfer stage inherits. **Put the version in the filename**; it goes into the provenance record, and a table whose version is unknown cannot be cited. |

> **⚠ The PubMed IDs in the collectors were not verified against PubMed.** A wrong one is harmless rather than silent: `collect_chembl_docs` logs `ChEMBL documents selected: NONE`, returns 0 rows, and the manual path still works. Check them with `--check` and correct them in `scripts/tnk1_harvest.py` if a lookup comes back empty for a paper you know is in ChEMBL.

> **⚠ Nothing here has verified that any of these tables contains a TNK1 column.** The row counts in `audit_data_gaps.py` are order-of-magnitude expectations from each paper's own description, and the script says so. Verify per source before planning around it.

### One real code gap these sources exposed

Their supplementary tables identify compounds by **drug name** (`dasatinib`, `GW-2580`) and carry no structures at all. The existing `remote_structure_lookup` resolves InChIKey → SMILES, which is what DTC needs and no help here — without name resolution every Klaeger/Anastassiadis/Karaman row would reach the master table with an empty structure and sit there unusable rather than being reported as a gap.

Added: `resolve_names_to_smiles` (PubChem PUG-REST, cached) and `name_structure_lookup`, wired into post-processing before the InChIKey pass. **Every hit is labelled `structure_source='pubchem_name_lookup'` and must be treated as provisional** — a name is a far weaker identifier than an InChIKey, and synonyms, salts and stereochemistry all collide. **Where a table offers a CAS number, PubChem CID or ChEMBL ID, prefer it**: the existing resolvers handle those exactly, and PUG-REST has no batch name endpoint so names cost one request each.

---

## 3. Also worth harvesting

Not yet coded; listed so the decision is explicit rather than forgotten.

- **ChEMBL release check.** The frozen table is built from **ChEMBL 34**. Confirm whether a newer release adds TNK1 rows — this is a one-query check and the collector already exists.
- **BindingDB** is already harvested, but confirm it ran against a current release rather than a cached TSV.
- **PubChem BioAssay** is already harvested with `PUBCHEM_INCLUDE_QUALITATIVE: True`. Worth auditing how many TNK1 rows are qualitative-only (`record_class='qualitative'`) — those are active/inactive calls, not potencies.
- **Open Targets** and **Probes & Drugs** aggregate differently and may surface rows the others miss.
- **SureChEMBL / patents.** The novelty audit already queries SureChEMBL through UniChem, so the plumbing is partly there. Patent examples are a genuine source of kinase SAR and are absent from every database above.

---

## 4. The only data that can validate anything

Every source in §1–3 is a **retrospective** measurement of chemistry someone else made. None tests a molecule this pipeline designed, so **none of them can validate it.** Stage 9 is a no-op because no wet-lab result exists, and the pipeline has never been assessed prospectively.

The 40-compound Cycle-01 contract (`runs/cycle_01/contract_novel_team_b.csv`) is ready. **Two things must be settled before commissioning synthesis:**

1. **`TNK1-NOV-C01-012` sits at p(hERG) = 0.82**, above the configured 0.85 ceiling, and is in the contract only because the hERG hard filter was never applied in the offline scoring path. It has needed a PI decision since 2026-11-09. For context: 29 of 40 exceed the TPP's 0.20 limit and 35 of 40 fail the full TPP.

2. **The contract was selected by the critic that lost the paired comparison.** Those 40 molecules were ranked by the deep ensemble that scores Spearman 0.501 against a random forest's 0.686, with uninformative uncertainty. **Re-score them against the stacked critic before committing synthesis money** — it costs minutes and may reorder them:
   ```bash
   python scripts/run_stage2b_zoo.py --version v3 --tag core --repeats 10
   python scripts/run_stage4b_stack.py --version v3 --metric spearman
   ```

**Measured inactives are worth as much as actives here.** The censored-row handling is already correct — `task_loss` penalises only `max(0, μ − bound)` — so a ">10 µM" result is usable training data, not waste. Ask the assay for every value, not just the hits.

---

## 5. The gap that stopped the harvest reaching the pipeline — now closed

**The harvester's output could not be ingested at all.** Stage 1's long-CSV reader,
`tnk1.data.extract.from_long_csv`, ends with a hard column selection:

```python
return df[["smiles", "uniprot_id", "standard_type", "relation", "value_nM", "source"]]
```

The harvester names three of those five differently — `std_smiles`, `uniprot_accession`,
`activity_type` — so handing it a master table raised `KeyError`. And the obvious
workaround, pointing one of the existing `pkis_csv` / `kinomescan_csv` / `dkk_csv` keys at
the harvest file, would have been worse than the error: `from_long_csv` does
`df["source"] = source`, overwriting every row's real provenance with one literal string.

So a harvest could either fail loudly or succeed while discarding the provenance the
harvester exists to produce. Both are fixed by `tnk1.data.extract.from_harvest_csv`:

- maps the harvester's column names onto Stage 1's, with a `std_smiles` →
  `smiles_used` → `smiles_raw` fallback;
- **preserves per-row `source`**;
- filters on the harvester's own criteria (`record_class == quantitative`, an allowed
  `activity_type`, `value_nM > 0`, a usable structure, `in_panel`, no contamination flag)
  but, unlike `is_training_ready_exact`, **keeps right-censored rows**, which the pipeline
  models and which `is_training_ready_exact` discards by demanding `relation == "="`;
- excludes semantics that are not a single direct measurement — KIBA scores (a
  model-integrated fusion, not a concentration), GtoPdb summary affinities (already a
  median over experiments, so aggregating them again understates `sigma_rep`), and
  outcome-only rows;
- attributes every dropped row to exactly one counted reason, so
  `sum(dropped) + kept == read` and the Stage 1 log says precisely what was excluded.

Reached from `configs/default.yaml: data.harvest_master_csv`. **Absent file → skipped**,
so the ingest is byte-identical to before until a harvest exists.

### The side effect: `source` now reaches the frozen table

§5 previously noted that the frozen table carries no per-row `source`, so the audit could
not say which source each TNK1 row came from. The reason was the branch Stage 1 took: with
no local raw dumps the `parts` list is empty and it falls through to the curated
feature-store path, which has no `source`. The `parts` path does
`aggregate_replicates`, which emits `source` as a `|`-joined set of contributors.

Once a harvest master exists, `parts` is non-empty, so Stage 1 takes that branch and
`kinome_long_v4.parquet` **gains a `source` column**. That resolves the gap in the only
safe way: as a new dataset version, not as a schema mutation of the frozen `v3` that
published results were computed against.

Still open, deliberately: a compound measured by two sources collapses to
`"chembl|bindingdb"` rather than keeping both rows, so a per-source noise floor still
cannot be estimated from the frozen table. That needs a replicate-level table, which is a
data-contract decision rather than a cleanup.

---

## 6. Suggested order

| # | Action | Cost | Expected gain |
|---|---|---|---|
| 1 | `audit_data_gaps.py` | minutes, offline | Know the real numbers before planning |
| 2 | `tnk1_harvest.py --check` | 1 min | Which endpoints work from your network; which manual files are missing |
| 3 | `--sources metz,dtc,kiba` | hours | The three that failed. Metz alone could be substantial |
| 4 | Download the 4 supplementary tables by hand | an afternoon | The largest untapped TNK1 source — **check Klaeger for a TNK1 column first** |
| 5 | `--sources klaeger,anastassiadis,karaman,papyrus` | hours | Parses what you dropped in |
| 6 | Re-run Stage 1, re-freeze, re-audit | hours | A new `kinome_long_v4` with an honest provenance record |
| 7 | Re-score the Cycle-01 contract against the stacked critic | minutes | Before any synthesis money is committed |
| 8 | Commission the 40-compound contract | weeks, real money | **The only data that can validate the pipeline** |

Steps 1–2 are free and tell you whether 3–5 are worth the time. Do not skip them.

### The exact commands, end to end

The dataset version is the thing to get right. **Use a new version; do not use `--force`.**
`--force` deletes the frozen table every published v3 number was computed from, which
destroys the baseline you need in order to show that the extra data helped.

```bash
# 1. harvest  (writes <out>/TNK1_kinase_master.csv)
python scripts/tnk1_harvest.py --check                 # which endpoints your network reaches
python scripts/tnk1_harvest.py --out data/raw/tnk1_data

# 2. point Stage 1 at it (configs/default.yaml: data.harvest_master_csv),
#    or just copy it to the configured default path
cp data/raw/tnk1_data/TNK1_kinase_master.csv data/raw/TNK1_kinase_master.csv

# 3. see what it would contribute, offline, before freezing anything
python scripts/audit_data_gaps.py --version v3 --harvest data/raw/tnk1_data

# 4. freeze a NEW version. v3 stays on disk as the comparison baseline.
python scripts/run_stage1_data.py --version v4

# 5. run everything against v4. TNK1_VERSION drives every HPC script.
export TNK1_VERSION=v4
bash hpc/preflight.sh                                  # checks v4, not v3
bash hpc/run_pipeline.sh full
bash hpc/run_sweeps.sh all                             # zoo -> stack -> 8 RL objectives

# 6. docking is a SEPARATE step, and needs a Vina binary (not in this repo)
conda install -c conda-forge vina                      # or set TNK1_VINA=/path/to/vina
python scripts/dock_controls.py --cycle 3
python scripts/dock_40_molecules_tnk1.py --cycle 3
```

Step 4 prints the harvest filter report — read it. It is the only place that tells you how
many of the harvested rows actually became training data, and why the rest did not.

Two things this sequence depends on, both of which were broken and are now fixed:

- **`TNK1_VERSION` is honoured by every HPC script.** `submit_cpu.sh`, `submit_gpu.sh`,
  `preflight.sh` and `run_tnk1.sh` hardcoded `--version v3`, while `submit_array.sh` and
  `run_sweeps.sh` already read the variable. With a `v4` table on disk the hardcoded
  scripts would have run every stage against **v3** — no error, because v3 still exists
  and its `sha256` still verifies, so the jobs would have completed and the results would
  silently have been the old 172-molecule ones.
- **`--cycle` actually selects a cycle in the docking scripts.** It was parsed and
  ignored: every path was a constant pinned to `cycle_01`, so `--cycle 3` re-docked
  Cycle-01's molecules and overwrote its published poses.

---

## 7. What is honestly unverified in this plan

- **No harvest has been run.** Every code change here compiles and is registered consistently (`audit_data_gaps.py` and the source-registry check confirm 14 sources wired through config, collectors, run order and master order), but **no collector has executed.**
- **The four new PubMed IDs are unchecked**, as flagged in §2 and in the code.
- **No table has been confirmed to contain TNK1.** The row-count expectations are from paper descriptions.
- **The DTC and KIBA repairs are hypotheses about why they failed** — plausible ones (dead URL; GitHub branch rename), but the 2026-10-26 run recorded only "failed", so the actual cause was not captured. If the repairs do not work, the new error messages at least name the files and paths, which the old ones did not.
- **`resolve_names_to_smiles` has never resolved a name.** It is wired in and compiles; its first real use will be its first test.
- **The harvest bridge has never seen a real master table.** `from_harvest_csv` is covered by 34 unit tests against a synthetic table built to exercise every filter branch, and the full suite is 465 passed / 13 skipped. But a synthetic fixture is written by the same person who wrote the filter: the column names and semantics strings come from reading `tnk1_harvest.py`, not from a file it produced. The first real harvest is the first genuine test, and the thing to read is the filter report Stage 1 prints — specifically whether `kept` is a plausible fraction of `read`, and whether any single exclusion reason dominates unexpectedly.
- **Nothing here has run on a GPU, and this machine has no rdkit or torch**, so every claim above is from static reading and the dependency-free tests. The pipeline stages themselves remain unexecuted since the v3 run.
- **Docking needs a Vina binary that is not in this repository.** `tools/` is gitignored and the old hardcoded path was a Windows `.exe`, so docking could never have run on a Linux HPC. `resolve_vina` now searches `--vina`, `$TNK1_VINA`, `PATH` and `tools/`, and fails with an install command instead of a `FileNotFoundError` from inside a worker thread — but the binary still has to be installed.
