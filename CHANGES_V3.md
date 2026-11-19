# Pipeline v3 — hardening pass

Goal of this round: make every stage runnable from anywhere, remove inert configuration,
and close the gaps that let a wrong answer pass silently. Thirteen defects, each of
which produced no exception and no log line.

Test suite **237 → 308**. Coverage **49% → 56%**, concentrated on the half that was
untested: `dmpnn.py` 22→100%, `trainer.py` 0→98%, `model.py` 32→96%, `activity.py` →100%,
`standardize.py` 0→97%.

---

## 1. The kinome store's `pActivity == 0.0` sentinel reached the loss as a measurement

**The worst defect found.** `run_stage1_data.py` filtered with `keep = np.isfinite(pact)`
and its comment asserted that %-inhibition and qualitative rows "are dropped with a printed
count". They were not: the store encodes "no dose-response value" as pActivity exactly
**0.0**, and `np.isfinite(0.0)` is `True`.

Of 4,798,165 rows in `kinome_all_measurements_3.95M.parquet`:

| | rows | |
| :--- | ---: | :--- |
| `pActivity == 0.0`, `censored == 0` | **3,374,532 (70.3%)** | reach `task_loss` as *exact* measurements of IC50 = **1 molar** |
| of which non-dose-response | 3,338,837 | `OUTCOME` 1.80M, `PCT_INHIBITION` 771k, `kon` 258k, `koff` 258k, `Residual Activity` 70k, `Activity` 60k, `PCT_CONTROL` 58k, `PCT_ACTIVITY` 29k — *every one* |
| `0 != pActivity < 2.0` | 108,450 | `value_nM` 1.0e7–1.3e20 (median 3.2e14 nM = 316,000 M); 107,795 are PubChem `Ki` rows not in nM |
| composite `uniprot_accession` | 211,546 | semicolon-joined sets of up to ~150 accessions (642 distinct), discarded silently downstream |

`kinome_long_v2.parquet` (4,796,129 rows) was built before the filter and contains
3,372,521 poisoned rows. A model trained on it is fitted to predict 0.0 for seven rows in
ten — and nothing would have complained: the checksum verifies, the schema is right, and
the row count merely looks impressive.

**Fixed:** `tnk1.data.activity.filter_measurable` drops all four classes with per-reason
counts, governed by `data.dose_response_types` / `pact_min` / `pact_max` /
`drop_composite_targets`. `Ctx._assert_targets_usable` refuses any table over 1%
contaminated (`$TNK1_ALLOW_SENTINEL_TARGETS=1` to inspect, not to train).

Filtered table `v3`: **1,103,498 rows**, mean pAct 6.67, 0 sentinels, 491 real kinases.
Without the composite drop it is 1,315,069 — the ~1.35M that `CHANGES_V2.md` documents as
usable, which is the cross-check that the filter is the intended one.

**Pinned by:** `tests/test_measurability.py` (14 tests, incl. that the drop reasons are
mutually exclusive and sum to the total).

## 2. Every config path resolved against the working directory

`configs/default.yaml` declares `data/processed`, `runs`, `data/raw` relatively, and `Ctx`
passed them straight to `Path(...)`. Launching a stage from `scripts/` did not fail — it
resolved `data/processed` to `scripts/data/processed`, created it with
`mkdir(parents=True)`, and wrote a parallel artifact tree there. Stage 0 would report the
real store as ready while the next stage read and wrote an empty one.

**Fixed:** `tnk1.utils.paths.resolve_path` anchors relative paths to the repo root;
`load_config` absolutises every path-valued key and records `_config_path`; the nine stage
scripts default `--config` to the package's own config (`$TNK1_CONFIG` overrides).
`features_dir()` now warns when a CWD-relative candidate wins.

**Pinned by:** `tests/test_pipeline_scorer.py::test_config_paths_are_absolute_and_cwd_independent`.

## 3. Stage 1 deleted the frozen artifact to bypass its own immutability guard

`save_locked` raises `FileExistsError` so a re-run cannot replace the table published
results were computed from. Stage 1 called `unlink()` on both the parquet **and its
`.sha256` sidecar** immediately before invoking it. A re-run therefore overwrote the frozen
table and wrote a new matching digest — `verify_locked` still passed, and nothing recorded
that the data had changed under every downstream number.

**Fixed:** replacing a frozen table requires `--force`, prints the old and new digests, and
states that downstream artifacts are now stale. The error names the next free version.

## 4. `PipelineScorer` existed, was exported, and was never instantiated

`rl_min/scorer.py` has shipped a `PipelineScorer` adapter onto the production reward since
v2. Nothing constructed it: `run_rl.py` hardcoded `StandaloneScorer` (QED × MW-window ×
motif). **No generator had ever been trained against the real objective**, and the
"7.8× reward improvement" in `CHANGES_V2.md` was measured on the toy proxy.

Three bugs in the adapter itself, found on wiring it up:

- **Beta staleness.** `Scorer.score` recomputes `beta` from its step counter and feeds it
  to `r_pot`. The cache was keyed on SMILES alone and persisted across steps, so a molecule
  sampled at step 5 kept its exploration-era reward (β=0.3) at step 900 (β=1.5) while a
  fresh molecule was judged pessimistically — two objectives at once, biased toward
  whatever the agent emitted early. Default is now `cache="step"` (de-duplicate within a
  batch only); `"run"` warns.
- **Stalled schedule.** `advance=True` only fired when something needed scoring, so an
  all-cached batch did not tick β. Added `Scorer.advance_step()` / `Scorer.beta()`.
- **`NameError`** on an empty input list with caching off (`out` unbound).

**Fixed:** `--scorer pipeline` builds the full stack from the Stage 3–5 checkpoints,
calibrates `pot_c`/`sel_c` on a fresh prior sample before step 1, and records
components/weights/calibration in `meta.json`. `PipelineScorer.static()` scores the prior
baseline without advancing β, so prior and agent are compared at the same β.

**Pinned by:** `tests/test_pipeline_scorer.py` (15 tests).

## 5. `Path.glob()` truthiness made an empty ensemble report READY

`run_stage0_check.py` had `not (proc / "ensemble").glob("member_*.pt")`. `Path.glob()`
returns a **generator**, which is always truthy, so the expression was unconditionally
`False` and the check never fired once. An `ensemble/` directory with zero members reported
`[READY] rl`; Stage 7 then died inside `Ensemble.__init__`'s assert, after the operator had
been told generation could run. Fixed with `any(...)`.

## 6. A module executed at import, hitting the network

`tnk1/rl/reinvent_client.py` ran `argparse.parse_args()`, read stdin to EOF and issued an
HTTP POST in its module body. Any package walk — pytest collection, a docs build,
`pkgutil.walk_packages` — tried to reach 127.0.0.1:8765 and died with `URLError`, and the
module could never be imported to be tested. Its caller located it by the CWD-relative
literal `Path("tnk1/rl/reinvent_client.py")`.

**Fixed:** `main()` behind `if __name__ == "__main__"`, plus `client_path()` and a testable
`score()`. All 13 package modules now import cleanly.

## 7. Eight config keys were declared and never read

A knob that does nothing is indistinguishable from a typo, and editing it appears to change
behaviour.

| Key | Was | Now |
| :--- | :--- | :--- |
| `al.exploit_min_sel`, `al.explore_min_mu` | not passed to `select_batch`; its defaults applied, and they happened to match — so editing the config changed nothing | passed |
| `rl_min.replay_size`, `replay_batch`, `grad_clip`, `temperature` | `DAPConfig` built from five CLI flags only | plumbed, and `rl.diversity` with them |
| `rl_min.baseline_n` | CLI default only | config default |
| `model.esm_dim`, `model.klifs_dim` | fusion width came from `feats.shape[1]`; a store built with a different ESM variant would train at the wrong width while the config still claimed 1280/425 | asserted against the store, raises on mismatch |
| `targets.tnk2` | inert; the panel repeats `Q07912` as a literal, so the two could drift | validated at preflight |
| `admet.epochs` | pinned at `train_endpoint`'s default of 12, unreachable from config | added and wired |

Audit script: **0 of 181** config leaf keys are now unreferenced.

## 8. Tautomer canonicalisation stripped stereochemistry

`rdMolStandardize.TautomerEnumerator().Canonicalize()` removes sp3 stereo adjacent to a
tautomerisable system unless told otherwise:

```
in   C[C@H](N)C(=O)Nc1ccc(Cl)cc1C(=O)NCc1ccccc1   YNNMCNCKOWYFQD-NSHDSACASA-N
out  CC(N)C(=O)Nc1ccc(Cl)cc1C(=O)NCc1ccccc1       YNNMCNCKOWYFQD-UHFFFAOYSA-N
```

`UHFFFAOYSA` is the hash for *no stereochemistry*, so both enantiomers came out with one
identical 27-character key — while the function asked for `isomericSmiles=True`.

**Measured blast radius: 0 of 40 contract compounds and 0 of 380 curated records change**,
though 32% and 27% respectively carry stereocentres. **No published result was affected and
no frozen table needs rebuilding.** Fixed because it is a live trap: conflated enantiomers
would be median-averaged by `aggregate_replicates`, and a pair spanning >1 log unit (what a
potent/inactive enantiomer pair *is*) would be dropped outright by
`replicate_policy: drop`. The at-risk motifs are the aminopyrimidine and aminopyrazole
hinge binders `HINGE_SMARTS` actively selects for.

**Fixed:** `SetRemoveSp3Stereo(False)`; `$TNK1_LEGACY_TAUTOMER_STEREO=1` restores the old
keys. The standardiser objects are also cached — they were being reconstructed per
molecule, i.e. 788,000 times over the full store.

## 9. Docking was unseeded, and probing the script started a campaign

Two defects in the same pair of scripts, found by running them.

**`dock_40_molecules_tnk1.py` took no arguments**, so `--help` was ignored and the script
immediately began docking all 40 contract compounds, overwriting the published poses in
`runs/cycle_01/docking_poses/`. The universal gesture for "tell me what this does" was a
destructive action; you found out it was unsafe to probe by probing it. (This was hit for
real during this pass: 11 poses and `docking_controls.json` were overwritten and had to be
restored from git.) Both dock scripts now have real argparse, and refuse to overwrite an
existing campaign without `--force`.

**Vina was invoked without `--seed`**, so it seeded from the clock and the campaign was not
reproducible. Re-running the three clinical controls on the identical receptor and box:

| control | published | re-run | published hinge | re-run hinge |
| :--- | ---: | ---: | ---: | ---: |
| TP-0903 | −9.40 | −9.29 | 1.98 Å | 2.06 Å |
| Dasatinib | −8.37 | −8.30 | 3.49 Å | **3.51 Å** |
| Bosutinib | −8.26 | −8.10 | 3.53 Å | 3.59 Å |

The affinity drift is small. The problem is that **`hinge_contact` is a hard threshold at
3.5 Å**: dasatinib crosses it between the two runs, flipping from contact to no-contact on
the RNG alone. The headline "60.0% overall hinge contact rate, reaching 75.0% in EXPLOIT
leads" is a count of exactly these threshold crossings, so its reproducibility was bounded
by an unseeded search. `DOCK_SEED = 42` is now passed through; `--seed -1` restores the old
clock-seeded behaviour.

`dock_controls.py` was also bare module-level code (same shape as defect 10 below), and
both scripts carried dead fallbacks to `tnk1-pipeline/tnk1-pipeline/`, a pre-reorganisation
path that had decayed to a cache-only stub and is now deleted.

### The pattern, not the instance

**All three scripts in the repository that write published artifacts ignored `--help` and
overwrote without asking:**

| script | writes | what `--help` did |
| :--- | :--- | :--- |
| `dock_40_molecules_tnk1.py` | `runs/cycle_01/docking_poses/` (43 files) | docked 40 compounds |
| `dock_controls.py` | `runs/cycle_01/docking_controls.json` | re-docked 3 controls |
| `generate_paper_figures.py` | `paper/figures/*.{png,pdf}` (13 files) | regenerated every manuscript figure |

The last had an `if __name__ == "__main__"` guard but no argument parsing, so it called all
seven generators regardless of argv. Every one was triggered for real during this pass by a
`--help` probe, and every one had to be reverted from git. All three now parse arguments,
print help when asked, and refuse to overwrite without `--force` (with `--out` to write
elsewhere).

## 10. Stage 9 was bare module-level code with CWD-relative globs

No `main()`, no argparse, `runs/cycle_*/results.csv` read and `runs/stage9_hit_rates.csv`
written relative to the caller. With no wet-lab returns yet — the actual situation —
`pd.concat([])` raised `ValueError: No objects to concatenate`, making "the experiment has
not returned" indistinguishable from "the stage is broken". Rewritten: config-driven paths,
required columns checked by name, Holm-adjusted slice comparisons, and a clean exit 0 with
an explanation when there is nothing to report.

## 11. Two more, found by actually running the stages

**The SSL checkpoint had no provenance check.** `kinome_stage` loaded `ssl_encoder.pt` with
a bare `load_state_dict`. Change `model.hidden` in the config and you got a ten-line torch
size-mismatch dump from inside `nn.Module` -- naming tensor shapes, never the actual
problem, which is that the checkpoint and the config describe different architectures. It
now names both widths and says to re-run `make pretrain` or delete the checkpoint.

**There was no fast end-to-end smoke path.** `LabelledSet.__init__` computes descriptors
for every row at construction, so even a 1-epoch / 2-member run over the 1.1M-row table
pays ~1.1M RDKit descriptor calculations before the first gradient step. Verifying a change
to Stage 4 therefore cost hours, which is the same pressure that lets defects survive into
a 24-hour run. `data.max_kinome_rows` deterministically samples the off-target rows while
**always keeping every TNK1 row** (they carry the labels every metric is computed on, and
the frozen splits are keyed by TNK1 compound). A run with it set prints a loud warning.

---

## Added

```
scripts/run_pipeline.py        one orchestrator: declares each stage's needs/makes,
                               refuses to start on missing inputs, treats "exited 0 but
                               produced nothing" as failure, tracks which dataset digest
                               each artifact came from, resumable, --plan / --from / --only
scripts/tnk1_harvest.py        the raw-data harvester, RECOVERED from ml_research.zip
                               (10 sources, resumable) — was referenced by the audit as
                               missing and existed nowhere in the repository
docs/DATA_PROVENANCE.md        the full chain, what is reproducible and what is not
pyproject.toml                 real dependencies; setup.py declared NONE
requirements-lock.txt          the exact environment (py 3.10.20, torch 2.12.0+cu130)
tests/test_measurability.py    14 tests  — defect 1
tests/test_pipeline_scorer.py  15 tests  — defects 2, 4
tests/test_training_core.py    21 tests  — D-MPNN, training loop, L2-SP, early stopping
tests/test_standardize.py      21 tests  — defect 8, compound identity, _mk row builder
```

`make plan` / `make pipeline` replace the nine hand-ordered `make` targets, which remain
for re-running one step.

---

## Still open

- **Nothing in the repository rebuilds the 12 feature-store files.** The harvester produces
  `data/raw/tnk1_data/`; Stage 1 consumes `data/features/`. The step between is missing —
  see `docs/DATA_PROVENANCE.md` for the acceptance test (regenerated sha256s must match
  `manifest.json`). Do not delete `ml_research.zip`: it is the only copy of the raw tree.
- **The RL critic still loses to the Stage-2 baseline.** RF: RMSE 0.440 / ρ 0.641.
  Deep ensemble, which drives `r_pot` and `r_sel`: RMSE 0.623 / ρ 0.264. Unchanged from
  `CHANGES_V2.md` defect 6; `make ablate` (the A0–A7 ladder, wired and never run) is the
  diagnosis, and `tnk1/rl/scorer.py` injects the critic through two duck-typed methods, so
  an RF/GP-backed critic is an adapter rather than a rewrite.
- **The 2026-10-26 harvest was incomplete and said so nowhere.** DTC and KIBA failed, METZ
  returned 0 rows: 8 of 11 sources. Stage 0 now prints per-source status.
- `TNK1-NOV-C01-012` remains in the Cycle-01 contract at p(hERG) = 0.8199, above the
  Tier-A gate of 0.50. 29/40 exceed the TPP's 0.20 and 35/40 fail the full TPP.
  `PIPELINE_AUDIT.md` §6.3 has wanted a decision since 2026-11-09.
- Audit blockers 1 and 2 are **closed**: 6/6 stages ready, feature store 12/12 verified,
  and the novelty re-audit shows 40/40 `novelty_fully_checked` / `NOVEL`.
