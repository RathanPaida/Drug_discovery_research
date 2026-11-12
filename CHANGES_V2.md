# Pipeline v2 — changes, and why

Six defects were found by auditing the Cycle-01 artifacts against the code that produced
them. Each was silent: none raised, none logged, and the notebook reported results as
though the affected stage had worked.

---

## 1. Stage 7 never ran — the TOML was unparseable

`runs/rl/` contained only `rl.toml`: no checkpoint, no step log, no TensorBoard directory.
`pipeline_execution.log` shows Stage 8 alone, total elapsed **0.1 minutes**.
`stage7_scored_candidates.csv` has no `step` column.

The cause: `write_rl_toml` interpolated Windows paths into TOML *basic* strings.

```
TOMLDecodeError: Unescaped '\' in a string (at line 9, column 21)
```

Line 9 is `prior_file = "data\processed\prior\kinase.prior"` — `\p` is an invalid escape,
so a spec-compliant parser (`tomllib`, which REINVENT 4 uses) rejects the whole file.
Worse, `...\tnk1\rl\...` parses `\t` as TAB and `\r` as CR, silently corrupting paths that
*do* parse.

**Fixed:** `tnk1/rl/reinvent_io.py` emits POSIX paths inside TOML *literal* strings.
**Pinned by:** `tests/test_regressions_v2.py::test_rl_toml_parses_when_paths_are_windows_native`

## 2. The 12,000 "RL candidates" were filtered, not generated

Even had the TOML parsed, Stage 7 as written performed *selection*: sample a fixed pool,
score once, keep the best. No generator parameter was ever updated.

**Fixed:** new `rl_min/` package implementing REINVENT's DAP loss, so the reward acts on
the generator's weights. Measured on a 12k corpus / 1.47M-parameter prior, 240 steps:

| | prior | agent |
|---|---|---|
| mean reward | 0.069 | **0.544** (7.8×) |
| validity | 0.44 | 0.91 |
| scaffolds/batch | 27 | 48 |

**Pinned by:** `tests/test_rl_min.py::test_one_dap_step_changes_the_agent_but_not_the_prior`

## 3. The applicability domain divided by ~zero

`ApplicabilityDomain` standardises a 300-dimensional latent fitted on ~304 molecules.
Dead ReLU units have σ exactly 0, and the code divided by a 1e-6 epsilon. Result: `d_lat`
reached **240,832** against a threshold of **1.007**, and `r_ad` — a nominal weight-1.5
regulariser — became the largest single driver of the RL reward (realised signal 0.380,
above `r_pot`'s 0.335). `exp(-(d_lat-thr)/0.15)` underflowed to exactly 0.0, zeroing those
molecules' entire geometric-mean reward.

**Fixed:** per-dimension σ floored against the latent's *global* scale. A median-relative
floor fails exactly when needed — if most dimensions are dead the median is itself ≈0.
Reproduction: `d_lat` max 43,528 → 1.3.

## 4. Reward sigmoids sat outside the population's range

Measured over the shipped 12,000-molecule pool:

| term | weight | mean | problem |
|---|---|---|---|
| `r_sel` | 2.0 | **0.0089** | centre 1.2, pool **max 0.25** — never reached |
| `r_pot` | 3.0 | 0.185 | centre 7.0, pool mean 6.16 |
| `r_adme` | 1.0 | 0.554 | flat — realised signal 0.019 |
| `r_qed` | 0.5 | 0.482 | flat — realised signal 0.016 |

The `r_sel` defect is **range compression, not variance starvation**. In a saturated tail
`sigmoid(x) ≈ exp(k(x−c))`, so `log r` is linear in x and log-variance is at its *maximum*
there. The damage is that every molecule gets a near-constant heavy discount, `R` tops out
at 0.399, and since DAP uses `prior_logP + sigma·R` the effective `sigma` collapses with it
(120 → ≈48).

**Fixed:** `calibrate_centre()` places centres from the agent's own output;
`component_signal()` audits realised vs. intended influence. Signal spread 23.0× → 5.9×,
max R 0.399 → 0.773.

## 5. Stage 2 claimed 5×5 nested CV and ran 2×2

`run_stage2_baselines.py` had `spl["cv"][:2]` / `f["inner"][:2]` — hyperparameters were
selected on **4 of 25 folds** while both the notebook and the manual reported 5×5 nested
scaffold cross-validation. 104 fits instead of 650.

**Fixed:** all folds by default; `splits.tune_outer_folds` / `tune_inner_folds` to dial back.

## 6. The RL reward model ranked worse than the baseline it replaced

| model | RMSE | Spearman |
|---|---|---|
| Stage 2 Random Forest | 0.436 | **0.626** |
| Stage 4 deep ensemble (used as RL critic) | 0.623 | **0.264** |

A policy optimised against this critic climbs noise. **Not fixed** — it needs modelling
work (shrink the D-MPNN: hidden 300→128, fusion [512,256]→[128,64], dropout 0.2→0.35; or
stack RF+XGB+GP). The v2 notebook now *gates* on it and prints a warning when the critic
loses to the baseline.

---

## Running on maximum available data

| cap | was | now | knob |
|---|---|---|---|
| kinome training records | 425,906 | **~1.35M** | `data.full_kinome: true` |
| Stage 2 tuning folds | 4 of 25 | all 25 | `splits.tune_*_folds` |
| descriptor scaler fit | 50,000 | 200,000 | `data.desc_scaler_n` |
| Stage 8 candidate pool | 50,000 | 200,000 | `al.pool_size` |
| RL prior corpus | — | **787,283** | `rl_min.corpus` |
| molecular-graph cache | unbounded (~10 GB, OOM) | 300k entries | `$TNK1_GRAPH_CACHE` |

**Why ~1.35M and not ~4M.** The full measurements file is roughly 946k exact + 408k
censored (both carry a pActivity) plus ~858k single-point %-inhibition and ~2.37M
qualitative active/inactive flags. The latter two have **no regression target**, so
`task_loss` cannot consume them; they need a classification head. Stage 1 drops them with a
printed count rather than coercing to NaN.

**`kinome_train.cap_n: 2000` is deliberately kept.** It is not a data cap — it is the
per-kinase importance weight `min(1, √(cap_n/N_k))`. Removing it lets EGFR (19,085 pairs)
and KDR (15,756) swamp TNK2 (1,350), the closest homolog to TNK1 and the one that
transfers best. With the full kinome ingested the imbalance grows, so it matters *more*.

---

## Notebook integrity

`notebooks/TNK1_Master_Pipeline_v2.ipynb` replaces the v1 notebook (archived alongside).

| | v1 | v2 |
|---|---|---|
| Stage 2 chart | hardcoded literals | computed frame |
| Stage 4 interval plot | `np.random.normal` under the legend *"True Experimental pIC50"* | real held-out predictions |
| Stage 5 reference table | invented fallback numbers when the file was absent | fails loudly |
| Stage 9 | r = 0.188 presented as a cross-check | reported as a **negative** result |
| Stage 10 | *"Patentability Verification"* | exact-match novelty audit, scope stated |
| Scorecard | narrative strings | computed from this run's artifacts |

---

## Files added

```
notebooks/TNK1_Master_Pipeline_v2.ipynb          32 cells, every stage computes
notebooks/TNK1_Master_Pipeline_v1_archived.ipynb for diffing
rl_min/                                          DAP generator (7 modules + README)
tests/test_regressions_v2.py                     13 tests pinning defects 1,3,4
tests/test_rl_min.py                             26 tests for the generator
configs/aizynth.yml                              template (model files not included)
priors/README.md                                 what must go there, and how to get it
```

Full suite: **212 passed**, 2 deselected (network).
