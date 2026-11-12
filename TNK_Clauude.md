# TNK1 Pipeline — Audit, Repair and v2 Rebuild

A step-by-step record of the work: what was found, what was changed, what was verified by
running it, and what is still unproven.

**Related:** [`CHANGES_V2.md`](CHANGES_V2.md) (defect detail) · [`docs/RUNBOOK.md`](docs/RUNBOOK.md) (how to run it)

---

## Step 0 — Starting point

The notebook `TNK1_Master_Pipeline.ipynb` presented a 10-stage pipeline with a polished
scorecard:

```
Generative Optimization Engine : Multi-Objective REINFORCE with LCB Scheduling
Delivered Active Learning Batch: 40 Novel Small Molecules
Worldwide Novelty Verification : 100% Novel (0/40 in PubChem, UniChem, Kinome Store)
```

The audit compared those claims against the artifacts that produced them.

---

## Step 1 — Audit: the RL stage had never run

Evidence, all from the repository's own files:

| file | what it showed |
|---|---|
| `runs/rl/` | contained **only** `rl.toml` — no checkpoint, no step log, no `tb_rl/` |
| `runs/pipeline_execution.log` | only Stage 8 ran. Total elapsed: **0.1 minutes** |
| `runs/stage7_scored_candidates.csv` | 12,000 rows, **no `step` column** |

The 12,000 "RL candidates" were one-shot samples from a prior, scored once and filtered.
That is *selection*, not reinforcement learning — no generator parameter was ever updated,
so the stage could not propose anything the prior was already unlikely to emit.

---

## Step 2 — Audit: found why it never ran

`tnk1/rl/reinvent_io.py` interpolated Windows paths into TOML **basic** strings. In TOML,
a basic string processes backslash escapes and only `\b \t \n \f \r \" \\ \uXXXX` are legal.

Running the shipped `runs/rl/rl.toml` through `tomllib` (what REINVENT 4 uses):

```
TOMLDecodeError: Unescaped '\' in a string (at line 9, column 21)
```

Line 9 is `prior_file = "data\processed\prior\kinase.prior"` — `\p` is invalid, so the
parser rejects the entire file. Two further hazards in the same file: `...\tnk1\rl\...`
parses `\t` as TAB and `\r` as carriage return, silently corrupting paths that *do* parse.

**Verified both directions:** the old file raises; the patched writer produces a file that
parses, with no stray TAB or CR.

---

## Step 3 — Audit: the reward was degenerate

Measured over the pipeline's own 12,000-molecule pool:

| term | intended weight | mean value | % below 0.01 | realised signal |
|---|---|---|---|---|
| `r_ad` | 1.5 | 0.745 | 18.2% | **0.380 — largest** |
| `r_pot` | 3.0 | 0.185 | 3.8% | 0.335 |
| `r_sel` | 2.0 | **0.0089** | **70.0%** | 0.157 |
| `r_safety` | 1.5 | 0.281 | 0% | 0.080 |
| `r_sa` | 1.0 | 0.698 | 0.3% | 0.037 |
| `r_adme` | 1.0 | 0.554 | 0% | 0.019 |
| `r_qed` | 0.5 | 0.482 | 0% | 0.016 |

Three separate problems:

1. **`r_sel` never reached its own midpoint.** Centre was 1.2; `mu_sel` across all 12,000
   molecules had mean −1.24 and **maximum 0.25**.
2. **`r_ad` was numerically broken and dominant.** See Step 4.
3. **`r_adme` and `r_qed` were at the noise floor** — they consumed weight and returned no
   gradient.

Maximum reward was **0.399**. REINVENT's DAP computes `augmented_logP = prior_logP +
sigma * score`, so a nominal `sigma = 120` behaved like **≈48**.

### A correction made mid-work

The first characterisation of `r_sel` was *"carries almost no gradient"*. Writing the
regression test disproved that: in a saturated tail `sigmoid(x) ≈ exp(k(x−c))`, so `log r`
is **linear** in x and log-variance is at its **maximum** there (0.82 vs 0.50 for the
recalibrated term). The real defect is **range compression** — every molecule receives a
near-constant heavy discount, `R` never approaches 1, and the effective `sigma` collapses
with it. The fix is the same; the explanation was wrong and was corrected in the code
comments and the test.

---

## Step 4 — Audit: the applicability domain divided by ~zero

`ApplicabilityDomain` standardises a **300-dimensional** latent fitted on **~304**
molecules. Dead ReLU units have σ exactly 0, and the code divided by a 1e-6 epsilon.

```
d_lat observed : mean 6,221.7   max 240,831.9
d_thr_lat      : 1.007
```

`r_ad = exp(-(d_lat - thr)/0.15)` underflows to exactly 0.0, zeroing those molecules'
entire geometric-mean reward. A nominal weight-1.5 regulariser was the single largest
driver of the objective.

**A wrong first fix, caught by its own test.** The initial patch floored each σ against the
*median* σ. That floors nothing in exactly the case it is needed: when most dimensions are
dead, the median is itself ≈0. Corrected to floor against the latent's **global** scale.

Reproduction with the real geometry (304×300, 160 dead units): `d_lat` max **43,528 → 1.3**.

---

## Step 5 — Audit: Stage 2 claimed 5×5 and ran 2×2

`scripts/run_stage2_baselines.py`:

```python
for f in spl["cv"][:2]:
    for inn in f["inner"][:2]:
```

Hyperparameters were selected on **4 of 25 folds** while both the notebook and the manual
reported "5×5 Nested Scaffold Cross-Validation" — 104 fits instead of 650.

---

## Step 6 — Audit: the reward model ranked worse than the baseline it replaced

| model | RMSE ↓ | Spearman ↑ | role |
|---|---|---|---|
| Random Forest (Stage 2) | **0.436** | **0.626** | discarded baseline |
| XGBoost (Stage 2) | 0.480 | 0.638 | discarded baseline |
| **10-member deep ensemble (Stage 4)** | **0.623** | **0.264** | **used as the RL critic** |

Root cause is structural: ~600k parameters fitted against **172 exact labels** (the other
208 of 380 TNK1 records are right-censored).

**This is not fixed.** It needs modelling work. The v2 notebook *gates* on it and prints a
warning when the surrogate loses to the baseline.

---

## Step 7 — Audit: notebook integrity

- **Cell 11** plotted `np.random.normal` synthetic data under the legend
  *"True Experimental pIC50"*.
- **Cell 7** loaded the benchmark CSV and then plotted **hardcoded literals**.
- **Cell 13** substituted invented numbers when its input file was absent.
- **Stage 9** reported 2D-vs-3D Pearson **r = 0.188** as a cross-check. That is no
  correlation — a negative result.
- **Stage 10** titled exact-InChIKey matching *"Patentability Verification"*.

---

## Step 8 — Fix: the four code defects

| file | change | verification |
|---|---|---|
| `tnk1/rl/reinvent_io.py` | POSIX paths inside TOML **literal** strings | old file raises, new parses; no TAB/CR |
| `tnk1/uq/applicability.py` | σ floored against the latent's global scale; `d_lat` clipped | 43,528 → 1.3 on a reproduction |
| `tnk1/rl/reward.py` | `calibrate_centre()`, `component_signal()`, `aggregate(components=)` | signal spread **23.0× → 5.9×**, max R **0.399 → 0.773** |
| `configs/default.yaml` | documented why each historical value is wrong | parses |

---

## Step 9 — Build: `rl_min`, a generator that actually trains

New package implementing REINVENT's DAP loss:

```
L = ( log P_prior(x) + sigma·R(x)  −  log P_agent(x) )²
```

The prior is frozen and acts as an implicit KL anchor; `sigma·R(x)` displaces the target
toward high-reward molecules; the agent regresses onto it. **Weights move.**

```
vocab.py        multi-character SMILES tokeniser (Cl, [nH], %10 stay single tokens)
model.py        SmilesRNN — whole-sequence NLL, EOS-masked sampling
train_prior.py  MLE prior; EncodedCorpus for the 788k store
dap.py          the policy-gradient loop, diversity filter, experience replay
scorer.py       StandaloneScorer (no feature store) / PipelineScorer (real surrogate)
run_rl.py       driver; measures the prior baseline before training
```

### Ran it end to end

Trained a prior on 12,000 real SMILES, then ran 240 DAP steps:

```
prior validity      0.044 → 0.447   (12 → 90 epochs)
prior baseline R    0.069
agent last-10 R     0.544            ← 7.8× over the prior baseline
validity            0.44 → 0.91      ← rising under reward pressure
uniqueness          ~0.98            ← no mode collapse
scaffolds/batch     27 → 48
```

Validity *increasing* under reward pressure indicates a policy that generalised rather than
one that memorised a single high-scoring scaffold.

**One real bug surfaced during that run:** `sample_with()` left the LSTM in `eval()` mode,
and cuDNN refuses to backpropagate an RNN in eval. Fixed.

**The decisive measurement** is the prior baseline, scored *before* training starts. Without
it you cannot distinguish "the policy improved" from "we drew more samples" — which is
precisely why the v1 scorecard cannot support its RL claim.

---

## Step 10 — Build: the v2 notebook

`notebooks/TNK1_Master_Pipeline_v2.ipynb` — 32 cells, every stage computes.

| | v1 | v2 |
|---|---|---|
| Stage 2 chart | hardcoded literals | computed frame |
| Stage 4 intervals | simulated noise labelled as measurements | real held-out predictions |
| Stage 5 reference | invented fallback | fails loudly |
| **Stage 7** | **sample → score → filter** | **DAP; the generator trains** |
| Stage 8 input | fixed pool | molecules from the trained agent |
| Stage 9 | r = 0.188 as confirmation | reported as a negative result |
| Stage 10 | "Patentability Verification" | exact-match audit, scope stated |
| Scorecard | narrative strings | computed from this run's artifacts |

Two wrong API calls were caught by static verification before they could fail at runtime:
`tnk1.context` exports `Ctx`, not `load_context`; Stage 8 takes `--cycle` / `--rl-csv`, not
`--candidates`. **38/38 API calls now resolve against the package.**

---

## Step 11 — Scale: caps found and lifted

The pipeline was not using the available data. Five caps:

| cap | was | now | knob |
|---|---|---|---|
| kinome training records | 425,906 | **~1.35M** | `data.full_kinome` |
| Stage 2 tuning folds | 4 of 25 | all 25 | `splits.tune_*_folds` |
| descriptor scaler fit | 50,000 | 200,000 | `data.desc_scaler_n` |
| Stage 8 candidate pool | 50,000 | 200,000 | `al.pool_size` |
| RL prior corpus | — | **787,283** | `rl_min.corpus` |

### Why ~1.35M and not the full ~4M

The full measurements file is approximately 946k exact + 408k censored (both carry a
pActivity) plus ~858k single-point %-inhibition and ~2.37M qualitative active/inactive
flags. The latter two have **no regression target**, so `task_loss` cannot consume them —
they need a classification head. Stage 1 drops them with a printed count rather than
coercing them to NaN.

*(The feature-store README is internally inconsistent here: its breakdown sums to 4.58M,
the filename says 3.95M, and a section header says 39.55 lakh. Worth reconciling before
quoting any count.)*

### One cap deliberately kept

`kinome_train.cap_n: 2000` looks like a data cap but is the per-kinase importance weight
`min(1, √(cap_n/N_k))`. Removing it does not add data; it lets EGFR (19,085 pairs) and KDR
(15,756) swamp TNK2 (1,350) — the closest homolog to TNK1 and the best transfer source.
With the full kinome ingested the imbalance grows, so it matters **more**.

---

## Step 12 — Scale: two caches that would have OOM'd

| cache | problem | fix |
|---|---|---|
| `_CACHE` (molecular graphs) | unbounded; ~20 kB × several hundred thousand structures ≈ **10 GB** | capped, `$TNK1_GRAPH_CACHE` (default 300k) |
| `DescStore.cache` | preloaded all 787,283 rows: ~1 GB resident + **1.26 GB transient spike**, then grew without limit | configurable `$TNK1_DESC_CACHE`, chunked transform |

Both fail **late** — hours into Stage 3 — which is the expensive way to find out.

**Measured memory fix** in `EncodedCorpus`, at true 788k scale (49,335,715 tokens):

```
list-of-lists   445 MB resident
EncodedCorpus   201 MB resident      2.2× smaller
```

**A bug found while fixing `DescStore`:** when a single call misses more rows than the cap,
evicting *before* the update cannot trim an empty dict, and evicting *after* can discard a
row that very call needs. Rewritten to build the answer first, then trim.

---

## Step 13 — Package: files added

```
TNK_Clauude.md                                   this document
CHANGES_V2.md                                    the six defects, with evidence
docs/RUNBOOK.md                                  how to run it; 8/16/32 GB profiles
tnk1-pipeline/tnk1-pipeline/
├── notebooks/TNK1_Master_Pipeline_v2.ipynb      32 cells
├── notebooks/TNK1_Master_Pipeline_v1_archived.ipynb
├── rl_min/                                      8 files — the DAP generator + README
├── tests/test_regressions_v2.py                 13 tests pinning defects 1, 3, 4
├── tests/test_rl_min.py                         26 tests for the generator
├── tests/test_memory_bounds.py                  8 tests pinning both caches
├── configs/aizynth.yml                          template (model files not included)
└── priors/README.md                             what goes there and how to get it
```

Test suite: **220 passed**, 2 deselected (network).

---

## Step 14 — Package: two runtime blockers caught at the last check

**1. Four scripts are outside the package with a different required working directory.**
`dock_controls.py`, `dock_40_molecules_tnk1.py`, `check_novelty_external.py` and
`scan_similarity_788k.py` live in the outer `Rathan_ML_Research/scripts/`. Worse,
`dock_controls.py` opens `data/tnk1_kinase.pdbqt` and `tools/vina.exe` **relative to the
outer root**. Stages 9 and 10 would have failed with "No such file" *after Stages 1–8 had
already run*. `run_stage()` now resolves across both roots with the correct cwd.

**2. `SKIP_RETRO=True` was a no-op.** It printed advice and then ran the retrosynthesis
screen anyway. `run_stage8_cycle.py` now has a real `--skip-retro` flag.

Also added a Stage 8–10 preflight in Cell 2, so a missing `vina.exe`, `meeko` or
`aizynthfinder` surfaces at minute two rather than hour fifteen.

---

## Step 15 — Current state

### Verified by execution
- The four code patches (each with a failing-before / passing-after reproduction)
- `rl_min` end to end: prior training, DAP learning, 7.8× over baseline
- Memory bounds, at true 788k scale
- 220 tests
- Notebook validity: 32 cells parse; 38/38 API calls resolve

### Verified by reading only
- Every stage-script invocation in the notebook. `run_stage1_data.py`,
  `run_stage3_transfer.py` and the rest have **never been executed** — the feature store is
  not on the development machine. The calls are shaped correctly and the known traps are
  removed, but a clean end-to-end run is not proven.

### Not fixed
- **Defect 6.** The Stage 4 surrogate still ranks worse than the Stage 2 baseline. Until
  that changes, every potency figure downstream is a ranking from a model that loses to
  Random Forest.

---

## Step 16 — What to do next

### Before launching
1. `python -m tnk1.utils.verify_features` → exit 0 only
2. `python -m pytest -q` → 220 passed
3. Export the memory profile for your box (see [`docs/RUNBOOK.md`](docs/RUNBOOK.md) §5).
   The default `TNK1_GRAPH_CACHE=300000` is ~6 GB and wrong for a 16 GB machine.

### Staged first run — do not launch all ten blind
```
Cell 1–2   ~3 min   preflight; reports anything missing
Cell 3     ~15 min  Stage 1 — check: ~2.6M rows dropped, frame ~1 GB
Cell 4     ~45 min  Stage 2
           ↑ stop and look. If these pass, the plumbing works.
Cell 5     5–9 h    Stage 3 — only commit after the above
```

### Runtime: 20–40 h
Stages **3, 7, 8** are ~80% of it. Stage 8's retrosynthesis screen is the widest band
(2–12 h); `SKIP_RETRO=True` removes it and invalidates nothing upstream.

### The gate that matters
Cell 6 prints `Critic beats baseline?`. If `NO`, fix the surrogate before reading anything
into the candidates — shrink the D-MPNN (hidden 300→128, fusion [512,256]→[128,64],
dropout 0.2→0.35), confirm the kinome pretraining transfers, or stack RF+XGB+GP.

### Then the ablations
| ablation | isolates |
|---|---|
| `sigma ∈ {30, 60, 120, 240}` | exploration/exploitation via reward scaling |
| diversity filter on/off | mode collapse |
| `beta` schedule on/off | optimism vs. pessimism under uncertainty |
| reward with/without `r_ad` | how one miscalibrated term hijacks an objective |
| **agent vs. prior at equal sample count** | **whether policy gradient beat plain sampling** |

That last row is the result worth writing up. If 1,000 RL steps do not beat 128,000 prior
samples, the honest finding is that the RL did not help — and documenting that clearly is
worth more than a scorecard implying otherwise.

---

## Closing note on method

Three of the fixes in this document were wrong on the first attempt and were caught by
writing a test that reproduced the original failure:

- the median-relative σ floor, which floors nothing when most dimensions are dead;
- the `r_sel` "no gradient" explanation, which a log-variance measurement disproved;
- the `DescStore` eviction order, which could discard a row the current call needed.

Each defect in the original pipeline shared one property: it was **silent**. Nothing raised,
nothing logged, and plausible numbers appeared in the scorecard. That is the failure mode
the new tests exist to prevent.
