# TNK1 Pipeline — v3 Full Run on HPC

**Report date:** 2026-11-20
**Cluster:** `172.17.120.214` (user3), single node `user`, NVIDIA H100 NVL (95 830 MiB)
**Environment:** conda env at `/home/user3/TO_RUN_ON_GPU/envs/tnk1`, Python 3.11, torch 2.14.1+cu130
**Config:** `configs/hpc.yaml` (`max_kinome_rows: null`, `n_members: 10`, `oof_members: 5`)
**Dataset:** `kinome_long_v3.parquet`, sha256 `76ab324c18254853a8302829ea2b5fc0b1f58024ca02b92e590a452bb22caa52`
**Resource cap:** 16 GPU shards held at any instant, enforced by a single-lane `afterany` dependency chain
**Job prefix:** `dug_*`

---

## 1. Executive summary

Five findings, in order of importance.

1. **The deep ensemble — the RL critic — is the worst model in the repository.** Over 5 paired outer folds, Random Forest reaches Spearman 0.686 / RMSE 0.675 against the ensemble's 0.501 / 0.954. Paired per-row absolute error favours RF at **p = 8.6×10⁻⁴** (n = 127). This is no longer the single-split artefact recorded as defect 6 in `CHANGES_V2.md`; it is now properly powered evidence.

2. **The ensemble's uncertainty is uninformative.** Error–uncertainty correlation is **−0.021** for the ensemble and **−0.142** for MC-dropout, versus **+0.258** for RF and **+0.283** for the Tanimoto GP. Its σ does not know where it is wrong, which undercuts the σ penalty in `r_pot`, the conformal interval, the applicability domain and EI/UCB acquisition.

3. **The run produced zero candidate molecules.** The cross-architecture audit in Stage 8 rejected all 89 unique generated molecules because the ensemble and the Tanimoto GP disagree by 1.18–2.37 log units on every one of them (tolerance 1.0). The gate behaved correctly — it caught the generator exploiting the weaker model.

4. **Transfer learning does work, within the neural family.** The ablation ladder shows scratch D-MPNN at RMSE 1.148 improving to 0.628–0.697 once pan-kinome multi-task pretraining is added — a 39–45 % reduction. The architecture's central claim is supported. It is just that the whole neural family's ceiling sits below a Morgan-count random forest at this data scale (172 exact TNK1 measurements).

5. **Everything ran without error.** All 12 completed stages exited 0 on full-scale data. The hardening work from this cycle held: no path errors, no sentinel contamination, `preflight --strict` passed, provenance fingerprints recorded for every stage.

**Overall verdict:** the engineering is sound and the experiment is informative, but the headline scientific result is negative. The honest paper reports that negative result rather than the originally intended positive one.

---

## 2. Run history

Three submissions were made. Only the third is the run of record.

| Jobs | Config | Purpose | Outcome |
|---|---|---|---|
| 2165–2176 | `hpc_smoke.yaml` | End-to-end smoke test (20 k rows, 2 members, 8 RL steps) | Passed. Exposed two real defects (below). |
| 2177–2187 | `hpc.yaml` | First full attempt | **Cancelled by us** — it held 48 shards (3 concurrent GPU jobs × 16). |
| **2190–2202** | `hpc.yaml` | **Full run of record, 16-shard cap** | 12 of 13 stages complete, all exit 0; `ablate` still running. |

Two defects the smoke run caught, both fixed before the full run:

- `rl.steps` / `batch_size` / `sigma` / `lr` were inert argparse literals, so the smoke config's `steps: 8, batch_size: 16` ran as 300 × 64 — 19.4 min instead of 0.4 min. Job 2174 is that wasted run.
- After the fix, a stale `a.steps // 30` raised `TypeError: unsupported operand type(s) for //: 'NoneType' and 'int'` **after** the expensive reward calibration. Job 2175 is that failure (exit 1); job 2176 is the 0.4-min pass after the second fix.

Both are now pinned by regression tests in `tests/test_pipeline_scorer.py`.

### 2.1 Stage timings, run of record

| Job | Stage | Device | Wall | Result |
|---|---|---|---|---|
| 2190 | preflight | CPU | 0.0 min | ok (`--strict`) |
| 2191 | baselines | CPU | 5.7 min | ok |
| 2192 | ens | GPU | 48.2 min | ok — 10 members, 71.1 MB |
| 2193 | oof | GPU | 109.3 min | ok — 5 folds × 5 members |
| 2194 | critic | CPU | 0.2 min | ok |
| 2195 | calibrate | GPU | 0.1 min | ok |
| 2196 | admet | GPU | — | skipped, already current for v3/`76ab324c1825` |
| 2197 | prior | GPU | 24.5 min | ok — 788 k-molecule corpus |
| 2198 | rl | GPU | 165.2 min | ok — 1000 DAP steps |
| 2199 | cycle | GPU | 0.3 min | ok, **but zero candidates** |
| 2200 | selectivity | GPU | 0.1 min | ok |
| 2201 | compare | GPU | 0.1 min | ok |
| 2202 | ablate | GPU | ~110 min+ | **running** (A0–A6 done, A7 pending) |

Total ≈ 6.1 h plus `ablate`. GPU occupancy never exceeded 16 shards.

---

## 3. Results

### 3.1 Classical baselines — `runs/stage2_baselines.csv`

Nested scaffold CV, held-out fold **n = 45**. The ± values are *seed* repeats on one split, not fold variance (`n_std = 0`, and kNN/GP std ≈ 1e-17 because those models are deterministic). Do not quote these as cross-validated error bars.

| Model | Best params | RMSE | MAE | Spearman | Pearson | NLL | cov90 | ECE | err–unc ρ |
|---|---|---|---|---|---|---|---|---|---|
| **RF** | `max_features 0.3, min_samples_leaf 3` | **0.439** | 0.326 | 0.628 | 0.691 | 0.531 | 0.844 | 0.042 | **+0.258** |
| XGB | `max_depth 3, lr 0.01, reg_lambda 10` | 0.449 | 0.328 | **0.640** | 0.679 | — | — | — | — |
| TanimotoGP | — | 0.474 | 0.366 | 0.474 | 0.640 | 0.609 | 0.867 | 0.026 | **+0.283** |
| kNN | `k = 5` | 0.564 | 0.425 | 0.195 | 0.487 | 1.546 | 0.733 | 0.132 | +0.186 |

Note the two models with calibrated uncertainty (RF, GP) both have *positive* error–uncertainty correlation. This is the benchmark the ensemble fails in §3.3.

### 3.2 Transfer ablation ladder — `runs/stage3_ablation.csv` (partial)

A0–A7 × 5 seeds, same held-out fold (**n = 45**), so directly comparable to §3.1. A7 — the production model — was still training at the time of writing.

| ID | Description | RMSE | Spearman |
|---|---|---|---|
| A0 | Scratch D-MPNN, TNK1 only | 1.148 ± 0.376 | 0.183 ± 0.075 |
| A1 | SSL pretrain + TNK1 fine-tune, no kinome | **5.026 ± 0.331** | 0.174 ± 0.205 |
| A2 | Kinome multi-head (Design A), zero-shot | 0.697 ± 0.061 | 0.329 ± 0.090 |
| A3 | Design A + TNK1 fine-tune | 0.673 ± 0.055 | 0.307 ± 0.064 |
| A4 | Target-conditioned (Design B), zero-shot | 0.802 ± 0.048 | **0.378 ± 0.068** |
| A5 | Design B + plain fine-tune | **0.628 ± 0.063** | 0.262 ± 0.083 |
| A6 | Design B + related-kinase weighting + fine-tune | 0.638 ± 0.026 | 0.319 ± 0.076 *(4/5 seeds)* |
| A7 | FULL: Design B + weighting + censored loss + 3-phase L2-SP | *pending* | *pending* |

Three readings:

- **Pan-kinome pretraining pays, and substantially.** A0 → A2/A3/A5 cuts RMSE from 1.148 to 0.63–0.70. That is the paper's main architectural claim and it holds.
- **The ladder is flat above A2.** Every refinement after multi-task pretraining — Design B, kinase weighting, fine-tuning — moves RMSE within 0.628–0.802 and Spearman within 0.262–0.378. No single component produces a step change. Notably, the two *zero-shot* variants (A2, A4) are competitive with the fine-tuned ones, and A4 has the best Spearman of the ladder.
- **A1 is broken, not merely weak.** RMSE 5.03 in pActivity units means predictions off by five orders of magnitude in concentration. SSL pretraining followed by fine-tuning *without* kinome conditioning diverges. This needs investigating before A1 is reported as an ablation result; right now it is a bug, not a finding.

Against §3.1 on the same fold: best ablation RMSE 0.628 vs RF 0.439; best ablation Spearman 0.378 vs XGB 0.640. The neural family loses on both axes.

### 3.3 Ensemble uncertainty quantification — `runs/stage4_uq_comparison.csv`

Held-out test fold, **n = 45**.

| Variant | RMSE | MAE | Spearman | NLL | cov90 | ECE | Sharpness | err–unc ρ |
|---|---|---|---|---|---|---|---|---|
| ensemble_raw | 0.672 | 0.503 | 0.176 | 0.944 | 0.822 | 0.0497 | 0.568 | **−0.021** |
| ensemble_cal | 0.672 | 0.503 | 0.176 | 0.933 | **0.889** | **0.0240** | 0.642 | **−0.021** |
| mc_dropout | 0.554 | 0.412 | 0.304 | 1.621 | 0.778 | 0.0532 | 0.392 | **−0.142** |

**What works:** calibration does its job. Coverage moves 0.822 → 0.889 against a 0.90 target and ECE halves, 0.0497 → 0.0240, while the point predictions are left *exactly* unchanged (identical RMSE/MAE/Spearman/Pearson). That is the correct signature for variance temperature scaling and confirms the calibrator is implemented properly.

**What does not work:** `err_unc_rho ≈ 0` means the calibrated σ is statistically independent of the actual error. The intervals have the right *average* width but point in no useful direction. MC-dropout is worse than useless at −0.142. Compare RF +0.258 and GP +0.283 from §3.1.

This fold is also a hard one — the ensemble scores ρ 0.176 here versus a 5-fold mean of 0.501 in §3.4. Another reason to prefer the 5-fold numbers.

### 3.4 Critic comparison — `runs/critic_comparison.json` ← **read this first**

The decisive experiment, added this cycle. Paired across all 5 outer folds, 172 exact TNK1 rows, 127 rows joinable per-row. The ensemble's predictions come from `stage4_oof_predictions.csv`, so no fold ever saw its own test compounds, and the baselines are fitted on exactly the same fold keys.

| Model | RMSE | MAE | Spearman | Pearson |
|---|---|---|---|---|
| **RF** | **0.675 ± 0.124** | 0.507 ± 0.099 | **0.686 ± 0.079** | **0.718 ± 0.105** |
| XGB | 0.696 ± 0.125 | 0.517 ± 0.099 | 0.683 ± 0.089 | 0.697 ± 0.109 |
| kNN | 0.793 ± 0.094 | 0.580 ± 0.099 | 0.589 ± 0.117 | 0.592 ± 0.122 |
| DeepEnsemble | 0.954 ± 0.063 | 0.693 ± 0.063 | 0.501 ± 0.121 | 0.535 ± 0.102 |

Paired fold-level tests (Wilcoxon; 5 folds cannot produce p < 0.0625, so read the direction and effect size, not the p-value):

| Comparison | Metric | mean_diff | p | Winner |
|---|---|---|---|---|
| vs RF | RMSE | −0.2785 | 0.0625 | baseline |
| vs RF | Spearman | −0.1846 | 0.1250 | baseline |
| vs XGB | RMSE | −0.2573 | 0.0625 | baseline |
| vs XGB | Spearman | −0.1816 | 0.1250 | baseline |
| vs kNN | RMSE | −0.1608 | 0.0625 | baseline |
| vs kNN | Spearman | −0.0882 | 0.4375 | baseline |

Paired per-row absolute error — the properly powered test, same compounds joined on InChIKey block 1:

| Comparison | n | mean \|err\| diff | p | Winner |
|---|---|---|---|---|
| vs RF | 127 | −0.1881 | **8.6×10⁻⁴** | baseline |
| vs XGB | 127 | −0.1798 | **3.6×10⁻³** | baseline |
| vs kNN | 127 | −0.1107 | 4.0×10⁻² | baseline |

Sign convention: negative means the baseline is better. **Every metric, every baseline, both test levels — the ensemble loses.**

For context against the previous data version: the ensemble improved markedly on v3 (Spearman 0.264 → 0.501), but RF improved too (0.641 → 0.686). The clean-data rebuild helped both models and did not close the gap.

### 3.5 Selectivity controls — `runs/stage5_selectivity_reference.csv`

A sanity check: known pan-kinase drugs must **not** be predicted TNK1-selective.

| Compound | μ_sel | σ_sel | Fold selectivity (10^μ) |
|---|---|---|---|
| dasatinib | −2.090 | 0.562 | 0.008× |
| bosutinib | −0.982 | 0.346 | 0.104× |
| TP-0903 | −1.144 | 0.313 | 0.072× |

All three are predicted 10–120× *less* potent on TNK1 than on the off-target panel. Correct behaviour — the check passes.

### 3.6 Prior and reinforcement learning — `runs/rl_min/`

Reward calibration against 10 000 prior samples: `pot_c = 6.165`, `sel_c = −0.455`, n = 9 638, target fraction 0.30, 636 s.

Prior baseline over 2 000 samples: mean reward **0.0743**, max 0.7504, validity 0.964, 1 841 distinct scaffolds.

DAP configuration: 1000 steps, batch 128, σ 120.0, lr 1e-4, max_len 128, temperature 1.0, grad_clip 5.0, diversity-filter bucket 25 at min score 0.55, replay 100/16, seed 42.

| Metric | Start | End |
|---|---|---|
| Mean reward | 0.0919 *(first-10-step mean)* | **0.5410** *(last-10-step mean)* |
| Max reward in batch | 0.644 | 0.880 |
| Best ever (replay) | — | 0.8957 |
| Validity | 0.961 | 0.977–1.000 |
| Uniqueness | 1.000 | 0.91–1.00 |
| Scaffolds per 128-batch | 123 | 117 |
| Novel fraction | 1.000 | 0.45–0.50 |
| Prior NLL | 20.18 | 42–44 |
| Agent NLL | 21.33 | 11.5 |

Verdict: **LEARNED** — 7.3× the prior baseline, measured against the prior rather than against the agent's own first step, which is the correct control.

**What is healthy:** validity stayed at or above 0.96 throughout, uniqueness never collapsed, and ~117 distinct scaffolds per 128-molecule batch means no mode collapse. There is no malformed or unkekulizable output.

**What is concerning:** prior NLL more than doubled, 20 → 43. DAP's entire purpose is to keep the agent anchored to the prior; a doubling means the agent has drifted far into chemical space the prior considers unlikely. Combined with §3.4, the generator spent 165 minutes optimising a model we now know ranks worse than a random forest. Novel fraction also fell to ~0.47, meaning roughly half of each late batch repeats molecules seen earlier in the run.

Top-scoring molecule (reward 0.8957):

```
N#Cc1cc(-c2ncnc3[nH]c(-c4ccc(C(=O)O)cc4F)cc23)nc(N2CCS(=O)(=O)CC2)c1
```

A 7-azaindole/pyrrolopyrimidine hinge binder with a fluorobenzoic acid and a sulfone-piperidine. Chemically sensible as a kinase scaffold.

### 3.7 Candidate funnel — `runs/cycle_02/` → **zero candidates**

The stage reported "ok in 0.3 min" and wrote a header-only CSV. Traced gate by gate:

| Step | Survivors | Note |
|---|---|---|
| A. Pool from `best_molecules.csv` | 100 | **`al.pool_size` is 200 000** |
| B. Standardised successfully | 100 | |
| C. Unique by InChIKey block 1 | 89 | 11 duplicates removed |
| D. Not already in ChEMBL / in-house | 89 | all novel |
| E. Hard gate `R > 0` | 89 | all pass |
| F. Cross-architecture audit `\|μ − μ_GP\| ≤ 1.0` | **0** | **all rejected** |

Reward components on the 89 survivors of step E:

| Component | Weight | min | median | max |
|---|---|---|---|---|
| pot | 3.0 | 0.582 | 0.766 | 0.898 |
| sel | 2.5 | 0.091 | **0.113** | 0.144 |
| hinge | 2.0 | 1.000 | 1.000 | 1.000 |
| ad | 1.0 | 0.908 | 1.000 | 1.000 |
| sa | 1.5 | 0.644 | 0.712 | 0.808 |
| safety | 2.5 | 0.864 | 0.905 | 0.923 |
| adme | *(not in V2)* | 0.000 | 0.000 | 0.000 | ← see §4.1 |
| **R** | | 0.533 | 0.567 | 0.616 |

The audit numbers:

```
|mu - mu_gp|   min 1.180   median 1.770   max 2.366      tolerance 1.000
mu     (ensemble)  median 8.214   ->  approx 6 nM
mu_gp  (Tanimoto GP) median 6.471   ->  approx 340 nM
```

A **systematic 1.8-log (≈60×) optimism gap on every single molecule**. The ensemble believes the generated compounds are single-digit nanomolar; an independent model class trained on the same 172 measurements says mid-nanomolar-to-micromolar. This is the signature of reward hacking, and the audit gate caught it. **The gate was right. Loosening `audit_tol` to obtain candidates would manufacture results, not discover them.**

A second problem is visible in the table above: **selectivity was never achieved.** `r_sel` sits at 0.091–0.144 against a weight of 2.5, i.e. at the floor. The generator maximised potency, hinge contact, applicability domain, synthesisability and safety, and abandoned the one property the project is named for.

### 3.8 ADMET profile of the generated set

Predicted endpoints over the 100 generated molecules (median, with range):

| Endpoint | min | median | max | Reading |
|---|---|---|---|---|
| solubility (logS) | −5.36 | −4.25 | −3.62 | acceptable |
| **caco2 (logPapp)** | −6.59 | **−6.03** | −5.55 | **poor — threshold is −5.15** |
| microsome (min) | 0.46 | 12.29 | 32.44 | moderate |
| p(hERG) | 0.012 | 0.032 | 0.073 | good, well under the 0.20 TPP limit |
| p(Ames) | 0.024 | 0.074 | 0.186 | good, under the 0.30 limit |
| p(CYP3A4) | 0.007 | 0.040 | 0.105 | good |
| p(CYP2D6) | 0.0002 | 0.004 | 0.031 | good |

Decomposing the ADME term: solubility sigmoid 0.906, **Caco-2 sigmoid 0.066**, microsomal-stability sigmoid 0.762, giving `adme = 0.358`. Permeability is the single bottleneck, and it is a direct consequence of the generator favouring carboxylic acids. The safety profile, by contrast, is genuinely good — notably better than the `TNK1-NOV-C01-012` liability at p(hERG) = 0.82 that remains in the old Cycle-01 contract.

---

## 4. Defects found during this run

### 4.1 `adme` and `qed` are reported as all-zeros — **fixed 2026-11-20**

`tnk1/rl/scorer.py:139`:

```python
total = R.aggregate(comps, w, gate, rl.get("veto"), components=self.components)
for c in self.components: out[c][vi] = comps[c]
```

`self.components` is `COMPONENTS_V2 = [pot, sel, hinge, ad, sa, safety]`. `comps["adme"]` and `comps["qed"]` are computed on the line above and then silently discarded — `out["adme"]` and `out["qed"]` keep their `np.zeros(n)` initialisation.

`scripts/run_stage8_cycle.py:67` then uses that column as one of six Pareto objectives:

```python
objs = np.c_[P.mu - P.sigma_epi, P.mu_sel - P.sigma_sel, P.adme, P.ad, P.sa, P.safety]
```

A constant column cannot affect Pareto dominance — every pair ties on it. **ADME has therefore never influenced candidate selection**, and `qed` is written into the candidate CSV as a column of zeros. Verified that the real values exist and are non-trivial: `adme` median 0.358 when computed directly.

This is the same defect class as the inert config keys fixed earlier this cycle: a quantity that is computed, reported, and never actually used. It did not change this run's outcome only because the audit gate emptied the pool before the Pareto step.

**Fixed.** `tnk1/rl/scorer.py` now scatters every key of `comps`, not just the ones
`self.components` aggregates, so a component added later cannot be silently discarded
either. Pinned by 6 tests in `tests/test_component_scatter.py`, 5 of which fail against
the old loop. **This run's `adme`/`qed` columns remain zeros** — the fix changes future
runs, not the artifacts already in `runs/`.

### 4.2 Stage 8 receives 100 molecules instead of 200 000 — **fixed 2026-11-20**

`al.pool_size` is 200 000, but the chain passes `--rl-csv runs/rl_min/best_molecules.csv`, which is the top-100 replay buffer. Nothing in the chain samples a large pool from the trained `agent.pt`. The funnel is starved by three orders of magnitude before any filter runs, so even a working audit gate would have little to select from.

**Fixed.** `scripts/run_stage8_cycle.py` gained `pool_from_agent`, which samples
`al.pool_size` unique SMILES on-policy from `agent.pt`, and the orchestrator now passes
`--agent`/`--vocab`. A pool short of `al.pool_size` raises rather than proceeding;
`--allow-small-pool` is the logged opt-out for smoke runs. Pinned by 11 tests in
`tests/test_stage8_pool.py`. The sampling itself is untested on real weights — no torch
or `agent.pt` exists off-cluster, so the first cluster run must check the reported
unique-sample count against `al.pool_size`.

### 4.3 Uneven statistical footing — **methodological**

`stage2_baselines.csv` and `stage4_uq_comparison.csv` report **n = 45**, a single held-out fold repeated over seeds. `critic_comparison.*` reports 5 genuinely paired outer folds and 127 per-row pairs. Only the latter supports a claim about model ranking. Quote the critic file in any write-up, not stage 2.

### 4.4 A1 ablation diverges — **new, needs investigation**

RMSE 5.03 means predictions wrong by five log units. SSL pretraining plus fine-tuning without kinome conditioning does not train. Until diagnosed, A1 is a bug and must not be reported as an ablation finding.

### 4.5 Defects fixed earlier this cycle, confirmed working here

| Defect | Evidence it is fixed |
|---|---|
| 70.3 % sentinel-value contamination in the activity table | v3 rebuilt clean, 1 103 498 rows; `preflight --strict` passes |
| CWD-dependent relative paths | all stages ran correctly from `SLURM_SUBMIT_DIR` |
| `PipelineScorer` never wired into RL | `meta.json` records `"scorer": "pipeline"` with the 6 production components |
| 8 inert config keys (of 181) | 0 remain; `rl.steps: 1000` and `batch_size: 128` took effect as logged |
| Stereochemistry stripped by the tautomer canonicaliser | blast radius was 0 of 40 and 0 of 380 molecules — no published result affected |
| Unseeded docking | `DOCK_SEED = 42` passed to Vina |
| `--help` ran destructive scripts | three scripts now require `--force` |
| Frozen-artifact immutability bypass | `--force` now required to replace a frozen table |

---

## 5. What may and may not be claimed

**Supported by this run:**

- Pan-kinome multi-task pretraining substantially improves a D-MPNN over training from scratch on TNK1 alone (RMSE 1.148 → 0.63–0.70).
- Variance temperature scaling and split-conformal calibration are correctly implemented and improve coverage and ECE without touching point predictions.
- A DAP policy-gradient generator trained against the production reward reaches 7.3× the prior's mean reward while maintaining validity ≥ 0.96 and without mode collapse.
- A cross-architecture audit gate successfully detects and blocks reward hacking by a generative model.
- On 172 exact TNK1 measurements, a heteroscedastic multi-task deep ensemble is **outperformed** by a Morgan-count random forest on ranking and error, and its uncertainty estimates are uninformative.

**Not supported — do not claim:**

- That the ensemble is a better TNK1 predictor than classical baselines. The paired evidence says the opposite.
- That the ensemble's uncertainty is meaningful. `err_unc_rho ≈ 0`.
- That any generated molecule is a TNK1 inhibitor. Zero candidates cleared the audit; nothing was docked; there is no wet-lab data.
- That the generated molecules are selective for TNK1. `r_sel` was at the floor throughout.
- That the generated molecules are synthesisable. `--skip-retro` was used; rows carry `retro_ok=True` by assertion, not evidence, and the stage log says so explicitly.
- That ADME influenced candidate selection. Per §4.1 it never has.
- Any number from `stage2_baselines.csv` or `stage4_uq_comparison.csv` as a cross-validated result. They are single-fold.

---

## 6. Recommended next steps

**Priority 1 — finish and read `ablate`.** `runs/stage3_ablation.csv` on job 2202 completion. A7 is the production model; its position relative to A5/A6 determines whether the three-phase L2-SP fine-tune, censored loss and kinase weighting earn their complexity. Partial data suggests they will not.

**Priority 2 — choose the paper's frame.** The recommendation is to **report the negative result as the headline, then fix the critic and show the pipeline works with it.** You now hold paired, 5-fold, per-row-significant evidence that a pan-kinome multi-task D-MPNN ensemble loses to a Morgan-count random forest at this data scale and produces uninformative uncertainty — plus a clean demonstration of an audit gate catching a generator exploiting a weak critic. Both are real, defensible contributions, and more valuable than a positive claim the data will not carry.

**Priority 3 — fix the three concrete defects** before any rerun. Items 1 and 2 are done; item 3 is the one that still blocks a rerun worth the GPU hours, since a rerun against the same critic reproduces the same 1.8-log optimism gap and the same empty funnel:

1. ~~The `adme`/`qed` scatter at `tnk1/rl/scorer.py:139`~~ — **done 2026-11-20**, §4.1.
2. ~~Sample a real pool from `agent.pt` for Stage 8~~ — **done 2026-11-20**, §4.2.
3. **Still open.** Back `r_pot` / `r_sel` with RF or a stacked RF/GP. `tnk1/rl/scorer.py` touches the critic through only two duck-typed methods, `predict_members` and `latent`, so the change is contained.

**Priority 4 — rerun only `rl` → `cycle`** (≈2.8 h GPU). `ens`, `oof`, `calibrate`, `prior` and `admet` are valid and provenance-cached against `76ab324c1825`; the orchestrator will skip them.

**Priority 5 — address the selectivity failure.** `r_sel` never rose above 0.144. Either `sel_c = −0.455` is mis-calibrated for the achievable range, or the selectivity signal is too weak in the off-target panel to optimise against. Investigate before the next RL run, since selectivity is the project's stated objective.

**Do not:** loosen `audit_tol`, drop the GP audit, or re-enable `--skip-retro` in anything reported as a result.

### 6.1 Open items unchanged by this run

- Nothing in the repository rebuilds the 12 feature-store files; a `scripts/build_feature_store.py` with a sha256 acceptance test against `manifest.json` is still missing.
- The 2026-10-26 data harvest was incomplete — 8 of 11 sources (DTC and KIBA failed, METZ returned 0).
- `TNK1-NOV-C01-012` remains in the Cycle-01 contract at p(hERG) = 0.8199.
- No wet-lab data exists, so Stage 9 is a no-op.
- The existing paper draft and figures quote v1 numbers throughout and must be regenerated.
- The local working tree holds ~40 modified and ~10 new files, uncommitted.

---

## 7. File index

Downloaded to `TO_RUN_ON_GPU/results_hpc/`; live at `/home/user3/TO_RUN_ON_GPU/runs/` on the cluster.

| File | Contents |
|---|---|
| `runs/critic_comparison.json` | **Start here.** Paired RF/XGB/kNN vs ensemble, 5 folds + per-row |
| `runs/critic_comparison.csv` | Per fold × model metrics |
| `runs/critic_comparison_per_row.csv` | Per-compound predictions for every baseline |
| `runs/pipeline/critic.log` | The same comparison, formatted, with the remediation options |
| `runs/stage4_uq_comparison.csv` | Ensemble raw vs calibrated vs MC-dropout; the `err_unc_rho` finding |
| `runs/stage4_oof_predictions.csv` | Out-of-fold ensemble predictions, 5 folds |
| `runs/stage2_baselines.csv` | Classical baselines (single fold — see §4.3) |
| `runs/stage3_ablation.csv` | A0–A7 ladder *(written on job 2202 completion)* |
| `runs/pipeline/ablate.log` | Per-seed ablation rows, readable while running |
| `runs/stage5_selectivity_reference.csv` | Control-drug selectivity sanity check |
| `runs/rl_min/history.csv` | 1000-step trace: reward, validity, uniqueness, scaffolds, NLLs |
| `runs/rl_min/meta.json` | Scorer, weights, reward calibration, DAP config, verdict |
| `runs/rl_min/best_molecules.csv` | The 100 generated molecules with scores |
| `runs/rl_min/agent.pt` | Trained generator (22.4 MB, on cluster only) |
| `runs/cycle_02/candidates_pre_docking.csv` | Header only — zero candidates |
| `runs/pipeline/cycle.log` | The zero-candidate message and the `--skip-retro` warning |
| `runs/pipeline/last_run.json` | Provenance record: stage versions and dataset fingerprints |
| `logs/dug_*.out` | Per-job Slurm output for all 31 jobs across the three submissions |

### 7.1 Reproduction

```bash
# on the cluster
cd /home/user3/TO_RUN_ON_GPU
./hpc/run_pipeline.sh                 # full chain, 16-shard cap, refuses to double-submit
./hpc/run_pipeline.sh --from rl       # resume from a stage
sbatch hpc/submit_gpu.sh <stage>      # single stage, bypasses the lane cap

# shard accounting
squeue -u $USER -o "%.7i %.16j %.2t %.8M %.20b"
```

`TNK1_GPU_LANES=2` would allow two concurrent GPU jobs at 32 shards. It defaults to 1. `TNK1_ALLOW_CONCURRENT=1` overrides the already-queued guard; neither should be needed.
