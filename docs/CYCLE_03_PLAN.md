# Cycle 03 — what was built, what to run, and what to collect

**Written:** 2026-11-20
**Status of the code:** committed and tested (364 hermetic tests pass; 13 skip without RDKit/torch). **Nothing in this document has been run on real data yet.**

This is the companion to [`RUN_REPORT_V3_HPC.md`](../RUN_REPORT_V3_HPC.md), which records what the v3 run established. That report's conclusion was negative and correct: the deep ensemble that drives the RL reward is the worst model in the repository, its uncertainty is uninformative, and the generator exploited it. Everything below is addressed at that.

---

## 1. The one number that governs everything

**172 exact TNK1 measurements.**

Every limitation in the v3 report follows from it, and no amount of modelling removes it:

| Consequence | The number |
|---|---|
| A scaffold-split test fold holds ~34 molecules | Spearman standard error ≈ 0.10, so **differences below ~0.15 are noise** |
| 5 outer folds | exact Wilcoxon floor is p ≥ 0.0625 — **it cannot reach p < 0.05 at all** |
| Model-selection overfitting | Cawley & Talbot (JMLR 11:2079, 2010) measure it as "comparable in magnitude to differences in performance between learning algorithms" — i.e. **the size of the 0.686-vs-0.501 gap this project reasons from** |
| Conformal calibration | Xu et al. (arXiv:2304.00970, Merck) **excluded their 203-molecule dataset as too small**. 172 is below the floor of the best QSAR conformal study. |

So read §4 before §3. More data is worth more than any algorithm in this repository.

---

## 2. What was added

Nine modules, five scripts, two HPC drivers, 153 tests. The design rule throughout: **the mathematics is numpy and tested; the torch is a thin wrapper.** That is why 153 tests run on a laptop with no GPU.

### Base models — `tnk1/zoo/`
26 registered regressors × 21 representations, behind one duck type (`fit`/`predict(return_std=)`) that `tnk1/rl/scorer.py` already consumes.

- **Gaussian processes work without gpytorch.** `MinMaxKernel` and `TanimotoKernel` are implemented as real `sklearn.gaussian_process.kernels.Kernel` subclasses, so all five GP variants run in the numpy-only environment. The MinMax/Tanimoto distinction is registered as two models because on *count* vectors they are not the same kernel, and nothing here had tested which is better.
- **SVR is prioritised over RF.** Jiang et al. (J Cheminform 13:12, 2021), 50-seed means: SVM is the best model on the two smallest regression sets and beats RF by 0.29 RMSE on FreeSolv, while **RF is second-worst or worst among descriptor models on all three**. This project's "RF beats the D-MPNN ensemble" matches their model-*class* finding; the within-class ordering is untested, and **RF is probably not the ceiling.**
- **Two representations the pipeline was missing.** Kamuntavičius et al. (J Cheminform 17:108, 2025), 25 datasets, Friedman + Nemenyi: the best regression combination is `rdkit_desc + ERG + ECFP4 + Avalon`, and they state explicitly that *"the popular rdkit_desc + ecfp4 combination was not the best choice"* — which is precisely this project's Stage-2 representation. **ERG and Avalon are now registered**, as is their combination (`admet_best_combo`).
- `mean_baseline` is registered deliberately. An RMSE of 0.954 means nothing until the reader knows the mean predictor's.

### Ensembling — `tnk1/ensembles/`
Nine combiners plus the guard that matters:

- **`DiscreteSuperLearner`** evaluates every combiner *and every individual model* as candidates and picks the lowest CV risk (Phillips, van der Laan, Lee & Gruber, Int J Epidemiol 2023, step 5). This project shipped a combination worse than a single random forest and optimised a generator against it for 6 GPU-hours. **That is now structurally impossible**, and a test asserts it over randomised trials.
- NNLS on the simplex (Breiman 1996 — the non-negativity constraint is what buys the guarantee), bagged Caruana greedy selection optimising **Spearman directly**, and a rank-NNLS variant invariant to member scale.
- `combine.py`: moment-matching with the aleatoric/epistemic split for the conformal layer, the exact mixture for scoring, Yao et al. (2018) log-score stacking on the simplex with a Pseudo-BMA+ warm start, and **`PoolThenCalibrate`** — Rahaman & Thiery (NeurIPS 2021) show temperature scaling must come *after* averaging, and the natural way to write it is the wrong way round.

### Uncertainty — `tnk1/uq/conformal.py`
Mondrian (difficulty-conditional) conformal, CQR (Romano et al. 2019), CV+ (Barber et al. 2021), plus `conditional_coverage` — **the diagnostic this project lacks.** A worked example in the test suite: split conformal gives 89.5% marginal coverage assembled from bins at 100%/98%/96%/**64%**. The 64% bin is the novel scaffolds the pipeline actually makes decisions about. Mondrian narrows the easy bins and widens the hard one, improving both marginal coverage *and* mean width.

`metrics.py` gains **calibrated NLL**: a σ can be mis-scaled (one scalar fixes it) or uninformative (nothing does), and plain NLL cannot distinguish them. The ensemble's error-σ correlation of −0.021 predicts recalibration will not rescue it; `cnll` makes that checkable rather than asserted.

### RL — `rl_min/algos/`
Six objectives behind the existing `train_dap` signature, so `--algo` is the only change and `dap` remains the default:

| Algorithm | Paper | Why for this project |
|---|---|---|
| `ahc` | Thomas et al., J Cheminform 14:68 (2022) | DAP on the top-k only. The reward costs an ensemble pass + 7 ADMET models per molecule; sample efficiency is the binding constraint |
| `grpo` | Shao et al. 2024 (DeepSeekMath) | Critic-free: the batch *is* the group. No value network fitted on the same expensive reward |
| `rloo` | Ahmadian et al., ACL 2024 | Unbiased leave-one-out baseline; reported to beat PPO for sequence alignment at a fraction of the complexity |
| `ppo` | Schulman et al. 2017 | The clipped-surrogate control |
| `gflownet` | Malkin et al., NeurIPS 2022 | **Samples proportional to reward rather than seeking the mode.** The structurally different answer to the diversity and selectivity collapse. On a SMILES tree P_B = 1, so trajectory balance reduces to `(log Z + log P(x) − β log R(x))²` using the existing `nll` |

And the two fixes for the actual failures, in `algos/common.py`:

- **`pessimistic_reward`** — `μ − κ·σ_epi`. MOPO (Yu et al. 2020) / MOReL (Kidambi et al. 2020): an agent optimising a learned model finds that model's errors, so the model's uncertainty must cost it something *during training*. With `StackedCritic`, `σ_epi` **is** the cross-architecture disagreement the audit gate measured at 1.8 log units after the fact.
- **`multiplicative_gate`** / **`lagrangian_reward`** — under the v3 weights, abandoning selectivity entirely forfeits only 2.5/12.5 = **20%** of a weighted-sum reward, so there is no structural pressure to lift a component that is hard to move. A product (Derringer & Suich 1980) makes it unforfeitable; a Lagrangian multiplier grows while the constraint is violated, which is the opposite of a fixed weight's behaviour.

### Critic swap — `tnk1/zoo/critic.py`
`ZooCritic` / `StackedCritic` provide `predict_members` and `latent`, the only two methods the scorer uses. **Synthetic-member contract, stated plainly:** two members at `μ ± σ_epi` reproduce `(μ, σ_epi, σ_ale)` exactly through `decompose`. They are *not* independently trained members and `report()` says so, so no run can be mistaken for a deep-ensemble run.

### Transfer — `tnk1/transfer/`
`finetune.py`: ten named strategies (LP-FT, surgical first/last, gradual unfreezing, SWA, EMA, mixout, head-only floor) each citing its paper, all wrapping the existing `finetune_tnk1`.

`task_selection.py` asks the question nothing here has asked: **which of the 491 kinases belong in the pretraining set at all?** The current stage weights them all by pocket similarity and selects none, so a kinase whose SAR runs *opposite* to TNK1's still contributes gradient. Numpy, tested, seconds to run.

---

## 3. What to run, in order

### Step 1 — `taskselect`. CPU, seconds. Do this first.
```bash
python scripts/run_stage3b_transfer.py --version v3 --task-selection-only
```
Per donor kinase: Spearman correlation with TNK1 over compounds measured on both. Watch for two numbers:
- **how many of the 491 can be judged at all** (donors with <10 shared compounds cannot, and are currently weighted on pocket geometry alone);
- **how many are anti-correlated.** Any donor at ρ ≤ −0.3 is being trained on right now, and the trunk is being asked to fit both directions at once.

This costs nothing and may change what Step 4 should be pretrained on.

### Step 2 — `zoo`. CPU, ~1–2 h. The question: is RF actually the ceiling?
```bash
bash hpc/run_sweeps.sh models          # zoo -> stack, both CPU
# or locally:
python scripts/run_stage2b_zoo.py --version v3 --tag core --repeats 10
```
Nested CV, hyperparameters on inner folds only, 10 repeats → 50 fold scores, every challenger tested against `rf` under Holm. **Install `lightgbm`, `catboost` and `ngboost` first** or six models are excluded; `run_sweeps.sh` prints what is missing before queueing anything.

Read `err_unc_rho` as carefully as `spearman`. A model with ρ(|error|, σ) > +0.20 has a σ worth putting in the reward and the conformal interval; the deep ensemble's is −0.021.

### Step 3 — `stack`. CPU, minutes.
```bash
python scripts/run_stage4b_stack.py --version v3 --metric spearman
```
The discrete super learner reports `stack_beats_best_single`. **If it is `false`, that is a real answer** — ship the single model and say so. Writes `data/processed/stack.pkl`.

### Step 4 — the critic swap, then `rl` → `cycle`. ~2.8 h GPU.
Set `critic.backend: zoo_stack` in the config, then:
```bash
bash hpc/run_pipeline.sh full rl
```
This is the only change in Cycle 03 that can alter a scientific outcome. A re-run against the *same* deep ensemble reproduces the same 1.8-log optimism gap and the same empty funnel, so do not spend the GPU hours until the critic has changed.

**Four things read the ensemble's σ** — `r_pot`'s penalty, the conformal `q̂`, the applicability domain, and EI/UCB acquisition. Swapping the critic means re-deriving all four, not just the point predictions. `calibrate` must re-run.

### Step 5 — the RL algorithm array. 6 × ~2.8 h GPU, 2 concurrent.
```bash
sbatch --array=0-5%2 hpc/submit_array.sh rlalgo
python scripts/compare_sweeps.py
```
**Do not rank these runs on `score_mean`.** The v3 agent reached 7.3× the prior's reward and produced zero candidates. Rank on `cycle_candidates` — the audit-gate survivors — once Stage 8 has run against each agent.

### Step 6 — the transfer ladder. Comparable in cost to `ablate`.
```bash
sbatch --array=0-9%2 hpc/submit_array.sh transfer
```
Ten fine-tuning strategies × 5 folds, each needing its own kinome trunk. Like `ablate`, this justifies an architecture choice rather than producing molecules — run it when the critic question is settled.

### Priority 1 from the v3 report is still open
`runs/stage3_ablation.csv` on job 2202. A7 is the production model and its position relative to A5/A6 decides whether the three-phase L2-SP fine-tune, censored loss and kinase weighting earn their complexity. **A1 also diverges at RMSE 5.03 and must not be reported as an ablation finding until diagnosed.**

---

## 4. Data collection — the part that actually matters

At 172 measurements, **40 new TNK1 data points is a 23% increase in the training set. No algorithm in §2 can do that.** In rough order of value per unit effort:

### 4a. Finish the harvest that already failed
The 2026-10-26 run got **8 of 11 sources**: DTC and KIBA failed outright, METZ returned 0 rows. `scripts/tnk1_harvest.py` is resumable. This is free data already scoped, already coded, and already known to be missing.

### 4b. Published kinase-profiling panels — the largest untapped TNK1 source
TNK1 appears as a column in several large panels that are not in the current table. Each contributes exact TNK1 values *and* pan-kinome rows:

- **Klaeger et al., "The target landscape of clinical kinase drugs", Science 358:eaan4368 (2017)** — 243 clinical kinase inhibitors × the kinome by chemical proteomics. Supplementary tables are downloadable.
- **Davis et al., Nat Biotechnol 29:1046 (2011)** — KINOMEscan, 72 inhibitors × 442 kinases.
- **Metz et al., Nat Chem Biol 7:200 (2011)** — the source the harvest got 0 rows from; worth re-attempting directly from the supplement rather than through an API.
- **Anastassiadis et al., Nat Biotechnol 29:1039 (2011)** — 178 inhibitors × 300 kinases.
- **Drug Target Commons** and **KIBA** — the two that failed.

> Verify each panel actually contains TNK1 (some kinome panels omit it) before building a loader. I have not checked these specific tables for a TNK1 column — treat the list as leads, not as confirmed sources.

### 4c. Curated aggregations
- **Papyrus** (Béquignon et al., J Cheminform 15:3, 2023) — a standardised, quality-controlled bioactivity set built from ChEMBL + ExCAPE-DB. Designed for exactly this problem and would also give a cleaner pan-kinome table than the current harvest.
- **ChEMBL release check.** The table is built from ChEMBL 34. Confirm whether a newer release adds TNK1 rows.
- **BindingDB**, **PubChem BioAssay**, **Open Targets**.

### 4d. Prospective measurement — the highest-value action available
No wet-lab data exists, so Stage 9 is a no-op and the whole pipeline has never been validated prospectively. The Cycle-01 contract (40 compounds) is ready. Two blockers to settle first:

1. **`TNK1-NOV-C01-012` at p(hERG) = 0.82**, above the configured 0.85 ceiling only because the hERG hard filter was never applied offline. It has needed a PI decision since 2026-11-09. 29 of 40 exceed the TPP's 0.20 limit and 35 of 40 fail the full TPP.
2. **The Cycle-01 contract was selected by the critic that lost the paired comparison.** If you are commissioning synthesis now, re-score the 40 against the stacked critic first — it costs minutes and may reorder them.

Measured values, including **inactives**, are worth more than actives: the current table's censored-row handling is correct (`task_loss` penalises only `max(0, μ − bound)`), so ">10 µM" results are usable training data, not waste.

### 4e. What will *not* help
- **More ensemble members.** Lakshminarayanan et al. (2017) recommend M=5 and note RMSE is sometimes worse than baselines because the method optimises NLL. A 10-member heteroscedastic D-MPNN on 172 points is near the worst case.
- **Pretrained molecular foundation models, for now.** Kamuntavičius et al. (2025) found Mol2vec, GROVER, MolFormer, BARTSMILES and MegaMolBART all "did not perform well" against descriptor baselines on 25 ADMET datasets. Run the classical sweep first; revisit only if the zoo plateaus well below usefulness.
- **Deep kernel learning and sparse GPs.** GAUCHE's own benchmarks report that deep Bayesian models "generally did not outperform the best shallow kernels", and at N=172 an exact GP is exact and costs microseconds.

---

## 5. Hard constraints — do not relax these

From the v3 report, unchanged:

- **Do not loosen `audit_tol`**, drop the GP audit, or re-enable `--skip-retro` in anything reported as a result. The gate was right; loosening it manufactures results.
- **Do not quote `stage2_baselines.csv` or `stage4_uq_comparison.csv`** as cross-validated. They are single-fold, n = 45.

New, from this cycle:

- **Do not report a `--repeats 1` zoo run.** It reuses only the frozen splits; the Wilcoxon floor applies and the corrected t-test has 5 points.
- **Do not report `in_sample_report_do_not_quote`** from `stage4b_stack.json`. The honest number is `discrete_super_learner.cv_risk`.
- **Do not describe `ZooCritic`'s two synthetic members as an ensemble.** They are a carrier for one model's own uncertainty.
- **A smoke config (`repeats: 1`, `min_rows: 5`) must never be reported.** It proves the chain runs.

---

## 6. Honest statement of what is and is not verified

**Verified here:** every module imports and runs; 364 hermetic tests pass, including randomised-trial assertions on the discrete-super-learner guard and a worked demonstration that Mondrian conformal fixes a conditional-coverage failure split conformal hides. The orchestrator builds its plan with all four new stages. All shell scripts parse. Both trees are byte-identical.

**Not verified:** nothing has been run on the real 172-molecule table or on a GPU. In particular —

- the model-zoo ranking is unmeasured; the claim that SVR/GP may beat RF is a literature expectation, not a result here;
- the six RL trainers have had their mathematics tested and their shapes checked, but **no gradient step has been taken with any of them**;
- `pool_from_agent` (the Stage-8 pool fix from earlier today) is still untested on real weights — the first cluster run must check the reported unique-sample count against `al.pool_size`;
- `ft_mixout` applies mixout per *epoch*, not per forward pass as the paper does. Flagged in its docstring.

Two of four literature reviews commissioned for this work are still outstanding (transfer-learning foundation models; the RL/reward-hacking survey). The RL algorithms above were implemented from primary knowledge of the papers cited and may need reconciling when those land.
