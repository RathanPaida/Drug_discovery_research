# TNK1 AI Pipeline (Team A)

Implementation of *"TNK1 AI Pipeline: Complete Model & Step-by-Step Technical Manual"*: kinome transfer learning
(D-MPNN + ESM-2/KLIFS conditioning) -> calibrated heteroscedastic ensemble -> selectivity/ADMET -> REINVENT 4 with an
LCB multi-objective reward -> batch active learning with Team B.

## Layout
| Stage | Code | Entry point |
|---|---|---|
| 0 env/seeds/hash-lock/tracking | `tnk1/utils/common.py`, `envs/*.yml` | `make stage0` |
| 1 data, RDKit standardisation, splits | `tnk1/data/{extract,standardize,activity,splits}.py` | `make stage1` |
| 2 baselines (kNN, RF, XGB, Tanimoto-GP; 2.5 = A0) | `tnk1/baselines/` | `make stage2` |
| 3 D-MPNN, SSL pretrain, Design A/B, weighting, censored loss, L2-SP, ablations | `tnk1/models/` | `make store pretrain lambda ablate` |
| 4 ensemble/MC-dropout/evidential, temperature + conformal, dual-space AD | `tnk1/uq/` | `make ens oof calibrate compare` |
| 5 selectivity, 7-endpoint ADMET, filters | `tnk1/selectivity/`, `tnk1/admet/` | `make admet selectivity` |
| 6 kinase prior (REINVENT 4 TL) | `tnk1/rl/reinvent_io.py` | `make prior-prep` -> `reinvent tl.toml` -> `make prior-qc` |
| 7 LCB reward, beta(t), scoring service | `tnk1/rl/{reward,scorer,server,reinvent_client}.py` | `make rl-config`, `make rl-serve` |
| 8 acquisition, Butina batch, contract, ingest | `tnk1/active_learning/` | `scripts/run_stage8_cycle.py {prepare,select,ingest}` |
| 9 stats (Clopper-Pearson, Wilcoxon, Holm) | `tnk1/evaluation/stats.py` | `scripts/run_stage9_report.py` |

Run order: stage1 -> stage2 -> store -> pretrain -> lambda -> ablate -> oof -> ens -> calibrate -> compare -> admet ->
selectivity -> prior-prep (+reinvent) -> prior-qc -> rl-serve & reinvent -> Stage 8 cycles.
RL runs across two conda envs: `reinvent` (tnk1-reinvent) calls `reinvent_client.py`, which POSTs to the REST scorer in tnk1-predict.

## Verification status (be aware)
* **Run and passing here:** `tests/test_core.py` (18 tests) - pActivity/censoring/replicates, scaffold-split leakage, variance
  decomposition, temperature + conformal coverage, selectivity, EI/UCB/Pareto, reward + beta schedule, batch quotas,
  contract validation, Clopper-Pearson.
* **Written but NOT executed** (build sandbox had no torch/RDKit/network): D-MPNN, trainers, SSL, ADMET, GP, scorer, server,
  REINVENT TOMLs, AiZynthFinder wrapper, data extraction. They compile and follow the manual's equations; expect a debug pass
  on first real run. Start with a 2-epoch / 200-molecule smoke run.

## Where the manual was ambiguous, truncated, or (I think) wrong - decisions made
1. **Reward does not behave as described.** Manual: failing hERG/SA gives "near-zero" reward. With exponents w_j/sum(w) and a 1e-4
   floor, a fully failed safety term only multiplies R by ~0.27. Implemented as written; optional `rl.veto` zeroes R below a
   per-component threshold (off by default). Unit-tested.
2. **Epistemic variance can be underestimated.** Members that all start from one kinome checkpoint theta0 and are L2-SP-anchored
   to it are barely diverse. Default `uq.independent_kinome_training: true` (each member trains its own kinome stage; SSL
   encoder shared). Costs ~10x Stage 3 compute; set false to share theta0.
3. **Truncated spans filled with defaults:** Step 1.3 replicate rule (drop if range > 1.0; configurable), Stage 3.3 Design B
   details, 3.6 ablation table (A0-A7 ladder in `models/ablations.py`, ending at the described production model), 4.2 conformal
   step 2 (split-conformal on normalised residuals), 4.3 AD (5-NN, 95th pct, latent + Morgan), Stage 6, Stage 9 pillars 1-3,
   and the end of the data-contract table.
4. "Phase 3" L2-SP is a regulariser, so it is applied during phases 1-2. Chemprop's D-MPNN shares W_m across steps, so "final
   message-passing layer" = W_m + readout (W_a, attention, norm).
5. In-repo D-MPNN (pure PyTorch) instead of Chemprop v2 so kinase fusion, L2-SP and freezing stay controllable; Chemprop is optional.
6. `<`-relation measurements are dropped (the manual only defines right-censoring). Single-dose % inhibition is not in the main table.
7. REINVENT 4 TOML keys (incl. `learning_strategy.type`, `ExternalProcess` endpoint format) are written from memory of v4.x:
   diff against the example configs of your pinned release before the first run.
8. Statistical power: ~60 TNK1 compounds -> a 20% scaffold test set is ~12 compounds, and 16 exploit / 12 explore / 4 control
   compounds per cycle give wide Clopper-Pearson intervals (e.g. 4/4 -> 40-100%). Pool cycles before claiming differences.
9. ADMET reward mapping for microsomal clearance and solubility/Caco2 sigmoid centres are my heuristics (`rl/scorer.py`).
