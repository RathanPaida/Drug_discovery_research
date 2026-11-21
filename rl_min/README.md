# `rl_min` — in-process policy-gradient generator

A self-contained DAP (REINVENT) trainer for SMILES. It exists because **Stage 7 of the
shipped pipeline performed selection, not reinforcement learning**: it sampled 12,000
molecules from a prior, scored them once, and kept the best. No parameter of any generator
was ever updated, so the stage could not propose anything the prior was already unlikely to
emit. `runs/rl/` contained only `rl.toml` — no checkpoint, no step log, no TensorBoard
directory — and `stage7_scored_candidates.csv` has no `step` column.

## The objective

```
augmented_logP = log P_prior(x) + sigma * R(x)
loss           = mean( (augmented_logP - log P_agent(x))^2 )
```

The prior is frozen and acts as an implicit KL anchor. `sigma * R(x)` displaces the target
toward high-reward molecules, and the agent regresses onto it. **Weights move.**

Because `sigma` and `R` multiply, the reward's *attainable range* is part of the
hyperparameter. Calibrate `R` to span [0,1] before sweeping `sigma` or the two are
confounded — the shipped reward topped out at 0.399, so a nominal `sigma=120` behaved
like ≈48.

## Files

| | |
|---|---|
| `vocab.py` | multi-character SMILES tokeniser (`Cl`, `[nH]`, `%10` stay single tokens) |
| `model.py` | `SmilesRNN` — the policy; whole-sequence NLL, EOS-masked sampling |
| `train_prior.py` | MLE prior training; `EncodedCorpus` for the 788k store |
| `dap.py` | the DAP loop **and the shared infrastructure**: `ReplayMemory`, `ScaffoldFilter`, `DAPConfig`. Every algorithm in `algos/` imports these rather than reimplementing them, so a fix to the replay buffer or the diversity filter reaches all eight |
| `algos/` | the seven alternative objectives + the dispatch table (below) |
| `algos/common.py` | objective-agnostic pieces: advantage transforms, PPO clipping, `approx_kl`, pessimistic reward, `weighted_geometric_mean`, Lagrangian multipliers, `intdiv`/`n_circles` diversity metrics |
| `scorer.py` | `StandaloneScorer` (no feature store) / `PipelineScorer` (real surrogate) |
| `run_rl.py` | driver; always measures the prior baseline first |

## The eight objectives

`--algo` selects one; all share the prior, the scorer, the replay buffer and the diversity
filter, so a comparison between them is a comparison of objectives and nothing else.

| `--algo` | what it optimises | why it is here |
|---|---|---|
| `dap` | `(logP_prior + sigma*R - logP_agent)^2` | **the default.** 1st of 25 on the PMO benchmark (Gao et al. 2022) |
| `augmem` | DAP + whole-buffer replay x A, SMILES-randomised | Augmented Memory (Guo & Schwaller); best reported sample efficiency |
| `ahc` | DAP on the top-k fraction of the batch only | Augmented Hill-Climb (Thomas et al. 2022) |
| `ppo` | clipped surrogate with batch reuse | Schulman et al. 2017; the baseline everyone knows |
| `grpo` | batch-**standardised** advantage, no critic | Shao et al. 2024 |
| `drgrpo` | centred advantage, no std division, no length norm | Dr. GRPO (Liu et al. 2025): removes two biases in GRPO |
| `rloo` | REINFORCE with a leave-one-out baseline | Ahmadian et al. 2024; unbiased, no critic |
| `gflownet` | trajectory balance, `P(x) ~ R(x)^beta` | Malkin et al. 2022. **Ranks 17th of 25 on PMO** — included to be falsifiable, not because it is expected to win |

```bash
python -m rl_min.run_rl --list-algos          # the table above, with each one's extra config keys
```

Per-algorithm hyperparameters live under `rl.<algo>` in the config and are listed by
`--list-algos`. `run_rl.py` **refuses to start** if it finds an `rl.<algo>` key it does not
read, so a typo in a sweep config fails immediately rather than silently using the default.

## Usage

```bash
# 1. train the prior on every molecule in the store (~2-3 h on an RTX 4060)
python -m rl_min.train_prior \
  --corpus ../../ml_research/features/unique_molecules_788k.parquet \
  --smiles-col canonical_smiles --out data/processed/rl_min \
  --epochs 40 --batch 512 --hidden 512 --layers 3

# 2. run RL against a reward that needs no feature store (validates the machinery)
python -m rl_min.run_rl \
  --prior data/processed/rl_min/prior.pt \
  --vocab data/processed/rl_min/vocab.txt \
  --out runs/rl_min --steps 300 --sigma 120
```

For the real surrogate reward, pass `--scorer pipeline`: `run_rl.py` then builds the
`Ensemble` / `Scorer` stack itself and wraps it in `PipelineScorer`. This is what the
pipeline's `rl` stage does — `scripts/run_pipeline.py` invokes `python -m rl_min.run_rl`
directly, so there is no notebook and no separate REINVENT process in the loop.

```bash
# a non-default objective, against the real surrogate
python -m rl_min.run_rl --scorer pipeline --algo augmem --version v4

# through the pipeline (writes runs/rl_<algo>/ for anything but dap)
python scripts/run_pipeline.py --only rl --rl-algo grpo --version v4

# all eight on the HPC, 2 concurrent GPU tasks
bash hpc/run_sweeps.sh rl
```

## The number that matters

`run_rl.py` scores 2,000 prior samples **before** training starts. Without that baseline you
cannot distinguish "the policy improved" from "we drew more samples" — which is exactly why
the v1 scorecard cannot support its RL claim.

Measured on a 12,000-molecule corpus with a 1.47M-parameter prior (RTX 3050, 240 steps):

```
prior baseline reward     0.069
agent, last 10 steps      0.544      7.8x
validity        0.44  ->  0.91
uniqueness              ~0.98        (no mode collapse)
scaffolds/batch   27  ->  48
```

Validity *rising* under reward pressure is the signature of a policy that generalised,
rather than one that memorised a single high-scoring scaffold. On the full 788k corpus
expect a stronger prior (85–95% validity) and correspondingly better agent output.

## Debugging a run that does not learn

In this order:

1. **Reward range.** If `R` never exceeds ~0.4, your effective `sigma` is a third of what
   you set. Use `tnk1.rl.reward.calibrate_centre` on a fresh sample from the prior.
2. **`sigma`.** Too low and the prior dominates; too high and the agent collapses onto one
   scaffold. Watch `n_scaffolds` per batch — a fall to 1–2 is mode collapse.
3. **The critic.** If your surrogate ranks worse than a Stage 2 baseline (it did: Spearman
   0.264 vs Random Forest's 0.626), the policy is climbing noise. Fix the surrogate first.
4. **Prior validity.** Below ~80%, most samples score 0 and the gradient sees only
   failures. Experience replay mitigates this but cannot substitute for a usable prior.

## Tests

`tests/test_rl_min.py` (19) covers the tokeniser, the model, DAP and the scorers;
`tests/test_rl_algos.py` (33) covers `algos/` and the shared transforms in `common.py`.

The decisive one is `test_one_dap_step_changes_the_agent_but_not_the_prior` — if it fails,
the stage has regressed to selection, which is the exact defect `rl_min` exists to fix.

Both files run without torch installed: `rl_min/__init__.py` uses a lazy PEP 562
`__getattr__`, so `rl_min.algos.common` and its transforms are importable and testable on a
machine with no GPU stack. Tests needing torch skip themselves.
