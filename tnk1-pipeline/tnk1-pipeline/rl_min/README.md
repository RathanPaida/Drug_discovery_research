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
| `dap.py` | the policy-gradient loop, diversity filter, experience replay |
| `scorer.py` | `StandaloneScorer` (no feature store) / `PipelineScorer` (real surrogate) |
| `run_rl.py` | driver; always measures the prior baseline first |

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

For the real surrogate reward, use the v2 notebook (Stage 7), which builds the `Ensemble` /
`Scorer` stack and wraps it in `PipelineScorer`.

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

`tests/test_rl_min.py` (26 tests). The decisive one is
`test_one_dap_step_changes_the_agent_but_not_the_prior` — if it fails, the stage has
regressed to selection.
