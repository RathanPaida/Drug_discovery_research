"""Policy-optimisation algorithms for the SMILES generator, behind one registry.

Every trainer honours the signature `rl_min.dap.train_dap` already established:

    train(prior, agent, vocab, scorer, cfg, device, log_every, logger)
        -> (history: list[dict], replay: ReplayMemory)

so `rl_min/run_rl.py` selects one with `--algo` and everything downstream -- the history
CSV, `best_molecules.csv`, `meta.json`, the Stage-8 hand-off -- is unchanged. That is the
point: the algorithms become a controlled variable instead of a rewrite.

    dap        the incumbent. REINVENT's Difference between Augmented and Posterior.
    augmem     Augmented Memory: whole-buffer replay with SMILES randomisation.
    ahc        DAP computed on the top fraction of each batch (Thomas et al. 2022).
    grpo       group-relative policy optimisation, critic-free (Shao et al. 2024).
    drgrpo     GRPO without the std division -- the variant to use here (Liu et al. 2025).
    rloo       REINFORCE with a leave-one-out baseline (Ahmadian et al. 2024).
    ppo        clipped surrogate with batch reuse (Schulman et al. 2017).
    gflownet   trajectory balance: samples PROPORTIONAL to reward (Malkin et al. 2022).

RANKING, FROM THE PRACTICAL MOLECULAR OPTIMISATION BENCHMARK (Gao, Fu, Sun & Coley,
NeurIPS 2022), because the obvious intuitions are wrong here:

    Augmented Memory   summed AUC Top-10 15.002, winning 19 of 23 tasks
    REINVENT (= dap)   14.016 -- and 1st of 25 methods on all six ranking metrics
    GFlowNet           ranks 17th of 25; GFlowNet-AL 21st

So the backbone this project already has is the reigning champion, and `augmem` is the
upgrade with the largest measured margin. **GFlowNet is kept, and is genuinely the right
SHAPE of answer to mode collapse, but it is a second-phase experiment, not a first move:**
the published molecular GFlowNet results are fragment/graph GFlowNets, not SMILES-sequence
ones, at this oracle budget they lose decisively to policy-gradient SMILES methods, and
trajectory balance adds a learned partition function with its own tuning surface. Fix the
reward first.

`common.py` holds their mathematics as pure numpy and imports no torch, so the selection
rules, advantage definitions, clipping and reward transforms are unit-testable without a
GPU. Importing THIS module imports the torch trainers, so a caller that only wants the
mathematics should import `rl_min.algos.common` directly.
"""
from __future__ import annotations

from . import common

#: name -> fully qualified trainer, resolved lazily so `import rl_min.algos` stays cheap
#: and so a missing torch does not break `rl_min.algos.common`.
_TRAINERS = {
    "dap": ("rl_min.dap", "train_dap"),
    "augmem": ("rl_min.algos.augmented_memory", "train_augmented_memory"),
    "ahc": ("rl_min.algos.ahc", "train_ahc"),
    "ppo": ("rl_min.algos.policy_grad", "train_ppo"),
    "grpo": ("rl_min.algos.policy_grad", "train_grpo"),
    "drgrpo": ("rl_min.algos.policy_grad", "train_drgrpo"),
    "rloo": ("rl_min.algos.policy_grad", "train_rloo"),
    "gflownet": ("rl_min.algos.gflownet", "train_gflownet"),
}

ALGOS = tuple(_TRAINERS)

#: Extra `DAPConfig` fields each algorithm reads, so `run_rl.py` can validate a config
#: instead of silently ignoring a key -- the defect class this repository has already hit
#: three times (`rl.steps` as an inert argparse literal, 8 unread config keys, and the
#: `adme`/`qed` components computed and discarded).
EXTRA_CONFIG_KEYS = {
    "dap": (),
    "augmem": ("am_rounds", "am_randomise", "am_purge"),
    "ahc": ("topk", "ahc_min_keep", "ahc_likelihood_penalty"),
    "ppo": ("pg_advantage", "pg_clip", "pg_epochs", "kl_coef", "kl_target"),
    "grpo": ("pg_advantage", "pg_clip", "pg_epochs", "kl_coef", "kl_target"),
    "drgrpo": ("pg_advantage", "pg_clip", "pg_epochs", "kl_coef", "kl_target"),
    "rloo": ("pg_advantage", "kl_coef", "kl_target"),
    "gflownet": ("gfn_beta", "gfn_r_floor", "gfn_logz_lr", "gfn_prior_weight"),
}


def get_trainer(name: str):
    """-> the trainer callable for `name`. Raises with the valid list, not a KeyError."""
    if name not in _TRAINERS:
        raise ValueError(f"unknown RL algorithm {name!r}; choose from {', '.join(ALGOS)}")
    mod, fn = _TRAINERS[name]
    from importlib import import_module
    return getattr(import_module(mod), fn)


def describe() -> dict:
    """One line per algorithm, for `--list-algos` and for the run's `meta.json`."""
    return {
        "dap": "REINVENT DAP: regress agent onto prior_logP + sigma*score. 1st of 25 on PMO",
        "augmem": "Augmented Memory: whole-buffer replay x A, SMILES-randomised. "
                  "PMO 15.002 vs REINVENT's 14.016, 19/23 tasks -- the best measured margin",
        "ahc": "Augmented Hill-Climb: DAP on the top-k fraction only (Thomas et al. 2022). "
               "Halve rl.sigma when using it",
        "ppo": "PPO clipped surrogate with batch reuse (Schulman et al. 2017)",
        "grpo": "GRPO: batch-STANDARDISED advantage, no critic (Shao et al. 2024). The std "
                "division misfires on a floored component -- prefer drgrpo",
        "drgrpo": "Dr. GRPO: centred advantage, no std division, no length norm "
                  "(Liu et al. 2025). The GRPO variant to use here",
        "rloo": "REINFORCE leave-one-out baseline, unbiased, no critic (Ahmadian et al. 2024)",
        "gflownet": "Trajectory balance: P(x) proportional to R(x)^beta (Malkin et al. 2022). "
                    "Ranks 17th of 25 on PMO -- second-phase experiment, not a first move",
    }


__all__ = ["common", "ALGOS", "EXTRA_CONFIG_KEYS", "get_trainer", "describe"]
