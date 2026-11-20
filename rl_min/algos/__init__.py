"""Policy-optimisation algorithms for the SMILES generator, behind one registry.

Every trainer honours the signature `rl_min.dap.train_dap` already established:

    train(prior, agent, vocab, scorer, cfg, device, log_every, logger)
        -> (history: list[dict], replay: ReplayMemory)

so `rl_min/run_rl.py` selects one with `--algo` and everything downstream -- the history
CSV, `best_molecules.csv`, `meta.json`, the Stage-8 hand-off -- is unchanged. That is the
point: the algorithms become a controlled variable instead of a rewrite.

    dap        the incumbent. REINVENT's Difference between Augmented and Posterior.
    ahc        DAP computed on the top fraction of each batch (Thomas et al. 2022).
    grpo       group-relative policy optimisation, critic-free (Shao et al. 2024).
    rloo       REINFORCE with a leave-one-out baseline (Ahmadian et al. 2024).
    ppo        clipped surrogate with batch reuse (Schulman et al. 2017).
    gflownet   trajectory balance: samples PROPORTIONAL to reward (Malkin et al. 2022).

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
    "ahc": ("rl_min.algos.ahc", "train_ahc"),
    "ppo": ("rl_min.algos.policy_grad", "train_ppo"),
    "grpo": ("rl_min.algos.policy_grad", "train_grpo"),
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
    "ahc": ("topk", "ahc_min_keep"),
    "ppo": ("pg_advantage", "pg_clip", "pg_epochs", "kl_coef", "kl_target"),
    "grpo": ("pg_advantage", "pg_clip", "pg_epochs", "kl_coef", "kl_target"),
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
        "dap": "REINVENT DAP: regress agent onto prior_logP + sigma*score (incumbent)",
        "ahc": "Augmented Hill-Climb: DAP on the top-k fraction only (Thomas et al. 2022)",
        "ppo": "PPO clipped surrogate with batch reuse (Schulman et al. 2017)",
        "grpo": "GRPO: batch-standardised advantage, no critic (Shao et al. 2024)",
        "rloo": "REINFORCE leave-one-out baseline, unbiased, no critic (Ahmadian et al. 2024)",
        "gflownet": "Trajectory balance: P(x) proportional to R(x)^beta (Malkin et al. 2022)",
    }


__all__ = ["common", "ALGOS", "EXTRA_CONFIG_KEYS", "get_trainer", "describe"]
