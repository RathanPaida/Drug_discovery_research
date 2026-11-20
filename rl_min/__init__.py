"""rl_min -- a self-contained policy-gradient generator for SMILES.

Exists because Stage 7 of the main pipeline samples a fixed pool and filters it, which
updates no generator parameters. This package does the other thing: the reward acts on the
generator's weights, so it proposes molecules it was never shown.

    train_prior.py   MLE prior (the KL anchor)
    dap.py           REINVENT's DAP policy-gradient loss
    algos/           the other policy-gradient objectives (AHC, PPO, GRPO, GFlowNet, BAR)
    algos/common.py  their mathematics as pure numpy -- importable with NO torch
    scorer.py        reward interface: StandaloneScorer | PipelineScorer
    run_rl.py        driver + logging

IMPORTS ARE LAZY, DELIBERATELY. Every name below used to be imported eagerly, so
`import rl_min` pulled in `dap` -> `torch`, and ANY access to this package failed outright
in a numpy-only environment:

    >>> from rl_min.algos import common
    ModuleNotFoundError: No module named 'torch'

`rl_min.algos.common` contains nothing but numpy arithmetic -- the selection rules,
advantage definitions, clipping and reward transforms that are the actual content of each
RL variant -- and it is precisely the part that should be unit-testable on a laptop with
no GPU. The module-level `__getattr__` below (PEP 562) keeps the public names working
exactly as before while deferring the torch import until one of them is touched.
"""
from __future__ import annotations

_LAZY = {
    "DAPConfig": ".dap",
    "ReplayMemory": ".dap",
    "ScaffoldFilter": ".dap",
    "train_dap": ".dap",
    "SmilesRNN": ".model",
    "encode_batch": ".model",
    "PipelineScorer": ".scorer",
    "StandaloneScorer": ".scorer",
    "Vocabulary": ".vocab",
    "tokenize": ".vocab",
}

__all__ = list(_LAZY)


def __getattr__(name):
    """Resolve a public name on first use, importing only the module that defines it."""
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(mod, __name__), name)
    globals()[name] = value          # cache, so the lookup happens once
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
