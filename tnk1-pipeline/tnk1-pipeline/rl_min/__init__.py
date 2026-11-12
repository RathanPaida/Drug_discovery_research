"""rl_min -- a self-contained policy-gradient generator for SMILES.

Exists because Stage 7 of the main pipeline samples a fixed pool and filters it, which
updates no generator parameters. This package does the other thing: the reward acts on the
generator's weights, so it proposes molecules it was never shown.

    train_prior.py   MLE prior (the KL anchor)
    dap.py           REINVENT's DAP policy-gradient loss
    scorer.py        reward interface: StandaloneScorer | PipelineScorer
    run_rl.py        driver + logging
"""
from .dap import DAPConfig, ReplayMemory, ScaffoldFilter, train_dap
from .model import SmilesRNN, encode_batch
from .scorer import PipelineScorer, StandaloneScorer
from .vocab import Vocabulary, tokenize

__all__ = ["DAPConfig", "ReplayMemory", "ScaffoldFilter", "train_dap", "SmilesRNN",
           "encode_batch", "PipelineScorer", "StandaloneScorer", "Vocabulary", "tokenize"]
