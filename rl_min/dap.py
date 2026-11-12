"""DAP (Difference between Augmented and Posterior) policy gradient -- REINVENT's RL loss.

    augmented_logP = prior_logP + sigma * score
    loss           = mean( (augmented_logP - agent_logP)^2 )

Reading the objective: the prior pins the agent to chemistry it already believes is
syntactically and structurally plausible, and `sigma * score` displaces that target toward
high-reward molecules. The squared term regresses the agent onto the displaced target. The
prior therefore acts as an implicit KL anchor -- which is why `sigma` is the real
exploration knob, and why a reward that only ever reaches 0.4 makes a nominal sigma=120
behave like ~48.

`sigma` and `score` multiply, so the reward's ATTAINABLE RANGE is part of the
hyperparameter. Calibrate the reward to span [0,1] (see tnk1.rl.reward.calibrate_centre)
before tuning sigma, or the two are confounded and no sigma sweep will be interpretable.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from .model import SmilesRNN, encode_batch
from .vocab import Vocabulary


@dataclass
class DAPConfig:
    steps: int = 300
    batch_size: int = 64
    sigma: float = 120.0
    lr: float = 1e-4
    max_len: int = 128
    temperature: float = 1.0
    grad_clip: float = 5.0
    # diversity filter: once `bucket` molecules scoring >= `min_score` share a Murcko
    # scaffold, further members of that scaffold score 0. Without it the agent reliably
    # collapses onto one scaffold -- the single most common failure mode of this loss.
    df_bucket: int = 25
    df_min_score: float = 0.4
    # experience replay: re-show the best molecules found so far. Rewards here are sparse
    # (most samples score near 0), so without replay the gradient is dominated by failures.
    replay_size: int = 100
    replay_batch: int = 16
    seed: int = 42


@dataclass
class ReplayMemory:
    size: int = 100
    smiles: list = field(default_factory=list)
    scores: list = field(default_factory=list)

    def add(self, smis, scores):
        for s, r in zip(smis, scores):
            if s and s not in self.smiles:
                self.smiles.append(s)
                self.scores.append(float(r))
        if len(self.smiles) > self.size:
            order = np.argsort(self.scores)[::-1][: self.size]
            self.smiles = [self.smiles[i] for i in order]
            self.scores = [self.scores[i] for i in order]

    def sample(self, n, rng):
        if not self.smiles:
            return [], np.array([])
        n = min(n, len(self.smiles))
        idx = rng.choice(len(self.smiles), size=n, replace=False)
        return [self.smiles[i] for i in idx], np.array([self.scores[i] for i in idx])


class ScaffoldFilter:
    def __init__(self, bucket=25, min_score=0.4):
        self.bucket, self.min_score, self.counts = bucket, min_score, defaultdict(int)

    def __call__(self, scaffolds, scores):
        out = np.asarray(scores, dtype=float).copy()
        for i, (sc, r) in enumerate(zip(scaffolds, scores)):
            if sc is None:
                continue
            if self.counts[sc] >= self.bucket:
                out[i] = 0.0
            elif r >= self.min_score:
                self.counts[sc] += 1
        return out


def train_dap(prior: SmilesRNN, agent: SmilesRNN, vocab: Vocabulary, scorer,
              cfg: DAPConfig, device="cpu", log_every=1, logger=None):
    """`scorer(smiles_list) -> dict` with keys 'score' (array) and optionally
    'valid', 'scaffold', plus any extra per-component arrays to log.

    Returns a list of per-step metric dicts."""
    rng = np.random.RandomState(cfg.seed)
    torch.manual_seed(cfg.seed)

    prior.to(device).eval()
    for p in prior.parameters():          # the anchor must never move
        p.requires_grad_(False)
    agent.to(device).train()

    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
    dfilter = ScaffoldFilter(cfg.df_bucket, cfg.df_min_score)
    replay = ReplayMemory(cfg.replay_size)
    history, seen = [], set()
    t0 = time.time()

    for step in range(1, cfg.steps + 1):
        seqs, _ = agent.sample_with(vocab, cfg.batch_size, cfg.max_len, device, cfg.temperature)
        smis = [vocab.decode(row) for row in seqs]

        res = scorer(smis)
        score = np.asarray(res["score"], dtype=float)
        valid = np.asarray(res.get("valid", [bool(s) for s in smis]), dtype=bool)
        scaffolds = res.get("scaffold", [None] * len(smis))

        score = dfilter(scaffolds, score)

        # splice replayed high-scorers into the batch so the gradient sees successes
        r_smis, r_scores = replay.sample(cfg.replay_batch, rng)
        if r_smis:
            r_seqs = encode_batch(vocab, r_smis, device)
            n = max(seqs.size(1), r_seqs.size(1))
            pad = lambda t: torch.nn.functional.pad(t, (0, n - t.size(1)), value=vocab.pad)
            batch_seqs = torch.cat([pad(seqs), pad(r_seqs)], 0)
            batch_score = np.concatenate([score, r_scores])
        else:
            batch_seqs, batch_score = seqs, score

        # sample_with() put the agent in eval() to sample without dropout. cuDNN refuses to
        # backprop through an RNN in eval mode, and leaving it there would also drop the
        # regularisation out of the update, so switch back before building the loss.
        agent.train()
        agent_nll = agent.nll(batch_seqs)
        with torch.no_grad():
            prior_nll = prior.nll(batch_seqs)
        s = torch.as_tensor(batch_score, dtype=torch.float32, device=device)

        augmented = -prior_nll + cfg.sigma * s        # prior_logP + sigma * score
        loss = ((augmented - (-agent_nll)) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip)
        opt.step()

        replay.add([s_ for s_, v in zip(smis, valid) if v],
                   [r for r, v in zip(score, valid) if v])
        fresh = {s_ for s_ in smis if s_} - seen
        seen |= fresh

        m = {
            "step": step,
            "loss": float(loss.item()),
            "score_mean": float(score.mean()),
            "score_max": float(score.max()) if score.size else 0.0,
            "validity": float(valid.mean()),
            "uniqueness": len(set(s_ for s_ in smis if s_)) / max(len(smis), 1),
            "novel_frac": len(fresh) / max(len(smis), 1),
            "n_scaffolds": len({s_ for s_ in scaffolds if s_}),
            "prior_nll": float(prior_nll.mean().item()),
            "agent_nll": float(agent_nll.mean().item()),
            "replay_best": float(max(replay.scores)) if replay.scores else 0.0,
            "elapsed_s": time.time() - t0,
        }
        history.append(m)
        if logger and step % log_every == 0:
            logger(m)
    return history, replay
