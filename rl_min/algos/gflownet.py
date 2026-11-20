"""GFlowNet with the trajectory-balance objective, for SMILES sequence generation.

Bengio, Jain, Korablyov, Precup & Bengio, "Flow Network based Generative Models for
Non-Iterative Diverse Candidate Generation", NeurIPS 2021.
Malkin, Jain, Bengio, Sun & Bengio, "Trajectory balance: Improved credit assignment in
GFlowNets", NeurIPS 2022.

WHY THIS IS THE RIGHT ALGORITHM FOR THIS PROJECT'S FAILURE

A policy-gradient method maximises expected reward, so its optimum is a point mass on the
single best molecule it can find. Every practical fix in the REINVENT lineage -- the
scaffold diversity filter, the replay memory's uniqueness check, sigma tuning -- exists to
fight that pull. A GFlowNet does not have the pull: it is trained so that the terminating
probability of an object is PROPORTIONAL to its reward,

    P_theta(x)  proportional to  R(x)^beta

so high-reward modes are all sampled, in proportion to how good they are. For a project
whose v3 run produced 89 unique molecules from 100 samples but whose selectivity term sat
pinned at the floor and whose candidates all shared the same exploitable failure, sampling
the reward distribution rather than its argmax is the structurally different thing to try.

THE OBJECTIVE, AND WHY IT COLLAPSES TO ONE LINE HERE

Trajectory balance, in general:

    L_TB(tau) = ( log Z_theta + sum_t log P_F(s_{t+1} | s_t)
                              - log R(x) - sum_t log P_B(s_t | s_{t+1}) )^2

For AUTOREGRESSIVE SEQUENCE generation the state graph is a TREE: a partial SMILES string
has exactly one parent (drop the last token), so the backward policy is deterministic,
P_B = 1, and every `log P_B` term is 0. The forward trajectory probability is just the
sequence likelihood. The loss therefore reduces to

    L = ( log Z + log P_theta(x) - beta * log R(x) )^2

with a single learnable scalar `log Z`. `log P_theta(x)` is exactly `-agent.nll(seq)`,
which `rl_min.model.SmilesRNN` already computes and already documents as being SUMMED over
tokens rather than averaged -- which is the form TB requires, so no change is needed there.

That makes this implementation genuinely small, and it is worth stating plainly that the
small size is a property of the tree-structured state space, not a simplification: on a
fragment- or graph-building MDP (where a molecule has many construction orders) P_B is not
1 and the full loss is required.

PRACTICAL NOTES

  * `beta` is the reward exponent / inverse temperature. beta = 1 samples proportional to
    R; larger beta sharpens toward the argmax and recovers policy-gradient behaviour in
    the limit. Sweep it -- it is the knob that trades diversity for potency, and it is a
    far more interpretable knob than DAP's `sigma`, which is confounded with the reward's
    attainable range (see `rl_min/dap.py`).
  * `log Z` needs a much larger learning rate than the policy; the TB paper and every
    subsequent implementation use a separate optimiser group for it. Default here is 1e-1
    against the policy's 1e-4.
  * Rewards must be POSITIVE for `log R` to exist. `r_floor` clamps them, and the floor is
    reported in the history so a run where most of the batch sat on the floor is visible
    rather than silently degenerate.
  * Off-policy training is allowed and helpful: TB is an off-policy objective, so replayed
    high-reward molecules can be mixed into the batch exactly as DAP does.
"""
from __future__ import annotations

import time

import numpy as np
import torch


def train_gflownet(prior, agent, vocab, scorer, cfg, device="cpu", log_every=1, logger=None):
    """Trajectory-balance GFlowNet. Same contract as `train_dap`: -> (history, replay).

    `prior` is used only for the optional KL-anchor term and for logging; set
    `cfg.gfn_prior_weight = 0.0` for pure TB. The anchor is not part of the TB objective
    and is off by default -- it is available because this project's prior encodes 788k
    molecules' worth of syntactic validity, and discarding that at step 1 wastes it.
    """
    from ..dap import ReplayMemory, ScaffoldFilter
    from ..model import encode_batch

    rng = np.random.RandomState(cfg.seed)
    torch.manual_seed(cfg.seed)
    prior.to(device).eval()
    for p in prior.parameters():
        p.requires_grad_(False)
    agent.to(device).train()

    beta = float(getattr(cfg, "gfn_beta", 1.0))
    r_floor = float(getattr(cfg, "gfn_r_floor", 1e-4))
    logz_lr = float(getattr(cfg, "gfn_logz_lr", 1e-1))
    prior_w = float(getattr(cfg, "gfn_prior_weight", 0.0))

    # The learnable partition function. Initialised at 0 rather than at a guess: its scale
    # is learned within a few dozen steps and a wrong initialisation only costs those.
    log_z = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([
        {"params": list(agent.parameters()), "lr": cfg.lr},
        {"params": [log_z], "lr": logz_lr},
    ])

    dfilter = ScaffoldFilter(cfg.df_bucket, cfg.df_min_score)
    replay = ReplayMemory(cfg.replay_size)
    history, seen = [], set()
    t0 = time.time()

    for step in range(1, cfg.steps + 1):
        seqs, _ = agent.sample_with(vocab, cfg.batch_size, cfg.max_len, device, cfg.temperature)
        smis = [vocab.decode(r) for r in seqs]
        res = scorer(smis)
        score = np.asarray(res["score"], dtype=float)
        valid = np.asarray(res.get("valid", [bool(s) for s in smis]), dtype=bool)
        scaffolds = res.get("scaffold", [None] * len(smis))
        score = dfilter(scaffolds, score)

        r_smis, r_scores = replay.sample(cfg.replay_batch, rng)
        if r_smis:
            r_seqs = encode_batch(vocab, r_smis, device)
            n = max(seqs.size(1), r_seqs.size(1))
            pad = lambda t: torch.nn.functional.pad(t, (0, n - t.size(1)), value=vocab.pad)
            batch_seqs = torch.cat([pad(seqs), pad(r_seqs)], 0)
            batch_score = np.concatenate([score, r_scores])
        else:
            batch_seqs, batch_score = seqs, score

        clamped = np.maximum(batch_score, r_floor)
        n_floored = int((batch_score < r_floor).sum())
        log_r = torch.as_tensor(np.log(clamped), dtype=torch.float32, device=device)

        agent.train()
        agent_nll = agent.nll(batch_seqs)                  # = -log P_theta(x), summed
        log_pf = -agent_nll
        # P_B == 1 on a sequence tree, so the backward terms vanish; see module docstring.
        tb = (log_z + log_pf - beta * log_r) ** 2
        loss = tb.mean()
        if prior_w > 0:
            with torch.no_grad():
                prior_nll = prior.nll(batch_seqs)
            # keep the agent near the prior's syntax without changing the TB fixed point
            # more than `prior_w` allows
            loss = loss + prior_w * torch.nn.functional.relu(agent_nll - prior_nll).mean()
        else:
            with torch.no_grad():
                prior_nll = prior.nll(batch_seqs)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip)
        opt.step()

        replay.add([s_ for s_, v in zip(smis, valid) if v],
                   [r for r, v in zip(score, valid) if v])
        fresh = {s_ for s_ in smis if s_} - seen
        seen |= fresh
        m = {"step": step, "loss": float(loss.item()),
             "log_z": float(log_z.detach().cpu().item()),
             "tb_loss": float(tb.mean().detach().cpu().item()),
             "score_mean": float(score.mean()),
             "score_max": float(score.max()) if score.size else 0.0,
             "n_floored": n_floored, "beta": beta,
             "validity": float(valid.mean()),
             "uniqueness": len({s_ for s_ in smis if s_}) / max(len(smis), 1),
             "novel_frac": len(fresh) / max(len(smis), 1),
             "n_scaffolds": len({s_ for s_ in scaffolds if s_}),
             "prior_nll": float(prior_nll.mean().item()),
             "agent_nll": float(agent_nll.mean().item()),
             "replay_best": float(max(replay.scores)) if replay.scores else 0.0,
             "elapsed_s": time.time() - t0}
        history.append(m)
        if logger and step % log_every == 0:
            logger(m)
    return history, replay
