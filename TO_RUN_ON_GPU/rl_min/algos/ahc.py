"""Augmented Hill-Climb. Thomas, O'Boyle, Bender & de Graaf, J Cheminform 14:68 (2022).

"Augmented Hill-Climb increases reinforcement learning efficiency for language-based de
novo molecule generation".

THE ALGORITHM, in one sentence: it is REINVENT's DAP loss computed on the TOP FRACTION of
each sampled batch instead of all of it.

    augmented_logP = prior_logP + sigma * score            (as in DAP)
    keep           = top `topk` fraction of the batch by score
    loss           = mean over `keep` of (augmented_logP - agent_logP)^2

Why that one change matters. DAP regresses the agent onto the augmented likelihood for
EVERY sampled molecule, including the ones that scored zero. For those, the target is just
the prior's likelihood, so the update actively pulls the agent back toward the prior -- it
spends gradient on reproducing chemistry it already knows. With a sparse reward most of a
batch is exactly that. AHC drops those samples, so every gradient step is spent on the
molecules that did well.

Why it matters HERE in particular. The v3 run's reward is expensive: each molecule costs a
D-MPNN ensemble forward pass over a 9-kinase panel plus seven ADMET endpoint models. 1000
DAP steps at batch 128 took 165 GPU-minutes. Sample efficiency is the binding constraint,
and this is the published variant that addresses it directly.

Relationship to hill-climbing (Neil et al. 2018) and to the `sigma` confound: classical
hill-climb fine-tunes on the top-k by MAXIMUM LIKELIHOOD, with no prior anchor, and
collapses onto one scaffold. AHC keeps the prior term, so the anchor survives the
selection -- which is exactly what the project's scaffold-diversity filter has been
compensating for by hand.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .common import topk_mask


def train_ahc(prior, agent, vocab, scorer, cfg, device="cpu", log_every=1, logger=None):
    """Same contract as `rl_min.dap.train_dap`: -> (history, replay).

    `cfg` is a `DAPConfig` carrying the extra fields `topk` (kept fraction, default 0.5),
    `ahc_min_keep` and `ahc_likelihood_penalty`. Everything else -- the replay memory, the
    scaffold filter -- behaves as in DAP so the two are comparable on one changed variable.

    ON SIGMA. The reference implementation drops sigma from REINVENT's 128 to 60 for AHC,
    and the reason is structural rather than empirical: discarding the bottom half of the
    batch already concentrates the update, so a smaller score multiplier suffices. This
    trainer inherits whatever `rl.sigma` the config sets, so **set `rl.sigma` to about half
    its DAP value when running `--algo ahc`**, or the effective step size is doubled
    relative to the published configuration and the comparison is confounded.
    """
    from ..dap import ReplayMemory, ScaffoldFilter
    from ..model import encode_batch

    rng = np.random.RandomState(cfg.seed)
    torch.manual_seed(cfg.seed)
    prior.to(device).eval()
    for p in prior.parameters():
        p.requires_grad_(False)
    agent.to(device).train()

    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
    dfilter = ScaffoldFilter(cfg.df_bucket, cfg.df_min_score)
    replay = ReplayMemory(cfg.replay_size)
    topk = float(getattr(cfg, "topk", 0.5))
    min_keep = int(getattr(cfg, "ahc_min_keep", 1))
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

        # THE ONE DIFFERENCE FROM DAP: top-k selection on the SAMPLED batch only.
        #
        # The replay loss is appended UNFILTERED afterwards, which is what the reference
        # implementation does and is not an accident: the replay buffer already holds the
        # best molecules found so far, so top-k-ing it a second time would narrow the
        # update to the best of the best and throw away the gradient signal replay exists
        # to provide. An earlier draft of this file filtered the concatenated batch and so
        # did exactly that.
        keep = topk_mask(score, topk, min_keep)
        idx = torch.as_tensor(np.where(keep)[0], dtype=torch.long, device=device)

        agent.train()
        agent_nll = agent.nll(seqs[idx])
        with torch.no_grad():
            prior_nll = prior.nll(seqs[idx])
        s = torch.as_tensor(score[keep], dtype=torch.float32, device=device)
        per_mol = ((-prior_nll + cfg.sigma * s) - (-agent_nll)) ** 2

        r_smis, r_scores = replay.sample(cfg.replay_batch, rng)
        if r_smis:
            r_seqs = encode_batch(vocab, r_smis, device)
            r_agent_nll = agent.nll(r_seqs)
            with torch.no_grad():
                r_prior_nll = prior.nll(r_seqs)
            rs = torch.as_tensor(r_scores, dtype=torch.float32, device=device)
            r_per_mol = ((-r_prior_nll + cfg.sigma * rs) - (-r_agent_nll)) ** 2
            per_mol = torch.cat([per_mol, r_per_mol], 0)
            agent_nll_all = torch.cat([agent_nll, r_agent_nll], 0)
        else:
            agent_nll_all = agent_nll

        loss = per_mol.mean()
        # The reference implementation adds a high-likelihood penalty, `-mean(1/agent_nll)`
        # at weight 5e3, which discourages the agent from driving any sequence's likelihood
        # arbitrarily high -- the mechanism behind hill-climbing's usual mode collapse.
        # Off by default here (`ahc_likelihood_penalty: 0.0`) so `ahc` differs from `dap`
        # in ONE variable and the comparison stays attributable; set it to 5e3 to reproduce
        # the published configuration.
        lp_w = float(getattr(cfg, "ahc_likelihood_penalty", 0.0))
        if lp_w:
            loss = loss + lp_w * (-(1.0 / agent_nll_all.clamp_min(1e-6)).mean())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip)
        opt.step()

        replay.add([s_ for s_, v in zip(smis, valid) if v],
                   [r for r, v in zip(score, valid) if v])
        fresh = {s_ for s_ in smis if s_} - seen
        seen |= fresh
        m = {"step": step, "loss": float(loss.item()),
             "score_mean": float(score.mean()),
             "score_max": float(score.max()) if score.size else 0.0,
             "score_kept_mean": float(score[keep].mean()),
             "n_kept": int(keep.sum()), "n_batch": int(len(score)),
             "n_replayed": len(r_smis),
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
