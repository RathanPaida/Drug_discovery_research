"""PPO, GRPO and RLOO for SMILES sequence generation. One module: they share a skeleton.

All three replace DAP's regression-onto-a-target with an explicit policy-gradient step on
an ADVANTAGE, and differ only in how the advantage is formed and how the step is
constrained. Writing them together makes the comparison honest -- the sampling, the
scoring, the replay memory, the diversity filter and the logging are literally the same
code, so a difference in the results is a difference in the objective.

    REINFORCE/RLOO   maximise  A_i * log P(x_i),  A from a baseline
    GRPO             maximise  clipped(ratio_i) * A_i,  A standardised WITHIN the batch
    PPO              maximise  clipped(ratio_i) * A_i,  A from a baseline, with epochs
                               of reuse over the same samples

THE CITATIONS

  * PPO -- Schulman, Wolski, Dhariwal, Radford & Klimov, "Proximal Policy Optimization
    Algorithms", arXiv:1707.06347 (2017). The clipped surrogate
        L = E[ min( r_t A_t, clip(r_t, 1-eps, 1+eps) A_t ) ],  r_t = pi_new/pi_old
    lets the same batch be reused for several epochs without the update running away from
    the sampling distribution.
  * GRPO -- Shao, Wang, Zhu, Xu, Song, Bi, Zhang, Zhang, Li, Wu & Guo, "DeepSeekMath:
    Pushing the Limits of Mathematical Reasoning in Open Language Models",
    arXiv:2402.03300 (2024). Drops the value network entirely: the advantage is the
    reward standardised within a GROUP of samples. Here the group is the sampled batch,
    which fits this setting exactly -- there is no prompt, so a batch IS one group, and
    the whole batch shares a single comparison set.
  * RLOO -- Kool, van Hoof & Welling (2019); Ahmadian, Cremer, Gallé, Fadaee, Kreutzer,
    Pietquin, Üstün & Hooker, "Back to Basics: Revisiting REINFORCE-style Optimization
    for Learning from Human Feedback in LLMs", ACL 2024. Leave-one-out baseline, unbiased,
    no critic, no clipping. Their finding -- that this beats PPO for sequence alignment at
    a fraction of the complexity -- transfers to our shape of problem: a sequence policy,
    one scalar terminal reward, an expensive oracle, no per-token credit to assign.

WHY A CRITIC-FREE METHOD IS THE RIGHT DEFAULT HERE. PPO's value network is an extra model
fitted on the same expensive reward evaluations, to predict a reward that arrives only at
the end of the sequence. There is no intermediate reward to bootstrap from, so the critic
is estimating a single scalar per molecule -- which is what the batch mean already
estimates, for free. GRPO and RLOO are the methods that recognise this.

THE KL ANCHOR IS NOT OPTIONAL IN THIS PROJECT. The prior encodes 788k kinase-like
molecules' worth of syntactic and structural plausibility. Policy gradient without an
anchor degrades validity within tens of steps. `kl_coef` applies a penalty on
`logP_prior - logP_agent`, which is the same role DAP's prior term plays implicitly.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .common import approx_kl, center, clip_fraction, leave_one_out_baseline, standardize


def _advantages(score, mode):
    """Dispatch the one line that actually differs between the three algorithms."""
    if mode == "grpo":
        return standardize(score)
    if mode == "rloo":
        return leave_one_out_baseline(score)
    if mode == "mean":
        return center(score)
    if mode == "none":
        return np.asarray(score, float)
    raise ValueError(f"unknown advantage mode {mode!r}")


def train_policy_gradient(prior, agent, vocab, scorer, cfg, device="cpu", log_every=1,
                          logger=None, algo="grpo"):
    """PPO / GRPO / RLOO. Same contract as `train_dap`: -> (history, replay).

    `cfg` fields consumed beyond the DAP ones:
        pg_advantage   "grpo" | "rloo" | "mean" | "none"   (default per `algo`)
        pg_clip        PPO/GRPO clipping epsilon           (default 0.2)
        pg_epochs      inner epochs over each batch        (default 1; PPO uses 4)
        kl_coef        weight on the KL-to-prior penalty   (default 0.02)
        kl_target      if set, adapt kl_coef toward it     (default None)
    """
    from ..dap import ReplayMemory, ScaffoldFilter

    defaults = {"ppo": ("mean", 4), "grpo": ("grpo", 1), "drgrpo": ("mean", 1),
                "rloo": ("rloo", 1)}
    adv_mode, def_epochs = defaults.get(algo, ("grpo", 1))
    adv_mode = getattr(cfg, "pg_advantage", None) or adv_mode
    clip = float(getattr(cfg, "pg_clip", 0.2))
    epochs = int(getattr(cfg, "pg_epochs", def_epochs))
    kl_coef = float(getattr(cfg, "kl_coef", 0.02))
    kl_target = getattr(cfg, "kl_target", None)
    use_clip = algo in ("ppo", "grpo", "drgrpo")

    rng = np.random.RandomState(cfg.seed)
    torch.manual_seed(cfg.seed)
    prior.to(device).eval()
    for p in prior.parameters():
        p.requires_grad_(False)
    agent.to(device).train()

    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
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

        adv = _advantages(score, adv_mode)
        A = torch.as_tensor(adv, dtype=torch.float32, device=device)

        # The sampling policy's log-probabilities, frozen. PPO's ratio is defined against
        # these, so they must be captured BEFORE the first inner epoch -- recomputing them
        # each epoch would make every ratio exactly 1 and silently disable the clipping.
        agent.train()
        with torch.no_grad():
            logp_old = -agent.nll(seqs)
            prior_logp = -prior.nll(seqs)

        last = {}
        for ep in range(max(1, epochs)):
            logp = -agent.nll(seqs)
            if use_clip:
                ratio = torch.exp(logp - logp_old)
                surr = torch.minimum(ratio * A,
                                     torch.clamp(ratio, 1 - clip, 1 + clip) * A)
                pg_loss = -surr.mean()
            else:
                ratio = torch.ones_like(logp)
                pg_loss = -(A * logp).mean()
            # KL(agent || prior) on the sampled sequences, as a penalty rather than a
            # constraint. Non-negative by construction via the k3 estimator's torch form.
            d = prior_logp - logp
            kl = (torch.exp(d) - 1 - d).mean()
            loss = pg_loss + kl_coef * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip)
            opt.step()
            last = {"pg_loss": float(pg_loss.item()), "kl": float(kl.item()),
                    "loss": float(loss.item()),
                    "ratio_mean": float(ratio.mean().item()),
                    "epoch": ep}

        with torch.no_grad():
            logp_new = (-agent.nll(seqs)).cpu().numpy()
        kl_hat = approx_kl(logp_new, logp_old.cpu().numpy())
        cf = clip_fraction(np.exp(logp_new - logp_old.cpu().numpy()), clip) if use_clip else 0.0

        # Adaptive KL coefficient (Schulman et al.'s dual-descent variant). Off unless a
        # target is given: a fixed coefficient is one fewer moving part, and a target is
        # worth setting only once a run has shown the KL drifting.
        if kl_target:
            if kl_hat > 1.5 * kl_target:
                kl_coef = min(kl_coef * 1.5, 10.0)
            elif kl_hat < kl_target / 1.5:
                kl_coef = max(kl_coef / 1.5, 1e-4)

        replay.add([s_ for s_, v in zip(smis, valid) if v],
                   [r for r, v in zip(score, valid) if v])
        fresh = {s_ for s_ in smis if s_} - seen
        seen |= fresh
        m = {"step": step, "algo": algo, "advantage": adv_mode,
             **last,
             "approx_kl": kl_hat, "kl_coef": kl_coef, "clip_frac": cf,
             "score_mean": float(score.mean()),
             "score_max": float(score.max()) if score.size else 0.0,
             "adv_sd": float(np.std(adv)),
             "validity": float(valid.mean()),
             "uniqueness": len({s_ for s_ in smis if s_}) / max(len(smis), 1),
             "novel_frac": len(fresh) / max(len(smis), 1),
             "n_scaffolds": len({s_ for s_ in scaffolds if s_}),
             "prior_nll": float((-prior_logp).mean().item()),
             "agent_nll": float(-logp_new.mean()),
             "replay_best": float(max(replay.scores)) if replay.scores else 0.0,
             "elapsed_s": time.time() - t0}
        history.append(m)
        if logger and step % log_every == 0:
            logger(m)
    return history, replay


def train_ppo(prior, agent, vocab, scorer, cfg, device="cpu", log_every=1, logger=None):
    return train_policy_gradient(prior, agent, vocab, scorer, cfg, device, log_every,
                                 logger, algo="ppo")


def train_grpo(prior, agent, vocab, scorer, cfg, device="cpu", log_every=1, logger=None):
    return train_policy_gradient(prior, agent, vocab, scorer, cfg, device, log_every,
                                 logger, algo="grpo")


def train_drgrpo(prior, agent, vocab, scorer, cfg, device="cpu", log_every=1, logger=None):
    """Dr. GRPO: GRPO with the group-std division and length normalisation removed.

    Liu, Chen, Li, Qi, Pang, Du, Lee & Lin, "Understanding R1-Zero-Like Training: A
    Critical Perspective" (arXiv:2503.20783). The advantage is simply `A_i = R_i - mean(R)`.

    THE VARIANT TO USE ON THIS PROJECT, for a concrete reason. GRPO's
    `A_i = (R_i - mean(R)) / std(R)` divides by the within-batch spread, and a reward
    component pinned at its floor has almost no spread -- r_sel sat at 0.091-0.144 for all
    1000 steps of the v3 run. Dividing by a near-zero std inflates floating-point noise
    into large advantages and the policy chases it. Removing the division removes that
    failure mode at no cost: the mean baseline still reduces variance.
    """
    return train_policy_gradient(prior, agent, vocab, scorer, cfg, device, log_every,
                                 logger, algo="drgrpo")


def train_rloo(prior, agent, vocab, scorer, cfg, device="cpu", log_every=1, logger=None):
    return train_policy_gradient(prior, agent, vocab, scorer, cfg, device, log_every,
                                 logger, algo="rloo")
