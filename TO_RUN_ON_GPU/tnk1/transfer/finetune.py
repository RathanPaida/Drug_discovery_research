"""Fine-tuning strategies for the pan-kinome trunk, as a registry of named ladders.

WHAT IS ALREADY HERE. `tnk1.models.pipeline.finetune_tnk1` implements one recipe:
phase 1 trains the head alone for 15 epochs; phase 2 unfreezes the last message-passing
block plus the readout with discriminative learning rates; L2-SP anchors both phases to
the pretrained weights. That is a good recipe -- it is close to LP-FT plus partial
unfreezing plus L2-SP -- but it is ONE point in a space, chosen once and never compared,
and the ablation ladder (`tnk1.models.ablations`) varies the pretraining stack rather than
the fine-tuning strategy. A0-A7 cannot tell you whether the three-phase schedule earns its
complexity; the v3 report says exactly that about A7 and calls it Priority 1.

WHAT THIS MODULE ADDS, and the paper behind each:

  lp_ft       **Linear probe, THEN fine-tune.** Kumar, Raghunathan, Jones, Ma & Liang,
              "Fine-Tuning can Distort Pretrained Features and Underperform
              Out-of-Distribution", ICLR 2022. Their finding: full fine-tuning from a
              RANDOM head distorts the pretrained features, because early gradients are
              large and mostly wrong; probing to CONVERGENCE first gives the backbone a
              sensible target and preserves the features. The existing recipe probes for a
              fixed 15 epochs, which is not the same thing as to convergence -- this
              variant probes with early stopping and reports where it stopped.
  surgical    **Fine-tune exactly ONE block.** Lee, Chang, Mishra, Finn, Yu & co.,
              "Surgical Fine-Tuning Improves Adaptation to Distribution Shifts",
              ICLR 2023. Which block to tune depends on the kind of shift: early blocks
              for input-level shift, later blocks for label-level shift. For TNK1 the
              shift from the pan-kinome distribution is mostly label-level (same chemical
              space, different target), which predicts the LAST block -- and that is a
              testable prediction, which is the point of having the variants.
  gradual     **Gradual unfreezing with discriminative learning rates.** Howard & Ruder,
              "Universal Language Model Fine-tuning for Text Classification", ACL 2018.
              Unfreeze one block per epoch group, from the top down, each with a learning
              rate divided by a constant factor per depth.
  swa         **Stochastic Weight Averaging.** Izmailov, Podoprikhin, Garipov, Vetrov &
              Wilson, "Averaging Weights Leads to Wider Optima and Better
              Generalization", UAI 2018. Average the weights along the tail of the
              trajectory instead of taking the last or best point. Nearly free, and it is
              the single most reliable small-data regulariser in the list.
  ema         Exponential moving average of the weights -- SWA's online cousin.
  mixout      **Mixout.** Lee, Cho & Kang, "Mixout: Effective Regularization to Finetune
              Large-scale Pretrained Language Models", ICLR 2020. Stochastically replace
              each weight with its PRETRAINED value rather than with zero, which makes
              the implicit regulariser an L2 pull toward the pretrained point instead of
              toward the origin. A stochastic relative of L2-SP, and it is reported to
              stabilise fine-tuning on very small target sets -- 172 molecules is very
              small.

EVERY STRATEGY IS A WRAPPER, NOT A FORK. Each returns a fitted model through the same
`finetune_tnk1`-compatible signature, so `tnk1/uq/ensemble.py` and the ablation runner can
select one by name without knowing what it does.

MODEL SELECTION IS THE UNSOLVED PART, AND THIS MODULE SAYS SO RATHER THAN HIDING IT. With
172 molecules a validation fold is ~30, so the early-stopping metric has a standard error
large enough that "best epoch" is substantially luck. Every strategy here therefore
reports the epoch it stopped at and the validation curve, and `swa`/`ema` exist precisely
because averaging is less sensitive to that choice than picking a point.
"""
from __future__ import annotations

import copy

import numpy as np


# --------------------------------------------------------------------------- utilities


def _set_grad(params, flag):
    for p in params:
        p.requires_grad_(flag)
    return params


def encoder_blocks(model):
    """Encoder parameter groups, shallow-to-deep, for surgical and gradual strategies.

    Named explicitly rather than derived from `named_parameters()` order: the D-MPNN's
    parameter order is an implementation detail, and a refactor that reorders it would
    silently change which block "the last block" means -- turning a surgical experiment
    into a different one with no error.
    """
    enc = model.enc
    groups = []
    for name in ("W_i", "W_m", "W_a", "attn", "norm"):
        mod = getattr(enc, name, None)
        if mod is not None:
            groups.append((name, list(mod.parameters())))
    return groups


class SWA:
    """Running average of the weights, with the count, as in Izmailov et al. (2018).

    Kept as an object rather than a closure so the caller can log `n` -- an SWA model
    averaged over 2 points is not SWA, and the count is the only way to notice.
    """

    def __init__(self):
        self.state = None
        self.n = 0

    def update(self, model):
        sd = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
        if self.state is None:
            self.state = sd
        else:
            for k in self.state:
                self.state[k] += (sd[k] - self.state[k]) / (self.n + 1)
        self.n += 1
        return self

    def apply_to(self, model):
        if self.state is None or self.n == 0:
            return model
        tgt = model.state_dict()
        model.load_state_dict({k: self.state[k].to(tgt[k].dtype) for k in tgt})
        return model


class EMA:
    """Exponential moving average of the weights. `decay` near 1 averages more history."""

    def __init__(self, decay=0.99):
        self.decay = float(decay)
        self.state = None
        self.n = 0

    def update(self, model):
        sd = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
        if self.state is None:
            self.state = sd
        else:
            d = self.decay
            for k in self.state:
                self.state[k].mul_(d).add_(sd[k], alpha=1 - d)
        self.n += 1
        return self

    def apply_to(self, model):
        if self.state is None:
            return model
        tgt = model.state_dict()
        model.load_state_dict({k: self.state[k].to(tgt[k].dtype) for k in tgt})
        return model


def apply_mixout(model, ref_state, p=0.1):
    """Replace each weight with its PRETRAINED value with probability `p`, in place.

    Mixout (Lee, Cho & Kang, ICLR 2020). Dropout pulls weights toward 0; mixout pulls them
    toward the pretrained point, so the implicit regulariser is an L2 penalty on the
    DEVIATION from pretraining -- the same target L2-SP penalises deterministically. Call
    once per step before the forward pass.

    Implemented as a weight perturbation rather than as a module replacement: the D-MPNN
    is a small custom encoder, and swapping its Linear layers for mixout variants would
    change its state_dict keys and break every existing checkpoint.
    """
    import torch
    if p <= 0:
        return model
    with torch.no_grad():
        for n, prm in model.named_parameters():
            if n not in ref_state or prm.dim() < 2:
                continue            # biases and norms left alone, as in the paper
            mask = torch.rand_like(prm) < p
            prm[mask] = ref_state[n].to(prm.device)[mask]
    return model


# ------------------------------------------------------------------------- strategies


def ft_plain(cfg, model, tr, va, store, desc_store, device="cuda", seed=42, log=None,
             **kw):
    """Unfreeze everything, one phase, no anchor. The A5 rung, and the control."""
    from ..models.pipeline import finetune_tnk1
    return finetune_tnk1(cfg, model, tr, va, store, desc_store, device, seed, log,
                         use_l2sp=False, three_phase=False, **kw)


def ft_l2sp(cfg, model, tr, va, store, desc_store, device="cuda", seed=42, log=None, **kw):
    """Three-phase L2-SP: the production recipe (A7). The incumbent to beat."""
    from ..models.pipeline import finetune_tnk1
    return finetune_tnk1(cfg, model, tr, va, store, desc_store, device, seed, log,
                         use_l2sp=True, three_phase=True, **kw)


def ft_lp_ft(cfg, model, tr, va, store, desc_store, device="cuda", seed=42, log=None,
             probe_epochs=200, probe_patience=20, **kw):
    """Linear probe to CONVERGENCE, then fine-tune. Kumar et al., ICLR 2022.

    The difference from the production recipe is the word "convergence": phase 1 there is
    15 epochs regardless. Here the probe runs with early stopping, so the backbone's first
    gradient sees a head that already fits as well as a linear head can -- which is the
    mechanism the paper credits for preserving pretrained features.
    """
    import torch
    from ..models.pipeline import _mk
    from ..models.trainer import run_epochs
    ft = cfg["finetune"]
    mode = kw.pop("loss_mode", None) or ft["loss"]
    kmap = {u: i for i, u in enumerate(store["ids"])}
    TR = _mk(tr, kmap, desc_store)
    VA = _mk(va, kmap, desc_store) if va is not None and len(va) else None
    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    alphas = {"enc": ft["alpha_enc"], "head": ft["alpha_head"]}

    # --- phase 1: the probe, to convergence -----------------------------------------
    _set_grad(model.parameters(), False)
    _set_grad(model.head_params(), True)
    opt = torch.optim.AdamW(model.head_params(), lr=ft["phase1"]["lr_head"])
    probe_best = run_epochs(model, TR, VA, device, probe_epochs, opt, None, mode,
                            ref_state=ref, alphas=alphas, patience=probe_patience,
                            batch_size=32, seed=seed, log=log)
    if log:
        log({"phase": "probe_converged", "val": probe_best})

    # --- phase 2: full fine-tune from the probed head --------------------------------
    _set_grad(model.parameters(), True)
    p2 = ft["phase2"]
    opt = torch.optim.AdamW([
        {"params": model.encoder_params(), "lr": p2["lr_enc"]},
        {"params": model.kproj_params() + model.head_params(), "lr": p2["lr_head"]},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, p2["epochs"])
    run_epochs(model, TR, VA, device, p2["epochs"], opt, sched, mode, ref_state=ref,
               alphas=alphas, patience=p2["patience"], batch_size=32, seed=seed, log=log)
    return model


def ft_surgical(cfg, model, tr, va, store, desc_store, device="cuda", seed=42, log=None,
                block="last", **kw):
    """Fine-tune ONE encoder block (plus the head). Lee et al., ICLR 2023.

    `block` is "first", "last", a block name from `encoder_blocks`, or "head_only".
    For a label-level shift -- same chemical space, different kinase -- the paper's
    analysis predicts the LATER blocks, which makes "last" the hypothesis and "first" the
    falsifier. Run both.
    """
    import torch
    from ..models.pipeline import _mk
    from ..models.trainer import run_epochs
    ft = cfg["finetune"]
    mode = kw.pop("loss_mode", None) or ft["loss"]
    kmap = {u: i for i, u in enumerate(store["ids"])}
    TR = _mk(tr, kmap, desc_store)
    VA = _mk(va, kmap, desc_store) if va is not None and len(va) else None
    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    alphas = {"enc": ft["alpha_enc"], "head": ft["alpha_head"]}

    blocks = encoder_blocks(model)
    if not blocks and block not in ("head_only",):
        raise RuntimeError("no named encoder blocks found; surgical fine-tuning needs them")
    if block == "head_only":
        chosen, chosen_name = [], "head_only"
    elif block == "first":
        chosen_name, chosen = blocks[0]
    elif block == "last":
        chosen_name, chosen = blocks[-1]
    else:
        match = [(n, ps) for n, ps in blocks if n == block]
        if not match:
            raise ValueError(f"unknown block {block!r}; have "
                             f"{[n for n, _ in blocks]} plus first/last/head_only")
        chosen_name, chosen = match[0]

    _set_grad(model.parameters(), False)
    _set_grad(chosen + model.kproj_params() + model.head_params(), True)
    p2 = ft["phase2"]
    groups = [{"params": model.kproj_params() + model.head_params(), "lr": p2["lr_head"]}]
    if chosen:
        groups.insert(0, {"params": chosen, "lr": p2["lr_enc"]})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, p2["epochs"])
    if log:
        log({"phase": "surgical", "block": chosen_name,
             "n_params": sum(p.numel() for p in chosen)})
    run_epochs(model, TR, VA, device, p2["epochs"], opt, sched, mode, ref_state=ref,
               alphas=alphas, patience=p2["patience"], batch_size=32, seed=seed, log=log)
    _set_grad(model.parameters(), True)
    return model


def ft_gradual(cfg, model, tr, va, store, desc_store, device="cuda", seed=42, log=None,
               lr_decay=2.6, epochs_per_block=10, **kw):
    """Gradual unfreezing, top block down, discriminative LRs. Howard & Ruder, ACL 2018.

    `lr_decay=2.6` is the ULMFiT value: each block deeper gets a learning rate divided by
    2.6. Kept as the published default rather than retuned, so a difference against the
    production recipe is attributable to the schedule and not to a learning-rate search.
    """
    import torch
    from ..models.pipeline import _mk
    from ..models.trainer import run_epochs
    ft = cfg["finetune"]
    mode = kw.pop("loss_mode", None) or ft["loss"]
    kmap = {u: i for i, u in enumerate(store["ids"])}
    TR = _mk(tr, kmap, desc_store)
    VA = _mk(va, kmap, desc_store) if va is not None and len(va) else None
    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    alphas = {"enc": ft["alpha_enc"], "head": ft["alpha_head"]}
    p2 = ft["phase2"]

    blocks = encoder_blocks(model)
    _set_grad(model.parameters(), False)
    live = model.kproj_params() + model.head_params()
    _set_grad(live, True)
    opt = torch.optim.AdamW([{"params": live, "lr": p2["lr_head"]}])
    run_epochs(model, TR, VA, device, epochs_per_block, opt, None, mode, ref_state=ref,
               alphas=alphas, batch_size=32, seed=seed, log=log)

    for depth, (name, params) in enumerate(reversed(blocks), start=1):
        _set_grad(params, True)
        lr = p2["lr_enc"] / (lr_decay ** (depth - 1))
        groups = [{"params": params, "lr": lr},
                  {"params": live, "lr": p2["lr_head"]}]
        opt = torch.optim.AdamW(groups)
        if log:
            log({"phase": "gradual_unfreeze", "block": name, "depth": depth, "lr": lr})
        run_epochs(model, TR, VA, device, epochs_per_block, opt, None, mode,
                   ref_state=ref, alphas=alphas, batch_size=32, seed=seed + depth, log=log)
        live = params + live
    _set_grad(model.parameters(), True)
    return model


def ft_swa(cfg, model, tr, va, store, desc_store, device="cuda", seed=42, log=None,
           swa_start_frac=0.5, mode_="swa", ema_decay=0.99, **kw):
    """The production recipe, with SWA (or EMA) over the tail of phase 2.

    Izmailov et al. (UAI 2018). Averaging the last half of the trajectory costs one extra
    state_dict of memory and no extra compute, and it removes the dependence on "best
    epoch" -- which at a ~30-molecule validation fold is substantially a coin flip.
    """
    import torch
    from ..models.pipeline import _mk
    from ..models.trainer import run_epochs
    ft = cfg["finetune"]
    loss_mode = kw.pop("loss_mode", None) or ft["loss"]
    kmap = {u: i for i, u in enumerate(store["ids"])}
    TR = _mk(tr, kmap, desc_store)
    VA = _mk(va, kmap, desc_store) if va is not None and len(va) else None
    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    alphas = {"enc": ft["alpha_enc"], "head": ft["alpha_head"]}

    # phase 1 exactly as in production, so only the averaging differs
    _set_grad(model.parameters(), False)
    _set_grad(model.head_params(), True)
    opt = torch.optim.AdamW(model.head_params(), lr=ft["phase1"]["lr_head"])
    run_epochs(model, TR, VA, device, ft["phase1"]["epochs"], opt, None, loss_mode,
               ref_state=ref, alphas=alphas, batch_size=32, seed=seed, log=log)

    _set_grad(model.parameters(), False)
    mp_tail = (list(model.enc.W_m.parameters()) + list(model.enc.W_a.parameters())
               + list(model.enc.attn.parameters()) + list(model.enc.norm.parameters()))
    _set_grad(mp_tail + model.head_params(), True)
    p2 = ft["phase2"]
    opt = torch.optim.AdamW([{"params": mp_tail, "lr": p2["lr_enc"]},
                             {"params": model.head_params(), "lr": p2["lr_head"]}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, p2["epochs"])
    avg = SWA() if mode_ == "swa" else EMA(ema_decay)
    start = int(swa_start_frac * p2["epochs"])

    def on_epoch(ep, val):
        if ep >= start:
            avg.update(model)

    run_epochs(model, TR, VA, device, p2["epochs"], opt, sched, loss_mode, ref_state=ref,
               alphas=alphas, patience=None, batch_size=32, seed=seed, log=log,
               on_epoch=on_epoch)
    # patience=None deliberately: early stopping and weight averaging are alternative
    # answers to the same question, and combining them averages over a truncated tail.
    avg.apply_to(model)
    if log:
        log({"phase": mode_, "n_averaged": avg.n, "start_epoch": start})
    _set_grad(model.parameters(), True)
    return model


def ft_mixout(cfg, model, tr, va, store, desc_store, device="cuda", seed=42, log=None,
              mixout_p=0.1, **kw):
    """The production recipe with Mixout applied each epoch. Lee, Cho & Kang, ICLR 2020.

    Applied per EPOCH rather than per step: per-step perturbation of a 939k-parameter
    trunk on 172 molecules was unstable in preliminary shape-checking, and the per-epoch
    form keeps the stochastic-anchor interpretation while leaving the within-epoch
    optimisation smooth. Flagged as a deviation from the paper, which perturbs per forward
    pass.
    """
    import torch
    from ..models.pipeline import _mk
    from ..models.trainer import run_epochs
    ft = cfg["finetune"]
    loss_mode = kw.pop("loss_mode", None) or ft["loss"]
    kmap = {u: i for i, u in enumerate(store["ids"])}
    TR = _mk(tr, kmap, desc_store)
    VA = _mk(va, kmap, desc_store) if va is not None and len(va) else None
    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    alphas = {"enc": ft["alpha_enc"], "head": ft["alpha_head"]}
    p2 = ft["phase2"]
    _set_grad(model.parameters(), True)
    opt = torch.optim.AdamW([
        {"params": model.encoder_params(), "lr": p2["lr_enc"]},
        {"params": model.kproj_params() + model.head_params(), "lr": p2["lr_head"]}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, p2["epochs"])

    def on_epoch(ep, val):
        apply_mixout(model, ref, mixout_p)

    run_epochs(model, TR, VA, device, p2["epochs"], opt, sched, loss_mode, ref_state=ref,
               alphas=alphas, patience=p2["patience"], batch_size=32, seed=seed, log=log,
               on_epoch=on_epoch)
    return model


#: name -> (callable, one-line description). `tnk1/uq/ensemble.py` and the Stage-3b runner
#: select by name, so adding a strategy needs no edit anywhere else.
STRATEGIES = {
    "plain": (ft_plain, "unfreeze everything, one phase, no anchor (A5 control)"),
    "l2sp": (ft_l2sp, "three-phase L2-SP -- the production recipe (A7 incumbent)"),
    "lp_ft": (ft_lp_ft, "linear probe to convergence, then fine-tune (Kumar et al. 2022)"),
    "surgical_last": (lambda *a, **k: ft_surgical(*a, block="last", **k),
                      "fine-tune only the last encoder block (Lee et al. 2023)"),
    "surgical_first": (lambda *a, **k: ft_surgical(*a, block="first", **k),
                       "fine-tune only the first encoder block (the falsifier)"),
    "head_only": (lambda *a, **k: ft_surgical(*a, block="head_only", **k),
                  "linear probe only: the frozen-features floor"),
    "gradual": (ft_gradual, "gradual unfreezing, discriminative LRs (Howard & Ruder 2018)"),
    "swa": (ft_swa, "production recipe + stochastic weight averaging (Izmailov et al. 2018)"),
    "ema": (lambda *a, **k: ft_swa(*a, mode_="ema", **k),
            "production recipe + EMA of the weights"),
    "mixout": (ft_mixout, "stochastic anchor to the pretrained weights (Lee et al. 2020)"),
}


def get_strategy(name: str):
    if name not in STRATEGIES:
        raise ValueError(f"unknown fine-tuning strategy {name!r}; "
                         f"choose from {', '.join(STRATEGIES)}")
    return STRATEGIES[name][0]


def describe() -> dict[str, str]:
    return {k: v[1] for k, v in STRATEGIES.items()}
