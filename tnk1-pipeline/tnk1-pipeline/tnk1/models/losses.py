"""Losses: masked/weighted Huber, right-censored hinge, heteroscedastic NLL, NIG evidential, L2-SP."""
import torch
import torch.nn.functional as F


def task_loss(mu, logvar, y, cens, w=None, mode="huber", tau=None, delta=1.0, cens_weight=0.5):
    """Exact rows (cens==0): Huber (mode='huber') or Gaussian NLL (mode='nll').
    Censored rows (cens==1, e.g. IC50>10uM => true pAct < bound): penalise only max(0, mu - bound)^2.
    `y` for censored rows already holds that row's bound. Per-sample weights `w` (similarity,
    majority cap, in-house x3.0) multiply each term.

    `tau` is an OVERRIDE, not a default: it replaces every censored row's own bound with the single
    constant `tau`. Leave it None (as `trainer.py` does) whenever the rows carry real per-row bounds
    -- a ">1 uM" record bounds pActivity at 6 and a ">100 uM" record at 4, and collapsing both to
    5.0 both over- and under-penalises. Pass it only for a dataset with one global threshold.
    """
    w = torch.ones_like(mu) if w is None else w
    ex, ce = cens == 0, cens == 1
    # `mu.sum() * 0`, not `mu.new_zeros(())`: the latter is detached from the autograd graph,
    # so a batch contributing neither an exact nor a censored term returned a loss with
    # requires_grad=False and `loss.backward()` raised "does not require grad". This keeps the
    # value at exactly zero while staying connected to mu.
    loss = mu.sum() * 0
    if ex.any():
        if mode == "nll":
            per = 0.5 * torch.exp(-logvar[ex]) * (y[ex] - mu[ex]) ** 2 + 0.5 * logvar[ex]
        else:
            per = F.huber_loss(mu[ex], y[ex], delta=delta, reduction="none")
        loss = loss + (w[ex] * per).sum() / w[ex].sum().clamp_min(1e-9)
    if ce.any():
        bound = y[ce] if tau is None else torch.full_like(y[ce], tau)
        per = F.relu(mu[ce] - bound) ** 2
        loss = loss + cens_weight * (w[ce] * per).sum() / w[ce].sum().clamp_min(1e-9)
    return loss


def evidential_loss(raw, y, lam=1e-2, eps=1e-2):
    """Amini et al. 2020 NIG NLL + evidence regulariser."""
    from .model import evidential_outputs
    g, nu, a, b = evidential_outputs(raw)
    om = 2 * b * (1 + nu)
    nll = (0.5 * torch.log(torch.pi / nu) - a * torch.log(om) + (a + 0.5) * torch.log(nu * (y - g) ** 2 + om)
           + torch.lgamma(a) - torch.lgamma(a + 0.5))
    reg = torch.abs(y - g) * (2 * nu + a)
    return (nll + lam * (reg - eps)).mean()


def l2sp(model, ref_state, alpha_enc, alpha_head):
    """alpha_enc*||theta_enc - theta0_enc||^2 + alpha_head*||theta_head - theta0_head||^2 (Sec 3.5);
    kproj is anchored with the encoder coefficient."""
    tot = 0.0
    for n, p in model.named_parameters():
        if not p.requires_grad or n not in ref_state:
            continue
        a = alpha_head if n.startswith("head") else alpha_enc
        tot = tot + a * torch.sum((p - ref_state[n].to(p.device)) ** 2)
    return tot


def compute_finetune_loss(mu, logvar, y, cens, w, model, ref_state, cfg_ft, mode="nll", tau=None):
    """`tau` now defaults to None, matching `trainer.py`: censored rows keep their own bounds.
    It previously defaulted to 5.0, which silently discarded every per-row censor bound."""
    return task_loss(mu, logvar, y, cens, w, mode, tau=tau, cens_weight=0.5) + l2sp(
        model, ref_state, cfg_ft["alpha_enc"], cfg_ft["alpha_head"])
