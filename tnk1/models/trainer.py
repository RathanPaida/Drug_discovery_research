"""Generic training loop + inference helpers."""
from __future__ import annotations
import copy
import numpy as np
import torch
from .losses import task_loss, l2sp, evidential_loss
from .data import loader, LabelledSet


@torch.no_grad()
def predict(model, ds: LabelledSet, device, batch_size=512, mc_dropout=False):
    """Returns (mu, logvar) numpy arrays (or raw 4-vector for evidential)."""
    model.eval()
    if mc_dropout: model.enable_dropout()
    outs = []
    for b, desc, asy, kin, *_ in loader(ds, device, batch_size, shuffle=False):
        o = model(b, desc, asy, kin)
        outs.append(torch.stack(o, 1).cpu() if isinstance(o, tuple) else o.cpu())
    arr = torch.cat(outs).numpy()
    return (arr[:, 0], arr[:, 1]) if arr.shape[1] == 2 else arr


def run_epochs(model, tr: LabelledSet, va: LabelledSet | None, device, epochs, optimizer, scheduler=None,
               loss_mode="huber", huber_delta=1.0, cens_weight=0.5, ref_state=None, alphas=None,
               patience=None, batch_size=256, seed=0, log=None, on_epoch=None, evidential=False):
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); tot = n = 0
        for b, desc, asy, kin, y, c, w in loader(tr, device, batch_size, True, seed + ep):
            optimizer.zero_grad()
            out = model(b, desc, asy, kin)
            if evidential:
                loss = evidential_loss(out, y)
            else:
                mu, lv = out
                loss = task_loss(mu, lv, y, c, w, loss_mode, delta=huber_delta, cens_weight=cens_weight)
            if ref_state is not None:
                loss = loss + l2sp(model, ref_state, alphas["enc"], alphas["head"])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            tot += loss.item() * len(y); n += len(y)
        if scheduler: scheduler.step()
        v = tot / max(n, 1)
        if va is not None and len(va):
            pr = predict(model, va, device)
            mu = pr[:, 0] if evidential else pr[0]
            ex = va.c == 0                                          # early-stopping metric: val RMSE on exact labels
            v = float(np.sqrt(np.mean((mu[ex] - va.y[ex]) ** 2))) if ex.any() else float(np.mean(np.maximum(mu - va.y, 0)))
        if log: log({"epoch": ep, "train_loss": tot / max(n, 1), "val": v})
        if on_epoch: on_epoch(ep, v)
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if patience and bad >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return best
