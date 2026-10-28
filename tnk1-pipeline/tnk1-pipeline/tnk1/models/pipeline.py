"""Stage 3 orchestration: kinome multi-task training (Design A/B), LOKO lambda tuning, 3-phase TNK1 fine-tuning."""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from .model import KinomeModel
from .data import LabelledSet, DescStore, assay_index
from .trainer import run_epochs, predict
from .weighting import sample_weights
from .kinase_features import load_store
from ..uq.metrics import full_report


def _mk(df, kin_idx_map, desc_store, weights=None, tau=5.0):
    valid = df.uniprot_id.isin(kin_idx_map)
    if not valid.all():
        df = df[valid].reset_index(drop=True)
        if weights is not None:
            weights = weights[valid]
    y = np.where(df.censored.values == 1, df.pact.values, df.pact.values)       # censored rows: pact = upper bound
    w = df.weight.values if weights is None else weights * df.weight.values
    return LabelledSet(df.smiles.values, df.uniprot_id.map(kin_idx_map).values, assay_index(df.standard_type.values),
                       y, df.censored.values, w, desc_store)


def kinome_stage(cfg, df, store, desc_store, ssl_ckpt=None, design=None, lam=None, seed=42, device="cuda",
                 val_frac_groups=0.05, log=None, head_type="gauss"):
    """Sub-stage 2 + 3: multi-task kinome training with similarity weights, majority cap, censored loss."""
    from ..data.splits import balanced_group_folds
    k = cfg["kinome_train"]; lam = k["lambda_sim"] if lam is None else lam
    kmap = {u: i for i, u in enumerate(store["ids"])}
    df = df[df.uniprot_id.isin(kmap)].reset_index(drop=True)
    S = dict(zip(store["ids"], store["S"])); N = df.uniprot_id.value_counts().to_dict()
    w = sample_weights(df.uniprot_id.values, S, N, lam, k["cap_n"])
    fold = balanced_group_folds(df.scaffold.values, int(1 / val_frac_groups), seed) == 0
    tr, va = _mk(df[~fold], kmap, desc_store, w[~fold]), _mk(df[fold], kmap, desc_store, w[fold])
    torch.manual_seed(seed)
    model = KinomeModel(cfg, store, design=design, head_type=head_type).to(device)
    if ssl_ckpt:
        model.enc.load_state_dict(torch.load(ssl_ckpt, map_location=device))
    opt = torch.optim.AdamW(model.parameters(), lr=k["lr"], weight_decay=k["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, k["epochs"])
    run_epochs(model, tr, va, device, k["epochs"], opt, sched, "huber", k["huber_delta"], k["cens_weight"],
               patience=8, batch_size=k["batch_size"], seed=seed, log=log)
    return model


def finetune_tnk1(cfg, model, tnk1_train: pd.DataFrame, tnk1_val: pd.DataFrame | None, store, desc_store,
                  device="cuda", seed=42, log=None, use_l2sp=True, three_phase=True, loss_mode=None):
    """Sec 3.5. Phase 1: linear probe (head only, lr 3e-4, 15 ep). Phase 2: unfreeze last MP layer (W_m) +
    readout (W_a, attention, norm) with discriminative LRs (1e-5 enc / 1e-4 head), cosine, early stopping.
    Phase 3 (L2-SP anchor to theta0) is applied as a regulariser throughout phases 1-2."""
    ft = cfg["finetune"]; mode = loss_mode or ft["loss"]
    kmap = {u: i for i, u in enumerate(store["ids"])}
    tr, va = _mk(tnk1_train, kmap, desc_store), (_mk(tnk1_val, kmap, desc_store) if tnk1_val is not None and len(tnk1_val) else None)
    ref = {n: p.detach().clone() for n, p in model.named_parameters()} if use_l2sp else None
    alphas = {"enc": ft["alpha_enc"], "head": ft["alpha_head"]}
    set_grad = lambda ps, flag: [p.requires_grad_(flag) for p in ps]

    if three_phase:
        set_grad(model.parameters(), False); set_grad(model.head_params(), True)
        opt = torch.optim.AdamW(model.head_params(), lr=ft["phase1"]["lr_head"])
        run_epochs(model, tr, va, device, ft["phase1"]["epochs"], opt, None, mode, ref_state=ref, alphas=alphas,
                   batch_size=32, seed=seed, log=log)
    set_grad(model.parameters(), False)
    mp_tail = list(model.enc.W_m.parameters()) + list(model.enc.W_a.parameters()) + list(model.enc.attn.parameters()) + list(model.enc.norm.parameters())
    set_grad(mp_tail + model.head_params(), True)
    p2 = ft["phase2"]
    opt = torch.optim.AdamW([{"params": mp_tail, "lr": p2["lr_enc"]}, {"params": model.head_params(), "lr": p2["lr_head"]}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, p2["epochs"])
    run_epochs(model, tr, va, device, p2["epochs"], opt, sched, mode, ref_state=ref, alphas=alphas,
               patience=p2["patience"], batch_size=32, seed=seed, log=log)
    set_grad(model.parameters(), True)
    return model


def loko_tune_lambda(cfg, df, store, desc_store, ssl_ckpt, related: list[str], device="cuda"):
    """Pick lambda in {0,1,3,10} maximising mean zero-shot Spearman on held-out *related* kinases (excluding TNK1)."""
    res = {}
    for lam in cfg["kinome_train"]["lambda_grid"]:
        sc = []
        for held in related:
            m = kinome_stage(cfg, df[df.uniprot_id != held], store, desc_store, ssl_ckpt, "B", lam, device=device)
            te = df[(df.uniprot_id == held) & (df.censored == 0)]
            kmap = {u: i for i, u in enumerate(store["ids"])}
            mu, _ = predict(m, _mk(te, kmap, desc_store), device)
            sc.append(full_report(te.pact.values, mu)["spearman"])
        res[lam] = float(np.nanmean(sc)); print(f"[LOKO] lambda={lam}: mean Spearman={res[lam]:.3f}")
    return max(res, key=res.get), res
