"""Heteroscedastic deep ensemble (M=10) wrapper: training, persistence and member-level inference."""
from __future__ import annotations
from pathlib import Path
import numpy as np, torch
from ..models.model import KinomeModel
from ..models.data import LabelledSet, DescStore
from ..models.trainer import predict
from ..models.pipeline import kinome_stage, finetune_tnk1
from ..utils.common import seed_everything


def train_member(cfg, kin_df, tnk1_train, tnk1_val, store, desc_store, ssl_ckpt, seed, device, shared_ckpt=None, log=None):
    seed_everything(seed)
    if cfg["uq"]["independent_kinome_training"] or shared_ckpt is None:
        model = kinome_stage(cfg, kin_df, store, desc_store, ssl_ckpt, "B", seed=seed, device=device, log=log)
    else:
        model = KinomeModel(cfg, store, design="B").to(device); model.load_state_dict(torch.load(shared_ckpt, map_location=device))
    return finetune_tnk1(cfg, model, tnk1_train, tnk1_val, store, desc_store, device, seed, log, use_l2sp=True, loss_mode="nll")


def train_ensemble(cfg, kin_df, tnk1_train, tnk1_val, store, desc_store, ssl_ckpt, out_dir, device="cuda", log=None, shared_ckpt=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    shared = shared_ckpt
    for m in range(cfg["uq"]["n_members"]):
        seed = 1000 + m
        member = train_member(cfg, kin_df, tnk1_train, tnk1_val, store, desc_store, ssl_ckpt, seed, device, shared, log)
        torch.save(member.state_dict(), out / f"member_{m:02d}.pt")
    return out


class Ensemble:
    def __init__(self, cfg, store, desc_store, ckpt_dir, device="cuda"):
        self.cfg, self.store, self.desc, self.device = cfg, store, desc_store, device
        self.kmap = {u: i for i, u in enumerate(store["ids"])}
        self.members = []
        for f in sorted(Path(ckpt_dir).glob("member_*.pt")):
            m = KinomeModel(cfg, store, design="B").to(device); m.load_state_dict(torch.load(f, map_location=device)); m.eval()
            self.members.append(m)
        assert self.members, f"no members in {ckpt_dir}"

    def predict_members(self, smiles, uniprots, assay_idx=0):
        """-> mu (M, n, K), logvar (M, n, K) for every compound x kinase."""
        n, K = len(smiles), len(uniprots)
        S = [s for s in smiles for _ in uniprots]; U = [u for _ in smiles for u in uniprots]
        ds = LabelledSet(S, [self.kmap[u] for u in U], np.full(len(S), assay_idx), np.zeros(len(S)),
                         np.zeros(len(S)), np.ones(len(S)), self.desc)
        mus, lvs = [], []
        for m in self.members:
            mu, lv = predict(m, ds, self.device); mus.append(mu.reshape(n, K)); lvs.append(lv.reshape(n, K))
        return np.stack(mus), np.stack(lvs)

    @torch.no_grad()
    def latent(self, smiles):
        """Member-0 z_mol(300)+desc(200) for AD computation."""
        from ..models.data import make_collate
        ds = LabelledSet(smiles, np.zeros(len(smiles), int), np.zeros(len(smiles), int), np.zeros(len(smiles)),
                         np.zeros(len(smiles)), np.ones(len(smiles)), self.desc)
        fn, out = make_collate(ds, self.device), []
        for i in range(0, len(smiles), 512):
            b, desc, *_ = fn(np.arange(i, min(i + 512, len(smiles))))
            out.append(torch.cat([self.members[0].enc(b), desc], 1).cpu())
        return torch.cat(out).numpy()
