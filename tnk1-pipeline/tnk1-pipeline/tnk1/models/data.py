"""Datasets / collation shared by all torch stages."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .graph import mol_to_graph, collate_graphs
from .dmpnn import to_torch
from ..baselines.features import ASSAYS

_CACHE: dict[str, dict] = {}


def get_graph(smi):
    g = _CACHE.get(smi)
    if g is None:
        g = _CACHE[smi] = mol_to_graph(smi)
    return g


class DescStore:
    """Scaled 200-d RDKit descriptors keyed by SMILES (scaler fit on unlabeled/train molecules only)."""
    def __init__(self, scaler):
        self.scaler, self.cache = scaler, {}
        from pathlib import Path
        candidates = [
            Path(__file__).resolve().parents[3] / "ml_research" / "features",
            Path(__file__).resolve().parents[4] / "features",
            Path("ml_research/features"),
            Path("d:/ML_Research/ml_research/ml_research/features"),
        ]
        p_npz, p_parq = None, None
        for c in candidates:
            npz_candidate = c / "unique_molecules_descriptors_200d.npz"
            parq_candidate = c / "unique_molecules_788k.parquet"
            if npz_candidate.exists() and parq_candidate.exists():
                p_npz, p_parq = npz_candidate, parq_candidate
                break
        if p_npz and p_parq and p_npz.exists() and p_parq.exists():
            try:
                import pandas as pd
                df_mol = pd.read_parquet(p_parq, columns=["canonical_smiles"])
                npz = np.load(p_npz)
                desc = npz["descriptors_200d"]
                scaled = self.scaler.transform(desc)
                self.cache = dict(zip(df_mol.canonical_smiles, scaled))
            except Exception:
                pass

    def get(self, smiles):
        miss = [s for s in dict.fromkeys(smiles) if s not in self.cache]
        if miss:
            from ..baselines.features import rdkit_descriptors
            X = self.scaler.transform(rdkit_descriptors(miss))
            self.cache.update(zip(miss, X))
        return np.stack([self.cache[s] for s in smiles])


class LabelledSet(Dataset):
    """Rows: smiles, kinase index, assay idx, y(pAct or censor bound), censored flag, weight."""
    def __init__(self, smiles, kin_idx, assay_idx, y, cens, w, desc_store: DescStore):
        self.smiles = list(smiles); self.kin = np.asarray(kin_idx, dtype=np.int64); self.asy = np.asarray(assay_idx, dtype=np.int64)
        self.y = np.asarray(y, np.float32); self.c = np.asarray(cens, np.int64); self.w = np.asarray(w, np.float32)
        self.desc = desc_store.get(self.smiles)

    def __len__(self): return len(self.smiles)
    def __getitem__(self, i): return i


def make_collate(ds: LabelledSet, device):
    def fn(idx):
        idx = np.asarray(idx)
        b = to_torch(collate_graphs([get_graph(ds.smiles[i]) for i in idx]), device)
        T = lambda a, dt=None: torch.as_tensor(a[idx], dtype=dt).to(device)
        asy = torch.zeros(len(idx), 3, device=device); asy[torch.arange(len(idx)), torch.as_tensor(ds.asy[idx], dtype=torch.long)] = 1
        return b, T(ds.desc, torch.float32), asy, T(ds.kin, torch.long), T(ds.y, torch.float32), T(ds.c, torch.long), T(ds.w, torch.float32)
    return fn


def loader(ds, device, batch_size=256, shuffle=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=make_collate(ds, device), generator=g)


def assay_index(types): return np.array([ASSAYS.index(t) if t in ASSAYS else 0 for t in types])
