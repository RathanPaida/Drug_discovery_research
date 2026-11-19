"""Datasets / collation shared by all torch stages."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .graph import mol_to_graph, collate_graphs
from .dmpnn import to_torch
from ..baselines.features import ASSAYS

# Molecular-graph cache.
#
# This was an unbounded dict. At ~425k training records over ~180k unique molecules that
# was tolerable, but with data.full_kinome=true the table carries ~1.35M records over
# several hundred thousand unique structures. A featurised graph is roughly 20 KB (N x 133
# atom features plus directed bond features and index arrays), so an uncapped cache
# reaches ~10 GB and OOMs a 16 GB machine partway through Stage 3 -- after hours of work.
#
# A bounded FIFO is the right trade here rather than an LRU: training shuffles over the
# whole corpus every epoch, so the access pattern has no locality for an LRU to exploit
# and any bounded policy degrades to the same hit rate. Recomputing a graph costs ~1 ms
# against a GPU step measured in tens of ms, so misses are cheap.
#
# Tune with $TNK1_GRAPH_CACHE (entries; 0 disables the bound). 300k entries is ~6 GB.
import os as _os
from collections import OrderedDict as _OrderedDict

_CACHE_MAX = int(_os.environ.get("TNK1_GRAPH_CACHE", "300000"))
_CACHE: "_OrderedDict[str, dict]" = _OrderedDict()


def get_graph(smi):
    g = _CACHE.get(smi)
    if g is None:
        g = _CACHE[smi] = mol_to_graph(smi)
        if _CACHE_MAX and len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)          # evict oldest
    return g


def cache_stats() -> dict:
    return {"entries": len(_CACHE), "max": _CACHE_MAX or None}


class DescStore:
    """Scaled 200-d RDKit descriptors keyed by SMILES (scaler fit on unlabeled/train molecules only)."""
    def __init__(self, scaler):
        self.scaler, self.cache = scaler, {}
        from ..utils.paths import features_dir
        # single resolver (honours $TNK1_FEATURES_DIR); the previous parents[3]/parents[4]
        # candidates were off by one and never resolved, so this always fell through to a
        # hardcoded d:/ path that exists only on the original author's machine.
        candidates = [features_dir()]
        p_npz, p_parq = None, None
        for c in candidates:
            npz_candidate = c / "unique_molecules_descriptors_200d.npz"
            parq_candidate = c / "unique_molecules_788k.parquet"
            if npz_candidate.exists() and parq_candidate.exists():
                p_npz, p_parq = npz_candidate, parq_candidate
                break
        # How much of the 787,283-row descriptor store to preload.
        #   unset -> all of it (the historical behaviour, ~1 GB resident)
        #   N     -> the first N rows only
        #   0     -> nothing; every descriptor is computed on demand by get()
        # The preloaded dict costs roughly 1.2 kB per molecule (a 200-float32 array plus
        # numpy and dict overhead), so the full store is ~1 GB resident on top of a
        # ~1.26 GB transient spike during construction. That is affordable at 16 GB and
        # not at 8 GB -- see docs/RUNBOOK.md. A miss costs one RDKit descriptor
        # calculation (~1-3 ms), so a small cache trades wall-clock, not correctness.
        _dc = _os.environ.get("TNK1_DESC_CACHE")
        max_entries = None if _dc in (None, "") else int(_dc)
        if max_entries != 0 and p_npz and p_parq and p_npz.exists() and p_parq.exists():
            try:
                import pandas as pd
                df_mol = pd.read_parquet(p_parq, columns=["canonical_smiles"])
                npz = np.load(p_npz)
                desc = npz["descriptors_200d"]
                keys = df_mol.canonical_smiles.values
                if max_entries is not None:
                    desc, keys = desc[:max_entries], keys[:max_entries]
                # Transform in chunks. Doing it in one call holds the raw and the scaled
                # matrices simultaneously -- 630 MB each at full size -- which is the
                # single largest transient allocation in the pipeline.
                step = 100_000
                for i in range(0, len(desc), step):
                    block = self.scaler.transform(desc[i:i + step])
                    self.cache.update(zip(keys[i:i + step], block))
                    del block
                del desc, npz, df_mol, keys
            except Exception as e:
                # Not fatal: `get()` recomputes any miss, so results are unchanged -- only
                # slower. But warn instead of swallowing, so a silently-dropped 787k-row
                # cache does not get mistaken for a performance mystery.
                import warnings
                warnings.warn(
                    f"precomputed descriptor cache at {p_npz} could not be loaded "
                    f"({type(e).__name__}: {e}); descriptors will be recomputed on demand",
                    RuntimeWarning, stacklevel=2,
                )

    def get(self, smiles):
        miss = [s for s in dict.fromkeys(smiles) if s not in self.cache]
        if not miss:
            return np.stack([self.cache[s] for s in smiles])
        from ..baselines.features import rdkit_descriptors
        local = dict(zip(miss, self.scaler.transform(rdkit_descriptors(miss))))

        # Build the answer BEFORE touching the cache. Trimming first cannot help when a
        # single call misses more rows than the ceiling (an empty cache has nothing to
        # evict), and trimming after the update can evict a row this very call needs.
        out = np.stack([self.cache[s] if s in self.cache else local[s] for s in smiles])

        _dc = _os.environ.get("TNK1_DESC_CACHE")
        cap = None if _dc in (None, "") else int(_dc)
        if cap == 0:
            return out                      # caching disabled: retain nothing
        self.cache.update(local)
        if cap is not None and len(self.cache) > cap:
            # evict in insertion order: training shuffles the corpus, so there is no
            # locality for a smarter policy to exploit (same reasoning as _CACHE above)
            for k in list(self.cache)[:len(self.cache) - cap]:
                self.cache.pop(k, None)
        return out


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
