"""Shared loader used by the stage scripts: config, frozen data, splits, kinase store, descriptor scaler."""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np, pandas as pd
from .utils.common import load_config, read_table, verify_locked


class Ctx:
    def __init__(self, config="configs/default.yaml", version="v1", device="cuda"):
        self.cfg = load_config(config); self.version, self.device = version, device
        self.proc = Path(self.cfg["paths"]["processed"]); self.runs = Path(self.cfg["paths"]["runs"]); self.runs.mkdir(exist_ok=True, parents=True)
        p = self.proc / f"kinome_long_{version}.parquet"; self.sha = verify_locked(p)
        self.df = read_table(p)
        self.splits = json.loads((self.proc / f"splits_{version}.json").read_text())
        self.tnk1 = self.cfg["targets"]["tnk1"]
        self.test_keys, self.dev_keys = set(self.splits["test_keys"]), set(self.splits["dev_keys"])

    @property
    def store(self):
        from .models.kinase_features import load_store
        return load_store(self.proc / "kinase_store.npz")

    def desc_scaler(self):
        p = self.proc / "desc_scaler.pkl"
        if p.exists(): return pickle.loads(p.read_bytes())
        from .baselines.features import rdkit_descriptors, DescriptorScaler
        smi = self.df.smiles.drop_duplicates().sample(min(50_000, self.df.smiles.nunique()), random_state=0)
        sc = DescriptorScaler().fit(rdkit_descriptors(list(smi), self.cfg["model"]["n_desc"]))
        p.write_bytes(pickle.dumps(sc)); return sc

    def desc_store(self):
        from .models.data import DescStore
        return DescStore(self.desc_scaler())

    def kinome_train_df(self, exclude_uniprots=()):
        """Leakage-safe kinome training table: TNK1 test compounds removed from ALL kinases."""
        from .data.splits import kinome_train_mask
        return self.df[kinome_train_mask(self.df, self.test_keys, exclude_uniprots)].reset_index(drop=True)

    def tnk1_rows(self, keys, censored=True):
        d = self.df[(self.df.uniprot_id == self.tnk1) & self.df.inchikey_block1.isin(set(keys))]
        return d if censored else d[d.censored == 0]
