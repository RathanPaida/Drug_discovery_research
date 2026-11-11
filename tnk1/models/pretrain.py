"""Sec 3.2: self-supervised D-MPNN pretraining on ~1.5M unlabeled ChEMBL molecules.
(1) masked-atom prediction (15% atoms + incident bonds zeroed; predict atomic number, degree, aromaticity)
(2) simultaneous regression of the 200 RDKit descriptors."""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .dmpnn import DMPNNEncoder, to_torch
from .graph import mol_to_graph, collate_graphs


class SSLHeads(nn.Module):
    def __init__(self, hidden, n_desc):
        super().__init__()
        self.z = nn.Linear(hidden, 101); self.deg = nn.Linear(hidden, 6); self.aro = nn.Linear(hidden, 2)
        self.desc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_desc))


class SSLSet(Dataset):
    def __init__(self, smiles, desc_fn): self.s, self.desc_fn = smiles, desc_fn
    def __len__(self): return len(self.s)
    def __getitem__(self, i): return mol_to_graph(self.s[i]), self.s[i]


def mask_batch(b: dict, frac: float, gen: torch.Generator):
    n = b["x"].size(0)
    m = torch.rand(n, generator=gen, device="cpu").to(b["x"].device) < frac
    if not m.any(): m[torch.randint(0, n, (1,))] = True
    x = b["x"].clone(); x[m] = 0
    ef = b["ef"].clone(); inc = m[b["src"]] | m[b["dst"]]; ef[inc] = 0       # incident edges masked too
    return {**b, "x": x, "ef": ef}, m


def pretrain(cfg, smiles, desc_scaler, out_path, device="cuda", seed=42):
    from ..baselines.features import rdkit_descriptors
    c, m = cfg["ssl_pretrain"], cfg["model"]
    enc, heads = DMPNNEncoder(m["hidden"], m["depth"], m["dropout"]).to(device), SSLHeads(m["hidden"], m["n_desc"]).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(heads.parameters()), lr=c["lr"])
    gen = torch.Generator().manual_seed(seed)

    def collate(items):
        gs, sm = zip(*items)
        d = desc_scaler.transform(rdkit_descriptors(list(sm), m["n_desc"]))
        return collate_graphs(list(gs)), torch.tensor(d)
    dl = DataLoader(SSLSet(smiles, None), batch_size=c["batch_size"], shuffle=True, collate_fn=collate,
                    num_workers=c["num_workers"], generator=gen)
    ce = nn.CrossEntropyLoss()
    for ep in range(c["epochs"]):
        tot = 0.0
        for raw, desc in dl:
            b = to_torch(raw, device); desc = desc.to(device)
            mb, msk = mask_batch(b, c["mask_frac"], gen)
            z, hv = enc(mb, return_atoms=True)
            lab = b["labels"][msk]
            l_atom = ce(heads.z(hv[msk]), lab[:, 0]) + ce(heads.deg(hv[msk]), lab[:, 1]) + ce(heads.aro(hv[msk]), lab[:, 2])
            z_full = enc(b)                                                    # unmasked pass for descriptors
            l_desc = nn.functional.mse_loss(heads.desc(z_full), desc)
            loss = l_atom + l_desc
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"[ssl] epoch {ep} loss {tot / len(dl):.4f}")
        torch.save(enc.state_dict(), out_path)
    return out_path
