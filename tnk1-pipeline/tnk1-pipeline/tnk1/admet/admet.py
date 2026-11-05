"""Sec 5.2: seven TDC endpoints, 5-member scaffold-split D-MPNN ensembles (one per endpoint)."""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn
from ..models.dmpnn import DMPNNEncoder, to_torch
from ..models.graph import mol_to_graph, collate_graphs

TPP = {"solubility": ("ge", -5.0), "caco2": ("gt", -5.15), "microsome": ("lt", 30.0),
       "cyp3a4": ("lt", 0.35), "cyp2d6": ("lt", 0.35), "herg": ("lt", 0.20), "ames": ("lt", 0.15)}


class ADMETNet(nn.Module):
    def __init__(self, cfg, task):
        super().__init__(); m = cfg["model"]; self.task = task
        self.enc = DMPNNEncoder(m["hidden"], m["depth"], m["dropout"])
        self.ffn = nn.Sequential(nn.Linear(m["hidden"] + m["n_desc"], 300), nn.ReLU(), nn.Dropout(m["dropout"]), nn.Linear(300, 1))
    def forward(self, b, desc): return self.ffn(torch.cat([self.enc(b), desc], 1))[:, 0]


def load_tdc(group, name, seed=42):
    from pathlib import Path
    import pandas as pd
    candidates = [
        Path(__file__).resolve().parents[3] / "ml_research" / "features" / "admet_7endpoints.parquet",
        Path("ml_research/features/admet_7endpoints.parquet"),
        Path("d:/ML_Research/ml_research/ml_research/features/admet_7endpoints.parquet"),
    ]
    local_p = next((p for p in candidates if p.exists()), Path("admet_7endpoints.parquet"))
    if local_p.exists():
        tdc_col_map = {
            "Solubility_AqSolDB": "aqsol_logS",
            "Caco2_Wang": "caco2_logPapp",
            "Clearance_Microsome_AZ": "clearance_microsome_logCLint",
            "CYP3A4_Veith": "cyp3a4_inhibition",
            "CYP2D6_Veith": "cyp2d6_inhibition",
            "hERG_Karim": "herg_blocker",
            "AMES": "ames_mutagenic"
        }
        col = tdc_col_map.get(name)
        if col:
            df = pd.read_parquet(local_p)
            sub = df[["canonical_smiles", col]].dropna().rename(columns={"canonical_smiles": "Drug", col: "Y"})
            sub = sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            n = len(sub)
            n_tr = int(0.80 * n); n_va = int(0.10 * n)
            return sub.iloc[:n_tr], sub.iloc[n_tr:n_tr+n_va], sub.iloc[n_tr+n_va:]
    from tdc.single_pred import ADME, Tox
    d = (ADME if group == "ADME" else Tox)(name=name)
    s = d.get_split(method="scaffold", seed=seed)
    return s["train"], s["valid"], s["test"]


def train_endpoint(cfg, endpoint, desc_scaler, device="cuda", seeds=(0, 1, 2, 3, 4), epochs=12):
    from ..baselines.features import rdkit_descriptors
    spec = cfg["admet"]["endpoints"][endpoint]; group, name = spec["tdc"]; task = spec["task"]
    tr, va, te = load_tdc(group, name)
    prep = lambda df: ([mol_to_graph(s) for s in df.Drug], desc_scaler.transform(rdkit_descriptors(list(df.Drug))), df.Y.values.astype(np.float32))
    (gtr, dtr, ytr), (gva, dva, yva) = prep(tr), prep(va)
    mu_, sd_ = (ytr.mean(), ytr.std() + 1e-6) if task == "reg" else (0.0, 1.0)
    members = []
    for s in seeds[:cfg["admet"]["n_members"]]:
        torch.manual_seed(s); net = ADMETNet(cfg, task).to(device); opt = torch.optim.AdamW(net.parameters(), 1e-3)
        lossf = nn.MSELoss() if task == "reg" else nn.BCEWithLogitsLoss(); best, bs = 1e9, None
        for ep in range(epochs):
            net.train(); perm = np.random.RandomState(s * 1000 + ep).permutation(len(gtr))
            for i in range(0, len(perm), 128):
                ix = perm[i:i + 128]; b = to_torch(collate_graphs([gtr[j] for j in ix]), device)
                y = torch.tensor((ytr[ix] - mu_) / sd_, device=device)
                loss = lossf(net(b, torch.tensor(dtr[ix], device=device)), y); opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                b = to_torch(collate_graphs(gva), device); p = net(b, torch.tensor(dva, device=device))
                v = float(lossf(p, torch.tensor((yva - mu_) / sd_, device=device)))
            if v < best: best, bs = v, {k: x.clone() for k, x in net.state_dict().items()}
        net.load_state_dict(bs); members.append(net.eval())
    return {"members": members, "task": task, "mu": mu_, "sd": sd_}


@torch.no_grad()
def predict_endpoint(bundle, smiles, desc_scaler, device="cuda", chunk_size=512):
    from ..baselines.features import rdkit_descriptors
    if len(smiles) <= chunk_size:
        b = to_torch(collate_graphs([mol_to_graph(s) for s in smiles]), device)
        d = torch.tensor(desc_scaler.transform(rdkit_descriptors(smiles)), device=device)
        outs = np.stack([m(b, d).cpu().numpy() for m in bundle["members"]])
        if bundle["task"] == "cls": outs = 1 / (1 + np.exp(-outs))
        else: outs = outs * bundle["sd"] + bundle["mu"]
        return outs.mean(0), outs.std(0)
    
    means, stds = [], []
    for i in range(0, len(smiles), chunk_size):
        chunk = smiles[i:i + chunk_size]
        b = to_torch(collate_graphs([mol_to_graph(s) for s in chunk]), device)
        d = torch.tensor(desc_scaler.transform(rdkit_descriptors(chunk)), device=device)
        outs = np.stack([m(b, d).cpu().numpy() for m in bundle["members"]])
        if bundle["task"] == "cls": outs = 1 / (1 + np.exp(-outs))
        else: outs = outs * bundle["sd"] + bundle["mu"]
        means.append(outs.mean(0))
        stds.append(outs.std(0))
    return np.concatenate(means), np.concatenate(stds)


def tpp_pass(endpoint, value) -> bool:
    op, thr = TPP[endpoint]
    return {"ge": value >= thr, "gt": value > thr, "lt": value < thr}[op]
