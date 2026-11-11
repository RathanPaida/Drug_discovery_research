"""Directed message-passing encoder (Sec 3.1) implemented directly in PyTorch (no PyG dependency).
h0_vw = ReLU(W_i[x_v || e_vw]);  m_vw = sum_{k in N(v)\\w} h_kv;  h_vw <- ReLU(h0_vw + W_m m_vw)
h_v = ReLU(W_a[x_v || sum_k h_kv]);  z_mol = LN(sum-pool + attention-pool)."""
import torch
import torch.nn as nn
from .graph import ATOM_DIM, BOND_DIM


def to_torch(b: dict, device) -> dict:
    out = {}
    for k, v in b.items():
        out[k] = torch.as_tensor(v).to(device) if not isinstance(v, int) else v
    return out


class DMPNNEncoder(nn.Module):
    def __init__(self, hidden=300, depth=3, dropout=0.2):
        super().__init__()
        self.hidden, self.depth = hidden, depth
        self.W_i = nn.Linear(ATOM_DIM + BOND_DIM, hidden, bias=False)
        self.W_m = nn.Linear(hidden, hidden, bias=False)       # shared across steps (as in Chemprop)
        self.W_a = nn.Linear(ATOM_DIM + hidden, hidden)
        self.attn = nn.Linear(hidden, 1)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def atom_states(self, b):
        x, ef, src, dst, rev = b["x"], b["ef"], b["src"], b["dst"], b["rev"]
        N = x.size(0)
        if src.numel() == 0:                                    # single-atom molecules
            return torch.relu(self.W_a(torch.cat([x, x.new_zeros(N, self.hidden)], 1)))
        h0 = torch.relu(self.W_i(torch.cat([x[src], ef], 1)))
        h = h0
        for _ in range(self.depth):
            a = torch.zeros(N, self.hidden, device=x.device, dtype=h.dtype).index_add_(0, dst, h)
            m = a[src] - h[rev]                                 # exclude the reverse bond w->v
            h = self.drop(torch.relu(h0 + self.W_m(m)))
        a = torch.zeros(N, self.hidden, device=x.device, dtype=h.dtype).index_add_(0, dst, h)
        return torch.relu(self.W_a(torch.cat([x, a], 1)))

    def pool(self, hv, batch, n_mol):
        s = torch.zeros(n_mol, self.hidden, device=hv.device).index_add_(0, batch, hv)
        w = torch.exp(self.attn(hv).clamp(-10, 10))
        den = torch.zeros(n_mol, 1, device=hv.device).index_add_(0, batch, w)
        att = torch.zeros(n_mol, self.hidden, device=hv.device).index_add_(0, batch, w * hv) / den.clamp_min(1e-9)
        return self.norm(s / 10.0 + att)                         # sum scaled by ~typical heavy-atom count

    def forward(self, b, return_atoms=False):
        hv = self.atom_states(b)
        z = self.pool(hv, b["batch"], b["n_mol"])
        return (z, hv) if return_atoms else z
