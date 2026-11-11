"""Directed-bond molecular graphs: 133-d atom features, 14-d bond features (Chemprop-style)."""
from __future__ import annotations
import numpy as np

ATOM_Z = list(range(1, 101))            # 100 + unknown = 101
DEG, CHG, CHI, NH = list(range(6)), [-2, -1, 0, 1, 2], [0, 1, 2, 3], list(range(5))
HYB = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
ATOM_DIM, BOND_DIM = 133, 14


def _oh(v, choices):
    o = [0.0] * (len(choices) + 1)
    o[choices.index(v) if v in choices else -1] = 1.0
    return o


def atom_features(a) -> list[float]:
    f = (_oh(a.GetAtomicNum(), ATOM_Z) + _oh(a.GetTotalDegree(), DEG) + _oh(a.GetFormalCharge(), CHG)
         + _oh(int(a.GetChiralTag()), CHI) + _oh(a.GetTotalNumHs(), NH) + _oh(str(a.GetHybridization()), HYB)
         + [1.0 if a.GetIsAromatic() else 0.0, a.GetMass() * 0.01])
    assert len(f) == ATOM_DIM
    return f


def bond_features(b) -> list[float]:
    from rdkit import Chem
    bt = b.GetBondType()
    f = [0.0,  # null-bond slot (unused for real bonds)
         bt == Chem.BondType.SINGLE, bt == Chem.BondType.DOUBLE, bt == Chem.BondType.TRIPLE, bt == Chem.BondType.AROMATIC,
         b.GetIsConjugated() if bt is not None else 0, b.IsInRing()]
    st = [0.0] * 7; st[min(int(b.GetStereo()), 6)] = 1.0     # none/any/Z/E/cis/trans (+slack)
    f = [float(x) for x in f] + st
    assert len(f) == BOND_DIM
    return f


def mol_to_graph(smiles: str) -> dict:
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    x = np.array([atom_features(a) for a in m.GetAtoms()], dtype=np.float32)
    labels = np.array([[min(max(a.GetAtomicNum(), 0), 100), min(a.GetTotalDegree(), 5), int(a.GetIsAromatic())]
                       for a in m.GetAtoms()], dtype=np.int64)
    src, dst, ef = [], [], []
    for b in m.GetBonds():
        i, j, bf = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_features(b)
        src += [i, j]; dst += [j, i]; ef += [bf, bf]
    E = len(src)
    rev = np.arange(E) ^ 1                                   # edge 2k <-> 2k+1
    return {"x": x, "labels": labels, "src": np.array(src, np.int64), "dst": np.array(dst, np.int64),
            "ef": np.array(ef, np.float32).reshape(E, BOND_DIM), "rev": rev.astype(np.int64)}


def collate_graphs(graphs: list[dict]) -> dict:
    """Concatenate with atom/edge offsets; returns numpy dict (converted to tensors by the caller)."""
    xs, lb, src, dst, ef, rev, batch = [], [], [], [], [], [], []
    ao = eo = 0
    for gi, g in enumerate(graphs):
        n, e = len(g["x"]), len(g["src"])
        xs.append(g["x"]); lb.append(g["labels"]); ef.append(g["ef"])
        src.append(g["src"] + ao); dst.append(g["dst"] + ao); rev.append(g["rev"] + eo)
        batch.append(np.full(n, gi, np.int64)); ao += n; eo += e
    cat = np.concatenate
    return {"x": cat(xs), "labels": cat(lb), "ef": cat(ef), "src": cat(src), "dst": cat(dst), "rev": cat(rev),
            "batch": cat(batch), "n_mol": len(graphs)}
