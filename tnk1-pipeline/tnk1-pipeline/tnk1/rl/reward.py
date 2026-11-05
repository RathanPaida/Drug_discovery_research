"""Sec 7.1: multi-objective reward. Every component is mapped to [0,1]; aggregated with a weighted geometric
mean (floor 1e-4) and multiplied by a hard med-chem gate. Pure numpy so it is unit-testable."""
from __future__ import annotations
import numpy as np

COMPONENTS = ["pot", "sel", "ad", "sa", "adme", "safety", "qed"]


def sigmoid(x, c, k): return 1.0 / (1.0 + np.exp(-k * (np.asarray(x, float) - c)))


def beta_schedule(step: int, b0=0.3, b1=1.5, explore_steps=300, total=1000) -> float:
    """beta=0.3 for steps 1..300, then linear 0.3 -> 1.5 over 301..1000."""
    if step <= explore_steps: return b0
    return float(b0 + (b1 - b0) * min(1.0, (step - explore_steps) / max(total - explore_steps, 1)))


def r_pot(mu, sigma_epi, beta, c=7.0, k=2.0): return sigmoid(mu - beta * sigma_epi, c, k)
def r_sel(mu_sel, sigma_sel, gamma=1.0, c=1.2, k=2.0): return sigmoid(mu_sel - gamma * sigma_sel, c, k)
def r_ad(d_lat, d_thr, scale=0.15): return np.exp(-np.maximum(0.0, np.asarray(d_lat) - d_thr) / scale)
def r_sa(sa, lo=2.0, hi=5.5): return np.clip((hi - np.asarray(sa)) / (hi - lo), 0, 1)


def r_adme(r_aqsol, r_caco2, r_micro):
    return (np.asarray(r_aqsol) * np.asarray(r_caco2) * np.asarray(r_micro)) ** (1 / 3)


def r_safety(p_herg, p_ames, p_cyp3a4):
    return (1 - np.asarray(p_herg)) * (1 - np.asarray(p_ames)) * (1 - 0.5 * np.asarray(p_cyp3a4))


def r_novelty(inchikeys, known_inchikeys_set) -> np.ndarray:
    """Strict cryptographic novelty verification gate: returns 1.0 if unseen, 0.0 if known."""
    return np.array([0.0 if ik in known_inchikeys_set or (ik[:14] in known_inchikeys_set if ik else False) else 1.0 for ik in inchikeys], dtype=float)


def hard_gate(valid, n_alerts, mw, p_ames, mw_max=600.0, p_herg=None, p_herg_max=0.85):
    res = (np.asarray(valid, bool) & (np.asarray(n_alerts) == 0) & (np.asarray(mw) <= mw_max)
           & (np.asarray(p_ames) <= 0.5))
    if p_herg is not None:
        res = res & (np.asarray(p_herg) <= p_herg_max)
    return res.astype(float)


def aggregate(comps: dict, weights: dict, gate, veto: dict | None = None) -> np.ndarray:
    """R = I_hard * prod_j max(r_j, 1e-4)^(w_j / sum w).
    CAUTION (spec check): with normalised exponents a component that collapses to the 1e-4 floor only scales R by
    (1e-4)^(w_j/sum w) -- e.g. 0.27 for safety (w=1.5 of 10.5), 0.07 for potency -- i.e. NOT 'near zero'.
    `veto={'safety':0.05,'sa':0.05}` optionally zeroes R when a component falls below its threshold; default off
    (= manual as written). Enable via cfg rl.veto after reviewing with the PI."""
    wsum = sum(weights[c] for c in COMPONENTS)
    logr = sum(weights[c] / wsum * np.log(np.maximum(np.asarray(comps[c], float), 1e-4)) for c in COMPONENTS)
    out = np.asarray(gate, float) * np.exp(logr)
    for c, thr in (veto or {}).items():
        out = np.where(np.asarray(comps[c], float) < thr, 0.0, out)
    return out


class DiversityFilter:
    """Bemis-Murcko bucket: after `bucket` molecules with R>=min_score share a scaffold, further ones get R=0.
    (REINVENT 4's built-in IdenticalMurckoScaffold filter does the same during RL; this copy is used for offline pools.)"""
    def __init__(self, bucket=25, min_score=0.55): self.bucket, self.min, self.mem = bucket, min_score, {}
    def __call__(self, scaffolds, scores):
        out = np.array(scores, float)
        for i, (s, r) in enumerate(zip(scaffolds, scores)):
            if self.mem.get(s, 0) >= self.bucket: out[i] = 0.0
            elif r >= self.min: self.mem[s] = self.mem.get(s, 0) + 1
        return out
