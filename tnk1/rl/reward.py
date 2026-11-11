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


# Endpoints that r_adme / r_safety / hard_gate all require to be real predictions.
REQUIRED_ENDPOINTS = ("solubility", "caco2", "microsome", "herg", "ames", "cyp3a4")


def require_endpoints(ep: dict, n: int, names=REQUIRED_ENDPOINTS) -> dict:
    """Fail CLOSED on missing ADMET predictions.

    The previous call sites used `ep.get(name, np.zeros(n))`, which is actively dangerous:
    p(hERG)=p(AMES)=p(CYP3A4)=0 is *perfect* safety (r_safety -> 1.0) and passes `hard_gate`,
    and logS=logPapp=0 / CLint->1e-3 are the *best* possible ADME values. A missing or
    failed-to-load endpoint model therefore produced a flawless safety profile rather than
    an error. Absence of evidence must not score as evidence of safety.

    Returns the validated arrays (float, length n) or raises KeyError/ValueError.
    """
    missing = [k for k in names if k not in ep or ep[k] is None]
    if missing:
        raise KeyError(
            f"missing ADMET endpoint prediction(s): {sorted(missing)}. Scoring is "
            f"fail-closed: train/load every endpoint in cfg['admet']['endpoints'] "
            f"(Stage 5 `make admet`) before scoring candidates."
        )
    out = {}
    for k in names:
        v = np.asarray(ep[k], dtype=float).reshape(-1)
        if v.shape[0] != n:
            raise ValueError(f"endpoint '{k}' has length {v.shape[0]}, expected {n}")
        if not np.isfinite(v).all():
            raise ValueError(f"endpoint '{k}' contains non-finite predictions")
        out[k] = v
    return out


def adme_safety_components(ep: dict, n: int) -> dict:
    """The single definition of the `adme` and `safety` reward terms.

    Both `tnk1.rl.scorer.Scorer.score` (online RL) and `scripts/run_full_pipeline.py`
    (offline scoring) call this, so the two paths cannot drift apart again -- they already
    had: the offline path omitted the p(hERG) argument to `hard_gate`, silently dropping
    the hERG hard filter for every candidate it scored.
    """
    e = require_endpoints(ep, n)
    adme = r_adme(
        sigmoid(e["solubility"], -5.0, 3.0),
        sigmoid(e["caco2"], -5.15, 3.0),
        1 - sigmoid(np.log10(np.maximum(e["microsome"], 1e-3)), np.log10(30.0), 3.0),
    )
    return {"adme": adme, "safety": r_safety(e["herg"], e["ames"], e["cyp3a4"]), "ep": e}


def r_novelty(inchikeys, known_inchikeys, known_skeletons=None) -> np.ndarray:
    """Strict InChIKey novelty gate: 1.0 if unseen, 0.0 if known (or unparseable).

    `known_inchikeys` holds full 27-character Standard InChIKeys; `known_skeletons` holds
    14-character skeletal (connectivity-layer) hashes. They are SEPARATE sets: a 14-char
    skeleton is never a member of a set of 27-char keys, so passing one set for both
    silently disables the skeletal check. When `known_skeletons` is omitted only the exact
    check is applied.

    An empty/None InChIKey means standardisation failed, which scores 0.0 (not novel) --
    novelty must be positively demonstrated, never inferred from missing data.

    NOTE: this component is deliberately NOT in COMPONENTS and is not applied by
    `aggregate`. Generation-time novelty is enforced by REINVENT's own prior-novelty check
    plus the post-hoc Stage 7 audit in `tnk1.novelty.verifier`. Wire it in as a multiplier
    only after agreeing the reward change with the PI, since it alters published rewards.
    """
    known = set(known_inchikeys)
    skel = set(known_skeletons) if known_skeletons is not None else set()
    out = []
    for ik in inchikeys:
        if not ik:
            out.append(0.0)
        elif ik in known or ik[:14] in skel:
            out.append(0.0)
        else:
            out.append(1.0)
    return np.array(out, dtype=float)


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
