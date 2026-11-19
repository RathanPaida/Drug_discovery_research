"""Sec 7.1: multi-objective reward. Every component is mapped to [0,1]; aggregated with a weighted geometric
mean (floor 1e-4) and multiplied by a hard med-chem gate. Pure numpy so it is unit-testable."""
from __future__ import annotations
import numpy as np

COMPONENTS = ["pot", "sel", "ad", "sa", "adme", "safety", "qed"]

# Cycle-02 component set. `COMPONENTS` above is the Cycle-01 set and stays the default so
# locked results reproduce; pass `components=COMPONENTS_V2` to `aggregate` to use this one.
#
# Two terms were dropped, not demoted. `component_signal` on the shipped pool measured adme
# at 0.019 and qed at 0.016 -- the noise floor -- so they consumed 1.5 of the 10.5 weight
# budget and returned no gradient in exchange. They are now enforced where a flat criterion
# costs nothing: qed in `hard_gate`/`tpp_select`, adme at final selection.
#
# `ad` is KEPT, at a reduced weight, against the brief's suggestion to drop it. It was the
# largest realised driver in Cycle-01 (0.380) only because d_lat had blown up to 240,832
# against a threshold of 1.007; with the applicability fix (sd_floor_frac=0.05,
# d_lat_clip=50.0) d_lat maxes at ~1.3 and the term is well-behaved. It is also the only
# thing stopping the policy from optimising the critic off-distribution, which matters a
# great deal here: the deep ensemble's test Spearman is 0.264, so its extrapolations are
# close to noise, and a generator rewarded for exploiting them will do exactly that.
COMPONENTS_V2 = ["pot", "sel", "hinge", "ad", "sa", "safety"]


def sigmoid(x, c, k): return 1.0 / (1.0 + np.exp(-k * (np.asarray(x, float) - c)))


def beta_schedule(step: int, b0=0.3, b1=1.5, explore_steps=300, total=1000) -> float:
    """beta=0.3 for steps 1..300, then linear 0.3 -> 1.5 over 301..1000."""
    if step <= explore_steps: return b0
    return float(b0 + (b1 - b0) * min(1.0, (step - explore_steps) / max(total - explore_steps, 1)))


def r_pot(mu, sigma_epi, beta, c=7.0, k=2.0): return sigmoid(mu - beta * sigma_epi, c, k)
def r_sel(mu_sel, sigma_sel, gamma=1.0, c=1.2, k=2.0): return sigmoid(mu_sel - gamma * sigma_sel, c, k)
def r_ad(d_lat, d_thr, scale=0.15): return np.exp(-np.maximum(0.0, np.asarray(d_lat) - d_thr) / scale)
def r_sa(sa, lo=2.0, hi=5.5): return np.clip((hi - np.asarray(sa)) / (hi - lo), 0, 1)


# ---------------------------------------------------------------- reward calibration
# A sigmoid component only produces a usable gradient near its centre. Pin the centre
# somewhere the population does not reach and the term degenerates into a constant
# multiplier that still consumes its share of the weight budget. That is what happened to
# r_sel in Cycle 01: c was 1.2 while mu_sel over the whole 12,000-molecule pool had
# mean -1.24 and MAXIMUM 0.25, so no molecule ever reached the midpoint, 70% of the pool
# scored below 0.01, and a nominal weight of 2.0/10.5 bought a near-flat ~0.38x penalty.

def calibrate_centre(values, target_frac=0.30) -> float:
    """Centre a sigmoid so ~`target_frac` of `values` scores above 0.5.

    Use this to set `pot_c` / `sel_c` from a representative sample of the generator's own
    output (e.g. 10k molecules from the prior) instead of hand-picking a target value.
    A target_frac near 0.3 keeps the component discriminative: most candidates sit on the
    rising part of the curve rather than pinned to either rail."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite values to calibrate against")
    return float(np.percentile(v, 100.0 * (1.0 - target_frac)))


def component_signal(comps: dict, weights: dict, components=None) -> dict:
    """Realised influence of each reward term: sd(log r) x normalised weight.

    `weights` states intended influence; this measures the delivered amount. They diverge
    badly in practice. On the shipped Cycle-01 pool the intended ranking was
    pot(3.0) > sel(2.0) > ad(1.5) = safety(1.5) > sa(1.0) = adme(1.0) > qed(0.5), but the
    realised ranking was ad 0.380 > pot 0.335 > sel 0.157 > safety 0.080 > sa 0.037 >
    adme 0.019 > qed 0.016 -- a weight-1.5 regulariser was steering the objective.

    READ IT WITH ITS BLIND SPOT. A LOW value means the term is flat and returns no
    gradient: adme 0.019 and qed 0.016 were effectively constants, and such terms belong
    in `hard_gate` where they cost nothing. A HIGH value does NOT certify a healthy term.
    A sigmoid pinned to its saturated tail has log r ~ k(x - c), linear in x, so its
    log-variance is at its MAXIMUM precisely when every molecule sits on one rail -- which
    is exactly the state r_sel was in. Pair this with the component's mean and range
    (`r.mean()`, `r.max() - r.min()`): a term whose mean is ~0.01 is crushing R's ceiling
    no matter how much log-variance it shows. Aim for the survivors to land within ~2x of
    each other AND to span a usable fraction of [0,1]."""
    components = components or COMPONENTS
    wsum = sum(weights[c] for c in components)
    out = {}
    for c in components:
        lv = np.log(np.maximum(np.asarray(comps[c], float), 1e-4))
        out[c] = float(lv.std() * weights[c] / wsum)
    return out


def r_adme(r_aqsol, r_caco2, r_micro):
    return (np.asarray(r_aqsol) * np.asarray(r_caco2) * np.asarray(r_micro)) ** (1 / 3)


def r_safety(p_herg, p_ames, p_cyp3a4):
    return (1 - np.asarray(p_herg)) * (1 - np.asarray(p_ames)) * (1 - 0.5 * np.asarray(p_cyp3a4))


def r_safety_tpp(p_herg, p_ames, p_cyp3a4, herg_c=0.20, ames_c=0.30, cyp_c=0.50, k=12.0):
    """TPP-anchored safety: geometric mean of descending sigmoids centred on each threshold.

    `r_safety` (above) is a linear product, so it discounts rather than discriminates. At
    the Cycle-01 contract's mean p(hERG) of 0.387 it returns a 0.613 factor -- a mild tax,
    not a signal -- and because the geometric-mean `aggregate` raises it to w/sum(w), even
    a catastrophic 0.865 only scales R by about 0.7x. That is why 29 of the shipped 40
    compounds exceed the TPP's hERG ceiling while still scoring well.

    Centring a steep sigmoid ON the TPP threshold puts the steepest gradient exactly in the
    decision region: a molecule at p(hERG)=0.20 scores 0.5, at 0.10 about 0.77, at 0.40
    about 0.10. The term now PUSHES toward the threshold from both sides instead of
    tolerating anything below 1.0, and it does so without deleting the molecule the way a
    hard gate would, so the gradient survives (see `hard_gate`'s note).

    k=12 is deliberately steep but still smooth: it spans roughly +/-0.25 probability units
    around the centre, which is wider than the endpoint models' own calibration error.

    The mean is WEIGHTED, not plain. An unweighted cube root dilutes each endpoint to a
    1/3 exponent, so a tenfold difference in the hERG factor survives as only 10**(1/3) =
    2.2x in the reward -- and hERG is the endpoint 29 of the 40 shipped compounds actually
    fail, while CYP3A4 inhibition is a DDI liability rather than a stop. Weighting hERG
    2.0 / AMES 1.0 / CYP3A4 0.5 keeps the term's overall scale while letting the endpoint
    that is failing dominate the gradient.
    """
    w_herg, w_ames, w_cyp = 2.0, 1.0, 0.5
    wsum = w_herg + w_ames + w_cyp
    terms = [
        (w_herg, 1.0 - sigmoid(p_herg, herg_c, k)),
        (w_ames, 1.0 - sigmoid(p_ames, ames_c, k)),
        (w_cyp, 1.0 - sigmoid(p_cyp3a4, cyp_c, k * 0.5)),  # liability, not a hard stop
    ]
    return np.exp(sum(wt / wsum * np.log(np.maximum(t, 1e-4)) for wt, t in terms))


def r_hinge(hinge_bool, floor=0.1):
    """Kinase hinge-binding motif present (1.0) or absent (`floor`).

    Deliberately NOT a hard gate. Only 27.5% of the Cycle-01 pool matched any hinge SMARTS,
    so gating on it would discard three quarters of every batch; and the six patterns in
    `tnk1.admet.filters.HINGE_SMARTS` are a known-motif whitelist, so a hard gate would also
    forbid the generator from ever discovering an unlisted hinge binder.

    MIND THE DILUTION. `aggregate` raises each term to w/sum(w), so the realised penalty is
    floor**(w/sum w), NOT floor. At the configured hinge weight of 2.0 out of 12.5 that is
    0.1**0.16 = 0.69 -- a hinge-less molecule keeps 69% of its reward, not 10%. Raising the
    weight barely helps (w=4.0 only reaches 0.53) because the exponent moves slowly, so if
    the hinge-match rate has to climb faster, LOWER THE FLOOR rather than raise the weight:
    floor=0.01 gives 0.48 at the same weight. A persistent 0.69x multiplier applied to the
    72.5% of the population that lacks a hinge motif is still a strong, consistent gradient
    over a 1000-step run; it is simply not the 10x penalty the floor value suggests.
    """
    return np.where(np.asarray(hinge_bool, dtype=bool), 1.0, float(floor))


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


def hard_gate(valid, n_alerts, mw, p_ames, mw_max=600.0, p_herg=None, p_herg_max=0.85,
              *, sa=None, sa_max=None, n_rot=None, rot_max=None, qed=None, qed_min=None,
              p_ames_max=0.5):
    """Tier A: the ABSURDITY gate. Binary; anything failing it scores R = 0.

    The new criteria are keyword-only and default to None (= not enforced), so every
    existing positional call site and the locked Cycle-01 numbers reproduce unchanged.

    SET THESE LOOSE, AND READ `tpp_select` BEFORE TIGHTENING THEM. A hard gate does not
    shape behaviour, it only deletes it: every molecule it rejects contributes R = 0 and
    therefore ZERO gradient, exactly like a molecule that is merely bad. Measured on the
    Cycle-01 contract -- the best 40 of 12,000 -- the TPP thresholds applied as a joint
    hard gate (hERG<=0.20, AMES<=0.30, SA<=4.5, QED>=0.35, rot<=7) pass only 5 of 40:

        hERG<=0.20   11/40   cumulative 11
        AMES<=0.30   27/40   cumulative 10
        SA<=4.5      25/40   cumulative  7
        QED>=0.35    34/40   cumulative  5
        rot<=7       32/40   cumulative  5

    On raw generator output the joint survival is far below 1%, so a DAP batch of 128
    would contain ~1 non-zero reward and the policy would learn nothing at all. That is a
    worse failure than the flat r_sel it was meant to fix: a flat term still leaves the
    other terms' gradient intact, whereas a gate that rejects everything removes all of it.

    The TPP belongs in two other places instead:
      * as the CENTRE of the shaped `r_safety_tpp` / `r_sa` terms, which carry gradient
        toward the threshold from both sides (Tier B), and
      * as a strict filter at final selection, `tpp_select`, where the pool is already
        1500 optimised candidates rather than a live generator batch (Tier C).
    """
    res = (np.asarray(valid, bool) & (np.asarray(n_alerts) == 0) & (np.asarray(mw) <= mw_max)
           & (np.asarray(p_ames) <= p_ames_max))
    if p_herg is not None:
        res = res & (np.asarray(p_herg) <= p_herg_max)
    if sa is not None and sa_max is not None:
        res = res & (np.asarray(sa, float) <= sa_max)
    if n_rot is not None and rot_max is not None:
        res = res & (np.asarray(n_rot, float) <= rot_max)
    if qed is not None and qed_min is not None:
        res = res & (np.asarray(qed, float) >= qed_min)
    return res.astype(float)


# ---------------------------------------------------------------- Tier C: final selection
# TPP compliance is a SHIPPING criterion, not a generation criterion. Applied here the
# thresholds cost nothing -- the pool is already optimised and we are choosing which 40 of
# ~1500 to synthesise -- whereas applied in `hard_gate` they starve the policy of gradient.
TPP = {"herg_max": 0.20, "ames_max": 0.30, "sa_max": 4.5, "qed_min": 0.35, "rot_max": 7,
       "mw_max": 600.0}


def tpp_select(df, tpp: dict | None = None, columns: dict | None = None):
    """Boolean mask of rows meeting the full TPP. Use at Stage 8 contract assembly.

    `columns` maps TPP keys to dataframe column names; the defaults match the Cycle-01
    contract schema (`runs/cycle_01/contract_to_team_b.csv`). Only criteria whose column is
    actually present are applied, and the ones that were skipped are reported so a caller
    can never mistake "column absent" for "criterion met" -- the same fail-closed discipline
    as `require_endpoints`.

    Returns (mask, report) where report lists per-criterion pass counts and any skips.
    """
    t = {**TPP, **(tpp or {})}
    cols = {"herg_max": "tox_herg_prob", "ames_max": "tox_ames_prob", "sa_max": "sa_score",
            "qed_min": "qed_score", "rot_max": "rot_bonds", "mw_max": "mol_weight",
            **(columns or {})}
    mask = np.ones(len(df), dtype=bool)
    report = {"applied": {}, "skipped": []}
    for key, thr in t.items():
        col = cols.get(key)
        if col is None or col not in df.columns:
            report["skipped"].append(key)
            continue
        v = np.asarray(df[col], dtype=float)
        ok = (v >= thr) if key.endswith("_min") else (v <= thr)
        report["applied"][key] = int(ok.sum())
        mask &= ok
    report["n_pass"] = int(mask.sum())
    report["n_total"] = int(len(df))
    return mask, report


def aggregate(comps: dict, weights: dict, gate, veto: dict | None = None,
              components=None) -> np.ndarray:
    """R = I_hard * prod_j max(r_j, 1e-4)^(w_j / sum w).
    CAUTION (spec check): with normalised exponents a component that collapses to the 1e-4 floor only scales R by
    (1e-4)^(w_j/sum w) -- e.g. 0.27 for safety (w=1.5 of 10.5), 0.07 for potency -- i.e. NOT 'near zero'.
    `veto={'safety':0.05,'sa':0.05}` optionally zeroes R when a component falls below its threshold; default off
    (= manual as written). Enable via cfg rl.veto after reviewing with the PI."""
    components = components or COMPONENTS
    wsum = sum(weights[c] for c in components)
    logr = sum(weights[c] / wsum * np.log(np.maximum(np.asarray(comps[c], float), 1e-4)) for c in components)
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
