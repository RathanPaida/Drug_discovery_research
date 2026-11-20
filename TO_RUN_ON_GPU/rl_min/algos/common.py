"""Algorithm mathematics, as pure numpy. No torch, so every rule here is unit-testable.

WHY THIS FILE EXISTS. The interesting part of a policy-gradient variant is almost never
the autograd: it is the selection rule, the advantage definition, the clipping and the
reward transform. Those are arithmetic, and in this repository arithmetic that cannot be
tested is where the defects lived -- an inert config key, a component computed and
discarded, a `//` on a None. Everything below is therefore plain numpy with a torch-free
contract, and `rl_min/algos/*.py` wraps these in one-line autograd losses.

The same split already paid off once: `collect_unique` in `scripts/run_stage8_cycle.py`
was factored out of the torch sampler for exactly this reason, and its 7 tests run on a
laptop with no GPU, no agent checkpoint and no torch at all.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ reward transforms


def standardize(r, eps=1e-8):
    """(r - mean) / std. The group-relative advantage of GRPO (Shao et al. 2024, eq. 3).

    PREFER `center` (the Dr. GRPO form) FOR THIS PROJECT. The division by the group
    standard deviation is actively harmful when a reward component is pinned at its floor,
    which is exactly the v3 run's situation: r_sel sat at 0.091-0.144 for 1000 steps. A
    floored component has near-zero within-batch variance, so dividing by `std` inflates
    what is left -- floating-point noise -- into large advantages, and the policy chases
    it. Liu, Chen, Li, Qi, Pang, Du, Lee & Lin, "Understanding R1-Zero-Like Training: A
    Critical Perspective" (arXiv:2503.20783), remove both the std division and the length
    normalisation; the molecular GRPO implementations follow them.

    Kept because it is the published GRPO objective and the comparison is worth running.
    Returns zeros for a degenerate batch rather than dividing by ~0.
    """
    r = np.asarray(r, float)
    sd = r.std()
    if sd < eps or not np.isfinite(sd):
        return np.zeros_like(r)
    return (r - r.mean()) / sd


def center(r):
    """r - mean(r). The Dr. GRPO advantage, and the recommended default here.

    Liu et al. (arXiv:2503.20783) remove GRPO's division by the group standard deviation
    and its trajectory-length normalisation, leaving

        A_i = R_i - mean(R)

    which is what the molecular GRPO implementations use (no std division, no length
    normalisation, and no PPO clipping when a single gradient step is taken per batch).
    See `standardize` for why the std division specifically misfires on a floored reward
    component -- the exact situation r_sel was in for 1000 steps.

    The leave-one-out form below is unbiased; this one is not, strictly, because r_i
    appears in its own baseline. At batch 128 that bias is O(1/n), and it is what
    REINFORCE with a mean baseline does in practice.
    """
    r = np.asarray(r, float)
    return r - r.mean()


def leave_one_out_baseline(r):
    """r_i - mean(r_{j != i}). Unbiased, and free: RLOO (Kool et al. 2019; Ahmadian et al.
    2024, "Back to Basics: Revisiting REINFORCE-style optimization"), which reports that
    this beats PPO for LLM alignment at a fraction of the complexity.

    Relevant here because our setting is the same shape as theirs: a sequence policy, a
    scalar terminal reward, an expensive oracle and no per-token credit to assign.
    """
    r = np.asarray(r, float)
    n = len(r)
    if n < 2:
        return np.zeros_like(r)
    total = r.sum()
    return r - (total - r) / (n - 1)


def rank_transform(r):
    """Map rewards to evenly spaced values in [0, 1] by rank.

    Scale-free, so it is immune to the confound documented in `rl_min/dap.py`: `sigma` and
    the reward MULTIPLY in the augmented likelihood, so a reward that only ever reaches
    0.4 makes a nominal sigma of 120 behave like 48, and no sigma sweep is interpretable
    until the reward's attainable range is fixed. Ranking fixes the range by construction.
    """
    from scipy.stats import rankdata
    r = np.asarray(r, float)
    if len(r) < 2:
        return np.zeros_like(r)
    return (rankdata(r) - 1) / (len(r) - 1)


# ---------------------------------------------------------------------- selection rules


def topk_mask(scores, frac=0.5, min_keep=1):
    """Boolean mask of the top `frac` of a batch by score.

    THE core of Augmented Hill-Climb (Thomas, O'Boyle, Bender & de Graaf, "Augmented
    Hill-Climb increases reinforcement learning efficiency for language-based de novo
    molecule generation", J Cheminform 2022). Standard REINVENT regresses the agent onto
    the augmented likelihood for EVERY sampled molecule, including the ones that scored
    zero; AHC keeps only the best fraction and updates on those. The reported effect is a
    large gain in sample efficiency, which is the binding constraint whenever the reward
    involves an expensive oracle -- here a D-MPNN ensemble forward pass plus 7 ADMET
    models per molecule.

    Ties are broken by original order, so the rule is deterministic given the scores.
    """
    s = np.asarray(scores, float)
    n = len(s)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = int(max(min_keep, min(n, round(frac * n))))
    # argsort on -s is stable, so equal scores keep input order
    keep = np.argsort(-s, kind="stable")[:k]
    m = np.zeros(n, dtype=bool)
    m[keep] = True
    return m


def ppo_clip_weights(ratio, adv, clip=0.2):
    """The PPO-clipped surrogate's per-sample weight (Schulman et al. 2017).

    Returns `min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv)` -- the objective to
    MAXIMISE, elementwise. Written here so the clipping behaviour can be tested directly:
    the asymmetry (a positive advantage is clipped above, a negative one below) is the part
    implementations get wrong, and it is invisible in a training curve.
    """
    ratio = np.asarray(ratio, float)
    adv = np.asarray(adv, float)
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1 - clip, 1 + clip) * adv
    return np.minimum(unclipped, clipped)


def clip_fraction(ratio, clip=0.2):
    """Fraction of samples whose ratio left the trust region. Log it every step.

    A clip fraction near 0 means the updates are too small to matter; near 1 means the
    policy has already run away from the sampling distribution and the surrogate is no
    longer a lower bound on anything. Either way the number tells you before the molecules
    do.
    """
    r = np.asarray(ratio, float)
    return float(np.mean((r < 1 - clip) | (r > 1 + clip))) if len(r) else 0.0


def approx_kl(logp_new, logp_old):
    """The k3 estimator of KL(old || new): E[exp(d) - 1 - d], d = logp_new - logp_old.

    Schulman's unbiased, always-non-negative estimator. The naive `mean(logp_old -
    logp_new)` can come out negative on a finite sample, which makes an early-stopping
    rule keyed on it fire at the wrong time.
    """
    d = np.asarray(logp_new, float) - np.asarray(logp_old, float)
    return float(np.mean(np.exp(d) - 1 - d))


# ----------------------------------------------------- uncertainty-penalised reward


def pessimistic_reward(mu, sigma, kappa=1.0, lo=None, hi=None):
    """mu - kappa * sigma: optimise the LOWER confidence bound, not the mean.

    THE ANSWER TO THIS PROJECT'S CENTRAL FAILURE. The v3 run's generator drove the deep
    ensemble to claim single-digit nanomolar potency while an independently trained
    Tanimoto GP put the same molecules 1.8 log units (~60x) weaker -- on every single one
    of the 89 candidates. A cross-architecture audit gate caught it afterwards and rejected
    100% of the pool. The gate was right, and a gate is the wrong instrument: it discards
    work already done instead of shaping the policy away from doing it.

    The offline-RL literature formalises precisely this. MOPO (Yu et al., "MOPO:
    Model-based Offline Policy Optimization", NeurIPS 2020) and MOReL (Kidambi et al.,
    NeurIPS 2020) construct a PESSIMISTIC MDP by subtracting an uncertainty penalty from
    the learned model's reward, and prove the resulting policy's true value is bounded
    below by what it achieves there. CQL (Kumar et al., NeurIPS 2020) is the value-based
    sibling. The point common to all three: an agent optimising a learned model will find
    that model's errors, so the fix is to make the model's own uncertainty cost the agent
    something DURING training.

    Here `sigma` should be the EPISTEMIC component -- model disagreement, not measurement
    noise -- because the former is what the generator can exploit and the latter is a
    property of the assay. `tnk1.zoo.critic.StackedCritic` makes this exact: its
    `sigma_epi` is the weighted spread of an RF, a GP and a D-MPNN about their stacked
    prediction, i.e. the same cross-architecture disagreement the audit gate measured, so
    using it as the critic turns the post-hoc gate into a training-time penalty.
    """
    mu = np.asarray(mu, float)
    sigma = np.maximum(np.asarray(sigma, float), 0.0)
    out = mu - kappa * sigma
    if lo is not None or hi is not None:
        out = np.clip(out, -np.inf if lo is None else lo, np.inf if hi is None else hi)
    return out


def disagreement_penalty(mu_a, mu_b, tol=1.0, scale=1.0, mode="hinge"):
    """Penalty on |mu_a - mu_b|, the gap between two independent model classes.

    `mode="hinge"`  -- zero inside `tol`, then linear. Reproduces the audit gate's
                       tolerance as a soft shoulder, so molecules the gate would have
                       accepted are untouched and the ones it would have killed are
                       discouraged with a gradient instead of deleted without one.
    `mode="linear"` -- proportional everywhere.
    `mode="gate"`   -- the old behaviour, 0/1, kept so the two can be compared rather
                       than swapped on faith.

    Returns a penalty in [0, inf); callers multiply by a weight or subtract it.
    """
    d = np.abs(np.asarray(mu_a, float) - np.asarray(mu_b, float))
    if mode == "gate":
        return (d > tol).astype(float)
    if mode == "linear":
        return scale * d
    if mode == "hinge":
        return scale * np.maximum(d - tol, 0.0)
    raise ValueError(f"unknown disagreement mode {mode!r}")


def weighted_geometric_mean(components, weights, floor=1e-8):
    """prod_i s_i^(w_i / sum w). The aggregator fix for a floored reward component.

    THE ARGUMENT, WHICH IS ABOUT GRADIENTS RATHER THAN TASTE.

    Under a weighted ARITHMETIC mean the partial derivative with respect to each component
    is a constant:

        S  = sum_i w_i_hat * s_i          =>   dS/ds_i = w_i_hat

    so moving selectivity from 0.09 to 0.19 is worth exactly as much as moving safety from
    0.89 to 0.99, and the optimiser rationally spends its capacity where improvement is
    cheap. That is the v3 run in one line: every component driven near its ceiling except
    the one that was hard to move, which stayed at 0.091-0.144 for 1000 steps.

    Under a weighted GEOMETRIC mean it is the ELASTICITY that is constant:

        S  = prod_i s_i^w_i_hat           =>   dlnS/dlns_i = w_i_hat

    so FRACTIONAL improvement is what pays. Doubling 0.09 -> 0.18 multiplies the total by
    2^w_hat, while 0.89 -> 0.98 multiplies it by 1.10^w_hat. At a 0.2 weight share that is
    1.15 against 1.02 -- roughly a 7x larger reward gradient on the floored component, for
    no change but the aggregator. (REINVENT 4 exposes both natively as `geometric_mean`
    / `custom_product` and `arithmetic_mean` / `custom_sum`.)

    `floor` clamps each component before exponentiating, matching REINVENT's own 1e-8. BE
    AWARE WHAT THAT MEANS: a zero component contributes floor^w_hat, which at a 0.2 weight
    share is about 0.025 -- a severe penalty, not a veto. For a true veto multiply by a
    0/1 term (`tnk1.uq.applicability.BoundingBoxAD.reward_term`) or use
    `multiplicative_gate` below.
    """
    keys = [k for k in components if k in weights]
    if not keys:
        raise ValueError("no component has a weight")
    w = np.array([float(weights[k]) for k in keys])
    if w.sum() <= 0:
        raise ValueError("component weights sum to zero")
    w = w / w.sum()
    out = None
    for k, wi in zip(keys, w):
        v = np.clip(np.asarray(components[k], float), floor, None)
        term = v ** wi
        out = term if out is None else out * term
    return out


def aggregate_gradient_share(components, weights, floor=1e-8) -> dict:
    """Per-component share of the reward gradient, under both aggregators.

    A diagnostic, not a reward. Run it on a scored batch BEFORE a long RL run and read the
    `arithmetic` column: a component whose share is tiny is one the optimiser will abandon,
    and the v3 run would have shown exactly that for `sel` at step 1 rather than after 165
    GPU-minutes.
    """
    keys = [k for k in components if k in weights]
    w = np.array([float(weights[k]) for k in keys])
    w = w / max(w.sum(), 1e-12)
    vals = {k: np.clip(np.asarray(components[k], float), floor, None) for k in keys}
    geo = weighted_geometric_mean(components, weights, floor)
    out = {}
    for k, wi in zip(keys, w):
        v = vals[k]
        # arithmetic: dS/ds_i = w_i, so the share of a 1-unit move is just w_i
        # geometric:  dS/ds_i = w_i * S / s_i, which is larger where s_i is small
        out[k] = {
            "weight_share": float(wi),
            "mean_value": float(v.mean()),
            "arith_dS_ds": float(wi),
            "geom_dS_ds": float(np.mean(wi * geo / v)),
            "geom_over_arith": float(np.mean(geo / v)),
        }
    return out


def multiplicative_gate(components, floors):
    """Product of per-component pass fractions -- a soft AND over objectives.

    WHAT A WEIGHTED SUM DOES WRONG, stated precisely. Under `COMPONENTS_V2` the weights are
    pot 3.0, sel 2.5, hinge 2.0, ad 1.0, sa 1.5, safety 2.5, summing to 12.5. A molecule
    that abandons selectivity ENTIRELY therefore forfeits at most 2.5/12.5 = 20% of the
    reward and keeps 80% of it attainable. There is no structural pressure to lift a
    component that is hard to move -- which is what the v3 run shows: r_sel sat at
    0.091-0.144 while every other component was driven near its ceiling, and the total
    reward still reached 0.533-0.616.

    (The stronger claim -- that the sum actively PREFERS abandoning selectivity -- does not
    follow from these weights: 3.0 and 2.5 are close enough that trading sel for pot is
    not a net gain. The problem is the absence of pressure, not a perverse incentive.)

    A product is the standard fix -- the desirability-function aggregation of Derringer &
    Suich, "Simultaneous optimization of several response variables", J Quality Technology
    1980: any component at zero zeroes the whole reward, so no component can be abandoned
    at any price.

    `floors` maps a component name to the value below which it contributes ~0, which keeps
    the product from being dominated by whichever component happens to have the smallest
    attainable range.
    """
    out = None
    for k, v in components.items():
        v = np.asarray(v, float)
        f = floors.get(k, 0.0)
        term = np.clip((v - f) / max(1.0 - f, 1e-9), 0.0, 1.0)
        out = term if out is None else out * term
    return np.ones(1) if out is None else out


def lagrangian_reward(objective, constraints, multipliers):
    """objective - sum_j lambda_j * violation_j: constrained RL instead of a weighted sum.

    `constraints[j]` is a non-negative VIOLATION (0 when satisfied), `multipliers[j]` the
    current lambda. Paired with `update_multipliers` this is the standard primal-dual
    treatment of a constrained MDP (Altman's constrained MDPs; Achiam et al.,
    "Constrained Policy Optimization", ICML 2017, and the Lagrangian baselines in Ray
    et al.'s Safety Gym).

    Why it fits this project better than weights: the goal is not "potency and selectivity
    in a 3.0 : 2.5 ratio", it is "the most potent molecule SUBJECT TO selectivity above
    some margin". A fixed weight cannot express that, and when the constraint is far from
    satisfied a fixed weight is also exactly when it exerts least pull. A multiplier that
    grows while the constraint is violated does the opposite, which is what the r_sel
    collapse called for.
    """
    obj = np.asarray(objective, float)
    tot = np.zeros_like(obj)
    for v, lam in zip(constraints, multipliers):
        tot = tot + float(lam) * np.maximum(np.asarray(v, float), 0.0)
    return obj - tot


def update_multipliers(multipliers, violations, lr=0.05, lam_max=10.0):
    """Dual ascent: lambda <- clip(lambda + lr * mean_violation, 0, lam_max).

    The cap is not cosmetic. An unbounded multiplier on a constraint the policy CANNOT
    satisfy grows without limit and eventually swamps the objective, so the agent stops
    optimising potency altogether and still fails the constraint. If a multiplier sits at
    `lam_max`, the honest reading is that the constraint is infeasible for this
    generator -- which for selectivity would be a finding worth reporting rather than a
    knob to raise.
    """
    out = []
    for lam, v in zip(multipliers, violations):
        g = float(np.mean(np.maximum(np.asarray(v, float), 0.0)))
        out.append(float(np.clip(lam + lr * g, 0.0, lam_max)))
    return out


# ------------------------------------------------------------------ diversity metrics


def _tanimoto(A, B):
    ab = A @ B.T
    aa = (A * A).sum(1)[:, None]
    bb = (B * B).sum(1)[None, :]
    return ab / np.maximum(aa + bb - ab, 1e-9)


def intdiv(fps, sample=2000, seed=0):
    """Mean pairwise Tanimoto DISTANCE within a generated set. Higher is more diverse.

    The standard internal-diversity number for generated sets (Benhenda, "ChemGAN
    challenge for drug discovery", 2017). Subsampled above `sample` molecules because the
    pairwise matrix is O(n^2): a 200k pool would need ~160 GB.
    """
    X = np.asarray(fps, float)
    if len(X) < 2:
        return 0.0
    if sample and len(X) > sample:
        X = X[np.random.RandomState(seed).choice(len(X), sample, replace=False)]
    S = _tanimoto(X, X)
    n = len(X)
    iu = np.triu_indices(n, 1)
    return float(1.0 - S[iu].mean())


def n_circles(fps, threshold=0.65, sample=5000, seed=0):
    """#Circles: the size of a greedy maximal set whose members are pairwise LESS similar
    than `threshold` (Xie et al., "How Much Space Has Been Explored?", ICLR 2023).

    Better than mean pairwise distance for the question this project actually has, which is
    "how many genuinely distinct chemotypes did the generator find" -- a set can have high
    mean distance while containing only three clusters.
    """
    X = np.asarray(fps, float)
    if not len(X):
        return 0
    if sample and len(X) > sample:
        X = X[np.random.RandomState(seed).choice(len(X), sample, replace=False)]
    chosen: list[int] = []
    S = _tanimoto(X, X)
    for i in range(len(X)):
        if all(S[i, j] < threshold for j in chosen):
            chosen.append(i)
    return len(chosen)
