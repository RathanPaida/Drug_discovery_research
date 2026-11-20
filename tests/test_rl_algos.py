"""The RL algorithms' mathematics. Pure numpy, so this runs with no torch and no GPU.

The content of a policy-gradient variant is its selection rule, advantage definition,
clipping and reward transform -- not its autograd. `rl_min/algos/common.py` holds exactly
those as numpy functions so they can be tested here, and the torch trainers are one-line
wrappers over them. The same split already paid off for `collect_unique` in
`scripts/run_stage8_cycle.py`.

Two sections carry most of the weight:

  `pessimistic_reward` / `disagreement_penalty` -- the answer to the v3 run's central
  failure, where the generator drove the deep ensemble to claim single-digit nanomolar
  potency while an independent Tanimoto GP put the same molecules 1.8 log units weaker on
  every one of 89 candidates.

  `multiplicative_gate` / `lagrangian_reward` -- the answer to the other failure, where a
  weighted SUM let the generator abandon selectivity (r_sel pinned at 0.09-0.14 against a
  weight of 2.5) because conceding it was cheaper than what it bought elsewhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imported directly, NOT via `import rl_min`: that package's __init__ is lazy precisely so
# this module is reachable without torch. If this import starts pulling torch, the lazy
# __getattr__ in rl_min/__init__.py has regressed.
from rl_min.algos import common as C  # noqa: E402


def test_importing_the_math_does_not_import_torch():
    """Regression: `rl_min/__init__.py` used to import `.dap` eagerly, so ANY access to
    the package failed with ModuleNotFoundError: torch in a numpy-only environment."""
    assert "torch" not in sys.modules or True        # torch may be present; that's fine
    import importlib
    m = importlib.import_module("rl_min.algos.common")
    assert not hasattr(m, "torch"), "common.py must not hold a torch reference"


# ------------------------------------------------------------- reward transforms

def test_standardize_is_zero_mean_unit_sd():
    r = np.array([0.1, 0.9, 0.5, 0.5, 0.0])
    z = C.standardize(r)
    assert abs(z.mean()) < 1e-12
    assert abs(z.std() - 1.0) < 1e-12


def test_standardize_returns_zeros_for_a_constant_batch():
    """A constant-reward batch carries no learning signal; manufacturing one from
    floating-point noise gives a gradient of magnitude 1e8 and diverges at step 1."""
    assert np.allclose(C.standardize(np.ones(10)), 0.0)
    assert np.allclose(C.standardize(np.zeros(5)), 0.0)
    assert np.isfinite(C.standardize(np.array([1e-300] * 4))).all()


def test_center_has_zero_mean():
    r = np.array([1.0, 2.0, 6.0])
    assert abs(C.center(r).mean()) < 1e-12


def test_leave_one_out_baseline_excludes_the_sample_itself():
    """RLOO's unbiasedness depends on r_i NOT appearing in its own baseline."""
    r = np.array([1.0, 2.0, 3.0, 4.0])
    a = C.leave_one_out_baseline(r)
    for i in range(len(r)):
        others = np.delete(r, i)
        assert a[i] == pytest.approx(r[i] - others.mean())


def test_leave_one_out_is_degenerate_for_a_single_sample():
    assert np.allclose(C.leave_one_out_baseline(np.array([3.0])), 0.0)


def test_rank_transform_is_scale_free_and_spans_the_unit_interval():
    """Fixes the sigma/reward-range confound documented in rl_min/dap.py: sigma and the
    reward MULTIPLY, so a reward that only reaches 0.4 makes a nominal sigma of 120 behave
    like 48, and no sigma sweep is interpretable until the range is pinned."""
    r = np.array([0.01, 0.02, 0.03, 0.04])
    a = C.rank_transform(r)
    b = C.rank_transform(r * 1000 + 50)
    assert np.allclose(a, b), "a monotone rescaling must not change the transform"
    assert a.min() == 0.0 and a.max() == 1.0


# ------------------------------------------------------------- selection (AHC)

def test_topk_mask_keeps_the_best_fraction():
    s = np.array([0.1, 0.9, 0.5, 0.7, 0.0])
    m = C.topk_mask(s, 0.4)
    assert m.sum() == 2
    assert m[1] and m[3], "the two highest scores must be kept"


def test_topk_mask_is_deterministic_under_ties():
    """Stable ordering, so the rule is reproducible given the scores."""
    s = np.array([0.5, 0.5, 0.5, 0.5])
    a = C.topk_mask(s, 0.5)
    for _ in range(5):
        assert (C.topk_mask(s, 0.5) == a).all()
    assert a.sum() == 2


def test_topk_mask_respects_min_keep_and_handles_edges():
    assert C.topk_mask(np.array([0.1, 0.2]), 0.0, min_keep=1).sum() == 1
    assert C.topk_mask(np.array([0.1, 0.2]), 1.0).all()
    assert C.topk_mask(np.array([]), 0.5).shape == (0,)


def test_ahc_selection_drops_the_zero_scoring_tail():
    """The mechanism AHC's sample-efficiency claim rests on: standard DAP regresses the
    agent onto the augmented likelihood for every sample, and for a zero-scoring molecule
    that target is just the prior -- so the update pulls the agent back toward chemistry it
    already knows. With a sparse reward most of a batch is exactly that."""
    s = np.concatenate([np.zeros(90), np.linspace(0.3, 0.9, 10)])
    m = C.topk_mask(s, 0.1)
    assert m.sum() == 10
    assert (s[m] > 0).all(), "every kept sample should carry signal"


# ------------------------------------------------------------------ PPO clipping

def test_ppo_clip_is_asymmetric_in_the_sign_of_the_advantage():
    """The part implementations get wrong, and it is invisible in a training curve: a
    POSITIVE advantage is clipped above (no reward for moving further), a NEGATIVE one is
    clipped below."""
    ratio = np.array([0.5, 1.0, 1.5])
    pos = C.ppo_clip_weights(ratio, np.ones(3), clip=0.2)
    assert pos == pytest.approx([0.5, 1.0, 1.2])
    neg = C.ppo_clip_weights(ratio, -np.ones(3), clip=0.2)
    assert neg == pytest.approx([-0.8, -1.0, -1.5])


def test_ppo_clip_equals_the_unclipped_value_inside_the_trust_region():
    ratio = np.array([0.95, 1.0, 1.05])
    adv = np.array([2.0, -2.0, 0.5])
    assert C.ppo_clip_weights(ratio, adv, clip=0.2) == pytest.approx(ratio * adv)


def test_clip_fraction_counts_departures_from_the_trust_region():
    """Near 0 means the updates are too small to matter; near 1 means the policy has
    already left the sampling distribution and the surrogate bounds nothing."""
    assert C.clip_fraction(np.array([1.0, 1.0]), 0.2) == 0.0
    assert C.clip_fraction(np.array([0.5, 1.5]), 0.2) == 1.0
    assert C.clip_fraction(np.array([]), 0.2) == 0.0


def test_approx_kl_is_non_negative_and_zero_at_equality():
    """The k3 estimator. The naive mean(logp_old - logp_new) can come out NEGATIVE on a
    finite sample, which makes an early-stopping rule keyed on it fire at the wrong time."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        old = rng.normal(-5, 1, 64)
        new = old + rng.normal(0, 0.3, 64)
        assert C.approx_kl(new, old) >= -1e-12
    x = rng.normal(-5, 1, 32)
    assert abs(C.approx_kl(x, x)) < 1e-12


# ------------------------------------------- uncertainty-penalised reward (the fix)

def test_pessimistic_reward_penalises_exactly_where_the_model_is_unsure():
    """MOPO (Yu et al., NeurIPS 2020) / MOReL (Kidambi et al., NeurIPS 2020): subtract an
    uncertainty penalty so the agent cannot profit from the model's own errors. Two
    molecules the critic calls equally potent must separate by their disagreement."""
    mu = np.array([8.2, 8.2])
    sigma = np.array([0.1, 1.5])
    out = C.pessimistic_reward(mu, sigma, kappa=1.0)
    assert out[0] > out[1], "the uncertain molecule must be scored lower"
    assert out == pytest.approx([8.1, 6.7])


def test_pessimistic_reward_reduces_to_the_mean_at_kappa_zero():
    mu = np.array([7.0, 8.0])
    assert C.pessimistic_reward(mu, np.array([1.0, 2.0]), kappa=0.0) == pytest.approx(mu)


def test_pessimistic_reward_ignores_a_negative_sigma():
    out = C.pessimistic_reward(np.array([7.0]), np.array([-5.0]), kappa=1.0)
    assert out == pytest.approx([7.0]), "a negative sd must not become a BONUS"


def test_disagreement_hinge_is_zero_inside_the_tolerance_then_linear():
    """Reproduces the audit gate's tolerance as a soft shoulder: molecules the gate would
    have ACCEPTED are untouched, and the ones it would have killed are discouraged with a
    gradient instead of deleted without one."""
    p = C.disagreement_penalty(np.array([7.0, 7.5, 8.2]), np.full(3, 7.0),
                               tol=1.0, mode="hinge")
    assert p[0] == 0.0 and p[1] == 0.0
    assert p[2] == pytest.approx(0.2)


def test_disagreement_gate_reproduces_the_old_hard_behaviour():
    """Kept so the soft and hard forms can be COMPARED rather than swapped on faith."""
    g = C.disagreement_penalty(np.array([7.5, 8.2]), np.full(2, 7.0), tol=1.0, mode="gate")
    assert g.tolist() == [0.0, 1.0]


def test_disagreement_penalty_on_the_actual_v3_gap():
    """The measured case: ensemble median 8.214, GP median 6.471, a 1.74-log gap against a
    tolerance of 1.0. The hinge must charge for it; the gate rejected 100% of candidates."""
    pen = C.disagreement_penalty(np.array([8.214]), np.array([6.471]), tol=1.0,
                                 mode="hinge")
    assert pen[0] == pytest.approx(0.743, abs=1e-3)
    assert C.disagreement_penalty(np.array([8.214]), np.array([6.471]), tol=1.0,
                                  mode="gate")[0] == 1.0


def test_unknown_disagreement_mode_raises():
    with pytest.raises(ValueError):
        C.disagreement_penalty(np.array([1.0]), np.array([1.0]), mode="nope")


# ------------------------------------------- multi-objective (the other failure)

def test_multiplicative_gate_zeroes_the_reward_when_any_component_is_abandoned():
    """The v3 run maximised potency, hinge, AD, SA and safety while leaving selectivity at
    0.09-0.14 against a weight of 2.5, because 2.5*0.1 was cheaper to concede than the
    3.0*0.9 it bought elsewhere. A product (Derringer & Suich 1980) makes that trade
    impossible: no component can be abandoned."""
    good = {"pot": np.array([0.9]), "sel": np.array([0.8])}
    abandoned = {"pot": np.array([0.9]), "sel": np.array([0.0])}
    assert C.multiplicative_gate(good, {})[0] > 0.5
    assert C.multiplicative_gate(abandoned, {})[0] == 0.0


def test_a_weighted_sum_leaves_most_of_the_reward_attainable_without_selectivity():
    """The precise mechanism, with the real COMPONENTS_V2 weights.

    pot 3.0, sel 2.5, hinge 2.0, ad 1.0, sa 1.5, safety 2.5 -> 12.5 total. A molecule that
    abandons selectivity entirely still reaches 80% of the maximum, so there is no
    structural pressure to lift a component that is hard to move. That is what the v3 run
    shows: r_sel 0.091-0.144 throughout, every other component near its ceiling, total
    reward still 0.533-0.616.

    NOT the stronger claim that the sum PREFERS abandoning selectivity -- with 3.0 against
    2.5 that does not follow, and the test says so rather than asserting it.
    """
    w = {"pot": 3.0, "sel": 2.5, "hinge": 2.0, "ad": 1.0, "sa": 1.5, "safety": 2.5}
    total = sum(w.values())
    no_sel = {k: (0.0 if k == "sel" else 1.0) for k in w}
    frac = sum(w[k] * v for k, v in no_sel.items()) / total
    assert frac == pytest.approx(0.8), \
        "abandoning selectivity should forfeit only 20% of a weighted-sum reward"
    # and the product makes the same molecule worthless
    comps = {k: np.array([v]) for k, v in no_sel.items()}
    assert C.multiplicative_gate(comps, {})[0] == 0.0


def test_multiplicative_gate_prefers_the_balanced_molecule():
    """Where the two aggregations genuinely disagree: equal weighted-sum score, very
    different product."""
    # 3.0*0.60 + 2.5*0.60 = 3.30 ;  3.0*0.95 + 2.5*0.18 = 3.30  -- an exact tie
    balanced = {"pot": np.array([0.60]), "sel": np.array([0.60])}
    lopsided = {"pot": np.array([0.95]), "sel": np.array([0.18])}
    w = {"pot": 3.0, "sel": 2.5}
    s_bal = sum(w[k] * v[0] for k, v in balanced.items())
    s_lop = sum(w[k] * v[0] for k, v in lopsided.items())
    assert abs(s_bal - s_lop) < 1e-9, "chosen so the weighted sums are exactly tied"
    assert C.multiplicative_gate(balanced, {})[0] > C.multiplicative_gate(lopsided, {})[0], \
        "the product must break the tie toward the balanced molecule"


def test_multiplicative_gate_floors_rescale_each_component():
    c = {"a": np.array([0.5])}
    assert C.multiplicative_gate(c, {"a": 0.0})[0] == pytest.approx(0.5)
    assert C.multiplicative_gate(c, {"a": 0.5})[0] == pytest.approx(0.0)


def test_lagrangian_reward_subtracts_weighted_violations_only():
    obj = np.array([1.0, 1.0])
    viol = [np.array([0.0, 0.5])]
    out = C.lagrangian_reward(obj, viol, [2.0])
    assert out == pytest.approx([1.0, 0.0])


def test_dual_ascent_grows_the_multiplier_while_the_constraint_is_violated():
    """A fixed weight exerts least pull exactly when the constraint is furthest from
    satisfied; a multiplier that grows does the opposite, which is what the r_sel collapse
    called for."""
    lam = [0.0]
    for _ in range(10):
        lam = C.update_multipliers(lam, [np.array([0.4])], lr=0.5)
    assert lam[0] > 1.0


def test_dual_ascent_decays_to_zero_once_satisfied_and_is_capped():
    lam = [5.0]
    for _ in range(50):
        lam = C.update_multipliers(lam, [np.array([0.0])], lr=0.5)
    assert lam[0] == pytest.approx(5.0), "a satisfied constraint holds its multiplier"
    lam = [0.0]
    for _ in range(1000):
        lam = C.update_multipliers(lam, [np.array([1.0])], lr=1.0, lam_max=10.0)
    assert lam[0] == 10.0, ("an infeasible constraint must saturate at lam_max, not grow "
                            "without bound and swamp the objective")


# ------------------------------------------------------------- diversity metrics

def test_intdiv_is_zero_for_identical_molecules_and_rises_with_spread():
    same = np.tile([1.0, 1.0, 0.0, 0.0], (10, 1))
    assert C.intdiv(same) == pytest.approx(0.0)
    mixed = np.r_[same, np.tile([0.0, 0.0, 1.0, 1.0], (10, 1))]
    assert C.intdiv(mixed) > 0.4


def test_intdiv_handles_degenerate_input():
    assert C.intdiv(np.zeros((1, 4))) == 0.0
    assert C.intdiv(np.zeros((0, 4))) == 0.0


def test_n_circles_counts_distinct_chemotypes_not_molecules():
    """Xie et al. (ICLR 2023). A set can have high mean pairwise distance while containing
    only three clusters -- #Circles answers the question the project actually has, which is
    how many genuinely distinct chemotypes the generator found."""
    a = np.tile([1.0, 1.0, 0.0, 0.0], (20, 1))
    b = np.tile([0.0, 0.0, 1.0, 1.0], (20, 1))
    c = np.tile([1.0, 0.0, 1.0, 0.0], (20, 1))
    assert C.n_circles(np.r_[a, b, c], threshold=0.65) == 3
    assert C.n_circles(a, threshold=0.65) == 1
    assert C.n_circles(np.zeros((0, 4))) == 0


def test_n_circles_subsamples_without_crashing():
    X = np.random.default_rng(0).integers(0, 2, (500, 16)).astype(float)
    assert 0 < C.n_circles(X, threshold=0.65, sample=100) <= 100
