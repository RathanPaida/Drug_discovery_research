"""Cycle-02 reward: the three-tier design (absurdity gate / shaped terms / TPP selection).

The thing these tests mainly defend against is the obvious-looking fix: promoting the TPP
thresholds straight into `hard_gate`. That change reads as "enforce the spec" and is in
fact a reward collapse, because a gate deletes molecules instead of shaping them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tnk1.rl import reward as RW


# ---------------------------------------------------------------- Tier A: hard gate
def test_new_gate_criteria_are_off_unless_configured():
    """Cycle-01 call sites pass positionally and must be bit-identical."""
    g = RW.hard_gate([True], [0], [400.0], [0.1], 600.0, p_herg=[0.80], p_herg_max=0.85)
    assert g[0] == 1.0                      # SA/rot/QED absent => not enforced
    assert RW.hard_gate([True], [0], [400.0], [0.1], 600.0)[0] == 1.0


@pytest.mark.parametrize("kw,expected", [
    ({"sa": [6.0], "sa_max": 5.5}, 0.0),
    ({"sa": [5.0], "sa_max": 5.5}, 1.0),
    ({"n_rot": [12], "rot_max": 10}, 0.0),
    ({"n_rot": [9], "rot_max": 10}, 1.0),
    ({"qed": [0.10], "qed_min": 0.20}, 0.0),
    ({"qed": [0.50], "qed_min": 0.20}, 1.0),
])
def test_each_tier_a_criterion_bites_when_configured(kw, expected):
    assert RW.hard_gate([True], [0], [400.0], [0.1], 600.0, **kw)[0] == expected


def test_ames_ceiling_is_configurable_and_defaults_to_the_legacy_value():
    assert RW.hard_gate([True], [0], [400.0], [0.45], 600.0)[0] == 1.0          # legacy 0.5
    assert RW.hard_gate([True], [0], [400.0], [0.45], 600.0, p_ames_max=0.40)[0] == 0.0


def test_tpp_thresholds_as_a_hard_gate_would_starve_the_policy():
    """The regression this whole design exists to prevent.

    Applying the TPP jointly as a hard gate passes 5 of the 40 BEST compounds the pipeline
    ever produced. On raw generator output survival is far lower, so a DAP batch would hold
    almost no non-zero rewards and the policy would have nothing to learn from.
    """
    rng = np.random.default_rng(0)
    n = 4000
    # A population deliberately easier than the real generator's: centred ON each TPP
    # threshold rather than beyond it.
    sa = rng.normal(4.5, 1.0, n)
    herg = rng.beta(2, 5, n)          # mean 0.29, near the Cycle-01 contract's 0.387
    qed = rng.normal(0.45, 0.15, n)
    rot = rng.poisson(6.5, n)

    tier_a = RW.hard_gate(np.ones(n), np.zeros(n), np.full(n, 400.0), np.full(n, 0.1),
                          600.0, p_herg=herg, p_herg_max=0.50,
                          sa=sa, sa_max=5.5, n_rot=rot, rot_max=10,
                          qed=qed, qed_min=0.20)
    tpp_gate = RW.hard_gate(np.ones(n), np.zeros(n), np.full(n, 400.0), np.full(n, 0.1),
                            600.0, p_herg=herg, p_herg_max=RW.TPP["herg_max"],
                            sa=sa, sa_max=RW.TPP["sa_max"],
                            n_rot=rot, rot_max=RW.TPP["rot_max"],
                            qed=qed, qed_min=RW.TPP["qed_min"])
    assert tier_a.mean() > 0.50, "Tier A must leave most of the batch alive to learn from"
    assert tpp_gate.mean() < 0.10, "TPP-as-gate should be shown to collapse the batch"
    assert tier_a.mean() > 6 * tpp_gate.mean()


# ---------------------------------------------------------------- Tier B: shaped terms
def test_shaped_safety_discriminates_where_the_linear_product_merely_discounts():
    """At the contract's mean p(hERG)=0.387 the old term returns a mild 0.6x factor."""
    ames = cyp = np.array([0.2, 0.2])
    herg = np.array([0.10, 0.40])          # one passing the TPP, one clearly failing it
    old = RW.r_safety(herg, ames, cyp)
    new = RW.r_safety_tpp(herg, ames, cyp)
    old_ratio, new_ratio = old[0] / old[1], new[0] / new[1]
    assert old_ratio < 1.6, "the linear product barely separates them"
    assert new_ratio > 3.0, "the shaped term must separate them sharply"
    assert new_ratio > 2 * old_ratio


def test_shaped_safety_is_centred_on_the_tpp_threshold():
    """p(hERG) exactly at the ceiling should sit mid-sigmoid, not near 1.0."""
    at = RW.r_safety_tpp([0.20], [0.0], [0.0], herg_c=0.20)
    # 0.5 on the hERG factor itself, lifted by the clean AMES/CYP terms it is mixed with.
    assert 0.5 < at[0] < 0.9
    assert RW.r_safety_tpp([0.05], [0.0], [0.0])[0] > at[0]
    assert RW.r_safety_tpp([0.50], [0.0], [0.0])[0] < 0.5 * at[0]


def test_r_hinge_penalises_without_deleting():
    r = RW.r_hinge([True, False])
    assert r[0] == 1.0 and r[1] == 0.1
    # Through the geometric mean a hinge-less molecule must stay rankable, not be zeroed.
    w = {"pot": 3.0, "sel": 2.5, "hinge": 2.0, "ad": 1.0, "sa": 1.5, "safety": 2.5}
    comps = {c: np.full(2, 0.6) for c in RW.COMPONENTS_V2}
    comps["hinge"] = r
    out = RW.aggregate(comps, w, np.ones(2), components=RW.COMPONENTS_V2)
    assert out[1] > 0.05, "a hinge-less molecule must still carry gradient"
    # The realised penalty is floor**(w/sum w) = 0.1**(2.0/12.5) = 0.69, NOT the floor
    # itself -- see r_hinge's docstring. Pin that so nobody reads 0.1 as a 10x penalty.
    assert out[1] < 0.75 * out[0]
    assert np.isclose(out[1] / out[0], 0.1 ** (2.0 / 12.5), rtol=1e-6)


def test_calibrating_sel_centre_rescues_the_flat_term():
    """The Cycle-01 failure, reproduced: sel_c=1.2 above a population maxing at 0.25."""
    rng = np.random.default_rng(1)
    mu_sel = rng.normal(-1.24, 0.5, 5000).clip(max=0.25)
    sig = np.full_like(mu_sel, 0.2)
    flat = RW.r_sel(mu_sel, sig, c=1.2)
    tuned = RW.r_sel(mu_sel, sig, c=RW.calibrate_centre(mu_sel, 0.30))
    assert flat.mean() < 0.05 and tuned.mean() > 0.30
    assert tuned.std() > 8 * flat.std(), "the whole point is restored gradient"


# ---------------------------------------------------------------- Tier C: selection
def test_tpp_select_reports_skipped_criteria_instead_of_passing_them():
    """Fail-closed reporting: a missing column must never read as a criterion met."""
    df = pd.DataFrame({"tox_herg_prob": [0.1, 0.5], "sa_score": [3.0, 3.0]})
    mask, rep = RW.tpp_select(df)
    assert mask.tolist() == [True, False]
    assert set(rep["skipped"]) == {"ames_max", "qed_min", "rot_max", "mw_max"}
    assert rep["n_pass"] == 1 and rep["n_total"] == 2


def test_tpp_select_applies_min_and_max_criteria_in_the_right_direction():
    df = pd.DataFrame({"qed_score": [0.10, 0.90], "sa_score": [3.0, 3.0]})
    mask, _ = RW.tpp_select(df)
    assert mask.tolist() == [False, True]


# ---------------------------------------------------------------- veto
def test_veto_zeroes_a_collapsed_component_which_the_geometric_mean_alone_does_not():
    w = {"pot": 3.0, "sel": 2.5, "hinge": 2.0, "ad": 1.0, "sa": 1.5, "safety": 2.5}
    comps = {c: np.full(2, 0.8) for c in RW.COMPONENTS_V2}
    comps["safety"] = np.array([0.8, 1e-4])
    no_veto = RW.aggregate(comps, w, np.ones(2), None, components=RW.COMPONENTS_V2)
    vetoed = RW.aggregate(comps, w, np.ones(2), {"safety": 0.10},
                          components=RW.COMPONENTS_V2)
    assert no_veto[1] > 0.1, "documented hazard: a failed component only scales R"
    assert vetoed[1] == 0.0


def test_v2_component_set_drops_the_noise_floor_terms_and_keeps_ad():
    assert "adme" not in RW.COMPONENTS_V2 and "qed" not in RW.COMPONENTS_V2
    assert "hinge" in RW.COMPONENTS_V2
    assert "ad" in RW.COMPONENTS_V2, "the only guard against exploiting a rho=0.264 critic"
