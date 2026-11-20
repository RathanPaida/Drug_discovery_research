"""The disagreement-gate calibration and the bounding-box applicability domain.

These two tests cover the two findings that most directly challenge the v3 run's
conclusions, and both are pure numpy.

1. `tnk1/rl/audit.py` -- the audit gate's threshold of 1.0 is a round number with no
   derivation anywhere in the repository. Langevin, Vuilleumier & Bianciotto (J Cheminform
   14:20, 2022) show that the optimise-then-rescore diagnostic this project implements can
   report exploitation that "can be explained by a disagreement between predictive models
   on the original data distribution". If the baseline held-out disagreement is already
   near 1.0, the 100% rejection rate is an artefact of the threshold, not evidence about
   the generator. `test_a_miscalibrated_gate_is_identified` constructs exactly that case.

2. `BoundingBoxAD` -- Langevin et al. (ACS Omega 8:23148, 2023) tested 16 applicability
   domains on a SMILES LSTM generator and found that similarity-to-training-set ADs, the
   family this project uses, FAIL, while a bounding box on physchem descriptors and
   COUNT-based Morgan fingerprints worked on every task. The count/binary distinction is
   load-bearing for this project specifically: the v3 generator's permeability collapsed
   because it drifted toward carboxylic acids, and a binary fingerprint cannot tell one
   carboxylic acid from five.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tnk1.rl import audit as A  # noqa: E402
from tnk1.uq.applicability import PHYSCHEM_AD_FEATURES, BoundingBoxAD  # noqa: E402


# ------------------------------------------------------- the disagreement null

def test_calibrate_null_summarises_both_populations():
    rng = np.random.default_rng(0)
    a_h, b_h = rng.normal(7, 1, 200), None
    b_h = a_h + rng.normal(0, 0.4, 200)
    a_p, b_p = rng.normal(6, 1, 500), None
    b_p = a_p + rng.normal(0, 0.9, 500)
    null = A.calibrate_null(a_h, b_h, a_p, b_p)
    assert null["n_heldout"] == 200 and null["n_prior"] == 500
    assert 0 < null["heldout"]["median"] < null["prior_samples"]["median"]
    assert null["recommended_threshold"] > null["heldout"]["median"]
    assert set(null["heldout"]["quantiles"]) == {"0.5", "0.9", "0.95", "0.99"}


def test_a_miscalibrated_gate_is_identified():
    """THE test. Baseline held-out disagreement around 1.2, gate tolerance 1.0.

    Here the generated molecules are drawn from the SAME disagreement distribution as the
    held-out measured compounds -- nothing is being exploited -- yet a tolerance of 1.0
    rejects most of them. The assessment must say the gate is mis-specified AND that the
    observed set is not anomalous, because that is the situation the v3 run may actually
    have been in.
    """
    rng = np.random.default_rng(1)
    held_a = rng.normal(7, 1, 300)
    held_b = held_a + rng.normal(0, 1.2, 300)          # baseline disagreement ~1.2 sd
    gen_a = rng.normal(8, 1, 89)
    gen_b = gen_a + rng.normal(0, 1.2, 89)             # SAME disagreement process
    null = A.calibrate_null(held_a, held_b)
    out = A.assess_gate(null, A.disagreement(gen_a, gen_b), current_tol=1.0)

    assert "MIS-SPECIFIED" in out["verdict_threshold"], out["verdict_threshold"]
    assert "NOT ANOMALOUS" in out["verdict_exploitation"], out["verdict_exploitation"]
    assert out["fraction_above_current_tol"] > 0.3, \
        "the round-number gate should reject a large fraction of honest molecules"
    assert out["fraction_above_null_q95"] < 0.25, \
        "against the calibrated threshold, the same molecules are unremarkable"


def test_genuine_exploitation_is_still_detected():
    """The calibration must not explain away a real signal. Generated molecules whose
    disagreement is 3x the baseline must come back ANOMALOUS."""
    rng = np.random.default_rng(2)
    held_a = rng.normal(7, 1, 300)
    held_b = held_a + rng.normal(0, 0.3, 300)          # tight baseline
    gen_a = rng.normal(8.2, 0.3, 89)
    gen_b = gen_a - 1.8 + rng.normal(0, 0.3, 89)       # a systematic 1.8-log gap
    null = A.calibrate_null(held_a, held_b)
    out = A.assess_gate(null, A.disagreement(gen_a, gen_b), current_tol=1.0)
    assert "DEFENSIBLE" in out["verdict_threshold"]
    assert "ANOMALOUS" in out["verdict_exploitation"]
    assert out["fraction_above_null_q95"] > 0.9
    assert out["median_ratio_to_null"] > 3


def test_tolerance_band_is_conditional_on_the_optimised_score():
    """Langevin et al.'s diagnostic is per-bin because inter-model disagreement is not
    constant across the prediction range -- it is largest where the optimiser pushes,
    at the top of the range where training data is thinnest."""
    rng = np.random.default_rng(3)
    a = rng.uniform(4, 9, 600)
    # disagreement grows with the predicted value, as it does in reality
    b = a + rng.normal(0, 0.1 + 0.25 * (a - 4), 600)
    bands = A.tolerance_band(a, b, n_bins=5, level=0.95)
    assert len(bands) == 5
    widths = [bd["control_hi"] - bd["control_lo"] for bd in bands]
    assert widths[-1] > widths[0] * 1.5, \
        "the band must widen where the model is least constrained"
    for bd in bands:
        assert bd["control_lo"] <= bd["control_median"] <= bd["control_hi"]


def test_in_tolerance_band_flags_only_the_outliers():
    rng = np.random.default_rng(4)
    a = rng.uniform(4, 9, 400)
    b = a + rng.normal(0, 0.3, 400)
    bands = A.tolerance_band(a, b, n_bins=4)
    inside = A.in_tolerance_band(a, b, bands)
    assert 0.85 < inside.mean() <= 1.0, "a 95% band should admit roughly 95%"
    # a molecule whose control score is far below its bin's band must be flagged
    assert not A.in_tolerance_band(np.array([8.0]), np.array([2.0]), bands)[0]


def test_in_tolerance_band_with_no_bands_is_all_false():
    assert not A.in_tolerance_band(np.array([7.0]), np.array([7.0]), []).any()


def test_max_pairwise_discrepancy_is_not_a_variance():
    """MOReL's USAD is the MAX pairwise discrepancy across differently-initialised models,
    not their variance. The distinction matters: a variance over 10 members of ONE
    architecture measures seed noise, which is what the deep ensemble's uninformative sigma
    reports; a max across model CLASSES measures the thing that tracks the failure, and
    needs no member to be well calibrated."""
    mu = np.array([[7.0, 7.0], [7.0, 9.0], [7.0, 8.0]])
    d = A.max_pairwise_discrepancy(mu)
    assert d == pytest.approx([0.0, 2.0])
    assert d[1] > mu[:, 1].std(), "the max exceeds the sd, by construction"
    assert A.max_pairwise_discrepancy(np.array([[1.0, 2.0]])) == pytest.approx([0.0, 0.0])


def test_combined_uncertainty_takes_the_max_so_neither_signal_can_be_silenced():
    mu = np.array([[7.0, 7.0], [7.0, 9.0]])
    sigma = np.array([1.5, 0.1])
    u = A.combined_uncertainty(mu, sigma)
    assert u == pytest.approx([1.5, 2.0]), \
        "each molecule must take whichever signal is larger"
    assert A.combined_uncertainty(mu) == pytest.approx([0.0, 2.0])


# ----------------------------------------------------- the bounding-box domain

def test_bounding_box_admits_training_data_and_rejects_outside():
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (300, 8))
    ad = BoundingBoxAD(percentile=1.0).fit(X)
    assert ad.in_domain(X).mean() > 0.80, "most training molecules must be in-domain"
    far = np.full((5, 8), 100.0)
    assert not ad.in_domain(far).any()
    assert (ad.violations(far) == 8).all(), "every feature should be violated"


def test_strict_box_is_widened_by_one_outlier_and_the_percentile_box_is_not():
    """Why `percentile=1` is the default here rather than the paper's strict min/max: at
    172 training molecules a single extreme compound widens a strict box enough to admit a
    region nothing supports."""
    X = np.zeros((100, 3))
    X[0] = 50.0                                        # one wild molecule
    strict = BoundingBoxAD(percentile=None).fit(X)
    robust = BoundingBoxAD(percentile=1.0).fit(X)
    probe = np.full((1, 3), 25.0)                      # halfway to the outlier
    assert strict.in_domain(probe)[0], "the strict box admits the unsupported region"
    assert not robust.in_domain(probe)[0], "the percentile box does not"


def test_violation_count_is_graded_not_boolean():
    """'out of domain on 1 of 215 features' and 'out on 60' are very different molecules,
    even though the reward term itself is 0/1."""
    X = np.random.default_rng(6).normal(0, 1, (200, 10))
    ad = BoundingBoxAD(percentile=1.0).fit(X)
    # start from the per-column MEDIAN, which is inside the box on every axis by
    # construction -- an arbitrary training row may already sit outside the 1st-99th
    # percentile box on one of ten features, which would make the count ambiguous.
    one_bad = np.median(X, axis=0)[None, :].copy()
    assert ad.violations(one_bad)[0] == 0, "the median must be in-domain on every axis"
    one_bad[0, 0] = 99.0
    many_bad = np.full((1, 10), 99.0)
    assert ad.violations(one_bad)[0] == 1
    assert ad.violations(many_bad)[0] == 10
    assert ad.in_domain(one_bad, max_violations=1)[0]
    assert not ad.in_domain(one_bad, max_violations=0)[0]


def test_nan_features_fail_closed():
    """A feature that cannot be judged counts as a violation, not a pass -- the direction
    this pipeline requires everywhere else after the fail-open audit."""
    X = np.random.default_rng(7).normal(0, 1, (100, 4))
    ad = BoundingBoxAD().fit(X)
    probe = np.zeros((1, 4))
    probe[0, 2] = np.nan
    assert ad.violations(probe)[0] >= 1
    assert not ad.in_domain(probe)[0]


def test_constant_training_feature_does_not_reject_everything():
    """A feature with no training variance would make the box a point on that axis and
    fail every generated molecule."""
    X = np.random.default_rng(8).normal(0, 1, (100, 3))
    X[:, 1] = 2.0
    ad = BoundingBoxAD().fit(X)
    probe = X[:5].copy()
    probe[:, 1] = 7.0                                  # far from the constant value
    assert ad.in_domain(probe).all(), \
        "a constant feature must be unconstrained, not infinitely strict"


def test_reward_term_is_the_multiplicative_zero_one_langevin_specifies():
    X = np.random.default_rng(9).normal(0, 1, (200, 5))
    ad = BoundingBoxAD(percentile=1.0).fit(X)
    probe = np.vstack([X[:3], np.full((2, 5), 50.0)])
    t = ad.reward_term(probe)
    assert set(np.unique(t)) <= {0.0, 1.0}
    assert t[-1] == 0.0 and t[-2] == 0.0


def test_worst_features_names_the_exploited_channel():
    """The diagnostic that turns 'rejected' into 'rejected because it has four more
    carboxylic acids than anything in training' -- which is what makes a targeted
    counter-term possible instead of a generic penalty."""
    rng = np.random.default_rng(10)
    X = rng.normal(0, 1, (300, 4))
    ad = BoundingBoxAD(percentile=1.0).fit(X)
    bad = rng.normal(0, 1, (50, 4))
    bad[:, 2] = 40.0                                   # one channel systematically abused
    worst = ad.worst_features(bad, names=["mw", "logp", "n_cooh", "tpsa"])
    assert worst[0]["feature"] == "n_cooh"
    assert worst[0]["frac"] == pytest.approx(1.0)


def test_physchem_feature_list_matches_the_paper():
    """The descriptor set is quoted verbatim in the docstring; this pins the count and a
    few members so a later edit cannot silently drop one."""
    assert len(PHYSCHEM_AD_FEATURES) == 17
    for f in ("NumHDonors", "NumHAcceptors", "RingCount", "NumRotatableBonds", "TPSA",
              "MolLogP", "MolMR", "MolWt", "FractionCSP3", "ScaffoldAtomRatio",
              "HeavyAtomCount", "TotalFormalCharge", "NumChiralCenters"):
        assert f in PHYSCHEM_AD_FEATURES


# ------------------------------------------- the aggregator fix, with its gradient claim

def test_weighted_geometric_mean_gives_the_floored_component_a_bigger_gradient():
    """The elasticity argument, checked numerically.

    Under a weighted ARITHMETIC mean dS/ds_i is a constant w_i, so a 0.1 move in a floored
    component is worth exactly as much as a 0.1 move in a saturated one -- and the
    optimiser spends its capacity where improvement is cheap. That is the v3 run: r_sel at
    0.091-0.144 for 1000 steps while everything else saturated.

    Under a weighted GEOMETRIC mean dlnS/dlns_i is constant, so FRACTIONAL improvement
    pays. This test measures the actual multiplier each aggregator gives for the same
    relative improvement on a floored component versus a saturated one.
    """
    from rl_min.algos import common as C
    w = {"sel": 2.5, "safety": 2.5}
    floored = {"sel": np.array([0.09]), "safety": np.array([0.89])}

    # double the floored component vs. a same-fractional move on the saturated one
    sel_doubled = {"sel": np.array([0.18]), "safety": np.array([0.89])}
    safety_up = {"sel": np.array([0.09]), "safety": np.array([0.98])}

    arith = lambda c: sum(w[k] * c[k][0] for k in w) / sum(w.values())
    geo = lambda c: C.weighted_geometric_mean(c, w)[0]

    a0, g0 = arith(floored), geo(floored)
    # the comparison the report makes is on the EXCESS gain -- how much each move
    # multiplies the total by, above 1. At an 0.2 weight share the paper's figures are
    # 1.15 vs 1.02, i.e. an excess of 0.15 against 0.02, "roughly a 7x larger reward
    # gradient on the floored component". With two equal weights the share is 0.5 and the
    # numbers are 2^0.5 = 1.414 against 1.10^0.5 = 1.049.
    exc = lambda f, c: f(c) / f(floored) - 1.0
    arith_sel, arith_saf = exc(arith, sel_doubled), exc(arith, safety_up)
    geo_sel, geo_saf = exc(geo, sel_doubled), exc(geo, safety_up)

    # under the ARITHMETIC mean the saturated move is worth at least as much, because
    # dS/ds_i is a constant and safety simply moves further in absolute terms
    assert arith_saf >= arith_sel * 0.95, (
        f"arithmetic: sel +{arith_sel:.4f} vs safety +{arith_saf:.4f}")
    # under the GEOMETRIC mean doubling the floored component dominates by several times
    assert geo_sel / geo_saf > 3.0, (
        f"geometric: doubling sel gave +{geo_sel:.4f}, saturating safety +{geo_saf:.4f} "
        f"(ratio {geo_sel / geo_saf:.1f}x) -- the floored component should dominate")
    # and the SWITCH of aggregator is what flips the ordering
    assert (arith_sel / arith_saf) < 1.0 < (geo_sel / geo_saf), \
        "the aggregator change must reverse which component is worth improving"


def test_weighted_geometric_mean_is_a_severe_but_finite_penalty_at_zero():
    """REINVENT clamps at 1e-8, so a zero component is not a veto. Worth pinning, because
    the natural assumption is that a product vetoes and it does not."""
    from rl_min.algos import common as C
    w = {"a": 1.0, "b": 4.0}
    v = C.weighted_geometric_mean({"a": np.array([0.0]), "b": np.array([1.0])}, w)[0]
    assert 0.0 < v < 0.1, f"expected a severe but non-zero value, got {v}"


def test_aggregate_gradient_share_exposes_the_component_that_will_be_abandoned():
    """The pre-run diagnostic: this would have shown `sel` as the loser at step 1 rather
    than after 165 GPU-minutes."""
    from rl_min.algos import common as C
    comps = {"pot": np.full(50, 0.77), "sel": np.full(50, 0.11),
             "safety": np.full(50, 0.90)}
    w = {"pot": 3.0, "sel": 2.5, "safety": 2.5}
    sh = C.aggregate_gradient_share(comps, w)
    assert set(sh) == set(w)
    # the geometric aggregator's gradient on the floored component is much larger
    assert sh["sel"]["geom_dS_ds"] > sh["safety"]["geom_dS_ds"] * 3
    # while the arithmetic one is flat in the component's value
    assert sh["sel"]["arith_dS_ds"] == pytest.approx(sh["safety"]["arith_dS_ds"])
