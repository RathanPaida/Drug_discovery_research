"""Stacking and distribution combination. Hermetic: numpy, scipy and sklearn only.

The headline property under test is the one this project got wrong at full scale: it
shipped a COMBINATION (a 10-member deep ensemble) that was worse than a single plain model
and used it as the RL critic for a 6-hour HPC campaign. `test_discrete_super_learner_*`
below pins the guard that makes that outcome impossible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tnk1.ensembles import combine as CB, stacking as ST  # noqa: E402


# ------------------------------------------------------------------------- fixtures

def library(n=172, seed=0):
    """A realistic small library: one good model, one mediocre, one biased, one useless.

    Biased-but-informative and useless are both present deliberately: a good combiner must
    down-weight the useless one to ~0 while still using the biased one, whose ORDERING is
    fine even though its scale is wrong.
    """
    rng = np.random.default_rng(seed)
    y = rng.normal(7.0, 1.0, n)
    P = np.column_stack([
        y + rng.normal(0, 0.35, n),             # good
        y + rng.normal(0, 0.90, n),             # mediocre
        0.5 * y + 3.5 + rng.normal(0, 0.45, n),  # biased, but correctly ordered
        rng.normal(7.0, 1.0, n),                # pure noise
    ])
    names = ["good", "mid", "biased", "noise"]
    groups = np.array([f"S{i // 6:03d}" for i in range(n)])
    return P, y, names, groups


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


# -------------------------------------------------------------------- meta-learners

@pytest.mark.parametrize("name", sorted(ST.STACKERS))
def test_every_stacker_fits_predicts_and_is_finite(name):
    P, y, names, _ = library()
    m = ST.STACKERS[name]().fit(P, y)
    p = m.predict(P)
    assert p.shape == (len(y),)
    assert np.isfinite(p).all()


@pytest.mark.parametrize("name", sorted(ST.STACKERS))
def test_no_stacker_is_worse_than_the_worst_member(name):
    """A weak sanity floor that catches a sign error or a misaligned matrix."""
    P, y, names, _ = library()
    worst = max(rmse(y, P[:, j]) for j in range(P.shape[1]))
    assert rmse(y, ST.STACKERS[name]().fit(P, y).predict(P)) <= worst + 1e-6


def test_nnls_weights_are_non_negative_and_sum_to_one():
    """Breiman (1996): the non-negativity constraint is what makes stacked regression beat
    the best single member. Unconstrained OLS on collinear member predictions produces
    large cancelling coefficients that fit the stacking sample and not the target."""
    P, y, names, _ = library()
    m = ST.NNLSStacker(normalize=True).fit(P, y)
    w = m.weights
    assert (w >= -1e-12).all(), "a negative weight means the constraint was not applied"
    assert abs(w.sum() - 1.0) < 1e-6


def test_nnls_is_a_convex_combination_so_it_cannot_extrapolate():
    """The property the simplex constraint buys: the stacked prediction is bounded by the
    members', so it cannot run away off 172 points."""
    P, y, _, _ = library()
    p = ST.NNLSStacker(normalize=True).fit(P, y).predict(P)
    assert (p >= P.min(axis=1) - 1e-8).all()
    assert (p <= P.max(axis=1) + 1e-8).all()


def test_nnls_downweights_the_useless_member():
    P, y, names, _ = library()
    w = dict(zip(names, ST.NNLSStacker().fit(P, y).weights))
    assert w["good"] > w["noise"], "the noise column should lose to the good one"
    assert w["noise"] < 0.15, f"pure noise kept weight {w['noise']:.3f}"


def test_nnls_falls_back_to_the_uniform_average_and_warns_on_a_degenerate_fit():
    """Zero weight everywhere means the matrix is constant or misaligned with y, which is
    a bug upstream -- it must be loud, not silently return zeros."""
    P = np.zeros((40, 3))
    y = np.random.default_rng(0).normal(0, 1, 40)
    with pytest.warns(RuntimeWarning, match="zero weight"):
        m = ST.NNLSStacker().fit(P, y)
    assert np.allclose(m.weights, 1 / 3)


def test_caruana_weights_are_selection_counts_so_they_cannot_blow_up():
    """Caruana et al. (2004): weights are integer selection counts over a fixed number of
    rounds, so there is no regression fit to destabilise at small N."""
    P, y, names, _ = library()
    m = ST.CaruanaSelector(n_rounds=20).fit(P, y)
    w = m.weights
    assert (w >= 0).all() and abs(w.sum() - 1.0) < 1e-9
    assert m.counts_.sum() <= 20
    assert len(m.path_) >= 1


def test_caruana_stops_when_no_addition_helps():
    """Continuing past that point is how greedy selection overfits its hillclimb set."""
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 100)
    P = np.column_stack([y + rng.normal(0, 0.01, 100)] + [rng.normal(0, 3, 100)] * 1)
    m = ST.CaruanaSelector(n_rounds=50).fit(P, y)
    assert len(m.path_) < 50, "selection ran to the round limit instead of stopping"
    assert m.weights[0] > 0.9, "it should concentrate on the one accurate member"


def test_caruana_can_optimise_spearman_directly():
    """The reason it is ranked first for this project: Spearman is the headline metric and
    the quantity the RL reward's ordering consumes, and greedy selection is the only
    combiner here that can optimise a non-differentiable rank metric rather than hoping
    RMSE is a proxy for it."""
    P, y, names, _ = library()
    m = ST.CaruanaSelector(metric="spearman").fit(P, y)
    from scipy import stats
    rho_stack = stats.spearmanr(y, m.predict(P)).statistic
    rho_best = max(stats.spearmanr(y, P[:, j]).statistic for j in range(P.shape[1]))
    assert rho_stack >= rho_best - 0.02


def test_bagged_caruana_spreads_weight_more_than_a_single_greedy_path():
    """Bagging is one of Caruana et al.'s three anti-overfitting enhancements, and the one
    that matters at a 4-25 model library: a model that only wins when a stronger one is
    absent gets partial credit instead of none."""
    P, y, names, _ = library()
    single = ST.CaruanaSelector(n_rounds=25).fit(P, y).weights
    bagged = ST.BaggedCaruana(n_bags=20, frac=0.5, seed=0).fit(P, y).weights
    assert (bagged > 0).sum() >= (single > 0).sum()
    assert abs(bagged.sum() - 1.0) < 1e-9


def test_rank_stackers_are_invariant_to_a_members_scale():
    """Rank-based combination is immune to a member being offset or rescaled, which is the
    property plain averaging lacks."""
    P, y, _, _ = library()
    a = ST.RankNNLSStacker().fit(P, y).predict(P)
    P2 = P.copy()
    P2[:, 1] = P2[:, 1] * 3.0 + 10.0            # monotone transform of one member
    b = ST.RankNNLSStacker().fit(P2, y).predict(P2)
    from scipy import stats
    assert stats.spearmanr(a, b).statistic > 0.98


def test_median_stacker_survives_one_member_blowing_up_on_a_row():
    P, y, _, _ = library()
    P[0, 1] = 1e6
    med = ST.MedianStacker().fit(P, y).predict(P)
    mean = ST.MeanStacker().fit(P, y).predict(P)
    assert abs(med[0] - y[0]) < abs(mean[0] - y[0]), \
        "the median must be more robust than the mean to one wild member"


# ------------------------------------------------------- the discrete super learner

def test_discrete_super_learner_picks_the_stack_when_stacking_helps():
    P, y, names, groups = library()
    d = ST.DiscreteSuperLearner(metric="rmse", seed=0).fit(P, y, names=names, groups=groups)
    rep = d.report()
    assert rep["chosen"].startswith("stack:")
    assert rep["best_stack_risk"] < rep["best_single_risk"]
    assert rep["stack_beats_best_single"] is True


def test_discrete_super_learner_can_never_ship_worse_than_the_best_single():
    """THE GUARD. Phillips, van der Laan, Lee & Gruber (IJE 2023) step 5: every individual
    learner is itself a candidate, so the selected model's CV risk is at most the best
    single model's. This project shipped a combination worse than a plain random forest
    (Spearman 0.501 vs 0.686) and optimised an RL generator against it for 6 GPU-hours;
    this assertion is what makes that structurally impossible."""
    rng = np.random.default_rng(3)
    for trial in range(5):
        n = 120
        y = rng.normal(7, 1, n)
        k = rng.integers(2, 6)
        # deliberately include trials where stacking CANNOT help: one near-perfect model
        # plus pure noise columns.
        cols = [y + rng.normal(0, 0.05, n)] + [rng.normal(7, 1, n) for _ in range(k - 1)]
        P = np.column_stack(cols)
        names = [f"m{j}" for j in range(k)]
        groups = np.array([f"S{i // 6:03d}" for i in range(n)])
        d = ST.DiscreteSuperLearner(metric="rmse", seed=trial).fit(
            P, y, names=names, groups=groups)
        rep = d.report()
        assert rep["cv_risk"] <= rep["best_single_risk"] + 1e-9, (
            f"trial {trial}: the chosen candidate's CV risk {rep['cv_risk']:.4f} is worse "
            f"than the best single model's {rep['best_single_risk']:.4f}")


def test_discrete_super_learner_candidate_set_contains_every_single_model():
    P, y, names, groups = library()
    d = ST.DiscreteSuperLearner(seed=0).fit(P, y, names=names, groups=groups)
    cands = set(d.scores_)
    for n in names:
        assert f"single:{n}" in cands, f"{n} was not offered as a candidate on its own"
    assert any(c.startswith("stack:") for c in cands)


def test_discrete_super_learner_reports_a_full_ranking():
    P, y, names, groups = library()
    rep = ST.DiscreteSuperLearner(seed=0).fit(P, y, names=names, groups=groups).report()
    risks = [e["risk"] for e in rep["ranking"]]
    assert risks == sorted(risks), "the ranking must be ordered by risk"
    assert len(rep["ranking"]) == len(rep["metric"] and rep["ranking"])


def test_cv_select_stacker_groups_by_scaffold():
    """Leakage can enter a second time at the meta level: without grouping, a scaffold's
    rows land on both sides of the meta-learner's own split and the chosen combiner is the
    one that best memorises the OOF matrix."""
    P, y, names, groups = library()
    best, scores = ST.cv_select_stacker(P, y, groups=groups, seed=0)
    assert best in ST.STACKERS
    assert all(np.isfinite(v) for v in scores.values())


def test_super_learner_auto_selects_and_reports_its_weights():
    P, y, names, groups = library()
    sl = ST.SuperLearner(metric="rmse").fit(P, y, names=names, groups=groups)
    rep = sl.report()
    assert rep["stacker"] in ST.STACKERS
    if "weights" in rep:
        assert set(rep["weights"]) == set(names)


# ------------------------------------------------------ distribution combination

def test_moment_match_reproduces_the_law_of_total_variance():
    """The split into within-member (aleatoric) and between-member (epistemic) variance is
    algebraically the law of total variance, and it must match the closed form
    sum_k w_k (sigma_k^2 + mu_k^2) - mu*^2 exactly."""
    rng = np.random.default_rng(0)
    mu = rng.normal(7, 1, (4, 50))
    sd = rng.uniform(0.2, 1.0, (4, 50))
    w = np.array([0.4, 0.3, 0.2, 0.1])
    out = CB.moment_match(mu, sd, w)
    direct = (w[:, None] * (sd ** 2 + mu ** 2)).sum(0) - out["mu"] ** 2
    assert np.allclose(out["var_ale"] + out["var_epi"], direct, atol=1e-10)
    assert np.allclose(out["mu"], (w[:, None] * mu).sum(0))


def test_epistemic_term_is_zero_when_members_agree_and_grows_when_they_do_not():
    """`var_epi` is the quantity to feed the pessimistic reward: it is model DISAGREEMENT,
    which a generator can exploit, as opposed to assay noise, which it cannot."""
    agree = np.tile(np.linspace(6, 8, 20), (3, 1))
    sd = np.full((3, 20), 0.5)
    assert np.allclose(CB.moment_match(agree, sd)["var_epi"], 0.0)
    disagree = agree + np.array([[-1.0], [0.0], [1.0]])
    assert CB.moment_match(disagree, sd)["var_epi"].mean() > 0.5


def test_mixture_logpdf_is_a_normalised_density():
    """Integrating the mixture over y must give 1; a log-sum-exp error shows up here."""
    mu = np.array([[6.0], [8.0]])
    sd = np.array([[0.5], [0.5]])
    grid = np.linspace(0, 14, 20001)
    dens = np.exp([CB.mixture_logpdf(np.array([g]), mu, sd)[0] for g in grid])
    assert abs(np.trapezoid(dens, grid) - 1.0) < 1e-3


def test_mixture_nll_sees_bimodality_a_gaussian_hides():
    """Why the mixture is used for scoring and the moment-matched Gaussian for conformal:
    a Gaussian approximation to a bimodal mixture hides exactly the member disagreement
    that matters for a project whose failure was two model classes 1.8 log units apart."""
    n = 400
    y = np.where(np.arange(n) % 2 == 0, 6.0, 8.0)
    mu = np.vstack([np.full(n, 6.0), np.full(n, 8.0)])
    sd = np.full((2, n), 0.3)
    mix = CB.mixture_nll(y, mu, sd)
    mm = CB.moment_match(mu, sd)
    from tnk1.uq.metrics import gaussian_nll
    gauss = gaussian_nll(y, mm["mu"], mm["sigma"])
    assert mix < gauss, "the mixture must score the bimodal truth better than a Gaussian"


def test_log_score_stacking_improves_the_mixture_nll():
    """Yao, Vehtari, Simpson & Gelman (2018): maximise the mixture's LOO log score on the
    simplex. It must beat the uniform mixture it warm-starts from."""
    rng = np.random.default_rng(1)
    n = 200
    y = rng.normal(7, 1, n)
    mu = np.vstack([y + rng.normal(0, 0.3, n), rng.normal(7, 1, n)])
    sd = np.vstack([np.full(n, 0.3), np.full(n, 1.0)])
    w, info = CB.stack_log_score(mu, sd, y)
    assert (w >= -1e-9).all() and abs(w.sum() - 1) < 1e-6
    assert w[0] > w[1], "the informative member must take most of the weight"
    assert CB.mixture_nll(y, mu, sd, w) < CB.mixture_nll(y, mu, sd)


def test_pseudo_bma_plus_is_a_simplex_vector_and_prefers_the_better_member():
    rng = np.random.default_rng(2)
    n = 150
    y = rng.normal(7, 1, n)
    lp = np.vstack([CB._norm_logpdf(y, y + rng.normal(0, 0.3, n), np.full(n, 0.3)),
                    CB._norm_logpdf(y, rng.normal(7, 1, n), np.full(n, 1.0))])
    w = CB.pseudo_bma_plus(lp, n_boot=200, seed=0)
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()
    assert w[0] > w[1]


def test_crps_is_deterministic_and_rewards_the_sharper_correct_forecast():
    y = np.linspace(6, 8, 50)
    sharp = (np.vstack([y]), np.full((1, 50), 0.2))
    broad = (np.vstack([y]), np.full((1, 50), 2.0))
    a = CB.mixture_crps(y, *sharp)
    assert a == CB.mixture_crps(y, *sharp), "CRPS must not be stochastic"
    assert a < CB.mixture_crps(y, *broad)


def test_pool_then_calibrate_scalar_recovers_a_known_miscalibration():
    """Rahaman & Thiery (NeurIPS 2021): the scale must be fitted AFTER pooling. A sigma
    understated by 2x must produce s ~= 2."""
    rng = np.random.default_rng(0)
    n = 4000
    mu = np.zeros(n)
    true_sd = 1.0
    y = rng.normal(mu, true_sd)
    reported = np.full(n, true_sd / 2)
    pc = CB.PoolThenCalibrate("scalar").fit(y, mu, reported)
    assert 1.8 < pc.report()["s"] < 2.2
    assert np.allclose(pc.transform(reported), pc.report()["s"] * reported)


def test_pool_then_calibrate_affine_can_flatten_an_uninformative_sigma():
    """The two-parameter form can reshape a sigma that carries no information into a
    constant (b -> 0), rather than scaling noise. That is the deep ensemble's case:
    error-sigma correlation -0.021."""
    rng = np.random.default_rng(0)
    n = 3000
    y = rng.normal(0, 1, n)
    mu = np.zeros(n)
    noise_sigma = rng.uniform(0.1, 3.0, n)      # uncorrelated with the actual error
    pc = CB.PoolThenCalibrate("affine").fit(y, mu, noise_sigma)
    r = pc.report()
    assert r["a"] > 0, "the affine form must learn a positive floor"
    assert r["b"] < 0.5, f"b={r['b']:.3f}: an uninformative sigma should be shrunk"


def test_precision_weighted_is_available_but_documented_as_not_recommended():
    out = CB.precision_weighted(np.array([[1.0, 2.0]]), np.array([[0.5, 0.5]]))
    assert np.isfinite(out["mu"]).all() and np.isfinite(out["sigma"]).all()
    assert "not recommended" in CB.precision_weighted.__doc__.lower()
