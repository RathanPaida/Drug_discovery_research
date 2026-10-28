import numpy as np, pandas as pd, pytest
from tnk1.data.activity import to_pactivity, prepare_records, aggregate_replicates
from tnk1.data import splits as sp
from tnk1.uq.decomposition import decompose, nig_summary
from tnk1.uq.calibration import fit_temperature, conformal_quantile, Calibrator
from tnk1.uq.applicability import ApplicabilityDomain
from tnk1.uq import metrics as M
from tnk1.selectivity.selectivity import selectivity_stats, member_margin, soft_margin
from tnk1.active_learning import acquisition as ACQ, batch as B, contract as K
from tnk1.rl import reward as R
from tnk1.models.weighting import sample_weights
from tnk1.models.kinase_features import klifs_encode, klifs_identity, composite_similarity
from tnk1.evaluation.stats import clopper_pearson, holm
from tnk1.baselines.models import KNNTanimoto, tanimoto_sim

rng = np.random.RandomState(0)


def test_pactivity_anchor_values():
    assert np.allclose(to_pactivity([1000, 100, 10]), [6, 7, 8])


def test_aggregate_replicates_and_censoring():
    d = pd.DataFrame({"inchikey_block1": ["A"] * 3 + ["B"] * 2 + ["C"], "uniprot_id": "Q13470", "standard_type": "IC50",
                      "smiles": "x", "scaffold": "s", "source": "t", "relation": ["="] * 5 + [">"],
                      "value_nM": [10, 12, 11, 10, 1000, 10000]})
    out = aggregate_replicates(prepare_records(d), 1.0, "drop").set_index("inchikey_block1")
    assert "B" not in out.index                       # replicate range 2.0 > 1.0 -> dropped
    assert out.loc["A", "n_rep"] == 3 and abs(out.loc["A", "pact"] - to_pactivity(11)) < 1e-9
    assert out.loc["C", "censored"] == 1 and abs(out.loc["C", "pact"] - 5.0) < 1e-9   # > 10 uM -> bound 5.0


def test_nested_scaffold_cv_has_no_scaffold_leakage():
    groups = np.array([f"s{i % 17}" for i in range(120)])
    cv = sp.nested_scaffold_cv(groups, 5, 5, seed=1)
    for f in cv:
        assert not set(groups[f["train"]]) & set(groups[f["test"]])
        for i in f["inner"]:
            assert not set(groups[i["train"]]) & set(groups[i["val"]])
    assert sorted(sum([f["test"] for f in cv], [])) == list(range(120))      # test folds partition the data
    assert (sp.balanced_group_folds(groups, 5, 1) == sp.balanced_group_folds(groups, 5, 1)).all()   # deterministic


def test_kinome_train_mask_removes_test_compounds_from_all_kinases():
    df = pd.DataFrame({"inchikey_block1": ["A", "B", "A"], "uniprot_id": ["SRC", "SRC", "ABL1"]})
    assert sp.kinome_train_mask(df, ["A"]).tolist() == [False, True, False]
    assert sp.kinome_train_mask(df, [], ["ABL1"]).tolist() == [True, True, False]


def test_variance_decomposition_matches_total_variance_law():
    mu_m = rng.randn(10, 50); lv_m = rng.randn(10, 50) * 0.3 - 1
    d = decompose(mu_m, lv_m)
    assert np.allclose(d["sigma_tot"] ** 2, np.exp(lv_m).mean(0) + mu_m.var(0))
    # identical members => zero epistemic variance
    assert np.allclose(decompose(np.tile(mu_m[:1], (10, 1)), lv_m)["var_epi"], 0)
    n = nig_summary(np.zeros(3), np.ones(3) * 2, np.ones(3) * 3, np.ones(3))
    assert np.allclose(n["var_ale"], 0.5) and np.allclose(n["var_epi"], 0.25)


def test_temperature_and_conformal_restore_coverage():
    n = 4000; mu = rng.randn(n); true_sd = 0.8; y = mu + rng.randn(n) * true_sd
    sig = np.full(n, 0.3)                                    # over-confident by ~2.7x
    assert M.coverage(y, mu, sig) < 0.5
    tau = fit_temperature(y, mu, sig); assert abs(tau * 0.3 - true_sd) < 0.05
    cal = Calibrator(0.10).fit(y[:2000], mu[:2000], sig[:2000])
    lo, hi = cal.interval(mu[2000:], sig[2000:])
    cov = np.mean((y[2000:] >= lo) & (y[2000:] <= hi)); assert 0.86 < cov < 0.94
    q = conformal_quantile(y, mu, sig * tau, 0.1); assert 1.5 < q < 1.8      # ~ z_0.95 = 1.645


def test_metrics_perfect_and_calibrated():
    y = rng.randn(2000); mu = y + rng.randn(2000) * 0.5
    r = M.full_report(y, mu, np.full(2000, 0.5)); assert r["ece"] < 0.03 and abs(r["cov90"] - 0.9) < 0.03


def test_selectivity_member_level_and_pan_kinase():
    mu_t = np.array([[8.0], [7.0]]); off = np.array([[[6.0, 6.5]], [[7.5, 5.0]]])     # M=2, n=1, K=2
    assert np.allclose(member_margin(mu_t, off), [[1.5], [-0.5]])
    s = selectivity_stats(mu_t, off); assert np.allclose(s["mu_sel"], 0.5) and np.allclose(s["sigma_sel"], 1.0)
    assert np.allclose(s["fold"], 10 ** 0.5)
    assert (soft_margin(mu_t, off) <= member_margin(mu_t, off) + 1e-9).all()          # LSE >= max => margin <=


def test_acquisition_functions():
    mu, s = np.array([7.0, 7.0, 6.0]), np.array([0.1, 1.0, 1.0])
    ei = ACQ.expected_improvement(mu, s, y_best=7.0, xi=0.05)
    assert ei[1] > ei[0] > 0 and ei[1] > ei[2]
    assert np.allclose(ACQ.ucb(mu, s, 1.5), mu + 1.5 * s) and np.allclose(ACQ.lcb(mu, s, 1.0), mu - s)
    objs = np.array([[1, 1], [2, 0], [0, 2], [0.5, 0.5], [0, 0]])
    assert ACQ.pareto_rank(objs).tolist() == [0, 0, 0, 1, 2]


def test_reward_components_and_schedule():
    assert R.beta_schedule(1) == 0.3 and R.beta_schedule(300) == 0.3
    assert abs(R.beta_schedule(1000) - 1.5) < 1e-9 and 0.3 < R.beta_schedule(650) < 1.5
    assert R.r_pot(9.0, 0.0, 0.3) > 0.98 and R.r_pot(9.0, 3.0, 1.5) < 0.05            # uncertain "hallucinated" potency is crushed
    assert abs(R.r_pot(7.0, 0, 1) - 0.5) < 1e-9
    assert R.r_ad(0.5, 0.5) == 1 and R.r_ad(0.8, 0.5) < 0.2
    assert R.r_sa(2.0) == 1 and R.r_sa(5.5) == 0 and R.r_sa(7) == 0
    one = np.ones(3); comps = {c: one.copy() for c in R.COMPONENTS}; w = {"pot": 3, "sel": 2, "ad": 1.5, "sa": 1, "adme": 1, "safety": 1.5, "qed": .5}
    assert np.allclose(R.aggregate(comps, w, one), 1.0)
    comps["safety"] = np.array([1, 1, 0.0])
    out = R.aggregate(comps, w, one)
    assert abs(out[2] - (1e-4) ** (1.5 / 10.5)) < 1e-9 and out[2] < out[0]                # as-written: penalised (~0.27) but NOT ~0
    assert R.aggregate(comps, w, one, veto={"safety": 0.05})[2] == 0                      # optional veto actually zeroes it
    assert R.aggregate(comps, w, np.array([1, 0, 1]))[1] == 0                           # hard gate
    g = R.hard_gate([True, True, True, True], [0, 1, 0, 0], [400, 400, 650, 400], [0.1, 0.1, 0.1, 0.6]); assert g.tolist() == [1, 0, 0, 0]


def test_diversity_filter_bucket():
    df = R.DiversityFilter(bucket=2, min_score=0.5)
    out = df(["a"] * 4 + ["b"], [0.9, 0.9, 0.9, 0.9, 0.9]); assert out.tolist() == [0.9, 0.9, 0, 0, 0.9]


def test_similarity_weighting():
    w = sample_weights(["TNK2", "EGFR", "SRC"], {"TNK2": 1.0, "EGFR": 0.2, "SRC": 0.5}, {"TNK2": 500, "EGFR": 18000, "SRC": 2000}, lam=3, cap_n=2000)
    assert np.isclose(w[0], 4.0) and np.isclose(w[1], (1 + 3 * .04) * np.sqrt(2000 / 18000)) and np.isclose(w[2], 1 + 3 * .25)


def test_klifs_and_similarity():
    p = "ACDEFGHIKLMNPQRSTVWY" * 4 + "ACDEF"; assert len(p) == 85 and klifs_encode(p).shape == (425,)
    q = "-" * 5 + p[5:]; assert klifs_identity(p, q) == 1.0
    assert composite_similarity(np.array([1.0]), np.array([0.5])).item() == pytest.approx(0.85)


def test_applicability_domain_flags_outliers():
    Z = rng.randn(300, 8); FP = (rng.rand(300, 64) > 0.8).astype(float)
    ad = ApplicabilityDomain(5, 95).fit(Z, FP)
    far = np.full((1, 8), 25.0); assert not ad.in_domain(far, FP[:1])[0]
    assert ad.in_domain(Z[:50], FP[:50]).mean() > 0.8


def test_batch_selection_respects_quota_clusters_and_filters():
    n = 400; r = np.random.RandomState(1)
    pool = pd.DataFrame({"smiles": [f"c{i}" for i in range(n)], "cluster": r.randint(0, 120, n), "scaffold": [f"s{i % 90}" for i in range(n)],
                         "mu": r.uniform(5, 9, n), "sigma_epi": r.uniform(.1, 1, n), "mu_sel": r.uniform(0, 3, n), "passes_filters": r.rand(n) > 0.1})
    pool["ei"] = ACQ.expected_improvement(pool.mu.values, pool.sigma_epi.values, 8.0); pool["ucb"] = ACQ.ucb(pool.mu.values, pool.sigma_epi.values)
    pool["lcb"] = ACQ.lcb(pool.mu.values, pool.sigma_epi.values)
    b = B.select_batch(pool, known_scaffolds={"s1", "s2"}, seed=0)
    assert len(b) == 40 and b.batch_slice.value_counts().to_dict() == {"EXPLOIT": 16, "EXPLORE": 12, "DIVERSITY": 8, "CONTROL": 4}
    assert b.cluster.value_counts().max() <= 2 and b.smiles.is_unique and b.passes_filters.all()
    assert (b[b.batch_slice == "EXPLOIT"].mu_sel >= 1.5).all() and (b[b.batch_slice == "EXPLORE"].mu >= 6.3).all()
    assert not set(b[b.batch_slice == "DIVERSITY"].scaffold) & {"s1", "s2"}


def test_contract_validation():
    cal = Calibrator(0.1); cal.tau, cal.q = 1.0, 1.645
    b = pd.DataFrame({"smiles": ["CCO", "CCN"], "batch_slice": ["EXPLOIT", "CONTROL"], "mu": [7.0, 6.0], "sigma_epi": [0.3, 0.5]})
    ct = K.build_contract(b, 1, cal, lambda s: "A" * 14 + "-" + "B" * 10 + "-N")
    assert ct.compound_id.tolist() == ["TNK1-C01-001", "TNK1-C01-002"] and (ct.pred_ci90_low < ct.pred_mu_tnk1).all()
    bad = ct.copy(); bad.loc[0, "batch_slice"] = "WRONG"
    with pytest.raises(AssertionError): K.validate_contract(bad)


def test_clopper_pearson_and_holm():
    lo, hi = clopper_pearson(0, 10); assert lo == 0 and abs(hi - 0.3085) < 1e-3
    lo, hi = clopper_pearson(5, 10); assert abs(lo - 0.1871) < 1e-3 and abs(hi - 0.8129) < 1e-3
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_knn_tanimoto_baseline_and_kernel_bounds():
    X = (rng.rand(30, 64) > .7).astype(np.float32); S = tanimoto_sim(X, X)
    assert np.allclose(np.diag(S)[X.sum(1) > 0], 1) and S.min() >= 0 and S.max() <= 1 + 1e-6
    y = rng.randn(30); assert np.allclose(KNNTanimoto(1).fit(X, y).predict(X[X.sum(1) > 0]), y[X.sum(1) > 0])
