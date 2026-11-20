"""The model zoo: registry, representations, GP kernels and the nested-CV harness.

Hermetic. The representations need RDKit and are skipped without it; everything else --
the registry, the kernels, the fold translation, the train-fold-only scaling and the
out-of-fold matrix -- runs on numpy and sklearn alone, which are core dependencies.

What these tests are FOR. The sweep's whole purpose is to produce a defensible ranking of
~26 models, and the two ways that can go silently wrong are a leak (a scaler or a
hyperparameter that saw the test fold) and a mis-assembled OOF matrix (a model credited
with a prediction it never made). Both are tested here directly, because neither shows up
as an error -- only as a number that is too good.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tnk1.zoo as Z  # noqa: E402
from tnk1.zoo import benchmark as B, features as FZ, gp as GP  # noqa: E402


# --------------------------------------------------------------------------- registry

def test_registry_is_populated_and_every_entry_is_well_formed():
    assert len(Z.REGISTRY) >= 20, "the zoo should hold the full candidate set"
    for name, spec in Z.REGISTRY.items():
        assert spec.name == name, f"{name} is registered under a different key"
        assert spec.rep in FZ.REGISTRY, f"{name} names an unregistered representation"
        assert spec.family, f"{name} has no family; member-diversity selection needs it"
        assert spec.cite, f"{name} has no citation -- every choice must be traceable"
        assert spec.cost > 0


def test_available_reports_missing_dependencies_without_raising():
    """The contract that makes a long sweep safe: discover absences BEFORE fitting."""
    av = Z.available()
    assert set(av) == set(Z.REGISTRY)
    assert all(v == "ok" or v.startswith("missing: ") for v in av.values())
    # sklearn is a core dependency, so at minimum these must be runnable
    for n in ("rf", "ridge", "gp_minmax", "mean_baseline"):
        assert av[n] == "ok", f"{n} should run in the core environment, got {av[n]}"


def test_build_raises_a_readable_error_for_a_missing_dependency():
    missing = [n for n, v in Z.available().items() if v != "ok"]
    if not missing:
        pytest.skip("every dependency is installed in this environment")
    with pytest.raises(ImportError) as e:
        Z.build(missing[0])
    assert "pip install" in str(e.value), "the error must say how to fix it"


def test_unknown_model_lists_the_known_ones():
    with pytest.raises(KeyError) as e:
        Z.build("not_a_model")
    assert "rf" in str(e.value)


def test_grid_points_of_an_empty_grid_is_one_fit_not_zero():
    """The subtler of the two possible bugs: an empty grid must mean 'fit once with the
    defaults', never 'perform no fits'."""
    assert Z.grid_points({}) == [{}]
    assert len(Z.grid_points({"a": [1, 2], "b": [3, 4, 5]})) == 6


def test_shortlist_is_ordered_by_cost_and_filtered_by_availability():
    core = Z.shortlist("core")
    assert core, "there must be a core shortlist"
    costs = [Z.REGISTRY[n].cost for n in core]
    assert costs == sorted(costs), "cheapest first, so a truncated sweep still finishes"
    assert all(Z.available([n])[n] == "ok" for n in core)


def test_families_partition_the_registry():
    fam = Z.families()
    assert sum(len(v) for v in fam.values()) == len(Z.REGISTRY)
    assert "tree" in fam and "gp" in fam


def test_mean_baseline_predicts_the_training_mean():
    """The floor every other model must clear. An RMSE means nothing without it."""
    m = Z.build("mean_baseline").fit(np.zeros((10, 3)), np.arange(10.0))
    mu, sd = m.predict(np.zeros((4, 3)), return_std=True)
    assert np.allclose(mu, 4.5)
    assert np.all(sd > 0)


# ------------------------------------------------------------------------- GP kernels

def _counts(n=40, d=64, seed=0):
    return np.random.default_rng(seed).poisson(0.15, (n, d)).astype(float)


def test_minmax_kernel_is_a_valid_kernel():
    X = _counts()
    K = GP.MinMaxKernel()(X)
    assert np.allclose(K, K.T), "a kernel gram matrix must be symmetric"
    assert np.allclose(np.diag(K), 1.0), "self-similarity is 1 for MinMax"
    assert np.linalg.eigvalsh(K).min() > -1e-8, "must be positive semi-definite"


def test_tanimoto_kernel_is_valid_and_differs_from_minmax_on_counts():
    """The distinction is why both are registered: on BINARY vectors they coincide, on
    COUNT vectors they do not, and nothing in this project had tested which is better."""
    X = _counts()
    Kt, Km = GP.TanimotoKernel()(X), GP.MinMaxKernel()(X)
    assert np.linalg.eigvalsh(Kt).min() > -1e-8
    assert not np.allclose(Kt, Km), "they must differ on count vectors"
    Xb = (X > 0).astype(float)
    assert np.allclose(GP.TanimotoKernel()(Xb), GP.MinMaxKernel()(Xb), atol=1e-9), \
        "they must coincide on binary vectors"


def test_kernel_gradient_shape_is_empty_for_a_parameter_free_kernel():
    """sklearn's optimiser reads this to learn there is nothing here to tune."""
    X = _counts(12, 16)
    K, grad = GP.MinMaxKernel()(X, eval_gradient=True)
    assert K.shape == (12, 12)
    assert grad.shape == (12, 12, 0)


def test_kernel_composes_with_sklearn_operators():
    """Regression: these kernels must SUBCLASS sklearn's Kernel. A duck-typed kernel is
    silently wrapped in ConstantKernel(<kernel object>) by the `*` operator, whose theta
    then calls np.log on it -- 'loop of ufunc does not support argument 0 of type
    MinMaxKernel'. The failure appears only when a GP is actually fitted."""
    from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, WhiteKernel
    assert isinstance(GP.MinMaxKernel(), Kernel)
    k = ConstantKernel(1.0) * GP.MinMaxKernel() + WhiteKernel(0.1)
    X = _counts(10, 8)
    assert k(X).shape == (10, 10)
    assert np.isfinite(k.theta).all()


@pytest.mark.parametrize("kernel", ["minmax", "tanimoto", "rbf", "matern", "minmax_rbf"])
def test_every_gp_fits_and_returns_a_positive_sigma(kernel):
    rng = np.random.default_rng(1)
    X = _counts(50, 32, seed=1)
    y = 7 + X[:, :4].sum(1) * 0.3 + rng.normal(0, 0.3, 50)
    m = GP.SkGP(kernel=kernel).fit(X[:35], y[:35])
    mu, sd = m.predict(X[35:], return_std=True)
    assert mu.shape == (15,) and sd.shape == (15,)
    assert np.all(sd > 0), "a GP must return a strictly positive predictive sd"
    assert np.isfinite(m.log_marginal_likelihood())


def test_gp_exposes_its_learned_noise_for_the_aleatoric_estimate():
    """`ZooCritic(aleatoric="model")` reads this: a GP ESTIMATES the irreducible noise as
    a hyperparameter instead of having it inferred from held-out residuals."""
    X = _counts(40, 16, seed=2)
    y = np.random.default_rng(2).normal(7, 1, 40)
    m = GP.SkGP(kernel="minmax").fit(X, y)
    assert m.noise_sd is not None and m.noise_sd > 0


def test_gp_normalizes_y_so_far_from_data_it_does_not_predict_zero():
    """A GP's prior mean is 0. Without centring, a molecule far from the training set is
    pulled toward pIC50 = 0 -- an IC50 of 1 molar. That is a catastrophic prior for the
    generated molecules the RL critic is actually asked about."""
    X = _counts(40, 16, seed=3)
    y = np.full(40, 7.5) + np.random.default_rng(3).normal(0, 0.1, 40)
    m = GP.SkGP(kernel="minmax", normalize_y=True).fit(X, y)
    far = np.zeros((1, 16))                     # similar to nothing
    mu = m.predict(far)
    assert 5.0 < mu[0] < 10.0, f"far-field prediction {mu[0]} collapsed toward zero"


# ------------------------------------------------------------------- the CV harness

def _problem(n=120, d=48, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(0.1, (n, d)).astype(np.float32)
    w = np.zeros(d)
    w[rng.choice(d, 6, replace=False)] = rng.normal(0, 1.5, 6)
    y = 7 + X @ w * 0.5 + rng.normal(0, 0.4, n)
    keys = np.array([f"K{i // 2:03d}" for i in range(n)])
    scaf = np.array([f"S{i // 6:03d}" for i in range(n)])
    return X, y, keys, scaf


def _splits(keys, scaf, seed=42):
    from tnk1.data.splits import nested_scaffold_cv
    cv = nested_scaffold_cv(scaf, 5, 3, seed=seed)
    return {"cv": [{"outer": f["outer"],
                    "train": sorted(set(keys[f["train"]])),
                    "test": sorted(set(keys[f["test"]])),
                    "inner": [{"train": sorted(set(keys[i["train"]])),
                               "val": sorted(set(keys[i["val"]]))} for i in f["inner"]]}
                   for f in cv]}


def test_folds_from_splits_never_puts_a_compound_on_both_sides():
    """The property the frozen splits exist to guarantee. A compound measured under both
    IC50 and Ki has two rows, and both must land on the same side."""
    X, y, keys, scaf = _problem()
    folds = B.folds_from_splits(_splits(keys, scaf), keys)
    assert len(folds) == 5
    for f in folds:
        assert not (set(keys[f.train]) & set(keys[f.test])), \
            f"fold {f.fold} shares compounds between train and test"
        assert not (set(f.train) & set(f.test))


def test_folds_from_splits_keeps_inner_folds_inside_the_outer_training_set():
    X, y, keys, scaf = _problem()
    for f in B.folds_from_splits(_splits(keys, scaf), keys):
        for tr, va in f.inner:
            assert set(tr) <= set(f.train), "an inner train fold escaped the outer train set"
            assert set(va) <= set(f.train), "an inner val fold overlaps the outer TEST set"
            assert not (set(tr) & set(va))


def test_scaled_matrix_fits_on_the_training_fold_only():
    """THE leak this harness exists to prevent. If the scaler sees the test rows, it
    imports their location and spread into every fit -- worth a visible amount of apparent
    skill at N=172, and completely invisible in the output."""
    rng = np.random.default_rng(0)
    X = np.hstack([rng.poisson(0.1, (60, 8)).astype(float),
                   rng.normal(0, 1, (60, 4))])
    tr = np.arange(40)
    dense = np.zeros(12, dtype=bool)
    dense[8:] = True
    # Move the test rows far away; a train-only fit must be unaffected by it.
    X2 = X.copy()
    X2[40:, 8:] += 1000.0
    A = B.scaled_matrix(X, tr, dense)
    C = B.scaled_matrix(X2, tr, dense)
    assert np.allclose(A[tr], C[tr]), \
        "the training rows' scaled values changed when only TEST rows moved"
    assert np.allclose(A[:, :8], X[:, :8]), "fingerprint columns must not be scaled"


def test_scaled_matrix_leaves_fingerprint_geometry_alone():
    """Standardising a sparse count vector destroys the substructure-overlap geometry the
    Tanimoto/MinMax kernels rely on, and centring a bit vector makes 'substructure absent'
    a nonzero value."""
    X = np.random.default_rng(0).poisson(0.1, (30, 10)).astype(float)
    assert np.allclose(B.scaled_matrix(X, np.arange(20), None), X)


@pytest.mark.parametrize("scaler", ["robust", "cdf"])
def test_both_scalers_bound_their_output(scaler):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 5))
    X[0, 0] = 1e6                                # one wild outlier
    out = B.scaled_matrix(X, np.arange(40), np.ones(5, dtype=bool), scaler=scaler)
    assert np.isfinite(out).all()
    if scaler == "cdf":
        assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6, \
            "the CDF transform is bounded in [0,1] by construction"


def test_run_model_produces_out_of_fold_predictions_for_every_covered_row():
    X, y, keys, scaf = _problem()
    folds = B.folds_from_splits(_splits(keys, scaf), keys)
    rows, oof = B.run_model("ridge", X, y, folds, None, seed=0, tune=True)
    ok = [r for r in rows if r.get("status") == "ok"]
    assert len(ok) == 5
    covered = np.concatenate([f.test for f in folds])
    assert oof["filled"][covered].all(), "a covered row has no OOF prediction"
    assert np.isfinite(oof["mu"][covered]).all()
    # every prediction came from a model that did NOT see that row
    for f in folds:
        assert set(oof["fold"][f.test]) == {f.fold}


def test_run_model_records_the_chosen_params_so_tuning_is_auditable():
    X, y, keys, scaf = _problem()
    folds = B.folds_from_splits(_splits(keys, scaf), keys)
    rows, _ = B.run_model("ridge", X, y, folds, None, seed=0, tune=True)
    ok = [r for r in rows if r.get("status") == "ok"]
    assert all("params" in r and r["inner_fits"] > 0 for r in ok), \
        "a tuned run must record what it chose and how many inner fits it cost"


def test_a_failing_model_is_recorded_not_silently_absent():
    """One unavailable dependency must not void nineteen valid results -- but the failure
    has to appear in the table, because an absent model and a failed model look identical
    in a results directory."""
    X, y, keys, scaf = _problem()
    folds = B.folds_from_splits(_splits(keys, scaf), keys)

    def X_of(name):
        if name == "ridge":
            raise RuntimeError("synthetic featurisation failure")
        return X, None
    rows, oofs = B.run_sweep(["ridge", "mean_baseline"], X_of, y, folds, log=None)
    failed = [r for r in rows if str(r.get("status", "")).startswith("FAILED")]
    assert len(failed) == 1 and failed[0]["model"] == "ridge"
    assert "mean_baseline" in oofs, "the other model must still have run"


def test_oof_matrix_restricts_to_complete_case_rows():
    """A stacker fitted on a matrix with holes either crashes or silently imputes -- and
    an imputed column credits a model with a prediction it never made."""
    X, y, keys, scaf = _problem()
    folds = B.folds_from_splits(_splits(keys, scaf), keys)
    _, oofs = B.run_sweep(["ridge", "mean_baseline"], lambda n: (X, None), y, folds, log=None)
    # punch a hole in one model's coverage
    oofs["ridge"]["filled"][:5] = False
    oofs["ridge"]["mu"][:5] = np.nan
    P, S, used, rows_ok = B.oof_matrix(oofs)
    assert P.shape == (len(y), len(used))
    assert not rows_ok[:5].any(), "rows with a hole must be excluded"
    assert np.isfinite(P[rows_ok]).all()


def test_oof_matrix_refuses_an_empty_model_set():
    with pytest.raises(ValueError):
        B.oof_matrix({})


def test_with_std_never_fabricates_a_sigma():
    """A fabricated sigma feeds r_pot's uncertainty penalty and the conformal interval.
    A model without one must return None, not zeros dressed as a scale."""
    X = np.random.default_rng(0).normal(size=(20, 4))
    y = np.random.default_rng(1).normal(size=20)
    gp = Z.build("gp_minmax").fit(np.abs(X), y)
    mu, sd = B._with_std(gp, np.abs(X))
    assert sd is not None and np.all(sd > 0)

    class NoStd:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))
    mu, sd = B._with_std(NoStd().fit(X, y), X)
    assert sd is None, "a model with no predictive scale must report None"


# ----------------------------------------------------------------- representations

class TestRepresentations:
    """RDKit-dependent. Skipped cleanly in the core environment."""

    @staticmethod
    def _rdkit():
        return pytest.importorskip("rdkit", reason="representations need RDKit")

    def test_registry_metadata_is_consistent(self):
        for name, rep in FZ.REGISTRY.items():
            assert rep.kind in ("count", "binary", "dense", "mixed"), name
            assert callable(rep.fn)

    def test_core_representations_featurise(self):
        self._rdkit()
        smis = ["CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1"]
        for name in ("ecfp4_count", "maccs", "avalon", "erg", "rdkit_desc_all"):
            X, ok = FZ.featurize(name, smis)
            assert ok.all(), f"{name} failed on a valid SMILES"
            assert X.shape[0] == 3 and X.shape[1] > 0

    def test_unparseable_smiles_is_flagged_not_zero_filled(self):
        """An all-zero fingerprint row is a molecule similar to nothing for a Tanimoto
        kernel and a legitimate-looking point at the origin for a tree."""
        self._rdkit()
        X, ok = FZ.featurize("ecfp4_count", ["CCO", "not-a-smiles"])
        assert ok.tolist() == [True, False]
        with pytest.raises(ValueError) as e:
            FZ.featurize_or_raise("ecfp4_count", ["CCO", "not-a-smiles"])
        assert "did not parse" in str(e.value)

    def test_dense_columns_is_derived_not_hardcoded(self):
        """Regression: the offsets used to be literals, so adding a block in the middle of
        a combination silently shifted the slice -- a scaler fitted on fingerprint counts
        and a descriptor block left unscaled, with no error."""
        self._rdkit()
        for name in ("ecfp4_desc200", "admet_best_combo", "ecfp4_maccs_desc"):
            mask = FZ.dense_columns(name)
            X, _ = FZ.featurize(name, ["CCO"])
            assert mask is not None and len(mask) == X.shape[1], name
            assert mask.any() and not mask.all(), f"{name} should be partly dense"

    def test_pure_fingerprint_has_no_dense_columns(self):
        self._rdkit()
        assert FZ.dense_columns("ecfp4_count") is None

    def test_combination_width_is_the_sum_of_its_blocks(self):
        self._rdkit()
        total = sum(FZ.block_width(c) for c in FZ._COMBOS["admet_best_combo"])
        X, _ = FZ.featurize("admet_best_combo", ["CCO"])
        assert X.shape[1] == total
