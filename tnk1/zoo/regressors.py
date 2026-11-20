"""Base regressors for the zoo. Registers itself on import.

SELECTION PRINCIPLE. At 172 training molecules the question is not "which architecture has
the most capacity" but "which estimator has the least variance for a given amount of
bias". That is why this file is mostly trees, kernels and penalised linear models, and why
the deep learning lives elsewhere. The empirical finding this project already holds -- a
Morgan-count random forest beating a 10-member multi-task D-MPNN ensemble at Spearman
0.686 vs 0.501 -- is the expected ordering at this N, not a surprise, and it is consistent
with the published comparisons of descriptor models against graph networks on small QSAR
sets.

EVERY model here honours the project's existing duck type, so any of them can become the
RL critic through `tnk1.zoo.critic.ZooCritic`:

    m.fit(X, y) -> self
    m.predict(X, return_std=False) -> mu | (mu, sigma)

`gives_std=True` in the spec means `return_std=True` yields a GENUINE predictive scale,
not a placeholder. Models without one return the point prediction and let the caller decide
-- `tnk1.zoo.benchmark._with_std` never fabricates a sigma, because a fabricated sigma
feeds `r_pot`'s uncertainty penalty and the conformal interval.
"""
from __future__ import annotations

import numpy as np

from .registry import ModelSpec, register


# --------------------------------------------------------------------------- utilities


class _Sk:
    """Thin adapter giving any sklearn regressor the project's predict(return_std=) shape."""

    def __init__(self, est, std=None):
        self.est, self._std = est, std

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def predict(self, X, return_std=False):
        if not return_std:
            return self.est.predict(X)
        if self._std == "native":
            return self.est.predict(X, return_std=True)
        mu = self.est.predict(X)
        return mu, np.zeros(len(mu))


# ------------------------------------------------------------------------- tree models


class ForestStd:
    """Random forest whose sigma is the spread ACROSS trees.

    Same construction as `tnk1.baselines.models.ForestWithStd`, re-expressed here with
    `oob_score=True` so `tnk1.zoo.critic.ZooCritic` can take the out-of-bag residual as an
    honest aleatoric estimate rather than an in-sample one (a forest drives in-sample
    residuals to nearly zero, which would make the noise term vanish).

    Per-tree spread is an epistemic quantity -- disagreement among equally plausible
    fits -- and it is the one that empirically correlates with error here (+0.258), which is
    more than can be said for the deep ensemble's (-0.021).
    """

    def __init__(self, n_estimators=500, max_features=0.3, min_samples_leaf=1,
                 random_state=0, oob_score=True, **kw):
        from sklearn.ensemble import RandomForestRegressor
        self.m = RandomForestRegressor(
            n_estimators=n_estimators, max_features=max_features,
            min_samples_leaf=min_samples_leaf, random_state=random_state,
            oob_score=oob_score, bootstrap=True, n_jobs=-1, **kw)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def _per_tree(self, X):
        return np.stack([t.predict(X) for t in self.m.estimators_])

    def predict(self, X, return_std=False):
        per = self._per_tree(X)
        return (per.mean(0), per.std(0)) if return_std else per.mean(0)


class ExtraTreesStd(ForestStd):
    """Extremely randomised trees (Geurts, Ernst & Wehenkel, Machine Learning 2006).

    Splits are chosen at random rather than optimised, which lowers variance further than
    bagging alone -- the trade that tends to pay at small N. Worth testing against the
    forest rather than assumed better.
    """

    def __init__(self, n_estimators=500, max_features=0.3, min_samples_leaf=1,
                 random_state=0, **kw):
        from sklearn.ensemble import ExtraTreesRegressor
        self.m = ExtraTreesRegressor(
            n_estimators=n_estimators, max_features=max_features,
            min_samples_leaf=min_samples_leaf, random_state=random_state,
            bootstrap=True, oob_score=True, n_jobs=-1, **kw)


class QuantileForest:
    """Quantile regression forest. Meinshausen, "Quantile Regression Forests", JMLR 2006.

    A forest's leaves hold the training labels that fell in them, so the forest defines a
    full conditional DISTRIBUTION, not just a mean -- and the quantiles of that
    distribution are a prediction interval that needs no Gaussian assumption and no
    calibration step. That matters here because the Gaussian assumption is doing real work
    in the current pipeline (the conformal step normalises by a sigma and the reward
    penalises it) while nothing has checked that the residuals are Gaussian at all.

    Implemented directly rather than pulled from a dependency: sklearn's forests expose
    `apply()` (leaf index per tree) which is all that is required.
    """

    def __init__(self, n_estimators=500, max_features=0.3, min_samples_leaf=5,
                 random_state=0, alpha=0.10):
        from sklearn.ensemble import RandomForestRegressor
        self.alpha = alpha
        self.m = RandomForestRegressor(
            n_estimators=n_estimators, max_features=max_features,
            min_samples_leaf=min_samples_leaf, random_state=random_state, n_jobs=-1)

    def fit(self, X, y):
        self.m.fit(X, y)
        self.y_ = np.asarray(y, float)
        leaves = self.m.apply(X)                       # (n, n_trees)
        # leaf -> training labels, per tree
        self.leaf_y_ = []
        for t in range(leaves.shape[1]):
            d: dict[int, list] = {}
            for i, lf in enumerate(leaves[:, t]):
                d.setdefault(int(lf), []).append(self.y_[i])
            self.leaf_y_.append({k: np.asarray(v) for k, v in d.items()})
        return self

    def _dists(self, X):
        leaves = self.m.apply(X)
        for i in range(len(X)):
            vals = [self.leaf_y_[t][int(leaves[i, t])]
                    for t in range(leaves.shape[1])
                    if int(leaves[i, t]) in self.leaf_y_[t]]
            yield np.concatenate(vals) if vals else self.y_

    def predict(self, X, return_std=False):
        mu = self.m.predict(X)
        if not return_std:
            return mu
        sd = np.array([d.std() for d in self._dists(X)])
        return mu, np.maximum(sd, 1e-6)

    def interval(self, X, alpha=None):
        a = self.alpha if alpha is None else alpha
        qs = np.array([np.quantile(d, [a / 2, 1 - a / 2]) for d in self._dists(X)])
        return qs[:, 0], qs[:, 1]


register(ModelSpec("rf", lambda p: ForestStd(**p), rep="ecfp4_desc200",
                   grid={"max_features": ["sqrt", 0.3, 0.5], "min_samples_leaf": [1, 3, 5]},
                   gives_std=True, family="tree", cost=1.0, needs=("sklearn",),
                   cite="Breiman 2001; this project's strongest model to date",
                   tags=("core", "control", "seeded")))
register(ModelSpec("rf_fp", lambda p: ForestStd(**p), rep="ecfp4_count",
                   grid={"max_features": ["sqrt", 0.3], "min_samples_leaf": [1, 3]},
                   gives_std=True, family="tree", cost=1.0, needs=("sklearn",),
                   cite="Breiman 2001", tags=("core", "seeded")))
register(ModelSpec("extratrees", lambda p: ExtraTreesStd(**p), rep="ecfp4_desc200",
                   grid={"max_features": ["sqrt", 0.3, 0.5], "min_samples_leaf": [1, 3]},
                   gives_std=True, family="tree", cost=1.0, needs=("sklearn",),
                   cite="Geurts et al. 2006", tags=("core", "seeded")))
register(ModelSpec("qrf", lambda p: QuantileForest(**p), rep="ecfp4_desc200",
                   grid={"min_samples_leaf": [3, 5, 10]},
                   gives_std=True, family="tree", cost=1.2, needs=("sklearn",),
                   cite="Meinshausen 2006", tags=("core", "seeded", "interval")))


# ---------------------------------------------------------------------- boosted trees


def _xgb(p):
    from xgboost import XGBRegressor
    kw = dict(n_estimators=500, subsample=0.8, colsample_bytree=0.7, tree_method="hist",
              max_depth=5, learning_rate=0.03, reg_lambda=1.0)
    kw.update(p)
    return _Sk(XGBRegressor(**kw))


def _lgbm(p):
    from lightgbm import LGBMRegressor
    kw = dict(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=5,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
    kw.update(p)
    return _Sk(LGBMRegressor(**kw))


def _catboost(p):
    from catboost import CatBoostRegressor
    kw = dict(iterations=600, learning_rate=0.03, depth=6, verbose=0,
              allow_writing_files=False)
    kw.update(p)
    return _Sk(CatBoostRegressor(**kw))


class NGBoostStd:
    """NGBoost: Duan et al., "NGBoost: Natural Gradient Boosting for Probabilistic
    Prediction", ICML 2020.

    Boosts the PARAMETERS of a predictive distribution (mean and scale of a Normal) with
    the natural gradient, so unlike XGBoost it returns a calibrated sigma natively rather
    than a point estimate someone has to wrap. For this project that is the interesting
    property: boosted trees are the right model family at this N, and this variant is the
    one that also produces the uncertainty the reward and the conformal layer need.
    """

    def __init__(self, n_estimators=2000, learning_rate=0.01, minibatch_frac=0.5,
                 max_depth=2, random_state=0, **kw):
        from ngboost import NGBRegressor
        from sklearn.tree import DecisionTreeRegressor
        # Defaults from the two sources that actually tuned this for chemistry: NGBoost's
        # own tuning guide reports its best configuration as a depth-2 base learner with
        # minibatch_frac=0.5, and Tom, Hickman, Zinzuwadia, Mohajeri, Sanchez-Lengeling &
        # Aspuru-Guzik, "Calibration and generalizability of probabilistic models on
        # low-data chemical datasets with DIONYSUS" (arXiv:2212.01574), use 2000 trees at
        # max_depth <= 3 and single out NGBoost as suiting "very small datasets (under
        # about 100 molecules)" -- the regime here. Their results also make it the
        # best-calibrated regressor on two of three datasets.
        #
        # The library's own documentation warns that deep trees + small leaves + a high
        # learning rate diverge and produce over-narrow predictive distributions, which is
        # the failure that would matter most here: a confidently wrong sigma feeding r_pot.
        base = DecisionTreeRegressor(criterion="friedman_mse", max_depth=int(max_depth),
                                     random_state=random_state)
        self.m = NGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                              minibatch_frac=minibatch_frac, Base=base,
                              random_state=random_state, verbose=False, **kw)

    def fit(self, X, y):
        self.m.fit(np.asarray(X), np.asarray(y, float))
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X)
        mu = self.m.predict(X)
        if not return_std:
            return mu
        d = self.m.pred_dist(X)
        return mu, np.maximum(np.asarray(d.scale, float), 1e-6)


# Grid narrowed to the small-N end of the search space that Jiang et al. 2021 used for
# XGBoost (max_depth 3-10, lr 0.01-0.2, min_child_weight 1-6, subsample/colsample 0.7-1.0).
# At 172 molecules a depth-10 tree has more leaves than training points, so the upper half
# of their range is unusable here and `reg_lambda` carries the regularisation instead.
register(ModelSpec("xgb", _xgb, rep="ecfp4_desc200",
                   grid={"max_depth": [2, 3, 5], "learning_rate": [0.01, 0.03, 0.1],
                         "reg_lambda": [1.0, 10.0, 100.0],
                         "min_child_weight": [1, 5, 20]},
                   family="boost", cost=1.5, needs=("xgboost",),
                   cite="Chen & Guestrin 2016; grid from Jiang et al. 2021 restricted to "
                        "small N; Boldini et al., J Cheminform 15:73 (2023) find XGBoost "
                        "the overall winner across 157,590 trained models",
                   tags=("core", "seeded")))
# LightGBM is the GBDT variant with a SMALL-DATA-specific endorsement: Boldini et al. 2023
# report that on small sets "LightGBM matches or beats XGBoost" -- contradicting the earlier
# concern that its leaf-wise growth overfits small datasets -- while CatBoost does worse
# there. Both are kept because the evidence genuinely conflicts across studies.
register(ModelSpec("lgbm", _lgbm, rep="ecfp4_desc200",
                   grid={"num_leaves": [4, 7, 15, 31], "learning_rate": [0.01, 0.03, 0.1],
                         "min_child_samples": [2, 5, 10, 20],
                         "reg_lambda": [0.0, 1.0, 100.0]},
                   family="boost", cost=1.2, needs=("lightgbm",),
                   cite="Ke et al. 2017; Boldini et al. 2023 -- matches/beats XGBoost on "
                        "SMALL datasets",
                   tags=("core", "seeded")))
# Kamuntavicius et al., J Cheminform 17:108 (2025) rank CatBoost the single best model
# across 25 ADMET datasets (best mean rank 2.07, significant under Nemenyi) with RF and a
# Chemprop MPNN both performing poorly -- the same ordering this project found, from an
# independent benchmark. Boldini et al. disagree for the smallest sets. Run it and see.
register(ModelSpec("catboost", _catboost, rep="ecfp4_desc200",
                   grid={"depth": [2, 4, 6], "learning_rate": [0.03, 0.1],
                         "l2_leaf_reg": [1.0, 10.0, 30.0]},
                   family="boost", cost=2.5, needs=("catboost",),
                   cite="Prokhorenkova et al. 2018; best mean rank over 25 ADMET sets in "
                        "Kamuntavicius et al. 2025",
                   tags=("core", "seeded")))
register(ModelSpec("ngboost", lambda p: NGBoostStd(**p), rep="ecfp4_desc200",
                   grid={"learning_rate": [0.005, 0.01, 0.05], "max_depth": [2, 3]},
                   gives_std=True, family="boost", cost=3.0, needs=("ngboost",),
                   cite="Duan et al., ICML 2020; DIONYSUS (arXiv:2212.01574) -- suits "
                        "datasets under ~100 molecules and was best-calibrated on 2 of 3",
                   tags=("core", "seeded", "uq")))


# ----------------------------------------------------------- penalised linear + sparse


def _ridge(p):
    from sklearn.linear_model import Ridge
    return _Sk(Ridge(alpha=p.get("alpha", 1.0), random_state=p.get("random_state", 0)))


def _lasso(p):
    from sklearn.linear_model import Lasso
    return _Sk(Lasso(alpha=p.get("alpha", 0.01), max_iter=20000))


def _enet(p):
    from sklearn.linear_model import ElasticNet
    return _Sk(ElasticNet(alpha=p.get("alpha", 0.01), l1_ratio=p.get("l1_ratio", 0.5),
                          max_iter=20000))


def _bayes_ridge(p):
    from sklearn.linear_model import BayesianRidge
    return _Sk(BayesianRidge(**p), std="native")


def _ard(p):
    from sklearn.linear_model import ARDRegression
    return _Sk(ARDRegression(**p), std="native")


def _pls(p):
    from sklearn.cross_decomposition import PLSRegression
    n = int(p.get("n_components", 4))

    class _PLS:
        def __init__(self):
            self.m = PLSRegression(n_components=n, scale=False)

        def fit(self, X, y):
            self.m.fit(X, np.asarray(y, float).reshape(-1, 1))
            return self

        def predict(self, X, return_std=False):
            mu = self.m.predict(X).ravel()
            return (mu, np.zeros(len(mu))) if return_std else mu
    return _PLS()


# BayesianRidge and ARD are the two penalised linear models that return a genuine
# predictive sd (they are exact Bayesian linear regression with learned precisions), which
# makes them a useful calibration CONTROL: if either produces better error-sigma
# correlation than the deep ensemble, the ensemble's uncertainty has no excuse.
register(ModelSpec("ridge", _ridge, rep="ecfp4_desc200",
                   grid={"alpha": [0.1, 1.0, 10.0, 100.0]},
                   family="linear", cost=0.2, needs=("sklearn",),
                   cite="Hoerl & Kennard 1970", tags=("core",)))
register(ModelSpec("lasso", _lasso, rep="ecfp4_desc200",
                   grid={"alpha": [0.001, 0.01, 0.1]},
                   family="linear", cost=0.3, needs=("sklearn",),
                   cite="Tibshirani 1996", tags=("probe",)))
register(ModelSpec("elasticnet", _enet, rep="ecfp4_desc200",
                   grid={"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]},
                   family="linear", cost=0.4, needs=("sklearn",),
                   cite="Zou & Hastie 2005", tags=("probe",)))
register(ModelSpec("bayes_ridge", _bayes_ridge, rep="ecfp4_desc200",
                   gives_std=True, family="linear", cost=0.3, needs=("sklearn",),
                   cite="MacKay 1992; Tipping 2001", tags=("core", "uq")))
register(ModelSpec("ard", _ard, rep="ecfp4_desc200",
                   gives_std=True, family="linear", cost=1.0, needs=("sklearn",),
                   cite="MacKay 1992 ARD", tags=("probe", "uq")))
register(ModelSpec("pls", _pls, rep="ecfp4_desc200",
                   grid={"n_components": [2, 4, 8, 16]},
                   family="linear", cost=0.2, needs=("sklearn",),
                   cite="Wold et al. 1984 -- the classical QSAR workhorse",
                   tags=("core", "control")))


# ------------------------------------------------------------------ nearest neighbours


class KNNTanimotoStd:
    """Similarity-weighted k-NN on a count-Tanimoto similarity, with a weighted-spread sd.

    Same estimator as `tnk1.baselines.models.KNNTanimoto` (kept registered so the sweep
    includes the published baseline), but exposed through the registry so it can be tuned
    over k on the same inner folds as everything else rather than in its own loop.
    """

    def __init__(self, k=5, power=1.0):
        self.k, self.power = int(k), float(power)

    def fit(self, X, y):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float)
        return self

    def predict(self, X, return_std=False):
        from ..baselines.models import tanimoto_sim
        S = tanimoto_sim(np.asarray(X, float), self.X)
        k = min(self.k, S.shape[1])
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        s = np.take_along_axis(S, idx, 1) ** self.power
        yy = self.y[idx]
        w = np.maximum(s.sum(1), 1e-9)
        mu = (s * yy).sum(1) / w
        if not return_std:
            return mu
        var = (s * (yy - mu[:, None]) ** 2).sum(1) / w
        return mu, np.sqrt(np.maximum(var, 1e-12))


register(ModelSpec("knn_tanimoto", lambda p: KNNTanimotoStd(**p), rep="ecfp4_count",
                   grid={"k": [1, 3, 5, 7, 11], "power": [1.0, 3.0]},
                   gives_std=True, family="neighbour", cost=0.2, needs=("sklearn",),
                   cite="this project's published Stage-2 baseline 2.1",
                   tags=("core", "control")))


# ------------------------------------------------------------ kernel methods (non-GP)


def _tanimoto_gram(A, B):
    from ..baselines.models import tanimoto_sim
    return tanimoto_sim(np.asarray(A, float), np.asarray(B, float))


class _PrecomputedKernel:
    """SVR / kernel-ridge on a precomputed Tanimoto gram matrix.

    The Tanimoto (MinMax) kernel is positive definite on count vectors, so it is a valid
    kernel for an SVM and for kernel ridge -- not merely a similarity heuristic. Using it
    puts the chemistry in the kernel rather than asking an RBF to rediscover substructure
    overlap from 2048 sparse counts, which is the standard argument for Tanimoto kernels in
    QSAR and the reason the Tanimoto GP already outperforms the deep ensemble here.
    """

    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.X_ = np.asarray(X, float)
        self.est.fit(_tanimoto_gram(self.X_, self.X_), np.asarray(y, float))
        return self

    def predict(self, X, return_std=False):
        mu = self.est.predict(_tanimoto_gram(np.asarray(X, float), self.X_))
        return (mu, np.zeros(len(mu))) if return_std else mu


def _svr_tan(p):
    from sklearn.svm import SVR
    return _PrecomputedKernel(SVR(kernel="precomputed", C=p.get("C", 10.0),
                                  epsilon=p.get("epsilon", 0.1)))


def _krr_tan(p):
    from sklearn.kernel_ridge import KernelRidge
    return _PrecomputedKernel(KernelRidge(kernel="precomputed", alpha=p.get("alpha", 1.0)))


def _svr_rbf(p):
    from sklearn.svm import SVR
    kw = dict(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)
    kw.update({k: v for k, v in p.items() if k != "random_state"})
    return _Sk(SVR(**kw))


# SVR-RBF ON SCALED DESCRIPTORS IS THE LITERATURE'S FIRST CHOICE, NOT AN AFTERTHOUGHT.
# Jiang, Wu, Hsieh, Chen, Liao, Wang, Shen, Cao, Wu & Hou, "Could graph neural networks
# learn better molecular representation for drug discovery? A comparison study of
# descriptor-based and graph-based models", J Cheminform 13:12 (2021), report 50-seed mean
# RMSE in which SVM is the BEST model on the two smallest regression sets and beats random
# forest by 0.29 RMSE on FreeSolv (N=639) -- and in which RF is the second-worst or worst
# of the descriptor-based models on all three. This project's "RF beats the D-MPNN
# ensemble" result matches the paper's model-CLASS finding (descriptors beat graphs at
# small N) while leaving the within-class ordering untested: RF is very likely NOT the
# ceiling here.
#
# Caveat recorded rather than buried: Jiang et al. used random 8:1:1 splits (not scaffold)
# and MOE + PubChem + substructure fingerprints (not Morgan + RDKit), so the result
# transfers by analogy to this project's descriptor block, not directly.
register(ModelSpec("svr_rbf", _svr_rbf, rep="rdkit_desc_all",
                   grid={"C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 0.001, 0.01],
                         "epsilon": [0.05, 0.1, 0.2]},
                   family="kernel", cost=0.5, needs=("sklearn",),
                   cite="Jiang et al. 2021 -- best regressor on the smallest sets",
                   tags=("core",)))
register(ModelSpec("svr_rbf_combo", _svr_rbf, rep="admet_best_combo",
                   grid={"C": [1.0, 10.0, 100.0], "gamma": ["scale", 0.001]},
                   family="kernel", cost=0.8, needs=("sklearn",),
                   cite="Jiang et al. 2021 model x Kamuntavicius et al. 2025 features",
                   tags=("core",)))
register(ModelSpec("svr_tanimoto", _svr_tan, rep="ecfp4_count",
                   grid={"C": [0.01, 1.0, 10.0, 100.0, 1000.0],
                         "epsilon": [0.01, 0.05, 0.1, 0.2]},
                   family="kernel", cost=0.5, needs=("sklearn",),
                   cite="Tanimoto/MinMax kernel SVR; NB no controlled QSAR benchmark of "
                        "Tanimoto-SVR vs RBF-SVR was found, so this pair settles it on "
                        "our own folds",
                   tags=("core",)))
register(ModelSpec("krr_tanimoto", _krr_tan, rep="ecfp4_count",
                   grid={"alpha": [1e-4, 1e-2, 0.1, 1.0, 10.0]},
                   family="kernel", cost=0.3, needs=("sklearn",),
                   cite="kernel ridge with a Tanimoto kernel: closed form, one "
                        "hyperparameter, no C/epsilon interaction",
                   tags=("core",)))


# --------------------------------------------------------------------------- the mean


class MeanBaseline:
    """Predicts the training mean. The floor every model must clear.

    Not a joke entry. With 172 molecules, scaffold-split so train and test share no
    scaffold, a model can land BELOW this -- and a reported RMSE of 0.95 means nothing
    until the reader knows the mean predictor's is 1.02. The v3 run reports the deep
    ensemble at RMSE 0.954 without that reference point anywhere in the document.
    """

    def fit(self, X, y):
        self.mu_ = float(np.mean(y))
        self.sd_ = float(np.std(y))
        return self

    def predict(self, X, return_std=False):
        mu = np.full(len(X), self.mu_)
        return (mu, np.full(len(X), self.sd_)) if return_std else mu


register(ModelSpec("mean_baseline", lambda p: MeanBaseline(), rep="ecfp4_count",
                   gives_std=True, family="trivial", cost=0.01, needs=(),
                   cite="the floor: train-set mean", tags=("core", "control")))
