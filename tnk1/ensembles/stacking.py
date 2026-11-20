"""Combining a library of base models into one predictor. Pure numpy/scipy/sklearn.

WHY NOT JUST AVERAGE. The project's existing ensemble is 10 members of ONE architecture,
which buys variance reduction and nothing else: the members share the D-MPNN's inductive
bias, so they share its errors, and averaging correlated errors does not remove them. The
bias-variance-COVARIANCE decomposition (Ueda & Nakano 1996; Brown, Wyatt, Harris & Yao,
"Diversity creation methods in classifier ensembles", Information Fusion 2005) makes the
point exactly: an ensemble's error contains a covariance term, and only members whose
errors are decorrelated reduce it. A Morgan-count random forest, a Tanimoto GP and a
D-MPNN are three different inductive biases over the same 172 measurements -- that is where
the covariance term has room to fall.

WHY THE WEIGHTS ARE CONSTRAINED. Breiman, "Stacked regressions", Machine Learning 1996,
found ordinary least squares on base-model predictions unusable -- the predictions are
collinear by construction (every model is estimating the same function), so OLS produces
large cancelling coefficients that fit the stacking sample and not the target. Breiman's
fix, which is what `NNLSStacker` implements, is to constrain the weights to be
NON-NEGATIVE; he reports this alone restores stability, and that the further constraint of
summing to one costs little. At N=172 with ~20 collinear columns this is not a refinement,
it is the difference between a stack that works and one that is worse than its best member.

WHY THE INPUT MUST BE OUT-OF-FOLD. Wolpert, "Stacked generalization", Neural Networks
1992, specifies that the meta-learner is trained on predictions made for points the base
model did not see. Train the meta-learner on in-sample base predictions and it learns to
trust whichever model overfits hardest, because that model's in-sample predictions look
best. `tnk1.zoo.benchmark.oof_matrix` is the only sanctioned source of that matrix here,
and it carries a `rows_ok` mask rather than imputing holes, because an imputed column
credits a model with a prediction it never made.

van der Laan, Polley & Hubbard, "Super Learner", Statistical Applications in Genetics and
Molecular Biology 2007, is the same construction with the asymptotic guarantee: the
cross-validated stack performs asymptotically as well as the best member of the library.
`SuperLearner` below is that, with the meta-learner itself chosen by cross-validation,
which is the part people usually skip.
"""
from __future__ import annotations

import warnings

import numpy as np


# ------------------------------------------------------------------------- meta-learners


class NNLSStacker:
    """Breiman (1996) stacked regression: non-negative least squares on OOF predictions.

    `normalize=True` additionally rescales the weights to sum to 1, which makes the stack
    an interpolation of its members (so it inherits their scale and cannot extrapolate past
    them) and makes the weight vector readable as "how much each model is worth".

    `fit_intercept` is off by default. The members are all predicting pIC50 on the same
    scale, so an intercept mostly absorbs fold-level mean shifts, which at N=172 is a real
    risk of fitting the calibration sample's mean rather than the target's.
    """

    def __init__(self, normalize=True, fit_intercept=False, nonneg=True):
        self.normalize, self.fit_intercept, self.nonneg = normalize, fit_intercept, nonneg
        self.w_ = None
        self.b_ = 0.0

    def fit(self, P, y):
        from scipy.optimize import nnls
        P = np.asarray(P, float)
        y = np.asarray(y, float)
        A = np.hstack([P, np.ones((len(P), 1))]) if self.fit_intercept else P
        if self.nonneg:
            # nnls cannot give a negative intercept, so centring is how an intercept is
            # allowed to go either way: solve on the residual from y's mean and add it back.
            if self.fit_intercept:
                self.b_ = float(y.mean())
                w, _ = nnls(P, y - self.b_)
            else:
                self.b_, (w, _) = 0.0, nnls(P, y)
        else:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            w, self.b_ = (sol[:-1], float(sol[-1])) if self.fit_intercept else (sol, 0.0)
        s = w.sum()
        if self.normalize and s > 1e-12:
            w = w / s
            # With weights renormalised to sum to 1 the fitted intercept no longer
            # corresponds to that solution; drop it rather than apply a stale shift.
            if self.nonneg and self.fit_intercept:
                self.b_ = float(y.mean() - (P @ w).mean())
        if s <= 1e-12:
            warnings.warn("NNLS put zero weight on every base model; falling back to the "
                          "uniform average. This usually means the OOF matrix is constant "
                          "or misaligned with y.", RuntimeWarning, stacklevel=2)
            w = np.full(P.shape[1], 1.0 / P.shape[1])
        self.w_ = w
        return self

    def predict(self, P):
        return np.asarray(P, float) @ self.w_ + self.b_

    @property
    def weights(self):
        return None if self.w_ is None else self.w_.copy()


class RidgeStacker:
    """L2-regularised stack; alpha chosen by internal CV. Allows negative weights.

    Kept as a comparator for `NNLSStacker` rather than a recommendation: negative weights
    let the stack extrapolate beyond its members, which occasionally helps and more often
    is how a stack ends up worse than its best member on a fresh fold.
    """

    def __init__(self, alphas=(0.01, 0.1, 1.0, 10.0, 100.0), positive=False):
        self.alphas, self.positive = alphas, positive
        self.m_ = None

    def fit(self, P, y):
        from sklearn.linear_model import RidgeCV
        n = len(y)
        cv = min(5, max(2, n // 10))
        self.m_ = RidgeCV(alphas=list(self.alphas), cv=cv).fit(np.asarray(P, float), y)
        return self

    def predict(self, P):
        return self.m_.predict(np.asarray(P, float))

    @property
    def weights(self):
        return None if self.m_ is None else np.asarray(self.m_.coef_, float)


class CaruanaSelector:
    """Greedy ensemble selection WITH REPLACEMENT.

    Caruana, Niculescu-Mizil, Crew & Ksikes, "Ensemble selection from libraries of models",
    ICML 2004. Start from the empty ensemble; repeatedly add the single model (replacement
    allowed) that most improves the ensemble's score; stop after `n_rounds`. Selecting with
    replacement is what turns a hard subset choice into a weighting -- a model added three
    times out of ten rounds carries weight 0.3 -- and the greedy path is far more robust at
    small N than fitting 20 free coefficients, because it only ever makes one discrete
    decision at a time.

    This is the method auto-sklearn uses to build its final ensemble, and it is the one I
    would trust most at N=172: it cannot produce a negative weight, cannot produce a large
    cancelling pair, and degrades to "pick the best single model" when nothing combines.
    """

    def __init__(self, n_rounds=25, metric="rmse", with_replacement=True, init_best=True):
        self.n_rounds, self.metric = n_rounds, metric
        self.with_replacement, self.init_best = with_replacement, init_best
        self.counts_ = None
        self.path_ = []

    def _loss(self, y, p):
        if self.metric == "rmse":
            return float(np.sqrt(np.mean((y - p) ** 2)))
        if self.metric == "mae":
            return float(np.mean(np.abs(y - p)))
        if self.metric == "spearman":
            from scipy import stats
            r = stats.spearmanr(y, p)[0]
            return -(0.0 if not np.isfinite(r) else float(r))
        raise ValueError(f"unknown metric {self.metric!r}")

    def fit(self, P, y):
        P = np.asarray(P, float)
        y = np.asarray(y, float)
        n_models = P.shape[1]
        counts = np.zeros(n_models)
        running = np.zeros(len(y))
        total = 0
        if self.init_best:
            j = int(np.argmin([self._loss(y, P[:, k]) for k in range(n_models)]))
            counts[j] += 1
            running = P[:, j].copy()
            total = 1
            self.path_.append(j)
        for _ in range(max(0, self.n_rounds - total)):
            best_j, best_l = None, float("inf")
            for j in range(n_models):
                if not self.with_replacement and counts[j] > 0:
                    continue
                cand = (running * total + P[:, j]) / (total + 1)
                l = self._loss(y, cand)
                if l < best_l:
                    best_j, best_l = j, l
            if best_j is None:
                break
            # Stop when no addition improves on the current ensemble: continuing past this
            # point is how greedy selection overfits the selection sample.
            if total and best_l >= self._loss(y, running) - 1e-12:
                break
            running = (running * total + P[:, best_j]) / (total + 1)
            counts[best_j] += 1
            total += 1
            self.path_.append(best_j)
        if total == 0:
            counts = np.full(n_models, 1.0)
            total = n_models
        self.counts_ = counts
        self.w_ = counts / counts.sum()
        return self

    def predict(self, P):
        return np.asarray(P, float) @ self.w_

    @property
    def weights(self):
        return None if self.counts_ is None else self.counts_ / self.counts_.sum()


class BaggedCaruana:
    """Caruana greedy selection, run on random subsets of the library and summed.

    Caruana et al. (2004) give three anti-overfitting enhancements, because forward
    selection "sometimes overfits to the hillclimbing (validation) set": selection WITH
    REPLACEMENT, sorted initialisation, and BAGGED ensemble selection. With a library of
    10-25 models -- tiny next to the ~2000 of the paper -- replacement and bagging are the
    two that matter; sorted initialisation has nothing to sort.

    Each bag draws `frac` of the library at random, runs the greedy selection on it, and
    the final weights are the summed selection counts. A model that only wins when a
    stronger one is absent therefore gets partial credit, and one that wins everywhere
    gets most of the weight -- which is a much softer decision than a single greedy path
    makes on 172 rows.
    """

    def __init__(self, n_bags=20, frac=0.5, n_rounds=25, metric="rmse", seed=0):
        self.n_bags, self.frac, self.n_rounds = int(n_bags), float(frac), int(n_rounds)
        self.metric, self.seed = metric, seed
        self.w_ = None

    def fit(self, P, y):
        P = np.asarray(P, float)
        y = np.asarray(y, float)
        K = P.shape[1]
        k = max(2, int(round(self.frac * K)))
        rng = np.random.RandomState(self.seed)
        counts = np.zeros(K)
        for _ in range(self.n_bags):
            cols = rng.choice(K, size=min(k, K), replace=False)
            sel = CaruanaSelector(self.n_rounds, self.metric).fit(P[:, cols], y)
            counts[cols] += sel.counts_
        if counts.sum() <= 0:
            counts = np.ones(K)
        self.w_ = counts / counts.sum()
        return self

    def predict(self, P):
        return np.asarray(P, float) @ self.w_

    @property
    def weights(self):
        return None if self.w_ is None else self.w_.copy()


class RankNNLSStacker:
    """Rank-normalise each member's column, then NNLS on the simplex.

    Makes the stack insensitive to each member's SCALE and CALIBRATION while still
    emitting a continuous value that can be conformalised -- the property plain rank
    averaging loses. A member whose ordering is good but whose predictions sit half a log
    unit low contributes its ordering without dragging the combined mean.

    The columns are mapped to [0, 1] by rank and the weighted combination is mapped back
    through the training y quantiles, so the output stays on a pIC50 scale.
    """

    def __init__(self):
        self.inner_ = NNLSStacker(normalize=True, fit_intercept=False)
        self.y_sorted_ = None

    @staticmethod
    def _ranks(P):
        from scipy.stats import rankdata
        P = np.asarray(P, float)
        n = len(P)
        if n < 2:
            return np.zeros_like(P)
        return np.column_stack([(rankdata(P[:, j]) - 0.5) / n for j in range(P.shape[1])])

    def fit(self, P, y):
        y = np.asarray(y, float)
        self.y_sorted_ = np.sort(y)
        # target on the same [0,1] rank scale, so NNLS weights are comparable across columns
        from scipy.stats import rankdata
        yq = (rankdata(y) - 0.5) / len(y)
        self.inner_.fit(self._ranks(P), yq)
        return self

    def predict(self, P):
        q = np.clip(self.inner_.predict(self._ranks(P)), 0.0, 1.0)
        if self.y_sorted_ is None:
            return q
        idx = np.clip((q * len(self.y_sorted_)).astype(int), 0, len(self.y_sorted_) - 1)
        return self.y_sorted_[idx]

    @property
    def weights(self):
        return self.inner_.weights


class MeanStacker:
    """The uniform average. The control every other stacker must beat to justify itself."""

    def __init__(self, trim=0.0):
        self.trim = trim
        self.n_ = None

    def fit(self, P, y=None):
        self.n_ = np.asarray(P).shape[1]
        return self

    def predict(self, P):
        P = np.asarray(P, float)
        if self.trim <= 0:
            return P.mean(axis=1)
        k = int(np.floor(self.trim * P.shape[1]))
        if k == 0:
            return P.mean(axis=1)
        S = np.sort(P, axis=1)
        return S[:, k:P.shape[1] - k].mean(axis=1)

    @property
    def weights(self):
        return None if self.n_ is None else np.full(self.n_, 1.0 / self.n_)


class MedianStacker:
    """Per-row median. Robust to one member blowing up on a row, which is the failure the
    uniform mean handles worst."""

    def fit(self, P, y=None):
        return self

    def predict(self, P):
        return np.median(np.asarray(P, float), axis=1)

    @property
    def weights(self):
        return None


class RankAverageStacker:
    """Average the per-model RANKS, then map back onto the observed y distribution.

    Spearman rho is this project's headline metric and the one the RL reward's ordering
    depends on, and rank averaging optimises ordering directly: it is invariant to each
    member's calibration, so a model that ranks well but is offset by half a log unit
    contributes its ordering without dragging the mean. The inverse mapping onto the
    training y quantiles restores a pIC50 scale so downstream conformal and reward code
    still receive a number in the right units.
    """

    def __init__(self):
        self.y_sorted_ = None

    def fit(self, P, y):
        self.y_sorted_ = np.sort(np.asarray(y, float))
        return self

    def predict(self, P):
        from scipy.stats import rankdata
        P = np.asarray(P, float)
        R = np.mean([rankdata(P[:, j]) for j in range(P.shape[1])], axis=0)
        if self.y_sorted_ is None:
            return R
        # rank -> quantile -> the training y at that quantile
        q = (rankdata(R) - 0.5) / len(R)
        idx = np.clip((q * len(self.y_sorted_)).astype(int), 0, len(self.y_sorted_) - 1)
        return self.y_sorted_[idx]

    @property
    def weights(self):
        return None


STACKERS = {
    "mean": MeanStacker,
    "trimmed_mean": lambda: MeanStacker(trim=0.2),
    "median": MedianStacker,
    "nnls": NNLSStacker,
    "nnls_free": lambda: NNLSStacker(normalize=False),
    "ridge": RidgeStacker,
    "caruana": CaruanaSelector,
    # Spearman is this project's headline metric and the one the RL reward's ordering
    # depends on, and greedy selection is the only combiner here that can optimise a
    # non-differentiable rank metric DIRECTLY rather than hoping RMSE is a proxy for it.
    "caruana_spearman": lambda: CaruanaSelector(metric="spearman"),
    "caruana_bagged": BaggedCaruana,
    "caruana_bagged_spearman": lambda: BaggedCaruana(metric="spearman"),
    "rank_nnls": RankNNLSStacker,
    "rank_mean": RankAverageStacker,
}


# ------------------------------------------------------------- the discrete Super Learner


class DiscreteSuperLearner:
    """Choose, by cross-validation, among EVERY stacker AND every individual base model.

    THE SINGLE HIGHEST-LEVERAGE GUARD IN THIS MODULE. Phillips, van der Laan, Lee & Gruber,
    "Practical considerations for specifying a super learner", International Journal of
    Epidemiology (2023), step 5 of their flowchart: when an ensemble super learner is used,
    "we recommend it be evaluated as another candidate in a dSL, along with the individual
    learners from which it was constructed. This allows the eSL's CV performance to be
    compared to that from the other learners, and for the final SL to be the candidate
    learner that achieved the lowest CV risk."

    Why it matters here specifically. This project's whole predicament is that it shipped a
    combination -- a 10-member deep ensemble -- that is WORSE than a single plain model
    (Spearman 0.501 against the Morgan-count random forest's 0.686), and used it as the RL
    critic for a full HPC campaign. A discrete super learner makes that outcome
    structurally impossible: the random forest is itself a candidate, so the selected model
    can never be worse than the best single member by more than cross-validation noise.

    It is also the honest answer to "is the stack worth it?". Sometimes the answer is no,
    and a construction that cannot return "no" is not an experiment.
    """

    def __init__(self, metric="rmse", n_splits=5, seed=0, candidates=None):
        self.metric, self.n_splits, self.seed = metric, n_splits, seed
        self.candidates = candidates
        self.chosen_ = None
        self.scores_ = {}
        self.stacker_ = None
        self.names_ = None
        self.col_ = None

    def _loss(self, y, p):
        if self.metric == "rmse":
            return float(np.sqrt(np.mean((y - p) ** 2)))
        if self.metric == "mae":
            return float(np.mean(np.abs(y - p)))
        if self.metric == "spearman":
            from scipy import stats
            r = stats.spearmanr(y, p)[0]
            return -(0.0 if not np.isfinite(r) else float(r))
        raise ValueError(f"unknown metric {self.metric!r}")

    def fit(self, P, y, names=None, groups=None):
        from sklearn.model_selection import GroupKFold, KFold
        P = np.asarray(P, float)
        y = np.asarray(y, float)
        self.names_ = list(names) if names is not None else [f"m{j}" for j in range(P.shape[1])]
        n_splits = max(2, min(self.n_splits,
                              len(y) // 5 if groups is None else len(np.unique(groups))))
        folds = list(GroupKFold(n_splits=n_splits).split(P, y, groups) if groups is not None
                     else KFold(n_splits=n_splits, shuffle=True,
                                random_state=self.seed).split(P))

        scores: dict[str, float] = {}
        # --- candidate set part 1: every individual base model, used alone -------------
        # No fitting needed: a single column IS its own prediction, so its CV risk is just
        # its loss. Evaluated over the same folds so the comparison is paired.
        for j, nm in enumerate(self.names_):
            ls = [self._loss(y[te], P[te, j]) for _, te in folds]
            scores[f"single:{nm}"] = float(np.mean(ls))
        # --- candidate set part 2: every stacker --------------------------------------
        for sname in (self.candidates or STACKERS):
            ls = []
            for tr, te in folds:
                try:
                    st = STACKERS[sname]().fit(P[tr], y[tr])
                    ls.append(self._loss(y[te], st.predict(P[te])))
                except Exception as e:
                    warnings.warn(f"candidate {sname} failed on a fold: {e}",
                                  RuntimeWarning)
            scores[f"stack:{sname}"] = float(np.mean(ls)) if ls else float("inf")

        self.scores_ = scores
        self.chosen_ = min(scores, key=scores.get)
        if self.chosen_.startswith("single:"):
            self.col_ = self.names_.index(self.chosen_.split(":", 1)[1])
            self.stacker_ = None
        else:
            self.col_ = None
            self.stacker_ = STACKERS[self.chosen_.split(":", 1)[1]]().fit(P, y)
        return self

    def predict(self, P):
        P = np.asarray(P, float)
        return P[:, self.col_] if self.col_ is not None else self.stacker_.predict(P)

    def report(self) -> dict:
        ranked = sorted(self.scores_.items(), key=lambda kv: kv[1])
        best_single = min((v for k, v in self.scores_.items() if k.startswith("single:")),
                          default=float("nan"))
        best_stack = min((v for k, v in self.scores_.items() if k.startswith("stack:")),
                         default=float("nan"))
        out = {"chosen": self.chosen_, "metric": self.metric,
               "cv_risk": self.scores_.get(self.chosen_),
               "best_single_risk": best_single, "best_stack_risk": best_stack,
               "stack_beats_best_single": bool(best_stack < best_single),
               "ranking": [{"candidate": k, "risk": v} for k, v in ranked]}
        if self.stacker_ is not None:
            w = getattr(self.stacker_, "weights", None)
            if w is not None:
                out["weights"] = {n: float(v) for n, v in zip(self.names_, w)}
        return out


# ----------------------------------------------------------------------- Super Learner


def cv_select_stacker(P, y, groups=None, candidates=None, n_splits=5, seed=0,
                      metric="rmse") -> tuple[str, dict]:
    """Choose the meta-learner by cross-validation over the OOF matrix itself.

    This is the step that makes the construction a Super Learner (van der Laan et al. 2007)
    rather than "I picked NNLS". It is also a second place leakage can enter: the splits
    here must group by the SAME scaffold groups as the outer CV, or a scaffold's rows land
    on both sides of the meta-learner's own split and the chosen stacker is the one that
    best memorises the OOF matrix.

    Returns (name, {name: mean_score}).
    """
    from sklearn.model_selection import GroupKFold, KFold
    P = np.asarray(P, float)
    y = np.asarray(y, float)
    cand = list(candidates or STACKERS)
    n_splits = max(2, min(n_splits, len(y) // 5 if groups is None else len(np.unique(groups))))
    splitter = (GroupKFold(n_splits=n_splits).split(P, y, groups) if groups is not None
                else KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(P))
    folds = list(splitter)

    def loss(yy, pp):
        if metric == "rmse":
            return float(np.sqrt(np.mean((yy - pp) ** 2)))
        if metric == "spearman":
            from scipy import stats
            r = stats.spearmanr(yy, pp)[0]
            return -(0.0 if not np.isfinite(r) else float(r))
        return float(np.mean(np.abs(yy - pp)))

    scores = {}
    for name in cand:
        ls = []
        for tr, te in folds:
            try:
                st = STACKERS[name]().fit(P[tr], y[tr])
                ls.append(loss(y[te], st.predict(P[te])))
            except Exception as e:
                warnings.warn(f"stacker {name} failed on a fold: {e}", RuntimeWarning)
        scores[name] = float(np.mean(ls)) if ls else float("inf")
    best = min(scores, key=scores.get)
    return best, scores


class SuperLearner:
    """Fit the base library's OOF matrix with the CV-selected meta-learner.

    Holds the base model NAMES and the fitted base models so `predict` can featurise new
    SMILES, run every member and combine -- which is what makes this usable as the RL
    critic. See `tnk1.zoo.critic.ZooCritic` for the adapter that gives it the
    `predict_members` / `latent` interface `tnk1/rl/scorer.py` expects.
    """

    def __init__(self, stacker="auto", metric="rmse", seed=0):
        self.stacker_name, self.metric, self.seed = stacker, metric, seed
        self.stacker_ = None
        self.scores_ = {}
        self.names_ = None

    def fit(self, P, y, names=None, groups=None):
        self.names_ = list(names) if names is not None else None
        if self.stacker_name == "auto":
            best, self.scores_ = cv_select_stacker(P, y, groups, seed=self.seed,
                                                   metric=self.metric)
            self.chosen_ = best
        else:
            self.chosen_ = self.stacker_name
        self.stacker_ = STACKERS[self.chosen_]().fit(np.asarray(P, float), np.asarray(y, float))
        return self

    def predict(self, P):
        return self.stacker_.predict(np.asarray(P, float))

    def report(self) -> dict:
        w = getattr(self.stacker_, "weights", None)
        out = {"stacker": self.chosen_, "cv_scores": self.scores_}
        if w is not None:
            out["weights"] = ({n: float(v) for n, v in zip(self.names_, w)}
                              if self.names_ else [float(v) for v in w])
        return out
