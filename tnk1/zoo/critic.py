"""Make any zoo model usable as the RL critic, without touching `tnk1/rl/scorer.py`.

THE OPENING THIS EXPLOITS. `tnk1/rl/scorer.py` reaches the critic through exactly two
duck-typed methods:

    ens.predict_members(smiles, panel)  -> (mu, logvar), each (M, n_mol, n_target)
    ens.latent(smiles)                  -> (n_mol, d)        [for the applicability domain]

Nothing else in the scoring path knows the critic is a deep ensemble. So replacing the
critic -- which is the only one of the v3 run's three recommended fixes that changes a
scientific outcome -- is an adapter, not a rewrite. `PIPELINE_AUDIT.md` and
`RUN_REPORT_V3_HPC.md` both say so; this module is that adapter.

WHY REPLACE IT. Over 5 paired outer folds the 10-member multi-task D-MPNN ensemble reaches
Spearman 0.501 / RMSE 0.954 on TNK1, against 0.686 / 0.675 for a Morgan-count random
forest (paired per-row absolute error favouring RF at p = 8.6e-4), and its error-sigma
correlation is -0.021 against +0.258 (RF) and +0.283 (Tanimoto GP). The generator was
being optimised against the weakest model in the repository, and the 1.8-log optimism gap
the audit gate caught is the predictable consequence.

THE SYNTHETIC-MEMBER CONTRACT, STATED PLAINLY. `tnk1.uq.decomposition.decompose` expects M
member predictions and derives

    mu       = mean_m(mu_m)
    var_epi  = mean_m((mu_m - mu)^2)          spread ACROSS members
    var_ale  = mean_m(exp(logvar_m))          each member's own noise estimate

A random forest or a GP is one model, not M. Rather than pretend otherwise, this adapter
emits **two synthetic members placed at `mu +- sigma_epi`**, with `logvar` set to
`log(var_ale)`, which reproduces `(mu, sigma_epi, sigma_ale)` through `decompose` EXACTLY.
These are not independently trained members and must never be reported as an ensemble of
two: they are a carrier for an uncertainty this model estimated its own way (per-tree
spread for a forest, posterior variance for a GP). The quantity that reaches `r_pot` is
therefore the model's genuine uncertainty, and `meta.json` records `critic: zoo:<name>` so
no run can be mistaken for a deep-ensemble run.

PER-TARGET MODELS, AND WHAT IT COSTS. `r_sel` needs predictions for the off-target panel,
so this fits one model per panel kinase on that kinase's own rows. Compared with the
multi-task trunk that is a real loss -- no cross-kinase transfer, and a panel member with
few measurements gets a weak model -- and a real gain: each kinase gets a model whose
hyperparameters were selected for it, and the D-MPNN's transfer advantage did not survive
the paired comparison anyway. The per-target training-set sizes are recorded in `report()`
and a kinase below `min_rows` is dropped loudly rather than modelled from nothing, because
a fabricated off-target prediction flows straight into the selectivity term.
"""
from __future__ import annotations

import warnings

import numpy as np

from .registry import REGISTRY, build


class ZooCritic:
    """A drop-in replacement for `tnk1.uq.ensemble.Ensemble` backed by zoo models.

    Parameters
    ----------
    model : str
        Registry name, e.g. "rf", "tanimoto_gp", "lgbm".
    rep : str | None
        Representation key; defaults to the model spec's own.
    params : dict | None
        Hyperparameters, normally the ones the nested CV selected.
    min_rows : int
        A panel kinase with fewer exact measurements than this is not modelled.
    aleatoric : {"residual", "model", "fixed"}
        Where `var_ale` comes from. "residual" uses the model's own out-of-bag or
        cross-validated residual variance on the training set -- an honest estimate of
        irreducible noise. "model" asks the model for it (only meaningful for a GP with a
        learned likelihood noise). "fixed" uses `fixed_sigma_ale` and is for ablations.
    """

    def __init__(self, model="rf", rep=None, params=None, min_rows=30,
                 aleatoric="residual", fixed_sigma_ale=0.4, seed=42):
        self.model_name = model
        self.rep = rep or REGISTRY[model].rep
        self.params = dict(params or {})
        self.min_rows = int(min_rows)
        self.aleatoric = aleatoric
        self.fixed_sigma_ale = float(fixed_sigma_ale)
        self.seed = seed
        self.models_: dict[str, object] = {}
        self.sigma_ale_: dict[str, float] = {}
        self.n_rows_: dict[str, int] = {}
        self.targets_: list[str] = []
        self._latent_rep = "ecfp4_count"

    # ------------------------------------------------------------------------ fitting

    def fit(self, df, targets, tnk1=None, latent_rep="ecfp4_count"):
        """Fit one model per target from the frozen long table.

        `df` is the pan-kinome table (`Ctx.df`, already leakage-filtered by the caller --
        this method does NOT remove test compounds, because it cannot know which split is
        in play; pass `C.kinome_train_df()` or a fold's training rows).
        """
        from .features import featurize
        self._latent_rep = latent_rep
        self.targets_ = list(targets)
        d = df[df.censored == 0] if "censored" in df.columns else df
        for u in self.targets_:
            rows = d[d.uniprot_id == u]
            self.n_rows_[u] = int(len(rows))
            if len(rows) < self.min_rows:
                warnings.warn(
                    f"ZooCritic: target {u} has {len(rows)} exact measurements, below "
                    f"min_rows={self.min_rows}; it will be ABSENT from predictions rather "
                    f"than modelled. A fabricated off-target prediction goes straight into "
                    f"r_sel, so absence is the safe failure.", RuntimeWarning, stacklevel=2)
                continue
            X, ok = featurize(self.rep, list(rows.smiles))
            y = rows.pact.values[ok]
            X = X[ok]
            if len(y) < self.min_rows:
                continue
            m = build(self.model_name, self.params).fit(X, y)
            self.models_[u] = m
            self.sigma_ale_[u] = self._estimate_noise(m, X, y)
        if not self.models_:
            raise RuntimeError(
                f"ZooCritic fitted no model: every target had fewer than {self.min_rows} "
                f"exact rows ({self.n_rows_}). Lower min_rows only if you accept the "
                f"off-target predictions being near-meaningless.")
        if tnk1 is not None and tnk1 not in self.models_:
            raise RuntimeError(
                f"ZooCritic has no model for the PRIMARY target {tnk1}: it had "
                f"{self.n_rows_.get(tnk1, 0)} exact rows. r_pot would be undefined.")
        return self

    def _estimate_noise(self, m, X, y) -> float:
        """Aleatoric sd. Never returns 0, because `var_ale = 0` makes the reward's sigma
        penalty vanish and is exactly the fail-open pattern the audit removed."""
        if self.aleatoric == "fixed":
            return self.fixed_sigma_ale
        if self.aleatoric == "model":
            noise = getattr(m, "noise_sd", None)
            if noise is not None:
                return max(float(noise), 1e-3)
            warnings.warn(f"{self.model_name} exposes no learned noise; falling back to "
                          f"the residual estimate", RuntimeWarning, stacklevel=2)
        # Out-of-bag where available (a forest), else 5-fold CV residuals. IN-SAMPLE
        # residuals would badly understate the noise -- that is the quantity a flexible
        # model drives to zero.
        oob = getattr(getattr(m, "m", None), "oob_prediction_", None)
        if oob is not None:
            return max(float(np.std(np.asarray(y) - np.asarray(oob))), 1e-3)
        try:
            from sklearn.model_selection import KFold
            res = []
            for tr, va in KFold(5, shuffle=True, random_state=self.seed).split(X):
                mm = build(self.model_name, self.params).fit(X[tr], y[tr])
                p = mm.predict(X[va])
                res.append(np.asarray(y[va]) - np.asarray(p[0] if isinstance(p, tuple) else p))
            return max(float(np.std(np.concatenate(res))), 1e-3)
        except Exception as e:
            warnings.warn(f"ZooCritic noise estimate failed ({e}); using "
                          f"fixed_sigma_ale={self.fixed_sigma_ale}", RuntimeWarning, stacklevel=2)
            return self.fixed_sigma_ale

    # --------------------------------------------------------------- the Ensemble API

    def predict_members(self, smiles, uniprots, assay_idx=0):
        """-> (mu, logvar), each (2, n_mol, n_target). See the synthetic-member note above.

        A target with no model yields NaN for every molecule, so a downstream selectivity
        term computed over it is visibly undefined rather than quietly optimistic.
        """
        from .features import featurize
        smiles = list(smiles)
        uniprots = list(uniprots)
        n, K = len(smiles), len(uniprots)
        mu = np.full((n, K), np.nan)
        sd_epi = np.full((n, K), np.nan)
        lv = np.full((n, K), np.nan)
        X, ok = featurize(self.rep, smiles)
        for j, u in enumerate(uniprots):
            m = self.models_.get(u)
            if m is None:
                continue
            if not ok.any():
                continue
            out = m.predict(X[ok], return_std=True)
            if isinstance(out, tuple):
                mm, ss = np.asarray(out[0], float), np.asarray(out[1], float)
            else:
                mm, ss = np.asarray(out, float), np.zeros(int(ok.sum()))
            mu[ok, j] = mm
            sd_epi[ok, j] = np.maximum(ss, 1e-6)
            lv[ok, j] = 2 * np.log(max(self.sigma_ale_.get(u, self.fixed_sigma_ale), 1e-3))
        # Two members at mu +- sigma_epi reproduce (mu, var_epi) through
        # tnk1.uq.decomposition.decompose exactly; logvar carries var_ale.
        mu_m = np.stack([mu - sd_epi, mu + sd_epi])
        lv_m = np.stack([lv, lv])
        return mu_m, lv_m

    def latent(self, smiles):
        """The representation itself is the latent space for the applicability domain.

        The deep ensemble returns `z_mol(300) + desc(200)` from member 0. There is no
        learned latent here, so the AD is computed in the model's own input space -- which
        is arguably the more defensible choice: the applicability domain of a Morgan-count
        random forest IS the region of Morgan-count space its training set covers, and the
        d_lat explosion that `tnk1/uq/applicability.py` had to floor (distances up to
        240,831 against a threshold of 1.007) was a pathology of standardising a
        300-dimensional learned latent fitted on ~300 molecules.
        """
        from .features import featurize
        X, _ = featurize(self._latent_rep, list(smiles))
        return X.astype(np.float64)

    # -------------------------------------------------------------------- bookkeeping

    def report(self) -> dict:
        return {"critic": f"zoo:{self.model_name}", "representation": self.rep,
                "params": self.params, "aleatoric": self.aleatoric,
                "targets_modelled": sorted(self.models_),
                "targets_absent": sorted(set(self.targets_) - set(self.models_)),
                "n_train_rows": dict(self.n_rows_),
                "sigma_ale": {k: float(v) for k, v in self.sigma_ale_.items()},
                "synthetic_members": 2,
                "note": "two synthetic members carry (mu, sigma_epi); NOT a trained ensemble"}


class StackedCritic(ZooCritic):
    """A critic whose per-target prediction is a stack of several zoo models.

    This is the v3 report's Priority-3 item 3 ("back r_pot / r_sel with RF or a stacked
    RF/GP") in its stronger form. Per target: fit every base model, build leakage-safe
    out-of-fold predictions, fit the meta-learner on those, then refit the bases on all
    rows for deployment.

    Epistemic sigma for the stack is the WEIGHTED SPREAD of its members around the stacked
    prediction, which is the quantity that matters for reward hacking: the generator is
    penalised precisely where the model classes disagree, and cross-architecture
    disagreement is what the audit gate caught at 1.8 log units after the fact. Putting it
    inside `sigma_epi` moves that signal from a post-hoc filter into the training reward.
    """

    def __init__(self, models=("rf", "tanimoto_gp", "lgbm"), stacker="nnls", rep=None,
                 params=None, min_rows=30, n_folds=5, seed=42, fixed_sigma_ale=0.4):
        super().__init__(model=models[0], rep=rep, params=params, min_rows=min_rows,
                         aleatoric="residual", fixed_sigma_ale=fixed_sigma_ale, seed=seed)
        self.model_names = list(models)
        self.stacker_name = stacker
        self.n_folds = int(n_folds)
        self.stack_: dict[str, object] = {}
        self.members_: dict[str, list] = {}
        self.weights_: dict[str, dict] = {}

    def fit(self, df, targets, tnk1=None, latent_rep="ecfp4_count"):
        from sklearn.model_selection import KFold
        from ..ensembles.stacking import STACKERS
        from .features import featurize
        self._latent_rep = latent_rep
        self.targets_ = list(targets)
        d = df[df.censored == 0] if "censored" in df.columns else df
        for u in self.targets_:
            rows = d[d.uniprot_id == u]
            self.n_rows_[u] = int(len(rows))
            if len(rows) < self.min_rows:
                warnings.warn(f"StackedCritic: target {u} has {len(rows)} exact rows "
                              f"(< {self.min_rows}); absent from predictions",
                              RuntimeWarning, stacklevel=2)
                continue
            fitted, oof_cols, names = [], [], []
            for name in self.model_names:
                rep = REGISTRY[name].rep if self.rep is None else self.rep
                X, ok = featurize(rep, list(rows.smiles))
                X, y = X[ok], rows.pact.values[ok]
                if len(y) < self.min_rows:
                    continue
                oof = np.full(len(y), np.nan)
                for tr, va in KFold(min(self.n_folds, len(y)), shuffle=True,
                                    random_state=self.seed).split(X):
                    mm = build(name, self.params).fit(X[tr], y[tr])
                    p = mm.predict(X[va])
                    oof[va] = np.asarray(p[0] if isinstance(p, tuple) else p, float)
                if not np.isfinite(oof).all():
                    continue
                fitted.append((name, rep, build(name, self.params).fit(X, y)))
                oof_cols.append(oof)
                names.append(name)
            if not fitted:
                continue
            P = np.column_stack(oof_cols)
            yv = rows.pact.values[:len(P)]
            st = STACKERS[self.stacker_name]().fit(P, yv)
            self.stack_[u] = st
            self.members_[u] = fitted
            w = getattr(st, "weights", None)
            self.weights_[u] = ({n: float(v) for n, v in zip(names, w)} if w is not None
                                else {n: 1.0 / len(names) for n in names})
            # Noise: residual of the STACK's out-of-fold prediction, not of any member.
            self.sigma_ale_[u] = max(float(np.std(yv - st.predict(P))), 1e-3)
        if not self.stack_:
            raise RuntimeError("StackedCritic fitted no target; check min_rows and the "
                               "available model dependencies")
        if tnk1 is not None and tnk1 not in self.stack_:
            raise RuntimeError(f"StackedCritic has no model for the primary target {tnk1}")
        return self

    def predict_members(self, smiles, uniprots, assay_idx=0):
        from .features import featurize
        smiles, uniprots = list(smiles), list(uniprots)
        n, K = len(smiles), len(uniprots)
        mu = np.full((n, K), np.nan)
        sd = np.full((n, K), np.nan)
        lv = np.full((n, K), np.nan)
        cache: dict[str, tuple] = {}
        for j, u in enumerate(uniprots):
            st = self.stack_.get(u)
            if st is None:
                continue
            cols, w = [], []
            weights = self.weights_[u]
            for name, rep, m in self.members_[u]:
                if rep not in cache:
                    cache[rep] = featurize(rep, smiles)
                X, ok = cache[rep]
                p = np.full(n, np.nan)
                if ok.any():
                    out = m.predict(X[ok])
                    p[ok] = np.asarray(out[0] if isinstance(out, tuple) else out, float)
                cols.append(p)
                w.append(weights.get(name, 0.0))
            P = np.column_stack(cols)
            good = np.isfinite(P).all(axis=1)
            mu[good, j] = st.predict(P[good])
            # weighted spread of the members about the stacked value: the disagreement term
            wv = np.asarray(w, float)
            wv = wv / max(wv.sum(), 1e-12)
            dev = (P[good] - mu[good, j][:, None]) ** 2
            sd[good, j] = np.maximum(np.sqrt((dev * wv[None, :]).sum(axis=1)), 1e-6)
            lv[good, j] = 2 * np.log(max(self.sigma_ale_.get(u, self.fixed_sigma_ale), 1e-3))
        return np.stack([mu - sd, mu + sd]), np.stack([lv, lv])

    def report(self) -> dict:
        r = super().report()
        r.update({"critic": f"zoo_stack:{'+'.join(self.model_names)}",
                  "stacker": self.stacker_name,
                  "weights_per_target": self.weights_,
                  "targets_modelled": sorted(self.stack_),
                  "targets_absent": sorted(set(self.targets_) - set(self.stack_)),
                  "note": "sigma_epi is the weighted member spread = cross-architecture "
                          "disagreement, so the reward penalises it during training rather "
                          "than filtering on it afterwards"})
        return r
