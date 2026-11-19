"""Reward interfaces for the DAP loop.

A scorer is any callable `f(list[str]) -> {"score": array, "valid": array, "scaffold": list}`.
Keeping it this narrow is deliberate: the RL loop never imports torch models, a feature
store, or a REST client, so it stays runnable while the heavy pipeline is being repaired,
and `StandaloneScorer` vs `PipelineScorer` become a one-line swap.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")        # invalid SMILES are expected and counted, not printed


def _parse(smiles: str):
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def murcko(mol) -> str | None:
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return None


class StandaloneScorer:
    """Reward with no dependency on the feature store or trained surrogates.

    Use it to validate the RL machinery itself -- that the agent can shift its output
    distribution under reward pressure -- before trusting a reward built on a surrogate
    whose test-set Spearman is 0.264. Debugging RL and debugging a weak critic at the same
    time is how a pipeline ends up with neither working.

    Geometric mean of QED, a molecular-weight window, and presence of `motif`, so the
    shape matches the production reward (multiplicative, every term on [0,1]).
    """

    def __init__(self, motif_smarts: str = "c1ncnc(N)c1", mw_lo=250.0, mw_hi=550.0):
        self.motif = Chem.MolFromSmarts(motif_smarts) if motif_smarts else None
        self.mw_lo, self.mw_hi = mw_lo, mw_hi

    def __call__(self, smiles: list[str]) -> dict:
        n = len(smiles)
        score = np.zeros(n)
        valid = np.zeros(n, dtype=bool)
        scaf: list = [None] * n
        qed_a, motif_a = np.zeros(n), np.zeros(n)
        for i, s in enumerate(smiles):
            m = _parse(s)
            if m is None or m.GetNumHeavyAtoms() < 6:
                continue
            valid[i] = True
            scaf[i] = murcko(m)
            q = float(QED.qed(m))
            mw = float(Descriptors.MolWt(m))
            # soft trapezoid on MW rather than a hard cut: a hard cut gives zero gradient
            # on either side, so the agent gets no signal about which direction to move.
            w = float(np.clip(min((mw - self.mw_lo + 100) / 100, (self.mw_hi + 100 - mw) / 100), 0, 1))
            hit = 1.0 if (self.motif is not None and m.HasSubstructMatch(self.motif)) else 0.0
            qed_a[i], motif_a[i] = q, hit
            terms = [max(q, 1e-4), max(w, 1e-4), max(hit, 1e-4)]
            score[i] = float(np.exp(np.mean(np.log(terms))))
        return {"score": score, "valid": valid, "scaffold": scaf,
                "qed": qed_a, "motif": motif_a}


class PipelineScorer:
    """Adapter onto the production `tnk1.rl.scorer.Scorer`.

    Needs the feature store and every Stage 3-5 checkpoint. `advance=True` makes each call
    one RL step so the beta schedule (exploration -> pessimism) tracks the agent's progress,
    which is the behaviour Stage 7 was specified to have.
    """

    # `cache` semantics -- the default changed, deliberately.
    #
    #   "step" (default) : de-duplicate WITHIN one call. A DAP batch routinely repeats the
    #                      same SMILES several times, so this still removes most of the
    #                      scoring cost, and it is exactly as correct as no cache at all.
    #   "run"            : persist across steps. FASTER AND WRONG unless beta is constant
    #                      -- see below. Opt in only for offline pools / fixed beta.
    #   False / None      : score every element, duplicates included.
    #
    # WHY "run" IS WRONG DURING RL. `Scorer.score` recomputes beta from its own step
    # counter (`tnk1/rl/scorer.py`, beta_schedule) and feeds it to `r_pot`, which scores
    # `mu - beta * sigma_epi`. Over a 1000-step run beta moves 0.3 -> 1.5: the whole point
    # of the schedule is that a molecule is judged optimistically while exploring and
    # pessimistically once converging. A SMILES-keyed cache that survives across steps
    # pins each molecule's reward to the beta in force the first time it was seen, so a
    # molecule sampled at step 5 keeps its exploration-era reward at step 900 while a
    # molecule first seen at step 900 is judged pessimistically. The agent is then trained
    # on two different objectives at once, with the bias favouring whatever it happened to
    # emit early -- silent, and in the direction of over-confidence.
    _CACHE_MODES = ("step", "run")

    def __init__(self, scorer, cache: str | bool = "step"):
        if cache is True:
            cache = "run"
        elif cache is False or cache is None:
            cache = None
        if cache is not None and cache not in self._CACHE_MODES:
            raise ValueError(f"cache must be one of {self._CACHE_MODES}, False or None; got {cache!r}")
        if cache == "run":
            import warnings
            warnings.warn(
                "PipelineScorer(cache='run') freezes each molecule's reward at the beta in "
                "force when it was first scored; beta_schedule moves 0.3 -> 1.5 during RL, "
                "so this biases the objective toward early samples. Use it only when beta "
                "is constant (offline scoring at beta_final).",
                RuntimeWarning, stacklevel=2,
            )
        self.scorer = scorer
        self.mode = cache
        self._run_cache: dict[str, float] = {}
        self.n_scored = 0          # molecules actually pushed through the surrogate
        self.n_calls = 0           # == RL steps, so the beta schedule can be audited

    def static(self) -> "_StaticView":
        """A non-advancing view, for scoring that is not an RL step.

        The prior baseline in `rl_min.run_rl` scores 2000 molecules before training starts.
        Routed through `__call__` that would tick the beta schedule and leave the agent's
        first real step running at step 2 -- the baseline would also be measured under a
        different beta than step 1, which is precisely the comparison it exists to make.
        `Scorer.score(advance=False)` uses `beta_final`, the converged value, so prior and
        agent are judged on the same scale.
        """
        return _StaticView(self)

    def __call__(self, smiles: list[str]) -> dict:
        self.n_calls += 1
        n = len(smiles)
        mols = [_parse(s) for s in smiles]

        known = self._run_cache if self.mode == "run" else {}
        if self.mode is None:
            todo = list(smiles)
        else:
            # dict.fromkeys preserves order and removes duplicates; the previous list
            # comprehension sent repeats through the surrogate once per occurrence.
            todo = [s for s in dict.fromkeys(smiles) if s not in known]

        fresh: dict[str, float] = {}
        if todo:
            # ONE score() call per __call__, so `advance=True` ticks the beta schedule
            # exactly once per RL step. Previously a batch whose every molecule was already
            # cached skipped score() entirely and the schedule stalled -- beta then lagged
            # the true step count by however many all-cached batches had gone by.
            out = self.scorer.score(todo, advance=True)
            r = np.asarray(out["R"], dtype=float)
            if r.shape[0] != len(todo):
                raise ValueError(f"scorer returned {r.shape[0]} rewards for {len(todo)} molecules")
            fresh = {s: float(v) for s, v in zip(todo, r)}
            self.n_scored += len(todo)
        elif self.mode is not None:
            # Nothing new to score, but the step still happened: advance the schedule on an
            # empty batch rather than letting beta fall behind.
            advance = getattr(self.scorer, "advance_step", None)
            if callable(advance):
                advance()

        lookup = {**known, **fresh}
        # An unparseable SMILES is never sent to the surrogate and must score 0.0, not
        # inherit a neighbour's reward. `.get(s, 0.0)` also removes the NameError the
        # previous `out["R"]` branch raised when `smiles` was empty and caching was off.
        if self.mode is None:
            score = np.array([fresh.get(s, 0.0) for s in smiles], dtype=float)
        else:
            score = np.array([lookup.get(s, 0.0) for s in smiles], dtype=float)
        if self.mode == "run":
            self._run_cache.update(fresh)

        return {"score": score,
                "valid": np.array([m is not None for m in mols], dtype=bool),
                "scaffold": [murcko(m) if m is not None else None for m in mols],
                "n_scored": self.n_scored,
                "n_unique_in_batch": len(set(smiles)) if n else 0}


class _StaticView:
    """Scores through a `PipelineScorer`'s surrogate without advancing the beta schedule.

    Returned by `PipelineScorer.static()`; see that method for why the distinction matters.
    """

    def __init__(self, parent: "PipelineScorer"):
        self.parent = parent

    def __call__(self, smiles: list[str]) -> dict:
        mols = [_parse(s) for s in smiles]
        uniq = list(dict.fromkeys(smiles))
        lookup: dict[str, float] = {}
        if uniq:
            out = self.parent.scorer.score(uniq, advance=False)
            lookup = {s: float(v) for s, v in zip(uniq, np.asarray(out["R"], dtype=float))}
        return {"score": np.array([lookup.get(s, 0.0) for s in smiles], dtype=float),
                "valid": np.array([m is not None for m in mols], dtype=bool),
                "scaffold": [murcko(m) if m is not None else None for m in mols]}
