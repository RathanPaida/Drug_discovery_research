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

    def __init__(self, scorer, cache: bool = True):
        self.scorer, self._cache = scorer, ({} if cache else None)

    def __call__(self, smiles: list[str]) -> dict:
        todo = [s for s in smiles if self._cache is None or s not in self._cache]
        if todo:
            out = self.scorer.score(todo, advance=True)
            if self._cache is not None:
                for j, s in enumerate(todo):
                    self._cache[s] = float(out["R"][j])
        if self._cache is None:
            score = np.asarray(out["R"], dtype=float)
        else:
            score = np.array([self._cache.get(s, 0.0) for s in smiles], dtype=float)
        mols = [_parse(s) for s in smiles]
        return {"score": score,
                "valid": np.array([m is not None for m in mols]),
                "scaffold": [murcko(m) if m is not None else None for m in mols]}
