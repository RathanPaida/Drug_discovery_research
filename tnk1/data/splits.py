"""Frozen evaluation splits: TNK1 scaffold hold-out + 5x5 nested scaffold CV + Leave-One-Kinase-Out.
Pure numpy/pandas (no RDKit) so it is unit-testable; scaffolds come from the standardization step."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def balanced_group_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Deterministic scaffold-group K-fold: largest groups first, each to the currently smallest fold."""
    rng = np.random.RandomState(seed)
    uniq, counts = np.unique(groups, return_counts=True)
    order = np.lexsort((rng.permutation(len(uniq)), -counts))      # size desc, random tie-break
    sizes = np.zeros(n_folds, dtype=int)
    assign = {}
    for i in order:
        f = int(np.argmin(sizes))
        assign[uniq[i]] = f
        sizes[f] += counts[i]
    return np.array([assign[g] for g in groups])


def scaffold_holdout(groups: np.ndarray, test_frac: float, seed: int) -> np.ndarray:
    """Boolean mask of the frozen test set (whole scaffold groups, ~test_frac of rows)."""
    k = max(2, int(round(1 / test_frac)))
    return balanced_group_folds(groups, k, seed) == 0


def nested_scaffold_cv(groups: np.ndarray, n_outer=5, n_inner=5, seed=42) -> list[dict]:
    outer = balanced_group_folds(groups, n_outer, seed)
    folds = []
    for o in range(n_outer):
        tr, te = np.where(outer != o)[0], np.where(outer == o)[0]
        inner = balanced_group_folds(groups[tr], n_inner, seed + 1 + o)
        folds.append({"outer": o, "train": tr.tolist(), "test": te.tolist(),
                      "inner": [{"train": tr[inner != i].tolist(), "val": tr[inner == i].tolist()}
                                for i in range(n_inner)]})
    return folds


def build_tnk1_splits(tnk1: pd.DataFrame, tnk1_uniprot: str, test_frac, n_outer, n_inner, seed) -> dict:
    """Operates on the TNK1 subset. Splits are stored by inchikey_block1 so they survive table rebuilds."""
    t = tnk1[tnk1.uniprot_id == tnk1_uniprot].reset_index(drop=True)
    keys = t.inchikey_block1.values
    # one group per scaffold; rows of the same compound (different assay types) share a scaffold => same fold
    groups = t.scaffold.values
    test_mask = scaffold_holdout(groups, test_frac, seed)
    dev = t[~test_mask].reset_index(drop=True)
    cv = nested_scaffold_cv(dev.scaffold.values, n_outer, n_inner, seed)
    to_keys = lambda idx: sorted(set(dev.inchikey_block1.values[idx]))
    out = {"seed": seed, "test_keys": sorted(set(keys[test_mask])),
           "dev_keys": sorted(set(dev.inchikey_block1)), "cv": []}
    for f in cv:
        out["cv"].append({"outer": f["outer"], "train": to_keys(f["train"]), "test": to_keys(f["test"]),
                          "inner": [{"train": to_keys(i["train"]), "val": to_keys(i["val"])} for i in f["inner"]]})
    return out


def kinome_train_mask(df: pd.DataFrame, exclude_compound_keys=(), exclude_uniprots=()) -> np.ndarray:
    """Leakage guard. Any compound that is in a TNK1 test fold is removed from ALL kinases' rows
    (the same molecule measured on SRC would otherwise leak its scaffold); for LOKO the held-out kinase's
    rows are removed entirely."""
    m = ~df.inchikey_block1.isin(set(exclude_compound_keys)).values
    if len(exclude_uniprots):
        m &= ~df.uniprot_id.isin(set(exclude_uniprots)).values
    return m


def loko_splits(df: pd.DataFrame, kinases: list[str]) -> dict:
    return {k: {"heldout_n": int((df.uniprot_id == k).sum()),
                "train_rows_excluded_uniprot": k} for k in kinases}


def assert_no_leakage(train_keys, test_keys, train_scaffolds=None, test_scaffolds=None):
    inter = set(train_keys) & set(test_keys)
    assert not inter, f"{len(inter)} compounds appear in both train and test"
    if train_scaffolds is not None:
        s = set(train_scaffolds) & set(test_scaffolds)
        assert not s, f"{len(s)} scaffolds shared between train and test"


def save_splits(splits: dict, path) -> None:
    Path(path).write_text(json.dumps(splits))
