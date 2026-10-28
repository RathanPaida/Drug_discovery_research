"""Sec 8.1 step 8.4: stratified 40-compound batch (16 exploit / 12 explore / 8 novel-scaffold / 4 control) over
Butina clusters with at most 2 analogs per cluster."""
from __future__ import annotations
import numpy as np
import pandas as pd


def butina_clusters(fps_or_smiles, cutoff=0.40) -> np.ndarray:
    """Butina sphere-exclusion on Morgan(r=2, 2048) Tanimoto distance; returns cluster id per molecule.
    cutoff=0.40 distance <=> similarity >= 0.60 to the centroid."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in fps_or_smiles]
    d = []
    for i in range(1, len(fps)):
        d.extend(1 - np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])))
    cl = Butina.ClusterData(d, len(fps), cutoff, isDistData=True)
    ids = np.zeros(len(fps), int)
    for c, members in enumerate(cl):
        ids[list(members)] = c                    # members[0] is the centroid
    return ids


def select_batch(pool: pd.DataFrame, n=None, max_per_cluster=2, known_scaffolds=(), seed=0) -> pd.DataFrame:
    """pool columns: smiles, cluster, scaffold, mu, sigma_epi, mu_sel, ei, ucb, lcb, passes_filters (bool),
    rf_pred (optional, for control). Returns the batch with a `batch_slice` column.
    Slices are filled in order EXPLOIT -> EXPLORE -> DIVERSITY -> CONTROL, never reusing a molecule and never
    exceeding `max_per_cluster` per cluster across the whole batch."""
    n = n or dict(exploit=16, explore=12, novel=8, control=4)
    pool = pool[pool.passes_filters].copy(); used, per_cluster, chosen = set(), {}, []
    known = set(known_scaffolds)

    if "mu" not in pool.columns and "mu_tnk1" in pool.columns:
        pool["mu"] = pool["mu_tnk1"]
    if "mu_sel" not in pool.columns:
        pool["mu_sel"] = 0.0

    def take(cands: pd.DataFrame, k, label):
        got = 0
        for idx, r in cands.iterrows():
            if got >= k: break
            if idx in used or per_cluster.get(r.cluster, 0) >= max_per_cluster: continue
            used.add(idx); per_cluster[r.cluster] = per_cluster.get(r.cluster, 0) + 1
            chosen.append((idx, label)); got += 1
        return got

    exploit_pool = pool[pool.mu_sel >= 1.5].sort_values("lcb", ascending=False)
    if len(exploit_pool) < n["exploit"]:
        exploit_pool = pool.sort_values("lcb", ascending=False)
    take(exploit_pool, n["exploit"], "EXPLOIT")

    explore_pool = pool[pool.mu >= 6.3].sort_values("ei", ascending=False)
    if len(explore_pool) < n["explore"]:
        explore_pool = pool.sort_values("ei", ascending=False)
    take(explore_pool, n["explore"], "EXPLORE")

    # centroids of Murcko scaffolds not represented in the training/in-house data or in the batch so far
    batch_scaf = {pool.loc[i, "scaffold"] for i, _ in chosen if "scaffold" in pool.columns}
    if "scaffold" in pool.columns:
        nov = pool[~pool.scaffold.isin(known | batch_scaf)].sort_values("ucb", ascending=False)
        nov = nov.drop_duplicates("scaffold")
    else:
        nov = pool.sort_values("ucb", ascending=False)
    take(nov, n["novel"], "DIVERSITY")
    ctrl = pool.sample(frac=1.0, random_state=seed)
    take(ctrl, n["control"], "CONTROL")             # random / baseline-RF pick passing identical filters
    out = pool.loc[[i for i, _ in chosen]].copy(); out["batch_slice"] = [l for _, l in chosen]
    return out
