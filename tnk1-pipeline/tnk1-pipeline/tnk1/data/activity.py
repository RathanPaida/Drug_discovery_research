"""Step 1.3: pActivity transform, replicate aggregation, right-censoring."""
from __future__ import annotations
import numpy as np
import pandas as pd

UNIT_TO_NM = {"nM": 1.0, "uM": 1e3, "µM": 1e3, "μM": 1e3, "pM": 1e-3, "mM": 1e6, "M": 1e9}


def to_pactivity(value_nM):
    """pActivity = 9 - log10(value in nM): 1 uM -> 6, 100 nM -> 7, 10 nM -> 8."""
    return 9.0 - np.log10(np.asarray(value_nM, dtype=float))


def prepare_records(df: pd.DataFrame, tau: float = 5.0) -> pd.DataFrame:
    """Input columns: inchikey_block1, smiles, scaffold, uniprot_id, standard_type, value_nM, relation, source.
    '=' -> exact; '>'/'>=' -> right-censored (true pActivity < bound); '<' records are dropped
    (left-censored potent values are not modelled by the manual's loss)."""
    d = df.copy()
    d["relation"] = d.get("relation", "=").fillna("=").replace({"": "="})
    d = d[d["value_nM"] > 0]
    d = d[~d["relation"].isin(["<", "<=", "<<"])]
    d["censored"] = d["relation"].isin([">", ">=", ">>"]).astype(int)
    d["pact"] = to_pactivity(d["value_nM"])
    return d


def aggregate_replicates(d: pd.DataFrame, range_max: float = 1.0, policy: str = "drop") -> pd.DataFrame:
    """Median pActivity per (inchikey_block1, uniprot_id, standard_type). If exact measurements exist they
    take precedence; else the tightest censor bound is kept. Replicate sets spanning > range_max are dropped
    (policy='drop') or kept at the median (policy='keep_median'); sigma_rep and n_rep are stored."""
    keys = ["inchikey_block1", "uniprot_id", "standard_type"]
    rows = []
    for k, g in d.groupby(keys, sort=False):
        ex = g[g.censored == 0]
        if len(ex):
            rng = ex.pact.max() - ex.pact.min()
            if len(ex) > 1 and rng > range_max and policy == "drop":
                continue
            rows.append({**dict(zip(keys, k)), "smiles": ex.smiles.iloc[0], "scaffold": ex.scaffold.iloc[0],
                         "pact": float(ex.pact.median()), "censored": 0, "n_rep": len(ex),
                         "sigma_rep": float(ex.pact.std(ddof=0)) if len(ex) > 1 else np.nan,
                         "source": "|".join(sorted(set(ex.source.astype(str))))})
        else:
            rows.append({**dict(zip(keys, k)), "smiles": g.smiles.iloc[0], "scaffold": g.scaffold.iloc[0],
                         "pact": float(g.pact.min()), "censored": 1, "n_rep": len(g), "sigma_rep": np.nan,
                         "source": "|".join(sorted(set(g.source.astype(str))))})
    out = pd.DataFrame(rows)
    out["weight"] = 1.0
    return out
