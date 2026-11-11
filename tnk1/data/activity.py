"""Step 1.3: pActivity transform, replicate aggregation, right-censoring."""
from __future__ import annotations
import numpy as np
import pandas as pd

UNIT_TO_NM = {"nM": 1.0, "uM": 1e3, "µM": 1e3, "μM": 1e3, "pM": 1e-3, "mM": 1e6, "M": 1e9}


def to_pactivity(value_nM):
    """pActivity = 9 - log10(value in nM): 1 uM -> 6, 100 nM -> 7, 10 nM -> 8."""
    return 9.0 - np.log10(np.asarray(value_nM, dtype=float))


def prepare_records(df: pd.DataFrame) -> pd.DataFrame:
    """Input columns: inchikey_block1, smiles, scaffold, uniprot_id, standard_type, value_nM, source,
    and optionally relation. '=' (or a missing/blank relation) -> exact; '>'/'>=' -> censored, meaning
    the true pActivity lies BELOW the recorded bound; '<' records are dropped (left-censored potent
    values are not modelled by the manual's loss).

    Each censored row keeps its OWN bound, derived from its value_nM -- a ">1 uM" record bounds
    pActivity at 6, a ">100 uM" record at 4. There is deliberately no global tau here: the previous
    signature accepted `tau` and never used it (`run_stage1_data.py` passed `cfg.data.censor_tau`
    expecting an effect). `censor_tau` applies to newly assayed active-learning rows, where the
    protocol has a single threshold -- see `tnk1.active_learning.ingest.to_training_rows`.
    """
    d = df.copy()
    # `df.get("relation", "=")` returns the *scalar* "=" when the column is absent, and the
    # following .fillna() then raised AttributeError on a str. The PKIS and KINOMEscan inputs
    # declared in configs/default.yaml have no `relation` column, so this crashed on exactly
    # the datasets it was meant to accept.
    if "relation" in d.columns:
        rel = d["relation"].astype("object").where(d["relation"].notna(), "=")
        rel = rel.replace({"": "=", None: "="})
    else:
        rel = pd.Series("=", index=d.index, dtype="object")
    d["relation"] = rel
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
