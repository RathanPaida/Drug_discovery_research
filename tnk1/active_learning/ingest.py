"""Sec 8.2: Evaluate-Before-Train protocol for returning wet-lab results."""
from __future__ import annotations
import numpy as np, pandas as pd
from ..uq.metrics import rmse, spearman


def evaluate_prospective(results: pd.DataFrame, calibrator) -> dict:
    """results: contract columns + measured pIC50 `pact` (+ `censored`) from Team B (purity >= 95%).
    Scored with the FROZEN pre-batch predictions already in the contract."""
    r = results[results.purity_pct >= 95].copy(); ex = r[r.censored == 0]
    inside = (ex.pact >= ex.pred_ci90_low) & (ex.pact <= ex.pred_ci90_high)
    out = {"n": len(r), "rmse": rmse(ex.pact, ex.pred_mu_tnk1) if len(ex) else np.nan,
           "spearman": spearman(ex.pact, ex.pred_mu_tnk1) if len(ex) > 2 else np.nan,
           "cov90_actual": float(inside.mean()) if len(ex) else np.nan,
           "hit_1uM": int(((r.pact >= 6.0) & (r.censored == 0)).sum()), "hit_100nM": int(((r.pact >= 7.0) & (r.censored == 0)).sum())}
    out["by_slice"] = r.groupby("batch_slice").apply(
        lambda g: pd.Series({"n": len(g), "hit_1uM": int(((g.pact >= 6.0) & (g.censored == 0)).sum()),
                             "hit_100nM": int(((g.pact >= 7.0) & (g.censored == 0)).sum())})).to_dict("index")
    return out


def to_training_rows(results: pd.DataFrame, inhouse_weight=3.0, tnk1_uniprot="Q13470", censor_tau=5.0) -> pd.DataFrame:
    """Inactives (> 10 uM) become right-censored labels at tau=5.0; in-house rows get sample weight 3.0."""
    r = results[results.purity_pct >= 95].copy()
    r["censored"] = r.get("censored", 0).fillna(0).astype(int)
    r.loc[r.censored == 1, "pact"] = censor_tau
    return pd.DataFrame({"smiles": r.canonical_smiles, "inchikey_block1": r.inchikey.str[:14], "scaffold": r.get("scaffold", r.inchikey.str[:14]),
                         "uniprot_id": tnk1_uniprot, "standard_type": r.get("assay_type", "IC50"), "pact": r.pact,
                         "censored": r.censored, "n_rep": 1, "sigma_rep": np.nan, "source": "inhouse",
                         "weight": inhouse_weight})
