"""Shared Team A <-> Team B data contract (frozen Week 1)."""
from __future__ import annotations
import re
import numpy as np, pandas as pd

SLICES = {"EXPLOIT", "EXPLORE", "DIVERSITY", "CONTROL", "REF_STD"}
COLUMNS = ["compound_id", "cycle_id", "canonical_smiles", "inchikey", "batch_slice",
           "pred_mu_tnk1", "pred_sigma_epi", "pred_ci90_low", "pred_ci90_high"]


def build_contract(batch: pd.DataFrame, cycle: int, calibrator, ik_fn) -> pd.DataFrame:
    mu_vals = batch.mu.values if "mu" in batch.columns else batch.mu_tnk1.values
    sig_epi = batch.sigma_epi.values
    sig_ale = batch.sigma_ale.values if "sigma_ale" in batch.columns else 0.0
    lo, hi = calibrator.interval(mu_vals, np.sqrt(sig_epi ** 2 + sig_ale ** 2))
    df = pd.DataFrame({
        "compound_id": [f"TNK1-C{cycle:02d}-{i + 1:03d}" for i in range(len(batch))], "cycle_id": cycle,
        "canonical_smiles": batch.smiles.values, "inchikey": [ik_fn(s) for s in batch.smiles],
        "batch_slice": batch.batch_slice.values, "pred_mu_tnk1": mu_vals,
        "predicted_pic50": mu_vals,
        "pred_sigma_epi": sig_epi, "pred_ci90_low": lo, "pred_ci90_high": hi})
    if "ppid_score" in batch.columns:
        df["ppid_score"] = batch.ppid_score.values
        df["ppi_score"] = batch.ppi_score.values if "ppi_score" in batch.columns else batch.ppid_score.values
    validate_contract(df); return df


def validate_contract(df: pd.DataFrame) -> None:
    missing = set(COLUMNS) - set(df.columns); assert not missing, f"missing columns: {missing}"
    assert df.compound_id.is_unique, "duplicate compound_id"
    assert df.compound_id.str.match(r"^TNK1-C\d{2}-\d{3}$").all(), "compound_id must look like TNK1-C01-001"
    assert df.inchikey.str.len().eq(27).all(), "InChIKey must be 27 characters"
    assert set(df.batch_slice) <= SLICES, f"bad batch_slice values {set(df.batch_slice) - SLICES}"
    assert (df.pred_ci90_low <= df.pred_mu_tnk1).all() and (df.pred_mu_tnk1 <= df.pred_ci90_high).all()
    assert df[["pred_mu_tnk1", "pred_sigma_epi"]].notna().all().all()
