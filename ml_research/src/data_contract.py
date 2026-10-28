#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Team A <-> Team B Shared Data Contract & Schema Validator (Page 10 Reference Blueprint)
Enforces the frozen interface schema for closed-loop DMTA cycles:
- compound_id / cycle_id
- canonical_smiles / inchikey
- batch_slice (EXPLOIT, EXPLORE, DIVERSITY, CONTROL, REF_STD)
- pred_mu_tnk1 / pred_sigma_epi
- pred_ci90_low / pred_ci90_high
"""

import re
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

VALID_BATCH_SLICES = {"EXPLOIT", "EXPLORE", "DIVERSITY", "CONTROL", "REF_STD"}

CONTRACT_COLUMNS = [
    "compound_id",
    "cycle_id",
    "canonical_smiles",
    "inchikey",
    "batch_slice",
    "pred_mu_tnk1",
    "pred_sigma_epi",
    "pred_ci90_low",
    "pred_ci90_high"
]

def validate_candidate_batch(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validates that a generated candidate batch adheres to the frozen interface contract."""
    errors = []
    
    # 1. Required columns
    for col in CONTRACT_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required contract column: {col}")
            
    if errors:
        return False, errors
        
    # 2. Check compound_id format (e.g. TNK1-C01-001)
    id_pattern = re.compile(r"^TNK1-C\d{2}-\d{3,}$")
    invalid_ids = [cid for cid in df['compound_id'] if not id_pattern.match(str(cid))]
    if invalid_ids:
        errors.append(f"Invalid compound_id format (expected TNK1-C01-001): {invalid_ids[:3]}")
        
    # 3. Check batch_slice enum values
    invalid_slices = set(df['batch_slice']) - VALID_BATCH_SLICES
    if invalid_slices:
        errors.append(f"Invalid batch_slice enum values: {invalid_slices}")
        
    # 4. Check confidence bounds: pred_ci90_low <= pred_mu_tnk1 <= pred_ci90_high
    invalid_bounds = df[
        (df['pred_ci90_low'] > df['pred_mu_tnk1']) | 
        (df['pred_mu_tnk1'] > df['pred_ci90_high'])
    ]
    if len(invalid_bounds) > 0:
        errors.append(f"Found {len(invalid_bounds)} rows where CI90 bounds violate pred_ci90_low <= pred_mu <= pred_ci90_high")
        
    return len(errors) == 0, errors

def format_candidate_batch(
    compound_smiles: List[str],
    inchikeys: List[str],
    cycle_id: int,
    batch_slices: List[str],
    pred_mu: np.ndarray,
    pred_sigma_epi: np.ndarray,
    conformal_quantile: float = 1.645
) -> pd.DataFrame:
    """Formats raw predictions into the Team A -> Team B Data Contract table."""
    n = len(compound_smiles)
    compound_ids = [f"TNK1-C{cycle_id:02d}-{i+1:03d}" for i in range(n)]
    
    ci_low = pred_mu - conformal_quantile * pred_sigma_epi
    ci_high = pred_mu + conformal_quantile * pred_sigma_epi
    
    df = pd.DataFrame({
        "compound_id": compound_ids,
        "cycle_id": [cycle_id] * n,
        "canonical_smiles": compound_smiles,
        "inchikey": inchikeys,
        "batch_slice": batch_slices,
        "pred_mu_tnk1": pred_mu.astype(float),
        "pred_sigma_epi": pred_sigma_epi.astype(float),
        "pred_ci90_low": ci_low.astype(float),
        "pred_ci90_high": ci_high.astype(float)
    })
    return df
