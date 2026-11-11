"""Sec 5.1: ensemble-propagated selectivity. Margins are computed INSIDE each member, then aggregated,
which preserves cross-kinase covariance."""
from __future__ import annotations
import numpy as np


def member_margin(mu_tnk1: np.ndarray, mu_off: np.ndarray) -> np.ndarray:
    """mu_tnk1: (M, n); mu_off: (M, n, K) -> dS_m(x) = mu_m(x,TNK1) - max_k mu_m(x,k)."""
    return mu_tnk1 - mu_off.max(-1)


def soft_margin(mu_tnk1, mu_off, temp=1.0):
    """Soft (LogSumExp) margin: mu_tnk1 - temp*log sum_k exp(mu_k/temp)."""
    m = mu_off / temp
    lse = temp * (np.log(np.exp(m - m.max(-1, keepdims=True)).sum(-1)) + m.max(-1))
    return mu_tnk1 - lse


def selectivity_stats(mu_tnk1, mu_off, soft=False, temp=1.0) -> dict:
    dS = soft_margin(mu_tnk1, mu_off, temp) if soft else member_margin(mu_tnk1, mu_off)
    mu_sel = dS.mean(0)
    return {"mu_sel": mu_sel, "sigma_sel": np.sqrt(((dS - mu_sel) ** 2).mean(0)), "fold": 10 ** mu_sel}


def sanity_check_pan_kinase(mu_sel_by_name: dict, names=("dasatinib", "bosutinib")) -> dict:
    """Known pan-kinase inhibitors must satisfy mu_sel <= 0."""
    return {n: bool(mu_sel_by_name[n] <= 0) for n in names if n in mu_sel_by_name}
