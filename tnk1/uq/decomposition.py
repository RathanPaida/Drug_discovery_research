"""Sec 4.1: law-of-total-variance decomposition for a heteroscedastic deep ensemble (numpy only)."""
from __future__ import annotations
import numpy as np


def decompose(mu_m: np.ndarray, logvar_m: np.ndarray) -> dict:
    """mu_m, logvar_m: (M, ...) member predictions. Returns mean, aleatoric var (mean exp(s)),
    epistemic var (variance of member means, population form 1/M), total std."""
    mu = mu_m.mean(0)
    ale = np.exp(logvar_m).mean(0)
    epi = ((mu_m - mu) ** 2).mean(0)
    return {"mu": mu, "var_ale": ale, "var_epi": epi, "sigma_ale": np.sqrt(ale), "sigma_epi": np.sqrt(epi),
            "sigma_tot": np.sqrt(ale + epi)}


def mc_dropout_summary(passes: np.ndarray) -> dict:
    """passes: (T, n) stochastic forward passes with dropout active."""
    return {"mu": passes.mean(0), "sigma_epi": passes.std(0), "var_epi": passes.var(0)}


def nig_summary(gamma, nu, alpha, beta) -> dict:
    """Deep evidential regression: E[mu]=gamma, E[s^2]=beta/(alpha-1), Var[mu]=beta/(nu(alpha-1))."""
    ale = beta / (alpha - 1); epi = beta / (nu * (alpha - 1))
    return {"mu": gamma, "var_ale": ale, "var_epi": epi, "sigma_tot": np.sqrt(ale + epi)}
