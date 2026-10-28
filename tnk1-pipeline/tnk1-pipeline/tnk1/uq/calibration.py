"""Sec 4.2: variance temperature scaling + split-conformal prediction on normalised residuals (numpy only)."""
from __future__ import annotations
import math
import numpy as np


def fit_temperature(y, mu, sigma_tot) -> float:
    """argmin_tau Gaussian NLL of N(mu, (tau*sigma)^2) has the closed form tau^2 = mean(((y-mu)/sigma)^2)."""
    r2 = ((np.asarray(y) - np.asarray(mu)) / np.maximum(sigma_tot, 1e-6)) ** 2
    return float(math.sqrt(max(r2.mean(), 1e-12)))


def conformal_quantile(y_cal, mu_cal, sigma_cal, alpha=0.10) -> float:
    """Split-conformal on normalised nonconformity s = |y-mu|/sigma_cal with finite-sample correction
    q_hat = the ceil((n+1)(1-alpha))/n empirical quantile."""
    s = np.abs(np.asarray(y_cal) - np.asarray(mu_cal)) / np.maximum(sigma_cal, 1e-6)
    n = len(s)
    k = min(n, math.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(s)[k - 1])


class Calibrator:
    """fit on out-of-fold validation predictions; apply to new predictions."""
    def __init__(self, alpha=0.10): self.alpha, self.tau, self.q = alpha, 1.0, None

    def fit(self, y, mu, sigma_tot, split=0.5, seed=0):
        """Temperature on one half, conformal quantile on the other (keeps exchangeability clean)."""
        rng = np.random.RandomState(seed); idx = rng.permutation(len(y)); h = len(y) // 2
        a, b = idx[:h], idx[h:]
        self.tau = fit_temperature(y[a], mu[a], sigma_tot[a])
        self.q = conformal_quantile(y[b], mu[b], self.tau * sigma_tot[b], self.alpha)
        return self

    def refit_quantile(self, y, mu, sigma_tot):
        """Per-cycle recalibration (Stage 8.2) keeping the temperature."""
        self.q = conformal_quantile(y, mu, self.tau * sigma_tot, self.alpha); return self

    def sigma(self, sigma_tot): return self.tau * np.asarray(sigma_tot)

    def interval(self, mu, sigma_tot):
        s = self.sigma(sigma_tot); return mu - self.q * s, mu + self.q * s
