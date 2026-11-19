"""Regression + uncertainty metrics (numpy/scipy only)."""
from __future__ import annotations
import numpy as np
from scipy import stats


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
def mae(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
def spearman(y, p): return float(stats.spearmanr(y, p)[0]) if len(y) > 2 else float("nan")
def pearson(y, p): return float(stats.pearsonr(y, p)[0]) if len(y) > 2 else float("nan")


def gaussian_nll(y, mu, sd):
    sd = np.maximum(sd, 1e-6)
    return float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * ((y - mu) / sd) ** 2))


def coverage(y, mu, sd, z=1.645):
    return float(np.mean(np.abs(y - mu) <= z * sd))


def calibration_curve(y, mu, sd, levels=np.linspace(0.05, 0.95, 19)):
    z = stats.norm.ppf(0.5 + levels / 2)
    obs = np.array([np.mean(np.abs(y - mu) <= zz * sd) for zz in z])
    return levels, obs


def ece_regression(y, mu, sd):
    lv, obs = calibration_curve(y, mu, sd); return float(np.mean(np.abs(lv - obs)))


def sharpness(sd): return float(np.mean(sd))


def error_uncertainty_rho(y, mu, sd):
    return spearman(np.abs(y - mu), sd)


def full_report(y, mu, sd=None) -> dict:
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    r = {"n": len(y), "rmse": rmse(y, mu), "mae": mae(y, mu), "spearman": spearman(y, mu), "pearson": pearson(y, mu)}
    if sd is not None:
        sd = np.asarray(sd, float)
        r.update(nll=gaussian_nll(y, mu, sd), cov90=coverage(y, mu, sd), ece=ece_regression(y, mu, sd),
                 sharpness=sharpness(sd), err_unc_rho=error_uncertainty_rho(y, mu, sd))
    return r
