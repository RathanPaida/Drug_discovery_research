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


def miscalibration_area(y, mu, sd) -> float:
    """Area between the calibration curve and the diagonal.

    The headline calibration number in Hirschfeld, Swanson, Yang, Barzilay & Coley,
    "Uncertainty Quantification Using Neural Networks for Molecular Property Prediction",
    J Chem Inf Model 60(8) (2020), whose metric set this module now matches. Integrated
    rather than averaged (`ece_regression` above takes the mean of the same absolute
    deviations), so it is comparable with the numbers that paper reports.
    """
    lv, obs = calibration_curve(y, mu, sd)
    return float(np.trapezoid(np.abs(lv - obs), lv)) if hasattr(np, "trapezoid") \
        else float(np.trapz(np.abs(lv - obs), lv))


def calibrated_nll(y, mu, sd) -> dict:
    """NLL after refitting a single multiplicative scale on sigma. Returns both and the scale.

    WHY THIS IS THE DIAGNOSTIC THAT MATTERS HERE. A sigma can be bad in two different
    ways, and plain NLL cannot tell them apart:

      * MIS-SCALED -- the sigma ranks the errors correctly but is uniformly too small or
        too large. Fixable by one scalar. The fix is already in the pipeline:
        `tnk1.uq.calibration.fit_temperature`.
      * UNINFORMATIVE -- the sigma carries no information about where the model is wrong.
        No scalar, and no monotone transform, repairs it.

    `cnll` is the NLL achievable after the optimal rescale, so `nll - cnll` is the part of
    the loss that was mere mis-scaling and `cnll` is the part that is irreducible. For this
    project the two cases are already distinguishable by the error-sigma correlation --
    -0.021 for the deep ensemble versus +0.258 (RF) and +0.283 (Tanimoto GP) -- which
    predicts that recalibration will NOT rescue the ensemble and WILL help the RF and GP if
    they are merely mis-scaled. Reporting `cnll` next to `nll` makes that prediction
    checkable instead of asserted.

    Hirschfeld et al. (2020) report exactly this pair. The closed-form optimal scale is the
    root-mean-square standardised residual, which is `fit_temperature`'s tau.
    """
    y, mu, sd = (np.asarray(a, float) for a in (y, mu, sd))
    sd = np.maximum(sd, 1e-6)
    tau = float(np.sqrt(max(np.mean(((y - mu) / sd) ** 2), 1e-12)))
    return {"nll": gaussian_nll(y, mu, sd), "cnll": gaussian_nll(y, mu, tau * sd),
            "sigma_scale": tau}


def full_report(y, mu, sd=None) -> dict:
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    r = {"n": len(y), "rmse": rmse(y, mu), "mae": mae(y, mu), "spearman": spearman(y, mu), "pearson": pearson(y, mu)}
    if sd is not None:
        sd = np.asarray(sd, float)
        c = calibrated_nll(y, mu, sd)
        r.update(nll=c["nll"], cnll=c["cnll"], sigma_scale=c["sigma_scale"],
                 cov90=coverage(y, mu, sd), ece=ece_regression(y, mu, sd),
                 mis_area=miscalibration_area(y, mu, sd),
                 sharpness=sharpness(sd), err_unc_rho=error_uncertainty_rho(y, mu, sd))
    return r
