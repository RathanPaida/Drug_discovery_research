"""Calibrate the cross-architecture disagreement gate against its own null distribution.

READ THIS BEFORE TRUSTING THE v3 RUN'S HEADLINE CONCLUSION.

`RUN_REPORT_V3_HPC.md` section 3.7 reports that the Stage-8 audit gate rejected all 89
generated molecules because `|mu_ensemble - mu_GP|` was 1.18-2.37 (median 1.77) against a
tolerance of 1.0, and reads that as "the signature of reward hacking". That reading may be
wrong, and the fix is cheap.

Langevin, Vuilleumier & Bianciotto, "Explaining and avoiding failure modes in goal-directed
generation of small molecules", J Cheminform 14:20 (2022), is a direct response to Renz,
Van Rompaey, Wegner, Hochreiter & Klambauer, "On failure modes in molecule generation and
optimization", Drug Discovery Today: Technologies 32-33:55 (2019/2020), which introduced
the optimise-then-rescore-with-a-control-model diagnostic this project's gate implements.
Langevin et al.'s central result: the apparent model-exploitation signal in that setup

    "can be explained by a disagreement between predictive models on the original data
     distribution"

-- that is, two independently trained models disagree by a BASELINE amount even on
molecules nobody optimised against, and that baseline is easy to mistake for exploitation.

Why that bites here specifically. The gate's threshold is 1.0, a round number with no
derivation anywhere in the repository. The two models are a 10-member heteroscedastic
multi-task D-MPNN ensemble and a Tanimoto-kernel GP, fitted on 172 measurements. Those are
very differently biased estimators at that sample size, and a ~1 log unit disagreement on
HELD-OUT chemistry would not be surprising. **If the baseline held-out disagreement is
already near 1.0, the gate is mis-specified, the 100% rejection rate carries far less
information than the report claims, and the entire "the generator exploited the critic"
narrative is partly an artefact of an uncalibrated threshold.**

This module measures that baseline. It is pure numpy: given two models' predictions on
(i) held-out measured TNK1 compounds and (ii) unoptimised prior samples, it returns the
null distribution of `|mu_a - mu_b|` and the quantile-derived threshold that should replace
the round number.

THE ORDER OF OPERATIONS MATTERS. Every pessimism scheme in `rl_min.algos.common`
(`pessimistic_reward`, `disagreement_penalty`) is parameterised by a disagreement
magnitude. Calibrating them against an uncalibrated null means penalising at a threshold
that may be mostly noise. Run `calibrate_null` first, then set `tol` from
`recommended_threshold`.
"""
from __future__ import annotations

import numpy as np


def disagreement(mu_a, mu_b) -> np.ndarray:
    """|mu_a - mu_b|, the gate's own statistic."""
    return np.abs(np.asarray(mu_a, float) - np.asarray(mu_b, float))


def calibrate_null(mu_a_heldout, mu_b_heldout, mu_a_prior=None, mu_b_prior=None,
                   quantiles=(0.50, 0.90, 0.95, 0.99)) -> dict:
    """The null distribution of cross-model disagreement, where nobody is gaming anything.

    Parameters
    ----------
    mu_a_heldout, mu_b_heldout
        The two models' predictions on MEASURED compounds that neither was fitted on.
        This is the reference population: these molecules came from medicinal chemists,
        not from an optimiser, so any disagreement here is irreducible inter-model
        disagreement. Use the frozen test fold, or the out-of-fold predictions both
        models already produce.
    mu_a_prior, mu_b_prior
        Optional, and worth including: the same two models on UNOPTIMISED samples from the
        generative prior. These are the distribution the agent starts from, so they
        separate "disagreement on novel chemistry" from "disagreement caused by
        optimisation". If the prior-sample disagreement already matches the generated
        molecules', the generator did not walk anywhere the prior was not already.

    Returns the two distributions' quantiles plus `recommended_threshold`.
    """
    d_held = disagreement(mu_a_heldout, mu_b_heldout)
    out = {
        "n_heldout": int(len(d_held)),
        "heldout": {
            "mean": float(d_held.mean()), "median": float(np.median(d_held)),
            "sd": float(d_held.std()), "max": float(d_held.max()) if len(d_held) else float("nan"),
            "quantiles": {str(q): float(np.quantile(d_held, q)) for q in quantiles},
        },
    }
    if mu_a_prior is not None and mu_b_prior is not None:
        d_prior = disagreement(mu_a_prior, mu_b_prior)
        out["n_prior"] = int(len(d_prior))
        out["prior_samples"] = {
            "mean": float(d_prior.mean()), "median": float(np.median(d_prior)),
            "sd": float(d_prior.std()),
            "quantiles": {str(q): float(np.quantile(d_prior, q)) for q in quantiles},
        }
    # The threshold the gate SHOULD use: an upper quantile of the held-out null, so a
    # molecule is flagged only when its disagreement exceeds what honest chemistry
    # produces. Langevin et al. use 95% tolerance intervals for the control-model score;
    # the 0.95 quantile of the null is the one-sided equivalent.
    out["recommended_threshold"] = float(np.quantile(d_held, 0.95)) if len(d_held) else float("nan")
    out["quantile_used"] = 0.95
    return out


def assess_gate(null: dict, observed_disagreement, current_tol: float = 1.0) -> dict:
    """Is the generated set's disagreement actually anomalous against the null?

    Returns the verdict the v3 report could not reach, because it never measured the null.
    `fraction_above_null_q95` is the number to read: if the generated molecules sit at the
    same place in the null distribution as held-out measured compounds do, there is no
    exploitation signal to explain.
    """
    d = np.asarray(observed_disagreement, float)
    q95 = null["recommended_threshold"]
    h = null["heldout"]
    out = {
        "n_observed": int(len(d)),
        "observed_median": float(np.median(d)),
        "observed_mean": float(d.mean()),
        "null_median": h["median"],
        "null_q95": q95,
        "current_tol": float(current_tol),
        "fraction_above_current_tol": float(np.mean(d > current_tol)),
        "fraction_above_null_q95": float(np.mean(d > q95)),
        "median_ratio_to_null": float(np.median(d) / h["median"]) if h["median"] > 0 else float("inf"),
    }
    # The three readings, spelled out so the caller cannot shrug one off.
    if current_tol < h["quantiles"]["0.95"]:
        out["verdict_threshold"] = (
            f"MIS-SPECIFIED: the gate's tolerance ({current_tol:.2f}) is BELOW the 95th "
            f"percentile of baseline held-out disagreement ({h['quantiles']['0.95']:.2f}). "
            f"The gate rejects molecules whose disagreement is normal for honest "
            f"chemistry, so its rejection rate is not evidence of exploitation. Raise the "
            f"tolerance to the null quantile before drawing any conclusion from it.")
    else:
        out["verdict_threshold"] = (
            f"DEFENSIBLE: the tolerance ({current_tol:.2f}) exceeds the baseline 95th "
            f"percentile ({h['quantiles']['0.95']:.2f}), so a rejection does carry "
            f"information.")
    if out["fraction_above_null_q95"] > 0.5:
        out["verdict_exploitation"] = (
            f"ANOMALOUS: {out['fraction_above_null_q95']:.0%} of the generated set exceeds "
            f"the baseline 95th percentile, against the 5% expected by construction. That "
            f"is a genuine exploitation signal.")
    else:
        out["verdict_exploitation"] = (
            f"NOT ANOMALOUS: only {out['fraction_above_null_q95']:.0%} of the generated set "
            f"exceeds the baseline 95th percentile (5% expected). The disagreement is "
            f"within what two differently-biased models trained on this much data produce "
            f"anyway -- the 'reward hacking' reading is not supported by this statistic.")
    return out


def tolerance_band(mu_opt, mu_control, n_bins=5, level=0.95) -> list[dict]:
    """Per-bin tolerance band for the control model's score as a function of the optimised one.

    Langevin et al.'s diagnostic, in the form their Mean Average Difference plots use:
    bin by the OPTIMISATION model's prediction and report the control model's distribution
    within each bin. The point is conditional: inter-model disagreement is not constant
    across the prediction range, and it is typically largest exactly where the optimiser
    pushes -- the top of the range, where the training data is thinnest.

    A generated set whose control-model scores fall inside these bands at the potency the
    agent claims is consistent with baseline disagreement. One that falls below the lower
    bound is exploiting.
    """
    a = np.asarray(mu_opt, float)
    b = np.asarray(mu_control, float)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    edges = np.quantile(a, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.inf
    out = []
    for i in range(n_bins):
        m = (a >= edges[i]) & (a < edges[i + 1])
        if m.sum() < 3:
            continue
        out.append({
            "bin": i, "n": int(m.sum()),
            "opt_lo": float(edges[i]),
            "opt_hi": float(edges[i + 1] if np.isfinite(edges[i + 1]) else a.max()),
            "control_median": float(np.median(b[m])),
            "control_lo": float(np.quantile(b[m], lo_q)),
            "control_hi": float(np.quantile(b[m], hi_q)),
            "mean_abs_diff": float(np.mean(np.abs(a[m] - b[m]))),
        })
    return out


def in_tolerance_band(mu_opt, mu_control, bands) -> np.ndarray:
    """Boolean mask: does each molecule's control score fall inside its bin's band?

    The per-molecule version of the diagnostic, usable as a gate that is calibrated by
    construction instead of by a round number.
    """
    a = np.asarray(mu_opt, float)
    b = np.asarray(mu_control, float)
    ok = np.zeros(len(a), dtype=bool)
    if not bands:
        return ok
    for i, x in enumerate(a):
        band = min(bands, key=lambda bd: abs((bd["opt_lo"] + bd["opt_hi"]) / 2 - x))
        ok[i] = band["control_lo"] <= b[i] <= band["control_hi"]
    return ok


def max_pairwise_discrepancy(mu_members) -> np.ndarray:
    """max_{i,j} |mu_i - mu_j| across models. MOReL's USAD, not a variance.

    Kidambi, Rajeswaran, Netrapalli & Joachims, "MOReL: Model-Based Offline Reinforcement
    Learning", NeurIPS 2020, define their unknown-state-action detector as the MAXIMUM
    pairwise discrepancy between differently-initialised models rather than their variance.

    The distinction matters here. A variance over 10 members of ONE architecture measures
    seed noise, which is what the deep ensemble's uninformative sigma (error-sigma
    correlation -0.021) actually reports. A max pairwise discrepancy over DIFFERENT model
    classes measures the thing that tracks the failure, and it does not require any single
    member to be well calibrated -- which is the property that makes it usable when none of
    them is.

    `mu_members` is (M, n). Returns (n,).
    """
    M = np.atleast_2d(np.asarray(mu_members, float))
    if M.shape[0] < 2:
        return np.zeros(M.shape[1])
    return (M.max(axis=0) - M.min(axis=0))


def combined_uncertainty(mu_members, sigma=None) -> np.ndarray:
    """max(max-pairwise-discrepancy, sigma): the uncertainty to penalise.

    MOPO's penalty with MOReL's detector substituted for the ensemble spread, which is
    what to use when the ensemble's own sigma is known to be uninformative. Taking the max
    of the two means the penalty fires when EITHER signal says the model is out of its
    depth, and cannot be defeated by the weaker one being silent.
    """
    u = max_pairwise_discrepancy(mu_members)
    if sigma is None:
        return u
    return np.maximum(u, np.asarray(sigma, float))
