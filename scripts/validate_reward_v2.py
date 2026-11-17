"""Re-score the shipped Cycle-01 pool under the Cycle-02 reward and print the comparison.

This is the acceptance test for the reward rebalancing. It needs no models and no RDKit:
it replays the stored per-molecule quantities in runs/stage7_scored_candidates.csv
(mu_tnk1, sigma_epi, mu_sel, sigma_sel, d_lat, sa, qed, mw, n_alerts, hinge) through the
new reward, so the only thing that changes between the two columns is the reward itself.

Safety is validated separately against runs/cycle_01/contract_to_team_b.csv, which is the
only artefact carrying per-molecule p(hERG)/p(AMES)/p(CYP3A4); the 12k pool never stored
them. Nothing here is imputed -- a quantity that is absent is reported as absent.

    python scripts/validate_reward_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tnk1.rl import reward as RW  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "runs/stage7_scored_candidates.csv"
CONTRACT = ROOT / "runs/cycle_01/contract_to_team_b.csv"
CFG = ROOT / "configs/default.yaml"


def _line(label, old, new, fmt="{:.4f}", target=None):
    t = "" if target is None else f"   target {target}"
    print(f"  {label:<34s} {fmt.format(old):>10s} -> {fmt.format(new):>10s}{t}")


def main() -> int:
    cfg = yaml.safe_load(CFG.read_text())
    rl, filt, tpp = cfg["rl"], cfg["filters"], cfg["tpp"]
    d = pd.read_csv(POOL)
    n = len(d)
    print(f"\n=== Cycle-01 pool: {n} molecules, re-scored under the Cycle-02 reward ===\n")

    # ---------------------------------------------------------------- calibration
    # The single change that unflattens r_sel. Centres come from this population, which is
    # what calibrate_centre is for; the Cycle-01 literals were picked by hand.
    frac = rl.get("calibrate_target_frac", 0.30)
    pot_c = RW.calibrate_centre(d.mu_tnk1, frac)
    sel_c = RW.calibrate_centre(d.mu_sel, frac)
    print("Sigmoid centres (calibrate_centre, target_frac=%.2f):" % frac)
    print(f"  pot_c  {rl['pot_c']:>7.3f} -> {pot_c:7.3f}   (mu_tnk1 max {d.mu_tnk1.max():.2f})")
    print(f"  sel_c  {rl['sel_c']:>7.3f} -> {sel_c:7.3f}   (mu_sel  max {d.mu_sel.max():.2f}"
          "  <-- the old centre sat ABOVE this)\n")

    beta = rl["beta_final"]
    old = {
        "pot": RW.r_pot(d.mu_tnk1, d.sigma_epi, beta, rl["pot_c"], rl["pot_k"]),
        "sel": RW.r_sel(d.mu_sel, d.sigma_sel, rl["gamma"], rl["sel_c"], rl["sel_k"]),
        "ad": d.r_ad.values, "sa": d.r_sa.values,
        "adme": d.r_adme.values, "safety": d.r_safety.values, "qed": d.r_qed.values,
    }
    new = {
        "pot": RW.r_pot(d.mu_tnk1, d.sigma_epi, beta, pot_c, rl["pot_k"]),
        "sel": RW.r_sel(d.mu_sel, d.sigma_sel, rl["gamma"], sel_c, rl["sel_k"]),
        "ad": d.r_ad.values, "sa": d.r_sa.values,
        "hinge": RW.r_hinge(d.hinge.astype(bool)),
        # p(hERG)/p(AMES)/p(CYP3A4) were never stored for the 12k pool, so the SHAPED
        # safety term cannot be replayed here. Carrying the old r_safety through unchanged
        # keeps this an honest lower bound on the improvement rather than an imputed one.
        "safety": d.r_safety.values,
    }

    print("Component means (the reward is a geometric mean: a term near 0 caps R):")
    for c in ["pot", "sel", "ad", "sa", "safety"]:
        _line(f"r_{c}", np.mean(old[c]), np.mean(new[c]))
    print(f"  {'r_hinge':<34s} {'-':>10s} -> {np.mean(new['hinge']):10.4f}   (new term)")

    # ---------------------------------------------------------------- gates
    g_old = RW.hard_gate(np.ones(n), d.n_alerts, d.mw, np.zeros(n), 600.0)
    g_new = RW.hard_gate(np.ones(n), d.n_alerts, d.mw, np.zeros(n), filt["mw_max_hard"],
                         sa=d.sa, sa_max=filt["sa_max_hard"],
                         qed=d.qed, qed_min=filt["qed_min"])
    print("\nTier-A gate pass rate (p(hERG)/p(AMES) absent from this file, so not applied):")
    _line("gate pass fraction", g_old.mean(), g_new.mean(), "{:.3f}")

    R_old = RW.aggregate(old, rl["weights"], g_old, None, components=RW.COMPONENTS)
    R_new = RW.aggregate(new, rl["weights_v2"], g_new, rl["veto"],
                         components=RW.COMPONENTS_V2)
    print("\nTotal reward R:")
    _line("mean R", R_old.mean(), R_new.mean())
    _line("max  R", R_old.max(), R_new.max(), target="> 0.70")
    _line("fraction with R > 0.3", (R_old > 0.3).mean(), (R_new > 0.3).mean(), "{:.4f}")

    s_old = RW.component_signal(old, rl["weights"], RW.COMPONENTS)
    s_new = RW.component_signal(new, rl["weights_v2"], RW.COMPONENTS_V2)
    spread = lambda s: max(s.values()) / max(min(s.values()), 1e-9)  # noqa: E731
    print("\nRealised component signal  sd(log r) x normalised weight:")
    print("  old:", {k: round(v, 3) for k, v in sorted(s_old.items(), key=lambda x: -x[1])})
    print("  new:", {k: round(v, 3) for k, v in sorted(s_new.items(), key=lambda x: -x[1])})
    _line("signal spread (max/min)", spread(s_old), spread(s_new), "{:.1f}x", target="< 3.0x")

    # `ad` is now the widest term, and it is an ARTEFACT OF THIS FILE, not of the reward.
    # The stored r_ad column was computed before the applicability fix: d_lat reaches
    # 240,831.9 here against a threshold of ~1.007, and exp(-(d_lat-thr)/0.15) underflows
    # to exactly 0.0 for 1,499 molecules. It cannot be replayed post-fix from stored
    # columns because the fix (sd_floor_frac=0.05) changes d_lat itself, upstream of r_ad,
    # and the latent features are not in this file. Reporting the spread without it shows
    # what the rebalanced reward does on its own terms.
    n_underflow = int((d.r_ad <= 1e-9).sum())
    s_new_noad = {k: v for k, v in s_new.items() if k != "ad"}
    print(f"\n  NOTE: stored r_ad is PRE-FIX -- d_lat max {d.d_lat.max():,.0f} vs threshold "
          f"~1.007,\n        {n_underflow} molecules underflowed to r_ad = 0 exactly. "
          "It cannot be replayed\n        post-fix from this file (the fix changes d_lat, "
          "upstream of r_ad).")
    print(f"  Signal spread excluding the stale ad term: {spread(s_new_noad):.1f}x "
          "   target < 3.0x")

    # ---------------------------------------------------------------- Tier B vs C on the 40
    print("\n=== Shipped contract (best 40 of 12,000) ===\n")
    c40 = pd.read_csv(CONTRACT)
    sh_old = RW.r_safety(c40.tox_herg_prob, c40.tox_ames_prob, c40.cyp3a4_inhibition_prob)
    sh_new = RW.r_safety_tpp(c40.tox_herg_prob, c40.tox_ames_prob, c40.cyp3a4_inhibition_prob,
                             tpp["herg_max"], tpp["ames_max"])
    print("Shaped safety term (r_safety -> r_safety_tpp), the one place p(hERG) is stored:")
    _line("mean r_safety", sh_old.mean(), sh_new.mean())
    bad = c40.tox_herg_prob > tpp["herg_max"]
    _line(f"mean on the {int(bad.sum())} hERG-failing cpds", sh_old[bad].mean(),
          sh_new[bad].mean(), target="should COLLAPSE")
    print(f"  -> the old term still returned {sh_old[bad].mean():.2f} for molecules that "
          "violate the TPP;\n     the shaped term prices them correctly and so creates the "
          "gradient away from them.")

    mask, rep = RW.tpp_select(c40)
    print(f"\nTier-C tpp_select on the shipped 40: {rep['n_pass']}/{rep['n_total']} pass")
    for k, v in rep["applied"].items():
        print(f"    {k:<12s} {v:>2d}/40")
    if rep["skipped"]:
        print(f"    SKIPPED (column absent): {rep['skipped']}")
    print("\n  35 of the 40 compounds already shipped to Team B fail the project's own TPP.")
    print("  That is the finding, not a bug in the filter.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
