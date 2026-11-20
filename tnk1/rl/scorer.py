"""Stage 7 scoring backend (runs in the tnk1-predict env). Computes each molecule's reward and per-component scores
from the ensemble, selectivity, AD, ADMET and med-chem filters. Used by the REST server / CLI wrapper."""
from __future__ import annotations
import numpy as np
from . import reward as R
from ..uq.decomposition import decompose
from ..selectivity.selectivity import selectivity_stats


class Scorer:
    def __init__(self, cfg, ensemble, calibrator, ad, ad_ref_fn, admet_bundles, desc_scaler, device="cuda"):
        self.cfg, self.ens, self.cal, self.ad, self.admet, self.ds, self.dev = cfg, ensemble, calibrator, ad, admet_bundles, desc_scaler, device
        self.step = 0
        self.tnk1, self.off = cfg["targets"]["tnk1"], cfg["offtarget_panel"]
        self.ad_ref_fn = ad_ref_fn
        self.use_v2 = bool(cfg["rl"].get("use_v2", False))
        self.components = R.COMPONENTS_V2 if self.use_v2 else R.COMPONENTS
        self.calibrated = False

    def _weights(self):
        rl = self.cfg["rl"]
        return rl["weights_v2"] if self.use_v2 else rl["weights"]

    def advance_step(self) -> int:
        """Tick the beta schedule without scoring anything.

        A caller that de-duplicates or caches (see `rl_min.scorer.PipelineScorer`) can
        reach an RL step where every molecule is already known and `score()` is never
        invoked. Without this, `self.step` stops tracking the true step count and beta
        silently lags -- the agent keeps being judged under exploration-era optimism long
        after the schedule should have turned pessimistic. Call it once per skipped step.
        """
        self.step += 1
        return self.step

    def beta(self) -> float:
        """The beta the NEXT scored batch will use. Log it; a drifting schedule is
        otherwise invisible until the results look wrong."""
        rl = self.cfg["rl"]
        return R.beta_schedule(self.step, rl["beta_explore"], rl["beta_final"],
                               rl["beta_explore_steps"], rl["steps"])

    def calibrate(self, smiles: list[str], target_frac: float | None = None) -> dict:
        """Re-centre the potency and selectivity sigmoids on this generator's own output.

        THIS IS NOT OPTIONAL, and until now it only ever happened inside the v2 notebook,
        which mutated its in-memory cfg. Every script entry point (run_stage7_rl.py,
        run_stage8_cycle.py) therefore still ran with the Cycle-01 literals pot_c=7.0 and
        sel_c=1.2 -- and sel_c=1.2 sits above the ENTIRE pool's maximum mu_sel of 0.25, so
        r_sel averaged 0.0089 and the 2.0 weight bought a near-constant 0.38x penalty with
        no gradient in it. Calibrating here closes that gap for every caller.

        Call once with a prior sample before training. Returns the centres it set.
        """
        rl = self.cfg["rl"]
        frac = rl.get("calibrate_target_frac", 0.30) if target_frac is None else target_frac
        res = self.score(smiles, advance=False)
        # Rows that failed to parse are left as exact 0.0 by `score`, and a real predicted
        # pIC50 is never exactly 0.0; calibrating on those zeros would drag both centres
        # down and undo the fix this method exists to apply.
        mu = np.asarray(res["mu"], float)
        use = np.isfinite(mu) & (mu != 0.0)
        if use.sum() < 50:
            raise ValueError(
                f"only {int(use.sum())} scorable molecules in the calibration sample; "
                "calibrating a sigmoid centre on this few is meaningless")
        rl["pot_c"] = R.calibrate_centre(mu[use], frac)
        rl["sel_c"] = R.calibrate_centre(np.asarray(res["mu_sel"])[use], frac)
        self.calibrated = True
        return {"pot_c": rl["pot_c"], "sel_c": rl["sel_c"], "n": int(use.sum()),
                "target_frac": frac}

    def score(self, smiles: list[str], advance: bool = True) -> dict:
        from ..admet.admet import predict_endpoint
        from ..admet.filters import medchem_profile
        rl, w = self.cfg["rl"], self._weights()
        tpp = self.cfg.get("tpp", {})
        if advance and rl.get("calibrate_at_runtime", False) and not self.calibrated:
            raise RuntimeError(
                "rl.calibrate_at_runtime is set but Scorer.calibrate() was never called. "
                "Scoring would use the Cycle-01 literals pot_c=7.0 / sel_c=1.2, and sel_c "
                "sits above the whole pool's maximum mu_sel, which flattens r_sel to a "
                "constant. Call scorer.calibrate(prior_sample) before training."
            )
        if advance:
            self.step += 1                                   # one REINVENT batch == one step
            beta = R.beta_schedule(self.step, rl["beta_explore"], rl["beta_final"], rl["beta_explore_steps"], rl["steps"])
        else:
            beta = rl["beta_final"]                          # offline scoring (Stage 8) uses the converged, pessimistic beta
        prof = [medchem_profile(s) for s in smiles]
        valid = np.array([p is not None for p in prof])
        n = len(smiles)
        out = {c: np.zeros(n) for c in set(R.COMPONENTS) | set(R.COMPONENTS_V2)}
        out["R"] = np.zeros(n); out["mu"] = np.zeros(n); out["sigma_epi"] = np.zeros(n)
        for k in ("sigma_ale", "mu_sel", "sigma_sel", "d_lat"): out[k] = np.zeros(n)
        if not valid.any():
            return {"beta": beta, **out}
        vi = np.where(valid)[0]; vs = [smiles[i] for i in vi]
        panel = [self.tnk1] + self.off
        mu_m, lv_m = self.ens.predict_members(vs, panel)
        d = decompose(mu_m[..., 0], lv_m[..., 0])
        sel = selectivity_stats(mu_m[..., 0], mu_m[..., 1:])
        Z, FP = self.ad_ref_fn(vs, self.ens)
        d_lat, _ = self.ad.distances(Z, FP)
        ep = {k: predict_endpoint(b, vs, self.ds, self.dev)[0] for k, b in self.admet.items()}
        P = lambda k: np.array([p[k] for p in (prof[i] for i in vi)])
        # fail-closed: raises if any endpoint model is absent (see reward.require_endpoints)
        aps = R.adme_safety_components(ep, len(vs))
        herg, ames = aps["ep"]["herg"], aps["ep"]["ames"]
        comps = {
            "pot": R.r_pot(d["mu"], d["sigma_epi"], beta, rl["pot_c"], rl["pot_k"]),
            "sel": R.r_sel(sel["mu_sel"], sel["sigma_sel"], rl["gamma"], rl["sel_c"], rl["sel_k"]),
            "ad": R.r_ad(d_lat, self.ad.d_thr_lat, rl["ad_scale"]),
            "sa": R.r_sa(P("sa")),
            "adme": aps["adme"],
            "safety": (R.r_safety_tpp(herg, ames, aps["ep"]["cyp3a4"],
                                      tpp.get("herg_max", 0.20), tpp.get("ames_max", 0.30))
                       if self.use_v2 else aps["safety"]),
            "qed": P("qed"),
            "hinge": R.r_hinge(P("hinge")),
        }
        f = self.cfg["filters"]
        gate = R.hard_gate(
            np.ones(len(vi)),
            P("n_alerts"),
            P("mw"),
            ames,
            f["mw_max_hard"],
            p_herg=herg,
            p_herg_max=f["herg_max"],
            # Tier-A criteria are only applied when configured, so the Cycle-01 config
            # (which has none of these keys) reproduces the old gate exactly.
            p_ames_max=f.get("ames_max", 0.5),
            sa=P("sa"), sa_max=f.get("sa_max_hard"),
            n_rot=(P("n_rot") if "n_rot" in prof[vi[0]] else None), rot_max=f.get("rot_max"),
            qed=P("qed"), qed_min=f.get("qed_min"),
        )
        total = R.aggregate(comps, w, gate, rl.get("veto"), components=self.components)
        # Scatter EVERY component that was computed, not only the ones `self.components`
        # aggregates into R. `adme` and `qed` are outside COMPONENTS_V2, so under the v2
        # reward they were computed on the lines above and then dropped on the floor --
        # `out["adme"]` and `out["qed"]` kept the np.zeros init from the top of this method
        # and were reported, and consumed, as all-zeros.
        #
        # That was not cosmetic. `scripts/run_stage8_cycle.py` feeds `P.adme` as one of six
        # Pareto objectives, and a constant column cannot affect dominance -- every pair
        # ties on it -- so ADME never influenced candidate selection at all, while `qed`
        # was written into the candidate CSV as a column of zeros. The real values are
        # non-trivial (adme median ~0.36 on the Cycle-02 generated set).
        #
        # Keep the loop over `comps`, not over any component list: a new component added to
        # `comps` is then reported automatically instead of being silently discarded until
        # someone notices it is suspiciously constant.
        for c, v in comps.items():
            if c not in out: out[c] = np.zeros(n)
            out[c][vi] = v
        out["R"][vi], out["mu"][vi], out["sigma_epi"][vi] = total, d["mu"], d["sigma_epi"]
        out["sigma_ale"][vi], out["mu_sel"][vi], out["sigma_sel"][vi], out["d_lat"][vi] = d["sigma_ale"], sel["mu_sel"], sel["sigma_sel"], d_lat
        return {"beta": beta, **out}
