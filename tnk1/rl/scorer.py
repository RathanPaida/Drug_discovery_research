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

    def score(self, smiles: list[str], advance: bool = True) -> dict:
        from ..admet.admet import predict_endpoint
        from ..admet.filters import medchem_profile
        rl, w = self.cfg["rl"], self.cfg["rl"]["weights"]
        if advance:
            self.step += 1                                   # one REINVENT batch == one step
            beta = R.beta_schedule(self.step, rl["beta_explore"], rl["beta_final"], rl["beta_explore_steps"], rl["steps"])
        else:
            beta = rl["beta_final"]                          # offline scoring (Stage 8) uses the converged, pessimistic beta
        prof = [medchem_profile(s) for s in smiles]
        valid = np.array([p is not None for p in prof])
        n = len(smiles)
        out = {c: np.zeros(n) for c in R.COMPONENTS}; out["R"] = np.zeros(n); out["mu"] = np.zeros(n); out["sigma_epi"] = np.zeros(n)
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
            "safety": aps["safety"],
            "qed": P("qed"),
        }
        gate = R.hard_gate(
            np.ones(len(vi)),
            P("n_alerts"),
            P("mw"),
            ames,
            self.cfg["filters"]["mw_max_hard"],
            p_herg=herg,
            p_herg_max=self.cfg["filters"]["herg_max"],
        )
        total = R.aggregate(comps, w, gate, rl.get("veto"))
        for c in R.COMPONENTS: out[c][vi] = comps[c]
        out["R"][vi], out["mu"][vi], out["sigma_epi"][vi] = total, d["mu"], d["sigma_epi"]
        out["sigma_ale"][vi], out["mu_sel"][vi], out["sigma_sel"][vi], out["d_lat"][vi] = d["sigma_ale"], sel["mu_sel"], sel["sigma_sel"], d_lat
        return {"beta": beta, **out}
