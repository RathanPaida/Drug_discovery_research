"""Stage 8: one DMTA cycle.
  prepare : pool -> hard filter -> dual-model audit -> Pareto top-1500 -> AiZynthFinder -> docking request for Team B
  select  : Team B docking verdicts -> EI/UCB/LCB -> Butina -> stratified 40-compound batch -> contract CSV
  ingest  : Team B results -> evaluate-before-train -> append (w=3.0) -> retrain + recalibrate + expand AD
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.context import Ctx
from tnk1.active_learning import acquisition as ACQ, batch as B, contract as K, ingest as I
from tnk1.baselines.features import morgan_count


def pool_from_reinvent(paths):
    dfs = [pd.read_csv(p) for p in paths]; d = pd.concat(dfs)
    col = next(c for c in d.columns if c.lower() == "smiles"); return pd.Series(d[col].dropna().unique())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--rl-csv", nargs="*"); ap.add_argument("--docking", default=None); ap.add_argument("--results", default=None)
    ap.add_argument("--aizynth-config", default="configs/aizynth.yml")
    # Retrosynthetic feasibility FILTERS candidates; it feeds no model. Skipping changes
    # which molecules reach docking and invalidates nothing upstream. It is also the
    # slowest step in the pipeline (5-30 s x al.pareto_keep candidates = 2-12 h), and
    # AiZynthFinder needs ~1 GB of model files that are not in this repository.
    ap.add_argument("--skip-retro", action="store_true",
                    help="bypass the AiZynthFinder synthesisability screen")
    ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--version", default="v1"); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    C = Ctx(a.config, a.version, a.device); cfg, al = C.cfg, C.cfg["al"]
    cyc = C.runs / f"cycle_{a.cycle:02d}"; cyc.mkdir(parents=True, exist_ok=True)

    def load_stack():
        import torch
        from tnk1.uq.ensemble import Ensemble
        from tnk1.rl.scorer import Scorer
        ens = Ensemble(cfg, C.store, C.desc_store(), C.proc / "ensemble", a.device)
        cal, ad = pickle.load(open(C.proc / "calibrator.pkl", "rb")), pickle.load(open(C.proc / "ad.pkl", "rb"))
        bundles = torch.load(C.proc / "admet_bundles.pt", weights_only=False)
        return ens, cal, ad, Scorer(cfg, ens, cal, ad, lambda s, e: (e.latent(s), morgan_count(s)), bundles, C.desc_scaler(), a.device)

    if a.cmd == "prepare":
        from tnk1.data.standardize import standardize_smiles
        from tnk1.baselines.models import TanimotoGP
        from tnk1.active_learning.retro import retro_screen
        ens, cal, ad, scorer = load_stack()
        pool = pool_from_reinvent(a.rl_csv).sample(frac=1.0, random_state=a.cycle).iloc[: al["pool_size"]]
        std = [standardize_smiles(s) for s in pool]; rows = [r for r in std if r]
        P = pd.DataFrame(rows).drop_duplicates("inchikey_block1")
        P = P[~P.inchikey_block1.isin(set(C.df.inchikey_block1))].reset_index(drop=True)      # vs ChEMBL + in-house
        sc = scorer.score(list(P.smiles), advance=False); P = P.assign(**{k: v for k, v in sc.items() if k != "beta"})
        P = P[P.R > 0].reset_index(drop=True)                                                 # I_hard(x) = 1
        # cross-architecture audit: D-MPNN ensemble vs Tanimoto GP
        tr = C.df[(C.df.uniprot_id == C.tnk1) & (C.df.censored == 0)]
        gp = TanimotoGP(device=a.device).fit(morgan_count(tr.smiles), tr.pact.values)
        P["mu_gp"] = gp.predict(morgan_count(P.smiles)); P = P[(P.mu - P.mu_gp).abs() <= al["audit_tol"]].reset_index(drop=True)
        objs = np.c_[P.mu - P.sigma_epi, P.mu_sel - P.sigma_sel, P.adme, P.ad, P.sa, P.safety]
        P = P.iloc[ACQ.top_by_pareto(objs, al["pareto_keep"], P.R.values)].reset_index(drop=True)
        if a.skip_retro:
            print(f"[stage8] --skip-retro: keeping all {len(P)} candidates WITHOUT a "
                  f"synthesisability screen. Downstream rows carry retro_ok=True by "
                  f"assertion, not by evidence -- do not report them as synthesisable.")
            P = P.assign(retro_ok=True, state_score=float("nan"), n_steps=-1)
        else:
            rs = retro_screen(list(P.smiles), a.aizynth_config, al["retro_max_steps"], al["retro_min_state_score"])
            P = P.merge(rs[["smiles", "retro_ok", "state_score", "n_steps"]], on="smiles"); P = P[P.retro_ok]
        P.to_csv(cyc / "candidates_pre_docking.csv", index=False)
        P[["smiles"]].assign(docking_ok="").to_csv(cyc / "docking_request.csv", index=False)
        print(f"{len(P)} retro-feasible candidates -> send {cyc/'docking_request.csv'} to Team B (docking: hinge H-bond, no clash)")

    elif a.cmd == "select":
        ens, cal, ad, _ = load_stack(); P = pd.read_csv(cyc / "candidates_pre_docking.csv")
        dk = pd.read_csv(a.docking); P = P.merge(dk[["smiles", "docking_ok"]], on="smiles"); P = P[P.docking_ok.astype(str).str.lower().isin(["1", "true", "yes", "y"])].copy()
        ytr = C.df[(C.df.uniprot_id == C.tnk1) & (C.df.censored == 0)]; ybest = float(ytr.pact.max())
        P["ei"] = ACQ.expected_improvement(P.mu.values, P.sigma_epi.values, ybest, al["ei_xi"])
        P["ucb"] = ACQ.ucb(P.mu.values, P.sigma_epi.values, al["ucb_kappa"]); P["lcb"] = ACQ.lcb(P.mu.values, P.sigma_epi.values, al["exploit_lcb_k"])
        P["cluster"] = B.butina_clusters(list(P.smiles), al["butina_dist"]); P["passes_filters"] = True
        sel = B.select_batch(P, al["batch"], al["max_per_cluster"], set(ytr.scaffold), seed=a.cycle)
        from rdkit import Chem
        ct = K.build_contract(sel, a.cycle, cal, lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)))
        ct.to_csv(cyc / "contract_to_team_b.csv", index=False); print(ct.batch_slice.value_counts().to_dict(), "->", cyc / "contract_to_team_b.csv")

    elif a.cmd == "ingest":
        from tnk1.utils.common import save_locked
        res = pd.read_csv(a.results); _, cal, ad, _ = load_stack()
        rep = I.evaluate_prospective(res, cal); print(rep); pd.Series({k: v for k, v in rep.items() if k != "by_slice"}).to_csv(cyc / "prospective_metrics.csv")
        new = I.to_training_rows(res, cfg["finetune"]["inhouse_weight"], C.tnk1, cfg["data"]["censor_tau"])
        merged = pd.concat([C.df, new], ignore_index=True)
        save_locked(merged, C.proc / f"kinome_long_{a.version}_c{a.cycle:02d}.parquet")
        print("dataset extended -> re-run: stage3 (kinome refresh optional), stage4 train/oof/calibrate with --version", f"{a.version}_c{a.cycle:02d}")
        # conformal quantile / AD refresh happen in `run_stage4_uq.py calibrate` after the ensemble is retrained (Stage 3.5 L2-SP)


if __name__ == "__main__":
    main()
