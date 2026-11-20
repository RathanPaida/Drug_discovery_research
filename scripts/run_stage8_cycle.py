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


def collect_unique(sample_fn, n, max_rounds=200):
    """Draw from `sample_fn()` until `n` unique non-empty SMILES are held, or the rounds run out.

    Factored out of `pool_from_agent` so the de-duplication and stopping rule are testable
    without torch, a trained agent or a GPU. `sample_fn` returns a list of raw decoded
    strings; duplicates and empties are expected (an autoregressive sampler emits plenty of
    both) and the draw order is preserved so a seeded sampler gives a reproducible pool.
    """
    seen, order = set(), []
    for _ in range(max_rounds):
        if len(order) >= n: break
        batch = sample_fn()
        if not batch: break                      # an exhausted sampler would otherwise spin
        for s in batch:
            s = (s or "").strip()
            if s and s not in seen:
                seen.add(s); order.append(s)
    return order[:n]


def pool_from_agent(agent_path, vocab_path, n, device="cuda", seed=0, batch=4096,
                    max_len=128, temperature=1.0):
    """Sample `n` unique SMILES on-policy from the trained DAP agent.

    This is the step the chain was missing. `configs/default.yaml` describes `al.pool_size`
    as "candidates drawn from the trained agent", but nothing sampled from `agent.pt`: the
    chain passed `--rl-csv runs/rl_min/best_molecules.csv`, which is the top-100 replay
    buffer, and then sliced `al.pool_size` out of it. The funnel was therefore starved by
    three orders of magnitude before any filter ran -- `.iloc[:200000]` on 100 rows is 100
    rows, and nothing said so.
    """
    import torch
    from rl_min.model import SmilesRNN
    from rl_min.vocab import Vocabulary
    torch.manual_seed(seed)
    vocab = Vocabulary.load(vocab_path)
    agent = SmilesRNN.load(agent_path, device=device)

    def draw():
        seqs, _ = agent.sample_with(vocab, batch, max_len=max_len, device=device,
                                    temperature=temperature)
        return [vocab.decode(r) for r in seqs]

    smis = collect_unique(draw, n, max_rounds=max(1, 4 * -(-n // batch)))
    print(f"[stage8] sampled {len(smis)} unique SMILES from {agent_path} "
          f"(target {n}, batch {batch}, seed {seed})")
    return pd.Series(smis)


def require_pool_size(got, want, allow_small):
    """Fail-closed on a pool that is nowhere near the configured size.

    `al.pool_size` silently meaning 100 is the defect this guards. A shortfall is either a
    wiring mistake (no `--agent`, so the pool is the replay buffer) or a sampler that could
    not produce enough unique molecules; both change what the funnel means, so neither may
    pass unremarked. `--allow-small-pool` is the explicit, logged opt-out for smoke runs.
    """
    if got >= want: return
    msg = (f"pool holds {got} molecules but al.pool_size is {want}. "
           f"Pass --agent/--vocab to sample a real pool from the trained agent, lower "
           f"al.pool_size to a number the run can actually supply, or pass "
           f"--allow-small-pool to proceed on {got} and accept that every downstream "
           f"survival rate is computed over that pool, not over {want}.")
    if not allow_small: raise SystemExit(f"[stage8] {msg}")
    print(f"[stage8] WARNING --allow-small-pool: {msg}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--rl-csv", nargs="*"); ap.add_argument("--docking", default=None); ap.add_argument("--results", default=None)
    # The pool. `--agent` is the scientifically correct source: al.pool_size molecules
    # sampled on-policy from the trained generator. `--rl-csv` alone is the top-100 replay
    # buffer and is now rejected unless --allow-small-pool says so out loud.
    ap.add_argument("--agent", default=None, help="trained agent.pt to sample the pool from "
                    "(default: <runs>/rl_min/agent.pt when it exists)")
    ap.add_argument("--vocab", default=None, help="vocabulary for --agent "
                    "(default: <processed>/rl_min/vocab.txt)")
    ap.add_argument("--sample-batch", type=int, default=4096, help="agent samples per draw")
    ap.add_argument("--sample-seed", type=int, default=None, help="default: --cycle")
    ap.add_argument("--allow-small-pool", action="store_true",
                    help="proceed when the pool is smaller than al.pool_size (smoke runs)")
    ap.add_argument("--aizynth-config", default=None,
                    help="default: <repo>/configs/aizynth.yml")
    # Retrosynthetic feasibility FILTERS candidates; it feeds no model. Skipping changes
    # which molecules reach docking and invalidates nothing upstream. It is also the
    # slowest step in the pipeline (5-30 s x al.pareto_keep candidates = 2-12 h), and
    # AiZynthFinder needs ~1 GB of model files that are not in this repository.
    ap.add_argument("--skip-retro", action="store_true",
                    help="bypass the AiZynthFinder synthesisability screen")
    ap.add_argument("--config", default=None)       # None -> tnk1.utils.paths.default_config()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    C = Ctx(a.config, a.version, a.device); cfg, al = C.cfg, C.cfg["al"]
    from tnk1.utils.paths import repo_root
    if a.aizynth_config is None:
        a.aizynth_config = str(repo_root() / "configs" / "aizynth.yml")
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
        want = int(al["pool_size"])
        seed = a.cycle if a.sample_seed is None else a.sample_seed
        agent_p = Path(a.agent) if a.agent else C.runs / "rl_min" / "agent.pt"
        vocab_p = Path(a.vocab) if a.vocab else C.proc / "rl_min" / "vocab.txt"
        if a.agent or (agent_p.exists() and vocab_p.exists()):
            if not agent_p.exists(): raise SystemExit(f"[stage8] --agent {agent_p} does not exist")
            if not vocab_p.exists(): raise SystemExit(f"[stage8] --vocab {vocab_p} does not exist")
            pool = pool_from_agent(agent_p, vocab_p, want, device=a.device, seed=seed,
                                   batch=a.sample_batch)
        else:
            if not a.rl_csv:
                raise SystemExit(f"[stage8] no pool source: {agent_p} is absent and no "
                                 f"--rl-csv was given")
            print(f"[stage8] NOTE no agent at {agent_p}; falling back to --rl-csv, which is "
                  f"the RL replay buffer (the top-scoring molecules kept during training), "
                  f"not a sample from the generator's current distribution.")
            pool = pool_from_reinvent(a.rl_csv).sample(frac=1.0, random_state=seed)
        require_pool_size(len(pool), want, a.allow_small_pool)
        pool = pool.iloc[:want]
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
        # al.exploit_min_sel and al.explore_min_mu were declared in configs/default.yaml but
        # NEVER PASSED here, so `select_batch`'s own defaults applied and the two config
        # knobs did nothing. They happened to hold the same values, which is why it was
        # invisible: editing the config changed no behaviour. Same defect class as the
        # `censor_tau` argument documented in tnk1/data/activity.py.
        sel = B.select_batch(P, al["batch"], al["max_per_cluster"], set(ytr.scaffold),
                             seed=a.cycle,
                             exploit_min_sel=al["exploit_min_sel"],
                             explore_min_mu=al["explore_min_mu"])
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
