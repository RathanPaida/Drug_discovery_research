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


def _std_one(smi):
    """Module-level so it can be pickled for a process pool.

    The import is inside the function on purpose: `standardize_smiles` is imported
    inside `main()`, so it is NOT in this module's namespace, and a module-level
    reference raises NameError in every worker. Catching that would have returned None
    for all 200,000 molecules and the stage would have reported "No candidate passed
    strict profile filters" -- a real-looking scientific result produced by an import
    error. Only the molecule-level failure is caught, and the import is left to raise.
    """
    from tnk1.data.standardize import standardize_smiles
    try:
        return standardize_smiles(smi)
    except (ValueError, RuntimeError, AttributeError, TypeError):
        # one unparseable structure, not a broken environment
        return None


def _standardize_pool(pool, jobs=None, chunk=256):
    """Standardize the candidate pool across cores, with progress.

    This was `[standardize_smiles(s) for s in pool]` -- one core, no output. It was
    invisible while the pool was ~89 molecules, which is all the starved funnel ever
    produced. Now that `--agent` supplies the real `al.pool_size` (200,000), it is the
    single most expensive step in the stage and the stage runs eight times, once per RL
    objective.

    Measured on this node: 12.8 ms/molecule for curated ChEMBL structures, but RL output
    runs several times slower -- generated SMILES carry more tautomeric centres and
    `tautomer.Canonicalize` is combinatorial in those. At ~70 ms/molecule a 200,000
    pool is ~4 hours on one core, so the eight cycles would spend well over a day inside
    this one list comprehension while 15 of the job's 16 cores sat idle.

    Progress is printed every `chunk*40` molecules because the old silence made a slow
    run indistinguishable from a hang -- there was no way to tell, from outside, whether
    four hours meant "nearly done" or "deadlocked".
    """
    import os
    import time
    n = len(pool)
    if jobs is None:
        jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    jobs = max(1, min(jobs, n))
    t0 = time.time()
    print(f"[stage8] standardizing {n:,} pooled molecules on {jobs} core(s)", flush=True)

    def _report(done):
        el = time.time() - t0
        rate = done / el if el > 0 else 0
        eta = (n - done) / rate if rate > 0 else float("nan")
        print(f"[stage8]   standardized {done:,}/{n:,} "
              f"({el/60:.1f} min elapsed, {rate:.0f}/s, ~{eta/60:.1f} min left)",
              flush=True)

    if jobs == 1:
        out = []
        for i, s in enumerate(pool, 1):
            out.append(_std_one(s))
            if i % (chunk * 40) == 0:
                _report(i)
        _report(n)
        return out

    import multiprocessing as mp
    try:
        with mp.Pool(jobs) as p:
            out = []
            for i, r in enumerate(p.imap(_std_one, pool, chunksize=chunk), 1):
                out.append(r)
                if i % (chunk * 40) == 0:
                    _report(i)
    except Exception as e:
        # A pool failure must not lose the stage; fall back rather than crash after the
        # expensive sampling step that produced this pool.
        print(f"[stage8] WARNING process pool failed ({type(e).__name__}: {e}); "
              f"falling back to one core", flush=True)
        return _standardize_pool(pool, jobs=1, chunk=chunk)
    _report(n)
    return out


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
    ap.add_argument("--std-jobs", type=int, default=None,
                    help="cores for standardizing the pool (default: SLURM_CPUS_PER_TASK, "
                         "else os.cpu_count). 1 restores the old serial behaviour, which "
                         "on a 200k pool of generated molecules is about four hours.")
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
        from tnk1.zoo.deployment import load_predictor
        ens = load_predictor(C)
        cal, ad = pickle.load(open(C.proc / ("selected_calibrator.pkl" if cfg.get("critic", {}).get("backend") == "selected" else "calibrator.pkl"), "rb")), pickle.load(open(C.proc / ("selected_ad.pkl" if cfg.get("critic", {}).get("backend") == "selected" else "ad.pkl"), "rb"))
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
        std = _standardize_pool(list(pool), jobs=a.std_jobs); rows = [r for r in std if r]
        P = pd.DataFrame(rows).drop_duplicates("inchikey_block1")
        P = P[~P.inchikey_block1.isin(set(C.df.inchikey_block1))].reset_index(drop=True)      # vs ChEMBL + in-house
        import json
        centres = C.runs / "reward_centres.json"
        if centres.exists():
            cfg["rl"].update(json.loads(centres.read_text()))
        sc = scorer.score(list(P.smiles), advance=False); P = P.assign(**{k: v for k, v in sc.items() if k != "beta"})
        from tnk1.rl.reward import tpp_select
        tpp_mask, tpp_report = tpp_select(P, cfg.get("tpp"))
        if tpp_report["skipped"]: raise ValueError(f"Missing TPP endpoints: {tpp_report['skipped']}")
        P = P[(P.R > 0) & tpp_mask].reset_index(drop=True)                                                 # I_hard(x) = 1
        if P.empty:
            P.to_csv(cyc / "candidates_pre_docking.csv", index=False)
            P[["smiles"]].assign(docking_ok="").to_csv(cyc / "docking_request.csv", index=False)
            print("No candidate passed strict profile filters")
            return
        # cross-architecture audit: D-MPNN ensemble vs Tanimoto GP
        tr = C.tnk1_rows(C.dev_keys, censored=False)
        gp = TanimotoGP(device=a.device).fit(morgan_count(tr.smiles), tr.pact.values)
        P["mu_gp"] = gp.predict(morgan_count(P.smiles)); P = P[(P.mu - P.mu_gp).abs() <= al["audit_tol"]].reset_index(drop=True)
        if P.empty:
            P.to_csv(cyc / "candidates_pre_docking.csv", index=False)
            P[["smiles"]].assign(docking_ok="").to_csv(cyc / "docking_request.csv", index=False)
            print("No candidate passed independent-model agreement")
            return
        objs = np.c_[P.mu - P.sigma_epi, P.mu_sel - P.sigma_sel, P.adme, P.ad, P.sa, P.safety]
        P = P.iloc[ACQ.top_by_pareto(objs, al["pareto_keep"], P.R.values)].reset_index(drop=True)
        if a.skip_retro:
            print(f"[stage8] --skip-retro: keeping all {len(P)} candidates WITHOUT a "
                  f"synthesisability screen. Downstream rows carry retro_ok=True by "
                  f"assertion, not by evidence -- do not report them as synthesisable.")
            P = P.assign(retro_ok=True, state_score=float("nan"), n_steps=-1)
        else:
            rs = retro_screen(list(P.smiles), a.aizynth_config, al["retro_max_steps"], al["retro_min_state_score"])
            P = P.merge(rs[["smiles", "retro_ok", "state_score", "n_steps", "routes_json", "route_method", "stock_name"]], on="smiles"); P = P[P.retro_ok]
        P.to_csv(cyc / "candidates_pre_docking.csv", index=False)
        P[["smiles"]].assign(docking_ok="").to_csv(cyc / "docking_request.csv", index=False)
        print(f"{len(P)} retro-feasible candidates -> send {cyc/'docking_request.csv'} to Team B (docking: hinge H-bond, no clash)")

    elif a.cmd == "select":
        ens, cal, ad, _ = load_stack(); P = pd.read_csv(cyc / "candidates_pre_docking.csv")
        dk = pd.read_csv(a.docking); P = P.merge(dk[["smiles", "docking_ok"]], on="smiles"); P = P[P.docking_ok.astype(str).str.lower().isin(["1", "true", "yes", "y"])].copy()
        ytr = C.tnk1_rows(C.dev_keys, censored=False); ybest = float(ytr.pact.max())
        P["ei"] = ACQ.expected_improvement(P.mu.values, P.sigma_epi.values, ybest, al["ei_xi"])
        P["ucb"] = ACQ.ucb(P.mu.values, P.sigma_epi.values, al["ucb_kappa"]); P["lcb"] = ACQ.lcb(P.mu.values, P.sigma_epi.values, al["exploit_lcb_k"])
        P["cluster"] = B.butina_clusters(list(P.smiles), al["butina_dist"]); P["passes_filters"] = True
        from tnk1.rl.reward import tpp_select
        mask, report = tpp_select(P, cfg.get("tpp"))
        if report["skipped"]:
            raise ValueError(f"Missing mandatory final-selection endpoints: {report['skipped']}")
        P["passes_filters"] = mask
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
