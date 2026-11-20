"""One entry point that runs the whole pipeline, in order, fail-closed and resumable.

    python scripts/run_pipeline.py --version v3                  # everything not yet done
    python scripts/run_pipeline.py --version v3 --from ens        # resume at the ensemble
    python scripts/run_pipeline.py --version v3 --only admet      # a single stage
    python scripts/run_pipeline.py --version v3 --plan            # print the plan, run nothing
    python scripts/run_pipeline.py --version v3 --smoke           # tiny config, minutes not hours

Why this exists
---------------
The stages were runnable only as nine separate `make` targets whose ordering,
preconditions and version flags lived in a README. Nothing checked that a stage's inputs
existed before it started, nothing checked that its outputs appeared afterwards, and a
failure at stage 5 of 9 left no record of what had completed. A long run therefore failed
hours in, silently, and could not be resumed except by hand.

Each stage here declares its `needs` (artifacts that must already exist) and `makes`
(artifacts it must produce). The runner refuses to start a stage whose inputs are missing,
and treats "the command exited 0 but the output is absent" as a failure -- that is the
exact shape of the Stage-7 defect in CHANGES_V2.md, where an unparseable TOML meant
REINVENT never ran while the pipeline log recorded a successful 0.1-minute stage.

State lives in `<runs>/pipeline/state.json`, so a re-run skips completed stages and
`--from` is rarely needed.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.utils.common import git_hash, load_config, sha256_file  # noqa: E402
from tnk1.utils.paths import repo_root  # noqa: E402

PY = sys.executable


@dataclass
class Stage:
    name: str
    argv: list[str]
    needs: list[str] = field(default_factory=list)      # paths, {proc}/{runs}/{ver} templated
    makes: list[str] = field(default_factory=list)
    note: str = ""
    optional: bool = False       # a failure is reported but does not stop the run
    gpu: bool = False
    # True for a stage whose OUTPUT does not depend on kinome_long_<version>.parquet, so
    # rebuilding the Stage-1 table does not invalidate it. `prior` reads the 788k corpus
    # straight from the feature store and never constructs a Ctx; `store` derives from
    # kinases.csv plus ESM-2. Without this they were reported STALE after every Stage-1
    # rebuild and would be retrained for hours to produce the same bytes.
    version_independent: bool = False


def build_plan(version: str, device: str, cycle: int, skip_retro: bool,
               zoo_tag: str = "core", zoo_repeats: int = 10, rl_algo: str = "dap",
               stack_metric: str = "spearman") -> list[Stage]:
    S = lambda *a: [str(x) for x in a]
    stage3 = "scripts/run_stage3_transfer.py"
    stage4 = "scripts/run_stage4_uq.py"
    stage5 = "scripts/run_stage5_selectivity_admet.py"
    return [
        Stage("preflight", S(PY, "scripts/run_stage0_check.py", "--version", version, "--strict"),
              note="environment, determinism, feature-store sha256, ADMET endpoint contract"),

        Stage("data", S(PY, "scripts/run_stage1_data.py", "--version", version),
              makes=["{proc}/kinome_long_{ver}.parquet", "{proc}/splits_{ver}.json"],
              note="extract -> standardize -> pActivity -> measurability filter -> freeze splits"),

        Stage("baselines", S(PY, "scripts/run_stage2_baselines.py", "--version", version),
              needs=["{proc}/kinome_long_{ver}.parquet", "{proc}/splits_{ver}.json"],
              makes=["{runs}/stage2_baselines.csv"],
              note="kNN / RF / XGB / TanimotoGP under nested scaffold CV"),

        Stage("store", S(PY, stage3, "store", "--device", device, "--version", version),
              needs=["{proc}/kinome_long_{ver}.parquet"],
              makes=["{proc}/kinase_store.npz"], gpu=True, version_independent=True,
              note="ESM-2 650M pocket embeddings + KLIFS similarity -> kinase store"),

        Stage("pretrain", S(PY, stage3, "pretrain", "--device", device, "--version", version),
              needs=["{proc}/kinome_long_{ver}.parquet"],
              makes=["{proc}/ssl_encoder.pt"], gpu=True, optional=True,
              note="masked-atom SSL on the unlabelled corpus (optional: trunk trains without it)"),

        Stage("lambda", S(PY, stage3, "lambda", "--device", device, "--version", version),
              needs=["{proc}/kinase_store.npz"],
              makes=["{runs}/loko_lambda.csv"], gpu=True, optional=True,
              note="leave-one-kinase-out tuning of the similarity weight lambda"),

        Stage("ablate", S(PY, stage3, "ablate", "--device", device, "--version", version),
              needs=["{proc}/kinase_store.npz"],
              makes=["{runs}/stage3_ablation.csv"], gpu=True, optional=True,
              note="A0-A7 ladder x 5 seeds: does the transfer stack beat a scratch D-MPNN?"),

        Stage("ens", S(PY, stage4, "train", "--device", device, "--version", version),
              needs=["{proc}/kinase_store.npz", "{proc}/kinome_long_{ver}.parquet"],
              makes=["{proc}/ensemble"], gpu=True,
              note="heteroscedastic deep ensemble, one fine-tune per member"),

        Stage("oof", S(PY, stage4, "oof", "--device", device, "--version", version),
              needs=["{proc}/ensemble"],
              makes=["{runs}/stage4_oof_predictions.csv"], gpu=True,
              note="out-of-fold predictions for the calibrator"),

        Stage("calibrate", S(PY, stage4, "calibrate", "--device", device, "--version", version),
              needs=["{runs}/stage4_oof_predictions.csv"],
              makes=["{proc}/calibrator.pkl", "{proc}/ad.pkl"], gpu=True,
              note="variance temperature + split-conformal quantile + applicability domain"),

        Stage("compare", S(PY, stage4, "compare", "--device", device, "--version", version),
              needs=["{proc}/ensemble", "{proc}/calibrator.pkl"],
              makes=["{runs}/stage4_uq_comparison.csv"], gpu=True, optional=True,
              note="ensemble vs MC-dropout vs calibrated, on the held-out test fold"),

        Stage("critic", S(PY, "scripts/compare_critic.py", "--version", version),
              needs=["{runs}/stage4_oof_predictions.csv"],
              makes=["{runs}/critic_comparison.csv"],
              note="paired RF/XGB/kNN vs the deep ensemble over the 5 outer folds -- "
                   "is the RL critic actually the best model?"),

        Stage("admet", S(PY, stage5, "admet", "--device", device, "--version", version),
              needs=["{proc}/kinome_long_{ver}.parquet"],
              makes=["{proc}/admet_bundles.pt", "{proc}/desc_scaler.pkl"], gpu=True,
              note="7 TDC endpoints, 5-member D-MPNN ensemble each"),

        Stage("selectivity", S(PY, stage5, "selectivity", "--device", device, "--version", version),
              needs=["{proc}/ensemble", "{proc}/kinase_store.npz"],
              makes=["{runs}/stage5_selectivity_reference.csv"], gpu=True,
              note="ensemble-propagated off-target margins + pan-kinase control check"),

        # ---- the model zoo and the stacked critic (Cycle-03 additions) ---------------
        # `zoo` is CPU-only and embarrassingly cheap next to the GPU stages: ~26 models x
        # nested CV on 172 molecules. It is on the critical path to a defensible critic,
        # because `stack` consumes its out-of-fold matrix and the RL stages can then use
        # the result instead of the deep ensemble that lost the paired comparison.
        Stage("zoo", S(PY, "scripts/run_stage2b_zoo.py", "--version", version,
                       "--tag", zoo_tag, "--repeats", zoo_repeats),
              needs=["{proc}/kinome_long_{ver}.parquet", "{proc}/splits_{ver}.json"],
              makes=["{runs}/stage2b_zoo_summary.csv", "{runs}/stage2b_zoo_oof.csv"],
              note="nested-CV benchmark of every registered base model x representation"),

        # `stack` needs `oof` only to include the deep ensemble as a candidate; it runs
        # without it and says so. Declared as a need anyway so the orchestrator orders the
        # two correctly when both are in the plan.
        Stage("stack", S(PY, "scripts/run_stage4b_stack.py", "--version", version,
                         "--metric", stack_metric),
              needs=["{runs}/stage2b_zoo_oof.csv"],
              makes=["{runs}/stage4b_stack.json", "{proc}/stack.pkl"],
              note="discrete super learner over the zoo + the deep ensemble"),

        # Task selection is CPU-only, takes seconds, and its answer changes what the
        # kinome stage should be pretrained ON -- so it is a separate, cheap stage rather
        # than part of the expensive ladder.
        Stage("taskselect", S(PY, "scripts/run_stage3b_transfer.py", "--version", version,
                              "--task-selection-only"),
              needs=["{proc}/kinome_long_{ver}.parquet"],
              makes=["{runs}/stage3b_transfer_task_selection.csv"],
              note="per-donor-kinase SAR correlation with TNK1: which of the 491 help"),

        Stage("transfer", S(PY, "scripts/run_stage3b_transfer.py", "--version", version,
                            "--device", device),
              needs=["{proc}/kinome_long_{ver}.parquet", "{proc}/kinase_store.npz"],
              makes=["{runs}/stage3b_transfer.csv"], gpu=True, optional=True,
              note="fine-tuning strategy ladder: LP-FT, surgical, gradual, SWA, mixout"),

        Stage("prior", S(PY, "-m", "rl_min.train_prior", "--device", device),
              makes=["{proc}/rl_min/prior.pt"], gpu=True, version_independent=True,
              note="maximum-likelihood SMILES prior on the 788k corpus (the RL anchor)"),

        # `--algo` defaults to dap, so a bare `--only rl` reproduces the v3 run exactly.
        # `--out` carries the algorithm name whenever it is NOT dap, so a comparison run
        # cannot overwrite the incumbent's artifacts -- and so `runs/rl_min/` keeps
        # meaning "the DAP run of record".
        Stage("rl", S(PY, "-m", "rl_min.run_rl", "--scorer", "pipeline",
                      "--algo", rl_algo,
                      "--prior", "{proc}/rl_min/prior.pt",
                      "--vocab", "{proc}/rl_min/vocab.txt",
                      *([] if rl_algo == "dap" else ["--out", f"{{runs}}/rl_{rl_algo}"]),
                      "--version", version, "--device", device),
              needs=["{proc}/rl_min/prior.pt", "{proc}/ensemble", "{proc}/calibrator.pkl",
                     "{proc}/ad.pkl", "{proc}/admet_bundles.pt"],
              makes=(["{runs}/rl_min/agent.pt", "{runs}/rl_min/history.csv"]
                     if rl_algo == "dap"
                     else [f"{{runs}}/rl_{rl_algo}/agent.pt",
                           f"{{runs}}/rl_{rl_algo}/history.csv"]),
              gpu=True,
              note=f"{rl_algo} policy optimisation against the PRODUCTION reward"),

        Stage("cycle", S(PY, "scripts/run_stage8_cycle.py", "prepare", "--cycle", cycle,
                         "--device", device, "--version", version,
                         *(["--skip-retro"] if skip_retro else []),
                         # The pool is sampled from the trained agent, which is what
                         # `al.pool_size` has always claimed. Passing only --rl-csv fed
                         # Stage 8 the top-100 replay buffer, so the funnel was starved by
                         # three orders of magnitude before any filter ran.
                         "--agent", "{runs}/rl_min/agent.pt",
                         "--vocab", "{proc}/rl_min/vocab.txt",
                         "--rl-csv", "{runs}/rl_min/best_molecules.csv"),
              needs=["{runs}/rl_min/agent.pt", "{proc}/rl_min/vocab.txt",
                     "{runs}/rl_min/best_molecules.csv", "{proc}/ensemble"],
              makes=["{runs}/cycle_{cyc:02d}/candidates_pre_docking.csv"], gpu=True,
              optional=True,
              note="pool -> hard gate -> dual-model audit -> Pareto -> docking request"),

        Stage("report", S(PY, "scripts/run_stage9_report.py"),
              optional=True,
              note="prospective hit rates from wet-lab returns (no-op until Team B reports)"),
    ]


def expand(p: str, ctx: dict) -> Path:
    return Path(p.format(**ctx))


def dataset_fingerprint(proc: Path, version: str) -> str:
    """The frozen Stage-1 table's sha256, read from its sidecar.

    This is the provenance key for every downstream stage. Without it, "the output file
    exists" is mistaken for "the output is current": `data/processed/ensemble/` was trained
    on kinome_long_v1.parquet (426,286 rows, the pre-filter table), so a v3 run that merely
    checked for the directory's existence would skip retraining and the 1,103,498-row
    filtered table would never reach a model. Results would then be attributed to v3 while
    actually coming from v1 -- a provenance error no checksum catches, because each
    individual file is internally consistent.
    """
    side = proc / f"kinome_long_{version}.parquet.sha256"
    if side.exists():
        return side.read_text().strip()
    p = proc / f"kinome_long_{version}.parquet"
    return sha256_file(p) if p.exists() else ""


def stage_is_current(state: dict, name: str, version: str, fp: str,
                     version_independent: bool = False) -> tuple[bool, str]:
    """Has this stage already run against THIS version and THIS dataset fingerprint?"""
    rec = state.get(name)
    if not rec:
        return False, "no provenance record"
    if version_independent:
        return True, "current (output does not depend on the dataset version)"
    if rec.get("version") != version:
        return False, f"built for version {rec.get('version')!r}, not {version!r}"
    if fp and rec.get("dataset_sha256") and rec["dataset_sha256"] != fp:
        return False, (f"built from dataset {rec['dataset_sha256'][:12]}..., "
                       f"current is {fp[:12]}...")
    return True, "current"


def artifact_ok(p: Path) -> bool:
    """Present and non-trivial. A directory must contain something; a file must be non-empty.

    An empty `ensemble/` directory used to satisfy an `exists()` check and let Stage 7
    start, which then died inside `Ensemble.__init__`'s assert. Same class of bug as the
    `Path.glob()` truthiness defect in run_stage0_check.py.
    """
    if not p.exists():
        return False
    if p.is_dir():
        return any(p.iterdir())
    return p.stat().st_size > 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v3",
                    help="dataset version for kinome_long_<v>.parquet (default v3, the "
                         "measurability-filtered table)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cycle", type=int, default=2)
    ap.add_argument("--config", default=None)
    ap.add_argument("--from", dest="from_stage", default=None)
    ap.add_argument("--to", dest="to_stage", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--force", action="store_true",
                    help="re-run stages whose outputs already exist")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a required stage fails (default: stop)")
    ap.add_argument("--skip-retro", action="store_true", default=True,
                    help="bypass AiZynthFinder in stage 8 (its model files are not shipped)")
    ap.add_argument("--assume-current", action="store_true",
                    help="adopt existing outputs whose provenance does not match this "
                         "version/dataset instead of rebuilding them")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    # --- Cycle-03 stage parameters ---------------------------------------------------
    ap.add_argument("--zoo-tag", default="core",
                    help="`zoo` stage: which shortlist to benchmark (core|slow|probe)")
    ap.add_argument("--zoo-repeats", type=int, default=10,
                    help="`zoo` stage: repeated-CV repeats. 10 gives 50 fold scores, which "
                         "is the minimum for a paired test that can reach p < 0.05; "
                         "1 reuses only the frozen splits")
    ap.add_argument("--stack-metric", default="spearman",
                    choices=["rmse", "mae", "spearman"],
                    help="`stack` stage: the CV risk the discrete super learner minimises")
    ap.add_argument("--rl-algo", default="dap",
                    help="`rl` stage: policy-optimisation objective "
                         "(dap|ahc|ppo|grpo|rloo|gflownet). Anything but dap writes to "
                         "runs/rl_<algo>/ so the DAP run of record is never overwritten")
    ap.add_argument("--timeout", type=float, default=None,
                    help="per-stage timeout in seconds (default: none)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    root = repo_root()
    proc, runs = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["runs"])
    ctx = {"proc": proc, "runs": runs, "ver": a.version, "cyc": a.cycle}

    fp = dataset_fingerprint(proc, a.version)
    plan = build_plan(a.version, a.device, a.cycle, a.skip_retro,
                      zoo_tag=a.zoo_tag, zoo_repeats=a.zoo_repeats,
                      rl_algo=a.rl_algo, stack_metric=a.stack_metric)
    # The `data` stage is SELF-CERTIFYING: its output carries a .sha256 sidecar written by
    # `save_locked`, and that digest is the very fingerprint downstream stages are compared
    # against. If the table verifies, the stage is current by definition -- no state record
    # needed. Seeding it here means a table built by running Stage 1 directly (rather than
    # through this runner) is not reported as stale.
    _seed_data_provenance = None
    names = [s.name for s in plan]

    # stage selection
    if a.only:
        unknown = [n for n in a.only if n not in names]
        if unknown:
            raise SystemExit(f"unknown stage(s) {unknown}; known: {names}")
        plan = [s for s in plan if s.name in a.only]
    else:
        if a.from_stage:
            if a.from_stage not in names:
                raise SystemExit(f"unknown --from {a.from_stage!r}; known: {names}")
            plan = plan[names.index(a.from_stage):]
        if a.to_stage:
            cur = [s.name for s in plan]
            if a.to_stage not in cur:
                raise SystemExit(f"unknown --to {a.to_stage!r}; remaining: {cur}")
            plan = plan[: cur.index(a.to_stage) + 1]
    plan = [s for s in plan if s.name not in set(a.skip)]

    state_dir = runs / "pipeline"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_p = state_dir / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}

    # see the note next to `_seed_data_provenance`
    if fp and "data" not in state:
        tbl = proc / f"kinome_long_{a.version}.parquet"
        if artifact_ok(tbl) and artifact_ok(proc / f"splits_{a.version}.json"):
            try:
                from tnk1.utils.common import verify_locked
                verify_locked(tbl)
                state["data"] = {"version": a.version, "dataset_sha256": fp,
                                 "at": "self-certified via .sha256 sidecar"}
                state_p.write_text(json.dumps(state, indent=2))
            except Exception as e:
                print(f"[data] sidecar does NOT verify ({type(e).__name__}: {e}); "
                      f"treating the table as stale")

    if a.plan:
        print(f"Plan  (version={a.version}, device={a.device}, cycle={a.cycle})")
        print(f"  config    {cfg['_config_path']}")
        print(f"  processed {proc}\n  runs      {runs}\n")
        print(f"  dataset   {fp[:16] or '(stage 1 not run)'}\n")
        for s in plan:
            present = all(artifact_ok(expand(m, ctx)) for m in s.makes) if s.makes else False
            cur, why = stage_is_current(state, s.name, a.version, fp,
                                        s.version_independent)
            if present and cur:
                flag = "done"
            elif present:
                flag = "STALE"
            elif s.optional:
                flag = "opt "
            else:
                flag = "    "
            print(f"  [{flag}] {s.name:<12} {s.note}")
            if flag == "STALE":
                print(f"            !! {why}")
            for m in s.makes:
                print(f"            -> {expand(m, ctx)}")
        return 0

    print(f"=== pipeline  version={a.version} device={a.device} ===")
    print(f"config {cfg['_config_path']}")
    print(f"git    {git_hash()}")
    print(f"stages {[s.name for s in plan]}\n")

    run_meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "version": a.version,
                "device": a.device, "git": git_hash(), "config": cfg["_config_path"],
                "stages": {}}
    overall = 0

    for s in plan:
        tag = f"[{s.name}]"
        made = [expand(m, ctx) for m in s.makes]
        present = bool(s.makes) and all(artifact_ok(m) for m in made)
        current, why = stage_is_current(state, s.name, a.version, fp,
                                        s.version_independent)
        if present and current and not a.force:
            print(f"{tag} skip - already run against {a.version}/{fp[:12] or 'n-a'}")
            run_meta["stages"][s.name] = {"status": "skipped", "reason": "current"}
            continue
        if present and not current and not a.force and not a.assume_current:
            # The artifacts exist but were not produced from this dataset. Re-running is the
            # safe default; `--assume-current` adopts them if you know they are fine.
            print(f"{tag} STALE - outputs exist but {why}; re-running")
        if present and not current and a.assume_current:
            print(f"{tag} adopt - outputs exist, {why}, --assume-current given")
            state[s.name] = {"version": a.version, "dataset_sha256": fp,
                             "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "adopted": True}
            run_meta["stages"][s.name] = {"status": "adopted", "reason": why}
            state_p.write_text(json.dumps(state, indent=2))
            continue

        missing = [str(expand(n, ctx)) for n in s.needs if not artifact_ok(expand(n, ctx))]
        if missing:
            msg = f"{tag} BLOCKED - missing input(s):\n    " + "\n    ".join(missing)
            print(msg)
            run_meta["stages"][s.name] = {"status": "blocked", "missing": missing}
            if not s.optional and not a.keep_going:
                overall = 1
                break
            overall = overall or (0 if s.optional else 1)
            continue

        argv_r = [x.format(**ctx) if isinstance(x, str) else x for x in s.argv]
        log_p = state_dir / f"{s.name}.log"
        print(f"{tag} {s.note}")
        print(f"{tag} $ {' '.join(shlex.quote(x) for x in argv_r)}")
        print(f"{tag} log -> {log_p}")
        t0 = time.time()
        with open(log_p, "w", encoding="utf-8") as fh:
            fh.write(f"# {' '.join(argv_r)}\n# cwd={root}\n\n")
            fh.flush()
            # Propagate the config to every child via $TNK1_CONFIG rather than appending
            # `--config` to each stage's argv. One place instead of eighteen, and it also
            # reaches the `python -m rl_min.*` stages, which resolve their config through
            # `default_config()`.
            #
            # This was a real bug while writing this file: `--config` was accepted here and
            # passed to NOTHING, so a run launched with a scratch config silently executed
            # every stage against configs/default.yaml -- i.e. against the real
            # data/processed tree. Exactly the class of defect this runner exists to catch,
            # and it was caught by watching which directory the first stage touched.
            env = {**os.environ, "PYTHONUNBUFFERED": "1",
                   "TNK1_CONFIG": str(cfg["_config_path"])}
            try:
                rc = subprocess.call(argv_r, cwd=root, stdout=fh,
                                     stderr=subprocess.STDOUT, env=env, timeout=a.timeout)
            except subprocess.TimeoutExpired:
                rc = 124
                fh.write(f"\n# TIMEOUT after {a.timeout}s\n")
        dt = time.time() - t0

        # "exited 0" is not success: the artifact must exist. This is the Stage-7 failure
        # mode from CHANGES_V2.md, where the stage logged success and produced nothing.
        absent = [str(m) for m in made if not artifact_ok(m)]
        ok = rc == 0 and not absent
        status = "ok" if ok else ("failed" if rc else "no-output")
        run_meta["stages"][s.name] = {
            "status": status, "returncode": rc, "seconds": round(dt, 1),
            "log": str(log_p), "missing_outputs": absent,
        }
        if ok:
            print(f"{tag} ok in {dt/60:.1f} min")
            for m in made:
                sz = m.stat().st_size if m.is_file() else sum(
                    f.stat().st_size for f in m.rglob('*') if f.is_file())
                print(f"{tag}   -> {m}  ({sz/1e6:.1f} MB)")
            if s.makes:
                # Re-read the fingerprint: the `data` stage is what creates it.
                cur_fp = dataset_fingerprint(proc, a.version)
                state[s.name] = {"version": a.version, "dataset_sha256": cur_fp,
                                 "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "seconds": round(dt, 1)}
        else:
            print(f"{tag} FAILED after {dt/60:.1f} min (exit {rc})")
            if absent:
                print(f"{tag}   command exited {rc} but these were not produced:")
                for m in absent:
                    print(f"{tag}     {m}")
            print(f"{tag}   last 25 log lines:")
            try:
                for line in log_p.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]:
                    print(f"{tag}   | {line}")
            except OSError:
                pass
            if s.optional:
                print(f"{tag}   stage is optional; continuing")
            else:
                overall = 1
                if not a.keep_going:
                    print(f"\nStopping. Fix the above, then resume with:\n"
                          f"  python scripts/run_pipeline.py --version {a.version} --from {s.name}")
                    break

        state_p.write_text(json.dumps(state, indent=2))
        (state_dir / "last_run.json").write_text(json.dumps(run_meta, indent=2))

    run_meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_meta["exit"] = overall
    (state_dir / "last_run.json").write_text(json.dumps(run_meta, indent=2))
    state_p.write_text(json.dumps(state, indent=2))

    print("\n=== summary ===")
    for n, m in run_meta["stages"].items():
        extra = f" ({m['seconds']/60:.1f} min)" if "seconds" in m else ""
        print(f"  {m['status']:<9} {n}{extra}")
    print(f"\nrun record -> {state_dir / 'last_run.json'}")
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
