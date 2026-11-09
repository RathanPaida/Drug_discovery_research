"""Stage 0 preflight: environment, determinism, feature-store integrity and per-stage readiness.

Previously this checked imports and determinism and always exited 0, so a run could start
with no data and no checkpoints and only fail hours later, deep inside a stage. It now
answers the question that actually matters -- *which stages can run right now, and what
exactly is missing for the rest* -- and exits non-zero if a requested stage is not ready.

    python scripts/run_stage0_check.py                 # full report, exit 0 unless env is broken
    python scripts/run_stage0_check.py --require rl    # exit 1 unless generation can run
    python scripts/run_stage0_check.py --version v1

Stage keys for --require: data, baselines, transfer, uq, admet, rl.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# Same bootstrap as every other stage script: make `tnk1` importable when this file is run
# directly (Python puts scripts/ on sys.path, not the package root). stage0 was the only
# stage script missing this, so `python scripts/run_stage0_check.py` failed on import
# unless the package happened to be pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.utils.common import load_config, seed_everything  # noqa: E402

# (module, needed-for) -- 'core' modules are mandatory, the rest gate specific stages.
MODULES = [
    ("numpy", "core"), ("pandas", "core"), ("scipy", "core"), ("sklearn", "core"),
    ("yaml", "core"), ("pyarrow", "core"),
    ("torch", "training + generation"), ("rdkit", "all chemistry stages"),
    ("xgboost", "stage 2 baselines"), ("gpytorch", "stage 2 Tanimoto GP"),
    ("esm", "stage 3 pocket embeddings (optional: precomputed store can substitute)"),
    ("tdc", "stage 5 ADMET download (optional: local parquet can substitute)"),
    ("aizynthfinder", "stage 8 retrosynthesis (optional)"),
]
CORE = {m for m, need in MODULES if need == "core"}

# Feature-store files each stage reads (names as they appear in features/manifest.json).
STAGE_FEATURES = {
    "data": ["kinome_all_measurements_3.95M.parquet", "tnk1_curated.parquet",
             "kinome_similarity_scores.parquet"],
    "baselines": ["tnk1_baseline_features_2248d.npz"],
    "transfer": ["kinome_multitask_300k.parquet", "kinome_similarity_scores.parquet"],
    "uq": [],
    "admet": ["admet_7endpoints.parquet"],
    "rl": ["unique_molecules_788k.parquet", "unique_molecules_fingerprints_2048bit.npz",
           "offtargets_curated.parquet"],
}

# Artifacts each stage needs its predecessors to have produced, under data/processed/.
def stage_artifacts(version: str) -> dict[str, list[str]]:
    return {
        "data": [],
        "baselines": [f"kinome_long_{version}.parquet", f"splits_{version}.json"],
        "transfer": [f"kinome_long_{version}.parquet", f"splits_{version}.json"],
        "uq": [f"kinome_long_{version}.parquet", "kinase_store.npz"],
        "admet": ["desc_scaler.pkl"],
        "rl": ["ensemble", "calibrator.pkl", "ad.pkl", "admet_bundles.pt",
               "kinase_store.npz", f"kinome_long_{version}.parquet"],
    }


def _check_modules() -> list[str]:
    print("Environment")
    missing_core = []
    for name, need in MODULES:
        try:
            mod = importlib.import_module(name)
            print(f"  [ok]   {name:<15} {getattr(mod, '__version__', '')}")
        except Exception:
            print(f"  [MISS] {name:<15} needed for: {need}")
            if name in CORE:
                missing_core.append(name)
    return missing_core


def _check_determinism() -> None:
    try:
        import torch
        seed_everything(42); a = torch.randn(4)
        seed_everything(42); b = torch.randn(4)
        ok = bool(torch.equal(a, b))
        print(f"\nDeterminism\n  [{'ok' if ok else 'FAIL'}]   seeded RNG reproducible")
        print(f"  [info] CUDA available: {torch.cuda.is_available()}"
              + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
        if not ok:
            raise SystemExit("seeded RNG is not reproducible; refusing to proceed")
    except ImportError:
        print("\nDeterminism\n  [skip] torch not installed")


def _check_feature_store() -> dict[str, str]:
    from tnk1.utils.paths import features_dir, verify_features
    d = features_dir()
    print(f"\nFeature store  ({d})")
    try:
        status = verify_features()
    except FileNotFoundError as e:
        print(f"  [FAIL] {e}")
        return {}
    ok = sum(v == "ok" for v in status.values())
    absent = sum(v == "absent" for v in status.values())
    corrupt = len(status) - ok - absent
    print(f"  verified={ok}  absent={absent}  corrupt={corrupt}  (of {len(status)})")
    for name, state in sorted(status.items()):
        if state not in ("ok", "absent"):
            print(f"  [CORRUPT] {name}: {state}")
    if corrupt:
        print("  -> do NOT report results computed from this store")
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--require", choices=sorted(STAGE_FEATURES), default=None,
                    help="exit 1 unless this stage is ready to run")
    args = ap.parse_args(argv)

    missing_core = _check_modules()
    _check_determinism()

    print("\nConfig")
    try:
        cfg = load_config(args.config)
        proc = Path(cfg["paths"]["processed"])
        # these are read via required-key access at runtime; surface a typo here instead
        for key in ("mw_max_hard", "herg_max", "sa_max"):
            cfg["filters"][key]
        print(f"  [ok]   {args.config} loaded; filters complete; processed={proc}")
    except Exception as e:
        print(f"  [FAIL] {args.config}: {type(e).__name__}: {e}")
        return 1

    feat_status = _check_feature_store()
    artifacts = stage_artifacts(args.version)

    print(f"\nStage readiness  (version={args.version}, processed={proc})")
    ready: dict[str, bool] = {}
    for stage in STAGE_FEATURES:
        need_feat = [f for f in STAGE_FEATURES[stage] if feat_status.get(f) != "ok"]
        need_art = [a for a in artifacts[stage] if not (proc / a).exists()]
        if stage == "rl" and not need_art and not (proc / "ensemble").glob("member_*.pt"):
            need_art = ["ensemble/member_*.pt"]
        ready[stage] = not need_feat and not need_art
        mark = "READY" if ready[stage] else "BLOCKED"
        print(f"  [{mark:<7}] {stage}")
        for f in need_feat:
            print(f"              missing feature file: {f}")
        for a in need_art:
            print(f"              missing artifact:     {proc / a}")

    print()
    if missing_core:
        print(f"FAIL: core modules missing: {missing_core}")
        return 1
    if args.require:
        if ready[args.require]:
            print(f"OK: stage '{args.require}' is ready to run.")
            return 0
        print(f"FAIL: stage '{args.require}' is NOT ready -- see missing items above.\n"
              f"      Restore the feature store (set $TNK1_FEATURES_DIR) and run the\n"
              f"      preceding stages; scoring and generation are fail-closed and will\n"
              f"      not substitute defaults for missing models or data.")
        return 1
    n_ready = sum(ready.values())
    print(f"Environment OK. {n_ready}/{len(ready)} stages ready to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
