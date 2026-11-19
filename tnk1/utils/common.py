"""Stage 0: determinism, hashing, config, tracking."""
from __future__ import annotations
import hashlib, json, os, random, subprocess
from pathlib import Path
import numpy as np

SEEDS = [42, 101, 202, 303, 404]


def seed_everything(seed: int) -> None:
    """Lock Python, NumPy, torch CPU/CUDA and cuDNN (Step 0.2)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# Config keys holding a filesystem location. Each is rewritten to an absolute path at load
# time so that no stage can resolve one against the process working directory. Adding a new
# path-valued key to configs/default.yaml means adding it here too, or it stays CWD-relative.
_PATH_KEYS = {
    "paths": ("raw", "processed", "runs"),
    "data": ("chembl_sqlite", "bindingdb_tsv", "pkis_csv", "kinomescan_csv", "dkk_csv",
             "kinase_table", "chembl_unlabelled_smiles"),
}


def load_config(path=None, *, resolve=True) -> dict:
    """Load the YAML config and (by default) absolutise every path-valued entry.

    `path=None` uses `tnk1.utils.paths.default_config()` -- the config that ships with the
    installed package, or `$TNK1_CONFIG`. It is NOT "configs/default.yaml" relative to the
    caller's CWD: that default is why every stage script had to be launched from the repo
    root, and why launching one from `scripts/` silently wrote a second artifact tree
    (see `tnk1.utils.paths.resolve_path`).

    Pass `resolve=False` to get the file verbatim -- only useful for round-tripping the
    config itself, never for running a stage.
    """
    import yaml
    from .paths import default_config, resolve_path

    p = Path(path).expanduser() if path is not None else default_config()
    if not p.is_absolute():
        p = resolve_path(p)
    if not p.exists():
        raise FileNotFoundError(
            f"config not found: {p}. Pass --config, set $TNK1_CONFIG, or run from a "
            f"checkout that still has configs/default.yaml."
        )
    with open(p) as fh:
        cfg = yaml.safe_load(fh)

    cfg["_config_path"] = str(p)
    if not resolve:
        return cfg
    for section, keys in _PATH_KEYS.items():
        if not isinstance(cfg.get(section), dict):
            continue
        for k in keys:
            v = cfg[section].get(k)
            if isinstance(v, str) and v:
                cfg[section][k] = str(resolve_path(v))
    return cfg


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_locked(df, path) -> str:
    """Write an immutable Parquet (CSV fallback) + .sha256 sidecar (Step 0.3). Refuses to overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} is frozen; write a new versioned file instead.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    digest = sha256_file(path)
    Path(str(path) + ".sha256").write_text(digest + "\n")
    return digest


def verify_locked(path) -> str:
    path = Path(path)
    expected = Path(str(path) + ".sha256").read_text().strip()
    actual = sha256_file(path)
    if expected != actual:
        raise RuntimeError(f"Checksum mismatch for {path}: {actual} != {expected}")
    return actual


def read_table(path):
    import pandas as pd
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


class Tracker:
    """Thin W&B / MLflow / null logger. Always records git hash, dataset hash, config."""

    def __init__(self, cfg: dict, run_name: str, dataset_sha: str = "", extra: dict | None = None):
        self.backend = cfg.get("tracking", {}).get("backend", "none")
        meta = {"git": git_hash(), "dataset_sha256": dataset_sha, "config": cfg, **(extra or {})}
        self.local = Path(cfg["paths"]["runs"]) / run_name
        self.local.mkdir(parents=True, exist_ok=True)
        (self.local / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        self._run = None
        if self.backend == "wandb":
            import wandb
            self._run = wandb.init(project=cfg["tracking"]["project"], name=run_name, config=meta)
        elif self.backend == "mlflow":
            import mlflow
            mlflow.set_experiment(cfg["tracking"]["project"]); mlflow.start_run(run_name=run_name)
            mlflow.log_dict(meta, "meta.json")

    def log(self, metrics: dict, step: int | None = None):
        with open(self.local / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps({"step": step, **metrics}, default=float) + "\n")
        if self.backend == "wandb":
            self._run.log(metrics, step=step)
        elif self.backend == "mlflow":
            import mlflow
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isscalar(v)}, step=step)
