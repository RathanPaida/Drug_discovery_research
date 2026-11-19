"""Single source of truth for locating the precomputed feature store, plus manifest
integrity verification.

Why this module exists
----------------------
The feature-store location was previously hardcoded in seven places as
`Path("d:/ML_Research/ml_research/ml_research/features")` -- in two of them as the PRIMARY
value rather than a fallback. On any machine that is not that one, those stages either
crashed or (worse, given the fail-open defaults elsewhere) silently proceeded without the
reference data. Resolution order is now:

    1. $TNK1_FEATURES_DIR                     (explicit override; wins everywhere)
    2. <repo root>/ml_research/features       (the committed layout)
    3. ./ml_research/features, ./features     (cwd-relative, for ad-hoc runs)

The store itself is gitignored (~280 MB of parquet/npz), so only `manifest.json` and the
summary JSONs are in the repository. `manifest.json` carries a sha256 per file, which
`verify_features` checks -- the difference between "the pipeline ran" and "the pipeline ran
on the data we claim it ran on".
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

__all__ = ["repo_root", "features_dir", "feature_file", "load_manifest", "verify_features",
           "resolve_path", "default_config"]

_ENV_VAR = "TNK1_FEATURES_DIR"


def repo_root() -> Path:
    """The checkout root (.../tnk1/utils -> up 2)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(p, *, root: Optional[Path] = None) -> Path:
    """Anchor a possibly-relative path to the REPO ROOT, not the current directory.

    Why this exists
    ---------------
    Every path in `configs/default.yaml` is relative (`data/processed`, `runs`, ...), and
    `Ctx` used to pass them straight to `Path(...)`. A relative Path resolves against the
    *process* working directory, so `python scripts/run_stage3_transfer.py` launched from
    `scripts/` did not fail -- it silently resolved `data/processed` to
    `scripts/data/processed`, created it (`mkdir(parents=True)`), and wrote a parallel set
    of artifacts there. Stage 0 would then report the real store as ready while the stage
    that followed read and wrote an empty one.

    That is the same failure shape as every defect in PIPELINE_AUDIT.md: no exception, no
    log line, a wrong answer. Anchoring to the repo root makes the pipeline's location
    depend on the installed package, not on where the operator happened to be standing.

    Absolute paths and `$TNK1_*` overrides are returned untouched, so an out-of-tree data
    directory still works. `~` is expanded.
    """
    q = Path(str(p)).expanduser()
    if q.is_absolute():
        return q
    return ((root or repo_root()) / q).resolve()


def default_config() -> Path:
    """The shipped config, located relative to the package rather than the caller's CWD.

    `$TNK1_CONFIG` overrides it, so an experiment can point every entry point at an
    alternative config without editing argv in nine scripts.
    """
    env = os.environ.get("TNK1_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "configs" / "default.yaml"


def features_dir(required: bool = False) -> Path:
    """Resolve the feature-store directory. See module docstring for the search order."""
    env = os.environ.get(_ENV_VAR)
    if env:
        p = Path(env).expanduser()
        if required and not p.exists():
            raise FileNotFoundError(f"${_ENV_VAR} points at a missing directory: {p}")
        return p

    # Repo-anchored candidates first. The two CWD-relative ones are kept for ad-hoc runs
    # from a data directory, but they WARN: a relative candidate resolves against the
    # process working directory, so it can silently point at a different feature store
    # than the one the repo ships -- the same class of failure as the CWD-relative config
    # paths that `resolve_path` exists to prevent. Everything the pipeline does resolves
    # the repo-anchored path first, so a warning here means something unusual is going on.
    anchored = [
        repo_root() / "data" / "features",
        repo_root() / "ml_research" / "features",      # pre-reorganisation layout
    ]
    relative = [Path("data/features"), Path("features")]
    for c in anchored:
        if c.exists():
            return c
    for c in relative:
        if c.exists():
            import warnings
            warnings.warn(
                f"feature store resolved to the CWD-relative path {c.resolve()} because "
                f"none of {[str(a) for a in anchored]} exist. This depends on where the "
                f"process was started; set ${_ENV_VAR} to make it explicit.",
                RuntimeWarning, stacklevel=2,
            )
            return c
    candidates = anchored + relative
    if required:
        raise FileNotFoundError(
            f"feature store not found; looked in {[str(c) for c in candidates]}. "
            f"Set ${_ENV_VAR} to its location."
        )
    return candidates[0]


def feature_file(name: str, required: bool = False) -> Path:
    """Path to one file inside the feature store."""
    p = features_dir() / name
    if required and not p.exists():
        raise FileNotFoundError(
            f"missing feature file '{name}' in {features_dir()}. It is gitignored "
            f"(see ml_research/features/README.md); restore it or set ${_ENV_VAR}."
        )
    return p


def load_manifest(directory: Optional[Path] = None) -> Dict[str, dict]:
    """The {filename: {size_bytes, sha256}} manifest shipped with the feature store."""
    d = Path(directory) if directory else features_dir()
    p = d / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"no manifest.json in {d}")
    return json.loads(p.read_text())


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_features(
    names: Optional[Iterable[str]] = None,
    directory: Optional[Path] = None,
    strict: bool = False,
) -> Dict[str, str]:
    """Check feature files against manifest.json.

    Returns {filename: "ok" | "absent" | "size-mismatch: …" | "sha256-mismatch: …"}.
    With `strict=True`, raises unless every requested file is "ok". `names=None` checks
    every entry in the manifest.
    """
    d = Path(directory) if directory else features_dir()
    manifest = load_manifest(d)
    wanted: List[str] = list(names) if names is not None else list(manifest)

    out: Dict[str, str] = {}
    for name in wanted:
        if name not in manifest:
            out[name] = "not-in-manifest"
            continue
        p = d / name
        if not p.exists():
            out[name] = "absent"
            continue
        exp = manifest[name]
        size = p.stat().st_size
        if "size_bytes" in exp and size != exp["size_bytes"]:
            out[name] = f"size-mismatch: {size} != {exp['size_bytes']}"
            continue
        digest = _sha256(p)
        out[name] = "ok" if digest == exp.get("sha256") else (
            f"sha256-mismatch: {digest[:16]}… != {str(exp.get('sha256'))[:16]}…"
        )

    if strict:
        bad = {k: v for k, v in out.items() if v != "ok"}
        if bad:
            raise RuntimeError(f"feature store verification failed in {d}: {bad}")
    return out
