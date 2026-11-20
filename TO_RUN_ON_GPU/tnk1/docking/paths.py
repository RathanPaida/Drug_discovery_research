"""Where the Vina binary and a cycle's docking artifacts live.

WHY THIS IS A MODULE AND NOT TWO FUNCTIONS IN THE SCRIPT. Both helpers used to be
module-level constants in `scripts/dock_40_molecules_tnk1.py`, which imports rdkit and
meeko at the top. That made them untestable on any machine without a chemistry stack --
i.e. exactly the machines where you would want to check that the Vina path resolves
before queueing a job. Both defects below were of the kind a single unit test catches:

  1. `VINA_EXE = ROOT / "tools" / "vina.exe"` -- a Windows binary, in a directory the
     repository does not contain and never did (`tools/` is gitignored). On Linux, which
     is every HPC this project targets, the first docking job died inside `subprocess`
     with a bare FileNotFoundError, after the receptor prep had already run.

  2. The per-cycle paths were constants pinned to `cycle_01` while `main` parsed a
     `--cycle` flag and never used it. `--cycle 3` was accepted and silently ignored: it
     read Cycle-01's contract, re-docked Cycle-01's molecules and overwrote Cycle-01's
     published poses. An inert flag is worse than a rejected one, because the output
     looks like a Cycle-03 result.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# Vina binary names worth trying, in order. The official Linux release ships under its
# full versioned name; a conda or apt install puts a bare `vina` on PATH; `.exe` is the
# Windows build the Cycle-01 campaign was originally run with.
VINA_NAMES = ("vina", "vina_1.2.7_linux_x86_64", "vina_1.2.5_linux_x86_64", "vina.exe")

_VINA_HELP = (
    "AutoDock Vina executable not found.\n"
    "  Docking is a separate step from the pipeline and needs the Vina binary, which is\n"
    "  NOT in this repository (it is platform-specific, and `tools/` is gitignored).\n\n"
    "  Install one of:\n"
    "    conda install -c conda-forge vina\n"
    "    # or download the release for your platform from\n"
    "    #   https://github.com/ccsb-scripps/AutoDock-Vina/releases\n"
    "    # then put it on PATH, pass --vina /path/to/vina, or set TNK1_VINA\n"
)


def repo_root() -> Path:
    """The repository root, three levels up from this file (tnk1/docking/paths.py)."""
    return Path(__file__).resolve().parents[2]


def pipeline_root(root: Path | None = None) -> Path:
    """The directory that holds `runs/`.

    Older checkouts nested the pipeline at `tnk1-pipeline/tnk1-pipeline/`; the scripts
    carried that fallback, so it is preserved rather than silently dropped.
    """
    r = Path(root) if root is not None else repo_root()
    return r if (r / "runs").exists() else r / "tnk1-pipeline" / "tnk1-pipeline"


def cycle_paths(cycle: int, root: Path | None = None) -> dict[str, Path]:
    """Every per-cycle docking path, derived from the cycle number.

    `contract` is the novelty-audited table; `contract_fallback` is the plain hand-off
    table that earlier cycles wrote. A caller should prefer the first and fall back.
    """
    if int(cycle) < 1:
        raise ValueError(f"cycle must be >= 1, got {cycle!r}")
    runs = pipeline_root(root) / "runs" / f"cycle_{int(cycle):02d}"
    return {
        "runs": runs,
        "contract": runs / "contract_novel_team_b.csv",
        "contract_fallback": runs / "contract_to_team_b.csv",
        "poses": runs / "docking_poses",
        "results": runs / "docking_results_40_compounds.csv",
        "summary": runs / "docking_summary.json",
    }


def resolve_vina(explicit: str | os.PathLike | None = None,
                 root: Path | None = None,
                 env: dict | None = None) -> Path:
    """Locate the AutoDock Vina executable, or raise SystemExit with a fix.

    Resolution order: `explicit` (the --vina flag), then $TNK1_VINA, then PATH, then
    `tools/`. The message names every path tried, because "Vina not found" with no list
    is indistinguishable from "you installed it somewhere I did not look".
    """
    environ = os.environ if env is None else env
    r = Path(root) if root is not None else repo_root()

    tried: list[str] = []
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit).expanduser())
    from_env = str(environ.get("TNK1_VINA", "") or "").strip()
    if from_env:
        cands.append(Path(from_env).expanduser())
    for name in VINA_NAMES:
        found = shutil.which(name)
        if found:
            cands.append(Path(found))
    for name in VINA_NAMES:
        cands.append(r / "tools" / name)

    for c in cands:
        tried.append(str(c))
        if c.is_file():
            return c

    raise SystemExit(_VINA_HELP + "\n  Looked in:\n" + "\n".join(f"    {t}" for t in tried))
