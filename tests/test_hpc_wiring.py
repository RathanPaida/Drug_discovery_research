"""The shell scripts and the Python they launch must agree.

WHY. The HPC scripts hardcode counts and ranges that mirror Python-side registries, and
nothing checked them against each other. Two drifted, and neither would have raised:

  1. `submit_gpu.sh` documented `sbatch --array=0-5%2 hpc/submit_array.sh rlalgo` after
     the objective set grew from 6 to 8. An array index ABOVE the range errors cleanly;
     one below simply never launches, so following that line ran six objectives and
     silently omitted `grpo` and `gflownet`. The results table would show six rows and
     look complete.

  2. `submit_cpu.sh`, `submit_gpu.sh`, `preflight.sh` and `run_tnk1.sh` hardcoded
     `--version v3` while `submit_array.sh` and `run_sweeps.sh` read `$TNK1_VERSION`. With
     a newer frozen table on disk those scripts ran every stage against v3 -- which still
     exists and whose sha256 still verifies, so the jobs completed on the old data.

Both are the same class of defect: a constant duplicated across a language boundary. These
tests put the boundary under test instead of under a comment.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from rl_min.algos import ALGOS

ROOT = Path(__file__).resolve().parents[1]

# The shell scripts live at TO_RUN_ON_GPU/hpc/ in the repository and at hpc/ when running
# from inside the bundle itself, which is a full mirror. Resolving both means these checks
# are live in the place that actually submits the jobs, not only in the repo.
HPC = next((p for p in (ROOT / "TO_RUN_ON_GPU" / "hpc", ROOT / "hpc") if p.is_dir()),
           ROOT / "TO_RUN_ON_GPU" / "hpc")

pytestmark = pytest.mark.skipif(not HPC.is_dir(), reason="hpc/ scripts not present")


def _text(name: str) -> str:
    p = HPC / name
    if not p.is_file():
        pytest.skip(f"{p} absent")
    return p.read_text()


def _bash_array(text: str, name: str) -> list[str]:
    m = re.search(rf"^{name}=\(([^)]*)\)", text, re.M)
    assert m, f"{name}=(...) not found"
    return m.group(1).split()


# ------------------------------------------------- the RL objective set, shell vs Python

def test_submit_array_lists_exactly_the_registered_algorithms():
    shell = _bash_array(_text("submit_array.sh"), "RL_ALGOS")
    assert sorted(shell) == sorted(ALGOS), (
        "RL_ALGOS in submit_array.sh has drifted from rl_min.algos.ALGOS; the array "
        "sweep would run the wrong set")


def test_run_sweeps_queues_one_task_per_algorithm():
    """REGRESSION. The range must cover every algorithm, not the count it had once."""
    txt = _text("run_sweeps.sh")
    m = re.search(r'submit_array\s+gpu\s+rlalgo\s+"\$\w+"\s+"0-(\d+)%', txt)
    assert m, "could not find the rlalgo submit_array call in run_sweeps.sh"
    assert int(m.group(1)) == len(ALGOS) - 1, (
        f"run_sweeps.sh queues 0-{m.group(1)} but there are {len(ALGOS)} algorithms; "
        f"an index below the range never launches, so the sweep would be silently short")


def test_documented_array_ranges_cover_every_algorithm():
    """Every `--array=0-N` for the rlalgo sweep in a comment must also be right.

    A wrong range in documentation is as damaging as one in code here, because the
    documented line is what a human copies.
    """
    want = len(ALGOS) - 1
    bad = []
    for name in ("submit_gpu.sh", "submit_array.sh", "run_sweeps.sh", "run_pipeline.sh"):
        p = HPC / name
        if not p.is_file():
            continue
        for ln, line in enumerate(p.read_text().splitlines(), 1):
            if "rlalgo" not in line:
                continue
            for m in re.finditer(r"--array=0-(\d+)", line):
                if int(m.group(1)) != want:
                    bad.append(f"{name}:{ln}: --array=0-{m.group(1)}, expected 0-{want}")
    assert not bad, "stale array range(s):\n  " + "\n  ".join(bad)


def test_representation_sweep_range_matches_its_own_array():
    """The same check for ZOO_REPS, which legitimately has a different length."""
    arr = _bash_array(_text("submit_array.sh"), "ZOO_REPS")
    m = re.search(r'submit_array\s+cpu\s+zoorep\s+""\s+"0-(\d+)%', _text("run_sweeps.sh"))
    assert m, "could not find the zoorep submit_array call"
    assert int(m.group(1)) == len(arr) - 1, (
        f"run_sweeps.sh queues 0-{m.group(1)} for {len(arr)} representations")


# ----------------------------------------------------------- the dataset version boundary

@pytest.mark.parametrize("name", ["submit_cpu.sh", "submit_gpu.sh", "preflight.sh",
                                  "run_sweeps.sh", "submit_array.sh"])
def test_no_script_hardcodes_a_dataset_version(name):
    """REGRESSION. A hardcoded `--version v3` runs the old frozen table without erroring,
    because v3 remains on disk and its sha256 still verifies."""
    txt = _text(name)
    hits = [f"line {i}: {l.strip()}" for i, l in enumerate(txt.splitlines(), 1)
            if re.search(r"--version\s+v\d", l)]
    assert not hits, (
        f"{name} hardcodes a dataset version; use \"${{TNK1_VERSION:-v3}}\":\n  "
        + "\n  ".join(hits))


@pytest.mark.parametrize("name", ["submit_cpu.sh", "submit_gpu.sh", "preflight.sh",
                                  "run_sweeps.sh", "submit_array.sh"])
def test_every_submitting_script_reads_tnk1_version(name):
    assert "TNK1_VERSION" in _text(name), (
        f"{name} never consults $TNK1_VERSION, so it cannot follow a re-freeze")


def test_run_tnk1_honours_tnk1_version():
    p = HPC.parent / "run_tnk1.sh"      # bundle root, wherever hpc/ was found
    if not p.is_file():
        pytest.skip("run_tnk1.sh absent")
    txt = p.read_text()
    assert "TNK1_VERSION" in txt
    assert not re.search(r"--version\s+v\d", txt), "run_tnk1.sh still pins a version"


# ------------------------------------------------------------------ stage-name agreement

def test_every_only_stage_in_the_shell_is_a_real_pipeline_stage():
    """`--only <stage>` names must exist in run_pipeline.py, or the job dies at startup
    after queueing, waiting and allocating a GPU."""
    plan = (ROOT / "scripts" / "run_pipeline.py").read_text()
    stages = set(re.findall(r'Stage\("([a-z0-9_]+)"', plan))
    assert stages, "no Stage(...) declarations found"
    bad = []
    for p in sorted(HPC.glob("*.sh")):
        for ln, line in enumerate(p.read_text().splitlines(), 1):
            for m in re.finditer(r"--only\s+([a-z0-9_]+)", line):
                if m.group(1) not in stages:
                    bad.append(f"{p.name}:{ln}: --only {m.group(1)}")
    assert not bad, ("shell passes a stage run_pipeline.py does not define:\n  "
                     + "\n  ".join(bad) + f"\n  known: {sorted(stages)}")
