"""The HPC submission scripts must hold to two caps that were agreed for this project:

  * at most 36 GPU shards held at any one instant, and
  * at most one of our jobs running at any one instant.

These live in shell, so nothing type-checks them and a stray `--gres=shard:48` in a
copy-pasted header would not fail until it was already running on a shared node. The
caps are cheap to assert textually, so they are asserted.

History worth keeping: the lane originally covered GPU jobs only. The CPU stages
(preflight, baselines, zoo, stack, critic, taskselect) request no GRES, so they never
queued behind a GPU job and several of our jobs were resident at once. `run_sweeps.sh`
separately defaulted to 2 concurrent GPU array tasks and 4 CPU ones.
"""
from __future__ import annotations

import pathlib
import re

import pytest

HPC = pathlib.Path(__file__).resolve().parents[1] / "hpc"
SHARD_CAP = 36

GPU_SUBMITTERS = ["submit_gpu.sh", "submit_campaign.sh"]


@pytest.mark.parametrize("name", GPU_SUBMITTERS)
def test_sbatch_shard_default_is_within_cap(name):
    """The #SBATCH fallback matters: a bare `sbatch hpc/submit_gpu.sh ens` does not go
    through run_pipeline.sh and gets exactly this value."""
    text = (HPC / name).read_text()
    found = re.findall(r"^#SBATCH\s+--gres=shard:(\d+)", text, re.M)
    assert found, f"{name} declares no #SBATCH --gres=shard:N"
    for n in found:
        assert int(n) <= SHARD_CAP, f"{name} requests shard:{n}, over the {SHARD_CAP} cap"


def test_no_script_requests_more_than_the_cap():
    """Catches every shard REQUEST, including ones in comments showing a user how to
    ask for more -- those get copy-pasted.

    Matches only `--gres=shard:N`, so prose about the node's capacity is left alone:
    preflight.sh legitimately prints `gpu:1,shard:94` as the total the node offers.
    """
    for path in sorted(HPC.glob("*.sh")):
        for n in re.findall(r'--gres="?shard:(\d+)', path.read_text()):
            assert int(n) <= SHARD_CAP, (
                f"{path.name} requests --gres=shard:{n}, over the {SHARD_CAP} cap")


def test_run_pipeline_refuses_a_shard_override_above_the_cap():
    text = (HPC / "run_pipeline.sh").read_text()
    assert 'SHARDS="${TNK1_SHARDS:-36}"' in text, "default shard budget is not 36"
    assert '[ "$SHARDS" -gt 36 ]' in text, (
        "run_pipeline.sh does not refuse TNK1_SHARDS above the cap")


def test_lane_serialises_every_job_not_only_gpu_ones():
    """The lane dependency must be applied unconditionally. If it is ever put back
    behind `[ "$kind" = gpu ]`, the CPU stages run alongside a GPU stage again."""
    text = (HPC / "run_pipeline.sh").read_text()
    assert 'LANES="${TNK1_LANES:-${TNK1_GPU_LANES:-1}}"' in text
    assert 'if [ "$LANES" -ge 1 ]; then' in text, (
        "the lane is no longer applied to every job")
    assert '[ "$kind" = gpu ] && [ "$GPU_LANES"' not in text, (
        "the lane is gated on job kind again -- CPU jobs would escape it")
    # every submitted id enters the FIFO, not just the GPU ones
    assert re.search(r'^\s*echo "\$jid" >> "\$LANE_FIFO"', text, re.M), (
        "not every job id is recorded in the lane FIFO")


def test_lane_uses_afterany_so_a_failure_does_not_cascade():
    text = (HPC / "run_pipeline.sh").read_text()
    assert "afterany:$lane_prev" in text, (
        "the ordering dependency must be afterany; afterok would cancel every later "
        "stage when one unrelated stage fails")


def test_sweeps_default_to_one_concurrent_task():
    text = (HPC / "run_sweeps.sh").read_text()
    assert 'GPU_CONC="${TNK1_ARRAY_GPU_CONCURRENCY:-1}"' in text
    assert 'CPU_CONC="${TNK1_ARRAY_CPU_CONCURRENCY:-1}"' in text


def test_dry_run_does_not_build_an_environment():
    """`--dry-run` is documented as queueing nothing. It also has to touch nothing:
    the resume path used to move envs/tnk1 aside and start a 6 GB conda build before
    printing a single sbatch line."""
    text = (HPC / "run_pipeline.sh").read_text()
    assert '[ "$DRY" != 1 ] && ! tnk1_env_ready' in text, (
        "env setup is not guarded by DRY")
    assert '[ "$DRY" = 1 ] || activate_env' in text, (
        "activate_env is not guarded by DRY")
    assert '[ "$DRY" != 1 ] && ! tnk1_data_ready' in text, (
        "the feature-store check is not guarded by DRY")


def test_campaign_header_does_not_document_the_wrong_script():
    """submit_campaign.sh began as a verbatim copy of submit_gpu.sh and told the reader
    to run `sbatch hpc/submit_gpu.sh ens`, which does something else entirely."""
    text = (HPC / "submit_campaign.sh").read_text()
    head = text.split("set -uo pipefail")[0]
    # Only usage LINES matter -- `#   sbatch hpc/submit_gpu.sh ens`. Prose that mentions
    # the other script to explain the difference is the point, not the defect.
    usage = [ln for ln in head.splitlines()
             if re.match(r"^#\s+(sbatch|TNK1_\w+=\S+ sbatch)\s", ln)]
    assert usage, "the header shows no usage example at all"
    for ln in usage:
        assert "submit_gpu.sh" not in ln, f"header tells you to run submit_gpu.sh: {ln}"
    assert "submit_campaign.sh" in head
    assert "--job-name=dug_campaign" in text, (
        "the campaign job shares submit_gpu.sh's job name, so logs collide")
