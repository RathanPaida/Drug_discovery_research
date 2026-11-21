#!/bin/bash
# One GPU pipeline stage.
#   cd ~/TO_RUN_ON_GPU
#   sbatch hpc/submit_gpu.sh ens
#   sbatch hpc/submit_gpu.sh oof
#   TNK1_CONFIG=$PWD/configs/hpc_fast.yaml sbatch hpc/submit_gpu.sh ens
#
# This node's GPU is split into 94 shards of ~1 GB. shard:16 is what was asked for and is
# ample: the D-MPNN trunk is 939k parameters and the measured peak on an 8 GB card was
# 7.5 GB with the full 1.1M-row table. Up to 5 such jobs can share the H100, which is why
# run_pipeline.sh can fan out the independent stages.
#   whole GPU instead:  sbatch --gres=gpu:1 hpc/submit_gpu.sh <stage>
#   more memory:        sbatch --gres=shard:32 hpc/submit_gpu.sh <stage>
# On CUDA OOM, lower kinome_train.batch_size in the config before raising shards.

# Fallback name for a bare `sbatch hpc/submit_gpu.sh <stage>`; run_pipeline.sh
# overrides it with --job-name=$TNK1_JOB_PREFIX_<stage> on the command line.
#SBATCH --job-name=dug_gpu
#SBATCH --gres=shard:16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# 36G not 64G: the node has 125 GB total, so 64G let only ONE GPU job run at a time and
# the DAG serialised. 36G lets three coexist (108 of 125 GB). Measured need is well
# under that: the 420k-entry graph cache is ~8 GB, the descriptor preload ~1 GB, the
# 1.1M-row frame ~1 GB, plus torch. Raise it if a job is OOM-killed (check `dmesg` or
# a 'Killed' line in the log).
#SBATCH --mem=36G
#SBATCH --time=7-00:00:00
# one log per job: stdout and stderr together, in order
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
_cfg=""
for _d in "${SLURM_SUBMIT_DIR:-.}/hpc" "${TNK1_ROOT:-$HOME/TO_RUN_ON_GPU}/hpc" "$(dirname "${BASH_SOURCE[0]}")"; do
    [ -f "$_d/_config.sh" ] && { _cfg="$_d/_config.sh"; break; }
done
[ -n "$_cfg" ] || { echo "ERROR: hpc/_config.sh not found - sbatch from the bundle root, or export TNK1_ROOT" >&2; exit 1; }
# shellcheck disable=SC1090
source "$_cfg"
activate_env

STAGE="${1:?usage: sbatch hpc/submit_gpu.sh <stage>   (ens|oof|calibrate|compare|admet|selectivity|prior|rl|lambda|ablate|cycle)}"; shift || true
# NB a non-default RL objective goes through run_pipeline.py's --rl-algo, which this
# script forwards along with any other extra flags:
#   sbatch hpc/submit_gpu.sh rl --rl-algo grpo
# or, for the whole 8-algorithm sweep at once, use the array. The range must match
# RL_ALGOS in submit_array.sh, which has 8 entries: `0-5` was left here when the set grew
# and would have run only the first six, silently skipping grpo and gflownet (an index
# ABOVE the range errors; one below just never launches).
#   sbatch --array=0-7%2 hpc/submit_array.sh rlalgo
# or let run_sweeps.sh compute the range for you:  bash hpc/run_sweeps.sh rl

# See the note in submit_cpu.sh: this was hardcoded `v3`, so a run against a newly
# frozen kinome_long_v4.parquet would have silently used the old table instead.
VERSION="${TNK1_VERSION:-v3}"

echo "[dug] node=$(hostname) stage=$STAGE version=$VERSION config=$TNK1_CONFIG"
echo "[dug] shards=${SLURM_JOB_GRES:-?} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}MB"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible in this job'; \
print('[tnk1] torch', torch.__version__, '| gpu', torch.cuda.get_device_name(0))"

python scripts/run_pipeline.py --version "$VERSION" --device cuda --only "$STAGE" "$@"
rc=$?

echo
echo "---- stage '$STAGE' exited $rc ----"
python - <<'EOF'
import json, pathlib
p = pathlib.Path("runs/pipeline/last_run.json")
if p.exists():
    for k, v in json.loads(p.read_text()).get("stages", {}).items():
        extra = f"  {v['seconds']/60:.1f} min" if "seconds" in v else ""
        print(f"  {v['status']:<9} {k}{extra}")
        for m in v.get("missing_outputs") or []:
            print(f"      MISSING OUTPUT: {m}")
EOF
exit $rc
