#!/bin/bash
# The WHOLE campaign in one job: 26 models, 12 stackers, 10 fine-tunes, 8 RL algorithms,
# driven by scripts/run_campaign.py rather than one stage per job.
#
#   cd ~/TO_RUN_ON_GPU
#   sbatch hpc/submit_campaign.sh
#   TNK1_CONFIG=$PWD/configs/hpc_fast.yaml sbatch hpc/submit_campaign.sh
#
# This header used to be a verbatim copy of submit_gpu.sh's, down to telling you to run
# `sbatch hpc/submit_gpu.sh ens` -- which does something else entirely. It is one job,
# it takes no stage argument, and $STAGE is hardcoded to "campaign" below.
#
# This node's GPU is split into 94 shards of ~1 GB and the budget for this project is
# 36 SHARDS TOTAL at any instant. This job is the only thing running when it runs
# (run_pipeline.sh full submits it alone), so it requests the whole budget: shard:36.
# Override in one place with TNK1_SHARDS, which run_pipeline.sh forwards as --gres.
# On CUDA OOM, lower kinome_train.batch_size in the config before raising shards.

#SBATCH --job-name=dug_campaign
#SBATCH --gres=shard:36
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

STAGE="campaign"
# run_campaign.py sweeps every RL objective itself, so there is no --rl-algo to pass and
# no array to launch here; the per-stage and array entry points are submit_gpu.sh and
# submit_array.sh respectively. Extra flags given to this script are forwarded to
# run_campaign.py unchanged.

# See the note in submit_cpu.sh: this was hardcoded `v3`, so a run against a newly
# frozen kinome_long_v4.parquet would have silently used the old table instead.
VERSION="${TNK1_VERSION:-v3}"

echo "[dug] node=$(hostname) stage=$STAGE version=$VERSION config=$TNK1_CONFIG"
echo "[dug] shards=${SLURM_JOB_GRES:-?} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}MB"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible in this job'; \
print('[tnk1] torch', torch.__version__, '| gpu', torch.cuda.get_device_name(0))"

python scripts/run_campaign.py --version "$VERSION" --device cuda "$@"
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
