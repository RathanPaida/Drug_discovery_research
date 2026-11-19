#!/bin/bash
# CPU pipeline step (prepare / sparse / features / prune / stack / validate):
#   cd ~/Amazon_ML/code/business_entity_resolution
#   sbatch hpc/submit_cpu.sh prepare
#   sbatch hpc/submit_cpu.sh features --split test
#   sbatch hpc/submit_cpu.sh prune gbdt.num_leaves=127        # trailing key=value = config overrides
# Sized for our development node (one node: 128 CPUs, 125 GB RAM, queue LocalQ).
# No --partition and no GPU: the default queue. A cluster whose only queue is a
# GPU queue needs e.g.:  sbatch -p gpu --gres=gpu:1 hpc/submit_cpu.sh ...

#SBATCH --job-name=ber_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
# one log per job: output and errors together, in order
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
_cfg=""
for _d in "${SLURM_SUBMIT_DIR:-.}/hpc" "${BER_CODE_DIR:-$HOME/Amazon_ML/code/business_entity_resolution}/hpc"; do
    [ -f "$_d/_config.sh" ] && { _cfg="$_d/_config.sh"; break; }
done
[ -n "$_cfg" ] || { echo "ERROR: hpc/_config.sh not found - sbatch from the code dir or set BER_CODE_DIR" >&2; exit 1; }
# shellcheck disable=SC1090
source "$_cfg"
activate_env

STEP="${1:?usage: sbatch hpc/submit_cpu.sh <step> [--split S] [key=value ...]}"; shift
echo "[ber] node=$(hostname) step=$STEP args=$* cpus=${SLURM_CPUS_PER_TASK:-?} work=${BER_WORK_DIR:-$BER_ROOT/work}"
python -m ber "$STEP" "$@"
