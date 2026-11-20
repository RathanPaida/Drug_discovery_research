#!/bin/bash
# One CPU-only pipeline stage (no GPU requested, so it never queues behind GPU work).
#   sbatch hpc/submit_cpu.sh preflight
#   sbatch hpc/submit_cpu.sh baselines
#   sbatch hpc/submit_cpu.sh critic
#   sbatch hpc/submit_cpu.sh report
#
# baselines is the only substantial one: kNN / RF / XGB / TanimotoGP over 5x5 nested
# scaffold CV, measured at 15.6 min on a laptop. 16 cores is plenty (sklearn/xgboost
# parallelise; OMP_NUM_THREADS is set from SLURM_CPUS_PER_TASK by activate_env).

# Fallback name for a bare `sbatch hpc/submit_cpu.sh <stage>`; run_pipeline.sh
# overrides it with --job-name=$TNK1_JOB_PREFIX_<stage> on the command line.
#SBATCH --job-name=dug_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# 24G: baselines holds the 380-row TNK1 frame plus 2248-d features -- tiny. critic is
# the same plus the oof CSV.
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
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

STAGE="${1:?usage: sbatch hpc/submit_cpu.sh <stage>   (preflight|baselines|zoo|stack|critic|report)}"; shift || true

# The dataset version was hardcoded `v3` here while submit_array.sh and run_sweeps.sh
# already read $TNK1_VERSION. That split mattered the moment Stage 1 produced a second
# frozen table: harvesting more measurements writes kinome_long_v4.parquet, but every
# stage submitted through this script would still have asked for v3 -- and v3 still
# exists, so verify_locked passes, nothing errors, and the job trains on the OLD
# 172-molecule table. A silently stale run is the one failure mode worse than a crash,
# because the results look like results.
VERSION="${TNK1_VERSION:-v3}"

echo "[dug] node=$(hostname) stage=$STAGE version=$VERSION config=$TNK1_CONFIG cpus=${SLURM_CPUS_PER_TASK:-?}"
python scripts/run_pipeline.py --version "$VERSION" --device cpu --only "$STAGE" "$@"
rc=$?
echo
echo "---- stage '$STAGE' exited $rc ----"
exit $rc
