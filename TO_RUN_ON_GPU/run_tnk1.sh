#!/usr/bin/env bash
# ==============================================================================
# TNK1 Master HPC Pipeline Runner (Single-Job & Interactive Entrypoint)
# Modeled after GAMA submission architecture (run_amazon_ml.sh).
# ==============================================================================
#
# Single-job submission on 1 GPU:
#   sbatch run_tnk1.sh smoke           # fast verification run using configs/smoke.yaml (~10 min)
#   sbatch run_tnk1.sh full            # full production pipeline
#   sbatch run_tnk1.sh full ens        # resume at a specific stage (earlier outputs exist)
#   sbatch run_tnk1.sh baseline        # CPU baselines only
#
# Interactive / login node usage (no GPU allocation consumed):
#   bash run_tnk1.sh setup             # builds python environment (envs/tnk1) and verifies packages
#   bash run_tnk1.sh preflight         # runs sha256 checksums, config checks, and unit tests
#   bash run_tnk1.sh plan              # shows pipeline stage status (done, stale, pending)
#
# Watch progress:
#   squeue -u $USER
#   tail -f logs/tnk1_*.out
# ==============================================================================

#SBATCH --job-name=tnk1_master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out

set -eEo pipefail
exec 2>&1
export PYTHONUNBUFFERED=1

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$_here/hpc/_config.sh" ]; then
    # shellcheck disable=SC1091
    source "$_here/hpc/_config.sh"
else
    echo "ERROR: hpc/_config.sh not found in $_here" >&2
    exit 1
fi

MODE="${1:-full}"
FROM="${2:-}"

case "$MODE" in
    setup)
        echo ">> Setting up Python environment..."
        bash hpc/setup_env.sh
        exit 0
        ;;
    preflight)
        echo ">> Running preflight and tests..."
        bash hpc/preflight.sh
        exit 0
        ;;
    plan)
        activate_env
        python scripts/run_pipeline.py --version "${TNK1_VERSION:-v3}" --plan
        exit 0
        ;;
    smoke|full|baseline)
        ;;
    *)
        echo "usage: sbatch run_tnk1.sh smoke|full|baseline [from_step]   or   bash run_tnk1.sh setup|preflight|plan" >&2
        exit 2
        ;;
esac

# Require Slurm allocation for long-running modes unless TNK1_ALLOW_INTERACTIVE is set
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${TNK1_ALLOW_INTERACTIVE:-0}" != "1" ]; then
    echo "NOTE: '$MODE' is best submitted as a batch job to the GPU queue." >&2
    echo "      Submit with:  sbatch run_tnk1.sh $MODE $FROM" >&2
    echo "      (Or set TNK1_ALLOW_INTERACTIVE=1 to run directly in this shell session)" >&2
    exit 2
fi

need_writable "$TNK1_LOGS_DIR" "$TNK1_RUNS_DIR" "$TNK1_DATA_DIR/processed" || exit 1

# Setup environment if not ready
if ! tnk1_env_ready; then
    echo ">> Python environment missing or incomplete - running hpc/setup_env.sh..."
    bash hpc/setup_env.sh
fi

activate_env

echo "=============================================================="
echo " job        : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-no-id})"
echo " node       : $(hostname)"
echo " root       : $TNK1_ROOT"
echo " mode       : $MODE"
echo " python     : $(command -v python)  ($(python --version 2>&1))"
echo " cpus       : ${SLURM_CPUS_PER_TASK:-?}  mem: ${SLURM_MEM_PER_NODE:-?} MB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo " gpu        : none visible"
echo " start time : $(date)"
echo "=============================================================="

trap 'echo "=================================================="; echo "!!! FAILED at $(date) - see the log above"; echo "Resume later with: sbatch run_tnk1.sh $MODE <from_stage>"' ERR

EXTRA_ARGS=()
if [ "$MODE" = "smoke" ]; then
    export TNK1_CONFIG="$TNK1_ROOT/configs/smoke.yaml"
    EXTRA_ARGS+=(--smoke)
elif [ "$MODE" = "baseline" ]; then
    EXTRA_ARGS+=(--only preflight baselines)
fi

if [ -n "$FROM" ]; then
    EXTRA_ARGS+=(--from "$FROM")
fi

# Dataset version. Was hardcoded `v3`: after Stage 1 freezes a larger kinome_long_v4,
# this script would still have run every stage against the old table, without erroring,
# because v3 remains on disk and its sha256 still verifies. Export TNK1_VERSION=v4.
VERSION="${TNK1_VERSION:-v3}"
echo " Dataset  : $VERSION"
python scripts/run_pipeline.py --version "$VERSION" --device cuda "${EXTRA_ARGS[@]}"

echo "=============================================================="
echo " Pipeline execution completed successfully!"
echo " End time : $(date)"
echo " Results  : $TNK1_RUNS_DIR"
echo "=============================================================="
