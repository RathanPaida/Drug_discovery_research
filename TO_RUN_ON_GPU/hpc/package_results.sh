#!/usr/bin/env bash
# ==============================================================================
# Package pipeline results into a portable archive for download from HPC.
# Modeled after GAMA submission packaging.
#
# Usage:
#   bash hpc/package_results.sh [archive_name.tar.gz]
# ==============================================================================

set -euo pipefail

_cfg=""
for _d in "$(dirname "${BASH_SOURCE[0]}")" "${SLURM_SUBMIT_DIR:-}/hpc" "${TNK1_ROOT:-$PWD}/hpc"; do
    [ -f "$_d/_config.sh" ] && { _cfg="$_d/_config.sh"; break; }
done
[ -n "$_cfg" ] || { echo "ERROR: hpc/_config.sh not found" >&2; exit 1; }
source "$_cfg"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_NAME="${1:-dug_results_${TIMESTAMP}.tar.gz}"

cd "$TNK1_ROOT"
echo ">> Packaging results from $TNK1_ROOT into $OUT_NAME..."

tar -czf "$OUT_NAME" \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude=".pytest_cache" \
    runs/ \
    logs/ \
    data/processed/*.parquet \
    data/processed/*.csv \
    data/processed/*.json \
    data/processed/*.pkl \
    data/processed/ensemble/ \
    data/processed/rl_min/ 2>/dev/null || true

echo ">> Created archive: $TNK1_ROOT/$OUT_NAME"
echo ">> Size: $(du -h "$OUT_NAME" 2>/dev/null | awk '{print $1}')"
echo ">> Download to local machine:"
echo "   scp <user>@<cluster>:$TNK1_ROOT/$OUT_NAME ."
