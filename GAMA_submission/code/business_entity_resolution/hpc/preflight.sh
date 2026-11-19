#!/bin/bash
# Checks before burning a GPU allocation. Safe on the login node.
#   cd ~/Amazon_ML/code/business_entity_resolution && bash hpc/preflight.sh
# (bash run_amazon_ml.sh setup runs it after building the env and staging the models)

set -euo pipefail
_cfg=""
for _d in "$(dirname "${BASH_SOURCE[0]}")" "${SLURM_SUBMIT_DIR:-}/hpc" \
          "${BER_CODE_DIR:-$HOME/Amazon_ML/code/business_entity_resolution}/hpc"; do
    [ -f "$_d/_config.sh" ] && { _cfg="$_d/_config.sh"; break; }
done
[ -n "$_cfg" ] || { echo "ERROR: hpc/_config.sh not found (set BER_CODE_DIR)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$_cfg"
activate_env
ber_need_writable "$CODE_DIR/logs" || exit 1

echo "== ber preflight (paths / data / packages / models / GPU) =="
python -m ber preflight "$@"

echo
echo "== unit tests (CPU, no weights) =="
if python -c "import pytest" 2>/dev/null; then
    python -m pytest tests -q -p no:cacheprovider
else
    python tests/test_core.py
fi

echo
echo "Ready. One job on one GPU:  cd $BER_ROOT && sbatch run_amazon_ml.sh smoke"
echo "      Chain of jobs:        bash hpc/run_pipeline.sh smoke"
