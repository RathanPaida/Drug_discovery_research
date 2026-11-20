#!/bin/bash
# Everything that can be checked before queueing hours of compute. Run on the login node.
#
#   bash hpc/preflight.sh
#
# Checks, in the order they would otherwise fail:
#   1. the env imports what the pipeline imports
#   2. all shipped feature files match their sha256 in manifest.json
#   3. the frozen v3 table verifies against its .sha256 sidecar and is not contaminated
#   4. the config is self-consistent (reward weights, off-target panel, filters)
#   5. the hermetic test suite (317 tests, ~30 s)
#   6. Slurm can see the node, the GPU and the shard GRES

set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_here/_config.sh"

rc=0
step() { echo; echo "=== $* ==="; }

# Preflight must check the SAME dataset version the run will use, or it green-lights v3
# while the jobs go on to use v4 (or the reverse). See the note in submit_cpu.sh.
VERSION="${TNK1_VERSION:-v3}"
echo "dataset version under test: $VERSION   (override with TNK1_VERSION=v4)"

step "1-4. environment, feature store, frozen table, config"
if ! env_python_runs; then
    echo "FAIL: no working env at $ENV_DIR -- run:  bash hpc/setup_env.sh" >&2
    exit 1
fi
activate_env
python scripts/run_stage0_check.py --version "$VERSION" --strict || rc=1

step "5. hermetic test suite"
python -m pytest -q || rc=1

step "6. Slurm view of this node"
sinfo -o "%20N %10c %10m %30G %10T" 2>&1 | head -5 || true
echo "shard GRES expected: gpu:1,shard:94   (1 shard ~ 1 GB of the H100's 94 GB)"
squeue -u "$USER" 2>&1 | head -5 || true

step "7. what the pipeline would do next"
python scripts/run_pipeline.py --version "$VERSION" --plan || rc=1

echo
if [ "$rc" = 0 ]; then
    echo "PREFLIGHT OK. Queue the run:   bash hpc/run_pipeline.sh full"
else
    echo "PREFLIGHT FAILED (rc=$rc) -- fix the above before queueing compute." >&2
fi
exit "$rc"
