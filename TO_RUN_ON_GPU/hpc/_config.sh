#!/bin/bash
# Single source of truth for cluster paths. Sourced by every hpc/*.sh.
# Paths derive from where this file lives, so the folder works wherever it is uploaded.
# Override any of them by exporting it first, e.g.
#   TNK1_CONFIG=$PWD/configs/hpc_fast.yaml bash hpc/run_pipeline.sh full
#
# Layout
#   $TNK1_ROOT                      this folder; ALSO the python package root
#   $TNK1_ROOT/tnk1  rl_min  scripts  configs  tests
#   $TNK1_ROOT/data/features        12-file feature store (sha256 in manifest.json)
#   $TNK1_ROOT/data/processed       frozen v3 table, kinase store, models
#   $TNK1_ROOT/envs/tnk1            python env (hpc/setup_env.sh)
#   $TNK1_ROOT/runs                 stage outputs + runs/pipeline/{state,last_run}.json
#   $TNK1_ROOT/logs                 one log per Slurm job
#
# DEVIATION FROM THE GAMA LAYOUT, AND WHY. GAMA keeps the package in
# $ROOT/code/<pkg> with data in $ROOT/data, because its code takes BER_ROOT from the
# environment. This pipeline instead derives every data path from the PACKAGE location --
# tnk1/utils/paths.py::repo_root() is `Path(__file__).parents[2]` -- so the package root
# and the data root have to be the same directory. Splitting them would make repo_root()
# resolve to code/<pkg> and the feature store would not be found. Everything else here
# follows the GAMA conventions: hpc/_config.sh, enter_env/activate_env, submit_{cpu,gpu}.sh
# with #SBATCH inside, run_pipeline.sh chaining with --dependency=afterok, logs/%x_%j.out.

_tnk1_hpc_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TNK1_ROOT="${TNK1_ROOT:-$(dirname "$_tnk1_hpc_dir")}"
CODE_DIR="$TNK1_ROOT"                       # same directory, see the note above
ENV_DIR="${TNK1_ENV_DIR:-$TNK1_ROOT/envs/tnk1}"
export TNK1_ROOT CODE_DIR ENV_DIR

# Feature store. tnk1.utils.paths.features_dir() honours this and warns if it ever has to
# fall back to a CWD-relative guess.
export TNK1_FEATURES_DIR="${TNK1_FEATURES_DIR:-$TNK1_ROOT/data/features}"

# Which config every stage uses. Read by tnk1.utils.paths.default_config(), so no stage
# script needs a --config flag. hpc.yaml = full data; hpc_fast.yaml = reduced.
export TNK1_CONFIG="${TNK1_CONFIG:-$TNK1_ROOT/configs/hpc.yaml}"

# Slurm job-name prefix. Every job appears in squeue as <prefix>_<stage>, and each log is
# logs/<prefix>_<stage>_<jobid>.out because the #SBATCH --output pattern is %x_%j.out.
# run_pipeline.sh passes --job-name on the sbatch command line, which overrides the
# #SBATCH --job-name fallback inside submit_{cpu,gpu}.sh.
export TNK1_JOB_PREFIX="${TNK1_JOB_PREFIX:-dug}"

# Molecular-graph cache, in ENTRIES. Must exceed the unique-compound count of the table
# or the FIFO thrashes and every epoch re-featurises. The v3 table has 331,620 unique
# compounds; ~20 KB per cached graph, so 420k entries is ~8 GB. This node has 125 GB.
export TNK1_GRAPH_CACHE="${TNK1_GRAPH_CACHE:-420000}"

# Descriptor cache: unset preloads all 787,283 rows (~1 GB resident, ~1.26 GB transient).
# Set an integer to cap it, or 0 to compute on demand (much slower).
# export TNK1_DESC_CACHE=400000

_tnk1_mload() { type module >/dev/null 2>&1 && module load "$1" >/dev/null 2>&1; }

# Put $ENV_DIR's python first on PATH. Works for venvs and conda envs alike.
enter_env() {
    local m=""
    [ -f "$ENV_DIR/.tnk1_module" ] && m="$(cat "$ENV_DIR/.tnk1_module")"
    if [ -n "$m" ] && ! _tnk1_mload "$m"; then
        echo "WARN: could not load module '$m' recorded for $ENV_DIR" >&2
    fi
    export VIRTUAL_ENV="$ENV_DIR"
    export PATH="$ENV_DIR/bin:$PATH"
    # A module-provided PYTHONPATH or ~/.local site-packages can shadow the env's own
    # packages; isolate so `import torch` cannot pick up a different build.
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1
}

env_python_runs() {
    ( enter_env >/dev/null 2>&1; [ -x "$ENV_DIR/bin/python" ] &&
      "$ENV_DIR/bin/python" -c "import sys" >/dev/null 2>&1 )
}

# The env has everything the pipeline imports, so setup_env.sh has nothing to do.
tnk1_env_ready() {
    ( enter_env >/dev/null 2>&1; [ -x "$ENV_DIR/bin/python" ] &&
      "$ENV_DIR/bin/python" -c "import numpy, pandas, scipy, sklearn, yaml, pyarrow, torch, rdkit, xgboost" \
          >/dev/null 2>&1 )
}

# The feature store is present and every sha256 matches its manifest entry.
tnk1_data_ready() {
    ( enter_env >/dev/null 2>&1
      cd "$TNK1_ROOT" || return 1
      "$ENV_DIR/bin/python" -c "
import sys
from tnk1.utils.paths import verify_features
bad = {k: v for k, v in verify_features().items() if v != 'ok'}
sys.exit(1 if bad else 0)" >/dev/null 2>&1 )
}

# Can this machine reach URL $1? (setup needs pypi)
tnk1_online() {
    if command -v curl >/dev/null 2>&1; then
        curl -sfL -o /dev/null -m 20 "$1" >/dev/null 2>&1
    else
        python3 -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=20)" "$1" >/dev/null 2>&1
    fi
}

# Stop with the reason and the fix when a folder the pipeline writes to is not writable.
tnk1_need_writable() {
    local d p err fix rc=0
    for d in "$@"; do
        case "$d" in /*) ;; *) d="$TNK1_ROOT/$d" ;; esac
        if err="$(mkdir -p "$d" 2>&1)" && [ -w "$d" ]; then continue; fi
        p="$d"; while [ ! -d "$p" ]; do p="$(dirname "$p")"; done
        case "$p/" in "$TNK1_ROOT"/*) fix="$TNK1_ROOT" ;; *) fix="$p" ;; esac
        {
            echo "ERROR: cannot write to $d"
            [ -n "$err" ] && echo "  $err"
            echo "  you are '$(id -un)'; the folder in the way is:"
            echo "    $(ls -ld "$p" 2>&1)"
            echo "  If that folder is yours:  chmod -R u+w $fix"
            echo "  If it belongs to another account, copy the project somewhere you own."
        } >&2
        rc=1
    done
    return "$rc"
}

activate_env() {
    if ! env_python_runs; then
        echo "ERROR: no working python env at $ENV_DIR" >&2
        echo "Build it once:  bash $TNK1_ROOT/hpc/setup_env.sh" >&2
        exit 1
    fi
    enter_env
    # Nothing in this pipeline downloads anything at run time: kinase_store.npz and
    # ssl_encoder.pt ship pre-built, and admet_7endpoints.parquet substitutes for the TDC
    # download. Setting these makes a stray hub lookup fail fast instead of hanging.
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
    export NUMEXPR_MAX_THREADS="${OMP_NUM_THREADS}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1   # logs arrive at once and in order
    cd "$CODE_DIR" || exit 1
    mkdir -p logs runs/pipeline 2>/dev/null || true
}
