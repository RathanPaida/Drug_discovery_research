#!/bin/bash
# Single source of truth for cluster paths. Sourced by every hpc/*.sh and by
# run_amazon_ml.sh. Paths derive from where this file lives, so the folder works
# wherever it is uploaded (for example ~/Amazon_ML).
# Override any of them by exporting it first, e.g.
#   BER_WORK_DIR=/scratch/$USER/ber_work bash hpc/run_pipeline.sh full
#
# Layout (same relative layout as the laptop):
#   $BER_ROOT/code/business_entity_resolution   this package  (CODE_DIR)
#   $BER_ROOT/data/student_resource/dataset     competition data (train/, test/)
#   $BER_ROOT/model_weights/{bge-m3,bge-reranker-v2-m3}   staged by hpc/download_models.sh
#   $BER_ROOT/work                              intermediate artefacts
#   $BER_ROOT/output                            matching_results.tsv, candidate_pairs.tsv
#   $BER_ROOT/envs/ber                          python env (hpc/setup_env.sh)

_ber_hpc_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${BER_CODE_DIR:-$(dirname "$_ber_hpc_dir")}"
BER_ROOT="${BER_ROOT:-$(cd "$CODE_DIR/../.." && pwd)}"
ENV_DIR="${BER_ENV_DIR:-$BER_ROOT/envs/ber}"
# Optional: an existing env that already has torch + transformers. setup_env.sh
# then layers a venv on top of it instead of downloading torch.
BASE_ENV_DIR="${BER_BASE_ENV_DIR:-}"
export BER_ROOT
export BER_MODEL_DIR="${BER_MODEL_DIR:-$BER_ROOT/model_weights}"

# quiet `module load` that also works where Lmod/`module` is absent
_ber_mload() { type module >/dev/null 2>&1 && module load "$1" >/dev/null 2>&1; }

# Put $ENV_DIR's python first on PATH. Works for conda envs and venvs alike;
# loads the module the env was built from (recorded by setup_env.sh), if any.
enter_env() {
    local m=""
    [ -f "$ENV_DIR/.ber_module" ] && m="$(cat "$ENV_DIR/.ber_module")"
    if [ -n "$m" ] && ! _ber_mload "$m"; then
        echo "WARN: could not load module '$m' recorded for $ENV_DIR" >&2
    fi
    export VIRTUAL_ENV="$ENV_DIR"
    export PATH="$ENV_DIR/bin:$PATH"
    # module-provided PYTHONPATH and ~/.local site-packages can shadow the env's
    # packages (seen before as "undefined symbol: ncclCommResume") - isolate
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1
}

env_python_runs() {
    ( enter_env >/dev/null 2>&1; [ -x "$ENV_DIR/bin/python" ] && "$ENV_DIR/bin/python" -c "import sys" >/dev/null 2>&1 )
}

# The env has everything the pipeline imports (setup_env.sh then has nothing to do).
ber_env_ready() {
    ( enter_env >/dev/null 2>&1; [ -x "$ENV_DIR/bin/python" ] &&
      "$ENV_DIR/bin/python" -c "import torch, transformers, huggingface_hub, safetensors, polars, numpy, scipy, sklearn, yaml" \
          >/dev/null 2>&1 )
}

# Both models are staged (download_models.sh then has nothing to do).
ber_models_ready() {
    local m f
    for m in bge-m3 bge-reranker-v2-m3; do
        for f in config.json tokenizer.json model.safetensors; do
            [ -s "$BER_MODEL_DIR/$m/$f" ] || return 1
        done
    done
}

# Can this machine reach URL $1? (setup and model download need internet)
ber_online() {
    if command -v curl >/dev/null 2>&1; then
        curl -sfL -o /dev/null -m 20 "$1" >/dev/null 2>&1
    else
        python3 -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=20)" "$1" >/dev/null 2>&1
    fi
}

# Stop with the reason and the fix when a folder the pipeline writes to is not
# writable (project uploaded by / owned by another account, or read-only).
ber_need_writable() {
    local d p err fix rc=0
    for d in "$@"; do
        case "$d" in /*) ;; *) d="$PWD/$d" ;; esac
        if err="$(mkdir -p "$d" 2>&1)" && [ -w "$d" ]; then continue; fi
        p="$d"; while [ ! -d "$p" ]; do p="$(dirname "$p")"; done
        case "$p/" in "$BER_ROOT"/*) fix="$BER_ROOT" ;; *) fix="$p" ;; esac
        {
            echo "ERROR: cannot write to $d"
            [ -n "$err" ] && echo "  $err"
            echo "  you are '$(id -un)'; the folder in the way is:"
            echo "    $(ls -ld "$p" 2>&1)"
            echo "  If that folder is yours, make it writable and run again:  chmod -R u+w $fix"
            echo "  If it belongs to another account, copy the project into a folder you own and run from the copy."
        } >&2
        rc=1
    done
    return "$rc"
}

activate_env() {
    if ! env_python_runs; then
        echo "ERROR: no working python env at $ENV_DIR" >&2
        echo "Build it once:  cd $BER_ROOT && bash run_amazon_ml.sh setup   (or: bash $CODE_DIR/hpc/setup_env.sh)" >&2
        exit 1
    fi
    enter_env
    export PYTHONPATH="$CODE_DIR/src"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=true
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1   # logs arrive at once and in order
    cd "$CODE_DIR" || exit 1
    # for jobs submitted straight from here; preflight.sh / run_pipeline.sh insist on it
    mkdir -p logs 2>/dev/null || true
}
