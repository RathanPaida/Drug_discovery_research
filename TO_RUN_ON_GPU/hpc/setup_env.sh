#!/bin/bash
# One-time environment setup: builds $TNK1_ROOT/envs/tnk1 (~6 GB). Needs internet.
#
#   bash hpc/setup_env.sh
#
# Re-running is safe: it installs only what is missing.
#
# This node (172.17.120.214) has no `module` command and no conda on PATH -- only
# /usr/bin/python3.12 -- so the default path is a venv from the system python plus pip
# wheels from pypi, which this node can reach. Driver 580.65.06 / CUDA 13.0 on an
# H100 NVL, glibc 2.39, gcc 13.3: every wheel we need has a cp312 manylinux build.
#
# NOT requirements-lock.txt. That file pins the exact versions the v3 dataset was built
# with on Python 3.10.20, and several pins have no cp312 wheel. The floating ranges in
# requirements-core.txt / requirements-chem.txt are what the code actually needs; the
# lock file is kept in the bundle as the record of the reference environment.
#
# Overrides:
#   TNK1_PYTHON=/path/to/python3.11      use that interpreter instead
#   TNK1_TORCH_INDEX=https://download.pytorch.org/whl/cu128
#   TNK1_TORCH_SPEC="torch==2.8.0"
#   TNK1_BASE_ENV_DIR=/env/with/torch    layer a venv on top of it (skips the torch download)

set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_here/_config.sh"

if ! tnk1_online https://pypi.org/simple/pip/; then
    echo "ERROR: $(hostname) cannot reach pypi.org, and the setup needs internet." >&2
    [ -n "${SLURM_JOB_ID:-}" ] && echo "  Run it on the login node instead:  bash $TNK1_ROOT/hpc/setup_env.sh" >&2
    echo "  Behind a proxy?  export https_proxy=http://<proxy>:<port>  and retry." >&2
    exit 1
fi

tnk1_need_writable "$(dirname "$ENV_DIR")" "$TNK1_ROOT/logs" "$TNK1_ROOT/runs" || exit 1
echo ">> free disk where the env goes: $(df -hP "$(dirname "$ENV_DIR")" | awk 'NR==2 {print $4}')  (needs ~6 GB)"
if command -v quota >/dev/null 2>&1; then
    echo ">> quota:"; quota -s 2>/dev/null | tail -2 | sed 's/^/     /'
    echo "   (223 G of a 300 G limit was in use when this bundle was built -- the env is ~6 GB)"
fi

# >= 3.10 AND able to create a venv. Debian/Ubuntu ship python3.12 without the
# python3.12-venv package, so `python -m venv` dies on "ensurepip is not available" --
# which is what /usr/bin/python3.12 does on this node. Requiring ensurepip here makes the
# detection fall through to conda instead of failing after it has already made a
# half-built env directory.
py_ok() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1       && "$1" -c 'import ensurepip' >/dev/null 2>&1
}
first_python() {
    local c p
    for c in python3.12 python3.11 python3.10 python3 python; do
        p="$(command -v "$c" 2>/dev/null)" || continue
        py_ok "$p" && { echo "$p"; return 0; }
    done
    return 1
}

# An env copied from another account or path will not run here: move it aside.
if [ -e "$ENV_DIR" ] && ! env_python_runs; then
    stale="$ENV_DIR.stale.$(date +%Y%m%d_%H%M%S)"
    echo ">> $ENV_DIR exists but its python does not run here -> $stale"
    mv "$ENV_DIR" "$stale"
fi

# First usable conda wins. This node has no `module` command, so look in /apps/compilers
# directly (anaconda3-2024.06 ships conda 24.5.0) before trying PATH and then modules.
find_conda() {
    local c m
    for c in ${TNK1_CONDA:-} /apps/compilers/anaconda3-2024.06/bin/conda \
             /apps/compilers/anaconda3-2022.05/bin/conda; do
        [ -x "$c" ] && "$c" --version >/dev/null 2>&1 && { echo "$c"; return 0; }
    done
    c="$(command -v conda 2>/dev/null)" && [ -n "$c" ] && { echo "$c"; return 0; }
    for m in ${TNK1_ANACONDA_MODULE:-} anaconda3 Anaconda3 miniconda3 miniforge3 conda; do
        if _tnk1_mload "$m"; then
            c="$(command -v conda 2>/dev/null)" && [ -n "$c" ] && { ENV_MODULE="$m"; echo "$c"; return 0; }
        fi
    done
    return 1
}

if [ ! -e "$ENV_DIR" ]; then
    ENV_MODULE=""
    if [ -n "${TNK1_BASE_ENV_DIR:-}" ] && [ -x "$TNK1_BASE_ENV_DIR/bin/python" ] && \
       env -u PYTHONPATH PYTHONNOUSERSITE=1 "$TNK1_BASE_ENV_DIR/bin/python" -c "import torch" >/dev/null 2>&1; then
        echo ">> venv on top of $TNK1_BASE_ENV_DIR (reuses its torch)"
        "$TNK1_BASE_ENV_DIR/bin/python" -m venv --system-site-packages "$ENV_DIR"
    elif [ -n "${TNK1_PYTHON:-}" ] && py_ok "$TNK1_PYTHON"; then
        echo ">> venv from $TNK1_PYTHON ($("$TNK1_PYTHON" --version 2>&1))  [TNK1_PYTHON]"
        "$TNK1_PYTHON" -m venv "$ENV_DIR"
    elif CONDA="$(find_conda)"; then
        # conda-forge ONLY: Anaconda's 'defaults' channel refuses non-interactive installs
        # until its Terms of Service are accepted, which fails inside a batch job.
        echo ">> conda env (python 3.11 from conda-forge) using $CONDA"
        quiet=""; [ -t 1 ] || quiet="--quiet"
        if ! "$CONDA" create -y $quiet -p "$ENV_DIR" --override-channels -c conda-forge python=3.11 pip; then
            rm -rf "$ENV_DIR"
            echo "ERROR: conda could not create the env at $ENV_DIR" >&2
            exit 1
        fi
    elif PY="$(first_python)"; then
        echo ">> venv from $PY ($("$PY" --version 2>&1))"
        "$PY" -m venv "$ENV_DIR"
    else
        echo "ERROR: found no conda, and no python >= 3.10 that can create a venv." >&2
        echo "  This node's /usr/bin/python3.12 lacks ensurepip (no python3.12-venv package)" >&2
        echo "  and there is no 'module' command, so the search looked in /apps/compilers." >&2
        echo "  Point the setup somewhere explicitly:" >&2
        echo "    TNK1_CONDA=/path/to/conda         bash hpc/setup_env.sh" >&2
        echo "    TNK1_PYTHON=/path/to/python3.11   bash hpc/setup_env.sh" >&2
        exit 1
    fi
    printf '%s\n' "$ENV_MODULE" > "$ENV_DIR/.tnk1_module"
fi

enter_env
export PIP_INDEX_URL="https://pypi.org/simple" PIP_RETRIES=3 PIP_TIMEOUT=60 PIP_PREFER_BINARY=1
echo ">> python: $(command -v python)  ($(python --version 2>&1))"
python -m pip install --upgrade -q pip setuptools wheel

# ---- torch -----------------------------------------------------------------------
# Pick the wheel index from the driver's CUDA version. A 580 driver runs cu126/cu128/cu130
# builds, so fall back down the list rather than failing.
if ! python -c "import torch" >/dev/null 2>&1; then
    CUDA_MAJ="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)"
    if [ -n "${TNK1_TORCH_INDEX:-}" ]; then
        INDEXES=("$TNK1_TORCH_INDEX")
    else
        case "$CUDA_MAJ" in
            13*) INDEXES=(https://download.pytorch.org/whl/cu130 https://download.pytorch.org/whl/cu128 https://download.pytorch.org/whl/cu126) ;;
            12*) INDEXES=(https://download.pytorch.org/whl/cu128 https://download.pytorch.org/whl/cu126) ;;
            *)   INDEXES=(https://pypi.org/simple) ;;
        esac
    fi
    ok=0
    for idx in "${INDEXES[@]}"; do
        echo ">> installing ${TNK1_TORCH_SPEC:-torch} from $idx"
        if python -m pip install ${TNK1_TORCH_SPEC:-torch} --index-url "$idx"; then ok=1; break; fi
        echo "   (that index did not work; trying the next)"
    done
    [ "$ok" = 1 ] || { echo "ERROR: could not install torch" >&2; exit 1; }
fi

# ---- everything else -------------------------------------------------------------
# The floating ranges, NOT the 3.10 lock file (see the header).
python -m pip install -r "$TNK1_ROOT/requirements-core.txt"
# requirements-chem.txt starts with `-r requirements-core.txt`, so install it from the
# directory that holds both or the relative include fails.
( cd "$TNK1_ROOT" && python -m pip install -r requirements-chem.txt )
# gpytorch powers the Tanimoto GP baseline; PyTDC is only needed if the local
# admet_7endpoints.parquet is absent, which it is not. Both are optional -- do not let
# either one fail the whole setup.
python -m pip install gpytorch || echo "WARN: gpytorch not installed -- the Stage-2 TanimotoGP baseline will be skipped"

# ---- verify ----------------------------------------------------------------------
echo
echo ">> verifying the environment"
python - <<'EOF'
import importlib, sys
need = ["numpy", "pandas", "scipy", "sklearn", "yaml", "pyarrow", "torch", "rdkit", "xgboost"]
bad = []
for m in need:
    try:
        mod = importlib.import_module(m)
        print(f"   {m:10} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"   {m:10} MISSING ({type(e).__name__})"); bad.append(m)
for m in ("gpytorch", "tdc"):
    try:
        mod = importlib.import_module(m); print(f"   {m:10} {getattr(mod, '__version__', 'ok')} (optional)")
    except Exception:
        print(f"   {m:10} absent (optional)")
import numpy as np
assert np.__version__.split(".")[0] >= "2", f"numpy must be >= 2.0 (np.bitwise_count); got {np.__version__}"
import torch
print(f"   torch cuda : {torch.version.cuda}   available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   gpu        : {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GB")
sys.exit(1 if bad else 0)
EOF

echo
echo ">> env ready: $ENV_DIR"
echo ">> next:  bash hpc/preflight.sh"
