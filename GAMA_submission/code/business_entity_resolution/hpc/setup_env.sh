#!/bin/bash
# One-time environment setup: builds $BER_ROOT/envs/ber (python 3.10+, torch,
# transformers, requirements.txt; ~7 GB). Needs internet.
#
#   bash hpc/setup_env.sh                  # directly (inside tmux)
#   bash run_amazon_ml.sh setup            # the same, plus the models and the checks
# run_amazon_ml.sh also calls it by itself when the env is missing, from inside
# the job. That works where jobs can reach the internet (our development node
# is a single node); where compute nodes are offline, run it on the login node.
#
# torch: the CUDA build follows this node's NVIDIA driver (nvidia-smi "CUDA
# Version"), and pip takes the newest torch on that index that fits this python
# and glibc (glibc 2.17 -> torch 2.6). With no GPU on this node (a login node):
# 2.6.0+cu118 where glibc < 2.28, else the newest torch on PyPI.
# Your own choice:  BER_TORCH_SPEC="torch==2.6.0+cu124" BER_TORCH_INDEX=https://download.pytorch.org/whl/cu124
#
# Base python, first found wins:
#   1. BER_BASE_ENV_DIR=/env/with/torch   venv on top of it (skips the torch download)
#   2. BER_PYTHON=/path/to/python3.11     venv from that interpreter
#   3. conda from a module or on PATH     new env, python 3.11 from conda-forge
#   4. python >= 3.10 on PATH, then from a `python` module
#   5. last resort: Miniforge (conda-forge, BSD) installed into $BER_ROOT/miniforge3
# Re-running is safe: it only installs what is missing.

set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_here/_config.sh"
REQ_FILE="$CODE_DIR/requirements.txt"

if ! ber_online https://pypi.org/simple/pip/; then
    echo "ERROR: $(hostname) cannot reach pypi.org, and the setup needs internet." >&2
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "  This cluster's job nodes are offline: run it on the login node:  bash $CODE_DIR/hpc/setup_env.sh" >&2
    else
        echo "  Behind a proxy?  export https_proxy=http://<proxy>:<port>  and run it again." >&2
    fi
    exit 1
fi
# fail now rather than after a 20-minute install: the env and the lock file go here
ber_need_writable "$(dirname "$ENV_DIR")" "$CODE_DIR" || exit 1
echo ">> free disk where the env goes: $(df -hP "$(dirname "$ENV_DIR")" | awk 'NR==2 {print $4}')  (the env needs ~7 GB)"

py_ok() {  # python >= 3.10 ?
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}
first_python() {  # first python >= 3.10 on PATH
    local c p
    for c in python3.12 python3.11 python3.10 python3 python; do
        p="$(command -v "$c" 2>/dev/null)" || continue
        py_ok "$p" && { echo "$p"; return 0; }
    done
    return 1
}
bootstrap_miniforge() {
    local dir="$BER_ROOT/miniforge3" arch inst url
    arch="$(uname -m)"
    url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$arch.sh"
    inst="$BER_ROOT/Miniforge3-Linux-$arch.sh"
    if [ ! -x "$dir/bin/conda" ]; then
        echo ">> no conda / python>=3.10 found - installing Miniforge into $dir"
        if command -v curl >/dev/null 2>&1; then
            curl -fL -o "$inst" "$url" || return 1
        elif command -v wget >/dev/null 2>&1; then
            wget -O "$inst" "$url" || return 1
        else
            local anypy
            anypy="$(command -v python3 || command -v python || true)"
            [ -n "$anypy" ] || { echo "   (no curl, wget or python to download it)" >&2; return 1; }
            "$anypy" -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
                "$url" "$inst" || return 1
        fi
        bash "$inst" -b -p "$dir" || return 1
        rm -f "$inst"
    fi
    export PATH="$dir/bin:$PATH"
    command -v conda >/dev/null 2>&1
}

# an env copied from another user / path does not run here: move it aside
if [ -e "$ENV_DIR" ] && ! env_python_runs; then
    stale="$ENV_DIR.stale.$(date +%Y%m%d_%H%M%S)"
    echo ">> $ENV_DIR exists but its python does not run here (copied from another account/path?) -> $stale"
    mv "$ENV_DIR" "$stale"
fi

if [ ! -e "$ENV_DIR" ]; then
    how="" PY="" ENV_MODULE=""
    if [ -n "$BASE_ENV_DIR" ] && [ -x "$BASE_ENV_DIR/bin/python" ] && \
       env -u PYTHONPATH PYTHONNOUSERSITE=1 "$BASE_ENV_DIR/bin/python" -c "import torch, transformers" >/dev/null 2>&1; then
        how="venv-base"
    elif [ -n "${BER_PYTHON:-}" ] && py_ok "$BER_PYTHON"; then
        how="venv" PY="$BER_PYTHON"
    else
        for m in ${BER_ANACONDA_MODULE:-} anaconda3-2024.2 anaconda3 Anaconda3 anaconda miniconda3 Miniconda3 miniforge3 Miniforge3 conda mamba; do
            if _ber_mload "$m"; then
                if command -v conda >/dev/null 2>&1; then how="conda"; echo ">> conda from module '$m'"; break; fi
                module unload "$m" >/dev/null 2>&1 || true
            fi
        done
        if [ -z "$how" ] && command -v conda >/dev/null 2>&1; then how="conda"; fi
        if [ -z "$how" ] && PY="$(first_python)"; then how="venv"; fi
        if [ -z "$how" ]; then
            for m in python Python python3 Python3; do
                if _ber_mload "$m"; then
                    if PY="$(first_python)"; then how="venv" ENV_MODULE="$m"; echo ">> python from module '$m'"; break; fi
                    module unload "$m" >/dev/null 2>&1 || true
                fi
            done
        fi
        if [ -z "$how" ] && bootstrap_miniforge; then how="conda"; fi
    fi

    case "$how" in
        venv-base)
            echo ">> venv on top of $BASE_ENV_DIR (reuses its torch/transformers)"
            "$BASE_ENV_DIR/bin/python" -m venv --system-site-packages "$ENV_DIR" ;;
        venv)
            echo ">> venv from $PY ($("$PY" --version 2>&1))"
            "$PY" -m venv "$ENV_DIR" ;;
        conda)
            # conda-forge only: Anaconda's 'defaults' channels now refuse
            # non-interactive installs until their Terms of Service are accepted
            echo ">> conda env (python 3.11 from conda-forge) using $(command -v conda)"
            quiet=""; [ -t 1 ] || quiet="--quiet"          # no progress bars in a log file
            if ! conda create -y $quiet -p "$ENV_DIR" --override-channels -c conda-forge python=3.11 pip; then
                rm -rf "$ENV_DIR"
                PY="$(first_python)" || { echo "ERROR: conda failed and no python >= 3.10 is on PATH to fall back on" >&2; exit 1; }
                echo ">> conda failed - venv from $PY ($("$PY" --version 2>&1)) instead"
                "$PY" -m venv "$ENV_DIR"
            fi ;;
        *)
            echo "ERROR: found no conda and no python >= 3.10, and Miniforge could not be installed." >&2
            echo "  See what the cluster offers:  module avail 2>&1 | grep -i -E 'conda|python'" >&2
            echo "  then re-run with e.g.  BER_ANACONDA_MODULE=<module>  or  BER_PYTHON=/path/to/python3.11" >&2
            exit 1 ;;
    esac
    printf '%s\n' "$ENV_MODULE" > "$ENV_DIR/.ber_module"
fi

enter_env
export PIP_INDEX_URL="https://pypi.org/simple" PIP_EXTRA_INDEX_URL="" PIP_RETRIES=3 PIP_TIMEOUT=60
# Take the newest version that has a ready-made wheel for this machine instead of
# compiling a newer one: on glibc 2.17 numpy >= 2.3 has no wheel, and
# compiling it fails there (binutils 2.27 cannot assemble its AVX-512 code).
export PIP_PREFER_BINARY=1
echo ">> using python: $(command -v python)  ($(python --version 2>&1))"
py_ok "$(command -v python)" || { echo "ERROR: env python is older than 3.10 - delete $ENV_DIR and re-run" >&2; exit 1; }

pip_install_with_retries() {   # retries network failures; gives up at once when retrying cannot help
    local attempt log
    log="$(mktemp)"
    for attempt in 1 2 3; do
        if python -m pip install --index-url https://pypi.org/simple --timeout 120 --retries 3 "$@" 2>&1 | tee "$log"; then
            rm -f "$log"; return 0
        fi
        if grep -qE "No matching distribution|Could not find a version" "$log"; then
            echo "!! no installable build of: $*  (for this python / glibc)" >&2
            rm -f "$log"; return 1
        fi
        if grep -qE "metadata-generation-failed|Failed to build|Failed building wheel|subprocess-exited-with-error" "$log"; then
            echo "!! no ready-made wheel of: $* fits this machine, and compiling it here failed" >&2
            rm -f "$log"; return 1
        fi
        echo "pip install failed (attempt $attempt/3) - retrying in 30s..." >&2
        sleep 30
    done
    rm -f "$log"
    echo "ERROR: pip install kept failing: $*" >&2
    return 1
}

python -m pip install --upgrade pip >/dev/null 2>&1 || true

# -- torch: a build this node's driver and glibc can run -------------------
# The largest download (~2.5 GB with the CUDA libraries), so first and alone.
# No PyPI extra index here: PyPI's newer default build would win over the
# CUDA build picked for this driver.
if python -c "import torch" 2>/dev/null; then
    echo ">> torch $(python -c 'import torch; print(torch.__version__)') already installed"
elif [ -n "${BER_TORCH_SPEC:-}${BER_TORCH_INDEX:-}" ]; then
    pip_install_with_retries --index-url "${BER_TORCH_INDEX:-https://pypi.org/simple}" "${BER_TORCH_SPEC:-torch}"
else
    DRV_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -n 1 || true)"
    GLIBC="$(python -c 'import platform; print(platform.libc_ver()[1] or "0")')"
    BUILDS="$(python - "$DRV_CUDA" "$GLIBC" <<'PY'
import sys
def ver(s):
    p = [int(x) for x in s.split(".")[:2] if x.isdigit()]
    return tuple(p + [0] * (2 - len(p)))
drv, glibc = sys.argv[1], ver(sys.argv[2])
cuda_builds = [((13, 0), "cu130"), ((12, 8), "cu128"), ((12, 6), "cu126"),
               ((12, 4), "cu124"), ((12, 1), "cu121"), ((11, 8), "cu118")]
if drv:    # GPU on this node: every build its driver can run, newest first
    print(" ".join(tag for need, tag in cuda_builds if ver(drv) >= need))
else:      # no GPU here (a login node): what is known to run on the GPU nodes
    print("cu118" if glibc < (2, 28) else "pypi")
PY
)"
    echo ">> NVIDIA driver CUDA ${DRV_CUDA:-? (no GPU on this node)}, glibc $GLIBC -> torch builds to try: ${BUILDS:-none}"
    got=0
    for b in $BUILDS; do
        if [ "$b" = pypi ]; then idx="https://pypi.org/simple"; else idx="https://download.pytorch.org/whl/$b"; fi
        echo ">> newest torch from $idx"
        if pip_install_with_retries --index-url "$idx" torch; then got=1; break; fi
    done
    if [ "$got" != 1 ]; then
        echo "ERROR: no torch build installs here (driver CUDA ${DRV_CUDA:-?}, glibc $GLIBC)." >&2
        echo "  Choose one:  BER_TORCH_SPEC=torch==<ver>+cu<xyz> BER_TORCH_INDEX=https://download.pytorch.org/whl/cu<xyz> bash $0" >&2
        exit 1
    fi
fi

# transformers 5 goes with a recent torch; stay on 4.x with an older one
TORCH_V="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
if python -c "import sys; v = tuple(int(x) for x in '$TORCH_V'.split('.')[:2]); sys.exit(0 if v >= (2, 7) else 1)"; then
    TF_SPEC="transformers>=4.44"
else
    TF_SPEC="transformers>=4.44,<5"
fi
python -c "import transformers" 2>/dev/null || pip_install_with_retries "$TF_SPEC" huggingface_hub safetensors

# -- requirements.txt, one package at a time ------------------------------
# Speed-ups are optional: without them the pipeline uses slower built-in
# fallbacks (sklearn GBDT, python string similarity, scipy sparse products).
# A core pin with no build for this machine falls back to the newest one that has.
OPTIONAL=" lightgbm rapidfuzz sparse_dot_topn sparse-dot-topn anyascii pytest "
while IFS= read -r line || [ -n "$line" ]; do
    req="$(printf '%s' "${line%%#*}" | tr -d '[:space:]')"
    [ -z "$req" ] && continue
    name="$(printf '%s' "$req" | sed -E 's/[<>=!~;[].*//' | tr 'A-Z' 'a-z')"
    case "$name" in torch|transformers|huggingface_hub) continue ;; esac   # installed above (CUDA build)
    if [[ "$OPTIONAL" == *" $name "* ]]; then
        pip_install_with_retries "$req" || \
            echo "WARN: optional '$req' not installed - the pipeline will use its slower fallback" >&2
    else
        pip_install_with_retries "$req" || {
            echo ">> '$req' has no build here - installing the newest compatible $name" >&2
            pip_install_with_retries "$name"
        }
    fi
done < "$REQ_FILE"

# exact versions actually installed -> reproducibility record for the submission zip
python -m pip freeze > "$CODE_DIR/requirements.lock.txt"

echo
echo "Environment at: $ENV_DIR   (lock file: $CODE_DIR/requirements.lock.txt)"
python - <<'PY'
import importlib, shutil, subprocess, sys
core = ["torch", "transformers", "huggingface_hub", "safetensors", "polars", "numpy", "scipy", "sklearn", "yaml"]
missing = []
for m in core + ["lightgbm", "rapidfuzz", "sparse_dot_topn", "anyascii"]:
    try:
        x = importlib.import_module(m)
        print(f"  {m:16s} {getattr(x, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {m:16s} MISSING ({e.__class__.__name__})")
        if m in core:
            missing.append(m)
if "torch" not in missing:
    import torch
    gpu_here = shutil.which("nvidia-smi") is not None and \
        subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
    if torch.cuda.is_available():
        print(f"  cuda             {torch.version.cuda} on {torch.cuda.get_device_name(0)}")
    elif gpu_here:
        print(f"!! this node has a GPU, but torch {torch.__version__} cannot use it (driver older than CUDA {torch.version.cuda}?)")
        missing.append("a torch that can use this GPU")
    else:
        print("  cuda             no GPU on this node (normal on a login node)")
if missing:
    sys.exit("ENV NOT READY - missing: " + ", ".join(missing))
print("ENV READY")
PY
