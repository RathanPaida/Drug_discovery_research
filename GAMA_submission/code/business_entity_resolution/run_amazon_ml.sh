#!/bin/bash

#SBATCH --job-name=amazon_ml_er

# =====================================================
# QUEUE
# =====================================================
# No --partition: the default queue (the only one on our development node).
# Other clusters: add it when submitting, e.g.  sbatch -p gpu ...

# =====================================================
# RESOURCES  (development node: one shared node, 128 CPUs, 125 GB RAM)
# =====================================================

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G

# =====================================================
# GPU REQUEST
# =====================================================
# The GPU is split into 94 shards (~1 GB each); half of it is plenty, the
# pipeline peaks near 40 GB. Whole GPU, when nobody else is using it:
#   sbatch --gres=gpu:1 run_amazon_ml.sh full

#SBATCH --gres=shard:47

# =====================================================
# TIME  (LocalQ has no limit; finished steps are kept, so re-submitting resumes)
# =====================================================

#SBATCH --time=3-00:00:00

# =====================================================
# LOG  (one file: output, warnings and errors together, in the order they happen)
# =====================================================

#SBATCH --output=amazon_ml_%j.out

# =====================================================
# AMAZON ML CHALLENGE 2026 - BUSINESS ENTITY RESOLUTION
# =====================================================
# Everything in one job on one GPU, starting from a bare account:
#   python env + models (first run only; needs internet, ~20 min)
#   -> prepare -> sparse -> train-retriever (fold 0, 1) -> dense -> features -> prune
#   -> train-ce (fold 0, 1) -> score-ce (train, test) -> stack -> frpost
#
# From the project folder (the one holding code/ and data/), e.g. ~/Amazon_ML :
#   sbatch run_amazon_ml.sh smoke         # 20k-entity sample + short training: tests every step (~1 h)
#   sbatch run_amazon_ml.sh full          # the real run
#   sbatch run_amazon_ml.sh full dense    # redo a step and everything after it
#   sbatch run_amazon_ml.sh retest        # test side only, after a test-side change (e.g. the
#                                         # France add-on fix): prepare/sparse/dense/features/prune/
#                                         # score-ce with --split test, saved models; then
#                                         #   sbatch hpc/submit_cpu.sh stack --split test
#   RETEST_COUNTRIES=France sbatch run_amazon_ml.sh retest
#                                         # same, but dense / prune / score-ce redo only France and
#                                         # keep US+India from the previous run (~45 min instead of ~3 h)
#   bash   run_amazon_ml.sh setup         # optional: env + models + checks now, without a GPU job
#
# Watch:  squeue -u $USER ;  tail -f amazon_ml_<jobid>.out
# Finished steps are remembered (<work>/.done/). If the job stops, submit the
# same command again: it skips finished steps, and training resumes from its
# last checkpoint.
# =====================================================

set -eEo pipefail
exec 2>&1                      # errors land in the log right where they happen, also when run with bash
export PYTHONUNBUFFERED=1      # python output reaches the log at once, in order

MODE="${1:-full}"
FROM="${2:-}"
case "$MODE" in
    setup|smoke|full|retest) ;;
    *) echo "usage: sbatch run_amazon_ml.sh smoke|full|retest [from_step]    or    bash run_amazon_ml.sh setup"; exit 2 ;;
esac
if [ "$MODE" != setup ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: '$MODE' needs a GPU from the queue. Submit it:  sbatch run_amazon_ml.sh $MODE $FROM" >&2
    exit 2
fi

# project folder: $BER_ROOT, else the folder you ran sbatch / bash from
PROJECT_DIR=""
for d in ${BER_ROOT:+"$BER_ROOT"} "${SLURM_SUBMIT_DIR:-$PWD}" "${SLURM_SUBMIT_DIR:-$PWD}/../.." "$(dirname "$0")" \
         "$HOME/Amazon_ML"; do
    if [ -f "$d/code/business_entity_resolution/hpc/_config.sh" ]; then PROJECT_DIR="$(cd "$d" && pwd)"; break; fi
done
if [ -z "$PROJECT_DIR" ]; then
    echo "ERROR: project not found. Run this from code/business_entity_resolution/ or from the project root" >&2
    echo "       (the folder with code/ and data/)," >&2
    echo "       or set BER_ROOT=/path/to/Amazon_ML" >&2
    exit 1
fi
CODE_DIR="$PROJECT_DIR/code/business_entity_resolution"
cd "$CODE_DIR"
# shellcheck disable=SC1091
source hpc/_config.sh          # env location, model dir, setup helpers

trap 'echo "=================================================="; echo "!!! FAILED - see the error just above"; echo "!!! Fix it, then run the same command again: finished steps are skipped."; echo "End Time  : $(date)"' ERR

echo "=================================================="
echo "AMAZON ML - ENTITY RESOLUTION  (mode: $MODE)"
echo "=================================================="

echo "Job ID           : ${SLURM_JOB_ID:-none (outside the queue)}"
echo "Job Name         : ${SLURM_JOB_NAME:-}"
echo "Partition        : ${SLURM_JOB_PARTITION:-}"
echo "Node             : ${SLURM_NODELIST:-$(hostname)}"
echo "Project          : $PROJECT_DIR"
echo "Start Time       : $(date)"

# =====================================================
# ONE-TIME SETUP: python env + models (skipped once present)
# =====================================================

echo "=================================================="
echo "SETUP (python env + models)"
echo "=================================================="

if ber_env_ready; then
    echo "Python env ready : $ENV_DIR"
else
    echo "Building the python env at $ENV_DIR (first run only, ~15 min)"
    bash hpc/setup_env.sh
fi
if ber_models_ready; then
    echo "Models ready     : $BER_MODEL_DIR"
else
    echo "Downloading bge-m3 + bge-reranker-v2-m3 into $BER_MODEL_DIR (first run only, ~4.6 GB)"
    bash hpc/download_models.sh
fi

if [ "$MODE" = setup ]; then
    bash hpc/preflight.sh
    exit 0
fi

# =====================================================
# WORK / OUTPUT LOCATIONS
# =====================================================

# big intermediate files (~50 GB): /scratch/$USER when the cluster has it, else inside the project
_me="${USER:-$(id -un)}"
if [ -n "${BER_SCRATCH_DIR:-}" ]; then
    SCRATCH_DIR="$BER_SCRATCH_DIR"
elif [ -d "/scratch/$_me" ] && [ -w "/scratch/$_me" ]; then
    SCRATCH_DIR="/scratch/$_me/amazon_ml"
else
    SCRATCH_DIR="$PROJECT_DIR"
fi

case "$MODE" in
    smoke)
        export BER_WORK_DIR="$SCRATCH_DIR/work_smoke"
        export BER_OUTPUT_DIR="$PROJECT_DIR/output_smoke"
        PREPARE_ARGS="data.sample_s1=20000"
        RETRIEVER_ARGS="retriever.max_steps=60 retriever.batch_queries=64"
        CE_ARGS="crossenc.max_steps=60"
        ;;
    full|retest)
        export BER_WORK_DIR="$SCRATCH_DIR/work"
        export BER_OUTPUT_DIR="$PROJECT_DIR/output"
        PREPARE_ARGS=""
        RETRIEVER_ARGS=""
        CE_ARGS=""
        ;;
esac

echo "=================================================="
echo "ACTIVATING ENV"
echo "=================================================="

activate_env                   # python env from hpc/setup_env.sh; offline HF, thread counts = allocated CPUs

echo "Current Directory: $(pwd)"
echo "Python Executable: $(command -v python)"
echo "Python Version   : $(python --version 2>&1)"
echo "Work dir         : $BER_WORK_DIR"
echo "Output dir       : $BER_OUTPUT_DIR"
echo "Models           : $BER_MODEL_DIR"
echo "CPU threads      : $OMP_NUM_THREADS"

echo "=================================================="
echo "GPU INFORMATION"
echo "=================================================="

nvidia-smi || true
python - <<'PY'
import os, sys
import torch
if not torch.cuda.is_available():
    sys.exit(f"ERROR: torch {torch.__version__} sees no GPU in this job "
             f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
free, total = torch.cuda.mem_get_info()
print(f"GPU              : {torch.cuda.get_device_name(0)}, {free / 2**30:.0f} of {total / 2**30:.0f} GB free "
      f"(torch {torch.__version__}, CUDA {torch.version.cuda})")
PY

echo "=================================================="
echo "STORAGE"
echo "=================================================="

DONE_DIR="$BER_WORK_DIR/.done"
ber_need_writable "$DONE_DIR" "$BER_OUTPUT_DIR" || exit 1
df -h "$BER_WORK_DIR" "$BER_OUTPUT_DIR" || true

# =====================================================
# PIPELINE
# =====================================================

STEPS=(prepare sparse retriever_f0 retriever_f1 dense features prune ce_f0 ce_f1 score_train score_test stack frpost)
if [ "$MODE" = retest ]; then
    # own markers, so the full run's finished steps stay untouched
    STEPS=(retest_prepare retest_sparse retest_dense retest_features retest_prune retest_score)
fi

# "full dense" -> forget 'dense' and every later step, so they run again
if [ -n "$FROM" ]; then
    found=0
    for s in "${STEPS[@]}"; do
        [ "$s" = "$FROM" ] && found=1
        [ "$found" = 1 ] && rm -f "$DONE_DIR/$s"
    done
    if [ "$found" = 0 ]; then
        echo "unknown step '$FROM' - one of: ${STEPS[*]}"
        exit 2
    fi
fi

run_step() {   # run_step <name> <ber step> [args...]
    local name="$1"; shift
    if [ -f "$DONE_DIR/$name" ]; then
        echo ">>> [$name] already done - skipping"
        return 0
    fi
    echo "=================================================="
    echo "STEP: $name    $(date)"
    echo "      python -m ber $*"
    echo "=================================================="
    local t0=$SECONDS
    python -m ber "$@"
    touch "$DONE_DIR/$name"
    echo ">>> [$name] finished in $(( (SECONDS - t0) / 60 )) min"
}

echo "=================================================="
echo "PREFLIGHT (data / packages / models)"
echo "=================================================="

python -m ber preflight

# shellcheck disable=SC2086  # the *_ARGS strings are meant to split into words
if [ "$MODE" = retest ]; then
ONLY_ARGS=""
if [ -n "${RETEST_COUNTRIES:-}" ]; then
    ONLY_ARGS="data.only_countries=$RETEST_COUNTRIES"
    echo ">>> partial test rerun: dense / prune / score-ce redo only $RETEST_COUNTRIES"
fi
{
run_step retest_prepare  prepare --split test
run_step retest_sparse   sparse --split test
run_step retest_dense    dense $ONLY_ARGS --split test
run_step retest_features features --split test
run_step retest_prune    prune $ONLY_ARGS --split test
run_step retest_score    score-ce $ONLY_ARGS --split test
echo ">>> test side redone. Submission: sbatch hpc/submit_cpu.sh stack --split test"
echo ">>> (after 'stack --split train' has finished, if new stackers are being trained)"
}
else
{
run_step prepare      prepare $PREPARE_ARGS
run_step sparse       sparse
run_step retriever_f0 train-retriever --fold 0 $RETRIEVER_ARGS
run_step retriever_f1 train-retriever --fold 1 $RETRIEVER_ARGS
run_step dense        dense
run_step features     features
run_step prune        prune
run_step ce_f0        train-ce --fold 0 $CE_ARGS
run_step ce_f1        train-ce --fold 1 $CE_ARGS
run_step score_train  score-ce --split train
run_step score_test   score-ce --split test
run_step stack        stack
run_step frpost       frpost    # France (no training labels): decoy drops + filler-copy adds, in place
}
fi

EXIT_CODE=$?

# =====================================================
# JOB FINISHED
# =====================================================

echo "=================================================="
echo "JOB FINISHED"
echo "=================================================="

echo "Exit Code : $EXIT_CODE"

echo "End Time  : $(date)"

echo "=================================================="
echo "VALIDATION (out-of-fold macro F0.5 on the training set)"
echo "=================================================="

python - <<'PY' || true
import json, os
p = os.path.join(os.environ["BER_WORK_DIR"], "stack", "report.json")
if os.path.exists(p):
    r = json.load(open(p))
    print("  out-of-fold :", json.dumps(r.get("oof"), indent=None))
    print("  stress test :", json.dumps(r.get("stress"), indent=None))
    print("  decision    :", r.get("decision"))
else:
    print("  (no report yet)", p)
PY

echo "=================================================="
echo "OUTPUT LOCATION"
echo "=================================================="

echo "$BER_OUTPUT_DIR"
ls -lh "$BER_OUTPUT_DIR"

exit $EXIT_CODE
