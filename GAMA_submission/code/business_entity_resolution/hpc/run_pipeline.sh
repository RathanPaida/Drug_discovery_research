#!/bin/bash
# Submit a whole pipeline as a chain of dependent SLURM jobs (the alternative
# to the single job run_amazon_ml.sh). Run on the login node:
#
#   bash hpc/run_pipeline.sh smoke           # 20k-S1 sample, capped training - exercises every step
#   bash hpc/run_pipeline.sh baseline        # CPU only: lexical retrieval + GBDT -> first valid submission
#   bash hpc/run_pipeline.sh full            # bge-m3 retrieval + bge-reranker cross-encoder (the real run)
#   bash hpc/run_pipeline.sh full dense      # resume a chain from a step (earlier outputs must exist)
#
# If the python env or the models are missing, it builds / downloads them first
# (needs internet, ~20 min), then queues the jobs.
# Steps: prepare sparse train-retriever dense features prune train-ce stack frpost
# Watch:  squeue -u $USER ;  tail -f logs/ber_<step>_<jobid>.out
# smoke writes to $BER_ROOT/work_smoke and $BER_ROOT/output_smoke only.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source hpc/_config.sh                   # BER_ROOT: <root>/code/business_entity_resolution -> <root>
ROOT="$BER_ROOT"
RUN_MODE="${1:-full}"
MODE="$RUN_MODE"
FROM="${2:-prepare}"
K=2                                     # must equal data.n_folds_nn
STEPS=" prepare sparse train-retriever dense features prune train-ce stack frpost "
case "$STEPS" in *" $FROM "*) ;; *) echo "unknown step '$FROM' (one of:$STEPS)" >&2; exit 1 ;; esac
case "$MODE" in smoke|baseline|full) ;; *) echo "mode must be smoke|baseline|full" >&2; exit 1 ;; esac

extra_prepare=() extra_ret=() extra_ce=()
if [ "$MODE" = "smoke" ]; then
    export BER_WORK_DIR="$ROOT/work_smoke" BER_OUTPUT_DIR="$ROOT/output_smoke"
    extra_prepare=(data.sample_s1=20000)
    extra_ret=(retriever.max_steps=60 retriever.batch_queries=64)
    extra_ce=(crossenc.max_steps=60)
    MODE=full
fi
# SLURM opens each job's log in logs/ before the job starts; if it can't, the job dies without a trace
ber_need_writable logs "${BER_WORK_DIR:-$ROOT/work}" "${BER_OUTPUT_DIR:-$ROOT/output}" || exit 1

# one-time setup before anything is queued (skipped once done)
ber_env_ready || bash hpc/setup_env.sh
[ "$RUN_MODE" = baseline ] || ber_models_ready || bash hpc/download_models.sh   # baseline uses no models

last=""                                  # job id(s) the next step depends on
# extra sbatch options for every job, e.g. if the cluster caps memory/time:
#   BER_SBATCH_ARGS="--mem=64G --time=12:00:00" bash hpc/run_pipeline.sh smoke
extra_sbatch=()
[ -n "${BER_SBATCH_ARGS:-}" ] && read -r -a extra_sbatch <<< "$BER_SBATCH_ARGS"
submit() {  # submit <cpu|gpu> <job-name> <deps> <args...>
    local kind="$1" name="$2" deps="$3"; shift 3
    local opts=(--parsable --job-name="ber_$name" ${extra_sbatch[@]+"${extra_sbatch[@]}"})
    [ -n "$deps" ] && opts+=(--dependency="afterok:$deps")
    sbatch "${opts[@]}" "hpc/submit_$kind.sh" "$@"
}

started=0
step() {  # true from FROM onwards
    if [ "$started" = 1 ] || [ "$1" = "$FROM" ]; then started=1; return 0; fi
    return 1
}

if step prepare; then last=$(submit cpu prepare "$last" prepare ${extra_prepare[@]+"${extra_prepare[@]}"}); echo "prepare   $last"; fi
if step sparse;  then last=$(submit cpu sparse "$last" sparse); echo "sparse    $last"; fi

if [ "$MODE" = "full" ]; then
    if step train-retriever; then
        ids=()
        for f in $(seq 0 $((K - 1))); do
            ids+=("$(submit gpu "ret_f$f" "$last" train-retriever --fold "$f" ${extra_ret[@]+"${extra_ret[@]}"})")
        done
        last=$(IFS=:; echo "${ids[*]}"); echo "retriever $last"
    fi
    if step dense; then last=$(submit gpu dense "$last" dense); echo "dense     $last"; fi
else
    step train-retriever || true; step dense || true
fi

if step features; then last=$(submit cpu features "$last" features); echo "features  $last"; fi
if step prune;    then last=$(submit cpu prune "$last" prune); echo "prune     $last"; fi

if [ "$MODE" = "full" ] && step train-ce; then
    ids=()
    for f in $(seq 0 $((K - 1))); do
        t=$(submit gpu "ce_f$f" "$last" train-ce --fold "$f" ${extra_ce[@]+"${extra_ce[@]}"})
        ids+=("$(submit gpu "sce_tr_f$f" "$t" score-ce --split train --fold "$f")")
        ids+=("$(submit gpu "sce_te_f$f" "$t" score-ce --split test --fold "$f")")
    done
    last=$(IFS=:; echo "${ids[*]}"); echo "cross-enc $last"
fi

if step stack; then last=$(submit cpu stack "$last" stack); echo "stack     $last"; fi
# France (countries without training labels): decoy drops + filler-copy adds, in place on output/
if step frpost; then last=$(submit cpu frpost "$last" frpost); echo "frpost    $last"; fi
echo "submitted ($RUN_MODE). Final job: ${last:-none}  -> ${BER_OUTPUT_DIR:-$ROOT/output}"
