#!/bin/bash
# GPU pipeline step (train-retriever / dense / train-ce / score-ce):
#   cd ~/Amazon_ML/code/business_entity_resolution
#   sbatch hpc/submit_gpu.sh train-retriever --fold 0
#   sbatch hpc/submit_gpu.sh dense
#   sbatch hpc/submit_gpu.sh train-ce --fold 1 crossenc.batch_size=96
#   sbatch hpc/submit_gpu.sh score-ce --split test --fold 0
#
# Development node: one GPU split into 94 shards (~1 GB each). Half of it
# (shard:47) is plenty: bge-m3 / bge-reranker-v2-m3 are 568M params (bf16
# autocast + gradient checkpointing) and the steps peak near 40 GB; the two
# fold jobs of run_pipeline.sh then share the GPU. Whole GPU:
#   sbatch --gres=gpu:1 hpc/submit_gpu.sh ...        (other clusters: -p gpu --gres=gpu:1)
# On CUDA OOM lower retriever.batch_queries / crossenc.batch_size / dense.query_chunk.

#SBATCH --job-name=ber_gpu
#SBATCH --gres=shard:47
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=3-00:00:00
# one log per job: output and errors together, in order
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
_cfg=""
for _d in "${SLURM_SUBMIT_DIR:-.}/hpc" "${BER_CODE_DIR:-$HOME/Amazon_ML/code/business_entity_resolution}/hpc"; do
    [ -f "$_d/_config.sh" ] && { _cfg="$_d/_config.sh"; break; }
done
[ -n "$_cfg" ] || { echo "ERROR: hpc/_config.sh not found - sbatch from the code dir or set BER_CODE_DIR" >&2; exit 1; }
# shellcheck disable=SC1090
source "$_cfg"
activate_env

python -c "import transformers" 2>/dev/null || {
    echo "ERROR: transformers missing from $ENV_DIR - run bash hpc/setup_env.sh" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

STEP="${1:?usage: sbatch hpc/submit_gpu.sh <step> [--fold F] [--split S] [key=value ...]}"; shift
echo "[ber] node=$(hostname) step=$STEP args=$* gpu=${CUDA_VISIBLE_DEVICES:-?} work=${BER_WORK_DIR:-$BER_ROOT/work}"
python -m ber "$STEP" "$@"
