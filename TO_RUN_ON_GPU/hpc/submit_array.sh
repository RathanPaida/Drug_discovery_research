#!/bin/bash
# One Slurm ARRAY task per item of a sweep. Submitted by hpc/run_sweeps.sh, not by hand.
#
# Why an array rather than the single-lane chain in hpc/run_pipeline.sh: that chain exists
# to cap GPU shards at 16 for stages that must run in a fixed order (ens -> oof ->
# calibrate -> rl -> cycle). A sweep is the opposite shape -- N independent runs with no
# dependencies between them -- so the right tool is an array with a concurrency limit,
# which Slurm enforces itself via the `%` suffix on --array.
#
#   sbatch --array=0-7%2 hpc/submit_array.sh rlalgo        # 8 algorithms, 2 at a time
#   sbatch --array=0-3%4 hpc/submit_array.sh zoo           # 4 model shortlists, CPU
#
# $1 selects the sweep; $SLURM_ARRAY_TASK_ID selects the item within it. The item lists
# live here, in one place, so a sweep's membership is reviewable in the diff rather than
# retyped on a login node.

#SBATCH --job-name=dug_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -uo pipefail
_cfg=""
for _d in "${SLURM_SUBMIT_DIR:-.}/hpc" "${TNK1_ROOT:-$HOME/TO_RUN_ON_GPU}/hpc" "$(dirname "${BASH_SOURCE[0]}")"; do
    [ -f "$_d/_config.sh" ] && { _cfg="$_d/_config.sh"; break; }
done
[ -n "$_cfg" ] || { echo "ERROR: hpc/_config.sh not found - sbatch from the bundle root, or export TNK1_ROOT" >&2; exit 1; }
# shellcheck disable=SC1090
source "$_cfg"
activate_env

SWEEP="${1:?usage: sbatch --array=0-N hpc/submit_array.sh <rlalgo|zoo|zoorep|transfer>}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
VERSION="${TNK1_VERSION:-v3}"

# ---------------------------------------------------------------- the sweep item lists
#
# RL ALGORITHMS. Each is a full policy-optimisation run against the production reward, so
# each costs roughly what the v3 DAP run cost (165 GPU-min at 1000 steps x batch 128).
# Ordered cheapest-insight-first: `ahc` is the one with a published sample-efficiency
# claim, `grpo` and `rloo` are critic-free and so cost no extra model, `gflownet` is the
# one that samples proportional to reward rather than seeking the mode.
# Ordered by expected value, not alphabetically. `augmem` first after the incumbent
# because it has the largest measured margin on the practical molecular optimisation
# benchmark (summed AUC Top-10 15.002 vs REINVENT's 14.016, 19 of 23 tasks); `gflownet`
# last because it ranks 17th of 25 there despite being the right SHAPE of answer to mode
# collapse. Submit 0-3 first if the budget is tight.
RL_ALGOS=(dap augmem ahc drgrpo rloo ppo grpo gflownet)

# MODEL-ZOO SHORTLISTS, by tag. `core` is the literature-backed set to run every time.
ZOO_TAGS=(core probe slow control)

# REPRESENTATION SWEEP: one task per representation, every `core` model fitted on it.
# `admet_best_combo` is the combination Kamuntavicius et al. (J Cheminform 17:108, 2025)
# found best for regression over 25 ADMET datasets; `ecfp4_desc200` is this project's
# incumbent and is included so the comparison has a control.
ZOO_REPS=(ecfp4_desc200 admet_best_combo ecfp4_count rdkit_desc_all fp_union desc_erg_avalon)

case "$SWEEP" in
  rlalgo)
    ALGO="${RL_ALGOS[$IDX]:-}"
    [ -n "$ALGO" ] || { echo "ERROR: array index $IDX exceeds the ${#RL_ALGOS[@]} RL algorithms" >&2; exit 2; }
    echo "[array] sweep=rlalgo idx=$IDX algo=$ALGO config=$TNK1_CONFIG"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    python -c "import torch; assert torch.cuda.is_available(), 'no GPU in this job'; \
print('[tnk1] torch', torch.__version__, '| gpu', torch.cuda.get_device_name(0))"
    python scripts/run_pipeline.py --version "$VERSION" --device cuda \
           --only rl --rl-algo "$ALGO" "${@:2}"
    ;;

  zoo)
    TAG="${ZOO_TAGS[$IDX]:-}"
    [ -n "$TAG" ] || { echo "ERROR: array index $IDX exceeds the ${#ZOO_TAGS[@]} zoo tags" >&2; exit 2; }
    echo "[array] sweep=zoo idx=$IDX tag=$TAG config=$TNK1_CONFIG"
    python scripts/run_stage2b_zoo.py --version "$VERSION" --tag "$TAG" \
           --repeats "${TNK1_ZOO_REPEATS:-10}" \
           --out "runs/stage2b_zoo_${TAG}" "${@:2}"
    ;;

  zoorep)
    REP="${ZOO_REPS[$IDX]:-}"
    [ -n "$REP" ] || { echo "ERROR: array index $IDX exceeds the ${#ZOO_REPS[@]} representations" >&2; exit 2; }
    echo "[array] sweep=zoorep idx=$IDX rep=$REP config=$TNK1_CONFIG"
    python scripts/run_stage2b_zoo.py --version "$VERSION" --tag core --rep "$REP" \
           --repeats "${TNK1_ZOO_REPEATS:-10}" \
           --out "runs/stage2b_rep_${REP}" "${@:2}"
    ;;

  transfer)
    # One task per fine-tuning strategy. The 10-entry LADDER lives in
    # scripts/run_stage3b_transfer.py and is printed by `--list`, so the index->strategy
    # mapping is reviewable rather than implied:
    #   sbatch --array=0-9%2 hpc/submit_array.sh transfer
    echo "[array] sweep=transfer idx=$IDX config=$TNK1_CONFIG"
    python scripts/run_stage3b_transfer.py --version "$VERSION" --device cuda \
           --index "$IDX" "${@:2}"
    ;;

  *)
    echo "ERROR: unknown sweep '$SWEEP' (rlalgo|zoo|zoorep|transfer)" >&2
    exit 2
    ;;
esac

rc=$?
echo
echo "---- array task $IDX of sweep '$SWEEP' exited $rc ----"
exit $rc
