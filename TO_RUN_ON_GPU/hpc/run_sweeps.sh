#!/bin/bash
# Submit the Cycle-03 sweeps: the model zoo, the stacked critic, and the RL algorithms.
# Run on the login node.
#
#   bash hpc/run_sweeps.sh models            # zoo (CPU) -> stack (CPU).  ~1-2 h, no GPU
#   bash hpc/run_sweeps.sh reps              # representation sweep, 6 CPU array tasks
#   bash hpc/run_sweeps.sh rl                # 8 RL objectives, GPU array, 2 concurrent
#   bash hpc/run_sweeps.sh all               # models -> reps -> rl, chained
#   bash hpc/run_sweeps.sh rl --dry-run      # print the sbatch lines, queue nothing
#
# WHY THIS IS SEPARATE FROM hpc/run_pipeline.sh
#
# run_pipeline.sh submits ONE dependency chain through a single GPU lane, because the
# production stages must run in order and together they would otherwise hold 48-64 of the
# node's 94 shards. The sweeps have the opposite structure: N independent runs that share
# no state. Forcing them through that lane would serialise 8 RL runs that could overlap,
# and putting them in the same chain would make an unrelated failure cancel them.
#
# So: Slurm job ARRAYS with a `%` concurrency cap, which Slurm enforces itself. The GPU
# cap defaults to 2 concurrent tasks x shard:16 = 32 shards, which is deliberately higher
# than run_pipeline.sh's single lane because a sweep is a one-off and the node is
# otherwise idle while it runs. Lower it with TNK1_ARRAY_GPU_CONCURRENCY=1 if someone
# else is using the machine.
#
# READ THE RESULTS WITH:
#   python scripts/compare_sweeps.py                    # the summary table
#   cat runs/stage2b_zoo_summary.csv
#   cat runs/stage4b_stack.json | python -m json.tool | head -40

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source hpc/_config.sh

WHAT="${1:-models}"
DRY=0
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; done

case "$WHAT" in models|reps|rl|stack|all) ;;
  *) echo "usage: bash hpc/run_sweeps.sh <models|reps|rl|stack|all> [--dry-run]" >&2; exit 2 ;;
esac

GPU_CONC="${TNK1_ARRAY_GPU_CONCURRENCY:-2}"
CPU_CONC="${TNK1_ARRAY_CPU_CONCURRENCY:-4}"
VERSION="${TNK1_VERSION:-v3}"
export TNK1_VERSION="$VERSION"

tnk1_need_writable logs runs runs/pipeline data/processed || exit 1

if ! tnk1_env_ready; then
    echo ">> python env not ready -- building it now (needs internet, ~10 min)"
    bash hpc/setup_env.sh || exit 1
fi
activate_env

# The sweeps add dependencies the production env does not have. Report which models the
# sweep will actually be able to run BEFORE queueing anything: a sweep that silently drops
# 6 of 26 models produces a results table whose absences look like failures.
echo "=============================================================="
echo " sweep     : $WHAT"
echo " config    : $TNK1_CONFIG"
echo " version   : $VERSION"
echo " root      : $TNK1_ROOT"
python - <<'EOF'
import sys
sys.path.insert(0, ".")
try:
    import tnk1.zoo as Z
    av = Z.available()
    ok = [k for k, v in av.items() if v == "ok"]
    bad = {k: v for k, v in av.items() if v != "ok"}
    print(f" zoo       : {len(ok)}/{len(av)} models runnable in this env")
    if bad:
        print("             MISSING, and therefore excluded from the sweep:")
        for k, v in bad.items():
            print(f"               {k:22} {v}")
        print("             install with:  pip install lightgbm catboost ngboost")
except Exception as e:
    print(f" zoo       : could not import tnk1.zoo ({e})")
EOF
echo "=============================================================="
echo

extra_sbatch=()
[ -n "${TNK1_SBATCH_ARGS:-}" ] && read -r -a extra_sbatch <<< "$TNK1_SBATCH_ARGS"

submit_one() {  # submit_one <cpu|gpu> <name> <deps> <stage...>
    local kind="$1" name="$2" deps="$3"; shift 3
    local opts=(--parsable --job-name="${TNK1_JOB_PREFIX}_$name" ${extra_sbatch[@]+"${extra_sbatch[@]}"})
    local clean="" id
    for id in ${deps//:/ }; do [ -n "$id" ] && clean="$clean:$id"; done
    [ -n "$clean" ] && opts+=(--dependency="afterok$clean" --kill-on-invalid-dep=yes)
    if [ "$DRY" = 1 ]; then
        echo "sbatch ${opts[*]} hpc/submit_$kind.sh $*" >&2
        echo "DRY_$name"; return 0
    fi
    local jid
    jid="$(sbatch "${opts[@]}" "hpc/submit_$kind.sh" "$@")" || { echo "submit failed: $name" >&2; exit 1; }
    printf '  %-16s job %-9s %s\n' "$name" "$jid" "${clean:+after$clean}" >&2
    echo "$jid"
}

submit_array() {  # submit_array <cpu|gpu> <name> <deps> <range%conc> <sweep>
    local kind="$1" name="$2" deps="$3" range="$4" sweep="$5"
    local opts=(--parsable --job-name="${TNK1_JOB_PREFIX}_$name" --array="$range"
                ${extra_sbatch[@]+"${extra_sbatch[@]}"})
    [ "$kind" = gpu ] && opts+=(--gres=shard:16)
    local clean="" id
    for id in ${deps//:/ }; do [ -n "$id" ] && clean="$clean:$id"; done
    [ -n "$clean" ] && opts+=(--dependency="afterok$clean" --kill-on-invalid-dep=yes)
    if [ "$DRY" = 1 ]; then
        echo "sbatch ${opts[*]} hpc/submit_array.sh $sweep" >&2
        echo "DRY_$name"; return 0
    fi
    local jid
    jid="$(sbatch "${opts[@]}" hpc/submit_array.sh "$sweep")" || { echo "submit failed: $name" >&2; exit 1; }
    printf '  %-16s job %-9s array %-10s %s\n' "$name" "$jid" "$range" "${clean:+after$clean}" >&2
    echo "$jid"
}

ZOO=""; REPS=""; STACK=""
echo "Queued:"
case "$WHAT" in
  models|all)
    # One CPU job, not an array: the `core` shortlist is ~17 runnable models x nested CV on
    # 172 molecules, which is minutes, and keeping it as one job keeps one summary CSV.
    ZOO=$(submit_one cpu zoo "" zoo)
    STACK=$(submit_one cpu stack "$ZOO" stack)
    ;;
esac
case "$WHAT" in
  reps|all)
    REPS=$(submit_array cpu zoorep "" "0-5%${CPU_CONC}" zoorep)
    ;;
esac
case "$WHAT" in
  stack)
    STACK=$(submit_one cpu stack "" stack)
    ;;
esac
case "$WHAT" in
  rl|all)
    # Depends on `stack` only when both are in this invocation: the RL runs are more
    # informative against the stacked critic than against the deep ensemble that lost the
    # paired comparison, but they will run against whatever critic the config selects.
    submit_array gpu rlalgo "$STACK" "0-7%${GPU_CONC}" rlalgo >/dev/null
    ;;
esac

echo
if [ "$DRY" = 1 ]; then
    echo "dry run -- nothing queued"
else
    echo "Watch:   squeue -u \$USER -o '%.10i %.16j %.2t %.11M %R'"
    echo "         tail -f logs/${TNK1_JOB_PREFIX}_*.out"
    echo "Compare: python scripts/compare_sweeps.py"
    echo "Cancel:  scancel -u \$USER"
fi
