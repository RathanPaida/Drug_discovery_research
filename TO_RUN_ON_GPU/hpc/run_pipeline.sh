#!/bin/bash
# Submit the whole pipeline as a chain of dependent Slurm jobs. Run on the login node:
#
#   bash hpc/run_pipeline.sh smoke          # every stage, tiny, ~30 min -- proves the chain
#   bash hpc/run_pipeline.sh fast           # 400k rows / 5 members / 3 oof  (hpc_fast.yaml)
#   bash hpc/run_pipeline.sh full           # 1.1M rows / 10 members / 5 oof (hpc.yaml)  <- report this
#   bash hpc/run_pipeline.sh full ens       # resume the chain from a stage
#   bash hpc/run_pipeline.sh full --dry-run # print the sbatch lines, queue nothing
#
# Builds the python env first if it is missing (needs internet; this node has it).
# Watch:  squeue -u $USER ;  tail -f logs/${TNK1_JOB_PREFIX}_*.out
# Status: python scripts/run_pipeline.py --version ${TNK1_VERSION:-v3} --plan
#
# smoke writes to runs_smoke/ and processed_smoke/ so it cannot touch real artifacts.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source hpc/_config.sh

MODE="${1:-full}"
FROM="${2:-}"
DRY=0
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; done
[ "${FROM:-}" = "--dry-run" ] && FROM=""

# Default full run now executes the complete isolated campaign in one GPU lane.
if [ "$MODE" = full ] && [ -z "$FROM" ]; then
    export TNK1_CONFIG="$TNK1_ROOT/configs/hpc.yaml"
    if [ "$DRY" = 1 ]; then
        echo "sbatch hpc/submit_campaign.sh (26 models, 12 stackers, 10 fine-tunes, 8 RL algorithms)"
        exit 0
    fi
    if ! tnk1_env_ready; then bash hpc/setup_env.sh || exit 1; fi
    mkdir -p logs
    campaign_opts=()
    [ -n "${TNK1_SBATCH_ARGS:-}" ] && read -r -a campaign_opts <<< "$TNK1_SBATCH_ARGS"
    sbatch "${campaign_opts[@]}" hpc/submit_campaign.sh
    exit $?
fi

STAGES=" preflight baselines zoo stack taskselect transfer ens oof critic calibrate admet prior rl cycle selectivity compare ablate "
case "$MODE" in smoke|fast|full) ;; *) echo "mode must be smoke|fast|full (got '$MODE')" >&2; exit 2 ;; esac
if [ -n "$FROM" ]; then
    case "$STAGES" in *" $FROM "*) ;; *) echo "unknown stage '$FROM' (one of:$STAGES)" >&2; exit 2 ;; esac
fi

case "$MODE" in
    smoke) export TNK1_CONFIG="$TNK1_ROOT/configs/hpc_smoke.yaml" ;;
    fast)  export TNK1_CONFIG="$TNK1_ROOT/configs/hpc_fast.yaml" ;;
    full)  export TNK1_CONFIG="$TNK1_ROOT/configs/hpc.yaml" ;;
esac
[ -f "$TNK1_CONFIG" ] || { echo "ERROR: $TNK1_CONFIG missing" >&2; exit 1; }

# smoke runs in its own tree (configs/hpc_smoke.yaml redirects paths.processed and
# paths.runs) so a toy 2-member ensemble can never be mistaken for the production one.
# The shipped read-only inputs are symlinked rather than copied -- the frozen table alone
# is 38 MB and the quota here is tight.
if [ "$MODE" = smoke ]; then
    mkdir -p data/processed_smoke runs_smoke/pipeline
    for f in kinome_long_v3.parquet kinome_long_v3.parquet.sha256 splits_v3.json \
             kinase_store.npz ssl_encoder.pt; do
        [ -e "data/processed_smoke/$f" ] || ln -s "../processed/$f" "data/processed_smoke/$f"
    done
    # same pre-seeded provenance as the real tree, so smoke also skips data/store/pretrain
    [ -f runs_smoke/pipeline/state.json ] || cp runs/pipeline/state.json runs_smoke/pipeline/state.json
    echo ">> smoke tree: data/processed_smoke (symlinks) + runs_smoke"
fi

# Slurm opens each job's log in logs/ BEFORE the job starts; if it cannot, the job dies
# with no trace at all. Check now, not then.
tnk1_need_writable logs runs runs/pipeline data/processed || exit 1

# ALREADY-QUEUED GUARD.
#
# The lane (see `submit` below) serialises only the jobs of ONE invocation: it tracks ids
# in a temp file created per run. A second `run_pipeline.sh` therefore starts a fresh lane
# whose first job has no dependency on the chain already queued, and the two would run
# together -- 2 x shard:36 = 72 of the node's 94, breaking both caps the lane exists to
# enforce. Refuse instead, and say what to do about it.
if [ "$DRY" != 1 ]; then
    _mine="$(squeue -u "$USER" -h -o "%i %j %t" 2>/dev/null | grep -c "${TNK1_JOB_PREFIX}_" || true)"
    if [ "${_mine:-0}" -gt 0 ]; then
        echo "ERROR: you already have $_mine ${TNK1_JOB_PREFIX}_* job(s) queued or running." >&2
        squeue -u "$USER" -o "  %.7i %.16j %.2t %.8M %.20b" >&2
        echo >&2
        echo "  Submitting another chain would start a second lane, so two of our jobs" >&2
        echo "  could run at once and hold 2 x shard:${TNK1_SHARDS:-36} of the node's 94." >&2
        echo >&2
        echo "  Wait for it, or:" >&2
        echo "    scancel -u $USER                       # drop the existing chain" >&2
        echo "    bash hpc/run_pipeline.sh $MODE <stage>   # resume from a stage instead" >&2
        echo "    TNK1_ALLOW_CONCURRENT=1 bash hpc/run_pipeline.sh $MODE   # override" >&2
        [ "${TNK1_ALLOW_CONCURRENT:-0}" = 1 ] || exit 1
        echo "  TNK1_ALLOW_CONCURRENT=1 set -- proceeding anyway." >&2
    fi
fi

# One-time setup, before anything is queued (skipped once done).
#
# Guarded by DRY. `--dry-run` is advertised as "print the sbatch lines, queue nothing",
# and the `full` path above honours that, but this resume path did not: on any machine
# without a working env -- a laptop, say -- `run_pipeline.sh full ens --dry-run` would
# move envs/tnk1 aside to envs/tnk1.stale.<timestamp> and start a ~6 GB conda build
# before printing a single sbatch line. A dry run must not touch the filesystem.
if [ "$DRY" != 1 ] && ! tnk1_env_ready; then
    echo ">> python env not ready -- building it now (needs internet, ~10 min)"
    bash hpc/setup_env.sh || exit 1
fi
[ "$DRY" = 1 ] || activate_env

if [ "$DRY" != 1 ] && ! tnk1_data_ready; then
    echo "ERROR: the feature store does not verify against data/features/manifest.json." >&2
    echo "  Re-upload data/features/ -- a truncated transfer is the usual cause." >&2
    echo "  Detail:  python scripts/run_stage0_check.py --version ${TNK1_VERSION:-v3} --strict" >&2
    exit 1
fi

echo "=============================================================="
echo " mode      : $MODE"
echo " config    : $TNK1_CONFIG"
# Needs the env's python, so a dry run on a machine without one skips it rather than
# failing on `python` not existing.
if [ "$DRY" != 1 ]; then
python - <<'EOF'
import os, yaml
c = yaml.safe_load(open(os.environ["TNK1_CONFIG"]))
print(f" rows      : {c['data']['max_kinome_rows'] or 'all 1,103,498'}")
print(f" members   : ens={c['uq']['n_members']}  oof={c['uq']['oof_members']}"
      f"  -> {c['uq']['n_members'] + 5 * c['uq']['oof_members']} kinome passes")
EOF
fi
echo " lanes     : ${TNK1_LANES:-${TNK1_GPU_LANES:-1}} job(s) at a time (CPU and GPU alike)"
echo " shards    : max $(( ${TNK1_LANES:-${TNK1_GPU_LANES:-1}} * ${TNK1_SHARDS:-36} )) of 94 held at any instant (shard:${TNK1_SHARDS:-36} per GPU job)"
echo " root      : $TNK1_ROOT"
echo " env       : $ENV_DIR"
echo "=============================================================="
echo

extra_sbatch=()
[ -n "${TNK1_SBATCH_ARGS:-}" ] && read -r -a extra_sbatch <<< "$TNK1_SBATCH_ARGS"

# EXECUTION LANE: ONE JOB AT A TIME, ACROSS THE WHOLE CHAIN.
#
# Two separate caps are being enforced here, and they are not the same thing:
#
#   1. At most TNK1_SHARDS (36) GPU shards are held at any instant.
#   2. At most TNK1_LANES (1) of OUR jobs is running at any instant.
#
# (1) alone is not enough, and (2) alone is not enough either. This lane used to apply
# to GPU jobs only, which left the CPU stages -- preflight, baselines, zoo, stack,
# critic, taskselect -- free to run alongside a GPU stage, because they request no GRES
# and so never queued behind it. That is still several of our jobs resident at once on a
# node we share, each holding 16 CPUs of the node's 128. With the lane now covering
# every job regardless of kind, the chain is strictly serial: exactly one dug_* job runs
# at a time, and it may hold the entire 36-shard budget.
#
# The cost is real. `ens`, `admet`, `prior` and `ablate` are all eligible after
# preflight and would previously have overlapped; now they queue. The submission order
# below is therefore the schedule, and it is ordered critical-path-first for that reason.
#
# Implemented by making the Nth job wait on the (N - lanes)th. The ids live in a FIFO
# FILE rather than a shell variable because `submit` is called inside $( ), which runs in
# a subshell -- a variable assigned in there would not survive the call.
#
# The lane dependency is `afterany`, not `afterok`: it exists only to ORDER jobs, so a
# failed stage must not cascade-cancel every later stage that does not actually depend on
# it. The genuine prerequisites stay in the `afterok` clause, and Slurm ANDs the two with
# a comma.
LANES="${TNK1_LANES:-${TNK1_GPU_LANES:-1}}"
SHARDS="${TNK1_SHARDS:-36}"
if [ "$SHARDS" -gt 36 ] 2>/dev/null; then
    echo "ERROR: TNK1_SHARDS=$SHARDS exceeds the 36-shard budget for this project." >&2
    echo "  36 is the agreed ceiling on shards held at any instant. Lower it, or if the" >&2
    echo "  allocation really did change, change the ceiling in hpc/run_pipeline.sh and" >&2
    echo "  the #SBATCH defaults in hpc/submit_gpu.sh and hpc/submit_campaign.sh too." >&2
    exit 1
fi
LANE_FIFO="$(mktemp)"
trap 'rm -f "$LANE_FIFO"' EXIT

submit() {  # submit <cpu|gpu> <name> <deps> <stage>
    local kind="$1" name="$2" deps="$3" stage="$4"
    local opts=(--parsable --job-name="${TNK1_JOB_PREFIX}_$name" ${extra_sbatch[@]+"${extra_sbatch[@]}"})
    # Every GPU job is pinned to the shard budget from one variable, so the cap cannot
    # drift between here and the #SBATCH default inside submit_gpu.sh.
    [ "$kind" = gpu ] && opts+=(--gres="shard:$SHARDS")
    # Drop empty dependency ids (stages skipped by the resume argument).
    local clean="" id
    for id in ${deps//:/ }; do [ -n "$id" ] && clean="$clean:$id"; done
    local dep="" lane_prev="" n
    [ -n "$clean" ] && dep="afterok$clean"
    if [ "$LANES" -ge 1 ]; then
        n=$(wc -l < "$LANE_FIFO" 2>/dev/null || echo 0)
        if [ "$n" -ge "$LANES" ]; then
            lane_prev="$(sed -n "$((n - LANES + 1))p" "$LANE_FIFO")"
            [ -n "$lane_prev" ] && dep="${dep:+$dep,}afterany:$lane_prev"
        fi
    fi
    [ -n "$dep" ] && opts+=(--dependency="$dep" --kill-on-invalid-dep=yes)
    if [ "$DRY" = 1 ]; then
        echo "sbatch ${opts[*]} hpc/submit_$kind.sh $stage" >&2
        echo "DRY_$name" >> "$LANE_FIFO"
        echo "DRY_$name"; return 0
    fi
    local jid
    jid="$(sbatch "${opts[@]}" "hpc/submit_$kind.sh" "$stage")" || {
        echo "submit failed: $name" >&2; exit 1; }
    echo "$jid" >> "$LANE_FIFO"
    printf '  %-13s job %-9s %s%s\n' "$stage" "$jid" "${clean:+after$clean}" \
           "${lane_prev:+ [lane after $lane_prev]}" >&2
    echo "$jid"
}

started=0
[ -z "$FROM" ] && started=1
want() {  # true from $FROM onwards
    if [ "$started" = 1 ] || [ "$1" = "$FROM" ]; then started=1; return 0; fi
    return 1
}

# ---- the DAG -------------------------------------------------------------------
# preflight ─┬─ baselines   (CPU)
#            ├─ ablate      (A0-A7 x 5 seeds: the architecture-justification table)
#            ├─ prior       (version-independent: reads only the 788k corpus)
#            ├─ admet       (independent: 7 endpoints from admet_7endpoints.parquet)
#            └─ ens ─┬─ selectivity
#                    └─ oof ─┬─ critic     (CPU: paired baselines vs the ensemble)
#                            └─ calibrate ─┬─ compare
#                                          └─ rl ── cycle
#                                             (rl also needs prior + admet)
#
# `ablate` and `cycle` were previously left out of the auto-submit as "optional and
# expensive". They are in now because each answers a question a reviewer asks first:
#   ablate  does the pan-kinome transfer stack beat a scratch D-MPNN at all? Without the
#           A0-A7 ladder the architecture has no counterfactual. Wired since v1, never run.
#   critic  is the deep ensemble -- which supplies r_pot and r_sel, 5.5 of 12.5 reward
#           weight units -- actually better than a random forest? On v1 it was not
#           (Spearman 0.264 vs 0.641) and that was measured on ONE split.
#   cycle   turns the generator's output into a selected compound set. Without it the run
#           ends at "a trained agent" rather than "these molecules".
# SUBMISSION ORDER MATTERS NOW. With TNK1_LANES=1 every job -- CPU and GPU alike -- runs
# strictly in the order it is submitted, so this sequence is the schedule. The critical path to a
# result -- ens -> oof -> calibrate -> admet -> prior -> rl -> cycle -- goes first.
#
# `ablate` is submitted LAST despite having no prerequisite beyond preflight: it is
# A0-A7 x 5 seeds, six of the eight variants calling kinome_stage, so roughly 30 further
# full kinome passes. Submitted early (as it was) it would have occupied the single lane
# and delayed every other stage behind it. It is a side analysis that justifies the
# architecture; it is not on the path to molecules.
#
# `selectivity` and `compare` are likewise moved after `rl`: both are cheap readouts that
# nothing else depends on.
PRE=""; BAS=""; ZOO=""; ENS=""; OOF=""; CAL=""; ADM=""; PRI=""; RL=""
want preflight   && PRE=$(submit cpu preflight   ""                 preflight)
want baselines   && BAS=$(submit cpu baselines   "$PRE"             baselines)
# zoo and stack are CPU-only and cost minutes, so they sit outside the GPU lane entirely
# and cannot delay the critical path. They are on it in the LOGICAL sense, though: `stack`
# produces data/processed/stack.pkl, which is the defensible critic the RL stages should
# be optimised against instead of the deep ensemble that lost the paired comparison
# (Spearman 0.501 vs the random forest's 0.686, per-row p = 8.6e-4).
want zoo         && ZOO=$(submit cpu zoo         "$PRE"             zoo)
# taskselect is seconds of pandas: it answers "which of the 491 donor kinases
# actually share SAR with TNK1", which changes what the kinome stage should be
# pretrained on. Cheap enough to run every time.
want taskselect  &&       submit cpu taskselect  "$PRE"             taskselect >/dev/null
want ens         && ENS=$(submit gpu ens         "$PRE"             ens)
want oof         && OOF=$(submit gpu oof         "$ENS"             oof)
want critic      &&       submit cpu critic      "$OOF"             critic >/dev/null
# stack wants `oof` too, for the DeepEnsemble candidate column, but runs without it
# and says so in its log -- so a missing oof degrades the comparison rather than
# blocking the stage.
want stack       &&       submit cpu stack       "$ZOO:$OOF"        stack >/dev/null
want calibrate   && CAL=$(submit gpu calibrate   "$OOF"             calibrate)
want admet       && ADM=$(submit gpu admet       "$PRE"             admet)
want prior       && PRI=$(submit gpu prior       "$PRE"             prior)
want rl          && RL=$(submit  gpu rl          "$CAL:$ADM:$PRI"   rl)
want cycle       &&       submit gpu cycle       "$RL"              cycle >/dev/null
want selectivity &&       submit gpu selectivity "$ENS"             selectivity >/dev/null
want compare     &&       submit gpu compare     "$CAL"             compare >/dev/null
want ablate      &&       submit gpu ablate      "$PRE"             ablate >/dev/null
# `transfer` is 10 fine-tuning strategies x 5 folds, each needing its own kinome
# trunk -- comparable in cost to `ablate`, and like `ablate` it justifies an
# architecture choice rather than producing molecules. Submitted last, and better
# run as an array:  sbatch --array=0-9%2 hpc/submit_array.sh transfer
want transfer    &&       submit gpu transfer    "$PRE"             transfer >/dev/null

echo
if [ "$DRY" = 1 ]; then
    echo "dry run -- nothing queued"
else
    echo "Queued. Watch:"
    echo "  squeue -u \$USER -o '%.8i %.16j %.2t %.11M %.6D %R'"
    echo "  tail -f logs/${TNK1_JOB_PREFIX}_*.out"
    echo "Status:  python scripts/run_pipeline.py --version ${TNK1_VERSION:-v3} --plan"
    echo "Cancel:  scancel -u \$USER"
fi
