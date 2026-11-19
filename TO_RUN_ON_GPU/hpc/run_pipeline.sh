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
# Status: python scripts/run_pipeline.py --version v3 --plan
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

STAGES=" preflight baselines ablate ens oof calibrate compare critic admet selectivity prior rl cycle "
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

# One-time setup, before anything is queued (skipped once done).
if ! tnk1_env_ready; then
    echo ">> python env not ready -- building it now (needs internet, ~10 min)"
    bash hpc/setup_env.sh || exit 1
fi
activate_env

if ! tnk1_data_ready; then
    echo "ERROR: the feature store does not verify against data/features/manifest.json." >&2
    echo "  Re-upload data/features/ -- a truncated transfer is the usual cause." >&2
    echo "  Detail:  python scripts/run_stage0_check.py --version v3 --strict" >&2
    exit 1
fi

echo "=============================================================="
echo " mode      : $MODE"
echo " config    : $TNK1_CONFIG"
python - <<'EOF'
import os, yaml
c = yaml.safe_load(open(os.environ["TNK1_CONFIG"]))
print(f" rows      : {c['data']['max_kinome_rows'] or 'all 1,103,498'}")
print(f" members   : ens={c['uq']['n_members']}  oof={c['uq']['oof_members']}"
      f"  -> {c['uq']['n_members'] + 5 * c['uq']['oof_members']} kinome passes")
EOF
echo " root      : $TNK1_ROOT"
echo " env       : $ENV_DIR"
echo "=============================================================="
echo

extra_sbatch=()
[ -n "${TNK1_SBATCH_ARGS:-}" ] && read -r -a extra_sbatch <<< "$TNK1_SBATCH_ARGS"

submit() {  # submit <cpu|gpu> <name> <deps> <stage>
    local kind="$1" name="$2" deps="$3" stage="$4"
    local opts=(--parsable --job-name="${TNK1_JOB_PREFIX}_$name" ${extra_sbatch[@]+"${extra_sbatch[@]}"})
    # Drop empty dependency ids (stages skipped by the resume argument).
    local clean="" id
    for id in ${deps//:/ }; do [ -n "$id" ] && clean="$clean:$id"; done
    [ -n "$clean" ] && opts+=(--dependency="afterok$clean" --kill-on-invalid-dep=yes)
    if [ "$DRY" = 1 ]; then
        echo "sbatch ${opts[*]} hpc/submit_$kind.sh $stage" >&2
        echo "DRY_$name"; return 0
    fi
    local jid
    jid="$(sbatch "${opts[@]}" "hpc/submit_$kind.sh" "$stage")" || {
        echo "submit failed: $name" >&2; exit 1; }
    printf '  %-13s job %-9s %s\n' "$stage" "$jid" "${clean:+after$clean}" >&2
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
PRE=""; BAS=""; ENS=""; OOF=""; CAL=""; ADM=""; PRI=""; RL=""
want preflight   && PRE=$(submit cpu preflight   ""                 preflight)
want baselines   && BAS=$(submit cpu baselines   "$PRE"             baselines)
want ablate      &&       submit gpu ablate      "$PRE"             ablate >/dev/null
want ens         && ENS=$(submit gpu ens         "$PRE"             ens)
want oof         && OOF=$(submit gpu oof         "$ENS"             oof)
want critic      &&       submit cpu critic      "$OOF"             critic >/dev/null
want calibrate   && CAL=$(submit gpu calibrate   "$OOF"             calibrate)
want compare     &&       submit gpu compare     "$CAL"             compare >/dev/null
want admet       && ADM=$(submit gpu admet       "$PRE"             admet)
want selectivity &&       submit gpu selectivity "$ENS"             selectivity >/dev/null
want prior       && PRI=$(submit gpu prior       "$PRE"             prior)
want rl          && RL=$(submit  gpu rl          "$CAL:$ADM:$PRI"   rl)
want cycle       &&       submit gpu cycle       "$RL"              cycle >/dev/null

echo
if [ "$DRY" = 1 ]; then
    echo "dry run -- nothing queued"
else
    echo "Queued. Watch:"
    echo "  squeue -u \$USER -o '%.8i %.16j %.2t %.11M %.6D %R'"
    echo "  tail -f logs/${TNK1_JOB_PREFIX}_*.out"
    echo "Status:  python scripts/run_pipeline.py --version v3 --plan"
    echo "Cancel:  scancel -u \$USER"
fi
