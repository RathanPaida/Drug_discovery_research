#!/bin/bash
# Prove the two resource caps actually hold on the queue right now.
#
#   bash hpc/verify_lane.sh
#
# Checks, and why each one exists:
#
#   1. At most one of our jobs is RUNNING.
#   2. No queued GPU job asks for more than TNK1_SHARDS (36).
#   3. Every pending job has a lane predecessor (an `afterany:` dependency).
#   4. No two jobs wait on the same predecessor.
#   5. Walking the lane from the running job reaches every queued job.
#
# (3)-(5) are not paranoia. The lane is built out of `afterany:<previous>` dependencies,
# and Slurm DROPS a dependency on a job that can never run. So when one stage fails and
# `--kill-on-invalid-dep=yes` cancels the stage behind it, the cancelled job takes the
# lane edge with it and the chain silently forks at that point.
#
# That is not hypothetical. On the v4 run, `zoo` failed on three missing dependencies,
# `stack` was cancelled because it needed zoo, and `calibrate` -- whose lane edge was
# `afterany:<stack>` -- was left with only its `afterok:<oof>` prerequisite. Once `oof`
# finished, `critic` and `calibrate` would both have become eligible and run at the same
# time, holding 36 shards and 16 extra CPUs between them. Nothing would have reported it:
# both jobs would have succeeded.
#
# Repair a broken edge by hand, pointing the orphan at the job that should precede it:
#   scontrol update job=<orphan> Dependency=afterok:<real prereqs>,afterany:<predecessor>
#
# This cluster has `Slurm accounting storage is disabled`, so sacct cannot be used to
# check any of it -- hence squeue plus the per-job logs.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source hpc/_config.sh

SHARDS="${TNK1_SHARDS:-36}"
Q="$(squeue -u "$USER" -h -o '%i|%j|%T|%E|%b' 2>/dev/null)"
if [ -z "$Q" ]; then
    echo "no ${TNK1_JOB_PREFIX}_* jobs queued -- nothing to check"
    exit 0
fi

fail=0
n_total="$(printf '%s\n' "$Q" | wc -l)"
running="$(printf '%s\n' "$Q" | awk -F'|' '$3=="RUNNING"{print $1}')"
n_run="$(printf '%s' "$running" | grep -c . || true)"

echo "queued: $n_total   running: ${n_run:-0} [${running//$'\n'/ }]"

# 1. one job at a time
if [ "${n_run:-0}" -gt 1 ]; then
    echo "FAIL: ${n_run} of our jobs are running at once; the cap is 1" >&2
    fail=1
fi

# 2. shard ceiling
over="$(printf '%s\n' "$Q" | awk -F'|' '{print $5}' | grep -oE 'shard:[0-9]+' \
        | cut -d: -f2 | awk -v c="$SHARDS" '$1>c' | sort -u || true)"
if [ -n "$over" ]; then
    echo "FAIL: shard requests above the ${SHARDS} cap: $over" >&2
    fail=1
fi

# 3. every pending job has a lane predecessor
orphans=""
while IFS='|' read -r jid name state dep gres; do
    [ "$state" = "RUNNING" ] && continue
    case "$dep" in *afterany:*) ;; *) orphans="$orphans $jid($name)" ;; esac
done <<< "$Q"
if [ -n "$orphans" ]; then
    echo "FAIL: pending job(s) with no lane predecessor -- these can start alongside" >&2
    echo "      another job:$orphans" >&2
    echo "      Repair: scontrol update job=<id> Dependency=afterok:<prereqs>,afterany:<predecessor>" >&2
    fail=1
fi

# 4. no two jobs wait on the same predecessor
forks="$(printf '%s\n' "$Q" | awk -F'|' '{print $4}' \
         | grep -oE 'afterany:[0-9]+' | cut -d: -f2 | sort | uniq -d || true)"
if [ -n "$forks" ]; then
    echo "FAIL: more than one job waits on each of: $forks (the lane forks there)" >&2
    fail=1
fi

# 5. the walk reaches everything
if [ "${n_run:-0}" -eq 1 ]; then
    reached=1; cur="$running"
    while :; do
        nxt="$(printf '%s\n' "$Q" | awk -F'|' -v p="afterany:$cur" '$4 ~ p {print $1; exit}')"
        [ -z "$nxt" ] && break
        reached=$((reached + 1)); cur="$nxt"
    done
    echo "lane walk from $running reaches $reached of $n_total queued job(s)"
    if [ "$reached" -ne "$n_total" ]; then
        echo "FAIL: $((n_total - reached)) job(s) are not on the chain" >&2
        fail=1
    fi
fi

if [ "$fail" = 0 ]; then
    echo "OK: one job at a time, <= ${SHARDS} shards, single unbroken lane"
else
    echo "the queue does NOT honour the caps -- see the FAIL lines above" >&2
fi
exit "$fail"
