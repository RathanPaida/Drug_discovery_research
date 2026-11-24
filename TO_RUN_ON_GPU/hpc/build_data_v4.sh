#!/bin/bash
# Build the v4 frozen table: the old 3.95M store UNIONED with the ChEMBL-37 harvest.
#
#   sbatch hpc/build_data_v4.sh
#
# Two steps, in one job because the second consumes the first:
#
#   1. scripts/build_kinome_chembl37.py
#        data/raw/kinome_raw_union.parquet  ->  data/features/kinome_chembl37.parquet
#      Re-aggregates the ChEMBL-37 harvest on (inchikey_block1, uniprot_id,
#      standard_type). The acquisition drop's own curation lost the kinase entirely;
#      see that script's header for the measurement.
#
#   2. scripts/run_stage1_data.py --version v4
#        + data/features/kinome_all_measurements_3.95M.parquet
#        -> data/processed/kinome_long_v4.parquet (+ .sha256, + splits_v4.json)
#      Unions both kinome sources, dedupes across them, freezes the result.
#
# CPU-only: it is RDKit standardisation and pandas, no GPU. It requests no GRES, so it
# holds 0 of the 36-shard budget. It is still submitted as a job rather than run on the
# login node -- 270k tautomer canonicalisations is not login-node work.
#
# Idempotent by refusal, not by overwrite: run_stage1_data.py will not replace an
# existing frozen kinome_long_v4.parquet without --force, because every downstream
# artifact is pinned to its sha256. To rebuild deliberately:
#   TNK1_STAGE1_FORCE=1 sbatch hpc/build_data_v4.sh

#SBATCH --job-name=dug_build_v4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
_cfg=""
for _d in "${SLURM_SUBMIT_DIR:-.}/hpc" "${TNK1_ROOT:-$HOME/TO_RUN_ON_GPU}/hpc" "$(dirname "${BASH_SOURCE[0]}")"; do
    [ -f "$_d/_config.sh" ] && { _cfg="$_d/_config.sh"; break; }
done
[ -n "$_cfg" ] || { echo "ERROR: hpc/_config.sh not found - sbatch from the bundle root, or export TNK1_ROOT" >&2; exit 1; }
# shellcheck disable=SC1090
source "$_cfg"
activate_env

VERSION="${TNK1_VERSION:-v4}"
JOBS="${SLURM_CPUS_PER_TASK:-16}"
RAW="${TNK1_RAW_UNION:-data/raw/kinome_raw_union.parquet}"
OUT="${TNK1_C37_OUT:-data/features/kinome_chembl37.parquet}"

echo "[build] node=$(hostname) cpus=$JOBS version=$VERSION config=$TNK1_CONFIG"
echo "[build] free memory: $(free -g 2>/dev/null | awk '/^Mem:/{print $7" GB"}')"

for f in "$RAW" data/features/kinome_all_measurements_3.95M.parquet; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f is missing. Both kinome sources have to be present:" >&2
        echo "  data/raw/kinome_raw_union.parquet              (ChEMBL-37 harvest, 11.5 MB)" >&2
        echo "  data/features/kinome_all_measurements_3.95M.parquet  (the old store, 195 MB)" >&2
        echo "A missing source would otherwise be a quietly smaller table." >&2
        exit 1
    fi
done

echo
echo "================ 1/2  re-aggregate the ChEMBL-37 harvest ================"
if [ -f "$OUT" ] && [ "${TNK1_C37_FORCE:-0}" != 1 ]; then
    echo "[build] $OUT already exists -- reusing it (TNK1_C37_FORCE=1 to rebuild)"
else
    python scripts/build_kinome_chembl37.py \
        --raw "$RAW" --out "$OUT" --jobs "$JOBS" || exit 1
fi

echo
echo "================ 2/2  union, dedupe and freeze $VERSION ================"
force_arg=()
[ "${TNK1_STAGE1_FORCE:-0}" = 1 ] && force_arg=(--force)
python scripts/run_stage1_data.py --version "$VERSION" "${force_arg[@]}" || exit 1
rc=$?

echo
echo "================ result ================"
ls -la "data/processed/kinome_long_${VERSION}.parquet" \
       "data/processed/kinome_long_${VERSION}.parquet.sha256" \
       "data/processed/splits_${VERSION}.json" 2>&1
echo "sha256: $(cat "data/processed/kinome_long_${VERSION}.parquet.sha256" 2>/dev/null)"
python - "$VERSION" <<'EOF'
import sys, pandas as pd
v = sys.argv[1]
d = pd.read_parquet(f"data/processed/kinome_long_{v}.parquet",
                    columns=["uniprot_id", "source", "censored", "inchikey_block1"])
print(f"\n{len(d):,} rows | {d.inchikey_block1.nunique():,} compounds "
      f"| {d.uniprot_id.nunique()} kinases "
      f"| {int((d.censored==0).sum()):,} exact | {int((d.censored==1).sum()):,} censored")
print("\nrows per source:")
print(d.source.value_counts().to_string())
EOF
exit $rc
