#!/bin/bash
# Build <team>_submission.zip in $BER_ROOT with the layout the organisers require:
#   output/{matching_results.tsv,candidate_pairs.tsv}
#   code/business_entity_resolution/{src/,README.md,requirements.txt,...}
#   Documentation_template.md
# Usage (login node, after `stack` finished):  bash hpc/package_submission.sh <team_name>

set -euo pipefail
TEAM="${1:?usage: bash hpc/package_submission.sh <team_name>}"
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_here/_config.sh"
activate_env

cd "$BER_ROOT"
for f in output/matching_results.tsv output/candidate_pairs.tsv Documentation_template.md; do
    [ -f "$f" ] || { echo "ERROR: missing $BER_ROOT/$f" >&2; exit 1; }
done
python -m ber validate
python - "$TEAM" <<'PY'
import sys, zipfile
from pathlib import Path
team = sys.argv[1]
root = Path(".")
skip_dirs = {"logs", "__pycache__", ".pytest_cache"}
out = root / f"{team}_submission.zip"
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for f in ("output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md"):
        z.write(root / f, f)
    code = root / "code" / "business_entity_resolution"
    for p in sorted(code.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and not skip_dirs.intersection(rel.parts) and p.suffix != ".pyc":
            z.write(p, str(rel))
print(f"wrote {out.resolve()}  ({out.stat().st_size/2**20:.1f} MB)")
PY
