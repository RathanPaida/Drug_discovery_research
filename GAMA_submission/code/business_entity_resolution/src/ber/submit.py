"""Step `validate`: run the organisers' validator (utils/validate_submission.py)
on output/matching_results.tsv + output/candidate_pairs.tsv."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .util import log


def validate(cfg, check_ids: bool = False) -> int:
    script = Path(cfg.paths.validator)
    out = Path(cfg.paths.output_dir)
    if not script.exists():
        log(f"validator not found at {script} - skipped")
        return 0
    cmd = [sys.executable, str(script), "--matching", str(out / "matching_results.tsv"),
           "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(Path(cfg.paths.data_dir) / "test")]
    if check_ids:
        cmd.append("--check-ids")
    log("validator: " + " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(script.parent.parent), capture_output=True, text=True)
    print(res.stdout, flush=True)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr, flush=True)
        log("validator FAILED - fix the issues above before submitting")
    return res.returncode


def run(cfg, args) -> None:
    code = validate(cfg, check_ids=True)
    if code:
        raise SystemExit(code)
