"""`python -m tnk1.utils.verify_features` -- sha256 the feature store against its manifest.

Exit codes: 0 every requested file verified; 1 something is absent or corrupt.

Use this before quoting any number that depends on the precomputed stores. The store is
gitignored (~280 MB), so a fresh checkout legitimately reports most files as `absent`;
that is a signal the heavy stages cannot be reproduced here, not a pass.
"""
from __future__ import annotations

import argparse
import sys

from .paths import features_dir, verify_features


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", help="specific filenames (default: all in manifest)")
    ap.add_argument("--dir", default=None, help="feature store dir (default: $TNK1_FEATURES_DIR or repo layout)")
    args = ap.parse_args(argv)

    d = args.dir or features_dir()
    try:
        status = verify_features(args.names or None, directory=d)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    width = max((len(k) for k in status), default=0)
    ok = absent = bad = 0
    for name, state in sorted(status.items()):
        mark = {"ok": "OK    ", "absent": "ABSENT"}.get(state, "FAIL  ")
        print(f"  [{mark}] {name:<{width}}  {'' if state in ('ok', 'absent') else state}")
        ok += state == "ok"
        absent += state == "absent"
        bad += state not in ("ok", "absent")

    print(f"\n{d}\n  verified={ok}  absent={absent}  corrupt={bad}  (of {len(status)})")
    if bad:
        print("  -> CORRUPT files present: do not report results computed from this store.")
    elif absent:
        print("  -> Files missing (they are gitignored); stages depending on them cannot run.")
    return 0 if (bad == 0 and absent == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
