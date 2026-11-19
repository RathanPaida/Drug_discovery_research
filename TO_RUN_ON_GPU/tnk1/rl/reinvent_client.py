"""REINVENT 4 ExternalProcess client (runs in the tnk1-reinvent env; stdlib only).

REINVENT pipes SMILES (one per line) to stdin and expects JSON on stdout:
    {"version": 1, "payload": {"predictions": [score, ...]}}

WHY THERE IS A `main()` HERE. This module's body used to run at import: it called
`argparse.parse_args()`, read stdin to EOF and issued an HTTP request, all as a side
effect of `import tnk1.rl.reinvent_client`. Anything that walks the package -- pytest
collection, a docs build, an IDE indexer, `pkgutil.walk_packages` -- therefore tried to
POST to 127.0.0.1:8765 and died with URLError, or hung waiting on stdin. It also meant
the module could never be imported to be tested. The executable part now lives behind
`if __name__ == "__main__"`, which is the only form safe for a script that sits inside an
importable package.

Locate this file with `client_path()` rather than a literal relative path: REINVENT is
handed an absolute path to run under a different interpreter, and
`Path("tnk1/rl/reinvent_client.py")` only resolved while the caller's CWD happened to be
the repo root.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def client_path() -> Path:
    """Absolute path to this file, for embedding in a REINVENT TOML."""
    return Path(__file__).resolve()


def score(url: str, smiles: list[str], key: str = "R", timeout: float = 600.0) -> list[float]:
    """POST `smiles` to the scoring server and return the `key` column."""
    endpoint = url if url.endswith("/score") else url.rstrip("/") + "/score"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps({"smiles": smiles}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        res = json.loads(fh.read())
    if key not in res:
        raise KeyError(f"scoring server returned no '{key}' column; got {sorted(res)}")
    return list(res[key])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8765/score")
    ap.add_argument("--key", default="R")
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args(argv)
    smiles = [l.strip() for l in sys.stdin if l.strip()]
    preds = score(a.url, smiles, a.key, a.timeout)
    print(json.dumps({"version": 1, "payload": {"predictions": preds}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
