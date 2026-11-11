"""REINVENT 4 ExternalProcess client (runs in the tnk1-reinvent env; stdlib only).
REINVENT pipes SMILES (one per line) to stdin and expects JSON on stdout:
    {"version": 1, "payload": {"predictions": [score, ...]}}"""
import argparse, json, sys, urllib.request

ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8765/score"); ap.add_argument("--key", default="R")
a = ap.parse_args()
smiles = [l.strip() for l in sys.stdin if l.strip()]
req = urllib.request.Request(a.url if a.url.endswith("/score") else a.url.rstrip("/") + "/score",
                             data=json.dumps({"smiles": smiles}).encode(), headers={"Content-Type": "application/json"})
res = json.loads(urllib.request.urlopen(req, timeout=600).read())
print(json.dumps({"version": 1, "payload": {"predictions": res[a.key]}}))
