"""Scoring service (tnk1-predict env). POST /score {"smiles":[...]} -> {"R":[...], components..., "beta":b, "step":n}.
Keeps the step counter (REINVENT calls the scorer once per RL step) so beta(t) can be annealed server-side, and
logs all component scores for post-hoc reward-hacking audits. Stdlib http.server => no extra deps."""
from __future__ import annotations
import json, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import numpy as np


def serve(scorer, host="127.0.0.1", port=8765, log_path="runs/rl_scores.jsonl", state_path="runs/rl_step.json"):
    lock = threading.Lock(); Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(state_path).exists():
        scorer.step = json.loads(Path(state_path).read_text()).get("step", 0)

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                res = scorer.score(body["smiles"])
                Path(state_path).write_text(json.dumps({"step": scorer.step}))
                with open(log_path, "a") as fh:
                    fh.write(json.dumps({"step": scorer.step, "t": time.time(), "smiles": body["smiles"],
                                         **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}}) + "\n")
            out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}; out["step"] = scorer.step
            data = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(data)
        def log_message(self, *a): pass

    print(f"scoring server on http://{host}:{port}")
    HTTPServer((host, port), H).serve_forever()
