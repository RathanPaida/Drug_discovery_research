"""Driver for the DAP run.

Always logs the PRIOR BASELINE before training starts. It is the only number that answers
the question the RL stage exists to answer -- did the policy improve, or did we simply draw
more samples? The shipped Cycle-01 result cannot answer it: 12,000 molecules were sampled
once and filtered, so there is no baseline and no trained policy to compare against.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .dap import DAPConfig, train_dap
from .model import SmilesRNN
from .scorer import StandaloneScorer
from .vocab import Vocabulary


def prior_baseline(prior, vocab, scorer, n=2000, device="cpu", seed=42):
    torch.manual_seed(seed)
    seqs, _ = prior.sample_with(vocab, n, device=device)
    smis = [vocab.decode(r) for r in seqs]
    res = scorer(smis)
    sc = np.asarray(res["score"], float)
    valid = np.asarray(res["valid"], bool)
    return {"n": n, "score_mean": float(sc.mean()), "score_max": float(sc.max()),
            "score_p99": float(np.percentile(sc, 99)), "validity": float(valid.mean()),
            "uniqueness": len({s for s in smis if s}) / max(n, 1),
            "n_scaffolds": len({s for s in res.get("scaffold", []) if s})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", default="runs/rl_min")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=120.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--motif", default="c1ncnc(N)c1")
    ap.add_argument("--baseline-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    vocab = Vocabulary.load(a.vocab)
    prior = SmilesRNN.load(a.prior, a.device)
    agent = copy.deepcopy(prior).to(a.device)        # agent starts AS the prior
    scorer = StandaloneScorer(a.motif)

    base = prior_baseline(prior, vocab, scorer, a.baseline_n, a.device, a.seed)
    print(f"[baseline] prior over {base['n']} samples: mean={base['score_mean']:.4f} "
          f"max={base['score_max']:.4f} validity={base['validity']:.3f} "
          f"scaffolds={base['n_scaffolds']}")

    cfg = DAPConfig(steps=a.steps, batch_size=a.batch_size, sigma=a.sigma,
                    lr=a.lr, seed=a.seed)

    def log(m):
        print(f"  step {m['step']:>4} | R {m['score_mean']:.4f} (max {m['score_max']:.3f}) "
              f"| valid {m['validity']:.2f} | uniq {m['uniqueness']:.2f} "
              f"| scaf {m['n_scaffolds']:>3} | loss {m['loss']:9.1f} "
              f"| best {m['replay_best']:.3f}")

    hist, replay = train_dap(prior, agent, vocab, scorer, cfg, a.device,
                             log_every=max(1, a.steps // 30), logger=log)

    agent.save(out / "agent.pt")
    import pandas as pd
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    pd.DataFrame({"smiles": replay.smiles, "score": replay.scores}).to_csv(
        out / "best_molecules.csv", index=False)
    (out / "baseline.json").write_text(json.dumps(base, indent=2))

    first = np.mean([h["score_mean"] for h in hist[:10]])
    last = np.mean([h["score_mean"] for h in hist[-10:]])
    print(f"\n[result] prior baseline {base['score_mean']:.4f} | "
          f"first-10-step mean {first:.4f} | last-10-step mean {last:.4f}")
    verdict = "LEARNED" if last > base["score_mean"] * 1.5 else "NO CLEAR IMPROVEMENT"
    print(f"[result] {verdict}  (agent must beat the prior baseline, not just its own start)")
    print(f"[result] artifacts -> {out}")


if __name__ == "__main__":
    main()
