"""Driver for the DAP run.

Always logs the PRIOR BASELINE before training starts. It is the only number that answers
the question the RL stage exists to answer -- did the policy improve, or did we simply draw
more samples? The shipped Cycle-01 result cannot answer it: 12,000 molecules were sampled
once and filtered, so there is no baseline and no trained policy to compare against.

Two rewards, selected with `--scorer`:

  standalone  QED x MW-window x motif. No feature store, no checkpoints. Use it to prove
              the RL machinery can move the output distribution at all. The 7.8x reward
              improvement reported in CHANGES_V2.md was measured HERE, on this toy
              objective -- it says the optimiser works, nothing about pharmacology.

  pipeline    The production reward: potency, selectivity, applicability domain, SA,
              ADMET/safety and hinge, through `tnk1.rl.scorer.Scorer`. Needs the feature
              store and every Stage 3-5 checkpoint. `rl_min.scorer.PipelineScorer` existed
              and was exported from day one but NOTHING EVER INSTANTIATED IT, so until now
              no generator had ever been trained against the real objective.

    python -m rl_min.run_rl --prior priors/kinase_788k.pt --vocab priors/kinase_788k.vocab.json
    python -m rl_min.run_rl --prior ... --vocab ... --scorer pipeline --steps 1000

Calibration is not optional in pipeline mode. `Scorer.calibrate()` re-centres the potency
and selectivity sigmoids on this generator's own output; with the Cycle-01 literals
(pot_c=7.0, sel_c=1.2) r_sel averages 0.0089 because sel_c sits above the entire pool's
maximum mu_sel. The driver calls it, on a fresh prior sample, before step 1.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .dap import DAPConfig, train_dap
from .model import SmilesRNN
from .scorer import PipelineScorer, StandaloneScorer
from .vocab import Vocabulary


def prior_baseline(prior, vocab, scorer, n=2000, device="cpu", seed=42, batch=512):
    """Score `n` fresh prior samples. Sampling is chunked so n=10000 does not allocate a
    single (10000, max_len) tensor on a 6-8 GB card."""
    torch.manual_seed(seed)
    smis: list[str] = []
    while len(smis) < n:
        k = min(batch, n - len(smis))
        seqs, _ = prior.sample_with(vocab, k, device=device)
        smis.extend(vocab.decode(r) for r in seqs)
    res = scorer(smis)
    sc = np.asarray(res["score"], float)
    valid = np.asarray(res["valid"], bool)
    return {"n": len(smis), "score_mean": float(sc.mean()), "score_max": float(sc.max()),
            "score_p99": float(np.percentile(sc, 99)), "validity": float(valid.mean()),
            "uniqueness": len({s for s in smis if s}) / max(len(smis), 1),
            "n_scaffolds": len({s for s in res.get("scaffold", []) if s}),
            "smiles": smis}


def _cfg_path_for_meta(config):
    """Which config file the run used, for the provenance record in meta.json."""
    from tnk1.utils.common import load_config
    try:
        return load_config(config).get("_config_path", str(config))
    except Exception:
        return str(config)


def build_pipeline_scorer(config, version, device, calibrate_n=None, cache="step"):
    """Assemble the production `Scorer` from the Stage 3-5 artifacts.

    Mirrors `scripts/run_stage7_rl.py serve` and `run_stage8_cycle.load_stack()` so the
    three entry points cannot drift apart -- that drift is defect #6 in PIPELINE_AUDIT.md,
    where the offline path silently omitted the hERG argument to `hard_gate`.
    """
    import pickle

    from tnk1.baselines.features import morgan_count
    from tnk1.context import Ctx
    from tnk1.rl.scorer import Scorer
    from tnk1.uq.ensemble import Ensemble

    C = Ctx(config, version, device)
    proc = C.proc
    missing = [str(p) for p in (proc / "calibrator.pkl", proc / "ad.pkl",
                                proc / "admet_bundles.pt", proc / "kinase_store.npz")
               if not p.exists()]
    if not any((proc / "ensemble").glob("member_*.pt")):
        missing.append(str(proc / "ensemble" / "member_*.pt"))
    if missing:
        raise FileNotFoundError(
            "pipeline scorer needs the Stage 3-5 artifacts; missing:\n  "
            + "\n  ".join(missing)
            + "\nRun `python scripts/run_stage0_check.py --require rl` for the full report."
        )

    ens = Ensemble(C.cfg, C.store, C.desc_store(), proc / "ensemble", device)
    cal = pickle.loads((proc / "calibrator.pkl").read_bytes())
    ad = pickle.loads((proc / "ad.pkl").read_bytes())
    bundles = torch.load(proc / "admet_bundles.pt", weights_only=False, map_location=device)
    ref = lambda smi, e: (e.latent(smi), morgan_count(smi))
    scorer = Scorer(C.cfg, ens, cal, ad, ref, bundles, C.desc_scaler(), device)
    n_cal = calibrate_n if calibrate_n is not None else C.cfg["rl"].get("calibrate_n", 10000)
    return PipelineScorer(scorer, cache=cache), scorer, C, n_cal


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Not `required=True`: that is enforced below instead, so `--list-algos` can run
    # without inventing a prior path. The check that replaces it is strictly louder --
    # it names the command that builds the missing artifact.
    ap.add_argument("--prior", default=None)
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--out", default=None, help="default: <runs>/rl_min")
    # default=None, then fall back to the config. With literal defaults here, rl.steps /
    # rl.batch_size / rl.sigma / rl.lr in configs/*.yaml were INERT for this entry point:
    # a smoke config asking for steps=8, batch_size=16 still ran 300 x 64 (19.4 min
    # instead of seconds), and configs/hpc.yaml's steps=1000 would have run 300. Same
    # defect class as the eight keys fixed in CHANGES_V3.md section 7 -- a knob that reads
    # as set and does nothing.
    ap.add_argument("--steps", type=int, default=None, help="default: rl.steps")
    ap.add_argument("--batch-size", type=int, default=None, help="default: rl.batch_size")
    ap.add_argument("--sigma", type=float, default=None, help="default: rl.sigma")
    ap.add_argument("--lr", type=float, default=None, help="default: rl.lr")
    # The policy-optimisation objective. `dap` is the incumbent and the default, so an
    # existing command line reproduces bit-for-bit; every alternative honours the same
    # trainer signature, so nothing downstream (history.csv, best_molecules.csv,
    # meta.json, the Stage-8 hand-off) changes shape.
    from .algos import ALGOS as _ALGOS, describe as _describe
    ap.add_argument("--algo", choices=_ALGOS, default="dap",
                    help="policy-optimisation objective. " +
                         " | ".join(f"{k}: {v}" for k, v in _describe().items()))
    ap.add_argument("--list-algos", action="store_true",
                    help="print the available RL algorithms and exit")
    ap.add_argument("--scorer", choices=("standalone", "pipeline"), default="standalone")
    ap.add_argument("--motif", default="c1ncnc(N)c1",
                    help="standalone scorer only: the SMARTS it rewards")
    ap.add_argument("--cache", choices=("step", "run", "off"), default="step",
                    help="pipeline scorer only; 'run' freezes beta per molecule (see scorer.py)")
    ap.add_argument("--calibrate-n", type=int, default=None,
                    help="pipeline scorer only; default rl.calibrate_n from the config")
    ap.add_argument("--baseline-n", type=int, default=None,
                    help="prior samples to score for the baseline; "
                         "default rl_min.baseline_n from the config (2000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=None)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    if a.list_algos:
        from .algos import EXTRA_CONFIG_KEYS
        for k, v in _describe().items():
            print(f"{k:10} {v}")
            keys = EXTRA_CONFIG_KEYS.get(k, ())
            print(f"{'':10} config block rl.{k}: "
                  f"{', '.join(keys) if keys else '(no extra keys)'}")
        return 0

    missing_args = [f"--{n}" for n in ("prior", "vocab") if not getattr(a, n)]
    if missing_args:
        ap.error(f"the following arguments are required: {', '.join(missing_args)}")

    for p in (Path(a.prior), Path(a.vocab)):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Train a prior first:\n"
                f"  python -m rl_min.train_prior --corpus data/features/"
                f"unique_molecules_788k.parquet --col canonical_smiles "
                f"--out priors/kinase_788k.pt"
            )

    vocab = Vocabulary.load(a.vocab)
    prior = SmilesRNN.load(a.prior, a.device)
    agent = copy.deepcopy(prior).to(a.device)        # agent starts AS the prior

    # Output directory from the config, so `--out` is the only way to land outside the
    # repo's runs/ tree and a bare run cannot scatter artifacts into the caller's CWD.
    if a.out:
        out = Path(a.out)
    else:
        from tnk1.utils.common import load_config
        out = Path(load_config(a.config)["paths"]["runs"]) / "rl_min"
    out.mkdir(parents=True, exist_ok=True)

    # steps / batch_size / sigma / lr are NOT recorded here: they are resolved from the
    # config further down and written to meta["dap_config"], which is the authoritative
    # record of what actually ran. Copying the argparse values here would write `null`
    # whenever the config supplied them.
    meta: dict = {"scorer": a.scorer, "seed": a.seed, "device": a.device,
                  "config": _cfg_path_for_meta(a.config),
                  "prior": str(Path(a.prior).resolve()), "vocab": str(Path(a.vocab).resolve())}

    if a.scorer == "standalone":
        scorer = StandaloneScorer(a.motif)
        baseline_scorer = scorer
        meta["motif"] = a.motif
        print(f"[scorer] standalone (QED x MW-window x motif {a.motif!r}) -- "
              f"validates the optimiser, NOT pharmacology")
    else:
        cache = None if a.cache == "off" else a.cache
        scorer, raw, C, n_cal = build_pipeline_scorer(
            a.config, a.version, a.device, a.calibrate_n, cache)
        baseline_scorer = scorer.static()
        meta["components"] = list(raw.components)
        meta["weights"] = dict(raw._weights())
        print(f"[scorer] pipeline: components={raw.components}")

        # Calibrate the potency/selectivity sigmoid centres on the PRIOR's own output.
        print(f"[calibrate] sampling {n_cal} molecules from the prior ...")
        t0 = time.time()
        # Sample only -- `Scorer.calibrate` does the scoring itself, so running the
        # molecules through `prior_baseline` first would double the surrogate cost.
        torch.manual_seed(a.seed)
        cal_smis: list[str] = []
        while len(cal_smis) < n_cal:
            k = min(512, n_cal - len(cal_smis))
            seqs, _ = prior.sample_with(vocab, k, device=a.device)
            cal_smis.extend(vocab.decode(r) for r in seqs)
        centres = raw.calibrate(cal_smis)
        print(f"[calibrate] pot_c={centres['pot_c']:.3f} sel_c={centres['sel_c']:.3f} "
              f"(n={centres['n']}, target_frac={centres['target_frac']}) "
              f"in {time.time() - t0:.0f}s")
        meta["calibration"] = centres

    from tnk1.utils.common import load_config as _lc0
    baseline_n = a.baseline_n
    if baseline_n is None:
        baseline_n = int((_lc0(a.config).get("rl_min") or {}).get("baseline_n", 2000))
    base = prior_baseline(prior, vocab, baseline_scorer, baseline_n, a.device, a.seed)
    base.pop("smiles", None)
    print(f"[baseline] prior over {base['n']} samples: mean={base['score_mean']:.4f} "
          f"max={base['score_max']:.4f} validity={base['validity']:.3f} "
          f"scaffolds={base['n_scaffolds']}")

    # Plumb the rl_min.* block through to DAPConfig. `replay_size`, `replay_batch`,
    # `grad_clip`, `temperature` and `baseline_n` were all declared in
    # configs/default.yaml and NONE of them reached the trainer: DAPConfig was built from
    # five CLI flags only, so the config's values were inert. replay in particular is not
    # cosmetic -- the config comment says "sparse rewards -> without replay the gradient
    # sees only failures", and the dataclass default happened to match, so editing the
    # config appeared to do nothing for no visible reason.
    from tnk1.utils.common import load_config as _lc
    _cfg_all = _lc(a.config)
    _rm = (_cfg_all.get("rl_min") or {})
    _rl = (_cfg_all.get("rl") or {})
    # CLI wins when given; otherwise the config; otherwise the historical literal.
    pick = lambda cli, key, default, cast: cast(cli if cli is not None else _rl.get(key, default))
    dap_kw = dict(steps=pick(a.steps, "steps", 300, int),
                  batch_size=pick(a.batch_size, "batch_size", 64, int),
                  sigma=pick(a.sigma, "sigma", 120.0, float),
                  lr=pick(a.lr, "lr", 1e-4, float),
                  seed=a.seed)
    for cli_name, cfg_key, cast in (
        ("replay_size", "replay_size", int),
        ("replay_batch", "replay_batch", int),
        ("grad_clip", "grad_clip", float),
        ("temperature", "temperature", float),
    ):
        if cfg_key in _rm and _rm[cfg_key] is not None:
            dap_kw[cli_name] = cast(_rm[cfg_key])
    # the diversity filter is configured under rl.diversity, and was equally inert here
    div = _rl.get("diversity") or {}
    if "bucket_size" in div:
        dap_kw["df_bucket"] = int(div["bucket_size"])
    if "min_score" in div:
        dap_kw["df_min_score"] = float(div["min_score"])
    cfg = DAPConfig(**dap_kw)
    meta["dap_config"] = {k: getattr(cfg, k) for k in
                          ("steps", "batch_size", "sigma", "lr", "max_len", "temperature",
                           "grad_clip", "df_bucket", "df_min_score", "replay_size",
                           "replay_batch", "seed")}

    # ---- the chosen algorithm's own hyperparameters -----------------------------------
    # Read from `rl.<algo>` in the config and ATTACHED to the DAPConfig instance, which is
    # what every trainer reads them off via getattr(cfg, name, default). A key present in
    # the config but not in EXTRA_CONFIG_KEYS is REFUSED rather than ignored: silently
    # unread config keys are this repository's most recurrent defect class -- eight of them
    # in CHANGES_V3.md section 7, `rl.steps` as an inert argparse literal, and the
    # adme/qed components computed and discarded.
    from .algos import EXTRA_CONFIG_KEYS, get_trainer
    algo_cfg = dict((_rl.get(a.algo) or {}))
    allowed = set(EXTRA_CONFIG_KEYS.get(a.algo, ()))
    unknown = sorted(set(algo_cfg) - allowed)
    if unknown:
        raise SystemExit(
            f"config block rl.{a.algo} has key(s) {unknown} that the {a.algo!r} trainer "
            f"never reads. It accepts {sorted(allowed) or 'no extra keys'}. Remove them, "
            f"or they will read as set and do nothing.")
    for k, v in algo_cfg.items():
        setattr(cfg, k, v)
    meta["algo"] = a.algo
    meta["algo_config"] = {k: getattr(cfg, k) for k in sorted(allowed)
                           if hasattr(cfg, k)}
    trainer = get_trainer(a.algo)
    print(f"[{a.algo}] {meta['dap_config']}")
    if meta["algo_config"]:
        print(f"[{a.algo}] algo-specific: {meta['algo_config']}")
    elif allowed:
        print(f"[{a.algo}] algo-specific: defaults (config block rl.{a.algo} is absent); "
              f"accepts {sorted(allowed)}")

    hist_path = out / "history.csv"

    def log(m):
        print(f"  step {m['step']:>4} | R {m['score_mean']:.4f} (max {m['score_max']:.3f}) "
              f"| valid {m['validity']:.2f} | uniq {m['uniqueness']:.2f} "
              f"| scaf {m['n_scaffolds']:>3} | loss {m['loss']:9.1f} "
              f"| best {m['replay_best']:.3f}")

    hist, replay = trainer(prior, agent, vocab, scorer, cfg, a.device,
                           # cfg.steps, not a.steps: the latter is None whenever the
                           # value came from the config rather than the command line.
                           log_every=max(1, cfg.steps // 30), logger=log)

    import pandas as pd
    agent.save(out / "agent.pt")
    pd.DataFrame(hist).to_csv(hist_path, index=False)
    pd.DataFrame({"smiles": replay.smiles, "score": replay.scores}).to_csv(
        out / "best_molecules.csv", index=False)
    (out / "baseline.json").write_text(json.dumps(base, indent=2))

    first = float(np.mean([h["score_mean"] for h in hist[:10]]))
    last = float(np.mean([h["score_mean"] for h in hist[-10:]]))
    verdict = "LEARNED" if last > base["score_mean"] * 1.5 else "NO CLEAR IMPROVEMENT"
    meta.update({"prior_baseline": base["score_mean"], "first_10_mean": first,
                 "last_10_mean": last, "verdict": verdict,
                 "replay_best": max(replay.scores) if replay.scores else 0.0})
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"\n[result] prior baseline {base['score_mean']:.4f} | "
          f"first-10-step mean {first:.4f} | last-10-step mean {last:.4f}")
    print(f"[result] {verdict}  (agent must beat the prior baseline, not just its own start)")
    if a.scorer == "standalone":
        print("[result] NOTE: standalone reward -- this says the optimiser works, not that "
              "the molecules are good. Re-run with --scorer pipeline for a claim.")
    print(f"[result] artifacts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
