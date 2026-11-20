"""Stage 3b: which fine-tuning strategy, and which pretraining tasks?

    python scripts/run_stage3b_transfer.py --version v3 --strategies l2sp,lp_ft,swa
    python scripts/run_stage3b_transfer.py --version v3 --index 2        # array mode
    python scripts/run_stage3b_transfer.py --task-selection-only         # CPU, no training
    python scripts/run_stage3b_transfer.py --list

TWO EXPERIMENTS, ONE STAGE
--------------------------
**B. The fine-tuning ladder.** `tnk1.models.ablations` varies the PRETRAINING stack
(A0-A7: SSL, kinome design, weighting, censored loss, L2-SP) and the v3 run's Priority 1
is to read A7's position in it. What no ladder in this repository varies is the FINE-TUNING
STRATEGY: the three-phase L2-SP recipe in `tnk1.models.pipeline.finetune_tnk1` was chosen
once and has never been compared with an alternative. `tnk1.transfer.finetune.STRATEGIES`
supplies the alternatives, each with a citation; this stage runs them over the frozen outer
folds and reports them paired.

**C. Pretraining-task selection.** The pan-kinome stage weights all 491 kinases by pocket
similarity and selects none, so a kinase whose structure-activity relationship is
unrelated to TNK1's still contributes gradient that competes with the 172 TNK1
measurements. `--task-selection-only` computes, from the frozen table alone and with no
training at all, each donor kinase's Spearman correlation with TNK1 over the compounds
measured on BOTH -- which is the only direct evidence of shared SAR available before
spending GPU time. Output: a ranked donor list, a weight vector that clamps
anti-correlated kinases to zero, and the count of donors that have too few shared
compounds to judge.

Run C first. It is CPU-only, takes seconds, and its answer changes what B should be
pretrained on.

OUTPUTS
-------
    runs/stage3b_transfer.csv        per (strategy, fold): metrics
    runs/stage3b_transfer.json       paired tests vs the l2sp incumbent, provenance
    runs/stage3b_task_selection.csv  per donor kinase: rho, overlap, weight
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.evaluation.stats import corrected_resampled_ttest, holm  # noqa: E402
from tnk1.transfer import task_selection as TS  # noqa: E402
from tnk1.utils.common import load_config, read_table, seed_everything, verify_locked  # noqa: E402


# The array-job order. Fixed here so `--index N` is reproducible and reviewable in the
# diff, rather than depending on dict iteration order at submission time.
LADDER = ["l2sp", "plain", "head_only", "lp_ft", "surgical_last", "surgical_first",
          "gradual", "swa", "ema", "mixout"]


def run_task_selection(cfg, df, tnk1, out_stem, min_overlap=10) -> dict:
    """Experiment C. Pure pandas/scipy -- no model, no GPU, no RDKit."""
    print(f"[transfer] === task selection: donors vs {tnk1} ===")
    corr = TS.activity_correlation(df, tnk1, min_overlap=min_overlap)
    n_donors = int(df[df.uniprot_id != tnk1].uniprot_id.nunique())
    weights = TS.task_weights_from_correlation(corr)
    rows = [{"uniprot_id": u, "rho": c["rho"], "p": c["p"], "n_overlap": c["n_overlap"],
             "weight": weights[u]} for u, c in corr.items()]
    T = pd.DataFrame(rows).sort_values("rho", ascending=False)
    T.to_csv(f"{out_stem}_task_selection.csv", index=False)

    judged = len(corr)
    pos = int((T.rho > 0).sum())
    strong = int((T.rho >= 0.3).sum())
    anti = int((T.rho <= -0.3).sum())
    print(f"[transfer] {n_donors} donor kinases; {judged} have >= {min_overlap} compounds "
          f"shared with {tnk1} and can be judged at all")
    print(f"[transfer]   {pos} positively correlated, {strong} with rho >= 0.30, "
          f"{anti} ANTI-correlated at rho <= -0.30")
    print(f"[transfer]   {n_donors - judged} cannot be judged -- too few shared compounds. "
          f"They are currently weighted by POCKET similarity alone.")
    if judged:
        print(f"\n[transfer] top 15 donors by SAR correlation:")
        print(T.head(15).round(3).to_string(index=False))
        if anti:
            print(f"\n[transfer] anti-correlated donors (rho <= -0.30) -- these are being "
                  f"trained on now:")
            print(T[T.rho <= -0.3].round(3).to_string(index=False))
            print("[transfer] A donor whose SAR runs OPPOSITE to the target's is worse "
                  "than useless:\n"
                  "           the trunk is asked to fit both directions at once. "
                  "`task_weights_from_correlation`\n"
                  "           clamps these to zero; the current pipeline does not.")
    return {"n_donors": n_donors, "n_judged": judged, "n_positive": pos,
            "n_strong": strong, "n_anticorrelated": anti, "min_overlap": min_overlap,
            "top_donors": T.head(25).to_dict(orient="records"),
            "anticorrelated": T[T.rho <= -0.3].to_dict(orient="records")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--version", default="v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--strategies", default=None,
                    help="comma-separated; default: the whole LADDER")
    ap.add_argument("--index", type=int, default=None,
                    help="array mode: run LADDER[index] only")
    ap.add_argument("--incumbent", default="l2sp",
                    help="the strategy every other is tested against")
    ap.add_argument("--task-selection-only", action="store_true",
                    help="experiment C only: CPU, seconds, no training")
    ap.add_argument("--min-overlap", type=int, default=10,
                    help="task selection: compounds shared with TNK1 needed to judge a donor")
    ap.add_argument("--folds", type=int, default=None,
                    help="limit to the first N outer folds (for a smoke run)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)

    if a.list:
        from tnk1.transfer import describe
        print("fine-tuning strategies (array index -> name):")
        for i, n in enumerate(LADDER):
            print(f"  {i:2}  {n:16} {describe().get(n, '')}")
        return 0

    cfg = load_config(a.config)
    proc, runs = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["runs"])
    runs.mkdir(parents=True, exist_ok=True)
    stem = Path(a.out) if a.out else runs / "stage3b_transfer"

    tbl = proc / f"kinome_long_{a.version}.parquet"
    sha = verify_locked(tbl)
    df = read_table(tbl)
    tnk1 = cfg["targets"]["tnk1"]

    # ---- experiment C -----------------------------------------------------------------
    sel = run_task_selection(cfg, df, tnk1, str(stem), a.min_overlap)
    if a.task_selection_only:
        Path(f"{stem}_task_selection.json").write_text(
            json.dumps({"dataset_sha256": sha, "target": tnk1, **sel},
                       indent=2, default=float))
        print(f"\n[transfer] wrote {stem}_task_selection.csv")
        print(f"[transfer] wrote {stem}_task_selection.json")
        print("[transfer] --task-selection-only: no training was run.")
        return 0

    # ---- experiment B -----------------------------------------------------------------
    from tnk1.context import Ctx
    from tnk1.models.pipeline import kinome_stage
    from tnk1.models.trainer import predict
    from tnk1.transfer import get_strategy
    from tnk1.uq.metrics import full_report

    if a.index is not None:
        if not 0 <= a.index < len(LADDER):
            raise SystemExit(f"--index {a.index} out of range; LADDER has {len(LADDER)} "
                             f"entries (see --list)")
        names = [LADDER[a.index]]
    elif a.strategies:
        names = [s.strip() for s in a.strategies.split(",") if s.strip()]
    else:
        names = list(LADDER)
    unknown = [n for n in names if n not in LADDER]
    if unknown:
        from tnk1.transfer import describe
        raise SystemExit(f"unknown strategy/strategies {unknown}; have "
                         f"{sorted(describe())}")
    print(f"\n[transfer] === fine-tuning ladder: {', '.join(names)} ===")

    C = Ctx(a.config, a.version, a.device)
    store, desc = C.store, C.desc_store()
    ssl_ckpt = proc / "ssl_encoder.pt"
    ssl = str(ssl_ckpt) if ssl_ckpt.exists() else None
    if ssl is None:
        print(f"[transfer] NOTE {ssl_ckpt} absent; the trunk starts from random init. "
              f"That changes what every strategy is adapting FROM, so do not compare "
              f"these numbers with a run that had it.")

    folds = C.splits["cv"][: a.folds] if a.folds else C.splits["cv"]
    rows = []
    for f in folds:
        fold = f["outer"]
        tr_keys, te_keys = set(f["train"]), set(f["test"])
        inner = f["inner"][0] if f.get("inner") else None
        tr = C.tnk1_rows(set(inner["train"]) if inner else tr_keys)
        va = C.tnk1_rows(set(inner["val"])) if inner else None
        te = C.tnk1_rows(te_keys, censored=False)
        if len(te) < 3:
            print(f"[transfer] fold {fold}: only {len(te)} exact test rows -- skipped")
            continue
        # The kinome trunk is trained ONCE per fold and deep-copied per strategy: the
        # strategies differ in how they ADAPT a pretrained trunk, so giving each its own
        # pretraining would confound the comparison with pretraining variance.
        print(f"[transfer] fold {fold}: pretraining the shared trunk "
              f"(train {len(tr)} / val {len(va) if va is not None else 0} / test {len(te)})")
        seed_everything(a.seed + fold)
        kin = C.kinome_train_df()
        kin = kin[~kin.inchikey_block1.isin(te_keys)]
        trunk = kinome_stage(cfg, kin, store, desc, ssl, "B", seed=a.seed + fold,
                             device=a.device)
        trunk_state = {k: v.detach().clone() for k, v in trunk.state_dict().items()}

        for name in names:
            import copy as _copy
            t0 = time.time()
            model = _copy.deepcopy(trunk)
            model.load_state_dict(trunk_state)
            seed_everything(a.seed + fold)
            logs = []
            try:
                fitted = get_strategy(name)(cfg, model, tr, va, store, desc,
                                           device=a.device, seed=a.seed + fold,
                                           log=logs.append)
            except Exception as e:
                print(f"[transfer]   {name:16} FAILED: {type(e).__name__}: {e}")
                rows.append({"strategy": name, "fold": fold,
                             "status": f"FAILED: {type(e).__name__}: {e}"})
                continue
            from tnk1.models.pipeline import _mk
            kmap = {u: i for i, u in enumerate(store["ids"])}
            mu, lv = predict(fitted, _mk(te, kmap, desc), a.device)
            rep = full_report(te.pact.values, mu, np.sqrt(np.exp(lv)))
            rows.append({"strategy": name, "fold": fold, "status": "ok",
                         "seconds": time.time() - t0,
                         "phases": json.dumps([l for l in logs if "phase" in l]),
                         **rep})
            print(f"[transfer]   {name:16} rho {rep['spearman']:+.3f}  "
                  f"rmse {rep['rmse']:.3f}  err-sig {rep.get('err_unc_rho', float('nan')):+.3f}  "
                  f"({time.time() - t0:.0f}s)")

    if not rows:
        raise SystemExit("[transfer] no fold produced a result")
    R = pd.DataFrame(rows)
    R.to_csv(f"{stem}.csv", index=False)
    ok = R[R.status == "ok"]

    out = {"dataset_sha256": sha, "version": a.version, "strategies": names,
           "incumbent": a.incumbent, "n_folds": int(ok.fold.nunique()),
           "ssl_encoder": ssl, "task_selection": sel, "comparisons": {}}
    if not ok.empty:
        S = ok.groupby("strategy")[["rmse", "spearman", "err_unc_rho"]].agg(["mean", "std"])
        print(f"\n[transfer] === per-fold mean over {ok.fold.nunique()} folds ===")
        print(S.round(3).to_string())
        out["summary"] = json.loads(S.to_json())

        if a.incumbent in set(ok.strategy) and ok.strategy.nunique() > 1:
            inc = ok[ok.strategy == a.incumbent].set_index("fold").sort_index()
            pv, order = [], []
            print(f"\n[transfer] === paired against `{a.incumbent}` "
                  f"(Nadeau-Bengio corrected) ===")
            for name in sorted(set(ok.strategy) - {a.incumbent}):
                ch = ok[ok.strategy == name].set_index("fold").sort_index()
                common = inc.index.intersection(ch.index)
                if len(common) < 3:
                    continue
                d = ch.loc[common, "spearman"].values - inc.loc[common, "spearman"].values
                c = corrected_resampled_ttest(d, np.zeros_like(d), k_folds=5)
                out["comparisons"][name] = c
                pv.append(c["p"])
                order.append(name)
                print(f"  {name:16} d_rho {c['mean_diff']:+.4f}  p={c['p']:.4f}  "
                      f"n={c['n']}  "
                      f"{'BETTER' if c['mean_diff'] > 0 else 'worse'}")
            if pv:
                for n, padj in zip(order, holm(pv)):
                    out["comparisons"][n]["p_holm"] = float(padj)
    Path(f"{stem}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[transfer] wrote {stem}.csv")
    print(f"[transfer] wrote {stem}.json")
    print(f"[transfer] wrote {stem}_task_selection.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
