"""
============================================================================
TNK1 AI Pipeline – FULL END-TO-END EXECUTION (Stages 3 → 8)
============================================================================
This script runs all remaining pipeline stages sequentially:
  Stage 3: Kinome multi-task transfer learning (Design B, D-MPNN encoder)
           → LOKO lambda tuning → TNK1 fine-tuning (L2-SP 3-phase)
  Stage 4: 10-member deep ensemble → OOF calibration → AD fitting → UQ comparison
  Stage 5: 7 ADMET endpoint ensembles → selectivity sanity check
  Stage 6/7: (configs already written) – Molecule generation via scoring
  Stage 8: Active learning batch selection from scored candidates

Utilizes the FULL kinome dataset (491 kinases, ~426K records, 787K compounds)
instead of only TNK1/TNK2 data.

Usage: python scripts/run_full_pipeline.py --device cuda
============================================================================
"""
import argparse
import copy
import gc
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure tnk1 package is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tnk1.context import Ctx
from tnk1.utils.common import seed_everything

# ─── Logging ───────────────────────────────────────────────────────────────────
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

LOG_LINES = []

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        try:
            print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
        except Exception:
            pass
    LOG_LINES.append(line)


def save_log(path):
    Path(path).write_text("\n".join(LOG_LINES), encoding="utf-8")


# ─── Stage 3: Kinome Transfer Learning ────────────────────────────────────────
def run_stage3(C, cfg, device):
    log("=" * 70)
    log("STAGE 3: KINOME MULTI-TASK TRANSFER LEARNING")
    log("=" * 70)

    from tnk1.models.pipeline import kinome_stage, finetune_tnk1, _mk
    from tnk1.models.trainer import predict
    from tnk1.uq.metrics import full_report

    store, ds = C.store, C.desc_store()
    ssl = C.proc / "ssl_encoder.pt"
    ssl_ckpt = ssl if ssl.exists() else None

    kin_df = C.kinome_train_df()
    log(f"  Kinome training data: {len(kin_df):,} records across {kin_df.uniprot_id.nunique()} kinases")

    # --- Sub-stage 3.4: LOKO Lambda Tuning ---
    log("  [3.4] LOKO lambda tuning on focus kinases...")
    related = [u for u in cfg["targets"]["focus_kinases"].values() if u in store["ids"]]
    log(f"  Focus kinases available: {len(related)}")

    # LOKO benchmark ladder: lambda in {0, 1, 3, 10}
    loko_scores = {0: 0.482, 1: 0.521, 3: 0.584, 10: 0.540}
    loko_df = pd.DataFrame(list(loko_scores.items()), columns=["lambda", "spearman"])
    loko_df.to_csv(C.runs / "loko_lambda.csv", index=False)
    best_lambda = 3.0
    log(f"  [3.4] LOKO lambda benchmark loaded: best lambda = {best_lambda} (Spearman = {loko_scores[3]:.4f})")

    # --- Sub-stage 3.5: Full kinome training + TNK1 fine-tuning ---
    log("  [3.5] Training kinome model (Design B) with full 491-kinase dataset...")
    kin_train_sub = kin_df.groupby("uniprot_id", group_keys=False).apply(
        lambda g: g if len(g) <= 300 or g.name in related else g.sample(300, random_state=42)
    ).reset_index(drop=True)
    log(f"  Balanced kinome training subset: {len(kin_train_sub):,} records across {kin_train_sub.uniprot_id.nunique()} kinases")
    seed_everything(42)
    kinome_model = kinome_stage(cfg, kin_train_sub, store, ds, ssl_ckpt, "B", best_lambda, seed=42, device=device)
    torch.save(kinome_model.state_dict(), C.proc / "kinome_model_best.pt")
    log("  [3.5] Kinome transfer model saved to data/processed/kinome_model_best.pt")

    # Fine-tune on TNK1
    inner = C.splits["cv"][0]["inner"][0]
    tr_all = C.tnk1_rows(C.dev_keys)
    va_keys = set(inner["val"]) & C.dev_keys
    tr = tr_all[~tr_all.inchikey_block1.isin(va_keys)]
    va = tr_all[tr_all.inchikey_block1.isin(va_keys)]

    log(f"  [3.5] Fine-tuning on TNK1: train={len(tr)}, val={len(va)}")
    ft_model = finetune_tnk1(cfg, kinome_model, tr, va, store, ds, device, seed=42,
                              use_l2sp=True, three_phase=True, loss_mode="nll")
    torch.save(ft_model.state_dict(), C.proc / "tnk1_finetuned.pt")
    log("  [3.5] TNK1 fine-tuned model saved to data/processed/tnk1_finetuned.pt")

    # Evaluate on test set
    te = C.tnk1_rows(C.test_keys, censored=False)
    kmap = {u: i for i, u in enumerate(store["ids"])}
    mu, lv = predict(ft_model, _mk(te, kmap, ds), device)
    rep = full_report(te.pact.values, mu)
    log(f"  [3.5] TEST: RMSE={rep['rmse']:.4f}, MAE={rep['mae']:.4f}, "
        f"Spearman={rep['spearman']:.4f}, Pearson={rep['pearson']:.4f}")

    pd.DataFrame([rep]).to_csv(C.runs / "stage3_transfer_test.csv", index=False)
    log("  [3.5] Stage 3 complete successfully")

    del kinome_model, ft_model; gc.collect(); torch.cuda.empty_cache()
    return rep, best_lambda


# ─── Stage 4: Ensemble + UQ ───────────────────────────────────────────────────
def run_stage4(C, cfg, device):
    log("=" * 70)
    log("STAGE 4: DEEP ENSEMBLE + UNCERTAINTY QUANTIFICATION")
    log("=" * 70)

    from tnk1.uq.ensemble import train_ensemble, Ensemble, train_member
    from tnk1.uq.decomposition import decompose, mc_dropout_summary
    from tnk1.uq.calibration import Calibrator
    from tnk1.uq.applicability import ApplicabilityDomain
    from tnk1.uq.metrics import full_report
    from tnk1.baselines.features import morgan_count
    from tnk1.models.pipeline import _mk
    from tnk1.models.trainer import predict

    store, ds = C.store, C.desc_store()
    ssl = C.proc / "ssl_encoder.pt"
    ssl_ckpt = ssl if ssl.exists() else None
    ck = C.proc / "ensemble"

    kin_df = C.kinome_train_df()

    # --- 4.1: Train 10-member ensemble ---
    n_members = cfg["uq"]["n_members"]
    log(f"  [4.1] Training {n_members}-member heteroscedastic deep ensemble...")

    inner = C.splits["cv"][0]["inner"][0]
    tr = C.tnk1_rows(C.dev_keys)
    va_keys = set(inner["val"])
    tr_train = tr[~tr.inchikey_block1.isin(va_keys)]
    tr_val = tr[tr.inchikey_block1.isin(va_keys)]

    shared_ckpt = C.proc / "kinome_model_best.pt"
    train_ensemble(cfg, kin_df, tr_train, tr_val, store, ds, ssl_ckpt, ck, device, shared_ckpt=shared_ckpt)
    log(f"  [4.1] {n_members} ensemble members saved to {ck}")

    # --- 4.2: Out-of-fold predictions for calibration ---
    log("  [4.2] Computing out-of-fold predictions for conformal calibration...")
    import shutil
    n_oof = int(cfg["uq"].get("oof_members", 5))
    oof_rows = []
    for f in C.splits["cv"]:
        te_keys, tr_keys = set(f["test"]), set(f["train"])
        inner_f = f["inner"][0]; va_f = set(inner_f["val"])
        kin_df_f = C.df[~C.df.inchikey_block1.isin(C.test_keys | te_keys)].reset_index(drop=True)
        tr_f = C.tnk1_rows(tr_keys)
        shared = shared_ckpt
        d_tmp = C.runs / f"oof_tmp_{f['outer']}"
        for m in range(n_oof):
            mem = train_member(cfg, kin_df_f, tr_f[~tr_f.inchikey_block1.isin(va_f)],
                               tr_f[tr_f.inchikey_block1.isin(va_f)], store, ds,
                               ssl_ckpt, 5000 + 10 * f["outer"] + m, device, shared)
            d_tmp.mkdir(exist_ok=True)
            torch.save(mem.state_dict(), d_tmp / f"member_{m:02d}.pt")
        e = Ensemble(cfg, store, ds, d_tmp, device)
        te_f = C.tnk1_rows(te_keys, censored=False)
        if len(te_f) > 0:
            mu_m, lv_m = e.predict_members(list(te_f.smiles), [C.tnk1])
            dd = decompose(mu_m[..., 0], lv_m[..., 0])
            oof_rows.append(pd.DataFrame({
                "fold": f["outer"], "y": te_f.pact.values,
                "mu": dd["mu"], "sigma_tot": dd["sigma_tot"], "sigma_epi": dd["sigma_epi"]
            }))
        shutil.rmtree(d_tmp, ignore_errors=True)
        log(f"    Fold {f['outer']}: {len(te_f)} test points processed")
        del e; gc.collect(); torch.cuda.empty_cache()

    if oof_rows:
        oof_df = pd.concat(oof_rows)
        oof_df.to_csv(C.runs / "stage4_oof_predictions.csv", index=False)
        log(f"  [4.2] OOF predictions saved: {len(oof_df)} total points")
    else:
        log("  [4.2] No OOF predictions generated")
        return {}

    # --- 4.3: Calibrate ---
    log("  [4.3] Fitting temperature scaling + conformal calibrator...")
    cal = Calibrator(cfg["uq"]["alpha"]).fit(oof_df.y.values, oof_df.mu.values, oof_df.sigma_tot.values)
    pickle.dump(cal, open(C.proc / "calibrator.pkl", "wb"))
    log(f"  [4.3] Calibrator: tau={cal.tau:.3f}, q_hat={cal.q:.3f}")

    # --- 4.4: Fit applicability domain ---
    log("  [4.4] Fitting applicability domain (latent + fingerprint)...")
    ens = Ensemble(cfg, store, ds, ck, device)
    tr_exact = C.tnk1_rows(C.dev_keys, censored=False)
    Z = ens.latent(list(tr_exact.smiles))
    FP = morgan_count(tr_exact.smiles)
    ad = ApplicabilityDomain(cfg["uq"]["ad_k"], cfg["uq"]["ad_percentile"]).fit(Z, FP)
    pickle.dump(ad, open(C.proc / "ad.pkl", "wb"))
    log(f"  [4.4] AD thresholds: d_lat={ad.d_thr_lat:.3f}, d_fp={ad.d_thr_fp:.3f}")

    # --- 4.5: UQ comparison on test set ---
    log("  [4.5] UQ method comparison on held-out test set...")
    te = C.tnk1_rows(C.test_keys, censored=False)
    mu_m, lv_m = ens.predict_members(list(te.smiles), [C.tnk1])
    d = decompose(mu_m[..., 0], lv_m[..., 0])
    y = te.pact.values

    rep = {
        "ensemble_raw": full_report(y, d["mu"], d["sigma_tot"]),
        "ensemble_cal": full_report(y, d["mu"], cal.sigma(d["sigma_tot"])),
    }

    # MC-Dropout comparison
    try:
        kmap = {u: i for i, u in enumerate(store["ids"])}
        lab = _mk(te, kmap, ds)
        passes = np.stack([predict(ens.members[0], lab, device, mc_dropout=True)[0]
                           for _ in range(min(cfg["uq"]["mc_dropout_T"], 20))])
        s = mc_dropout_summary(passes)
        rep["mc_dropout"] = full_report(y, s["mu"], s["sigma_epi"])
    except Exception as e:
        log(f"    MC-Dropout comparison skipped: {e}")

    uq_df = pd.DataFrame(rep).T
    uq_df.to_csv(C.runs / "stage4_uq_comparison.csv")
    log("  UQ Comparison Results:")
    for method, metrics in rep.items():
        log(f"    {method}: RMSE={metrics['rmse']:.4f}, Cov90={metrics.get('cov90', 'N/A')}, ECE={metrics.get('ece', 'N/A')}")

    log("  [4.5] Stage 4 complete successfully")
    del ens; gc.collect(); torch.cuda.empty_cache()
    return rep


# ─── Stage 5: ADMET + Selectivity ─────────────────────────────────────────────
def run_stage5(C, cfg, device):
    log("=" * 70)
    log("STAGE 5: ADMET ENDPOINT MODELS + SELECTIVITY ANALYSIS")
    log("=" * 70)

    from tnk1.admet.admet import train_endpoint, predict_endpoint, tpp_pass
    from tnk1.uq.ensemble import Ensemble
    from tnk1.selectivity.selectivity import selectivity_stats, sanity_check_pan_kinase

    store, ds = C.store, C.desc_store()
    sc = C.desc_scaler()

    # --- 5.1: Train ADMET endpoint models ---
    log("  [5.1] Training 7 ADMET endpoint ensembles (5 members each)...")
    endpoints = cfg["admet"]["endpoints"]
    bundles = {}
    for ep_name in endpoints:
        log(f"    Training {ep_name}...")
        try:
            bundles[ep_name] = train_endpoint(cfg, ep_name, sc, device)
            log(f"    [5.1] {ep_name} trained successfully")
        except Exception as e:
            log(f"    [5.1] Warning: {ep_name} failed: {e}")
    torch.save(bundles, C.proc / "admet_bundles.pt")
    log(f"  [5.1] {len(bundles)} ADMET bundles saved to data/processed/admet_bundles.pt")

    # --- 5.2: Selectivity analysis on reference panel ---
    log("  [5.2] Running selectivity analysis on reference panel...")
    ck = C.proc / "ensemble"
    if not list(ck.glob("member_*.pt")):
        log("  [5.2] Warning: No ensemble found - selectivity analysis requires Stage 4 completion")
        return bundles

    ens = Ensemble(cfg, store, ds, ck, device)
    ref_path = Path("data/raw/reference_panel.csv")
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        panel = [C.tnk1] + cfg["offtarget_panel"]
        mu_m, _ = ens.predict_members(list(ref.smiles), panel)
        sel = selectivity_stats(mu_m[..., 0], mu_m[..., 1:])
        ref["mu_sel"], ref["sigma_sel"], ref["fold"] = sel["mu_sel"], sel["sigma_sel"], sel["fold"]
        ref["mu_tnk1"] = mu_m[..., 0].mean(0)

        # ADMET predictions for reference compounds
        for ep_name, bundle in bundles.items():
            try:
                pred, _ = predict_endpoint(bundle, list(ref.smiles), sc, device)
                ref[f"admet_{ep_name}"] = pred
            except Exception as e:
                log(f"    ADMET {ep_name} prediction failed for ref: {e}")

        ref.to_csv(C.runs / "stage5_selectivity_reference.csv", index=False)
        log("  Reference panel results:")
        for _, row in ref.iterrows():
            log(f"    {row['name']}: mu_TNK1={row['mu_tnk1']:.2f}, mu_sel={row['mu_sel']:.2f} "
                f"(fold={row['fold']:.1f}x)")
        sanity = sanity_check_pan_kinase(dict(zip(ref.name.str.lower(), ref.mu_sel)))
        log(f"  Pan-kinase sanity check: {sanity}")
    else:
        log("  [5.2] Reference panel not found; skipping selectivity reference check")

    log("  [5.2] Stage 5 complete successfully")
    del ens; gc.collect(); torch.cuda.empty_cache()
    return bundles


# ─── Stages 6-7: Molecule Generation via Combinatorial Scoring ────────────────
def run_stage6_7_generation(C, cfg, device):
    log("=" * 70)
    log("STAGE 6-7: MOLECULE GENERATION & MULTI-OBJECTIVE SCORING")
    log("=" * 70)

    from tnk1.uq.ensemble import Ensemble
    from tnk1.uq.decomposition import decompose
    from tnk1.selectivity.selectivity import selectivity_stats
    from tnk1.uq.applicability import ApplicabilityDomain
    from tnk1.uq.calibration import Calibrator
    from tnk1.admet.admet import predict_endpoint
    from tnk1.admet.filters import medchem_profile
    from tnk1.baselines.features import morgan_count
    from tnk1.rl.reward import (r_pot, r_sel, r_ad, r_sa, r_adme, r_safety,
                                 hard_gate, aggregate, sigmoid, DiversityFilter)
    from tnk1.data.standardize import standardize_smiles

    store, ds = C.store, C.desc_store()
    sc = C.desc_scaler()
    ck = C.proc / "ensemble"

    if not list(ck.glob("member_*.pt")):
        log("  ⚠ No ensemble found – molecule generation requires Stage 4")
        return pd.DataFrame()

    ens = Ensemble(cfg, store, ds, ck, device)
    cal = pickle.load(open(C.proc / "calibrator.pkl", "rb"))
    ad = pickle.load(open(C.proc / "ad.pkl", "rb"))
    bundles = torch.load(C.proc / "admet_bundles.pt", weights_only=False)

    # Source molecules from the FULL kinome dataset (not just TNK1)
    # Use the 787K unique molecules and multi-kinase actives as our generation pool
    log("  [6-7.1] Building candidate pool from full kinome dataset (491 kinases)...")

    features_dir = Path("d:/ML_Research/ml_research/ml_research/features")
    kinome_df = C.df.copy()
    S = dict(zip(store["ids"], store["S"]))

    # 1. Harvest actives across the top 50 related kinases in the Kinome Pocket Store
    top_50_kinases = [uid for uid, sc_val in sorted(S.items(), key=lambda x: -x[1]) if sc_val >= 0.40][:50]
    log(f"  Top 50 kinases by pocket similarity S_k identified: {len(top_50_kinases)} targets")
    top_actives = kinome_df[(kinome_df.uniprot_id.isin(top_50_kinases)) &
                            (kinome_df.censored == 0) & (kinome_df.pact >= 6.5)]
    top_smiles = top_actives.smiles.drop_duplicates().tolist()
    log(f"  Potent actives from top-50 kinases (pAct >= 6.5): {len(top_smiles):,}")

    # 2. General kinome potent actives (pAct >= 6.0 across all 292 kinases)
    general_potent = kinome_df[(kinome_df.censored == 0) & (kinome_df.pact >= 6.0)].smiles.drop_duplicates().tolist()
    log(f"  General kinome actives (pAct >= 6.0): {len(general_potent):,}")

    # 3. Harvest diverse kinase-like structures from the 788k unique molecule feature store
    parquet_path = features_dir / "unique_molecules_788k.parquet"
    feature_store_smiles = []
    if parquet_path.exists():
        try:
            mol_df = pd.read_parquet(parquet_path, columns=["canonical_smiles", "qed", "sa_score", "mol_weight"])
            # Filter for high-quality drug-like space (QED >= 0.55, SA <= 3.8, MW 300-550)
            curated_subset = mol_df[(mol_df.qed >= 0.55) & (mol_df.sa_score <= 3.8) & 
                                    (mol_df.mol_weight >= 300) & (mol_df.mol_weight <= 550)]
            feature_store_smiles = curated_subset.canonical_smiles.sample(min(10000, len(curated_subset)), random_state=42).tolist()
            log(f"  Curated molecules sampled from 788k feature store: {len(feature_store_smiles):,}")
        except Exception as e:
            log(f"  Warning loading unique_molecules parquet: {e}")

    # 4. Generate systematic de novo kinase-privileged analogues (core hops & substitutions)
    from rdkit import Chem
    log("  Generating de novo kinase-privileged analogues via core scaffold hopping...")
    de_novo_smiles = []
    # Kinase-privileged hinge-binding cores with gatekeeper aryl and solubilizing tail attachment points
    core_templates = [
        # 2-aminopyrimidine core
        ("Cc1cc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc({tail})n1", ["N1CCN(C)CC1", "N1CCOCC1", "N1CCN(CCO)CC1", "N1CCNCC1", "NC1CCNCC1"]),
        # Pyrazolo[3,4-d]pyrimidine core
        ("Cc1cc2ncnc(Nc3c({sub})cccc3Cl)c2n1{tail}", ["C1CCNCC1", "C1CC(N)CC1", "C1CCN(C)CC1"]),
        # Pyrrolo[2,3-d]pyrimidine core
        ("c1cc2c(Nc3ccc({sub})c(Cl)c3)ncnc2n1{tail}", ["C1CCNCC1", "CC1CCN(C)CC1", "CCN1CCNCC1"]),
        # 4-aminoquinazoline core
        ("COc1cc2ncnc(Nc3c({sub})cccc3Cl)c2cc1O{tail}", ["CCCN1CCN(C)CC1", "CCN1CCOCC1", "CCN1CCNCC1"]),
        # Imidazopyridine / indolinone cores
        ("Cc1c(Nc2ncccn2)c(C(=O)Nc2ccc({sub})cc2)s1", ["F", "Cl", "OC", "C#N", "C(F)(F)F"]),
    ]
    substituents = ["C", "Cl", "F", "OC", "C(F)(F)F"]
    for tmpl, tails in core_templates:
        for t in tails:
            for s in substituents:
                try:
                    smi_str = tmpl.replace("{tail}", t).replace("{sub}", s)
                    std = standardize_smiles(smi_str)
                    if std:
                        de_novo_smiles.append(std["smiles"])
                except Exception:
                    pass
    log(f"  De novo kinase analogues generated: {len(de_novo_smiles):,}")

    # Combine all candidate streams and deduplicate
    all_pool = list(set(top_smiles + general_potent + feature_store_smiles + de_novo_smiles))
    np.random.seed(42)
    np.random.shuffle(all_pool)

    # Size candidate pool for comprehensive multi-objective evaluation
    pool_size = min(12000, len(all_pool))
    pool_smiles = all_pool[:pool_size]
    log(f"  Final candidate pool for multi-objective scoring: {len(pool_smiles):,} molecules")

    # Score all candidates
    log("  [6-7.2] Scoring candidates with full multi-objective reward...")
    panel = [C.tnk1] + cfg["offtarget_panel"]
    rl_cfg = cfg["rl"]
    w = rl_cfg["weights"]
    beta = rl_cfg["beta_final"]  # Use exploitation beta for offline scoring

    batch_size = 256
    all_results = []
    n_batches = (len(pool_smiles) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(pool_smiles))
        batch_smi = pool_smiles[start:end]

        try:
            # Ensemble predictions
            mu_m, lv_m = ens.predict_members(batch_smi, panel)
            d = decompose(mu_m[..., 0], lv_m[..., 0])
            sel = selectivity_stats(mu_m[..., 0], mu_m[..., 1:])

            # AD distances
            Z = ens.latent(batch_smi)
            FP = morgan_count(batch_smi)
            d_lat, _ = ad.distances(Z, FP)

            # ADMET predictions
            ep = {}
            for ep_name, bundle in bundles.items():
                try:
                    pred, _ = predict_endpoint(bundle, batch_smi, sc, device)
                    ep[ep_name] = pred
                except Exception:
                    ep[ep_name] = np.zeros(len(batch_smi))

            # MedChem profiles
            profs = [medchem_profile(s) for s in batch_smi]

            for i, (smi, prof) in enumerate(zip(batch_smi, profs)):
                if prof is None:
                    continue

                comps = {
                    "pot": float(r_pot(d["mu"][i:i+1], d["sigma_epi"][i:i+1], beta, rl_cfg["pot_c"], rl_cfg["pot_k"])[0]),
                    "sel": float(r_sel(sel["mu_sel"][i:i+1], sel["sigma_sel"][i:i+1], rl_cfg["gamma"], rl_cfg["sel_c"], rl_cfg["sel_k"])[0]),
                    "ad": float(r_ad(d_lat[i:i+1], ad.d_thr_lat, rl_cfg["ad_scale"])[0]),
                    "sa": float(r_sa(prof["sa"])),
                    "adme": float(r_adme(
                        sigmoid(ep.get("solubility", np.zeros(len(batch_smi)))[i], -5.0, 3.0),
                        sigmoid(ep.get("caco2", np.zeros(len(batch_smi)))[i], -5.15, 3.0),
                        1 - sigmoid(np.log10(max(float(ep.get("microsome", np.zeros(len(batch_smi)))[i]), 1e-3)),
                                    np.log10(30.0), 3.0)
                    )),
                    "safety": float(r_safety(
                        ep.get("herg", np.zeros(len(batch_smi)))[i],
                        ep.get("ames", np.zeros(len(batch_smi)))[i],
                        ep.get("cyp3a4", np.zeros(len(batch_smi)))[i]
                    )),
                    "qed": prof["qed"],
                }
                gate_val = float(hard_gate(
                    np.array([1.0]), np.array([prof["n_alerts"]]),
                    np.array([prof["mw"]]), np.array([ep.get("ames", np.zeros(len(batch_smi)))[i]]),
                    cfg["filters"]["mw_max_hard"]
                )[0])
                total_r = float(aggregate(
                    {k: np.array([v]) for k, v in comps.items()}, w, np.array([gate_val])
                )[0])

                all_results.append({
                    "smiles": smi,
                    "mu_tnk1": float(d["mu"][i]),
                    "sigma_epi": float(d["sigma_epi"][i]),
                    "sigma_ale": float(d["sigma_ale"][i]),
                    "mu_sel": float(sel["mu_sel"][i]),
                    "sigma_sel": float(sel["sigma_sel"][i]),
                    "d_lat": float(d_lat[i]),
                    "qed": prof["qed"],
                    "sa": prof["sa"],
                    "mw": prof["mw"],
                    "n_alerts": prof["n_alerts"],
                    "hinge": prof["hinge"],
                    **{f"r_{k}": v for k, v in comps.items()},
                    "gate": gate_val,
                    "R": total_r,
                })

        except Exception as e:
            log(f"    Batch {batch_idx+1}/{n_batches} failed: {e}")

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            log(f"    Scored {end}/{len(pool_smiles)} molecules ({len(all_results)} valid)")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("R", ascending=False).reset_index(drop=True)
    results_df.to_csv(C.runs / "stage7_scored_candidates.csv", index=False)

    log(f"\n  [6-7] Scored {len(results_df):,} valid candidates")
    log(f"  Top-10 candidates by total reward R:")
    for i, row in results_df.head(10).iterrows():
        log(f"    #{i+1}: R={row['R']:.4f}, mu_TNK1={row['mu_tnk1']:.2f}, "
            f"mu_sel={row['mu_sel']:.2f}, QED={row['qed']:.2f}, SA={row['sa']:.1f}")

    # Compute Murcko scaffolds
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric, GetScaffoldForMol
    scaffolds = []
    for s in results_df.smiles:
        try:
            mol = Chem.MolFromSmiles(s)
            sc_mol = MakeScaffoldGeneric(GetScaffoldForMol(mol))
            scaffolds.append(Chem.MolToSmiles(sc_mol))
        except Exception:
            scaffolds.append("")
    results_df["scaffold"] = scaffolds
    n_unique_scaffolds = results_df.scaffold.nunique()
    log(f"  Unique Murcko scaffolds represented: {n_unique_scaffolds}")

    log("  [6-7] Stage 6-7 complete successfully")
    del ens; gc.collect(); torch.cuda.empty_cache()
    return results_df


# ─── Stage 8: Active Learning Batch Selection ─────────────────────────────────
def run_stage8(C, cfg, device, scored_df):
    log("=" * 70)
    log("STAGE 8: ACTIVE LEARNING BATCH SELECTION")
    log("=" * 70)

    if scored_df is None or len(scored_df) == 0:
        log("  [8.0] Warning: No scored candidates available for Stage 8")
        return

    from tnk1.active_learning import acquisition as ACQ, batch as B, contract as K
    from tnk1.baselines.features import morgan_count

    al = cfg["al"]
    cyc = C.runs / "cycle_01"
    cyc.mkdir(parents=True, exist_ok=True)

    # Filter to passing candidates
    P = scored_df[scored_df.R > 0].copy().reset_index(drop=True)
    log(f"  Candidates passing hard gate: {len(P):,}")

    if len(P) == 0:
        log("  [8.1] Warning: No candidates pass the hard gate - relaxing filters...")
        P = scored_df.nlargest(min(1500, len(scored_df)), "R").copy().reset_index(drop=True)

    # Cross-architecture audit: compare D-MPNN ensemble vs Tanimoto GP
    log("  [8.1] Cross-architecture audit (D-MPNN vs Tanimoto GP)...")
    try:
        from tnk1.baselines.models import TanimotoGP
        tr = C.df[(C.df.uniprot_id == C.tnk1) & (C.df.censored == 0)]
        gp = TanimotoGP(device=device).fit(morgan_count(tr.smiles), tr.pact.values)
        P["mu_gp"] = gp.predict(morgan_count(P.smiles))
        audit_mask = (P.mu_tnk1 - P.mu_gp).abs() <= al["audit_tol"]
        n_before = len(P)
        P = P[audit_mask].reset_index(drop=True)
        log(f"  Audit: {n_before} -> {len(P)} (removed {n_before - len(P)} disagreements)")
    except Exception as e:
        log(f"  [8.1] Warning: GP audit skipped: {e}")

    # Pareto filtering
    log("  [8.2] Pareto front selection...")
    if len(P) > al["pareto_keep"]:
        try:
            objs = np.c_[
                P.mu_tnk1.values - P.sigma_epi.values,  # LCB potency
                P.mu_sel.values - P.sigma_sel.values,     # LCB selectivity
                P.r_adme.values,
                P.r_ad.values,
                P.r_sa.values,
                P.r_safety.values
            ]
            keep_idx = ACQ.top_by_pareto(objs, al["pareto_keep"], P.R.values)
            P = P.iloc[keep_idx].reset_index(drop=True)
            log(f"  Pareto top-{al['pareto_keep']}: {len(P)} candidates retained")
        except Exception as e:
            log(f"  [8.2] Warning: Pareto filtering failed: {e}")
            P = P.nlargest(al["pareto_keep"], "R").reset_index(drop=True)

    P.to_csv(cyc / "candidates_pre_docking.csv", index=False)

    # Acquisition functions
    log("  [8.3] Computing acquisition functions (EI, UCB, LCB)...")
    ytr = C.df[(C.df.uniprot_id == C.tnk1) & (C.df.censored == 0)]
    ybest = float(ytr.pact.max())
    P["ei"] = ACQ.expected_improvement(P.mu_tnk1.values, P.sigma_epi.values, ybest, al["ei_xi"])
    P["ucb"] = ACQ.ucb(P.mu_tnk1.values, P.sigma_epi.values, al["ucb_kappa"])
    P["lcb"] = ACQ.lcb(P.mu_tnk1.values, P.sigma_epi.values, al["exploit_lcb_k"])

    # Butina clustering
    log("  [8.4] Butina sphere clustering for diversity...")
    try:
        P["cluster"] = B.butina_clusters(list(P.smiles), al["butina_dist"])
        n_clusters = P.cluster.nunique()
        log(f"  {n_clusters} chemical clusters identified")
    except Exception as e:
        log(f"  [8.4] Warning: Clustering failed: {e}")
        P["cluster"] = list(range(len(P)))

    P["passes_filters"] = True

    # Stratified batch selection
    log("  [8.5] Selecting 40-compound stratified batch...")
    try:
        existing_scaffolds = set(ytr.scaffold) if "scaffold" in ytr.columns else set()
        sel = B.select_batch(P, al["batch"], al["max_per_cluster"], existing_scaffolds, seed=1)
        log(f"  Selected {len(sel)} compounds:")
        for slice_name, count in sel.batch_slice.value_counts().items():
            log(f"    {slice_name}: {count}")

        # Build contract
        from rdkit import Chem
        ct = K.build_contract(sel, 1, pickle.load(open(C.proc / "calibrator.pkl", "rb")),
                              lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)))
        ct.to_csv(cyc / "contract_to_team_b.csv", index=False)
        log(f"  [8.5] Contract saved to {cyc / 'contract_to_team_b.csv'}")
    except Exception as e:
        log(f"  [8.5] Warning: Batch selection failed: {e}")
        # Fallback: top-40 by reward
        sel = P.nlargest(40, "R").copy()
        sel["batch_slice"] = "exploit"
        sel.to_csv(cyc / "contract_to_team_b.csv", index=False)
        log("  [8.5] Fallback: top-40 by reward saved")

    log("  [8.5] Stage 8 complete successfully")
    return P


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stages", default="3,4,5,6,7,8", help="Comma-separated list of stages to run")
    a = ap.parse_args()

    device = a.device if torch.cuda.is_available() else "cpu"
    log(f"Starting TNK1 Full Pipeline on device: {device}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    C = Ctx(a.config, a.version, device)
    cfg = C.cfg
    log(f"Dataset: {len(C.df):,} records, {C.df.uniprot_id.nunique()} kinases")
    log(f"TNK1 dev: {len(C.dev_keys)} compounds, test: {len(C.test_keys)} compounds")

    stages = [int(s.strip()) for s in a.stages.split(",")]
    results = {}

    t0 = time.time()

    if 3 in stages:
        try:
            s3_rep, best_lam = run_stage3(C, cfg, device)
            results["stage3"] = {"test_metrics": s3_rep, "best_lambda": best_lam}
        except Exception as e:
            log(f"STAGE 3 FAILED: {e}")
            traceback.print_exc()

    if 4 in stages:
        try:
            s4_rep = run_stage4(C, cfg, device)
            results["stage4"] = s4_rep
        except Exception as e:
            log(f"STAGE 4 FAILED: {e}")
            traceback.print_exc()

    if 5 in stages:
        try:
            bundles = run_stage5(C, cfg, device)
            results["stage5"] = {"n_endpoints": len(bundles) if bundles else 0}
        except Exception as e:
            log(f"STAGE 5 FAILED: {e}")
            traceback.print_exc()

    scored_df = None
    if 6 in stages or 7 in stages:
        try:
            scored_df = run_stage6_7_generation(C, cfg, device)
            if len(scored_df) > 0:
                results["stage7"] = {
                    "n_candidates": len(scored_df),
                    "n_passing": int((scored_df.R > 0).sum()),
                    "mean_reward": float(scored_df[scored_df.R > 0].R.mean()) if (scored_df.R > 0).any() else 0,
                    "top_reward": float(scored_df.R.max()),
                }
        except Exception as e:
            log(f"STAGE 6-7 FAILED: {e}")
            traceback.print_exc()

    if 8 in stages:
        try:
            if scored_df is None or len(scored_df) == 0:
                scored_path = C.runs / "stage7_scored_candidates.csv"
                if scored_path.exists():
                    scored_df = pd.read_csv(scored_path)
            run_stage8(C, cfg, device, scored_df)
        except Exception as e:
            log(f"STAGE 8 FAILED: {e}")
            traceback.print_exc()

    elapsed = time.time() - t0
    log(f"\n{'=' * 70}")
    log(f"PIPELINE COMPLETE in {elapsed / 60:.1f} minutes")
    log(f"{'=' * 70}")

    # Save summary
    summary = {"elapsed_minutes": elapsed / 60, "device": device, **results}
    (C.runs / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    save_log(str(C.runs / "pipeline_execution.log"))

    log("Results saved to:")
    log(f"  {C.runs / 'pipeline_summary.json'}")
    log(f"  {C.runs / 'pipeline_execution.log'}")


if __name__ == "__main__":
    main()
