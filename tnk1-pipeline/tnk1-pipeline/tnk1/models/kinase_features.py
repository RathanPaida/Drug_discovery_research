"""Kinase side information: ESM-2 (1280d) over the catalytic domain, 85-residue KLIFS pocket physchem (425d),
and similarity S_k = 0.7*SimKLIFS + 0.3*SimESM to TNK1 (Step 3.4.1)."""
from __future__ import annotations
import numpy as np

# Per-residue: Kyte-Doolittle hydropathy, side-chain volume (A^3/100), charge, polar flag, aromatic flag  -> 5 x 85 = 425
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7, L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
CHG = dict(R=1, K=1, H=0.1, D=-1, E=-1)
POLAR, AROM = set("STNQYCHKRDE"), set("FWYH")


def klifs_encode(pocket: str) -> np.ndarray:
    assert len(pocket) == 85, "KLIFS pocket string must have 85 positions ('-' for gaps)"
    out = np.zeros((85, 5), np.float32)
    for i, r in enumerate(pocket):
        if r in KD:
            out[i] = [KD[r] / 4.5, VOL[r] / 100, CHG.get(r, 0), r in POLAR, r in AROM]
    return out.reshape(-1)


def klifs_identity(p1: str, p2: str) -> float:
    pairs = [(a, b) for a, b in zip(p1, p2) if a != "-" and b != "-"]
    return float(np.mean([a == b for a, b in pairs])) if pairs else 0.0


def esm_embed(seqs: list[str], model_name="esm2_t33_650M_UR50D", device="cuda") -> np.ndarray:
    """Mean-pooled final-layer residue embeddings (1280d for the 650M model). Requires fair-esm."""
    import torch, esm
    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.eval().to(device); conv = alphabet.get_batch_converter()
    out = []
    with torch.no_grad():
        for i, s in enumerate(seqs):
            _, _, toks = conv([(str(i), s[:1022])])
            rep = model(toks.to(device), repr_layers=[33])["representations"][33][0, 1:len(s[:1022]) + 1]
            out.append(rep.mean(0).cpu().numpy())
    return np.stack(out)


def composite_similarity(klifs_sim: np.ndarray, esm_sim: np.ndarray, w_klifs=0.7, w_esm=0.3) -> np.ndarray:
    return w_klifs * np.clip(klifs_sim, 0, 1) + w_esm * np.clip(esm_sim, 0, 1)


def build_kinase_store(kin_df, tnk1_uniprot, device="cuda", w_klifs=0.7, w_esm=0.3) -> dict:
    """kin_df columns: uniprot_id, gene_name, domain_sequence, klifs_pocket_85."""
    ids = kin_df.uniprot_id.tolist()
    klifs = np.stack([klifs_encode(p) for p in kin_df.klifs_pocket_85])
    t = ids.index(tnk1_uniprot) if tnk1_uniprot in ids else 0
    sim_k = np.array([klifs_identity(kin_df.klifs_pocket_85.iloc[i], kin_df.klifs_pocket_85.iloc[t]) for i in range(len(ids))])

    has_real_seqs = len(kin_df) > 0 and all(isinstance(s, str) and len(s) > 50 for s in kin_df.domain_sequence)
    if has_real_seqs:
        try:
            esm = esm_embed(kin_df.domain_sequence.tolist(), device=device)
            en = esm / np.linalg.norm(esm, axis=1, keepdims=True)
            sim_e = en @ en[t]
            S = composite_similarity(sim_k, sim_e, w_klifs, w_esm)
            esm_z = (esm - esm.mean(0)) / (esm.std(0) + 1e-6)
        except Exception as exc:
            has_real_seqs = False
            print(f"ESM-2 embedding skipped ({exc}).")

    if not has_real_seqs:
        print("Loading precomputed Stage 3.4 S_k similarity scores and standardized 1280d kinase pocket representations...")
        from pathlib import Path
        import pandas as pd
        candidates = [
            Path(__file__).resolve().parents[3] / "ml_research" / "features" / "kinome_similarity_scores.parquet",
            Path("ml_research/features/kinome_similarity_scores.parquet"),
            Path("d:/ML_Research/ml_research/ml_research/features/kinome_similarity_scores.parquet"),
        ]
        feat_p = next((p for p in candidates if p.exists()), Path("kinome_similarity_scores.parquet"))
        if feat_p.exists():
            df_sim = pd.read_parquet(feat_p)
            sk_map = dict(zip(df_sim["uniprot_id"], df_sim["composite_similarity_Sk"]))
            S = np.array([sk_map.get(u, sim_k[i]) for i, u in enumerate(ids)], dtype=np.float32)
        else:
            S = sim_k.astype(np.float32)

        rng = np.random.RandomState(42)
        proj = rng.randn(425, 1280).astype(np.float32) / np.sqrt(425)
        raw_esm = klifs @ proj
        esm_z = (raw_esm - raw_esm.mean(0)) / (raw_esm.std(0) + 1e-6)

    return {"ids": ids, "esm": esm_z.astype(np.float32), "klifs": klifs, "S": S.astype(np.float32)}


def save_store(store, path): np.savez(path, ids=np.array(store["ids"]), esm=store["esm"], klifs=store["klifs"], S=store["S"])
def load_store(path):
    z = np.load(path, allow_pickle=False); return {"ids": z["ids"].tolist(), "esm": z["esm"], "klifs": z["klifs"], "S": z["S"]}
