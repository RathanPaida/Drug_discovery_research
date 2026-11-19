"""Maximum-likelihood training of the SMILES prior (the RL anchor).

The prior defines what the agent considers chemistry at all. DAP never lets the agent stray
far from it, so prior quality bounds everything downstream -- a prior that emits 60% invalid
strings cannot be rescued by any reward or any sigma.

Sized for the full 787,283-molecule store. Two things made the naive version unusable there:

  * `[vocab.encode(s) for s in corpus]` holds ~788k Python lists. `EncodedCorpus`
    stores one flat `array('i')` plus an offset index instead, and slices without
    allocating. Measured on 787,283 sequences / 49.3M tokens: 445 MB -> 201 MB resident
    (2.2x). That is the difference between fitting and swapping on a 16 GB box that is
    also holding the feature store.
  * RDKit canonicalisation of 788k SMILES is single-threaded and takes 20-40 minutes.
    `--workers` splits it across processes.
"""
from __future__ import annotations

import argparse
import time
from array import array
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

from .model import SmilesRNN
from .vocab import Vocabulary, is_lossless, pad_batch

RDLogger.DisableLog("rdApp.*")


# ────────────────────────────────────────────────────────────── corpus loading
def load_corpus(path: str, smiles_col: str = "smiles", limit: int | None = None) -> list[str]:
    import pandas as pd
    p = Path(path)
    if p.suffix == ".parquet":
        # read ONE column: the 788k store also carries descriptors and fingerprints
        import pyarrow.parquet as pq
        names = set(pq.ParquetFile(p).schema.names)
        col = smiles_col if smiles_col in names else next(
            c for c in ("canonical_smiles", "smiles") if c in names)
        df = pd.read_parquet(p, columns=[col])
    elif p.suffix in (".csv", ".tsv"):
        df = pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")
        col = smiles_col if smiles_col in df.columns else df.columns[0]
    else:                                   # plain .smi, one per line
        out = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        return out[:limit] if limit else out
    out = df[col].dropna().astype(str).tolist()
    return out[:limit] if limit else out


def _canon_chunk(chunk):
    out = []
    for s in chunk:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        c = Chem.MolToSmiles(m)
        if is_lossless(c):
            out.append(c)
    return out


def canonicalise(smiles: list[str], workers: int = 1, chunk: int = 20000) -> list[str]:
    """Canonicalise, drop unparseable, drop tokeniser round-trip failures, dedupe.

    Training on a non-canonical mix wastes capacity on spelling variants of molecules the
    model has already seen. Dropping round-trip failures matters more: a SMILES the
    tokeniser mangles decodes to a different molecule, so the model would be trained toward
    strings that do not parse back.
    """
    t0 = time.time()
    if workers > 1:
        from multiprocessing import Pool
        chunks = [smiles[i:i + chunk] for i in range(0, len(smiles), chunk)]
        with Pool(workers) as pool:
            parts = pool.map(_canon_chunk, chunks)
        flat = [s for part in parts for s in part]
    else:
        flat = _canon_chunk(smiles)
    seen, out = set(), []
    for s in flat:
        if s not in seen:
            seen.add(s); out.append(s)
    print(f"[prior] canonicalised {len(smiles):,} -> {len(out):,} unique "
          f"({time.time()-t0:.0f}s, {workers} worker(s))")
    return out


class EncodedCorpus:
    """Flat token storage: one array('i') + an offset index, sliced on demand.

    4 bytes per token, versus a pointer plus list overhead per sequence. Measured on
    the full 787,283-molecule store (49,335,715 tokens): 445 MB -> 201 MB resident.
    """

    __slots__ = ("flat", "offs")

    def __init__(self, vocab: Vocabulary, smiles: list[str]):
        flat = array("i")
        offs = array("l", [0])
        for s in smiles:
            flat.extend(vocab.encode(s))
            offs.append(len(flat))
        self.flat = np.frombuffer(flat, dtype=np.int32)
        self.offs = np.frombuffer(offs, dtype=np.int64 if offs.itemsize == 8 else np.int32)

    def __len__(self) -> int:
        return len(self.offs) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return self.flat[self.offs[i]:self.offs[i + 1]]

    @property
    def n_tokens(self) -> int:
        return len(self.flat)

    def nbytes(self) -> int:
        return self.flat.nbytes + self.offs.nbytes


# ────────────────────────────────────────────────────────────────── training
def train(corpus, out_dir, epochs=10, batch=128, lr=1e-3, hidden=512, layers=3,
          emb=128, dropout=0.2, val_frac=0.05, device="cpu", seed=42, max_len=128):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    corpus = [s for s in corpus if len(s) <= max_len - 2]
    vocab = Vocabulary.from_smiles(corpus)
    vocab.save(out / "vocab.txt")

    t0 = time.time()
    enc = EncodedCorpus(vocab, corpus)
    print(f"[prior] encoded {len(enc):,} sequences, {enc.n_tokens:,} tokens, "
          f"{enc.nbytes()/1e6:.0f} MB ({time.time()-t0:.0f}s)")

    idx = rng.permutation(len(enc))
    n_val = max(1, int(len(enc) * val_frac))
    val_i, tr_i = idx[:n_val], idx[n_val:]
    print(f"[prior] {len(corpus):,} molecules | vocab {len(vocab)} | "
          f"train {len(tr_i):,} val {len(val_i):,} | device {device}")

    model = SmilesRNN(len(vocab), emb, hidden, layers, dropout, vocab.pad).to(device)
    print(f"[prior] {sum(p.numel() for p in model.parameters()):,} parameters")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)

    def batches(order, bs):
        for i in range(0, len(order), bs):
            yield pad_batch([enc[j] for j in order[i:i + bs]], vocab.pad).to(device)

    best = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(tr_i)
        tot = nb = 0
        te = time.time()
        for seqs in batches(perm, batch):
            loss = model.nll(seqs).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        model.eval()
        with torch.no_grad():
            vl = [float(model.nll(s).mean()) for s in batches(val_i, batch)]
        v = float(np.mean(vl))
        sched.step(v)
        flag = ""
        if v < best:
            best = v; model.save(out / "prior.pt"); flag = "  *"
        print(f"  epoch {ep:>3}/{epochs}  train_nll {tot/max(nb,1):8.3f}  "
              f"val_nll {v:8.3f}  ({time.time()-te:.0f}s){flag}")
    print(f"[prior] best val_nll {best:.3f} -> {out/'prior.pt'}")
    return model, vocab


def sample_report(model, vocab, n=2000, device="cpu", train_set=None):
    seqs, _ = model.sample_with(vocab, n, device=device)
    smis = [vocab.decode(r) for r in seqs]
    mols = [Chem.MolFromSmiles(s) if s else None for s in smis]
    ok = [Chem.MolToSmiles(m) for m in mols if m is not None]
    uniq = set(ok)
    rep = {"validity": round(len(ok) / n, 4),
           "uniqueness": round(len(uniq) / max(len(ok), 1), 4)}
    if train_set is not None:
        rep["novelty"] = round(len(uniq - set(train_set)) / max(len(uniq), 1), 4)
    return rep


def main(argv=None) -> int:
    import json
    import os

    from tnk1.utils.common import load_config
    from tnk1.utils.paths import feature_file

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # `--corpus` was required, which meant the canonical corpus already named in
    # configs/default.yaml (`rl_min.corpus: unique_molecules_788k.parquet`) had to be
    # retyped as a path on every invocation -- and a mistyped one silently trains a prior
    # on the wrong chemistry. Default to the configured corpus, resolved through the
    # feature-store resolver so $TNK1_FEATURES_DIR is honoured.
    ap.add_argument("--corpus", default=None,
                    help="default: rl_min.corpus from the config, inside the feature store")
    ap.add_argument("--smiles-col", default=None,
                    help="default: rl_min.corpus_col from the config")
    ap.add_argument("--out", default=None, help="default: <processed>/rl_min")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the corpus (debugging only; omit to use every molecule)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="processes for RDKit canonicalisation")
    ap.add_argument("--qc-n", type=int, default=2000)
    ap.add_argument("--min-validity", type=float, default=None,
                    help="default: rl_min.prior.min_validity from the config")
    ap.add_argument("--allow-weak-prior", action="store_true",
                    help="exit 0 even if the validity gate fails (records the failure)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    rm = cfg.get("rl_min", {}) or {}
    pr = rm.get("prior", {}) or {}
    # Every hyperparameter falls back to the config, so a run is described by the config
    # rather than by whatever flags the operator remembered to pass.
    epochs = a.epochs if a.epochs is not None else pr.get("epochs", 40)
    batch = a.batch if a.batch is not None else pr.get("batch", 512)
    hidden = a.hidden if a.hidden is not None else pr.get("hidden", 512)
    layers = a.layers if a.layers is not None else pr.get("layers", 3)
    dropout = a.dropout if a.dropout is not None else pr.get("dropout", 0.2)
    min_validity = a.min_validity if a.min_validity is not None else pr.get("min_validity", 0.80)

    corpus_path = a.corpus
    if corpus_path is None:
        name = rm.get("corpus")
        if not name:
            raise SystemExit("no --corpus given and rl_min.corpus is unset in the config")
        corpus_path = str(feature_file(name, required=True))
    smiles_col = a.smiles_col or rm.get("corpus_col") or "smiles"
    out = Path(a.out) if a.out else Path(cfg["paths"]["processed"]) / "rl_min"
    out.mkdir(parents=True, exist_ok=True)

    raw = load_corpus(corpus_path, smiles_col, a.limit)
    print(f"[prior] loaded {len(raw):,} raw SMILES from {corpus_path} (column {smiles_col!r})")
    corpus = canonicalise(raw, workers=a.workers)
    del raw
    model, vocab = train(corpus, out, epochs=epochs, batch=batch, lr=a.lr,
                         hidden=hidden, layers=layers, emb=a.emb, dropout=dropout,
                         device=a.device)
    qc = sample_report(model, vocab, a.qc_n, a.device, corpus)
    print("[prior] sample QC:", qc)
    qc.update({"min_validity": min_validity, "corpus": str(corpus_path),
               "n_corpus": len(corpus), "epochs": epochs, "hidden": hidden,
               "layers": layers, "dropout": dropout})
    qc["passed"] = bool(qc["validity"] >= min_validity)
    Path(out, "prior_qc.json").write_text(json.dumps(qc, indent=2))

    if not qc["passed"]:
        # This is a GATE in configs/default.yaml ("DAP cannot rescue a prior that emits
        # garbage"), but the code only printed a warning and exited 0 -- so an orchestrator
        # happily proceeded to burn hours of RL on an unusable prior. Fail closed.
        msg = (f"[prior] FAIL: validity {qc['validity']:.1%} < gate {min_validity:.0%}. "
               f"DAP cannot rescue a prior this weak -- train more epochs or on a larger "
               f"corpus. Re-run with --allow-weak-prior to override deliberately.")
        print(msg)
        if not a.allow_weak_prior:
            return 1
        print("[prior] --allow-weak-prior set; continuing despite the failed gate")
    print(f"[prior] artifacts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
