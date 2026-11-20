"""Tier 5 -- the unlabelled corpus that trains the generator prior and backs
the novelty scan.

Two artefacts:
  * `unique_molecules.parquet`   -- standardised, deduplicated structures
  * `unique_molecules_fp.npy`    -- packed 2048-bit Morgan fingerprints,
                                    row-aligned to the parquet

The fingerprint matrix is stored packed (`np.packbits`), which is 8x smaller
than uint8 and still fast to Tanimoto against via popcount. Row alignment to the
parquet is asserted on write, because a novelty scan that silently reports the
wrong neighbour is worse than no novelty scan.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from .util import LOG, Http, Layout, Manifest, human, try_import, write_parquet

MOSES_URL = ("https://media.githubusercontent.com/media/molecularsets/moses/"
             "master/data/dataset_v1.csv")


# --------------------------------------------------------------------------- #
# drug-likeness
# --------------------------------------------------------------------------- #
def druglike_mask(smiles: pd.Series, *, max_mw: float = 600.0,
                  min_mw: float = 150.0, max_logp: float = 6.0,
                  max_rotb: int = 12, max_hba: int = 12, max_hbd: int = 6,
                  max_tpsa: float = 160.0, require_ro5_violations: int = 2
                  ) -> tuple[pd.Series, pd.DataFrame]:
    """Lead-like-to-drug-like window, with the computed properties returned.

    Deliberately wider than Lipinski: kinase inhibitors routinely sit at
    MW 450-550 with 3-4 rings, and a strict Ro5 filter would throw away the very
    chemotypes this project is trying to generate around.
    """
    from rdkit import Chem                                          # noqa: PLC0415
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors   # noqa: PLC0415

    recs = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            recs.append((np.nan,) * 8)
            continue
        recs.append((
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumRings(mol),
            mol.GetNumHeavyAtoms(),
        ))
    props = pd.DataFrame(recs, index=smiles.index,
                         columns=["mw", "logp", "rotb", "hba", "hbd", "tpsa",
                                  "rings", "heavy_atoms"])
    violations = (
        (props["mw"] > 500).astype(int) + (props["logp"] > 5).astype(int)
        + (props["hba"] > 10).astype(int) + (props["hbd"] > 5).astype(int)
    )
    props["ro5_violations"] = violations
    mask = (
        props["mw"].between(min_mw, max_mw)
        & (props["logp"] <= max_logp)
        & (props["rotb"] <= max_rotb)
        & (props["hba"] <= max_hba)
        & (props["hbd"] <= max_hbd)
        & (props["tpsa"] <= max_tpsa)
        & (violations <= require_ro5_violations)
    ).fillna(False)
    LOG.info("drug-like filter: %d/%d kept (%.1f%%)",
             int(mask.sum()), len(mask), 100 * mask.mean())
    return mask, props


# --------------------------------------------------------------------------- #
# corpus build
# --------------------------------------------------------------------------- #
def build_corpus(chemreps_gz: Path, layout: Layout, manifest: Manifest, *,
                 extra_smi: list[Path] | None = None,
                 limit: int | None = None,
                 drug_like: bool = True,
                 batch: int = 200_000) -> tuple[Path, pd.DataFrame]:
    """Stream chemreps.txt.gz (+ any extra .smi files) -> standardised corpus."""
    from .curate import Standardizer                                # noqa: PLC0415
    std = Standardizer(min_heavy=8, max_heavy=70)

    seen: set[str] = set()
    keep_smiles: list[str] = []
    keep_keys: list[str] = []
    keep_src: list[str] = []

    def ingest(rows: list[tuple[str, str]]) -> None:
        for smi, src in rows:
            res = std(smi)
            if res.smiles is None or res.inchikey is None:
                continue
            if res.inchikey in seen:
                continue
            seen.add(res.inchikey)
            keep_smiles.append(res.smiles)
            keep_keys.append(res.inchikey)
            keep_src.append(src)

    buf: list[tuple[str, str]] = []
    n_read = 0
    LOG.info("reading %s", chemreps_gz)
    with gzip.open(chemreps_gz, "rt", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            smi_i = header.index("canonical_smiles")
        except ValueError:
            smi_i = 1
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= smi_i:
                continue
            buf.append((parts[smi_i], "chembl"))
            n_read += 1
            if len(buf) >= batch:
                ingest(buf); buf.clear()
                LOG.info("  corpus: read %d kept %d", n_read, len(keep_smiles))
            if limit and n_read >= limit:
                break
    ingest(buf); buf.clear()

    for p in (extra_smi or []):
        LOG.info("reading extra corpus file %s", p)
        opener = gzip.open if p.name.endswith(".gz") else open
        with opener(p, "rt", errors="replace") as fh:               # type: ignore[operator]
            rows = []
            for line in fh:
                tok = line.split()
                if not tok:
                    continue
                rows.append((tok[0], p.stem))
                if len(rows) >= batch:
                    ingest(rows); rows.clear()
            ingest(rows)

    df = pd.DataFrame({"canonical_smiles": keep_smiles,
                       "inchikey": keep_keys,
                       "corpus_source": keep_src})
    LOG.info("corpus: %d unique structures from %d input records", len(df), n_read)

    if drug_like:
        mask, props = druglike_mask(df["canonical_smiles"])
        df = pd.concat([df, props], axis=1)
        df["drug_like"] = mask.values

    out = write_parquet(df, layout.processed / "unique_molecules.parquet",
                        manifest, tier="5", source="chembl_chemreps",
                        notes="standardised, InChIKey-deduplicated corpus; "
                              "`drug_like` flags the generator-prior subset")
    return out, df


# --------------------------------------------------------------------------- #
# fingerprints
# --------------------------------------------------------------------------- #
def morgan_matrix(smiles: list[str], *, n_bits: int = 2048, radius: int = 2,
                  batch: int = 50_000) -> np.ndarray:
    """Packed (n, n_bits/8) uint8 Morgan fingerprint matrix."""
    from rdkit import Chem                                          # noqa: PLC0415
    from rdkit.Chem import rdFingerprintGenerator                   # noqa: PLC0415

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    packed = np.zeros((len(smiles), n_bits // 8), dtype=np.uint8)
    for start in range(0, len(smiles), batch):
        chunk = smiles[start:start + batch]
        block = np.zeros((len(chunk), n_bits), dtype=np.uint8)
        for i, smi in enumerate(chunk):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = gen.GetFingerprintAsNumPy(mol)
            block[i] = fp.astype(np.uint8)
        packed[start:start + len(chunk)] = np.packbits(block, axis=1)
        LOG.info("  fingerprints %d/%d", start + len(chunk), len(smiles))
    return packed


def build_fingerprints(df: pd.DataFrame, layout: Layout, manifest: Manifest, *,
                       smiles_col: str = "canonical_smiles",
                       n_bits: int = 2048) -> Path:
    smiles = df[smiles_col].tolist()
    packed = morgan_matrix(smiles, n_bits=n_bits)
    if packed.shape[0] != len(df):
        raise RuntimeError(
            f"fingerprint/parquet row misalignment: {packed.shape[0]} vs {len(df)}")
    path = layout.features / f"unique_molecules_fp{n_bits}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, packed)
    # Row-alignment key: the InChIKey order the matrix was built from.
    keys = layout.features / f"unique_molecules_fp{n_bits}.keys.txt.gz"
    with gzip.open(keys, "wt") as fh:
        fh.write("\n".join(df["inchikey"].astype(str)))
    manifest.record(path, tier="5", source="derived", rows=len(df),
                    notes=f"np.packbits Morgan r=2 {n_bits}-bit, row-aligned to "
                          f"{keys.name}")
    manifest.record(keys, tier="5", source="derived", rows=len(df),
                    notes="InChIKey row order for the fingerprint matrix")
    LOG.info("wrote %s %s (%s)", path.name, packed.shape, human(path.stat().st_size))
    return path


def tanimoto_vs_corpus(query_fp: np.ndarray, corpus_packed: np.ndarray
                       ) -> np.ndarray:
    """Max Tanimoto of one packed query fp against every corpus row."""
    q = np.unpackbits(query_fp.reshape(1, -1), axis=1).astype(np.uint16)
    c = np.unpackbits(corpus_packed, axis=1).astype(np.uint16)
    inter = c @ q.ravel()
    union = c.sum(1) + q.sum() - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def download_moses(http: Http, layout: Layout, manifest: Manifest, *,
                   force: bool = False) -> Path | None:
    """MOSES dataset_v1.csv -- the standard generative-model benchmark split.

    Optional. Useful only to sanity-check that a generator trains at all; it is
    a ZINC-derived lead-like set and is NOT kinase-relevant, so do not pretrain
    the production prior on it alone.
    """
    try:
        p = http.download(MOSES_URL, layout.download_dir("moses") / "dataset_v1.csv",
                          force=force)
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("MOSES download failed: %s", exc)
        return None
    manifest.record(p, tier="5", source="moses", url=MOSES_URL,
                    notes="generative benchmark; not kinase-specific")
    return p
