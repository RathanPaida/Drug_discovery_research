"""The missing step: data/raw + data/processed  ->  data/features/.

Nothing in the repo rebuilt `data/features/`. `tnk1_harvest.py` produced
`data/raw/tnk1_data/`, Stage 1 consumed `data/features/`, and the assembly
between them existed only as scattered pieces. This module is that assembly,
and it is written so the hard acceptance test can actually pass: every output
is deterministic given the inputs (fixed bit sizes, fixed descriptor list in a
fixed order, sorted rows, no hash seeds, no timestamps inside the arrays), so
re-running it reproduces the manifest's SHA-256 values byte for byte.

Determinism notes -- these are the things that silently break a checksum test:
  * rows are sorted by InChIKey before featurisation, never by dict order;
  * the descriptor list is an explicit tuple, not `Descriptors.descList`
    (which changes with the RDKit version and is not even stably ordered);
  * fingerprints use `rdFingerprintGenerator` with explicit radius/fpSize;
  * floats are cast to float32 once, at the end;
  * .npz is written with `np.savez` (not `savez_compressed`) because zlib
    output varies across zlib builds, which would change the file hash on a
    different machine while the arrays were identical.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .util import LOG, Layout, Manifest, sha256_file, write_parquet

#: Explicit, ordered descriptor set. Append only -- never reorder or remove,
#: or every previously recorded checksum becomes meaningless.
DESCRIPTORS: tuple[str, ...] = (
    "MolWt", "HeavyAtomCount", "NumHAcceptors", "NumHDonors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "NumAliphaticRings", "FractionCSP3",
    "TPSA", "MolLogP", "MolMR", "BalabanJ", "BertzCT", "Chi0v", "Chi1v",
    "Chi2v", "Chi3v", "Chi4v", "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
    "LabuteASA", "NumHeteroatoms", "NumSaturatedRings", "NOCount", "NHOHCount",
    "qed",
)

FP_RADIUS = 2
FP_BITS = 2048


def _descriptor_matrix(smiles: list[str]) -> np.ndarray:
    from rdkit import Chem                                          # noqa: PLC0415
    from rdkit.Chem import Descriptors                              # noqa: PLC0415

    fns = []
    for name in DESCRIPTORS:
        fn = getattr(Descriptors, name, None)
        if fn is None:
            raise RuntimeError(
                f"descriptor `{name}` not available in this RDKit build -- "
                f"pin the RDKit version, do not silently drop a column")
        fns.append(fn)

    out = np.full((len(smiles), len(fns)), np.nan, dtype="float64")
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for j, fn in enumerate(fns):
            try:
                out[i, j] = float(fn(mol))
            except Exception:                                      # noqa: BLE001
                out[i, j] = np.nan
    n_bad = int(np.isnan(out).any(axis=1).sum())
    if n_bad:
        LOG.warning("%d/%d rows have at least one NaN descriptor", n_bad, len(smiles))
    return out.astype("float32")


def _fingerprint_matrix(smiles: list[str], *, radius: int = FP_RADIUS,
                        n_bits: int = FP_BITS) -> np.ndarray:
    from rdkit import Chem                                          # noqa: PLC0415
    from rdkit.Chem import rdFingerprintGenerator                   # noqa: PLC0415
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype="uint8")
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        out[i] = gen.GetFingerprintAsNumPy(mol).astype("uint8")
    return out


# --------------------------------------------------------------------------- #
# graph tensors for the D-MPNN
# --------------------------------------------------------------------------- #
ATOM_SYMBOLS = ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si",
                "Se", "other")
HYBRIDIZATIONS = ("SP", "SP2", "SP3", "SP3D", "SP3D2", "other")
BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")


def atom_features(atom) -> list[float]:
    sym = atom.GetSymbol()
    f = [1.0 if sym == s else 0.0 for s in ATOM_SYMBOLS[:-1]]
    f.append(0.0 if sym in ATOM_SYMBOLS[:-1] else 1.0)
    hyb = str(atom.GetHybridization())
    f += [1.0 if hyb == h else 0.0 for h in HYBRIDIZATIONS[:-1]]
    f.append(0.0 if hyb in HYBRIDIZATIONS[:-1] else 1.0)
    f += [
        atom.GetTotalDegree() / 4.0,
        atom.GetFormalCharge() / 2.0,
        atom.GetTotalNumHs() / 4.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetMass() / 100.0,
        float(atom.GetChiralTag() != 0),
    ]
    return f


def bond_features(bond) -> list[float]:
    bt = str(bond.GetBondType())
    f = [1.0 if bt == b else 0.0 for b in BOND_TYPES]
    f += [float(bond.GetIsConjugated()), float(bond.IsInRing()),
          float(bond.GetStereo() != 0)]
    return f


def graph_tensors(smiles: list[str]) -> dict[str, np.ndarray]:
    """Flat, concatenated graph arrays with offset indices.

    Flat-with-offsets rather than a list of per-molecule arrays: it is one .npz,
    it loads in one read, and the batching logic in the model stays trivial.
    """
    from rdkit import Chem                                          # noqa: PLC0415

    atom_feats: list[list[float]] = []
    bond_feats: list[list[float]] = []
    edges: list[tuple[int, int]] = []
    atom_offsets = [0]
    bond_offsets = [0]
    failed: list[int] = []

    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed.append(idx)
            atom_offsets.append(atom_offsets[-1])
            bond_offsets.append(bond_offsets[-1])
            continue
        base = atom_offsets[-1]
        for a in mol.GetAtoms():
            atom_feats.append(atom_features(a))
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx() + base, b.GetEndAtomIdx() + base
            bf = bond_features(b)
            # Both directions, so a directed message-passing net needs no extra
            # bookkeeping.
            edges.append((i, j)); bond_feats.append(bf)
            edges.append((j, i)); bond_feats.append(bf)
        atom_offsets.append(base + mol.GetNumAtoms())
        bond_offsets.append(bond_offsets[-1] + 2 * mol.GetNumBonds())

    if failed:
        LOG.warning("%d SMILES failed graph construction (empty graphs kept as "
                    "zero-atom entries so row indices stay aligned)", len(failed))
    return {
        "atom_features": np.asarray(atom_feats, dtype="float32")
                         if atom_feats else np.zeros((0, len(atom_features_dim())),
                                                     dtype="float32"),
        "bond_features": np.asarray(bond_feats, dtype="float32")
                         if bond_feats else np.zeros((0, 7), dtype="float32"),
        "edge_index": np.asarray(edges, dtype="int32").T
                      if edges else np.zeros((2, 0), dtype="int32"),
        "atom_offsets": np.asarray(atom_offsets, dtype="int64"),
        "bond_offsets": np.asarray(bond_offsets, dtype="int64"),
        "failed_rows": np.asarray(failed, dtype="int64"),
    }


def atom_features_dim() -> list[float]:
    from rdkit import Chem                                          # noqa: PLC0415
    mol = Chem.MolFromSmiles("C")
    return atom_features(mol.GetAtomWithIdx(0))


# --------------------------------------------------------------------------- #
# the build
# --------------------------------------------------------------------------- #
def build_features(layout: Layout, manifest: Manifest, *,
                   curated_name: str = "tnk1_curated.parquet",
                   offtarget_name: str = "offtargets_curated.parquet",
                   kinome_name: str | None = "kinome_measurements_v3.parquet",
                   kinases_csv: str = "kinases.csv",
                   with_graphs: bool = True) -> dict[str, Path]:
    """Rebuild data/features/ deterministically from the curated tables."""
    layout.features.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    def _read(p: Path) -> pd.DataFrame | None:
        for cand in (p, p.with_suffix(".csv.gz")):
            if cand.exists():
                return (pd.read_parquet(cand) if cand.suffix == ".parquet"
                        else pd.read_csv(cand))
        LOG.warning("missing input %s", p.name)
        return None

    jobs: list[tuple[str, pd.DataFrame]] = []
    for label, name in (("tnk1", curated_name), ("offtargets", offtarget_name),
                        ("kinome", kinome_name)):
        if name is None:
            continue
        df = _read(layout.processed / name)
        if df is not None and not df.empty:
            jobs.append((label, df))

    if not jobs:
        raise RuntimeError("no curated tables found in data/processed -- run "
                           "tier1/tier2/tier3 first")

    for label, df in jobs:
        if "canonical_smiles" not in df.columns:
            LOG.error("%s has no canonical_smiles column; skipping", label)
            continue
        # Determinism: sort before featurising.
        sort_keys = [c for c in ("inchikey", "canonical_smiles", "target_gene",
                                 "accession", "endpoint") if c in df.columns]
        df = df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

        uniq = (df[["canonical_smiles"]].drop_duplicates()
                .sort_values("canonical_smiles", kind="mergesort")
                .reset_index(drop=True))
        smiles = uniq["canonical_smiles"].tolist()
        LOG.info("featurising %s: %d rows, %d unique structures",
                 label, len(df), len(smiles))

        fp = _fingerprint_matrix(smiles)
        desc = _descriptor_matrix(smiles)
        arrays = {"ecfp4": fp, "descriptors": desc}
        if with_graphs:
            arrays.update(graph_tensors(smiles))

        npz = layout.features / f"{label}_features.npz"
        np.savez(npz, **arrays)       # NOT savez_compressed -- see module docstring
        manifest.record(npz, tier="features", source="derived", rows=len(smiles),
                        notes=f"ecfp4 r={FP_RADIUS} {FP_BITS}b + "
                              f"{len(DESCRIPTORS)} descriptors"
                              + (" + graph tensors" if with_graphs else ""))
        out[f"{label}_npz"] = npz

        # The row-order key. Without this the arrays are unusable and any
        # downstream join is a guess.
        idx = layout.features / f"{label}_index.parquet"
        write_parquet(uniq.assign(row=np.arange(len(uniq), dtype="int64")),
                      idx, manifest, tier="features", source="derived",
                      notes="row order of every array in "
                            f"{npz.name}; join on canonical_smiles")
        out[f"{label}_index"] = idx

        labels_path = layout.features / f"{label}_labels.parquet"
        write_parquet(df, labels_path, manifest, tier="features",
                      source="derived",
                      notes="label table aligned to the index by "
                            "canonical_smiles")
        out[f"{label}_labels"] = labels_path

    kin = layout.raw / kinases_csv
    if kin.exists():
        from .descriptors import pocket_features                    # noqa: PLC0415
        pf = pocket_features(kin)
        p = layout.features / "kinase_pocket_features.parquet"
        write_parquet(pf.sort_values("uniprot", kind="mergesort"), p, manifest,
                      tier="features", source="derived",
                      notes="85-residue KLIFS pocket as integer residue indices; "
                            "feed as an embedding, not as a target one-hot")
        out["pocket"] = p

    spec = {
        "descriptors": list(DESCRIPTORS),
        "fp_radius": FP_RADIUS, "fp_bits": FP_BITS,
        "atom_feature_dim": len(atom_features_dim()),
        "bond_feature_dim": 7,
        "atom_symbols": list(ATOM_SYMBOLS),
        "hybridizations": list(HYBRIDIZATIONS),
        "bond_types": list(BOND_TYPES),
        "notes": "changing anything in this file invalidates every recorded "
                 "feature checksum; bump a version instead of editing in place",
        "version": 1,
    }
    spec_path = layout.features / "feature_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True))
    manifest.record(spec_path, tier="features", source="derived",
                    notes="featuriser contract")
    out["spec"] = spec_path
    return out


def verify_features(layout: Layout, manifest: Manifest) -> int:
    """The hard acceptance test: rebuilt files must match recorded SHA-256."""
    ok, bad, missing = manifest.verify()
    LOG.info("manifest verify: %d ok, %d mismatched, %d missing",
             len(ok), len(bad), len(missing))
    for k in bad:
        LOG.error("  CHECKSUM MISMATCH %s", k)
    for k in missing:
        LOG.error("  MISSING          %s", k)
    if bad:
        LOG.error("A mismatch on a derived feature file means the featuriser is "
                  "not deterministic. Usual causes: RDKit version change, an "
                  "edited DESCRIPTORS tuple, savez_compressed, or rows not "
                  "sorted before featurisation.")
    return 1 if (bad or missing) else 0
