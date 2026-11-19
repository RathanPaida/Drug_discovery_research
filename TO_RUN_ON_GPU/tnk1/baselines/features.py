"""Step 2.1: Morgan count FP (2048) + 200 RDKit 2D descriptors + assay one-hot."""
from __future__ import annotations
import numpy as np

ASSAYS = ["IC50", "Ki", "Kd"]


def morgan_count(smiles, radius=2, n_bits=2048) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles):
        out[i] = gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s))
    return out


def rdkit_descriptors(smiles, n=200) -> np.ndarray:
    """First `n` descriptors of rdkit.Chem.Descriptors._descList (includes MolLogP, TPSA, MolWt, NumHDonors,
    NumHAcceptors, BalabanJ, BertzCT, partial-charge descriptors). NaN/inf left for the imputer."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    fns = [f for _, f in Descriptors._descList][:n]
    out = np.full((len(smiles), len(fns)), np.nan, dtype=np.float64)
    n_bad_mols = 0
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            # row stays all-NaN; DescriptorScaler imputes to the train-fold median, which
            # is a neutral value (NOT an optimistic one), so this is safe -- but it must
            # be visible rather than indistinguishable from a molecule that parsed.
            n_bad_mols += 1
            continue
        for j, f in enumerate(fns):
            try:
                out[i, j] = f(m)      # individual descriptors legitimately fail on some
            except Exception:         # valences/elements; NaN -> median-imputed downstream
                pass
    out[~np.isfinite(out)] = np.nan
    if n_bad_mols:
        import warnings
        warnings.warn(
            f"rdkit_descriptors: {n_bad_mols}/{len(smiles)} SMILES failed to parse; "
            f"their descriptor rows are all-NaN and will be median-imputed",
            RuntimeWarning, stacklevel=2,
        )
    return out


class DescriptorScaler:
    """Median impute (train-fold median) + RobustScaler; clipped to +-5 for stability."""
    def fit(self, X):
        from sklearn.preprocessing import RobustScaler
        self.med = np.nanmedian(X, axis=0); self.med[~np.isfinite(self.med)] = 0.0
        self.sc = RobustScaler().fit(self._imp(X)); return self

    def _imp(self, X):
        X = X.copy(); idx = np.where(~np.isfinite(X)); X[idx] = np.take(self.med, idx[1]); return X

    def transform(self, X):
        return np.clip(self.sc.transform(self._imp(X)), -5, 5).astype(np.float32)


def assay_onehot(assay_types) -> np.ndarray:
    return np.array([[a == t for t in ASSAYS] for a in assay_types], dtype=np.float32)
