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


class CDFScaler:
    """Empirical-CDF (quantile) normalisation of descriptor columns to [0, 1].

    WHY, AND WHO ELSE DOES THIS. Chemprop normalises its 200 global RDKit descriptors by
    the empirical CDF of a ~100k-compound reference population rather than by min-max or a
    z-score (Yang, Swanson, Jin, Coley et al., "Analyzing Learned Molecular Representations
    for Property Prediction", J Chem Inf Model 59:3370, 2019). The transformed value then
    means "the fraction of the reference population below this molecule", which is bounded
    in [0, 1] by construction.

    That bound is the point at N=172. `DescriptorScaler` above uses median + RobustScaler
    + a clip to +-5, which is already outlier-aware, but its SCALE is still estimated from
    the training fold's interquartile range -- and descriptors like `MaxPartialCharge` or
    `BertzCT` have ranges that a handful of molecules can move substantially when the fold
    holds 140 of them. A rank-based transform cannot be moved at all: adding one extreme
    molecule changes one rank.

    Monotone and rank-preserving, so it leaves tree models' split ORDER untouched while
    making kernel methods' distances comparable across columns -- which is exactly what a
    mixed Tanimoto/RBF kernel needs.

    `DescriptorScaler` remains the default everywhere it is already used, so no locked
    result changes; pass `scaler="cdf"` to `tnk1.zoo.benchmark.scaled_matrix` to use this.
    """

    def __init__(self, n_quantiles: int = 1000):
        self.n_quantiles = int(n_quantiles)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.med_ = np.nanmedian(X, axis=0)
        self.med_[~np.isfinite(self.med_)] = 0.0
        Xi = self._imp(X)
        # Store the sorted reference column (subsampled to n_quantiles) and interpolate
        # against it at transform time. Equivalent to sklearn's QuantileTransformer but
        # without its dependency on a fitted estimator object, so a fitted CDFScaler
        # pickles to a couple of small arrays.
        n = Xi.shape[0]
        k = min(self.n_quantiles, max(n, 1))
        qs = np.linspace(0.0, 1.0, k)
        self.quantiles_ = np.nanquantile(Xi, qs, axis=0)      # (k, d)
        self.qs_ = qs
        return self

    def _imp(self, X):
        X = np.asarray(X, dtype=np.float64).copy()
        idx = np.where(~np.isfinite(X))
        if len(idx[0]):
            X[idx] = np.take(self.med_, idx[1])
        return X

    def transform(self, X):
        Xi = self._imp(X)
        out = np.empty_like(Xi)
        for j in range(Xi.shape[1]):
            ref = self.quantiles_[:, j]
            if not np.isfinite(ref).any() or ref[-1] - ref[0] <= 0:
                # A constant descriptor carries no information; map it to the midpoint
                # rather than divide by a zero range.
                out[:, j] = 0.5
                continue
            out[:, j] = np.interp(Xi[:, j], ref, self.qs_)
        return out.astype(np.float32)


def assay_onehot(assay_types) -> np.ndarray:
    return np.array([[a == t for t in ASSAYS] for a in assay_types], dtype=np.float32)
