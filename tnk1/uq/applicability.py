"""Sec 4.3: dual-space 5-NN applicability domain. d_lat on the D-MPNN latent z_mol (Euclidean, standardised),
d_fp = 1 - Tanimoto on Morgan FP; threshold d_thr = 95th percentile of leave-one-out 5-NN distances in the training set."""
from __future__ import annotations
import numpy as np


def _knn_mean(D: np.ndarray, k: int) -> np.ndarray:
    k = min(k, D.shape[1]); return np.sort(D, axis=1)[:, :k].mean(1)


def _eud(A, B): return np.sqrt(np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T, 0))


def _tan_dist(A, B):
    ab = A @ B.T; d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - ab
    return 1 - ab / np.maximum(d, 1e-9)


class ApplicabilityDomain:
    def __init__(self, k=5, pct=95): self.k, self.pct = k, pct

    def fit(self, Z_train, FP_train):
        self.mu, self.sd = Z_train.mean(0), Z_train.std(0) + 1e-6
        self.Z, self.FP = (Z_train - self.mu) / self.sd, FP_train
        Dl, Df = _eud(self.Z, self.Z), _tan_dist(FP_train, FP_train)
        np.fill_diagonal(Dl, np.inf); np.fill_diagonal(Df, np.inf)          # leave-one-out
        # normalise latent distances by sqrt(dim) so d_lat is O(1)
        self.scale = np.sqrt(self.Z.shape[1])
        self.d_thr_lat = float(np.percentile(_knn_mean(Dl, self.k) / self.scale, self.pct))
        self.d_thr_fp = float(np.percentile(_knn_mean(Df, self.k), self.pct))
        return self

    def distances(self, Z, FP):
        Zs = (Z - self.mu) / self.sd
        return (_knn_mean(_eud(Zs, self.Z), self.k) / self.scale, _knn_mean(_tan_dist(FP, self.FP), self.k))

    def in_domain(self, Z, FP):
        dl, df = self.distances(Z, FP); return (dl <= self.d_thr_lat) & (df <= self.d_thr_fp)

    def expand(self, Z_new, FP_new):
        """Stage 8.2: add newly validated scaffolds to the reference set (thresholds recomputed on next fit)."""
        self.Z = np.vstack([self.Z, (Z_new - self.mu) / self.sd]); self.FP = np.vstack([self.FP, FP_new])
