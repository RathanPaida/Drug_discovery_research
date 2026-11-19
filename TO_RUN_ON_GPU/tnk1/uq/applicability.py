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
    """Dual-space 5-NN applicability domain.

    `sd_floor_frac` guards the per-dimension standardisation. Z has ~300 latent dimensions
    but is fitted on only ~304 TNK1 molecules, so a good number of dimensions are nearly
    constant across the training set. `(Z - mu) / std` then divides by ~0 and a single such
    dimension can dominate the Euclidean distance: the shipped Cycle-01 pool had
    d_lat up to 240831.9 against a d_thr_lat of 1.007, which turned r_ad -- nominally a
    weight-1.5 regulariser -- into the single largest source of variance in the RL reward
    (sd(log r_ad) = 2.66, above even r_pot). Flooring each sd at `sd_floor_frac` x the
    median sd keeps low-variance dimensions from exploding while leaving informative ones
    untouched. Set sd_floor_frac=0.0 to reproduce the old (broken) behaviour exactly.
    """

    def __init__(self, k=5, pct=95, sd_floor_frac=0.05, d_lat_clip=50.0):
        self.k, self.pct = k, pct
        self.sd_floor_frac, self.d_lat_clip = sd_floor_frac, d_lat_clip

    def fit(self, Z_train, FP_train):
        self.mu = Z_train.mean(0)
        sd = Z_train.std(0)
        # Floor against the latent's GLOBAL scalar spread, not a per-dimension statistic.
        # A median- or percentile-based floor is useless exactly when it is needed most:
        # if more than half the dimensions are dead the median sd is itself ~0, so the floor
        # sits below every degenerate dimension and floors nothing. Z.std() over all entries
        # keeps a stable reference to the scale of the *informative* directions no matter
        # how many dead ones surround them.
        floor = self.sd_floor_frac * float(Z_train.std()) if self.sd_floor_frac else 0.0
        self.sd = np.maximum(sd, max(floor, 1e-6))
        self.n_floored = int((sd < floor).sum()) if floor else 0
        self.Z, self.FP = (Z_train - self.mu) / self.sd, FP_train
        Dl, Df = _eud(self.Z, self.Z), _tan_dist(FP_train, FP_train)
        np.fill_diagonal(Dl, np.inf); np.fill_diagonal(Df, np.inf)          # leave-one-out
        # normalise latent distances by sqrt(dim) so d_lat is O(1)
        self.scale = np.sqrt(self.Z.shape[1])
        self.d_thr_lat = float(np.percentile(_knn_mean(Dl, self.k) / self.scale, self.pct))
        self.d_thr_fp = float(np.percentile(_knn_mean(Df, self.k), self.pct))
        return self

    def distances(self, Z, FP):
        """Returns (d_lat, d_fp). d_fp is a Tanimoto distance and so is bounded on [0,1] by
        construction; d_lat is a standardised Euclidean distance and is only *typically* O(1),
        so it is clipped at `d_lat_clip` x d_thr_lat. Without the clip a single out-of-domain
        candidate can carry a distance five orders of magnitude above threshold into
        r_ad = exp(-(d_lat - d_thr)/0.15), which underflows to exactly 0.0 and silently
        zeroes that molecule's whole geometric-mean reward. Clipped, it still scores ~0 but
        the value stays finite and comparable, and the reward keeps a usable gradient.
        Prefer d_fp when you need a hard in/out decision."""
        Zs = (Z - self.mu) / self.sd
        d_lat = _knn_mean(_eud(Zs, self.Z), self.k) / self.scale
        clip = getattr(self, "d_lat_clip", 50.0)
        if clip:
            d_lat = np.minimum(d_lat, clip * self.d_thr_lat)
        return (d_lat, _knn_mean(_tan_dist(FP, self.FP), self.k))

    def in_domain(self, Z, FP):
        dl, df = self.distances(Z, FP); return (dl <= self.d_thr_lat) & (df <= self.d_thr_fp)

    def expand(self, Z_new, FP_new):
        """Stage 8.2: add newly validated scaffolds to the reference set (thresholds recomputed on next fit)."""
        self.Z = np.vstack([self.Z, (Z_new - self.mu) / self.sd]); self.FP = np.vstack([self.FP, FP_new])
