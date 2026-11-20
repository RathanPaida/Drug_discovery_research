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


# ---------------------------------------------------------------------------------------
# The bounding-box applicability domain, and why the one above is probably the wrong kind.
# ---------------------------------------------------------------------------------------


class BoundingBoxAD:
    """Per-feature min/max box over the training set. In/out is a hard, cheap decision.

    WHY THIS EXISTS, AND WHY IT MATTERS MORE THAN IT LOOKS.

    Langevin, Grebner, Guessregen, Sauer, Li, Matter & Bianciotto, "Impact of Applicability
    Domains to Generative Artificial Intelligence", ACS Omega 8:23148 (2023), tested 16
    applicability-domain definitions on a SMILES LSTM generator across four targets. Their
    findings bear directly on `ApplicabilityDomain` above:

      * The CLASSIC QSAR AD -- maximum Tanimoto similarity to the training set -- FAILS.
        Verbatim: "some of the classic applicability domain definitions used in QSAR
        modeling, such as ensuring similarity to the training set (corresponding to the
        applicability domain definitions 'maxsim ECFP4', 'maxsim ECFP6', and 'maxsim ap'),
        fail to generate an high enough proportion of drug-like molecules."
        The dual-space 5-NN domain above is a distance-to-training-set AD of exactly that
        family.
      * QED-based and unwanted-substructure ADs were "not very efficient either", and
        "measures of drug-likeness commonly used, such as QED, can be optimized in
        unintended ways and do not constitute a valid applicability domain for generative
        models either."
      * The ONLY definition that worked across every task was a BOUNDING BOX on
        physicochemical descriptors AND on COUNT-BASED Morgan r=2 fingerprints.
      * Count-based, not binary, is load-bearing. Their Figure 3 is titled "Limitations of
        binary fingerprints to discriminate unusual chemical moieties": a binary ECFP
        cannot distinguish "has a carboxylic acid" from "has five carboxylic acids".
        **The v3 generator's permeability collapsed because it drifted toward carboxylic
        acids (Caco-2 sigmoid 0.066). A binary-fingerprint domain is structurally blind to
        precisely that failure; a count box is not.**
      * A box, not a convex hull, for a stated reason: the hull is Omega(n^d) and
        "impractical as soon as the number of descriptors used grows".

    They also report two things to plan around, which is why they are in this docstring and
    not just in a paper:

      * "Counterintuitively, good applicability domains are associated with lower scores."
        EXPECT THE REWARD CURVE TO DROP when this replaces a similarity AD, and expect the
        score distribution to widen. That is the signature of success. Do not tune it back out.
      * "good applicability domain definitions lead to higher diversity in addition to
        better drug-likeness" -- so the AD fix and the diversity fix are the same fix.

    USED AS A MULTIPLICATIVE 0/1 TERM, not as a soft penalty. Same paper: "During
    generation, the AD is taken into account in the reward function as a multiplicative
    term that yields 0 if the molecule generated is out of the AD and 1 otherwise." And the
    conclusion that licenses it: "This reward hacking behavior is prevented by good AD
    definitions: the constraints associated with good applicability domains prevent
    generative algorithms for overexploiting a specific region of chemical space, leading
    to a wider distribution of scores."

    Parameters
    ----------
    percentile : float | None
        None uses the strict observed min/max, as the paper's default does. A value like 1
        uses the 1st-99th percentiles instead, which the paper names as the more
        outlier-robust option and leaves to future work. At 172 training molecules one
        extreme compound can widen a strict box enough to admit a region nothing supports,
        so `percentile=1` is the safer choice here and the reason this knob exists.
    slack : float
        Fractional widening of each side of the box, relative to that feature's range.
        0.0 is the paper's box. A small slack avoids rejecting a molecule for being 0.1%
        outside on one of 200 features.
    """

    def __init__(self, percentile: float | None = 1.0, slack: float = 0.0):
        self.percentile = percentile
        self.slack = float(slack)
        self.lo = None
        self.hi = None

    def fit(self, X):
        X = np.asarray(X, float)
        if self.percentile is None:
            lo, hi = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
        else:
            p = float(self.percentile)
            lo = np.nanpercentile(X, p, axis=0)
            hi = np.nanpercentile(X, 100 - p, axis=0)
        rng = np.where(np.isfinite(hi - lo), hi - lo, 0.0)
        self.lo = lo - self.slack * rng
        self.hi = hi + self.slack * rng
        # A feature that is constant in training admits only that value; without this the
        # box would be a point and every generated molecule would fail on it.
        flat = ~np.isfinite(self.hi - self.lo) | (self.hi - self.lo <= 0)
        self.lo = np.where(flat, -np.inf, self.lo)
        self.hi = np.where(flat, np.inf, self.hi)
        self.n_features_ = X.shape[1]
        return self

    def violations(self, X) -> np.ndarray:
        """Per-molecule count of features outside the box. 0 means in-domain.

        A COUNT rather than a boolean, deliberately: "out of domain on 1 of 215 features"
        and "out on 60 of 215" are very different molecules, and the count is what makes a
        graded diagnostic possible even though the reward term itself is 0/1.
        """
        X = np.asarray(X, float)
        below = X < self.lo[None, :]
        above = X > self.hi[None, :]
        # A NaN feature cannot be judged; count it as a violation rather than a pass, which
        # is the fail-closed direction this pipeline requires everywhere else.
        nan = ~np.isfinite(X)
        return (below | above | nan).sum(axis=1)

    def in_domain(self, X, max_violations: int = 0) -> np.ndarray:
        return self.violations(X) <= int(max_violations)

    def reward_term(self, X, max_violations: int = 0) -> np.ndarray:
        """The multiplicative 0/1 term Langevin et al. put inside the reward."""
        return self.in_domain(X, max_violations).astype(float)

    def worst_features(self, X, names=None, top=10) -> list[dict]:
        """Which features the out-of-domain molecules most often violate.

        The diagnostic that turns "rejected" into "rejected because it has four more
        carboxylic acids than anything in the training set". That is the information needed
        to add a counter-term against the exploited channel rather than a generic penalty.
        """
        X = np.asarray(X, float)
        bad = (X < self.lo[None, :]) | (X > self.hi[None, :]) | ~np.isfinite(X)
        counts = bad.sum(axis=0)
        order = np.argsort(-counts)[:top]
        return [{"feature": (names[j] if names is not None and j < len(names) else int(j)),
                 "n_violating": int(counts[j]),
                 "frac": float(counts[j] / max(len(X), 1)),
                 "lo": float(self.lo[j]), "hi": float(self.hi[j])}
                for j in order if counts[j] > 0]


#: The physicochemical descriptor set Langevin et al. (2023) used, verbatim from the paper:
#: "the number of hydrogen bonds donors and acceptors, the number of rings, the number of
#: rotatable bonds, the total polar surface area, Crippen descriptors (clogP and molar
#: refractivity), the molecular weight, the fraction of sp3 carbons, the ratio of atoms in
#: the Murcko scaffold on the total number of heavy atoms, the number of heavy atoms, the
#: maximum and minimum ring size, the minimal, maximal and total charge, and the number of
#: chiral centers."
#:
#: Mapped to RDKit callables in `tnk1.admet.filters.physchem_ad_descriptors`. Listed here,
#: in the numpy-only module, so the set is reviewable without RDKit installed.
PHYSCHEM_AD_FEATURES = (
    "NumHDonors", "NumHAcceptors", "RingCount", "NumRotatableBonds", "TPSA",
    "MolLogP", "MolMR", "MolWt", "FractionCSP3", "ScaffoldAtomRatio", "HeavyAtomCount",
    "MaxRingSize", "MinRingSize", "MinPartialCharge", "MaxPartialCharge",
    "TotalFormalCharge", "NumChiralCenters",
)
