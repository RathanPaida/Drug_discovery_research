"""Molecular representations, as a registry.

WHY A REGISTRY. `tnk1/baselines/features.py` hardcodes one representation -- 2048-bit
Morgan counts plus the first 200 RDKit descriptors plus an assay one-hot -- and every
Stage-2 model gets that or nothing. That is a fixed choice made once, and the whole
retrospective benchmark inherits it: when the Morgan-count random forest beat the D-MPNN
ensemble, nothing in the repository could say whether a different fingerprint would have
beaten it further. A representation is a hyperparameter, so it belongs in a sweep.

Each entry here is a pure function `list[str] -> np.ndarray (n, d)` plus metadata. They are
deliberately uniform so `tnk1/zoo/benchmark.py` can cross any model with any
representation, and so a new one is a single dict entry rather than an edit to every
caller.

FAIL-CLOSED ON UNPARSEABLE INPUT. A SMILES RDKit cannot parse yields an all-zero
fingerprint row, which for a Tanimoto kernel is a molecule similar to nothing and for a
tree model is a legitimate-looking point at the origin. Every featuriser here therefore
returns `(X, ok)` where `ok` is a boolean mask of the rows that parsed, and
`featurize_or_raise` refuses to hand back a matrix with silent dead rows. This is the same
discipline the audit imposed on the ADMET endpoints: absence of evidence must not be
scored as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------- metadata


@dataclass(frozen=True)
class Representation:
    """One molecular representation.

    `fn` takes a list of SMILES and returns `(X, ok)`.
    `kind` groups representations by what kernel/model they suit:
        "count"  -- integer count vectors; Tanimoto/MinMax kernels are defined on these
        "binary" -- 0/1 bit vectors; Jaccard/Tanimoto on bits
        "dense"  -- real-valued descriptors; need scaling, suit RBF/Matern kernels and GBDTs
    `needs` lists importable modules required beyond RDKit, so `available()` can report
    what this environment can actually compute instead of failing mid-sweep.
    """
    name: str
    fn: Callable[[list[str]], tuple[np.ndarray, np.ndarray]]
    kind: str
    dim: int | None = None
    needs: tuple[str, ...] = ()
    cite: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


REGISTRY: dict[str, Representation] = {}


def register(rep: Representation) -> Representation:
    if rep.name in REGISTRY:
        raise ValueError(f"representation {rep.name!r} is already registered")
    REGISTRY[rep.name] = rep
    return rep


def available(names=None) -> dict[str, str]:
    """-> {name: "ok" | "missing: <module>"}. Call before a sweep, not during one."""
    import importlib.util
    out = {}
    for n in (names or REGISTRY):
        rep = REGISTRY[n]
        missing = [m for m in ("rdkit",) + rep.needs
                   if importlib.util.find_spec(m.split(".")[0]) is None]
        out[n] = "ok" if not missing else f"missing: {','.join(missing)}"
    return out


def featurize(name: str, smiles) -> tuple[np.ndarray, np.ndarray]:
    if name not in REGISTRY:
        raise KeyError(f"unknown representation {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name].fn(list(smiles))


def featurize_or_raise(name: str, smiles) -> np.ndarray:
    """`featurize` but refuses to return a matrix containing unparseable rows.

    Use this everywhere a model is about to be fitted. The masked variant exists for
    callers that genuinely want to drop rows (a generated-molecule pool), and those
    callers must drop `y` alongside.
    """
    X, ok = featurize(name, smiles)
    if not ok.all():
        bad = np.where(~ok)[0][:5].tolist()
        raise ValueError(
            f"representation {name!r}: {int((~ok).sum())} of {len(ok)} SMILES did not "
            f"parse (first few row indices: {bad}). Their feature rows would be all-zero, "
            f"which is a valid-looking point at the origin for a tree model and a molecule "
            f"similar to nothing for a Tanimoto kernel. Standardise the input first "
            f"(tnk1.data.standardize.standardize_smiles) or use `featurize` and drop the "
            f"rows together with their labels.")
    return X


# ------------------------------------------------------------------ RDKit fingerprints


def _mols(smiles):
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")        # a failed parse is reported through `ok`, not stderr
    return [Chem.MolFromSmiles(s) if s else None for s in smiles]


def _fp_matrix(smiles, gen_fn, count: bool, n_bits: int):
    """Shared driver: build the generator once, then one row per molecule."""
    gen = gen_fn()
    get = (gen.GetCountFingerprintAsNumPy if count else gen.GetFingerprintAsNumPy)
    ms = _mols(smiles)
    X = np.zeros((len(ms), n_bits), dtype=np.float32)
    ok = np.zeros(len(ms), dtype=bool)
    for i, m in enumerate(ms):
        if m is None:
            continue
        X[i] = get(m)
        ok[i] = True
    return X, ok


def _morgan(radius: int, n_bits: int, count: bool, features: bool):
    def fn(smiles):
        from rdkit.Chem import rdFingerprintGenerator as rfg
        inv = rfg.GetMorganFeatureAtomInvGen() if features else None
        return _fp_matrix(
            smiles,
            lambda: rfg.GetMorganGenerator(radius=radius, fpSize=n_bits,
                                           atomInvariantsGenerator=inv),
            count, n_bits)
    return fn


# ECFP/FCFP. Rogers & Hahn, "Extended-Connectivity Fingerprints", J Chem Inf Model 2010.
# Radius 2 (= ECFP4 in the diameter convention) is the standard QSAR default; radius 3
# (ECFP6) is included because the kinase-hinge motifs this project cares about span more
# than four bonds. COUNT vectors are kept as the primary variants: the count Tanimoto is
# what `tnk1.baselines.models.tanimoto_sim` and the GP kernel implement, and binary
# folding discards the multiplicity that distinguishes a mono- from a tri-substituted
# scaffold.
register(Representation("ecfp4_count", _morgan(2, 2048, True, False), "count", 2048,
                        cite="Rogers & Hahn 2010", tags=("fingerprint", "default")))
register(Representation("ecfp4_bits", _morgan(2, 2048, False, False), "binary", 2048,
                        cite="Rogers & Hahn 2010", tags=("fingerprint",)))
register(Representation("ecfp6_count", _morgan(3, 2048, True, False), "count", 2048,
                        cite="Rogers & Hahn 2010", tags=("fingerprint",)))
register(Representation("ecfp4_count_4096", _morgan(2, 4096, True, False), "count", 4096,
                        cite="Rogers & Hahn 2010", tags=("fingerprint",)))
# FCFP: pharmacophoric atom invariants rather than atom identity, so it generalises across
# bioisosteres -- which is what scaffold-hopping asks for.
register(Representation("fcfp4_count", _morgan(2, 2048, True, True), "count", 2048,
                        cite="Rogers & Hahn 2010 (feature invariants)", tags=("fingerprint",)))


def _atom_pair(n_bits=2048, count=True):
    def fn(smiles):
        from rdkit.Chem import rdFingerprintGenerator as rfg
        return _fp_matrix(smiles, lambda: rfg.GetAtomPairGenerator(fpSize=n_bits), count, n_bits)
    return fn


def _torsion(n_bits=2048, count=True):
    def fn(smiles):
        from rdkit.Chem import rdFingerprintGenerator as rfg
        return _fp_matrix(smiles, lambda: rfg.GetTopologicalTorsionGenerator(fpSize=n_bits),
                          count, n_bits)
    return fn


def _rdkit_fp(n_bits=2048):
    def fn(smiles):
        from rdkit.Chem import rdFingerprintGenerator as rfg
        return _fp_matrix(smiles, lambda: rfg.GetRDKitFPGenerator(fpSize=n_bits), False, n_bits)
    return fn


# Carhart et al., "Atom pairs as molecular features in structure-activity studies",
# J Chem Inf Comput Sci 1985. Encodes through-bond distance between atom types, so it sees
# the shape of a molecule where ECFP sees only local environments.
register(Representation("atompair_count", _atom_pair(), "count", 2048,
                        cite="Carhart et al. 1985", tags=("fingerprint",)))
# Nilakantan et al., "Topological torsion", J Chem Inf Comput Sci 1987.
register(Representation("torsion_count", _torsion(), "count", 2048,
                        cite="Nilakantan et al. 1987", tags=("fingerprint",)))
register(Representation("rdkit_fp", _rdkit_fp(), "binary", 2048,
                        cite="RDKit path-based fingerprint", tags=("fingerprint",)))


def _maccs(smiles):
    from rdkit.Chem import MACCSkeys
    ms = _mols(smiles)
    X = np.zeros((len(ms), 167), dtype=np.float32)
    ok = np.zeros(len(ms), dtype=bool)
    for i, m in enumerate(ms):
        if m is None:
            continue
        X[i] = np.asarray(MACCSkeys.GenMACCSKeys(m), dtype=np.float32)
        ok[i] = True
    return X, ok


# MACCS: 166 expert-curated substructure keys (+1 unused bit 0). Low-dimensional and
# therefore well suited to N=172 -- the opposite trade-off from a 2048-bit ECFP.
register(Representation("maccs", _maccs, "binary", 167,
                        cite="MDL MACCS structural keys", tags=("fingerprint", "low_dim")))


def _avalon(n_bits=1024, count=False):
    def fn(smiles):
        from rdkit.Avalon import pyAvalonTools
        ms = _mols(smiles)
        X = np.zeros((len(ms), n_bits), dtype=np.float32)
        ok = np.zeros(len(ms), dtype=bool)
        for i, m in enumerate(ms):
            if m is None:
                continue
            if count:
                fp = pyAvalonTools.GetAvalonCountFP(m, nBits=n_bits)
                for k, v in fp.GetNonzeroElements().items():
                    X[i, k] = v
            else:
                X[i] = np.asarray(pyAvalonTools.GetAvalonFP(m, nBits=n_bits), dtype=np.float32)
            ok[i] = True
        return X, ok
    return fn


# AVALON IS NOT OPTIONAL HERE. Kamuntavicius, Silins, Jankauskas & Krasauskas,
# "Benchmarking ML in ADMET predictions: the practical impact of feature representations in
# ligand-based models", J Cheminform 17:108 (2025) -- 25 datasets from ~200 to ~36,000
# compounds, 10-fold CV, Friedman + Nemenyi -- finds the best REGRESSION feature
# combination to be `rdkit_desc + ERG + ECFP4 + Avalon`, and states explicitly that "the
# popular rdkit_desc + ecfp4 combination was not the best choice". That popular pairing is
# exactly this project's Stage-2 representation (`ecfp4_desc200`), so Avalon and ERG are
# the two blocks the pipeline has been missing.
register(Representation("avalon", _avalon(), "binary", 1024,
                        cite="Gedeck, Rohde & Bartels 2006; in the best regression "
                             "combination of Kamuntavicius et al. 2025",
                        tags=("fingerprint", "core")))
register(Representation("avalon_count", _avalon(count=True), "count", 1024,
                        cite="Gedeck et al. 2006, count variant", tags=("fingerprint",)))


def _erg(fuzz=0.3, max_path=15):
    def fn(smiles):
        from rdkit.Chem import rdReducedGraphs
        ms = _mols(smiles)
        rows, ok = [], np.zeros(len(ms), dtype=bool)
        dim = None
        for i, m in enumerate(ms):
            if m is None:
                rows.append(None)
                continue
            v = np.asarray(rdReducedGraphs.GetErGFingerprint(m, fuzzIncrement=fuzz,
                                                             maxPath=max_path),
                           dtype=np.float32)
            dim = len(v) if dim is None else dim
            rows.append(v)
            ok[i] = True
        if dim is None:
            return np.zeros((len(ms), 0), dtype=np.float32), ok
        X = np.zeros((len(ms), dim), dtype=np.float32)
        for i, v in enumerate(rows):
            if v is not None:
                X[i] = v
        return X, ok
    return fn


# ERG: Stiefl, Watson, Baumann & Zaliani, "ErG: 2D pharmacophore descriptions for scaffold
# hopping", J Chem Inf Model 46:208 (2006). A REDUCED graph -- pharmacophoric node types
# with fuzzy through-bond distances -- so it abstracts away the exact scaffold while keeping
# the pharmacophore. That is precisely the representation a scaffold-hopping objective
# wants, and this project's contract has a NOVEL_SCAFFOLD cohort it cannot currently
# represent. It is also the other half of the best regression combination in
# Kamuntavicius et al. 2025 and was absent from this repository entirely.
register(Representation("erg", _erg(), "dense", 315,
                        cite="Stiefl et al. 2006; in the best regression combination of "
                             "Kamuntavicius et al. 2025",
                        tags=("fingerprint", "pharmacophore", "core")))


# ------------------------------------------------------------------------- descriptors


def _rdkit_desc_all(smiles):
    """EVERY RDKit 2D descriptor, not the first 200.

    `tnk1.baselines.features.rdkit_descriptors` takes `_descList[:200]`, which is an
    alphabetical prefix, not a selection: it includes all 43 BCUT2D and both Chi4 variants
    while cutting everything from `NumAliphaticCarbocycles` onward -- i.e. most of the ring
    and H-bond counts a medicinal chemist would reach for first. Nothing documents that as
    a choice. Here the full list is computed and the sweep decides.
    """
    from rdkit.Chem import Descriptors
    fns = list(Descriptors._descList)
    ms = _mols(smiles)
    X = np.full((len(ms), len(fns)), np.nan, dtype=np.float64)
    ok = np.zeros(len(ms), dtype=bool)
    for i, m in enumerate(ms):
        if m is None:
            continue
        ok[i] = True
        for j, (_, f) in enumerate(fns):
            try:
                X[i, j] = f(m)
            except Exception:
                pass                 # individual descriptors fail on exotic valences
    X[~np.isfinite(X)] = np.nan      # NaN is imputed by DescriptorScaler, not here
    return X.astype(np.float64), ok


register(Representation("rdkit_desc_all", _rdkit_desc_all, "dense", None,
                        cite="RDKit Descriptors._descList (full)", tags=("descriptor",)))


def _rdkit_desc_200(smiles):
    """The legacy prefix, kept so the sweep can show whether the full set actually helps."""
    from ..baselines.features import rdkit_descriptors
    X = rdkit_descriptors(list(smiles), 200)
    ok = np.isfinite(X).any(axis=1)
    return X, ok


register(Representation("rdkit_desc_200", _rdkit_desc_200, "dense", 200,
                        cite="legacy: first 200 of _descList", tags=("descriptor", "legacy")))


def _mordred(smiles):
    from mordred import Calculator, descriptors as mdesc
    calc = Calculator(mdesc, ignore_3D=True)
    ms = _mols(smiles)
    ok = np.array([m is not None for m in ms])
    rows = calc.pandas([m for m in ms if m is not None], quiet=True, nproc=1)
    X = np.full((len(ms), rows.shape[1]), np.nan)
    X[ok] = rows.to_numpy(dtype=np.float64, na_value=np.nan)
    X[~np.isfinite(X)] = np.nan
    return X, ok


# ~1600 2D descriptors. Moriwaki et al., "Mordred: a molecular descriptor calculator",
# J Cheminform 2018. Optional: at N=172 this is 10x more features than molecules, so it is
# registered for completeness and expected to need aggressive regularisation.
register(Representation("mordred", _mordred, "dense", None, needs=("mordred",),
                        cite="Moriwaki et al. 2018", tags=("descriptor", "high_dim")))


# ------------------------------------------------------------------------ combinations


_COMBOS: dict[str, tuple[str, ...]] = {}


def combined(*names: str):
    """Horizontally stack several representations into one.

    The blocks are NOT scaled here, deliberately: scaling must be fitted on the TRAINING
    fold only, so it belongs to the cross-validation loop (`tnk1.zoo.benchmark`). A
    featuriser that standardised its own output would leak the test fold's location and
    spread into every fit -- worth a visible amount of apparent skill at N=172.

    The component names are recorded in `_COMBOS` so `dense_columns` can DERIVE which
    columns need scaling instead of hardcoding offsets. The hardcoded version was already
    wrong once: adding a block in the middle of a combination silently shifted the slice,
    and the symptom would have been a scaler fitted on fingerprint counts and a descriptor
    block left unscaled -- no error, just quietly worse models.
    """
    def fn(smiles):
        Xs, oks = [], []
        for n in names:
            X, ok = featurize(n, smiles)
            Xs.append(X)
            oks.append(ok)
        return np.hstack(Xs), np.logical_and.reduce(oks)
    fn._components = tuple(names)
    return fn


# The Stage-2 production representation, named so the sweep can include it as a control.
register(Representation("ecfp4_desc200", combined("ecfp4_count", "rdkit_desc_200"),
                        "mixed", 2248, cite="this project's Stage-2 baseline",
                        tags=("combined", "control")))
register(Representation("ecfp4_desc_all", combined("ecfp4_count", "rdkit_desc_all"),
                        "mixed", None, tags=("combined",)))
register(Representation("ecfp4_maccs_desc", combined("ecfp4_count", "maccs", "rdkit_desc_all"),
                        "mixed", None, tags=("combined",)))
register(Representation("fp_union", combined("ecfp4_count", "fcfp4_count", "atompair_count",
                                             "torsion_count"),
                        "count", 8192, tags=("combined", "high_dim")))

# THE LITERATURE'S BEST REGRESSION COMBINATION, as a named representation so it is a
# one-word change rather than a reimplementation: `rdkit_desc + ERG + ECFP4 + Avalon`
# (Kamuntavicius et al., J Cheminform 17:108, 2025, best mean rank over 25 datasets).
# Block order is fixed here because `dense_columns` below slices on it.
register(Representation("admet_best_combo",
                        combined("ecfp4_count", "avalon", "rdkit_desc_all", "erg"),
                        "mixed", None,
                        cite="Kamuntavicius et al. 2025, best regression combination",
                        tags=("combined", "core")))
# The same paper's best BINARY-endpoint combination, minus ECFP4: kept because it is a
# genuinely different block mix and cheap to evaluate alongside.
register(Representation("desc_erg_avalon", combined("rdkit_desc_all", "erg", "avalon"),
                        "mixed", None,
                        cite="Kamuntavicius et al. 2025, best classification combination",
                        tags=("combined",)))


_WIDTH_CACHE: dict[str, int] = {}


def block_width(name: str, probe: str = "CCO") -> int:
    """Column count of a representation, measured once on a probe molecule.

    Measured rather than declared: `rdkit_desc_all` and `mordred` have no fixed width (it
    depends on the installed RDKit/mordred version), and a declared `dim` that drifts from
    the real one is the kind of silent mismatch this codebase has been bitten by. Ethanol
    parses under every toolkit version, so the probe never fails.
    """
    if name not in _WIDTH_CACHE:
        X, _ = featurize(name, [probe])
        _WIDTH_CACHE[name] = int(X.shape[1])
    return _WIDTH_CACHE[name]


def dense_columns(name: str):
    """Boolean mask of the columns of `name` that need train-fold scaling. None if none do.

    Derived from the component blocks' own `kind`, so it stays correct when a combination's
    block order or content changes. `dense` blocks (RDKit/Mordred descriptors, ERG's fuzzy
    pharmacophore counts) are scaled; `count` and `binary` fingerprint blocks are not --
    standardising a sparse count vector destroys the substructure-overlap geometry the
    Tanimoto/MinMax kernels rely on, and centring a bit vector makes "substructure absent"
    a nonzero value.
    """
    rep = REGISTRY[name]
    if rep.kind == "dense":
        return np.ones(block_width(name), dtype=bool)
    comps = _COMBOS.get(name)
    if not comps:
        return None
    mask, any_dense = [], False
    for c in comps:
        w = block_width(c)
        is_dense = REGISTRY[c].kind == "dense"
        any_dense = any_dense or is_dense
        mask.append(np.full(w, is_dense, dtype=bool))
    return np.concatenate(mask) if any_dense else None


# Populate `_COMBOS` from the registered combination featurisers. Done after every
# `register` call above so a combination added later is picked up by re-running this.
for _n, _r in REGISTRY.items():
    _c = getattr(_r.fn, "_components", None)
    if _c:
        _COMBOS[_n] = _c
