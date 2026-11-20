"""Standardisation + the activity guard rails.

Two jobs:

1. `standardize_smiles` -- RDKit cleanup -> canonical parent SMILES + InChIKey,
   so the same compound from ChEMBL/BindingDB/PubChem collapses to one row.

2. `harmonize_activity` / `filter_measurable` -- unit conversion to pActivity
   and the fail-closed checks that stop the v2 poisoning from recurring.

The v2 incident, restated so the fix is readable next to it: rows with *no*
measured value were written as `pActivity == 0.0`, which reads as an exact
IC50 of 1 M (1e9 nM) and is a perfectly plausible-looking number. 3,372,521 of
4,798,165 rows (70.3%) were affected. The guard below therefore:

  * never derives pActivity from a null/zero/sentinel standard_value;
  * rejects pActivity outside [MIN_PACTIVITY, MAX_PACTIVITY];
  * refuses any frame whose exact-zero pActivity fraction exceeds
    `MAX_ZERO_FRACTION` (default 0.1%) -- a loud failure, not a warning;
  * keeps `relation` so censored (>/<) points are modelled as censored and
    never silently treated as exact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .util import LOG, try_import

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
#: Accepted activity endpoint names (ChEMBL `standard_type` vocabulary).
ACTIVITY_TYPES = ("IC50", "Ki", "Kd", "EC50", "XC50", "AC50", "Potency")

#: Concentration units -> multiplier to nanomolar.
TO_NM = {
    "M": 1e9, "mol/L": 1e9, "mol.l-1": 1e9,
    "mM": 1e6, "mmol/L": 1e6,
    "uM": 1e3, "um": 1e3, "µM": 1e3, "umol/L": 1e3, "μM": 1e3,
    "nM": 1.0, "nmol/L": 1.0,
    "pM": 1e-3, "pmol/L": 1e-3,
    "fM": 1e-6,
}

#: pActivity window. 2.0 == 10 mM, 12.0 == 1 pM. Anything outside is an
#: encoding error, not chemistry.
MIN_PACTIVITY = 2.0
MAX_PACTIVITY = 12.0

#: Values that mean "no measurement" in some upstream dumps.
SENTINELS = {0.0, -1.0, 1e9, 1e10, 999999.0, 1000000000.0}

#: Exact-zero pActivity fraction above which we refuse the frame outright.
MAX_ZERO_FRACTION = 1e-3

CENSORED_RELATIONS = {">", ">=", "<", "<=", ">>", "<<"}
EXACT_RELATIONS = {"=", "==", "~"}


class PoisonedDataError(RuntimeError):
    """Raised when a frame shows the v2 zero-encoding signature."""


# --------------------------------------------------------------------------- #
# structure standardisation
# --------------------------------------------------------------------------- #
@dataclass
class StdResult:
    smiles: str | None
    inchikey: str | None
    reason: str = ""


class Standardizer:
    """RDKit-based parent-structure standardiser.

    Steps: sanitize -> remove salts/solvates (largest organic fragment) ->
    neutralise -> remove isotopes -> canonical SMILES (no stereo loss) ->
    InChIKey. Rejects inorganics, mixtures that lose all organic content, and
    anything outside a sane heavy-atom window.
    """

    def __init__(self, min_heavy: int = 6, max_heavy: int = 100):
        rd = try_import("rdkit")
        if rd is None:
            raise SystemExit("rdkit is required for standardisation: "
                             "pip install rdkit")
        from rdkit import Chem, RDLogger                            # noqa: PLC0415
        from rdkit.Chem.MolStandardize import rdMolStandardize      # noqa: PLC0415

        RDLogger.DisableLog("rdApp.*")
        self.Chem = Chem
        self._lfc = rdMolStandardize.LargestFragmentChooser()
        self._uncharger = rdMolStandardize.Uncharger()
        self._normalizer = rdMolStandardize.Normalizer()
        self.min_heavy = min_heavy
        self.max_heavy = max_heavy

    def __call__(self, smiles: str | None) -> StdResult:
        if not smiles or not isinstance(smiles, str):
            return StdResult(None, None, "empty")
        Chem = self.Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return StdResult(None, None, "unparseable")
        try:
            mol = self._normalizer.normalize(mol)
            mol = self._lfc.choose(mol)
            mol = self._uncharger.uncharge(mol)
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)
            Chem.SanitizeMol(mol)
        except Exception as exc:                                   # noqa: BLE001
            return StdResult(None, None, f"standardize_error:{type(exc).__name__}")
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < self.min_heavy:
            return StdResult(None, None, f"too_small:{n_heavy}")
        if n_heavy > self.max_heavy:
            return StdResult(None, None, f"too_large:{n_heavy}")
        if not any(a.GetSymbol() == "C" for a in mol.GetAtoms()):
            return StdResult(None, None, "no_carbon")
        try:
            can = Chem.MolToSmiles(mol, canonical=True)
            key = Chem.MolToInchiKey(mol)
        except Exception as exc:                                   # noqa: BLE001
            return StdResult(None, None, f"serialize_error:{type(exc).__name__}")
        if not key:
            return StdResult(can, None, "no_inchikey")
        return StdResult(can, key)

    def apply(self, df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
        """Add canonical_smiles / inchikey / std_reason; drop failures."""
        uniq = df[smiles_col].dropna().astype(str).unique()
        LOG.info("standardising %d unique SMILES", len(uniq))
        cache = {s: self(s) for s in uniq}
        df = df.copy()
        df["canonical_smiles"] = df[smiles_col].map(
            lambda s: cache.get(s, StdResult(None, None, "empty")).smiles)
        df["inchikey"] = df[smiles_col].map(
            lambda s: cache.get(s, StdResult(None, None, "empty")).inchikey)
        df["std_reason"] = df[smiles_col].map(
            lambda s: cache.get(s, StdResult(None, None, "empty")).reason)
        bad = df["canonical_smiles"].isna()
        if bad.any():
            LOG.warning("dropping %d rows failing standardisation (%s)",
                        int(bad.sum()),
                        df.loc[bad, "std_reason"].value_counts().head(5).to_dict())
        return df.loc[~bad].drop(columns=["std_reason"])


# --------------------------------------------------------------------------- #
# activity harmonisation
# --------------------------------------------------------------------------- #
def _to_nm(value: float, units: str | None) -> float | None:
    if value is None or not np.isfinite(value) or value <= 0:
        return None
    if units is None:
        return None
    mult = TO_NM.get(str(units).strip())
    if mult is None:
        return None
    return value * mult


def harmonize_activity(df: pd.DataFrame, *,
                       value_col: str = "standard_value",
                       units_col: str = "standard_units",
                       type_col: str = "standard_type",
                       relation_col: str = "standard_relation",
                       pchembl_col: str | None = "pchembl_value") -> pd.DataFrame:
    """Add `p_activity` (-log10 M), `relation`, `censored`, `value_nm`.

    Precedence: a *measured* concentration in a convertible unit wins. An
    upstream pChEMBL value is used only as a cross-check and as a fallback when
    units are missing -- never to invent a value where none was measured.
    """
    df = df.copy()
    v = pd.to_numeric(df.get(value_col), errors="coerce")
    # Blank out sentinels BEFORE any maths.
    v = v.where(~v.isin(SENTINELS))
    v = v.where(v > 0)

    units = df.get(units_col)
    value_nm = [
        _to_nm(float(x), u) if pd.notna(x) else None
        for x, u in zip(v, units if units is not None else [None] * len(df))
    ]
    df["value_nm"] = value_nm
    p_from_value = np.where(
        pd.notna(df["value_nm"]),
        9.0 - np.log10(pd.to_numeric(df["value_nm"], errors="coerce")
                       .replace(0, np.nan)),
        np.nan,
    )

    p = pd.Series(p_from_value, index=df.index, dtype="float64")

    if pchembl_col and pchembl_col in df.columns:
        pch = pd.to_numeric(df[pchembl_col], errors="coerce")
        pch = pch.where((pch > MIN_PACTIVITY - 0.001) & (pch < MAX_PACTIVITY + 0.001))
        # Disagreement audit: >1 log unit apart means one of them is wrong.
        both = p.notna() & pch.notna()
        if both.any():
            delta = (p[both] - pch[both]).abs()
            n_bad = int((delta > 1.0).sum())
            if n_bad:
                LOG.warning("pActivity disagrees with upstream pChEMBL by >1 log "
                            "for %d/%d rows -- trusting the measured value",
                            n_bad, int(both.sum()))
        p = p.fillna(pch)
        df["p_activity_source"] = np.where(
            pd.notna(p_from_value), "value",
            np.where(pch.notna(), "pchembl", "none"))
    else:
        df["p_activity_source"] = np.where(pd.notna(p_from_value), "value", "none")

    df["p_activity"] = p

    rel = (df.get(relation_col, pd.Series(["="] * len(df), index=df.index))
           .fillna("=").astype(str).str.strip())
    df["relation"] = rel
    df["censored"] = rel.isin(CENSORED_RELATIONS)
    df["endpoint"] = df.get(type_col, pd.Series([None] * len(df), index=df.index))
    return df


def filter_measurable(df: pd.DataFrame, *, require_types: bool = True,
                      drop_censored: bool = False,
                      max_zero_fraction: float = MAX_ZERO_FRACTION,
                      label: str = "frame") -> pd.DataFrame:
    """Fail-closed row filter. Raises `PoisonedDataError` on the v2 signature."""
    n0 = len(df)
    if "p_activity" not in df.columns:
        raise ValueError("call harmonize_activity() before filter_measurable()")

    # --- poison detector (run on the INPUT, before we drop anything) -------- #
    p = pd.to_numeric(df["p_activity"], errors="coerce")
    n_zero = int((p == 0.0).sum())
    frac_zero = n_zero / max(1, n0)
    if frac_zero > max_zero_fraction:
        raise PoisonedDataError(
            f"{label}: {n_zero:,}/{n0:,} rows ({frac_zero:.1%}) have "
            f"p_activity == 0.0 exactly. This is the v2 'no value encoded as "
            f"zero' signature (reads as IC50 = 1 M). Refusing to use this file. "
            f"Use the v3 extract, or re-derive from standard_value/standard_units."
        )
    if n_zero:
        LOG.warning("%s: %d exact-zero p_activity rows dropped", label, n_zero)

    keep = p.notna() & (p > MIN_PACTIVITY) & (p < MAX_PACTIVITY)
    reasons = {"nan_or_out_of_range": int((~keep).sum())}

    if require_types and "endpoint" in df.columns:
        ok_type = df["endpoint"].astype(str).str.upper().isin(
            {t.upper() for t in ACTIVITY_TYPES})
        reasons["bad_endpoint"] = int((keep & ~ok_type).sum())
        keep &= ok_type

    if drop_censored and "censored" in df.columns:
        reasons["censored"] = int((keep & df["censored"]).sum())
        keep &= ~df["censored"]

    out = df.loc[keep].copy()
    LOG.info("filter_measurable[%s]: %d -> %d rows (%s)", label, n0, len(out), reasons)
    if len(out) == 0 and n0 > 0:
        raise RuntimeError(f"{label}: filter_measurable removed every row -- "
                           f"check units/endpoint columns, do not proceed")
    return out


def aggregate_replicates(df: pd.DataFrame, *,
                         key: tuple[str, ...] = ("inchikey", "target_gene", "endpoint"),
                         max_spread: float = 2.0) -> pd.DataFrame:
    """Collapse replicate measurements to one row per (compound, target, endpoint).

    Exact and censored points are aggregated separately: taking the median of a
    `>10 uM` and a `=50 nM` would be meaningless. Returns median p_activity,
    n_measurements, spread (max-min), and a `discordant` flag where the spread
    exceeds `max_spread` log units -- those are the rows worth re-assaying, and
    the ones most likely to be the reason a model's ceiling is low.
    """
    key = tuple(k for k in key if k in df.columns)
    specs: dict[str, tuple] = {
        "p_activity": ("p_activity", "median"),
        "p_activity_min": ("p_activity", "min"),
        "p_activity_max": ("p_activity", "max"),
        "n_measurements": ("p_activity", "size"),
        "canonical_smiles": ("canonical_smiles", "first"),
        "relation": ("relation", lambda s: s.mode().iat[0] if len(s.mode()) else "="),
    }
    if "source" in df.columns:
        specs["sources"] = ("source", lambda s: ";".join(sorted(set(s.dropna().astype(str)))))
    agg = df.groupby([*key, "censored"], dropna=False).agg(**specs).reset_index()
    agg["spread"] = agg["p_activity_max"] - agg["p_activity_min"]
    agg["discordant"] = agg["spread"] > max_spread
    n_disc = int(agg["discordant"].sum())
    if n_disc:
        LOG.warning("%d aggregated points have >%.1f log spread across replicates",
                    n_disc, max_spread)
    LOG.info("aggregate_replicates: %d -> %d rows (exact=%d, censored=%d)",
             len(df), len(agg), int((~agg["censored"]).sum()),
             int(agg["censored"].sum()))
    return agg


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def murcko_scaffold(smiles: str) -> str | None:
    from rdkit import Chem                                          # noqa: PLC0415
    from rdkit.Chem.Scaffolds import MurckoScaffold                 # noqa: PLC0415
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:                                               # noqa: BLE001
        return None


def add_scaffold_split(df: pd.DataFrame, *, smiles_col: str = "canonical_smiles",
                       fracs: tuple[float, float, float] = (0.8, 0.1, 0.1),
                       seed: int = 0) -> pd.DataFrame:
    """Deterministic scaffold split. With ~285 unique scaffolds over 380 rows a
    random split is near-meaningless -- almost every test compound has a train
    twin. Scaffold split is what makes the reported rho honest.
    """
    df = df.copy()
    df["murcko_scaffold"] = df[smiles_col].map(murcko_scaffold).fillna("")
    groups = df.groupby("murcko_scaffold").indices
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    n = len(df)
    n_train, n_valid = int(fracs[0] * n), int(fracs[1] * n)
    split = pd.Series(index=df.index, dtype="object")
    counts = {"train": 0, "valid": 0, "test": 0}
    for _scaf, idx in order:
        if counts["train"] < n_train:
            which = "train"
        elif counts["valid"] < n_valid:
            which = "valid"
        else:
            which = "test"
        split.iloc[list(idx)] = which
        counts[which] += len(idx)
    df["split"] = split.values
    LOG.info("scaffold split: %s over %d unique scaffolds",
             counts, df["murcko_scaffold"].nunique())
    return df
