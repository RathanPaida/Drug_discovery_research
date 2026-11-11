"""Sec 5.3: QED, Ertl SA score, PAINS/Brenk/NIH alerts, hinge-binder SMARTS."""
from __future__ import annotations
import os, sys

HINGE_SMARTS = {
    "2-aminopyrimidine": "[NX3;H1,H2]c1nccc(n1)", "aminopyrimidine_b": "c1cnc(N)nc1",
    "pyrrolo[2,3-d]pyrimidine": "c1ncc2cc[nH]c2n1", "quinazoline": "c1ncnc2ccccc12",
    "indazole": "c1ccc2[nH]ncc2c1", "pyrazolo[3,4-d]pyrimidine": "c1ncc2cn[nH]c2n1",
}
_CAT = None


def _catalog():
    global _CAT
    if _CAT is None:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams as P
        p = P()
        for c in (P.FilterCatalogs.PAINS_A, P.FilterCatalogs.PAINS_B, P.FilterCatalogs.PAINS_C,
                  P.FilterCatalogs.BRENK, P.FilterCatalogs.NIH):
            p.AddCatalog(c)
        _CAT = FilterCatalog(p)
    return _CAT


_SASCORER = None


def _sascorer():
    """Import RDKit's contrib SA_Score once.

    `sys.path.append(...)` used to run on *every* call, so scoring a 50k-molecule pool grew
    sys.path by 50k duplicate entries and slowed every subsequent import in the process.
    """
    global _SASCORER
    if _SASCORER is None:
        from rdkit.Chem import RDConfig
        p = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if p not in sys.path:
            sys.path.append(p)
        import sascorer
        _SASCORER = sascorer
    return _SASCORER


def sa_score(mol) -> float:
    return float(_sascorer().calculateScore(mol))


def alerts(mol) -> list[str]:
    return [e.GetDescription() for e in _catalog().GetMatches(mol)]


_HINGE_PATTERNS = None


def _hinge_patterns():
    """Compile the hinge SMARTS once instead of once per molecule.

    `hinge_match` previously called MolFromSmarts on all six patterns for every molecule --
    300k compilations over a 50k pool -- and silently treated an unparseable SMARTS as a
    non-match. Bad patterns now fail loudly at first use.
    """
    global _HINGE_PATTERNS
    if _HINGE_PATTERNS is None:
        from rdkit import Chem
        pats = {}
        for name, smarts in HINGE_SMARTS.items():
            q = Chem.MolFromSmarts(smarts)
            if q is None:
                raise ValueError(f"HINGE_SMARTS[{name!r}] is not a valid SMARTS: {smarts!r}")
            pats[name] = q
        _HINGE_PATTERNS = pats
    return _HINGE_PATTERNS


def hinge_match(mol) -> bool:
    return any(mol.HasSubstructMatch(q) for q in _hinge_patterns().values())


def hinge_matches(mol) -> list[str]:
    """Which hinge motifs matched -- useful for the contract's provenance columns."""
    return [n for n, q in _hinge_patterns().items() if mol.HasSubstructMatch(q)]


def medchem_profile(smiles: str) -> dict | None:
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None
    a = alerts(m)
    return {"qed": float(QED.qed(m)), "sa": sa_score(m), "mw": float(Descriptors.MolWt(m)),
            "n_alerts": len(a), "alerts": ";".join(a), "hinge": hinge_match(m)}
