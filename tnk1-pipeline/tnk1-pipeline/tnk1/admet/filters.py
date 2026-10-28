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


def sa_score(mol) -> float:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    return float(sascorer.calculateScore(mol))


def alerts(mol) -> list[str]:
    return [e.GetDescription() for e in _catalog().GetMatches(mol)]


def hinge_match(mol) -> bool:
    from rdkit import Chem
    return any(mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in HINGE_SMARTS.values())


def medchem_profile(smiles: str) -> dict | None:
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None
    a = alerts(m)
    return {"qed": float(QED.qed(m)), "sa": sa_score(m), "mw": float(Descriptors.MolWt(m)),
            "n_alerts": len(a), "alerts": ";".join(a), "hinge": hinge_match(m)}
