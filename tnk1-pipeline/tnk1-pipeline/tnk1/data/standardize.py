"""Step 1.2: six-step RDKit standardization + identity keys."""
from __future__ import annotations


def _rd():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    return Chem, Descriptors, rdMolStandardize, MurckoScaffold


def standardize_smiles(smi: str, mw_range=(150, 900), hac_range=(10, 65),
                       allowed=("C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "H")) -> dict | None:
    Chem, Descriptors, S, Murcko = _rd()
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi, sanitize=True)                        # 1. sanitize
    if mol is None:
        return None
    try:
        mol = S.LargestFragmentChooser().choose(mol)                    # 2. salts / waters
        mol = S.Normalizer().normalize(mol)                             # 3. normalize
        mol = S.Reionizer().reionize(mol)
        mol = S.Uncharger().uncharge(mol)
        mol = S.TautomerEnumerator().Canonicalize(mol)                  # 4. tautomer
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    mw, hac = Descriptors.MolWt(mol), mol.GetNumHeavyAtoms()            # 5. med-chem filter
    if not (mw_range[0] <= mw <= mw_range[1] and hac_range[0] <= hac <= hac_range[1]):
        return None
    if any(a.GetSymbol() not in allowed for a in mol.GetAtoms()):
        return None
    can = Chem.MolToSmiles(mol, isomericSmiles=True)                    # 6. identity keys
    ik = Chem.MolToInchiKey(mol)
    if not ik:
        return None
    try:
        scaf = Chem.MolToSmiles(Murcko.GetScaffoldForMol(mol), isomericSmiles=False)
    except Exception:
        scaf = ""
    return {"smiles": can, "inchikey": ik, "inchikey_block1": ik[:14], "scaffold": scaf or ik[:14]}
