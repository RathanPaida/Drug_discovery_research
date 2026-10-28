#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDKit Chemical Standardization Pipeline (Stage 1.2 Specification)
Strictly adheres to Step 1.2 of TNK1 AI Pipeline Technical Manual:
1. Sanitization & Parsing
2. Fragment & Salt Stripping (LargestFragmentChooser)
3. Functional Group Normalization & Reionization (Normalizer, Reionizer, Uncharger)
4. Tautomer Canonicalization (TautomerEnumerator)
5. Organic Medicinal Chemistry Filter (150 <= MW <= 900, 10 <= HAC <= 65, allowed elements)
6. Identity Key Generation (Canonical SMILES, InChIKey, InChIKey_block1)
"""

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import Descriptors, inchi
from typing import Optional, Dict, Any

# Disable verbose C++ RDKit logs
RDLogger.DisableLog("rdApp.*")

ALLOWED_ELEMENTS = {"C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "H"}

class ChemicalStandardizer:
    def __init__(self, max_tautomers: int = 50):
        self.lfc = rdMolStandardize.LargestFragmentChooser()
        self.normalizer = rdMolStandardize.Normalizer()
        self.reionizer = rdMolStandardize.Reionizer()
        self.uncharger = rdMolStandardize.Uncharger()
        
        self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
        self.tautomer_enumerator.SetMaxTautomers(max_tautomers)
        
    def standardize_mol(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Runs the 6-step standardization protocol on an RDKit Mol object."""
        result = {
            "status": "failed",
            "fail_reason": "",
            "canonical_smiles": "",
            "inchikey": "",
            "inchikey_block1": "",
            "mol_weight": 0.0,
            "heavy_atom_count": 0,
            "mol": None
        }
        
        if mol is None:
            result["fail_reason"] = "mol_is_none"
            return result

        try:
            # 1. Sanitization & Parsing
            Chem.SanitizeMol(mol)
        except Exception as e:
            result["fail_reason"] = f"sanitization_failed: {str(e)}"
            return result

        try:
            # 2. Fragment & Salt Stripping
            core_frag = self.lfc.choose(mol)
            if core_frag is None or core_frag.GetNumHeavyAtoms() == 0:
                result["fail_reason"] = "empty_fragment_after_stripping"
                return result

            # 3. Functional Group Normalization & Reionization
            norm = self.normalizer.normalize(core_frag)
            reion = self.reionizer.reionize(norm)
            uncharged = self.uncharger.uncharge(reion)
            if uncharged is None:
                uncharged = reion

            # 4. Tautomer Canonicalization
            canonical_tautomer = self.tautomer_enumerator.Canonicalize(uncharged)
            if canonical_tautomer is None:
                canonical_tautomer = uncharged

            # 5. Organic Medicinal Chemistry Filter
            # Allowed elements: {C, N, O, S, F, Cl, Br, I, P, B, H}
            for atom in canonical_tautomer.GetAtoms():
                sym = atom.GetSymbol()
                if sym not in ALLOWED_ELEMENTS:
                    result["fail_reason"] = f"disallowed_element_{sym}"
                    return result

            mw = Descriptors.MolWt(canonical_tautomer)
            hac = canonical_tautomer.GetNumHeavyAtoms()
            if not (150.0 <= mw <= 900.0):
                result["fail_reason"] = f"mw_out_of_bounds_{mw:.1f}"
                return result
            if not (10 <= hac <= 65):
                result["fail_reason"] = f"hac_out_of_bounds_{hac}"
                return result

            # 6. Identity Key Generation
            can_smi = Chem.MolToSmiles(canonical_tautomer, isomericSmiles=True, canonical=True)
            try:
                ikey = inchi.MolToInchiKey(canonical_tautomer) or ""
            except Exception:
                ikey = ""

            if not ikey and can_smi:
                # Fallback if inchi fails
                m_temp = Chem.MolFromSmiles(can_smi)
                if m_temp:
                    ikey = inchi.MolToInchiKey(m_temp) or ""

            ikey_b1 = ikey[:14] if len(ikey) >= 14 else ""

            result["status"] = "success"
            result["canonical_smiles"] = can_smi
            result["inchikey"] = ikey
            result["inchikey_block1"] = ikey_b1
            result["mol_weight"] = float(mw)
            result["heavy_atom_count"] = int(hac)
            result["mol"] = canonical_tautomer
            return result

        except Exception as e:
            result["fail_reason"] = f"standardization_exception: {str(e)}"
            return result

    def standardize_smiles(self, smi: str) -> Dict[str, Any]:
        """Standardizes a SMILES string."""
        if not smi or not isinstance(smi, str) or not smi.strip():
            return {"status": "failed", "fail_reason": "empty_smiles"}
        try:
            mol = Chem.MolFromSmiles(smi.strip(), sanitize=True)
            return self.standardize_mol(mol)
        except Exception as e:
            return {"status": "failed", "fail_reason": f"invalid_smiles_parse: {str(e)}"}
