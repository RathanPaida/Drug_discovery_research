#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medicinal Chemistry Structural Filters & Hinge Pharmacophore (Stage 5.3 Specification)
Implements:
1. Quantitative Estimate of Drug-likeness (QED in [0, 1])
2. Ertl Synthetic Accessibility Score (SA in [1, 10], target <= 4.2)
3. MedChem Structural Alerts: PAINS (A, B, C), Brenk, NIH (zero matches allowed)
4. Hinge-Binder SMARTS Verification:
   - 2-aminopyrimidine
   - pyrrolo[2,3-d]pyrimidine
   - quinazoline
   - indazole
   - pyrazolopyrimidine
"""

import os
import sys
from typing import Dict, Any, List

from rdkit import Chem, RDConfig
from rdkit.Chem import QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Import Ertl SA scorer from RDKit Contrib
sa_dir = os.path.join(RDConfig.RDContribDir, 'SA_Score')
if sa_dir not in sys.path:
    sys.path.append(sa_dir)
import sascorer

# Hinge-binder SMARTS definitions matching Stage 5.3
HINGE_SMARTS = {
    "2_aminopyrimidine": Chem.MolFromSmarts("c1ncnc(N)c1"),
    "pyrrolo_pyrimidine": Chem.MolFromSmarts("c1c[nH]c2ncnc12"),
    "quinazoline": Chem.MolFromSmarts("c1ccc2ncncc2c1"),
    "indazole": Chem.MolFromSmarts("c1ccc2[nH]ncc2c1"),
    "pyrazolopyrimidine": Chem.MolFromSmarts("c1ncnc2cc[nH]c12"),
    "pyrazolopyrimidine_v2": Chem.MolFromSmarts("c1cnnc2ncnc12")
}

class MedChemFilterSuite:
    def __init__(self):
        # Initialize FilterCatalog with PAINS, Brenk, and NIH
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
        self.catalog = FilterCatalog(params)
        
    def evaluate_mol(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Evaluates a molecule against all Stage 5.3 filters."""
        if mol is None:
            return {
                "valid": False,
                "qed": 0.0,
                "sa_score": 10.0,
                "has_alerts": True,
                "alert_reasons": ["invalid_mol"],
                "has_hinge_motif": False,
                "matched_hinge_motifs": [],
                "pass_all_filters": False
            }
            
        # 1. QED
        qed_val = float(QED.qed(mol))
        
        # 2. SA Score
        try:
            sa_val = float(sascorer.calculateScore(mol))
        except Exception:
            sa_val = 10.0
            
        # 3. Structural Alerts (PAINS, Brenk, NIH)
        matches = self.catalog.GetMatches(mol)
        has_alerts = len(matches) > 0
        alert_reasons = [m.GetDescription() for m in matches]
        
        # 4. Hinge-Binder SMARTS Verification
        matched_hinge = []
        for name, pattern in HINGE_SMARTS.items():
            if pattern is not None and mol.HasSubstructMatch(pattern):
                matched_hinge.append(name)
        has_hinge = len(matched_hinge) > 0
        
        # Pass criteria: zero alerts, SA <= 4.2
        pass_all = (not has_alerts) and (sa_val <= 4.2)
        
        return {
            "valid": True,
            "qed": qed_val,
            "sa_score": sa_val,
            "has_alerts": has_alerts,
            "n_alerts": len(alert_reasons),
            "alert_reasons": alert_reasons,
            "has_hinge_motif": has_hinge,
            "matched_hinge_motifs": matched_hinge,
            "pass_all_filters": pass_all
        }
        
    def evaluate_smiles(self, smi: str) -> Dict[str, Any]:
        """Evaluates a SMILES string."""
        if not smi:
            return self.evaluate_mol(None)
        mol = Chem.MolFromSmiles(smi)
        return self.evaluate_mol(mol)

if __name__ == "__main__":
    suite = MedChemFilterSuite()
    test_smi = "Cc1cc(Nc2ncc(Cl)c(-c3ccc(S(N)(=O)=O)cc3)n2)ccc1C(=O)Nc1ccc(F)c(Cl)c1"
    res = suite.evaluate_smiles(test_smi)
    print("Test Molecule Evaluation:")
    for k, v in res.items():
        print(f"  {k}: {v}")
