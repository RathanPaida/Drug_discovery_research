"""Step 1.2: six-step RDKit standardization + identity keys."""
from __future__ import annotations

import os

# Cached RDKit standardiser objects.
#
# `_rd()` used to be called per molecule and each call constructed a fresh
# LargestFragmentChooser / Normalizer / Reionizer / Uncharger / TautomerEnumerator.
# The TautomerEnumerator in particular builds its transform catalogue on construction, so
# standardising the 788k-molecule store paid that cost 788,000 times. Same fix as
# `tnk1.admet.filters._catalog` and `_hinge_patterns`.
_RD = None

# Keep sp3 stereochemistry through tautomer canonicalisation.
#
# THE DEFECT THIS FIXES. `rdMolStandardize.TautomerEnumerator().Canonicalize()` strips
# stereochemistry from sp3 centres adjacent to a tautomerisable system unless told not to.
# Measured on an alpha-amino amide:
#
#     input   C[C@H](N)C(=O)Nc1ccc(Cl)cc1C(=O)NCc1ccccc1   YNNMCNCKOWYFQD-NSHDSACASA-N
#     output  CC(N)C(=O)Nc1ccc(Cl)cc1C(=O)NCc1ccccc1       YNNMCNCKOWYFQD-UHFFFAOYSA-N
#
# `UHFFFAOYSA` is the InChIKey hash for "no stereochemical information", so BOTH
# enantiomers came out with one identical 27-character key.
#
# MEASURED BLAST RADIUS -- state it precisely, because it is smaller than it first looks.
# Re-standardising both datasets under the old and the new behaviour and diffing the
# InChIKeys gives **0 changed molecules of 40** in the Cycle-01 contract and **0 of 380**
# in `tnk1_curated.parquet`, even though 32% and 27% of them respectively carry
# stereocentres. The removal only fires for sp3 centres the enumerator considers adjacent
# to a tautomerisable system, which none of those molecules happen to have. So this did
# NOT corrupt any published result, and the frozen tables need no rebuild on its account.
#
# It is fixed because it is a live trap rather than a past error. Where it would bite:
#
#   * `aggregate_replicates` groups on (inchikey_block1, uniprot_id, standard_type) and
#     takes the MEDIAN pActivity. Kinase inhibitors are routinely stereospecific -- one
#     enantiomer potent, its mirror inactive -- so a conflated (R)/(S) pair would be
#     averaged into a value describing neither. Worse, `replicate_range_max: 1.0` with
#     `replicate_policy: drop` DISCARDS a pair spanning more than one log unit, which is
#     exactly what a potent/inactive enantiomer pair does: the most informative
#     stereochemistry would be the most likely to be thrown away.
#   * aminopyrimidine and aminopyrazole hinge binders -- the motifs this project's
#     `HINGE_SMARTS` actively selects for -- are tautomerisable, so a generated molecule
#     with an adjacent stereocentre is precisely the at-risk case.
#   * the returned `smiles` would lose its stereo tags, leaving the D-MPNN's chiral-atom
#     feature block (`_oh(int(a.GetChiralTag()), CHI)` in tnk1/models/graph.py) constant
#     and a contract SMILES under-specifying what the chemist should make.
#
# The function already asked for `isomericSmiles=True` when writing the canonical SMILES,
# so preserving stereochemistry was plainly the intent; the enumerator silently overrode
# it. Tautomer canonicalisation itself is unaffected: 2-hydroxypyridine still converges to
# 2-pyridone with the flag set.
#
# Set $TNK1_LEGACY_TAUTOMER_STEREO=1 to restore the old stereo-destroying behaviour, which
# is what the frozen v1/v2 tables and the Cycle-01 feature store were built with.
_LEGACY_ENV = "TNK1_LEGACY_TAUTOMER_STEREO"


def _rd():
    global _RD
    if _RD is None:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors
        from rdkit.Chem.MolStandardize import rdMolStandardize
        from rdkit.Chem.Scaffolds import MurckoScaffold
        RDLogger.DisableLog("rdApp.*")

        taut = rdMolStandardize.TautomerEnumerator()
        if os.environ.get(_LEGACY_ENV) != "1":
            taut.SetRemoveSp3Stereo(False)
        _RD = {
            "Chem": Chem,
            "Descriptors": Descriptors,
            "Murcko": MurckoScaffold,
            "largest": rdMolStandardize.LargestFragmentChooser(),
            "normalizer": rdMolStandardize.Normalizer(),
            "reionizer": rdMolStandardize.Reionizer(),
            "uncharger": rdMolStandardize.Uncharger(),
            "tautomer": taut,
        }
    return _RD


def preserves_stereo() -> bool:
    """Whether tautomer canonicalisation is configured to keep sp3 stereo. Stage scripts
    log this: it determines compound identity, so it belongs in a run's provenance."""
    return os.environ.get(_LEGACY_ENV) != "1"


def standardize_smiles(smi: str, mw_range=(150, 900), hac_range=(10, 65),
                       allowed=("C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "H")) -> dict | None:
    R = _rd()
    Chem = R["Chem"]
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi, sanitize=True)                        # 1. sanitize
    if mol is None:
        return None
    try:
        mol = R["largest"].choose(mol)                                  # 2. salts / waters
        mol = R["normalizer"].normalize(mol)                            # 3. normalize
        mol = R["reionizer"].reionize(mol)
        mol = R["uncharger"].uncharge(mol)
        mol = R["tautomer"].Canonicalize(mol)                           # 4. tautomer
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    mw, hac = R["Descriptors"].MolWt(mol), mol.GetNumHeavyAtoms()       # 5. med-chem filter
    if not (mw_range[0] <= mw <= mw_range[1] and hac_range[0] <= hac <= hac_range[1]):
        return None
    if any(a.GetSymbol() not in allowed for a in mol.GetAtoms()):
        return None
    can = Chem.MolToSmiles(mol, isomericSmiles=True)                    # 6. identity keys
    ik = Chem.MolToInchiKey(mol)
    if not ik:
        return None
    try:
        scaf = Chem.MolToSmiles(R["Murcko"].GetScaffoldForMol(mol), isomericSmiles=False)
    except Exception:
        scaf = ""
    return {"smiles": can, "inchikey": ik, "inchikey_block1": ik[:14], "scaffold": scaf or ik[:14]}
