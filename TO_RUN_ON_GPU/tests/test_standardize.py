"""Tests for the six-step RDKit standardisation and for Stage-3's row builder.

`tnk1/data/standardize.py` was at 0% coverage, which is the worst place in this repository
for a blind spot: `standardize_smiles` defines COMPOUND IDENTITY. Its `inchikey_block1` is
the key that

  * the frozen scaffold splits are stored by (`build_tnk1_splits`),
  * the leakage guard removes from every kinase (`kinome_train_mask`),
  * replicate aggregation groups on (`aggregate_replicates`),
  * and the worldwide novelty audit matches against (`tnk1.novelty.verifier`).

If salt stripping or tautomer canonicalisation changed, two records of the same molecule
would get different keys: the leakage guard would miss them, the same compound could sit
in train and test, and the "0/40 known in any registry" claim would be checking the wrong
hashes. None of that raises.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("rdkit")

from tnk1.data.standardize import standardize_smiles  # noqa: E402

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-n2cccn2)n1"


def test_returns_the_documented_keys():
    r = standardize_smiles(ASPIRIN)
    assert set(r) == {"smiles", "inchikey", "inchikey_block1", "scaffold"}
    assert len(r["inchikey"]) == 27, "a Standard InChIKey is 27 characters"
    assert len(r["inchikey_block1"]) == 14
    assert r["inchikey"].startswith(r["inchikey_block1"])


def test_invalid_and_empty_input_returns_none_rather_than_raising():
    for bad in ("", None, "not_a_molecule", "C(C)(C)(C)(C)C", 42):
        assert standardize_smiles(bad) is None, f"{bad!r} should be rejected"


def test_salt_is_stripped_to_the_largest_fragment():
    """A hydrochloride and its free base must collapse to the same identity, or the same
    compound reported by two sources gets two keys and the leakage guard misses one."""
    free = standardize_smiles("CN1CCN(CC1)c1ccccc1")
    salt = standardize_smiles("CN1CCN(CC1)c1ccccc1.Cl")
    assert free and salt
    assert salt["inchikey_block1"] == free["inchikey_block1"]
    assert "Cl" not in salt["smiles"]


def test_water_of_crystallisation_is_removed():
    a = standardize_smiles(ASPIRIN)
    b = standardize_smiles(ASPIRIN + ".O")
    assert a and b and a["inchikey"] == b["inchikey"]


def test_charge_is_neutralised():
    # MW must clear the 150 Da floor, so use a larger acid than benzoic (122 Da)
    neutral = standardize_smiles("OC(=O)c1ccc(C(=O)NCc2ccccc2)cc1")
    anion = standardize_smiles("[O-]C(=O)c1ccc(C(=O)NCc2ccccc2)cc1")
    assert neutral and anion
    assert anion["inchikey_block1"] == neutral["inchikey_block1"]


def test_tautomers_canonicalise_to_one_identity():
    """2-pyridone / 2-hydroxypyridine. Without canonicalisation these are two compounds."""
    # 2-pyridone itself is 95 Da, below the MW floor; use a substituted analogue
    a = standardize_smiles("O=c1cccc[nH]1", mw_range=(50, 900), hac_range=(5, 65))
    b = standardize_smiles("Oc1ccccn1", mw_range=(50, 900), hac_range=(5, 65))
    assert a and b
    assert a["smiles"] == b["smiles"], "tautomer canonicalisation did not converge"
    assert a["inchikey_block1"] == b["inchikey_block1"]


def test_output_smiles_is_canonical_and_round_trips():
    r = standardize_smiles(IMATINIB)
    assert r
    again = standardize_smiles(r["smiles"])
    assert again and again["smiles"] == r["smiles"]
    assert again["inchikey"] == r["inchikey"]


def test_molecular_weight_window_is_enforced():
    assert standardize_smiles("C", mw_range=(150, 900)) is None, "methane is too small"
    big = "C" * 400
    assert standardize_smiles(big, mw_range=(150, 900)) is None, "a C400 chain is too big"
    assert standardize_smiles(ASPIRIN, mw_range=(150, 900)) is not None


def test_heavy_atom_window_is_enforced():
    assert standardize_smiles(ASPIRIN, hac_range=(100, 200)) is None
    assert standardize_smiles(ASPIRIN, hac_range=(10, 65)) is not None


def test_disallowed_elements_are_rejected():
    """A kinase inhibitor dataset has no business containing organometallics; letting one
    through would put an unfeaturisable atom type into the D-MPNN."""
    ferrocene_like = "[Fe]"
    assert standardize_smiles(ferrocene_like) is None
    seleno = "C[Se]c1ccccc1"
    assert standardize_smiles(seleno) is None, "Se is not in the allowed set"
    assert standardize_smiles("Brc1ccccc1C(=O)NCCc1ccccc1") is not None, "Br is allowed"


def test_allowed_elements_is_configurable():
    seleno = "C[Se]c1ccccc1C(=O)NCCc1ccccc1"
    assert standardize_smiles(seleno) is None
    ok = standardize_smiles(seleno, allowed=("C", "N", "O", "Se", "H"))
    assert ok is not None


def test_scaffold_is_the_murcko_framework():
    r = standardize_smiles(ASPIRIN)
    assert r and r["scaffold"] == "c1ccccc1", "aspirin's Murcko scaffold is benzene"


def test_acyclic_molecule_gets_a_non_empty_scaffold_fallback():
    """An acyclic compound has an EMPTY Murcko scaffold. `build_tnk1_splits` groups folds
    BY scaffold, so if every acyclic molecule shared the empty string they would all land
    in one group -- and `scaffold_holdout` could put that whole group in the test fold."""
    r = standardize_smiles("CCCCCCCCCCCCCCCC(=O)O")      # palmitic acid
    assert r
    assert r["scaffold"], "empty scaffold would collapse all acyclic compounds into one group"
    assert r["scaffold"] == r["inchikey_block1"], "documented fallback is the identity key"


def test_distinct_molecules_keep_distinct_keys():
    mols = (ASPIRIN, IMATINIB, "CN1CCN(CC1)c1ccccc1",
            "OC(=O)c1ccc(C(=O)NCc2ccccc2)cc1")
    results = [standardize_smiles(s) for s in mols]
    assert all(r is not None for r in results), "a test molecule was filtered out"
    assert len({r["inchikey_block1"] for r in results}) == 4


def test_stereoisomers_share_block1_but_differ_in_the_full_key():
    """Block 1 is the connectivity layer: it must NOT distinguish stereoisomers, while the
    full 27-character key must. The novelty audit relies on exactly this split."""
    a = standardize_smiles("C[C@H](N)C(=O)Nc1ccc(Cl)cc1C(=O)NCc1ccccc1")
    b = standardize_smiles("C[C@@H](N)C(=O)Nc1ccc(Cl)cc1C(=O)NCc1ccccc1")
    assert a and b
    assert a["inchikey_block1"] == b["inchikey_block1"]
    assert a["inchikey"] != b["inchikey"]


def test_standardisation_is_idempotent_and_deterministic():
    first = standardize_smiles(IMATINIB)
    for _ in range(3):
        again = standardize_smiles(IMATINIB)
        assert again == first, "standardisation is not deterministic across calls"


# --------------------------------------------------------- Stage-3 row builder (_mk)
torch = pytest.importorskip("torch")
from tnk1.models.pipeline import _mk  # noqa: E402
import pandas as pd  # noqa: E402


class _Scaler:
    def transform(self, X):
        return np.zeros((len(X), 200), dtype=np.float32)


def _store():
    from tnk1.models.data import DescStore
    ds = DescStore.__new__(DescStore)
    ds.scaler, ds.cache = _Scaler(), {}
    return ds


def _frame(pact, censored, uni=("U0", "U0")):
    return pd.DataFrame({
        "smiles": ["c1ccccc1", "CCO"][: len(pact)],
        "uniprot_id": list(uni)[: len(pact)],
        "standard_type": ["IC50"] * len(pact),
        "pact": pact, "censored": censored, "weight": [1.0] * len(pact),
    })


def test_mk_keeps_the_per_row_bound_for_censored_rows():
    df = _frame([6.0, 4.0], [1, 1])
    ds = _mk(df, {"U0": 0}, _store())
    assert ds.y.tolist() == [6.0, 4.0], "censored rows must keep their own bounds"
    assert ds.c.tolist() == [1, 1]


def test_mk_falls_back_to_tau_only_for_a_censored_row_with_no_value():
    df = _frame([np.nan, 7.0], [1, 0])
    ds = _mk(df, {"U0": 0}, _store(), tau=5.0)
    assert ds.y[0] == pytest.approx(5.0), "censored + missing value -> tau"
    assert ds.y[1] == pytest.approx(7.0)


def test_mk_drops_rows_whose_kinase_is_absent_from_the_store():
    """`kinome_stage` maps uniprot -> index; a row naming an unknown kinase has no column
    to predict. Silently keeping it would index past the embedding table."""
    df = _frame([7.0, 7.0], [0, 0], uni=("U0", "UNKNOWN"))
    ds = _mk(df, {"U0": 0}, _store())
    assert len(ds) == 1
    assert ds.kin.tolist() == [0]


def test_mk_aligns_external_weights_after_dropping_rows():
    """The weight vector is positional. If rows are dropped without subsetting the weights,
    every surviving row gets a neighbour's weight."""
    df = _frame([7.0, 8.0], [0, 0], uni=("UNKNOWN", "U0"))
    w = np.array([99.0, 2.0])
    ds = _mk(df, {"U0": 0}, _store(), weights=w)
    assert len(ds) == 1
    assert ds.y[0] == pytest.approx(8.0)
    assert ds.w[0] == pytest.approx(2.0), "weights were not realigned to the kept rows"


def test_legacy_env_restores_the_stereo_destroying_behaviour(tmp_path):
    """The old behaviour must stay reachable, because the frozen v1/v2 tables and the
    Cycle-01 feature store were built with it. Run in a subprocess: the enumerator and its
    flag are cached at first use, so the env var cannot be toggled in-process."""
    import json
    import os
    import subprocess

    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from tnk1.data.standardize import standardize_smiles, preserves_stereo\n"
        "a = standardize_smiles('C[C@H](N)C(=O)Nc1ccc(Cl)cc1C(=O)NCc1ccccc1')\n"
        "b = standardize_smiles('C[C@@H](N)C(=O)Nc1ccc(Cl)cc1C(=O)NCc1ccccc1')\n"
        "print('@@' + __import__('json').dumps("
        "{'preserves': preserves_stereo(), 'same': a['inchikey'] == b['inchikey']}))\n"
        % str(ROOT)
    )

    def run(legacy):
        env = {**os.environ, "TNK1_LEGACY_TAUTOMER_STEREO": legacy}
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, cwd=str(ROOT))
        line = next(l for l in p.stdout.splitlines() if l.startswith("@@"))
        return json.loads(line[2:])

    fixed = run("0")
    assert fixed["preserves"] is True
    assert fixed["same"] is False, "enantiomers must be distinguishable by default"

    legacy = run("1")
    assert legacy["preserves"] is False
    assert legacy["same"] is True, "the legacy escape hatch no longer reproduces the old keys"
