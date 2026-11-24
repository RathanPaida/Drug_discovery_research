"""The Stage-8 candidate pool must standardize across cores, and identically.

`[standardize_smiles(s) for s in pool]` was one core with no output. That was invisible
while the pool held ~89 molecules, which is all the starved funnel ever produced. With
`--agent` supplying the real `al.pool_size` of 200,000, it became the most expensive step
in the pipeline, and the stage runs once per RL objective -- eight times.

Measured on the cluster: 12.8 ms/molecule for curated ChEMBL structures. Generated SMILES
run several times slower, because they carry more tautomeric centres and
`tautomer.Canonicalize` is combinatorial in those. The observed rate on a real pool was
about 70 ms/molecule, i.e. ~4 h per cycle on one core while 15 of the job's 16 sat idle.

Two things are pinned here:

* parallel output equals serial output, element for element and in order -- the pool is
  positional, and a reordering would silently mis-associate every downstream score;
* `_std_one` does not swallow an import error. The first version caught bare Exception,
  and because `standardize_smiles` is imported inside `main()` rather than at module
  scope, every worker raised NameError and returned None. All 200,000 molecules would
  have been dropped and the stage would have printed "No candidate passed strict profile
  filters" -- a plausible-looking scientific result manufactured by an import bug.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Imported by NAME, not by file location: a process pool started with the "spawn" method
# re-imports the worker's module in each child, and a module registered under an ad-hoc
# name from spec_from_file_location cannot be re-created there.
import run_stage8_cycle as s8  # noqa: E402

JUNK = ["NOTASMILES", "", "@@@", None]

# standardize_smiles is not only a parser: it applies the med-chem filter too
# (MW 150-900, HAC 10-65, allowed elements). Every molecule here clears it, so a None
# in the output means a real failure rather than an intended rejection.
#   paracetamol MW 151 HAC 11 | caffeine MW 194 HAC 14 | ibuprofen MW 206 HAC 15
#   imatinib    MW 494 HAC 37 | aspirin   MW 180 HAC 13
REAL = ["CC(=O)Nc1ccc(O)cc1",
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Cc1ccc(cc1Nc1nccc(n1)-c1cccnc1)C(=O)Nc1ccc(CN2CCN(C)CC2)c(C)c1",
        "CC(=O)Oc1ccccc1C(=O)O"]

# Below the MW/HAC floor: dropped on purpose, not a malfunction.
TOO_SMALL = ["CCO", "c1ccccc1O", "C"]


def _keys(rows):
    return [None if r is None else r["inchikey_block1"] for r in rows]


def test_parallel_matches_serial_exactly():
    pool = (REAL * 4) + JUNK
    serial = s8._standardize_pool(pool, jobs=1)
    par = s8._standardize_pool(pool, jobs=4)
    assert len(serial) == len(pool), "one output per input, including failures"
    assert len(par) == len(pool)
    assert _keys(serial) == _keys(par), "parallel reordered or altered the pool"


def test_real_molecules_survive():
    """Guards the failure mode where everything returns None and the stage reports
    'No candidate passed strict profile filters' from an environment error."""
    rows = s8._standardize_pool(list(REAL), jobs=2)
    kept = [r for r in rows if r]
    assert len(kept) == len(REAL), (
        f"only {len(kept)} of {len(REAL)} valid molecules survived -- if this is 0, "
        f"_std_one is swallowing an import error again")
    for r in kept:
        assert r["inchikey_block1"] and len(r["inchikey_block1"]) == 14


def test_junk_is_dropped_not_fatal():
    rows = s8._standardize_pool(list(JUNK), jobs=2)
    assert len(rows) == len(JUNK)
    assert all(r is None for r in rows)


def test_med_chem_filter_still_applies_in_parallel():
    """standardize_smiles filters as well as parses, and parallelising must not change
    which molecules it rejects. Ethanol and phenol are below the MW/HAC floor."""
    rows = s8._standardize_pool(list(TOO_SMALL), jobs=2)
    assert all(r is None for r in rows), (
        "sub-threshold molecules are passing the med-chem filter under the pool")
    serial = s8._standardize_pool(list(TOO_SMALL), jobs=1)
    assert _keys(serial) == _keys(rows)


def test_std_one_lets_an_import_error_through():
    """A broken environment must crash, not silently empty the pool."""
    import inspect
    src = inspect.getsource(s8._std_one)
    assert "except Exception" not in src, (
        "_std_one catches bare Exception again; a NameError or ImportError in a worker "
        "would return None for every molecule and fake an empty candidate set")
    assert "from tnk1.data.standardize import standardize_smiles" in src, (
        "the import must be inside the worker: standardize_smiles is imported inside "
        "main(), so a module-level reference is a NameError in every child process")


def test_jobs_defaults_to_the_slurm_allocation(monkeypatch):
    """16 allocated cores should mean 16 used, not 1."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    captured = {}
    real = s8._std_one

    def spy(smi):
        captured["called"] = True
        return real(smi)

    monkeypatch.setattr(s8, "_std_one", spy)
    # jobs=None must read the allocation; one molecule keeps it to a single worker.
    out = s8._standardize_pool(["CCO"], jobs=None)
    assert len(out) == 1


@pytest.mark.parametrize("jobs", [1, 3])
def test_output_length_always_matches_input(jobs):
    pool = REAL + JUNK + REAL
    assert len(s8._standardize_pool(pool, jobs=jobs)) == len(pool)
