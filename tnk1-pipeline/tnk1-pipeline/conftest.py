"""Make the `tnk1` package importable without an editable install.

The suite previously only collected when pytest happened to be invoked from this
directory (`python -m pytest tests` adds the cwd to sys.path). Running it from the repo
root -- as the top-level README instructs (`pytest tnk1-pipeline/tnk1-pipeline/tests/`) --
failed with ModuleNotFoundError: tnk1. Prepending the package root here makes both forms
work; `pip install -e tnk1-pipeline/tnk1-pipeline` remains the preferred setup.
"""
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
