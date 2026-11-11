"""Make the `tnk1` package importable without an editable install.

Prepending the package root here makes running pytest from any directory work seamlessly;
`pip install -e .` remains the preferred editable setup.
"""
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
