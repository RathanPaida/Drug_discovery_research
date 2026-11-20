"""Transfer-learning strategies and pretraining-task selection.

    finetune.py        named fine-tuning ladders (LP-FT, surgical, gradual, SWA, mixout)
    task_selection.py  WHICH of the 491 kinases belong in the pretraining set -- pure numpy

Split that way because the two answer different questions. `finetune.py` asks how to adapt
a trunk that has already been pretrained; `task_selection.py` asks what it should have been
pretrained on, and is the one this project has never asked: the pan-kinome stage WEIGHTS
all 491 kinases by pocket similarity but SELECTS none, so a kinase whose SAR is unrelated
to TNK1's still contributes gradient.

Only `task_selection` is imported here. `finetune` pulls in torch through
`tnk1.models.pipeline`, and importing it eagerly would make `import tnk1.transfer` fail in
the numpy-only environment where the selection logic is perfectly runnable and tested.
"""
from __future__ import annotations

from . import task_selection

__all__ = ["task_selection", "finetune", "get_strategy", "describe"]


def __getattr__(name):
    """Resolve `finetune` and its helpers on first use, so torch stays optional."""
    if name in ("finetune", "get_strategy", "describe", "STRATEGIES"):
        from importlib import import_module
        mod = import_module(".finetune", __name__)
        return mod if name == "finetune" else getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
