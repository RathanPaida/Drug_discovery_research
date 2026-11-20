"""The model registry: machinery only. `regressors.py` and `gp.py` fill it.

Keeping the machinery separate from the entries means a new model is one `register(...)`
call next to its implementation, and `tnk1/zoo/benchmark.py` never needs editing to pick it
up. It also keeps this module importable with nothing but numpy installed, so
`available()` can report what the environment is missing instead of raising an ImportError
halfway through a sweep that has already burned an hour of GPU time.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One candidate base model.

    name      registry key, used in configs, CSVs and `--models` arguments
    ctor      params dict -> an object honouring fit/predict(return_std=)
    rep       default representation key from `tnk1.zoo.features.REGISTRY`
    grid      hyperparameter grid for the inner CV; {} means "no tuning"
    needs     importable top-level modules required, beyond numpy/sklearn
    gives_std True iff `predict(return_std=True)` returns a genuine predictive sd
    family    coarse grouping, used for ensemble member DIVERSITY selection: picking the
              best k models by score alone tends to pick k variants of one algorithm,
              whose errors are correlated, and a correlated ensemble gains nothing
              (Brown et al., "Diversity creation methods", Information Fusion 2005)
    cost      rough relative fit cost at N~200; the sweep orders by this so a cheap
              shortlist finishes even if the expensive tail is cut off
    cite      paper, so the choice is traceable from the code
    """
    name: str
    ctor: Callable[[dict], Any]
    rep: str = "ecfp4_count"
    grid: dict[str, list] = field(default_factory=dict)
    needs: tuple[str, ...] = ()
    gives_std: bool = False
    family: str = "other"
    cost: float = 1.0
    cite: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    if spec.name in REGISTRY:
        raise ValueError(f"model {spec.name!r} is already registered")
    REGISTRY[spec.name] = spec
    return spec


def build(name: str, params: dict | None = None):
    """Instantiate a registered model. Raises a readable error on a missing dependency."""
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; have {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    missing = _missing(spec)
    if missing:
        raise ImportError(
            f"model {name!r} needs {', '.join(missing)}, which this environment does not "
            f"have. Install the extra (`pip install -e '.[chem]'`) or exclude it from the "
            f"sweep; `tnk1.zoo.available()` lists what is runnable here.")
    return spec.ctor(dict(params or {}))


def _missing(spec: ModelSpec) -> list[str]:
    return [m for m in spec.needs if importlib.util.find_spec(m.split(".")[0]) is None]


def available(names=None) -> dict[str, str]:
    """-> {name: "ok" | "missing: a,b"}. Call this FIRST in any sweep and log the result.

    A sweep that discovers at model 14 of 20 that lightgbm is absent has wasted the
    13 fits before it and produced a results table whose absences look like failures.
    """
    return {n: ("ok" if not _missing(REGISTRY[n]) else f"missing: {','.join(_missing(REGISTRY[n]))}")
            for n in (names or REGISTRY)}


def shortlist(tag: str = "core", only_available: bool = True) -> list[str]:
    """Registered models carrying `tag`, cheapest first.

    Tags in use:
        "core"      the literature-backed shortlist to run every time
        "slow"      worth running, but budget for it
        "control"   reproduces an existing published number; always include
        "probe"     exploratory; may well lose
    """
    names = [n for n, s in REGISTRY.items() if tag in s.tags]
    if only_available:
        names = [n for n in names if not _missing(REGISTRY[n])]
    return sorted(names, key=lambda n: (REGISTRY[n].cost, n))


def families(names=None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for n in (names or REGISTRY):
        out.setdefault(REGISTRY[n].family, []).append(n)
    return {k: sorted(v) for k, v in out.items()}


def grid_points(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a hyperparameter grid -> list of param dicts.

    Mirrors `scripts/run_stage2_baselines.py::grid` so the two cannot drift. An empty grid
    yields a single empty dict, i.e. "fit once with the defaults", rather than no fits at
    all -- the subtler of the two possible bugs.
    """
    import itertools
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
