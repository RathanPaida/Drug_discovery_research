"""Tests for the two unbounded caches that made a full-data run impossible.

Both were fine at the original 425,906-record scale and fail at the ~1.35M-record scale
that `data.full_kinome: true` ingests. Both fail LATE -- hours into Stage 3 -- which is the
expensive way to find out.

  * `_CACHE` in tnk1/models/data.py held every featurised molecular graph. At ~20 kB each
    over several hundred thousand unique structures that is ~10 GB.
  * `DescStore.cache` preloaded all 787,283 descriptor rows (~1 GB resident, ~1.26 GB
    transient) and then grew without limit on every miss.

Neither bound changes results: a miss is recomputed, so the cap trades wall-clock only.
"""
import importlib
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is required for the data-module tests")
pytest.importorskip("rdkit", reason="rdkit is required to featurise molecules")

SMILES = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCC", "c1ccncc1", "CCOCC", "CC(C)O"]


def _reload_data(monkeypatch, **env):
    """The caches read their ceiling at import time, so the module must be reimported."""
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, str(v))
    import tnk1.models.data as d
    return importlib.reload(d)


# ───────────────────────────────────────────────────────── graph cache

def test_graph_cache_evicts_once_past_its_ceiling(monkeypatch):
    d = _reload_data(monkeypatch, TNK1_GRAPH_CACHE=3)
    for s in SMILES:
        d.get_graph(s)
    assert d.cache_stats()["entries"] <= 3, "graph cache grew past its ceiling"


def test_graph_cache_still_returns_correct_graphs_under_eviction(monkeypatch):
    d = _reload_data(monkeypatch, TNK1_GRAPH_CACHE=2)
    first = d.get_graph("CCO")
    for s in SMILES:
        d.get_graph(s)
    again = d.get_graph("CCO")          # evicted, so recomputed
    assert again["x"].shape == first["x"].shape
    assert np.array_equal(again["x"], first["x"]), "recomputed graph differs from cached"


def test_graph_cache_bound_can_be_disabled(monkeypatch):
    d = _reload_data(monkeypatch, TNK1_GRAPH_CACHE=0)
    for s in SMILES:
        d.get_graph(s)
    assert d.cache_stats()["entries"] == len(SMILES)
    assert d.cache_stats()["max"] is None


def test_graph_cache_defaults_to_a_finite_ceiling(monkeypatch):
    """The shipped default was unbounded, which OOMs a 16 GB box partway through Stage 3."""
    d = _reload_data(monkeypatch, TNK1_GRAPH_CACHE=None)
    assert d.cache_stats()["max"] is not None


# ───────────────────────────────────────────────────── descriptor cache

class _IdentityScaler:
    def transform(self, X):
        return np.asarray(X, dtype=np.float32)


def test_desc_store_respects_its_cap_on_misses(monkeypatch):
    d = _reload_data(monkeypatch, TNK1_DESC_CACHE=4, TNK1_FEATURES_DIR="/nonexistent")
    store = d.DescStore(_IdentityScaler())
    out = store.get(SMILES)
    assert out.shape[0] == len(SMILES)
    assert len(store.cache) <= 4, "descriptor cache grew past its ceiling"


def test_desc_store_with_caching_disabled_retains_nothing(monkeypatch):
    d = _reload_data(monkeypatch, TNK1_DESC_CACHE=0, TNK1_FEATURES_DIR="/nonexistent")
    store = d.DescStore(_IdentityScaler())
    out = store.get(SMILES)
    assert out.shape[0] == len(SMILES)
    assert len(store.cache) == 0, "cache disabled but entries were retained"


def test_desc_store_returns_the_same_values_whatever_the_cap(monkeypatch):
    """The cap must trade wall-clock, never correctness."""
    d = _reload_data(monkeypatch, TNK1_DESC_CACHE=0, TNK1_FEATURES_DIR="/nonexistent")
    a = d.DescStore(_IdentityScaler()).get(SMILES)
    d = _reload_data(monkeypatch, TNK1_DESC_CACHE=3, TNK1_FEATURES_DIR="/nonexistent")
    b = d.DescStore(_IdentityScaler()).get(SMILES)
    assert np.allclose(a, b, equal_nan=True)


def test_desc_store_handles_repeated_smiles_in_one_call(monkeypatch):
    d = _reload_data(monkeypatch, TNK1_DESC_CACHE=2, TNK1_FEATURES_DIR="/nonexistent")
    store = d.DescStore(_IdentityScaler())
    out = store.get(["CCO", "CCO", "c1ccccc1", "CCO"])
    assert out.shape[0] == 4
    assert np.allclose(out[0], out[1]) and np.allclose(out[0], out[3])
