"""Pairwise string similarity for aligned lists (a[i] vs b[i]), 0..1.

Uses RapidFuzz's multi-threaded `process.cpdist` when available (the cluster
env); otherwise a slow pure-Python fallback so the pipeline still runs for
smoke tests. Empty-vs-anything scores 0.
"""
from __future__ import annotations

import difflib
from typing import Sequence

import numpy as np

try:
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler
    _RF = True
    _CPDIST = hasattr(process, "cpdist")
except ImportError:           # pragma: no cover - exercised only without rapidfuzz
    _RF = _CPDIST = False

SCORERS = ("ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio", "jaro_winkler")


def backend() -> str:
    return "rapidfuzz.cpdist" if _CPDIST else ("rapidfuzz" if _RF else "python-fallback")


# -- fallback implementations ---------------------------------------------
def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0


def _jaro_winkler(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    win = max(len(a), len(b)) // 2 - 1
    ma, mb = [False] * len(a), [False] * len(b)
    m = 0
    for i, ch in enumerate(a):
        for j in range(max(0, i - win), min(len(b), i + win + 1)):
            if not mb[j] and b[j] == ch:
                ma[i] = mb[j] = True
                m += 1
                break
    if m == 0:
        return 0.0
    ta = [c for c, f in zip(a, ma) if f]
    tb = [c for c, f in zip(b, mb) if f]
    t = sum(x != y for x, y in zip(ta, tb)) / 2
    jaro = (m / len(a) + m / len(b) + (m - t) / m) / 3
    pre = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        pre += 1
    return jaro + pre * 0.1 * (1 - jaro)


def _fallback(name: str, a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if name == "ratio":
        return _ratio(a, b)
    if name == "jaro_winkler":
        return _jaro_winkler(a, b)
    ta, tb = a.split(), b.split()
    if name == "token_sort_ratio":
        return _ratio(" ".join(sorted(ta)), " ".join(sorted(tb)))
    if name == "token_set_ratio":
        sa, sb = set(ta), set(tb)
        inter = " ".join(sorted(sa & sb))
        ra = (inter + " " + " ".join(sorted(sa - sb))).strip()
        rb = (inter + " " + " ".join(sorted(sb - sa))).strip()
        return max(_ratio(inter, ra), _ratio(inter, rb), _ratio(ra, rb))
    if name == "partial_ratio":
        # smoke-test approximation: best block-aligned window (RapidFuzz is exact)
        s, l = (a, b) if len(a) <= len(b) else (b, a)
        blocks = difflib.SequenceMatcher(None, s, l, autojunk=False).get_matching_blocks()
        starts = {max(0, min(len(l) - len(s), bj - bi)) for bi, bj, _ in blocks}
        return max(_ratio(s, l[i:i + len(s)]) for i in starts)
    raise ValueError(name)


# -- public API -----------------------------------------------------------
def _rf_scorer(name: str):
    return {"ratio": fuzz.ratio, "partial_ratio": fuzz.partial_ratio,
            "token_sort_ratio": fuzz.token_sort_ratio, "token_set_ratio": fuzz.token_set_ratio,
            "jaro_winkler": JaroWinkler.normalized_similarity}[name]


def pairwise(name: str, a: Sequence[str], b: Sequence[str], workers: int = 1,
             empty: np.ndarray | None = None) -> np.ndarray:
    """`empty` (optional): precomputed mask of pairs where either side is ''."""
    if name not in SCORERS:
        raise ValueError(name)
    n = len(a)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    scale = 1.0 if name == "jaro_winkler" else 100.0
    if _CPDIST:
        out = process.cpdist(a, b, scorer=_rf_scorer(name), workers=workers, dtype=np.float32)
        out = np.asarray(out, dtype=np.float32) / scale
    elif _RF:
        sc = _rf_scorer(name)
        out = np.fromiter((sc(x, y) for x, y in zip(a, b)), dtype=np.float32, count=n) / scale
    else:
        out = np.fromiter((_fallback(name, x, y) for x, y in zip(a, b)), dtype=np.float32, count=n)
    # identical conventions for empties regardless of backend
    if empty is None:
        empty = np.fromiter(((not x) or (not y) for x, y in zip(a, b)), dtype=bool, count=n)
    out[empty] = 0.0
    return out
