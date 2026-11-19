"""Unit tests for the level-2 features, the grouped expected-F decision and model
save/load (CPU only). Run with `python -m pytest tests -q`."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber import collective, decide, gbdt  # noqa: E402
from ber.config import load_config  # noqa: E402


def test_house_rel_classes():
    df = pl.DataFrame({"a": ["17", "1505", "17", "17", None, "17"],
                       "b": ["17", "505", "19", "71", "3", "300"]})
    out = df.select(collective.house_rel(pl.col("a"), pl.col("b")).alias("h"))["h"].to_list()
    assert out == [0.0, 1.0, 2.0, 3.0, None, 3.0]


def test_competition_rivals_and_margins():
    # record 10 is wanted by S1 1 (p .9) and S1 2 (p .6); record 11 only by S1 1
    df = pl.DataFrame({"q": [1, 2, 1], "r": [10, 10, 11], "r_src": [2, 2, 3], "p1": [0.9, 0.6, 0.2]})
    out = collective.competition(df).sort("q", "r")
    row = {(a, b): i for i, (a, b) in enumerate(zip(out["q"], out["r"]))}
    assert abs(out["c_r_rival"][row[(1, 10)]] - 0.6) < 1e-6          # best other S1 for record 10
    assert abs(out["c_r_rival"][row[(2, 10)]] - 0.9) < 1e-6
    assert out["c_r_rival"][row[(1, 11)]] == 0.0                     # alone
    assert abs(out["c_r_margin"][row[(1, 10)]] - 0.3) < 1e-6
    assert out["c_q_n05"][row[(1, 10)]] == 1.0 and out["c_q_rank"][row[(1, 11)]] == 2.0


def test_drift_flags_neighbour_of_exact_copy():
    df = pl.DataFrame({"q": [1, 1, 1], "r": [10, 11, 12], "p1": [0.95, 0.9, 0.8], "h_rel": [0.0, 2.0, None]})
    out = collective.drift(df).sort("r")
    assert out["sib_exact_n"].to_list() == [0.0, 1.0, 1.0]
    assert out["drift_vs_exact_sib"].to_list() == [0.0, 1.0, None]


def test_namesake_gap():
    keys = pl.DataFrame({"idx": [1, 2, 10, 11, 12], "src": [1, 1, 2, 2, 3],
                         "nk": ["US|summit", "US|summit", None, None, None], "house": ["1", "2", "1", "1", None],
                         "name_s": [""] * 5, "addr_s": [""] * 5, "addr_empty": [False, False, False, False, True]})
    # S1 1 wins two address-bearing records, S1 2 none; record 12 (no address) is contested
    df = pl.DataFrame({"q": [1, 1, 1, 2], "r": [10, 11, 12, 12], "p1": [0.9, 0.9, 0.6, 0.55]})
    out = collective.namesake(df, keys, pl.Series([1, 2])).sort("q", "r")
    g = dict(zip(zip(out["q"], out["r"]), out["ns_gap"]))
    assert g[(1, 12)] == 2.0 and g[(2, 12)] == -2.0
    assert out["ns_size"].unique().to_list() == [2.0]


def test_expected_f_grouped_choice():
    P = np.zeros((3, decide.MAX_K), dtype=np.float32)
    P[0, :3] = [0.99, 0.98, 0.01]            # keep two
    P[1, :1] = [0.01]                        # keep none (likely a singleton)
    P[2, :2] = [0.97, 0.96]                  # keep both
    counts = np.array([3, 1, 2])
    k = decide._choose_k(P, counts, 0.5, "expected_f", 512, 0)
    assert k.tolist() == [2, 0, 2]


def test_only_countries_parsing():
    from ber.util import as_list, only_countries
    assert as_list("France") == ["France"] and as_list("[France, US]") == ["France", "US"]
    assert as_list(["France"]) == ["France"] and as_list(None) == [] and as_list([]) == []
    cfg = load_config(None, ["data.only_countries=France"])
    assert only_countries(cfg, "test") == ["France"] and only_countries(cfg, "train") == []


def test_model_save_load_roundtrip():
    cfg = load_config(None, ["gbdt.backend=sklearn", "gbdt.num_boost_round=30"])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3)).astype(np.float32)
    X[:, 1] = np.nan                          # all-missing column: dropped at fit, kept in the sidecar
    y = (X[:, 0] > 0).astype(int)
    m = gbdt.fit(cfg, X[:300], y[:300], X[300:], y[300:], ["a", "b", "c"])
    with tempfile.TemporaryDirectory() as d:
        m.save(Path(d) / "stack_fold0")
        m2 = gbdt.load_model(Path(d) / "stack_fold0", [])
        assert m2.all_features == ["a", "b", "c"] and m2.features == ["a", "c"]
        assert np.allclose(m.predict(X), m2.predict(X))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
