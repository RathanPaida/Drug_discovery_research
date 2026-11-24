"""Regression tests for the old+new kinome union.

Two defects are pinned here.

1. THE TARGET-COLLAPSE BUG that the second acquisition drop shipped. Its
   `aggregate_replicates` filtered its own grouping key down to the columns that
   happened to be present:

       key = tuple(k for k in key if k in df.columns)

   so a frame carrying `accession` but not `target_gene` was grouped by
   (inchikey, endpoint) alone and every kinase for a compound collapsed into one
   median. `tnk1.data.activity.aggregate_replicates` must never do that: its key is
   hardcoded, and a frame missing a key column has to raise rather than silently
   produce a smaller, wrong table.

2. CROSS-SOURCE DOUBLE COUNTING. Each source is deduped internally, so until a
   second kinome source existed nothing deduped across them. The two sources
   overlap on 90.6% of compounds, so the same (compound, kinase, endpoint) cell
   arrives twice with two different values. The resolution order is exact-beats-
   censored, then more-replicates, then tighter-spread.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tnk1.data import activity


def _rec(ik, uni, typ, pact, censored=0, n=1, smiles="CCO", scaffold="c1ccccc1",
         source="A"):
    return {"inchikey_block1": ik, "uniprot_id": uni, "standard_type": typ,
            "pact": pact, "censored": censored, "smiles": smiles,
            "scaffold": scaffold, "source": source, "value_nM": 10.0,
            "relation": "=" if not censored else ">", "n_rep": n}


# --------------------------------------------------------------------------- #
# 1. the kinase must survive aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_keeps_one_row_per_kinase():
    """One compound measured on three kinases stays three rows, with three values.

    This is the exact shape the shipped file got wrong: it would return a single
    row at the median of 9.0/5.0/5.0.
    """
    d = pd.DataFrame([
        _rec("AAAAAAAAAAAAAA", "Q13470", "IC50", 9.0),   # TNK1, potent
        _rec("AAAAAAAAAAAAAA", "P00519", "IC50", 5.0),   # ABL1, weak
        _rec("AAAAAAAAAAAAAA", "P12931", "IC50", 5.0),   # SRC,  weak
    ])
    out = activity.aggregate_replicates(d)

    assert len(out) == 3, "one row per (compound, kinase, endpoint)"
    assert set(out["uniprot_id"]) == {"Q13470", "P00519", "P12931"}
    got = dict(zip(out["uniprot_id"], out["pact"]))
    assert got["Q13470"] == pytest.approx(9.0)
    # The selectivity signal -- 4 log units between TNK1 and the off-targets -- is
    # what the collapse destroyed. If this ever reads ~5.2 the bug is back.
    assert got["Q13470"] - got["P00519"] == pytest.approx(4.0)


def test_aggregate_key_is_not_silently_narrowed():
    """A frame missing a key column must raise, not group on what is left.

    The acquisition drop's version returned a table with no target column and no
    error. pandas raises KeyError for a missing groupby key, and this test exists
    so nobody 'fixes' that by filtering the key to present columns.
    """
    d = pd.DataFrame([_rec("AAAAAAAAAAAAAA", "Q13470", "IC50", 9.0)])
    d = d.drop(columns=["uniprot_id"])
    with pytest.raises(KeyError):
        activity.aggregate_replicates(d)


def test_replicates_on_same_kinase_still_collapse():
    """The fix must not disable replicate collapsing on a single kinase."""
    d = pd.DataFrame([
        _rec("AAAAAAAAAAAAAA", "Q13470", "IC50", 7.0),
        _rec("AAAAAAAAAAAAAA", "Q13470", "IC50", 7.2),
    ])
    out = activity.aggregate_replicates(d)
    assert len(out) == 1
    assert out["pact"].iloc[0] == pytest.approx(7.1)
    assert int(out["n_rep"].iloc[0]) == 2


# --------------------------------------------------------------------------- #
# 2. cross-source dedupe, as Stage 1 applies it
# --------------------------------------------------------------------------- #
KEY = ["inchikey_block1", "uniprot_id", "standard_type"]


def _dedupe(final: pd.DataFrame) -> pd.DataFrame:
    """The resolution rule from run_stage1_data.py, in isolation.

    Kept in step with that block by `test_dedupe_matches_stage1_source`, which
    greps the script for the same sort keys.
    """
    order = pd.DataFrame({
        "censored": pd.to_numeric(final["censored"], errors="coerce").fillna(1),
        "n_rep": -pd.to_numeric(final["n_rep"], errors="coerce").fillna(1),
        "sigma_rep": pd.to_numeric(final["sigma_rep"], errors="coerce").fillna(0.0),
    })
    out = (final.assign(**{f"_o_{c}": order[c] for c in order})
                .sort_values([f"_o_{c}" for c in order], kind="stable"))
    out = out[~out.duplicated(subset=KEY, keep="first")]
    return out.drop(columns=[c for c in out.columns if c.startswith("_o_")])


def _row(source, pact, censored=0, n_rep=1, sigma=0.0):
    return {"inchikey_block1": "AAAAAAAAAAAAAA", "uniprot_id": "Q13470",
            "standard_type": "IC50", "smiles": "CCO", "scaffold": "c1ccccc1",
            "pact": pact, "censored": censored, "n_rep": n_rep,
            "sigma_rep": sigma, "source": source, "weight": 1.0,
            "gene_name": "TNK1"}


def test_exact_beats_censored_regardless_of_order():
    """A censored bound must never displace a real measurement, whichever source
    is listed first -- otherwise the table's contents depend on config ordering."""
    for rows in ([_row("OLD", 6.0, censored=1), _row("NEW", 8.0, censored=0)],
                 [_row("NEW", 8.0, censored=0), _row("OLD", 6.0, censored=1)]):
        out = _dedupe(pd.DataFrame(rows))
        assert len(out) == 1
        assert int(out["censored"].iloc[0]) == 0
        assert out["pact"].iloc[0] == pytest.approx(8.0)
        assert out["source"].iloc[0] == "NEW"


def test_more_replicates_wins_when_both_exact():
    out = _dedupe(pd.DataFrame([_row("OLD", 7.0, n_rep=6),
                                _row("NEW", 7.9, n_rep=1)]))
    assert len(out) == 1
    assert int(out["n_rep"].iloc[0]) == 6
    assert out["pact"].iloc[0] == pytest.approx(7.0)


def test_tighter_spread_wins_at_equal_support():
    out = _dedupe(pd.DataFrame([_row("OLD", 7.0, n_rep=3, sigma=0.9),
                                _row("NEW", 7.4, n_rep=3, sigma=0.1)]))
    assert len(out) == 1
    assert out["sigma_rep"].iloc[0] == pytest.approx(0.1)
    assert out["source"].iloc[0] == "NEW"


def test_distinct_cells_are_not_merged():
    """Dedupe must only collapse genuine collisions. Different kinase, different
    endpoint and different compound are all distinct cells."""
    rows = [_row("OLD", 7.0)]
    for field, value in [("uniprot_id", "P00519"), ("standard_type", "Ki"),
                         ("inchikey_block1", "BBBBBBBBBBBBBB")]:
        r = _row("NEW", 5.0)
        r[field] = value
        rows.append(r)
    out = _dedupe(pd.DataFrame(rows))
    assert len(out) == 4, "no two of these share the full key"


def test_blank_identity_rows_are_dropped_before_dedupe():
    """A blank InChIKey would collide with every other blank one, so blanks have
    to go before the dedupe, not after."""
    rows = [_row("OLD", 7.0), _row("NEW", 5.0), _row("NEW", 6.0)]
    rows[1]["inchikey_block1"] = ""
    rows[2]["inchikey_block1"] = None
    final = pd.DataFrame(rows)

    blank = pd.Series(False, index=final.index)
    for c in ["inchikey_block1", "uniprot_id", "smiles"]:
        col = final[c].astype(str)
        blank |= final[c].isna() | (col.str.strip() == "") | (col == "nan")
    _p = pd.to_numeric(final["pact"], errors="coerce").astype("float64")
    blank |= pd.Series(~np.isfinite(_p.to_numpy()), index=final.index)

    assert int(blank.sum()) == 2
    out = _dedupe(final[~blank])
    assert len(out) == 1
    assert out["pact"].iloc[0] == pytest.approx(7.0)


def test_nonfinite_pact_is_dropped():
    final = pd.DataFrame([_row("OLD", np.nan), _row("NEW", np.inf),
                          _row("X", 7.0)])
    _p = pd.to_numeric(final["pact"], errors="coerce").astype("float64")
    blank = pd.Series(~np.isfinite(_p.to_numpy()), index=final.index)
    assert int(blank.sum()) == 2
    assert _dedupe(final[~blank])["pact"].iloc[0] == pytest.approx(7.0)


# --------------------------------------------------------------------------- #
# 3. the config and the script must stay in step
# --------------------------------------------------------------------------- #
def test_dedupe_matches_stage1_source():
    """The helper above duplicates Stage 1's rule. If that block changes its sort
    keys, this test fails and says so, instead of the two drifting apart."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "run_stage1_data.py").read_text()
    assert 'final.duplicated(subset=key, keep="first")' in src
    for token in ['"censored": pd.to_numeric', '"n_rep": -pd.to_numeric',
                  '"sigma_rep": pd.to_numeric']:
        assert token in src, f"Stage 1 no longer sorts by {token!r}"
    assert 'key = ["inchikey_block1", "uniprot_id", "standard_type"]' in src


@pytest.mark.parametrize("name", ["hpc.yaml", "hpc_fast.yaml", "hpc_smoke.yaml",
                                  "default.yaml"])
def test_configs_declare_the_extra_kinome_source(name):
    """Every config that exists in THIS tree must list the source.

    The repository root and the TO_RUN_ON_GPU bundle hold parallel copies of scripts,
    tnk1/ and tests/, but the hpc*.yaml configs are bundle-only -- the root tree has
    just default.yaml. An absent config is therefore skipped, not failed; a present
    one that forgets the source is failed.
    """
    import pathlib
    import yaml
    p = pathlib.Path(__file__).resolve().parents[1] / "configs" / name
    if not p.exists():
        pytest.skip(f"{name} does not exist in this tree")
    cfg = yaml.safe_load(p.read_text())
    extra = cfg["data"].get("extra_kinome_parquets")
    assert extra, f"{name} does not list extra_kinome_parquets"
    assert "kinome_chembl37.parquet" in extra
