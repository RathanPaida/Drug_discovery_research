"""Step 1.3: pActivity transform, replicate aggregation, right-censoring, and the
measurability filter that decides which rows the regression head may consume."""
from __future__ import annotations
import numpy as np
import pandas as pd

UNIT_TO_NM = {"nM": 1.0, "uM": 1e3, "µM": 1e3, "μM": 1e3, "pM": 1e-3, "mM": 1e6, "M": 1e9}

# Assay readouts that ARE a concentration at which an effect is defined, so
# pActivity = 9 - log10(value_nM) is meaningful. Everything else in the kinome store
# (OUTCOME, PCT_INHIBITION, PCT_CONTROL, PCT_ACTIVITY, Residual Activity, Activity,
# kon, koff, Delta Tm, FC, ...) is either a qualitative flag, a single-point percentage
# or a rate constant. Those need a classification or a separate head; `task_loss` cannot
# consume them.
DOSE_RESPONSE_TYPES = ("IC50", "Ki", "Kd", "EC50", "Potency", "AC50", "GI50")

# `pActivity == 0.0` is the kinome store's "no dose-response value" SENTINEL, not a
# measurement. 0.0 corresponds to value_nM = 1e9 = 1 molar, which no binding assay reports.
SENTINEL_PACT = 0.0

# Physically defensible window for a kinase binding/inhibition constant.
#   pAct 2  == 10 mM   -- weaker than any reported kinase inhibitor
#   pAct 12 == 1 pM    -- tighter than the tightest known small-molecule kinase binder
PACT_MIN, PACT_MAX = 2.0, 12.0


def filter_measurable(df: pd.DataFrame, *, type_col: str = "activity_type",
                      pact_col: str = "pActivity", types=DOSE_RESPONSE_TYPES,
                      pact_min: float = PACT_MIN, pact_max: float = PACT_MAX,
                      sentinel: float = SENTINEL_PACT,
                      target_col: str | None = None,
                      drop_composite_targets: bool = True) -> tuple[pd.DataFrame, dict]:
    """Keep only rows whose pActivity is a real, physically plausible dose-response value.

    Returns `(kept, report)`. `report` counts every exclusion by reason, so a caller can
    print exactly what was dropped instead of asserting it in a comment.

    WHY THIS FUNCTION EXISTS -- two silent defects it closes
    --------------------------------------------------------
    1. THE ZERO SENTINEL. `run_stage1_data.py` filtered with `keep = np.isfinite(pact)`
       and its comment stated that %-inhibition and qualitative rows "are dropped with a
       printed count". They were not: the store encodes them as pActivity **0.0**, and
       `np.isfinite(0.0)` is True. Measured on
       `kinome_all_measurements_3.95M.parquet` (4,798,165 rows): 3,374,532 rows (70.3%)
       carry pActivity exactly 0.0, every one of the 3,338,837 non-dose-response rows
       among them, and all of them with `is_censored = 0` -- i.e. presented to the loss as
       an EXACT measurement of an IC50 of one molar. `kinome_long_v2.parquet` contains
       3,372,521 such rows. A model trained on that table is fitted to predict 0.0 for
       seven rows in ten.

    2. UNIT-CORRUPTED ROWS. 108,450 rows have `0 != pActivity < 2.0`, i.e. value_nM
       between 1.0e7 and 1.3e20 (median 3.2e14 nM = 316,000 M). 107,795 are `Ki` records
       from PubChem whose value was evidently not in nM. They pass any finite-value check
       and read as very weak binders, dragging predictions down.

    Applying both filters to the 4.8M store keeps **1,315,069** rows, which is the
    ~1.35M that CHANGES_V2.md documents as usable. The previous code kept 4,796,129.
    """
    d = df
    n0 = len(d)
    report: dict = {"n_input": n0, "dropped": {}, "kept_by_type": {}}

    pact = pd.to_numeric(d[pact_col], errors="coerce")

    # 1. unusable readout kinds
    if type_col in d.columns:
        t = d[type_col].astype(str)
        is_dose = t.isin(set(types))
        report["dropped"]["not_dose_response"] = int((~is_dose).sum())
        report["dropped_types"] = (
            t[~is_dose].value_counts().head(20).to_dict() if (~is_dose).any() else {}
        )
    else:
        is_dose = pd.Series(True, index=d.index)
        report["dropped"]["not_dose_response"] = 0
        report["note"] = f"no {type_col!r} column; readout kind not filtered"

    # 2. non-numeric / NaN / inf
    finite = np.isfinite(pact)
    report["dropped"]["pact_not_finite"] = int((is_dose & ~finite).sum())

    # 3. the sentinel
    not_sentinel = pact != sentinel
    report["dropped"]["pact_sentinel_zero"] = int((is_dose & finite & ~not_sentinel).sum())

    # 4. physically impossible
    in_window = (pact >= pact_min) & (pact <= pact_max)
    report["dropped"]["pact_out_of_window"] = int(
        (is_dose & finite & not_sentinel & ~in_window).sum())
    report["pact_window"] = [pact_min, pact_max]

    keep = is_dose & finite & not_sentinel & in_window

    # 5. composite / multi-protein targets.
    #
    # The store carries 642 distinct `uniprot_accession` values that are semicolon-joined
    # lists of up to ~150 accessions -- panel aggregates and protein complexes, e.g.
    # "Q07912;P00519;Q06187;...". 211,546 rows of the filtered table carry one. They are
    # not measurements of a single kinase, so there is no defensible label: attributing the
    # value to one accession is wrong, and duplicating the row across 150 accessions
    # fabricates 150 measurements from one.
    #
    # They were already being discarded, but SILENTLY -- `kinome_stage` does
    # `df[df.uniprot_id.isin(kmap)]` against the 491 single accessions in kinase_store, so
    # 16% of the table vanished with no count anywhere, and `df.uniprot_id.nunique()`
    # reported 1054 "kinases" when only 491 are real. Dropping them here makes it a
    # reported decision. Set drop_composite_targets=False to keep them for a model that
    # can represent a target set.
    if target_col and drop_composite_targets and target_col in d.columns:
        tgt = d[target_col].astype(str)
        composite = tgt.str.contains(";", regex=False)
        report["dropped"]["composite_target"] = int((keep & composite).sum())
        report["n_composite_target_values"] = int(tgt[composite].nunique())
        keep = keep & ~composite
    elif target_col and not drop_composite_targets:
        report["note_composite"] = "composite targets RETAINED by request"

    kept = d[keep]
    report["n_kept"] = int(len(kept))
    report["n_dropped"] = n0 - int(len(kept))
    if type_col in d.columns and len(kept):
        report["kept_by_type"] = kept[type_col].astype(str).value_counts().to_dict()
    return kept, report


def format_filter_report(report: dict) -> str:
    """Human-readable summary; stage scripts print this so the exclusions are in the log."""
    lines = [f"  input rows            : {report['n_input']:,}"]
    for reason, n in report["dropped"].items():
        if n:
            lines.append(f"  dropped {reason:<22}: {n:,}")
    lo, hi = report.get("pact_window", (PACT_MIN, PACT_MAX))
    lines.append(f"  kept (pAct in [{lo}, {hi}], dose-response only): {report['n_kept']:,}")
    if report.get("kept_by_type"):
        kinds = ", ".join(f"{k}={v:,}" for k, v in sorted(
            report["kept_by_type"].items(), key=lambda kv: -kv[1]))
        lines.append(f"  kept by type          : {kinds}")
    if report.get("dropped_types"):
        kinds = ", ".join(f"{k}={v:,}" for k, v in sorted(
            report["dropped_types"].items(), key=lambda kv: -kv[1])[:8])
        lines.append(f"  dropped readout kinds : {kinds}")
    return "\n".join(lines)


def to_pactivity(value_nM):
    """pActivity = 9 - log10(value in nM): 1 uM -> 6, 100 nM -> 7, 10 nM -> 8."""
    return 9.0 - np.log10(np.asarray(value_nM, dtype=float))


def prepare_records(df: pd.DataFrame) -> pd.DataFrame:
    """Input columns: inchikey_block1, smiles, scaffold, uniprot_id, standard_type, value_nM, source,
    and optionally relation. '=' (or a missing/blank relation) -> exact; '>'/'>=' -> censored, meaning
    the true pActivity lies BELOW the recorded bound; '<' records are dropped (left-censored potent
    values are not modelled by the manual's loss).

    Each censored row keeps its OWN bound, derived from its value_nM -- a ">1 uM" record bounds
    pActivity at 6, a ">100 uM" record at 4. There is deliberately no global tau here: the previous
    signature accepted `tau` and never used it (`run_stage1_data.py` passed `cfg.data.censor_tau`
    expecting an effect). `censor_tau` applies to newly assayed active-learning rows, where the
    protocol has a single threshold -- see `tnk1.active_learning.ingest.to_training_rows`.
    """
    d = df.copy()
    # `df.get("relation", "=")` returns the *scalar* "=" when the column is absent, and the
    # following .fillna() then raised AttributeError on a str. The PKIS and KINOMEscan inputs
    # declared in configs/default.yaml have no `relation` column, so this crashed on exactly
    # the datasets it was meant to accept.
    if "relation" in d.columns:
        rel = d["relation"].astype("object").where(d["relation"].notna(), "=")
        rel = rel.replace({"": "=", None: "="})
    else:
        rel = pd.Series("=", index=d.index, dtype="object")
    d["relation"] = rel
    d = d[d["value_nM"] > 0]
    d = d[~d["relation"].isin(["<", "<=", "<<"])]
    d["censored"] = d["relation"].isin([">", ">=", ">>"]).astype(int)
    d["pact"] = to_pactivity(d["value_nM"])
    return d


def aggregate_replicates(d: pd.DataFrame, range_max: float = 1.0, policy: str = "drop") -> pd.DataFrame:
    """Median pActivity per (inchikey_block1, uniprot_id, standard_type). If exact measurements exist they
    take precedence; else the tightest censor bound is kept. Replicate sets spanning > range_max are dropped
    (policy='drop') or kept at the median (policy='keep_median'); sigma_rep and n_rep are stored."""
    keys = ["inchikey_block1", "uniprot_id", "standard_type"]
    rows = []
    for k, g in d.groupby(keys, sort=False):
        ex = g[g.censored == 0]
        if len(ex):
            rng = ex.pact.max() - ex.pact.min()
            if len(ex) > 1 and rng > range_max and policy == "drop":
                continue
            rows.append({**dict(zip(keys, k)), "smiles": ex.smiles.iloc[0], "scaffold": ex.scaffold.iloc[0],
                         "pact": float(ex.pact.median()), "censored": 0, "n_rep": len(ex),
                         "sigma_rep": float(ex.pact.std(ddof=0)) if len(ex) > 1 else np.nan,
                         "source": "|".join(sorted(set(ex.source.astype(str))))})
        else:
            rows.append({**dict(zip(keys, k)), "smiles": g.smiles.iloc[0], "scaffold": g.scaffold.iloc[0],
                         "pact": float(g.pact.min()), "censored": 1, "n_rep": len(g), "sigma_rep": np.nan,
                         "source": "|".join(sorted(set(g.source.astype(str))))})
    out = pd.DataFrame(rows)
    out["weight"] = 1.0
    return out
