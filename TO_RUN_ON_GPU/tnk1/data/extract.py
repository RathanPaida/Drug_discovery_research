"""Step 1.1: multi-source extraction -> long table. All inputs are LOCAL files (no network in-pipeline)."""
from __future__ import annotations
import re, sqlite3
import numpy as np
import pandas as pd
from .activity import UNIT_TO_NM

CHEMBL_SQL = """
SELECT cs.canonical_smiles AS smiles, cp.chembl_id AS mol_id, cseq.accession AS uniprot_id,
       act.standard_type, act.standard_relation AS relation, act.standard_value, act.standard_units
FROM activities act
JOIN assays a            ON act.assay_id = a.assay_id
JOIN target_dictionary t ON a.tid = t.tid
JOIN target_components tc ON t.tid = tc.tid
JOIN component_sequences cseq ON tc.component_id = cseq.component_id
JOIN compound_structures cs ON act.molregno = cs.molregno
JOIN molecule_dictionary cp ON act.molregno = cp.molregno
WHERE t.target_type = 'SINGLE PROTEIN' AND t.organism = 'Homo sapiens'
  AND a.assay_type = 'B'
  AND act.standard_type IN ({types})
  AND act.data_validity_comment IS NULL
  AND act.standard_value IS NOT NULL
  AND cseq.accession IN ({uniprots})
"""


def _to_nm(value, units):
    return value * units.map(UNIT_TO_NM).astype(float)


def from_chembl(sqlite_path, kinase_uniprots, types=("IC50", "Ki", "Kd")) -> pd.DataFrame:
    q = CHEMBL_SQL.format(types=",".join(f"'{t}'" for t in types),
                          uniprots=",".join(f"'{u}'" for u in kinase_uniprots))
    with sqlite3.connect(sqlite_path) as con:
        df = pd.read_sql_query(q, con)
    df = df[df.standard_units.isin(UNIT_TO_NM)]
    df["value_nM"] = _to_nm(df.standard_value.astype(float), df.standard_units)
    df["source"] = "chembl"
    return df[["smiles", "uniprot_id", "standard_type", "relation", "value_nM", "source"]]


def from_bindingdb(tsv_path, kinase_uniprots) -> pd.DataFrame:
    """BindingDB flat TSV. Dose-response Ki/Kd/IC50 only; single-dose % inhibition is stored separately."""
    use = None
    chunks = []
    for ch in pd.read_csv(tsv_path, sep="\t", chunksize=200_000, low_memory=False, on_bad_lines="skip"):
        if use is None:
            col = lambda pat: next(c for c in ch.columns if re.search(pat, c))
            use = dict(smiles=col(r"^Ligand SMILES"), uni=col(r"UniProt \(SwissProt\) Primary ID of Target"),
                       ki=col(r"^Ki \(nM\)"), kd=col(r"^Kd \(nM\)"), ic=col(r"^IC50 \(nM\)"))
        ch = ch[ch[use["uni"]].isin(kinase_uniprots)]
        for t, c in (("Ki", use["ki"]), ("Kd", use["kd"]), ("IC50", use["ic"])):
            sub = ch[[use["smiles"], use["uni"], c]].dropna()
            vals = sub[c].astype(str)
            rel = vals.str.extract(r"^([<>]=?)")[0].fillna("=")
            num = pd.to_numeric(vals.str.replace(r"[<>=\s]", "", regex=True), errors="coerce")
            chunks.append(pd.DataFrame({"smiles": sub[use["smiles"]], "uniprot_id": sub[use["uni"]],
                                        "standard_type": t, "relation": rel, "value_nM": num, "source": "bindingdb"}))
    return pd.concat(chunks).dropna(subset=["value_nM"])


def from_long_csv(path, source) -> pd.DataFrame:
    """PKIS/PKIS2/KINOMEscan/DKK: columns smiles,uniprot_id,value_nM,standard_type[,relation]."""
    df = pd.read_csv(path)
    if "relation" not in df:
        df["relation"] = "="
    df["source"] = source
    return df[["smiles", "uniprot_id", "standard_type", "relation", "value_nM", "source"]]


def single_dose_to_aux(path) -> pd.DataFrame:
    """Single-dose %-inhibition screens are kept out of the main table (auxiliary pretraining only)."""
    return pd.read_csv(path)


# --------------------------------------------------------------------------------------
#  scripts/tnk1_harvest.py  ->  Stage 1
# --------------------------------------------------------------------------------------
#
# WHY THIS EXISTS. The harvester produces a 40-column table whose names were chosen for
# provenance (`uniprot_accession`, `activity_type`, `std_smiles`), while Stage 1's ingest
# path wants the five-column PKIS shape (`uniprot_id`, `standard_type`, `smiles`). Three
# of the five names differ, so a harvest CSV handed to `from_long_csv` raises KeyError on
# its final column selection -- and `from_long_csv` additionally does `df["source"] = source`,
# which would overwrite the real per-row provenance with a single literal string.
#
# So the harvester's output could not reach the pipeline at all, and the obvious
# workaround would have silently destroyed the source attribution.

HARVEST_RENAME = {"uniprot_accession": "uniprot_id", "activity_type": "standard_type"}

# Preferred SMILES column, best first. `std_smiles` is the standardised structure written
# by the harvester's post-processing; `smiles_used` is whatever it resolved (possibly via a
# name or CID lookup); `smiles_raw` is the string as deposited.
HARVEST_SMILES_COLS = ("std_smiles", "smiles_used", "smiles_raw")

# `measure_semantics` values whose number is NOT a single direct measurement of the
# quantity its `activity_type` names, and which therefore must not be pooled with one by
# `aggregate_replicates` (which groups on standard_type and takes a median):
#
#   kiba_score               a model-integrated score fusing Ki/Kd/IC50 (Tang et al. 2014).
#                            Not a concentration. Already excluded by the `types` filter;
#                            named here so the exclusion is explicit, not incidental.
#   gtopdb_curated_summary   already a MEDIAN over several curated experiments. Passing a
#                            pre-aggregated value into aggregate_replicates aggregates it
#                            a second time and records n_rep=1 for what may be a dozen
#                            underlying measurements, understating sigma_rep.
#   activity_outcome_only    an active/inactive flag carrying no value.
#
# Matching is a case-insensitive substring test, so a semantics string that later gains a
# citation or parenthetical still matches.
HARVEST_EXCLUDE_SEMANTICS = ("kiba_score", "gtopdb_curated_summary", "activity_outcome_only")

_STAGE1_COLS = ["smiles", "uniprot_id", "standard_type", "relation", "value_nM", "source"]


def from_harvest_csv(path, *, types=("IC50", "Ki", "Kd"), allow_censored: bool = True,
                     require_panel: bool = True, drop_composite: bool = True,
                     exclude_semantics=HARVEST_EXCLUDE_SEMANTICS,
                     distinguish_semantics: bool = False,
                     chunksize: int = 200_000) -> tuple[pd.DataFrame, dict]:
    """Ingest `TNK1_kinase_master.csv` from scripts/tnk1_harvest.py into Stage 1's schema.

    Returns `(rows, report)`. `report` counts every exclusion by reason so the caller can
    print what was dropped instead of asserting it in a comment -- the same contract as
    `tnk1.data.activity.filter_measurable`.

    The filter mirrors the harvester's own `is_training_ready_exact` flag
    (record_class == quantitative, an allowed activity_type, value_nM > 0, a usable
    structure, in_panel, no contamination flag) with one deliberate difference: that flag
    also demands `relation == "="`, which would discard every right-censored row. The
    pipeline models those -- `prepare_records` keeps each censored row's own bound and the
    loss has a censored term -- so `allow_censored=True` (the default) admits `>` and `>=`
    as well. `<` is dropped here rather than downstream, so its count is reported.

    Per-row `source` is preserved. That is the whole point of reading this table rather
    than routing it through `from_long_csv`.

    `distinguish_semantics=True` appends the semantics tag to `standard_type`, e.g.
    `Kd@kd_nM_matrix_Davis2011`. Use it when you want competition-binding Kd_app kept in
    its own replicate group rather than pooled with solution Kd. It changes the values in
    a grouping key, so a table built with it is NOT comparable to one built without.
    """
    want_num = {"value_nM"}
    report: dict = {"n_input": 0, "dropped": {}, "kept_by_type": {}, "kept_by_source": {},
                    "kept_by_semantics": {}}
    # Declared in the order they are applied, so the printed report reads top-down.
    for k in ("not_allowed_type", "not_quantitative", "missing_smiles", "bad_value",
              "left_censored", "censored_excluded", "not_in_panel", "invalid_structure",
              "contaminated", "composite_target", "excluded_semantics"):
        report["dropped"][k] = 0

    allowed_rel = {"=", ">", ">="} if allow_censored else {"="}
    kept_parts = []
    first = True
    for ch in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if first:
            missing = [c for c in ("uniprot_accession", "activity_type", "relation", "value_nM")
                       if c not in ch.columns]
            if not any(c in ch.columns for c in HARVEST_SMILES_COLS):
                missing.append(" or ".join(HARVEST_SMILES_COLS))
            if missing:
                raise ValueError(
                    f"{path} does not look like a tnk1_harvest master table; missing "
                    f"column(s): {', '.join(missing)}. Expected the file written as "
                    f"TNK1_kinase_master.csv by scripts/tnk1_harvest.py.")
            first = False
        report["n_input"] += len(ch)

        smi_col = next(c for c in HARVEST_SMILES_COLS if c in ch.columns)
        d = pd.DataFrame({
            "smiles": ch[smi_col].astype("string").fillna("").str.strip(),
            "uniprot_id": ch["uniprot_accession"].astype("string").fillna("").str.strip(),
            "standard_type": ch["activity_type"].astype("string").fillna("").str.strip(),
            "relation": ch["relation"].astype("string").fillna("=").str.strip().replace("", "="),
            "value_nM": pd.to_numeric(ch["value_nM"], errors="coerce"),
            "source": ch["source"].astype("string").fillna("").str.strip()
                      if "source" in ch.columns else pd.Series("harvest", index=ch.index, dtype="string"),
        })
        keep = pd.Series(True, index=ch.index)

        def _drop(mask, reason):
            nonlocal keep
            n = int((keep & mask).sum())
            report["dropped"][reason] += n
            keep = keep & ~mask

        # Each row is attributed to EXACTLY ONE reason -- the first that matches -- so the
        # counts partition the input and `sum(dropped) + n_kept == n_input` holds.
        #
        # The order is therefore part of the diagnostic, not an implementation detail. The
        # readout-kind checks come first because a PCT_INHIBITION or KIBA_score row has no
        # nM value by construction: testing `value_nM` before `activity_type` attributed
        # those rows to "bad_value", which reads as a unit-corrupted measurement and would
        # have someone hunting a conversion bug in rows that are simply not concentrations.
        # `bad_value` now means what it says: a row that claims to be a quantitative
        # concentration and is not one.
        _drop(~d.standard_type.isin(set(types)), "not_allowed_type")
        if "record_class" in ch.columns:
            _drop(ch["record_class"].astype("string").fillna("").ne("quantitative"), "not_quantitative")
        _drop(d.smiles.eq("") | d.uniprot_id.eq(""), "missing_smiles")
        _drop(~np.isfinite(d.value_nM) | (d.value_nM <= 0), "bad_value")
        _drop(d.relation.isin(["<", "<=", "<<"]), "left_censored")
        _drop(~d.relation.isin(allowed_rel), "censored_excluded")
        if require_panel and "in_panel" in ch.columns:
            _drop(~ch["in_panel"].astype(str).str.lower().isin(["true", "1", "1.0", "yes"]),
                  "not_in_panel")
        if "structure_status" in ch.columns:
            _drop(ch["structure_status"].astype("string").fillna("").eq("invalid_smiles"),
                  "invalid_structure")
        if "flag_contamination" in ch.columns:
            _drop(ch["flag_contamination"].astype("string").fillna("").ne(""), "contaminated")
        if drop_composite:
            _drop(d.uniprot_id.str.contains(";", regex=False), "composite_target")

        sem = (ch["measure_semantics"].astype("string").fillna("")
               if "measure_semantics" in ch.columns else pd.Series("", index=ch.index, dtype="string"))
        if exclude_semantics:
            low = sem.str.lower()
            bad = pd.Series(False, index=ch.index)
            for tag in exclude_semantics:
                bad = bad | low.str.contains(str(tag).lower(), regex=False, na=False)
            _drop(bad, "excluded_semantics")

        d = d[keep]
        if not len(d):
            continue
        s = sem[keep]
        if distinguish_semantics:
            d = d.assign(standard_type=d.standard_type.str.cat(s.where(s.ne(""), "direct"), sep="@"))
        for col, key in (("standard_type", "kept_by_type"), ("source", "kept_by_source")):
            for k, v in d[col].value_counts().items():
                report[key][str(k)] = report[key].get(str(k), 0) + int(v)
        for k, v in s.where(s.ne(""), "unspecified").value_counts().items():
            report["kept_by_semantics"][str(k)] = report["kept_by_semantics"].get(str(k), 0) + int(v)
        kept_parts.append(d.astype({c: "object" for c in ("smiles", "uniprot_id",
                                                          "standard_type", "relation", "source")}))

    out = (pd.concat(kept_parts, ignore_index=True) if kept_parts
           else pd.DataFrame(columns=_STAGE1_COLS))
    report["n_kept"] = int(len(out))
    report["n_dropped"] = report["n_input"] - report["n_kept"]
    report["n_censored"] = int(out.relation.isin([">", ">="]).sum()) if len(out) else 0
    report["allow_censored"] = bool(allow_censored)
    report["distinguish_semantics"] = bool(distinguish_semantics)
    return out[_STAGE1_COLS], report


def format_harvest_report(report: dict) -> str:
    """Human-readable summary of `from_harvest_csv`, for the Stage 1 log."""
    L = [f"  harvest rows read     : {report['n_input']:,}"]
    for reason, n in report["dropped"].items():
        if n:
            L.append(f"  dropped {reason:<22}: {n:,}")
    L.append(f"  kept                  : {report['n_kept']:,} "
             f"({report.get('n_censored', 0):,} right-censored)")
    for key, label in (("kept_by_type", "by assay type"),
                       ("kept_by_source", "by source"),
                       ("kept_by_semantics", "by semantics")):
        if report.get(key):
            top = sorted(report[key].items(), key=lambda kv: -kv[1])[:8]
            L.append(f"  {label:<21}: " + ", ".join(f"{k}={v:,}" for k, v in top))
    return "\n".join(L)
