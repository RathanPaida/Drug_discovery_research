"""Tier 6 -- ADMET training sets (seven TDC endpoints).

The reward now fails closed on any missing endpoint, so this tier is not
optional: a missing file stops the whole RL loop rather than quietly scoring
compounds on six of seven axes. `build_admet()` therefore asserts that all
seven datasets are present and writes a `coverage.json` the reward can check at
startup.

hERG gets extra attention because it is what is driving the TPP failures (29 of
40 contract compounds above 0.20). Two hERG sets are pulled, not one:
  * `hERG`          -- Wang et al., binary blocker/non-blocker, ~650 compounds
  * `hERG_Karim`    -- ~13 k compounds, binary, much better covered chemically
A classifier trained on the 650-compound set alone will be confidently wrong on
kinase-inhibitor chemotypes, which is the most likely reason the predicted hERG
risk is failing the TPP so broadly. The larger set is the one to model on; the
small one is a held-out sanity check.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .util import LOG, Layout, Manifest, try_import, write_parquet

#: (tdc_group, tdc_name, our_endpoint_name, task)
ENDPOINTS: tuple[tuple[str, str, str, str], ...] = (
    ("adme", "Solubility_AqSolDB",     "solubility",        "regression"),
    ("adme", "Caco2_Wang",             "caco2",             "regression"),
    ("adme", "Clearance_Microsome_AZ", "microsomal_cl",     "regression"),
    ("adme", "CYP3A4_Veith",           "cyp3a4",            "classification"),
    ("adme", "CYP2D6_Veith",           "cyp2d6",            "classification"),
    ("tox",  "hERG_Karim",             "herg",              "classification"),
    ("tox",  "AMES",                   "ames",              "classification"),
)

#: Pulled alongside hERG_Karim as an independent check set, not for training.
EXTRA = (("tox", "hERG", "herg_wang_small", "classification"),)

REQUIRED = tuple(e[2] for e in ENDPOINTS)


def _load_tdc(group: str, name: str, path: Path) -> pd.DataFrame | None:
    tdc = try_import("tdc")
    if tdc is None:
        raise SystemExit("PyTDC is required for tier 6: pip install PyTDC")
    try:
        if group == "adme":
            from tdc.single_pred import ADME                        # noqa: PLC0415
            data = ADME(name=name, path=str(path))
        else:
            from tdc.single_pred import Tox                         # noqa: PLC0415
            data = Tox(name=name, path=str(path))
        df = data.get_data()
    except Exception as exc:                                       # noqa: BLE001
        LOG.error("TDC %s/%s failed: %s", group, name, exc)
        return None
    return df.rename(columns={"Drug": "smiles", "Y": "y", "Drug_ID": "compound_id"})


def build_admet(layout: Layout, manifest: Manifest, *,
                standardize: bool = True,
                strict: bool = True) -> tuple[Path, dict]:
    from .curate import Standardizer                                # noqa: PLC0415

    d = layout.download_dir("tdc")
    std = Standardizer() if standardize else None

    frames: list[pd.DataFrame] = []
    coverage: dict[str, dict] = {}
    missing: list[str] = []

    for group, name, endpoint, task in (*ENDPOINTS, *EXTRA):
        df = _load_tdc(group, name, d)
        if df is None or df.empty:
            missing.append(endpoint)
            coverage[endpoint] = {"status": "missing", "tdc_name": name}
            continue
        df = df[["smiles", "y"] + [c for c in ("compound_id",) if c in df.columns]]
        if std is not None:
            df = std.apply(df, smiles_col="smiles")
        df["endpoint"] = endpoint
        df["task"] = task
        df["tdc_name"] = name
        df["train_eligible"] = endpoint in REQUIRED
        # Collapse replicate measurements of the same structure.
        n_before = len(df)
        agg = "mean" if task == "regression" else "max"
        df = (df.groupby(["inchikey" if "inchikey" in df else "smiles",
                          "endpoint", "task", "tdc_name", "train_eligible"],
                         dropna=False)
                .agg(y=("y", agg),
                     canonical_smiles=("canonical_smiles", "first")
                     if "canonical_smiles" in df else ("smiles", "first"),
                     n_measurements=("y", "size"))
                .reset_index())
        coverage[endpoint] = {
            "status": "ok", "tdc_name": name, "task": task,
            "rows": len(df), "rows_before_dedup": n_before,
            "positive_rate": (float(df["y"].mean()) if task == "classification"
                              else None),
            "y_min": float(df["y"].min()), "y_max": float(df["y"].max()),
        }
        LOG.info("ADMET %-16s %-24s %6d rows", endpoint, name, len(df))
        frames.append(df)

    if missing and strict:
        raise RuntimeError(
            f"tier 6 incomplete -- missing endpoints: {missing}. The reward "
            f"fails closed on a missing endpoint, so a partial build would stop "
            f"the RL loop at run time instead of here. Re-run with "
            f"--no-strict-admet to build anyway.")

    all_df = pd.concat(frames, ignore_index=True)
    out = write_parquet(all_df, layout.processed / "admet_7endpoints.parquet",
                        manifest, tier="6", source="tdc",
                        notes="long format: one row per (structure, endpoint); "
                              "`train_eligible` excludes the hERG check set")

    cov_path = layout.processed / "admet_coverage.json"
    cov = {"required": list(REQUIRED), "missing": missing, "endpoints": coverage}
    cov_path.write_text(json.dumps(cov, indent=2))
    manifest.record(cov_path, tier="6", source="derived",
                    notes="read this at reward startup to fail closed early")
    return out, cov


def herg_risk_note(layout: Layout) -> str:
    """Diagnostic text for the hERG/TPP problem, written into the README."""
    return (
        "hERG: 29/40 contract compounds predicted >0.20 is a *model* signal as "
        "much as a chemistry signal. Check in this order: (1) is the classifier "
        "trained on hERG_Karim (~13k) or on the ~650-compound Wang set? "
        "(2) what is the base rate of the training set vs your 0.20 threshold -- "
        "a 50% positive-rate training set produces a poorly calibrated score and "
        "0.20 is then not a 20% risk; (3) are the 29 within the applicability "
        "domain, by max Tanimoto to the training set? Recalibrate on a kinase-"
        "inhibitor subset before redesigning chemistry against the score."
    )
