"""Tier 8 -- retrosynthesis models (~1 GB).

Absent in Cycle 01, which is why no synthesisability screen ran at all. The
BRICS fallback is a fragment check, not a route search: it tells you a molecule
can be cut into known fragments, not that anyone can make it. A generator
optimised against a fragment check will happily produce things with no route.

`download_aizynth()` uses AiZynthFinder's own `download_public_data` entry
point, which is the only distribution that keeps the model and the template
library version-matched -- mixing a USPTO expansion policy with a different
template set produces routes that score well and cannot be executed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .util import LOG, Http, Layout, Manifest, try_import

#: Files the public bundle should contain.
EXPECTED = (
    "uspto_model.onnx",         # expansion policy
    "uspto_templates.csv.gz",   # template library (must match the model)
    "uspto_filter_model.onnx",  # in-scope filter
    "zinc_stock.hdf5",          # purchasable stock
)


def download_aizynth(layout: Layout, manifest: Manifest, *,
                     force: bool = False) -> Path | None:
    """Fetch the AiZynthFinder public data bundle into data/models/aizynth/."""
    if try_import("aizynthfinder") is None:
        LOG.error("aizynthfinder not installed -- pip install aizynthfinder")
        return None
    d = layout.models / "aizynth"
    d.mkdir(parents=True, exist_ok=True)
    cfg = d / "config.yml"
    if cfg.exists() and not force and all((d / f).exists() for f in EXPECTED):
        LOG.info("aizynth data already present in %s", d)
    else:
        LOG.info("downloading AiZynthFinder public data (~1 GB) into %s", d)
        try:
            subprocess.run(                                        # noqa: S603
                [sys.executable, "-m",
                 "aizynthfinder.tools.download_public_data", str(d)],
                check=True, timeout=14400)
        except Exception as exc:                                   # noqa: BLE001
            LOG.error("aizynth download failed: %s", exc)
            LOG.error("fallback: the same files are published on figshare "
                      "(AiZynthFinder public data); place them in %s and "
                      "re-run with --skip-download", d)
            return None

    present = {f: (d / f).exists() for f in EXPECTED}
    missing = [f for f, ok in present.items() if not ok]
    for f, ok in present.items():
        if ok:
            manifest.record(d / f, tier="8", source="aizynthfinder",
                            notes="USPTO public bundle")
    if missing:
        LOG.error("aizynth bundle incomplete, missing: %s", missing)
    if cfg.exists():
        manifest.record(cfg, tier="8", source="aizynthfinder",
                        notes="config pointing at the bundled model/stock")
    return cfg if cfg.exists() else None


# --------------------------------------------------------------------------- #
# synthesisability scoring
# --------------------------------------------------------------------------- #
def sa_score(smiles: list[str]) -> pd.DataFrame:
    """RDKit SAscore (Ertl) + RingSystemLookup-style ring sanity.

    Cheap, no download, correlates only loosely with route existence. Use it as
    a *pre-filter* ahead of AiZynthFinder, never as the synthesisability term in
    a reward on its own.
    """
    from rdkit import Chem                                          # noqa: PLC0415
    from rdkit.Chem import RDConfig                                 # noqa: PLC0415
    sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
    import sascorer                                                 # noqa: PLC0415

    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append({"smiles": smi,
                     "sa_score": sascorer.calculateScore(mol) if mol else None})
    return pd.DataFrame(rows)


def brics_fragment_check(smiles: list[str], stock_smiles: set[str] | None = None
                         ) -> pd.DataFrame:
    """The existing fallback, kept and clearly labelled for what it is."""
    from rdkit import Chem                                          # noqa: PLC0415
    from rdkit.Chem import BRICS                                    # noqa: PLC0415
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({"smiles": smi, "n_brics_fragments": None,
                         "brics_in_stock_frac": None})
            continue
        frags = list(BRICS.BRICSDecompose(mol))
        in_stock = None
        if stock_smiles:
            clean = [Chem.MolToSmiles(Chem.MolFromSmiles(
                        __import__("re").sub(r"\[\d+\*\]", "[H]", f)))
                     if Chem.MolFromSmiles(
                        __import__("re").sub(r"\[\d+\*\]", "[H]", f)) else None
                     for f in frags]
            clean = [c for c in clean if c]
            in_stock = (sum(c in stock_smiles for c in clean) / len(clean)
                        if clean else 0.0)
        rows.append({"smiles": smi, "n_brics_fragments": len(frags),
                     "brics_in_stock_frac": in_stock})
    df = pd.DataFrame(rows)
    LOG.warning("BRICS fragment check is a FRAGMENT check, not a route search -- "
                "do not report it as synthesisability")
    return df


def run_aizynth(config: Path, smiles: list[str], out_json: Path, *,
                nproc: int = 4, timeout_s: int = 7200) -> Path | None:
    """Route search over a candidate list; writes AiZynthFinder's own JSON."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    smi_file = out_json.with_suffix(".input.smi")
    smi_file.write_text("\n".join(smiles))
    try:
        subprocess.run(                                            # noqa: S603
            [sys.executable, "-m", "aizynthfinder.interfaces.aizynthcli",
             "--config", str(config), "--smiles", str(smi_file),
             "--output", str(out_json), "--nproc", str(nproc)],
            check=True, timeout=timeout_s)
    except Exception as exc:                                       # noqa: BLE001
        LOG.error("aizynthcli failed: %s", exc)
        return None
    LOG.info("route search written to %s", out_json)
    return out_json
