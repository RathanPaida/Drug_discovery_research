#!/usr/bin/env python3
"""TNK1 data acquisition CLI.

    python acquire.py --list                      # what each tier costs
    python acquire.py tier4 tier1 tier2           # the cheap, high-value core
    python acquire.py tier9                       # generative anchoring
    python acquire.py --all                       # everything, in order
    python acquire.py features                    # rebuild data/features/
    python acquire.py verify                      # SHA-256 acceptance test

Every tier is idempotent: already-downloaded files are skipped unless --force.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tnk1_acquire import pipeline
from tnk1_acquire.util import LOG, Http, Layout, Manifest, setup_logging

TIERS: dict[str, tuple[str, str]] = {
    "tier1": ("TNK1 bioactivity (ChEMBL+BindingDB+PubChem+DTC)", "~10 min, <50 MB"),
    "tier2": ("Off-target panel, 8 kinases", "~30 min, <200 MB"),
    "tier3": ("Kinome-wide multi-task (ChEMBL SQLite + BindingDB + Papyrus)",
              "hours, ~8 GB download / ~30 GB unpacked"),
    "tier4": ("Kinase descriptors (KLIFS pockets + UniProt)", "~2 min, <10 MB"),
    "tier5": ("Unlabelled corpus + fingerprints", "~1-3 h, ~1 GB"),
    "tier6": ("ADMET, 7 TDC endpoints", "~10 min, <200 MB"),
    "tier7": ("Structures: AlphaFold TNK1 + ACK1/TNK2 ensemble", "~5 min, <100 MB"),
    "tier8": ("Retrosynthesis models (AiZynthFinder)", "~20 min, ~1 GB"),
    "tier9": ("Generative anchoring around known TNK1/TNK2 chemotypes",
              "~20 min, <100 MB"),
    "features": ("Rebuild data/features/ from data/processed/", "~10-60 min"),
    "verify": ("SHA-256 acceptance test against MANIFEST.json", "minutes"),
}

ORDER = ("tier4", "tier1", "tier2", "tier3", "tier5", "tier6", "tier7",
         "tier8", "tier9", "features", "verify")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="acquire.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stages", nargs="*", choices=list(TIERS) + [], metavar="STAGE",
                   help="one or more of: " + ", ".join(TIERS))
    p.add_argument("--all", action="store_true",
                   help="run every stage in dependency order")
    p.add_argument("--list", action="store_true", help="describe the stages and exit")
    p.add_argument("--root", type=Path, default=Path(__file__).parent,
                   help="data root (default: this directory)")
    p.add_argument("--force", action="store_true",
                   help="re-download even if the file is already present")
    p.add_argument("--quiet", action="store_true")

    g = p.add_argument_group("source toggles")
    g.add_argument("--no-bindingdb", action="store_true")
    g.add_argument("--no-pubchem", action="store_true")
    g.add_argument("--no-dtc", action="store_true")
    g.add_argument("--no-papyrus", action="store_true")
    g.add_argument("--no-sqlite", action="store_true",
                   help="tier3: skip the ~6 GB ChEMBL SQLite dump "
                        "(leaves only BindingDB/DTC/Papyrus breadth)")
    g.add_argument("--bindingdb-url", default=None,
                   help="pin a BindingDB release, or file:///path/to/local.zip")
    g.add_argument("--dtc-csv", type=Path, default=None,
                   help="hand-downloaded DTC bulk CSV")

    o = p.add_argument_group("options")
    o.add_argument("--corpus-limit", type=int, default=None,
                   help="tier5: cap input records (for a smoke test)")
    o.add_argument("--with-moses", action="store_true")
    o.add_argument("--no-strict-admet", action="store_true",
                   help="tier6: build even with endpoints missing (the reward "
                        "will then fail closed at run time instead)")
    o.add_argument("--no-receptor-prep", action="store_true",
                   help="tier7: download structures but skip PDBQT conversion")
    o.add_argument("--timeout", type=int, default=180)
    o.add_argument("--rate", type=float, default=0.2,
                   help="minimum seconds between HTTP requests (be polite to "
                        "EBI/NCBI; 0.2 is ~5 req/s)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list or (not args.stages and not args.all):
        print(f"\n{'stage':10}  {'what it gets':62}  cost")
        print("-" * 100)
        for name in ORDER:
            desc, cost = TIERS[name]
            print(f"{name:10}  {desc:62}  {cost}")
        print("\nrun order matters: tier4 defines the kinome that tier3 filters "
              "on, and tier9 needs tier1/tier2.\n")
        return 0

    layout = Layout(args.root.resolve())
    layout.ensure()
    setup_logging(layout, verbose=not args.quiet)
    manifest = Manifest(layout)
    http = Http(timeout=args.timeout, min_interval=args.rate)

    stages = list(ORDER) if args.all else list(args.stages)
    LOG.info("root=%s stages=%s", layout.root, stages)

    common = dict(force=args.force)
    kw_by_stage = {
        "tier1": dict(include_bindingdb=not args.no_bindingdb,
                      include_pubchem=not args.no_pubchem,
                      include_dtc=not args.no_dtc),
        "tier2": dict(include_bindingdb=not args.no_bindingdb,
                      include_pubchem=not args.no_pubchem,
                      include_dtc=not args.no_dtc),
        "tier3": dict(use_sqlite=not args.no_sqlite,
                      bindingdb_url=args.bindingdb_url,
                      dtc_csv=args.dtc_csv,
                      include_papyrus=not args.no_papyrus),
        "tier5": dict(limit=args.corpus_limit, with_moses=args.with_moses),
        "tier6": dict(strict=not args.no_strict_admet),
        "tier7": dict(prepare_receptors=not args.no_receptor_prep),
    }
    needs_http = {"tier1", "tier2", "tier3", "tier4", "tier5", "tier7", "tier9"}

    rc = 0
    for stage in stages:
        fn = {
            "tier1": pipeline.tier1, "tier2": pipeline.tier2,
            "tier3": pipeline.tier3, "tier4": pipeline.tier4,
            "tier5": pipeline.tier5, "tier6": pipeline.tier6,
            "tier7": pipeline.tier7, "tier8": pipeline.tier8,
            "tier9": pipeline.tier9,
            "features": pipeline.build_features, "verify": pipeline.verify,
        }[stage]
        kw = {**common, **kw_by_stage.get(stage, {})}
        t0 = time.monotonic()
        try:
            if stage in needs_http:
                result = fn(http, layout, manifest, **kw)
            else:
                result = fn(layout, manifest, **kw)
            if stage == "verify" and result:
                rc = int(result)
            LOG.info("[%s] completed in %.1f min", stage,
                     (time.monotonic() - t0) / 60)
        except KeyboardInterrupt:
            LOG.error("[%s] interrupted", stage)
            return 130
        except Exception as exc:                                   # noqa: BLE001
            LOG.exception("[%s] FAILED after %.1f min: %s", stage,
                          (time.monotonic() - t0) / 60, exc)
            rc = 1
            if stage in ("tier4",):
                LOG.error("tier4 is a hard dependency of tier3 -- stopping")
                break

    pipeline.write_report(layout)
    manifest.flush()
    ok, bad, missing = manifest.verify()
    LOG.info("manifest: %d artefacts recorded (%d verified, %d mismatched, "
             "%d missing)", len(ok) + len(bad) + len(missing), len(ok),
             len(bad), len(missing))
    LOG.info("manifest written to %s", manifest.path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
