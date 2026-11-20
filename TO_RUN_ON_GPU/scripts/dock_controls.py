"""Dock the three clinical control inhibitors into the same pocket as the candidates.

The controls are what make the candidates' Vina scores interpretable: an absolute
`dG = -7.3 kcal/mol` means nothing without knowing that TP-0903 scores -9.40 and dasatinib
-8.37 on the identical receptor and box.

    python scripts/dock_controls.py
    python scripts/dock_controls.py --cpu 8 --cycle 1

Two things changed here. The body used to run at import, so `from scripts.dock_controls
import ...` -- or any tool that walks the package -- would start a docking campaign; and
it fell back to a `tnk1-pipeline/tnk1-pipeline/runs/...` path left over from the
pre-reorganisation layout, a directory that no longer exists (it had become a cache-only
stub). Missing inputs are now reported up front instead of failing inside Vina.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dock_40_molecules_tnk1 import resolve_vina, run_docking_single  # noqa: E402

# name, batch_slice, SMILES. All three are approved or clinical-stage kinase inhibitors
# with published TNK1/ACK-family activity, used as the benchmark for the campaign.
CONTROLS = [
    ("CTRL_DASATINIB", "CONTROL",
     "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1"),
    ("CTRL_BOSUTINIB", "CONTROL",
     "COc1cc(Nc2c(C#N)cnc3cc(OCCCN4CCN(C)CC4)c(OC)cc23)c(Cl)cc1Cl"),
    ("CTRL_TP0903", "CONTROL",
     "CC1=CN(C2CCNCC2)C(=O)C2=C1NC1=NC(NC3=CC=C(OCCN4CCCC4)C=C3)=NC=C21"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycle", type=int, default=1)
    ap.add_argument("--cpu", type=int, default=4, help="Vina CPU threads per job")
    ap.add_argument("--receptor", default=None,
                    help="default: data/structures/tnk1_kinase.pdbqt")
    ap.add_argument("--vina", default=None,
                    help="path to the Vina executable; else $TNK1_VINA, PATH, tools/")
    ap.add_argument("--out", default=None, help="default: runs/cycle_<cycle>/")
    ap.add_argument("--seed", type=int, default=None,
                    help="Vina RNG seed; default is dock_40_molecules_tnk1.DOCK_SEED. "
                         "-1 seeds from the clock and makes the run irreproducible")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing docking_controls.json for this cycle")
    a = ap.parse_args(argv)

    receptor = Path(a.receptor) if a.receptor else ROOT / "data" / "structures" / "tnk1_kinase.pdbqt"
    runs_base = Path(a.out) if a.out else ROOT / "runs" / f"cycle_{a.cycle:02d}"

    # Fail closed, naming every missing input, instead of letting Vina fail opaquely.
    # The binary itself goes through the shared resolver (--vina / $TNK1_VINA / PATH /
    # tools/) so this script and dock_40_molecules_tnk1.py cannot disagree about where
    # Vina lives -- and so neither of them assumes a Windows .exe on a Linux HPC.
    if not receptor.exists():
        print("Cannot dock; missing input:", file=sys.stderr)
        print(f"  {receptor}", file=sys.stderr)
        print("\nThe receptor is produced by the Stage-8 docking prep.", file=sys.stderr)
        return 1
    vina = resolve_vina(a.vina)

    # Published Cycle-01 numbers (-9.40 / -8.37 / -8.26 kcal/mol) are quoted in README.md
    # and the master document. Re-running shifts them, because Vina's search is stochastic.
    out = runs_base / "docking_controls.json"
    if out.exists() and not a.force:
        print(f"Refusing to run: {out} already exists.")
        print("Its values are quoted in README.md and TNK1_PROJECT_MASTER_DOCUMENT.md as the")
        print("benchmark the candidate affinities are interpreted against. Vina results shift")
        print("between runs, so re-docking changes a published comparison. Pass --force.")
        return 1

    poses = runs_base / "docking_poses"
    poses.mkdir(parents=True, exist_ok=True)

    from scripts.dock_40_molecules_tnk1 import DOCK_SEED
    seed = DOCK_SEED if a.seed is None else (None if int(a.seed) < 0 else int(a.seed))

    results = []
    for cid, bslice, smi in CONTROLS:
        print(f"Docking {cid}... (seed={seed if seed is not None else 'clock'})", flush=True)
        res = run_docking_single(cid, bslice, smi, receptor, vina, poses,
                                 cpu_per_job=a.cpu, seed=seed)
        print(f"{cid}: Affinity={res['docking_affinity_kcal_mol']} kcal/mol, "
              f"Hinge_dist={res['hinge_min_dist_A']} A, Contact={res['hinge_contact']}",
              flush=True)
        results.append(res)

    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} control results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
