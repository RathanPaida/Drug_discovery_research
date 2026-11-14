from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
import json
from scripts.dock_40_molecules_tnk1 import run_docking_single

receptor = (ROOT / 'data' / 'structures' / 'tnk1_kinase.pdbqt') if (ROOT / 'data' / 'structures' / 'tnk1_kinase.pdbqt').exists() else (ROOT / 'data' / 'tnk1_kinase.pdbqt')
vina = ROOT / 'tools' / 'vina.exe'
runs_base = (ROOT / 'runs' / 'cycle_01') if (ROOT / 'runs' / 'cycle_01').exists() else (ROOT / 'tnk1-pipeline' / 'tnk1-pipeline' / 'runs' / 'cycle_01')
poses = runs_base / 'docking_poses'
poses.mkdir(parents=True, exist_ok=True)

controls = [
    ('CTRL_DASATINIB', 'CONTROL', 'Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1'),
    ('CTRL_BOSUTINIB', 'CONTROL', 'COc1cc(Nc2c(C#N)cnc3cc(OCCCN4CCN(C)CC4)c(OC)cc23)c(Cl)cc1Cl'),
    ('CTRL_TP0903', 'CONTROL', 'CC1=CN(C2CCNCC2)C(=O)C2=C1NC1=NC(NC3=CC=C(OCCN4CCCC4)C=C3)=NC=C21')
]

results = []
for cid, bslice, smi in controls:
    print(f"Docking {cid}...", flush=True)
    res = run_docking_single(cid, bslice, smi, receptor, vina, poses, cpu_per_job=4)
    print(f"{cid}: Affinity={res['docking_affinity_kcal_mol']} kcal/mol, Hinge_dist={res['hinge_min_dist_A']} A, Contact={res['hinge_contact']}", flush=True)
    results.append(res)

with open(runs_base / 'docking_controls.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved docking controls successfully.")

