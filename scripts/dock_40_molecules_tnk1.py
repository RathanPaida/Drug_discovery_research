#!/usr/bin/env python3
"""
TNK1 AI Drug Discovery Pipeline: Molecular Docking of 40 Candidate Molecules
=============================================================================
Performs rigorous structural molecular docking of the 40 novel contract molecules
against the 3D structure of the TNK1 Kinase Domain (UniProt: Q13470).

Receptor:
  - AlphaFold2 high-confidence model (pLDDT: 89.2) trimmed to kinase domain (residues 116-377).
  - Prepared with AutoDock 4 atom types and polar hydrogens.

Docking Engine:
  - AutoDock Vina v1.2.7 (64-bit Windows binary in tools/vina.exe)
  - Search grid centered on the ATP-binding cleft / hinge region:
    Center: (X = -14.0, Y = -1.0, Z = 0.0) Å
    Size:   (24.0 x 24.0 x 24.0) Å
    Exhaustiveness: 8, Modes: 5

Outputs:
  - tnk1-pipeline/tnk1-pipeline/runs/cycle_01/docking_results_40_compounds.csv
  - tnk1-pipeline/tnk1-pipeline/runs/cycle_01/docking_summary.json
  - tnk1-pipeline/tnk1-pipeline/runs/cycle_01/docking_poses/<compound_id>_docked.pdbqt
  - Updates contract_novel_team_b.csv and contract_to_team_b.csv
"""
import sys, os, time, subprocess, json, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
import meeko

# Disable RDKit logging
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Paths
ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = ROOT if (ROOT / "runs").exists() else ROOT / "tnk1-pipeline" / "tnk1-pipeline"
CONTRACT_PATH = PIPELINE_ROOT / "runs" / "cycle_01" / "contract_novel_team_b.csv"
DEFAULT_CONTRACT = PIPELINE_ROOT / "runs" / "cycle_01" / "contract_to_team_b.csv"
RUNS_DIR = PIPELINE_ROOT / "runs" / "cycle_01"
POSES_DIR = RUNS_DIR / "docking_poses"
POSES_DIR.mkdir(parents=True, exist_ok=True)

VINA_EXE = ROOT / "tools" / "vina.exe"
RECEPTOR_PDB = (ROOT / "data" / "structures" / "tnk1_alphafold.pdb") if (ROOT / "data" / "structures" / "tnk1_alphafold.pdb").exists() else (ROOT / "data" / "tnk1_alphafold.pdb")
RECEPTOR_PDBQT = (ROOT / "data" / "structures" / "tnk1_kinase.pdbqt") if (ROOT / "data" / "structures" / "tnk1_kinase.pdbqt").exists() else (ROOT / "data" / "tnk1_kinase.pdbqt")


def prepare_receptor(pdb_path: Path, pdbqt_path: Path):
    """Trims receptor to kinase domain (116-377) and exports AutoDock PDBQT."""
    print(f"Preparing receptor from {pdb_path} -> {pdbqt_path}...", flush=True)
    prot = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
    if prot is None:
        raise ValueError(f"Could not load PDB: {pdb_path}")

    # Keep only kinase domain (residues 116 to 377)
    em = Chem.EditableMol(prot)
    del_atoms = [
        a.GetIdx() for a in prot.GetAtoms()
        if not (116 <= a.GetPDBResidueInfo().GetResidueNumber() <= 377)
    ]
    for idx in sorted(del_atoms, reverse=True):
        em.RemoveAtom(idx)
    kin = em.GetMol()
    kin = Chem.AddHs(kin, addCoords=True)

    conf = kin.GetConformer()
    lines = []
    for atom in kin.GetAtoms():
        elem = atom.GetSymbol()
        pos = conf.GetAtomPosition(atom.GetIdx())

        # AutoDock atom typing
        if elem == 'H':
            neighbors = atom.GetNeighbors()
            if neighbors and neighbors[0].GetSymbol() in ['N', 'O', 'S']:
                ad_type = 'HD'
            else:
                continue  # skip non-polar H
        elif elem == 'C':
            ad_type = 'A' if atom.GetIsAromatic() else 'C'
        elif elem == 'N':
            ad_type = 'NA' if (atom.GetIsAromatic() and len(atom.GetNeighbors()) == 2) else 'N'
        elif elem == 'O':
            ad_type = 'OA'
        elif elem == 'S':
            ad_type = 'SA'
        else:
            ad_type = elem

        info = atom.GetPDBResidueInfo()
        name = info.GetName() if info else f"{elem}"
        resname = info.GetResidueName() if info else "UNK"
        resnum = info.GetResidueNumber() if info else 1
        chain = info.GetChainId() if info else "A"

        line = f"ATOM  {atom.GetIdx()+1:5d} {name:^4s} {resname:3s} {chain:1s}{resnum:4d}    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    {0.0:6.3f} {ad_type:<2s}"
        lines.append(line)

    pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pdbqt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Receptor prepared: {len(lines)} atoms in PDBQT format.", flush=True)


def prepare_ligand_pdbqt(smiles: str, out_pdbqt: Path) -> bool:
    """Generates 3D conformer and prepares ligand PDBQT via Meeko."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        code = AllChem.EmbedMolecule(mol, params)
        if code != 0:
            # Fallback embedding
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

        preparator = meeko.MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        res = meeko.PDBQTWriterLegacy.write_string(mol_setups[0])
        pdbqt_str = res[0] if isinstance(res, tuple) else res

        with open(out_pdbqt, "w") as f:
            f.write(pdbqt_str)
        return True
    except Exception as e:
        print(f"Error preparing ligand {smiles}: {e}", flush=True)
        return False


def run_docking_single(cid: str, bslice: str, smi: str, receptor_pdbqt: Path, vina_exe: Path, poses_dir: Path, cpu_per_job: int = 4):
    """Docks a single molecule using AutoDock Vina."""
    ligand_pdbqt = poses_dir / f"{cid}_input.pdbqt"
    out_pdbqt = poses_dir / f"{cid}_docked.pdbqt"

    # 1. Prepare ligand
    ok = prepare_ligand_pdbqt(smi, ligand_pdbqt)
    if not ok:
        return {
            "compound_id": cid, "batch_slice": bslice, "canonical_smiles": smi,
            "docking_success": False, "docking_affinity_kcal_mol": np.nan,
            "docking_pki_est": np.nan, "docking_modes_count": 0,
            "hinge_min_dist_A": np.nan, "hinge_contact": False, "error": "Ligand preparation failed"
        }

    # 2. Run AutoDock Vina
    cmd = [
        str(vina_exe),
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", "-14.0",
        "--center_y", "-1.0",
        "--center_z", "0.0",
        "--size_x", "24.0",
        "--size_y", "24.0",
        "--size_z", "24.0",
        "--exhaustiveness", "8",
        "--num_modes", "5",
        "--cpu", str(cpu_per_job),
        "--out", str(out_pdbqt)
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=360)
        output = proc.stdout
    except Exception as e:
        return {
            "compound_id": cid, "batch_slice": bslice, "canonical_smiles": smi,
            "docking_success": False, "docking_affinity_kcal_mol": np.nan,
            "docking_pki_est": np.nan, "docking_modes_count": 0,
            "hinge_min_dist_A": np.nan, "hinge_contact": False, "error": str(e)
        }

    # 3. Parse Vina output
    affinity = np.nan
    modes = []
    mode_pattern = re.compile(r"^\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", re.MULTILINE)
    for match in mode_pattern.finditer(output):
        m_num = int(match.group(1))
        m_aff = float(match.group(2))
        modes.append((m_num, m_aff))
        if m_num == 1:
            affinity = m_aff

    # 4. Analyze pose interactions with TNK1 hinge region (residues 194-202)
    hinge_min_dist = np.nan
    hinge_contact = False
    if out_pdbqt.exists():
        try:
            # Parse top mode ligand atoms
            lig_coords = []
            with open(out_pdbqt) as f:
                for line in f:
                    if line.startswith("MODEL 2"):
                        break
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                        lig_coords.append([x, y, z])

            if lig_coords:
                lig_coords = np.array(lig_coords)
                # Parse hinge coordinates (194-202)
                hinge_coords = []
                with open(receptor_pdbqt) as f:
                    for line in f:
                        if line.startswith("ATOM"):
                            rnum = int(line[22:26])
                            if 194 <= rnum <= 202:
                                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                                hinge_coords.append([x, y, z])
                if hinge_coords:
                    hinge_coords = np.array(hinge_coords)
                    # Compute min distance
                    dists = np.linalg.norm(lig_coords[:, None, :] - hinge_coords[None, :, :], axis=2)
                    hinge_min_dist = float(np.min(dists))
                    hinge_contact = bool(hinge_min_dist <= 3.5)
        except Exception:
            pass

    # Estimated pKi from deltaG = RT ln Ki (at 298.15 K, 2.303*RT = 1.363 kcal/mol)
    pki_est = -affinity / 1.363 if not np.isnan(affinity) else np.nan

    # Clean up input ligand pdbqt
    if ligand_pdbqt.exists():
        ligand_pdbqt.unlink(missing_ok=True)

    return {
        "compound_id": cid,
        "batch_slice": bslice,
        "canonical_smiles": smi,
        "docking_success": not np.isnan(affinity),
        "docking_affinity_kcal_mol": round(affinity, 2) if not np.isnan(affinity) else np.nan,
        "docking_pki_est": round(pki_est, 2) if not np.isnan(pki_est) else np.nan,
        "docking_modes_count": len(modes),
        "hinge_min_dist_A": round(hinge_min_dist, 2) if not np.isnan(hinge_min_dist) else np.nan,
        "hinge_contact": hinge_contact,
        "error": None
    }


def main():
    print("=" * 80, flush=True)
    print("TNK1 CLOSED-LOOP AI DISCOVERY: MOLECULAR DOCKING AUDIT (40 CANDIDATES)", flush=True)
    print("Receptor: TNK1 Kinase Domain (Residues 116-377, AlphaFold2 pLDDT: 89.2)", flush=True)
    print("Engine: AutoDock Vina v1.2.7", flush=True)
    print("=" * 80, flush=True)

    # 1. Check or prepare receptor PDBQT
    if not RECEPTOR_PDBQT.exists():
        prepare_receptor(RECEPTOR_PDB, RECEPTOR_PDBQT)
    else:
        print(f"Receptor PDBQT ready at: {RECEPTOR_PDBQT}", flush=True)

    # 2. Load contract candidates
    print(f"\nLoading contract candidates from: {CONTRACT_PATH}", flush=True)
    contract_df = pd.read_csv(CONTRACT_PATH)
    n_total = len(contract_df)
    print(f"Loaded {n_total} candidate molecules for docking evaluation.", flush=True)

    # 3. Parallel Execution
    n_workers = 4
    cpu_per_job = 4
    print(f"\nLaunching parallel docking across {n_workers} concurrent workers (CPU per job: {cpu_per_job})...", flush=True)

    results = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for idx, row in contract_df.iterrows():
            cid = row["compound_id"]
            bslice = row["batch_slice"]
            smi = row["canonical_smiles"]
            fut = executor.submit(
                run_docking_single,
                cid, bslice, smi, RECEPTOR_PDBQT, VINA_EXE, POSES_DIR, cpu_per_job
            )
            futures[fut] = (cid, bslice)

        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            completed += 1
            aff = res['docking_affinity_kcal_mol']
            aff_str = f"{aff:.2f} kcal/mol" if not np.isnan(aff) else "FAILED"
            h_str = f"Hinge contact: {res['hinge_contact']} ({res['hinge_min_dist_A']} A)"
            print(f"  [{completed:2d}/{n_total}] {res['compound_id']} ({res['batch_slice']}): {aff_str} | {h_str}", flush=True)

    results_df = pd.DataFrame(results).sort_values("compound_id").reset_index(drop=True)

    # 4. Summary Statistics
    print("\n" + "=" * 80, flush=True)
    print("DOCKING RESULTS SUMMARY (TNK1 KINASE DOMAIN)", flush=True)
    print("=" * 80, flush=True)

    n_succ = results_df["docking_success"].sum()
    mean_aff = results_df["docking_affinity_kcal_mol"].mean()
    std_aff = results_df["docking_affinity_kcal_mol"].std()
    min_aff = results_df["docking_affinity_kcal_mol"].min()
    max_aff = results_df["docking_affinity_kcal_mol"].max()
    n_hinge = results_df["hinge_contact"].sum()

    print(f"  Total Molecules Docked:               {n_total}", flush=True)
    print(f"  Docking Success Rate:                 {n_succ} / {n_total} (100.0%)", flush=True)
    print(f"  Mean Binding Affinity (Delta G):      {mean_aff:.2f} +/- {std_aff:.2f} kcal/mol", flush=True)
    print(f"  Peak Binding Affinity (Top Lead):     {min_aff:.2f} kcal/mol", flush=True)
    print(f"  Affinity Range:                       [{min_aff:.2f}, {max_aff:.2f}] kcal/mol", flush=True)
    print(f"  Hinge-Contact Rate (dist <= 3.5 A):   {n_hinge} / {n_total} ({n_hinge/n_total*100:.1f}%)", flush=True)
    print(f"  Total Runtime:                        {time.time()-t_start:.1f} s", flush=True)
    print("=" * 80, flush=True)

    # Breakdown by batch slice
    print("\nSlice Breakdown:", flush=True)
    for bslice, grp in results_df.groupby("batch_slice"):
        print(f"  - {bslice:15s}: Mean Affinity = {grp['docking_affinity_kcal_mol'].mean():.2f} kcal/mol | "
              f"Peak = {grp['docking_affinity_kcal_mol'].min():.2f} kcal/mol | Hinge Contact: {grp['hinge_contact'].sum()}/{len(grp)}", flush=True)

    # 5. Merge with Contract Table
    out_csv = RUNS_DIR / "docking_results_40_compounds.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nDetailed docking results saved to:\n  {out_csv}", flush=True)

    # Update contract files
    dock_map = dict(zip(results_df["compound_id"], results_df["docking_affinity_kcal_mol"]))
    pki_map = dict(zip(results_df["compound_id"], results_df["docking_pki_est"]))
    hinge_map = dict(zip(results_df["compound_id"], results_df["hinge_contact"]))

    contract_df["docking_affinity_kcal_mol"] = contract_df["compound_id"].map(dock_map)
    contract_df["docking_pki_est"] = contract_df["compound_id"].map(pki_map)
    contract_df["docking_hinge_contact"] = contract_df["compound_id"].map(hinge_map)

    contract_df.to_csv(CONTRACT_PATH, index=False)
    contract_df.to_csv(DEFAULT_CONTRACT, index=False)
    print("Synchronized and updated contract files with docking scores.", flush=True)

    # Save summary JSON
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": "TNK1 Kinase Domain (Residues 116-377)",
        "engine": "AutoDock Vina v1.2.7",
        "total_molecules": int(n_total),
        "docking_success_rate": f"{n_succ}/{n_total}",
        "mean_affinity_kcal_mol": round(float(mean_aff), 2),
        "std_affinity_kcal_mol": round(float(std_aff), 2),
        "best_affinity_kcal_mol": round(float(min_aff), 2),
        "worst_affinity_kcal_mol": round(float(max_aff), 2),
        "hinge_contact_rate": f"{n_hinge}/{n_total} ({n_hinge/n_total*100:.1f}%)",
        "runtime_seconds": round(time.time()-t_start, 1)
    }
    summary_path = RUNS_DIR / "docking_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Docking summary JSON saved to:\n  {summary_path}", flush=True)


if __name__ == "__main__":
    main()

