"""Compare novel, route-screened candidates with references using matched docking.

Produces a computational shortlist, not a claim of superior biological activity.
Does not prepare a receptor automatically or convert docking scores into measured Ki.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import yaml


def heavy_coordinates(path, first_pose=False):
    coords = []
    for line in Path(path).read_text().splitlines():
        if first_pose and line.startswith('ENDMDL'): break
        if line.startswith(('ATOM', 'HETATM')):
            atom_type = line.split()[-1]
            if atom_type not in ('H', 'HD', 'HS'):
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords: raise ValueError(f'No heavy atom coordinates: {path}')
    return np.asarray(coords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--protocol', default='configs/final_screen.yaml')
    a = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    protocol_path = Path(a.protocol)
    if not protocol_path.is_absolute(): protocol_path = root / protocol_path
    cfg = yaml.safe_load(protocol_path.read_text())
    receptor = Path(cfg['receptor'])
    if not receptor.is_absolute(): receptor = root / receptor
    if not receptor.is_file(): raise FileNotFoundError(f'Prepared TNK1 receptor required: {receptor}')
    dest = Path(a.campaign) / 'final_screen'; dest.mkdir(exist_ok=True)
    from scripts.dock_40_molecules_tnk1 import run_docking_single, resolve_vina
    from scripts.dock_controls import CONTROLS
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    vina = resolve_vina()
    receptor_xyz = heavy_coordinates(receptor)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    refs = [Chem.MolFromSmiles(s) for _, _, s in CONTROLS]
    if any(m is None for m in refs): raise ValueError('Invalid reference control structure')
    reffps = [fpgen.GetFingerprint(m) for m in refs]
    tables = []
    for path in sorted(Path(a.campaign).glob('*/novel_route_screened_candidates.csv')):
        table = pd.read_csv(path); table['algorithm'] = path.parent.name; tables.append(table)
    if not tables:
        pd.DataFrame(columns=['canonical_smiles', 'reason']).to_csv(dest / 'ranked_candidates.csv', index=False)
        (dest / 'summary.json').write_text(json.dumps({'n_selected': 0, 'reason': 'No route-screened novel candidates'}))
        return 0
    pool = pd.concat(tables, ignore_index=True).sort_values('R', ascending=False)
    pool = pool.drop_duplicates('inchikey14').reset_index(drop=True)
    if pool.empty:
        pool.to_csv(dest / 'ranked_candidates.csv', index=False)
        (dest / 'summary.json').write_text(json.dumps({'n_selected': 0, 'reason': 'No candidates passed novelty and route screening'}))
        return 0
    def dock(cid, smi):
        rows = []
        for seed in cfg['seeds']:
            poses = dest / 'poses' / str(seed); poses.mkdir(parents=True, exist_ok=True)
            row = run_docking_single(cid, 'SCREEN', smi, receptor, vina, poses,
                                     cpu_per_job=cfg['cpu_per_job'], seed=int(seed))
            # Keep a score and a geometry heuristic; neither is an experimental Ki/H-bond.
            row.pop('docking_pki_est', None); row['seed'] = seed; rows.append(row)
            if row['docking_success']:
                ligand_xyz = heavy_coordinates(poses / f'{cid}_docked.pdbqt', first_pose=True)
                distance = np.linalg.norm(ligand_xyz[:, None, :] - receptor_xyz[None, :, :], axis=2)
                row['min_heavy_atom_separation'] = float(distance.min())
                row['severe_clash'] = bool(distance.min() < cfg['min_heavy_atom_separation'])
        return rows
    controls = []
    for cid, _, smi in CONTROLS: controls.extend(dock(cid, smi))
    pd.DataFrame(controls).to_csv(dest / 'reference_docking.csv', index=False)
    if not all(r['docking_success'] for r in controls):
        raise RuntimeError('Reference docking incomplete; comparison cannot be interpreted')
    reference = pd.DataFrame(controls).groupby('compound_id').docking_affinity_kcal_mol.median().min()
    results, trials = [], []
    for i, row in pool.iterrows():
        record = row.to_dict(); reasons = []
        mol = Chem.MolFromSmiles(row.canonical_smiles)
        if mol is None:
            record['rejection_reasons'] = 'invalid_structure'; results.append(record); continue
        sim = max(DataStructs.BulkTanimotoSimilarity(fpgen.GetFingerprint(mol), reffps))
        record['max_similarity_to_controls'] = sim
        if not bool(row.novelty_fully_checked) or not bool(row.is_globally_novel): reasons.append('novelty_unverified')
        if float(row.max_tanimoto_kinome788k) >= cfg['max_reference_similarity'] or sim >= cfg['max_reference_similarity']:
            reasons.append('too_similar_to_reference_chemistry')
        if not np.isfinite(row.mu) or row.mu < cfg['min_predicted_pactivity']: reasons.append('predicted_potency')
        if not np.isfinite(row.mu_sel) or row.mu_sel < cfg['min_selectivity_margin']: reasons.append('predicted_selectivity')
        routes = json.loads(row.get('routes_json', '[]')) if isinstance(row.get('routes_json'), str) else []
        if not routes: reasons.append('route_evidence_missing')
        record['scaffold'] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if not reasons:
            cid = f'NOVEL-{i:05d}'; ds = dock(cid, row.canonical_smiles); trials.extend(ds)
            if not all(r['docking_success'] for r in ds): reasons.append('docking_failed')
            else:
                scores = np.array([r['docking_affinity_kcal_mol'] for r in ds])
                record.update(docking_median=float(np.median(scores)), docking_std=float(scores.std()),
                              improvement_over_best_reference=float(reference - np.median(scores)), pose_id=cid)
                record['ligand_efficiency'] = float(-np.median(scores) / mol.GetNumHeavyAtoms())
                if record['ligand_efficiency'] < cfg['min_ligand_efficiency']: reasons.append('size_driven_docking_score')
                if any(r.get('severe_clash', True) for r in ds): reasons.append('severe_pose_clash')
                if record['improvement_over_best_reference'] < cfg['min_docking_improvement']: reasons.append('reference_comparison')
                if record['docking_std'] > cfg['max_docking_std']: reasons.append('unstable_docking')
            (dest / f'{cid}_routes.json').write_text(json.dumps(routes, indent=2))
        record['rejection_reasons'] = ';'.join(reasons); results.append(record)
    audit = pd.DataFrame(results)
    audit.to_csv(dest / 'candidate_audit.csv', index=False)
    pd.DataFrame(trials).to_csv(dest / 'candidate_docking.csv', index=False)
    selected = audit[audit.rejection_reasons == ''].copy()
    if len(selected):
        selected = selected.sort_values(['docking_median', 'n_steps', 'sa_score'])
        selected = selected.groupby('scaffold', sort=False).head(cfg['max_per_scaffold']).head(cfg['max_final_candidates'])
    selected.to_csv(dest / 'ranked_candidates.csv', index=False)
    (dest / 'summary.json').write_text(json.dumps({'n_evaluated': len(audit), 'n_selected': len(selected),
        'protocol': cfg, 'receptor_sha256': hashlib.sha256(receptor.read_bytes()).hexdigest(),
        'best_reference_median': float(reference),
        'interpretation': 'Computational screening only. No experimental superiority, inhibition, synthetic yield or patentability established.'}, indent=2))
    return 0

if __name__ == '__main__': raise SystemExit(main())
