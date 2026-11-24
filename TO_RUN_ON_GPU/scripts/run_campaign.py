"""Complete isolated HPC experiment: all models, stackers, fine-tunes and RL objectives.
No docking or wet-lab activity is inferred from a generated candidate.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import yaml
from tnk1.utils.common import load_config
from tnk1.zoo import REGISTRY, available
from rl_min.algos import ALGOS

STAGES = ['baselines', 'ens', 'oof', 'calibrate', 'compare', 'critic', 'zoo',
          'stack', 'taskselect', 'transfer', 'ablate', 'admet', 'prior']

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config'); ap.add_argument('--version', default='v3')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--plan', action='store_true')
    ap.add_argument('--skip-retro', action='store_true', help='Explicitly omit synthesis screening; candidates remain unverified')
    a = ap.parse_args(); cfg = load_config(a.config)
    root = Path(__file__).resolve().parent.parent
    print(json.dumps({'stages': STAGES, 'models': list(REGISTRY), 'rl': list(ALGOS),
                      'candidate_stage_per_algorithm': True, 'retro_enabled': not a.skip_retro}, indent=2))
    if a.plan: return 0
    missing = {n: s for n, s in available().items() if s != 'ok'}
    if missing: raise SystemExit(f'All-model campaign cannot omit models: {missing}')
    screen = yaml.safe_load((root / 'configs/final_screen.yaml').read_text())
    receptor = Path(screen['receptor'])
    if not receptor.is_absolute(): receptor = root / receptor
    from tnk1.docking.paths import resolve_vina
    resolve_vina()
    if not receptor.is_file():
        raise SystemExit(f'Final screening requires a prepared TNK1 receptor: {receptor}')
    if not a.skip_retro:
        import importlib.util
        rcfg = yaml.safe_load((root / 'configs/aizynth.yml').read_text())
        paths = [*rcfg['expansion']['uspto'], rcfg['filter']['uspto'], rcfg['stock']['zinc']]
        absent = [v for v in paths if not Path(v).is_file()]
        if importlib.util.find_spec('aizynthfinder') is None or absent:
            raise SystemExit(f'Retrosynthesis requires AiZynthFinder and real paths in configs/aizynth.yml. Missing files: {absent}. See docs/HPC_COMPLETE_CAMPAIGN.md')
    # Immutable experiment namespace includes source, config and dataset identity.
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    h.update((root / 'configs/final_screen.yaml').read_bytes())
    h.update((root / 'configs/aizynth.yml').read_bytes())
    h.update(receptor.read_bytes())
    for folder in ['tnk1', 'rl_min', 'scripts']:
        for p in sorted((root / folder).rglob('*.py')): h.update(p.read_bytes())
    source_proc = Path(cfg['paths']['processed'])
    h.update((source_proc / f'kinome_long_{a.version}.parquet.sha256').read_bytes())
    h.update((source_proc / f'splits_{a.version}.json').read_bytes())
    dest = Path(cfg['paths']['runs']) / ('campaign_' + h.hexdigest()[:16])
    proc = dest / 'processed'; proc.mkdir(parents=True, exist_ok=True)
    for name in [f'kinome_long_{a.version}.parquet', f'kinome_long_{a.version}.parquet.sha256',
                 f'splits_{a.version}.json', 'kinase_store.npz', 'ssl_encoder.pt']:
        src = source_proc / name
        if src.exists() and not (proc / name).exists(): shutil.copy2(src, proc / name)
    cfg['paths'].update(processed=str(proc), runs=str(dest))
    cfg.setdefault('critic', {})['backend'] = 'ensemble'
    config = dest / 'campaign.yaml'; config.write_text(yaml.safe_dump(cfg))
    statuses = {}
    def run(key, args, config_path=config):
        env = dict(os.environ, TNK1_CONFIG=str(config_path), PYTHONUNBUFFERED='1')
        log = dest / (key + '.log')
        print(f'[campaign] {key} -> {log}', flush=True)
        with log.open('w') as f:
            rc = subprocess.call([sys.executable, *args], cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT)
        statuses[key] = {'exit': rc, 'log': str(log)}
        (dest / 'campaign_status.json').write_text(json.dumps(statuses, indent=2))
        record = Path(yaml.safe_load(config_path.read_text())['paths']['runs']) / 'pipeline' / 'last_run.json'
        if args[0] == 'scripts/run_pipeline.py' and record.exists():
            report = json.loads(record.read_text())
            if any(v['status'] in ('failed', 'blocked', 'no-output') for v in report.get('stages', {}).values()): return False
        return rc == 0
    base = ['scripts/run_pipeline.py', '--version', a.version, '--device', a.device]
    failures = []
    for stage in STAGES:
        ok = run(stage, base + ['--only', stage, '--zoo-tag', 'all'])
        if not ok:
            failures.append(stage)
            if stage not in ['transfer', 'ablate']: return 1
    # A sweep must report every requested model, including failures.
    import pandas as pd
    f = pd.read_csv(dest / 'stage2b_zoo_folds.csv')
    bad = sorted(set(REGISTRY) - set(f.loc[f.status == 'ok', 'model']))
    if bad or (f.status != 'ok').any():
        raise SystemExit(f'Incomplete zoo; inspect stage2b_zoo_folds.csv. No successful folds: {bad}')
    if not run('export', ['scripts/export_selected_predictor.py', '--version', a.version]): return 1
    cfg['critic']['backend'] = 'selected'
    for algo in ALGOS:
        run_dir = dest / algo; run_dir.mkdir(exist_ok=True)
        cfg['paths']['runs'] = str(run_dir)
        cp = run_dir / 'config.yaml'; cp.write_text(yaml.safe_dump(cfg))
        if not run('rl_' + algo, base + ['--only', 'rl', '--rl-algo', algo], cp):
            failures.append('rl_' + algo); continue
        agent_dir = run_dir / ('rl_min' if algo == 'dap' else 'rl_' + algo)
        args = ['scripts/run_stage8_cycle.py', 'prepare', '--version', a.version,
                '--cycle', '3', '--device', a.device, '--agent', str(agent_dir / 'agent.pt'),
                '--vocab', str(proc / 'rl_min/vocab.txt')]
        if a.skip_retro: args.append('--skip-retro')
        if not run('candidates_' + algo, args, cp):
            failures.append('candidates_' + algo); continue
        candidates = pd.read_csv(run_dir / 'cycle_03' / 'candidates_pre_docking.csv')
        if len(candidates):
            # Audit the top 40 per algorithm; this is a screening shortlist, not a lab contract.
            candidates = candidates.sort_values('R', ascending=False).head(40).copy()
            candidates['canonical_smiles'] = candidates.smiles
            candidates['compound_id'] = [f'{algo}-{i:03d}' for i in range(len(candidates))]
            shortlist = run_dir / 'screening_shortlist.csv'; candidates.to_csv(shortlist, index=False)
            if not run('novelty_' + algo, ['scripts/check_novelty_external.py', '--contract',
                       str(shortlist), '--outdir', str(run_dir / 'novelty')], cp):
                failures.append('novelty_' + algo)
            audited = run_dir / 'novelty' / 'novelty_verification_results.csv'
            if audited.exists():
                checked = pd.read_csv(audited)
                keep = checked.novelty_fully_checked & checked.is_globally_novel
                if a.skip_retro: keep &= False
                checked[keep].to_csv(run_dir / 'novel_route_screened_candidates.csv', index=False)
    if not run('final_screen', ['scripts/finalize_candidates.py', '--campaign', str(dest)]):
        failures.append('final_screen')
    print(f'Campaign artifacts: {dest}; failed stages: {failures}', flush=True)
    return int(bool(failures))

if __name__ == '__main__': raise SystemExit(main())
