"""Refit selected zoo library on development data and export its production adapter."""
import argparse, ast, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from tnk1.context import Ctx
from tnk1.zoo import REGISTRY, build
from tnk1.zoo.features import featurize, dense_columns
from tnk1.zoo.deployment import FittedMember
from tnk1.baselines.features import DescriptorScaler
from tnk1.uq.calibration import Calibrator

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config'); ap.add_argument('--version', default='v3')
    a = ap.parse_args(); C = Ctx(a.config, a.version, 'cpu')
    b = pickle.loads((C.proc / 'stack.pkl').read_bytes())
    t = C.tnk1_rows(C.dev_keys, censored=False)
    folds = pd.read_csv(C.runs / 'stage2b_zoo_folds.csv')
    members = {}
    for name in b['names']:
        if name == 'DeepEnsemble': continue
        X, ok = featurize(REGISTRY[name].rep, list(t.smiles))
        if not ok.all(): raise ValueError(f'Invalid training features: {name}')
        dense = dense_columns(REGISTRY[name].rep)
        sc = None
        if dense is not None and np.asarray(dense).any():
            sc = DescriptorScaler().fit(X[:, dense]); X = X.copy()
            X[:, dense] = sc.transform(X[:, dense])
        rows = folds[(folds.model == name) & (folds.status == 'ok')]
        params = ast.literal_eval(rows.params.mode().iloc[0])
        if 'seeded' in REGISTRY[name].tags: params['random_state'] = 42
        members[name] = FittedMember(name, build(name, params).fit(X, t.pact.values), dense, sc)
    oof = pd.read_csv(C.runs / 'stage4b_stack_oof.csv')
    residual = oof.y.values - oof.stacked.values
    sd = max(float(np.sqrt(np.mean(residual ** 2))), 0.1)
    b.update(members=members, residual_sd=sd,
             uncertainty_note='Empirical cross-fitted residual calibration; no novel-scaffold coverage guarantee',
             off_target_backend='deep ensemble')
    (C.proc / 'selected_predictor.pkl').write_bytes(pickle.dumps(b))
    cal = Calibrator(C.cfg['uq']['alpha']).fit(oof.y.values, oof.stacked.values, np.full(len(oof), sd))
    (C.proc / 'selected_calibrator.pkl').write_bytes(pickle.dumps(cal))
    (C.proc / 'selected_ad.pkl').write_bytes((C.proc / 'ad.pkl').read_bytes())

if __name__ == '__main__': main()
