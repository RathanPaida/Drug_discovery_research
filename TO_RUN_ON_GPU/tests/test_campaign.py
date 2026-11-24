import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tnk1.ensembles.stacking import STACKERS, DiscreteSuperLearner
from tnk1.zoo.deployment import SelectedPredictor
from scripts.run_pipeline import build_plan

def test_campaign_plan_has_every_model_and_algorithm():
    p = subprocess.run([sys.executable, 'scripts/run_campaign.py', '--plan'], capture_output=True, text=True, check=True)
    plan = json.loads(p.stdout)
    assert len(plan['models']) == 26 and len(plan['rl']) == 8
    assert {'transfer', 'ablate', 'stack'} <= set(plan['stages'])
    assert plan['retro_enabled']

def test_algorithm_candidate_paths_follow_training():
    for algo in ['ppo', 'augmem', 'gflownet']:
        stages = {s.name:s for s in build_plan('v3','cpu',3,False,rl_algo=algo)}
        cycle = stages['cycle'].argv
        assert cycle[cycle.index('--agent')+1] == '{runs}/rl_' + algo + '/agent.pt'

def test_rank_predictors_are_batch_independent():
    rng = np.random.RandomState(9); P = rng.randn(80,3); y = P[:,0]+6
    for name in ['rank_nnls', 'rank_mean']:
        m = STACKERS[name]().fit(P,y)
        batch = m.predict(P[:8])
        single = np.array([m.predict(P[i:i+1])[0] for i in range(8)])
        np.testing.assert_allclose(batch, single)

def test_selected_predictor_overrides_only_primary():
    class Neural:
        def predict_members(self, smiles, targets, assay_idx=0):
            return np.full((3,len(smiles),len(targets)),2.), np.zeros((3,len(smiles),len(targets)))
    class Member:
        def predict(self, smiles): return np.full(len(smiles),8.)
    d = DiscreteSuperLearner(); d.col_=0; d.stacker_=None
    p = SelectedPredictor({'names':['rf'], 'members':{'rf':Member()}, 'dsl':d, 'residual_sd':.4}, Neural(),'TNK1')
    mu, lv = p.predict_members(['CCO'],['TNK1','SRC'])
    assert mu[:,0,0].mean() == 8 and mu[:,0,1].mean() == 2
    assert np.isfinite(lv).all()
