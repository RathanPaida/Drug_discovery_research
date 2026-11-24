"""Deploy the selected TNK1 predictor; off-target predictions remain explicitly neural."""
import pickle
import numpy as np
from .features import featurize

class FittedMember:
    def __init__(self, name, model, dense, scaler):
        self.name, self.model, self.dense, self.scaler = name, model, dense, scaler
    def predict(self, smiles):
        from .registry import REGISTRY
        X, ok = featurize(REGISTRY[self.name].rep, smiles)
        if not ok.all():
            raise ValueError("Invalid structure in deployed critic")
        if self.scaler is not None:
            X = X.copy(); X[:, self.dense] = self.scaler.transform(X[:, self.dense])
        p = self.model.predict(X)
        return np.asarray(p[0] if isinstance(p, tuple) else p)

class SelectedPredictor:
    def __init__(self, bundle, neural, target):
        self.bundle, self.neural, self.target = bundle, neural, target
    def latent(self, smiles):
        return self.neural.latent(smiles)
    def predict_members(self, smiles, uniprots, assay_idx=0):
        mu, lv = self.neural.predict_members(smiles, uniprots, assay_idx)
        if self.target not in uniprots:
            return mu, lv
        j = list(uniprots).index(self.target)
        cols = []
        for name in self.bundle['names']:
            cols.append(mu[..., j].mean(0) if name == 'DeepEnsemble'
                        else self.bundle['members'][name].predict(smiles))
        P = np.column_stack(cols)
        pred = self.bundle['dsl'].predict(P)
        # Cross-architecture spread is an empirical disagreement penalty, not a
        # posterior uncertainty guarantee. Preserve all neural off-target members.
        spread = np.std(P, axis=1)
        offsets = np.linspace(-1, 1, len(mu)); offsets /= max(offsets.std(), 1e-6)
        mu = mu.copy(); lv = lv.copy()
        mu[..., j] = pred[None, :] + offsets[:, None] * spread[None, :]
        lv[..., j] = 2 * np.log(self.bundle['residual_sd'])
        return mu, lv

def load_predictor(C):
    from ..uq.ensemble import Ensemble
    neural = Ensemble(C.cfg, C.store, C.desc_store(), C.proc / 'ensemble', C.device)
    backend = C.cfg.get('critic', {}).get('backend', 'ensemble')
    if backend == 'ensemble':
        return neural
    if backend != 'selected':
        raise ValueError(f"Unsupported production critic {backend}; use selected or ensemble")
    b = pickle.loads((C.proc / 'selected_predictor.pkl').read_bytes())
    if b['dataset_sha256'] != C.sha:
        raise ValueError('Selected critic belongs to a different dataset')
    return SelectedPredictor(b, neural, C.tnk1)
