"""Stage 2 baselines 2.1-2.4 (2.5 = scratch D-MPNN is trained via tnk1.models.trainer)."""
from __future__ import annotations
import numpy as np


def tanimoto_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Count-vector Tanimoto <a,b> / (|a|^2 + |b|^2 - <a,b>)."""
    ab = A @ B.T
    aa, bb = (A * A).sum(1)[:, None], (B * B).sum(1)[None, :]
    return ab / np.maximum(aa + bb - ab, 1e-9)


class KNNTanimoto:
    """Model 2.1: similarity-weighted mean of top-k neighbours."""
    def __init__(self, k=5): self.k = k
    def fit(self, X, y): self.X, self.y = X, np.asarray(y, float); return self
    def predict(self, X, return_std=False):
        S = tanimoto_sim(X, self.X); k = min(self.k, S.shape[1])
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        s = np.take_along_axis(S, idx, 1); yy = self.y[idx]
        mu = (s * yy).sum(1) / np.maximum(s.sum(1), 1e-9)
        if return_std:
            return mu, np.sqrt(((s * (yy - mu[:, None]) ** 2).sum(1)) / np.maximum(s.sum(1), 1e-9))
        return mu


class ForestWithStd:
    """Model 2.2: RF; std across the individual trees = baseline uncertainty."""
    def __init__(self, **kw):
        from sklearn.ensemble import RandomForestRegressor
        self.m = RandomForestRegressor(n_estimators=kw.pop("n_estimators", 500), n_jobs=-1, **kw)
    def fit(self, X, y): self.m.fit(X, y); return self
    def predict(self, X, return_std=False):
        per = np.stack([t.predict(X) for t in self.m.estimators_])
        return (per.mean(0), per.std(0)) if return_std else per.mean(0)


def make_xgb(**kw):
    """Model 2.3."""
    from xgboost import XGBRegressor
    return XGBRegressor(subsample=0.8, colsample_bytree=0.7, n_estimators=kw.pop("n_estimators", 500),
                        tree_method="hist", **kw)


class TanimotoGP:
    """Model 2.4: exact GP, Tanimoto kernel k = s_f^2 <x,x'>/(|x|^2+|x'|^2-<x,x'>), trained by exact MLL (GPyTorch)."""
    def __init__(self, iters=200, lr=0.05, device="cpu"):
        self.iters, self.lr, self.device = iters, lr, device

    def _build(self, X, y):
        import torch, gpytorch

        class TanimotoKernel(gpytorch.kernels.Kernel):
            is_stationary = False
            def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **p):
                ab = x1 @ x2.transpose(-1, -2)
                aa = (x1 * x1).sum(-1, keepdim=True); bb = (x2 * x2).sum(-1, keepdim=True).transpose(-1, -2)
                k = ab / (aa + bb - ab).clamp_min(1e-9)
                return k.diagonal(dim1=-2, dim2=-1) if diag else k

        class GP(gpytorch.models.ExactGP):
            def __init__(s, X, y, lik):
                super().__init__(X, y, lik)
                s.mean_module = gpytorch.means.ConstantMean()
                s.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
            def forward(s, x):
                return gpytorch.distributions.MultivariateNormal(s.mean_module(x), s.covar_module(x))
        return GP, gpytorch

    def fit(self, X, y):
        import torch
        self._t = torch
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32, device=self.device)
        GP, gp = self._build(X, y)
        self.lik = gp.likelihoods.GaussianLikelihood().to(self.device)
        self.model = GP(Xt, yt, self.lik).to(self.device)
        self.model.train(); self.lik.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = gp.mlls.ExactMarginalLogLikelihood(self.lik, self.model)
        for _ in range(self.iters):
            opt.zero_grad(); loss = -mll(self.model(Xt), yt); loss.backward(); opt.step()
        self.model.eval(); self.lik.eval(); self.gp = gp
        return self

    def predict(self, X, return_std=False, include_noise=False):
        torch = self._t
        with torch.no_grad(), self.gp.settings.fast_pred_var(False):
            f = self.model(torch.tensor(X, dtype=torch.float32, device=self.device))
            d = self.lik(f) if include_noise else f
            mu, sd = d.mean.cpu().numpy(), d.variance.clamp_min(1e-12).sqrt().cpu().numpy()
        return (mu, sd) if return_std else mu
