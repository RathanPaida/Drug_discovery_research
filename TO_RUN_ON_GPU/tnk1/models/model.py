"""Design A (multi-head) and Design B (target-conditioned, 631-d fusion) predictors + heads for UQ variants."""
import numpy as np
import torch
import torch.nn as nn
from .dmpnn import DMPNNEncoder


class KinomeModel(nn.Module):
    """design='B': [z_mol(300)+desc(200) || z_kin(128) || assay(3)] -> 512 -> 256 -> (mu, log var)
       design='A': [z_mol+desc || assay] -> K parallel linear outputs (mu only; log var fixed at 0).
       head_type='evidential' -> NIG (gamma, nu, alpha, beta)."""

    def __init__(self, cfg: dict, kin_store: dict | None = None, n_kinases: int | None = None,
                 design: str | None = None, head_type: str = "gauss"):
        super().__init__()
        m = cfg["model"]; self.design = design or m["design"]; self.head_type = head_type
        self.enc = DMPNNEncoder(m["hidden"], m["depth"], m["dropout"])
        d_mol = m["hidden"] + m["n_desc"]
        self.n_assay = 3
        if self.design == "B":
            assert kin_store is not None
            # cfg.model.esm_dim / klifs_dim were declared in configs/default.yaml but never
            # read: the fusion width came from `feats.shape[1]`, so a kinase store built
            # with a different ESM variant (650M -> 1280-d vs 3B -> 2560-d) or a different
            # KLIFS encoding would train happily at the wrong width and the config would
            # still claim 1280/425. Check them instead of ignoring them.
            e_dim, k_dim = kin_store["esm"].shape[1], kin_store["klifs"].shape[1]
            for name, got, want in (("esm_dim", e_dim, m.get("esm_dim")),
                                    ("klifs_dim", k_dim, m.get("klifs_dim"))):
                if want is not None and int(want) != got:
                    raise ValueError(
                        f"kinase store disagrees with cfg.model.{name}: store has {got}, "
                        f"config declares {want}. Either the store was built with a "
                        f"different encoder than the config documents, or the config is "
                        f"stale -- do not train until they agree."
                    )
            feats = np.hstack([kin_store["esm"], kin_store["klifs"]]).astype(np.float32)
            self.register_buffer("kin_feat", torch.tensor(feats))
            self.kproj = nn.Sequential(nn.Linear(feats.shape[1], m["kin_proj_dim"]), nn.ReLU(), nn.Dropout(m["dropout"]))
            d_in = d_mol + m["kin_proj_dim"] + self.n_assay            # 500 + 128 + 3 = 631
            h1, h2 = m["fusion_hidden"]
            out = 4 if head_type == "evidential" else 2
            self.head = nn.Sequential(nn.Linear(d_in, h1), nn.ReLU(), nn.Dropout(m["dropout"]),
                                      nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(m["dropout"]), nn.Linear(h2, out))
        else:
            K = n_kinases or len(kin_store["ids"])
            self.head = nn.Sequential(nn.Dropout(m["dropout"]), nn.Linear(d_mol + self.n_assay, K))

    # parameter groups used by fine-tuning / L2-SP
    def encoder_params(self): return list(self.enc.parameters())
    def kproj_params(self): return list(self.kproj.parameters()) if hasattr(self, "kproj") else []
    def head_params(self): return list(self.head.parameters())

    def forward(self, b: dict, desc, assay, kin_idx):
        z = torch.cat([self.enc(b), desc], 1)
        if self.design == "B":
            k_idx = kin_idx.clamp(0, len(self.kin_feat) - 1)
            zk = self.kproj(self.kin_feat[k_idx])
            o = self.head(torch.cat([z, zk, assay], 1))
            if self.head_type == "evidential":
                return o
            return o[:, 0], o[:, 1].clamp(-6, 4)                     # mu, log sigma_ale^2
        o = self.head(torch.cat([z, assay], 1))
        mu = o.gather(1, kin_idx[:, None])[:, 0]
        return mu, torch.zeros_like(mu)

    def enable_dropout(self):
        for mod in self.modules():
            if isinstance(mod, nn.Dropout):
                mod.train()


def evidential_outputs(raw):
    gamma, nu, alpha, beta = raw[:, 0], *[nn.functional.softplus(raw[:, i]) for i in (1, 2, 3)]
    return gamma, nu + 1e-6, alpha + 1.0 + 1e-6, beta + 1e-6
