"""Tests for the learning machinery: D-MPNN message passing, the training loop, and
three-phase fine-tuning.

Why these are worth having. The hermetic suite covered the numpy scientific core very
well (reward, uq, splits, stats at 97-100%) and the torch half not at all:
`trainer.py` 0%, `pipeline.py` 0%, `pretrain.py` 0%, `model.py` 32%, `dmpnn.py` 22%.
That is the half that actually produces the numbers, and the parts most easily wrong in a
way no assertion catches -- a message-passing step that leaks the reverse bond still
trains, converges, and reports a plausible RMSE.

Everything here runs on CPU with a handful of molecules, in seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from tnk1.models.data import DescStore, LabelledSet, assay_index, get_graph  # noqa: E402
from tnk1.models.dmpnn import DMPNNEncoder, to_torch  # noqa: E402
from tnk1.models.graph import ATOM_DIM, BOND_DIM, collate_graphs, mol_to_graph  # noqa: E402
from tnk1.models.losses import l2sp, task_loss  # noqa: E402
from tnk1.models.model import KinomeModel  # noqa: E402
from tnk1.models.trainer import predict, run_epochs  # noqa: E402

SMILES = ["c1ccccc1", "CCO", "CC(=O)Nc1ccc(O)cc1", "C", "c1ccc2[nH]ncc2c1",
          "CN1CCN(CC1)c1ccccc1", "OC(=O)c1ccccc1", "Cc1ccccc1N"]


# ----------------------------------------------------------------- graph featurisation
def test_graph_shapes_and_directed_edges():
    g = mol_to_graph("CCO")
    assert g["x"].shape == (3, ATOM_DIM)
    assert g["ef"].shape == (4, BOND_DIM), "2 bonds -> 4 directed edges"
    assert g["src"].shape == g["dst"].shape == (4,)


def test_reverse_edge_index_is_an_involution():
    """`rev` must map each directed edge to its twin. The D-MPNN update subtracts
    `h[rev]` to exclude the reverse bond; if `rev` is wrong the model silently passes a
    message back along the edge it came from and still trains."""
    g = mol_to_graph("CC(=O)Nc1ccc(O)cc1")
    rev = g["rev"]
    assert np.array_equal(rev[rev], np.arange(len(rev))), "rev(rev(e)) != e"
    # the twin of an edge u->v must be v->u
    assert np.array_equal(g["src"][rev], g["dst"])
    assert np.array_equal(g["dst"][rev], g["src"])


def test_single_atom_molecule_does_not_crash_the_encoder():
    """Methane has zero bonds, so the message-passing tensors are empty."""
    enc = DMPNNEncoder(hidden=16, depth=2, dropout=0.0)
    b = to_torch(collate_graphs([mol_to_graph("C")]), "cpu")
    z = enc(b)
    assert z.shape == (1, 16)
    assert torch.isfinite(z).all()


def test_unparseable_smiles_yields_a_placeholder_not_an_exception():
    g = mol_to_graph("this is not a molecule")
    assert g["x"].shape == (1, ATOM_DIM)
    assert len(g["src"]) == 0


def test_collate_offsets_keep_molecules_separate():
    graphs = [mol_to_graph(s) for s in ("CCO", "c1ccccc1")]
    b = collate_graphs(graphs)
    assert b["n_mol"] == 2
    assert b["x"].shape[0] == len(graphs[0]["x"]) + len(graphs[1]["x"])
    # atom indices of molecule 1 must be offset past molecule 0
    assert b["batch"].tolist() == [0] * 3 + [1] * 6
    assert b["src"].max() < b["x"].shape[0]


def test_encoder_is_permutation_equivariant_over_batch_order():
    """Pooling must not mix molecules: encoding [a, b] then [b, a] gives swapped rows."""
    enc = DMPNNEncoder(hidden=24, depth=3, dropout=0.0).eval()
    ga, gb = mol_to_graph("CCO"), mol_to_graph("c1ccccc1")
    with torch.no_grad():
        z1 = enc(to_torch(collate_graphs([ga, gb]), "cpu"))
        z2 = enc(to_torch(collate_graphs([gb, ga]), "cpu"))
    assert torch.allclose(z1[0], z2[1], atol=1e-5)
    assert torch.allclose(z1[1], z2[0], atol=1e-5)


def test_graph_cache_returns_equal_graphs():
    a, b = get_graph("CCO"), get_graph("CCO")
    assert np.array_equal(a["x"], b["x"])


# ----------------------------------------------------------------- losses
def test_censored_hinge_only_penalises_predictions_above_the_bound():
    """A right-censored row (IC50 > 10 uM) says the true pAct is BELOW the bound, so a
    prediction under the bound must be free and one above must be penalised."""
    y = torch.tensor([6.0, 6.0])
    cens = torch.tensor([1, 1])
    lv = torch.zeros(2)
    under = task_loss(torch.tensor([5.0, 5.0]), lv, y, cens)
    over = task_loss(torch.tensor([7.0, 7.0]), lv, y, cens)
    assert float(under) == pytest.approx(0.0, abs=1e-6)
    assert float(over) > 0.1


def test_loss_with_no_rows_of_either_kind_still_has_a_gradient():
    """Pins the `mu.sum() * 0` fix: `mu.new_zeros(())` is detached from the graph, so a
    batch contributing neither an exact nor a censored term returned requires_grad=False
    and `loss.backward()` raised 'does not require grad'."""
    mu = torch.zeros(0, requires_grad=True)
    loss = task_loss(mu, torch.zeros(0), torch.zeros(0), torch.zeros(0, dtype=torch.long))
    assert loss.requires_grad
    loss.backward()


def test_exact_rows_use_the_requested_mode():
    mu = torch.tensor([5.0]); y = torch.tensor([7.0])
    c = torch.tensor([0]); lv = torch.zeros(1)
    huber = float(task_loss(mu, lv, y, c, mode="huber"))
    nll = float(task_loss(mu, lv, y, c, mode="nll"))
    assert huber != nll
    assert huber > 0 and nll > 0


def test_per_sample_weights_change_the_loss():
    mu = torch.tensor([5.0, 5.0]); y = torch.tensor([7.0, 7.0])
    c = torch.tensor([0, 0]); lv = torch.zeros(2)
    a = float(task_loss(mu, lv, y, c, w=torch.tensor([1.0, 1.0])))
    b = float(task_loss(mu, lv, y, c, w=torch.tensor([1.0, 9.0])))
    assert a == pytest.approx(b, rel=1e-6), "equal residuals -> weighting is a no-op"
    y2 = torch.tensor([7.0, 5.0])
    c1 = float(task_loss(mu, lv, y2, c, w=torch.tensor([9.0, 1.0])))
    c2 = float(task_loss(mu, lv, y2, c, w=torch.tensor([1.0, 9.0])))
    assert c1 > c2, "weighting the larger residual must raise the loss"


def test_tau_override_replaces_every_per_row_bound():
    """`tau` is an override, not a default: it must flatten all bounds to one value."""
    mu = torch.tensor([5.5, 5.5]); lv = torch.zeros(2)
    y = torch.tensor([6.0, 4.0])              # different real bounds
    c = torch.tensor([1, 1])
    own = float(task_loss(mu, lv, y, c))                 # only the 4.0 row is violated
    flat = float(task_loss(mu, lv, y, c, tau=5.0))       # both rows violated equally
    assert own != flat


def test_l2sp_is_zero_at_the_anchor_and_grows_away_from_it():
    enc = DMPNNEncoder(hidden=8, depth=1, dropout=0.0)
    ref = {n: p.detach().clone() for n, p in enc.named_parameters()}
    assert l2sp(enc, ref, 1.0, 1.0).detach().item() == pytest.approx(0.0, abs=1e-9)
    with torch.no_grad():
        next(iter(enc.parameters())).add_(1.0)
    assert l2sp(enc, ref, 1.0, 1.0).detach().item() > 0.0


# ----------------------------------------------------------------- training loop
class _Scaler:
    """Stand-in for DescriptorScaler: no RDKit descriptor computation needed."""
    def transform(self, X):
        return np.zeros((len(X), 200), dtype=np.float32)


def _tiny_store(n_kin=3):
    return {"ids": [f"U{i}" for i in range(n_kin)],
            "esm": np.zeros((n_kin, 4), np.float32),
            "klifs": np.zeros((n_kin, 2), np.float32),
            "S": np.linspace(0.1, 0.9, n_kin).astype(np.float32)}


def _tiny_cfg():
    return {"model": {"hidden": 16, "depth": 2, "dropout": 0.0, "n_desc": 200,
                      "kin_proj_dim": 8, "fusion_hidden": [16, 8], "design": "B"}}


def _tiny_set(n=8, kin=0):
    ds = DescStore.__new__(DescStore)      # bypass the 788k preload
    ds.scaler, ds.cache = _Scaler(), {}
    smiles = SMILES[:n]
    return LabelledSet(smiles, np.full(n, kin), assay_index(["IC50"] * n),
                       np.linspace(5.0, 8.0, n), np.zeros(n), np.ones(n), ds)


def test_model_forward_returns_mu_and_clamped_logvar():
    m = KinomeModel(_tiny_cfg(), _tiny_store(), design="B")
    ds = _tiny_set()
    mu, lv = predict(m, ds, "cpu", batch_size=4)
    assert mu.shape == lv.shape == (len(ds),)
    assert np.isfinite(mu).all() and np.isfinite(lv).all()
    assert (lv >= -6).all() and (lv <= 4).all(), "log-variance must stay clamped"


def test_training_reduces_the_loss_on_data_it_can_memorise():
    torch.manual_seed(0)
    m = KinomeModel(_tiny_cfg(), _tiny_store(), design="B")
    tr = _tiny_set()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    seen = []
    run_epochs(m, tr, None, "cpu", epochs=12, optimizer=opt, loss_mode="huber",
               batch_size=4, log=lambda d: seen.append(d["train_loss"]))
    assert len(seen) == 12
    assert seen[-1] < seen[0], f"loss did not fall: {seen[0]:.4f} -> {seen[-1]:.4f}"


def test_early_stopping_restores_the_best_state_not_the_last():
    """`run_epochs` deep-copies the best state and reloads it at the end. If that path is
    broken the returned model is whatever the final (possibly worse) epoch produced, and
    nothing downstream can tell."""
    torch.manual_seed(0)
    m = KinomeModel(_tiny_cfg(), _tiny_store(), design="B")
    tr, va = _tiny_set(), _tiny_set()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    best = run_epochs(m, tr, va, "cpu", epochs=10, optimizer=opt, loss_mode="huber",
                      batch_size=4, patience=3)
    mu, _ = predict(m, va, "cpu")
    ex = va.c == 0
    final = float(np.sqrt(np.mean((mu[ex] - va.y[ex]) ** 2)))
    assert final == pytest.approx(best, abs=1e-4), (
        "model does not hold the state that achieved the reported best score")


def test_patience_stops_the_run_early():
    torch.manual_seed(0)
    m = KinomeModel(_tiny_cfg(), _tiny_store(), design="B")
    tr, va = _tiny_set(), _tiny_set()
    opt = torch.optim.AdamW(m.parameters(), lr=0.0)      # cannot improve
    epochs = []
    run_epochs(m, tr, va, "cpu", epochs=50, optimizer=opt, loss_mode="huber",
               batch_size=4, patience=2, on_epoch=lambda e, v: epochs.append(e))
    assert len(epochs) < 50, "patience never triggered"


def test_design_a_uses_a_per_kinase_head_and_zero_logvar():
    cfg = _tiny_cfg(); cfg["model"]["design"] = "A"
    m = KinomeModel(cfg, _tiny_store(), design="A", n_kinases=3)
    mu, lv = predict(m, _tiny_set(), "cpu", batch_size=4)
    assert np.allclose(lv, 0.0), "design A is homoscedastic by construction"
    assert np.isfinite(mu).all()


def test_scheduler_and_grad_clipping_do_not_break_the_step():
    torch.manual_seed(0)
    m = KinomeModel(_tiny_cfg(), _tiny_store(), design="B")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 4)
    run_epochs(m, _tiny_set(), None, "cpu", epochs=4, optimizer=opt, scheduler=sched,
               loss_mode="nll", batch_size=4)
    assert all(torch.isfinite(p).all() for p in m.parameters())


def test_l2sp_anchor_keeps_parameters_near_theta0():
    """With a large anchor coefficient and a few steps the weights must barely move."""
    torch.manual_seed(0)
    m = KinomeModel(_tiny_cfg(), _tiny_store(), design="B")
    ref = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    run_epochs(m, _tiny_set(), None, "cpu", epochs=5, optimizer=opt, loss_mode="huber",
               ref_state=ref, alphas={"enc": 1e4, "head": 1e4}, batch_size=4)
    drift = max(float((p - ref[n]).abs().max()) for n, p in m.named_parameters())
    assert drift < 0.5, f"L2-SP did not hold the trunk: max drift {drift:.3f}"


def test_mc_dropout_prediction_varies_while_eval_is_deterministic():
    cfg = _tiny_cfg(); cfg["model"]["dropout"] = 0.5
    m = KinomeModel(cfg, _tiny_store(), design="B")
    ds = _tiny_set()
    a, _ = predict(m, ds, "cpu")
    b, _ = predict(m, ds, "cpu")
    assert np.allclose(a, b), "eval() must be deterministic"
    torch.manual_seed(0); c, _ = predict(m, ds, "cpu", mc_dropout=True)
    torch.manual_seed(1); d, _ = predict(m, ds, "cpu", mc_dropout=True)
    assert not np.allclose(c, d), "mc_dropout did not enable dropout at inference"
