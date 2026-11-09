"""Tests for the training losses: masked/weighted Huber, censored hinge, heteroscedastic
NLL, NIG evidential and L2-SP.

torch-gated, so this module skips in a core-only install. Writing these surfaced defect
#25 (tau override discarding per-row censor bounds) and #26 (detached zero loss).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is required for the loss tests")

from tnk1.models.losses import evidential_loss, l2sp, task_loss  # noqa: E402

# ------------------------------------------------------------------------------ task_loss

def test_exact_rows_huber_is_zero_on_a_perfect_fit():
    mu = torch.tensor([7.0, 6.0])
    loss = task_loss(mu, torch.zeros(2), mu.clone(), torch.tensor([0, 0]), mode="huber")
    assert loss.item() == pytest.approx(0.0)


def test_censored_row_below_its_bound_is_not_penalised():
    """Predicting pAct=4 for a ">10 uM" record (bound 5) is consistent with the observation."""
    loss = task_loss(torch.tensor([4.0]), torch.zeros(1), torch.tensor([5.0]),
                     torch.tensor([1]), mode="huber")
    assert loss.item() == pytest.approx(0.0)


def test_censored_row_above_its_bound_is_penalised_quadratically():
    """mu exceeds the bound by 2.0 -> relu(2)^2 * cens_weight(0.5) = 2.0"""
    loss = task_loss(torch.tensor([7.0]), torch.zeros(1), torch.tensor([5.0]),
                     torch.tensor([1]), mode="huber", cens_weight=0.5)
    assert loss.item() == pytest.approx(2.0)


def test_censored_penalty_scales_with_cens_weight():
    args = (torch.tensor([7.0]), torch.zeros(1), torch.tensor([5.0]), torch.tensor([1]))
    full = task_loss(*args, mode="huber", cens_weight=1.0)
    half = task_loss(*args, mode="huber", cens_weight=0.5)
    assert full.item() == pytest.approx(2 * half.item())


def test_tau_override_discards_per_row_censor_bounds():
    """Regression: `tau` REPLACES every row's bound with one constant. compute_finetune_loss
    used to default to tau=5.0, silently collapsing a >1 uM bound (6) and a >100 uM bound (4)
    onto 5.0 -- over-penalising one and under-penalising the other. It now defaults to None."""
    mu = torch.tensor([7.0, 7.0])
    y = torch.tensor([6.0, 4.0])          # genuine per-row bounds
    cens = torch.tensor([1, 1])
    per_row = task_loss(mu, torch.zeros(2), y, cens, mode="huber", tau=None)
    forced = task_loss(mu, torch.zeros(2), y, cens, mode="huber", tau=5.0)
    assert per_row.item() == pytest.approx(0.5 * (1.0 ** 2 + 3.0 ** 2) / 2)
    assert forced.item() == pytest.approx(0.5 * (2.0 ** 2 + 2.0 ** 2) / 2)
    assert per_row.item() != pytest.approx(forced.item())

    from tnk1.models.losses import compute_finetune_loss
    import inspect
    assert inspect.signature(compute_finetune_loss).parameters["tau"].default is None


def test_sample_weights_reweight_the_exact_term():
    mu = torch.tensor([7.0, 7.0])
    y = torch.tensor([6.0, 8.0])
    cens = torch.tensor([0, 0])
    # weighting the first residual 3x shifts the weighted mean toward it
    w_eq = task_loss(mu, torch.zeros(2), y, cens, torch.ones(2), mode="huber")
    w_sk = task_loss(mu, torch.zeros(2), y, cens, torch.tensor([3.0, 1.0]), mode="huber")
    assert w_eq.item() == pytest.approx(w_sk.item())      # symmetric residuals -> equal
    # asymmetric residuals do respond to the weights
    y2 = torch.tensor([6.0, 7.0])
    a = task_loss(mu, torch.zeros(2), y2, cens, torch.ones(2), mode="huber")
    b = task_loss(mu, torch.zeros(2), y2, cens, torch.tensor([3.0, 1.0]), mode="huber")
    assert b.item() > a.item()


def test_mixed_batch_sums_both_terms():
    mu = torch.tensor([7.0, 7.0])
    y = torch.tensor([7.0, 5.0])
    cens = torch.tensor([0, 1])
    loss = task_loss(mu, torch.zeros(2), y, cens, mode="huber", cens_weight=0.5)
    assert loss.item() == pytest.approx(0.0 + 0.5 * 4.0)


def test_nll_mode_rewards_calibrated_variance():
    """At a fixed residual, NLL is minimised when exp(logvar) matches the squared error."""
    mu, y = torch.tensor([7.0]), torch.tensor([8.0])     # residual 1.0 -> optimal logvar 0
    cens = torch.tensor([0])
    at_opt = task_loss(mu, torch.tensor([0.0]), y, cens, mode="nll")
    too_small = task_loss(mu, torch.tensor([-2.0]), y, cens, mode="nll")
    too_large = task_loss(mu, torch.tensor([2.0]), y, cens, mode="nll")
    assert at_opt.item() < too_small.item()
    assert at_opt.item() < too_large.item()


def test_empty_batch_returns_zero_and_stays_differentiable():
    mu = torch.zeros(0, requires_grad=True)
    loss = task_loss(mu, torch.zeros(0), torch.zeros(0), torch.zeros(0), mode="huber")
    assert loss.item() == pytest.approx(0.0)
    assert loss.requires_grad


def test_all_censored_batch_skips_the_exact_term():
    loss = task_loss(torch.tensor([9.0]), torch.zeros(1), torch.tensor([5.0]),
                     torch.tensor([1]), mode="nll", cens_weight=1.0)
    assert loss.item() == pytest.approx(16.0)       # relu(9-5)^2, no NLL contribution


def test_task_loss_gradient_flows_to_mu():
    mu = torch.tensor([7.0], requires_grad=True)
    task_loss(mu, torch.zeros(1), torch.tensor([5.0]), torch.tensor([1]), mode="huber").backward()
    assert mu.grad is not None and mu.grad.item() > 0      # pushes mu back below the bound


# ----------------------------------------------------------------------------------- l2sp

class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Linear(2, 2, bias=False)
        self.head = torch.nn.Linear(2, 1, bias=False)


def test_l2sp_is_zero_at_the_reference_point():
    m = _Tiny()
    ref = {k: v.detach().clone() for k, v in m.named_parameters()}
    assert l2sp(m, ref, 1.0, 1.0).item() == pytest.approx(0.0)


def test_l2sp_penalises_encoder_and_head_with_their_own_coefficients():
    m = _Tiny()
    ref = {k: torch.zeros_like(v) for k, v in m.named_parameters()}
    with torch.no_grad():
        m.enc.weight.fill_(1.0)      # 4 params -> sum sq 4
        m.head.weight.fill_(1.0)     # 2 params -> sum sq 2
    assert l2sp(m, ref, 1.0, 0.0).item() == pytest.approx(4.0)
    assert l2sp(m, ref, 0.0, 1.0).item() == pytest.approx(2.0)
    assert l2sp(m, ref, 1.0, 1.0).item() == pytest.approx(6.0)


def test_l2sp_ignores_parameters_absent_from_the_reference():
    m = _Tiny()
    with torch.no_grad():
        m.enc.weight.fill_(5.0)
    assert l2sp(m, {}, 1.0, 1.0) == 0.0


# --------------------------------------------------------------------------- evidential

def test_evidential_loss_is_finite_and_penalises_a_worse_fit():
    raw = torch.zeros(4, 4)
    near = evidential_loss(raw, torch.zeros(4))
    far = evidential_loss(raw, torch.full((4,), 5.0))
    assert np.isfinite(near.item()) and np.isfinite(far.item())
    assert far.item() > near.item()


