"""TanimotoGP.predict must batch, and batching must not change the answer.

An ExactGP in eval mode builds the full joint posterior over whatever it is handed, so
the Tanimoto kernel materialises a dense (n_test x n_test) matrix. `predict` reads only
the mean and the diagonal of the covariance, so that block is built and discarded.

Harmless at tens of molecules. Stage 8's cross-architecture audit scores the whole
candidate pool, and once the funnel stopped being starved -- 200,000 sampled instead of
~89 -- the audit reached ~117k survivors and the stage died:

    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 51.29 GiB
      tnk1/baselines/models.py  k = ab / (aa + bb - ab).clamp_min(1e-9)

after 236 minutes. Because the cycle stage is `optional=True`, run_pipeline.py printed
"stage is optional; continuing" and the job still exited 0, so the chain moved on and
nothing recorded that the run had produced no molecules.

Batching is exact for what is read: the posterior mean at a test point and its MARGINAL
variance depend only on the training set and that point. Only the cross-covariance
between test points changes, and nothing reads it. These tests hold that line.
"""
from __future__ import annotations

import numpy as np
import pytest

from tnk1.baselines.models import TanimotoGP

pytest.importorskip("gpytorch")
pytest.importorskip("torch")


def _fp(n, bits=64, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, bits)) < 0.15).astype(np.float32)


@pytest.fixture(scope="module")
def fitted():
    X = _fp(40, seed=1)
    y = X.sum(1) * 0.1 + 5.0
    return TanimotoGP(iters=15, lr=0.1, device="cpu").fit(X, y)


def test_batched_mean_matches_single_pass(fitted):
    Xt = _fp(37, seed=2)
    one = fitted.predict(Xt, batch_size=len(Xt))
    many = fitted.predict(Xt, batch_size=8)
    assert one.shape == many.shape == (len(Xt),)
    np.testing.assert_allclose(one, many, rtol=1e-5, atol=1e-5)


def test_batched_std_matches_single_pass(fitted):
    Xt = _fp(37, seed=3)
    m1, s1 = fitted.predict(Xt, return_std=True, batch_size=len(Xt))
    m2, s2 = fitted.predict(Xt, return_std=True, batch_size=5)
    np.testing.assert_allclose(m1, m2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(s1, s2, rtol=1e-5, atol=1e-5)
    assert np.all(s2 > 0)


def test_batch_size_does_not_change_results(fitted):
    """Any batch size must agree -- a size-dependent answer means the cross-covariance
    is leaking into the per-point output."""
    Xt = _fp(31, seed=4)
    ref = fitted.predict(Xt, batch_size=len(Xt))
    for bs in (1, 2, 7, 16, 1000):
        np.testing.assert_allclose(fitted.predict(Xt, batch_size=bs), ref,
                                   rtol=1e-5, atol=1e-5,
                                   err_msg=f"batch_size={bs} disagrees")


def test_default_batch_is_bounded(fitted):
    """The default must not be 'everything' -- that is the OOM."""
    assert TanimotoGP.PREDICT_BATCH <= 8192, (
        "the default predict batch is large enough to rebuild the n^2 block that "
        "killed Stage 8")
    assert TanimotoGP.PREDICT_BATCH >= 256, "pathologically small batches will crawl"


def test_more_test_points_than_batch_size(fitted):
    """The path that actually mattered: n_test well above the batch size."""
    Xt = _fp(500, seed=5)
    mu = fitted.predict(Xt, batch_size=64)
    assert mu.shape == (500,)
    assert np.isfinite(mu).all()


def test_empty_input_is_not_a_crash(fitted):
    """Stage 8 can hand the audit an empty frame when every candidate is filtered out
    earlier; that must not raise on a concatenate of nothing."""
    mu = fitted.predict(np.zeros((0, 64), dtype=np.float32))
    assert mu.shape == (0,)
    mu, sd = fitted.predict(np.zeros((0, 64), dtype=np.float32), return_std=True)
    assert mu.shape == (0,) and sd.shape == (0,)
