from __future__ import annotations

import numpy as np

from src.anchored_mbir_lite import AnchoredStreamingSubsetTVReconstructor
from src.config import MBIRLiteConfig


class IdentityOperator:
    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.volume_shape = shape
        self.ordered_subset_count = 1

    def forward(self, x):
        return np.asarray(x, dtype=np.float32)

    def normal(self, x):
        return np.asarray(x, dtype=np.float32)

    def data_residual_squared(self, x, b, projection_weights=None):
        residual = np.asarray(x, dtype=np.float32) - np.asarray(b, dtype=np.float32)
        if projection_weights is not None:
            residual = residual * np.asarray(projection_weights, dtype=np.float32)[:, None, None]
        return float(np.sum(residual * residual))

    def data_gradient_batches(self, x, b, scale_to_full=True, projection_weights=None):
        residual = np.asarray(x, dtype=np.float32) - np.asarray(b, dtype=np.float32)
        if projection_weights is not None:
            weights = np.asarray(projection_weights, dtype=np.float32)
            residual = residual * weights[:, None, None] * weights[:, None, None]
        yield residual, 1, 1

    def begin_iteration(self, iteration: int) -> None:
        return None


def test_anchored_mbir_lite_moves_toward_data_prior_compromise() -> None:
    shape = (3, 3, 3)
    cfg = MBIRLiteConfig(n_sweeps=20, lambda_tv=0.0, rho_prior=1.0, positivity=False, subset_tv_power_iterations=1)
    b = np.ones(shape, dtype=np.float32)
    prior = np.zeros(shape, dtype=np.float32)
    confidence = np.ones(shape, dtype=np.float32)

    result = AnchoredStreamingSubsetTVReconstructor(
        IdentityOperator(shape),
        cfg,
        prior,
        confidence,
        projection_weights=np.ones(shape[0], dtype=np.float32),
    ).reconstruct(b, x0=prior)

    assert result.volume.mean() > 0.3
    assert result.volume.mean() < 0.7
    assert result.metrics[-1].prior_anchor_term > 0.0


def test_zero_confidence_removes_prior_effect() -> None:
    shape = (3, 3, 3)
    cfg = MBIRLiteConfig(n_sweeps=8, lambda_tv=0.0, rho_prior=100.0, positivity=False, subset_tv_power_iterations=1)
    b = np.ones(shape, dtype=np.float32)
    prior = np.zeros(shape, dtype=np.float32)
    confidence = np.zeros(shape, dtype=np.float32)

    result = AnchoredStreamingSubsetTVReconstructor(
        IdentityOperator(shape),
        cfg,
        prior,
        confidence,
    ).reconstruct(b, x0=prior)

    assert result.volume.mean() > 0.7
    assert np.isclose(result.metrics[-1].prior_anchor_term, 0.0)
