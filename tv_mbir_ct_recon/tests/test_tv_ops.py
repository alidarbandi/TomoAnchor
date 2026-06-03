from __future__ import annotations

import numpy as np

from src.tv_ops import adjointness_error, gradient_3d, isotropic_shrinkage_3d, tv_norm_isotropic_3d


def test_gradient_adjointness() -> None:
    assert adjointness_error((4, 5, 6), seed=123) < 1e-6


def test_isotropic_shrinkage_zeroes_small_vectors() -> None:
    z = np.full((2, 2, 2), 0.1, dtype=np.float32)
    sx, sy, sz = isotropic_shrinkage_3d(z, z, z, threshold=1.0)
    assert np.allclose(sx, 0.0)
    assert np.allclose(sy, 0.0)
    assert np.allclose(sz, 0.0)


def test_tv_norm_positive() -> None:
    x = np.zeros((3, 3, 3), dtype=np.float32)
    x[1, 1, 1] = 1.0
    assert tv_norm_isotropic_3d(x) > 0.0
