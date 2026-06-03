from __future__ import annotations

import numpy as np

from src.tv_ops import gradient_3d, gradient_adjoint_3d, inner_product


def test_explicit_gradient_adjoint_identity() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 4, 5)).astype(np.float32)
    p = tuple(rng.normal(size=x.shape).astype(np.float32) for _ in range(3))
    gx = gradient_3d(x)
    lhs = sum(inner_product(a, b) for a, b in zip(gx, p))
    rhs = inner_product(x, gradient_adjoint_3d(*p))
    assert np.isclose(lhs, rhs, rtol=1e-6, atol=1e-6)
