from __future__ import annotations

import numpy as np

from src.cg_solver import conjugate_gradient


def test_conjugate_gradient_diagonal_system() -> None:
    diag = np.array([2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    rhs = diag * np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    result = conjugate_gradient(lambda x: diag * x, rhs, max_iterations=20, tolerance=1e-8)
    assert result.converged
    assert np.allclose(result.x, np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32), atol=1e-5)
