from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .cancel_utils import raise_if_cancelled
from .tv_ops import get_array_module


@dataclass
class CGResult:
    x: object
    residual_norm: float
    relative_residual: float
    iterations: int
    converged: bool
    residual_history: list[float]


def conjugate_gradient(
    matvec: Callable[[object], object],
    rhs: object,
    x0: object | None = None,
    max_iterations: int = 20,
    tolerance: float = 1e-5,
    callback: Callable[[int, float], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> CGResult:
    xp = get_array_module(rhs)
    b = xp.asarray(rhs)
    x = xp.zeros_like(b) if x0 is None else xp.asarray(x0).copy()
    r = b - matvec(x)
    p = r.copy()
    rsold = _dot(r, r)
    bnorm = max(_sqrt(_dot(b, b)), 1e-30)
    history = [_sqrt(rsold)]
    converged = history[-1] / bnorm <= tolerance
    if converged:
        return CGResult(x, history[-1], history[-1] / bnorm, 0, True, history)

    iterations = 0
    for iteration in range(1, max_iterations + 1):
        raise_if_cancelled(cancel_check, "ADMM conjugate-gradient solve cancelled.")
        ap = matvec(p)
        denom = _dot(p, ap)
        if abs(denom) <= 1e-30:
            break
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * ap
        rsnew = _dot(r, r)
        residual = _sqrt(rsnew)
        history.append(residual)
        iterations = iteration
        if callback is not None:
            callback(iteration, residual / bnorm)
        if residual / bnorm <= tolerance:
            converged = True
            rsold = rsnew
            break
        beta = rsnew / max(rsold, 1e-30)
        p = r + beta * p
        rsold = rsnew
    residual_norm = history[-1] if history else _sqrt(rsold)
    return CGResult(x, residual_norm, residual_norm / bnorm, iterations, converged, history)


def _dot(a, b) -> float:
    xp = get_array_module(a)
    value = xp.vdot(a, b).real
    get = getattr(value, "get", None)
    if callable(get):  # pragma: no cover
        value = get()
    return float(value)


def _sqrt(value: float) -> float:
    return float(np.sqrt(max(value, 0.0)))
