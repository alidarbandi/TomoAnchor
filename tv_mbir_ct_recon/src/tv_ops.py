from __future__ import annotations

from typing import Any

import numpy as np


def get_array_module(array: Any):
    module = type(array).__module__.split(".")[0]
    if module == "cupy":  # pragma: no cover - depends on CuPy
        import cupy as cp

        return cp
    return np


def gradient_3d(x):
    """Forward differences for a `(z, y, x)` volume.

    Returns `(gx, gy, gz)` where `gx` is the column/x gradient, `gy` is row/y,
    and `gz` is slice/z. Boundary forward differences are zero.
    """
    xp = get_array_module(x)
    x = xp.asarray(x)
    gx = xp.zeros_like(x)
    gy = xp.zeros_like(x)
    gz = xp.zeros_like(x)
    gx[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
    gy[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
    gz[:-1, :, :] = x[1:, :, :] - x[:-1, :, :]
    return gx, gy, gz


def gradient_adjoint_3d(gx, gy, gz):
    """Adjoint of `gradient_3d`.

    This returns `G^T p`, so `<Gx, p> = <x, G^T p>`. It is the negative of the
    usual continuous divergence sign convention.
    """
    xp = get_array_module(gx)
    out = xp.zeros_like(gx)
    out[:, :, :-1] -= gx[:, :, :-1]
    out[:, :, 1:] += gx[:, :, :-1]
    out[:, :-1, :] -= gy[:, :-1, :]
    out[:, 1:, :] += gy[:, :-1, :]
    out[:-1, :, :] -= gz[:-1, :, :]
    out[1:, :, :] += gz[:-1, :, :]
    return out


def divergence_3d(gx, gy, gz):
    """Classical divergence sign: `div = -gradient_adjoint_3d`."""
    return -gradient_adjoint_3d(gx, gy, gz)


def gradient_laplacian_3d(x):
    return gradient_adjoint_3d(*gradient_3d(x))


def tv_norm_isotropic_3d(x, epsilon: float = 1e-8) -> float:
    xp = get_array_module(x)
    gx, gy, gz = gradient_3d(x)
    value = xp.sum(xp.sqrt(gx * gx + gy * gy + gz * gz + float(epsilon) ** 2))
    return _as_float(value)


def isotropic_shrinkage_3d(zx, zy, zz, threshold: float, epsilon: float = 1e-12):
    xp = get_array_module(zx)
    magnitude = xp.sqrt(zx * zx + zy * zy + zz * zz)
    scale = xp.maximum(1.0 - float(threshold) / xp.maximum(magnitude, float(epsilon)), 0.0)
    return scale * zx, scale * zy, scale * zz


def squared_norm(arrays) -> float:
    total = 0.0
    for array in arrays:
        xp = get_array_module(array)
        total += _as_float(xp.sum(xp.asarray(array) * xp.asarray(array)))
    return total


def l2_norm(array) -> float:
    xp = get_array_module(array)
    return _as_float(xp.sqrt(xp.sum(xp.asarray(array) * xp.asarray(array))))


def inner_product(a, b) -> float:
    xp = get_array_module(a)
    return _as_float(xp.vdot(a, b).real)


def adjointness_error(shape: tuple[int, int, int] = (5, 6, 7), seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=shape).astype(np.float32)
    p = tuple(rng.normal(size=shape).astype(np.float32) for _ in range(3))
    gx = gradient_3d(x)
    lhs = sum(inner_product(g, q) for g, q in zip(gx, p))
    rhs = inner_product(x, gradient_adjoint_3d(*p))
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)


def _as_float(value) -> float:
    get = getattr(value, "get", None)
    if callable(get):  # pragma: no cover - CuPy scalar
        value = get()
    return float(value)
