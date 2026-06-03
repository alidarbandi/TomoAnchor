from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from .admm_tv_mbir import ADMMIterationMetrics, ADMMResult
from .cancel_utils import raise_if_cancelled
from .config import MBIRConfig


@dataclass
class PDHGStepSizes:
    tau: float
    sigma: float
    operator_norm: float


class PDHGTVMBIRReconstructor:
    """Low-memory TV-MBIR solver using a Chambolle-Pock/PDHG update."""

    def __init__(
        self,
        operator,
        config: MBIRConfig | None = None,
        logger: Callable[[str], None] | None = None,
        progress_callback: Callable[[ADMMIterationMetrics, np.ndarray], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self.operator = operator
        self.config = config or MBIRConfig()
        self.logger = logger
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check

    def reconstruct(self, b: np.ndarray, x0: np.ndarray | None = None) -> ADMMResult:
        cfg = self.config
        b = np.ascontiguousarray(b, dtype=np.float32)
        x = np.zeros(self.operator.volume_shape, dtype=np.float32) if x0 is None else np.ascontiguousarray(x0, dtype=np.float32)
        if cfg.positivity:
            np.maximum(x, 0.0, out=x)

        self._log("Starting low-memory PDHG TV-MBIR reconstruction.")
        self._log(
            "PDHG replaces the ADMM/CG inner solve with first-order primal-dual updates, "
            "so it keeps far fewer full-volume arrays in RAM."
        )

        steps = self._estimate_step_sizes(x.shape)
        self._log(
            f"PDHG step sizes: tau={steps.tau:.6g}, sigma={steps.sigma:.6g}, "
            f"estimated ||[A;grad]||={steps.operator_norm:.6g}"
        )

        dual_dtype = _dual_dtype(cfg.pdhg_dual_dtype)
        chunk_slices = max(1, int(cfg.pdhg_chunk_slices))
        x_bar = x.copy()
        data_dual = np.zeros(self.operator.projection_shape, dtype=np.float32)
        tv_dual = tuple(np.zeros(x.shape, dtype=dual_dtype) for _ in range(3))
        metrics: list[ADMMIterationMetrics] = []
        started = perf_counter()
        converged = False
        message = "maximum iterations reached"

        for iteration in range(1, int(cfg.max_admm_iterations) + 1):
            raise_if_cancelled(self.cancel_check, "PDHG reconstruction cancelled.")
            begin_iteration = getattr(self.operator, "begin_iteration", None)
            if callable(begin_iteration):
                begin_iteration(iteration)
            self._log(f"Starting low-memory PDHG iteration {iteration}/{cfg.max_admm_iterations}.")

            ax_bar = self.operator.forward(x_bar)
            data_dual += steps.sigma * np.asarray(ax_bar, dtype=np.float32)
            del ax_bar
            data_dual -= steps.sigma * b
            data_dual /= 1.0 + steps.sigma

            _update_tv_dual_inplace(
                tv_dual,
                x_bar,
                sigma=steps.sigma,
                radius=float(cfg.lambda_tv),
                chunk_slices=chunk_slices,
            )

            x_old = x_bar
            np.copyto(x_old, x)
            primal_gradient = self.operator.backproject(data_dual)
            _add_gradient_adjoint_from_dual_inplace(primal_gradient, tv_dual, chunk_slices=chunk_slices)
            x -= steps.tau * np.asarray(primal_gradient, dtype=np.float32)
            del primal_gradient
            if cfg.positivity:
                np.maximum(x, 0.0, out=x)

            np.subtract(x, x_old, out=x_bar)
            relative_change = _relative_norm_from_delta(x_bar, x_old)
            x_bar *= float(cfg.pdhg_theta)
            x_bar += x

            metric = self._evaluate_metrics(iteration, b, x, relative_change, perf_counter() - started)
            metrics.append(metric)
            self._log(
                f"PDHG {iteration:03d}: obj={metric.objective:.6g}, data={metric.data_fidelity:.6g}, "
                f"tv={metric.tv_term:.6g}, rel_dx={metric.relative_x_change:.3g}"
            )
            if self.progress_callback is not None:
                self.progress_callback(metric, x)
            if iteration > 1 and relative_change <= float(cfg.tolerance_relative_change):
                converged = True
                message = "converged"
                break

        return ADMMResult(np.ascontiguousarray(x, dtype=np.float32), metrics, converged, message)

    def _estimate_step_sizes(self, shape: tuple[int, int, int]) -> PDHGStepSizes:
        iterations = max(1, int(self.config.pdhg_power_iterations))
        rng = np.random.default_rng(17)
        x = rng.normal(size=shape).astype(np.float32)
        x /= max(float(np.linalg.norm(x.reshape(-1))), 1e-12)
        norm_estimate = 1.0
        for iteration in range(1, iterations + 1):
            raise_if_cancelled(self.cancel_check, "PDHG operator-norm estimation cancelled.")
            self._log(f"Estimating PDHG operator norm: power iteration {iteration}/{iterations}.")
            y = self.operator.normal(x)
            _add_gradient_laplacian_inplace(y, x)
            norm_estimate = max(float(np.linalg.norm(y.reshape(-1))), 1e-12)
            x = np.ascontiguousarray(y / norm_estimate, dtype=np.float32)
        operator_norm = float(np.sqrt(norm_estimate))
        step = float(self.config.pdhg_step_safety) / max(operator_norm, 1e-12)
        return PDHGStepSizes(tau=step, sigma=step, operator_norm=operator_norm)

    def _evaluate_metrics(
        self,
        iteration: int,
        b: np.ndarray,
        x: np.ndarray,
        relative_change: float,
        elapsed_s: float,
    ) -> ADMMIterationMetrics:
        data_residual_squared = getattr(self.operator, "data_residual_squared", None)
        if callable(data_residual_squared):
            self._log("Evaluating full-data residual/objective in projection batches.")
            data = 0.5 * float(data_residual_squared(x, b))
        else:
            residual = self.operator.forward(x) - b
            data = 0.5 * float(np.sum(np.asarray(residual, dtype=np.float64) ** 2))
        tv = float(self.config.lambda_tv) * _tv_norm_isotropic_low_memory(
            x,
            epsilon=float(self.config.tv_epsilon),
            chunk_slices=max(1, int(self.config.pdhg_chunk_slices)),
        )
        return ADMMIterationMetrics(
            iteration=iteration,
            objective=float(data + tv),
            data_fidelity=float(data),
            tv_term=float(tv),
            primal_residual=float("nan"),
            dual_residual=float("nan"),
            relative_x_change=float(relative_change),
            cg_residual=float("nan"),
            cg_iterations=0,
            elapsed_s=float(elapsed_s),
            solver="pdhg_low_memory",
        )

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)


def _dual_dtype(value: str):
    normalized = str(value or "float16").strip().lower()
    if normalized in {"float32", "single", "fp32"}:
        return np.float32
    return np.float16


def _iter_z_chunks(nz: int, chunk_slices: int):
    for z0 in range(0, int(nz), int(chunk_slices)):
        yield z0, min(int(nz), z0 + int(chunk_slices))


def _update_tv_dual_inplace(tv_dual, x: np.ndarray, sigma: float, radius: float, chunk_slices: int) -> None:
    px, py, pz = tv_dual
    if radius <= 0:
        for p in tv_dual:
            p.fill(0)
        return
    nz = x.shape[0]
    for z0, z1 in _iter_z_chunks(nz, chunk_slices):
        if x.shape[2] > 1:
            current = px[z0:z1, :, :-1].astype(np.float32, copy=True)
            current += float(sigma) * (x[z0:z1, :, 1:] - x[z0:z1, :, :-1])
            px[z0:z1, :, :-1] = current
            px[z0:z1, :, -1] = 0
        if x.shape[1] > 1:
            current = py[z0:z1, :-1, :].astype(np.float32, copy=True)
            current += float(sigma) * (x[z0:z1, 1:, :] - x[z0:z1, :-1, :])
            py[z0:z1, :-1, :] = current
            py[z0:z1, -1, :] = 0
        if z0 < nz - 1:
            stop = min(z1, nz - 1)
            current = pz[z0:stop, :, :].astype(np.float32, copy=True)
            current += float(sigma) * (x[z0 + 1 : stop + 1, :, :] - x[z0:stop, :, :])
            pz[z0:stop, :, :] = current
    pz[-1, :, :] = 0

    for z0, z1 in _iter_z_chunks(nz, chunk_slices):
        cx = px[z0:z1].astype(np.float32, copy=True)
        cy = py[z0:z1].astype(np.float32, copy=True)
        cz = pz[z0:z1].astype(np.float32, copy=True)
        magnitude = np.sqrt(cx * cx + cy * cy + cz * cz, dtype=np.float32)
        scale = np.maximum(1.0, magnitude / float(radius))
        cx /= scale
        cy /= scale
        cz /= scale
        px[z0:z1] = cx
        py[z0:z1] = cy
        pz[z0:z1] = cz


def _add_gradient_adjoint_from_dual_inplace(out: np.ndarray, tv_dual, chunk_slices: int) -> None:
    px, py, pz = tv_dual
    nz = out.shape[0]
    for z0, z1 in _iter_z_chunks(nz, chunk_slices):
        if out.shape[2] > 1:
            current = px[z0:z1, :, :-1].astype(np.float32, copy=False)
            out[z0:z1, :, :-1] -= current
            out[z0:z1, :, 1:] += current
        if out.shape[1] > 1:
            current = py[z0:z1, :-1, :].astype(np.float32, copy=False)
            out[z0:z1, :-1, :] -= current
            out[z0:z1, 1:, :] += current
        if z0 < nz - 1:
            stop = min(z1, nz - 1)
            current = pz[z0:stop, :, :].astype(np.float32, copy=False)
            out[z0:stop, :, :] -= current
            out[z0 + 1 : stop + 1, :, :] += current


def _add_gradient_laplacian_inplace(out: np.ndarray, x: np.ndarray) -> None:
    out[:, :, :-1] += x[:, :, :-1]
    out[:, :, :-1] -= x[:, :, 1:]
    out[:, :, 1:] += x[:, :, 1:]
    out[:, :, 1:] -= x[:, :, :-1]
    out[:, :-1, :] += x[:, :-1, :]
    out[:, :-1, :] -= x[:, 1:, :]
    out[:, 1:, :] += x[:, 1:, :]
    out[:, 1:, :] -= x[:, :-1, :]
    out[:-1, :, :] += x[:-1, :, :]
    out[:-1, :, :] -= x[1:, :, :]
    out[1:, :, :] += x[1:, :, :]
    out[1:, :, :] -= x[:-1, :, :]


def _tv_norm_isotropic_low_memory(x: np.ndarray, epsilon: float, chunk_slices: int) -> float:
    total = 0.0
    nz = x.shape[0]
    eps2 = float(epsilon) ** 2
    for z0, z1 in _iter_z_chunks(nz, chunk_slices):
        gx2 = np.zeros((z1 - z0, x.shape[1], x.shape[2]), dtype=np.float32)
        if x.shape[2] > 1:
            gx2[:, :, :-1] += (x[z0:z1, :, 1:] - x[z0:z1, :, :-1]) ** 2
        if x.shape[1] > 1:
            gx2[:, :-1, :] += (x[z0:z1, 1:, :] - x[z0:z1, :-1, :]) ** 2
        if z0 < nz - 1:
            stop = min(z1, nz - 1)
            gx2[: stop - z0, :, :] += (x[z0 + 1 : stop + 1, :, :] - x[z0:stop, :, :]) ** 2
        total += float(np.sum(np.sqrt(gx2 + eps2, dtype=np.float32), dtype=np.float64))
    return total


def _relative_norm_from_delta(delta: np.ndarray, reference: np.ndarray) -> float:
    delta_sum = 0.0
    reference_sum = 0.0
    for z0, z1 in _iter_z_chunks(delta.shape[0], 16):
        d = np.asarray(delta[z0:z1], dtype=np.float32)
        r = np.asarray(reference[z0:z1], dtype=np.float32)
        delta_sum += float(np.sum(d * d, dtype=np.float64))
        reference_sum += float(np.sum(r * r, dtype=np.float64))
    delta_norm = float(np.sqrt(delta_sum))
    reference_norm = float(np.sqrt(reference_sum))
    return delta_norm / max(reference_norm, 1e-12)
