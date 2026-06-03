from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np

from .admm_tv_mbir import ADMMIterationMetrics, ADMMResult
from .cancel_utils import raise_if_cancelled
from .config import MBIRConfig
from .pdhg_tv_mbir import _iter_z_chunks, _tv_norm_isotropic_low_memory


class StreamingSubsetTVReconstructor:
    """Very low-memory ordered-subset gradient solver with smoothed TV updates."""

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

        self._log("Starting streaming subset-TV reconstruction.")
        self._log(
            "This mode streams one projection batch, applies an immediate image update, and discards the batch gradient. "
            "It is less exact per iteration than ADMM/PDHG, but has the smallest RAM footprint."
        )
        data_lipschitz = self._estimate_data_lipschitz(x.shape)
        tau = float(cfg.subset_tv_step_safety) / max(data_lipschitz, 1e-12)
        self._log(f"Streaming subset-TV step size: tau={tau:.6g}, estimated ||A||^2={data_lipschitz:.6g}")

        metrics: list[ADMMIterationMetrics] = []
        started = perf_counter()
        converged = False
        message = "maximum iterations reached"
        chunk_slices = max(1, int(cfg.pdhg_chunk_slices))

        for iteration in range(1, int(cfg.max_admm_iterations) + 1):
            raise_if_cancelled(self.cancel_check, "Streaming subset-TV reconstruction cancelled.")
            begin_iteration = getattr(self.operator, "begin_iteration", None)
            if callable(begin_iteration):
                begin_iteration(iteration)
            self._log(f"Starting streaming subset-TV iteration {iteration}/{cfg.max_admm_iterations}.")
            update_square_sum = 0.0
            reference_square_sum = _squared_norm_low_memory(x, chunk_slices)
            for gradient, batch_number, total_batches in self.operator.data_gradient_batches(x, b, scale_to_full=True):
                raise_if_cancelled(self.cancel_check, "Streaming subset-TV reconstruction cancelled.")
                _add_smoothed_tv_gradient_inplace(
                    gradient,
                    x,
                    weight=float(cfg.lambda_tv),
                    epsilon=float(cfg.tv_epsilon),
                    chunk_slices=chunk_slices,
                )
                batch_tau = tau / float(max(1, total_batches))
                update_square_sum += (batch_tau * batch_tau) * _squared_norm_low_memory(gradient, chunk_slices)
                x -= batch_tau * gradient
                if cfg.positivity:
                    np.maximum(x, 0.0, out=x)
                del gradient
            relative_change = float(np.sqrt(update_square_sum)) / max(float(np.sqrt(reference_square_sum)), 1e-12)
            metric = self._evaluate_metrics(iteration, b, x, relative_change, perf_counter() - started)
            metrics.append(metric)
            self._log(
                f"Subset-TV {iteration:03d}: obj={metric.objective:.6g}, data={metric.data_fidelity:.6g}, "
                f"tv={metric.tv_term:.6g}, rel_dx={metric.relative_x_change:.3g}"
            )
            if self.progress_callback is not None:
                self.progress_callback(metric, x)
            if iteration > 1 and relative_change <= float(cfg.tolerance_relative_change):
                converged = True
                message = "converged"
                break

        return ADMMResult(np.ascontiguousarray(x, dtype=np.float32), metrics, converged, message)

    def _estimate_data_lipschitz(self, shape: tuple[int, int, int]) -> float:
        iterations = max(1, int(self.config.subset_tv_power_iterations))
        rng = np.random.default_rng(23)
        x = rng.normal(size=shape).astype(np.float32)
        x /= max(float(np.linalg.norm(x.reshape(-1))), 1e-12)
        estimate = 1.0
        for iteration in range(1, iterations + 1):
            raise_if_cancelled(self.cancel_check, "Subset-TV data-step estimation cancelled.")
            self._log(f"Estimating subset-TV data step: power iteration {iteration}/{iterations}.")
            y = self.operator.normal(x)
            estimate = max(float(np.linalg.norm(y.reshape(-1))), 1e-12)
            x = np.ascontiguousarray(y / estimate, dtype=np.float32)
        return estimate

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
            solver="streaming_subset_tv",
        )

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)


def _add_smoothed_tv_gradient_inplace(
    out: np.ndarray,
    x: np.ndarray,
    weight: float,
    epsilon: float,
    chunk_slices: int,
) -> None:
    if weight <= 0:
        return
    nz = x.shape[0]
    eps2 = float(epsilon) ** 2
    for z0, z1 in _iter_z_chunks(nz, chunk_slices):
        gx = np.zeros((z1 - z0, x.shape[1], x.shape[2]), dtype=np.float32)
        gy = np.zeros_like(gx)
        gz = np.zeros_like(gx)
        if x.shape[2] > 1:
            gx[:, :, :-1] = x[z0:z1, :, 1:] - x[z0:z1, :, :-1]
        if x.shape[1] > 1:
            gy[:, :-1, :] = x[z0:z1, 1:, :] - x[z0:z1, :-1, :]
        if z0 < nz - 1:
            stop = min(z1, nz - 1)
            gz[: stop - z0, :, :] = x[z0 + 1 : stop + 1, :, :] - x[z0:stop, :, :]
        magnitude = np.sqrt(gx * gx + gy * gy + gz * gz + eps2, dtype=np.float32)
        gx /= magnitude
        gy /= magnitude
        gz /= magnitude
        _add_gradient_adjoint_chunk_inplace(out, gx, gy, gz, z0, z1, float(weight))


def _add_gradient_adjoint_chunk_inplace(
    out: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    z0: int,
    z1: int,
    weight: float,
) -> None:
    if out.shape[2] > 1:
        current = weight * gx[:, :, :-1]
        out[z0:z1, :, :-1] -= current
        out[z0:z1, :, 1:] += current
    if out.shape[1] > 1:
        current = weight * gy[:, :-1, :]
        out[z0:z1, :-1, :] -= current
        out[z0:z1, 1:, :] += current
    if z0 < out.shape[0] - 1:
        stop = min(z1, out.shape[0] - 1)
        current = weight * gz[: stop - z0, :, :]
        out[z0:stop, :, :] -= current
        out[z0 + 1 : stop + 1, :, :] += current


def _squared_norm_low_memory(x: np.ndarray, chunk_slices: int) -> float:
    total = 0.0
    for z0, z1 in _iter_z_chunks(x.shape[0], chunk_slices):
        chunk = np.asarray(x[z0:z1], dtype=np.float32)
        total += float(np.sum(chunk * chunk, dtype=np.float64))
    return total
