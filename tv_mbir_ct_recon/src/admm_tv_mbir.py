from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from .cancel_utils import raise_if_cancelled
from .cg_solver import CGResult, conjugate_gradient
from .config import MBIRConfig
from .tv_ops import (
    gradient_3d,
    gradient_adjoint_3d,
    isotropic_shrinkage_3d,
    l2_norm,
    squared_norm,
    tv_norm_isotropic_3d,
)


@dataclass
class ADMMIterationMetrics:
    iteration: int
    objective: float
    data_fidelity: float
    tv_term: float
    primal_residual: float
    dual_residual: float
    relative_x_change: float
    cg_residual: float
    cg_iterations: int
    elapsed_s: float
    solver: str = "admm"


@dataclass
class ADMMResult:
    volume: np.ndarray
    metrics: list[ADMMIterationMetrics]
    converged: bool
    message: str


class ADMMTVMBIRReconstructor:
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
        if cfg.rho <= 0:
            raise ValueError("ADMM rho must be positive.")
        if cfg.lambda_tv < 0:
            raise ValueError("lambda_tv must be non-negative.")
        b = np.ascontiguousarray(b, dtype=np.float32)
        operator_mode = getattr(self.operator, "memory_mode", "full_gpu")
        projection_batch_size = getattr(self.operator, "projection_batch_size", "n/a")
        ordered_subset_count = getattr(self.operator, "ordered_subset_count", 1)
        uses_ordered_subsets = operator_mode == "ordered_subsets" and getattr(self.operator, "ordered_subset_count", 1) > 1
        self._log(
            "MBIR projection operator: "
            f"mode={operator_mode}, batch_size={projection_batch_size}, ordered_subsets={ordered_subset_count}"
        )
        atb = None
        if not uses_ordered_subsets:
            if operator_mode == "projection_streaming":
                self._log("Computing full-data A^T b in projection batches for the MBIR right-hand side.")
            else:
                self._log("Computing full-data A^T b backprojection for the MBIR right-hand side.")
            try:
                atb = self.operator.backproject(b)
            except Exception as exc:
                raise RuntimeError(
                    "TIGRE failed while computing A^T b for MBIR. This often means the requested detector/projection "
                    "stack and reconstruction volume do not fit in available GPU and system memory. Try projection "
                    "streaming, reducing volume voxels, using fewer projections for a debug run, closing memory-heavy "
                    "applications, or selecting a machine with more GPU/RAM capacity."
                ) from exc
        else:
            self._log(
                "Ordered-subset MBIR enabled. Each ADMM iteration uses one interleaved projection subset "
                "scaled to approximate the full data term; full-data residual is still evaluated for progress."
            )
        x = np.zeros(self.operator.volume_shape, dtype=np.float32) if x0 is None else np.ascontiguousarray(x0, dtype=np.float32)
        if cfg.positivity:
            x = np.maximum(x, 0.0).astype(np.float32, copy=False)

        gx, gy, gz = gradient_3d(x)
        d = tuple(np.zeros_like(g, dtype=np.float32) for g in (gx, gy, gz))
        u = tuple(np.zeros_like(g, dtype=np.float32) for g in (gx, gy, gz))
        metrics: list[ADMMIterationMetrics] = []
        started = perf_counter()
        converged = False
        message = "maximum iterations reached"

        self._log("Starting ADMM TV-MBIR reconstruction.")
        self._log(
            f"lambda_tv={cfg.lambda_tv:g}, rho={cfg.rho:g}, "
            f"ADMM iterations={cfg.max_admm_iterations}, CG iterations={cfg.inner_cg_iterations}"
        )
        for iteration in range(1, cfg.max_admm_iterations + 1):
            raise_if_cancelled(self.cancel_check, "ADMM reconstruction cancelled.")
            begin_iteration = getattr(self.operator, "begin_iteration", None)
            if callable(begin_iteration):
                begin_iteration(iteration)
            projection_indices = getattr(self.operator, "current_projection_indices", None)
            if callable(projection_indices):
                active_indices = projection_indices()
                if operator_mode in {"projection_streaming", "ordered_subsets"}:
                    self._log(
                        f"ADMM iteration {iteration}: using {len(active_indices)} projection views "
                        f"in batches of {projection_batch_size}."
                    )
            self._log(f"Starting ADMM iteration {iteration}/{cfg.max_admm_iterations}.")
            x_previous = x.copy()
            d_previous = tuple(comp.copy() for comp in d)
            if uses_ordered_subsets:
                try:
                    atb_iter = self.operator.backproject(b)
                except Exception as exc:
                    raise RuntimeError(
                        "TIGRE failed while computing the ordered-subset A^T b term. Reduce projection batch size "
                        "or volume size, close memory-heavy applications, or select projection_streaming/full_gpu "
                        "mode for diagnosis."
                    ) from exc
            else:
                atb_iter = atb
            rhs = atb_iter + cfg.rho * gradient_adjoint_3d(d[0] - u[0], d[1] - u[1], d[2] - u[2])

            def matvec(v):
                vg = gradient_3d(v)
                return self.operator.normal(v) + cfg.rho * gradient_adjoint_3d(*vg)

            def cg_callback(cg_iteration: int, cg_relative_residual: float) -> None:
                self._log(
                    f"ADMM {iteration:03d} CG {int(cg_iteration)}/{cfg.inner_cg_iterations}: "
                    f"residual={float(cg_relative_residual):.3e}"
                )
                self._emit_progress(
                    {
                        "kind": "cg_progress",
                        "solver": "admm",
                        "admm_iteration": int(iteration),
                        "max_admm_iterations": int(cfg.max_admm_iterations),
                        "cg_iteration": int(cg_iteration),
                        "cg_max_iterations": int(cfg.inner_cg_iterations),
                        "cg_residual": float(cg_relative_residual),
                        "elapsed_s": float(perf_counter() - started),
                    }
                )

            cg: CGResult = conjugate_gradient(
                matvec,
                rhs,
                x0=x,
                max_iterations=cfg.inner_cg_iterations,
                tolerance=cfg.cg_tolerance,
                callback=cg_callback,
                cancel_check=self.cancel_check,
            )
            x = np.ascontiguousarray(cg.x, dtype=np.float32)
            if cfg.positivity:
                x = np.maximum(x, 0.0).astype(np.float32, copy=False)

            gx, gy, gz = gradient_3d(x)
            z = (gx + u[0], gy + u[1], gz + u[2])
            d = isotropic_shrinkage_3d(z[0], z[1], z[2], cfg.lambda_tv / cfg.rho)
            u = (u[0] + gx - d[0], u[1] + gy - d[1], u[2] + gz - d[2])

            iter_metrics = self._evaluate_metrics(
                iteration,
                b,
                x,
                x_previous,
                d,
                d_previous,
                cg,
                perf_counter() - started,
            )
            metrics.append(iter_metrics)
            self._log(
                f"ADMM {iteration:03d}: obj={iter_metrics.objective:.6g}, "
                f"data={iter_metrics.data_fidelity:.6g}, tv={iter_metrics.tv_term:.6g}, "
                f"r={iter_metrics.primal_residual:.3g}, s={iter_metrics.dual_residual:.3g}, "
                f"rel_dx={iter_metrics.relative_x_change:.3g}, "
                f"cg_iter={int(iter_metrics.cg_iterations)}, cg={iter_metrics.cg_residual:.3g}"
            )
            self._emit_progress(iter_metrics, x)
            if (
                iter_metrics.primal_residual <= cfg.tolerance_primal
                and iter_metrics.dual_residual <= cfg.tolerance_dual
                and iter_metrics.relative_x_change <= cfg.tolerance_relative_change
            ):
                converged = True
                message = "converged"
                break

        return ADMMResult(np.ascontiguousarray(x, dtype=np.float32), metrics, converged, message)

    def _evaluate_metrics(
        self,
        iteration: int,
        b: np.ndarray,
        x: np.ndarray,
        x_previous: np.ndarray,
        d,
        d_previous,
        cg: CGResult,
        elapsed_s: float,
    ) -> ADMMIterationMetrics:
        cfg = self.config
        data_residual_squared = getattr(self.operator, "data_residual_squared", None)
        if callable(data_residual_squared):
            if getattr(self.operator, "memory_mode", "full_gpu") in {"projection_streaming", "ordered_subsets"}:
                self._log("Evaluating full-data residual/objective in projection batches.")
            data = 0.5 * float(data_residual_squared(x, b))
        else:
            residual = self.operator.forward(x) - b
            data = 0.5 * squared_norm((residual,))
        tv = cfg.lambda_tv * tv_norm_isotropic_3d(x, cfg.tv_epsilon)
        gx, gy, gz = gradient_3d(x)
        primal = np.sqrt(squared_norm((gx - d[0], gy - d[1], gz - d[2])))
        dual_vec = gradient_adjoint_3d(d[0] - d_previous[0], d[1] - d_previous[1], d[2] - d_previous[2])
        dual = cfg.rho * l2_norm(dual_vec)
        rel_dx = l2_norm(x - x_previous) / max(l2_norm(x_previous), 1e-12)
        return ADMMIterationMetrics(
            iteration=iteration,
            objective=float(data + tv),
            data_fidelity=float(data),
            tv_term=float(tv),
            primal_residual=float(primal),
            dual_residual=float(dual),
            relative_x_change=float(rel_dx),
            cg_residual=float(cg.relative_residual),
            cg_iterations=int(cg.iterations),
            elapsed_s=float(elapsed_s),
            solver="admm",
        )

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)

    def _emit_progress(self, payload: object, volume: np.ndarray | None = None) -> None:
        if self.progress_callback is not None:
            self.progress_callback(payload, volume)
