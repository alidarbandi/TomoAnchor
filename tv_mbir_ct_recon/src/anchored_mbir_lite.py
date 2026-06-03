from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np

from .cancel_utils import raise_if_cancelled
from .config import MBIRLiteConfig
from .fast_types import MBIRLiteMetric, MBIRLiteResult
from .pdhg_tv_mbir import _tv_norm_isotropic_low_memory
from .subset_tv_mbir import _add_smoothed_tv_gradient_inplace, _squared_norm_low_memory


class AnchoredStreamingSubsetTVReconstructor:
    """Fast ordered-subset TV solver anchored by a conservative prior volume."""

    def __init__(
        self,
        operator,
        config: MBIRLiteConfig | None,
        x_prior: np.ndarray,
        confidence: np.ndarray,
        projection_weights: np.ndarray | None = None,
        qc_operator=None,
        qc_projections: np.ndarray | None = None,
        qc_weights: np.ndarray | None = None,
        logger: Callable[[str], None] | None = None,
        progress_callback: Callable[[MBIRLiteMetric, np.ndarray], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self.operator = operator
        self.config = config or MBIRLiteConfig()
        self.x_prior = np.ascontiguousarray(x_prior, dtype=np.float32).copy()
        self.confidence = np.ascontiguousarray(confidence, dtype=np.float32).copy()
        self.projection_weights = None if projection_weights is None else np.asarray(projection_weights, dtype=np.float32)
        self.qc_operator = qc_operator
        self.qc_projections = None if qc_projections is None else np.ascontiguousarray(qc_projections, dtype=np.float32)
        self.qc_weights = None if qc_weights is None else np.asarray(qc_weights, dtype=np.float32)
        self.logger = logger
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        if self.x_prior.shape != tuple(self.operator.volume_shape):
            raise ValueError(f"Prior shape {self.x_prior.shape} does not match operator volume shape {self.operator.volume_shape}.")
        if self.confidence.shape != self.x_prior.shape:
            raise ValueError(f"Confidence shape {self.confidence.shape} does not match prior shape {self.x_prior.shape}.")

    def reconstruct(self, b: np.ndarray, x0: np.ndarray | None = None) -> MBIRLiteResult:
        cfg = self.config
        b = np.ascontiguousarray(b, dtype=np.float32)
        x = self.x_prior.copy() if x0 is None else np.ascontiguousarray(x0, dtype=np.float32).copy()
        if x.shape != self.x_prior.shape:
            raise ValueError(f"Initial volume shape {x.shape} does not match prior shape {self.x_prior.shape}.")
        if cfg.positivity:
            np.maximum(x, 0.0, out=x)

        self._log("Starting anchored MBIR-lite reconstruction.")
        data_lipschitz = self._estimate_data_lipschitz(x.shape)
        max_c = float(np.nanmax(self.confidence)) if self.confidence.size else 0.0
        tau = float(cfg.subset_tv_step_safety) / max(data_lipschitz + float(cfg.rho_prior) * max_c, 1e-12)
        self._log(
            "Anchored MBIR-lite step size: "
            f"tau={tau:.6g}, estimated ||A||^2={data_lipschitz:.6g}, rho_prior={cfg.rho_prior:g}"
        )

        metrics: list[MBIRLiteMetric] = []
        started = perf_counter()
        chunk_slices = max(1, int(cfg.pdhg_chunk_slices))
        converged = False
        message = "maximum sweeps reached"
        for sweep in range(1, int(cfg.n_sweeps) + 1):
            raise_if_cancelled(self.cancel_check, "Anchored MBIR-lite reconstruction cancelled.")
            begin_iteration = getattr(self.operator, "begin_iteration", None)
            if callable(begin_iteration):
                begin_iteration(sweep)
            self._log(f"Starting anchored MBIR-lite sweep {sweep}/{cfg.n_sweeps}.")
            reference_square_sum = _squared_norm_low_memory(x, chunk_slices)
            update_square_sum = 0.0
            for gradient, batch_number, total_batches in self.operator.data_gradient_batches(
                x,
                b,
                scale_to_full=True,
                projection_weights=self.projection_weights if cfg.use_projection_weights else None,
            ):
                raise_if_cancelled(self.cancel_check, "Anchored MBIR-lite reconstruction cancelled.")
                _add_smoothed_tv_gradient_inplace(
                    gradient,
                    x,
                    weight=float(cfg.lambda_tv),
                    epsilon=float(cfg.tv_epsilon),
                    chunk_slices=chunk_slices,
                )
                gradient += float(cfg.rho_prior) * self.confidence * (x - self.x_prior)
                batch_tau = tau / float(max(1, total_batches))
                update_square_sum += (batch_tau * batch_tau) * _squared_norm_low_memory(gradient, chunk_slices)
                x -= batch_tau * gradient
                if cfg.positivity:
                    np.maximum(x, 0.0, out=x)
                del gradient
            relative_change = float(np.sqrt(update_square_sum)) / max(float(np.sqrt(reference_square_sum)), 1e-12)
            metric = self._evaluate_metrics(sweep, b, x, relative_change, perf_counter() - started)
            metrics.append(metric)
            self._log(
                f"MBIR-lite {sweep:03d}: obj={metric.objective_total:.6g}, data={metric.data_weighted:.6g}, "
                f"tv={metric.tv_term:.6g}, prior={metric.prior_anchor_term:.6g}, rel_dx={metric.relative_change:.3g}"
            )
            if self.progress_callback is not None:
                self.progress_callback(metric, x)
            if sweep > 1 and relative_change <= 1e-5:
                converged = True
                message = "converged"
                break
        return MBIRLiteResult(np.ascontiguousarray(x, dtype=np.float32), metrics, converged, message)

    def _estimate_data_lipschitz(self, shape: tuple[int, int, int]) -> float:
        iterations = max(1, int(self.config.subset_tv_power_iterations))
        rng = np.random.default_rng(23)
        x = rng.normal(size=shape).astype(np.float32)
        x /= max(float(np.linalg.norm(x.reshape(-1))), 1e-12)
        estimate = 1.0
        for iteration in range(1, iterations + 1):
            raise_if_cancelled(self.cancel_check, "MBIR-lite data-step estimation cancelled.")
            self._log(f"Estimating MBIR-lite data step: power iteration {iteration}/{iterations}.")
            y = self.operator.normal(x)
            estimate = max(float(np.linalg.norm(y.reshape(-1))), 1e-12)
            x = np.ascontiguousarray(y / estimate, dtype=np.float32)
        if self.projection_weights is not None and self.config.use_projection_weights:
            max_weight = float(np.nanmax(np.asarray(self.projection_weights, dtype=np.float32)))
            estimate *= max(max_weight * max_weight, 1e-6)
        return estimate

    def _evaluate_metrics(
        self,
        iteration: int,
        b: np.ndarray,
        x: np.ndarray,
        relative_change: float,
        elapsed_s: float,
    ) -> MBIRLiteMetric:
        data_residual_squared = getattr(self.operator, "data_residual_squared", None)
        if callable(data_residual_squared):
            data = 0.5 * float(
                data_residual_squared(
                    x,
                    b,
                    projection_weights=self.projection_weights if self.config.use_projection_weights else None,
                )
            )
        else:
            residual = self.operator.forward(x) - b
            if self.projection_weights is not None and self.config.use_projection_weights:
                residual = residual * self.projection_weights[:, None, None]
            data = 0.5 * float(np.sum(np.asarray(residual, dtype=np.float64) ** 2))
        tv = float(self.config.lambda_tv) * _tv_norm_isotropic_low_memory(
            x,
            epsilon=float(self.config.tv_epsilon),
            chunk_slices=max(1, int(self.config.pdhg_chunk_slices)),
        )
        delta = x - self.x_prior
        prior = 0.5 * float(self.config.rho_prior) * float(np.sum(self.confidence * delta * delta, dtype=np.float64))
        qc = self._qc_residual(x)
        return MBIRLiteMetric(
            iteration=int(iteration),
            objective_total=float(data + tv + prior),
            data_weighted=float(data),
            tv_term=float(tv),
            prior_anchor_term=float(prior),
            qc_anchor_residual=float(qc),
            relative_change=float(relative_change),
            elapsed_s=float(elapsed_s),
        )

    def _qc_residual(self, x: np.ndarray) -> float:
        if self.qc_operator is None or self.qc_projections is None or not self.qc_projections.size:
            return float("nan")
        pred = self.qc_operator.forward(x)
        residual = pred - self.qc_projections
        denom_source = self.qc_projections
        if self.qc_weights is not None:
            weights = np.asarray(self.qc_weights, dtype=np.float32)
            residual = residual * weights[:, None, None]
            denom_source = denom_source * weights[:, None, None]
        return float(np.sum(np.asarray(residual, dtype=np.float64) ** 2) / max(float(np.sum(np.asarray(denom_source, dtype=np.float64) ** 2)), 1e-12))

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)
