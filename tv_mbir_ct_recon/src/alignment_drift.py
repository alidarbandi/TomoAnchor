from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, shift as ndi_shift

from .cancel_utils import raise_if_cancelled
from .cpu_utils import bounded_thread_map, default_cpu_parallel_plan


SHIFT_MODES: tuple[tuple[str, str, int, int], ...] = (
    ("A", "Mode A: row = +y_shift, col = +x_shift", 1, 1),
    ("B", "Mode B: row = -y_shift, col = +x_shift", 1, -1),
    ("C", "Mode C: row = +y_shift, col = -x_shift", -1, 1),
    ("D", "Mode D: row = -y_shift, col = -x_shift", -1, -1),
)


@dataclass
class DriftTable:
    x_shift_px: np.ndarray
    y_shift_px: np.ndarray
    source: str = ""


def apply_projection_shifts(
    stack: np.ndarray,
    x_shift_px: Sequence[float],
    y_shift_px: Sequence[float],
    x_sign: int = 1,
    y_sign: int = 1,
    interpolation_order: int = 1,
    mode: str = "nearest",
    logger: Callable[[str], None] | object | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    stack = np.asarray(stack, dtype=np.float32)
    x_shift_px = np.asarray(x_shift_px, dtype=np.float32)
    y_shift_px = np.asarray(y_shift_px, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"Stack must have shape (num_proj, rows, cols), got {stack.shape}")
    if len(x_shift_px) != stack.shape[0] or len(y_shift_px) != stack.shape[0]:
        raise ValueError("Shift arrays must match number of projections.")
    if x_sign not in (-1, 1) or y_sign not in (-1, 1):
        raise ValueError("Shift signs must be +1 or -1.")
    raise_if_cancelled(cancel_check, "Projection drift correction cancelled.")
    shifted = np.empty_like(stack, dtype=np.float32)
    plan = default_cpu_parallel_plan(stack.shape[1:])
    use_parallel = stack.shape[0] >= 4 and plan.compute_workers > 1
    _emit_log(logger, f"Projection drift-shift mode: {plan.compute_workers} workers" if use_parallel else "Projection drift-shift mode: serial")

    def _shift_one(index: int) -> tuple[int, np.ndarray, float, float]:
        raise_if_cancelled(cancel_check, "Projection drift correction cancelled.")
        row_shift = float(y_sign * y_shift_px[index])
        col_shift = float(x_sign * x_shift_px[index])
        image = ndi_shift(
            stack[index],
            shift=(row_shift, col_shift),
            order=interpolation_order,
            mode=mode,
            prefilter=False,
        ).astype(np.float32, copy=False)
        return index, image, row_shift, col_shift

    if use_parallel:
        completed = 0
        for _, (index, image, row_shift, col_shift) in bounded_thread_map(
            _shift_one,
            list(range(stack.shape[0])),
            max_workers=plan.compute_workers,
            cancel_check=cancel_check,
        ):
            raise_if_cancelled(cancel_check, "Projection drift correction cancelled.")
            shifted[index] = image
            completed += 1
            if completed <= 5 or completed == stack.shape[0]:
                _emit_log(
                    logger,
                    f"Projection {index}: applied row_shift={row_shift:.4f}, col_shift={col_shift:.4f}",
                )
    else:
        for index in range(stack.shape[0]):
            raise_if_cancelled(cancel_check, "Projection drift correction cancelled.")
            index, image, row_shift, col_shift = _shift_one(index)
            shifted[index] = image
            if index < 5 or index == stack.shape[0] - 1:
                _emit_log(
                    logger,
                    f"Projection {index}: applied row_shift={row_shift:.4f}, col_shift={col_shift:.4f}",
                )
    return np.ascontiguousarray(shifted, dtype=np.float32)


def effective_shift_signs(x_sign: int, y_sign: int, interpretation_sign: int) -> tuple[int, int]:
    if x_sign not in (-1, 1) or y_sign not in (-1, 1) or interpretation_sign not in (-1, 1):
        raise ValueError("Shift signs must be +1 or -1.")
    return interpretation_sign * x_sign, interpretation_sign * y_sign


def load_shift_csv(path: str | Path, x_column: str = "x_shift_px", y_column: str = "y_shift_px") -> DriftTable:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    for column in (x_column, y_column):
        if column not in frame.columns:
            raise ValueError(f"Missing shift column '{column}' in {csv_path}")
    x = pd.to_numeric(frame[x_column], errors="coerce")
    y = pd.to_numeric(frame[y_column], errors="coerce")
    if x.isna().any() or y.isna().any():
        raise ValueError(f"Shift CSV contains non-numeric values: {csv_path}")
    return DriftTable(x.astype(np.float32).to_numpy(), y.astype(np.float32).to_numpy(), str(csv_path))


def save_shift_csv(path: str | Path, x_shift_px: Sequence[float], y_shift_px: Sequence[float]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"x_shift_px": np.asarray(x_shift_px), "y_shift_px": np.asarray(y_shift_px)})
    frame.to_csv(output, index=False)
    return output


def estimate_projection_drift_phase_correlation(
    stack: np.ndarray,
    reference_index: int = 0,
    upsample_factor: int = 10,
    smooth_window: int = 1,
    logger: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> DriftTable:
    try:
        from skimage.registration import phase_cross_correlation
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("scikit-image is required for phase-correlation drift estimation.") from exc
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"Stack must be 3-D, got {stack.shape}")
    raise_if_cancelled(cancel_check, "Projection drift estimation cancelled.")
    reference_index = int(np.clip(reference_index, 0, stack.shape[0] - 1))
    reference = stack[reference_index]
    y = np.zeros(stack.shape[0], dtype=np.float32)
    x = np.zeros(stack.shape[0], dtype=np.float32)
    plan = default_cpu_parallel_plan(stack.shape[1:])
    use_parallel = stack.shape[0] >= 4 and plan.compute_workers > 1
    if logger is not None:
        logger(f"Phase-correlation drift estimation mode: {plan.compute_workers} workers" if use_parallel else "Phase-correlation drift estimation mode: serial")

    def _estimate_one(index: int) -> tuple[int, float, float]:
        raise_if_cancelled(cancel_check, "Projection drift estimation cancelled.")
        image = stack[index]
        shift, _, _ = phase_cross_correlation(reference, image, upsample_factor=upsample_factor)
        return index, float(shift[0]), float(shift[1])

    if use_parallel:
        completed = 0
        for _, (index, row_shift, col_shift) in bounded_thread_map(
            _estimate_one,
            list(range(stack.shape[0])),
            max_workers=plan.compute_workers,
            cancel_check=cancel_check,
        ):
            raise_if_cancelled(cancel_check, "Projection drift estimation cancelled.")
            y[index] = row_shift
            x[index] = col_shift
            completed += 1
            if logger is not None and (completed <= 5 or completed == stack.shape[0]):
                logger(f"Estimated drift projection {index}: x={x[index]:.4f}, y={y[index]:.4f} px")
    else:
        for index in range(stack.shape[0]):
            raise_if_cancelled(cancel_check, "Projection drift estimation cancelled.")
            index, row_shift, col_shift = _estimate_one(index)
            y[index] = row_shift
            x[index] = col_shift
            if logger is not None and (index < 5 or index == stack.shape[0] - 1):
                logger(f"Estimated drift projection {index}: x={x[index]:.4f}, y={y[index]:.4f} px")
    if smooth_window and smooth_window > 1:
        size = int(smooth_window)
        if size % 2 == 0:
            size += 1
        x = median_filter(x, size=size).astype(np.float32)
        y = median_filter(y, size=size).astype(np.float32)
    return DriftTable(x, y, "phase_correlation")


def build_shift_sanity_report(
    x_shift_px: np.ndarray | None,
    y_shift_px: np.ndarray | None,
    enabled: bool,
    shift_stage: str,
    x_sign: int,
    y_sign: int,
    interpretation_sign: int,
    interpolation_order: int,
    boundary_mode: str,
) -> list[str]:
    effective_x_sign, effective_y_sign = effective_shift_signs(x_sign, y_sign, interpretation_sign)
    return [
        "",
        "Per-projection drift correction",
        "--------------------------------",
        f"Enabled: {enabled}",
        f"Shift stage: {shift_stage}",
        f"x_shift_px min/max/mean/std: {_stats(x_shift_px)}",
        f"y_shift_px min/max/mean/std: {_stats(y_shift_px)}",
        f"Applied row formula: row_shift = {_signed(effective_y_sign)} * y_shift_px",
        f"Applied col formula: col_shift = {_signed(effective_x_sign)} * x_shift_px",
        f"Interpolation: scipy.ndimage.shift order={interpolation_order} mode={boundary_mode}",
        "Drift is applied directly to projection images in version 1.",
    ]


def _stats(values: np.ndarray | None) -> str:
    if values is None or len(values) == 0:
        return "unavailable"
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "unavailable"
    return f"{np.min(finite):.8g} / {np.max(finite):.8g} / {np.mean(finite):.8g} / {np.std(finite):.8g}"


def _signed(value: int) -> str:
    return "+1" if value >= 0 else "-1"


def _emit_log(logger: Callable[[str], None] | object | None, message: str) -> None:
    if logger is None:
        return
    if callable(logger):
        logger(message)
        return
    info = getattr(logger, "info", None)
    if callable(info):
        info(message)
