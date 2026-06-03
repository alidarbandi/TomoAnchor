from __future__ import annotations

from dataclasses import dataclass, replace
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from .cancel_utils import raise_if_cancelled
from .geometry_tigre import GeometryParams


METRIC_EPSILON = 1e-12


@dataclass
class CenterShiftPreviewResult:
    shift_px: float
    preview_image: np.ndarray
    preview_index: int
    center_shift_sign: float
    detector_offset_mm: float
    metrics: dict[str, float] | None = None
    reconstruction_time_s: float | None = None


def generate_shift_values(start_px: float, end_px: float, step_px: float) -> list[float]:
    if not np.isfinite(start_px) or not np.isfinite(end_px) or not np.isfinite(step_px):
        raise ValueError("Start shift, end shift, and step must be finite numbers.")
    if step_px <= 0:
        raise ValueError("Center-offset search step must be greater than zero.")
    direction = 1.0 if end_px >= start_px else -1.0
    signed_step = direction * abs(step_px)
    tolerance = max(abs(step_px) * 1e-7, 1e-9)
    values: list[float] = []
    current = float(start_px)
    while direction * (current - end_px) <= tolerance:
        values.append(round(current, 10))
        current += signed_step
        if len(values) > 100000:
            raise ValueError("Center-offset search generated too many values.")
    if values and abs(values[-1] - end_px) <= tolerance:
        values[-1] = round(float(end_px), 10)
    elif not values:
        values.append(round(float(start_px), 10))
    return values


def center_offset_mm(params: GeometryParams) -> float:
    return params.center_shift_sign * params.center_offset_pixels * params.detector_pixel_size_mm[1]


def make_params_with_center_shift(params: GeometryParams, shift_px: float, center_shift_sign: float = 1.0) -> GeometryParams:
    return replace(
        params,
        center_offset_pixels=float(shift_px),
        center_shift_sign=float(center_shift_sign),
        use_center_correction=True,
        center_offset_method="detector_offset",
    )


def reconstruct_center_shift_preview(
    projections: np.ndarray,
    angles_rad: np.ndarray,
    base_params: GeometryParams,
    shift_px: float,
    center_shift_sign: float = 1.0,
    gpu_ids: Sequence[int] | None = None,
    use_gpu: bool = True,
    progress: Callable[[str], None] | None = None,
) -> CenterShiftPreviewResult:
    from .tigre_ops import run_fdk_reconstruction

    preview_params = make_params_with_center_shift(base_params, shift_px, center_shift_sign)
    z, y, x = preview_params.volume_voxels
    preview_params = replace(preview_params, volume_voxels=(1, y, x))
    started = perf_counter()
    volume = run_fdk_reconstruction(
        projections,
        angles_rad,
        preview_params,
        gpu_ids=gpu_ids,
        use_gpu=use_gpu,
        progress=progress,
    )
    elapsed = perf_counter() - started
    image = np.asarray(volume[volume.shape[0] // 2], dtype=np.float32)
    return CenterShiftPreviewResult(
        shift_px=float(shift_px),
        preview_image=np.ascontiguousarray(image, dtype=np.float32),
        preview_index=0,
        center_shift_sign=float(center_shift_sign),
        detector_offset_mm=center_shift_sign * shift_px * base_params.detector_pixel_size_mm[1],
        reconstruction_time_s=elapsed,
    )


def reconstruct_center_shift_previews(
    projections: np.ndarray,
    angles_rad: np.ndarray,
    base_params: GeometryParams,
    shift_values_px: Sequence[float],
    center_shift_sign: float = 1.0,
    gpu_ids: Sequence[int] | None = None,
    use_gpu: bool = True,
    progress: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[CenterShiftPreviewResult]:
    normalized_gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids) if gpu_ids is not None else ()
    shifts = [float(shift_px) for shift_px in shift_values_px]
    if not shifts:
        return []
    if not use_gpu or len(normalized_gpu_ids) <= 1 or len(shifts) <= 1:
        results: list[CenterShiftPreviewResult] = []
        for index, shift_px in enumerate(shifts):
            raise_if_cancelled(cancel_check, "Center-offset search cancelled.")
            preview = reconstruct_center_shift_preview(
                projections,
                angles_rad,
                base_params,
                shift_px,
                center_shift_sign=center_shift_sign,
                gpu_ids=normalized_gpu_ids or None,
                use_gpu=use_gpu,
                progress=None,
            )
            preview.preview_index = index
            results.append(preview)
        return results
    return _reconstruct_center_shift_previews_parallel(
        projections,
        angles_rad,
        base_params,
        shifts,
        center_shift_sign=center_shift_sign,
        gpu_ids=normalized_gpu_ids,
        use_gpu=use_gpu,
        progress=progress,
        cancel_check=cancel_check,
    )


@lru_cache(maxsize=8)
def _load_preview_memmaps(projections_path: str, angles_path: str) -> tuple[np.ndarray, np.ndarray]:
    projections = np.load(projections_path, mmap_mode="r")
    angles = np.load(angles_path, mmap_mode="r")
    return projections, angles


def _center_shift_preview_worker(
    projections_path: str,
    angles_path: str,
    base_params: GeometryParams,
    shift_px: float,
    center_shift_sign: float,
    gpu_id: int,
    use_gpu: bool,
) -> CenterShiftPreviewResult:
    projections, angles_rad = _load_preview_memmaps(projections_path, angles_path)
    return reconstruct_center_shift_preview(
        projections,
        angles_rad,
        base_params,
        shift_px,
        center_shift_sign=center_shift_sign,
        gpu_ids=(int(gpu_id),),
        use_gpu=use_gpu,
        progress=None,
    )


def _reconstruct_center_shift_previews_parallel(
    projections: np.ndarray,
    angles_rad: np.ndarray,
    base_params: GeometryParams,
    shift_values_px: Sequence[float],
    center_shift_sign: float,
    gpu_ids: Sequence[int],
    use_gpu: bool,
    progress: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[CenterShiftPreviewResult]:
    gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)
    worker_count = min(len(gpu_ids), len(shift_values_px))
    ordered_results: list[CenterShiftPreviewResult | None] = [None] * len(shift_values_px)
    with TemporaryDirectory(prefix="tv_mbir_center_search_") as temp_dir:
        temp_path = Path(temp_dir)
        projections_path = temp_path / "center_search_projections.npy"
        angles_path = temp_path / "center_search_angles.npy"
        np.save(projections_path, np.ascontiguousarray(projections, dtype=np.float32))
        np.save(angles_path, np.ascontiguousarray(angles_rad, dtype=np.float32))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            pending: dict[object, tuple[int, int, float]] = {}
            next_index = 0

            def submit_next() -> bool:
                nonlocal next_index
                if next_index >= len(shift_values_px):
                    return False
                shift_px = float(shift_values_px[next_index])
                gpu_id = int(gpu_ids[next_index % len(gpu_ids)])
                if progress is not None:
                    progress(
                        f"Dispatching center preview {next_index + 1}/{len(shift_values_px)} to GPU {gpu_id}: "
                        f"shift={shift_px:.4f} px"
                    )
                future = executor.submit(
                    _center_shift_preview_worker,
                    str(projections_path),
                    str(angles_path),
                    base_params,
                    shift_px,
                    float(center_shift_sign),
                    gpu_id,
                    bool(use_gpu),
                )
                pending[future] = (next_index, gpu_id, shift_px)
                next_index += 1
                return True

            for _ in range(worker_count):
                submit_next()

            while pending:
                if cancel_check is not None and cancel_check():
                    for future in pending:
                        future.cancel()
                    raise_if_cancelled(cancel_check, "Center-offset search cancelled.")
                done, _ = wait(tuple(pending), timeout=0.1, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for future in done:
                    index, gpu_id, shift_px = pending.pop(future)
                    result = future.result()
                    result.preview_index = index
                    ordered_results[index] = result
                    if progress is not None:
                        progress(
                            f"Center preview {index + 1}/{len(shift_values_px)} finished on GPU {gpu_id} "
                            f"in {result.reconstruction_time_s:.2f} s"
                        )
                    submit_next()
    return [result for result in ordered_results if result is not None]


def prepare_image_for_metrics(
    image: np.ndarray,
    crop_fraction: float = 0.1,
    percentile_clip: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Metric image must be 2-D, got shape {array.shape}.")
    crop_fraction = min(max(float(crop_fraction), 0.0), 0.45)
    rows, cols = array.shape
    row_crop = int(rows * crop_fraction)
    col_crop = int(cols * crop_fraction)
    if row_crop > 0 or col_crop > 0:
        array = array[row_crop : rows - row_crop or rows, col_crop : cols - col_crop or cols]
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros((1, 1), dtype=np.float64)
    low, high = np.percentile(finite, percentile_clip)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros_like(array, dtype=np.float64)
    clipped = np.clip(np.nan_to_num(array, nan=low, posinf=high, neginf=low), low, high)
    return (clipped - low) / (high - low + METRIC_EPSILON)


def metric_gradient_energy(image: np.ndarray) -> float:
    prepared = prepare_image_for_metrics(image)
    if prepared.shape[0] < 2 or prepared.shape[1] < 2:
        return 0.0
    gy, gx = np.gradient(prepared)
    value = float(np.mean(gx * gx + gy * gy))
    return value if np.isfinite(value) else 0.0


def metric_laplacian_variance(image: np.ndarray) -> float:
    prepared = prepare_image_for_metrics(image)
    laplacian = np.zeros_like(prepared)
    laplacian[1:-1, 1:-1] = (
        prepared[:-2, 1:-1]
        + prepared[2:, 1:-1]
        + prepared[1:-1, :-2]
        + prepared[1:-1, 2:]
        - 4.0 * prepared[1:-1, 1:-1]
    )
    value = float(np.var(laplacian))
    return value if np.isfinite(value) else 0.0


def metric_shannon_entropy(image: np.ndarray, bins: int = 256) -> float:
    prepared = prepare_image_for_metrics(image)
    hist, _ = np.histogram(prepared[np.isfinite(prepared)], bins=bins, range=(0.0, 1.0), density=False)
    total = np.sum(hist)
    if total <= 0:
        return 0.0
    probabilities = hist.astype(np.float64) / float(total)
    probabilities = probabilities[probabilities > 0]
    value = float(-np.sum(probabilities * np.log(probabilities + METRIC_EPSILON)))
    return max(value, 0.0) if np.isfinite(value) else 0.0


def compute_metrics_for_results(results: list[CenterShiftPreviewResult]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for result in results:
        metrics = {
            "shift_px": float(result.shift_px),
            "gradient_energy": metric_gradient_energy(result.preview_image),
            "laplacian_variance": metric_laplacian_variance(result.preview_image),
            "entropy": metric_shannon_entropy(result.preview_image),
        }
        rows.append(metrics)
    gradient_scores = _normalize_metric([row["gradient_energy"] for row in rows], True)
    laplacian_scores = _normalize_metric([row["laplacian_variance"] for row in rows], True)
    entropy_scores = _normalize_metric([row["entropy"] for row in rows], False)
    for index, row in enumerate(rows):
        row["combined_score"] = 0.4 * gradient_scores[index] + 0.4 * laplacian_scores[index] + 0.2 * entropy_scores[index]
        results[index].metrics = {key: value for key, value in row.items() if key != "shift_px"}
    return rows


def recommend_metric_index(metric_table: list[dict[str, float]], metric_key: str = "combined_score") -> int | None:
    if not metric_table:
        return None
    values = np.asarray([row.get(metric_key, np.nan) for row in metric_table], dtype=np.float64)
    if not np.any(np.isfinite(values)):
        return None
    return int(np.nanargmin(values) if metric_key == "entropy" else np.nanargmax(values))


def _normalize_metric(values: list[float], higher_is_better: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float64)
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    normalized = (np.nan_to_num(array, nan=min_value) - min_value) / (max_value - min_value + METRIC_EPSILON)
    return normalized if higher_is_better else 1.0 - normalized
