from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

import numpy as np

from .config import AlignmentConfig, GeometryConfig, PreprocessingConfig
from .geometry_tigre import detector_pixel_from_effective_um
from .metadata_zeiss import (
    MetadataValidation,
    ProjectionRecord,
    describe_angle_direction,
    validation_preview_rows,
)


def normalize_debug_binning(value: Sequence[int] | int | None) -> tuple[int, int]:
    if value is None:
        return 1, 1
    if isinstance(value, int):
        by = bx = int(value)
    else:
        if len(value) != 2:
            raise ValueError("MBIR debug pixel binning must contain Y and X factors.")
        by, bx = int(value[0]), int(value[1])
    if by < 1 or bx < 1:
        raise ValueError("MBIR debug pixel binning factors must be positive integers.")
    return by, bx


def normalize_projection_stride(value: int | None) -> int:
    stride = 1 if value is None else int(value)
    if stride < 1:
        raise ValueError("MBIR projection stride must be a positive integer.")
    return stride


def mbir_debug_is_active(pixel_binning: Sequence[int] | int | None, projection_stride: int | None) -> bool:
    by, bx = normalize_debug_binning(pixel_binning)
    return by > 1 or bx > 1 or normalize_projection_stride(projection_stride) > 1


def projection_axis_binning(pixel_binning: Sequence[int] | int | None, transpose_for_tigre: bool) -> tuple[int, int]:
    by, bx = normalize_debug_binning(pixel_binning)
    if transpose_for_tigre:
        return bx, by
    return by, bx


def volume_axis_binning(pixel_binning: Sequence[int] | int | None, transpose_for_tigre: bool) -> tuple[int, int, int]:
    row_factor, col_factor = projection_axis_binning(pixel_binning, transpose_for_tigre)
    return row_factor, col_factor, col_factor


def effective_projection_count(num_projections: int, projection_stride: int | None) -> int:
    count = max(0, int(num_projections))
    stride = normalize_projection_stride(projection_stride)
    if count == 0:
        return 0
    return int(math.ceil(count / stride))


def effective_detector_shape(
    detector_shape: tuple[int, int],
    pixel_binning: Sequence[int] | int | None,
    transpose_for_tigre: bool = False,
) -> tuple[int, int]:
    row_factor, col_factor = projection_axis_binning(pixel_binning, transpose_for_tigre)
    rows, cols = int(detector_shape[0]), int(detector_shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("Detector shape must be positive.")
    return max(1, rows // row_factor), max(1, cols // col_factor)


def effective_volume_shape(
    volume_shape: tuple[int, int, int],
    pixel_binning: Sequence[int] | int | None,
    transpose_for_tigre: bool = False,
) -> tuple[int, int, int]:
    factors = volume_axis_binning(pixel_binning, transpose_for_tigre)
    return tuple(max(1, int(shape) // int(factor)) for shape, factor in zip(volume_shape, factors))


def mbir_debug_report_lines(
    original_projection_count: int,
    original_detector_shape: tuple[int, int],
    original_volume_shape: tuple[int, int, int],
    pixel_binning: Sequence[int] | int | None,
    projection_stride: int | None,
    transpose_for_tigre: bool = False,
) -> list[str]:
    by, bx = normalize_debug_binning(pixel_binning)
    stride = normalize_projection_stride(projection_stride)
    effective_count = effective_projection_count(original_projection_count, stride)
    effective_detector = effective_detector_shape(original_detector_shape, (by, bx), transpose_for_tigre)
    effective_volume = effective_volume_shape(original_volume_shape, (by, bx), transpose_for_tigre)
    return [
        "MBIR debug sampling",
        "-------------------",
        f"Extra MBIR pixel binning y/x: {by} x {bx}",
        f"MBIR projection stride: every {stride} projection(s)",
        f"Projection views used for MBIR: {effective_count} / {int(original_projection_count)}",
        f"Estimated MBIR detector rows/cols: {effective_detector[0]} x {effective_detector[1]}",
        f"Estimated MBIR volume voxels z/y/x: {effective_volume[0]} x {effective_volume[1]} x {effective_volume[2]}",
    ]


def preprocessing_for_mbir_debug(
    preprocessing: PreprocessingConfig,
    pixel_binning: Sequence[int] | int | None,
) -> PreprocessingConfig:
    by, bx = normalize_debug_binning(pixel_binning)
    if by == 1 and bx == 1:
        return preprocessing
    current_by, current_bx = normalize_debug_binning(preprocessing.binning)
    top, bottom, left, right = [int(value) for value in preprocessing.crop]
    return replace(
        preprocessing,
        binning=(current_by * by, current_bx * bx),
        crop=(
            _ceil_div(top, by),
            _ceil_div(bottom, by),
            _ceil_div(left, bx),
            _ceil_div(right, bx),
        ),
    )


def validation_for_mbir_debug(
    validation: MetadataValidation,
    pixel_binning: Sequence[int] | int | None,
    projection_stride: int | None,
) -> MetadataValidation:
    by, bx = normalize_debug_binning(pixel_binning)
    stride = normalize_projection_stride(projection_stride)
    selected_records = list(validation.records[::stride])
    if not selected_records:
        raise ValueError("MBIR projection stride leaves no projection records.")
    records = _scale_record_shifts(selected_records, by=by, bx=bx)
    angles_rad = np.asarray([record.angle_rad for record in records], dtype=np.float32)
    angles_deg = np.asarray([record.angle_deg for record in records], dtype=np.float32)
    warnings = list(validation.warnings)
    if stride > 1:
        warnings.append(f"MBIR debug run uses every {stride} projection from the validated metadata.")
    if by > 1 or bx > 1:
        warnings.append(f"MBIR debug run applies extra pixel binning {by} x {bx}.")
    return replace(
        validation,
        records=records,
        angle_direction=describe_angle_direction(angles_deg.astype(np.float64)),
        angle_min_deg=float(np.min(angles_deg)) if angles_deg.size else math.nan,
        angle_max_deg=float(np.max(angles_deg)) if angles_deg.size else math.nan,
        angle_min_rad=float(np.min(angles_rad)) if angles_rad.size else math.nan,
        angle_max_rad=float(np.max(angles_rad)) if angles_rad.size else math.nan,
        angles_rad=angles_rad,
        x_shift_px=_records_shift_array(records, "x_shift_px"),
        y_shift_px=_records_shift_array(records, "y_shift_px"),
        table_rows=validation_preview_rows(records),
        warnings=warnings,
    )


def geometry_for_mbir_debug(
    geometry: GeometryConfig,
    pixel_binning: Sequence[int] | int | None,
    detector_shape: tuple[int, int],
    transpose_for_tigre: bool = False,
) -> GeometryConfig:
    by, bx = normalize_debug_binning(pixel_binning)
    if by == 1 and bx == 1:
        return replace(geometry, detector_pixels=(int(detector_shape[0]), int(detector_shape[1])))
    row_factor, col_factor = projection_axis_binning((by, bx), transpose_for_tigre)
    volume_factors = volume_axis_binning((by, bx), transpose_for_tigre)
    detector_pixel_size = _scaled_detector_pixel_size(geometry, row_factor, col_factor)
    voxel_size = _scaled_voxel_size(geometry.voxel_size_mm, volume_factors)
    volume_voxels = _scaled_volume_voxels(geometry.volume_voxels, volume_factors)
    det_v, det_u = geometry.detector_offset_pixels
    return replace(
        geometry,
        detector_pixels=(int(detector_shape[0]), int(detector_shape[1])),
        detector_pixel_size_mm=detector_pixel_size,
        voxel_size_mm=voxel_size,
        volume_voxels=volume_voxels,
        detector_offset_pixels=(float(det_v) / row_factor, float(det_u) / col_factor),
    )


def alignment_for_mbir_debug(
    alignment: AlignmentConfig,
    pixel_binning: Sequence[int] | int | None,
    transpose_for_tigre: bool = False,
) -> AlignmentConfig:
    by, bx = normalize_debug_binning(pixel_binning)
    if by == 1 and bx == 1:
        return alignment
    _, col_factor = projection_axis_binning((by, bx), transpose_for_tigre)
    return replace(alignment, center_offset_pixels=float(alignment.center_offset_pixels) / col_factor)


def _scaled_detector_pixel_size(geometry: GeometryConfig, row_factor: int, col_factor: int) -> tuple[float | None, float | None]:
    d_v, d_u = geometry.detector_pixel_size_mm
    if d_v is not None and d_u is not None:
        return float(d_v) * row_factor, float(d_u) * col_factor
    if (
        geometry.effective_pixel_size_um is not None
        and geometry.source_origin_distance_mm is not None
        and geometry.source_detector_distance_mm is not None
    ):
        base = detector_pixel_from_effective_um(
            float(geometry.effective_pixel_size_um),
            float(geometry.source_origin_distance_mm),
            float(geometry.source_detector_distance_mm),
        )
        return base * row_factor, base * col_factor
    return d_v, d_u


def _scaled_voxel_size(
    voxel_size: tuple[float | None, float | None, float | None],
    factors: tuple[int, int, int],
) -> tuple[float | None, float | None, float | None]:
    return tuple(None if value is None else float(value) * factor for value, factor in zip(voxel_size, factors))


def _scaled_volume_voxels(
    volume_voxels: tuple[int | None, int | None, int | None],
    factors: tuple[int, int, int],
) -> tuple[int | None, int | None, int | None]:
    return tuple(None if value is None else max(1, int(value) // factor) for value, factor in zip(volume_voxels, factors))


def _scale_record_shifts(records: list[ProjectionRecord], by: int, bx: int) -> list[ProjectionRecord]:
    if by == 1 and bx == 1:
        return records
    scaled: list[ProjectionRecord] = []
    for record in records:
        x_shift = None if record.x_shift_px is None else float(record.x_shift_px) / bx
        y_shift = None if record.y_shift_px is None else float(record.y_shift_px) / by
        scaled.append(replace(record, x_shift_px=x_shift, y_shift_px=y_shift))
    return scaled


def _records_shift_array(records: Sequence[ProjectionRecord], attribute: str) -> np.ndarray | None:
    values = [getattr(record, attribute) for record in records]
    if not values or any(value is None for value in values):
        return None
    return np.asarray(values, dtype=np.float32)


def _ceil_div(value: int, divisor: int) -> int:
    return int(math.ceil(max(0, int(value)) / max(1, int(divisor))))
