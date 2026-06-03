from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import AlignmentConfig, GeometryConfig


@dataclass
class GeometryParams:
    source_detector_distance_mm: float
    source_origin_distance_mm: float
    detector_pixel_size_mm: tuple[float, float]  # vertical, horizontal
    detector_pixels: tuple[int, int]  # rows, cols
    voxel_size_mm: tuple[float, float, float]  # z, y, x
    volume_voxels: tuple[int, int, int]  # z, y, x
    detector_offset_pixels: tuple[float, float] = (0.0, 0.0)
    origin_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    center_offset_pixels: float = 0.0
    center_shift_sign: float = 1.0
    use_center_correction: bool = True
    center_offset_method: str = "detector_offset"
    fdk_filter: str = "ram_lak"


def detector_pixel_from_effective_um(effective_pixel_size_um: float, dso_mm: float, dsd_mm: float) -> float:
    if effective_pixel_size_um <= 0 or dso_mm <= 0 or dsd_mm <= 0:
        raise ValueError("Effective pixel size, DSO, and DSD must be positive.")
    return effective_pixel_size_um * 1e-3 * (dsd_mm / dso_mm)


def params_from_config(
    config: GeometryConfig,
    detector_shape: tuple[int, int] | None = None,
    alignment: AlignmentConfig | None = None,
) -> GeometryParams:
    dsd = config.source_detector_distance_mm
    dso = config.source_origin_distance_mm
    if dsd is None or dso is None:
        raise ValueError("Geometry requires source_detector_distance_mm and source_origin_distance_mm.")
    if dsd <= dso:
        raise ValueError("source_detector_distance_mm must be larger than source_origin_distance_mm.")
    rows_cols = _resolve_detector_pixels(config.detector_pixels, detector_shape)
    detector_pixel = _resolve_detector_pixel_size(config, dso, dsd)
    voxels = _resolve_volume_voxels(config.volume_voxels, detector_shape)
    voxel_size = _resolve_voxel_size(config.voxel_size_mm)
    align = alignment or AlignmentConfig()
    return GeometryParams(
        source_detector_distance_mm=float(dsd),
        source_origin_distance_mm=float(dso),
        detector_pixel_size_mm=detector_pixel,
        detector_pixels=rows_cols,
        voxel_size_mm=voxel_size,
        volume_voxels=voxels,
        detector_offset_pixels=(float(config.detector_offset_pixels[0]), float(config.detector_offset_pixels[1])),
        origin_offset_mm=tuple(float(v) for v in config.origin_offset_mm),
        center_offset_pixels=float(align.center_offset_pixels),
        center_shift_sign=float(align.center_shift_sign),
        use_center_correction=bool(align.use_center_correction),
        center_offset_method=str(align.center_offset_method),
        fdk_filter=str(config.fdk_filter),
    )


def build_angles_from_config(config: GeometryConfig, n_angles: int) -> np.ndarray:
    if config.angle_file:
        data = np.loadtxt(config.angle_file, dtype=np.float32)
        angles = np.asarray(data, dtype=np.float32).reshape(-1)
        if config.angle_units.lower().startswith("deg"):
            angles = np.deg2rad(angles).astype(np.float32)
        if len(angles) != n_angles:
            raise ValueError(f"Angle file contains {len(angles)} angles, expected {n_angles}.")
        return _apply_angle_direction(angles, config)
    endpoint = bool(config.endpoint_included)
    raw = np.linspace(float(config.angle_start), float(config.angle_end), n_angles, endpoint=endpoint, dtype=np.float32)
    if config.angle_units.lower().startswith("deg"):
        raw = np.deg2rad(raw).astype(np.float32)
    return _apply_angle_direction(raw, config)


def angles_for_tigre(angles_rad: np.ndarray, invert_angle_sign: bool = True) -> np.ndarray:
    sign = -1.0 if invert_angle_sign else 1.0
    return np.ascontiguousarray(sign * np.asarray(angles_rad, dtype=np.float32), dtype=np.float32)


def build_tigre_geometry(params: GeometryParams):
    validate_geometry_params(params)
    try:
        import tigre
    except Exception as exc:  # pragma: no cover - depends on local TIGRE
        raise RuntimeError("TIGRE is not importable. Install CERN/TIGRE for this environment.") from exc

    try:
        geo = tigre.geometry(mode="cone", default=False)
    except TypeError:
        geo = tigre.geometry(mode="cone")

    detector_rows, detector_cols = params.detector_pixels
    d_v, d_u = params.detector_pixel_size_mm
    n_z, n_y, n_x = params.volume_voxels
    dz, dy, dx = params.voxel_size_mm

    geo.DSD = float(params.source_detector_distance_mm)
    geo.DSO = float(params.source_origin_distance_mm)
    geo.accuracy = 0.5
    geo.nDetector = np.array([detector_rows, detector_cols], dtype=np.int32)
    geo.dDetector = np.array([d_v, d_u], dtype=np.float64)
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.array([n_z, n_y, n_x], dtype=np.int32)
    geo.dVoxel = np.array([dz, dy, dx], dtype=np.float64)
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.offOrigin = np.array(params.origin_offset_mm, dtype=np.float64)

    off_v_mm = params.detector_offset_pixels[0] * d_v
    off_u_mm = params.detector_offset_pixels[1] * d_u
    if params.use_center_correction and params.center_offset_method == "detector_offset":
        off_u_mm += params.center_shift_sign * params.center_offset_pixels * d_u
    geo.offDetector = np.array([off_v_mm, off_u_mm], dtype=np.float64)
    return geo


def geometry_report(
    params: GeometryParams,
    angles_rad: np.ndarray | None = None,
    projection_stack: np.ndarray | None = None,
    angle_sign: int = -1,
) -> list[str]:
    rows, cols = params.detector_pixels
    d_v, d_u = params.detector_pixel_size_mm
    n_z, n_y, n_x = params.volume_voxels
    dz, dy, dx = params.voxel_size_mm
    center_offset_mm = params.center_shift_sign * params.center_offset_pixels * d_u if params.use_center_correction else 0.0
    if angles_rad is None or len(angles_rad) == 0:
        angle_summary = ["Number of angles: unavailable", "Angle range: unavailable", "Median angle step: unavailable"]
    else:
        angles_deg = np.rad2deg(np.asarray(angles_rad, dtype=np.float64))
        step = np.median(np.abs(np.diff(angles_deg))) if len(angles_deg) > 1 else np.nan
        angle_summary = [
            f"Number of angles: {len(angles_deg)}",
            f"Angle range: {np.min(angles_deg):.8g} to {np.max(angles_deg):.8g} degrees",
            f"Median angle step: {step:.8g} degrees" if np.isfinite(step) else "Median angle step: unavailable",
        ]
    shape = "unavailable" if projection_stack is None else str(tuple(projection_stack.shape))
    minmax = "unavailable"
    if projection_stack is not None and projection_stack.size:
        minmax = f"{np.nanmin(projection_stack):.8g} / {np.nanmax(projection_stack):.8g}"
    return [
        "Geometry sanity report:",
        f"DSO: {params.source_origin_distance_mm:g} mm",
        f"DSD: {params.source_detector_distance_mm:g} mm",
        f"Geometric magnification DSD/DSO: {params.source_detector_distance_mm / params.source_origin_distance_mm:.8g}",
        f"TIGRE detector pixels rows/cols: {rows} x {cols}",
        f"TIGRE dDetector v/u: {d_v:.12g}, {d_u:.12g} mm",
        f"TIGRE sDetector v/u: {rows * d_v:.12g}, {cols * d_u:.12g} mm",
        *angle_summary,
        f"TIGRE angle sign multiplier: {angle_sign:+d}",
        f"Projection stack shape: {shape}",
        f"TIGRE projection input min/max: {minmax}",
        f"TIGRE nVoxel z/y/x: {n_z} x {n_y} x {n_x}",
        f"TIGRE dVoxel z/y/x: {dz:g}, {dy:g}, {dx:g} mm",
        f"Origin offset z/y/x: {params.origin_offset_mm}",
        f"Detector offset pixels v/u: {params.detector_offset_pixels}",
        f"Center offset enabled: {params.use_center_correction}",
        f"Center offset: {params.center_offset_pixels:g} px = {center_offset_mm:.8g} mm",
        f"Center offset method: {params.center_offset_method}",
        f"FDK filter: {params.fdk_filter}",
    ]


def validate_geometry_params(params: GeometryParams) -> None:
    if params.source_origin_distance_mm <= 0 or params.source_detector_distance_mm <= 0:
        raise ValueError("DSO and DSD must be positive.")
    if params.source_detector_distance_mm <= params.source_origin_distance_mm:
        raise ValueError("DSD must be larger than DSO.")
    if min(params.detector_pixel_size_mm) <= 0:
        raise ValueError("Detector pixel sizes must be positive.")
    if min(params.detector_pixels) <= 0:
        raise ValueError("Detector pixel counts must be positive.")
    if min(params.voxel_size_mm) <= 0:
        raise ValueError("Voxel sizes must be positive.")
    if min(params.volume_voxels) <= 0:
        raise ValueError("Volume voxel counts must be positive.")
    if params.center_offset_method not in {"detector_offset", "image_shift"}:
        raise ValueError("Center offset method must be detector_offset or image_shift.")
    if params.center_shift_sign not in (-1, 1, -1.0, 1.0):
        raise ValueError("Center-shift sign convention must be +1 or -1.")


def _resolve_detector_pixels(
    configured: tuple[int | None, int | None],
    detector_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    rows, cols = configured
    if rows is None or cols is None:
        if detector_shape is None:
            raise ValueError("Detector pixels were not configured and no projection shape was supplied.")
        return int(detector_shape[0]), int(detector_shape[1])
    return int(rows), int(cols)


def _resolve_detector_pixel_size(config: GeometryConfig, dso: float, dsd: float) -> tuple[float, float]:
    d_v, d_u = config.detector_pixel_size_mm
    if d_v is not None and d_u is not None:
        return float(d_v), float(d_u)
    if config.effective_pixel_size_um is None:
        raise ValueError("Need detector_pixel_size_mm or effective_pixel_size_um.")
    computed = detector_pixel_from_effective_um(float(config.effective_pixel_size_um), float(dso), float(dsd))
    return computed, computed


def _resolve_volume_voxels(
    configured: tuple[int | None, int | None, int | None],
    detector_shape: tuple[int, int] | None,
) -> tuple[int, int, int]:
    z, y, x = configured
    if z is not None and y is not None and x is not None:
        return int(z), int(y), int(x)
    if detector_shape is None:
        raise ValueError("Volume voxels were not configured and no detector shape was supplied.")
    rows, cols = detector_shape
    return int(rows), int(cols), int(cols)


def _resolve_voxel_size(values: tuple[float | None, float | None, float | None]) -> tuple[float, float, float]:
    if any(value is None for value in values):
        raise ValueError("voxel_size_mm must contain z, y, and x values.")
    return tuple(float(value) for value in values)  # type: ignore[arg-type]


def _apply_angle_direction(angles: np.ndarray, config: GeometryConfig) -> np.ndarray:
    result = np.asarray(angles, dtype=np.float32)
    if config.angle_direction.lower() in {"cw", "clockwise", "negative"}:
        result = -result
    return np.ascontiguousarray(result, dtype=np.float32)
