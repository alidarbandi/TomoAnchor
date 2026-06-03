from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .alignment_drift import apply_projection_shifts, effective_shift_signs
from .cancel_utils import raise_if_cancelled
from .config import AlignmentConfig, PreprocessingConfig
from .cpu_utils import cpu_parallel_report_lines, default_cpu_parallel_plan
from .io_utils import load_projection_stack
from .metadata_zeiss import MetadataValidation
from .reference_images import compute_average_dark_field, compute_average_flat_field


@dataclass
class PreprocessingResult:
    raw_stack: np.ndarray
    flat_field: np.ndarray | None
    dark_field: np.ndarray | None
    transmission_stack: np.ndarray | None
    attenuation_stack: np.ndarray
    report_lines: list[str]


def apply_orientation_to_image(image: np.ndarray, flip_horizontal: bool, flip_vertical: bool) -> np.ndarray:
    oriented = image
    if flip_vertical:
        oriented = np.flip(oriented, axis=0)
    if flip_horizontal:
        oriented = np.flip(oriented, axis=1)
    return np.ascontiguousarray(oriented, dtype=np.float32)


def apply_orientation_to_stack(stack: np.ndarray, flip_horizontal: bool, flip_vertical: bool) -> np.ndarray:
    oriented = stack
    if flip_vertical:
        oriented = np.flip(oriented, axis=1)
    if flip_horizontal:
        oriented = np.flip(oriented, axis=2)
    return np.ascontiguousarray(oriented, dtype=np.float32)


def flat_dark_correct(
    projection_stack: np.ndarray,
    flat_field: np.ndarray | None,
    dark_field: np.ndarray | None = None,
    epsilon: float = 1e-6,
    clip_transmission: bool = True,
    min_transmission: float = 1e-6,
    max_transmission: float = 2.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> np.ndarray:
    if projection_stack.ndim != 3:
        raise ValueError(f"Projection stack must have shape (angles, rows, cols), got {projection_stack.shape}")
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")
    oriented_stack = apply_orientation_to_stack(projection_stack, flip_horizontal, flip_vertical)
    if flat_field is None:
        transmission = oriented_stack.astype(np.float32, copy=False)
    else:
        if flat_field.shape != projection_stack.shape[1:]:
            raise ValueError(f"Flat-field shape {flat_field.shape} does not match projections {projection_stack.shape[1:]}")
        oriented_flat = apply_orientation_to_image(flat_field, flip_horizontal, flip_vertical)
        if dark_field is not None:
            if dark_field.shape != projection_stack.shape[1:]:
                raise ValueError(f"Dark-field shape {dark_field.shape} does not match projections {projection_stack.shape[1:]}")
            oriented_dark = apply_orientation_to_image(dark_field, flip_horizontal, flip_vertical)
            numerator = oriented_stack - oriented_dark[None, :, :]
            denominator = oriented_flat - oriented_dark
            transmission = numerator / np.maximum(denominator[None, :, :], np.float32(epsilon))
        else:
            transmission = oriented_stack / np.maximum(oriented_flat[None, :, :], np.float32(epsilon))
    if clip_transmission:
        if min_transmission <= 0:
            raise ValueError("Minimum transmission must be positive when clipping is enabled.")
        transmission = np.clip(transmission, min_transmission, max_transmission)
    return np.ascontiguousarray(transmission, dtype=np.float32)


def compute_attenuation(
    transmission: np.ndarray,
    epsilon: float = 1e-6,
    max_transmission_for_log: float = 2.0,
    clip_negative_to_zero: bool = True,
) -> np.ndarray:
    if transmission.ndim != 3:
        raise ValueError(f"Transmission stack must have shape (angles, rows, cols), got {transmission.shape}")
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")
    if max_transmission_for_log <= epsilon:
        raise ValueError("Maximum transmission for log must be larger than epsilon.")
    transmission_safe = np.clip(transmission, epsilon, max_transmission_for_log)
    attenuation = -np.log(transmission_safe).astype(np.float32)
    if clip_negative_to_zero:
        attenuation = np.maximum(attenuation, 0.0)
    return np.ascontiguousarray(attenuation, dtype=np.float32)


def truncation_correction(projections: np.ndarray, extension_fraction: float = 0.1) -> np.ndarray:
    if projections.ndim != 3:
        raise ValueError(f"Projection stack must have shape (angles, rows, cols), got {projections.shape}")
    if extension_fraction <= 0:
        raise ValueError("Truncation correction extension fraction must be positive.")
    _, _, cols = projections.shape
    n_ext = int(cols * extension_fraction)
    if n_ext <= 0:
        raise ValueError(f"Extension fraction {extension_fraction:g} is too small for {cols} columns.")
    extended = np.pad(np.asarray(projections, dtype=np.float32), ((0, 0), (0, 0), (n_ext, n_ext)), mode="symmetric")
    ramp = np.linspace(0.0, 1.0, n_ext, endpoint=True, dtype=np.float32)
    extended[:, :, :n_ext] *= ramp[None, None, :]
    extended[:, :, -n_ext:] *= ramp[::-1][None, None, :]
    return np.ascontiguousarray(extended, dtype=np.float32)


def prepare_tigre_projection_input(stack: np.ndarray, transpose_for_tigre: bool) -> tuple[np.ndarray, list[str]]:
    lines = [
        "",
        "TIGRE projection axis mapping",
        "-----------------------------",
        f"Transpose attenuation stack for TIGRE: {transpose_for_tigre}",
        f"Input shape before optional transpose: {tuple(stack.shape)}",
    ]
    if not transpose_for_tigre:
        lines.append("TIGRE receives stack as [n_angles, rows, cols].")
        return np.ascontiguousarray(stack, dtype=np.float32), lines
    transposed = np.ascontiguousarray(np.transpose(stack, (0, 2, 1)), dtype=np.float32)
    lines.extend(
        [
            "Applied np.transpose(stack, (0, 2, 1)).",
            f"Input shape after transpose: {tuple(transposed.shape)}",
        ]
    )
    return transposed, lines


def run_preprocessing(
    validation: MetadataValidation,
    projection_folder: str,
    flat_folder: str = "",
    dark_folder: str = "",
    preprocessing: PreprocessingConfig | None = None,
    alignment: AlignmentConfig | None = None,
    progress: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> PreprocessingResult:
    prep = preprocessing or PreprocessingConfig()
    align = alignment or AlignmentConfig()
    binning = prep.binning
    crop = prep.crop
    raise_if_cancelled(cancel_check, "Preprocessing cancelled.")
    first_raw_shape = None
    if validation.records:
        from .io_utils import read_tiff_float32

        first_raw_shape = read_tiff_float32(validation.records[0].path, binning=1).shape
    raw = load_projection_stack(validation.records, binning=binning, crop=crop, progress=progress, cancel_check=cancel_check)
    cpu_plan = default_cpu_parallel_plan(raw.shape[1:])
    report = [
        "Preprocessing",
        "-------------",
        f"Raw projection stack shape: {tuple(raw.shape)}",
        f"Binning: {tuple(binning)}",
        f"Crop after binning (top, bottom, left, right): {tuple(crop)}",
        f"Flip horizontal: {prep.flip_horizontal}",
        f"Flip vertical: {prep.flip_vertical}",
    ]
    report.extend(cpu_parallel_report_lines(cpu_plan))
    flat = None
    dark = None
    if prep.use_flat:
        if not flat_folder:
            raise ValueError("Flat-field correction is enabled but no flat folder was supplied.")
        flat, flat_count = compute_average_flat_field(
            flat_folder,
            first_raw_shape,
            binning,
            crop,
            progress,
            cancel_check=cancel_check,
        )
        report.append(f"Flat-field images averaged: {flat_count}")
    if prep.use_dark and dark_folder:
        dark, dark_count = compute_average_dark_field(
            dark_folder,
            first_raw_shape,
            binning,
            crop,
            progress,
            cancel_check=cancel_check,
        )
        report.append(f"Dark-field images averaged: {dark_count}")
    elif prep.use_dark:
        report.append("Dark-field correction requested but no dark folder was supplied; using no dark field.")

    transmission = flat_dark_correct(
        raw,
        flat,
        dark,
        epsilon=prep.epsilon,
        clip_transmission=prep.clip_transmission,
        min_transmission=prep.transmission_clip_min,
        max_transmission=prep.transmission_clip_max,
        flip_horizontal=prep.flip_horizontal,
        flip_vertical=prep.flip_vertical,
    )
    raise_if_cancelled(cancel_check, "Preprocessing cancelled.")
    report.append("Computed transmission using (I-D)/(F-D) when dark is available, otherwise I/F.")
    report.append(f"Transmission clipping: {prep.clip_transmission}")

    if align.use_drift_correction and align.drift_stage == "transmission":
        if validation.x_shift_px is not None and validation.y_shift_px is not None:
            x_sign, y_sign = effective_shift_signs(align.x_shift_sign, align.y_shift_sign, align.shift_interpretation_sign)
            transmission = apply_projection_shifts(
                transmission,
                validation.x_shift_px,
                validation.y_shift_px,
                x_sign=x_sign,
                y_sign=y_sign,
                interpolation_order=align.shift_interpolation_order,
                mode=align.shift_boundary_mode,
                logger=progress,
                cancel_check=cancel_check,
            )
            report.append("Applied per-projection drift correction to transmission stack.")
        else:
            report.append("Drift correction enabled but metadata shift columns are incomplete.")

    if prep.negative_log:
        attenuation = compute_attenuation(
            transmission,
            epsilon=prep.epsilon,
            max_transmission_for_log=prep.max_transmission_for_log,
            clip_negative_to_zero=prep.clip_negative_attenuation_to_zero,
        )
        report.append("Computed line-integral attenuation b = -log(transmission).")
    else:
        attenuation = np.ascontiguousarray(transmission, dtype=np.float32)
        if prep.clip_negative_attenuation_to_zero:
            attenuation = np.maximum(attenuation, 0.0).astype(np.float32, copy=False)
        report.append("Skipped negative log; using the corrected stack directly as reconstruction input.")
    raise_if_cancelled(cancel_check, "Preprocessing cancelled.")

    if align.use_drift_correction and align.drift_stage == "attenuation":
        if validation.x_shift_px is not None and validation.y_shift_px is not None:
            x_sign, y_sign = effective_shift_signs(align.x_shift_sign, align.y_shift_sign, align.shift_interpretation_sign)
            attenuation = apply_projection_shifts(
                attenuation,
                validation.x_shift_px,
                validation.y_shift_px,
                x_sign=x_sign,
                y_sign=y_sign,
                interpolation_order=align.shift_interpolation_order,
                mode=align.shift_boundary_mode,
                logger=progress,
                cancel_check=cancel_check,
            )
            report.append("Applied per-projection drift correction to attenuation stack.")

    if prep.truncation_correction:
        before = attenuation.shape
        raise_if_cancelled(cancel_check, "Preprocessing cancelled.")
        attenuation = truncation_correction(attenuation, prep.truncation_extension_fraction)
        report.append(f"Applied truncation correction: {before} -> {attenuation.shape}")

    return PreprocessingResult(raw, flat, dark, transmission, attenuation, report)
