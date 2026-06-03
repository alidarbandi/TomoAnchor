from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .cancel_utils import raise_if_cancelled
from .cpu_utils import bounded_thread_map, default_cpu_parallel_plan
from .io_utils import list_tiff_files, preprocess_image_2d, read_tiff_float32


def compute_average_image(
    folder: str | Path,
    expected_raw_shape: tuple[int, int] | None = None,
    binning: int | tuple[int, int] = 1,
    crop: Sequence[int] = (0, 0, 0, 0),
    label: str = "reference",
    progress: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, int]:
    raise_if_cancelled(cancel_check, f"{label.title()} averaging cancelled.")
    files = list_tiff_files(folder)
    if not files:
        raise FileNotFoundError(f"No TIFF {label} images found in {folder}")

    raw0 = read_tiff_float32(files[0], binning=1)
    if expected_raw_shape is not None and raw0.shape != expected_raw_shape:
        raise ValueError(f"{label.title()} shape mismatch for {files[0].name}: {raw0.shape}, expected {expected_raw_shape}")
    first_image = preprocess_image_2d(raw0, binning=binning, crop=crop)
    accumulator = np.zeros_like(first_image, dtype=np.float64)
    accumulator += first_image.astype(np.float64, copy=False)
    expected_binned_shape = first_image.shape
    plan = default_cpu_parallel_plan(expected_binned_shape)
    remaining = list(files[1:])
    use_parallel = len(remaining) >= 4 and plan.io_workers > 1
    if progress is not None:
        mode = f"{plan.io_workers} workers" if use_parallel else "serial"
        progress(f"{label.title()} averaging mode: {mode}")
        if len(files) == 1:
            progress(f"Averaged {label} 1/1: {files[0].name}")

    def _load_one(path: Path) -> tuple[Path, np.ndarray]:
        raise_if_cancelled(cancel_check, f"{label.title()} averaging cancelled.")
        raw = read_tiff_float32(path, binning=1)
        if expected_raw_shape is not None and raw.shape != expected_raw_shape:
            raise ValueError(f"{label.title()} shape mismatch for {path.name}: {raw.shape}, expected {expected_raw_shape}")
        image = preprocess_image_2d(raw, binning=binning, crop=crop)
        if image.shape != expected_binned_shape:
            raise ValueError(f"{label.title()} shape mismatch for {path.name}: {image.shape}, expected {expected_binned_shape}")
        return path, image

    if use_parallel:
        completed = 1
        for _, (path, image) in bounded_thread_map(
            _load_one,
            remaining,
            max_workers=plan.io_workers,
            cancel_check=cancel_check,
        ):
            raise_if_cancelled(cancel_check, f"{label.title()} averaging cancelled.")
            accumulator += image.astype(np.float64, copy=False)
            completed += 1
            if progress is not None and (completed == len(files) or completed % 10 == 0):
                progress(f"Averaged {label} {completed}/{len(files)}: {path.name}")
    else:
        for index, path in enumerate(remaining, start=2):
            raise_if_cancelled(cancel_check, f"{label.title()} averaging cancelled.")
            _, image = _load_one(path)
            accumulator += image.astype(np.float64, copy=False)
            if progress is not None and (index == len(files) or index % 10 == 0):
                progress(f"Averaged {label} {index}/{len(files)}: {path.name}")

    return (accumulator / len(files)).astype(np.float32), len(files)


def compute_average_flat_field(
    reference_folder: str | Path,
    expected_raw_shape: tuple[int, int] | None = None,
    binning: int | tuple[int, int] = 1,
    crop: Sequence[int] = (0, 0, 0, 0),
    progress: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, int]:
    return compute_average_image(reference_folder, expected_raw_shape, binning, crop, "flat-field", progress, cancel_check)


def compute_average_dark_field(
    dark_folder: str | Path,
    expected_raw_shape: tuple[int, int] | None = None,
    binning: int | tuple[int, int] = 1,
    crop: Sequence[int] = (0, 0, 0, 0),
    progress: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, int]:
    return compute_average_image(dark_folder, expected_raw_shape, binning, crop, "dark-field", progress, cancel_check)
