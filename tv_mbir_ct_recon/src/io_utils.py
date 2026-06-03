from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import tifffile

from .cancel_utils import raise_if_cancelled
from .cpu_utils import bounded_thread_map, default_cpu_parallel_plan


TIFF_PATTERNS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")


def list_tiff_files(folder: str | Path) -> list[Path]:
    path = Path(folder)
    files: list[Path] = []
    for pattern in TIFF_PATTERNS:
        files.extend(path.glob(pattern))
    return sorted(set(files))


def read_tiff_float32(path: str | Path, binning: int | tuple[int, int] = 1) -> np.ndarray:
    array = tifffile.imread(str(path)).astype(np.float32, copy=False)
    if array.ndim > 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D TIFF image, got shape {array.shape} for {path}")
    return preprocess_image_2d(array, binning=binning)


def preprocess_image_2d(
    image: np.ndarray,
    binning: int | tuple[int, int] = 1,
    crop: Sequence[int] = (0, 0, 0, 0),
) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {array.shape}")
    by, bx = _normalize_binning(binning)
    if by > 1 or bx > 1:
        array = block_average_2d(array, by, bx)
    return crop_2d(array, crop)


def read_tiff_preprocessed(
    path: str | Path,
    binning: int | tuple[int, int] = 1,
    crop: Sequence[int] = (0, 0, 0, 0),
) -> np.ndarray:
    return preprocess_image_2d(read_tiff_float32(path, binning=1), binning=binning, crop=crop)


def block_average_2d(array: np.ndarray, by: int, bx: int | None = None) -> np.ndarray:
    bx = by if bx is None else bx
    if by <= 1 and bx <= 1:
        return array.astype(np.float32, copy=False)
    if by <= 0 or bx <= 0:
        raise ValueError("Binning factors must be positive integers.")
    rows, cols = array.shape
    trim_rows = rows - rows % by
    trim_cols = cols - cols % bx
    if trim_rows <= 0 or trim_cols <= 0:
        raise ValueError(f"Image shape {array.shape} is too small for binning {(by, bx)}.")
    trimmed = array[:trim_rows, :trim_cols].astype(np.float32, copy=False)
    return trimmed.reshape(trim_rows // by, by, trim_cols // bx, bx).mean(axis=(1, 3))


def crop_2d(array: np.ndarray, crop: Sequence[int]) -> np.ndarray:
    top, bottom, left, right = [int(value) for value in crop]
    if min(top, bottom, left, right) < 0:
        raise ValueError("Crop values must be non-negative.")
    rows, cols = array.shape
    row_end = rows - bottom if bottom else rows
    col_end = cols - right if right else cols
    if top >= row_end or left >= col_end:
        raise ValueError(f"Crop {tuple(crop)} removes all pixels from shape {array.shape}.")
    return np.ascontiguousarray(array[top:row_end, left:col_end], dtype=np.float32)


def crop_stack(stack: np.ndarray, crop: Sequence[int]) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError(f"Stack must be 3-D, got {stack.shape}")
    top, bottom, left, right = [int(value) for value in crop]
    if min(top, bottom, left, right) < 0:
        raise ValueError("Crop values must be non-negative.")
    rows, cols = stack.shape[1:]
    row_end = rows - bottom if bottom else rows
    col_end = cols - right if right else cols
    if top >= row_end or left >= col_end:
        raise ValueError(f"Crop {tuple(crop)} removes all pixels from stack shape {stack.shape}.")
    return np.ascontiguousarray(stack[:, top:row_end, left:col_end], dtype=np.float32)


def estimate_stack_memory_gb(num_images: int, rows: int, cols: int, bytes_per_pixel: int = 4) -> float:
    return num_images * rows * cols * bytes_per_pixel / 1024**3


def load_projection_stack(
    records: Sequence[object],
    binning: int | tuple[int, int] = 1,
    crop: Sequence[int] = (0, 0, 0, 0),
    progress: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    if not records:
        raise ValueError("No projection records are available.")
    raise_if_cancelled(cancel_check, "Projection stack loading cancelled.")
    first_path = getattr(records[0], "path")
    first = read_tiff_preprocessed(first_path, binning=binning, crop=crop)
    rows, cols = first.shape
    stack = np.empty((len(records), rows, cols), dtype=np.float32)
    stack[0] = first
    if progress is not None:
        progress(f"Loaded projection 1/{len(records)}: {getattr(records[0], 'filename', first_path)}")
    remaining = list(records[1:])
    plan = default_cpu_parallel_plan((rows, cols))
    use_parallel = len(remaining) >= 4 and plan.io_workers > 1
    if progress is not None:
        mode = f"{plan.io_workers} workers" if use_parallel else "serial"
        progress(f"Projection stack load mode: {mode}")

    def _load_one(record: object) -> tuple[object, np.ndarray]:
        raise_if_cancelled(cancel_check, "Projection stack loading cancelled.")
        path = getattr(record, "path")
        if not Path(path).exists():
            raise FileNotFoundError(f"Projection listed in metadata is missing: {getattr(record, 'filename', path)}")
        image = read_tiff_preprocessed(path, binning=binning, crop=crop)
        if image.shape != (rows, cols):
            raise ValueError(
                f"Projection shape mismatch for {getattr(record, 'filename', path)}: "
                f"{image.shape}, expected {(rows, cols)}"
            )
        return record, image

    if use_parallel:
        completed = 1
        for local_index, (record, image) in bounded_thread_map(
            _load_one,
            remaining,
            max_workers=plan.io_workers,
            cancel_check=cancel_check,
        ):
            raise_if_cancelled(cancel_check, "Projection stack loading cancelled.")
            stack[local_index + 1] = image
            completed += 1
            if progress is not None and (completed == len(records) or completed % 25 == 0):
                progress(f"Loaded projection {completed}/{len(records)}: {getattr(record, 'filename', getattr(record, 'path', ''))}")
    else:
        for index, record in enumerate(remaining, start=1):
            raise_if_cancelled(cancel_check, "Projection stack loading cancelled.")
            record, image = _load_one(record)
            stack[index] = image
            if progress is not None and (index + 1 == len(records) or (index + 1) % 25 == 0):
                progress(f"Loaded projection {index + 1}/{len(records)}: {getattr(record, 'filename', getattr(record, 'path', ''))}")
    return stack


def save_image(path: str | Path, image: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output), np.asarray(image, dtype=np.float32), photometric="minisblack")
    return output


def save_stack_tiff(path: str | Path, stack: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output), np.asarray(stack, dtype=np.float32), photometric="minisblack")
    return output


def save_stack_folder(
    folder: str | Path,
    stack: np.ndarray,
    prefix: str,
    progress: Callable[[str], None] | None = None,
) -> Path:
    output = Path(folder)
    output.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(np.asarray(stack)):
        path = output / f"{prefix}_{index:06d}.tif"
        tifffile.imwrite(str(path), np.asarray(image, dtype=np.float32), photometric="minisblack")
        if progress is not None and (index + 1 == stack.shape[0] or (index + 1) % 25 == 0):
            progress(f"Saved {index + 1}/{stack.shape[0]} slices to {output}")
    return output


def save_png_image(path: str | Path, image: np.ndarray, cmap: str = "gray") -> Path:
    from matplotlib import image as mpimg

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size:
        low, high = np.percentile(finite, [1, 99])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(finite))
            high = float(np.max(finite))
    else:
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1.0
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    mpimg.imsave(str(output), scaled, cmap=cmap, vmin=0.0, vmax=1.0)
    return output


def _normalize_binning(binning: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(binning, int):
        return int(binning), int(binning)
    if len(binning) != 2:
        raise ValueError("Binning must be an int or a two-element tuple.")
    return int(binning[0]), int(binning[1])
