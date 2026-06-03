from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from .cancel_utils import raise_if_cancelled
from .distributed_tigre import DistributedFullDataTigrePool
from .geometry_tigre import GeometryParams, build_tigre_geometry, geometry_report


@dataclass
class OperatorTiming:
    forward_calls: int = 0
    backproject_calls: int = 0
    forward_seconds: float = 0.0
    backproject_seconds: float = 0.0


def tigre_available() -> tuple[bool, str]:
    try:
        import tigre  # noqa: F401
        from tigre.algorithms import fdk  # noqa: F401
    except Exception as exc:
        return False, str(exc)
    return True, "TIGRE import succeeded."


def make_tigre_gpuids(gpu_ids: Sequence[int] | None = None, use_gpu: bool = True):
    if not use_gpu:
        return None
    try:
        from tigre.utilities.gpu import GpuIds
    except Exception:
        return None
    gpuids = GpuIds()
    if gpu_ids is not None:
        normalized = [int(gpu_id) for gpu_id in gpu_ids]
        if normalized:
            gpuids.devices = normalized
    return gpuids


def gpu_diagnostics(gpu_ids: Sequence[int] | None = None, use_gpu: bool = True) -> list[str]:
    selected_gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids) if gpu_ids is not None else ()
    lines = ["GPU diagnostics", "---------------", f"Use TIGRE GPU: {use_gpu}"]
    if not use_gpu:
        lines.append("TIGRE GPU acceleration disabled in configuration.")
        return lines
    try:
        from tigre.utilities.gpu import getGpuNames

        names = list(getGpuNames())
    except Exception as exc:
        lines.append(f"Could not query TIGRE GPU names: {exc}")
        return lines
    if names:
        lines.append(f"Available TIGRE GPUs: {', '.join(f'{idx}: {name}' for idx, name in enumerate(names))}")
        if selected_gpu_ids:
            valid = [gpu_id for gpu_id in selected_gpu_ids if 0 <= int(gpu_id) < len(names)]
            invalid = [gpu_id for gpu_id in selected_gpu_ids if gpu_id not in valid]
            if valid:
                lines.append(
                    "Selected TIGRE GPUs: "
                    + ", ".join(f"{gpu_id}: {names[int(gpu_id)]}" for gpu_id in valid)
                )
            if invalid:
                lines.append(f"Warning: requested GPU IDs are outside the detected TIGRE GPU list: {invalid}")
        else:
            lines.append("Selected TIGRE GPUs: default TIGRE selection.")
    else:
        lines.append("No TIGRE GPUs were reported by the current environment.")
    return lines


def _normalize_memory_mode(memory_mode: str) -> str:
    mode = str(memory_mode or "auto").strip().lower()
    if mode == "auto":
        return "projection_streaming"
    if mode in {"full", "full_gpu", "monolithic"}:
        return "full_gpu"
    if mode in {"distributed", "distributed_full", "distributed_full_data", "multi_gpu_exact"}:
        return "distributed_full_data"
    if mode in {"stream", "streaming", "projection_streaming"}:
        return "projection_streaming"
    if mode in {"ordered_subset", "ordered_subsets", "os"}:
        return "ordered_subsets"
    raise ValueError(f"Unknown MBIR memory mode: {memory_mode}")


def _enumerate_batches(indices: np.ndarray, batch_size: int):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(indices), batch_size):
        yield start, np.ascontiguousarray(indices[start : start + batch_size], dtype=np.int64)


class TigreConeBeamOperator:
    """Matrix-free TIGRE cone-beam operator.

    Shape conventions:
    - volume arrays are `(z, y, x)` and match `geo.nVoxel`.
    - projection arrays are `(n_angles, detector_rows, detector_cols)`.
    """

    def __init__(
        self,
        params: GeometryParams,
        angles_rad: np.ndarray,
        gpu_ids: Sequence[int] | None = None,
        use_gpu: bool = True,
        memory_mode: str = "auto",
        projection_batch_size: int = 64,
        ordered_subset_count: int = 1,
        logger: Callable[[str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self.params = params
        self.angles = np.ascontiguousarray(angles_rad, dtype=np.float32)
        self.gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids) if gpu_ids is not None else ()
        self.use_gpu = bool(use_gpu)
        self.gpuids = make_tigre_gpuids(self.gpu_ids, self.use_gpu)
        self.geo = build_tigre_geometry(params)
        self.timing = OperatorTiming()
        self.memory_mode = _normalize_memory_mode(memory_mode)
        self.projection_batch_size = max(1, int(projection_batch_size))
        self.ordered_subset_count = max(1, int(ordered_subset_count))
        self.current_subset_index = 0
        self.logger = logger
        self.cancel_check = cancel_check
        self._distributed_pool: DistributedFullDataTigrePool | None = None
        if self.memory_mode == "distributed_full_data":
            if not self.use_gpu or len(self.gpu_ids) <= 1:
                self._log(
                    "Distributed full-data mode requested, but fewer than two GPUs are available in the active selection. "
                    "Falling back to full_gpu."
                )
                self.memory_mode = "full_gpu"
            else:
                self._distributed_pool = DistributedFullDataTigrePool(
                    self.params,
                    self.angles,
                    self.gpu_ids,
                    logger=self.logger,
                )
        if self.memory_mode != "ordered_subsets":
            self.ordered_subset_count = 1

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.params.volume_voxels)

    @property
    def projection_shape(self) -> tuple[int, int, int]:
        rows, cols = self.params.detector_pixels
        return (len(self.angles), int(rows), int(cols))

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.memory_mode == "distributed_full_data" and self._distributed_pool is not None:
            self._log(f"MBIR distributed full-data forward projection: {len(self.angles)} views across {len(self.gpu_ids)} GPUs.")
            started = perf_counter()
            projections = self._distributed_pool.forward(x, cancel_check=self.cancel_check)
            self.timing.forward_calls += 1
            self.timing.forward_seconds += perf_counter() - started
            return np.ascontiguousarray(projections, dtype=np.float32)
        if self._uses_batched_projection():
            return self._forward_batched(x, self._active_indices())
        self._log(f"MBIR full forward projection: {len(self.angles)} views.")
        return self._forward_angles(x, self.angles)

    def backproject(self, y: np.ndarray) -> np.ndarray:
        if self.memory_mode == "distributed_full_data" and self._distributed_pool is not None:
            self._log(f"MBIR distributed full-data backprojection: {len(self.angles)} views across {len(self.gpu_ids)} GPUs.")
            started = perf_counter()
            volume = self._distributed_pool.backproject(y, cancel_check=self.cancel_check)
            self.timing.backproject_calls += 1
            self.timing.backproject_seconds += perf_counter() - started
            return np.ascontiguousarray(volume, dtype=np.float32)
        if self._uses_batched_projection():
            indices = self._active_indices()
            return self._backproject_batched(y, indices, scale=self._subset_scale(indices))
        self._log(f"MBIR full backprojection: {len(self.angles)} views.")
        return self._backproject_angles(y, self.angles)

    def normal(self, x: np.ndarray) -> np.ndarray:
        if self.memory_mode == "distributed_full_data" and self._distributed_pool is not None:
            self._log(f"MBIR distributed exact A^T A: {len(self.angles)} views across {len(self.gpu_ids)} GPUs.")
            started = perf_counter()
            volume = self._distributed_pool.normal(x, cancel_check=self.cancel_check)
            self.timing.forward_calls += 1
            self.timing.backproject_calls += 1
            elapsed = perf_counter() - started
            self.timing.forward_seconds += 0.5 * elapsed
            self.timing.backproject_seconds += 0.5 * elapsed
            return np.ascontiguousarray(volume, dtype=np.float32)
        if self._uses_batched_projection():
            indices = self._active_indices()
            volume = np.zeros(self.volume_shape, dtype=np.float32)
            for batch_number, total_batches, batch in self._iter_logged_batches(indices, "normal A^T A"):
                projections = self._forward_angles(x, self.angles[batch])
                volume += self._backproject_angles(projections, self.angles[batch])
            volume *= self._subset_scale(indices)
            return np.ascontiguousarray(volume, dtype=np.float32)
        return self.backproject(self.forward(x))

    def data_residual_squared(
        self,
        x: np.ndarray,
        b: np.ndarray,
        projection_weights: np.ndarray | None = None,
    ) -> float:
        if self.memory_mode == "distributed_full_data" and self._distributed_pool is not None:
            if projection_weights is not None:
                residual = self.forward(x) - np.ascontiguousarray(b, dtype=np.float32)
                weighted = _apply_projection_weights(residual, projection_weights, squared=False)
                return float(np.sum(np.asarray(weighted, dtype=np.float64) ** 2))
            return self._distributed_pool.data_residual_squared(x, b, cancel_check=self.cancel_check)
        if not self._uses_batched_projection():
            residual = self.forward(x) - np.ascontiguousarray(b, dtype=np.float32)
            if projection_weights is not None:
                residual = _apply_projection_weights(residual, projection_weights, squared=False)
            return float(np.sum(np.asarray(residual, dtype=np.float64) ** 2))
        total = 0.0
        projections = np.asarray(b, dtype=np.float32)
        full_indices = np.arange(len(self.angles), dtype=np.int64)
        for batch_number, total_batches, batch in self._iter_logged_batches(full_indices, "full-data residual"):
            residual = self._forward_angles(x, self.angles[batch]) - projections[batch]
            if projection_weights is not None:
                residual = _apply_projection_weights(residual, projection_weights, squared=False, indices=batch)
            total += float(np.sum(np.asarray(residual, dtype=np.float64) ** 2))
        return total

    def data_gradient_batches(
        self,
        x: np.ndarray,
        b: np.ndarray,
        scale_to_full: bool = True,
        projection_weights: np.ndarray | None = None,
    ):
        """Yield A_i^T(A_i x - b_i) one projection batch at a time."""
        indices = self._active_indices()
        projections = np.asarray(b, dtype=np.float32)
        if projections.shape[0] not in {len(self.angles), len(indices)}:
            raise ValueError(
                f"Projection stack has {projections.shape[0]} views, but expected {len(self.angles)} full views "
                f"or {len(indices)} active subset views."
            )
        subset_scale = self._subset_scale(indices) if scale_to_full else 1.0
        for batch_number, total_batches, batch in self._iter_logged_batches(indices, "streaming data gradient"):
            if projections.shape[0] == len(self.angles):
                target = projections[batch]
            else:
                local_start = (batch_number - 1) * max(1, int(self.projection_batch_size))
                target = projections[local_start : local_start + len(batch)]
            residual = self._forward_angles(x, self.angles[batch]) - target
            if projection_weights is not None:
                if projections.shape[0] == len(self.angles):
                    residual = _apply_projection_weights(residual, projection_weights, squared=True, indices=batch)
                else:
                    local_start = (batch_number - 1) * max(1, int(self.projection_batch_size))
                    local_weights = np.asarray(projection_weights, dtype=np.float32)[local_start : local_start + len(batch)]
                    residual = _apply_projection_weights(residual, local_weights, squared=True)
            gradient = self._backproject_angles(residual, self.angles[batch])
            scale = subset_scale * float(len(indices)) / float(max(1, len(batch))) if scale_to_full else 1.0
            if scale != 1.0:
                gradient *= scale
            yield np.ascontiguousarray(gradient, dtype=np.float32), batch_number, total_batches

    def begin_iteration(self, iteration: int) -> None:
        if self.memory_mode == "ordered_subsets" and self.ordered_subset_count > 1:
            self.current_subset_index = (max(1, int(iteration)) - 1) % self.ordered_subset_count
            active = self._active_indices()
            self._log(
                f"MBIR ordered subset {self.current_subset_index + 1}/{self.ordered_subset_count}: "
                f"{len(active)} active projection views."
            )
        else:
            self.current_subset_index = 0

    def current_projection_indices(self) -> np.ndarray:
        return self._active_indices()

    def _uses_batched_projection(self) -> bool:
        return self.memory_mode in {"projection_streaming", "ordered_subsets"}

    def _active_indices(self) -> np.ndarray:
        if self.memory_mode == "ordered_subsets" and self.ordered_subset_count > 1:
            return np.arange(self.current_subset_index, len(self.angles), self.ordered_subset_count, dtype=np.int64)
        return np.arange(len(self.angles), dtype=np.int64)

    def _subset_scale(self, indices: np.ndarray) -> float:
        if self.memory_mode == "ordered_subsets" and len(indices) > 0:
            return float(len(self.angles)) / float(len(indices))
        return 1.0

    def _batch_indices(self, indices: np.ndarray):
        batch_size = max(1, int(self.projection_batch_size))
        for start in range(0, len(indices), batch_size):
            yield np.ascontiguousarray(indices[start : start + batch_size], dtype=np.int64)

    def _iter_logged_batches(self, indices: np.ndarray, operation: str):
        indices = np.ascontiguousarray(indices, dtype=np.int64)
        total_batches = max(1, int(np.ceil(len(indices) / max(1, int(self.projection_batch_size)))))
        for batch_number, (_, batch) in enumerate(_enumerate_batches(indices, self.projection_batch_size), start=1):
            raise_if_cancelled(self.cancel_check, "MBIR projection operation cancelled.")
            self._log_batch(operation, batch_number, total_batches, batch)
            yield batch_number, total_batches, batch

    def _forward_batched(self, x: np.ndarray, indices: np.ndarray) -> np.ndarray:
        rows, cols = self.params.detector_pixels
        output = np.empty((len(indices), int(rows), int(cols)), dtype=np.float32)
        total_batches = max(1, int(np.ceil(len(indices) / max(1, int(self.projection_batch_size)))))
        for batch_number, (local_start, batch) in enumerate(_enumerate_batches(indices, self.projection_batch_size), start=1):
            self._log_batch("forward projection", batch_number, total_batches, batch)
            batch_projection = self._forward_angles(x, self.angles[batch])
            output[local_start : local_start + len(batch)] = batch_projection
        return np.ascontiguousarray(output, dtype=np.float32)

    def _backproject_batched(self, y: np.ndarray, indices: np.ndarray, scale: float = 1.0) -> np.ndarray:
        projections = np.asarray(y, dtype=np.float32)
        if projections.shape[0] == len(self.angles):
            projection_source = projections
        elif projections.shape[0] == len(indices):
            projection_source = None
        else:
            raise ValueError(
                f"Projection stack has {projections.shape[0]} views, but expected {len(self.angles)} full views "
                f"or {len(indices)} active subset views."
            )
        volume = np.zeros(self.volume_shape, dtype=np.float32)
        total_batches = max(1, int(np.ceil(len(indices) / max(1, int(self.projection_batch_size)))))
        for batch_number, (local_start, batch) in enumerate(_enumerate_batches(indices, self.projection_batch_size), start=1):
            self._log_batch("backprojection", batch_number, total_batches, batch)
            if projection_source is None:
                chunk = projections[local_start : local_start + len(batch)]
            else:
                chunk = projection_source[batch]
            volume += self._backproject_angles(chunk, self.angles[batch])
        volume *= scale
        return np.ascontiguousarray(volume, dtype=np.float32)

    def _forward_angles(self, x: np.ndarray, angles: np.ndarray) -> np.ndarray:
        try:
            import tigre
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("TIGRE is not importable.") from exc
        raise_if_cancelled(self.cancel_check, "MBIR forward projection cancelled.")
        volume = np.ascontiguousarray(x, dtype=np.float32)
        started = perf_counter()
        try:
            if self.gpuids is None:
                projections = tigre.Ax(volume, self.geo, np.ascontiguousarray(angles, dtype=np.float32))
            else:
                projections = tigre.Ax(volume, self.geo, np.ascontiguousarray(angles, dtype=np.float32), gpuids=self.gpuids)
        except TypeError:
            try:
                projections = tigre.Ax(volume, self.geo, np.ascontiguousarray(angles, dtype=np.float32))
            except Exception as exc:
                raise _tigre_operator_error("forward projection", exc) from exc
        except Exception as exc:
            raise _tigre_operator_error("forward projection", exc) from exc
        self.timing.forward_calls += 1
        self.timing.forward_seconds += perf_counter() - started
        return np.ascontiguousarray(projections, dtype=np.float32)

    def _backproject_angles(self, y: np.ndarray, angles: np.ndarray) -> np.ndarray:
        try:
            import tigre
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("TIGRE is not importable.") from exc
        raise_if_cancelled(self.cancel_check, "MBIR backprojection cancelled.")
        projections = np.ascontiguousarray(y, dtype=np.float32)
        started = perf_counter()
        try:
            if self.gpuids is None:
                volume = tigre.Atb(
                    projections,
                    self.geo,
                    np.ascontiguousarray(angles, dtype=np.float32),
                    backprojection_type="matched",
                )
            else:
                volume = tigre.Atb(
                    projections,
                    self.geo,
                    np.ascontiguousarray(angles, dtype=np.float32),
                    backprojection_type="matched",
                    gpuids=self.gpuids,
                )
        except TypeError:
            try:
                volume = tigre.Atb(projections, self.geo, np.ascontiguousarray(angles, dtype=np.float32), "matched")
            except Exception as exc:
                raise _tigre_operator_error("backprojection", exc) from exc
        except Exception as exc:
            raise _tigre_operator_error("backprojection", exc) from exc
        self.timing.backproject_calls += 1
        self.timing.backproject_seconds += perf_counter() - started
        return np.ascontiguousarray(volume, dtype=np.float32)

    def report(self, projections: np.ndarray | None = None) -> list[str]:
        lines = geometry_report(self.params, self.angles, projections, angle_sign=1)
        lines.extend(
            [
                f"MBIR operator memory mode: {self.memory_mode}",
                f"Projection batch size: {self.projection_batch_size}",
                f"Ordered subset count: {self.ordered_subset_count}",
                "Backprojection type: matched adjoint",
            ]
        )
        return lines

    def close(self) -> None:
        if self._distributed_pool is not None:
            self._distributed_pool.close()
            self._distributed_pool = None

    def _log_batch(self, operation: str, batch_number: int, total_batches: int, batch: np.ndarray) -> None:
        if len(batch) == 0:
            return
        subset_text = ""
        if self.memory_mode == "ordered_subsets" and self.ordered_subset_count > 1:
            subset_text = f", subset {self.current_subset_index + 1}/{self.ordered_subset_count}"
        first_view = int(batch[0])
        last_view = int(batch[-1])
        self._log(
            f"MBIR {operation} batch {batch_number}/{total_batches}: "
            f"views {first_view}-{last_view} ({len(batch)} views{subset_text})."
        )

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)


def run_fdk_reconstruction(
    attenuation_stack: np.ndarray,
    angles_rad: np.ndarray,
    params: GeometryParams,
    gpu_ids: Sequence[int] | None = None,
    use_gpu: bool = True,
    progress: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    available, message = tigre_available()
    if not available:
        raise RuntimeError(f"TIGRE import failed: {message}")
    from scipy.ndimage import shift as ndi_shift
    from tigre.algorithms import fdk

    if attenuation_stack.ndim != 3:
        raise ValueError(f"Attenuation stack must have shape (angles, rows, cols), got {attenuation_stack.shape}")
    if attenuation_stack.shape[0] != len(angles_rad):
        raise ValueError("Number of projections does not match number of angles.")
    raise_if_cancelled(cancel_check, "FDK reconstruction cancelled.")
    projections = np.ascontiguousarray(attenuation_stack, dtype=np.float32)
    if params.use_center_correction and params.center_offset_method == "image_shift":
        image_shift_px = params.center_shift_sign * params.center_offset_pixels
        if progress is not None:
            progress(f"Applying image-space horizontal center shift: {image_shift_px:g} px")
        projections = ndi_shift(projections, shift=(0.0, 0.0, float(image_shift_px)), order=1, mode="nearest", prefilter=False)
        projections = projections.astype(np.float32, copy=False)
    raise_if_cancelled(cancel_check, "FDK reconstruction cancelled.")
    geo = build_tigre_geometry(params)
    gpuids = make_tigre_gpuids(gpu_ids, use_gpu)
    if progress is not None:
        for line in geometry_report(params, angles_rad, projections, angle_sign=1):
            progress(line)
        for line in gpu_diagnostics(gpu_ids, use_gpu):
            progress(line)
        progress("Starting TIGRE FDK reconstruction.")
    try:
        if gpuids is None:
            volume = fdk(projections, geo, np.ascontiguousarray(angles_rad, dtype=np.float32), filter=params.fdk_filter)
        else:
            volume = fdk(
                projections,
                geo,
                np.ascontiguousarray(angles_rad, dtype=np.float32),
                filter=params.fdk_filter,
                gpuids=gpuids,
            )
    except TypeError:
        volume = fdk(projections, geo, np.ascontiguousarray(angles_rad, dtype=np.float32), filter=params.fdk_filter)
    raise_if_cancelled(cancel_check, "FDK reconstruction cancelled.")
    if progress is not None:
        progress("TIGRE FDK reconstruction finished.")
    return np.ascontiguousarray(volume, dtype=np.float32)


def _apply_projection_weights(
    residual: np.ndarray,
    projection_weights: np.ndarray,
    squared: bool,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    weights = np.asarray(projection_weights, dtype=np.float32)
    if indices is not None and weights.ndim == 1:
        weights = weights[np.asarray(indices, dtype=np.int64)]
    elif indices is not None and weights.ndim == 3:
        weights = weights[np.asarray(indices, dtype=np.int64)]
    if weights.ndim == 1:
        if weights.shape[0] != residual.shape[0]:
            raise ValueError(f"Projection weight count {weights.shape[0]} does not match residual views {residual.shape[0]}.")
        factors = weights[:, None, None]
    elif weights.ndim == 3:
        if weights.shape != residual.shape:
            raise ValueError(f"Projection weight shape {weights.shape} does not match residual shape {residual.shape}.")
        factors = weights
    else:
        raise ValueError(f"Projection weights must be 1-D or 3-D, got shape {weights.shape}.")
    if squared:
        factors = factors * factors
    return np.ascontiguousarray(np.asarray(residual, dtype=np.float32) * factors, dtype=np.float32)


def _tigre_operator_error(operation: str, exc: Exception) -> RuntimeError:
    detail = str(exc)
    hint = (
        "TIGRE/CUDA failed during "
        f"{operation}. If the message mentions cudaMalloc or out of memory, use projection_streaming, "
        "reduce projection batch size, reduce volume voxels for a debug run, or select a GPU with more memory. "
        "If system RAM is near full, close other applications or reduce the volume; TIGRE Atb also needs host RAM "
        "and page-locked host buffers, so CUDA can report out-of-memory even when VRAM is not visibly full."
    )
    return RuntimeError(f"{hint}\nOriginal TIGRE error: {detail}")


def _load_existing_fdk_volume(
    existing_path: str,
    target_shape: tuple[int, int, int],
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    volume = np.asarray(np.load(existing_path), dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Existing FDK volume must be 3-D, got shape {volume.shape}.")
    source_shape = tuple(int(value) for value in volume.shape)
    target_shape = tuple(int(value) for value in target_shape)
    if min(target_shape) <= 0:
        raise ValueError(f"Target MBIR volume shape must be positive, got {target_shape}.")
    if source_shape == target_shape:
        if progress is not None:
            progress(f"Loaded existing FDK initialization with matching shape {source_shape}.")
        return np.ascontiguousarray(volume, dtype=np.float32)
    if progress is not None:
        progress(f"Existing FDK shape {source_shape} does not match MBIR grid {target_shape}; resizing initialization.")
    if _can_block_average_to_shape(source_shape, target_shape):
        factors = tuple(source // target for source, target in zip(source_shape, target_shape))
        if progress is not None:
            progress(f"Downsampling existing FDK initialization by block averages: factors {factors}.")
        return _block_average_volume(volume, factors)
    try:
        from scipy.ndimage import zoom as ndi_zoom
    except Exception as exc:  # pragma: no cover - scipy is part of the app environment
        raise RuntimeError(
            "Existing FDK initialization must be resized for the selected MBIR debug grid, but scipy is unavailable."
        ) from exc
    zoom_factors = tuple(target / source for source, target in zip(source_shape, target_shape))
    if progress is not None:
        progress(f"Resampling existing FDK initialization with linear interpolation: zoom factors {zoom_factors}.")
    resized = ndi_zoom(volume, zoom_factors, order=1, mode="nearest", prefilter=False)
    if resized.shape != target_shape:
        resized = _crop_or_pad_volume(resized, target_shape)
    return np.ascontiguousarray(resized, dtype=np.float32)


def _can_block_average_to_shape(source_shape: tuple[int, int, int], target_shape: tuple[int, int, int]) -> bool:
    return all(source >= target and source % target == 0 for source, target in zip(source_shape, target_shape))


def _block_average_volume(volume: np.ndarray, factors: tuple[int, int, int]) -> np.ndarray:
    fz, fy, fx = [int(value) for value in factors]
    if min(fz, fy, fx) <= 0:
        raise ValueError(f"Invalid block-average factors: {factors}")
    z, y, x = volume.shape
    reshaped = np.ascontiguousarray(volume, dtype=np.float32).reshape(z // fz, fz, y // fy, fy, x // fx, fx)
    return np.ascontiguousarray(reshaped.mean(axis=(1, 3, 5), dtype=np.float32), dtype=np.float32)


def _crop_or_pad_volume(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    output = np.zeros(target_shape, dtype=np.float32)
    source_slices = tuple(slice(0, min(source, target)) for source, target in zip(volume.shape, target_shape))
    target_slices = tuple(slice(0, min(source, target)) for source, target in zip(volume.shape, target_shape))
    output[target_slices] = np.asarray(volume, dtype=np.float32)[source_slices]
    return output


def fdk_initialization(
    attenuation_stack: np.ndarray,
    angles_rad: np.ndarray,
    params: GeometryParams,
    mode: str = "fdk",
    existing_path: str | None = None,
    constant_value: float = 0.0,
    gpu_ids: Sequence[int] | None = None,
    use_gpu: bool = True,
    progress: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    mode = mode.lower()
    if mode in {"existing_fdk", "existing", "file"}:
        if not existing_path:
            raise ValueError("Existing FDK initialization requires fdk_volume_path.")
        return _load_existing_fdk_volume(existing_path, tuple(params.volume_voxels), progress=progress)
    if mode == "zeros":
        return np.zeros(params.volume_voxels, dtype=np.float32)
    if mode == "constant":
        return np.full(params.volume_voxels, float(constant_value), dtype=np.float32)
    if mode == "fdk":
        return run_fdk_reconstruction(
            attenuation_stack,
            angles_rad,
            params,
            gpu_ids=gpu_ids,
            use_gpu=use_gpu,
            progress=progress,
            cancel_check=cancel_check,
        )
    raise ValueError(f"Unknown initialization mode: {mode}")
