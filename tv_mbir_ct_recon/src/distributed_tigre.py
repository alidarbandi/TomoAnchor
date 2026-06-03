from __future__ import annotations

import gc
import multiprocessing as mp
from multiprocessing import shared_memory
from queue import Empty
import traceback
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from .cancel_utils import OperationCancelled
from .geometry_tigre import GeometryParams, build_tigre_geometry


_QUEUE_POLL_SECONDS = 0.1
_PROCESS_JOIN_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class DistributedFullDataMemoryEstimate:
    shared_buffers_gb: float
    transient_peak_gb: float


@dataclass(frozen=True)
class _SharedArraySpec:
    name: str
    shape: tuple[int, ...]
    dtype_str: str

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.dtype_str)


@dataclass(frozen=True)
class _WorkerShard:
    worker_id: int
    gpu_id: int
    start: int
    stop: int

    @property
    def count(self) -> int:
        return max(0, int(self.stop) - int(self.start))


class _SharedArray:
    def __init__(self, shm: shared_memory.SharedMemory, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self.shm = shm
        self.shape = tuple(int(value) for value in shape)
        self.dtype = np.dtype(dtype)

    @classmethod
    def create(cls, shape: Sequence[int], dtype: np.dtype | str = np.float32) -> "_SharedArray":
        normalized_shape = tuple(int(value) for value in shape)
        dtype = np.dtype(dtype)
        size = int(np.prod(normalized_shape, dtype=np.int64)) * int(dtype.itemsize)
        shm = shared_memory.SharedMemory(create=True, size=max(1, size))
        return cls(shm, normalized_shape, dtype)

    @property
    def spec(self) -> _SharedArraySpec:
        return _SharedArraySpec(self.shm.name, self.shape, self.dtype.str)

    def array(self) -> np.ndarray:
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def close(self, unlink: bool = False) -> None:
        try:
            self.shm.close()
        finally:
            if unlink:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass


class DistributedFullDataTigrePool:
    def __init__(
        self,
        params: GeometryParams,
        angles_rad: np.ndarray,
        gpu_ids: Sequence[int],
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.params = params
        self.angles = np.ascontiguousarray(angles_rad, dtype=np.float32)
        self.gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)
        self.logger = logger
        if len(self.gpu_ids) <= 1:
            raise ValueError("Distributed full-data TIGRE pool requires at least two GPUs.")
        if len(self.angles) == 0:
            raise ValueError("Distributed full-data TIGRE pool requires at least one projection angle.")

        self.volume_shape = tuple(int(value) for value in params.volume_voxels)
        self.projection_shape = (len(self.angles), int(params.detector_pixels[0]), int(params.detector_pixels[1]))
        self._ctx = mp.get_context("spawn")
        self._result_queue = self._ctx.Queue()
        self._commands: list[object] = []
        self._processes: list[mp.Process] = []
        self._shared_arrays: list[_SharedArray] = []
        self._closed = False

        self._volume_shared = _SharedArray.create(self.volume_shape, np.float32)
        self._shared_arrays.append(self._volume_shared)
        self._shards = _partition_angle_shards(len(self.angles), self.gpu_ids)
        self._projection_buffers: dict[int, _SharedArray] = {}
        self._partial_volume_buffers: dict[int, _SharedArray] = {}

        try:
            for shard in self._shards:
                projection_buffer = _SharedArray.create((shard.count, self.projection_shape[1], self.projection_shape[2]), np.float32)
                partial_volume_buffer = _SharedArray.create(self.volume_shape, np.float32)
                self._shared_arrays.extend([projection_buffer, partial_volume_buffer])
                self._projection_buffers[shard.worker_id] = projection_buffer
                self._partial_volume_buffers[shard.worker_id] = partial_volume_buffer
                command_queue = self._ctx.Queue()
                process = self._ctx.Process(
                    target=_distributed_tigre_worker,
                    args=(
                        shard,
                        self.params,
                        np.ascontiguousarray(self.angles[shard.start : shard.stop], dtype=np.float32),
                        self._volume_shared.spec,
                        projection_buffer.spec,
                        partial_volume_buffer.spec,
                        command_queue,
                        self._result_queue,
                    ),
                    daemon=True,
                )
                process.start()
                self._commands.append(command_queue)
                self._processes.append(process)
            self._log(
                "Distributed exact full-data TIGRE enabled: "
                + ", ".join(
                    f"worker {shard.worker_id} -> GPU {shard.gpu_id}, views {shard.start}-{shard.stop - 1} ({shard.count})"
                    for shard in self._shards
                )
            )
        except Exception:
            self.close()
            raise

    def forward(self, volume: np.ndarray, cancel_check: Callable[[], bool] | None = None) -> np.ndarray:
        started_total = perf_counter()
        started_phase = perf_counter()
        self._copy_volume(volume)
        copy_volume_s = perf_counter() - started_phase
        started_phase = perf_counter()
        self._dispatch_to_all("forward")
        responses = self._wait_for_all("forward", cancel_check=cancel_check)
        wait_s = perf_counter() - started_phase
        started_phase = perf_counter()
        output = np.empty(self.projection_shape, dtype=np.float32)
        for shard in self._shards:
            output[shard.start : shard.stop] = self._projection_buffers[shard.worker_id].array()
        gather_s = perf_counter() - started_phase
        self._log_timing_summary(
            "forward",
            {
                "copy_volume_s": copy_volume_s,
                "wait_workers_s": wait_s,
                "gather_projection_s": gather_s,
                "total_s": perf_counter() - started_total,
            },
            responses,
        )
        return np.ascontiguousarray(output, dtype=np.float32)

    def backproject(self, projections: np.ndarray, cancel_check: Callable[[], bool] | None = None) -> np.ndarray:
        started_total = perf_counter()
        projection_array = np.ascontiguousarray(projections, dtype=np.float32)
        if projection_array.shape != self.projection_shape:
            raise ValueError(f"Distributed full-data backprojection expects shape {self.projection_shape}, got {projection_array.shape}.")
        started_phase = perf_counter()
        for shard in self._shards:
            np.copyto(
                self._projection_buffers[shard.worker_id].array(),
                projection_array[shard.start : shard.stop],
                casting="no",
            )
        copy_projection_s = perf_counter() - started_phase
        started_phase = perf_counter()
        self._dispatch_to_all("backproject")
        responses = self._wait_for_all("backproject", cancel_check=cancel_check)
        wait_s = perf_counter() - started_phase
        started_phase = perf_counter()
        volume = np.zeros(self.volume_shape, dtype=np.float32)
        for shard in self._shards:
            volume += self._partial_volume_buffers[shard.worker_id].array()
        reduce_s = perf_counter() - started_phase
        self._log_timing_summary(
            "backproject",
            {
                "copy_projection_s": copy_projection_s,
                "wait_workers_s": wait_s,
                "reduce_volume_s": reduce_s,
                "total_s": perf_counter() - started_total,
            },
            responses,
        )
        return np.ascontiguousarray(volume, dtype=np.float32)

    def normal(self, volume: np.ndarray, cancel_check: Callable[[], bool] | None = None) -> np.ndarray:
        started_total = perf_counter()
        started_phase = perf_counter()
        self._copy_volume(volume)
        copy_volume_s = perf_counter() - started_phase
        started_phase = perf_counter()
        self._dispatch_to_all("normal")
        responses = self._wait_for_all("normal", cancel_check=cancel_check)
        wait_s = perf_counter() - started_phase
        started_phase = perf_counter()
        result = np.zeros(self.volume_shape, dtype=np.float32)
        for shard in self._shards:
            result += self._partial_volume_buffers[shard.worker_id].array()
        reduce_s = perf_counter() - started_phase
        self._log_timing_summary(
            "normal",
            {
                "copy_volume_s": copy_volume_s,
                "wait_workers_s": wait_s,
                "reduce_volume_s": reduce_s,
                "total_s": perf_counter() - started_total,
            },
            responses,
        )
        return np.ascontiguousarray(result, dtype=np.float32)

    def data_residual_squared(
        self,
        volume: np.ndarray,
        projections: np.ndarray,
        cancel_check: Callable[[], bool] | None = None,
    ) -> float:
        started_total = perf_counter()
        projection_array = np.ascontiguousarray(projections, dtype=np.float32)
        if projection_array.shape != self.projection_shape:
            raise ValueError(
                f"Distributed full-data residual expects projections with shape {self.projection_shape}, got {projection_array.shape}."
            )
        started_phase = perf_counter()
        self._copy_volume(volume)
        copy_volume_s = perf_counter() - started_phase
        started_phase = perf_counter()
        for shard in self._shards:
            np.copyto(
                self._projection_buffers[shard.worker_id].array(),
                projection_array[shard.start : shard.stop],
                casting="no",
            )
        copy_projection_s = perf_counter() - started_phase
        started_phase = perf_counter()
        self._dispatch_to_all("residual_squared")
        responses = self._wait_for_all("residual_squared", cancel_check=cancel_check)
        wait_s = perf_counter() - started_phase
        started_phase = perf_counter()
        total_value = float(sum(float(response.get("value", 0.0)) for response in responses))
        reduce_s = perf_counter() - started_phase
        self._log_timing_summary(
            "residual_squared",
            {
                "copy_volume_s": copy_volume_s,
                "copy_projection_s": copy_projection_s,
                "wait_workers_s": wait_s,
                "accumulate_scalar_s": reduce_s,
                "total_s": perf_counter() - started_total,
            },
            responses,
        )
        return total_value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for command_queue in self._commands:
            try:
                command_queue.put({"op": "close"})
            except Exception:
                pass
        for process in self._processes:
            try:
                process.join(timeout=_PROCESS_JOIN_TIMEOUT_SECONDS)
            except Exception:
                pass
            if process.is_alive():
                try:
                    process.terminate()
                except Exception:
                    pass
                try:
                    process.join(timeout=1.0)
                except Exception:
                    pass
        for shared_array in reversed(self._shared_arrays):
            shared_array.close(unlink=True)

    def __del__(self) -> None:  # pragma: no cover - destructor timing is interpreter-dependent
        try:
            self.close()
        except Exception:
            pass

    def _dispatch_to_all(self, operation: str) -> None:
        for command_queue in self._commands:
            command_queue.put({"op": operation})

    def _wait_for_all(
        self,
        operation: str,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[dict[str, object]]:
        responses: list[dict[str, object]] = []
        pending = len(self._shards)
        cancellation_requested = False
        while pending > 0:
            if cancel_check is not None and cancel_check():
                cancellation_requested = True
            try:
                response = self._result_queue.get(timeout=_QUEUE_POLL_SECONDS)
            except Empty:
                dead_processes = [process.pid for process in self._processes if not process.is_alive()]
                if dead_processes:
                    self.close()
                    raise RuntimeError(
                        f"Distributed TIGRE worker process exited unexpectedly during {operation}. "
                        f"Dead worker PIDs: {dead_processes}"
                    )
                continue
            if not isinstance(response, dict):
                raise RuntimeError(f"Distributed TIGRE worker returned an invalid response during {operation}.")
            if response.get("status") != "ok":
                self.close()
                raise RuntimeError(
                    f"Distributed TIGRE worker failed during {operation} on GPU {response.get('gpu_id')}:\n"
                    f"{response.get('traceback', response.get('message', 'unknown error'))}"
                )
            responses.append(response)
            pending -= 1
        if cancellation_requested:
            raise OperationCancelled(
                "Distributed TIGRE operation reached a cancellation request after the current GPU call completed."
            )
        return responses

    def _copy_volume(self, volume: np.ndarray) -> None:
        volume_array = np.ascontiguousarray(volume, dtype=np.float32)
        if volume_array.shape != self.volume_shape:
            raise ValueError(f"Distributed full-data operator expects volume shape {self.volume_shape}, got {volume_array.shape}.")
        np.copyto(self._volume_shared.array(), volume_array, casting="no")

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)

    def _log_timing_summary(
        self,
        operation: str,
        host_timings: dict[str, float],
        responses: list[dict[str, object]],
    ) -> None:
        if self.logger is None:
            return
        host_text = ", ".join(
            f"{label}={float(value):.2f}s" for label, value in host_timings.items() if float(value) >= 0.0
        )
        worker_parts: list[str] = []
        for response in sorted(responses, key=lambda item: int(item.get("worker_id", 0))):
            timings = response.get("timings")
            if not isinstance(timings, dict):
                continue
            timing_text = ", ".join(
                f"{label}={float(value):.2f}s" for label, value in timings.items() if float(value) >= 0.0
            )
            worker_parts.append(f"GPU {int(response.get('gpu_id', -1))}: {timing_text}")
        suffix = f" | {' | '.join(worker_parts)}" if worker_parts else ""
        self._log(f"Distributed {operation} timing: {host_text}{suffix}")


def estimate_distributed_full_data_overhead_gb(
    projection_gb: float,
    volume_gb: float,
    gpu_count: int,
) -> DistributedFullDataMemoryEstimate:
    gpu_count = max(1, int(gpu_count))
    projection_gb = max(0.0, float(projection_gb))
    volume_gb = max(0.0, float(volume_gb))
    shared_buffers = projection_gb + (gpu_count + 1.0) * volume_gb
    transient_peak = projection_gb + (2.0 * gpu_count + 1.0) * volume_gb
    return DistributedFullDataMemoryEstimate(shared_buffers_gb=shared_buffers, transient_peak_gb=transient_peak)


def _partition_angle_shards(total_angles: int, gpu_ids: Sequence[int]) -> list[_WorkerShard]:
    gpu_ids = [int(gpu_id) for gpu_id in gpu_ids]
    worker_count = min(len(gpu_ids), max(1, int(total_angles)))
    sizes = [int(total_angles // worker_count)] * worker_count
    for index in range(int(total_angles) % worker_count):
        sizes[index] += 1
    shards: list[_WorkerShard] = []
    start = 0
    for worker_id in range(worker_count):
        stop = start + sizes[worker_id]
        shards.append(_WorkerShard(worker_id=worker_id, gpu_id=gpu_ids[worker_id], start=start, stop=stop))
        start = stop
    return shards


def _distributed_tigre_worker(
    shard: _WorkerShard,
    params: GeometryParams,
    local_angles: np.ndarray,
    volume_spec: _SharedArraySpec,
    projection_spec: _SharedArraySpec,
    partial_volume_spec: _SharedArraySpec,
    command_queue,
    result_queue,
) -> None:
    volume_shm = shared_memory.SharedMemory(name=volume_spec.name)
    projection_shm = shared_memory.SharedMemory(name=projection_spec.name)
    partial_shm = shared_memory.SharedMemory(name=partial_volume_spec.name)
    try:
        volume = np.ndarray(volume_spec.shape, dtype=volume_spec.dtype, buffer=volume_shm.buf)
        projection_buffer = np.ndarray(projection_spec.shape, dtype=projection_spec.dtype, buffer=projection_shm.buf)
        partial_volume = np.ndarray(partial_volume_spec.shape, dtype=partial_volume_spec.dtype, buffer=partial_shm.buf)

        import tigre
        from tigre.utilities.gpu import GpuIds

        gpuids = GpuIds()
        gpuids.devices = [int(shard.gpu_id)]
        geo = build_tigre_geometry(params)
        local_angles = np.ascontiguousarray(local_angles, dtype=np.float32)

        while True:
            command = command_queue.get()
            operation = str(command.get("op", "")).strip().lower()
            if operation == "close":
                break
            try:
                op_started = perf_counter()
                timings: dict[str, float] = {}
                if operation == "forward":
                    phase_started = perf_counter()
                    result = tigre.Ax(
                        np.ascontiguousarray(volume, dtype=np.float32),
                        geo,
                        local_angles,
                        gpuids=gpuids,
                    )
                    timings["ax_s"] = perf_counter() - phase_started
                    phase_started = perf_counter()
                    np.copyto(projection_buffer, np.asarray(result, dtype=np.float32), casting="no")
                    timings["copy_out_s"] = perf_counter() - phase_started
                    del result
                elif operation == "backproject":
                    phase_started = perf_counter()
                    result = tigre.Atb(
                        np.ascontiguousarray(projection_buffer, dtype=np.float32),
                        geo,
                        local_angles,
                        backprojection_type="matched",
                        gpuids=gpuids,
                    )
                    timings["atb_s"] = perf_counter() - phase_started
                    phase_started = perf_counter()
                    np.copyto(partial_volume, np.asarray(result, dtype=np.float32), casting="no")
                    timings["copy_out_s"] = perf_counter() - phase_started
                    del result
                elif operation == "normal":
                    phase_started = perf_counter()
                    local_projection = tigre.Ax(
                        np.ascontiguousarray(volume, dtype=np.float32),
                        geo,
                        local_angles,
                        gpuids=gpuids,
                    )
                    timings["ax_s"] = perf_counter() - phase_started
                    phase_started = perf_counter()
                    result = tigre.Atb(
                        np.ascontiguousarray(local_projection, dtype=np.float32),
                        geo,
                        local_angles,
                        backprojection_type="matched",
                        gpuids=gpuids,
                    )
                    timings["atb_s"] = perf_counter() - phase_started
                    phase_started = perf_counter()
                    np.copyto(partial_volume, np.asarray(result, dtype=np.float32), casting="no")
                    timings["copy_out_s"] = perf_counter() - phase_started
                    del local_projection
                    del result
                elif operation == "residual_squared":
                    phase_started = perf_counter()
                    local_projection = tigre.Ax(
                        np.ascontiguousarray(volume, dtype=np.float32),
                        geo,
                        local_angles,
                        gpuids=gpuids,
                    )
                    timings["ax_s"] = perf_counter() - phase_started
                    phase_started = perf_counter()
                    residual = np.asarray(local_projection, dtype=np.float32) - np.asarray(projection_buffer, dtype=np.float32)
                    value = float(np.sum(np.asarray(residual, dtype=np.float64) ** 2))
                    timings["residual_s"] = perf_counter() - phase_started
                    del local_projection
                    del residual
                    timings["total_s"] = perf_counter() - op_started
                    result_queue.put(
                        {
                            "status": "ok",
                            "op": operation,
                            "worker_id": int(shard.worker_id),
                            "gpu_id": int(shard.gpu_id),
                            "value": value,
                            "timings": timings,
                        }
                    )
                    gc.collect()
                    continue
                else:
                    raise ValueError(f"Unknown distributed TIGRE worker operation: {operation}")
                timings["total_s"] = perf_counter() - op_started
                gc.collect()
                result_queue.put(
                    {
                        "status": "ok",
                        "op": operation,
                        "worker_id": int(shard.worker_id),
                        "gpu_id": int(shard.gpu_id),
                        "timings": timings,
                    }
                )
            except Exception:
                result_queue.put(
                    {
                        "status": "error",
                        "op": operation,
                        "worker_id": int(shard.worker_id),
                        "gpu_id": int(shard.gpu_id),
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        volume_shm.close()
        projection_shm.close()
        partial_shm.close()
