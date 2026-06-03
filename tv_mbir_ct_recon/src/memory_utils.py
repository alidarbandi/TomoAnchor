from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
import ctypes
from dataclasses import dataclass


BYTES_PER_FLOAT32 = 4
TIGRE_VOLUME_WORKING_FACTOR = 1.35
TIGRE_PROJECTION_WORKING_FACTOR = 2.0
TIGRE_BACKPROJECTION_KERNEL_VIEWS = 32
TIGRE_GPU_SAFETY_FRACTION = 0.70
SYSTEM_RAM_SAFETY_FRACTION = 0.85


@dataclass(frozen=True)
class MBIRMemoryEstimate:
    projection_gb: float
    volume_gb: float
    cpu_working_set_gb: float
    gpu_operator_estimate_gb: float

    def report_lines(self) -> list[str]:
        return [
            "MBIR memory estimate",
            "--------------------",
            f"Projection stack float32: {self.projection_gb:.2f} GB",
            f"Single volume float32: {self.volume_gb:.2f} GB",
            f"Estimated MBIR CPU working arrays: {self.cpu_working_set_gb:.2f} GB",
            f"Estimated low-memory PDHG CPU working arrays: {estimate_low_memory_pdhg_cpu_gb(self.projection_gb, self.volume_gb):.2f} GB",
            f"Estimated streaming subset-TV CPU working arrays: {estimate_streaming_subset_tv_cpu_gb(self.projection_gb, self.volume_gb):.2f} GB",
            f"Estimated TIGRE GPU operator working set: {self.gpu_operator_estimate_gb:.2f} GB",
            "CPU estimate is the full ADMM/CG working set and must fit in physical system RAM.",
            "Low-memory PDHG avoids the ADMM/CG volume pile-up and is selected automatically when ADMM is too large.",
            "Streaming subset-TV updates one projection batch at a time and keeps the smallest CPU working set.",
            "GPU estimate is the transient peak for one TIGRE Ax/Atb call, not all MBIR CPU arrays.",
            "GPU estimate uses the configured projection batch size when projection streaming is enabled.",
            "These are conservative estimates; TIGRE/CUDA may need additional internal buffers.",
        ]


@dataclass(frozen=True)
class GPUMemoryInfo:
    gpu_id: int
    name: str
    total_gb: float
    free_gb: float


@dataclass(frozen=True)
class SystemMemoryInfo:
    total_gb: float
    available_gb: float


def estimate_mbir_memory(
    num_projections: int,
    detector_shape: tuple[int, int],
    volume_shape: tuple[int, int, int],
    projection_batch_size: int | None = None,
    gpu_count: int = 1,
) -> MBIRMemoryEstimate:
    rows, cols = [int(v) for v in detector_shape]
    nz, ny, nx = [int(v) for v in volume_shape]
    projection_elems = max(0, int(num_projections)) * max(rows, 0) * max(cols, 0)
    volume_elems = max(nz, 0) * max(ny, 0) * max(nx, 0)
    projection_gb = _bytes_to_gb(projection_elems * BYTES_PER_FLOAT32)
    volume_gb = _bytes_to_gb(volume_elems * BYTES_PER_FLOAT32)

    # ADMM+CG keeps several full-volume arrays alive: x, x_previous, rhs, atb,
    # CG r/p/ap, TV gradients, split variables, and Bregman variables.
    cpu_working_set_gb = 18.0 * volume_gb + 3.0 * projection_gb

    effective_projection_count = int(num_projections)
    if projection_batch_size is not None and int(projection_batch_size) > 0:
        effective_projection_count = min(effective_projection_count, int(projection_batch_size))
    if projection_batch_size is not None:
        effective_projection_count = min(
            int(num_projections),
            max(effective_projection_count, min(int(num_projections), TIGRE_BACKPROJECTION_KERNEL_VIEWS)),
        )
    gpu_count = max(1, int(gpu_count))
    effective_projection_count_per_gpu = int(math.ceil(effective_projection_count / float(gpu_count)))
    effective_projection_gb = _bytes_to_gb(
        effective_projection_count_per_gpu * max(rows, 0) * max(cols, 0) * BYTES_PER_FLOAT32
    )

    gpu_operator_estimate_gb = estimate_tigre_operator_gpu_gb(volume_gb, effective_projection_gb)
    return MBIRMemoryEstimate(projection_gb, volume_gb, cpu_working_set_gb, gpu_operator_estimate_gb)


def estimate_tigre_operator_gpu_gb(volume_gb: float, projection_gb: float) -> float:
    # TIGRE Ax and Atb are called sequentially in the MBIR normal operator, so
    # the GPU peak is a single operator allocation rather than the sum of both.
    # Keep extra headroom for CUDA texture/output buffers and allocator rounding.
    return (
        TIGRE_VOLUME_WORKING_FACTOR * max(0.0, float(volume_gb))
        + TIGRE_PROJECTION_WORKING_FACTOR * max(0.0, float(projection_gb))
    )


def estimate_low_memory_pdhg_cpu_gb(projection_gb: float, volume_gb: float, dual_dtype: str = "float16") -> float:
    dual_scale = 0.5 if str(dual_dtype).lower() in {"float16", "half", "fp16"} else 1.0
    # x, extrapolated x, x-old/scratch, A^T dual-data gradient, TV dual field,
    # input projections, data dual, and one forward-projection work array.
    return (4.0 + 3.0 * dual_scale) * max(0.0, float(volume_gb)) + 3.0 * max(0.0, float(projection_gb))


def estimate_streaming_subset_tv_cpu_gb(projection_gb: float, volume_gb: float) -> float:
    # x plus one TIGRE backprojection/update volume, the resident projection stack,
    # and transient batch/TV chunk buffers. This solver updates after each batch
    # instead of holding a full projection dual or a full accumulated gradient.
    return 2.35 * max(0.0, float(volume_gb)) + 1.25 * max(0.0, float(projection_gb))


def recommended_projection_batch_size(
    num_projections: int,
    detector_shape: tuple[int, int],
    volume_shape: tuple[int, int, int],
    gpu_free_gb: float,
    gpu_count: int = 1,
    safety_fraction: float = TIGRE_GPU_SAFETY_FRACTION,
) -> int:
    """Choose a projection batch size that leaves headroom for CUDA internals."""
    rows, cols = [max(0, int(v)) for v in detector_shape]
    projection_count = max(1, int(num_projections))
    gpu_count = max(1, int(gpu_count))
    per_view_gb = _bytes_to_gb(rows * cols * BYTES_PER_FLOAT32)
    if per_view_gb <= 0:
        return projection_count

    nz, ny, nx = [max(0, int(v)) for v in volume_shape]
    volume_gb = _bytes_to_gb(nz * ny * nx * BYTES_PER_FLOAT32)
    minimum_projection_views_per_gpu = int(
        math.ceil(min(projection_count, TIGRE_BACKPROJECTION_KERNEL_VIEWS) / float(gpu_count))
    )
    minimum_projection_gb = _bytes_to_gb(
        minimum_projection_views_per_gpu * rows * cols * BYTES_PER_FLOAT32
    )
    target_gb = max(0.0, float(gpu_free_gb) * float(safety_fraction))
    projection_budget_gb = max(
        0.0,
        (target_gb - TIGRE_VOLUME_WORKING_FACTOR * volume_gb - TIGRE_PROJECTION_WORKING_FACTOR * minimum_projection_gb)
        / TIGRE_PROJECTION_WORKING_FACTOR,
    )
    recommended = gpu_count * int(math.floor(projection_budget_gb / per_view_gb))
    return max(1, min(projection_count, recommended))


def minimum_available_system_memory_gb(estimate: MBIRMemoryEstimate) -> float:
    # TIGRE Atb allocates a host output volume and page-locks host buffers.
    # Keep room for that transient allocation even before the full ADMM peak.
    return max(4.0, 2.0 * estimate.volume_gb + estimate.projection_gb)


def query_all_nvidia_gpu_memory() -> list[GPUMemoryInfo]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    gpu_infos: list[GPUMemoryInfo] = []
    for raw_line in completed.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 4:
            continue
        try:
            index = int(parts[0])
            total_gb = float(parts[2]) / 1024.0
            free_gb = float(parts[3]) / 1024.0
        except ValueError:
            continue
        gpu_infos.append(GPUMemoryInfo(index, parts[1], total_gb, free_gb))
    return gpu_infos


def query_nvidia_gpu_memory(gpu_id: int = 0) -> GPUMemoryInfo | None:
    for info in query_all_nvidia_gpu_memory():
        if info.gpu_id == int(gpu_id):
            return info
    return None


def query_system_memory() -> SystemMemoryInfo | None:
    if platform.system().lower() == "windows":
        return _query_windows_system_memory()
    return _query_posix_system_memory()


def _query_windows_system_memory() -> SystemMemoryInfo | None:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
    except Exception:
        return None
    if not ok:
        return None
    return SystemMemoryInfo(_bytes_to_gb(status.ullTotalPhys), _bytes_to_gb(status.ullAvailPhys))


def _query_posix_system_memory() -> SystemMemoryInfo | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    total_gb = _bytes_to_gb(float(page_size) * float(phys_pages))
    available_gb = _read_meminfo_available_gb()
    if available_gb is None:
        try:
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            available_gb = _bytes_to_gb(float(page_size) * float(available_pages))
        except (AttributeError, OSError, ValueError):
            available_gb = total_gb
    return SystemMemoryInfo(total_gb, available_gb)


def _read_meminfo_available_gb() -> float | None:
    path = "/proc/meminfo"
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return _bytes_to_gb(float(parts[1]) * 1024.0)
    except OSError:
        return None
    return None


def _bytes_to_gb(num_bytes: float) -> float:
    return float(num_bytes) / 1024.0**3
