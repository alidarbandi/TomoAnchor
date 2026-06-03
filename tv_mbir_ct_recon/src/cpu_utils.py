from __future__ import annotations

import math
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

from .cancel_utils import raise_if_cancelled
from .memory_utils import query_system_memory


T = TypeVar("T")
R = TypeVar("R")

MAX_IO_WORKERS = 8
MAX_COMPUTE_WORKERS = 6
MIN_AVAILABLE_RAM_RESERVE_GB = 4.0
PARALLEL_RAM_FRACTION = 0.20


@dataclass(frozen=True)
class CPUParallelPlan:
    logical_cpus: int
    available_ram_gb: float | None
    io_workers: int
    compute_workers: int


def default_cpu_parallel_plan(image_shape: Sequence[int] | None = None) -> CPUParallelPlan:
    logical_cpus = max(1, int(os.cpu_count() or 1))
    reserved_cpus = 2 if logical_cpus >= 8 else 1 if logical_cpus >= 4 else 0
    usable_cpus = max(1, logical_cpus - reserved_cpus)
    io_workers = min(MAX_IO_WORKERS, usable_cpus)
    compute_workers = min(MAX_COMPUTE_WORKERS, usable_cpus)

    memory = query_system_memory()
    available_ram_gb = memory.available_gb if memory is not None else None
    memory_cap = _memory_capped_workers(image_shape, available_ram_gb)
    if memory_cap is not None:
        io_workers = min(io_workers, memory_cap)
        compute_workers = min(compute_workers, memory_cap)
    return CPUParallelPlan(
        logical_cpus=logical_cpus,
        available_ram_gb=available_ram_gb,
        io_workers=max(1, int(io_workers)),
        compute_workers=max(1, int(compute_workers)),
    )


def bounded_thread_map(
    fn: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    max_workers: int,
    max_pending_factor: int = 2,
    cancel_check: Callable[[], bool] | None = None,
) -> Iterator[tuple[int, R]]:
    sequence = list(items) if not isinstance(items, Sequence) else items
    if max_workers <= 1 or len(sequence) <= 1:
        for index, item in enumerate(sequence):
            raise_if_cancelled(cancel_check)
            yield index, fn(item)
        return

    max_workers = max(1, int(max_workers))
    max_pending = max(max_workers, max_workers * max(1, int(max_pending_factor)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending: dict[object, int] = {}
        next_index = 0

        def submit_until_full() -> None:
            nonlocal next_index
            while next_index < len(sequence) and len(pending) < max_pending:
                future = executor.submit(fn, sequence[next_index])
                pending[future] = next_index
                next_index += 1

        submit_until_full()
        while pending:
            raise_if_cancelled(cancel_check)
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                yield index, future.result()
            submit_until_full()


def cpu_parallel_report_lines(plan: CPUParallelPlan) -> list[str]:
    lines = [
        "",
        "CPU parallelism",
        "---------------",
        f"Detected logical CPUs: {plan.logical_cpus}",
        f"Projection/reference I/O workers: {plan.io_workers}",
        f"CPU compute workers: {plan.compute_workers}",
    ]
    if plan.available_ram_gb is not None:
        lines.append(f"Available system RAM at planning time: {plan.available_ram_gb:.2f} GB")
    else:
        lines.append("Available system RAM at planning time: unavailable")
    lines.append("Worker counts are capped conservatively to avoid oversubscribing CPU and RAM.")
    return lines


def _memory_capped_workers(image_shape: Sequence[int] | None, available_ram_gb: float | None) -> int | None:
    if image_shape is None or available_ram_gb is None:
        return None
    if len(image_shape) < 2:
        return None
    rows = max(1, int(image_shape[0]))
    cols = max(1, int(image_shape[1]))
    image_gb = float(rows * cols * 4) / 1024.0**3
    if image_gb <= 0:
        return None
    usable_ram_gb = min(float(available_ram_gb) * PARALLEL_RAM_FRACTION, max(0.0, float(available_ram_gb) - MIN_AVAILABLE_RAM_RESERVE_GB))
    if usable_ram_gb <= 0:
        return 1
    # Keep room for source/output arrays and library scratch buffers.
    return max(1, int(math.floor(usable_ram_gb / max(2.0 * image_gb, 1e-9))))
