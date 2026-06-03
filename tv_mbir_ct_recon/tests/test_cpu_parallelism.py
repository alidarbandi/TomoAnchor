from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.alignment_drift import apply_projection_shifts
from src.cancel_utils import OperationCancelled
from src.cpu_utils import CPUParallelPlan, bounded_thread_map
from src.io_utils import load_projection_stack
from src.reference_images import compute_average_image


def test_bounded_thread_map_returns_all_items() -> None:
    results = list(bounded_thread_map(lambda value: value * value, [0, 1, 2, 3], max_workers=2))
    assert sorted(results) == [(0, 0), (1, 1), (2, 4), (3, 9)]


def test_bounded_thread_map_honors_cancellation() -> None:
    with pytest.raises(OperationCancelled):
        list(bounded_thread_map(lambda value: value, [0, 1, 2], max_workers=2, cancel_check=lambda: True))


def test_load_projection_stack_parallel_matches_expected(tmp_path: Path, monkeypatch) -> None:
    from src import io_utils

    monkeypatch.setattr(
        io_utils,
        "default_cpu_parallel_plan",
        lambda image_shape=None: CPUParallelPlan(logical_cpus=8, available_ram_gb=16.0, io_workers=2, compute_workers=2),
    )
    records = []
    for index in range(5):
        image = np.arange(16, dtype=np.float32).reshape(4, 4) + index
        path = tmp_path / f"projection_{index:03d}.tif"
        tifffile.imwrite(str(path), image)
        records.append(type("Record", (), {"path": path, "filename": path.name})())

    stack = load_projection_stack(records, binning=(2, 2), crop=(0, 0, 0, 0))
    expected = np.stack(
        [
            np.array([[2.5, 4.5], [10.5, 12.5]], dtype=np.float32) + index
            for index in range(5)
        ],
        axis=0,
    )
    np.testing.assert_allclose(stack, expected)


def test_load_projection_stack_raises_when_cancelled(tmp_path: Path) -> None:
    records = []
    for index in range(2):
        image = np.arange(16, dtype=np.float32).reshape(4, 4) + index
        path = tmp_path / f"projection_{index:03d}.tif"
        tifffile.imwrite(str(path), image)
        records.append(type("Record", (), {"path": path, "filename": path.name})())
    with pytest.raises(OperationCancelled):
        load_projection_stack(records, cancel_check=lambda: True)


def test_compute_average_image_parallel_matches_expected(tmp_path: Path, monkeypatch) -> None:
    from src import reference_images

    monkeypatch.setattr(
        reference_images,
        "default_cpu_parallel_plan",
        lambda image_shape=None: CPUParallelPlan(logical_cpus=8, available_ram_gb=16.0, io_workers=2, compute_workers=2),
    )
    expected_images = []
    for index in range(5):
        image = np.full((4, 4), float(index + 1), dtype=np.float32)
        expected_images.append(image)
        path = tmp_path / f"flat_{index:03d}.tif"
        tifffile.imwrite(str(path), image)

    average, count = compute_average_image(tmp_path, expected_raw_shape=(4, 4), binning=1, crop=(0, 0, 0, 0))
    assert count == 5
    np.testing.assert_allclose(average, np.mean(expected_images, axis=0))


def test_apply_projection_shifts_parallel_matches_serial(monkeypatch) -> None:
    from src import alignment_drift

    stack = np.zeros((5, 6, 6), dtype=np.float32)
    stack[:, 2:4, 2:4] = 1.0
    x_shift = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    y_shift = np.array([0.0, -0.5, 0.5, -1.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(
        alignment_drift,
        "default_cpu_parallel_plan",
        lambda image_shape=None: CPUParallelPlan(logical_cpus=8, available_ram_gb=16.0, io_workers=2, compute_workers=2),
    )
    parallel = apply_projection_shifts(stack, x_shift, y_shift)

    monkeypatch.setattr(
        alignment_drift,
        "default_cpu_parallel_plan",
        lambda image_shape=None: CPUParallelPlan(logical_cpus=1, available_ram_gb=16.0, io_workers=1, compute_workers=1),
    )
    serial = apply_projection_shifts(stack, x_shift, y_shift)
    np.testing.assert_allclose(parallel, serial)


def test_apply_projection_shifts_raises_when_cancelled() -> None:
    stack = np.zeros((5, 6, 6), dtype=np.float32)
    x_shift = np.zeros(5, dtype=np.float32)
    y_shift = np.zeros(5, dtype=np.float32)
    with pytest.raises(OperationCancelled):
        apply_projection_shifts(stack, x_shift, y_shift, cancel_check=lambda: True)
