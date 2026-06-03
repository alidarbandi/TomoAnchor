from __future__ import annotations

import numpy as np

from src.distributed_tigre import estimate_distributed_full_data_overhead_gb
from src.geometry_tigre import GeometryParams
from src.tigre_ops import TigreConeBeamOperator


def test_distributed_full_data_memory_overhead_grows_with_gpu_count() -> None:
    dual = estimate_distributed_full_data_overhead_gb(1.5, 4.0, gpu_count=2)
    quad = estimate_distributed_full_data_overhead_gb(1.5, 4.0, gpu_count=4)

    assert dual.shared_buffers_gb > 0.0
    assert dual.transient_peak_gb > dual.shared_buffers_gb
    assert quad.shared_buffers_gb > dual.shared_buffers_gb
    assert quad.transient_peak_gb > dual.transient_peak_gb


def test_distributed_full_data_operator_routes_exact_calls_to_pool(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []

    class FakePool:
        def __init__(self, params, angles_rad, gpu_ids, logger=None):
            self.gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)
            calls.append(("init", self.gpu_ids))

        def forward(self, volume, cancel_check=None):
            calls.append(("forward", tuple(np.asarray(volume).shape)))
            return np.full((4, 2, 2), 3.0, dtype=np.float32)

        def backproject(self, projections, cancel_check=None):
            calls.append(("backproject", tuple(np.asarray(projections).shape)))
            return np.full((2, 2, 2), 5.0, dtype=np.float32)

        def normal(self, volume, cancel_check=None):
            calls.append(("normal", tuple(np.asarray(volume).shape)))
            return np.full((2, 2, 2), 7.0, dtype=np.float32)

        def data_residual_squared(self, volume, projections, cancel_check=None):
            calls.append(("residual", tuple(np.asarray(projections).shape)))
            return 11.0

        def close(self):
            calls.append(("close", ()))

    monkeypatch.setattr("src.tigre_ops.DistributedFullDataTigrePool", FakePool)
    monkeypatch.setattr("src.tigre_ops.build_tigre_geometry", lambda params: object())
    monkeypatch.setattr("src.tigre_ops.make_tigre_gpuids", lambda gpu_ids=None, use_gpu=True: object())

    params = GeometryParams(
        source_detector_distance_mm=200.0,
        source_origin_distance_mm=100.0,
        detector_pixel_size_mm=(0.02, 0.02),
        detector_pixels=(2, 2),
        voxel_size_mm=(0.02, 0.02, 0.02),
        volume_voxels=(2, 2, 2),
    )
    angles = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    operator = TigreConeBeamOperator(
        params,
        angles,
        gpu_ids=(0, 1),
        use_gpu=True,
        memory_mode="distributed_full_data",
    )

    x = np.ones((2, 2, 2), dtype=np.float32)
    y = np.ones((4, 2, 2), dtype=np.float32)

    forward = operator.forward(x)
    backproject = operator.backproject(y)
    normal = operator.normal(x)
    residual = operator.data_residual_squared(x, y)
    operator.close()

    assert operator.memory_mode == "distributed_full_data"
    np.testing.assert_allclose(forward, 3.0)
    np.testing.assert_allclose(backproject, 5.0)
    np.testing.assert_allclose(normal, 7.0)
    assert residual == 11.0
    assert ("init", (0, 1)) in calls
    assert ("forward", (2, 2, 2)) in calls
    assert ("backproject", (4, 2, 2)) in calls
    assert ("normal", (2, 2, 2)) in calls
    assert ("residual", (4, 2, 2)) in calls
    assert ("close", ()) in calls


def test_distributed_full_data_falls_back_to_full_gpu_without_multiple_gpus(monkeypatch) -> None:
    logged: list[str] = []

    monkeypatch.setattr("src.tigre_ops.build_tigre_geometry", lambda params: object())
    monkeypatch.setattr("src.tigre_ops.make_tigre_gpuids", lambda gpu_ids=None, use_gpu=True: object())

    params = GeometryParams(
        source_detector_distance_mm=200.0,
        source_origin_distance_mm=100.0,
        detector_pixel_size_mm=(0.02, 0.02),
        detector_pixels=(2, 2),
        voxel_size_mm=(0.02, 0.02, 0.02),
        volume_voxels=(2, 2, 2),
    )
    angles = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    operator = TigreConeBeamOperator(
        params,
        angles,
        gpu_ids=(0,),
        use_gpu=True,
        memory_mode="distributed_full_data",
        logger=logged.append,
    )

    assert operator.memory_mode == "full_gpu"
    assert any("Falling back to full_gpu" in message for message in logged)
