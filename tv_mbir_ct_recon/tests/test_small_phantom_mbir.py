from __future__ import annotations

import numpy as np

from src.admm_tv_mbir import ADMMTVMBIRReconstructor
from src.config import MBIRConfig
from src.pdhg_tv_mbir import PDHGTVMBIRReconstructor
from src.subset_tv_mbir import StreamingSubsetTVReconstructor


class IdentityOperator:
    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.volume_shape = shape
        self.projection_shape = shape

    def forward(self, x):
        return np.asarray(x, dtype=np.float32)

    def backproject(self, y):
        return np.asarray(y, dtype=np.float32)

    def normal(self, x):
        return np.asarray(x, dtype=np.float32)

    def data_gradient_batches(self, x, b, scale_to_full=True):
        yield np.asarray(x, dtype=np.float32) - np.asarray(b, dtype=np.float32), 1, 1

    def data_residual_squared(self, x, b):
        residual = np.asarray(x, dtype=np.float32) - np.asarray(b, dtype=np.float32)
        return float(np.sum(np.asarray(residual, dtype=np.float64) ** 2))


def test_small_phantom_identity_mbir_smoke() -> None:
    phantom = np.zeros((6, 6, 6), dtype=np.float32)
    phantom[2:4, 2:4, 2:4] = 1.0
    cfg = MBIRConfig(lambda_tv=1e-3, rho=0.1, max_admm_iterations=3, inner_cg_iterations=8)
    result = ADMMTVMBIRReconstructor(IdentityOperator(phantom.shape), cfg).reconstruct(phantom)
    assert result.volume.shape == phantom.shape
    assert np.all(result.volume >= 0.0)
    assert result.metrics


def test_admm_reports_inner_cg_progress() -> None:
    phantom = np.zeros((6, 6, 6), dtype=np.float32)
    phantom[2:4, 2:4, 2:4] = 1.0
    cfg = MBIRConfig(lambda_tv=1e-3, rho=0.1, max_admm_iterations=1, inner_cg_iterations=3)
    logs: list[str] = []
    events: list[object] = []

    def progress_callback(payload, _volume) -> None:
        events.append(payload)

    result = ADMMTVMBIRReconstructor(
        IdentityOperator(phantom.shape),
        cfg,
        logger=logs.append,
        progress_callback=progress_callback,
    ).reconstruct(phantom)

    assert result.metrics
    assert any("ADMM 001 CG 1/3" in line for line in logs)
    assert any("cg_iter=" in line for line in logs)
    assert any(isinstance(event, dict) and event.get("kind") == "cg_progress" for event in events)


def test_small_phantom_identity_low_memory_pdhg_smoke() -> None:
    phantom = np.zeros((6, 6, 6), dtype=np.float32)
    phantom[2:4, 2:4, 2:4] = 1.0
    cfg = MBIRConfig(
        solver="pdhg_low_memory",
        lambda_tv=1e-3,
        max_admm_iterations=3,
        pdhg_power_iterations=2,
        pdhg_dual_dtype="float16",
        pdhg_chunk_slices=2,
    )
    result = PDHGTVMBIRReconstructor(IdentityOperator(phantom.shape), cfg).reconstruct(phantom)
    assert result.volume.shape == phantom.shape
    assert np.all(result.volume >= 0.0)
    assert result.metrics


def test_small_phantom_identity_streaming_subset_tv_smoke() -> None:
    phantom = np.zeros((6, 6, 6), dtype=np.float32)
    phantom[2:4, 2:4, 2:4] = 1.0
    cfg = MBIRConfig(
        solver="streaming_subset_tv",
        lambda_tv=1e-3,
        max_admm_iterations=3,
        subset_tv_power_iterations=2,
        pdhg_chunk_slices=2,
    )
    result = StreamingSubsetTVReconstructor(IdentityOperator(phantom.shape), cfg).reconstruct(phantom)
    assert result.volume.shape == phantom.shape
    assert np.all(result.volume >= 0.0)
    assert result.metrics
