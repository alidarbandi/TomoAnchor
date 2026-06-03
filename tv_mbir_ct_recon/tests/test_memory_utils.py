from __future__ import annotations

from src.memory_utils import (
    estimate_low_memory_pdhg_cpu_gb,
    estimate_streaming_subset_tv_cpu_gb,
    estimate_mbir_memory,
    minimum_available_system_memory_gb,
    recommended_projection_batch_size,
)


def test_streamed_gpu_estimate_uses_batch_size() -> None:
    full = estimate_mbir_memory(1600, (1024, 1024), (256, 256, 256))
    streamed = estimate_mbir_memory(1600, (1024, 1024), (256, 256, 256), projection_batch_size=32)
    assert streamed.projection_gb == full.projection_gb
    assert streamed.volume_gb == full.volume_gb
    assert streamed.gpu_operator_estimate_gb < full.gpu_operator_estimate_gb


def test_one_view_1024_cubed_streaming_estimate_fits_8gb_gpu() -> None:
    estimate = estimate_mbir_memory(400, (1024, 1024), (1024, 1024, 1024), projection_batch_size=1)
    assert estimate.volume_gb == 4.0
    assert estimate.gpu_operator_estimate_gb < 8.0


def test_full_stack_1024_cubed_estimate_still_warns_for_8gb_gpu() -> None:
    estimate = estimate_mbir_memory(400, (1024, 1024), (1024, 1024, 1024))
    assert estimate.projection_gb > 1.0
    assert estimate.gpu_operator_estimate_gb > 8.0


def test_recommended_batch_is_conservative_for_1024_cubed_on_8gb_gpu() -> None:
    recommended = recommended_projection_batch_size(
        400,
        (1024, 1024),
        (1024, 1024, 1024),
        gpu_free_gb=7.77,
    )
    assert recommended == 1


def test_multi_gpu_streaming_estimate_reduces_per_gpu_projection_pressure() -> None:
    single_gpu = estimate_mbir_memory(400, (1024, 1024), (512, 512, 512), projection_batch_size=64, gpu_count=1)
    dual_gpu = estimate_mbir_memory(400, (1024, 1024), (512, 512, 512), projection_batch_size=64, gpu_count=2)
    assert dual_gpu.gpu_operator_estimate_gb < single_gpu.gpu_operator_estimate_gb


def test_recommended_batch_scales_up_when_projection_work_is_split_across_gpus() -> None:
    single_gpu = recommended_projection_batch_size(
        400,
        (1024, 1024),
        (512, 512, 512),
        gpu_free_gb=20.0,
        gpu_count=1,
    )
    dual_gpu = recommended_projection_batch_size(
        400,
        (1024, 1024),
        (512, 512, 512),
        gpu_free_gb=20.0,
        gpu_count=2,
    )
    assert dual_gpu >= single_gpu


def test_minimum_available_system_memory_includes_tigre_atb_transient() -> None:
    estimate = estimate_mbir_memory(400, (1024, 1024), (1024, 1024, 1024), projection_batch_size=1)
    assert minimum_available_system_memory_gb(estimate) > 9.0


def test_low_memory_pdhg_estimate_is_much_smaller_than_admm() -> None:
    estimate = estimate_mbir_memory(400, (1024, 1024), (1024, 1024, 1024), projection_batch_size=1)
    low_memory = estimate_low_memory_pdhg_cpu_gb(estimate.projection_gb, estimate.volume_gb)
    assert low_memory < 32.0
    assert low_memory < 0.5 * estimate.cpu_working_set_gb


def test_streaming_subset_tv_estimate_is_smaller_than_pdhg() -> None:
    estimate = estimate_mbir_memory(400, (1024, 1024), (1024, 1024, 1024), projection_batch_size=1)
    low_memory = estimate_low_memory_pdhg_cpu_gb(estimate.projection_gb, estimate.volume_gb)
    streaming = estimate_streaming_subset_tv_cpu_gb(estimate.projection_gb, estimate.volume_gb)
    assert streaming < 16.0
    assert streaming < low_memory
