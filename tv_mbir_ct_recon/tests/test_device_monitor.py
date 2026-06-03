from __future__ import annotations

import pytest

from src.device_monitor import format_gpu_snapshot_summary, parse_nvidia_smi_snapshot


def test_parse_nvidia_smi_snapshot_parses_gpu_metrics() -> None:
    snapshot = parse_nvidia_smi_snapshot(
        "0, NVIDIA RTX A5000, 8, 11, 1049, 24564, 42, 20.54\n"
        "1, NVIDIA RTX A5000, 77, 63, 12000, 24564, 68, 205.12\n",
        sampled_at_s=12.5,
    )

    assert snapshot.timestamp_s == pytest.approx(12.5)
    assert len(snapshot.gpus) == 2
    assert snapshot.gpus[0].gpu_id == 0
    assert snapshot.gpus[0].utilization_gpu_pct == pytest.approx(8.0)
    assert snapshot.gpus[0].utilization_memory_pct == pytest.approx(11.0)
    assert snapshot.gpus[0].memory_used_gb == pytest.approx(1049.0 / 1024.0)
    assert snapshot.gpus[0].memory_total_gb == pytest.approx(24564.0 / 1024.0)
    assert snapshot.gpus[0].memory_percent == pytest.approx(100.0 * 1049.0 / 24564.0)
    assert snapshot.gpus[1].temperature_c == pytest.approx(68.0)
    assert snapshot.gpus[1].power_w == pytest.approx(205.12)


def test_parse_nvidia_smi_snapshot_handles_missing_optional_values() -> None:
    snapshot = parse_nvidia_smi_snapshot(
        "0, Test GPU, N/A, [Not Supported], 512, 1024, N/A, N/A\n",
        sampled_at_s=1.0,
    )

    assert len(snapshot.gpus) == 1
    sample = snapshot.gpus[0]
    assert sample.utilization_gpu_pct is None
    assert sample.utilization_memory_pct is None
    assert sample.temperature_c is None
    assert sample.power_w is None
    assert sample.memory_percent == pytest.approx(50.0)


def test_format_gpu_snapshot_summary_filters_selected_gpu_ids() -> None:
    snapshot = parse_nvidia_smi_snapshot(
        "0, GPU 0, 3, 4, 200, 1000, 30, 10.0\n"
        "1, GPU 1, 91, 88, 900, 1000, 71, 215.5\n",
        sampled_at_s=5.0,
    )

    summary = format_gpu_snapshot_summary(snapshot, gpu_ids=[1])

    assert "GPU 1" in summary
    assert "GPU 0" not in summary
    assert "91% util" in summary
    assert "0.9/1.0 GB (90%)" in summary
