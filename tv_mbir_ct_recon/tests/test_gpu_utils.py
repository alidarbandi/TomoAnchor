from __future__ import annotations

import pytest

from src.gpu_utils import format_gpu_selector, resolve_gpu_selection


def test_auto_selector_uses_all_detected_gpus() -> None:
    selection = resolve_gpu_selection(True, "auto", legacy_gpu_id=0, available_gpu_ids=[0, 1])
    assert selection.gpu_ids == (0, 1)
    assert selection.uses_multiple_gpus


def test_all_selector_falls_back_to_primary_when_detection_is_unavailable() -> None:
    selection = resolve_gpu_selection(True, "all", legacy_gpu_id=3, available_gpu_ids=[])
    assert selection.gpu_ids == (3,)
    assert selection.warnings


def test_explicit_selector_preserves_order_and_removes_duplicates() -> None:
    selection = resolve_gpu_selection(True, "1,0,1", legacy_gpu_id=0, available_gpu_ids=[0, 1, 2])
    assert selection.gpu_ids == (1, 0)


def test_single_selector_uses_legacy_primary_gpu() -> None:
    selection = resolve_gpu_selection(True, "single", legacy_gpu_id=2, available_gpu_ids=[0, 1, 2])
    assert selection.gpu_ids == (2,)


def test_explicit_selector_rejects_missing_gpu_ids() -> None:
    with pytest.raises(ValueError):
        resolve_gpu_selection(True, "0,4", legacy_gpu_id=0, available_gpu_ids=[0, 1])


def test_format_gpu_selector_round_trips_ids() -> None:
    assert format_gpu_selector((0, 1, 0)) == "0,1"
