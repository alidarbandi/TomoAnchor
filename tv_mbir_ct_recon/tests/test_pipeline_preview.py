from __future__ import annotations

import numpy as np

from src.pipeline import extract_preview_slice_for_gui, preview_slice_counts_for_shape


def test_preview_slice_counts_for_shape_reports_orientation_lengths() -> None:
    counts = preview_slice_counts_for_shape((24, 18, 12))

    assert counts == {"Axial": 24, "Coronal": 18, "Sagittal": 12}


def test_extract_preview_slice_for_gui_returns_requested_full_resolution_slice() -> None:
    volume = np.arange(24 * 18 * 12, dtype=np.float32).reshape(24, 18, 12)

    image, view, slice_index, slice_counts = extract_preview_slice_for_gui(
        volume,
        {"view": "coronal", "slice_index": 7},
    )

    assert view == "Coronal"
    assert slice_index == 7
    assert image.shape == (24, 12)
    assert np.array_equal(image, volume[:, 7, :])
    assert slice_counts == {"Axial": 24, "Coronal": 18, "Sagittal": 12}


def test_extract_preview_slice_for_gui_clamps_slice_index() -> None:
    volume = np.arange(10 * 8 * 6, dtype=np.float32).reshape(10, 8, 6)

    image, view, slice_index, _slice_counts = extract_preview_slice_for_gui(
        volume,
        {"view": "sagittal", "slice_index": 1000},
    )

    assert view == "Sagittal"
    assert slice_index == 5
    assert np.array_equal(image, volume[:, :, 5])
