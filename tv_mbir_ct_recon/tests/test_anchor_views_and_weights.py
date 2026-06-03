from __future__ import annotations

import numpy as np
import pandas as pd

from src.anchor_views import split_anchor_views
from src.config import AnchorSplitConfig
from src.fast_pipeline import _apply_drift_file, load_drift_shift_file
from src.projection_weights import (
    combine_duplicate_angle_projections,
    estimate_projection_scalar_weights,
    normalize_projection_weights,
)


def test_interleaved_anchor_split_is_deterministic_and_disjoint() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False, dtype=np.float32)
    cfg = AnchorSplitConfig(tune_count=3, qc_count=2, recon_count=4)

    first = split_anchor_views(angles, cfg)
    second = split_anchor_views(angles, cfg)

    assert np.array_equal(first.recon_indices, second.recon_indices)
    assert np.array_equal(first.tune_indices, second.tune_indices)
    assert np.array_equal(first.qc_indices, second.qc_indices)
    all_indices = np.concatenate([first.recon_indices, first.tune_indices, first.qc_indices])
    assert len(np.unique(all_indices)) == len(all_indices)


def test_duplicate_angle_combine_uses_inverse_variance_average() -> None:
    projections = np.asarray(
        [
            np.full((2, 2), 2.0, dtype=np.float32),
            np.full((2, 2), 10.0, dtype=np.float32),
            np.full((2, 2), 5.0, dtype=np.float32),
        ]
    )
    angles = np.asarray([0.0, 1.0e-5, 1.0], dtype=np.float32)
    weights = np.asarray([1.0, 3.0, 1.0], dtype=np.float32)

    merged, merged_angles, merged_weights, report = combine_duplicate_angle_projections(
        projections,
        angles,
        weights,
        angle_tolerance_rad=1.0e-4,
    )

    expected = (1.0 * 2.0 + 9.0 * 10.0) / 10.0
    assert merged.shape[0] == 2
    assert np.allclose(merged[0], expected)
    assert np.isclose(merged_weights[0], np.sqrt(10.0))
    assert len(report) == 1
    assert merged_angles.shape == (2,)


def test_projection_weights_normalize_main_median_and_cap_anchor() -> None:
    raw_main = np.ones((3, 2, 2), dtype=np.float32) * np.asarray([1, 4, 9], dtype=np.float32)[:, None, None]
    raw_anchor = np.ones((2, 2, 2), dtype=np.float32) * 10000.0
    main = estimate_projection_scalar_weights(raw_main, None, exposure_s=None)
    anchor = estimate_projection_scalar_weights(raw_anchor, None, exposure_s=None, fallback_multiplier=2.0)

    main_norm, anchor_norm = normalize_projection_weights(main, anchor, max_weight_ratio=5.0)

    assert np.isclose(np.median(main_norm), 1.0)
    assert anchor_norm is not None
    assert np.max(anchor_norm) <= 5.0


def test_drift_shift_file_accepts_alias_columns(tmp_path) -> None:
    path = tmp_path / "drift.csv"
    pd.DataFrame({"shift_x": [1.0, 2.0], "row_shift_px": [-1.0, -2.0]}).to_csv(path, index=False)

    x, y = load_drift_shift_file(path)

    assert np.allclose(x, [1.0, 2.0])
    assert np.allclose(y, [-1.0, -2.0])


def test_drift_file_drops_removed_duplicate_endpoint_row(tmp_path) -> None:
    path = tmp_path / "drift.csv"
    pd.DataFrame({"x_shift_px": [1.0, 2.0, 3.0], "y_shift_px": [-1.0, -2.0, -3.0]}).to_csv(path, index=False)
    validation = type(
        "Validation",
        (),
        {
            "records": [object(), object()],
            "duplicate_endpoint_removed": True,
            "x_shift_px": None,
            "y_shift_px": None,
            "x_shift_column_found": False,
            "y_shift_column_found": False,
            "x_shift_column": "",
            "y_shift_column": "",
            "warnings": [],
        },
    )()

    _apply_drift_file(validation, path, logger=None)

    assert np.allclose(validation.x_shift_px, [1.0, 2.0])
    assert np.allclose(validation.y_shift_px, [-1.0, -2.0])
