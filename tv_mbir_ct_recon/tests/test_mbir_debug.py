from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import AlignmentConfig, GeometryConfig, PreprocessingConfig
from src.geometry_tigre import GeometryParams
from src.mbir_debug import (
    alignment_for_mbir_debug,
    effective_detector_shape,
    effective_projection_count,
    effective_volume_shape,
    geometry_for_mbir_debug,
    preprocessing_for_mbir_debug,
    validation_for_mbir_debug,
)
from src.metadata_zeiss import MetadataValidation, ProjectionRecord
from src.tigre_ops import fdk_initialization


def test_mbir_projection_stride_and_shift_scaling() -> None:
    validation = _validation(5)

    sampled = validation_for_mbir_debug(validation, pixel_binning=(2, 4), projection_stride=2)

    assert [record.index for record in sampled.records] == [0, 2, 4]
    np.testing.assert_allclose(sampled.angles_rad, np.deg2rad([0.0, 20.0, 40.0]).astype(np.float32))
    np.testing.assert_allclose(sampled.x_shift_px, np.array([0.0, 2.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(sampled.y_shift_px, np.array([0.0, 2.0, 4.0], dtype=np.float32))


def test_mbir_extra_binning_adjusts_preprocessing_crop() -> None:
    preprocessing = PreprocessingConfig(binning=(2, 3), crop=(5, 6, 7, 8))

    effective = preprocessing_for_mbir_debug(preprocessing, pixel_binning=(2, 4))

    assert effective.binning == (4, 12)
    assert effective.crop == (3, 3, 2, 2)


def test_mbir_debug_effective_shapes() -> None:
    assert effective_projection_count(400, 3) == 134
    assert effective_detector_shape((1024, 768), (2, 3)) == (512, 256)
    assert effective_volume_shape((1024, 768, 768), (2, 3)) == (512, 256, 256)


def test_mbir_debug_geometry_scales_grid_and_pixel_sizes() -> None:
    geometry = GeometryConfig(
        source_detector_distance_mm=200.0,
        source_origin_distance_mm=100.0,
        effective_pixel_size_um=10.0,
        detector_pixel_size_mm=(None, None),
        detector_pixels=(100, 150),
        voxel_size_mm=(0.01, 0.01, 0.01),
        volume_voxels=(100, 200, 300),
        detector_offset_pixels=(4.0, 9.0),
    )

    effective = geometry_for_mbir_debug(geometry, pixel_binning=(2, 3), detector_shape=(50, 75))
    alignment = alignment_for_mbir_debug(AlignmentConfig(center_offset_pixels=12.0), pixel_binning=(2, 3))

    assert effective.detector_pixels == (50, 75)
    np.testing.assert_allclose(effective.detector_pixel_size_mm, (0.04, 0.06))
    np.testing.assert_allclose(effective.voxel_size_mm, (0.02, 0.03, 0.03))
    assert effective.volume_voxels == (50, 66, 100)
    np.testing.assert_allclose(effective.detector_offset_pixels, (2.0, 3.0))
    assert alignment.center_offset_pixels == 4.0


def test_existing_fdk_initialization_downsamples_to_mbir_grid(tmp_path: Path) -> None:
    source = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    path = tmp_path / "fdk.npy"
    np.save(path, source)
    params = GeometryParams(
        source_detector_distance_mm=200.0,
        source_origin_distance_mm=100.0,
        detector_pixel_size_mm=(0.02, 0.02),
        detector_pixels=(2, 2),
        voxel_size_mm=(0.02, 0.02, 0.02),
        volume_voxels=(2, 2, 2),
    )

    initialized = fdk_initialization(
        np.empty((0, 0, 0), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        params,
        mode="existing_fdk",
        existing_path=str(path),
    )

    expected = source.reshape(2, 2, 2, 2, 2, 2).mean(axis=(1, 3, 5), dtype=np.float32)
    assert initialized.shape == (2, 2, 2)
    np.testing.assert_allclose(initialized, expected)


def _validation(count: int) -> MetadataValidation:
    records = [
        ProjectionRecord(
            index=index,
            filename=f"projection_{index:04d}.tif",
            path=Path(f"projection_{index:04d}.tif"),
            angle_deg=float(index * 10.0),
            angle_rad=float(np.deg2rad(index * 10.0)),
            x_shift_px=float(index * 4.0),
            y_shift_px=float(index * 2.0),
            exists=True,
        )
        for index in range(count)
    ]
    angles_rad = np.asarray([record.angle_rad for record in records], dtype=np.float32)
    return MetadataValidation(
        records=records,
        missing_files=[],
        duplicate_filenames=[],
        duplicate_endpoint_detected=False,
        duplicate_endpoint_removed=False,
        angle_direction="ascending",
        angle_min_deg=0.0,
        angle_max_deg=float((count - 1) * 10.0),
        angle_min_rad=0.0,
        angle_max_rad=float(np.deg2rad((count - 1) * 10.0)),
        angles_rad=angles_rad,
        angle_input_source="test",
        x_shift_px=np.asarray([record.x_shift_px for record in records], dtype=np.float32),
        y_shift_px=np.asarray([record.y_shift_px for record in records], dtype=np.float32),
        x_shift_column_found=True,
        y_shift_column_found=True,
        x_shift_column="x_shift_px",
        y_shift_column="y_shift_px",
        table_rows=[],
        warnings=[],
    )
