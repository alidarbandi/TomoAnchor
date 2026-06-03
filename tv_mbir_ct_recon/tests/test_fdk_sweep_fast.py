from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.config import AppConfig
from src.fast_types import AnchorSplit, ProcessedProjectionSet
from src.fdk_sweep import run_fdk_filter_sweep
from src.geometry_tigre import GeometryParams


class FakeOperator:
    def __init__(self, params, angles_rad, **kwargs) -> None:
        self.params = params
        self.angles = np.asarray(angles_rad, dtype=np.float32)

    def forward(self, volume):
        rows, cols = self.params.detector_pixels
        value = float(np.mean(volume))
        return np.full((len(self.angles), rows, cols), value, dtype=np.float32)

    def close(self) -> None:
        return None


def test_fdk_sweep_selects_lowest_anchor_residual_and_saves_outputs(tmp_path, monkeypatch) -> None:
    def fake_fdk(projections, angles_rad, params, **kwargs):
        value = 1.0 if params.fdk_filter == "ram_lak" else 2.0
        return np.full(params.volume_voxels, value, dtype=np.float32)

    monkeypatch.setattr("src.fdk_sweep.run_fdk_reconstruction", fake_fdk)
    monkeypatch.setattr("src.fdk_sweep.TigreConeBeamOperator", FakeOperator)

    cfg = AppConfig()
    cfg.geometry.invert_angle_sign_for_tigre = False
    cfg.fdk_sweep.filters = ["hann", "ram_lak"]
    cfg.fdk_sweep.cutoffs = [1.0, 0.8]
    cfg.fdk_sweep.lowres_factor = 2
    cfg.fdk_sweep.save_candidate_previews = False
    cfg.anchors.split.tune_count = 1
    main = ProcessedProjectionSet(
        name="main",
        attenuation=np.ones((3, 4, 4), dtype=np.float32),
        angles_rad=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        validation=SimpleNamespace(),  # type: ignore[arg-type]
        preprocessing_report=[],
        weights=np.ones(3, dtype=np.float32),
    )
    anchor = ProcessedProjectionSet(
        name="anchor",
        attenuation=np.ones((2, 4, 4), dtype=np.float32),
        angles_rad=np.asarray([0.5, 1.5], dtype=np.float32),
        validation=SimpleNamespace(),  # type: ignore[arg-type]
        preprocessing_report=[],
        weights=np.ones(2, dtype=np.float32),
    )
    split = AnchorSplit(
        recon_indices=np.asarray([0], dtype=np.int64),
        tune_indices=np.asarray([1], dtype=np.int64),
        qc_indices=np.asarray([], dtype=np.int64),
    )
    params = GeometryParams(
        source_detector_distance_mm=10.0,
        source_origin_distance_mm=5.0,
        detector_pixel_size_mm=(1.0, 1.0),
        detector_pixels=(4, 4),
        voxel_size_mm=(1.0, 1.0, 1.0),
        volume_voxels=(4, 4, 4),
    )

    result = run_fdk_filter_sweep(main, anchor, split, params, cfg, tmp_path, None, False, None, None)

    assert result.best_filter == "ram_lak"
    assert result.best_volume.shape == (4, 4, 4)
    assert (tmp_path / "scores.csv").exists()
    assert (tmp_path / "best_filter.yaml").exists()
    assert (tmp_path / "fdk_best.npy").exists()
    assert all(float(row["cutoff"]) == 1.0 for row in result.scores)


def test_fdk_sweep_manual_filter_override_ignores_auto_score(tmp_path, monkeypatch) -> None:
    def fake_fdk(projections, angles_rad, params, **kwargs):
        value = 1.0 if params.fdk_filter == "ram_lak" else 2.0
        return np.full(params.volume_voxels, value, dtype=np.float32)

    monkeypatch.setattr("src.fdk_sweep.run_fdk_reconstruction", fake_fdk)
    monkeypatch.setattr("src.fdk_sweep.TigreConeBeamOperator", FakeOperator)

    cfg = AppConfig()
    cfg.geometry.invert_angle_sign_for_tigre = False
    cfg.fdk_sweep.filters = ["ram_lak", "hann"]
    cfg.fdk_sweep.selection_filter = "hann"
    cfg.fdk_sweep.lowres_factor = 2
    cfg.fdk_sweep.save_candidate_previews = False
    main = ProcessedProjectionSet(
        name="main",
        attenuation=np.ones((3, 4, 4), dtype=np.float32),
        angles_rad=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        validation=SimpleNamespace(),  # type: ignore[arg-type]
        preprocessing_report=[],
        weights=np.ones(3, dtype=np.float32),
    )
    anchor = ProcessedProjectionSet(
        name="anchor",
        attenuation=np.ones((2, 4, 4), dtype=np.float32),
        angles_rad=np.asarray([0.5, 1.5], dtype=np.float32),
        validation=SimpleNamespace(),  # type: ignore[arg-type]
        preprocessing_report=[],
        weights=np.ones(2, dtype=np.float32),
    )
    split = AnchorSplit(
        recon_indices=np.asarray([0], dtype=np.int64),
        tune_indices=np.asarray([1], dtype=np.int64),
        qc_indices=np.asarray([], dtype=np.int64),
    )
    params = GeometryParams(
        source_detector_distance_mm=10.0,
        source_origin_distance_mm=5.0,
        detector_pixel_size_mm=(1.0, 1.0),
        detector_pixels=(4, 4),
        voxel_size_mm=(1.0, 1.0, 1.0),
        volume_voxels=(4, 4, 4),
    )

    result = run_fdk_filter_sweep(main, anchor, split, params, cfg, tmp_path, None, False, None, None)

    assert result.best_filter == "hann"
    assert np.all(result.best_volume == 2.0)
    text = (tmp_path / "best_filter.yaml").read_text(encoding="utf-8")
    assert "selection_mode: manual_override" in text
    assert "auto_score_filter: ram_lak" in text
