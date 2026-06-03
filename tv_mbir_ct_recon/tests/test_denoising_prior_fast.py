from __future__ import annotations

import numpy as np

from src.config import AppConfig
from src.denoising_prior import make_training_free_prior
from src.geometry_tigre import GeometryParams


def test_training_free_prior_outputs_finite_confidence_and_files(tmp_path) -> None:
    cfg = AppConfig()
    cfg.prior.tv_weights = [0.001, 0.002]
    cfg.prior.tv_iterations = 2
    cfg.prior.support_dilation_voxels = 1
    cfg.confidence.C_min = 0.2
    x = np.zeros((6, 6, 6), dtype=np.float32)
    x[2:4, 2:4, 2:4] = 1.0
    x[3, 3, 3] = 1.5
    params = GeometryParams(
        source_detector_distance_mm=10.0,
        source_origin_distance_mm=5.0,
        detector_pixel_size_mm=(1.0, 1.0),
        detector_pixels=(6, 6),
        voxel_size_mm=(1.0, 1.0, 1.0),
        volume_voxels=(6, 6, 6),
    )

    result = make_training_free_prior(
        x,
        tune_anchor_data=None,
        geometry_params=params,
        config=cfg,
        output_folder=tmp_path,
        gpu_ids=None,
        use_gpu=False,
        logger=None,
        cancel_check=None,
    )

    assert result.x_prior.shape == x.shape
    assert np.all(np.isfinite(result.x_prior))
    assert np.min(result.confidence) >= cfg.confidence.C_min
    assert np.max(result.confidence) <= cfg.confidence.C_max
    assert (tmp_path / "prior_selected.npy").exists()
    assert (tmp_path / "prior_confidence.npy").exists()
    assert (tmp_path / "prior_difference.npy").exists()
    assert (tmp_path / "prior_metrics.csv").exists()
