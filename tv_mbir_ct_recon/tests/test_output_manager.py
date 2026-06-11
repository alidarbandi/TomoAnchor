from __future__ import annotations

import csv

import numpy as np

from src.admm_tv_mbir import ADMMIterationMetrics
from src.metrics import write_metrics_csv
from src.output_manager import OutputManager


def test_save_volume_outputs_can_skip_tif(tmp_path) -> None:
    manager = OutputManager(tmp_path)
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    npy_path, tif_path = manager.save_volume_outputs(tmp_path / "mbir", "mbir_final_volume", volume, save_tif=False)
    assert npy_path.exists()
    assert tif_path is None
    assert not (tmp_path / "mbir" / "mbir_final_volume.tif").exists()
    loaded = np.load(npy_path, allow_pickle=False)
    assert loaded.shape == volume.shape
    assert np.allclose(loaded, volume)


def test_write_metrics_csv_adds_monitoring_columns(tmp_path) -> None:
    metrics = [
        ADMMIterationMetrics(
            iteration=1,
            objective=5.0,
            data_fidelity=4.5,
            tv_term=1.0,
            primal_residual=0.1,
            dual_residual=0.05,
            relative_x_change=0.01,
            cg_residual=1e-3,
            cg_iterations=4,
            elapsed_s=2.0,
            solver="admm",
        )
    ]
    path = write_metrics_csv(tmp_path / "metrics.csv", metrics)
    assert path.exists()
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["iteration"] == "1"
    assert "data_residual" in rows[0]
    assert "iteration_time_s" in rows[0]
