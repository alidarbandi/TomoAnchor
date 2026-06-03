from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .admm_tv_mbir import ADMMIterationMetrics


def write_metrics_csv(path: str | Path, metrics: Iterable[ADMMIterationMetrics]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(metric) for metric in metrics]
    previous_elapsed = 0.0
    for row in rows:
        row["data_residual"] = float(np.sqrt(max(2.0 * float(row.get("data_fidelity", 0.0)), 0.0)))
        elapsed = float(row.get("elapsed_s", 0.0))
        row["iteration_time_s"] = max(0.0, elapsed - previous_elapsed)
        previous_elapsed = elapsed
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def relative_rmse(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.sqrt(np.mean(np.asarray(b, dtype=np.float64) ** 2)))
    return rmse(a, b) / max(denom, 1e-12)


def ssim_2d(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return None
    data_range = float(np.max(b) - np.min(b))
    if data_range <= 0:
        data_range = 1.0
    return float(structural_similarity(np.asarray(a), np.asarray(b), data_range=data_range))
