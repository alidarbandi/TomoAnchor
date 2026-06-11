from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .admm_tv_mbir import ADMMIterationMetrics
from .io_utils import replace_file_atomically, save_png_image, temporary_output_path


def save_volume_preview(path: str | Path, volume: np.ndarray, title: str = "Volume preview") -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    vol = np.asarray(volume, dtype=np.float32)
    z = vol.shape[0] // 2
    y = vol.shape[1] // 2
    x = vol.shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    slices = [(vol[z], "Axial"), (vol[:, y, :], "Coronal"), (vol[:, :, x], "Sagittal")]
    for axis, (image, label) in zip(axes, slices):
        finite = image[np.isfinite(image)]
        if finite.size:
            low, high = np.percentile(finite, [1, 99])
        else:
            low, high = 0.0, 1.0
        if high <= low:
            high = low + 1.0
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle(title)
    temp = temporary_output_path(output, suffix=".png.tmp")
    fig.savefig(temp, dpi=150)
    plt.close(fig)
    return replace_file_atomically(temp, output)


def save_metrics_plots(folder: str | Path, metrics: list[ADMMIterationMetrics]) -> list[Path]:
    output = Path(folder)
    output.mkdir(parents=True, exist_ok=True)
    if not metrics:
        return []
    iterations = [m.iteration for m in metrics]
    data_residual = [float(np.sqrt(max(2.0 * m.data_fidelity, 0.0))) for m in metrics]
    solver = str(getattr(metrics[-1], "solver", "admm") or "admm").strip().lower()
    admm_metrics = solver == "admm"
    saved: list[Path] = []
    saved.append(_plot_series(output / "objective_plot.png", iterations, [m.objective for m in metrics], "Objective"))
    saved.append(_plot_series(output / "data_residual_plot.png", iterations, data_residual, "Full-data residual ||Ax-b||", logy=True))
    saved.append(
        _plot_multi(
            output / "data_tv_terms_plot.png",
            iterations,
            {"Data term": [m.data_fidelity for m in metrics], "TV term": [m.tv_term for m in metrics]},
            "Data and TV terms",
        )
    )
    saved.append(_plot_series(output / "relative_change_plot.png", iterations, [m.relative_x_change for m in metrics], "Relative x-change", logy=True))
    if admm_metrics:
        saved.append(_plot_series(output / "residuals_plot.png", iterations, [m.primal_residual for m in metrics], "ADMM primal residual", logy=True))
        saved.append(_plot_series(output / "cg_residual_plot.png", iterations, [m.cg_residual for m in metrics], "CG residual", logy=True))
    return saved


def _plot_series(path: Path, x: list[int], y: list[float], title: str, logy: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(x, y, marker="o", linewidth=1.4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if logy:
        ax.set_yscale("log")
    temp = temporary_output_path(path, suffix=".png.tmp")
    fig.savefig(temp, dpi=150)
    plt.close(fig)
    return replace_file_atomically(temp, path)


def _plot_multi(path: Path, x: list[int], series: dict[str, list[float]], title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for label, y in series.items():
        ax.plot(x, y, marker="o", linewidth=1.4, label=label)
    ax.set_xlabel("Iteration")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    temp = temporary_output_path(path, suffix=".png.tmp")
    fig.savefig(temp, dpi=150)
    plt.close(fig)
    return replace_file_atomically(temp, path)
