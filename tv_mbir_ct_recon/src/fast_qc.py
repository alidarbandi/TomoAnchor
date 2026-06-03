from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import AppConfig
from .fast_types import AnchorSplit, FDKSweepResult, MBIRLiteResult, PriorResult, ProcessedProjectionSet
from .geometry_tigre import GeometryParams, angles_for_tigre
from .io_utils import save_image
from .tigre_ops import TigreConeBeamOperator


def make_fast_recon_qc_report(
    main_set: ProcessedProjectionSet | None,
    anchor_set: ProcessedProjectionSet | None,
    anchor_split: AnchorSplit | None,
    geometry_params: GeometryParams,
    config: AppConfig,
    output_folder: Path,
    fdk_result: FDKSweepResult | None,
    prior_result: PriorResult | None,
    mbir_lite_result: MBIRLiteResult | None,
    gpu_ids: Sequence[int] | None,
    use_gpu: bool,
    logger: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Path:
    output_folder = Path(output_folder)
    residual_folder = output_folder / "residuals"
    residual_folder.mkdir(parents=True, exist_ok=True)
    volumes = {
        "fdk": None if fdk_result is None else fdk_result.best_volume,
        "prior": None if prior_result is None else prior_result.x_prior,
        "final": None if mbir_lite_result is None else mbir_lite_result.volume,
    }
    metrics: list[dict[str, object]] = []
    if anchor_set is not None and anchor_split is not None:
        for split_name, indices in {"tune": anchor_split.tune_indices, "qc": anchor_split.qc_indices}.items():
            for volume_name, volume in volumes.items():
                if volume is None:
                    continue
                residual_value, residual_stack = _anchor_residual_stack(
                    volume,
                    anchor_set,
                    indices,
                    geometry_params,
                    config,
                    gpu_ids,
                    use_gpu,
                    logger,
                    cancel_check,
                )
                metrics.append({"volume": volume_name, "split": split_name, "residual": residual_value, "count": int(len(indices))})
                if split_name == "qc" and residual_stack is not None:
                    for local_index, residual_image in enumerate(residual_stack):
                        anchor_index = int(np.asarray(indices, dtype=np.int64)[local_index])
                        save_image(residual_folder / f"{volume_name}_qc_anchor_{anchor_index:04d}.tif", residual_image)
    _write_dicts_csv(output_folder / "qc_metrics.csv", metrics)
    _save_preview_panel(output_folder / "preview_panel.png", volumes, prior_result)
    report = _report_lines(main_set, anchor_set, anchor_split, config, fdk_result, prior_result, mbir_lite_result, metrics)
    report_path = output_folder / "report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    _log(logger, f"Saved fast reconstruction QC report: {report_path}")
    return report_path


def _anchor_residual_stack(
    volume: np.ndarray,
    anchor_set: ProcessedProjectionSet,
    indices: np.ndarray,
    geometry_params: GeometryParams,
    config: AppConfig,
    gpu_ids: Sequence[int] | None,
    use_gpu: bool,
    logger: Callable[[str], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> tuple[float, np.ndarray | None]:
    indices = np.asarray(indices, dtype=np.int64)
    if not indices.size:
        return float("nan"), None
    projections = np.ascontiguousarray(anchor_set.attenuation[indices], dtype=np.float32)
    angles = np.ascontiguousarray(anchor_set.angles_rad[indices], dtype=np.float32)
    weights = None if anchor_set.weights is None else np.asarray(anchor_set.weights[indices], dtype=np.float32)
    operator = TigreConeBeamOperator(
        geometry_params,
        angles_for_tigre(angles, config.geometry.invert_angle_sign_for_tigre),
        gpu_ids=gpu_ids,
        use_gpu=use_gpu,
        memory_mode="projection_streaming",
        projection_batch_size=max(1, int(config.mbir_lite.projection_batch_size)),
        logger=logger,
        cancel_check=cancel_check,
    )
    try:
        predicted = operator.forward(volume)
    finally:
        operator.close()
    residual = predicted - projections
    denom = projections
    if weights is not None:
        residual_weighted = residual * weights[:, None, None]
        denom = denom * weights[:, None, None]
    else:
        residual_weighted = residual
    value = float(np.sum(np.asarray(residual_weighted, dtype=np.float64) ** 2) / max(float(np.sum(np.asarray(denom, dtype=np.float64) ** 2)), 1e-12))
    return value, np.ascontiguousarray(residual, dtype=np.float32)


def _save_preview_panel(path: Path, volumes: dict[str, np.ndarray | None], prior_result: PriorResult | None) -> None:
    images: list[tuple[str, np.ndarray]] = []
    fdk = volumes.get("fdk")
    prior = volumes.get("prior")
    final = volumes.get("final")
    if fdk is not None:
        images.append(("FDK", _central_slice(fdk)))
    if prior is not None:
        images.append(("Prior", _central_slice(prior)))
    if final is not None:
        images.append(("Final", _central_slice(final)))
    if fdk is not None and prior is not None:
        images.append(("Prior - FDK", _central_slice(prior - fdk)))
    if final is not None and prior is not None:
        images.append(("Final - Prior", _central_slice(final - prior)))
    if final is not None and fdk is not None:
        images.append(("Final - FDK", _central_slice(final - fdk)))
    if prior_result is not None:
        images.append(("Confidence", _central_slice(prior_result.confidence)))
    if not images:
        path.write_text("No preview volumes available.", encoding="utf-8")
        return
    cols = min(4, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for axis, (title, image) in zip(axes_arr, images):
        finite = image[np.isfinite(image)]
        low, high = (np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0))
        if high <= low:
            high = low + 1.0
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
        axis.set_title(title)
        axis.axis("off")
    for axis in axes_arr[len(images) :]:
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _report_lines(
    main_set: ProcessedProjectionSet | None,
    anchor_set: ProcessedProjectionSet | None,
    anchor_split: AnchorSplit | None,
    config: AppConfig,
    fdk_result: FDKSweepResult | None,
    prior_result: PriorResult | None,
    mbir_lite_result: MBIRLiteResult | None,
    metrics: list[dict[str, object]],
) -> list[str]:
    lines = [
        "# Fast Anchor-Guided MBIR-Lite QC Report",
        "",
        "## Parameters",
        "",
        f"- Main projections: {0 if main_set is None else len(main_set.angles_rad)}",
        f"- Anchor projections: {0 if anchor_set is None else len(anchor_set.angles_rad)}",
        f"- Anchor recon/tune/qc counts: {_split_counts(anchor_split)}",
        f"- Selected FDK filter: {'' if fdk_result is None else fdk_result.best_filter}",
        f"- Selected prior: {'' if prior_result is None else prior_result.chosen_method} {'' if prior_result is None else prior_result.chosen_strength:g}",
        f"- MBIR-lite sweeps: {config.mbir_lite.n_sweeps}",
        f"- lambda_tv: {config.mbir_lite.lambda_tv:g}",
        f"- rho_prior: {config.mbir_lite.rho_prior:g}",
        f"- Projection batch size: {config.mbir_lite.projection_batch_size}",
        f"- Ordered subset count: {config.mbir_lite.ordered_subset_count}",
        "",
        "## Anchor Residuals",
        "",
        "Residual formula: `R = ||M W(Ax - y)||_2^2 / (||M W y||_2^2 + eps)`",
        "",
    ]
    if not metrics:
        lines.append("No anchor residuals were available.")
    else:
        lines.extend(["| Volume | Split | Count | Residual |", "|---|---:|---:|---:|"])
        for row in metrics:
            lines.append(f"| {row['volume']} | {row['split']} | {row['count']} | {float(row['residual']):.6g} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- Preview panel: `preview_panel.png`",
            "- Metrics: `qc_metrics.csv`",
            "- QC residual TIFFs: `residuals/`",
        ]
    )
    if mbir_lite_result is not None:
        lines.extend(["", f"MBIR-lite status: {mbir_lite_result.message}"])
    return lines


def _central_slice(volume: np.ndarray) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32)
    return np.ascontiguousarray(vol[vol.shape[0] // 2], dtype=np.float32)


def _split_counts(split: AnchorSplit | None) -> str:
    if split is None:
        return "0 / 0 / 0"
    return f"{len(split.recon_indices)} / {len(split.tune_indices)} / {len(split.qc_indices)}"


def _write_dicts_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _log(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)
