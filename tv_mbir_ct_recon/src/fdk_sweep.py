from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .cancel_utils import raise_if_cancelled
from .config import AppConfig
from .fast_types import AnchorSplit, FDKSweepResult, ProcessedProjectionSet
from .geometry_tigre import GeometryParams, angles_for_tigre
from .io_utils import block_average_2d, save_stack_tiff
from .plotting import save_volume_preview
from .projection_weights import combine_duplicate_angle_projections
from .tigre_ops import TigreConeBeamOperator, run_fdk_reconstruction


def run_fdk_filter_sweep(
    main_set: ProcessedProjectionSet,
    anchor_set: ProcessedProjectionSet | None,
    anchor_split: AnchorSplit | None,
    geometry_params: GeometryParams,
    config: AppConfig,
    output_folder: Path,
    gpu_ids: Sequence[int] | None,
    use_gpu: bool,
    logger: Callable[[str], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> FDKSweepResult:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    candidate_folder = output_folder / "candidate_previews"
    if config.fdk_sweep.save_candidate_previews:
        candidate_folder.mkdir(parents=True, exist_ok=True)

    recon_proj, recon_angles, recon_weights = _build_reconstruction_stack(main_set, anchor_set, anchor_split, config)
    tune_proj, tune_angles, tune_weights = _build_tune_stack(anchor_set, anchor_split, config)
    if config.anchors.merge.combine_duplicate_angles:
        tol = np.deg2rad(float(config.anchors.merge.duplicate_angle_tolerance_deg))
        recon_proj, recon_angles, recon_weights, merge_report = combine_duplicate_angle_projections(
            recon_proj,
            recon_angles,
            recon_weights,
            tol,
        )
        if merge_report:
            _log(logger, f"FDK sweep merged {len(merge_report)} duplicate-angle projection groups.")
    if config.anchors.merge.sort_by_angle:
        order = np.argsort(recon_angles.astype(np.float64), kind="mergesort")
        recon_proj = np.ascontiguousarray(recon_proj[order], dtype=np.float32)
        recon_angles = np.ascontiguousarray(recon_angles[order], dtype=np.float32)
        if recon_weights is not None:
            recon_weights = np.ascontiguousarray(recon_weights[order], dtype=np.float32)

    lowres_proj, lowres_params, lowres_tune_proj, lowres_factor = _prepare_lowres(
        recon_proj,
        geometry_params,
        tune_proj,
        max(1, int(config.fdk_sweep.lowres_factor)),
        logger,
    )
    lowres_recon_angles = recon_angles
    lowres_tune_angles = tune_angles
    raw_scores: list[dict[str, object]] = []
    candidate_volumes: list[np.ndarray] = []
    candidate_keys: list[tuple[str, float]] = []
    selected_override = _manual_selection_filter(config)
    candidate_filter_names = [str(name) for name in (config.fdk_sweep.filters or ["ram_lak"])]
    if selected_override is not None and selected_override not in {_normalize_fdk_filter_name(name) for name in candidate_filter_names}:
        candidate_filter_names.append(selected_override)
    allowed_cutoffs = [float(value) for value in (config.fdk_sweep.cutoffs or [1.0])]
    for filter_name in candidate_filter_names:
        for cutoff in allowed_cutoffs:
            raise_if_cancelled(cancel_check, "FDK filter sweep cancelled.")
            if abs(float(cutoff) - 1.0) > 1e-8:
                _log(logger, f"TIGRE FDK cutoff sweep is unavailable; skipping cutoff={cutoff:g} for filter {filter_name}.")
                continue
            candidate_filter = _normalize_fdk_filter_name(str(filter_name))
            candidate_params = replace(lowres_params, fdk_filter=candidate_filter)
            _log(logger, f"Running low-resolution FDK candidate filter={candidate_filter}, cutoff={cutoff:g}.")
            volume = run_fdk_reconstruction(
                lowres_proj,
                angles_for_tigre(lowres_recon_angles, config.geometry.invert_angle_sign_for_tigre),
                candidate_params,
                gpu_ids=gpu_ids,
                use_gpu=use_gpu,
                progress=logger,
                cancel_check=cancel_check,
            )
            candidate_volumes.append(volume)
            candidate_keys.append((candidate_filter, float(cutoff)))
            metrics = _score_fdk_candidate(
                volume,
                lowres_tune_proj,
                lowres_tune_angles,
                tune_weights,
                lowres_params,
                config,
                gpu_ids,
                use_gpu,
                logger,
                cancel_check,
            )
            metrics.update({"filter": candidate_filter, "cutoff": float(cutoff), "lowres_factor": int(lowres_factor)})
            raw_scores.append(metrics)
            if config.fdk_sweep.save_candidate_previews:
                safe_name = candidate_filter.replace("/", "_").replace("\\", "_")
                save_volume_preview(candidate_folder / f"{safe_name}_cutoff_{float(cutoff):.3g}.png", volume, f"FDK {candidate_filter}")

    if not raw_scores:
        raise RuntimeError("FDK sweep did not run any candidates. Check fdk_sweep.filters and cutoffs.")
    scores = _add_robust_scores(raw_scores, config)
    auto_best_index = int(np.argmin([float(row["score_total"]) for row in scores]))
    auto_best_filter, auto_best_cutoff = candidate_keys[auto_best_index]
    best_index = auto_best_index
    selection_mode = "auto_score"
    if selected_override is not None:
        manual_matches = [
            index
            for index, (filter_name, cutoff) in enumerate(candidate_keys)
            if _normalize_fdk_filter_name(filter_name) == selected_override and abs(float(cutoff) - 1.0) <= 1e-8
        ]
        if not manual_matches:
            raise RuntimeError(f"Selected FDK filter '{selected_override}' was not evaluated by the FDK sweep.")
        best_index = manual_matches[0]
        selection_mode = "manual_override"
    best_filter, best_cutoff = candidate_keys[best_index]
    _write_dicts_csv(output_folder / "scores.csv", scores)
    (output_folder / "best_filter.yaml").write_text(
        "\n".join(
            [
                f"best_filter: {best_filter}",
                f"best_cutoff: {best_cutoff}",
                f"selection_mode: {selection_mode}",
                f"auto_score_filter: {auto_best_filter}",
                f"auto_score_cutoff: {auto_best_cutoff}",
                f"lowres_factor: {lowres_factor}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if selection_mode == "manual_override":
        _log(logger, f"Manual FDK filter override selected {best_filter}; auto score winner was {auto_best_filter}. Running full-resolution FDK.")
    else:
        _log(logger, f"Selected FDK filter {best_filter} with cutoff {best_cutoff:g}. Running full-resolution FDK.")
    full_params = replace(geometry_params, fdk_filter=_normalize_fdk_filter_name(best_filter))
    best_volume = run_fdk_reconstruction(
        recon_proj,
        angles_for_tigre(recon_angles, config.geometry.invert_angle_sign_for_tigre),
        full_params,
        gpu_ids=gpu_ids,
        use_gpu=use_gpu,
        progress=logger,
        cancel_check=cancel_check,
    )
    np.save(output_folder / "fdk_best.npy", best_volume.astype(np.float32))
    save_stack_tiff(output_folder / "fdk_best.tif", best_volume)
    save_volume_preview(output_folder / "fdk_best_preview.png", best_volume, "Best FDK")
    return FDKSweepResult(
        best_volume=np.ascontiguousarray(best_volume, dtype=np.float32),
        best_filter=str(best_filter),
        best_cutoff=float(best_cutoff),
        scores=scores,
        recon_projections=np.ascontiguousarray(recon_proj, dtype=np.float32),
        recon_angles_rad=np.ascontiguousarray(recon_angles, dtype=np.float32),
        recon_weights=None if recon_weights is None else np.ascontiguousarray(recon_weights, dtype=np.float32),
        tune_anchor_projections=tune_proj,
        tune_anchor_angles_rad=tune_angles,
        tune_anchor_weights=tune_weights,
    )


def _build_reconstruction_stack(
    main_set: ProcessedProjectionSet,
    anchor_set: ProcessedProjectionSet | None,
    anchor_split: AnchorSplit | None,
    config: AppConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    projections = [np.asarray(main_set.attenuation, dtype=np.float32)]
    angles = [np.asarray(main_set.angles_rad, dtype=np.float32)]
    weight_parts: list[np.ndarray] = []
    if main_set.weights is not None:
        weight_parts.append(np.asarray(main_set.weights, dtype=np.float32))
    if anchor_set is not None and anchor_split is not None:
        anchor_indices = np.asarray(anchor_split.recon_indices, dtype=np.int64)
        if anchor_indices.size:
            projections.append(np.asarray(anchor_set.attenuation[anchor_indices], dtype=np.float32))
            angles.append(np.asarray(anchor_set.angles_rad[anchor_indices], dtype=np.float32))
            if anchor_set.weights is not None:
                weight_parts.append(np.asarray(anchor_set.weights[anchor_indices], dtype=np.float32))
    merged_proj = np.ascontiguousarray(np.concatenate(projections, axis=0), dtype=np.float32)
    merged_angles = np.ascontiguousarray(np.concatenate(angles, axis=0), dtype=np.float32)
    if not weight_parts:
        return merged_proj, merged_angles, None
    if len(weight_parts) != len(projections):
        weight_parts = [np.ones(part.shape[0], dtype=np.float32) for part in projections]
    return merged_proj, merged_angles, np.ascontiguousarray(np.concatenate(weight_parts), dtype=np.float32)


def _build_tune_stack(
    anchor_set: ProcessedProjectionSet | None,
    anchor_split: AnchorSplit | None,
    config: AppConfig,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not config.fdk_sweep.use_tune_anchors or anchor_set is None or anchor_split is None:
        return None, None, None
    tune_indices = np.asarray(anchor_split.tune_indices, dtype=np.int64)
    if not tune_indices.size:
        return None, None, None
    projections = np.ascontiguousarray(anchor_set.attenuation[tune_indices], dtype=np.float32)
    angles = np.ascontiguousarray(anchor_set.angles_rad[tune_indices], dtype=np.float32)
    weights = None if anchor_set.weights is None else np.ascontiguousarray(anchor_set.weights[tune_indices], dtype=np.float32)
    return projections, angles, weights


def _prepare_lowres(
    projections: np.ndarray,
    params: GeometryParams,
    tune_projections: np.ndarray | None,
    factor: int,
    logger: Callable[[str], None] | None,
) -> tuple[np.ndarray, GeometryParams, np.ndarray | None, int]:
    if factor <= 1:
        return projections, params, tune_projections, 1
    rows, cols = projections.shape[1:]
    nz, ny, nx = params.volume_voxels
    if rows % factor or cols % factor or nz % factor or ny % factor or nx % factor:
        _log(logger, f"Low-resolution FDK factor {factor} is not divisible for detector/volume shapes; using factor 1.")
        return projections, params, tune_projections, 1
    low_proj = _block_average_stack(projections, factor)
    low_tune = None if tune_projections is None else _block_average_stack(tune_projections, factor)
    dv, du = params.detector_pixel_size_mm
    dz, dy, dx = params.voxel_size_mm
    low_params = replace(
        params,
        detector_pixel_size_mm=(dv * factor, du * factor),
        detector_pixels=(rows // factor, cols // factor),
        voxel_size_mm=(dz * factor, dy * factor, dx * factor),
        volume_voxels=(nz // factor, ny // factor, nx // factor),
        detector_offset_pixels=(params.detector_offset_pixels[0] / factor, params.detector_offset_pixels[1] / factor),
        center_offset_pixels=params.center_offset_pixels / factor,
    )
    _log(logger, f"Using low-resolution FDK sweep factor {factor}.")
    return low_proj, low_params, low_tune, factor


def _block_average_stack(stack: np.ndarray, factor: int) -> np.ndarray:
    return np.ascontiguousarray(np.stack([block_average_2d(image, factor, factor) for image in stack], axis=0), dtype=np.float32)


def _score_fdk_candidate(
    volume: np.ndarray,
    tune_projections: np.ndarray | None,
    tune_angles: np.ndarray | None,
    tune_weights: np.ndarray | None,
    params: GeometryParams,
    config: AppConfig,
    gpu_ids: Sequence[int] | None,
    use_gpu: bool,
    logger: Callable[[str], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> dict[str, object]:
    residual = 0.0
    if tune_projections is not None and tune_angles is not None and tune_projections.size:
        operator = TigreConeBeamOperator(
            params,
            angles_for_tigre(tune_angles, config.geometry.invert_angle_sign_for_tigre),
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
        diff = predicted - tune_projections
        denom = tune_projections
        if tune_weights is not None:
            weights = np.asarray(tune_weights, dtype=np.float32)
            diff = diff * weights[:, None, None]
            denom = denom * weights[:, None, None]
        residual = float(np.sum(np.asarray(diff, dtype=np.float64) ** 2) / max(float(np.sum(np.asarray(denom, dtype=np.float64) ** 2)), 1e-12))
    tv = _tv_roughness(volume)
    negative = float(np.sum(np.minimum(volume, 0.0).astype(np.float64) ** 2) / max(float(np.sum(np.asarray(volume, dtype=np.float64) ** 2)), 1e-12))
    return {"anchor_residual": residual, "tv_roughness": tv, "negative_penalty": negative}


def _tv_roughness(volume: np.ndarray) -> float:
    x = np.asarray(volume, dtype=np.float32)
    gx = np.abs(x[:, :, 1:] - x[:, :, :-1]).sum(dtype=np.float64)
    gy = np.abs(x[:, 1:, :] - x[:, :-1, :]).sum(dtype=np.float64)
    gz = np.abs(x[1:, :, :] - x[:-1, :, :]).sum(dtype=np.float64)
    denom = max(float(np.sum(np.abs(x), dtype=np.float64)), 1e-12)
    return float((gx + gy + gz) / denom)


def _add_robust_scores(rows: list[dict[str, object]], config: AppConfig) -> list[dict[str, object]]:
    keys = ["anchor_residual", "tv_roughness", "negative_penalty"]
    z_values: dict[str, np.ndarray] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        q25, q75 = np.percentile(values, [25, 75])
        z_values[key] = (values - float(np.median(values))) / max(float(q75 - q25), 1e-12)
    scored: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["z_anchor_residual"] = float(z_values["anchor_residual"][index])
        item["z_tv_roughness"] = float(z_values["tv_roughness"][index])
        item["z_negative_penalty"] = float(z_values["negative_penalty"][index])
        item["score_total"] = float(
            config.fdk_sweep.metric_anchor_weight * item["z_anchor_residual"]
            + config.fdk_sweep.metric_tv_weight * item["z_tv_roughness"]
            + config.fdk_sweep.metric_negative_weight * item["z_negative_penalty"]
        )
        scored.append(item)
    return scored


def _normalize_fdk_filter_name(name: str) -> str:
    clean = str(name or "ram_lak").strip().lower()
    if clean == "shep_logan":
        return "shepp_logan"
    return clean


def _manual_selection_filter(config: AppConfig) -> str | None:
    value = _normalize_fdk_filter_name(getattr(config.fdk_sweep, "selection_filter", "auto"))
    if value in {"", "auto", "automatic", "score", "score_total"}:
        return None
    return value


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
