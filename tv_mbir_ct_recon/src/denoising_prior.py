from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from .cancel_utils import raise_if_cancelled
from .config import AppConfig, save_config
from .fast_types import AnchorValidationData, PriorResult
from .geometry_tigre import GeometryParams, angles_for_tigre
from .io_utils import save_stack_tiff
from .plotting import save_volume_preview
from .subset_tv_mbir import _add_smoothed_tv_gradient_inplace
from .tigre_ops import TigreConeBeamOperator


def make_training_free_prior(
    x_fdk: np.ndarray,
    tune_anchor_data: AnchorValidationData | None,
    geometry_params: GeometryParams,
    config: AppConfig,
    output_folder: Path,
    gpu_ids: Sequence[int] | None,
    use_gpu: bool,
    logger: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> PriorResult:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    x_fdk = np.ascontiguousarray(x_fdk, dtype=np.float32)
    support = _build_support_mask(x_fdk, config)
    x_norm, normalization = _normalize_volume(x_fdk, support, config)
    sigma_hat = _estimate_noise_highpass_mad(x_norm, support, config)
    _log(logger, f"Estimated normalized FDK noise sigma by high-pass MAD: {sigma_hat:.6g}")

    weights = list(config.prior.tv_weights) or [0.005]
    metrics: list[dict[str, object]] = []
    candidates: list[tuple[float, np.ndarray, dict[str, object]]] = []
    _log(logger, f"Prior generation will evaluate {len(weights)} TV candidate(s): {', '.join(f'{float(value):g}' for value in weights)}.")
    started = perf_counter()
    _log(logger, "Scoring baseline FDK projection residual for prior selection.")
    fdk_anchor_residual = _anchor_residual(x_norm, tune_anchor_data, geometry_params, config, gpu_ids, use_gpu, logger, cancel_check)
    _log(logger, f"Baseline FDK prior residual: {_format_float(fdk_anchor_residual)} in {perf_counter() - started:.1f} s.")
    for index, strength in enumerate(weights, start=1):
        raise_if_cancelled(cancel_check, "Prior generation cancelled.")
        candidate_started = perf_counter()
        _log(logger, f"Prior candidate {index}/{len(weights)}: TV strength {float(strength):g}; denoising volume.")
        candidate = _tv_denoise_3d(
            x_norm,
            weight=float(strength),
            iterations=max(1, int(config.prior.tv_iterations)),
            epsilon=float(config.prior.tv_epsilon),
            slab_depth=max(0, int(config.prior.slab_depth)),
            slab_overlap=max(0, int(config.prior.slab_overlap)),
            cancel_check=cancel_check,
        )
        _log(
            logger,
            f"Prior candidate {index}/{len(weights)}: denoising finished in {perf_counter() - candidate_started:.1f} s; scoring tune-anchor residual.",
        )
        score_started = perf_counter()
        metric = _score_prior_candidate(
            x_norm,
            candidate,
            support,
            tune_anchor_data,
            geometry_params,
            config,
            gpu_ids,
            use_gpu,
            fdk_anchor_residual,
            logger,
            cancel_check,
        )
        metric["method"] = "tv"
        metric["strength"] = float(strength)
        metrics.append(metric)
        candidates.append((float(strength), candidate, metric))
        _write_dicts_csv(output_folder / "prior_metrics.csv", metrics)
        _log(
            logger,
            "Prior candidate "
            f"{index}/{len(weights)} done in {perf_counter() - candidate_started:.1f} s "
            f"(anchor scoring {perf_counter() - score_started:.1f} s): "
            f"residual_ratio={_format_float(metric['anchor_residual_ratio'])}, "
            f"edge_retention={_format_float(metric['edge_retention'])}, "
            f"correction_fraction={_format_float(metric['correction_fraction'])}, "
            f"background_noise={_format_float(metric['background_noise'])}, "
            f"accepted={bool(metric['accepted'])}.",
        )

    accepted = [
        item
        for item in candidates
        if bool(item[2]["accepted"])
    ]
    if accepted and config.prior_scoring.choose_lowest_background_noise_among_valid:
        chosen_strength, chosen_norm, chosen_metric = min(accepted, key=lambda item: float(item[2]["background_noise"]))
    elif accepted:
        chosen_strength, chosen_norm, chosen_metric = accepted[0]
    else:
        chosen_strength, chosen_norm, chosen_metric = min(candidates, key=lambda item: item[0])
        chosen_metric["chosen_by_fallback"] = True
        _log(logger, "No prior candidate passed conservative checks; using the weakest TV prior.")

    x_prior = _unnormalize_volume(chosen_norm, normalization)
    confidence = _build_confidence_map(x_norm, chosen_norm, support, config)
    result = PriorResult(
        x_prior=np.ascontiguousarray(x_prior, dtype=np.float32),
        confidence=np.ascontiguousarray(confidence, dtype=np.float32),
        support_mask=np.asarray(support, dtype=bool),
        metrics=metrics,
        chosen_method="tv",
        chosen_strength=float(chosen_strength),
        normalization=normalization,
    )
    _save_prior_outputs(output_folder, x_fdk, x_norm, chosen_norm, result, config)
    _log(logger, f"Selected TV prior strength {chosen_strength:g}.")
    return result


def _build_support_mask(x: np.ndarray, config: AppConfig) -> np.ndarray:
    try:
        from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
    except Exception:
        return np.ones_like(x, dtype=bool)
    finite = x[np.isfinite(x)]
    if not finite.size:
        return np.ones_like(x, dtype=bool)
    smooth = gaussian_filter(np.nan_to_num(x, nan=0.0), sigma=max(float(config.prior.support_gaussian_sigma_voxels), 0.0))
    low = float(np.percentile(finite, 5))
    high = float(np.percentile(finite, 95))
    threshold = low + 0.02 * max(high - low, 1e-6)
    mask = smooth > threshold
    if not np.any(mask):
        return np.ones_like(x, dtype=bool)
    mask = binary_closing(mask, iterations=1)
    dilation = max(0, int(config.prior.support_dilation_voxels))
    if dilation:
        mask = binary_dilation(mask, iterations=dilation)
    return np.asarray(mask, dtype=bool)


def _normalize_volume(x: np.ndarray, support: np.ndarray, config: AppConfig) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(x[support], dtype=np.float32)
    if values.size == 0:
        values = np.asarray(x, dtype=np.float32).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        p_low, p_high = 0.0, 1.0
    else:
        p_low = float(np.percentile(finite, float(config.prior.normalize_low_percentile)))
        p_high = float(np.percentile(finite, float(config.prior.normalize_high_percentile)))
    scale = max(p_high - p_low, 1e-12)
    normalized = (np.asarray(x, dtype=np.float32) - p_low) / scale
    normalized = np.clip(normalized, float(config.prior.clip_min), float(config.prior.clip_max))
    return np.ascontiguousarray(normalized, dtype=np.float32), {"p_low": p_low, "p_high": p_high, "scale": scale}


def _unnormalize_volume(x_norm: np.ndarray, normalization: dict[str, float]) -> np.ndarray:
    return np.asarray(x_norm, dtype=np.float32) * float(normalization["scale"]) + float(normalization["p_low"])


def _estimate_noise_highpass_mad(x_norm: np.ndarray, support: np.ndarray, config: AppConfig) -> float:
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        values = np.asarray(x_norm[~support] if np.any(~support) else x_norm, dtype=np.float32)
    else:
        smooth = gaussian_filter(x_norm, sigma=max(float(config.prior.noise_highpass_sigma_voxels), 0.0))
        highpass = x_norm - smooth
        values = np.asarray(highpass[~support] if np.any(~support) else highpass, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    med = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - med)))


def _tv_denoise_3d(
    x_norm: np.ndarray,
    weight: float,
    iterations: int,
    epsilon: float,
    slab_depth: int = 0,
    slab_overlap: int = 0,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    if slab_depth > 0 and x_norm.shape[0] > slab_depth:
        overlap = min(max(int(slab_overlap), 0), max(int(slab_depth) - 1, 0))
        step = max(int(slab_depth) - overlap, 1)
        output = np.zeros_like(x_norm, dtype=np.float32)
        counts = np.zeros(x_norm.shape[0], dtype=np.float32)
        for z0 in range(0, x_norm.shape[0], step):
            raise_if_cancelled(cancel_check, "TV prior generation cancelled.")
            z1 = min(z0 + int(slab_depth), x_norm.shape[0])
            denoised = _tv_denoise_3d(
                x_norm[z0:z1],
                weight=weight,
                iterations=iterations,
                epsilon=epsilon,
                slab_depth=0,
                slab_overlap=0,
                cancel_check=cancel_check,
            )
            output[z0:z1] += denoised
            counts[z0:z1] += 1.0
            if z1 >= x_norm.shape[0]:
                break
        output /= np.maximum(counts[:, None, None], 1.0)
        return np.ascontiguousarray(output, dtype=np.float32)
    try:
        from skimage.restoration import denoise_tv_chambolle

        try:
            return np.ascontiguousarray(
                denoise_tv_chambolle(x_norm, weight=float(weight), max_num_iter=int(iterations), channel_axis=None),
                dtype=np.float32,
            )
        except TypeError:
            return np.ascontiguousarray(
                denoise_tv_chambolle(x_norm, weight=float(weight), n_iter_max=int(iterations), multichannel=False),
                dtype=np.float32,
            )
    except Exception:
        z = np.ascontiguousarray(x_norm, dtype=np.float32)
        step = min(0.2, 1.0 / (1.0 + 6.0 * max(float(weight), 0.0)))
        for _ in range(int(iterations)):
            raise_if_cancelled(cancel_check, "TV prior generation cancelled.")
            gradient = z - x_norm
            _add_smoothed_tv_gradient_inplace(gradient, z, float(weight), float(epsilon), max(1, min(16, z.shape[0])))
            z -= step * gradient
        return np.ascontiguousarray(z, dtype=np.float32)


def _score_prior_candidate(
    x_norm: np.ndarray,
    candidate: np.ndarray,
    support: np.ndarray,
    tune_anchor_data: AnchorValidationData | None,
    geometry_params: GeometryParams,
    config: AppConfig,
    gpu_ids: Sequence[int] | None,
    use_gpu: bool,
    fdk_anchor_residual: float,
    logger: Callable[[str], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> dict[str, object]:
    residual = _anchor_residual(candidate, tune_anchor_data, geometry_params, config, gpu_ids, use_gpu, logger, cancel_check)
    residual_ratio = 1.0 if not np.isfinite(residual) or not np.isfinite(fdk_anchor_residual) else residual / max(fdk_anchor_residual, 1e-12)
    correction_fraction = float(
        np.sum(np.abs((candidate - x_norm)[support]), dtype=np.float64)
        / max(float(np.sum(np.abs(x_norm[support]), dtype=np.float64)), 1e-12)
    )
    g_fdk = _gradient_magnitude(x_norm)
    g_candidate = _gradient_magnitude(candidate)
    support_grad = g_fdk[support]
    if support_grad.size:
        threshold = float(np.percentile(support_grad, 95))
        edge_mask = support & (g_fdk >= threshold)
    else:
        edge_mask = support
    edge_retention = float(np.median(g_candidate[edge_mask]) / max(float(np.median(g_fdk[edge_mask])), 1e-12)) if np.any(edge_mask) else 1.0
    background = candidate[~support] if np.any(~support) else candidate - _safe_gaussian(candidate, 1.5)
    background_noise = _mad(background)
    accepted = (
        residual_ratio <= float(config.prior_scoring.max_anchor_residual_ratio)
        and edge_retention >= float(config.prior_scoring.min_edge_retention)
        and correction_fraction <= float(config.prior_scoring.max_correction_fraction)
    )
    return {
        "anchor_residual_ratio": float(residual_ratio),
        "correction_fraction": float(correction_fraction),
        "edge_retention": float(edge_retention),
        "background_noise": float(background_noise),
        "accepted": bool(accepted),
    }


def _anchor_residual(
    volume: np.ndarray,
    anchor_data: AnchorValidationData | None,
    geometry_params: GeometryParams,
    config: AppConfig,
    gpu_ids: Sequence[int] | None,
    use_gpu: bool,
    logger: Callable[[str], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> float:
    if anchor_data is None or anchor_data.projections is None or not anchor_data.projections.size:
        return float("nan")
    operator = TigreConeBeamOperator(
        geometry_params,
        angles_for_tigre(anchor_data.angles_rad, config.geometry.invert_angle_sign_for_tigre),
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
    residual = predicted - anchor_data.projections
    denom = anchor_data.projections
    if anchor_data.weights is not None:
        weights = np.asarray(anchor_data.weights, dtype=np.float32)
        residual = residual * weights[:, None, None]
        denom = denom * weights[:, None, None]
    return float(np.sum(np.asarray(residual, dtype=np.float64) ** 2) / max(float(np.sum(np.asarray(denom, dtype=np.float64) ** 2)), 1e-12))


def _build_confidence_map(x_norm: np.ndarray, x_prior_norm: np.ndarray, support: np.ndarray, config: AppConfig) -> np.ndarray:
    delta = np.abs(x_prior_norm - x_norm)
    support_delta = delta[support]
    tau_delta = 2.0 * (float(np.median(support_delta)) if support_delta.size else float(np.median(delta))) + 1e-12
    c_delta = np.exp(-((delta / tau_delta) ** 2))
    g_fdk = _gradient_magnitude(x_norm)
    g_prior = _gradient_magnitude(x_prior_norm)
    loss = np.maximum(g_fdk - g_prior, 0.0) / np.maximum(g_fdk, 1e-12)
    c_grad = np.exp(-((loss / max(float(config.confidence.tau_gradient_loss), 1e-12)) ** 2))
    confidence = c_delta * c_grad
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        pass
    else:
        sigma = max(float(config.confidence.blur_sigma_voxels), 0.0)
        if sigma:
            confidence = gaussian_filter(confidence, sigma=sigma)
    return np.clip(confidence, float(config.confidence.C_min), float(config.confidence.C_max)).astype(np.float32)


def _gradient_magnitude(x: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(x, dtype=np.float32)
    gy = np.zeros_like(x, dtype=np.float32)
    gz = np.zeros_like(x, dtype=np.float32)
    gx[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
    gy[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
    gz[:-1, :, :] = x[1:, :, :] - x[:-1, :, :]
    return np.sqrt(gx * gx + gy * gy + gz * gz, dtype=np.float32)


def _safe_gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return np.zeros_like(x, dtype=np.float32)
    return np.asarray(gaussian_filter(x, sigma=sigma), dtype=np.float32)


def _mad(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    med = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - med)))


def _save_prior_outputs(
    output_folder: Path,
    x_fdk: np.ndarray,
    x_norm: np.ndarray,
    chosen_norm: np.ndarray,
    result: PriorResult,
    config: AppConfig,
) -> None:
    np.save(output_folder / "prior_selected.npy", result.x_prior.astype(np.float32))
    save_stack_tiff(output_folder / "prior_selected.tif", result.x_prior)
    np.save(output_folder / "prior_confidence.npy", result.confidence.astype(np.float32))
    save_stack_tiff(output_folder / "prior_confidence.tif", result.confidence)
    np.save(output_folder / "support_mask.npy", result.support_mask.astype(np.uint8))
    difference = result.x_prior - np.asarray(x_fdk, dtype=np.float32)
    np.save(output_folder / "prior_difference.npy", difference.astype(np.float32))
    save_stack_tiff(output_folder / "prior_difference.tif", difference)
    _write_dicts_csv(output_folder / "prior_metrics.csv", result.metrics)
    save_config(config, output_folder / "prior_config_used.yaml")
    save_volume_preview(output_folder / "prior_preview.png", result.x_prior, "Selected prior")
    save_volume_preview(output_folder / "prior_difference_preview.png", difference, "Prior minus FDK")
    save_volume_preview(output_folder / "prior_confidence_preview.png", result.confidence, "Prior confidence")
    np.save(output_folder / "prior_selected_normalized.npy", chosen_norm.astype(np.float32))
    np.save(output_folder / "fdk_normalized.npy", x_norm.astype(np.float32))


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


def _format_float(value: object) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(number):
        return str(number)
    return f"{number:.6g}"


def _log(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)
