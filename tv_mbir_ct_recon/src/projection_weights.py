from __future__ import annotations

import numpy as np


def estimate_projection_scalar_weights(
    raw_stack: np.ndarray | None,
    transmission_stack: np.ndarray | None,
    exposure_s: float | None,
    fallback_multiplier: float = 1.0,
) -> np.ndarray:
    stack = raw_stack if raw_stack is not None and np.asarray(raw_stack).size else transmission_stack
    if stack is None or not np.asarray(stack).size:
        raise ValueError("Need raw_stack or transmission_stack to infer the projection count for weights.")
    n_views = int(np.asarray(stack).shape[0])
    fallback = float(fallback_multiplier)
    if exposure_s is not None and float(exposure_s) > 0:
        return np.full(n_views, fallback * np.sqrt(float(exposure_s)), dtype=np.float32)
    if raw_stack is not None and np.asarray(raw_stack).size:
        raw = np.asarray(raw_stack, dtype=np.float32)
        robust_mean = np.nanmean(np.clip(raw, 0.0, None), axis=(1, 2))
        robust_mean = np.nan_to_num(robust_mean, nan=0.0, posinf=0.0, neginf=0.0)
        positive = robust_mean[robust_mean > 0]
        if positive.size:
            floor = max(float(np.percentile(positive, 5)), 1e-6)
            return (fallback * np.sqrt(np.maximum(robust_mean, floor))).astype(np.float32)
    return np.full(n_views, fallback, dtype=np.float32)


def normalize_projection_weights(
    main_weights: np.ndarray,
    anchor_weights: np.ndarray | None = None,
    max_weight_ratio: float = 10.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    main = np.asarray(main_weights, dtype=np.float32).reshape(-1)
    finite_main = main[np.isfinite(main) & (main > 0)]
    median_main = float(np.median(finite_main)) if finite_main.size else 1.0
    median_main = max(median_main, 1e-6)
    main = np.nan_to_num(main / median_main, nan=1.0, posinf=1.0, neginf=1.0)
    main = np.clip(main, 1.0 / max(float(max_weight_ratio), 1.0), float(max_weight_ratio)).astype(np.float32)
    if anchor_weights is None:
        return main, None
    anchor = np.asarray(anchor_weights, dtype=np.float32).reshape(-1)
    anchor = np.nan_to_num(anchor / median_main, nan=1.0, posinf=1.0, neginf=1.0)
    anchor = np.clip(anchor, 1.0 / max(float(max_weight_ratio), 1.0), float(max_weight_ratio)).astype(np.float32)
    return main, anchor


def combine_duplicate_angle_projections(
    projections: np.ndarray,
    angles: np.ndarray,
    weights: np.ndarray | None,
    angle_tolerance_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[dict[str, object]]]:
    proj = np.asarray(projections, dtype=np.float32)
    ang = np.asarray(angles, dtype=np.float32).reshape(-1)
    if proj.ndim != 3:
        raise ValueError(f"Projection stack must have shape (views, rows, cols), got {proj.shape}")
    if proj.shape[0] != ang.size:
        raise ValueError("Projection and angle counts do not match.")
    w = None if weights is None else np.asarray(weights, dtype=np.float32).reshape(-1)
    if w is not None and w.size != ang.size:
        raise ValueError("Projection weight count does not match projection count.")
    if ang.size == 0:
        return proj.copy(), ang.copy(), None if w is None else w.copy(), []
    order = np.argsort(ang.astype(np.float64), kind="mergesort")
    used = np.zeros(ang.size, dtype=bool)
    merged_proj: list[np.ndarray] = []
    merged_angles: list[float] = []
    merged_weights: list[float] = []
    report: list[dict[str, object]] = []
    tol = max(float(angle_tolerance_rad), 0.0)
    for sorted_idx in order:
        idx = int(sorted_idx)
        if used[idx]:
            continue
        cluster = [idx]
        used[idx] = True
        for other_sorted_idx in order:
            other = int(other_sorted_idx)
            if used[other]:
                continue
            distance = _angle_distance_rad(float(ang[other]), float(ang[idx]))
            if distance <= tol:
                cluster.append(other)
                used[other] = True
        cluster_arr = np.asarray(cluster, dtype=np.int64)
        if w is None:
            cluster_weights = np.ones(cluster_arr.size, dtype=np.float32)
            out_weight = 1.0
        else:
            cluster_weights = np.maximum(w[cluster_arr], 1e-6)
            out_weight = float(np.sqrt(np.sum(cluster_weights * cluster_weights)))
        variance_weights = cluster_weights * cluster_weights
        denom = float(np.sum(variance_weights))
        weighted = np.sum(proj[cluster_arr] * variance_weights[:, None, None], axis=0) / max(denom, 1e-12)
        angle_value = float(np.sum(ang[cluster_arr] * variance_weights) / max(denom, 1e-12))
        merged_proj.append(np.asarray(weighted, dtype=np.float32))
        merged_angles.append(angle_value)
        merged_weights.append(out_weight)
        if cluster_arr.size > 1:
            report.append(
                {
                    "merged_count": int(cluster_arr.size),
                    "indices": " ".join(str(int(value)) for value in cluster_arr),
                    "angle_rad": angle_value,
                }
            )
    out_proj = np.ascontiguousarray(np.stack(merged_proj, axis=0), dtype=np.float32)
    out_angles = np.ascontiguousarray(np.asarray(merged_angles, dtype=np.float32), dtype=np.float32)
    out_weights = None if weights is None else np.asarray(merged_weights, dtype=np.float32)
    return out_proj, out_angles, out_weights, report


def _angle_distance_rad(a: float, b: float) -> float:
    return abs(((a - b + np.pi) % (2.0 * np.pi)) - np.pi)
