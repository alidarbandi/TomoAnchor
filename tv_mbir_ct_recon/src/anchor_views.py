from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import AnchorSplitConfig
from .fast_types import AnchorSplit


def split_anchor_views(angles_rad: np.ndarray, split_config: AnchorSplitConfig) -> AnchorSplit:
    angles = np.asarray(angles_rad, dtype=np.float64).reshape(-1)
    n_views = int(angles.size)
    if n_views <= 0:
        return AnchorSplit(_empty_indices(), _empty_indices(), _empty_indices())
    mode = str(split_config.mode or "interleaved").strip().lower()
    if mode not in {"interleaved", "even", "random"}:
        raise ValueError(f"Unsupported anchor split mode: {split_config.mode}")

    tune_count = _bounded_count(split_config.tune_count, n_views)
    qc_count = _bounded_count(split_config.qc_count, n_views - tune_count)
    remaining = n_views - tune_count - qc_count
    recon_count = remaining if split_config.recon_count is None else _bounded_count(split_config.recon_count, remaining)

    order = _angular_order(angles)
    if mode == "random":
        rng = np.random.default_rng(int(split_config.random_seed))
        order = np.asarray(order, dtype=np.int64).copy()
        rng.shuffle(order)

    tune = _pick_evenly_spaced(order, tune_count)
    remaining_after_tune = np.asarray([idx for idx in order if idx not in set(tune.tolist())], dtype=np.int64)
    qc = _pick_evenly_spaced(remaining_after_tune, qc_count)
    used = set(tune.tolist()) | set(qc.tolist())
    recon_candidates = np.asarray([idx for idx in order if idx not in used], dtype=np.int64)
    recon = _pick_evenly_spaced(recon_candidates, recon_count)
    return AnchorSplit(
        recon_indices=np.sort(recon).astype(np.int64),
        tune_indices=np.sort(tune).astype(np.int64),
        qc_indices=np.sort(qc).astype(np.int64),
    )


def save_anchor_split(path: str | Path, split: AnchorSplit, angles_rad: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels: dict[int, str] = {}
    labels.update({int(idx): "recon" for idx in split.recon_indices})
    labels.update({int(idx): "tune" for idx in split.tune_indices})
    labels.update({int(idx): "qc" for idx in split.qc_indices})
    angles = np.asarray(angles_rad, dtype=np.float64)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "angle_rad", "angle_deg", "split"])
        writer.writeheader()
        for index in range(len(angles)):
            writer.writerow(
                {
                    "index": index,
                    "angle_rad": f"{angles[index]:.10g}",
                    "angle_deg": f"{np.rad2deg(angles[index]):.10g}",
                    "split": labels.get(index, "unused"),
                }
            )
    return output


def anchor_split_report(split: AnchorSplit) -> list[str]:
    return [
        "Anchor split",
        "------------",
        f"Reconstruction anchors: {len(split.recon_indices)}",
        f"Tuning anchors: {len(split.tune_indices)}",
        f"QC anchors: {len(split.qc_indices)}",
        f"Recon indices: {_format_indices(split.recon_indices)}",
        f"Tune indices: {_format_indices(split.tune_indices)}",
        f"QC indices: {_format_indices(split.qc_indices)}",
    ]


def _bounded_count(value: int | None, upper: int) -> int:
    if value is None:
        return max(0, int(upper))
    return min(max(0, int(value)), max(0, int(upper)))


def _angular_order(angles_rad: np.ndarray) -> np.ndarray:
    wrapped = np.mod(np.asarray(angles_rad, dtype=np.float64), 2.0 * np.pi)
    return np.argsort(wrapped, kind="mergesort").astype(np.int64)


def _pick_evenly_spaced(candidates: np.ndarray, count: int) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if count <= 0 or candidates.size == 0:
        return _empty_indices()
    count = min(int(count), int(candidates.size))
    if count == candidates.size:
        return np.asarray(candidates, dtype=np.int64)
    positions = np.linspace(0, candidates.size - 1, count, endpoint=True)
    chosen_positions = np.unique(np.rint(positions).astype(np.int64))
    if chosen_positions.size < count:
        missing = [idx for idx in range(candidates.size) if idx not in set(chosen_positions.tolist())]
        chosen_positions = np.concatenate([chosen_positions, np.asarray(missing[: count - chosen_positions.size])])
    return np.asarray(candidates[np.sort(chosen_positions[:count])], dtype=np.int64)


def _empty_indices() -> np.ndarray:
    return np.asarray([], dtype=np.int64)


def _format_indices(indices: np.ndarray) -> str:
    values = [str(int(value)) for value in np.asarray(indices, dtype=np.int64)]
    return ", ".join(values) if values else "(none)"
