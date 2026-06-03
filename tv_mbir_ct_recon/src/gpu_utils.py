from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ResolvedGPUSelection:
    selector: str
    gpu_ids: tuple[int, ...]
    warnings: tuple[str, ...] = ()

    @property
    def primary_gpu_id(self) -> int | None:
        return self.gpu_ids[0] if self.gpu_ids else None

    @property
    def uses_multiple_gpus(self) -> bool:
        return len(self.gpu_ids) > 1

    def describe(self) -> str:
        if not self.gpu_ids:
            return "CPU only"
        if len(self.gpu_ids) == 1:
            return f"GPU {self.gpu_ids[0]}"
        return f"GPUs {', '.join(str(gpu_id) for gpu_id in self.gpu_ids)}"


def resolve_gpu_selection(
    use_gpu: bool,
    selector: str | None = None,
    legacy_gpu_id: int = 0,
    available_gpu_ids: Sequence[int] | None = None,
) -> ResolvedGPUSelection:
    normalized_selector = str(selector or "auto").strip() or "auto"
    available = _normalize_gpu_ids(available_gpu_ids or ())
    warnings: list[str] = []
    if not use_gpu:
        return ResolvedGPUSelection(normalized_selector, ())

    lower = normalized_selector.lower()
    if lower in {"auto", "all"}:
        if available:
            return ResolvedGPUSelection(normalized_selector, available)
        warnings.append(
            "Could not enumerate NVIDIA GPUs for auto/all selection; falling back to the legacy primary GPU ID."
        )
        return ResolvedGPUSelection(normalized_selector, (int(legacy_gpu_id),), tuple(warnings))

    if lower in {"single", "primary", "legacy"}:
        return ResolvedGPUSelection(normalized_selector, (int(legacy_gpu_id),))

    gpu_ids = _parse_gpu_id_text(normalized_selector)
    if available:
        missing = [gpu_id for gpu_id in gpu_ids if gpu_id not in available]
        if missing:
            raise ValueError(
                f"Requested GPU IDs {missing} are not available. Detected NVIDIA GPU IDs: {list(available)}"
            )
    return ResolvedGPUSelection(normalized_selector, gpu_ids)


def format_gpu_selector(gpu_ids: Sequence[int]) -> str:
    normalized = _normalize_gpu_ids(gpu_ids)
    return ",".join(str(gpu_id) for gpu_id in normalized) if normalized else "auto"


def _parse_gpu_id_text(text: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in str(text).replace(";", ",").split(",")]
    values: list[int] = []
    for token in tokens:
        if not token:
            continue
        try:
            gpu_id = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid GPU selector '{text}'. Use 'auto', 'all', 'single', or a comma-separated list like '0,1'."
            ) from exc
        if gpu_id < 0:
            raise ValueError(f"GPU IDs must be non-negative integers, got {gpu_id}.")
        if gpu_id not in values:
            values.append(gpu_id)
    if not values:
        raise ValueError(
            f"Invalid GPU selector '{text}'. Use 'auto', 'all', 'single', or a comma-separated list like '0,1'."
        )
    return tuple(values)


def _normalize_gpu_ids(gpu_ids: Sequence[int]) -> tuple[int, ...]:
    values: list[int] = []
    for value in gpu_ids:
        gpu_id = int(value)
        if gpu_id < 0:
            raise ValueError(f"GPU IDs must be non-negative integers, got {gpu_id}.")
        if gpu_id not in values:
            values.append(gpu_id)
    return tuple(values)
