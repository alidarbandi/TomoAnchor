from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Sequence


_NVIDIA_SMI_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)


@dataclass(frozen=True)
class GPULiveSample:
    gpu_id: int
    name: str
    utilization_gpu_pct: float | None
    utilization_memory_pct: float | None
    memory_used_gb: float | None
    memory_total_gb: float | None
    temperature_c: float | None
    power_w: float | None

    @property
    def memory_percent(self) -> float | None:
        if self.memory_used_gb is None or self.memory_total_gb is None or self.memory_total_gb <= 0:
            return None
        return 100.0 * float(self.memory_used_gb) / float(self.memory_total_gb)


@dataclass(frozen=True)
class DeviceMonitorSnapshot:
    timestamp_s: float
    gpus: tuple[GPULiveSample, ...]
    source: str = "nvidia-smi"


def nvidia_gpu_monitor_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def query_nvidia_gpu_snapshot(timeout_s: float = 3.0) -> DeviceMonitorSnapshot | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                f"--query-gpu={','.join(_NVIDIA_SMI_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_s)),
        )
    except Exception:
        return None
    snapshot = parse_nvidia_smi_snapshot(completed.stdout, sampled_at_s=time.time())
    return snapshot if snapshot.gpus else None


def parse_nvidia_smi_snapshot(text: str, sampled_at_s: float | None = None) -> DeviceMonitorSnapshot:
    samples: list[GPULiveSample] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < len(_NVIDIA_SMI_FIELDS):
            continue
        try:
            gpu_id = int(parts[0])
        except ValueError:
            continue
        samples.append(
            GPULiveSample(
                gpu_id=gpu_id,
                name=parts[1],
                utilization_gpu_pct=_parse_optional_float(parts[2]),
                utilization_memory_pct=_parse_optional_float(parts[3]),
                memory_used_gb=_mb_to_gb(_parse_optional_float(parts[4])),
                memory_total_gb=_mb_to_gb(_parse_optional_float(parts[5])),
                temperature_c=_parse_optional_float(parts[6]),
                power_w=_parse_optional_float(parts[7]),
            )
        )
    return DeviceMonitorSnapshot(
        timestamp_s=float(sampled_at_s if sampled_at_s is not None else time.time()),
        gpus=tuple(samples),
    )


def format_gpu_snapshot_summary(
    snapshot: DeviceMonitorSnapshot,
    gpu_ids: Sequence[int] | None = None,
) -> str:
    selected_ids = None if gpu_ids is None else {int(gpu_id) for gpu_id in gpu_ids}
    gpus = [
        gpu for gpu in snapshot.gpus if selected_ids is None or int(gpu.gpu_id) in selected_ids
    ]
    return " | ".join(_format_gpu_sample_summary(sample) for sample in gpus)


def _format_gpu_sample_summary(sample: GPULiveSample) -> str:
    parts = [f"GPU {sample.gpu_id}"]
    util_text = _format_optional_value(sample.utilization_gpu_pct, suffix="% util")
    if util_text:
        parts.append(util_text)
    mem_text = _format_memory_text(sample)
    if mem_text:
        parts.append(mem_text)
    mem_bw_text = _format_optional_value(sample.utilization_memory_pct, suffix="% mem-bw")
    if mem_bw_text:
        parts.append(mem_bw_text)
    temp_text = _format_optional_value(sample.temperature_c, suffix=" C")
    if temp_text:
        parts.append(temp_text)
    power_text = _format_optional_value(sample.power_w, suffix=" W")
    if power_text:
        parts.append(power_text)
    return ", ".join(parts)


def _format_memory_text(sample: GPULiveSample) -> str:
    if sample.memory_used_gb is None or sample.memory_total_gb is None:
        return ""
    percent = sample.memory_percent
    if percent is None:
        return f"{sample.memory_used_gb:.1f}/{sample.memory_total_gb:.1f} GB"
    return f"{sample.memory_used_gb:.1f}/{sample.memory_total_gb:.1f} GB ({percent:.0f}%)"


def _format_optional_value(value: float | None, suffix: str) -> str:
    if value is None:
        return ""
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1e-6:
        return f"{int(round(numeric))}{suffix}"
    return f"{numeric:.1f}{suffix}"


def _parse_optional_float(value: str) -> float | None:
    cleaned = str(value).strip()
    if not cleaned:
        return None
    lowered = cleaned.lower().strip("[]")
    if lowered in {"n/a", "na", "not supported", "not applicable", "unknown"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _mb_to_gb(value_mb: float | None) -> float | None:
    if value_mb is None:
        return None
    return float(value_mb) / 1024.0
