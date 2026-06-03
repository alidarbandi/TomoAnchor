from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

try:
    import yaml
except ImportError:  # pragma: no cover - depends on the active environment
    yaml = None


T = TypeVar("T")


AVAILABLE_FDK_FILTERS: tuple[str, ...] = ("ram_lak", "shepp_logan", "cosine", "hann", "hamming")


@dataclass
class InputConfig:
    projection_folder: str = ""
    flat_folder: str = ""
    dark_folder: str = ""
    metadata_path: str = ""
    output_folder: str = ""


@dataclass
class MetadataConfig:
    filename_column: str = ""
    angle_column: str = ""
    angle_rad_column: str = ""
    x_shift_column: str = ""
    y_shift_column: str = ""
    remove_duplicate_endpoint: bool = True
    reverse_angle_order: bool = False


@dataclass
class PreprocessingConfig:
    use_dark: bool = True
    use_flat: bool = True
    negative_log: bool = True
    epsilon: float = 1e-6
    transmission_clip_min: float = 1e-6
    transmission_clip_max: float = 2.0
    max_transmission_for_log: float = 2.0
    clip_transmission: bool = True
    clip_negative_attenuation_to_zero: bool = True
    crop: tuple[int, int, int, int] = (0, 0, 0, 0)  # top, bottom, left, right after binning
    binning: tuple[int, int] = (1, 1)
    flip_horizontal: bool = False
    flip_vertical: bool = False
    transpose_for_tigre: bool = False
    truncation_correction: bool = False
    truncation_extension_fraction: float = 0.1


@dataclass
class GeometryConfig:
    source_detector_distance_mm: float | None = None
    source_origin_distance_mm: float | None = None
    effective_pixel_size_um: float | None = 11.0
    detector_pixel_size_mm: tuple[float | None, float | None] = (None, None)  # vertical, horizontal
    detector_pixels: tuple[int | None, int | None] = (None, None)  # rows, cols
    voxel_size_mm: tuple[float | None, float | None, float | None] = (0.011, 0.011, 0.011)  # z, y, x
    volume_voxels: tuple[int | None, int | None, int | None] = (None, None, None)  # z, y, x
    detector_offset_pixels: tuple[float, float] = (0.0, 0.0)  # vertical, horizontal
    origin_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)  # z, y, x
    angle_units: str = "degrees"
    angle_start: float = 0.0
    angle_end: float = 360.0
    endpoint_included: bool = False
    angle_direction: str = "ccw"
    angle_file: str | None = None
    invert_angle_sign_for_tigre: bool = True
    fdk_filter: str = "ram_lak"


@dataclass
class AlignmentConfig:
    center_offset_pixels: float = 0.0
    center_shift_sign: float = 1.0
    center_offset_method: str = "detector_offset"
    use_center_correction: bool = True
    use_drift_correction: bool = True
    drift_shift_file: str | None = None
    estimate_drift: bool = False
    drift_units: str = "pixels_after_binning"
    drift_stage: str = "transmission"
    x_shift_sign: int = 1
    y_shift_sign: int = 1
    shift_interpretation_sign: int = 1
    shift_interpolation_order: int = 1
    shift_boundary_mode: str = "nearest"


@dataclass
class AnchorInputConfig:
    projection_folder: str = ""
    flat_folder: str = ""
    dark_folder: str = ""
    metadata_path: str = ""


@dataclass
class AnchorExposureConfig:
    main_exposure_s: float | None = None
    anchor_exposure_s: float | None = None
    anchor_weight_multiplier: float = 1.0
    max_weight_ratio: float = 10.0


@dataclass
class AnchorSplitConfig:
    mode: str = "interleaved"
    recon_count: int | None = None
    tune_count: int = 6
    qc_count: int = 6
    random_seed: int = 23
    use_tune_anchors_in_final: bool = True
    use_qc_anchors_in_final: bool = False


@dataclass
class AnchorMergeConfig:
    duplicate_angle_tolerance_deg: float = 0.01
    combine_duplicate_angles: bool = True
    sort_by_angle: bool = True


@dataclass
class AnchorPreprocessingConfig(PreprocessingConfig):
    inherit_main: bool = True


@dataclass
class AnchorAlignmentConfig(AlignmentConfig):
    inherit_main: bool = True


@dataclass
class AnchorConfig:
    enabled: bool = False
    input: AnchorInputConfig = field(default_factory=AnchorInputConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    preprocessing: AnchorPreprocessingConfig = field(default_factory=AnchorPreprocessingConfig)
    alignment: AnchorAlignmentConfig = field(default_factory=AnchorAlignmentConfig)
    exposure: AnchorExposureConfig = field(default_factory=AnchorExposureConfig)
    split: AnchorSplitConfig = field(default_factory=AnchorSplitConfig)
    merge: AnchorMergeConfig = field(default_factory=AnchorMergeConfig)


@dataclass
class FDKSweepConfig:
    enabled: bool = True
    lowres_factor: int = 2
    filters: list[str] = field(default_factory=lambda: list(AVAILABLE_FDK_FILTERS))
    selection_filter: str = "auto"
    cutoffs: list[float] = field(default_factory=lambda: [1.0])
    metric_anchor_weight: float = 1.0
    metric_tv_weight: float = 0.10
    metric_negative_weight: float = 0.10
    use_tune_anchors: bool = True
    detector_mask_path: str | None = None
    save_candidate_previews: bool = True


@dataclass
class PriorConfig:
    enabled: bool = True
    method_order: list[str] = field(default_factory=lambda: ["tv"])
    normalize_low_percentile: float = 0.5
    normalize_high_percentile: float = 99.5
    clip_min: float = -0.1
    clip_max: float = 1.2
    support_gaussian_sigma_voxels: float = 2.0
    support_dilation_voxels: int = 8
    noise_estimation: str = "highpass_mad"
    noise_highpass_sigma_voxels: float = 1.5
    tv_weights: list[float] = field(default_factory=lambda: [0.005, 0.01, 0.02, 0.04])
    tv_epsilon: float = 1e-4
    tv_iterations: int = 50
    optional_bm4d_enabled: bool = False
    bm4d_sigma_multipliers: list[float] = field(default_factory=lambda: [0.5, 0.7, 1.0])
    slab_depth: int = 96
    slab_overlap: int = 12


@dataclass
class PriorScoringConfig:
    max_anchor_residual_ratio: float = 1.10
    min_edge_retention: float = 0.85
    max_correction_fraction: float = 0.15
    choose_lowest_background_noise_among_valid: bool = True


@dataclass
class ConfidenceConfig:
    enabled: bool = True
    method: str = "correction_and_edge"
    C_min: float = 0.05
    C_max: float = 1.0
    tau_delta_mode: str = "median_times_2"
    tau_gradient_loss: float = 0.35
    blur_sigma_voxels: float = 1.5


@dataclass
class MBIRLiteConfig:
    enabled: bool = True
    solver: str = "anchored_streaming_subset_tv"
    n_sweeps: int = 5
    projection_batch_size: int = 32
    ordered_subset_count: int = 8
    lambda_tv: float = 1e-4
    lambda_tv_mode: str = "absolute"
    rho_prior: float = 0.05
    rho_mode: str = "absolute"
    rho_beta: float = 0.15
    tv_epsilon: float = 1e-4
    positivity: bool = True
    support_mask: bool = True
    subset_tv_power_iterations: int = 2
    subset_tv_step_safety: float = 0.5
    pdhg_chunk_slices: int = 16
    use_projection_weights: bool = True
    stop_on_qc_anchor_residual: bool = True
    save_every: int = 1


@dataclass
class FastReconConfig:
    enabled: bool = False
    output_subfolder: str = "fast_recon"
    save_intermediate: bool = True
    make_qc_report: bool = True
    resume_run_folder: str | None = None


@dataclass
class InitializationConfig:
    mode: str = "fdk"  # fdk, existing_fdk, zeros, constant
    fdk_volume_path: str | None = None
    constant_value: float = 0.0


@dataclass
class MBIRConfig:
    solver: str = "auto"  # auto, admm, pdhg_low_memory, streaming_subset_tv
    lambda_tv: float = 1e-3
    rho: float = 5e-2
    tv_type: str = "isotropic_3d"
    tv_epsilon: float = 1e-8
    positivity: bool = True
    max_admm_iterations: int = 50
    inner_cg_iterations: int = 10
    tolerance_primal: float = 1e-4
    tolerance_dual: float = 1e-4
    tolerance_relative_change: float = 1e-5
    cg_tolerance: float = 1e-4
    save_every: int = 5
    debug_pixel_binning: tuple[int, int] = (1, 1)  # extra MBIR-only detector binning, y/x
    projection_stride: int = 1  # use every Nth projection for MBIR debug runs
    memory_mode: str = "auto"  # auto, full_gpu, distributed_full_data, projection_streaming, ordered_subsets
    projection_batch_size: int = 64
    ordered_subset_count: int = 1
    pdhg_power_iterations: int = 3
    pdhg_step_safety: float = 0.95
    pdhg_theta: float = 1.0
    pdhg_dual_dtype: str = "float16"  # float16 or float32 for TV dual arrays
    pdhg_chunk_slices: int = 16
    subset_tv_power_iterations: int = 2
    subset_tv_step_safety: float = 0.7
    use_weights: bool = False
    weight_mode: str = "none"
    weight_file: str | None = None
    photon_count_based_weights: bool = False
    flat_intensity_based_weights: bool = False


@dataclass
class GPUConfig:
    use_gpu: bool = True
    gpu_id: int = 0
    gpu_selector: str = "auto"  # auto, all, single, or comma-separated IDs such as "0,1"
    use_cupy_for_tv_ops: bool = True


@dataclass
class LoggingConfig:
    verbose: bool = True
    save_plots: bool = True
    save_intermediate: bool = True


@dataclass
class AppConfig:
    input: InputConfig = field(default_factory=InputConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    mbir: MBIRConfig = field(default_factory=MBIRConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    anchors: AnchorConfig = field(default_factory=AnchorConfig)
    fdk_sweep: FDKSweepConfig = field(default_factory=FDKSweepConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    prior_scoring: PriorScoringConfig = field(default_factory=PriorScoringConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    mbir_lite: MBIRLiteConfig = field(default_factory=MBIRLiteConfig)
    fast_recon: FastReconConfig = field(default_factory=FastReconConfig)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AppConfig":
        return _dataclass_from_dict(cls, values)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain_dict(self)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        raw = yaml.safe_load(text) or {}
    else:
        raw = _parse_simple_yaml(text)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return AppConfig.from_dict(raw)


def save_config(config: AppConfig, path: str | Path) -> Path:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        text = yaml.safe_dump(config.to_dict(), sort_keys=False)
    else:
        text = _dump_simple_yaml(config.to_dict())
    config_path.write_text(text, encoding="utf-8")
    return config_path


def _dataclass_from_dict(cls: type[T], values: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for field in fields(cls):
        if field.name not in values:
            continue
        value = values[field.name]
        field_type = type_hints.get(field.name, field.type)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[field.name] = _dataclass_from_dict(field_type, value)
        else:
            kwargs[field.name] = _coerce_value(field_type, value)
    return cls(**kwargs)


def _coerce_value(field_type: Any, value: Any) -> Any:
    origin = get_origin(field_type)
    if origin is tuple and isinstance(value, list):
        return tuple(value)
    if origin is tuple and isinstance(value, tuple):
        return value
    if origin is None:
        return value
    args = get_args(field_type)
    if type(None) in args and value is None:
        return None
    return value


def _to_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_plain_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain_dict(item) for key, item in value.items()}
    return value


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple nested YAML emitted by this project.

    This fallback exists so the GUI can start in the original TIGRE conda
    environment even when PyYAML is not installed. It intentionally supports
    only mappings with scalar or inline-list values.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = _strip_inline_comment(raw_line.rstrip())
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if ":" not in content:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key, value_text = content.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid YAML indentation near: {raw_line}")
        parent = stack[-1][1]
        if value_text == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_simple_yaml_value(value_text)
    return root


def _parse_simple_yaml_value(text: str) -> Any:
    lower = text.lower()
    if lower in {"null", "none", "~"}:
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_yaml_value(part.strip()) for part in _split_inline_list(inner)]
    try:
        if all(ch not in text for ch in ".eE"):
            return int(text)
        return float(text)
    except ValueError:
        return text


def _split_inline_list(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for char in text:
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
        if char == "," and quote is None:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return parts


def _strip_inline_comment(line: str) -> str:
    quote: str | None = None
    for index, char in enumerate(line):
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
        if char == "#" and quote is None and (index == 0 or line[index - 1].isspace()):
            return line[:index].rstrip()
    return line


def _dump_simple_yaml(values: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in values.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_dump_simple_yaml(value, indent + 2).rstrip())
        else:
            lines.append(f"{prefix}{key}: {_format_simple_yaml_value(value)}")
    return "\n".join(lines) + "\n"


def _format_simple_yaml_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_simple_yaml_value(item) for item in value) + "]"
    text = str(value).replace('"', '\\"')
    return f'"{text}"'
