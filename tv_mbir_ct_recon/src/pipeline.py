from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .admm_tv_mbir import ADMMResult, ADMMTVMBIRReconstructor
from .cancel_utils import raise_if_cancelled
from .config import AppConfig
from .distributed_tigre import estimate_distributed_full_data_overhead_gb
from .geometry_tigre import angles_for_tigre, geometry_report, params_from_config
from .gpu_utils import resolve_gpu_selection
from .io_utils import read_tiff_float32, save_image
from .logging_utils import MemoryLog, write_text_lines
from .metadata_zeiss import (
    ANGLE_HINTS,
    ANGLE_RAD_HINTS,
    FILENAME_HINTS,
    X_SHIFT_HINTS,
    Y_SHIFT_HINTS,
    find_metadata_csv,
    load_metadata_csv,
    suggest_column,
    suggest_optional_column,
    validate_metadata,
)
from .metrics import write_metrics_csv
from .memory_utils import (
    SYSTEM_RAM_SAFETY_FRACTION,
    estimate_mbir_memory,
    estimate_low_memory_pdhg_cpu_gb,
    minimum_available_system_memory_gb,
    query_all_nvidia_gpu_memory,
    query_system_memory,
    recommended_projection_batch_size,
    estimate_streaming_subset_tv_cpu_gb,
)
from .mbir_debug import (
    alignment_for_mbir_debug,
    geometry_for_mbir_debug,
    mbir_debug_is_active,
    preprocessing_for_mbir_debug,
    validation_for_mbir_debug,
)
from .output_manager import OutputManager, RunFolders
from .pdhg_tv_mbir import PDHGTVMBIRReconstructor
from .plotting import save_metrics_plots, save_volume_preview
from .preprocessing import prepare_tigre_projection_input, run_preprocessing
from .subset_tv_mbir import StreamingSubsetTVReconstructor
from .tigre_ops import TigreConeBeamOperator, fdk_initialization, gpu_diagnostics, run_fdk_reconstruction


@dataclass
class PipelineResult:
    run_folders: RunFolders
    attenuation_shape: tuple[int, int, int]
    fdk_volume: np.ndarray | None
    mbir_result: ADMMResult | None
    report_lines: list[str]


def preview_slice_counts_for_shape(volume_shape: tuple[int, int, int]) -> dict[str, int]:
    if len(volume_shape) != 3:
        raise ValueError(f"Expected a 3-D volume shape, got {volume_shape}")
    return {
        "Axial": max(1, int(volume_shape[0])),
        "Coronal": max(1, int(volume_shape[1])),
        "Sagittal": max(1, int(volume_shape[2])),
    }


def extract_preview_slice_for_gui(
    volume: np.ndarray,
    request: object | None = None,
) -> tuple[np.ndarray, str, int, dict[str, int]]:
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3-D MBIR preview volume, got shape {vol.shape}")
    slice_counts = preview_slice_counts_for_shape(tuple(int(v) for v in vol.shape))
    request_map = request if isinstance(request, dict) else {}
    view_text = str(request_map.get("view") or "Axial").strip().lower()
    if view_text.startswith("cor"):
        view = "Coronal"
    elif view_text.startswith("sag"):
        view = "Sagittal"
    else:
        view = "Axial"
    default_index = max(0, slice_counts[view] // 2)
    slice_index = int(request_map.get("slice_index", default_index))
    slice_index = min(max(slice_index, 0), slice_counts[view] - 1)
    if view == "Coronal":
        image = vol[:, slice_index, :]
    elif view == "Sagittal":
        image = vol[:, :, slice_index]
    else:
        image = vol[slice_index]
    return np.ascontiguousarray(image, dtype=np.float32), view, slice_index, slice_counts


def execute_pipeline(
    config: AppConfig,
    run_fdk: bool = False,
    run_mbir: bool = False,
    preprocess_only: bool = False,
    logger: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    mbir_progress_callback: Callable[[object], None] | None = None,
    mbir_preview_request_callback: Callable[[], object] | None = None,
) -> PipelineResult:
    output = OutputManager(config.input.output_folder or "output")
    folders = output.create_run()
    output.save_config(folders, config)
    log = MemoryLog(
        callback=logger,
        verbose=logger is None and config.logging.verbose,
        persist_path=folders.root / "log.txt",
    )
    log.write(f"Created run folder: {folders.root}")
    raise_if_cancelled(cancel_check, "Pipeline cancelled.")

    metadata_csv = find_metadata_csv(config.input.metadata_path)
    frame = load_metadata_csv(metadata_csv)
    columns = [str(column) for column in frame.columns]
    filename_column = config.metadata.filename_column or suggest_column(columns, FILENAME_HINTS)
    angle_column = config.metadata.angle_column or suggest_column(columns, ANGLE_HINTS)
    angle_rad_column = config.metadata.angle_rad_column or suggest_optional_column(columns, ANGLE_RAD_HINTS)
    x_shift_column = config.metadata.x_shift_column or suggest_optional_column(columns, X_SHIFT_HINTS)
    y_shift_column = config.metadata.y_shift_column or suggest_optional_column(columns, Y_SHIFT_HINTS)
    validation = validate_metadata(
        frame,
        config.input.projection_folder,
        filename_column,
        angle_column,
        angle_rad_column,
        x_shift_column,
        y_shift_column,
        remove_duplicate_endpoint=config.metadata.remove_duplicate_endpoint,
        reverse_angle_order=config.metadata.reverse_angle_order,
    )
    if validation.duplicate_filenames:
        raise ValueError(f"Duplicate projection filenames in metadata: {validation.duplicate_filenames[:10]}")
    if validation.missing_files:
        raise FileNotFoundError(f"Missing projection files: {validation.missing_files[:10]}")
    log.write(f"Validated {len(validation.records)} projection records from {metadata_csv}")

    mbir_debug_active = bool(
        run_mbir and mbir_debug_is_active(config.mbir.debug_pixel_binning, config.mbir.projection_stride)
    )
    run_validation = validation
    run_preprocessing_config = config.preprocessing
    run_alignment_config = config.alignment
    if mbir_debug_active:
        run_validation = validation_for_mbir_debug(
            validation,
            config.mbir.debug_pixel_binning,
            config.mbir.projection_stride,
        )
        run_preprocessing_config = preprocessing_for_mbir_debug(
            config.preprocessing,
            config.mbir.debug_pixel_binning,
        )
        log.write(
            "MBIR debug sampling active: "
            f"using {len(run_validation.records)} / {len(validation.records)} projections, "
            f"extra pixel binning {tuple(config.mbir.debug_pixel_binning)}."
        )
        if run_preprocessing_config.binning != config.preprocessing.binning:
            log.write(
                "MBIR effective preprocessing binning/crop: "
                f"binning {tuple(config.preprocessing.binning)} -> {tuple(run_preprocessing_config.binning)}, "
                f"crop {tuple(config.preprocessing.crop)} -> {tuple(run_preprocessing_config.crop)}"
            )

    preprocessing = run_preprocessing(
        run_validation,
        config.input.projection_folder,
        config.input.flat_folder,
        config.input.dark_folder,
        run_preprocessing_config,
        run_alignment_config,
        progress=log.write,
        cancel_check=cancel_check,
    )
    write_text_lines(folders.preprocessing / "preprocessed_projection_stack_info.txt", preprocessing.report_lines)
    if preprocessing.attenuation_stack.size:
        preview_index = min(preprocessing.attenuation_stack.shape[0] // 2, preprocessing.attenuation_stack.shape[0] - 1)
        save_image(folders.preprocessing / "preprocessed_projection_preview.tif", preprocessing.attenuation_stack[preview_index])
    log.extend(preprocessing.report_lines)

    tigre_input, transpose_report = prepare_tigre_projection_input(
        preprocessing.attenuation_stack,
        config.preprocessing.transpose_for_tigre,
    )
    raise_if_cancelled(cancel_check, "Pipeline cancelled after preprocessing.")
    preprocessing.raw_stack = np.empty((0, 0, 0), dtype=np.float32)
    preprocessing.transmission_stack = None
    preprocessing.attenuation_stack = np.empty((0, 0, 0), dtype=np.float32)
    gc.collect()
    detector_shape = tigre_input.shape[1:]
    run_geometry_config = config.geometry
    if mbir_debug_active:
        run_geometry_config = geometry_for_mbir_debug(
            config.geometry,
            config.mbir.debug_pixel_binning,
            detector_shape,
            transpose_for_tigre=config.preprocessing.transpose_for_tigre,
        )
        run_alignment_config = alignment_for_mbir_debug(
            config.alignment,
            config.mbir.debug_pixel_binning,
            transpose_for_tigre=config.preprocessing.transpose_for_tigre,
        )
    params = params_from_config(run_geometry_config, detector_shape=detector_shape, alignment=run_alignment_config)
    angles = angles_for_tigre(run_validation.angles_rad, run_geometry_config.invert_angle_sign_for_tigre)
    geom_lines = geometry_report(params, angles, tigre_input, angle_sign=-1 if config.geometry.invert_angle_sign_for_tigre else 1)
    detected_gpu_infos = query_all_nvidia_gpu_memory() if config.gpu.use_gpu else []
    gpu_selection = resolve_gpu_selection(
        config.gpu.use_gpu,
        config.gpu.gpu_selector,
        config.gpu.gpu_id,
        [info.gpu_id for info in detected_gpu_infos],
    )
    selected_gpu_ids = gpu_selection.gpu_ids
    selected_gpu_infos = [info for info in detected_gpu_infos if info.gpu_id in selected_gpu_ids]
    selected_gpu_count = max(1, len(selected_gpu_ids)) if config.gpu.use_gpu else 1
    selected_gpu_min_free_gb = min((info.free_gb for info in selected_gpu_infos), default=None)
    selected_gpu_min_total_gb = min((info.total_gb for info in selected_gpu_infos), default=None)
    selected_gpu_total_free_gb = sum(info.free_gb for info in selected_gpu_infos)
    selected_gpu_total_total_gb = sum(info.total_gb for info in selected_gpu_infos)
    requested_memory_mode = str(config.mbir.memory_mode or "auto").strip().lower()
    streaming_mode = requested_memory_mode in {"auto", "projection_streaming", "ordered_subsets"}
    system_memory = query_system_memory()
    effective_projection_batch_size = max(1, int(config.mbir.projection_batch_size))
    batch_adjustment_lines: list[str] = []
    if run_mbir and streaming_mode and len(angles) > 64 and effective_projection_batch_size >= len(angles):
        batch_adjustment_lines.extend(
            [
                "Projection batch size was capped so projection streaming does not silently become a full-stack TIGRE call.",
                f"Requested projection batch size: {effective_projection_batch_size}",
                "Effective projection batch size before GPU-memory tuning: 64",
            ]
        )
        effective_projection_batch_size = 64
    if run_mbir and streaming_mode and selected_gpu_min_total_gb is not None:
        one_view_estimate = estimate_mbir_memory(
            len(angles),
            detector_shape,
            tuple(params.volume_voxels),
            projection_batch_size=1,
            gpu_count=selected_gpu_count,
        )
        if one_view_estimate.gpu_operator_estimate_gb > 0.98 * selected_gpu_min_total_gb:
            raise RuntimeError(
                "The requested full-resolution MBIR volume is estimated to exceed this GPU's total memory even with "
                "one projection per TIGRE batch. Projection streaming cannot fix this because each TIGRE call still "
                "needs the full reconstruction volume on the GPU. Use a GPU with more "
                "memory, reduce the reconstruction volume grid for a debug run, or choose a smaller physical ROI."
            )
        if selected_gpu_min_free_gb is not None and one_view_estimate.gpu_operator_estimate_gb > 0.90 * selected_gpu_min_free_gb:
            batch_adjustment_lines.extend(
                [
                    "Warning: even one projection per TIGRE batch is close to the currently free GPU memory.",
                    "Close other GPU applications or reduce the reconstruction volume if TIGRE reports cudaMalloc failures.",
                ]
            )
        recommended_batch = recommended_projection_batch_size(
            len(angles),
            detector_shape,
            tuple(params.volume_voxels),
            selected_gpu_min_free_gb if selected_gpu_min_free_gb is not None else 0.0,
            gpu_count=selected_gpu_count,
        )
        if recommended_batch < effective_projection_batch_size:
            batch_adjustment_lines.extend(
                [
                    "Projection batch size was reduced automatically to avoid a TIGRE/CUDA allocation failure.",
                    f"Requested projection batch size: {effective_projection_batch_size}",
                    f"Effective projection batch size: {recommended_batch}",
                ]
            )
            effective_projection_batch_size = recommended_batch
        elif effective_projection_batch_size == 1 and recommended_batch > 1:
            batch_adjustment_lines.extend(
                [
                    "Projection batch size is 1. This is memory-safe, but it can underutilize the GPU because each "
                    "TIGRE call contains very little projection work.",
                    f"Current GPU-memory estimate suggests up to about {recommended_batch} views per batch may fit.",
                ]
            )
    memory_estimate = estimate_mbir_memory(
        len(angles),
        detector_shape,
        tuple(params.volume_voxels),
        projection_batch_size=effective_projection_batch_size if streaming_mode else None,
        gpu_count=selected_gpu_count,
    )
    requested_solver = _normalize_solver(config.mbir.solver)
    low_memory_cpu_gb = estimate_low_memory_pdhg_cpu_gb(
        memory_estimate.projection_gb,
        memory_estimate.volume_gb,
        dual_dtype=config.mbir.pdhg_dual_dtype,
    )
    subset_tv_cpu_gb = estimate_streaming_subset_tv_cpu_gb(memory_estimate.projection_gb, memory_estimate.volume_gb)
    distributed_overhead = estimate_distributed_full_data_overhead_gb(
        memory_estimate.projection_gb,
        memory_estimate.volume_gb,
        selected_gpu_count,
    )
    effective_solver = requested_solver
    if effective_solver == "auto":
        effective_solver = "admm"
        if system_memory is not None and memory_estimate.cpu_working_set_gb > SYSTEM_RAM_SAFETY_FRACTION * system_memory.total_gb:
            effective_solver = "pdhg_low_memory"
        elif system_memory is not None and system_memory.available_gb < minimum_available_system_memory_gb(memory_estimate):
            effective_solver = "pdhg_low_memory"
        if (
            system_memory is not None
            and effective_solver == "pdhg_low_memory"
            and low_memory_cpu_gb > 0.65 * system_memory.total_gb
        ):
            effective_solver = "streaming_subset_tv"
    minimum_available_ram_gb = minimum_available_system_memory_gb(memory_estimate)
    if run_mbir and system_memory is not None:
        if effective_solver == "admm" and memory_estimate.cpu_working_set_gb > SYSTEM_RAM_SAFETY_FRACTION * system_memory.total_gb:
            raise RuntimeError(
                "The requested ADMM/CG MBIR run is estimated to exceed safe physical system RAM. "
                f"Estimated MBIR CPU working arrays: {memory_estimate.cpu_working_set_gb:.2f} GB; "
                f"system RAM total: {system_memory.total_gb:.2f} GB. "
                "Use solver=auto or solver=pdhg_low_memory to avoid the ADMM/CG volume pile-up, or reduce the "
                "reconstruction volume."
            )
        if effective_solver == "admm" and system_memory.available_gb < minimum_available_ram_gb:
            raise RuntimeError(
                "Available system RAM is too low to call TIGRE Atb safely. "
                f"Available RAM: {system_memory.available_gb:.2f} GB; "
                f"minimum transient headroom estimate: {minimum_available_ram_gb:.2f} GB. "
                "Close other applications or reduce the reconstruction volume before running MBIR. "
                "Low physical RAM can make TIGRE/CUDA report cudaMalloc out-of-memory even when Task Manager does "
                "not show VRAM as full."
            )
        if effective_solver == "pdhg_low_memory" and low_memory_cpu_gb > SYSTEM_RAM_SAFETY_FRACTION * system_memory.total_gb:
            raise RuntimeError(
                "Even the low-memory PDHG solver is estimated to exceed safe physical system RAM. "
                f"Estimated low-memory PDHG CPU working arrays: {low_memory_cpu_gb:.2f} GB; "
                f"system RAM total: {system_memory.total_gb:.2f} GB. "
                "Use fewer voxels, crop the ROI, or choose float16 PDHG TV dual arrays."
            )
        if effective_solver == "streaming_subset_tv" and subset_tv_cpu_gb > SYSTEM_RAM_SAFETY_FRACTION * system_memory.total_gb:
            raise RuntimeError(
                "Even streaming subset-TV is estimated to exceed safe physical system RAM. "
                f"Estimated streaming subset-TV CPU working arrays: {subset_tv_cpu_gb:.2f} GB; "
                f"system RAM total: {system_memory.total_gb:.2f} GB. "
                "Use fewer voxels, crop the ROI, or close memory-heavy applications."
            )
    fdk_init_estimate = estimate_mbir_memory(
        len(angles),
        detector_shape,
        tuple(params.volume_voxels),
        gpu_count=selected_gpu_count,
    )
    operator_memory_mode = config.mbir.memory_mode
    if effective_solver in {"pdhg_low_memory", "streaming_subset_tv"} and config.mbir.memory_mode == "ordered_subsets":
        operator_memory_mode = "projection_streaming"
    distributed_mode_notes: list[str] = []
    if requested_memory_mode == "distributed_full_data":
        distributed_mode_notes.extend(
            [
                f"Estimated distributed full-data shared CPU buffers: {distributed_overhead.shared_buffers_gb:.2f} GB",
                f"Estimated distributed full-data transient CPU peak: {distributed_overhead.transient_peak_gb:.2f} GB",
            ]
        )
        if not config.gpu.use_gpu or len(selected_gpu_ids) <= 1:
            operator_memory_mode = "full_gpu"
            distributed_mode_notes.append(
                "Distributed full-data mode requested, but fewer than two GPUs are selected. Falling back to full_gpu."
            )
        elif effective_solver == "streaming_subset_tv":
            operator_memory_mode = "projection_streaming"
            distributed_mode_notes.append(
                "streaming_subset_tv uses projection_streaming internally; distributed_full_data is reserved for exact full-data Ax/A^T calls."
            )
        elif system_memory is not None:
            solver_cpu_gb = (
                memory_estimate.cpu_working_set_gb
                if effective_solver == "admm"
                else low_memory_cpu_gb
                if effective_solver == "pdhg_low_memory"
                else subset_tv_cpu_gb
            )
            if solver_cpu_gb + distributed_overhead.shared_buffers_gb > SYSTEM_RAM_SAFETY_FRACTION * system_memory.total_gb:
                operator_memory_mode = "full_gpu"
                distributed_mode_notes.append(
                    "Distributed full-data mode was downgraded to full_gpu because the extra shared "
                    "CPU buffers would exceed the safe physical RAM budget."
                )
            elif system_memory.available_gb < minimum_available_ram_gb + distributed_overhead.shared_buffers_gb:
                operator_memory_mode = "full_gpu"
                distributed_mode_notes.append(
                    "Distributed full-data mode was downgraded to full_gpu because currently available "
                    "system RAM is too low for the additional shared CPU buffers."
                )
    memory_lines = memory_estimate.report_lines()
    if mbir_debug_active:
        memory_lines.extend(
            [
                "",
                "MBIR debug sampling",
                "-------------------",
                f"Extra MBIR pixel binning y/x: {tuple(config.mbir.debug_pixel_binning)[0]} x {tuple(config.mbir.debug_pixel_binning)[1]}",
                f"MBIR projection stride: every {int(config.mbir.projection_stride)} projection(s)",
                f"Projection views used for MBIR: {len(run_validation.records)} / {len(validation.records)}",
                f"Effective preprocessing binning y/x: {tuple(run_preprocessing_config.binning)[0]} x {tuple(run_preprocessing_config.binning)[1]}",
                f"Effective preprocessing crop top/bottom/left/right: {tuple(run_preprocessing_config.crop)}",
                f"MBIR detector rows/cols: {detector_shape[0]} x {detector_shape[1]}",
                f"MBIR volume voxels z/y/x: {params.volume_voxels[0]} x {params.volume_voxels[1]} x {params.volume_voxels[2]}",
                "FDK initialization for this MBIR run uses the same sampled projection stack and geometry.",
                "Existing FDK initialization volumes are resized to this MBIR volume grid when needed.",
            ]
        )
    memory_lines.extend(
        [
            "",
            "MBIR memory strategy",
            "--------------------",
            f"Memory mode: {config.mbir.memory_mode}",
            f"Effective operator memory mode: {operator_memory_mode}",
            f"Requested projection batch size: {config.mbir.projection_batch_size}",
            f"Effective projection batch size: {effective_projection_batch_size if streaming_mode else 'not used'}",
            f"Ordered subset count: {config.mbir.ordered_subset_count}",
            f"Requested MBIR solver: {config.mbir.solver}",
            f"Effective MBIR solver: {effective_solver}",
            f"Resolved GPU selection: {gpu_selection.describe()} (selector='{gpu_selection.selector}')",
            f"Estimated low-memory PDHG CPU working arrays: {low_memory_cpu_gb:.2f} GB",
            f"Estimated streaming subset-TV CPU working arrays: {subset_tv_cpu_gb:.2f} GB",
            f"Estimated per-GPU TIGRE working set: {memory_estimate.gpu_operator_estimate_gb:.2f} GB",
            f"Estimated monolithic FDK initialization per-GPU working set: {fdk_init_estimate.gpu_operator_estimate_gb:.2f} GB",
            "Backprojection type: TIGRE matched adjoint for MBIR",
            "Default projection streaming evaluates the full data term in batches and preserves full-data MBIR quality.",
        ]
    )
    if requested_solver == "auto" and effective_solver == "pdhg_low_memory":
        memory_lines.extend(
            [
                "Auto solver selected low-memory PDHG because ADMM/CG would exceed the available RAM budget.",
                "PDHG uses projection batching plus first-order primal-dual TV updates instead of CG inner solves.",
            ]
        )
    if requested_solver == "auto" and effective_solver == "streaming_subset_tv":
        memory_lines.extend(
            [
                "Auto solver selected streaming subset-TV because ADMM/CG and PDHG would use too much RAM.",
                "Streaming subset-TV updates after each projection batch and discards the batch gradient immediately.",
            ]
        )
    if effective_solver in {"pdhg_low_memory", "streaming_subset_tv"} and config.mbir.memory_mode == "ordered_subsets":
        memory_lines.append(f"{effective_solver} uses projection_streaming internally for stable batch scheduling.")
    memory_lines.extend(distributed_mode_notes)
    memory_lines.extend(batch_adjustment_lines)
    memory_lines.extend(gpu_selection.warnings)
    if selected_gpu_infos:
        memory_lines.extend(
            [
                f"Selected GPU count: {len(selected_gpu_infos)}",
                "Selected GPUs: " + ", ".join(f"{info.gpu_id}: {info.name}" for info in selected_gpu_infos),
                f"Aggregate selected GPU memory free/total: {selected_gpu_total_free_gb:.2f} / {selected_gpu_total_total_gb:.2f} GB",
                f"Minimum per-GPU free/total across selection: {selected_gpu_min_free_gb:.2f} / {selected_gpu_min_total_gb:.2f} GB",
            ]
        )
    elif config.gpu.use_gpu:
        memory_lines.append("Selected GPU memory could not be queried with nvidia-smi.")
    if system_memory is not None:
        memory_lines.extend(
            [
                f"System RAM available/total: {system_memory.available_gb:.2f} / {system_memory.total_gb:.2f} GB",
                f"Minimum available RAM before TIGRE Atb: {minimum_available_ram_gb:.2f} GB",
            ]
        )
    init_mode = config.initialization.mode.lower()
    implicit_fdk_init = bool(run_mbir and not run_fdk and init_mode == "fdk")
    fdk_init_risky = False
    if implicit_fdk_init and streaming_mode:
        if selected_gpu_min_free_gb is not None:
            fdk_init_risky = fdk_init_estimate.gpu_operator_estimate_gb > 0.85 * selected_gpu_min_free_gb
        else:
            fdk_init_risky = fdk_init_estimate.gpu_operator_estimate_gb >= 8.0
    if fdk_init_risky:
        memory_lines.extend(
            [
                "",
                "FDK initialization fallback",
                "---------------------------",
                "Implicit FDK initialization will be skipped for this MBIR streaming run.",
                "Zero initialization will be used unless an existing FDK .npy is selected.",
            ]
        )
    gpu_lines = gpu_diagnostics(selected_gpu_ids or None, config.gpu.use_gpu)
    gpu_lines.extend(gpu_selection.warnings)
    gpu_lines.append(
        "TV/CG operations stay on NumPy arrays in the TIGRE path to avoid repeated CPU/GPU transfers around TIGRE calls."
    )
    write_text_lines(folders.root / "diagnostics.txt", transpose_report + geom_lines + [""] + memory_lines + [""] + gpu_lines)
    log.extend(transpose_report + geom_lines + [""] + memory_lines + [""] + gpu_lines)

    fdk_volume = None
    mbir_result = None
    report_geometry_lines = transpose_report + geom_lines
    if preprocess_only:
        report = _report_lines(
            config,
            preprocessing.report_lines,
            report_geometry_lines,
            memory_lines,
            "Preprocess-only run completed.",
        )
        output.write_report(folders, report)
        return PipelineResult(folders, tuple(tigre_input.shape), None, None, report)

    if fdk_init_risky:
        log.write(
            "Skipping implicit FDK initialization for MBIR because TIGRE FDK uses a monolithic full-stack "
            "allocation that is likely to exceed available GPU memory."
        )
        log.write(
            "MBIR will use zero initialization for this run. This preserves the MBIR objective and projection "
            "streaming path, but may require more ADMM/CG iterations than FDK initialization."
        )

    if run_fdk or (run_mbir and init_mode == "fdk" and not fdk_init_risky):
        raise_if_cancelled(cancel_check, "FDK reconstruction cancelled before start.")
        if run_mbir and config.mbir.memory_mode in {"auto", "projection_streaming", "ordered_subsets"}:
            log.write(
                "Note: MBIR projection operations use memory-aware batching, but FDK initialization still uses TIGRE FDK. "
                "For very large scans, existing_fdk or zeros initialization may be more memory robust."
            )
        fdk_volume = run_fdk_reconstruction(
            tigre_input,
            angles,
            params,
            gpu_ids=selected_gpu_ids or None,
            use_gpu=config.gpu.use_gpu,
            progress=log.write,
            cancel_check=cancel_check,
        )
        output.save_volume_outputs(folders.fdk, "fdk_initial_volume", fdk_volume)
        save_volume_preview(folders.fdk / "fdk_preview.png", fdk_volume, "FDK initialization")
        log.write("Saved FDK initialization volume.")

    if run_mbir:
        raise_if_cancelled(cancel_check, "Cancelled before MBIR reconstruction.")
        if init_mode == "fdk":
            if fdk_volume is None:
                x0 = np.zeros(params.volume_voxels, dtype=np.float32)
                log.write("Using zero initialization for MBIR because FDK initialization was skipped.")
            else:
                x0 = fdk_volume
        else:
            x0 = fdk_initialization(
                tigre_input,
                angles,
                params,
                mode=config.initialization.mode,
                existing_path=config.initialization.fdk_volume_path,
                constant_value=config.initialization.constant_value,
                gpu_ids=selected_gpu_ids or None,
                use_gpu=config.gpu.use_gpu,
                progress=log.write,
                cancel_check=cancel_check,
            )
        operator = TigreConeBeamOperator(
            params,
            angles,
            gpu_ids=selected_gpu_ids or None,
            use_gpu=config.gpu.use_gpu,
            memory_mode=operator_memory_mode,
            projection_batch_size=effective_projection_batch_size,
            ordered_subset_count=config.mbir.ordered_subset_count,
            logger=log.write,
            cancel_check=cancel_check,
        )
        try:
            log.write(
                "MBIR operator configured: "
                f"mode={operator.memory_mode}, batch_size={operator.projection_batch_size}, "
                f"ordered_subsets={operator.ordered_subset_count}"
            )

            def solver_progress(progress_event, volume=None) -> None:
                if mbir_progress_callback is None:
                    return
                if isinstance(progress_event, dict) and progress_event.get("kind") == "cg_progress":
                    payload = dict(progress_event)
                    payload.setdefault("solver", effective_solver)
                    mbir_progress_callback(payload)
                    return
                metrics = progress_event
                preview_image = None
                preview_view = None
                preview_slice_index = None
                preview_slice_counts = None
                preview_period = max(1, int(config.mbir.save_every))
                if metrics.iteration == 1 or metrics.iteration % preview_period == 0:
                    preview_request = mbir_preview_request_callback() if callable(mbir_preview_request_callback) else None
                    preview_image, preview_view, preview_slice_index, preview_slice_counts = extract_preview_slice_for_gui(
                        volume,
                        preview_request,
                    )
                mbir_progress_callback(
                    {
                        "metrics": metrics,
                        "preview_image": preview_image,
                        "preview_view": preview_view,
                        "preview_slice_index": preview_slice_index,
                        "preview_slice_counts": preview_slice_counts,
                        "solver": effective_solver,
                        "max_iterations": int(config.mbir.max_admm_iterations),
                    }
                )

            if effective_solver == "pdhg_low_memory":
                solver_class = PDHGTVMBIRReconstructor
            elif effective_solver == "streaming_subset_tv":
                solver_class = StreamingSubsetTVReconstructor
            else:
                solver_class = ADMMTVMBIRReconstructor
            solver = solver_class(
                operator,
                config.mbir,
                logger=log.write,
                progress_callback=solver_progress if mbir_progress_callback is not None else None,
                cancel_check=cancel_check,
            )
            mbir_result = solver.reconstruct(tigre_input, x0=x0)
            log.write(
                "TIGRE operator timing: "
                f"forward {operator.timing.forward_calls} calls / {operator.timing.forward_seconds:.2f} s, "
                f"backproject {operator.timing.backproject_calls} calls / {operator.timing.backproject_seconds:.2f} s"
            )
            output.save_volume_outputs(folders.mbir, "mbir_final_volume", mbir_result.volume)
            save_volume_preview(folders.mbir / "mbir_preview.png", mbir_result.volume, "TomoAnchor MBIR final")
            if fdk_volume is not None:
                save_volume_preview(
                    folders.mbir / "mbir_minus_fdk_preview.png",
                    mbir_result.volume - fdk_volume,
                    "MBIR minus FDK",
                )
            write_metrics_csv(folders.metrics / "metrics.csv", mbir_result.metrics)
            save_metrics_plots(folders.metrics, mbir_result.metrics)
            log.write(f"MBIR finished: {mbir_result.message}")
        finally:
            operator.close()

    status = "Run completed."
    report = _report_lines(config, preprocessing.report_lines, report_geometry_lines, memory_lines, status)
    output.write_report(folders, report)
    return PipelineResult(folders, tuple(tigre_input.shape), fdk_volume, mbir_result, report)


def _report_lines(
    config: AppConfig,
    preprocessing_lines: list[str],
    geometry_lines: list[str],
    memory_lines: list[str],
    status: str,
) -> list[str]:
    return [
        "# TomoAnchor Reconstruction Report",
        "",
        status,
        "",
        "## Input",
        "",
        f"- Projection folder: `{config.input.projection_folder}`",
        f"- Flat folder: `{config.input.flat_folder}`",
        f"- Dark folder: `{config.input.dark_folder}`",
        f"- Metadata path: `{config.input.metadata_path}`",
        "",
        "## Preprocessing",
        "",
        *[f"- {line}" if line else "" for line in preprocessing_lines],
        "",
        "## Geometry",
        "",
        *[f"- {line}" if line else "" for line in geometry_lines],
        "",
        "## Memory and Execution",
        "",
        *[f"- {line}" if line else "" for line in memory_lines],
        "",
        "## Reuse From FDK",
        "",
        "- CSV metadata validation, duplicate endpoint handling, and angle sign convention.",
        "- TIFF stack loading, binning, flat-field correction, and line-integral conversion.",
        "- Detector-offset center correction and per-projection drift correction conventions.",
        "- TIGRE geometry mapping with Zeiss effective object-plane pixel handling.",
    ]


def _normalize_solver(value: str) -> str:
    solver = str(value or "auto").strip().lower()
    if solver in {"auto", "automatic"}:
        return "auto"
    if solver in {"admm", "admm_cg", "cg"}:
        return "admm"
    if solver in {"pdhg", "pdhg_low_memory", "low_memory", "chambolle_pock", "cp"}:
        return "pdhg_low_memory"
    if solver in {"subset_tv", "streaming_subset_tv", "os_tv", "os_tv_low_memory", "sart_tv"}:
        return "streaming_subset_tv"
    raise ValueError(f"Unknown MBIR solver: {value}")
