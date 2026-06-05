from __future__ import annotations

import copy
import csv
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd

from .anchored_mbir_lite import AnchoredStreamingSubsetTVReconstructor
from .anchor_views import anchor_split_report, save_anchor_split, split_anchor_views
from .cancel_utils import raise_if_cancelled
from .config import AlignmentConfig, AppConfig, PreprocessingConfig
from .denoising_prior import make_training_free_prior
from .fast_qc import make_fast_recon_qc_report
from .fast_types import (
    AnchorSplit,
    AnchorValidationData,
    FDKSweepResult,
    FastPipelineResult,
    MBIRLiteResult,
    PriorResult,
    ProcessedProjectionSet,
)
from .fdk_sweep import run_fdk_filter_sweep
from .geometry_tigre import GeometryParams, angles_for_tigre, params_from_config
from .gpu_utils import resolve_gpu_selection
from .io_utils import save_stack_tiff
from .logging_utils import MemoryLog, write_text_lines
from .metadata_zeiss import (
    ANGLE_HINTS,
    ANGLE_RAD_HINTS,
    FILENAME_HINTS,
    X_SHIFT_HINTS,
    Y_SHIFT_HINTS,
    MetadataValidation,
    find_metadata_csv,
    load_metadata_csv,
    suggest_column,
    suggest_optional_column,
    validate_metadata,
)
from .output_manager import OutputManager, RunFolders
from .plotting import save_volume_preview
from .preprocessing import prepare_tigre_projection_input, run_preprocessing
from .projection_weights import (
    combine_duplicate_angle_projections,
    estimate_projection_scalar_weights,
    normalize_projection_weights,
)
from .tigre_ops import TigreConeBeamOperator


DRIFT_X_ALIASES = ("x_shift_px", "shift_x_px", "x_shift", "shift_x", "u_shift_px", "col_shift_px")
DRIFT_Y_ALIASES = ("y_shift_px", "shift_y_px", "y_shift", "shift_y", "v_shift_px", "row_shift_px")
_FAST_RECON_CHILD_FOLDERS = {"anchors", "fdk_sweep", "intermediate", "mbir_lite", "prior", "qc"}


def execute_fast_pipeline(
    config: AppConfig,
    run_fdk_sweep: bool = False,
    make_prior: bool = False,
    run_mbir_lite: bool = False,
    run_qc: bool = False,
    run_all: bool = False,
    fast_run_folder: str | Path | None = None,
    logger: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    mbir_lite_progress_callback: Callable[[object], None] | None = None,
) -> FastPipelineResult:
    if run_all:
        run_fdk_sweep = make_prior = run_mbir_lite = True
        run_qc = bool(config.fast_recon.make_qc_report)
    requested_stage_only = (make_prior or run_mbir_lite or run_qc) and not run_fdk_sweep and not run_all
    resume_folder = Path(fast_run_folder or config.fast_recon.resume_run_folder) if (fast_run_folder or config.fast_recon.resume_run_folder) else None
    if requested_stage_only and resume_folder is None:
        raise ValueError("Stage-only fast commands require --fast-run-folder or fast_recon.resume_run_folder.")

    folders = _resolve_run_folders(config, resume_folder)
    output = OutputManager(config.input.output_folder or "output")
    output.save_config(folders, config)
    log = MemoryLog(
        callback=logger,
        verbose=logger is None and config.logging.verbose,
        persist_path=folders.root / "log.txt",
    )
    if resume_folder is None:
        log.write(f"Created fast run folder: {folders.root}")
    else:
        log.write(f"Resuming fast run folder: {folders.root}")
    raise_if_cancelled(cancel_check, "Fast pipeline cancelled.")

    selected_gpu_ids = _selected_gpu_ids(config)
    main_set: ProcessedProjectionSet | None = None
    anchor_set: ProcessedProjectionSet | None = None
    anchor_split: AnchorSplit | None = None
    fdk_result: FDKSweepResult | None = None
    prior_result: PriorResult | None = None
    mbir_result: MBIRLiteResult | None = None
    qc_report_path: Path | None = None
    fdk_rerun_for_manual_selection = False

    make_prior_can_score_anchors = bool(
        make_prior
        and config.anchors.enabled
        and config.anchors.input.projection_folder
        and config.anchors.input.metadata_path
    )
    needs_projection_data = run_fdk_sweep or run_all or make_prior_can_score_anchors or run_mbir_lite or run_qc
    if needs_projection_data:
        main_set, anchor_set = _load_or_preprocess_projection_sets(config, folders, log.write, cancel_check)
        geometry_params = params_from_config(config.geometry, detector_shape=main_set.attenuation.shape[1:], alignment=config.alignment)
        if anchor_set is not None:
            _assert_detector_match(main_set, anchor_set)
            anchor_split = split_anchor_views(anchor_set.angles_rad, config.anchors.split)
            save_anchor_split(folders.fast_anchors / "anchor_split.csv", anchor_split, anchor_set.angles_rad)
            write_text_lines(folders.fast_anchors / "anchor_split_report.txt", anchor_split_report(anchor_split))
        else:
            anchor_split = None
    else:
        geometry_params = _geometry_from_existing_outputs(config, folders)

    if run_fdk_sweep:
        if main_set is None:
            raise RuntimeError("FDK sweep requires main projections.")
        fdk_result = run_fdk_filter_sweep(
            main_set,
            anchor_set,
            anchor_split,
            geometry_params,
            config,
            folders.fast_fdk_sweep,
            selected_gpu_ids or None,
            config.gpu.use_gpu,
            log.write,
            cancel_check,
        )
    else:
        fdk_result = _load_fdk_sweep_result_if_available(folders)
        manual_filter = _manual_fdk_filter(config)
        if manual_filter is not None and (fdk_result is None or _normalize_fdk_filter_name(fdk_result.best_filter) != manual_filter):
            if main_set is None:
                raise RuntimeError(
                    "The selected FDK filter does not match the saved fast FDK output. "
                    "Rerun FDK Sweep with projection data available before making the prior or running MBIR-lite."
                )
            log.write(f"Saved fast FDK does not match selected filter {manual_filter}; rerunning FDK sweep for the selected filter.")
            fdk_result = run_fdk_filter_sweep(
                main_set,
                anchor_set,
                anchor_split,
                geometry_params,
                config,
                folders.fast_fdk_sweep,
                selected_gpu_ids or None,
                config.gpu.use_gpu,
                log.write,
                cancel_check,
            )
            fdk_rerun_for_manual_selection = True

    if make_prior or (fdk_rerun_for_manual_selection and (run_mbir_lite or run_qc)):
        fdk_volume = fdk_result.best_volume if fdk_result is not None else _load_required_volume(folders.fast_fdk_sweep / "fdk_best.npy")
        tune_data = _anchor_validation_data(anchor_set, anchor_split.tune_indices if anchor_split is not None else None, "tune") if anchor_set is not None and anchor_split is not None else None
        if fdk_rerun_for_manual_selection and not make_prior:
            log.write("Rebuilding prior because the selected manual FDK filter changed the saved FDK output.")
        prior_result = make_training_free_prior(
            fdk_volume,
            tune_data,
            geometry_params,
            config,
            folders.fast_prior,
            selected_gpu_ids or None,
            config.gpu.use_gpu,
            log.write,
            cancel_check,
        )
    else:
        prior_result = _load_prior_result_if_available(folders)

    if run_mbir_lite:
        if main_set is None:
            raise RuntimeError("MBIR-lite requires projection data; provide config paths or saved intermediates.")
        if prior_result is None:
            prior_result = _load_prior_result_required(folders)
        fdk_volume = fdk_result.best_volume if fdk_result is not None else _load_optional_volume(folders.fast_fdk_sweep / "fdk_best.npy")
        mbir_initial_volume, mbir_initial_label = _load_mbir_lite_initial_volume(folders, config, fdk_volume, prior_result)
        projections, angles, weights = _build_mbir_lite_stack(main_set, anchor_set, anchor_split, config)
        qc_data = _anchor_validation_data(anchor_set, anchor_split.qc_indices, "qc") if anchor_set is not None and anchor_split is not None else None
        operator = TigreConeBeamOperator(
            geometry_params,
            angles_for_tigre(angles, config.geometry.invert_angle_sign_for_tigre),
            gpu_ids=selected_gpu_ids or None,
            use_gpu=config.gpu.use_gpu,
            memory_mode="ordered_subsets" if int(config.mbir_lite.ordered_subset_count) > 1 else "projection_streaming",
            projection_batch_size=max(1, int(config.mbir_lite.projection_batch_size)),
            ordered_subset_count=max(1, int(config.mbir_lite.ordered_subset_count)),
            logger=log.write,
            cancel_check=cancel_check,
        )
        qc_operator = None
        if qc_data is not None and qc_data.projections.size:
            qc_operator = TigreConeBeamOperator(
                geometry_params,
                angles_for_tigre(qc_data.angles_rad, config.geometry.invert_angle_sign_for_tigre),
                gpu_ids=selected_gpu_ids or None,
                use_gpu=config.gpu.use_gpu,
                memory_mode="projection_streaming",
                projection_batch_size=max(1, int(config.mbir_lite.projection_batch_size)),
                logger=log.write,
                cancel_check=cancel_check,
            )
        try:
            solver = AnchoredStreamingSubsetTVReconstructor(
                operator,
                config.mbir_lite,
                prior_result.x_prior,
                prior_result.confidence,
                projection_weights=weights if config.mbir_lite.use_projection_weights else None,
                qc_operator=qc_operator,
                qc_projections=None if qc_data is None else qc_data.projections,
                qc_weights=None if qc_data is None else qc_data.weights,
                logger=log.write,
                progress_callback=mbir_lite_progress_callback,
                cancel_check=cancel_check,
            )
            log.write(f"MBIR-lite start mode: {mbir_initial_label}.")
            mbir_result = solver.reconstruct(projections, x0=mbir_initial_volume)
        finally:
            operator.close()
            if qc_operator is not None:
                qc_operator.close()
        _save_mbir_lite_outputs(folders, mbir_result, fdk_volume, prior_result.x_prior)
    else:
        mbir_result = _load_mbir_lite_result_if_available(folders)

    if run_qc:
        if prior_result is None:
            prior_result = _load_prior_result_if_available(folders)
        if mbir_result is None:
            mbir_result = _load_mbir_lite_result_if_available(folders)
        qc_report_path = make_fast_recon_qc_report(
            main_set,
            anchor_set,
            anchor_split,
            geometry_params,
            config,
            folders.fast_qc,
            fdk_result,
            prior_result,
            mbir_result,
            selected_gpu_ids or None,
            config.gpu.use_gpu,
            log.write,
            cancel_check,
        )
    return FastPipelineResult(folders, main_set, anchor_set, anchor_split, fdk_result, prior_result, mbir_result, qc_report_path)


def load_drift_shift_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    x_col = _find_alias_column(frame.columns, DRIFT_X_ALIASES, "X drift shift")
    y_col = _find_alias_column(frame.columns, DRIFT_Y_ALIASES, "Y drift shift")
    x = pd.to_numeric(frame[x_col], errors="coerce")
    y = pd.to_numeric(frame[y_col], errors="coerce")
    if x.isna().any() or y.isna().any():
        raise ValueError(f"Drift shift file contains empty or non-numeric values: {path}")
    return x.astype(np.float32).to_numpy(), y.astype(np.float32).to_numpy()


def _resolve_run_folders(config: AppConfig, resume_folder: Path | None) -> RunFolders:
    if resume_folder is None:
        return OutputManager(config.input.output_folder or "output").create_run()
    root = _normalize_fast_run_root(Path(resume_folder))
    if not root.exists():
        raise FileNotFoundError(f"Fast run folder does not exist: {root}")
    return RunFolders(
        root=root,
        preprocessing=root / "preprocessing",
        alignment=root / "alignment",
        fdk=root / "fdk",
        mbir=root / "mbir",
        snapshots=root / "mbir" / "iteration_snapshots",
        metrics=root / "metrics",
        fast_recon=root / "fast_recon",
        fast_anchors=root / "fast_recon" / "anchors",
        fast_fdk_sweep=root / "fast_recon" / "fdk_sweep",
        fast_prior=root / "fast_recon" / "prior",
        fast_mbir_lite=root / "fast_recon" / "mbir_lite",
        fast_qc=root / "fast_recon" / "qc",
    )


def _normalize_fast_run_root(path: Path) -> Path:
    root = Path(path)
    if root.parent.name == "fast_recon" and root.name in _FAST_RECON_CHILD_FOLDERS:
        root = root.parent
    while root.name == "fast_recon":
        root = root.parent
    return root


def _fast_folder_with_legacy_fallback(folders: RunFolders, child: str) -> Path:
    folder = folders.fast_recon / child
    legacy = folders.fast_recon / "fast_recon" / child
    if not folder.exists() and legacy.exists():
        return legacy
    return folder


def _load_or_preprocess_projection_sets(
    config: AppConfig,
    folders: RunFolders,
    logger: Callable[[str], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> tuple[ProcessedProjectionSet, ProcessedProjectionSet | None]:
    intermediate = folders.fast_recon / "intermediate"
    main_cache = intermediate / "main_set.npz"
    anchor_cache = intermediate / "anchor_set.npz"
    if config.fast_recon.save_intermediate and main_cache.exists():
        main_set = _load_processed_set_cache("main", main_cache)
        anchor_set = _load_processed_set_cache("anchor", anchor_cache) if anchor_cache.exists() else None
        _log(logger, "Loaded saved fast preprocessing intermediates.")
        return main_set, anchor_set
    main_set, main_raw, main_transmission = _preprocess_named_set(
        "main",
        config,
        config.input.projection_folder,
        config.input.flat_folder,
        config.input.dark_folder,
        config.input.metadata_path,
        config.metadata,
        config.preprocessing,
        config.alignment,
        logger,
        cancel_check,
    )
    anchor_set = None
    anchor_raw = None
    anchor_transmission = None
    if config.anchors.enabled:
        anchor_prep = _effective_anchor_preprocessing(config)
        anchor_alignment = _effective_anchor_alignment(config)
        anchor_set, anchor_raw, anchor_transmission = _preprocess_named_set(
            "anchor",
            config,
            config.anchors.input.projection_folder,
            config.anchors.input.flat_folder,
            config.anchors.input.dark_folder,
            config.anchors.input.metadata_path,
            config.anchors.metadata,
            anchor_prep,
            anchor_alignment,
            logger,
            cancel_check,
            drift_shift_file=anchor_alignment.drift_shift_file,
        )
    main_weights = estimate_projection_scalar_weights(
        main_raw,
        main_transmission,
        config.anchors.exposure.main_exposure_s,
        fallback_multiplier=1.0,
    )
    anchor_weights = None
    if anchor_set is not None:
        anchor_weights = estimate_projection_scalar_weights(
            anchor_raw,
            anchor_transmission,
            config.anchors.exposure.anchor_exposure_s,
            fallback_multiplier=float(config.anchors.exposure.anchor_weight_multiplier),
        )
    main_weights, anchor_weights = normalize_projection_weights(
        main_weights,
        anchor_weights,
        max_weight_ratio=float(config.anchors.exposure.max_weight_ratio),
    )
    main_set.weights = main_weights
    if anchor_set is not None:
        anchor_set.weights = anchor_weights
    if config.fast_recon.save_intermediate:
        intermediate.mkdir(parents=True, exist_ok=True)
        _save_processed_set_cache(main_cache, main_set)
        if anchor_set is not None:
            _save_processed_set_cache(anchor_cache, anchor_set)
    return main_set, anchor_set


def _preprocess_named_set(
    name: str,
    config: AppConfig,
    projection_folder: str,
    flat_folder: str,
    dark_folder: str,
    metadata_path: str,
    metadata_config,
    preprocessing: PreprocessingConfig,
    alignment: AlignmentConfig,
    logger: Callable[[str], None] | None,
    cancel_check: Callable[[], bool] | None,
    drift_shift_file: str | None = None,
) -> tuple[ProcessedProjectionSet, np.ndarray, np.ndarray | None]:
    if not projection_folder:
        raise ValueError(f"{name} projection folder is required.")
    if not metadata_path:
        raise ValueError(f"{name} metadata path is required.")
    metadata_csv = find_metadata_csv(metadata_path)
    frame = load_metadata_csv(metadata_csv)
    columns = [str(column) for column in frame.columns]
    validation = validate_metadata(
        frame,
        projection_folder,
        metadata_config.filename_column or suggest_column(columns, FILENAME_HINTS),
        metadata_config.angle_column or suggest_column(columns, ANGLE_HINTS),
        metadata_config.angle_rad_column or suggest_optional_column(columns, ANGLE_RAD_HINTS),
        metadata_config.x_shift_column or suggest_optional_column(columns, X_SHIFT_HINTS),
        metadata_config.y_shift_column or suggest_optional_column(columns, Y_SHIFT_HINTS),
        remove_duplicate_endpoint=metadata_config.remove_duplicate_endpoint,
        reverse_angle_order=metadata_config.reverse_angle_order,
    )
    if validation.duplicate_filenames:
        raise ValueError(f"Duplicate {name} projection filenames in metadata: {validation.duplicate_filenames[:10]}")
    if validation.missing_files:
        raise FileNotFoundError(f"Missing {name} projection files: {validation.missing_files[:10]}")
    if drift_shift_file:
        _apply_drift_file(
            validation,
            drift_shift_file,
            logger,
            reverse_angle_order=bool(metadata_config.reverse_angle_order),
        )
    _log(logger, f"Validated {len(validation.records)} {name} projection records from {metadata_csv}.")
    preprocessing_result = run_preprocessing(
        validation,
        projection_folder,
        flat_folder,
        dark_folder,
        preprocessing,
        alignment,
        progress=logger,
        cancel_check=cancel_check,
    )
    tigre_input, transpose_report = prepare_tigre_projection_input(
        preprocessing_result.attenuation_stack,
        preprocessing.transpose_for_tigre,
    )
    report = [f"{name.title()} dataset", *preprocessing_result.report_lines, *transpose_report]
    processed = ProcessedProjectionSet(
        name=name,
        attenuation=np.ascontiguousarray(tigre_input, dtype=np.float32),
        angles_rad=np.ascontiguousarray(validation.angles_rad, dtype=np.float32),
        validation=validation,
        preprocessing_report=report,
        weights=None,
        detector_mask=None,
    )
    return processed, preprocessing_result.raw_stack, preprocessing_result.transmission_stack


def _effective_anchor_preprocessing(config: AppConfig) -> PreprocessingConfig:
    if config.anchors.preprocessing.inherit_main:
        return copy.deepcopy(config.preprocessing)
    values = {
        field.name: getattr(config.anchors.preprocessing, field.name)
        for field in fields(PreprocessingConfig)
    }
    return PreprocessingConfig(**values)


def _effective_anchor_alignment(config: AppConfig) -> AlignmentConfig:
    if config.anchors.alignment.inherit_main:
        align = copy.deepcopy(config.alignment)
        if config.anchors.alignment.drift_shift_file:
            align.drift_shift_file = config.anchors.alignment.drift_shift_file
        return align
    values = {
        field.name: getattr(config.anchors.alignment, field.name)
        for field in fields(AlignmentConfig)
    }
    return AlignmentConfig(**values)


def _apply_drift_file(
    validation: MetadataValidation,
    drift_shift_file: str | Path,
    logger: Callable[[str], None] | None,
    reverse_angle_order: bool = False,
) -> None:
    x_shift, y_shift = load_drift_shift_file(drift_shift_file)
    if validation.duplicate_endpoint_removed and len(x_shift) == len(validation.records) + 1:
        x_shift = x_shift[:-1]
        y_shift = y_shift[:-1]
        _log(
            logger,
            "Drift shift file has one extra row matching a removed duplicate endpoint; "
            "dropping the last drift-shift row.",
        )
    if len(x_shift) != len(validation.records) or len(y_shift) != len(validation.records):
        raise ValueError(
            f"Drift shift file has {len(x_shift)} rows, but metadata validation has {len(validation.records)} projections."
        )
    if reverse_angle_order:
        x_shift = x_shift[::-1]
        y_shift = y_shift[::-1]
        _log(logger, "Reversed drift-shift file rows to match reversed metadata order.")
    if validation.x_shift_px is not None or validation.y_shift_px is not None:
        _log(logger, "Anchor metadata shift columns and drift_shift_file are both present; using drift_shift_file.")
    validation.x_shift_px = np.ascontiguousarray(x_shift, dtype=np.float32)
    validation.y_shift_px = np.ascontiguousarray(y_shift, dtype=np.float32)
    validation.x_shift_column_found = True
    validation.y_shift_column_found = True
    validation.x_shift_column = "drift_shift_file"
    validation.y_shift_column = "drift_shift_file"
    validation.warnings.append(f"Used drift shifts from {drift_shift_file}.")


def _assert_detector_match(main_set: ProcessedProjectionSet, anchor_set: ProcessedProjectionSet) -> None:
    if main_set.attenuation.shape[1:] != anchor_set.attenuation.shape[1:]:
        raise ValueError(
            "Main and anchor detector shapes differ after preprocessing: "
            f"{main_set.attenuation.shape[1:]} vs {anchor_set.attenuation.shape[1:]}. "
            "Anchor resampling is not implemented in v1."
        )


def _build_mbir_lite_stack(
    main_set: ProcessedProjectionSet,
    anchor_set: ProcessedProjectionSet | None,
    anchor_split: AnchorSplit | None,
    config: AppConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    projections = [np.asarray(main_set.attenuation, dtype=np.float32)]
    angles = [np.asarray(main_set.angles_rad, dtype=np.float32)]
    weights: list[np.ndarray] = []
    if main_set.weights is not None:
        weights.append(np.asarray(main_set.weights, dtype=np.float32))
    if anchor_set is not None and anchor_split is not None:
        parts = [np.asarray(anchor_split.recon_indices, dtype=np.int64)]
        if config.anchors.split.use_tune_anchors_in_final:
            parts.append(np.asarray(anchor_split.tune_indices, dtype=np.int64))
        if config.anchors.split.use_qc_anchors_in_final:
            parts.append(np.asarray(anchor_split.qc_indices, dtype=np.int64))
        anchor_indices = np.unique(np.concatenate(parts)) if parts else np.asarray([], dtype=np.int64)
        if anchor_indices.size:
            projections.append(np.asarray(anchor_set.attenuation[anchor_indices], dtype=np.float32))
            angles.append(np.asarray(anchor_set.angles_rad[anchor_indices], dtype=np.float32))
            if anchor_set.weights is not None:
                weights.append(np.asarray(anchor_set.weights[anchor_indices], dtype=np.float32))
    stack = np.ascontiguousarray(np.concatenate(projections, axis=0), dtype=np.float32)
    angle_stack = np.ascontiguousarray(np.concatenate(angles, axis=0), dtype=np.float32)
    weight_stack = np.ascontiguousarray(np.concatenate(weights), dtype=np.float32) if weights and len(weights) == len(projections) else None
    if config.anchors.merge.combine_duplicate_angles:
        stack, angle_stack, weight_stack, _ = combine_duplicate_angle_projections(
            stack,
            angle_stack,
            weight_stack,
            np.deg2rad(float(config.anchors.merge.duplicate_angle_tolerance_deg)),
        )
    if config.anchors.merge.sort_by_angle:
        order = np.argsort(angle_stack.astype(np.float64), kind="mergesort")
        stack = np.ascontiguousarray(stack[order], dtype=np.float32)
        angle_stack = np.ascontiguousarray(angle_stack[order], dtype=np.float32)
        if weight_stack is not None:
            weight_stack = np.ascontiguousarray(weight_stack[order], dtype=np.float32)
    return stack, angle_stack, weight_stack


def _anchor_validation_data(
    anchor_set: ProcessedProjectionSet | None,
    indices: np.ndarray | None,
    label: str,
) -> AnchorValidationData | None:
    if anchor_set is None or indices is None:
        return None
    idx = np.asarray(indices, dtype=np.int64)
    if not idx.size:
        return None
    return AnchorValidationData(
        projections=np.ascontiguousarray(anchor_set.attenuation[idx], dtype=np.float32),
        angles_rad=np.ascontiguousarray(anchor_set.angles_rad[idx], dtype=np.float32),
        weights=None if anchor_set.weights is None else np.ascontiguousarray(anchor_set.weights[idx], dtype=np.float32),
        detector_mask=anchor_set.detector_mask,
        label=label,
    )


def _save_mbir_lite_outputs(
    folders: RunFolders,
    result: MBIRLiteResult,
    fdk_volume: np.ndarray | None,
    prior_volume: np.ndarray,
) -> None:
    folders.fast_mbir_lite.mkdir(parents=True, exist_ok=True)
    np.save(folders.fast_mbir_lite / "mbir_lite_final.npy", result.volume.astype(np.float32))
    save_stack_tiff(folders.fast_mbir_lite / "mbir_lite_final.tif", result.volume)
    save_volume_preview(folders.fast_mbir_lite / "mbir_lite_preview.png", result.volume, "MBIR-lite final")
    if fdk_volume is not None:
        diff = result.volume - fdk_volume
        np.save(folders.fast_mbir_lite / "final_minus_fdk.npy", diff.astype(np.float32))
    diff_prior = result.volume - prior_volume
    np.save(folders.fast_mbir_lite / "final_minus_prior.npy", diff_prior.astype(np.float32))
    _write_dataclass_csv(folders.fast_mbir_lite / "metrics.csv", result.metrics)


def _load_fdk_sweep_result_if_available(folders: RunFolders) -> FDKSweepResult | None:
    folder = _fast_folder_with_legacy_fallback(folders, "fdk_sweep")
    path = folder / "fdk_best.npy"
    if not path.exists():
        return None
    best_filter = "unknown"
    best_cutoff = 1.0
    best_yaml = folder / "best_filter.yaml"
    if best_yaml.exists():
        for line in best_yaml.read_text(encoding="utf-8").splitlines():
            if line.startswith("best_filter:"):
                best_filter = line.split(":", 1)[1].strip()
            elif line.startswith("best_cutoff:"):
                best_cutoff = float(line.split(":", 1)[1].strip())
    return FDKSweepResult(
        best_volume=np.asarray(np.load(path), dtype=np.float32),
        best_filter=best_filter,
        best_cutoff=best_cutoff,
        scores=[],
        recon_projections=np.empty((0, 0, 0), dtype=np.float32),
        recon_angles_rad=np.empty((0,), dtype=np.float32),
        recon_weights=None,
        tune_anchor_projections=None,
        tune_anchor_angles_rad=None,
        tune_anchor_weights=None,
    )


def _load_prior_result_if_available(folders: RunFolders) -> PriorResult | None:
    folder = _fast_folder_with_legacy_fallback(folders, "prior")
    prior = folder / "prior_selected.npy"
    confidence = folder / "prior_confidence.npy"
    if not prior.exists() or not confidence.exists():
        return None
    support_path = folder / "support_mask.npy"
    x_prior = np.asarray(np.load(prior), dtype=np.float32)
    support = np.asarray(np.load(support_path), dtype=bool) if support_path.exists() else np.ones_like(x_prior, dtype=bool)
    return PriorResult(
        x_prior=x_prior,
        confidence=np.asarray(np.load(confidence), dtype=np.float32),
        support_mask=support,
        metrics=[],
        chosen_method="loaded",
        chosen_strength=float("nan"),
        normalization={},
    )


def _load_prior_result_required(folders: RunFolders) -> PriorResult:
    result = _load_prior_result_if_available(folders)
    if result is None:
        legacy = folders.fast_recon / "fast_recon" / "prior"
        raise FileNotFoundError(f"Prior outputs were not found in {folders.fast_prior} or {legacy}")
    return result


def _load_mbir_lite_result_if_available(folders: RunFolders) -> MBIRLiteResult | None:
    path = _fast_folder_with_legacy_fallback(folders, "mbir_lite") / "mbir_lite_final.npy"
    if not path.exists():
        return None
    return MBIRLiteResult(np.asarray(np.load(path), dtype=np.float32), [], False, "loaded")


def _load_mbir_lite_initial_volume(
    folders: RunFolders,
    config: AppConfig,
    fdk_volume: np.ndarray | None,
    prior_result: PriorResult,
) -> tuple[np.ndarray, str]:
    start_mode = _normalize_mbir_lite_start_mode(getattr(config.mbir_lite, "start_mode", "fresh_from_fdk"))
    if start_mode == "resume_previous_final":
        path = _fast_folder_with_legacy_fallback(folders, "mbir_lite") / "mbir_lite_final.npy"
        if not path.exists():
            raise FileNotFoundError(
                "MBIR-lite start mode 'resume_previous_final' requires an existing "
                f"{path}. Choose 'fresh_from_fdk' or point Resume run folder to a run that already has MBIR-lite output."
            )
        volume = np.asarray(np.load(path), dtype=np.float32)
        if volume.shape != prior_result.x_prior.shape:
            raise ValueError(
                "Existing MBIR-lite final volume shape "
                f"{volume.shape} does not match the current prior shape {prior_result.x_prior.shape}."
            )
        return volume, "resume previous MBIR-lite final"
    if fdk_volume is not None:
        return fdk_volume, "fresh from FDK"
    return np.asarray(prior_result.x_prior, dtype=np.float32), "fresh from prior (FDK unavailable)"


def _load_required_volume(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required volume is missing: {path}")
    return np.asarray(np.load(path), dtype=np.float32)


def _load_optional_volume(path: Path) -> np.ndarray | None:
    return np.asarray(np.load(path), dtype=np.float32) if path.exists() else None


def _normalize_fdk_filter_name(name: str) -> str:
    clean = str(name or "ram_lak").strip().lower()
    if clean == "shep_logan":
        return "shepp_logan"
    return clean


def _manual_fdk_filter(config: AppConfig) -> str | None:
    value = _normalize_fdk_filter_name(getattr(config.fdk_sweep, "selection_filter", "auto"))
    if value in {"", "auto", "automatic", "score", "score_total"}:
        return None
    return value


def _normalize_mbir_lite_start_mode(value: object) -> str:
    clean = str(value or "fresh_from_fdk").strip().lower()
    if clean in {"resume_previous_final", "resume", "warm_start", "mbir_lite_final"}:
        return "resume_previous_final"
    return "fresh_from_fdk"


def _geometry_from_existing_outputs(config: AppConfig, folders: RunFolders) -> GeometryParams:
    prior_path = _fast_folder_with_legacy_fallback(folders, "prior") / "prior_selected.npy"
    fdk_path = _fast_folder_with_legacy_fallback(folders, "fdk_sweep") / "fdk_best.npy"
    source_path = prior_path if prior_path.exists() else fdk_path
    if not source_path.exists():
        raise FileNotFoundError(f"Cannot infer fast geometry because no prior or FDK volume exists in {folders.fast_recon}.")
    volume = np.load(source_path)
    try:
        return params_from_config(config.geometry, detector_shape=(int(volume.shape[1]), int(volume.shape[2])), alignment=config.alignment)
    except Exception:
        shape = tuple(int(value) for value in volume.shape)
        return GeometryParams(
            source_detector_distance_mm=2.0,
            source_origin_distance_mm=1.0,
            detector_pixel_size_mm=(1.0, 1.0),
            detector_pixels=(shape[1], shape[2]),
            voxel_size_mm=(1.0, 1.0, 1.0),
            volume_voxels=shape,
        )


def _save_processed_set_cache(path: Path, data: ProcessedProjectionSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "attenuation": data.attenuation.astype(np.float32),
        "angles_rad": data.angles_rad.astype(np.float32),
    }
    if data.weights is not None:
        arrays["weights"] = data.weights.astype(np.float32)
    np.savez_compressed(path, **arrays)


def _load_processed_set_cache(name: str, path: Path) -> ProcessedProjectionSet:
    data = np.load(path)
    weights = np.asarray(data["weights"], dtype=np.float32) if "weights" in data.files else None
    validation = SimpleNamespace(records=[], angles_rad=np.asarray(data["angles_rad"], dtype=np.float32))
    return ProcessedProjectionSet(
        name=name,
        attenuation=np.asarray(data["attenuation"], dtype=np.float32),
        angles_rad=np.asarray(data["angles_rad"], dtype=np.float32),
        validation=validation,  # type: ignore[arg-type]
        preprocessing_report=[],
        weights=weights,
        detector_mask=None,
    )


def _selected_gpu_ids(config: AppConfig) -> tuple[int, ...]:
    if not config.gpu.use_gpu:
        return ()
    try:
        from .memory_utils import query_all_nvidia_gpu_memory

        detected = query_all_nvidia_gpu_memory()
        selection = resolve_gpu_selection(
            config.gpu.use_gpu,
            config.gpu.gpu_selector,
            config.gpu.gpu_id,
            [info.gpu_id for info in detected],
        )
        return tuple(selection.gpu_ids)
    except Exception:
        return (int(config.gpu.gpu_id),)


def _find_alias_column(columns, aliases: tuple[str, ...], label: str) -> str:
    normalized = [(str(column), str(column).lower().strip()) for column in columns]
    for alias in aliases:
        for original, lower in normalized:
            if lower == alias:
                return original
    for alias in aliases:
        for original, lower in normalized:
            if alias in lower:
                return original
    raise ValueError(f"{label} column not found. Accepted aliases: {', '.join(aliases)}")


def _write_dataclass_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plain_rows = [asdict(row) for row in rows]
    if not plain_rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plain_rows[0].keys()))
        writer.writeheader()
        writer.writerows(plain_rows)


def _log(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)
