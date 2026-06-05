from __future__ import annotations

import csv
import math
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from ..alignment_center import (
    CenterShiftPreviewResult,
    compute_metrics_for_results,
    generate_shift_values,
    recommend_metric_index,
    reconstruct_center_shift_previews,
)
from ..geometry_tigre import angles_for_tigre, params_from_config
from ..gpu_utils import resolve_gpu_selection
from ..io_utils import crop_2d, estimate_stack_memory_gb, read_tiff_float32, save_stack_folder
from ..metadata_zeiss import (
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
from ..memory_utils import (
    SYSTEM_RAM_SAFETY_FRACTION,
    TIGRE_BACKPROJECTION_KERNEL_VIEWS,
    estimate_low_memory_pdhg_cpu_gb,
    estimate_mbir_memory,
    minimum_available_system_memory_gb,
    query_all_nvidia_gpu_memory,
    query_system_memory,
    recommended_projection_batch_size,
    estimate_streaming_subset_tv_cpu_gb,
)
from ..config import AVAILABLE_FDK_FILTERS, AppConfig, load_config, save_config
from ..cancel_utils import OperationCancelled
from ..distributed_tigre import estimate_distributed_full_data_overhead_gb
from ..mbir_debug import (
    effective_detector_shape,
    effective_projection_count,
    effective_volume_shape,
    mbir_debug_report_lines,
)
from ..device_monitor import format_gpu_snapshot_summary, nvidia_gpu_monitor_available, query_nvidia_gpu_snapshot
from ..logging_utils import append_text_line, timestamp
from ..fast_pipeline import execute_fast_pipeline
from ..output_manager import RunFolders
from ..pipeline import execute_pipeline, extract_preview_slice_for_gui, preview_slice_counts_for_shape
from ..preprocessing import PreprocessingResult, prepare_tigre_projection_input, run_preprocessing
from ..worker import CancellationToken
from .parameter_panels import PathPicker
from .help_manual import HelpManualWidget
from .plot_widgets import DeviceMonitorWidget, MetricsPlotWidget
from .preview_widgets import HistogramLevelWidget, RequestedSlicePreviewWidget, TomogramViewWidget, VolumePreviewWidget
from .style import APP_ICON_PATH, APP_STYLE_SHEET
from .qt_compat import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QIcon,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QThread,
    QTimer,
    QVBoxLayout,
    QWidget,
    QEvent,
    QObject,
    Qt,
    Signal,
    Slot,
    exec_app,
)


@dataclass
class TomogramSource:
    label: str
    path: Path | None = None
    volume: np.ndarray | None = None


@dataclass(frozen=True)
class MemoryEstimateRuntime:
    system_info: Any | None
    gpu_selection: Any
    selected_gpu_infos: list[Any]
    selected_gpu_count: int
    selected_gpu_min_free_gb: float | None
    selected_gpu_min_total_gb: float | None
    selected_gpu_total_free_gb: float
    selected_gpu_total_total_gb: float


_TOMOGRAM_SOURCE_DISPLAY_ORDER = (
    "fdk",
    "existing_fdk",
    "mbir",
    "fast_fdk",
    "fast_prior",
    "mbir_lite",
)


def _tomogram_source_is_available(source: TomogramSource) -> bool:
    return source.volume is not None or (source.path is not None and source.path.exists())


def _append_tomogram_source(
    sources: dict[str, TomogramSource],
    source_id: str,
    label: str,
    *,
    path: Path | None = None,
    volume: np.ndarray | None = None,
) -> None:
    source = TomogramSource(str(label), path=Path(path) if path is not None else None, volume=volume)
    if _tomogram_source_is_available(source):
        sources[str(source_id)] = source


def _merge_tomogram_source_maps(*source_maps: dict[str, TomogramSource]) -> dict[str, TomogramSource]:
    merged: dict[str, TomogramSource] = {}
    for source_id in _TOMOGRAM_SOURCE_DISPLAY_ORDER:
        for source_map in source_maps:
            source = source_map.get(source_id)
            if source is not None and _tomogram_source_is_available(source):
                merged[source_id] = source
    for source_map in source_maps:
        for source_id, source in source_map.items():
            if source_id not in merged and _tomogram_source_is_available(source):
                merged[str(source_id)] = source
    return merged


class CheckableComboBox(QComboBox):
    def __init__(self, items: tuple[str, ...] | list[str]) -> None:
        super().__init__()
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText("Select filters")
        self.view().viewport().installEventFilter(self)
        self.add_checkable_items(items)

    def add_checkable_items(self, items: tuple[str, ...] | list[str]) -> None:
        self.clear()
        for text in items:
            self.addItem(str(text))
            item = self.model().item(self.count() - 1)
            if hasattr(item, "setCheckable"):
                item.setCheckable(True)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setCheckState(Qt.Checked)
        self._update_summary()

    def checked_items(self) -> list[str]:
        values: list[str] = []
        for index in range(self.count()):
            item = self.model().item(index)
            if item is not None and self._is_checked(item):
                values.append(str(self.itemText(index)))
        return values

    def set_checked_items(self, values: list[str] | tuple[str, ...]) -> None:
        requested = {str(value).strip().lower() for value in values}
        for index in range(self.count()):
            item = self.model().item(index)
            if item is not None:
                checked = str(self.itemText(index)).strip().lower() in requested
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self._update_summary()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.view().viewport() and event.type() == QEvent.MouseButtonRelease:
            index = self.view().indexAt(event.pos())
            if index.isValid():
                self._toggle_index(index.row())
                return True
        if watched is self.view().viewport() and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
                index = self.view().currentIndex()
                if index.isValid():
                    self._toggle_index(index.row())
                    return True
        return super().eventFilter(watched, event)

    def _toggle_index(self, row: int) -> None:
        item = self.model().item(int(row))
        if item is None:
            return
        item.setCheckState(Qt.Unchecked if self._is_checked(item) else Qt.Checked)
        self._update_summary()

    def _is_checked(self, item) -> bool:
        state = item.checkState()
        if state == Qt.Checked:
            return True
        value = getattr(state, "value", state)
        checked_value = getattr(Qt.Checked, "value", Qt.Checked)
        try:
            return int(value) == int(checked_value)
        except Exception:
            return False

    def _update_summary(self) -> None:
        checked = self.checked_items()
        if not checked:
            text = "No filters selected"
        elif len(checked) == self.count():
            text = "All filters"
        else:
            text = ", ".join(checked)
        self.lineEdit().setText(text)


class Worker(QObject):
    finished = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)
    log = Signal(str)
    progress = Signal(object)

    def __init__(self, fn: Callable[..., Any], expects_progress: bool = False) -> None:
        super().__init__()
        self.fn = fn
        self.expects_progress = expects_progress

    @Slot()
    def run(self) -> None:
        try:
            if self.expects_progress:
                result = self.fn(self.log.emit, self.progress.emit)
            else:
                result = self.fn(self.log.emit)
        except OperationCancelled as exc:
            self.cancelled.emit(str(exc) or "Operation cancelled.")
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.finished.emit(result)


class TVMBIRMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TomoAnchor")
        if APP_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.resize(1480, 900)
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self._cancel = CancellationToken()
        self._last_result = None
        self._active_run_root: Path | None = None
        self.metadata_frame = None
        self.validation: MetadataValidation | None = None
        self.expected_raw_shape: tuple[int, int] | None = None
        self.raw_stack = None
        self.flat_field = None
        self.dark_field = None
        self.transmission_stack = None
        self.attenuation_stack = None
        self.center_shift_results: list[CenterShiftPreviewResult] = []
        self.current_center_shift_preview_index = 0
        self.auto_metric_table: list[dict[str, float]] | None = None
        self.auto_best_shift_px: float | None = None
        self.auto_best_preview_index: int | None = None
        self._live_mbir_metrics: list[Any] = []
        self._mbir_max_iterations = 0
        self._mbir_preview_started = False
        self._device_monitor_selected_gpu_ids: tuple[int, ...] = ()
        self._device_monitor_runtime_logging_enabled = False
        self._device_monitor_runtime_log_interval_s = 15.0
        self._last_device_runtime_log_s = 0.0
        self._config_template = AppConfig()
        self._tomogram_sources: dict[str, TomogramSource] = {}
        self._mbir_preview_request_lock = threading.Lock()
        self._mbir_preview_request = {"view": "Axial", "slice_index": 0}
        self._mbir_preview_volume_source: tuple[Path, str] | None = None
        self._build_ui()
        self._initialize_device_monitor()
        self._log("TomoAnchor GUI started.")

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        left = QWidget()
        self.form_layout = QVBoxLayout(left)
        self.form_layout.setSpacing(10)
        scroll.setWidget(left)
        splitter.addWidget(scroll)

        self._build_input_group()
        self._build_metadata_group()
        self._build_preprocessing_group()
        self._build_leveling_group()
        self._build_geometry_group()
        self._build_alignment_group()
        self._build_initialization_group()
        self._build_mbir_group()
        self._build_fast_recon_group()
        self._build_run_group()
        self.form_layout.addStretch(1)

        self.tabs = QTabWidget()
        self.help_manual = HelpManualWidget()
        self.tabs.addTab(self.help_manual, "Help")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.progress_bar = QProgressBar()
        log_page = QWidget()
        log_layout = QVBoxLayout(log_page)
        self._build_mbir_progress_panel(log_layout)
        log_layout.addWidget(self.progress_bar)
        log_layout.addWidget(self.log_text)
        self.tabs.addTab(log_page, "Progress / Log")
        self.device_monitor = DeviceMonitorWidget()
        self.tabs.addTab(self.device_monitor, "Device Monitor")
        validation_page = QWidget()
        validation_layout = QVBoxLayout(validation_page)
        self.validation_table = QTableWidget(0, 7)
        self.validation_table.setHorizontalHeaderLabels(
            ["Index", "Filename", "Angle deg", "Angle rad", "X shift px", "Y shift px", "Exists"]
        )
        self.validation_table.horizontalHeader().setStretchLastSection(True)
        validation_layout.addWidget(self.validation_table)
        self.tabs.addTab(validation_page, "Validation")
        self.metrics_plot = MetricsPlotWidget(self._active_image_changed)
        self.tabs.addTab(self.metrics_plot, "Metrics")
        self.preview = VolumePreviewWidget()
        self.tabs.addTab(self.preview, "Preview")
        self.tomogram_view = TomogramViewWidget(self._active_image_changed, self._tomogram_source_selected)
        self.tabs.addTab(self.tomogram_view, "Tomogram View")
        self.mbir_preview = RequestedSlicePreviewWidget(self._active_image_changed, self._mbir_preview_request_changed)
        self.mbir_preview.show_message("MBIR preview will appear during reconstruction")
        self.tabs.addTab(self.mbir_preview, "MBIR Preview")
        self.tabs.currentChanged.connect(lambda *_: self._active_image_changed())
        splitter.addWidget(self.tabs)
        splitter.setSizes([520, 900])

    def _initialize_device_monitor(self) -> None:
        self._device_monitor_timer = QTimer(self)
        self._device_monitor_timer.setInterval(2000)
        self._device_monitor_timer.timeout.connect(self._poll_device_monitor)
        self._refresh_device_monitor_selection()
        if nvidia_gpu_monitor_available():
            self.device_monitor.set_status_message("Waiting for the first NVIDIA GPU sample.")
            self._device_monitor_timer.start()
            self._poll_device_monitor()
        else:
            self.device_monitor.show_message("NVIDIA device monitor unavailable: `nvidia-smi` was not found.")

    def _refresh_device_monitor_selection(self, cfg: AppConfig | None = None) -> None:
        try:
            current_cfg = cfg or self.config_from_ui()
        except Exception:
            return
        selected_gpu_ids: tuple[int, ...] = ()
        if current_cfg.gpu.use_gpu:
            detected_gpu_infos = query_all_nvidia_gpu_memory()
            try:
                selection = resolve_gpu_selection(
                    current_cfg.gpu.use_gpu,
                    current_cfg.gpu.gpu_selector,
                    current_cfg.gpu.gpu_id,
                    [info.gpu_id for info in detected_gpu_infos],
                )
                selected_gpu_ids = tuple(int(gpu_id) for gpu_id in selection.gpu_ids)
            except Exception:
                selected_gpu_ids = ()
        self._device_monitor_selected_gpu_ids = selected_gpu_ids
        self.device_monitor.set_selected_gpu_ids(selected_gpu_ids)

    def _poll_device_monitor(self) -> None:
        snapshot = query_nvidia_gpu_snapshot()
        if snapshot is None:
            if not self.device_monitor.has_history():
                if nvidia_gpu_monitor_available():
                    self.device_monitor.show_message("Live NVIDIA telemetry is temporarily unavailable.")
                else:
                    self.device_monitor.show_message("NVIDIA device monitor unavailable: `nvidia-smi` was not found.")
            else:
                self.device_monitor.set_status_message(
                    "Waiting for the next NVIDIA GPU sample. Showing the last successful history."
                )
            return
        self.device_monitor.append_snapshot(snapshot)
        if self._device_monitor_runtime_logging_enabled and snapshot.gpus:
            sample_time = float(snapshot.timestamp_s)
            if self._last_device_runtime_log_s <= 0 or (
                sample_time - self._last_device_runtime_log_s
            ) >= self._device_monitor_runtime_log_interval_s:
                summary = format_gpu_snapshot_summary(snapshot, gpu_ids=self._device_monitor_selected_gpu_ids or None)
                if summary:
                    self._log(f"[{timestamp()}] Device monitor: {summary}")
                self._last_device_runtime_log_s = sample_time

    def _set_device_monitor_runtime_logging(self, enabled: bool, cfg: AppConfig | None = None) -> None:
        self._device_monitor_runtime_logging_enabled = bool(enabled)
        self._last_device_runtime_log_s = 0.0
        if cfg is not None:
            self._refresh_device_monitor_selection(cfg)

    def _build_mbir_progress_panel(self, parent: QVBoxLayout) -> None:
        group = QGroupBox("Live MBIR progress")
        layout = QFormLayout(group)
        self.mbir_iteration_label = QLabel("Not running")
        self.mbir_cg_label = QLabel("Not running")
        self.mbir_time_label = QLabel("Elapsed 00:00:00 | ETA unavailable")
        self.mbir_objective_label = QLabel("Objective unavailable")
        self.mbir_data_residual_label = QLabel("Data residual unavailable")
        self.mbir_relative_change_label = QLabel("Relative change unavailable")
        for label in (
            self.mbir_iteration_label,
            self.mbir_cg_label,
            self.mbir_time_label,
            self.mbir_objective_label,
            self.mbir_data_residual_label,
            self.mbir_relative_change_label,
        ):
            label.setWordWrap(True)
        layout.addRow("Iteration", self.mbir_iteration_label)
        layout.addRow("Solver", self.mbir_cg_label)
        layout.addRow("Time", self.mbir_time_label)
        layout.addRow("Objective", self.mbir_objective_label)
        layout.addRow("Data residual", self.mbir_data_residual_label)
        layout.addRow("Relative change", self.mbir_relative_change_label)
        parent.addWidget(group)

    def _build_input_group(self) -> None:
        group = QGroupBox("Dataset input")
        layout = QFormLayout(group)
        self.projection_folder = PathPicker("folder")
        self.flat_folder = PathPicker("folder")
        self.dark_folder = PathPicker("folder")
        self.metadata_path = PathPicker("file")
        self.output_folder = PathPicker("folder")
        layout.addRow("Projection folder", self.projection_folder)
        layout.addRow("Flat/reference folder", self.flat_folder)
        layout.addRow("Dark folder", self.dark_folder)
        layout.addRow("Metadata CSV", self.metadata_path)
        layout.addRow("Output folder", self.output_folder)
        self.form_layout.addWidget(group)

    def _build_metadata_group(self) -> None:
        group = QGroupBox("Metadata")
        layout = QFormLayout(group)
        self.filename_column = self._editable_combo()
        self.angle_column = self._editable_combo()
        self.angle_rad_column = self._editable_combo()
        self.x_shift_column = self._editable_combo()
        self.y_shift_column = self._editable_combo()
        self.remove_endpoint = QCheckBox("Remove duplicate 0/360 endpoint")
        self.remove_endpoint.setChecked(True)
        self.reverse_angle_order = QCheckBox("Reverse angle order")
        self.load_metadata_button = QPushButton("Load Metadata CSV")
        self.metadata_folder_button = QPushButton("Choose Metadata Folder")
        self.validate_metadata_button = QPushButton("Validate Metadata")
        self.preview_raw_button = QPushButton("Preview Raw Projection")
        self.preview_projection_index = self._spin(0, 1000000, 0)
        row1 = QHBoxLayout()
        row1.addWidget(self.load_metadata_button)
        row1.addWidget(self.metadata_folder_button)
        row2 = QHBoxLayout()
        row2.addWidget(self.validate_metadata_button)
        row2.addWidget(self.preview_raw_button)
        layout.addRow("Filename column", self.filename_column)
        layout.addRow("Angle column deg", self.angle_column)
        layout.addRow("Angle column rad", self.angle_rad_column)
        layout.addRow("X shift column", self.x_shift_column)
        layout.addRow("Y shift column", self.y_shift_column)
        layout.addRow("", self.remove_endpoint)
        layout.addRow("", self.reverse_angle_order)
        layout.addRow(row1)
        layout.addRow("Projection preview index", self.preview_projection_index)
        layout.addRow(row2)
        self.load_metadata_button.clicked.connect(self.load_metadata)
        self.metadata_folder_button.clicked.connect(self._browse_metadata_folder)
        self.validate_metadata_button.clicked.connect(self.validate_metadata)
        self.preview_raw_button.clicked.connect(self.preview_raw_projection)
        self.form_layout.addWidget(group)

    def _build_preprocessing_group(self) -> None:
        group = QGroupBox("Preprocessing")
        layout = QFormLayout(group)
        self.use_flat = QCheckBox("Use flat correction")
        self.use_flat.setChecked(True)
        self.use_dark = QCheckBox("Use dark correction if folder exists")
        self.use_dark.setChecked(True)
        self.negative_log = QCheckBox("Negative log")
        self.negative_log.setChecked(True)
        self.clip_transmission = QCheckBox("Clip transmission")
        self.clip_transmission.setChecked(True)
        self.clip_negative = QCheckBox("Set negative attenuation to zero")
        self.clip_negative.setChecked(True)
        self.flip_h = QCheckBox("Flip horizontal")
        self.flip_v = QCheckBox("Flip vertical")
        self.transpose_for_tigre = QCheckBox("Transpose attenuation for TIGRE")
        self.epsilon = self._double(1e-12, 1.0, 1e-6, 12)
        self.tmin = self._double(1e-12, 10.0, 1e-6, 12)
        self.tmax = self._double(1e-12, 100.0, 2.0, 6)
        self.binning_y = self._spin(1, 16, 1)
        self.binning_x = self._spin(1, 16, 1)
        self.crop_top = self._spin(0, 100000, 0)
        self.crop_bottom = self._spin(0, 100000, 0)
        self.crop_left = self._spin(0, 100000, 0)
        self.crop_right = self._spin(0, 100000, 0)
        self.preprocess_preview_kind = QComboBox()
        self.preprocess_preview_kind.addItems(
            [
                "Raw projection",
                "Averaged flat field",
                "Flat-field corrected projection",
                "Attenuation projection",
            ]
        )
        self.compute_preprocess_preview_button = QPushButton("Compute Preprocessed Preview Stack")
        self.preview_preprocess_button = QPushButton("Preview Selected Image")
        layout.addRow("", self.use_flat)
        layout.addRow("", self.use_dark)
        layout.addRow("", self.negative_log)
        layout.addRow("Epsilon", self.epsilon)
        layout.addRow("", self.clip_transmission)
        layout.addRow("Transmission min", self.tmin)
        layout.addRow("Transmission max", self.tmax)
        layout.addRow("", self.clip_negative)
        layout.addRow("Binning Y", self.binning_y)
        layout.addRow("Binning X", self.binning_x)
        layout.addRow("Crop top", self.crop_top)
        layout.addRow("Crop bottom", self.crop_bottom)
        layout.addRow("Crop left", self.crop_left)
        layout.addRow("Crop right", self.crop_right)
        layout.addRow("", self.flip_h)
        layout.addRow("", self.flip_v)
        layout.addRow("", self.transpose_for_tigre)
        layout.addRow("Preview image", self.preprocess_preview_kind)
        layout.addRow(self.compute_preprocess_preview_button)
        layout.addRow(self.preview_preprocess_button)
        self.compute_preprocess_preview_button.clicked.connect(self.compute_preprocessed_previews)
        self.preview_preprocess_button.clicked.connect(self.preview_selected_preprocessed_image)
        self.form_layout.addWidget(group)

    def _build_leveling_group(self) -> None:
        group = QGroupBox("Image display")
        layout = QVBoxLayout(group)
        self.level_histogram = HistogramLevelWidget(self._preview_levels_changed)
        self.level_histogram.setMinimumHeight(240)
        self.auto_level_button = QPushButton("Auto 1-99%")
        self.full_level_button = QPushButton("Full Range")
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_out_button = QPushButton("Zoom Out")
        self.fit_view_button = QPushButton("Fit")
        self.zoom_label = QLabel("Zoom: 100%")
        level_row = QHBoxLayout()
        level_row.addWidget(self.auto_level_button)
        level_row.addWidget(self.full_level_button)
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(self.zoom_in_button)
        zoom_row.addWidget(self.zoom_out_button)
        zoom_row.addWidget(self.fit_view_button)
        zoom_row.addWidget(self.zoom_label)
        layout.addWidget(self.level_histogram)
        layout.addLayout(level_row)
        layout.addLayout(zoom_row)
        self.auto_level_button.clicked.connect(self._auto_level_active_preview)
        self.full_level_button.clicked.connect(self._full_range_active_preview)
        self.zoom_in_button.clicked.connect(lambda: self._zoom_active_image(1.25))
        self.zoom_out_button.clicked.connect(lambda: self._zoom_active_image(1.0 / 1.25))
        self.fit_view_button.clicked.connect(self._fit_active_image)
        self.form_layout.addWidget(group)

    def _build_geometry_group(self) -> None:
        group = QGroupBox("Geometry")
        layout = QFormLayout(group)
        self.dsd = self._double(0.0, 1e6, 0.0, 6)
        self.dso = self._double(0.0, 1e6, 0.0, 6)
        self.effective_pixel_um = self._double(1e-6, 1e6, 11.0, 6)
        self.voxel_z = self._double(1e-9, 1e6, 0.011, 9)
        self.voxel_y = self._double(1e-9, 1e6, 0.011, 9)
        self.voxel_x = self._double(1e-9, 1e6, 0.011, 9)
        self.detector_rows = self._spin(0, 20000, 0)
        self.detector_cols = self._spin(0, 20000, 0)
        self.detector_rows.setReadOnly(True)
        self.detector_cols.setReadOnly(True)
        self.nz = self._spin(0, 20000, 0)
        self.ny = self._spin(0, 20000, 0)
        self.nx = self._spin(0, 20000, 0)
        self.det_v_offset = self._double(-1e6, 1e6, 0.0, 6)
        self.det_u_offset = self._double(-1e6, 1e6, 0.0, 6)
        self.invert_angle = QCheckBox("Invert angle sign for TIGRE")
        self.invert_angle.setChecked(True)
        layout.addRow("DSD mm", self.dsd)
        layout.addRow("DSO mm", self.dso)
        layout.addRow("Effective pixel um", self.effective_pixel_um)
        layout.addRow("Voxel z mm", self.voxel_z)
        layout.addRow("Voxel y mm", self.voxel_y)
        layout.addRow("Voxel x mm", self.voxel_x)
        layout.addRow("Detector rows", self.detector_rows)
        layout.addRow("Detector columns", self.detector_cols)
        layout.addRow("Nz", self.nz)
        layout.addRow("Ny", self.ny)
        layout.addRow("Nx", self.nx)
        layout.addRow("Detector offset v px", self.det_v_offset)
        layout.addRow("Detector offset u px", self.det_u_offset)
        layout.addRow("", self.invert_angle)
        self.form_layout.addWidget(group)

    def _build_alignment_group(self) -> None:
        group = QGroupBox("Alignment")
        layout = QFormLayout(group)
        self.use_center = QCheckBox("Use center correction")
        self.use_center.setChecked(True)
        self.center_offset = self._double(-1e6, 1e6, 0.0, 6)
        self.center_sign = self._double(-1.0, 1.0, 1.0, 0)
        self.use_drift = QCheckBox("Use drift correction")
        self.use_drift.setChecked(True)
        self.drift_stage = QComboBox()
        self.drift_stage.addItems(["transmission", "attenuation"])
        self.export_drift_transmission_button = QPushButton("Save Drift-Corrected Transmission TIFFs")
        self.x_shift_sign = self._spin(-1, 1, 1)
        self.y_shift_sign = self._spin(-1, 1, 1)
        self.center_search_start = self._double(-1e6, 1e6, -10.0, 4)
        self.center_search_end = self._double(-1e6, 1e6, 10.0, 4)
        self.center_search_step = self._double(1e-6, 1e6, 1.0, 4)
        self.center_search_metric = QComboBox()
        self.center_search_metric.addItem("Combined score", "combined_score")
        self.center_search_metric.addItem("Gradient energy", "gradient_energy")
        self.center_search_metric.addItem("Laplacian variance", "laplacian_variance")
        self.center_search_metric.addItem("Entropy", "entropy")
        self.run_center_manual_button = QPushButton("Run Manual Preview Search")
        self.run_center_auto_button = QPushButton("Run Automatic Search")
        self.run_center_fine_button = QPushButton("Run Fine Search Around Selected Shift")
        self.run_center_fine_button.setEnabled(False)
        self.center_preview_slider = QSlider(Qt.Horizontal)
        self.center_preview_slider.setRange(0, 0)
        self.center_preview_slider.setSingleStep(1)
        self.center_preview_slider.setPageStep(1)
        self.center_preview_slider.setEnabled(False)
        self.center_preview_prev_button = QPushButton("<")
        self.center_preview_prev_button.setEnabled(False)
        self.center_preview_next_button = QPushButton(">")
        self.center_preview_next_button.setEnabled(False)
        self.center_selected_label = QLabel("No center-offset previews")
        self.center_selected_label.setWordWrap(True)
        self.center_auto_label = QLabel("Automatic recommendation unavailable")
        self.center_auto_label.setWordWrap(True)
        self.apply_center_selected_button = QPushButton("Apply Selected Shift")
        self.apply_center_selected_button.setEnabled(False)
        self.apply_center_auto_button = QPushButton("Use Automatic Best Shift")
        self.apply_center_auto_button.setEnabled(False)
        center_run_row = QHBoxLayout()
        center_run_row.addWidget(self.run_center_manual_button)
        center_run_row.addWidget(self.run_center_auto_button)
        center_preview_row = QHBoxLayout()
        center_preview_row.addWidget(self.center_preview_prev_button)
        center_preview_row.addWidget(self.center_preview_slider, 1)
        center_preview_row.addWidget(self.center_preview_next_button)
        center_apply_row = QHBoxLayout()
        center_apply_row.addWidget(self.apply_center_selected_button)
        center_apply_row.addWidget(self.apply_center_auto_button)
        layout.addRow("", self.use_center)
        layout.addRow("Center offset px", self.center_offset)
        layout.addRow("Center sign", self.center_sign)
        layout.addRow("Search start px", self.center_search_start)
        layout.addRow("Search end px", self.center_search_end)
        layout.addRow("Search step px", self.center_search_step)
        layout.addRow("Automatic metric", self.center_search_metric)
        layout.addRow(center_run_row)
        layout.addRow(self.run_center_fine_button)
        layout.addRow("Preview selector", center_preview_row)
        layout.addRow(self.center_selected_label)
        layout.addRow(self.center_auto_label)
        layout.addRow(center_apply_row)
        layout.addRow("", self.use_drift)
        layout.addRow("Drift stage", self.drift_stage)
        layout.addRow(self.export_drift_transmission_button)
        layout.addRow("X shift sign", self.x_shift_sign)
        layout.addRow("Y shift sign", self.y_shift_sign)
        self.run_center_manual_button.clicked.connect(lambda: self.run_center_shift_search(automatic=False))
        self.run_center_auto_button.clicked.connect(lambda: self.run_center_shift_search(automatic=True))
        self.run_center_fine_button.clicked.connect(self.run_fine_center_shift_search)
        self.center_preview_prev_button.clicked.connect(lambda: self._step_center_shift_preview(-1))
        self.center_preview_next_button.clicked.connect(lambda: self._step_center_shift_preview(1))
        self.center_preview_slider.valueChanged.connect(self._center_shift_preview_changed)
        self.center_search_metric.currentIndexChanged.connect(self._update_auto_center_shift_recommendation)
        self.apply_center_selected_button.clicked.connect(self.apply_selected_center_shift)
        self.apply_center_auto_button.clicked.connect(self.apply_auto_center_shift)
        self.export_drift_transmission_button.clicked.connect(self.export_drift_corrected_transmission_stack)
        self.form_layout.addWidget(group)

    def _build_initialization_group(self) -> None:
        group = QGroupBox("FDK initialization")
        layout = QFormLayout(group)
        self.init_mode = QComboBox()
        self.init_mode.addItems(["fdk", "existing_fdk", "zeros", "constant"])
        self.fdk_path = PathPicker("file")
        self.init_constant = self._double(0.0, 1e6, 0.0, 6)
        layout.addRow("Mode", self.init_mode)
        layout.addRow("Existing FDK volume .npy", self.fdk_path)
        layout.addRow("Constant value", self.init_constant)
        self.form_layout.addWidget(group)

    def _build_mbir_group(self) -> None:
        group = QGroupBox("MBIR solver")
        layout = QFormLayout(group)
        self.lambda_tv = self._double(0.0, 1e6, 0.001, 9)
        self.rho = self._double(1e-12, 1e6, 0.05, 9)
        self.admm_iterations = self._spin(1, 10000, 50)
        self.cg_iterations = self._spin(1, 10000, 10)
        self.tolerance_primal = self._double(1e-12, 1.0, 1e-4, 10)
        self.tolerance_dual = self._double(1e-12, 1.0, 1e-4, 10)
        self.tolerance_relative_change = self._double(1e-12, 1.0, 1e-5, 10)
        self.cg_tolerance = self._double(1e-12, 1.0, 1e-4, 10)
        self.positivity = QCheckBox("Positivity")
        self.positivity.setChecked(True)
        self.tv_epsilon = self._double(0.0, 1.0, 1e-8, 12)
        self.mbir_solver = QComboBox()
        self.mbir_solver.addItems(["auto", "admm", "pdhg_low_memory", "streaming_subset_tv"])
        self.memory_mode = QComboBox()
        self.memory_mode.addItems(["auto", "full_gpu", "distributed_full_data", "projection_streaming", "ordered_subsets"])
        self.mbir_binning_y = self._spin(1, 16, 1)
        self.mbir_binning_x = self._spin(1, 16, 1)
        self.mbir_projection_stride = self._spin(1, 100000, 1)
        self.projection_batch_size = self._spin(1, 100000, 64)
        self.ordered_subset_count = self._spin(1, 1024, 1)
        self.pdhg_dual_dtype = QComboBox()
        self.pdhg_dual_dtype.addItems(["float16", "float32"])
        self.subset_tv_step_safety = self._double(1e-6, 10.0, 0.7, 6)
        self.use_gpu = QCheckBox("Use TIGRE GPU acceleration")
        self.use_gpu.setChecked(True)
        self.gpu_selector = QLineEdit("auto")
        self.gpu_id = self._spin(0, 32, 0)
        self.use_gpu.toggled.connect(lambda *_: self._refresh_device_monitor_selection())
        self.gpu_selector.editingFinished.connect(self._refresh_device_monitor_selection)
        self.gpu_id.valueChanged.connect(lambda *_: self._refresh_device_monitor_selection())
        layout.addRow("lambda TV", self.lambda_tv)
        layout.addRow("rho", self.rho)
        layout.addRow("ADMM iterations", self.admm_iterations)
        layout.addRow("Inner CG iterations", self.cg_iterations)
        layout.addRow("Primal tolerance", self.tolerance_primal)
        layout.addRow("Dual tolerance", self.tolerance_dual)
        layout.addRow("Relative x-change tol.", self.tolerance_relative_change)
        layout.addRow("CG tolerance", self.cg_tolerance)
        layout.addRow("", self.positivity)
        layout.addRow("TV epsilon", self.tv_epsilon)
        layout.addRow("Solver", self.mbir_solver)
        layout.addRow("MBIR extra binning Y", self.mbir_binning_y)
        layout.addRow("MBIR extra binning X", self.mbir_binning_x)
        layout.addRow("Use every Nth projection", self.mbir_projection_stride)
        layout.addRow("Memory mode", self.memory_mode)
        layout.addRow("Projection batch size", self.projection_batch_size)
        layout.addRow("Ordered subsets", self.ordered_subset_count)
        layout.addRow("PDHG TV dual dtype", self.pdhg_dual_dtype)
        layout.addRow("Subset-TV step safety", self.subset_tv_step_safety)
        layout.addRow("", self.use_gpu)
        layout.addRow("GPU selection", self.gpu_selector)
        layout.addRow("Primary GPU ID", self.gpu_id)
        self.form_layout.addWidget(group)

    def _build_fast_recon_group(self) -> None:
        group = QGroupBox("Fast anchor-guided reconstruction")
        layout = QFormLayout(group)
        self.fast_enabled = QCheckBox("Enable fast anchor workflow")
        self.fast_anchor_enabled = QCheckBox("Use high-exposure anchor dataset")
        self.fast_anchor_enabled.setChecked(False)
        self.fast_anchor_projection_folder = PathPicker("folder")
        self.fast_anchor_flat_folder = PathPicker("folder")
        self.fast_anchor_dark_folder = PathPicker("folder")
        self.fast_anchor_metadata_path = PathPicker("file")
        self.fast_anchor_drift_file = PathPicker("file")
        self.fast_anchor_inherit_preprocessing = QCheckBox("Inherit main preprocessing")
        self.fast_anchor_inherit_preprocessing.setChecked(True)
        self.fast_anchor_inherit_alignment = QCheckBox("Inherit main alignment")
        self.fast_anchor_inherit_alignment.setChecked(True)
        self.fast_main_exposure = self._double(0.0, 1000000.0, 0.0, 4)
        self.fast_anchor_exposure = self._double(0.0, 1000000.0, 0.0, 4)
        self.fast_anchor_weight_multiplier = self._double(0.01, 1000.0, 1.0, 3)
        self.fast_anchor_max_weight_ratio = self._double(1.0, 1000.0, 10.0, 2)
        self.fast_recon_count = self._spin(0, 1000000, 0)
        self.fast_tune_count = self._spin(0, 1000000, 6)
        self.fast_qc_count = self._spin(0, 1000000, 6)
        self.fast_use_tune_final = QCheckBox("Use tune anchors in MBIR-lite")
        self.fast_use_tune_final.setChecked(True)
        self.fast_use_qc_final = QCheckBox("Use QC anchors in MBIR-lite")
        self.fast_lowres_factor = self._spin(1, 16, 2)
        self.fast_filters = CheckableComboBox(AVAILABLE_FDK_FILTERS)
        self.fast_filter_selection = QComboBox()
        self.fast_filter_selection.addItem("Auto: lowest score_total", "auto")
        for filter_name in AVAILABLE_FDK_FILTERS:
            self.fast_filter_selection.addItem(filter_name, filter_name)
        self.fast_prior_tv_weights = QLineEdit("0.005, 0.01, 0.02, 0.04")
        self.fast_mbir_sweeps = self._spin(1, 1000, 5)
        self.fast_mbir_batch_size = self._spin(1, 1000000, 32)
        self.fast_mbir_subsets = self._spin(1, 1000000, 8)
        self.fast_mbir_start_mode = QComboBox()
        self.fast_mbir_start_mode.addItem("Fresh from FDK (reproducible rerun)", "fresh_from_fdk")
        self.fast_mbir_start_mode.addItem("Resume previous MBIR-lite final", "resume_previous_final")
        self.fast_lambda_tv = self._double(0.0, 1000.0, 1.0e-4, 8)
        self.fast_rho_prior = self._double(0.0, 1000.0, 0.05, 6)
        self.fast_resume_run_folder = PathPicker("folder")
        self.fast_save_intermediate = QCheckBox("Save reusable fast intermediates")
        self.fast_save_intermediate.setChecked(True)
        self.fast_make_qc = QCheckBox("Make QC report after Run Fast")
        self.fast_make_qc.setChecked(True)
        self.fast_run_button = QPushButton("Run Fast")
        self.fast_fdk_sweep_button = QPushButton("FDK Sweep")
        self.fast_prior_button = QPushButton("Make Prior")
        self.fast_mbir_lite_button = QPushButton("Run MBIR-lite")
        self.fast_qc_button = QPushButton("QC Report")
        fast_buttons = QHBoxLayout()
        for button in (
            self.fast_run_button,
            self.fast_fdk_sweep_button,
            self.fast_prior_button,
            self.fast_mbir_lite_button,
            self.fast_qc_button,
        ):
            fast_buttons.addWidget(button)
        layout.addRow("", self.fast_enabled)
        layout.addRow("", self.fast_anchor_enabled)
        layout.addRow("Anchor projections", self.fast_anchor_projection_folder)
        layout.addRow("Anchor flat/reference", self.fast_anchor_flat_folder)
        layout.addRow("Anchor dark", self.fast_anchor_dark_folder)
        layout.addRow("Anchor metadata CSV", self.fast_anchor_metadata_path)
        layout.addRow("Anchor drift file", self.fast_anchor_drift_file)
        layout.addRow("", self.fast_anchor_inherit_preprocessing)
        layout.addRow("", self.fast_anchor_inherit_alignment)
        layout.addRow("Main exposure s", self.fast_main_exposure)
        layout.addRow("Anchor exposure s", self.fast_anchor_exposure)
        layout.addRow("Anchor weight multiplier", self.fast_anchor_weight_multiplier)
        layout.addRow("Max weight ratio", self.fast_anchor_max_weight_ratio)
        layout.addRow("Recon anchor count (0 auto)", self.fast_recon_count)
        layout.addRow("Tune anchor count", self.fast_tune_count)
        layout.addRow("QC anchor count", self.fast_qc_count)
        layout.addRow("", self.fast_use_tune_final)
        layout.addRow("", self.fast_use_qc_final)
        layout.addRow("FDK lowres factor", self.fast_lowres_factor)
        layout.addRow("FDK filters", self.fast_filters)
        layout.addRow("FDK filter choice", self.fast_filter_selection)
        layout.addRow("Prior TV weights", self.fast_prior_tv_weights)
        layout.addRow("MBIR-lite sweeps", self.fast_mbir_sweeps)
        layout.addRow("MBIR-lite batch size", self.fast_mbir_batch_size)
        layout.addRow("MBIR-lite subsets", self.fast_mbir_subsets)
        layout.addRow("MBIR-lite start volume", self.fast_mbir_start_mode)
        layout.addRow("MBIR-lite lambda TV", self.fast_lambda_tv)
        layout.addRow("MBIR-lite rho prior", self.fast_rho_prior)
        layout.addRow("Resume run folder", self.fast_resume_run_folder)
        layout.addRow("", self.fast_save_intermediate)
        layout.addRow("", self.fast_make_qc)
        layout.addRow(fast_buttons)
        self.fast_run_button.clicked.connect(lambda: self._run_fast(run_all=True))
        self.fast_fdk_sweep_button.clicked.connect(lambda: self._run_fast(run_fdk_sweep=True))
        self.fast_prior_button.clicked.connect(lambda: self._run_fast(make_prior=True))
        self.fast_mbir_lite_button.clicked.connect(lambda: self._run_fast(run_mbir_lite=True))
        self.fast_qc_button.clicked.connect(lambda: self._run_fast(run_qc=True))
        self.form_layout.addWidget(group)

    def _build_run_group(self) -> None:
        group = QGroupBox("Run controls")
        layout = QVBoxLayout(group)
        self.preprocess_button = QPushButton("Preprocess Only")
        self.fdk_button = QPushButton("Run FDK")
        self.mbir_button = QPushButton("Run MBIR")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.estimate_memory_button = QPushButton("Estimate Memory")
        self.preload_results_button = QPushButton("Preload Existing Results")
        self.save_config_button = QPushButton("Save Config")
        self.load_config_button = QPushButton("Load Config")
        row1 = QHBoxLayout()
        row1.addWidget(self.preprocess_button)
        row1.addWidget(self.fdk_button)
        row1.addWidget(self.mbir_button)
        row1.addWidget(self.cancel_button)
        row2 = QHBoxLayout()
        row2.addWidget(self.estimate_memory_button)
        row2.addWidget(self.preload_results_button)
        row2.addWidget(self.save_config_button)
        row2.addWidget(self.load_config_button)
        layout.addLayout(row1)
        layout.addLayout(row2)
        self.preprocess_button.clicked.connect(lambda: self._run(preprocess_only=True))
        self.fdk_button.clicked.connect(lambda: self._run(run_fdk=True))
        self.mbir_button.clicked.connect(lambda: self._run(run_mbir=True))
        self.cancel_button.clicked.connect(self._request_cancel)
        self.estimate_memory_button.clicked.connect(self._estimate_mbir_lite_memory)
        self.preload_results_button.clicked.connect(self._preload_existing_results)
        self.save_config_button.clicked.connect(self._save_config)
        self.load_config_button.clicked.connect(self._load_config)
        self.form_layout.addWidget(group)

    def config_from_ui(self) -> AppConfig:
        cfg = AppConfig.from_dict(self._config_template.to_dict()) if hasattr(self, "_config_template") else AppConfig()
        cfg.input.projection_folder = self.projection_folder.text()
        cfg.input.flat_folder = self.flat_folder.text()
        cfg.input.dark_folder = self.dark_folder.text()
        cfg.input.metadata_path = self.metadata_path.text()
        cfg.input.output_folder = self.output_folder.text()
        cfg.metadata.filename_column = self._combo_text(self.filename_column)
        cfg.metadata.angle_column = self._combo_text(self.angle_column)
        cfg.metadata.angle_rad_column = self._optional_combo_text(self.angle_rad_column)
        cfg.metadata.x_shift_column = self._optional_combo_text(self.x_shift_column)
        cfg.metadata.y_shift_column = self._optional_combo_text(self.y_shift_column)
        cfg.metadata.remove_duplicate_endpoint = self.remove_endpoint.isChecked()
        cfg.metadata.reverse_angle_order = self.reverse_angle_order.isChecked()
        cfg.preprocessing.use_flat = self.use_flat.isChecked()
        cfg.preprocessing.use_dark = self.use_dark.isChecked()
        cfg.preprocessing.negative_log = self.negative_log.isChecked()
        cfg.preprocessing.epsilon = self.epsilon.value()
        cfg.preprocessing.transmission_clip_min = self.tmin.value()
        cfg.preprocessing.transmission_clip_max = self.tmax.value()
        cfg.preprocessing.max_transmission_for_log = self.tmax.value()
        cfg.preprocessing.clip_transmission = self.clip_transmission.isChecked()
        cfg.preprocessing.clip_negative_attenuation_to_zero = self.clip_negative.isChecked()
        cfg.preprocessing.binning = (self.binning_y.value(), self.binning_x.value())
        cfg.preprocessing.crop = (self.crop_top.value(), self.crop_bottom.value(), self.crop_left.value(), self.crop_right.value())
        cfg.preprocessing.flip_horizontal = self.flip_h.isChecked()
        cfg.preprocessing.flip_vertical = self.flip_v.isChecked()
        cfg.preprocessing.transpose_for_tigre = self.transpose_for_tigre.isChecked()
        cfg.geometry.source_detector_distance_mm = self.dsd.value() or None
        cfg.geometry.source_origin_distance_mm = self.dso.value() or None
        cfg.geometry.effective_pixel_size_um = self.effective_pixel_um.value()
        cfg.geometry.voxel_size_mm = (self.voxel_z.value(), self.voxel_y.value(), self.voxel_x.value())
        cfg.geometry.volume_voxels = (
            self.nz.value() or None,
            self.ny.value() or None,
            self.nx.value() or None,
        )
        cfg.geometry.detector_pixels = (
            self.detector_rows.value() or None,
            self.detector_cols.value() or None,
        )
        cfg.geometry.detector_offset_pixels = (self.det_v_offset.value(), self.det_u_offset.value())
        cfg.geometry.invert_angle_sign_for_tigre = self.invert_angle.isChecked()
        cfg.alignment.use_center_correction = self.use_center.isChecked()
        cfg.alignment.center_offset_pixels = self.center_offset.value()
        cfg.alignment.center_shift_sign = self._center_shift_sign_value()
        cfg.alignment.use_drift_correction = self.use_drift.isChecked()
        cfg.alignment.drift_stage = self.drift_stage.currentText()
        cfg.alignment.x_shift_sign = 1 if self.x_shift_sign.value() >= 0 else -1
        cfg.alignment.y_shift_sign = 1 if self.y_shift_sign.value() >= 0 else -1
        cfg.initialization.mode = self.init_mode.currentText()
        cfg.initialization.fdk_volume_path = self.fdk_path.text() or None
        cfg.initialization.constant_value = self.init_constant.value()
        cfg.mbir.lambda_tv = self.lambda_tv.value()
        cfg.mbir.rho = self.rho.value()
        cfg.mbir.max_admm_iterations = self.admm_iterations.value()
        cfg.mbir.inner_cg_iterations = self.cg_iterations.value()
        cfg.mbir.tolerance_primal = self.tolerance_primal.value()
        cfg.mbir.tolerance_dual = self.tolerance_dual.value()
        cfg.mbir.tolerance_relative_change = self.tolerance_relative_change.value()
        cfg.mbir.cg_tolerance = self.cg_tolerance.value()
        cfg.mbir.positivity = self.positivity.isChecked()
        cfg.mbir.tv_epsilon = self.tv_epsilon.value()
        cfg.mbir.solver = self.mbir_solver.currentText()
        cfg.mbir.debug_pixel_binning = (self.mbir_binning_y.value(), self.mbir_binning_x.value())
        cfg.mbir.projection_stride = self.mbir_projection_stride.value()
        cfg.mbir.memory_mode = self.memory_mode.currentText()
        cfg.mbir.projection_batch_size = self.projection_batch_size.value()
        cfg.mbir.ordered_subset_count = self.ordered_subset_count.value()
        cfg.mbir.pdhg_dual_dtype = self.pdhg_dual_dtype.currentText()
        cfg.mbir.subset_tv_step_safety = self.subset_tv_step_safety.value()
        cfg.gpu.use_gpu = self.use_gpu.isChecked()
        cfg.gpu.gpu_selector = self.gpu_selector.text().strip() or "auto"
        cfg.gpu.gpu_id = self.gpu_id.value()
        cfg.anchors.enabled = self.fast_anchor_enabled.isChecked()
        cfg.anchors.input.projection_folder = self.fast_anchor_projection_folder.text()
        cfg.anchors.input.flat_folder = self.fast_anchor_flat_folder.text()
        cfg.anchors.input.dark_folder = self.fast_anchor_dark_folder.text()
        cfg.anchors.input.metadata_path = self.fast_anchor_metadata_path.text()
        cfg.anchors.preprocessing.inherit_main = self.fast_anchor_inherit_preprocessing.isChecked()
        cfg.anchors.alignment.inherit_main = self.fast_anchor_inherit_alignment.isChecked()
        cfg.anchors.alignment.drift_shift_file = self.fast_anchor_drift_file.text() or None
        cfg.anchors.exposure.main_exposure_s = self.fast_main_exposure.value() or None
        cfg.anchors.exposure.anchor_exposure_s = self.fast_anchor_exposure.value() or None
        cfg.anchors.exposure.anchor_weight_multiplier = self.fast_anchor_weight_multiplier.value()
        cfg.anchors.exposure.max_weight_ratio = self.fast_anchor_max_weight_ratio.value()
        cfg.anchors.split.recon_count = None if self.fast_recon_count.value() <= 0 else self.fast_recon_count.value()
        cfg.anchors.split.tune_count = self.fast_tune_count.value()
        cfg.anchors.split.qc_count = self.fast_qc_count.value()
        cfg.anchors.split.use_tune_anchors_in_final = self.fast_use_tune_final.isChecked()
        cfg.anchors.split.use_qc_anchors_in_final = self.fast_use_qc_final.isChecked()
        cfg.fdk_sweep.lowres_factor = self.fast_lowres_factor.value()
        cfg.fdk_sweep.filters = self.fast_filters.checked_items() or ["ram_lak"]
        cfg.fdk_sweep.selection_filter = str(self.fast_filter_selection.currentData() or "auto")
        if cfg.fdk_sweep.selection_filter != "auto" and cfg.fdk_sweep.selection_filter not in cfg.fdk_sweep.filters:
            cfg.fdk_sweep.filters.append(cfg.fdk_sweep.selection_filter)
        cfg.prior.tv_weights = self._parse_float_list(self.fast_prior_tv_weights.text())
        cfg.mbir_lite.n_sweeps = self.fast_mbir_sweeps.value()
        cfg.mbir_lite.projection_batch_size = self.fast_mbir_batch_size.value()
        cfg.mbir_lite.ordered_subset_count = self.fast_mbir_subsets.value()
        cfg.mbir_lite.start_mode = str(self.fast_mbir_start_mode.currentData() or "fresh_from_fdk")
        cfg.mbir_lite.lambda_tv = self.fast_lambda_tv.value()
        cfg.mbir_lite.rho_prior = self.fast_rho_prior.value()
        cfg.fast_recon.enabled = self.fast_enabled.isChecked()
        cfg.fast_recon.resume_run_folder = self.fast_resume_run_folder.text() or None
        cfg.fast_recon.save_intermediate = self.fast_save_intermediate.isChecked()
        cfg.fast_recon.make_qc_report = self.fast_make_qc.isChecked()
        return cfg

    def apply_config(self, cfg: AppConfig) -> None:
        self._config_template = AppConfig.from_dict(cfg.to_dict())
        self.projection_folder.setText(cfg.input.projection_folder)
        self.flat_folder.setText(cfg.input.flat_folder)
        self.dark_folder.setText(cfg.input.dark_folder)
        self.metadata_path.setText(cfg.input.metadata_path)
        self.output_folder.setText(cfg.input.output_folder)
        self.filename_column.setCurrentText(cfg.metadata.filename_column)
        self.angle_column.setCurrentText(cfg.metadata.angle_column)
        self.angle_rad_column.setCurrentText(cfg.metadata.angle_rad_column)
        self.x_shift_column.setCurrentText(cfg.metadata.x_shift_column)
        self.y_shift_column.setCurrentText(cfg.metadata.y_shift_column)
        self.remove_endpoint.setChecked(cfg.metadata.remove_duplicate_endpoint)
        self.reverse_angle_order.setChecked(cfg.metadata.reverse_angle_order)
        self.use_flat.setChecked(cfg.preprocessing.use_flat)
        self.use_dark.setChecked(cfg.preprocessing.use_dark)
        self.negative_log.setChecked(cfg.preprocessing.negative_log)
        self.epsilon.setValue(cfg.preprocessing.epsilon)
        self.tmin.setValue(cfg.preprocessing.transmission_clip_min)
        self.tmax.setValue(cfg.preprocessing.transmission_clip_max)
        self.clip_transmission.setChecked(cfg.preprocessing.clip_transmission)
        self.clip_negative.setChecked(cfg.preprocessing.clip_negative_attenuation_to_zero)
        self.binning_y.setValue(int(cfg.preprocessing.binning[0]))
        self.binning_x.setValue(int(cfg.preprocessing.binning[1]))
        self.crop_top.setValue(int(cfg.preprocessing.crop[0]))
        self.crop_bottom.setValue(int(cfg.preprocessing.crop[1]))
        self.crop_left.setValue(int(cfg.preprocessing.crop[2]))
        self.crop_right.setValue(int(cfg.preprocessing.crop[3]))
        self.flip_h.setChecked(cfg.preprocessing.flip_horizontal)
        self.flip_v.setChecked(cfg.preprocessing.flip_vertical)
        self.transpose_for_tigre.setChecked(cfg.preprocessing.transpose_for_tigre)
        self.dsd.setValue(cfg.geometry.source_detector_distance_mm or 0.0)
        self.dso.setValue(cfg.geometry.source_origin_distance_mm or 0.0)
        self.effective_pixel_um.setValue(cfg.geometry.effective_pixel_size_um or 11.0)
        self.voxel_z.setValue(cfg.geometry.voxel_size_mm[0] or 0.011)
        self.voxel_y.setValue(cfg.geometry.voxel_size_mm[1] or 0.011)
        self.voxel_x.setValue(cfg.geometry.voxel_size_mm[2] or 0.011)
        self.detector_rows.setValue(cfg.geometry.detector_pixels[0] or 0)
        self.detector_cols.setValue(cfg.geometry.detector_pixels[1] or 0)
        self.nz.setValue(cfg.geometry.volume_voxels[0] or 0)
        self.ny.setValue(cfg.geometry.volume_voxels[1] or 0)
        self.nx.setValue(cfg.geometry.volume_voxels[2] or 0)
        self.det_v_offset.setValue(cfg.geometry.detector_offset_pixels[0])
        self.det_u_offset.setValue(cfg.geometry.detector_offset_pixels[1])
        self.invert_angle.setChecked(cfg.geometry.invert_angle_sign_for_tigre)
        self.use_center.setChecked(cfg.alignment.use_center_correction)
        self.center_offset.setValue(cfg.alignment.center_offset_pixels)
        self.center_sign.setValue(cfg.alignment.center_shift_sign)
        self.use_drift.setChecked(cfg.alignment.use_drift_correction)
        self.drift_stage.setCurrentText(cfg.alignment.drift_stage)
        self.x_shift_sign.setValue(cfg.alignment.x_shift_sign)
        self.y_shift_sign.setValue(cfg.alignment.y_shift_sign)
        self.init_mode.setCurrentText(cfg.initialization.mode)
        self.fdk_path.setText(cfg.initialization.fdk_volume_path or "")
        self.init_constant.setValue(cfg.initialization.constant_value)
        self.lambda_tv.setValue(cfg.mbir.lambda_tv)
        self.rho.setValue(cfg.mbir.rho)
        self.admm_iterations.setValue(cfg.mbir.max_admm_iterations)
        self.cg_iterations.setValue(cfg.mbir.inner_cg_iterations)
        self.tolerance_primal.setValue(cfg.mbir.tolerance_primal)
        self.tolerance_dual.setValue(cfg.mbir.tolerance_dual)
        self.tolerance_relative_change.setValue(cfg.mbir.tolerance_relative_change)
        self.cg_tolerance.setValue(cfg.mbir.cg_tolerance)
        self.positivity.setChecked(cfg.mbir.positivity)
        self.tv_epsilon.setValue(cfg.mbir.tv_epsilon)
        self.mbir_solver.setCurrentText(cfg.mbir.solver)
        self.mbir_binning_y.setValue(int(cfg.mbir.debug_pixel_binning[0]))
        self.mbir_binning_x.setValue(int(cfg.mbir.debug_pixel_binning[1]))
        self.mbir_projection_stride.setValue(int(cfg.mbir.projection_stride))
        self.memory_mode.setCurrentText(cfg.mbir.memory_mode)
        self.projection_batch_size.setValue(cfg.mbir.projection_batch_size)
        self.ordered_subset_count.setValue(cfg.mbir.ordered_subset_count)
        self.pdhg_dual_dtype.setCurrentText(cfg.mbir.pdhg_dual_dtype)
        self.subset_tv_step_safety.setValue(cfg.mbir.subset_tv_step_safety)
        self.use_gpu.setChecked(cfg.gpu.use_gpu)
        self.gpu_selector.setText(cfg.gpu.gpu_selector or "auto")
        self.gpu_id.setValue(cfg.gpu.gpu_id)
        self.fast_enabled.setChecked(cfg.fast_recon.enabled)
        self.fast_anchor_enabled.setChecked(cfg.anchors.enabled)
        self.fast_anchor_projection_folder.setText(cfg.anchors.input.projection_folder)
        self.fast_anchor_flat_folder.setText(cfg.anchors.input.flat_folder)
        self.fast_anchor_dark_folder.setText(cfg.anchors.input.dark_folder)
        self.fast_anchor_metadata_path.setText(cfg.anchors.input.metadata_path)
        self.fast_anchor_drift_file.setText(cfg.anchors.alignment.drift_shift_file or "")
        self.fast_anchor_inherit_preprocessing.setChecked(cfg.anchors.preprocessing.inherit_main)
        self.fast_anchor_inherit_alignment.setChecked(cfg.anchors.alignment.inherit_main)
        self.fast_main_exposure.setValue(cfg.anchors.exposure.main_exposure_s or 0.0)
        self.fast_anchor_exposure.setValue(cfg.anchors.exposure.anchor_exposure_s or 0.0)
        self.fast_anchor_weight_multiplier.setValue(cfg.anchors.exposure.anchor_weight_multiplier)
        self.fast_anchor_max_weight_ratio.setValue(cfg.anchors.exposure.max_weight_ratio)
        self.fast_recon_count.setValue(cfg.anchors.split.recon_count or 0)
        self.fast_tune_count.setValue(cfg.anchors.split.tune_count)
        self.fast_qc_count.setValue(cfg.anchors.split.qc_count)
        self.fast_use_tune_final.setChecked(cfg.anchors.split.use_tune_anchors_in_final)
        self.fast_use_qc_final.setChecked(cfg.anchors.split.use_qc_anchors_in_final)
        self.fast_lowres_factor.setValue(cfg.fdk_sweep.lowres_factor)
        self.fast_filters.set_checked_items([str(value) for value in cfg.fdk_sweep.filters])
        self._set_combo_by_data(self.fast_filter_selection, getattr(cfg.fdk_sweep, "selection_filter", "auto") or "auto")
        self.fast_prior_tv_weights.setText(", ".join(f"{float(value):g}" for value in cfg.prior.tv_weights))
        self.fast_mbir_sweeps.setValue(cfg.mbir_lite.n_sweeps)
        self.fast_mbir_batch_size.setValue(cfg.mbir_lite.projection_batch_size)
        self.fast_mbir_subsets.setValue(cfg.mbir_lite.ordered_subset_count)
        self._set_combo_by_data(self.fast_mbir_start_mode, getattr(cfg.mbir_lite, "start_mode", "fresh_from_fdk") or "fresh_from_fdk")
        self.fast_lambda_tv.setValue(cfg.mbir_lite.lambda_tv)
        self.fast_rho_prior.setValue(cfg.mbir_lite.rho_prior)
        self.fast_resume_run_folder.setText(cfg.fast_recon.resume_run_folder or "")
        self.fast_save_intermediate.setChecked(cfg.fast_recon.save_intermediate)
        self.fast_make_qc.setChecked(cfg.fast_recon.make_qc_report)
        self._refresh_device_monitor_selection(cfg)
        self._log("Loaded config into GUI.")

    def load_metadata(self) -> None:
        try:
            path = find_metadata_csv(self.metadata_path.text())
            frame = load_metadata_csv(path)
        except Exception as exc:
            self._error("Metadata load failed", str(exc))
            return

        self.metadata_frame = frame
        self.validation = None
        self._clear_preprocessed_arrays()
        self._clear_center_shift_results()
        self.metadata_path.setText(str(path))
        columns = [str(column) for column in frame.columns]
        current_filename = self._combo_text(self.filename_column)
        current_angle = self._combo_text(self.angle_column)
        current_angle_rad = self._optional_combo_text(self.angle_rad_column)
        current_x_shift = self._optional_combo_text(self.x_shift_column)
        current_y_shift = self._optional_combo_text(self.y_shift_column)
        self._set_combo_items(self.filename_column, columns)
        self._set_combo_items(self.angle_column, columns)
        self._set_combo_items(self.angle_rad_column, columns, optional_label="No radian column / use degrees")
        self._set_combo_items(self.x_shift_column, columns, optional_label="No X shift column")
        self._set_combo_items(self.y_shift_column, columns, optional_label="No Y shift column")
        self.filename_column.setCurrentText(current_filename or suggest_column(columns, FILENAME_HINTS))
        self.angle_column.setCurrentText(current_angle or suggest_column(columns, ANGLE_HINTS))
        self._set_optional_combo_guess(self.angle_rad_column, current_angle_rad or suggest_optional_column(columns, ANGLE_RAD_HINTS))
        self._set_optional_combo_guess(self.x_shift_column, current_x_shift or suggest_optional_column(columns, X_SHIFT_HINTS))
        self._set_optional_combo_guess(self.y_shift_column, current_y_shift or suggest_optional_column(columns, Y_SHIFT_HINTS))
        self._log(f"Loaded metadata CSV: {path}")
        self._log(f"Metadata rows: {len(frame)}")
        self._log(f"Metadata columns: {', '.join(columns)}")

    def validate_metadata(self) -> None:
        if self.metadata_frame is None:
            self.load_metadata()
            if self.metadata_frame is None:
                return
        try:
            validation = validate_metadata(
                self.metadata_frame,
                self.projection_folder.text(),
                self._combo_text(self.filename_column),
                self._combo_text(self.angle_column),
                self._optional_combo_text(self.angle_rad_column),
                self._optional_combo_text(self.x_shift_column),
                self._optional_combo_text(self.y_shift_column),
                remove_duplicate_endpoint=self.remove_endpoint.isChecked(),
                reverse_angle_order=self.reverse_angle_order.isChecked(),
            )
            self._populate_validation_table(validation)
            if validation.duplicate_filenames:
                raise ValueError(f"Duplicate projection filenames in metadata: {validation.duplicate_filenames[:10]}")
            if validation.missing_files:
                raise FileNotFoundError(f"Missing projection files: {validation.missing_files[:10]}")
            first_raw = read_tiff_float32(validation.records[0].path, binning=1)
            first_binned = crop_2d(read_tiff_float32(validation.records[0].path, binning=self._binning_tuple()), self._crop_tuple())
        except Exception as exc:
            self.validation = None
            self._error("Metadata validation failed", str(exc))
            return

        self.validation = validation
        self._clear_preprocessed_arrays()
        self._clear_center_shift_results()
        self.expected_raw_shape = first_raw.shape
        rows, cols = first_binned.shape
        self.detector_rows.setValue(rows)
        self.detector_cols.setValue(cols)
        if self.nx.value() == 0:
            self.nx.setValue(cols)
        if self.ny.value() == 0:
            self.ny.setValue(cols)
        if self.nz.value() == 0:
            self.nz.setValue(rows)
        self.preview_projection_index.setRange(0, max(0, len(validation.records) - 1))
        self._log(f"Validated {len(validation.records)} projection/angle pairs.")
        self._log(f"Projection raw image size: rows={first_raw.shape[0]}, cols={first_raw.shape[1]}")
        self._log(f"Projection image size after binning/crop: rows={rows}, cols={cols}")
        self._log(f"Estimated float32 projection stack memory: {estimate_stack_memory_gb(len(validation.records), rows, cols):.3f} GB")
        self._log(f"Angle input source: {validation.angle_input_source}")
        self._log(f"Angle range degrees: {validation.angle_min_deg:.8g} to {validation.angle_max_deg:.8g}")
        self._log(f"Angle direction: {validation.angle_direction}")
        self._log(f"Duplicate endpoint detected: {validation.duplicate_endpoint_detected}")
        self._log(f"Duplicate endpoint removed: {validation.duplicate_endpoint_removed}")
        self._log(f"X shift column: {validation.x_shift_column or '(none)'}, found={validation.x_shift_column_found}")
        self._log(f"Y shift column: {validation.y_shift_column or '(none)'}, found={validation.y_shift_column_found}")
        for warning in validation.warnings:
            self._log(f"Warning: {warning}")
        self.preview_raw_projection()

    def preview_raw_projection(self) -> None:
        if self.validation is None:
            self.validate_metadata()
            if self.validation is None:
                return
        index = min(max(self.preview_projection_index.value(), 0), len(self.validation.records) - 1)
        record = self.validation.records[index]
        try:
            image = crop_2d(read_tiff_float32(record.path, binning=self._binning_tuple()), self._crop_tuple())
        except Exception as exc:
            self._error("Projection preview failed", str(exc))
            return
        self._show_preview_image(image, f"Raw projection {index}: {record.filename}")
        self._log(f"Displayed raw projection preview {index}: {record.filename}, shape={tuple(image.shape)}")

    def compute_preprocessed_previews(self) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, "Busy", "A processing task is already running.")
            return
        if self.validation is None:
            self.validate_metadata()
            if self.validation is None:
                return
        cfg = self.config_from_ui()
        validation = self.validation
        self._cancel = CancellationToken()
        cancel_token = self._cancel
        self.progress_bar.setRange(0, 0)
        self._set_busy(True)

        def job(log: Callable[[str], None]) -> dict[str, object]:
            log("Computing preprocessed preview stacks.")
            result = run_preprocessing(
                validation,
                cfg.input.projection_folder,
                cfg.input.flat_folder,
                cfg.input.dark_folder,
                cfg.preprocessing,
                cfg.alignment,
                progress=log,
                cancel_check=cancel_token.is_cancelled,
            )
            return {"mode": "preprocess_preview", "result": result}

        self._start_worker(job, self._preprocessing_preview_success)

    def _preprocessing_preview_success(self, payload: dict[str, object]) -> None:
        result = payload["result"]
        if not isinstance(result, PreprocessingResult):
            self._error("Preprocessing preview failed", "Unexpected preprocessing result type.")
            return
        self.raw_stack = result.raw_stack
        self.flat_field = result.flat_field
        self.dark_field = result.dark_field
        self.transmission_stack = result.transmission_stack
        self.attenuation_stack = result.attenuation_stack
        self.preview_projection_index.setRange(0, max(0, self.raw_stack.shape[0] - 1))
        self._log("Preprocessed preview stacks are ready.")
        for line in result.report_lines:
            self._log(line)
        self.preview_selected_preprocessed_image()

    def export_drift_corrected_transmission_stack(self) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, "Busy", "A processing task is already running.")
            return
        if self.validation is None:
            self.validate_metadata()
            if self.validation is None:
                return
        validation = self.validation
        if validation.x_shift_px is None or validation.y_shift_px is None:
            self._error(
                "Drift shifts unavailable",
                "Select valid X and Y shift columns, then validate metadata before exporting drift-corrected transmission.",
            )
            return
        cfg = self.config_from_ui()
        if not cfg.input.flat_folder:
            self._error(
                "Flat folder required",
                "A flat/reference folder is required because this export writes flat-field corrected transmission TIFFs.",
            )
            return
        export_folder = QFileDialog.getExistingDirectory(self, "Choose folder for drift-corrected transmission TIFFs")
        if not export_folder:
            return
        target = Path(export_folder)
        has_existing_tiffs = any(target.glob("*.tif")) or any(target.glob("*.tiff")) or any(target.glob("*.TIF")) or any(
            target.glob("*.TIFF")
        )
        if has_existing_tiffs:
            answer = QMessageBox.question(
                self,
                "Folder contains TIFFs",
                "The selected folder already contains TIFF files. Existing files with the export prefix may be overwritten. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        cfg.preprocessing.use_flat = True
        cfg.alignment.use_drift_correction = True
        cfg.alignment.drift_stage = "transmission"
        self._cancel = CancellationToken()
        cancel_token = self._cancel
        self.progress_bar.setRange(0, 0)
        self._set_busy(True)

        def job(log: Callable[[str], None]) -> dict[str, object]:
            log("Exporting flat-field corrected, drift-corrected transmission TIFF stack.")
            log("Export uses drift stage: transmission, so shifts are applied before attenuation -log.")
            result = run_preprocessing(
                validation,
                cfg.input.projection_folder,
                cfg.input.flat_folder,
                cfg.input.dark_folder,
                cfg.preprocessing,
                cfg.alignment,
                progress=log,
                cancel_check=cancel_token.is_cancelled,
            )
            if result.transmission_stack is None:
                raise RuntimeError("Preprocessing did not produce a transmission stack.")
            output = save_stack_folder(target, result.transmission_stack, "drift_corrected_transmission", progress=log)
            manifest = self._write_drift_transmission_manifest(output, validation)
            return {"mode": "export_drift_transmission", "result": result, "folder": output, "manifest": manifest}

        self._start_worker(job, self._drift_transmission_export_success)

    def _drift_transmission_export_success(self, payload: dict[str, object]) -> None:
        result = payload["result"]
        if not isinstance(result, PreprocessingResult):
            self._error("Export failed", "Unexpected preprocessing result type.")
            return
        self.raw_stack = result.raw_stack
        self.flat_field = result.flat_field
        self.dark_field = result.dark_field
        self.transmission_stack = result.transmission_stack
        self.attenuation_stack = result.attenuation_stack
        if self.raw_stack is not None:
            self.preview_projection_index.setRange(0, max(0, self.raw_stack.shape[0] - 1))
        folder = Path(payload["folder"])
        manifest = Path(payload["manifest"])
        count = 0 if self.transmission_stack is None else int(self.transmission_stack.shape[0])
        self._log(f"Saved {count} drift-corrected transmission TIFFs to: {folder}")
        self._log(f"Saved drift export manifest: {manifest}")
        for line in result.report_lines:
            self._log(line)
        self.preprocess_preview_kind.setCurrentText("Flat-field corrected projection")
        self.preview_selected_preprocessed_image()

    @staticmethod
    def _write_drift_transmission_manifest(folder: Path, validation: MetadataValidation) -> Path:
        manifest = Path(folder) / "drift_corrected_transmission_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "index",
                    "exported_filename",
                    "source_filename",
                    "angle_deg",
                    "angle_rad",
                    "x_shift_px",
                    "y_shift_px",
                ]
            )
            for index, record in enumerate(validation.records):
                writer.writerow(
                    [
                        index,
                        f"drift_corrected_transmission_{index:06d}.tif",
                        record.filename,
                        f"{record.angle_deg:.12g}",
                        f"{record.angle_rad:.12g}",
                        "" if record.x_shift_px is None else f"{record.x_shift_px:.12g}",
                        "" if record.y_shift_px is None else f"{record.y_shift_px:.12g}",
                    ]
                )
        return manifest

    def preview_selected_preprocessed_image(self) -> None:
        kind = self.preprocess_preview_kind.currentText()
        if kind == "Raw projection":
            if self.raw_stack is None:
                self.preview_raw_projection()
                return
            image = self._stack_preview_image(self.raw_stack)
            self._show_preview_image(image, self._preview_title("Raw projection"))
            self._log(f"Displayed raw projection from loaded preview stack at index {self._preview_index()}.")
            return
        if kind == "Averaged flat field":
            if self.flat_field is None:
                self._error("Flat field unavailable", "Compute the preprocessed preview stack first.")
                return
            self._show_preview_image(self.flat_field, "Averaged flat field")
            self._log(f"Displayed averaged flat field, shape={tuple(self.flat_field.shape)}.")
            return
        if kind == "Flat-field corrected projection":
            if self.transmission_stack is None:
                self._error("Transmission unavailable", "Compute the preprocessed preview stack first.")
                return
            image = self._stack_preview_image(self.transmission_stack)
            self._show_preview_image(image, self._preview_title("Flat-field corrected transmission"))
            self._log(f"Displayed flat-field corrected transmission at index {self._preview_index()}.")
            return
        if kind == "Attenuation projection":
            if self.attenuation_stack is None:
                self._error("Attenuation unavailable", "Compute the preprocessed preview stack first.")
                return
            image = self._stack_preview_image(self.attenuation_stack)
            self._show_preview_image(image, self._preview_title("Attenuation projection"))
            self._log(f"Displayed attenuation projection at index {self._preview_index()}.")

    def _stack_preview_image(self, stack):
        index = self._preview_index(stack.shape[0])
        return stack[index]

    def _preview_index(self, count: int | None = None) -> int:
        index = self.preview_projection_index.value()
        if count is None:
            return index
        return min(max(index, 0), max(0, count - 1))

    def _preview_title(self, label: str) -> str:
        index = self._preview_index()
        filename = ""
        if self.validation is not None and 0 <= index < len(self.validation.records):
            filename = f": {self.validation.records[index].filename}"
        return f"{label} {index}{filename}"

    def _clear_preprocessed_arrays(self) -> None:
        self.raw_stack = None
        self.flat_field = None
        self.dark_field = None
        self.transmission_stack = None
        self.attenuation_stack = None

    def _show_preview_image(self, image: np.ndarray | None, title: str, preserve_zoom: bool = False) -> None:
        self.preview.show_image(image, title, preserve_zoom=preserve_zoom)
        self.tabs.setCurrentWidget(self.preview)
        self._active_image_changed()

    def _show_preview_volume(self, volume: np.ndarray | None, title: str) -> None:
        self._show_tomogram_volume(volume, title)

    def _show_tomogram_volume(self, volume: np.ndarray | None, title: str) -> None:
        self.tomogram_view.show_volume(volume, title)
        self.tabs.setCurrentWidget(self.tomogram_view)
        self._active_image_changed()

    def _set_tomogram_sources(
        self,
        sources: dict[str, TomogramSource],
        selected_id: str | None = None,
        *,
        replace: bool = False,
    ) -> None:
        source_maps = [sources] if replace else [self._tomogram_sources, sources]
        available = _merge_tomogram_source_maps(*source_maps)
        self._tomogram_sources = available
        current_id = self.tomogram_view.current_tomogram_source_id()
        target_id = selected_id if selected_id in available else current_id if current_id in available else None
        self.tomogram_view.set_tomogram_choices(
            [(source_id, source.label) for source_id, source in available.items()],
            selected_id=target_id,
        )

    def _tomogram_source_selected(self, source_id: str) -> None:
        self._load_tomogram_source(source_id, switch_to_tab=True, preserve_view_state=True)

    def _load_tomogram_source(
        self,
        source_id: str | None,
        *,
        switch_to_tab: bool,
        preserve_view_state: bool = True,
    ) -> bool:
        if not source_id:
            return False
        source = self._tomogram_sources.get(str(source_id))
        if source is None:
            return False
        try:
            volume = source.volume
            if volume is None:
                if source.path is None:
                    raise FileNotFoundError("No volume path is available for this tomogram.")
                if not source.path.exists():
                    raise FileNotFoundError(f"Volume file is missing: {source.path}")
                volume = np.load(source.path, mmap_mode="r", allow_pickle=False)
            self.tomogram_view.select_tomogram_choice(str(source_id))
            self.tomogram_view.show_volume(volume, source.label, preserve_view_state=preserve_view_state)
            if switch_to_tab:
                self.tabs.setCurrentWidget(self.tomogram_view)
            self._active_image_changed()
            return True
        except Exception as exc:
            self._log(f"Tomogram load failed for {source.label}: {exc}")
            self.tomogram_view.show_message(f"Could not load {source.label}")
            self._error("Tomogram load failed", str(exc))
            return False

    def _tomogram_sources_from_result(self, result) -> dict[str, TomogramSource]:
        folders = getattr(result, "run_folders", None)
        path_sources = self._tomogram_sources_from_run_root(Path(folders.root)) if folders is not None else {}
        live_sources: dict[str, TomogramSource] = {}
        explicit_sources: dict[str, TomogramSource] = {}

        fdk_path = Path(folders.fdk) / "fdk_initial_volume.npy" if folders is not None else None
        mbir_path = Path(folders.mbir) / "mbir_final_volume.npy" if folders is not None else None
        fast_fdk_path = Path(folders.fast_fdk_sweep) / "fdk_best.npy" if folders is not None else None
        fast_prior_path = Path(folders.fast_prior) / "prior_selected.npy" if folders is not None else None
        fast_mbir_path = Path(folders.fast_mbir_lite) / "mbir_lite_final.npy" if folders is not None else None

        _append_tomogram_source(
            live_sources,
            "fdk",
            "FDK reconstruction",
            path=fdk_path,
            volume=getattr(result, "fdk_volume", None),
        )
        _append_tomogram_source(
            live_sources,
            "mbir",
            "TomoAnchor MBIR final",
            path=mbir_path,
            volume=getattr(getattr(result, "mbir_result", None), "volume", None),
        )
        _append_tomogram_source(
            live_sources,
            "fast_fdk",
            "Fast FDK reconstruction",
            path=fast_fdk_path,
            volume=getattr(getattr(result, "fdk_sweep_result", None), "best_volume", None),
        )
        _append_tomogram_source(
            live_sources,
            "fast_prior",
            "Fast prior",
            path=fast_prior_path,
            volume=getattr(getattr(result, "prior_result", None), "x_prior", None),
        )
        _append_tomogram_source(
            live_sources,
            "mbir_lite",
            "MBIR-lite final",
            path=fast_mbir_path,
            volume=getattr(getattr(result, "mbir_lite_result", None), "volume", None),
        )

        try:
            cfg = self.config_from_ui()
        except Exception:
            cfg = None
        explicit_fdk_path = (
            Path(cfg.initialization.fdk_volume_path)
            if cfg is not None and cfg.initialization.fdk_volume_path
            else None
        )
        if explicit_fdk_path is not None and explicit_fdk_path.exists():
            normalized_fdk_path = None
            if fdk_path is not None and fdk_path.exists():
                try:
                    normalized_fdk_path = fdk_path.resolve()
                except Exception:
                    normalized_fdk_path = fdk_path
            try:
                normalized_explicit = explicit_fdk_path.resolve()
            except Exception:
                normalized_explicit = explicit_fdk_path
            if normalized_fdk_path is None or normalized_explicit != normalized_fdk_path:
                _append_tomogram_source(
                    explicit_sources,
                    "existing_fdk",
                    "Existing FDK volume",
                    path=explicit_fdk_path,
                )

        return _merge_tomogram_source_maps(path_sources, explicit_sources, live_sources)

    def _preload_existing_results(self) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, "Busy", "Wait for the current reconstruction task to finish before preloading results.")
            return
        cfg = self.config_from_ui()
        roots = self._preload_candidate_roots(cfg)
        explicit_fdk = Path(cfg.initialization.fdk_volume_path) if cfg.initialization.fdk_volume_path else None
        explicit_fdk = explicit_fdk if explicit_fdk is not None and explicit_fdk.exists() else None

        loaded_root: Path | None = None
        loaded_sources: dict[str, TomogramSource] = {}
        for root in roots:
            sources = self._tomogram_sources_from_run_root(root)
            if sources or self._run_has_metrics(root) or self._preferred_mbir_preview_path(root)[0] is not None:
                loaded_root = root
                loaded_sources = sources
                break

        if explicit_fdk is not None:
            loaded_sources.setdefault("existing_fdk", TomogramSource("Existing FDK volume", path=explicit_fdk))

        if loaded_root is None and not loaded_sources:
            checked = [str(root) for root in roots[:5]]
            if explicit_fdk is not None:
                checked.append(str(explicit_fdk))
            detail = "\n".join(checked) if checked else "No output folder or resume folder was available."
            QMessageBox.information(
                self,
                "No existing results found",
                "No saved reconstruction results were found to preload.\n\nChecked:\n" + detail,
            )
            return

        loaded_parts: list[str] = []
        if loaded_root is not None:
            self.metrics_plot.show_metrics([])
            self.metrics_plot.set_run_folder(loaded_root)
            category = self._preferred_metrics_category(loaded_root)
            if category:
                self.metrics_plot.select_category(category)
            standard_metrics = self._load_standard_metrics_from_run(loaded_root)
            if standard_metrics and category is None:
                self.metrics_plot.select_category("MBIR-lite")
                self.metrics_plot.show_metrics(standard_metrics)
            elif standard_metrics and category == "MBIR-lite" and self._fast_child_path(loaded_root, "mbir_lite", "metrics.csv") is None:
                self.metrics_plot.show_metrics(standard_metrics)
            loaded_parts.append(f"run folder {loaded_root}")

        if loaded_sources:
            preferred_tomogram_id = self._preferred_tomogram_id(loaded_sources)
            self._set_tomogram_sources(loaded_sources, preferred_tomogram_id, replace=True)
            if preferred_tomogram_id is not None:
                self._load_tomogram_source(preferred_tomogram_id, switch_to_tab=False, preserve_view_state=False)
            loaded_parts.append(f"{len(loaded_sources)} tomogram source(s)")

        if loaded_root is not None:
            mbir_path, mbir_title = self._preferred_mbir_preview_path(loaded_root)
            if mbir_path is not None:
                self._load_mbir_preview_from_path(mbir_path, mbir_title)
                loaded_parts.append("MBIR preview slice")
        elif explicit_fdk is not None:
            self._load_mbir_preview_from_path(explicit_fdk, "Existing FDK volume")
            loaded_parts.append("preview slice from existing FDK")

        if loaded_parts:
            self._log("Preloaded existing results: " + "; ".join(loaded_parts) + ".")
        self._active_image_changed()

    def _preload_candidate_roots(self, cfg: AppConfig) -> list[Path]:
        roots: list[Path] = []

        def add_root(path: str | Path | None) -> None:
            if not path:
                return
            root = self._normalize_preload_run_root(Path(path))
            if not root.exists():
                return
            if root not in roots:
                roots.append(root)

        add_root(cfg.fast_recon.resume_run_folder)
        output_root = Path(cfg.input.output_folder) if cfg.input.output_folder else None
        if output_root is not None and output_root.exists():
            if self._looks_like_run_root(output_root):
                add_root(output_root)
            try:
                children = [child for child in output_root.iterdir() if child.is_dir()]
            except Exception:
                children = []
            children.sort(key=lambda path: (path.name.startswith("run_"), _safe_mtime(path), path.name), reverse=True)
            for child in children[:50]:
                add_root(child)
        if cfg.initialization.fdk_volume_path:
            fdk_path = Path(cfg.initialization.fdk_volume_path)
            if fdk_path.exists():
                for parent in fdk_path.parents:
                    if self._looks_like_run_root(parent):
                        add_root(parent)
                        break
        return roots

    def _normalize_preload_run_root(self, path: Path) -> Path:
        root = Path(path)
        fast_children = {"anchors", "fdk_sweep", "intermediate", "mbir_lite", "prior", "qc"}
        if root.parent.name == "fast_recon" and root.name in fast_children:
            root = root.parent
        while root.name == "fast_recon":
            root = root.parent
        if root.name in {"fdk", "mbir", "metrics", "preprocessing", "alignment"}:
            root = root.parent
        return root

    def _run_folders_from_root(self, root: Path) -> RunFolders:
        root = self._normalize_preload_run_root(root)
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
            fast_fdk_sweep=self._fast_child_folder(root, "fdk_sweep"),
            fast_prior=self._fast_child_folder(root, "prior"),
            fast_mbir_lite=self._fast_child_folder(root, "mbir_lite"),
            fast_qc=self._fast_child_folder(root, "qc"),
        )

    def _fast_child_folder(self, root: Path, child: str) -> Path:
        base = self._normalize_preload_run_root(root) / "fast_recon"
        folder = base / child
        legacy = base / "fast_recon" / child
        if not folder.exists() and legacy.exists():
            return legacy
        return folder

    def _fast_child_path(self, root: Path, child: str, filename: str) -> Path | None:
        path = self._fast_child_folder(root, child) / filename
        return path if path.exists() else None

    def _tomogram_sources_from_run_root(self, root: Path) -> dict[str, TomogramSource]:
        folders = self._run_folders_from_root(root)
        candidates = [
            ("fdk", "FDK reconstruction", folders.fdk / "fdk_initial_volume.npy"),
            ("mbir", "TomoAnchor MBIR final", folders.mbir / "mbir_final_volume.npy"),
            ("fast_fdk", "Fast FDK reconstruction", folders.fast_fdk_sweep / "fdk_best.npy"),
            ("fast_prior", "Fast prior", folders.fast_prior / "prior_selected.npy"),
            ("mbir_lite", "MBIR-lite final", folders.fast_mbir_lite / "mbir_lite_final.npy"),
        ]
        sources: dict[str, TomogramSource] = {}
        for source_id, label, path in candidates:
            if path.exists():
                sources[source_id] = TomogramSource(label, path=path)
        return sources

    def _preferred_tomogram_id(self, sources: dict[str, TomogramSource]) -> str | None:
        for source_id in ("mbir_lite", "mbir", "fast_prior", "fast_fdk", "existing_fdk", "fdk"):
            if source_id in sources:
                return source_id
        return next(iter(sources), None)

    def _preferred_mbir_preview_path(self, root: Path) -> tuple[Path | None, str]:
        folders = self._run_folders_from_root(root)
        candidates = [
            (folders.fast_mbir_lite / "mbir_lite_final.npy", "Fast MBIR-lite final"),
            (folders.mbir / "mbir_final_volume.npy", "TomoAnchor MBIR final"),
        ]
        for path, title in candidates:
            if path.exists():
                return path, title
        return None, ""

    def _preferred_metrics_category(self, root: Path) -> str | None:
        if self._fast_child_path(root, "mbir_lite", "metrics.csv") is not None:
            return "MBIR-lite"
        if self._fast_child_path(root, "prior", "prior_metrics.csv") is not None:
            return "Make prior"
        if self._fast_child_path(root, "fdk_sweep", "scores.csv") is not None:
            return "FDK sweep"
        if self._fast_child_path(root, "qc", "qc_metrics.csv") is not None:
            return "QC report"
        return None

    def _run_has_metrics(self, root: Path) -> bool:
        return any(
            path is not None
            for path in (
                self._fast_child_path(root, "mbir_lite", "metrics.csv"),
                self._fast_child_path(root, "prior", "prior_metrics.csv"),
                self._fast_child_path(root, "fdk_sweep", "scores.csv"),
                self._fast_child_path(root, "qc", "qc_metrics.csv"),
            )
        ) or (self._normalize_preload_run_root(root) / "metrics" / "metrics.csv").exists()

    def _looks_like_run_root(self, root: Path) -> bool:
        if not root.exists() or not root.is_dir():
            return False
        folders = self._run_folders_from_root(root)
        paths = [
            folders.fdk / "fdk_initial_volume.npy",
            folders.mbir / "mbir_final_volume.npy",
            folders.metrics / "metrics.csv",
            folders.fast_fdk_sweep / "fdk_best.npy",
            folders.fast_prior / "prior_selected.npy",
            folders.fast_prior / "prior_metrics.csv",
            folders.fast_mbir_lite / "mbir_lite_final.npy",
            folders.fast_mbir_lite / "metrics.csv",
            folders.fast_qc / "qc_metrics.csv",
        ]
        return any(path.exists() for path in paths)

    def _load_standard_metrics_from_run(self, root: Path) -> list[object]:
        path = self._normalize_preload_run_root(root) / "metrics" / "metrics.csv"
        if not path.exists() or path.stat().st_size == 0:
            return []
        rows: list[object] = []
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    values = {key: _coerce_metric_value(value) for key, value in row.items()}
                    rows.append(SimpleNamespace(**values))
        except Exception as exc:
            self._log(f"Could not preload standard MBIR metrics from {path}: {exc}")
            return []
        return rows

    def _load_mbir_preview_from_path(self, path: Path, title: str) -> bool:
        try:
            volume = np.load(path, mmap_mode="r", allow_pickle=False)
            self._show_mbir_requested_slice_from_volume(volume, title)
            self._mbir_preview_volume_source = (Path(path), str(title))
            return True
        except Exception as exc:
            self._log(f"MBIR preview preload failed for {path}: {exc}")
            return False

    def _mbir_preview_request_changed(self, view: str, slice_index: int) -> None:
        with self._mbir_preview_request_lock:
            self._mbir_preview_request = {"view": str(view), "slice_index": int(slice_index)}
        if self._thread is not None:
            self.mbir_preview.mark_request_pending(view, slice_index)
            return
        fast_volume = getattr(getattr(self._last_result, "mbir_lite_result", None), "volume", None)
        if fast_volume is not None:
            self._show_mbir_requested_slice(fast_volume, "Fast MBIR-lite final")
            return
        volume = getattr(getattr(self._last_result, "mbir_result", None), "volume", None)
        if volume is not None:
            self._show_mbir_requested_slice(volume, "TomoAnchor MBIR final")
            return
        if self._mbir_preview_volume_source is not None:
            path, title = self._mbir_preview_volume_source
            if path.exists():
                self._load_mbir_preview_from_path(path, title)

    def _mbir_preview_request_snapshot(self) -> dict[str, object]:
        with self._mbir_preview_request_lock:
            return dict(self._mbir_preview_request)

    def _show_mbir_requested_slice(self, volume: np.ndarray, title: str) -> None:
        self._mbir_preview_volume_source = None
        self._show_mbir_requested_slice_from_volume(volume, title)

    def _show_mbir_requested_slice_from_volume(self, volume: np.ndarray, title: str) -> None:
        image, view, slice_index, slice_counts = self._extract_preview_slice_from_volume(
            volume,
            self._mbir_preview_request_snapshot(),
        )
        self.mbir_preview.show_slice(image, title, view, slice_index, slice_counts=slice_counts)

    def _extract_preview_slice_from_volume(
        self,
        volume: np.ndarray,
        request: object | None = None,
    ) -> tuple[np.ndarray, str, int, dict[str, int]]:
        if getattr(volume, "ndim", None) != 3:
            raise ValueError(f"Expected a 3-D MBIR preview volume, got shape {getattr(volume, 'shape', 'unknown')}")
        shape = tuple(int(value) for value in volume.shape)
        slice_counts = preview_slice_counts_for_shape(shape)
        request_map = request if isinstance(request, dict) else {}
        view_text = str(request_map.get("view") or "Axial").strip().lower()
        if view_text.startswith("cor"):
            view = "Coronal"
        elif view_text.startswith("sag"):
            view = "Sagittal"
        else:
            view = "Axial"
        default_index = max(0, slice_counts[view] // 2)
        slice_index = min(max(int(request_map.get("slice_index", default_index)), 0), slice_counts[view] - 1)
        if view == "Coronal":
            image = volume[:, slice_index, :]
        elif view == "Sagittal":
            image = volume[:, :, slice_index]
        else:
            image = volume[slice_index]
        return np.ascontiguousarray(np.asarray(image, dtype=np.float32)), view, slice_index, slice_counts

    def _active_image_widget(self):
        if hasattr(self, "tabs"):
            current = self.tabs.currentWidget()
            if hasattr(self, "mbir_preview") and current is self.mbir_preview:
                return self.mbir_preview
            if hasattr(self, "tomogram_view") and current is self.tomogram_view:
                return self.tomogram_view
            if hasattr(self, "metrics_plot") and current is self.metrics_plot:
                return self.metrics_plot
            if hasattr(self, "preview") and current is self.preview:
                return self.preview
        if hasattr(self, "preview"):
            return self.preview
        if hasattr(self, "metrics_plot"):
            return self.metrics_plot
        return None

    def _active_image_changed(self) -> None:
        self._sync_level_histogram()
        self._update_zoom_label()

    def _sync_level_histogram(self) -> None:
        if not hasattr(self, "level_histogram"):
            return
        active = self._active_image_widget()
        if active is None:
            return
        image = active.current_image()
        levels = active.current_levels()
        self.level_histogram.set_image(image, levels)

    def _preview_levels_changed(self, low: float, high: float) -> None:
        active = self._active_image_widget()
        if active is not None:
            active.set_levels(low, high)

    def _auto_level_active_preview(self) -> None:
        active = self._active_image_widget()
        if active is None:
            return
        levels = active.reset_auto_levels()
        if levels is not None:
            self.level_histogram.set_image(active.current_image(), levels)
        self._update_zoom_label()

    def _full_range_active_preview(self) -> None:
        active = self._active_image_widget()
        if active is None:
            return
        if active.current_image() is None:
            return
        levels = self.level_histogram.full_range_levels()
        self.level_histogram.set_levels(levels[0], levels[1], emit=False)
        active.set_levels(levels[0], levels[1])
        self._update_zoom_label()

    def _zoom_active_image(self, factor: float) -> None:
        active = self._active_image_widget()
        if active is None:
            return
        if active.current_image() is None:
            return
        current = active.current_zoom_factor()
        active.set_zoom_factor(current * factor)
        self._update_zoom_label()

    def _fit_active_image(self) -> None:
        active = self._active_image_widget()
        if active is None:
            return
        if active.current_image() is None:
            return
        active.fit_to_view()
        self._update_zoom_label()

    def _update_zoom_label(self) -> None:
        if not hasattr(self, "zoom_label"):
            return
        active = self._active_image_widget()
        if active is None:
            return
        has_image = active.current_image() is not None
        zoom = active.current_zoom_factor() if has_image else 1.0
        self.zoom_label.setText(f"Zoom: {zoom * 100:.0f}%")
        self.zoom_in_button.setEnabled(has_image)
        self.zoom_out_button.setEnabled(has_image and zoom > 1.0001)
        self.fit_view_button.setEnabled(has_image and zoom > 1.0001)

    def run_center_shift_search(self, automatic: bool = False) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, "Busy", "A processing task is already running.")
            return
        if self.validation is None:
            self.validate_metadata()
            if self.validation is None:
                return
        shifts = self._center_shift_values_from_ui()
        if not shifts:
            return
        cfg = self.config_from_ui()
        validation = self.validation
        existing_attenuation = self.attenuation_stack
        metric_key = str(self.center_search_metric.currentData() or "combined_score")
        self._cancel = CancellationToken()
        cancel_token = self._cancel
        self.progress_bar.setRange(0, 0)
        self._set_busy(True)
        mode_label = "automatic" if automatic else "manual preview"

        def job(log: Callable[[str], None]) -> dict[str, object]:
            log(f"Center-offset {mode_label} search started.")
            log(f"Shift range: {shifts[0]:.4f} to {shifts[-1]:.4f} px, count={len(shifts)}")
            log(f"Center-shift sign convention: {cfg.alignment.center_shift_sign:+g}")
            preprocess_result = None
            if existing_attenuation is None:
                log("No attenuation preview stack is loaded; computing preprocessing first.")
                preprocess_result = run_preprocessing(
                    validation,
                    cfg.input.projection_folder,
                    cfg.input.flat_folder,
                    cfg.input.dark_folder,
                    cfg.preprocessing,
                    cfg.alignment,
                    progress=log,
                    cancel_check=cancel_token.is_cancelled,
                )
                attenuation = preprocess_result.attenuation_stack
            else:
                attenuation = existing_attenuation
                log("Using already computed attenuation preview stack for center search.")
            tigre_input, transpose_report = prepare_tigre_projection_input(attenuation, cfg.preprocessing.transpose_for_tigre)
            for line in transpose_report:
                log(line)
            params = params_from_config(cfg.geometry, detector_shape=tigre_input.shape[1:], alignment=cfg.alignment)
            angles = angles_for_tigre(validation.angles_rad, cfg.geometry.invert_angle_sign_for_tigre)
            detected_gpu_infos = query_all_nvidia_gpu_memory() if cfg.gpu.use_gpu else []
            gpu_selection = resolve_gpu_selection(
                cfg.gpu.use_gpu,
                cfg.gpu.gpu_selector,
                cfg.gpu.gpu_id,
                [info.gpu_id for info in detected_gpu_infos],
            )
            log(f"Center-search GPU selection: {gpu_selection.describe()} (selector='{gpu_selection.selector}')")
            for warning in gpu_selection.warnings:
                log(warning)
            for index, shift_px in enumerate(shifts, start=1):
                detector_offset = cfg.alignment.center_shift_sign * shift_px * params.detector_pixel_size_mm[1]
                log(
                    f"Prepared center preview {index}/{len(shifts)}: "
                    f"shift={shift_px:.4f} px, detector offset={detector_offset:.8g} mm"
                )
            results = reconstruct_center_shift_previews(
                tigre_input,
                angles,
                params,
                shifts,
                center_shift_sign=cfg.alignment.center_shift_sign,
                gpu_ids=gpu_selection.gpu_ids or None,
                use_gpu=cfg.gpu.use_gpu,
                progress=log,
                cancel_check=cancel_token.is_cancelled,
            )
            if not results:
                raise RuntimeError("No center-offset previews were reconstructed.")
            metric_table = None
            if automatic:
                metric_table = compute_metrics_for_results(results)
                for row in metric_table:
                    log(
                        f"Shift {row['shift_px']:.4f} px: "
                        f"gradient={row['gradient_energy']:.8g}, "
                        f"laplacian={row['laplacian_variance']:.8g}, "
                        f"entropy={row['entropy']:.8g}, "
                        f"combined={row['combined_score']:.8g}"
                    )
                best = recommend_metric_index(metric_table, metric_key)
                if best is not None:
                    log(f"Automatic recommended center shift: {metric_table[best]['shift_px']:.4f} px")
            return {
                "results": results,
                "metric_table": metric_table,
                "automatic": automatic,
                "preprocess_result": preprocess_result,
            }

        self._start_worker(job, self._center_shift_search_success)

    def _center_shift_search_success(self, payload: dict[str, object]) -> None:
        preprocess_result = payload.get("preprocess_result")
        if isinstance(preprocess_result, PreprocessingResult):
            self.raw_stack = preprocess_result.raw_stack
            self.flat_field = preprocess_result.flat_field
            self.dark_field = preprocess_result.dark_field
            self.transmission_stack = preprocess_result.transmission_stack
            self.attenuation_stack = preprocess_result.attenuation_stack
        self.center_shift_results = list(payload["results"])
        self.auto_metric_table = payload.get("metric_table")
        self.current_center_shift_preview_index = len(self.center_shift_results) // 2
        self.center_preview_slider.blockSignals(True)
        self.center_preview_slider.setRange(0, max(0, len(self.center_shift_results) - 1))
        self.center_preview_slider.setValue(self.current_center_shift_preview_index)
        self.center_preview_slider.blockSignals(False)
        if self.auto_metric_table:
            self._update_auto_center_shift_recommendation()
        else:
            self.auto_best_shift_px = None
            self.auto_best_preview_index = None
            self.center_auto_label.setText("Automatic recommendation unavailable")
            self.metrics_plot.show_center_shift_metrics([], str(self.center_search_metric.currentData() or "combined_score"))
        self._update_center_shift_controls()
        self._show_selected_center_shift_preview()
        self._log("Center-offset search finished.")

    def run_fine_center_shift_search(self) -> None:
        selected = self._selected_center_shift_result()
        selected_shift = selected.shift_px if selected is not None else self.center_offset.value()
        previous_step = max(self.center_search_step.value(), 1e-6)
        half_width = max(1.0, 2.0 * previous_step)
        fine_step = max(previous_step / 5.0, 1e-6)
        self.center_search_start.setValue(selected_shift - half_width)
        self.center_search_end.setValue(selected_shift + half_width)
        self.center_search_step.setValue(fine_step)
        self._log(
            "Prepared fine center-offset search: "
            f"{selected_shift - half_width:.4f} to {selected_shift + half_width:.4f} px, step {fine_step:.4f} px"
        )
        self.run_center_shift_search(automatic=self.auto_metric_table is not None)

    def apply_selected_center_shift(self) -> None:
        result = self._selected_center_shift_result()
        if result is None:
            self._error("No center-offset preview", "Run a center-offset search before applying a selected shift.")
            return
        self.use_center.setChecked(True)
        self.center_offset.setValue(result.shift_px)
        self.center_sign.setValue(1.0 if result.center_shift_sign >= 0 else -1.0)
        self._log(f"Applied selected center shift: {result.shift_px:.4f} px")

    def apply_auto_center_shift(self) -> None:
        if self.auto_best_shift_px is None:
            self._error("No automatic recommendation", "Run automatic center-offset search first.")
            return
        self.use_center.setChecked(True)
        self.center_offset.setValue(self.auto_best_shift_px)
        if self.auto_best_preview_index is not None and self.auto_best_preview_index < len(self.center_shift_results):
            sign = self.center_shift_results[self.auto_best_preview_index].center_shift_sign
            self.center_sign.setValue(1.0 if sign >= 0 else -1.0)
        self._log(f"Applied automatic best center shift: {self.auto_best_shift_px:.4f} px")

    def _center_shift_preview_changed(self, index: int) -> None:
        if not self.center_shift_results:
            return
        self.current_center_shift_preview_index = min(max(int(index), 0), len(self.center_shift_results) - 1)
        self._update_center_shift_controls()
        self._show_selected_center_shift_preview()
        self._refresh_center_shift_metric_plot()

    def _step_center_shift_preview(self, step: int) -> None:
        if not self.center_shift_results:
            return
        current = self.center_preview_slider.value()
        maximum = max(0, len(self.center_shift_results) - 1)
        target = min(max(int(current) + int(step), 0), maximum)
        if target != current:
            self.center_preview_slider.setValue(target)

    def _show_selected_center_shift_preview(self) -> None:
        result = self._selected_center_shift_result()
        if result is None:
            return
        self._show_preview_image(
            result.preview_image,
            f"Center offset preview: {result.shift_px:.4f} px",
            preserve_zoom=True,
        )

    def _selected_center_shift_result(self) -> CenterShiftPreviewResult | None:
        if not self.center_shift_results:
            return None
        index = min(max(self.current_center_shift_preview_index, 0), len(self.center_shift_results) - 1)
        return self.center_shift_results[index]

    def _update_center_shift_controls(self) -> None:
        result = self._selected_center_shift_result()
        has_results = result is not None
        self.center_preview_slider.setEnabled(has_results)
        self.center_preview_prev_button.setEnabled(has_results and self.current_center_shift_preview_index > 0)
        self.center_preview_next_button.setEnabled(
            has_results and self.current_center_shift_preview_index < len(self.center_shift_results) - 1
        )
        self.apply_center_selected_button.setEnabled(has_results)
        self.run_center_fine_button.setEnabled(has_results)
        self.apply_center_auto_button.setEnabled(self.auto_best_shift_px is not None)
        if result is None:
            self.center_selected_label.setText("No center-offset previews")
        else:
            self.center_selected_label.setText(
                f"Preview {self.current_center_shift_preview_index + 1}/{len(self.center_shift_results)}    "
                f"Shift = {result.shift_px:.4f} px    Detector offset = {result.detector_offset_mm:.6g} mm"
            )

    def _update_auto_center_shift_recommendation(self, *_: object) -> None:
        if not self.auto_metric_table:
            self.auto_best_shift_px = None
            self.auto_best_preview_index = None
            self.center_auto_label.setText("Automatic recommendation unavailable")
            self._update_center_shift_controls()
            self._refresh_center_shift_metric_plot()
            return
        metric_key = str(self.center_search_metric.currentData() or "combined_score")
        best_index = recommend_metric_index(self.auto_metric_table, metric_key)
        if best_index is None:
            self.auto_best_shift_px = None
            self.auto_best_preview_index = None
            self.center_auto_label.setText("Automatic recommendation unavailable")
            self._update_center_shift_controls()
            self._refresh_center_shift_metric_plot()
            return
        row = self.auto_metric_table[best_index]
        self.auto_best_shift_px = float(row["shift_px"])
        self.auto_best_preview_index = best_index
        self.center_auto_label.setText(
            f"Automatic recommended shift: {self.auto_best_shift_px:.4f} px\n"
            f"Combined score: {row['combined_score']:.6g}    "
            f"{self.center_search_metric.currentText()}: {row[metric_key]:.6g}"
        )
        self._update_center_shift_controls()
        self._refresh_center_shift_metric_plot()

    def _refresh_center_shift_metric_plot(self) -> None:
        metric_key = str(self.center_search_metric.currentData() or "combined_score")
        selected = self._selected_center_shift_result()
        selected_shift = selected.shift_px if selected is not None else None
        self.metrics_plot.show_center_shift_metrics(
            self.auto_metric_table or [],
            metric_key,
            selected_shift=selected_shift,
            recommended_shift=self.auto_best_shift_px,
        )

    def _center_shift_values_from_ui(self) -> list[float] | None:
        try:
            shifts = generate_shift_values(
                self.center_search_start.value(),
                self.center_search_end.value(),
                self.center_search_step.value(),
            )
        except Exception as exc:
            self._error("Invalid center-offset search", str(exc))
            return None
        if len(shifts) > 101:
            answer = QMessageBox.question(
                self,
                "Large center-offset search",
                f"This search will reconstruct {len(shifts)} previews. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return None
        elif len(shifts) > 51:
            QMessageBox.warning(
                self,
                "Center-offset search",
                f"This search will reconstruct {len(shifts)} previews. A coarser step may be faster.",
            )
        return shifts

    def _clear_center_shift_results(self) -> None:
        self.center_shift_results = []
        self.current_center_shift_preview_index = 0
        self.auto_metric_table = None
        self.auto_best_shift_px = None
        self.auto_best_preview_index = None
        if hasattr(self, "center_preview_slider"):
            self.center_preview_slider.blockSignals(True)
            self.center_preview_slider.setRange(0, 0)
            self.center_preview_slider.setValue(0)
            self.center_preview_slider.blockSignals(False)
            self.center_auto_label.setText("Automatic recommendation unavailable")
            self._update_center_shift_controls()

    def _populate_validation_table(self, validation: MetadataValidation) -> None:
        rows = validation.table_rows
        self.validation_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, key in enumerate(
                ("index", "filename", "angle_deg", "angle_rad", "x_shift_px", "y_shift_px", "exists")
            ):
                self.validation_table.setItem(row_index, column_index, QTableWidgetItem(row[key]))
        self.validation_table.resizeColumnsToContents()

    def _browse_metadata_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose metadata folder")
        if not path:
            return
        self.metadata_path.setText(path)
        self.load_metadata()

    def _editable_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        return combo

    def _set_combo_items(self, combo: QComboBox, columns: list[str], optional_label: str | None = None) -> None:
        current = combo.currentText().strip()
        combo.clear()
        if optional_label is not None:
            combo.addItem(optional_label)
        combo.addItems(columns)
        if current:
            combo.setCurrentText(current)

    def _set_optional_combo_guess(self, combo: QComboBox, guess: str) -> None:
        if guess:
            combo.setCurrentText(guess)
        elif combo.count() > 0:
            combo.setCurrentIndex(0)

    def _combo_text(self, combo: QComboBox) -> str:
        return combo.currentText().strip()

    def _optional_combo_text(self, combo: QComboBox) -> str:
        text = combo.currentText().strip()
        return "" if text.startswith("No ") else text

    def _set_combo_by_data(self, combo: QComboBox, value: object) -> None:
        target = str(value)
        for index in range(combo.count()):
            if str(combo.itemData(index)) == target:
                combo.setCurrentIndex(index)
                return
        combo.setCurrentIndex(0)

    def _parse_string_list(self, text: str) -> list[str]:
        values = [part.strip() for part in str(text).split(",")]
        return [value for value in values if value]

    def _parse_float_list(self, text: str) -> list[float]:
        values: list[float] = []
        for part in str(text).split(","):
            clean = part.strip()
            if clean:
                values.append(float(clean))
        return values

    def _binning_tuple(self) -> tuple[int, int]:
        return self.binning_y.value(), self.binning_x.value()

    def _crop_tuple(self) -> tuple[int, int, int, int]:
        return self.crop_top.value(), self.crop_bottom.value(), self.crop_left.value(), self.crop_right.value()

    def _center_shift_sign_value(self) -> float:
        return 1.0 if self.center_sign.value() >= 0 else -1.0

    def _error(self, title: str, message: str) -> None:
        self._log(f"{title}: {message}")
        QMessageBox.critical(self, title, message)

    def _request_cancel(self) -> None:
        if self._thread is None:
            return
        self._cancel.cancel()
        self._log("Cancellation requested. Waiting for the current operation to reach a safe stopping point.")

    def _run(self, run_fdk: bool = False, run_mbir: bool = False, preprocess_only: bool = False) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, "Busy", "A reconstruction task is already running.")
            return
        cfg = self.config_from_ui()
        if run_mbir and not self._confirm_mbir_memory_budget(cfg):
            return
        self._cancel = CancellationToken()
        cancel_token = self._cancel
        self.progress_bar.setRange(0, 0)
        self._set_busy(True)
        self._set_device_monitor_runtime_logging(bool(cfg.gpu.use_gpu and (run_fdk or run_mbir)), cfg)
        if self._device_monitor_runtime_logging_enabled:
            self._poll_device_monitor()
        if run_mbir:
            self._prepare_live_mbir_monitor(
                cfg.mbir.max_admm_iterations,
                cfg.mbir.solver,
                tuple(int(value or 0) for value in cfg.geometry.volume_voxels),
            )

        def job(log: Callable[[str], None], mbir_progress: Callable[[object], None] | None = None):
            return execute_pipeline(
                cfg,
                run_fdk=run_fdk,
                run_mbir=run_mbir,
                preprocess_only=preprocess_only,
                logger=log,
                cancel_check=cancel_token.is_cancelled,
                mbir_progress_callback=mbir_progress if run_mbir else None,
                mbir_preview_request_callback=self._mbir_preview_request_snapshot if run_mbir else None,
            )

        self._start_worker(job, on_progress=self._mbir_progress_update if run_mbir else None)

    def _run_fast(
        self,
        run_all: bool = False,
        run_fdk_sweep: bool = False,
        make_prior: bool = False,
        run_mbir_lite: bool = False,
        run_qc: bool = False,
    ) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, "Busy", "A reconstruction task is already running.")
            return
        cfg = self.config_from_ui()
        cfg.fast_recon.enabled = True
        if cfg.fast_recon.resume_run_folder:
            self.metrics_plot.set_run_folder(cfg.fast_recon.resume_run_folder)
        if run_qc and not run_mbir_lite and not run_all:
            self.metrics_plot.select_category("QC report")
        elif make_prior and not run_mbir_lite and not run_all:
            self.metrics_plot.select_category("Make prior")
        elif run_fdk_sweep and not run_all:
            self.metrics_plot.select_category("FDK sweep")
        self._cancel = CancellationToken()
        cancel_token = self._cancel
        self.progress_bar.setRange(0, 0)
        self._set_busy(True)
        self._set_device_monitor_runtime_logging(bool(cfg.gpu.use_gpu and (run_all or run_fdk_sweep or run_mbir_lite)), cfg)
        wants_live_mbir = bool(run_all or run_mbir_lite)
        if wants_live_mbir:
            self._prepare_live_mbir_monitor(
                cfg.mbir_lite.n_sweeps,
                cfg.mbir_lite.solver,
                tuple(int(value or 0) for value in cfg.geometry.volume_voxels),
            )

        def job(log: Callable[[str], None], mbir_progress: Callable[[object], None] | None = None):
            def fast_progress(metric, volume) -> None:
                if mbir_progress is None:
                    return
                preview_request = self._mbir_preview_request_snapshot()
                preview_image, preview_view, preview_slice_index, preview_slice_counts = extract_preview_slice_for_gui(
                    volume,
                    preview_request,
                )
                mbir_progress(
                    {
                        "metrics": metric,
                        "preview_image": preview_image,
                        "preview_view": preview_view,
                        "preview_slice_index": preview_slice_index,
                        "preview_slice_counts": preview_slice_counts,
                        "solver": "anchored_streaming_subset_tv",
                        "max_iterations": int(cfg.mbir_lite.n_sweeps),
                    }
                )

            return execute_fast_pipeline(
                cfg,
                run_fdk_sweep=run_fdk_sweep,
                make_prior=make_prior,
                run_mbir_lite=run_mbir_lite,
                run_qc=run_qc,
                run_all=run_all,
                fast_run_folder=cfg.fast_recon.resume_run_folder,
                logger=log,
                cancel_check=cancel_token.is_cancelled,
                mbir_lite_progress_callback=fast_progress if wants_live_mbir else None,
            )

        self._start_worker(job, on_progress=self._mbir_progress_update if wants_live_mbir else None)

    def _estimate_mbir_lite_memory(self) -> None:
        cfg = self.config_from_ui()
        try:
            runtime = self._memory_estimate_runtime(cfg)
            lines = [
                *self._standard_mbir_memory_overview_lines(cfg, runtime),
                "",
                *self._mbir_lite_memory_overview_lines(cfg, runtime),
                "",
                *self._memory_estimate_machine_lines(cfg, runtime),
                "",
                "Notes",
                "-----",
                "Standard MBIR RAM is the full-run estimate for the current MBIR settings.",
                "MBIR-lite RAM uses the current anchor, subset, and batch settings.",
                "TIGRE GPU peak is the estimated peak for one Ax/Atb call, not the full host RAM working set.",
                "Run MBIR still performs a stricter preflight RAM/VRAM safety check before reconstruction starts.",
            ]
            self._show_memory_details_dialog(
                "Memory Estimate Overview",
                "This overview includes both standard MBIR and MBIR-lite using the current GUI settings.",
                "Use the Standard MBIR section for a full MBIR run and the MBIR-lite section for the fast anchor-guided workflow.",
                lines,
                critical=False,
                ask=False,
            )
        except Exception as exc:
            self._error("Memory estimate unavailable", str(exc))

    def _memory_estimate_runtime(self, cfg: AppConfig) -> MemoryEstimateRuntime:
        detected_gpu_infos = query_all_nvidia_gpu_memory() if cfg.gpu.use_gpu else []
        gpu_selection = resolve_gpu_selection(
            cfg.gpu.use_gpu,
            cfg.gpu.gpu_selector,
            cfg.gpu.gpu_id,
            [info.gpu_id for info in detected_gpu_infos],
        )
        selected_gpu_infos = [info for info in detected_gpu_infos if info.gpu_id in gpu_selection.gpu_ids]
        return MemoryEstimateRuntime(
            system_info=query_system_memory(),
            gpu_selection=gpu_selection,
            selected_gpu_infos=selected_gpu_infos,
            selected_gpu_count=max(1, len(gpu_selection.gpu_ids)) if cfg.gpu.use_gpu else 1,
            selected_gpu_min_free_gb=min((info.free_gb for info in selected_gpu_infos), default=None),
            selected_gpu_min_total_gb=min((info.total_gb for info in selected_gpu_infos), default=None),
            selected_gpu_total_free_gb=sum(info.free_gb for info in selected_gpu_infos),
            selected_gpu_total_total_gb=sum(info.total_gb for info in selected_gpu_infos),
        )

    def _standard_mbir_memory_overview_lines(
        self,
        cfg: AppConfig,
        runtime: MemoryEstimateRuntime,
    ) -> list[str]:
        validation, detector_shape_pre_tigre = self._main_validation_for_estimate(cfg)
        configured_volume_shape = (int(self.nz.value()), int(self.ny.value()), int(self.nx.value()))
        if min(configured_volume_shape) <= 0:
            raise ValueError("Set positive Nz, Ny, and Nx before estimating standard MBIR memory.")
        transpose_for_tigre = bool(cfg.preprocessing.transpose_for_tigre)
        base_detector_shape = (
            (detector_shape_pre_tigre[1], detector_shape_pre_tigre[0])
            if transpose_for_tigre
            else detector_shape_pre_tigre
        )
        projection_count = effective_projection_count(len(validation.records), cfg.mbir.projection_stride)
        detector_shape = effective_detector_shape(
            base_detector_shape,
            cfg.mbir.debug_pixel_binning,
            transpose_for_tigre=transpose_for_tigre,
        )
        volume_shape = effective_volume_shape(
            configured_volume_shape,
            cfg.mbir.debug_pixel_binning,
            transpose_for_tigre=transpose_for_tigre,
        )
        requested_memory_mode = str(cfg.mbir.memory_mode or "auto").strip().lower()
        streaming_mode = requested_memory_mode in {"auto", "projection_streaming", "ordered_subsets"}
        estimate = estimate_mbir_memory(
            projection_count,
            detector_shape,
            volume_shape,
            projection_batch_size=cfg.mbir.projection_batch_size if streaming_mode else None,
            gpu_count=runtime.selected_gpu_count,
        )
        low_memory_cpu_gb = estimate_low_memory_pdhg_cpu_gb(
            estimate.projection_gb,
            estimate.volume_gb,
            dual_dtype=cfg.mbir.pdhg_dual_dtype,
        )
        subset_tv_cpu_gb = estimate_streaming_subset_tv_cpu_gb(estimate.projection_gb, estimate.volume_gb)
        effective_solver = self._effective_auto_mbir_solver(
            cfg,
            estimate,
            low_memory_cpu_gb,
            subset_tv_cpu_gb,
            runtime.system_info,
        )
        batch_text = (
            f"Projection-streaming GPU estimate uses batch size {max(1, int(cfg.mbir.projection_batch_size))}"
            if streaming_mode
            else f"GPU estimate uses the full per-call projection stack for memory mode {cfg.mbir.memory_mode}"
        )
        lines = [
            "Standard MBIR",
            "-------------",
            f"Projection views: {projection_count}",
            f"Detector rows x cols: {detector_shape[0]} x {detector_shape[1]}",
            f"Volume voxels z/y/x: {volume_shape[0]} x {volume_shape[1]} x {volume_shape[2]}",
            f"Resident data: projection stack {estimate.projection_gb:.2f} GB | single volume {estimate.volume_gb:.2f} GB",
            f"Estimated system RAM: ADMM/CG {estimate.cpu_working_set_gb:.2f} GB | PDHG {low_memory_cpu_gb:.2f} GB | subset-TV {subset_tv_cpu_gb:.2f} GB",
            f"Estimated TIGRE GPU peak: {estimate.gpu_operator_estimate_gb:.2f} GB",
            f"Solver setting: {cfg.mbir.solver} | auto would likely use {_solver_display_name(effective_solver)}",
            batch_text,
        ]
        debug_binning = tuple(int(value) for value in cfg.mbir.debug_pixel_binning)
        if debug_binning != (1, 1) or int(cfg.mbir.projection_stride) > 1:
            lines.append(
                f"Debug estimate modifiers included: projection stride {int(cfg.mbir.projection_stride)}, binning {debug_binning[0]} x {debug_binning[1]}"
            )
        return lines

    def _mbir_lite_memory_overview_lines(
        self,
        cfg: AppConfig,
        runtime: MemoryEstimateRuntime,
    ) -> list[str]:
        validation, detector_shape_pre_tigre = self._main_validation_for_estimate(cfg)
        configured_volume_shape = (int(self.nz.value()), int(self.ny.value()), int(self.nx.value()))
        if min(configured_volume_shape) <= 0:
            raise ValueError("Set positive Nz, Ny, and Nx before estimating MBIR-lite memory.")
        transpose_for_tigre = bool(cfg.preprocessing.transpose_for_tigre)
        detector_shape = (
            (detector_shape_pre_tigre[1], detector_shape_pre_tigre[0])
            if transpose_for_tigre
            else detector_shape_pre_tigre
        )
        volume_shape = configured_volume_shape
        main_views = len(validation.records)
        anchor_total, anchor_final, anchor_lines = self._estimate_anchor_views_for_mbir_lite(cfg)
        total_views = max(1, int(main_views) + int(anchor_final))
        subset_count = max(1, int(cfg.mbir_lite.ordered_subset_count))
        batch_size = max(1, int(cfg.mbir_lite.projection_batch_size))
        active_subset_views = max(1, int(math.ceil(total_views / float(subset_count))))
        batches_per_sweep = max(1, int(math.ceil(active_subset_views / float(batch_size))))
        full_estimate = estimate_mbir_memory(
            total_views,
            detector_shape,
            volume_shape,
            gpu_count=runtime.selected_gpu_count,
        )
        subset_estimate = estimate_mbir_memory(
            active_subset_views,
            detector_shape,
            volume_shape,
            projection_batch_size=batch_size,
            gpu_count=runtime.selected_gpu_count,
        )
        active_subset_projection_gb = estimate_mbir_memory(
            active_subset_views,
            detector_shape,
            volume_shape,
            gpu_count=runtime.selected_gpu_count,
        ).projection_gb
        mbir_lite_ram_gb = 4.85 * full_estimate.volume_gb + 1.25 * full_estimate.projection_gb
        min_available_ram_gb = max(4.0, 2.0 * full_estimate.volume_gb + full_estimate.projection_gb)
        effective_gpu_views = _estimated_tigre_effective_views(active_subset_views, batch_size)
        lines = [
            "MBIR-lite",
            "---------",
            f"Main views: {main_views} | anchor views in metadata: {anchor_total} | anchor views used: {anchor_final}",
            *anchor_lines,
            f"Total MBIR-lite views: {total_views}",
            f"Detector rows x cols: {detector_shape[0]} x {detector_shape[1]}",
            f"Volume voxels z/y/x: {volume_shape[0]} x {volume_shape[1]} x {volume_shape[2]}",
            f"Resident data: full projection stack {full_estimate.projection_gb:.2f} GB | single volume {full_estimate.volume_gb:.2f} GB",
            f"Subsets {subset_count} | batch size {batch_size} | active views/sweep about {active_subset_views} | batches/sweep about {batches_per_sweep}",
            f"Active-subset projection stack: {active_subset_projection_gb:.2f} GB | estimated TIGRE views at peak: about {effective_gpu_views}",
            f"Estimated system RAM: {mbir_lite_ram_gb:.2f} GB",
            f"Minimum RAM before TIGRE calls: {min_available_ram_gb:.2f} GB",
            f"Estimated TIGRE GPU peak: {subset_estimate.gpu_operator_estimate_gb:.2f} GB",
        ]
        return lines

    def _memory_estimate_machine_lines(
        self,
        cfg: AppConfig,
        runtime: MemoryEstimateRuntime,
    ) -> list[str]:
        lines = [
            "Current Machine",
            "---------------",
            f"GPU selection: {runtime.gpu_selection.describe()} (selector='{runtime.gpu_selection.selector}')",
        ]
        if runtime.system_info is not None:
            lines.append(
                f"System RAM available/total: {runtime.system_info.available_gb:.2f} / {runtime.system_info.total_gb:.2f} GB"
            )
        else:
            lines.append("System RAM available/total: unavailable")
        if runtime.selected_gpu_infos:
            lines.extend(
                [
                    f"Selected GPU count: {len(runtime.selected_gpu_infos)}",
                    "Selected GPUs: " + ", ".join(f"{info.gpu_id}: {info.name}" for info in runtime.selected_gpu_infos),
                    f"Aggregate selected GPU VRAM free/total: {runtime.selected_gpu_total_free_gb:.2f} / {runtime.selected_gpu_total_total_gb:.2f} GB",
                    f"Minimum per-GPU VRAM free/total: {runtime.selected_gpu_min_free_gb:.2f} / {runtime.selected_gpu_min_total_gb:.2f} GB",
                ]
            )
        elif cfg.gpu.use_gpu:
            lines.append("Selected GPU VRAM: unavailable from nvidia-smi")
        else:
            lines.append("GPU acceleration is disabled; VRAM numbers are informational only")
        lines.extend(str(warning) for warning in runtime.gpu_selection.warnings)
        return lines

    def _effective_auto_mbir_solver(
        self,
        cfg: AppConfig,
        estimate,
        low_memory_cpu_gb: float,
        subset_tv_cpu_gb: float,
        system_info,
    ) -> str:
        effective_solver = _normalize_solver_name(cfg.mbir.solver)
        if effective_solver != "auto":
            return effective_solver
        effective_solver = "admm"
        if system_info is not None and estimate.cpu_working_set_gb > SYSTEM_RAM_SAFETY_FRACTION * system_info.total_gb:
            effective_solver = "pdhg_low_memory"
        elif system_info is not None and system_info.available_gb < minimum_available_system_memory_gb(estimate):
            effective_solver = "pdhg_low_memory"
        if system_info is not None and effective_solver == "pdhg_low_memory" and low_memory_cpu_gb > 0.65 * system_info.total_gb:
            effective_solver = "streaming_subset_tv"
        return effective_solver

    def _main_validation_for_estimate(self, cfg: AppConfig) -> tuple[MetadataValidation, tuple[int, int]]:
        if self.metadata_frame is None:
            self.load_metadata()
            if self.metadata_frame is None:
                raise ValueError("Load metadata before estimating memory.")
        validation = validate_metadata(
            self.metadata_frame,
            self.projection_folder.text(),
            self._combo_text(self.filename_column),
            self._combo_text(self.angle_column),
            self._optional_combo_text(self.angle_rad_column),
            self._optional_combo_text(self.x_shift_column),
            self._optional_combo_text(self.y_shift_column),
            remove_duplicate_endpoint=self.remove_endpoint.isChecked(),
            reverse_angle_order=self.reverse_angle_order.isChecked(),
        )
        if validation.duplicate_filenames:
            raise ValueError(f"Duplicate projection filenames in metadata: {validation.duplicate_filenames[:10]}")
        if validation.missing_files:
            raise FileNotFoundError(f"Missing projection files: {validation.missing_files[:10]}")
        if not validation.records:
            raise ValueError("No projection records are available.")
        first_binned = crop_2d(read_tiff_float32(validation.records[0].path, binning=self._binning_tuple()), self._crop_tuple())
        rows, cols = (int(first_binned.shape[0]), int(first_binned.shape[1]))
        self.validation = validation
        self._populate_validation_table(validation)
        self.detector_rows.setValue(rows)
        self.detector_cols.setValue(cols)
        if self.nx.value() == 0:
            self.nx.setValue(cols)
        if self.ny.value() == 0:
            self.ny.setValue(cols)
        if self.nz.value() == 0:
            self.nz.setValue(rows)
        return validation, (rows, cols)

    def _estimate_anchor_views_for_mbir_lite(self, cfg: AppConfig) -> tuple[int, int, list[str]]:
        if not cfg.anchors.enabled:
            return 0, 0, ["Anchor dataset disabled: MBIR-lite estimate uses main projections only."]
        if not cfg.anchors.input.projection_folder or not cfg.anchors.input.metadata_path:
            raise ValueError("Anchor projection folder and anchor metadata CSV are required for an MBIR-lite memory estimate.")
        frame = load_metadata_csv(find_metadata_csv(cfg.anchors.input.metadata_path))
        columns = [str(column) for column in frame.columns]
        validation = validate_metadata(
            frame,
            cfg.anchors.input.projection_folder,
            cfg.anchors.metadata.filename_column or suggest_column(columns, FILENAME_HINTS),
            cfg.anchors.metadata.angle_column or suggest_column(columns, ANGLE_HINTS),
            cfg.anchors.metadata.angle_rad_column or suggest_optional_column(columns, ANGLE_RAD_HINTS),
            cfg.anchors.metadata.x_shift_column or suggest_optional_column(columns, X_SHIFT_HINTS),
            cfg.anchors.metadata.y_shift_column or suggest_optional_column(columns, Y_SHIFT_HINTS),
            remove_duplicate_endpoint=cfg.anchors.metadata.remove_duplicate_endpoint,
            reverse_angle_order=cfg.anchors.metadata.reverse_angle_order,
        )
        if validation.duplicate_filenames:
            raise ValueError(f"Duplicate anchor projection filenames in metadata: {validation.duplicate_filenames[:10]}")
        if validation.missing_files:
            raise FileNotFoundError(f"Missing anchor projection files: {validation.missing_files[:10]}")
        total = len(validation.records)
        tune = min(max(0, int(cfg.anchors.split.tune_count)), total)
        qc = min(max(0, int(cfg.anchors.split.qc_count)), max(0, total - tune))
        remaining = max(0, total - tune - qc)
        recon = remaining if cfg.anchors.split.recon_count is None else min(max(0, int(cfg.anchors.split.recon_count)), remaining)
        final = recon
        if cfg.anchors.split.use_tune_anchors_in_final:
            final += tune
        if cfg.anchors.split.use_qc_anchors_in_final:
            final += qc
        return total, final, [
            f"Anchor split estimate: recon={recon}, tune={tune}, qc={qc}.",
            f"Use tune anchors in MBIR-lite: {bool(cfg.anchors.split.use_tune_anchors_in_final)}",
            f"Use QC anchors in MBIR-lite: {bool(cfg.anchors.split.use_qc_anchors_in_final)}",
        ]

    def _confirm_mbir_memory_budget(self, cfg: AppConfig) -> bool:
        if self.validation is None:
            self.validate_metadata()
            if self.validation is None:
                return False
        detector_rows = int(self.detector_rows.value())
        detector_cols = int(self.detector_cols.value())
        if detector_rows <= 0 or detector_cols <= 0:
            QMessageBox.warning(self, "Memory estimate unavailable", "Validate metadata/geometry before running MBIR.")
            return False
        configured_volume_shape = (int(self.nz.value()), int(self.ny.value()), int(self.nx.value()))
        if min(configured_volume_shape) <= 0:
            QMessageBox.warning(self, "Memory estimate unavailable", "Set positive Nz, Ny, and Nx before running MBIR.")
            return False
        transpose_for_tigre = bool(cfg.preprocessing.transpose_for_tigre)
        base_detector_shape = (detector_cols, detector_rows) if transpose_for_tigre else (detector_rows, detector_cols)
        projection_count = effective_projection_count(len(self.validation.records), cfg.mbir.projection_stride)
        detector_shape = effective_detector_shape(
            base_detector_shape,
            cfg.mbir.debug_pixel_binning,
            transpose_for_tigre=transpose_for_tigre,
        )
        volume_shape = effective_volume_shape(
            configured_volume_shape,
            cfg.mbir.debug_pixel_binning,
            transpose_for_tigre=transpose_for_tigre,
        )
        requested_memory_mode = str(cfg.mbir.memory_mode or "auto").strip().lower()
        streaming_mode = requested_memory_mode in {"auto", "projection_streaming", "ordered_subsets"}
        detected_gpu_infos = query_all_nvidia_gpu_memory() if cfg.gpu.use_gpu else []
        gpu_selection = resolve_gpu_selection(
            cfg.gpu.use_gpu,
            cfg.gpu.gpu_selector,
            cfg.gpu.gpu_id,
            [info.gpu_id for info in detected_gpu_infos],
        )
        selected_gpu_infos = [info for info in detected_gpu_infos if info.gpu_id in gpu_selection.gpu_ids]
        selected_gpu_count = max(1, len(gpu_selection.gpu_ids)) if cfg.gpu.use_gpu else 1
        selected_gpu_min_free_gb = min((info.free_gb for info in selected_gpu_infos), default=None)
        selected_gpu_min_total_gb = min((info.total_gb for info in selected_gpu_infos), default=None)
        selected_gpu_total_free_gb = sum(info.free_gb for info in selected_gpu_infos)
        selected_gpu_total_total_gb = sum(info.total_gb for info in selected_gpu_infos)
        estimate = estimate_mbir_memory(
            projection_count,
            detector_shape,
            volume_shape,
            projection_batch_size=cfg.mbir.projection_batch_size if streaming_mode else None,
            gpu_count=selected_gpu_count,
        )
        fdk_init_estimate = estimate_mbir_memory(
            projection_count,
            detector_shape,
            volume_shape,
            gpu_count=selected_gpu_count,
        )
        system_info = query_system_memory()
        minimum_available_ram_gb = minimum_available_system_memory_gb(estimate)
        low_memory_cpu_gb = estimate_low_memory_pdhg_cpu_gb(
            estimate.projection_gb,
            estimate.volume_gb,
            dual_dtype=cfg.mbir.pdhg_dual_dtype,
        )
        subset_tv_cpu_gb = estimate_streaming_subset_tv_cpu_gb(estimate.projection_gb, estimate.volume_gb)
        distributed_overhead = estimate_distributed_full_data_overhead_gb(
            estimate.projection_gb,
            estimate.volume_gb,
            selected_gpu_count,
        )
        requested_solver = _normalize_solver_name(cfg.mbir.solver)
        effective_solver = requested_solver
        if effective_solver == "auto":
            effective_solver = "admm"
            if system_info is not None and estimate.cpu_working_set_gb > SYSTEM_RAM_SAFETY_FRACTION * system_info.total_gb:
                effective_solver = "pdhg_low_memory"
            elif system_info is not None and system_info.available_gb < minimum_available_ram_gb:
                effective_solver = "pdhg_low_memory"
            if system_info is not None and effective_solver == "pdhg_low_memory" and low_memory_cpu_gb > 0.65 * system_info.total_gb:
                effective_solver = "streaming_subset_tv"
        operator_memory_mode = cfg.mbir.memory_mode
        if effective_solver in {"pdhg_low_memory", "streaming_subset_tv"} and cfg.mbir.memory_mode == "ordered_subsets":
            operator_memory_mode = "projection_streaming"
        distributed_mode_notes: list[str] = []
        if requested_memory_mode == "distributed_full_data":
            distributed_mode_notes.extend(
                [
                    f"Estimated distributed full-data shared CPU buffers: {distributed_overhead.shared_buffers_gb:.2f} GB",
                    f"Estimated distributed full-data transient CPU peak: {distributed_overhead.transient_peak_gb:.2f} GB",
                ]
            )
            if not cfg.gpu.use_gpu or len(gpu_selection.gpu_ids) <= 1:
                operator_memory_mode = "full_gpu"
                distributed_mode_notes.append(
                    "Distributed full-data mode requested, but fewer than two GPUs are selected. The run will fall back to full_gpu."
                )
            elif effective_solver == "streaming_subset_tv":
                operator_memory_mode = "projection_streaming"
                distributed_mode_notes.append(
                    "streaming_subset_tv uses projection_streaming internally; distributed_full_data is reserved for exact full-data Ax/A^T calls."
                )
            elif system_info is not None:
                solver_cpu_gb = (
                    estimate.cpu_working_set_gb
                    if effective_solver == "admm"
                    else low_memory_cpu_gb
                    if effective_solver == "pdhg_low_memory"
                    else subset_tv_cpu_gb
                )
                if solver_cpu_gb + distributed_overhead.shared_buffers_gb > SYSTEM_RAM_SAFETY_FRACTION * system_info.total_gb:
                    operator_memory_mode = "full_gpu"
                    distributed_mode_notes.append(
                        "Distributed full-data mode would exceed the safe physical RAM budget with its additional "
                        "shared CPU buffers, so the run will fall back to full_gpu."
                    )
                elif system_info.available_gb < minimum_available_ram_gb + distributed_overhead.shared_buffers_gb:
                    operator_memory_mode = "full_gpu"
                    distributed_mode_notes.append(
                        "Currently available system RAM is too low for distributed full-data shared buffers, so the "
                        "run will fall back to full_gpu."
                    )
        lines = estimate.report_lines()
        debug_binning = tuple(int(value) for value in cfg.mbir.debug_pixel_binning)
        debug_active = debug_binning != (1, 1) or int(cfg.mbir.projection_stride) > 1
        if debug_active:
            lines.extend(
                [
                    "",
                    *mbir_debug_report_lines(
                        len(self.validation.records),
                        base_detector_shape,
                        configured_volume_shape,
                        cfg.mbir.debug_pixel_binning,
                        cfg.mbir.projection_stride,
                        transpose_for_tigre=transpose_for_tigre,
                    ),
                    "Existing FDK .npy initialization will be resized to the effective MBIR volume grid if needed.",
                ]
            )
        critical = False
        risky = estimate.volume_gb >= 2.0 or estimate.projection_gb >= 1.0 or estimate.cpu_working_set_gb >= 32.0
        lines.extend(
            [
                "",
                f"Requested MBIR solver: {cfg.mbir.solver}",
                f"Effective MBIR solver: {effective_solver}",
                f"Effective operator memory mode: {operator_memory_mode}",
                f"Resolved GPU selection: {gpu_selection.describe()} (selector='{gpu_selection.selector}')",
                f"Estimated low-memory PDHG CPU working arrays: {low_memory_cpu_gb:.2f} GB",
                f"Estimated streaming subset-TV CPU working arrays: {subset_tv_cpu_gb:.2f} GB",
            ]
        )
        lines.extend(gpu_selection.warnings)
        lines.extend(distributed_mode_notes)
        if requested_solver == "auto" and effective_solver == "pdhg_low_memory":
            risky = True
            lines.append(
                "Auto solver will use low-memory PDHG for this run because ADMM/CG is too large for the RAM budget."
            )
        if requested_solver == "auto" and effective_solver == "streaming_subset_tv":
            risky = True
            lines.append(
                "Auto solver will use streaming subset-TV because ADMM/CG and PDHG are too large for the RAM budget."
            )
        if system_info is not None:
            lines.extend(
                [
                    "",
                    f"System RAM available/total: {system_info.available_gb:.2f} / {system_info.total_gb:.2f} GB",
                    f"Minimum available RAM before TIGRE Atb: {minimum_available_ram_gb:.2f} GB",
                ]
            )
            if effective_solver == "admm" and estimate.cpu_working_set_gb > SYSTEM_RAM_SAFETY_FRACTION * system_info.total_gb:
                critical = True
                risky = True
                lines.append("")
                lines.append(
                    "Critical: estimated ADMM/CG CPU working arrays exceed the safe physical RAM budget. "
                    "Use solver=auto or pdhg_low_memory to avoid the ADMM/CG volume pile-up."
                )
            elif estimate.cpu_working_set_gb > 0.70 * system_info.total_gb:
                risky = True
                lines.append("")
                lines.append(
                    "Warning: estimated MBIR CPU working arrays will use most physical RAM. "
                    "This can force paging and make TIGRE/CUDA allocations fail."
                )
            if effective_solver == "admm" and system_info.available_gb < minimum_available_ram_gb:
                critical = True
                risky = True
                lines.append("")
                lines.append(
                    "Critical: currently available system RAM is too low for TIGRE Atb's transient host allocation "
                    "and page-locked buffers. This can produce cudaMalloc out-of-memory even when VRAM is not full."
                )
            if effective_solver == "pdhg_low_memory" and low_memory_cpu_gb > SYSTEM_RAM_SAFETY_FRACTION * system_info.total_gb:
                critical = True
                risky = True
                lines.append("")
                lines.append(
                    "Critical: even low-memory PDHG is estimated to exceed the safe physical RAM budget. "
                    "Use fewer volume voxels, crop the ROI, or keep PDHG TV dual dtype at float16."
                )
            if effective_solver == "streaming_subset_tv" and subset_tv_cpu_gb > SYSTEM_RAM_SAFETY_FRACTION * system_info.total_gb:
                critical = True
                risky = True
                lines.append("")
                lines.append(
                    "Critical: even streaming subset-TV is estimated to exceed the safe physical RAM budget. "
                    "Use fewer volume voxels, crop the ROI, or close memory-heavy applications."
                )
        if (
            streaming_mode
            and projection_count > 64
            and int(cfg.mbir.projection_batch_size) >= projection_count
        ):
            lines.append("")
            lines.append(
                "Projection streaming batch size will be capped at 64 before GPU-memory tuning so the run does not "
                "silently become a full-stack TIGRE call."
            )
        if selected_gpu_infos:
            lines.extend(
                [
                    "",
                    f"Selected GPU count: {len(selected_gpu_infos)}",
                    "Selected GPUs: " + ", ".join(f"{info.gpu_id}: {info.name}" for info in selected_gpu_infos),
                    f"Aggregate selected GPU memory free/total: {selected_gpu_total_free_gb:.2f} / {selected_gpu_total_total_gb:.2f} GB",
                    f"Minimum per-GPU free/total across selection: {selected_gpu_min_free_gb:.2f} / {selected_gpu_min_total_gb:.2f} GB",
                ]
            )
            if selected_gpu_min_free_gb is not None and estimate.gpu_operator_estimate_gb > 0.85 * selected_gpu_min_free_gb:
                risky = True
                lines.append("")
                lines.append("Warning: estimated TIGRE GPU working set is close to or above available free GPU memory.")
            if streaming_mode:
                one_view_estimate = estimate_mbir_memory(
                    projection_count,
                    detector_shape,
                    volume_shape,
                    projection_batch_size=1,
                    gpu_count=selected_gpu_count,
                )
                if selected_gpu_min_total_gb is not None and one_view_estimate.gpu_operator_estimate_gb > 0.98 * selected_gpu_min_total_gb:
                    risky = True
                    lines.append("")
                    lines.append(
                        "Critical: even one projection per TIGRE batch is estimated to exceed this GPU's total memory. "
                        "The pipeline will stop before calling TIGRE instead of letting CUDA crash."
                    )
                elif selected_gpu_min_free_gb is not None and one_view_estimate.gpu_operator_estimate_gb > 0.90 * selected_gpu_min_free_gb:
                    risky = True
                    lines.append("")
                    lines.append(
                        "Warning: even one projection per TIGRE batch is close to currently free GPU memory. "
                        "Close other GPU applications if TIGRE reports cudaMalloc failures."
                    )
                recommended_batch = recommended_projection_batch_size(
                    projection_count,
                    detector_shape,
                    volume_shape,
                    selected_gpu_min_free_gb if selected_gpu_min_free_gb is not None else 0.0,
                    gpu_count=selected_gpu_count,
                )
                if recommended_batch < int(cfg.mbir.projection_batch_size):
                    lines.append("")
                    lines.append(
                        "Projection streaming batch size will be reduced automatically for this run: "
                        f"{cfg.mbir.projection_batch_size} -> {recommended_batch}."
                    )
                elif int(cfg.mbir.projection_batch_size) == 1 and recommended_batch > 1:
                    lines.append("")
                    lines.append(
                        "Projection batch size 1 is memory-safe but can underutilize the GPU. "
                        f"This estimate suggests up to about {recommended_batch} views per batch may fit."
                    )
            if (
                cfg.initialization.mode.lower() == "fdk"
                and selected_gpu_min_free_gb is not None
                and fdk_init_estimate.gpu_operator_estimate_gb > 0.85 * selected_gpu_min_free_gb
            ):
                risky = True
                lines.append("")
                lines.append(
                    "Warning: FDK initialization requires a monolithic TIGRE allocation. During MBIR streaming runs, "
                    "the pipeline will skip implicit FDK initialization and use zero initialization instead. "
                    "Use existing_fdk if you already have a saved .npy initialization."
                )
        elif cfg.gpu.use_gpu:
            lines.extend(["", "Selected GPU memory could not be queried with nvidia-smi."])
        if not risky:
            return True
        if critical:
            self._show_memory_details_dialog(
                "MBIR memory request cannot run safely",
                "Run blocked before calling TIGRE.",
                (
                    "The selected solver is still estimated to exceed safe RAM limits. "
                    "Use Solver=auto or streaming_subset_tv, reduce voxels, crop the ROI, or close memory-heavy apps."
                ),
                lines,
                critical=True,
                ask=False,
            )
            return False
        lines.extend(
            [
                "",
                "Recommended first fixes:",
                "- Keep Solver on auto so large runs use low-memory PDHG instead of ADMM/CG.",
                "- Use streaming_subset_tv for the smallest RAM footprint.",
                "- Use projection_streaming or ordered_subsets memory mode when exact distributed full-data is too large.",
                "- Use distributed_full_data when you want exact full-data MBIR with better multi-GPU occupancy and have enough host RAM.",
                "- Use the largest projection batch that fits GPU memory; batch size 1 is safest but usually slow.",
                "- Reduce volume voxels, for example 512 x 512 x 512 before trying 1024 cubed.",
                "- Use fewer projections or a cropped ROI for a debug MBIR run.",
                "- Use an existing FDK .npy file when FDK initialization is too large to run inside MBIR.",
                "",
                "Continue anyway?",
            ]
        )
        return self._show_memory_details_dialog(
            "Large MBIR memory request",
            f"Memory estimate: ADMM {estimate.cpu_working_set_gb:.1f} GB, PDHG {low_memory_cpu_gb:.1f} GB, subset-TV {subset_tv_cpu_gb:.1f} GB.",
            f"Effective solver: {effective_solver}. Continue anyway?",
            lines,
            critical=False,
            ask=True,
        )

    def _show_memory_details_dialog(
        self,
        title: str,
        summary: str,
        guidance: str,
        details: list[str],
        critical: bool = False,
        ask: bool = False,
    ) -> bool:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(720, 520)
        layout = QVBoxLayout(dialog)
        summary_label = QLabel(summary)
        summary_label.setWordWrap(True)
        guidance_label = QLabel(guidance)
        guidance_label.setWordWrap(True)
        details_box = QTextEdit()
        details_box.setReadOnly(True)
        details_box.setPlainText("\n".join(details))
        details_box.setMinimumHeight(260)
        layout.addWidget(summary_label)
        layout.addWidget(guidance_label)
        layout.addWidget(details_box, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        if ask:
            yes = QPushButton("Continue")
            no = QPushButton("Cancel")
            yes.clicked.connect(dialog.accept)
            no.clicked.connect(dialog.reject)
            row.addWidget(yes)
            row.addWidget(no)
        else:
            ok = QPushButton("OK")
            ok.clicked.connect(dialog.accept)
            row.addWidget(ok)
        layout.addLayout(row)
        result = dialog.exec() if hasattr(dialog, "exec") else dialog.exec_()
        return bool(ask and result == QDialog.Accepted)

    def _start_worker(
        self,
        fn: Callable[..., Any],
        on_success: Callable[[Any], None] | None = None,
        on_progress: Callable[[Any], None] | None = None,
    ) -> None:
        thread = QThread(self)
        worker = Worker(fn, expects_progress=on_progress is not None)
        success = on_success or self._worker_success
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._worker_log)
        if on_progress is not None:
            worker.progress.connect(on_progress)
        worker.finished.connect(success)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.cancelled.connect(self._worker_cancelled)
        worker.cancelled.connect(thread.quit)
        worker.cancelled.connect(worker.deleteLater)
        worker.failed.connect(self._worker_failed)
        worker.failed.connect(thread.quit)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._worker_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _prepare_live_mbir_monitor(
        self,
        max_iterations: int,
        solver: str = "auto",
        volume_shape_hint: tuple[int, int, int] | None = None,
    ) -> None:
        self._live_mbir_metrics = []
        self._mbir_max_iterations = max(1, int(max_iterations))
        self._mbir_preview_started = False
        self._mbir_preview_volume_source = None
        solver_label = _solver_display_name(solver)
        if volume_shape_hint is not None and min(int(v) for v in volume_shape_hint) > 0:
            self.mbir_preview.set_volume_shape_hint(volume_shape_hint)
        self.metrics_plot.select_category("MBIR-lite")
        self.metrics_plot.show_metrics([])
        self.progress_bar.setRange(0, self._mbir_max_iterations)
        self.progress_bar.setValue(0)
        self.mbir_iteration_label.setText(f"Waiting for {solver_label} iteration 1/{self._mbir_max_iterations}")
        self.mbir_cg_label.setText(f"{solver_label} not started")
        self.mbir_time_label.setText("Elapsed 00:00:00 | ETA unavailable")
        self.mbir_objective_label.setText("Objective unavailable")
        self.mbir_data_residual_label.setText("Data residual unavailable")
        self.mbir_relative_change_label.setText("Relative change unavailable")
        self.mbir_preview.show_message("Waiting for first MBIR preview snapshot")
        self.tabs.setCurrentWidget(self.mbir_preview)
        self._active_image_changed()

    def _mbir_progress_update(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("kind") == "cg_progress":
            self._mbir_cg_progress_update(payload)
            return
        metric = payload.get("metrics")
        if metric is None:
            return
        self._live_mbir_metrics.append(metric)
        self.metrics_plot.show_metrics(self._live_mbir_metrics)
        max_iterations = int(payload.get("max_iterations") or self._mbir_max_iterations or int(metric.iteration))
        self._mbir_max_iterations = max(1, max_iterations)
        self.progress_bar.setRange(0, self._mbir_max_iterations)
        self.progress_bar.setValue(min(int(metric.iteration), self.progress_bar.maximum()))

        data_residual = float(np.sqrt(max(2.0 * float(metric.data_fidelity), 0.0)))
        elapsed = float(metric.elapsed_s)
        eta_text = "ETA unavailable"
        if metric.iteration > 0 and self._mbir_max_iterations > 0:
            seconds_per_iteration = elapsed / max(int(metric.iteration), 1)
            remaining = max(self._mbir_max_iterations - int(metric.iteration), 0) * seconds_per_iteration
            eta_text = f"ETA {_format_seconds(remaining)}"

        solver = str(payload.get("solver") or "mbir")
        solver_label = _solver_display_name(solver)
        self.mbir_iteration_label.setText(f"{solver_label} {int(metric.iteration)} / {self._mbir_max_iterations}")
        self.mbir_cg_label.setText(_solver_detail_text(solver, metric))
        self.mbir_time_label.setText(f"Elapsed {_format_seconds(elapsed)} | {eta_text}")
        self.mbir_objective_label.setText(f"{metric.objective:.6g}")
        self.mbir_data_residual_label.setText(f"{data_residual:.6g}")
        self.mbir_relative_change_label.setText(f"{metric.relative_x_change:.3e}")

        preview_image = payload.get("preview_image")
        preview_view = str(payload.get("preview_view") or "Axial")
        preview_slice_index = int(payload.get("preview_slice_index") or 0)
        preview_slice_counts = payload.get("preview_slice_counts")
        requested = self._mbir_preview_request_snapshot()
        if (
            preview_image is not None
            and preview_view == str(requested.get("view") or "Axial")
            and preview_slice_index == int(requested.get("slice_index") or 0)
        ):
            slice_counts = (
                {key: int(value) for key, value in dict(preview_slice_counts).items()}
                if isinstance(preview_slice_counts, dict)
                else preview_slice_counts_for_shape((1, 1, 1))
            )
            self.mbir_preview.show_slice(
                np.asarray(preview_image, dtype=np.float32),
                f"MBIR iteration {int(metric.iteration)} | live MBIR slice",
                preview_view,
                preview_slice_index,
                slice_counts=slice_counts,
            )
            if not self._mbir_preview_started:
                self.tabs.setCurrentWidget(self.mbir_preview)
                self._mbir_preview_started = True
            elif self.tabs.currentWidget() is self.mbir_preview:
                self._active_image_changed()
        elif self._thread is not None:
            self.mbir_preview.mark_request_pending(
                str(requested.get("view") or "Axial"),
                int(requested.get("slice_index") or 0),
            )

    def _mbir_cg_progress_update(self, payload: dict[str, object]) -> None:
        solver = str(payload.get("solver") or "admm")
        solver_label = _solver_display_name(solver)
        admm_iteration = max(1, int(payload.get("admm_iteration") or 1))
        max_iterations = int(payload.get("max_admm_iterations") or self._mbir_max_iterations or admm_iteration)
        self._mbir_max_iterations = max(1, max_iterations)
        self.progress_bar.setRange(0, self._mbir_max_iterations)
        self.progress_bar.setValue(min(max(admm_iteration - 1, 0), self.progress_bar.maximum()))
        cg_iteration = max(0, int(payload.get("cg_iteration") or 0))
        cg_max_iterations = max(1, int(payload.get("cg_max_iterations") or 1))
        cg_residual = float(payload.get("cg_residual") or 0.0)
        elapsed = float(payload.get("elapsed_s") or 0.0)
        self.mbir_iteration_label.setText(f"{solver_label} {admm_iteration} / {self._mbir_max_iterations}")
        self.mbir_cg_label.setText(f"CG {cg_iteration}/{cg_max_iterations} in progress | residual {cg_residual:.3e}")
        self.mbir_time_label.setText(f"Elapsed {_format_seconds(elapsed)} | inner CG in progress")

    def _worker_success(self, result) -> None:
        self._last_result = result
        fast_mbir_result = getattr(result, "mbir_lite_result", None)
        fast_fdk_result = getattr(result, "fdk_sweep_result", None)
        fast_prior_result = getattr(result, "prior_result", None)
        fast_qc_report = getattr(result, "qc_report_path", None)
        standard_mbir_result = getattr(result, "mbir_result", None)
        standard_fdk_volume = getattr(result, "fdk_volume", None)
        preferred_tomogram_id = (
            "mbir_lite"
            if fast_mbir_result is not None
            else "fast_fdk"
            if fast_fdk_result is not None
            else "fast_prior"
            if fast_prior_result is not None
            else "mbir"
            if standard_mbir_result is not None
            else "fdk"
            if standard_fdk_volume is not None
            else None
        )
        self.metrics_plot.set_run_folder(result.run_folders.root)
        self._set_tomogram_sources(self._tomogram_sources_from_result(result), preferred_tomogram_id)
        if getattr(getattr(result, "mbir_result", None), "message", "") == "cancelled":
            self._log(f"Run cancelled. Partial output folder: {result.run_folders.root}")
        else:
            self._log(f"Run finished. Output: {result.run_folders.root}")
        if fast_qc_report is not None:
            self._log(f"Fast QC report: {fast_qc_report}")
        if fast_mbir_result is not None:
            self.metrics_plot.show_metrics(fast_mbir_result.metrics)
            self._show_mbir_requested_slice(fast_mbir_result.volume, "Fast MBIR-lite final")
            self._load_tomogram_source("mbir_lite", switch_to_tab=False, preserve_view_state=False)
            self.tabs.setCurrentWidget(self.mbir_preview)
            self._active_image_changed()
        elif fast_prior_result is not None:
            self._load_tomogram_source("fast_prior", switch_to_tab=True, preserve_view_state=False)
        elif fast_fdk_result is not None:
            self._load_tomogram_source("fast_fdk", switch_to_tab=True, preserve_view_state=False)
        elif standard_mbir_result is not None:
            self.metrics_plot.show_metrics(standard_mbir_result.metrics)
            self._show_mbir_requested_slice(standard_mbir_result.volume, "TomoAnchor MBIR final")
            self._load_tomogram_source("mbir", switch_to_tab=False, preserve_view_state=False)
            self.tabs.setCurrentWidget(self.mbir_preview)
            self._active_image_changed()
        elif standard_fdk_volume is not None:
            self._load_tomogram_source("fdk", switch_to_tab=True, preserve_view_state=False)
        self._thread = None
        self._worker = None

    def _worker_failed(self, text: str) -> None:
        self._log(text)
        QMessageBox.critical(self, "Run failed", text.splitlines()[-1] if text else "Run failed")
        self._thread = None
        self._worker = None

    def _worker_cancelled(self, message: str) -> None:
        self._log(message or "Operation cancelled.")
        self._thread = None
        self._worker = None

    def _worker_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._set_device_monitor_runtime_logging(False)
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        for button in (
            self.load_metadata_button,
            self.metadata_folder_button,
            self.validate_metadata_button,
            self.preview_raw_button,
            self.compute_preprocess_preview_button,
            self.preview_preprocess_button,
            self.run_center_manual_button,
            self.run_center_auto_button,
            self.run_center_fine_button,
            self.apply_center_selected_button,
            self.apply_center_auto_button,
            self.export_drift_transmission_button,
            self.preprocess_button,
            self.fdk_button,
            self.mbir_button,
            self.fast_run_button,
            self.fast_fdk_sweep_button,
            self.fast_prior_button,
            self.fast_mbir_lite_button,
            self.fast_qc_button,
            self.estimate_memory_button,
            self.preload_results_button,
            self.save_config_button,
            self.load_config_button,
        ):
            button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.progress_bar.setRange(0, 0 if busy else 1)
        if not busy:
            self.progress_bar.setValue(1)
            self._update_center_shift_controls()

    def _save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save YAML config", "", "YAML files (*.yaml *.yml);;All files (*)")
        if not path:
            return
        try:
            save_config(self.config_from_ui(), path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
        else:
            self._log(f"Saved config: {path}")

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load YAML config", "", "YAML files (*.yaml *.yml);;All files (*)")
        if not path:
            return
        try:
            self.apply_config(load_config(path))
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def _display_log_message(self, message: str) -> None:
        self.log_text.append(message)
        self.statusBar().showMessage(message[-180:])
        self._update_metrics_run_folder_from_log(message)

    def _worker_log(self, message: str) -> None:
        self._display_log_message(message)

    def _log(self, message: str) -> None:
        self._display_log_message(message)
        self._append_active_run_log(message)

    def _append_active_run_log(self, message: str) -> None:
        if self._active_run_root is None:
            return
        line = str(message).rstrip("\r\n")
        if not line:
            return
        try:
            append_text_line(self._active_run_root / "log.txt", line)
        except Exception:
            pass

    def _update_metrics_run_folder_from_log(self, message: str) -> None:
        prefixes = (
            "Created run folder:",
            "Created fast run folder:",
            "Resuming fast run folder:",
            "Run finished. Output:",
            "Run cancelled. Partial output folder:",
        )
        for prefix in prefixes:
            if prefix in message:
                folder = message.split(prefix, 1)[1].strip()
                if folder:
                    self._active_run_root = Path(folder)
                    self.metrics_plot.set_run_folder(folder)
                return

    def _double(self, minimum: float, maximum: float, value: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin


def _safe_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _coerce_metric_value(value: object) -> object:
    text = str(value).strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        number = float(text)
    except Exception:
        return value
    if np.isfinite(number) and number.is_integer():
        return int(number)
    return number


def _estimated_tigre_effective_views(active_views: int, batch_size: int) -> int:
    active = max(1, int(active_views))
    batch = max(1, int(batch_size))
    effective = min(active, batch)
    kernel_floor = min(active, TIGRE_BACKPROJECTION_KERNEL_VIEWS)
    return min(active, max(effective, kernel_floor))


def _format_seconds(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _normalize_solver_name(value: str) -> str:
    solver = str(value or "auto").strip().lower()
    if solver in {"auto", "automatic"}:
        return "auto"
    if solver in {"admm", "admm_cg", "cg"}:
        return "admm"
    if solver in {"pdhg", "pdhg_low_memory", "low_memory", "chambolle_pock", "cp"}:
        return "pdhg_low_memory"
    if solver in {"subset_tv", "streaming_subset_tv", "os_tv", "os_tv_low_memory", "sart_tv"}:
        return "streaming_subset_tv"
    if solver in {"anchored_streaming_subset_tv", "mbir_lite", "fast_mbir_lite"}:
        return "anchored_streaming_subset_tv"
    return "auto"


def _solver_display_name(value: str) -> str:
    solver = _normalize_solver_name(value)
    if solver == "admm":
        return "ADMM/CG"
    if solver == "pdhg_low_memory":
        return "PDHG"
    if solver == "streaming_subset_tv":
        return "Subset-TV"
    if solver == "anchored_streaming_subset_tv":
        return "MBIR-lite"
    if solver == "auto":
        return "Auto MBIR"
    return "MBIR"


def _solver_detail_text(solver: str, metric: object) -> str:
    normalized = _normalize_solver_name(solver)
    if normalized == "admm":
        return f"CG {int(metric.cg_iterations)} iterations | residual {float(metric.cg_residual):.3e}"
    if normalized == "pdhg_low_memory":
        return "PDHG primal-dual update | CG not used"
    if normalized == "streaming_subset_tv":
        return "Streaming projection-batch TV update | CG not used"
    if normalized == "anchored_streaming_subset_tv":
        qc = getattr(metric, "qc_anchor_residual", float("nan"))
        return f"Anchored subset update | QC residual {float(qc):.3e}"
    return "Solver detail unavailable"


def run_app(argv: list[str] | None = None) -> int:
    app = QApplication(argv or [])
    app.setStyleSheet(APP_STYLE_SHEET)
    window = TVMBIRMainWindow()
    window.show()
    return exec_app(app)
