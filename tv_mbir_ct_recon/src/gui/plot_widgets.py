from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading
import traceback

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from matplotlib.transforms import blended_transform_factory
import numpy as np

from ..admm_tv_mbir import ADMMIterationMetrics
from ..device_monitor import DeviceMonitorSnapshot, GPULiveSample
from .qt_compat import (
    QColor,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QObject,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QThread,
    QTimer,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
    Slot,
)
from .style import (
    ACCENT_COLOR,
    GRID_COLOR,
    MUTED_TEXT_COLOR,
    PANEL_BACKGROUND,
    TEXT_COLOR,
    style_axis,
    style_canvas,
    style_figure,
)

_DEVICE_COLORS = [
    ACCENT_COLOR,
    "#80D6B6",
    "#FFCC80",
    "#FF8A80",
    "#A8C7F7",
    "#DDE8F7",
]
_FAST_RECON_CHILD_FOLDERS = {"anchors", "fdk_sweep", "intermediate", "mbir_lite", "prior", "qc"}
_FAST_CATEGORY_CHILDREN = {
    "FDK sweep": ("fdk_sweep", "scores.csv"),
    "Make prior": ("prior", "prior_metrics.csv"),
    "MBIR-lite": ("mbir_lite", "metrics.csv"),
    "QC report": ("qc", "qc_metrics.csv"),
}
_CSV_CACHE_LOCK = threading.Lock()
_CSV_CACHE: dict[str, tuple[object, list[dict[str, str]]]] = {}
_TEXT_CACHE_LOCK = threading.Lock()
_TEXT_CACHE: dict[str, tuple[object, dict[str, str]]] = {}
_PREVIEW_CACHE_LOCK = threading.Lock()
_PREVIEW_CACHE: dict[tuple[str | None, str | None], tuple[object, np.ndarray | None]] = {}


@dataclass(frozen=True)
class _MetricsRefreshRequest:
    request_id: int
    category: str
    sync_detail: bool
    current_detail: str
    category_run_roots: dict[str, Path]
    live_mbir_rows: list[dict[str, object]]
    live_mbir_lite_rows: list[dict[str, object]]
    waiting_for_live_mbir_metrics: bool
    waiting_for_live_mbir_lite_metrics: bool


@dataclass(frozen=True)
class _MetricsRefreshResult:
    request_id: int
    category: str
    detail_choices: list[str]
    detail_label: str
    selected_detail: str
    status_text: str
    table_note_text: str
    rows: list[dict[str, object]]
    highlight_index: int | None
    render_kind: str
    render_payload: dict[str, object]
    render_key: tuple[object, ...]


class _MetricsRefreshWorker(QObject):
    finished = Signal(object)
    failed = Signal(int, str)

    def __init__(self, request: _MetricsRefreshRequest) -> None:
        super().__init__()
        self.request = request

    @Slot()
    def run(self) -> None:
        try:
            result = _load_metrics_refresh_result(self.request)
        except Exception:
            self.failed.emit(int(self.request.request_id), traceback.format_exc())
        else:
            self.finished.emit(result)


class MetricsPlotWidget(QWidget):
    def __init__(self, on_image_changed=None) -> None:
        super().__init__()
        self.on_image_changed = on_image_changed
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        style_figure(self.figure)
        style_canvas(self.canvas)
        self.canvas_scroll = QScrollArea()
        self.canvas_scroll.setWidget(self.canvas)
        self.canvas_scroll.setWidgetResizable(False)
        self.canvas_scroll.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.category_combo = QComboBox()
        self.category_combo.addItems(["FDK sweep", "Make prior", "MBIR", "MBIR-lite", "QC report"])
        self.detail_label = QLabel("Filter")
        self.detail_combo = QComboBox()
        self.status_label = QLabel("No fast reconstruction run is selected.")
        self.status_label.setWordWrap(True)
        self.table = QTableWidget(0, 0)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table_note_label = QLabel("")
        self.table_note_label.setWordWrap(True)
        self._run_folder: Path | None = None
        self._category_run_roots: dict[str, Path] = {}
        self._live_mbir_metrics: list[object] = []
        self._live_mbir_lite_metrics: list[object] = []
        self._waiting_for_live_mbir_metrics = False
        self._waiting_for_live_mbir_lite_metrics = False
        self._block_detail_signal = False
        self._current_image: np.ndarray | None = None
        self._levels: tuple[float, float] | None = None
        self._zoom_factor = 1.0
        self._pan_center: tuple[float, float] | None = None
        self._image_signature = ""
        self._image_artists: list[tuple[object, np.ndarray]] = []
        self._pan_drag: dict[str, object] | None = None
        self._last_layout_category = ""
        self._polling_enabled = True
        self._refresh_thread: QThread | None = None
        self._refresh_worker: _MetricsRefreshWorker | None = None
        self._refresh_request_id = 0
        self._active_refresh_request_id = 0
        self._pending_refresh_sync_detail = False
        self._last_render_key: tuple[object, ...] | None = None

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Metrics"))
        controls.addWidget(self.category_combo)
        controls.addWidget(self.detail_label)
        controls.addWidget(self.detail_combo, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.status_label)
        self.metrics_splitter = QSplitter(Qt.Vertical)
        table_panel = QWidget()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self.table)
        table_layout.addWidget(self.table_note_label)
        self.metrics_splitter.addWidget(self.canvas_scroll)
        self.metrics_splitter.addWidget(table_panel)
        self.metrics_splitter.setStretchFactor(0, 6)
        self.metrics_splitter.setStretchFactor(1, 1)
        self.metrics_splitter.setSizes([760, 180])
        layout.addWidget(self.metrics_splitter, 1)
        self._set_canvas_pixel_size(960, 700)

        self.category_combo.currentTextChanged.connect(lambda *_: self._refresh_category(sync_detail=True))
        self.detail_combo.currentTextChanged.connect(lambda *_: self._detail_changed())
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1500)
        self._poll_timer.timeout.connect(self.refresh_from_files)
        self._poll_timer.start()
        self.canvas.mpl_connect("button_press_event", self._on_pan_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_pan_motion)
        self.canvas.mpl_connect("button_release_event", self._on_pan_release)
        self._request_refresh(sync_detail=True)
        self.set_polling_enabled(self.isVisible())

    def set_run_folder(self, run_folder: str | Path | None) -> None:
        if not run_folder:
            self._run_folder = None
            self._category_run_roots = {}
        else:
            root = _normalize_fast_run_root(Path(run_folder))
            self._run_folder = root
            self._remember_run_folder(root)
        self.refresh_from_files()

    def select_category(self, category: str) -> None:
        if category != self.category_combo.currentText():
            self.category_combo.setCurrentText(category)
        else:
            self._refresh_category(sync_detail=True)

    def refresh_from_files(self) -> None:
        self._request_refresh(sync_detail=True)

    def show_metrics(self, metrics: list[object], category: str | None = None) -> None:
        metric_category = category or _infer_metrics_category(metrics)
        if metric_category == "MBIR-lite":
            self._live_mbir_lite_metrics = list(metrics or [])
            if self._live_mbir_lite_metrics:
                self._waiting_for_live_mbir_lite_metrics = False
            if not self._live_mbir_lite_metrics and self.category_combo.currentText() == "MBIR-lite":
                self._request_refresh(sync_detail=True)
                return
            if self.category_combo.currentText() == "MBIR-lite":
                self._request_refresh(sync_detail=False)
            return
        self._live_mbir_metrics = list(metrics or [])
        if self._live_mbir_metrics:
            self._waiting_for_live_mbir_metrics = False
        if not self._live_mbir_metrics and self.category_combo.currentText() == "MBIR":
            self._request_refresh(sync_detail=True)
            return
        if self.category_combo.currentText() == "MBIR":
            self._request_refresh(sync_detail=False)

    def begin_live_mbir_session(self, category: str = "MBIR") -> None:
        if category == "MBIR-lite":
            self._live_mbir_lite_metrics = []
            self._waiting_for_live_mbir_lite_metrics = True
            if self.category_combo.currentText() == "MBIR-lite":
                self._request_refresh(sync_detail=False)
            return
        self._live_mbir_metrics = []
        self._waiting_for_live_mbir_metrics = True
        if self.category_combo.currentText() == "MBIR":
            self._request_refresh(sync_detail=False)

    def clear_live_mbir_metrics(self) -> None:
        self._live_mbir_metrics = []
        self._live_mbir_lite_metrics = []
        self._waiting_for_live_mbir_metrics = False
        self._waiting_for_live_mbir_lite_metrics = False
        if self.category_combo.currentText() in {"MBIR", "MBIR-lite"}:
            self._request_refresh(sync_detail=True)

    def end_live_mbir_session(self, category: str | None = None) -> None:
        if category in {None, "MBIR"}:
            self._waiting_for_live_mbir_metrics = False
        if category in {None, "MBIR-lite"}:
            self._waiting_for_live_mbir_lite_metrics = False
        if self.category_combo.currentText() in {"MBIR", "MBIR-lite"}:
            self._request_refresh(sync_detail=False)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.set_polling_enabled(True)

    def hideEvent(self, event) -> None:
        self.set_polling_enabled(False)
        super().hideEvent(event)

    def set_polling_enabled(self, enabled: bool) -> None:
        self._polling_enabled = bool(enabled)
        if self._polling_enabled:
            if not self._poll_timer.isActive():
                self._poll_timer.start()
            self._request_refresh(sync_detail=False)
        else:
            self._poll_timer.stop()

    def _request_refresh(self, sync_detail: bool = False) -> None:
        self._refresh_request_id += 1
        self._pending_refresh_sync_detail = self._pending_refresh_sync_detail or bool(sync_detail)
        if self._refresh_thread is not None:
            return
        next_sync_detail = self._pending_refresh_sync_detail
        self._pending_refresh_sync_detail = False
        self._start_refresh_worker(next_sync_detail, self._refresh_request_id)

    def _start_refresh_worker(self, sync_detail: bool, request_id: int) -> None:
        request = _MetricsRefreshRequest(
            request_id=int(request_id),
            category=str(self.category_combo.currentText()),
            sync_detail=bool(sync_detail),
            current_detail=str(self.detail_combo.currentText() or "").strip(),
            category_run_roots=dict(self._category_run_roots),
            live_mbir_rows=_mbir_rows_from_objects(self._live_mbir_metrics),
            live_mbir_lite_rows=[_mbir_lite_metric_row_from_object(metric) for metric in self._live_mbir_lite_metrics],
            waiting_for_live_mbir_metrics=bool(self._waiting_for_live_mbir_metrics),
            waiting_for_live_mbir_lite_metrics=bool(self._waiting_for_live_mbir_lite_metrics),
        )
        thread = QThread(self)
        worker = _MetricsRefreshWorker(request)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._refresh_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(self._refresh_failed)
        worker.failed.connect(thread.quit)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._refresh_thread_finished)
        self._active_refresh_request_id = int(request_id)
        self._refresh_thread = thread
        self._refresh_worker = worker
        thread.start()

    def _refresh_finished(self, result: object) -> None:
        if isinstance(result, _MetricsRefreshResult) and int(result.request_id) == int(self._refresh_request_id):
            self._apply_refresh_result(result)

    def _refresh_failed(self, request_id: int, _traceback_text: str) -> None:
        if int(request_id) == int(self._refresh_request_id):
            self.table_note_label.setText("Metrics refresh failed. Keeping the last successful view.")

    def _refresh_thread_finished(self) -> None:
        self._refresh_thread = None
        self._refresh_worker = None
        if self._active_refresh_request_id != self._refresh_request_id or self._pending_refresh_sync_detail:
            next_sync_detail = self._pending_refresh_sync_detail
            self._pending_refresh_sync_detail = False
            self._start_refresh_worker(next_sync_detail, self._refresh_request_id)

    def _apply_refresh_result(self, result: _MetricsRefreshResult) -> None:
        if result.category != self.category_combo.currentText():
            return
        if result.render_key == self._last_render_key:
            self._set_detail_choices(result.detail_choices, result.detail_label, current_choice=result.selected_detail)
            self.status_label.setText(result.status_text)
            self.table_note_label.setText(result.table_note_text)
            return
        self._set_detail_choices(result.detail_choices, result.detail_label, current_choice=result.selected_detail)
        self.status_label.setText(result.status_text)
        self.table_note_label.setText(result.table_note_text)
        self._fill_table(result.rows, highlight_index=result.highlight_index)
        if result.render_kind == "fdk":
            self._render_fdk_sweep(result)
        elif result.render_kind == "prior":
            self._render_prior(result)
        elif result.render_kind == "mbir":
            self._draw_mbir(result.rows)
        elif result.render_kind == "mbir_lite":
            self._draw_mbir_lite(result.rows)
        elif result.render_kind == "qc":
            self._render_qc_report(result)
        else:
            self._draw_message("No metrics category selected.")
        self._last_render_key = result.render_key

    def current_image(self) -> np.ndarray | None:
        return self._current_image

    def current_levels(self) -> tuple[float, float] | None:
        return self._levels

    def current_zoom_factor(self) -> float:
        return self._zoom_factor

    def set_levels(self, low: float, high: float) -> None:
        if self._current_image is None:
            return
        low = float(low)
        high = float(high)
        if not np.isfinite(low) or not np.isfinite(high):
            return
        if high <= low:
            high = low + 1.0
        self._levels = (low, high)
        for artist, _image in self._image_artists:
            artist.set_clim(low, high)
        self.canvas.draw_idle()
        self._emit_image_changed()

    def reset_auto_levels(self) -> tuple[float, float] | None:
        if self._current_image is None:
            return None
        self._levels = self._auto_limits(self._current_image)
        self.set_levels(self._levels[0], self._levels[1])
        return self._levels

    def set_zoom_factor(self, zoom_factor: float) -> float:
        if self._current_image is None:
            return self._zoom_factor
        self._zoom_factor = min(max(float(zoom_factor), 1.0), 32.0)
        if self._zoom_factor <= 1.0001:
            self._pan_center = None
        self._apply_image_view()
        self.canvas.draw_idle()
        self._emit_image_changed()
        return self._zoom_factor

    def fit_to_view(self) -> float:
        self._zoom_factor = 1.0
        self._pan_center = None
        self._apply_image_view()
        self.canvas.draw_idle()
        self._emit_image_changed()
        return self._zoom_factor

    def show_center_shift_metrics(
        self,
        metric_table: list[dict[str, float]],
        metric_key: str = "combined_score",
        selected_shift: float | None = None,
        recommended_shift: float | None = None,
    ) -> None:
        self._last_render_key = None
        self._clear_image_display(emit=False)
        self.status_label.setText("Center-search metrics are displayed for the active alignment preview.")
        self._set_detail_choices([])
        self._fill_table(metric_table)
        self.figure.clear()
        style_figure(self.figure)
        axis = self.figure.add_subplot(111)
        style_axis(axis, grid=True)
        if not metric_table:
            axis.text(
                0.5,
                0.5,
                "No center-search metrics yet",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color=TEXT_COLOR,
            )
            axis.axis("off")
        else:
            shifts = np.asarray([row["shift_px"] for row in metric_table], dtype=np.float64)
            values = np.asarray([row.get(metric_key, np.nan) for row in metric_table], dtype=np.float64)
            finite = np.isfinite(shifts) & np.isfinite(values)
            if np.any(finite):
                axis.plot(shifts[finite], values[finite], linewidth=1.4, marker="o", color=ACCENT_COLOR, label=_metric_label(metric_key))
                if selected_shift is not None and np.isfinite(selected_shift):
                    axis.axvline(float(selected_shift), linestyle="--", linewidth=1.2, color="#FF8A80", label="Selected")
                if recommended_shift is not None and np.isfinite(recommended_shift):
                    best_index = int(np.argmin(np.abs(shifts - float(recommended_shift))))
                    axis.scatter([shifts[best_index]], [values[best_index]], s=70, zorder=5, color="#80D6B6", label="Recommended")
                axis.set_xlabel("Center offset shift (px)")
                axis.set_ylabel(_metric_label(metric_key))
                axis.set_title(f"{_metric_label(metric_key)} vs center shift", color=TEXT_COLOR)
                _style_legend(axis.legend())
            else:
                axis.text(0.5, 0.5, "No finite metric values", ha="center", va="center", transform=axis.transAxes, color=TEXT_COLOR)
                axis.axis("off")
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self._emit_image_changed()

    def _detail_changed(self) -> None:
        if not self._block_detail_signal:
            self._refresh_category(sync_detail=False)

    def _apply_category_layout(self, category: str) -> None:
        if category == self._last_layout_category:
            return
        self._last_layout_category = category
        if category == "Make prior":
            self.metrics_splitter.setSizes([940, 140])
        elif category == "MBIR":
            self.metrics_splitter.setSizes([1000, 180])
        elif category == "MBIR-lite":
            self.metrics_splitter.setSizes([980, 170])
        elif category == "QC report":
            self.metrics_splitter.setSizes([960, 160])
        else:
            self.metrics_splitter.setSizes([860, 170])

    def _refresh_category(self, sync_detail: bool = False) -> None:
        category = self.category_combo.currentText()
        self._apply_category_layout(category)
        self._request_refresh(sync_detail=sync_detail)

    def _show_fdk_sweep(self, sync_detail: bool) -> None:
        self._request_refresh(sync_detail=sync_detail)

    def _show_prior(self, sync_detail: bool) -> None:
        self._request_refresh(sync_detail=sync_detail)

    def _show_mbir(self) -> None:
        self._request_refresh(sync_detail=False)

    def _show_mbir_lite(self) -> None:
        self._request_refresh(sync_detail=False)

    def _show_qc_report(self) -> None:
        self._request_refresh(sync_detail=False)

    def _render_fdk_sweep(self, result: _MetricsRefreshResult) -> None:
        preview_items = result.render_payload.get("preview_items") or []
        selected_filter = str(result.selected_detail or "")
        self._start_image_display(f"fdk_sweep:{selected_filter}")
        if not preview_items:
            self._draw_message("Select a filter after FDK sweep scores are available.")
            return
        image_arrays = [image for _row, _path, image in preview_items if image is not None]
        cols = min(3, max(1, len(preview_items)))
        row_count = max(1, int(np.ceil(len(preview_items) / cols)))
        self._size_canvas_for_image_grid(
            image_arrays,
            cols=cols,
            rows=row_count,
            extra_height=60 + 28 * row_count,
            minimum_width=940,
            fallback_width=1320,
        )
        self.figure.clear()
        style_figure(self.figure)
        axes = np.asarray(self.figure.subplots(row_count, cols)).reshape(-1)
        for axis, (row, path, image) in zip(axes, preview_items):
            style_axis(axis)
            if image is not None:
                score = _to_float(row.get("score_total"))
                suffix = f" | score {score:.4g}" if np.isfinite(score) else ""
                self._register_image(axis, image, f"{selected_filter}{suffix}")
            else:
                axis.text(0.5, 0.5, f"Preview not found\n{path.name}", ha="center", va="center", transform=axis.transAxes, color=TEXT_COLOR)
                axis.axis("off")
        for axis in axes[len(preview_items) :]:
            axis.axis("off")
        self.figure.subplots_adjust(left=0.035, right=0.99, top=0.955, bottom=0.06, wspace=0.06, hspace=0.12)
        self._finish_image_display()
        self.canvas.draw_idle()

    def _render_prior(self, result: _MetricsRefreshResult) -> None:
        selected = str(result.selected_detail or "Selected prior")
        preview_image = result.render_payload.get("preview_image")
        rows = result.rows
        if not bool(result.render_payload.get("folder_available", False)):
            self._draw_message("No fast reconstruction run is selected.")
            return
        self._start_image_display(f"prior:{selected}")
        if preview_image is not None:
            self._size_canvas_for_image(preview_image, extra_height=260, minimum_width=960, fallback_width=1280)
        else:
            self._size_canvas_for_plot_grid(minimum_width=960, fallback_width=1180, height_ratio=0.72)
        self.figure.clear()
        style_figure(self.figure)
        grid = self.figure.add_gridspec(2, 1, height_ratios=[4.85, 1.35], hspace=0.18)
        image_axis = self.figure.add_subplot(grid[0, 0])
        plot_axis = self.figure.add_subplot(grid[1, 0])
        style_axis(image_axis)
        if preview_image is None:
            image_axis.text(0.5, 0.5, f"{selected} is not available yet.", ha="center", va="center", transform=image_axis.transAxes, color=TEXT_COLOR)
            image_axis.axis("off")
        else:
            self._register_image(image_axis, preview_image, selected)
        self._plot_prior_metrics(plot_axis, rows)
        self.figure.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.07, hspace=0.23)
        self._finish_image_display()
        self.canvas.draw_idle()

    def _plot_prior_metrics(self, axis, rows: list[dict[str, str]]) -> None:
        style_axis(axis, grid=True)
        if not rows:
            axis.text(0.5, 0.5, "Waiting for prior_metrics.csv", ha="center", va="center", transform=axis.transAxes, color=TEXT_COLOR)
            axis.axis("off")
            return
        x = np.asarray([_to_float(row.get("strength")) for row in rows], dtype=np.float64)
        fields = [
            ("background_noise", ACCENT_COLOR),
            ("edge_retention", "#80D6B6"),
            ("correction_fraction", "#FFCC80"),
            ("anchor_residual_ratio", "#FF8A80"),
        ]
        for key, color in fields:
            y = np.asarray([_to_float(row.get(key)) for row in rows], dtype=np.float64)
            finite = np.isfinite(x) & np.isfinite(y)
            if np.any(finite):
                axis.plot(x[finite], y[finite], marker="o", linewidth=1.3, markersize=3, label=key, color=color)
        axis.set_xlabel("TV strength")
        axis.set_title("Prior candidate metrics", color=TEXT_COLOR, fontsize=9)
        _style_legend(axis.legend(fontsize=7))

    def _draw_mbir(self, rows: list[dict[str, object]]) -> None:
        self._clear_image_display(emit=False)
        self.figure.clear()
        style_figure(self.figure)
        if not rows:
            self._draw_message("Waiting for MBIR metrics.")
            return
        panels = [
            ("objective", "Objective", ACCENT_COLOR, True),
            ("data_residual", "Data residual", "#80D6B6", True),
            ("data_fidelity", "Data term", "#A8C7F7", True),
            ("tv_term", "TV term", "#FFCC80", True),
            ("relative_x_change", "Relative change", "#DDE8F7", True),
            ("elapsed_s", "Elapsed seconds", "#B39DDB", False),
        ]
        if _rows_have_finite_values(rows, "primal_residual"):
            panels.append(("primal_residual", "Primal residual", "#FF8A80", True))
        if _rows_have_finite_values(rows, "dual_residual"):
            panels.append(("dual_residual", "Dual residual", "#F48FB1", True))
        if _rows_have_finite_values(rows, "cg_residual"):
            panels.append(("cg_residual", "CG residual", "#90CAF9", True))
        if _rows_have_finite_values(rows, "cg_iterations"):
            panels.append(("cg_iterations", "CG iterations", "#A5D6A7", False))
        self._size_canvas_for_plot_grid(
            minimum_width=1120,
            fallback_width=1460,
            height_ratio=0.72 if len(panels) <= 6 else 0.9,
        )
        iterations = np.asarray([_to_float(row.get("iteration"), index + 1) for index, row in enumerate(rows)], dtype=np.float64)
        for spec, (key, title, color, log_y) in zip(_panel_specs_for_count(self.figure, len(panels)), panels):
            axis = self.figure.add_subplot(spec)
            _plot_metric_panel(axis, iterations, rows, key, title, color, log_y, "Iteration")
        self.figure.subplots_adjust(left=0.055, right=0.985, top=0.955, bottom=0.06, hspace=0.28, wspace=0.22)
        self.canvas.draw_idle()
        self._emit_image_changed()

    def _draw_mbir_lite(self, rows: list[dict[str, object]]) -> None:
        self._clear_image_display(emit=False)
        self.figure.clear()
        style_figure(self.figure)
        if not rows:
            self._draw_message("Waiting for MBIR-lite live metrics.")
            return
        self._size_canvas_for_plot_grid(minimum_width=1080, fallback_width=1400, height_ratio=0.84)
        panels = [
            ("objective_total", "Objective", ACCENT_COLOR, True),
            ("data_weighted", "Data weighted", "#80D6B6", True),
            ("tv_term", "TV term", "#FFCC80", True),
            ("prior_anchor_term", "Prior anchor", "#FF8A80", True),
            ("qc_anchor_residual", "QC anchor residual", "#A8C7F7", True),
            ("relative_change", "Relative change", "#DDE8F7", True),
            ("elapsed_s", "Elapsed seconds", "#B39DDB", False),
        ]
        iterations = np.asarray([_to_float(row.get("iteration"), index + 1) for index, row in enumerate(rows)], dtype=np.float64)
        for spec, (key, title, color, log_y) in zip(_panel_specs_for_count(self.figure, len(panels)), panels):
            axis = self.figure.add_subplot(spec)
            _plot_metric_panel(axis, iterations, rows, key, title, color, log_y, "Sweep")
        self.figure.subplots_adjust(left=0.06, right=0.985, top=0.955, bottom=0.06, hspace=0.28, wspace=0.22)
        self.canvas.draw_idle()
        self._emit_image_changed()

    def _render_qc_report(self, result: _MetricsRefreshResult) -> None:
        rows = result.rows
        image = result.render_payload.get("preview_image")
        self._start_image_display("qc")
        if image is not None:
            self._size_canvas_for_image(image, extra_height=300, minimum_width=1040, fallback_width=1480)
        else:
            self._size_canvas_for_plot_grid(minimum_width=1040, fallback_width=1400, height_ratio=0.88)
        self.figure.clear()
        style_figure(self.figure)
        grid = self.figure.add_gridspec(2, 1, height_ratios=[5.35, 1.95], hspace=0.14)
        image_axis = self.figure.add_subplot(grid[0, 0])
        bar_axis = self.figure.add_subplot(grid[1, 0])
        style_axis(image_axis)
        if image is not None:
            self._register_image(image_axis, image, "QC preview panel")
        else:
            image_axis.text(0.5, 0.5, "QC preview_panel.png is not available yet.", ha="center", va="center", transform=image_axis.transAxes, color=TEXT_COLOR)
            image_axis.axis("off")
        style_axis(bar_axis, grid=True)
        if rows:
            grouped = _qc_residual_groups(rows)
            if grouped:
                volumes = list(grouped.keys())
                x = np.arange(len(volumes), dtype=np.float64)
                width = 0.34
                tune_values = np.asarray([grouped[volume].get("tune", float("nan")) for volume in volumes], dtype=np.float64)
                qc_values = np.asarray([grouped[volume].get("qc", float("nan")) for volume in volumes], dtype=np.float64)
                tune_bars = bar_axis.bar(
                    x - width / 2.0,
                    np.where(np.isfinite(tune_values), np.maximum(tune_values, 1e-30), np.nan),
                    width=width,
                    color="#80D6B6",
                    label="Tune",
                )
                qc_bars = bar_axis.bar(
                    x + width / 2.0,
                    np.where(np.isfinite(qc_values), np.maximum(qc_values, 1e-30), np.nan),
                    width=width,
                    color="#A8C7F7",
                    label="QC",
                )
                bar_axis.set_yscale("log")
                bar_axis.set_xticks(x)
                bar_axis.set_xticklabels([_display_volume_name(volume) for volume in volumes], fontsize=11)
                bar_axis.tick_params(axis="x", colors=MUTED_TEXT_COLOR, labelsize=11, pad=10)
                bar_axis.tick_params(axis="y", colors=MUTED_TEXT_COLOR, labelsize=11)
                bar_axis.set_ylabel("Normalized residual (lower is better)", fontsize=11, color=TEXT_COLOR)
                bar_axis.set_title("Tune vs held-out QC residuals", color=TEXT_COLOR, fontsize=12)
                _style_legend(bar_axis.legend(loc="upper right", fontsize=10))
                annotation_levels = np.linspace(0.68, 0.44, len(volumes), dtype=np.float64)
                annotation_transform = blended_transform_factory(bar_axis.transData, bar_axis.transAxes)
                for index, (volume, bar) in enumerate(zip(volumes, qc_bars)):
                    tune_value = tune_values[index]
                    qc_value = qc_values[index]
                    if not np.isfinite(tune_value) or not np.isfinite(qc_value) or tune_value == 0.0:
                        continue
                    percent_change = 100.0 * (tune_value - qc_value) / tune_value
                    color = "#80D6B6" if percent_change >= 0.0 else "#FF8A80"
                    bar_axis.text(
                        x[index],
                        float(annotation_levels[index]),
                        f"{percent_change:+.1f}% vs tune",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=color,
                        transform=annotation_transform,
                        clip_on=True,
                        bbox={
                            "facecolor": PANEL_BACKGROUND,
                            "edgecolor": color,
                            "alpha": 0.9,
                            "boxstyle": "round,pad=0.2",
                        },
                    )
        else:
            bar_axis.text(0.5, 0.5, "Waiting for qc_metrics.csv", ha="center", va="center", transform=bar_axis.transAxes, color=TEXT_COLOR)
            bar_axis.axis("off")
        self.figure.subplots_adjust(left=0.065, right=0.985, top=0.965, bottom=0.065, hspace=0.18)
        self._finish_image_display()
        self.canvas.draw_idle()

    def _draw_message(self, message: str) -> None:
        self._clear_image_display(emit=False)
        self.figure.clear()
        style_figure(self.figure)
        axis = self.figure.add_subplot(111)
        style_axis(axis, grid=True)
        axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes, color=TEXT_COLOR)
        axis.axis("off")
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self._emit_image_changed()

    def _start_image_display(self, signature: str) -> None:
        if signature != self._image_signature:
            self._levels = None
            self._zoom_factor = 1.0
            self._pan_center = None
            self._image_signature = signature
        self._image_artists = []
        self._current_image = None

    def _register_image(self, axis, image: np.ndarray, title: str) -> None:
        array = _as_display_image(image)
        if array is None:
            axis.text(0.5, 0.5, "Image unavailable", ha="center", va="center", transform=axis.transAxes, color=TEXT_COLOR)
            axis.axis("off")
            return
        if self._current_image is None:
            self._current_image = array
            if self._levels is None:
                self._levels = self._auto_limits(array)
            if self._pan_center is None:
                self._pan_center = ((array.shape[1] - 1) / 2.0, (array.shape[0] - 1) / 2.0)
        low, high = self._levels or self._auto_limits(array)
        artist = axis.imshow(array, cmap="gray", vmin=low, vmax=high, origin="upper")
        axis.set_title(title, color=TEXT_COLOR, fontsize=9)
        axis.axis("off")
        self._image_artists.append((artist, array))

    def _finish_image_display(self) -> None:
        if not self._image_artists:
            self._current_image = None
        self._apply_image_view()
        self._emit_image_changed()

    def _clear_image_display(self, emit: bool = True) -> None:
        self._current_image = None
        self._levels = None
        self._zoom_factor = 1.0
        self._pan_center = None
        self._image_signature = ""
        self._image_artists = []
        self._pan_drag = None
        if emit:
            self._emit_image_changed()

    def _apply_image_view(self) -> None:
        for artist, image in self._image_artists:
            axis = artist.axes
            rows, cols = image.shape[:2]
            if self._zoom_factor <= 1.0001:
                axis.set_xlim(-0.5, cols - 0.5)
                axis.set_ylim(rows - 0.5, -0.5)
                continue
            center_col, center_row = self._clamped_pan_center(image.shape)
            half_width = max(0.5, cols / (2.0 * self._zoom_factor))
            half_height = max(0.5, rows / (2.0 * self._zoom_factor))
            axis.set_xlim(center_col - half_width, center_col + half_width)
            axis.set_ylim(center_row + half_height, center_row - half_height)

    def _set_canvas_pixel_size(self, width: int, height: int) -> None:
        width = max(720, int(width))
        height = max(520, int(height))
        dpi = float(self.figure.dpi or 100.0)
        self.figure.set_size_inches(width / dpi, height / dpi, forward=True)
        self.canvas.setMinimumSize(width, height)
        self.canvas.resize(width, height)
        self.canvas.updateGeometry()

    def _canvas_target_width(self, minimum_width: int = 920, fallback_width: int = 1280) -> int:
        viewport_width = 0
        try:
            viewport_width = int(self.canvas_scroll.viewport().width())
        except Exception:
            viewport_width = 0
        if viewport_width >= 240:
            return max(minimum_width, viewport_width - 24)
        widget_width = int(self.width()) if self.width() > 0 else 0
        if widget_width >= 320:
            return max(minimum_width, widget_width - 32)
        return max(minimum_width, int(fallback_width))

    def _size_canvas_for_image(self, image: np.ndarray, *, extra_height: int, minimum_width: int, fallback_width: int) -> None:
        array = _as_display_image(image)
        if array is None:
            self._size_canvas_for_plot_grid(minimum_width=minimum_width, fallback_width=fallback_width, height_ratio=0.72)
            return
        width = self._canvas_target_width(minimum_width=minimum_width, fallback_width=fallback_width)
        aspect = float(array.shape[0]) / max(float(array.shape[1]), 1.0)
        height = int(np.clip(width * aspect + float(extra_height), 560.0, 1800.0))
        self._set_canvas_pixel_size(width, height)

    def _size_canvas_for_image_grid(
        self,
        images: list[np.ndarray],
        *,
        cols: int,
        rows: int,
        extra_height: int,
        minimum_width: int,
        fallback_width: int,
    ) -> None:
        displays = [array for array in (_as_display_image(image) for image in images) if array is not None]
        if not displays:
            self._size_canvas_for_plot_grid(minimum_width=minimum_width, fallback_width=fallback_width, height_ratio=0.62)
            return
        mean_height = float(np.mean([array.shape[0] for array in displays]))
        mean_width = float(np.mean([array.shape[1] for array in displays]))
        width = self._canvas_target_width(minimum_width=minimum_width, fallback_width=fallback_width)
        content_aspect = (max(1, rows) * mean_height) / max(float(max(1, cols)) * mean_width, 1.0)
        height = int(np.clip(width * content_aspect + float(extra_height), 520.0, 1700.0))
        self._set_canvas_pixel_size(width, height)

    def _size_canvas_for_plot_grid(self, *, minimum_width: int, fallback_width: int, height_ratio: float) -> None:
        width = self._canvas_target_width(minimum_width=minimum_width, fallback_width=fallback_width)
        height = int(np.clip(width * float(height_ratio), 560.0, 1600.0))
        self._set_canvas_pixel_size(width, height)

    def _clamped_pan_center(self, shape: tuple[int, ...]) -> tuple[float, float]:
        rows, cols = int(shape[0]), int(shape[1])
        default = ((cols - 1) / 2.0, (rows - 1) / 2.0)
        if self._pan_center is None:
            self._pan_center = default
            return default
        center_col, center_row = float(self._pan_center[0]), float(self._pan_center[1])
        if self._zoom_factor <= 1.0001:
            self._pan_center = default
            return default
        half_width = max(0.5, cols / (2.0 * self._zoom_factor))
        half_height = max(0.5, rows / (2.0 * self._zoom_factor))
        min_col = min(max(half_width - 0.5, 0.0), max(cols - 1, 0))
        max_col = max(min(cols - 0.5 - half_width, cols - 1), min_col)
        min_row = min(max(half_height - 0.5, 0.0), max(rows - 1, 0))
        max_row = max(min(rows - 0.5 - half_height, rows - 1), min_row)
        center = (min(max(center_col, min_col), max_col), min(max(center_row, min_row), max_row))
        self._pan_center = center
        return center

    def _on_pan_press(self, event) -> None:
        if self._zoom_factor <= 1.0001 or event.inaxes is None or event.button != 1:
            return
        for _artist, image in self._image_artists:
            if event.inaxes is _artist.axes:
                self._pan_drag = {
                    "x": float(event.x),
                    "y": float(event.y),
                    "center": self._clamped_pan_center(image.shape),
                    "shape": tuple(image.shape),
                    "bbox": event.inaxes.bbox.bounds,
                }
                return

    def _on_pan_motion(self, event) -> None:
        if self._pan_drag is None or event.x is None or event.y is None:
            return
        shape = tuple(self._pan_drag["shape"])
        rows, cols = int(shape[0]), int(shape[1])
        _bbox_x, _bbox_y, bbox_width, bbox_height = self._pan_drag["bbox"]
        if bbox_width <= 0 or bbox_height <= 0:
            return
        start_col, start_row = self._pan_drag["center"]
        visible_width = cols / max(self._zoom_factor, 1.0)
        visible_height = rows / max(self._zoom_factor, 1.0)
        dx = (float(event.x) - float(self._pan_drag["x"])) * visible_width / float(bbox_width)
        dy = (float(event.y) - float(self._pan_drag["y"])) * visible_height / float(bbox_height)
        self._pan_center = (float(start_col) - dx, float(start_row) + dy)
        self._apply_image_view()
        self.canvas.draw_idle()
        self._emit_image_changed()

    def _on_pan_release(self, _event) -> None:
        self._pan_drag = None

    def _auto_limits(self, array: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(array, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            low, high = np.percentile(finite, [1, 99])
        else:
            low, high = 0.0, 1.0
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            if finite.size:
                low = float(np.min(finite))
                high = float(np.max(finite))
            else:
                low, high = 0.0, 1.0
        if high <= low:
            high = low + 1.0
        return float(low), float(high)

    def _emit_image_changed(self) -> None:
        if self.on_image_changed is not None:
            self.on_image_changed()

    def _standard_metrics_path(self) -> Path | None:
        root = self._category_run_roots.get("MBIR")
        path = _standard_metrics_csv_path(root)
        if path is None:
            return None
        return path if path.exists() else None

    def _fast_folder(self, name: str) -> Path | None:
        category = self.category_combo.currentText()
        root = self._category_run_roots.get(category)
        return _fast_folder_for_root(root, name)

    def _remember_run_folder(self, root: Path) -> None:
        normalized = _normalize_fast_run_root(Path(root))
        standard_metrics = normalized / "metrics" / "metrics.csv"
        if standard_metrics.exists():
            self._category_run_roots["MBIR"] = normalized
        for category, (child, filename) in _FAST_CATEGORY_CHILDREN.items():
            if _fast_child_metric_path(normalized, child, filename) is not None:
                self._category_run_roots[category] = normalized

    def _set_detail_choices(self, choices: list[str], label: str = "", current_choice: str | None = None) -> None:
        current = str(current_choice if current_choice is not None else self.detail_combo.currentText())
        self._block_detail_signal = True
        self.detail_combo.clear()
        for choice in choices:
            self.detail_combo.addItem(str(choice))
        if current in choices:
            self.detail_combo.setCurrentText(current)
        self.detail_combo.setEnabled(bool(choices))
        self.detail_combo.setVisible(bool(choices))
        self.detail_label.setText(label)
        self.detail_label.setVisible(bool(choices))
        self._block_detail_signal = False

    def _fill_table(self, rows: list[dict[str, object]], highlight_index: int | None = None) -> None:
        if not rows:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        columns: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in columns:
                    columns.append(str(key))
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, column in enumerate(columns):
                item = QTableWidgetItem(_format_table_value(row.get(column, "")))
                item.setTextAlignment(Qt.AlignCenter)
                if highlight_index is not None and row_index == highlight_index:
                    item.setBackground(QColor("#314B42"))
                    item.setForeground(QColor("#F5FFF9"))
                self.table.setItem(row_index, col_index, item)
        if highlight_index is not None and 0 <= highlight_index < len(rows):
            self.table.selectRow(highlight_index)


def _standard_metrics_csv_path(root: Path | None) -> Path | None:
    if root is None:
        return None
    normalized = _normalize_fast_run_root(Path(root))
    return normalized / "metrics" / "metrics.csv"


def _fast_folder_for_root(root: Path | None, name: str) -> Path | None:
    if root is None:
        return None
    normalized = _normalize_fast_run_root(Path(root))
    folder = normalized / "fast_recon" / name
    legacy = normalized / "fast_recon" / "fast_recon" / name
    if not folder.exists() and legacy.exists():
        return legacy
    return folder


def _path_cache_key(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve())
    except Exception:
        return str(path)


def _path_stamp(path: Path | None) -> object:
    if path is None:
        return None
    try:
        stat = Path(path).stat()
    except OSError:
        return (False, 0, 0)
    return (True, int(stat.st_mtime_ns), int(stat.st_size))


def _preview_stamp(image_path: Path | None, npy_path: Path | None) -> tuple[object, object]:
    return (_path_stamp(image_path), _path_stamp(npy_path))


def _read_csv_rows_cached(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    key = _path_cache_key(path)
    stamp = _path_stamp(path)
    with _CSV_CACHE_LOCK:
        cached = _CSV_CACHE.get(str(key))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    rows = _read_csv_rows(path)
    with _CSV_CACHE_LOCK:
        _CSV_CACHE[str(key)] = (stamp, rows)
    return rows


def _read_text_mapping_cached(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    key = _path_cache_key(path)
    stamp = _path_stamp(path)
    with _TEXT_CACHE_LOCK:
        cached = _TEXT_CACHE.get(str(key))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    values = _read_text_mapping(path)
    with _TEXT_CACHE_LOCK:
        _TEXT_CACHE[str(key)] = (stamp, values)
    return values


def _load_preview_image_cached(image_path: Path | None, npy_path: Path | None) -> np.ndarray | None:
    key = (_path_cache_key(image_path), _path_cache_key(npy_path))
    stamp = _preview_stamp(image_path, npy_path)
    with _PREVIEW_CACHE_LOCK:
        cached = _PREVIEW_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    image = _load_preview_image(image_path, npy_path)
    with _PREVIEW_CACHE_LOCK:
        _PREVIEW_CACHE[key] = (stamp, image)
    return image


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _normalize_fast_run_root(path: Path) -> Path:
    root = Path(path)
    if root.parent.name == "fast_recon" and root.name in _FAST_RECON_CHILD_FOLDERS:
        root = root.parent
    while root.name == "fast_recon":
        root = root.parent
    return root


def _fast_child_metric_path(root: Path, child: str, filename: str) -> Path | None:
    normalized = _normalize_fast_run_root(Path(root))
    folder = normalized / "fast_recon" / child
    legacy = normalized / "fast_recon" / "fast_recon" / child
    candidate = folder / filename
    if candidate.exists():
        return candidate
    legacy_candidate = legacy / filename
    if legacy_candidate.exists():
        return legacy_candidate
    return None


def _unique_nonempty(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        clean = str(value).strip()
        if clean and clean not in unique:
            unique.append(clean)
    return unique


def _best_numeric_row(rows: list[dict[str, object]], key: str, minimize: bool = True) -> int | None:
    best_index: int | None = None
    best_value = np.inf if minimize else -np.inf
    for index, row in enumerate(rows):
        value = _to_float(row.get(key))
        if not np.isfinite(value):
            continue
        if (minimize and value < best_value) or (not minimize and value > best_value):
            best_value = value
            best_index = index
    return best_index


def _matching_filter_row(rows: list[dict[str, object]], filter_name: str) -> int | None:
    target = str(filter_name).strip().lower()
    for index, row in enumerate(rows):
        if str(row.get("filter", "")).strip().lower() == target:
            return index
    return None


def _read_text_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    except Exception:
        return {}
    return values


def _read_fdk_selection(folder: Path) -> dict[str, str]:
    return _read_text_mapping(folder / "best_filter.yaml")


def _prior_selected_row(rows: list[dict[str, object]]) -> int | None:
    accepted = [
        (index, _to_float(row.get("background_noise")))
        for index, row in enumerate(rows)
        if str(row.get("accepted", "")).strip().lower() in {"true", "1", "yes"}
    ]
    finite = [(index, value) for index, value in accepted if np.isfinite(value)]
    if finite:
        return min(finite, key=lambda item: item[1])[0]
    fallback = [
        index
        for index, row in enumerate(rows)
        if str(row.get("chosen_by_fallback", "")).strip().lower() in {"true", "1", "yes"}
    ]
    if fallback:
        return fallback[0]
    return 0 if rows else None


def _qc_residual_groups(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        volume = str(row.get("volume", "")).strip().lower()
        split = str(row.get("split", "")).strip().lower()
        value = _to_float(row.get("residual"))
        if not volume or split not in {"tune", "qc"}:
            continue
        entry = grouped.setdefault(volume, {})
        if np.isfinite(value):
            entry[split] = float(value)
    ordered: dict[str, dict[str, float]] = {}
    for volume in ("fdk", "prior", "final"):
        if volume in grouped:
            ordered[volume] = grouped[volume]
    for volume, entry in grouped.items():
        if volume not in ordered:
            ordered[volume] = entry
    return ordered


def _display_volume_name(volume: str) -> str:
    name = str(volume).strip().lower()
    return {
        "fdk": "FDK",
        "prior": "Prior",
        "final": "Final",
    }.get(name, name.title())


def _fdk_preview_path(folder: Path, row: dict[str, object]) -> Path:
    filter_name = str(row.get("filter", "candidate")).replace("/", "_").replace("\\", "_")
    cutoff = _to_float(row.get("cutoff"), 1.0)
    return folder / "candidate_previews" / f"{filter_name}_cutoff_{cutoff:.3g}.png"


def _prior_display_paths(folder: Path, selected: str) -> tuple[Path | None, Path | None]:
    mapping = {
        "Selected prior": (folder / "prior_preview.png", folder / "prior_selected.npy"),
        "Confidence": (folder / "prior_confidence_preview.png", folder / "prior_confidence.npy"),
        "Difference": (folder / "prior_difference_preview.png", folder / "prior_difference.npy"),
        "Support mask": (None, folder / "support_mask.npy"),
        "Selected prior normalized": (None, folder / "prior_selected_normalized.npy"),
        "FDK normalized": (None, folder / "fdk_normalized.npy"),
    }
    return mapping.get(selected, mapping["Selected prior"])


def _load_preview_image(image_path: Path | None, npy_path: Path | None) -> np.ndarray | None:
    try:
        if image_path is not None and image_path.exists():
            return mpimg.imread(image_path)
        if npy_path is not None and npy_path.exists():
            volume = np.load(npy_path, mmap_mode="r", allow_pickle=False)
            if volume.ndim == 3:
                image = np.asarray(volume[int(volume.shape[0]) // 2], dtype=np.float32)
            elif volume.ndim == 2:
                image = np.asarray(volume, dtype=np.float32)
            else:
                return None
            finite = image[np.isfinite(image)]
            if finite.size:
                low, high = np.percentile(finite, [1, 99])
                if high > low:
                    image = np.clip((image - low) / (high - low), 0.0, 1.0)
            return image
    except Exception:
        return None
    return None


def _rows_render_signature(rows: list[dict[str, object]]) -> tuple[object, ...]:
    signature_rows: list[tuple[object, ...]] = []
    for row in rows:
        signature_rows.append(tuple((str(key), repr(row.get(key))) for key in row.keys()))
    return tuple(signature_rows)


def _load_metrics_refresh_result(request: _MetricsRefreshRequest) -> _MetricsRefreshResult:
    category = str(request.category or "")
    if category == "FDK sweep":
        root = request.category_run_roots.get("FDK sweep")
        folder = _fast_folder_for_root(root, "fdk_sweep")
        scores_path = None if folder is None else folder / "scores.csv"
        rows = _read_csv_rows_cached(scores_path)
        filters = _unique_nonempty([str(row.get("filter", "")) for row in rows])
        selected_detail = str(request.current_detail or "").strip()
        if selected_detail not in filters and filters:
            selected_detail = filters[0]
        auto_best_index = _best_numeric_row(rows, "score_total", minimize=True)
        selection_path = None if folder is None else folder / "best_filter.yaml"
        selection = _read_text_mapping_cached(selection_path)
        selected_filter = str(selection.get("best_filter") or "").strip()
        selection_mode = str(selection.get("selection_mode") or "auto_score").strip()
        highlight_index = _matching_filter_row(rows, selected_filter) if selected_filter else auto_best_index
        if highlight_index is None:
            highlight_index = auto_best_index
        best_text = ""
        if auto_best_index is not None:
            best = rows[auto_best_index]
            if selection_mode == "manual_override" and selected_filter:
                best_text = f" Auto score winner: {best.get('filter', '')}; selected filter: {selected_filter}."
            else:
                best_text = f" Best: {best.get('filter', '')} (score {best.get('score_total', '')})."
        filtered_rows = [row for row in rows if str(row.get("filter", "")) == selected_detail]
        preview_items: list[tuple[dict[str, str], Path, np.ndarray | None]] = []
        preview_tokens: list[tuple[str, object]] = []
        if folder is not None:
            for row in filtered_rows:
                path = _fdk_preview_path(folder, row)
                preview_items.append((row, path, _load_preview_image_cached(path, None)))
                preview_tokens.append((str(path), _preview_stamp(path, None)))
        return _MetricsRefreshResult(
            request_id=request.request_id,
            category=category,
            detail_choices=filters,
            detail_label="Filter",
            selected_detail=selected_detail,
            status_text=f"FDK sweep candidates: {len(rows)}.{best_text}" if rows else "No FDK sweep scores are available yet.",
            table_note_text="FDK score_total is a cost: lower is better. Highlighted row is the filter saved for full-resolution FDK.",
            rows=rows,
            highlight_index=highlight_index,
            render_kind="fdk",
            render_payload={"preview_items": preview_items},
            render_key=(
                category,
                _path_stamp(scores_path),
                _path_stamp(selection_path),
                selected_detail,
                tuple(preview_tokens),
            ),
        )
    if category == "Make prior":
        root = request.category_run_roots.get("Make prior")
        folder = _fast_folder_for_root(root, "prior")
        metrics_path = None if folder is None else folder / "prior_metrics.csv"
        rows = _read_csv_rows_cached(metrics_path)
        choices = [
            "Selected prior",
            "Confidence",
            "Difference",
            "Support mask",
            "Selected prior normalized",
            "FDK normalized",
        ]
        selected_detail = str(request.current_detail or "").strip() or choices[0]
        if selected_detail not in choices:
            selected_detail = choices[0]
        image_path, npy_path = _prior_display_paths(folder, selected_detail) if folder is not None else (None, None)
        return _MetricsRefreshResult(
            request_id=request.request_id,
            category=category,
            detail_choices=choices,
            detail_label="Display",
            selected_detail=selected_detail,
            status_text=f"Prior candidates: {len(rows)}." if rows else "No prior metrics are available yet.",
            table_note_text="",
            rows=rows,
            highlight_index=_prior_selected_row(rows),
            render_kind="prior",
            render_payload={
                "folder_available": folder is not None,
                "preview_image": _load_preview_image_cached(image_path, npy_path),
            },
            render_key=(
                category,
                _path_stamp(metrics_path),
                selected_detail,
                _preview_stamp(image_path, npy_path),
            ),
        )
    if category == "MBIR":
        rows = list(request.live_mbir_rows)
        source_key: object = ("live", _rows_render_signature(rows))
        if not rows and not request.waiting_for_live_mbir_metrics:
            path = _standard_metrics_csv_path(request.category_run_roots.get("MBIR"))
            rows = _read_csv_rows_cached(path)
            source_key = ("file", _path_stamp(path))
        solver_text = _solver_display_name_from_key(_solver_from_rows(rows))
        if rows:
            status_text = f"MBIR iterations: {len(rows)} | solver {solver_text}."
        elif request.waiting_for_live_mbir_metrics:
            status_text = "Waiting for MBIR metrics."
        else:
            status_text = "No MBIR metrics are available yet."
        return _MetricsRefreshResult(
            request_id=request.request_id,
            category=category,
            detail_choices=[],
            detail_label="",
            selected_detail="",
            status_text=status_text,
            table_note_text=(
                "Standard MBIR shows solver-specific panels only when those values are available. "
                "ADMM residual and CG plots appear automatically for ADMM runs."
            ),
            rows=rows,
            highlight_index=None,
            render_kind="mbir",
            render_payload={},
            render_key=(category, source_key, bool(request.waiting_for_live_mbir_metrics)),
        )
    if category == "MBIR-lite":
        rows = list(request.live_mbir_lite_rows)
        source_key = ("live", _rows_render_signature(rows))
        if not rows and not request.waiting_for_live_mbir_lite_metrics:
            folder = _fast_folder_for_root(request.category_run_roots.get("MBIR-lite"), "mbir_lite")
            metrics_path = None if folder is None else folder / "metrics.csv"
            rows = _read_csv_rows_cached(metrics_path)
            source_key = ("file", _path_stamp(metrics_path))
        return _MetricsRefreshResult(
            request_id=request.request_id,
            category=category,
            detail_choices=[],
            detail_label="",
            selected_detail="",
            status_text=f"MBIR-lite iterations: {len(rows)}." if rows else "Waiting for MBIR-lite metrics.",
            table_note_text="",
            rows=rows,
            highlight_index=None,
            render_kind="mbir_lite",
            render_payload={},
            render_key=(category, source_key, bool(request.waiting_for_live_mbir_lite_metrics)),
        )
    if category == "QC report":
        root = request.category_run_roots.get("QC report")
        folder = _fast_folder_for_root(root, "qc")
        metrics_path = None if folder is None else folder / "qc_metrics.csv"
        preview_path = None if folder is None else folder / "preview_panel.png"
        rows = _read_csv_rows_cached(metrics_path)
        return _MetricsRefreshResult(
            request_id=request.request_id,
            category=category,
            detail_choices=[],
            detail_label="",
            selected_detail="",
            status_text=f"QC metric rows: {len(rows)}." if rows else "No QC report metrics are available yet.",
            table_note_text=(
                "Lower residual is better. Compare Final against FDK and Prior on both tune and QC splits. "
                "QC bar annotations show the percent change of each held-out QC residual relative to the matching tune residual."
            ),
            rows=rows,
            highlight_index=None,
            render_kind="qc",
            render_payload={"preview_image": _load_preview_image_cached(preview_path, None)},
            render_key=(category, _path_stamp(metrics_path), _preview_stamp(preview_path, None)),
        )
    return _MetricsRefreshResult(
        request_id=request.request_id,
        category=category,
        detail_choices=[],
        detail_label="",
        selected_detail="",
        status_text="No metrics category selected.",
        table_note_text="",
        rows=[],
        highlight_index=None,
        render_kind="message",
        render_payload={},
        render_key=(category, "message"),
    )


def _as_display_image(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    array = np.asarray(image)
    if array.ndim == 3:
        rgb = np.asarray(array[..., :3], dtype=np.float32)
        if rgb.shape[-1] < 3:
            return np.asarray(rgb[..., 0], dtype=np.float32)
        array = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    elif array.ndim != 2:
        return None
    return np.asarray(array, dtype=np.float32)


def _infer_metrics_category(metrics: list[object]) -> str:
    if any(hasattr(metric, "prior_anchor_term") or hasattr(metric, "objective_total") for metric in metrics or []):
        return "MBIR-lite"
    return "MBIR"


def _mbir_rows_from_objects(metrics: list[object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous_elapsed = 0.0
    for metric in metrics or []:
        row = _mbir_metric_row_from_object(metric, previous_elapsed)
        elapsed_value = _to_float(row.get("elapsed_s"), previous_elapsed)
        previous_elapsed = elapsed_value if np.isfinite(elapsed_value) else previous_elapsed
        rows.append(row)
    return rows


def _mbir_metric_row_from_object(metric: object, previous_elapsed: float = 0.0) -> dict[str, object]:
    data_fidelity = getattr(metric, "data_fidelity", "")
    data_fidelity_value = _to_float(data_fidelity)
    elapsed = getattr(metric, "elapsed_s", "")
    elapsed_value = _to_float(elapsed)
    row: dict[str, object] = {
        "iteration": getattr(metric, "iteration", ""),
        "objective": getattr(metric, "objective", getattr(metric, "objective_total", "")),
        "data_fidelity": data_fidelity,
        "data_residual": float(np.sqrt(max(2.0 * data_fidelity_value, 0.0))) if np.isfinite(data_fidelity_value) else "",
        "tv_term": getattr(metric, "tv_term", ""),
        "primal_residual": getattr(metric, "primal_residual", ""),
        "dual_residual": getattr(metric, "dual_residual", ""),
        "relative_x_change": getattr(metric, "relative_x_change", getattr(metric, "relative_change", "")),
        "cg_residual": getattr(metric, "cg_residual", ""),
        "cg_iterations": getattr(metric, "cg_iterations", ""),
        "elapsed_s": elapsed,
        "solver": getattr(metric, "solver", ""),
    }
    if np.isfinite(elapsed_value):
        row["iteration_time_s"] = max(0.0, elapsed_value - max(float(previous_elapsed), 0.0))
    else:
        row["iteration_time_s"] = ""
    return row


def _mbir_lite_metric_row_from_object(metric: object) -> dict[str, object]:
    return {
        "iteration": getattr(metric, "iteration", ""),
        "objective_total": getattr(metric, "objective_total", getattr(metric, "objective", "")),
        "data_weighted": getattr(metric, "data_weighted", getattr(metric, "data_fidelity", "")),
        "tv_term": getattr(metric, "tv_term", ""),
        "prior_anchor_term": getattr(metric, "prior_anchor_term", ""),
        "qc_anchor_residual": getattr(metric, "qc_anchor_residual", ""),
        "relative_change": getattr(metric, "relative_change", getattr(metric, "relative_x_change", "")),
        "elapsed_s": getattr(metric, "elapsed_s", ""),
    }


def _to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _format_table_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    numeric = _to_float(value)
    if np.isfinite(numeric) and str(value).strip() not in {"0", "1"}:
        return f"{numeric:.6g}"
    return str(value)


def _rows_have_finite_values(rows: list[dict[str, object]], key: str) -> bool:
    return any(np.isfinite(_to_float(row.get(key))) for row in rows)


def _solver_from_rows(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "unknown"
    return str(rows[-1].get("solver", "unknown") or "unknown").strip().lower()


def _solver_display_name_from_key(solver: str) -> str:
    key = str(solver or "unknown").strip().lower()
    return {
        "admm": "ADMM",
        "pdhg_low_memory": "PDHG",
        "streaming_subset_tv": "Streaming subset TV",
        "anchored_streaming_subset_tv": "Anchored subset TV",
    }.get(key, key.replace("_", " ").title() or "Unknown")


def _panel_specs_for_count(figure: Figure, count: int) -> list[object]:
    panel_count = max(1, int(count))
    if panel_count <= 3:
        rows, cols = 1, panel_count
    elif panel_count == 4:
        rows, cols = 2, 2
    elif panel_count <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = int(np.ceil(panel_count / 3.0)), 3
    grid = figure.add_gridspec(rows, cols, hspace=0.3, wspace=0.24)
    specs: list[object] = []
    full_rows = panel_count // cols
    remainder = panel_count % cols
    for row in range(full_rows):
        for col in range(cols):
            specs.append(grid[row, col])
    if remainder:
        row = full_rows
        if remainder == cols:
            for col in range(cols):
                specs.append(grid[row, col])
        elif remainder == 1:
            specs.append(grid[row, :])
        else:
            subgrid = grid[row, :].subgridspec(1, remainder, wspace=0.24)
            for col in range(remainder):
                specs.append(subgrid[0, col])
    return specs


def _plot_metric_panel(
    axis,
    iterations: np.ndarray,
    rows: list[dict[str, object]],
    key: str,
    title: str,
    color: str,
    log_y: bool,
    x_label: str,
) -> None:
    style_axis(axis, grid=True)
    values = np.asarray([_to_float(row.get(key)) for row in rows], dtype=np.float64)
    finite = np.isfinite(iterations) & np.isfinite(values)
    if np.any(finite):
        plot_values = np.maximum(values[finite], 1e-30) if log_y else values[finite]
        axis.plot(iterations[finite], plot_values, linewidth=1.45, marker="o", markersize=3, color=color)
        if log_y:
            axis.set_yscale("log")
        min_iteration = float(np.min(iterations[finite]))
        max_iteration = float(np.max(iterations[finite]))
        if max_iteration <= min_iteration:
            axis.set_xlim(min_iteration - 0.5, max_iteration + 0.5)
        else:
            axis.set_xlim(min_iteration, max_iteration)
        axis.margins(y=0.16)
    else:
        axis.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=axis.transAxes, color=MUTED_TEXT_COLOR)
        axis.axis("off")
        return
    axis.set_title(title, color=TEXT_COLOR, fontsize=11)
    axis.set_xlabel(x_label, fontsize=9)
    axis.tick_params(labelsize=8)


class DeviceMonitorWidget(QWidget):
    def __init__(self, history_limit: int = 300) -> None:
        super().__init__()
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        style_figure(self.figure)
        style_canvas(self.canvas)
        self.status_label = QLabel("Waiting for the first NVIDIA GPU sample.")
        self.status_label.setWordWrap(True)
        self.device_table = QTableWidget(0, 8)
        self.device_table.setHorizontalHeaderLabels(
            ["GPU", "Name", "Selected", "Compute %", "Mem BW %", "VRAM", "VRAM %", "Temp / Power"]
        )
        self.device_table.horizontalHeader().setStretchLastSection(True)
        self._history: list[DeviceMonitorSnapshot] = []
        self._selected_gpu_ids: tuple[int, ...] = ()
        self._history_limit = max(30, int(history_limit))
        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.device_table)
        layout.addWidget(self.canvas)
        self.show_message("Waiting for the first NVIDIA GPU sample.")

    def has_history(self) -> bool:
        return bool(self._history)

    def set_selected_gpu_ids(self, gpu_ids) -> None:
        unique_ids = []
        for gpu_id in gpu_ids or ():
            gpu_id = int(gpu_id)
            if gpu_id not in unique_ids:
                unique_ids.append(gpu_id)
        self._selected_gpu_ids = tuple(unique_ids)
        self._refresh_current_table()
        self._draw_history()

    def set_status_message(self, message: str) -> None:
        self.status_label.setText(message)

    def clear_history(self, message: str = "Waiting for the first NVIDIA GPU sample.") -> None:
        self._history = []
        self.show_message(message)

    def show_message(self, message: str) -> None:
        self.status_label.setText(message)
        self.device_table.setRowCount(0)
        self.figure.clear()
        style_figure(self.figure)
        axis = self.figure.add_subplot(111)
        style_axis(axis, grid=False)
        axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes, color=TEXT_COLOR)
        axis.axis("off")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def append_snapshot(self, snapshot: DeviceMonitorSnapshot) -> None:
        if not snapshot.gpus:
            if not self._history:
                self.show_message("No NVIDIA GPU samples are available yet.")
            return
        self._history.append(snapshot)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit :]
        self.status_label.setText(self._status_text(snapshot))
        self._refresh_current_table(snapshot)
        self._draw_history()

    def _status_text(self, snapshot: DeviceMonitorSnapshot) -> str:
        timestamp_text = datetime.fromtimestamp(float(snapshot.timestamp_s)).strftime("%H:%M:%S")
        history_seconds = 0.0
        if len(self._history) >= 2:
            history_seconds = max(0.0, float(self._history[-1].timestamp_s) - float(self._history[0].timestamp_s))
        return (
            f"Live NVIDIA polling via {snapshot.source} | last update {timestamp_text} | "
            f"{len(snapshot.gpus)} GPU(s) | history {history_seconds:.0f} s"
        )

    def _refresh_current_table(self, snapshot: DeviceMonitorSnapshot | None = None) -> None:
        if snapshot is None:
            snapshot = self._history[-1] if self._history else None
        if snapshot is None:
            self.device_table.setRowCount(0)
            return
        gpus = sorted(snapshot.gpus, key=lambda sample: sample.gpu_id)
        self.device_table.setRowCount(len(gpus))
        selected_ids = set(self._selected_gpu_ids)
        for row, sample in enumerate(gpus):
            selected = not selected_ids or sample.gpu_id in selected_ids
            values = [
                f"GPU {sample.gpu_id}",
                sample.name,
                "Yes" if sample.gpu_id in selected_ids else "",
                _display_number(sample.utilization_gpu_pct, "%", digits=0),
                _display_number(sample.utilization_memory_pct, "%", digits=0),
                _display_memory(sample),
                _display_number(sample.memory_percent, "%", digits=0),
                _display_thermal_power(sample),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if column in {0, 2, 3, 4, 5, 6, 7}:
                    item.setTextAlignment(int(Qt.AlignCenter))
                if selected:
                    item.setForeground(Qt.white)
                self.device_table.setItem(row, column, item)

    def _draw_history(self) -> None:
        self.figure.clear()
        style_figure(self.figure)
        if not self._history:
            axis = self.figure.add_subplot(111)
            style_axis(axis, grid=False)
            axis.text(0.5, 0.5, "No device-monitor history yet", ha="center", va="center", transform=axis.transAxes, color=TEXT_COLOR)
            axis.axis("off")
            self.figure.tight_layout()
            self.canvas.draw_idle()
            return

        axes = self.figure.subplots(2, 1, sharex=True)
        axes = np.atleast_1d(axes)
        util_axis, memory_axis = axes[0], axes[1]
        for axis in axes:
            style_axis(axis, grid=True)
            axis.tick_params(labelsize=7)

        start_time = float(self._history[0].timestamp_s)
        gpu_ids = sorted({sample.gpu_id for snapshot in self._history for sample in snapshot.gpus})
        selected_ids = set(self._selected_gpu_ids)

        for color_index, gpu_id in enumerate(gpu_ids):
            samples = [sample for snapshot in self._history for sample in snapshot.gpus if sample.gpu_id == gpu_id]
            if not samples:
                continue
            times = np.asarray(
                [
                    float(snapshot.timestamp_s) - start_time
                    for snapshot in self._history
                    for sample in snapshot.gpus
                    if sample.gpu_id == gpu_id
                ],
                dtype=np.float64,
            )
            compute_util = np.asarray(
                [
                    np.nan if sample.utilization_gpu_pct is None else float(sample.utilization_gpu_pct)
                    for sample in samples
                ],
                dtype=np.float64,
            )
            memory_percent = np.asarray(
                [
                    np.nan if sample.memory_percent is None else float(sample.memory_percent)
                    for sample in samples
                ],
                dtype=np.float64,
            )
            color = _DEVICE_COLORS[color_index % len(_DEVICE_COLORS)]
            is_selected = not selected_ids or gpu_id in selected_ids
            line_width = 2.0 if is_selected else 1.0
            alpha = 1.0 if is_selected else 0.35
            label = f"GPU {gpu_id}"
            util_axis.plot(times, compute_util, color=color, linewidth=line_width, alpha=alpha, label=label)
            memory_axis.plot(times, memory_percent, color=color, linewidth=line_width, alpha=alpha, label=label)

        util_axis.set_title("GPU compute utilization", color=TEXT_COLOR, fontsize=9)
        util_axis.set_ylabel("Utilization %", fontsize=8)
        util_axis.set_ylim(0.0, 100.0)
        memory_axis.set_title("VRAM occupancy", color=TEXT_COLOR, fontsize=9)
        memory_axis.set_ylabel("VRAM %", fontsize=8)
        memory_axis.set_xlabel("Elapsed seconds", fontsize=8)
        memory_axis.set_ylim(0.0, 100.0)
        _style_legend(util_axis.legend(loc="upper right", fontsize=7))
        self.figure.tight_layout(pad=1.0)
        self.canvas.draw_idle()


def _metric_label(metric_key: str) -> str:
    return {
        "combined_score": "Combined score",
        "gradient_energy": "Gradient energy",
        "laplacian_variance": "Laplacian variance",
        "entropy": "Entropy",
    }.get(metric_key, metric_key)


def _solver_from_metrics(metrics: list[ADMMIterationMetrics]) -> str:
    if not metrics:
        return "unknown"
    return str(getattr(metrics[-1], "solver", "admm") or "admm").strip().lower()


def _style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor(PANEL_BACKGROUND)
    frame.set_edgecolor(GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(MUTED_TEXT_COLOR)


def _display_memory(sample: GPULiveSample) -> str:
    if sample.memory_used_gb is None or sample.memory_total_gb is None:
        return "-"
    return f"{sample.memory_used_gb:.1f} / {sample.memory_total_gb:.1f} GB"


def _display_thermal_power(sample: GPULiveSample) -> str:
    temperature = _display_number(sample.temperature_c, " C", digits=0)
    power = _display_number(sample.power_w, " W", digits=1)
    if temperature == "-" and power == "-":
        return "-"
    if temperature == "-":
        return power
    if power == "-":
        return temperature
    return f"{temperature} / {power}"


def _display_number(value: float | None, suffix: str = "", digits: int = 0) -> str:
    if value is None or not np.isfinite(float(value)):
        return "-"
    numeric = float(value)
    if digits <= 0:
        return f"{numeric:.0f}{suffix}"
    return f"{numeric:.{digits}f}{suffix}"
