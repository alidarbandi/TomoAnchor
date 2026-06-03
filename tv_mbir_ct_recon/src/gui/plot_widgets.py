from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import numpy as np

from ..admm_tv_mbir import ADMMIterationMetrics
from ..device_monitor import DeviceMonitorSnapshot, GPULiveSample
from .qt_compat import QColor, QComboBox, QHBoxLayout, QLabel, QSplitter, QTableWidget, QTableWidgetItem, QTimer, QVBoxLayout, QWidget, Qt
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


class MetricsPlotWidget(QWidget):
    def __init__(self, on_image_changed=None) -> None:
        super().__init__()
        self.on_image_changed = on_image_changed
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        style_figure(self.figure)
        style_canvas(self.canvas)
        self.category_combo = QComboBox()
        self.category_combo.addItems(["FDK sweep", "Make prior", "MBIR-lite", "QC report"])
        self.detail_label = QLabel("Filter")
        self.detail_combo = QComboBox()
        self.status_label = QLabel("No fast reconstruction run is selected.")
        self.status_label.setWordWrap(True)
        self.table = QTableWidget(0, 0)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table_note_label = QLabel("")
        self.table_note_label.setWordWrap(True)
        self._run_folder: Path | None = None
        self._live_mbir_metrics: list[object] = []
        self._block_detail_signal = False
        self._current_image: np.ndarray | None = None
        self._levels: tuple[float, float] | None = None
        self._zoom_factor = 1.0
        self._pan_center: tuple[float, float] | None = None
        self._image_signature = ""
        self._image_artists: list[tuple[object, np.ndarray]] = []
        self._pan_drag: dict[str, object] | None = None
        self._last_layout_category = ""

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
        self.metrics_splitter.addWidget(self.canvas)
        self.metrics_splitter.addWidget(table_panel)
        self.metrics_splitter.setStretchFactor(0, 5)
        self.metrics_splitter.setStretchFactor(1, 2)
        self.metrics_splitter.setSizes([640, 220])
        layout.addWidget(self.metrics_splitter, 1)

        self.category_combo.currentTextChanged.connect(lambda *_: self._refresh_category(sync_detail=True))
        self.detail_combo.currentTextChanged.connect(lambda *_: self._detail_changed())
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1500)
        self._poll_timer.timeout.connect(self.refresh_from_files)
        self._poll_timer.start()
        self.canvas.mpl_connect("button_press_event", self._on_pan_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_pan_motion)
        self.canvas.mpl_connect("button_release_event", self._on_pan_release)
        self._refresh_category(sync_detail=True)

    def set_run_folder(self, run_folder: str | Path | None) -> None:
        self._run_folder = None if not run_folder else _normalize_fast_run_root(Path(run_folder))
        self.refresh_from_files()

    def select_category(self, category: str) -> None:
        if category != self.category_combo.currentText():
            self.category_combo.setCurrentText(category)
        else:
            self._refresh_category(sync_detail=True)

    def refresh_from_files(self) -> None:
        self._refresh_category(sync_detail=True)

    def show_metrics(self, metrics: list[ADMMIterationMetrics]) -> None:
        self._live_mbir_metrics = list(metrics or [])
        if not self._live_mbir_metrics and self.category_combo.currentText() == "MBIR-lite":
            self._refresh_category(sync_detail=True)
            return
        if self.category_combo.currentText() == "MBIR-lite":
            self._show_mbir_lite()

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
            self.metrics_splitter.setSizes([820, 120])
        else:
            self.metrics_splitter.setSizes([640, 220])

    def _refresh_category(self, sync_detail: bool = False) -> None:
        category = self.category_combo.currentText()
        self._apply_category_layout(category)
        if category == "FDK sweep":
            self._show_fdk_sweep(sync_detail)
        elif category == "Make prior":
            self._show_prior(sync_detail)
        elif category == "MBIR-lite":
            self._show_mbir_lite()
        elif category == "QC report":
            self._show_qc_report()
        else:
            self._draw_message("No metrics category selected.")

    def _show_fdk_sweep(self, sync_detail: bool) -> None:
        folder = self._fast_folder("fdk_sweep")
        rows = _read_csv_rows(folder / "scores.csv") if folder is not None else []
        filters = _unique_nonempty([str(row.get("filter", "")) for row in rows])
        if sync_detail:
            self._set_detail_choices(filters, "Filter")
        preview_filter = self.detail_combo.currentText().strip() if filters else ""
        if preview_filter not in filters and filters:
            preview_filter = filters[0]
        auto_best_index = _best_numeric_row(rows, "score_total", minimize=True)
        selection = _read_fdk_selection(folder) if folder is not None else {}
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
        self.status_label.setText(f"FDK sweep candidates: {len(rows)}.{best_text}" if rows else "No FDK sweep scores are available yet.")
        self.table_note_label.setText("FDK score_total is a cost: lower is better. Highlighted row is the filter saved for full-resolution FDK.")
        self._fill_table(rows, highlight_index=highlight_index)
        if not rows:
            self._draw_message("Waiting for FDK sweep scores.csv.")
            return
        selected_rows = [row for row in rows if str(row.get("filter", "")) == preview_filter]
        self._draw_fdk_candidate_previews(folder, preview_filter, selected_rows)

    def _show_prior(self, sync_detail: bool) -> None:
        folder = self._fast_folder("prior")
        rows = _read_csv_rows(folder / "prior_metrics.csv") if folder is not None else []
        choices = [
            "Selected prior",
            "Confidence",
            "Difference",
            "Support mask",
            "Selected prior normalized",
            "FDK normalized",
        ]
        if sync_detail:
            self._set_detail_choices(choices, "Display")
        selected = self.detail_combo.currentText().strip() or choices[0]
        highlight = _prior_selected_row(rows)
        self.status_label.setText(f"Prior candidates: {len(rows)}." if rows else "No prior metrics are available yet.")
        self.table_note_label.setText("")
        self._fill_table(rows, highlight_index=highlight)
        if folder is None:
            self._draw_message("No fast reconstruction run is selected.")
            return
        self._draw_prior_panel(folder, selected, rows)

    def _show_mbir_lite(self) -> None:
        self._set_detail_choices([])
        rows = [_metric_row_from_object(metric) for metric in self._live_mbir_metrics]
        if not rows:
            folder = self._fast_folder("mbir_lite")
            rows = _read_csv_rows(folder / "metrics.csv") if folder is not None else []
        self.status_label.setText(f"MBIR-lite iterations: {len(rows)}." if rows else "Waiting for MBIR-lite metrics.")
        self.table_note_label.setText("")
        self._fill_table(rows)
        self._draw_mbir_lite(rows)

    def _show_qc_report(self) -> None:
        self._set_detail_choices([])
        folder = self._fast_folder("qc")
        rows = _read_csv_rows(folder / "qc_metrics.csv") if folder is not None else []
        self.status_label.setText(f"QC metric rows: {len(rows)}." if rows else "No QC report metrics are available yet.")
        self.table_note_label.setText("")
        self._fill_table(rows)
        self._draw_qc_report(folder, rows)

    def _draw_fdk_candidate_previews(self, folder: Path | None, selected_filter: str, rows: list[dict[str, str]]) -> None:
        self._start_image_display(f"fdk_sweep:{folder}")
        self.figure.clear()
        style_figure(self.figure)
        if folder is None or not rows:
            self._draw_message("Select a filter after FDK sweep scores are available.")
            return
        preview_paths = [_fdk_preview_path(folder, row) for row in rows]
        cols = min(3, max(1, len(preview_paths)))
        axes = np.asarray(self.figure.subplots(1, cols)).reshape(-1)
        for axis, row, path in zip(axes, rows, preview_paths):
            style_axis(axis)
            image = _load_preview_image(path, None)
            if image is not None:
                score = _to_float(row.get("score_total"))
                suffix = f" | score {score:.4g}" if np.isfinite(score) else ""
                self._register_image(axis, image, f"{selected_filter}{suffix}")
            else:
                axis.text(0.5, 0.5, f"Preview not found\n{path.name}", ha="center", va="center", transform=axis.transAxes, color=TEXT_COLOR)
                axis.axis("off")
        for axis in axes[len(rows) :]:
            axis.axis("off")
        self.figure.tight_layout()
        self._finish_image_display()
        self.canvas.draw_idle()

    def _draw_prior_panel(self, folder: Path, selected: str, rows: list[dict[str, str]]) -> None:
        self._start_image_display(f"prior:{selected}")
        image_path, npy_path = _prior_display_paths(folder, selected)
        self.figure.clear()
        style_figure(self.figure)
        grid = self.figure.add_gridspec(2, 1, height_ratios=[4.2, 1.15], hspace=0.28)
        image_axis = self.figure.add_subplot(grid[0, 0])
        plot_axis = self.figure.add_subplot(grid[1, 0])
        style_axis(image_axis)
        image = _load_preview_image(image_path, npy_path)
        if image is None:
            image_axis.text(0.5, 0.5, f"{selected} is not available yet.", ha="center", va="center", transform=image_axis.transAxes, color=TEXT_COLOR)
            image_axis.axis("off")
        else:
            self._register_image(image_axis, image, selected)
        self._plot_prior_metrics(plot_axis, rows)
        self.figure.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.08, hspace=0.34)
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

    def _draw_mbir_lite(self, rows: list[dict[str, object]]) -> None:
        self._clear_image_display(emit=False)
        self.figure.clear()
        style_figure(self.figure)
        if not rows:
            self._draw_message("Waiting for MBIR-lite live metrics.")
            return
        panels = [
            ("objective_total", "Objective", ACCENT_COLOR, True),
            ("data_weighted", "Data weighted", "#80D6B6", True),
            ("tv_term", "TV term", "#FFCC80", True),
            ("prior_anchor_term", "Prior anchor", "#FF8A80", True),
            ("qc_anchor_residual", "QC anchor residual", "#A8C7F7", True),
            ("relative_change", "Relative change", "#DDE8F7", True),
            ("elapsed_s", "Elapsed seconds", "#B39DDB", False),
        ]
        axes = np.asarray(self.figure.subplots(2, 4)).reshape(-1)
        iterations = np.asarray([_to_float(row.get("iteration"), index + 1) for index, row in enumerate(rows)], dtype=np.float64)
        for axis, (key, title, color, log_y) in zip(axes, panels):
            style_axis(axis, grid=True)
            values = np.asarray([_to_float(row.get(key)) for row in rows], dtype=np.float64)
            finite = np.isfinite(iterations) & np.isfinite(values)
            if np.any(finite):
                plot_values = np.maximum(values, 1e-30) if log_y else values
                axis.plot(iterations[finite], plot_values[finite], linewidth=1.4, marker="o", markersize=3, color=color)
                if log_y:
                    axis.set_yscale("log")
            axis.set_title(title, color=TEXT_COLOR, fontsize=9)
            axis.set_xlabel("Sweep", fontsize=8)
            axis.tick_params(labelsize=7)
        for axis in axes[len(panels) :]:
            axis.axis("off")
        self.figure.tight_layout(pad=1.0)
        self.canvas.draw_idle()
        self._emit_image_changed()

    def _draw_qc_report(self, folder: Path | None, rows: list[dict[str, str]]) -> None:
        self._start_image_display("qc")
        self.figure.clear()
        style_figure(self.figure)
        axes = np.asarray(self.figure.subplots(1, 2)).reshape(-1)
        image_axis, bar_axis = axes[0], axes[1]
        style_axis(image_axis)
        preview_path = None if folder is None else folder / "preview_panel.png"
        image = _load_preview_image(preview_path, None) if preview_path is not None else None
        if image is not None:
            self._register_image(image_axis, image, "QC preview panel")
        else:
            image_axis.text(0.5, 0.5, "QC preview_panel.png is not available yet.", ha="center", va="center", transform=image_axis.transAxes, color=TEXT_COLOR)
            image_axis.axis("off")
        style_axis(bar_axis, grid=True)
        if rows:
            labels = [f"{row.get('volume', '')}/{row.get('split', '')}" for row in rows]
            values = np.asarray([_to_float(row.get("residual")) for row in rows], dtype=np.float64)
            finite = np.isfinite(values)
            if np.any(finite):
                indices = np.arange(len(labels))[finite]
                bar_axis.bar(indices, np.maximum(values[finite], 1e-30), color=ACCENT_COLOR)
                bar_axis.set_yscale("log")
                bar_axis.set_xticks(indices)
                bar_axis.set_xticklabels([labels[index] for index in indices], rotation=35, ha="right", fontsize=7)
                bar_axis.set_title("Anchor residuals", color=TEXT_COLOR, fontsize=9)
        else:
            bar_axis.text(0.5, 0.5, "Waiting for qc_metrics.csv", ha="center", va="center", transform=bar_axis.transAxes, color=TEXT_COLOR)
            bar_axis.axis("off")
        self.figure.tight_layout()
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

    def _fast_folder(self, name: str) -> Path | None:
        if self._run_folder is None:
            return None
        folder = self._run_folder / "fast_recon" / name
        legacy = self._run_folder / "fast_recon" / "fast_recon" / name
        if not folder.exists() and legacy.exists():
            return legacy
        return folder

    def _set_detail_choices(self, choices: list[str], label: str = "") -> None:
        current = self.detail_combo.currentText()
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


def _read_fdk_selection(folder: Path) -> dict[str, str]:
    path = folder / "best_filter.yaml"
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


def _metric_row_from_object(metric: object) -> dict[str, object]:
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
