from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from typing import Callable, Sequence

from .qt_compat import QComboBox, QHBoxLayout, QLabel, QSlider, QSpinBox, QVBoxLayout, QWidget, Qt
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


class VolumePreviewWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        style_figure(self.figure)
        style_canvas(self.canvas)
        self._current_image: np.ndarray | None = None
        self._current_volume: np.ndarray | None = None
        self._current_title = ""
        self._current_mode = "message"
        self._levels: tuple[float, float] | None = None
        self._zoom_factor = 1.0
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.show_message("No volume")

    def show_message(self, text: str) -> None:
        self.figure.clear()
        style_figure(self.figure)
        self._current_image = None
        self._current_volume = None
        self._current_title = text
        self._current_mode = "message"
        self._levels = None
        self._zoom_factor = 1.0
        axis = self.figure.add_subplot(111)
        style_axis(axis)
        axis.text(0.5, 0.5, text, ha="center", va="center", transform=axis.transAxes, color=TEXT_COLOR)
        axis.axis("off")
        self.canvas.draw_idle()

    def show_image(
        self,
        image: np.ndarray | None,
        title: str = "Image preview",
        preserve_zoom: bool = False,
    ) -> None:
        if image is None:
            self.show_message("No image")
            return
        array = np.asarray(image, dtype=np.float32)
        previous_zoom = self._zoom_factor
        self._current_image = array
        self._current_volume = None
        self._current_title = title
        self._current_mode = "image"
        self._levels = self._auto_limits(array)
        self._zoom_factor = previous_zoom if preserve_zoom else 1.0
        self._draw_current()

    def show_volume(
        self,
        volume: np.ndarray,
        title: str = "Volume preview",
        preserve_zoom: bool = False,
    ) -> None:
        if volume is None:
            self.show_message("No volume")
            return
        vol = np.asarray(volume, dtype=np.float32)
        previous_zoom = self._zoom_factor
        self._current_volume = vol
        self._current_image = vol[vol.shape[0] // 2]
        self._current_title = title
        self._current_mode = "volume"
        self._levels = self._auto_limits(vol)
        self._zoom_factor = previous_zoom if preserve_zoom else 1.0
        self._draw_current()

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
        self._draw_current()

    def reset_auto_levels(self) -> tuple[float, float] | None:
        data = self._current_volume if self._current_volume is not None else self._current_image
        if data is None:
            return None
        self._levels = self._auto_limits(data)
        self._draw_current()
        return self._levels

    def current_image(self) -> np.ndarray | None:
        return self._current_image

    def current_levels(self) -> tuple[float, float] | None:
        return self._levels

    def current_zoom_factor(self) -> float:
        return self._zoom_factor

    def set_zoom_factor(self, zoom_factor: float) -> float:
        self._zoom_factor = min(max(float(zoom_factor), 1.0), 32.0)
        self._draw_current()
        return self._zoom_factor

    def fit_to_view(self) -> float:
        return self.set_zoom_factor(1.0)

    def _draw_current(self) -> None:
        if self._current_mode == "image" and self._current_image is not None:
            self._draw_image(self._current_image, self._current_title)
        elif self._current_mode == "volume" and self._current_volume is not None:
            self._draw_volume(self._current_volume, self._current_title)

    def _draw_image(self, array: np.ndarray, title: str) -> None:
        self.figure.clear()
        style_figure(self.figure)
        axis = self.figure.add_subplot(111)
        style_axis(axis)
        low, high = self._levels or self._auto_limits(array)
        view = axis.imshow(array, cmap="gray", vmin=low, vmax=high, origin="upper")
        _apply_image_zoom(axis, array, self._zoom_factor)
        axis.set_title(title, color=TEXT_COLOR)
        axis.axis("off")
        colorbar = self.figure.colorbar(view, ax=axis, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(colors=MUTED_TEXT_COLOR)
        colorbar.outline.set_edgecolor(GRID_COLOR)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _draw_volume(self, vol: np.ndarray, title: str) -> None:
        self.figure.clear()
        style_figure(self.figure)
        axes = self.figure.subplots(1, 3)
        low, high = self._levels or self._auto_limits(vol)
        planes = [
            (vol[vol.shape[0] // 2], "Axial"),
            (vol[:, vol.shape[1] // 2, :], "Coronal"),
            (vol[:, :, vol.shape[2] // 2], "Sagittal"),
        ]
        for axis, (image, label) in zip(axes, planes):
            style_axis(axis)
            axis.imshow(image, cmap="gray", vmin=low, vmax=high)
            _apply_image_zoom(axis, image, self._zoom_factor)
            axis.set_title(label, color=TEXT_COLOR)
            axis.axis("off")
        self.figure.suptitle(title, color=TEXT_COLOR)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _auto_limits(self, array: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(array)[np.isfinite(array)]
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


class TomogramViewWidget(QWidget):
    """Single-plane tomogram browser for reconstructed volumes."""

    VIEW_LABELS = ("Axial", "Coronal", "Sagittal")

    def __init__(
        self,
        on_image_changed: Callable[[], None] | None = None,
        on_tomogram_selected: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.on_image_changed = on_image_changed
        self.on_tomogram_selected = on_tomogram_selected
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        style_figure(self.figure)
        style_canvas(self.canvas)
        self.tomogram_combo = QComboBox()
        self.tomogram_combo.addItem("No tomograms available", "")
        self.tomogram_combo.setEnabled(False)
        self.view_combo = QComboBox()
        self.view_combo.addItems(self.VIEW_LABELS)
        self.slice_spin = QSpinBox()
        self.slice_spin.setRange(0, 0)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_label = QLabel("No tomogram loaded")
        self.slice_label.setWordWrap(True)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Tomogram"))
        controls.addWidget(self.tomogram_combo, 1)
        controls.addWidget(QLabel("View"))
        controls.addWidget(self.view_combo)
        controls.addWidget(QLabel("Slice"))
        controls.addWidget(self.slice_spin)
        controls.addWidget(self.slice_slider, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.slice_label)
        layout.addWidget(self.canvas, 1)

        self._volume: np.ndarray | None = None
        self._title = "Tomogram"
        self._levels: tuple[float, float] | None = None
        self._zoom_factor = 1.0
        self._slice_indices = {"Axial": 0, "Coronal": 0, "Sagittal": 0}
        self._cached_images: dict[str, np.ndarray] = {}
        self._block_controls = False
        self._block_tomogram_selection = False

        self.tomogram_combo.currentIndexChanged.connect(self._tomogram_selection_changed)
        self.view_combo.currentTextChanged.connect(self._view_changed)
        self.slice_spin.valueChanged.connect(self._slice_spin_changed)
        self.slice_slider.valueChanged.connect(self._slice_slider_changed)
        self.show_message("No tomogram")

    def set_tomogram_choices(
        self,
        choices: Sequence[tuple[str, str]],
        selected_id: str | None = None,
    ) -> None:
        self._block_tomogram_selection = True
        self.tomogram_combo.clear()
        if choices:
            for source_id, label in choices:
                self.tomogram_combo.addItem(str(label), str(source_id))
            self.tomogram_combo.setEnabled(True)
            if selected_id is not None:
                self.select_tomogram_choice(selected_id)
        else:
            self.tomogram_combo.addItem("No tomograms available", "")
            self.tomogram_combo.setEnabled(False)
        self._block_tomogram_selection = False

    def select_tomogram_choice(self, source_id: str) -> None:
        self._block_tomogram_selection = True
        try:
            for index in range(self.tomogram_combo.count()):
                if str(self.tomogram_combo.itemData(index) or "") == str(source_id):
                    self.tomogram_combo.setCurrentIndex(index)
                    break
        finally:
            self._block_tomogram_selection = False

    def current_tomogram_source_id(self) -> str:
        return str(self.tomogram_combo.currentData() or "")

    def show_message(self, text: str) -> None:
        self._volume = None
        self._cached_images = {}
        self._levels = None
        self._zoom_factor = 1.0
        self.slice_label.setText(text)
        self._set_slice_controls(0, 0)
        self.figure.clear()
        style_figure(self.figure)
        axis = self.figure.add_subplot(111)
        style_axis(axis)
        axis.text(0.5, 0.5, text, ha="center", va="center", transform=axis.transAxes, color=TEXT_COLOR)
        axis.axis("off")
        self.canvas.draw_idle()
        self._emit_image_changed()

    def show_volume(
        self,
        volume: np.ndarray | None,
        title: str = "Tomogram",
        preserve_view_state: bool = False,
    ) -> None:
        if volume is None:
            self.show_message("No tomogram")
            return
        vol = np.asarray(volume)
        if vol.ndim != 3:
            raise ValueError(f"Tomogram volume must be 3-D, got shape {vol.shape}")
        if not np.issubdtype(vol.dtype, np.number):
            raise ValueError(f"Tomogram volume must be numeric, got dtype {vol.dtype}")
        previous_shape = tuple(self._volume.shape) if self._volume is not None else None
        previous_levels = self._levels
        previous_zoom = self._zoom_factor
        previous_slices = dict(self._slice_indices)
        self._volume = vol
        self._title = title
        if preserve_view_state and previous_shape is not None:
            self._slice_indices = {
                "Axial": min(max(int(previous_slices.get("Axial", vol.shape[0] // 2)), 0), int(vol.shape[0] - 1)),
                "Coronal": min(max(int(previous_slices.get("Coronal", vol.shape[1] // 2)), 0), int(vol.shape[1] - 1)),
                "Sagittal": min(max(int(previous_slices.get("Sagittal", vol.shape[2] // 2)), 0), int(vol.shape[2] - 1)),
            }
            self._levels = previous_levels if previous_levels is not None else self._auto_limits(vol)
            self._zoom_factor = previous_zoom
        else:
            self._slice_indices = {
                "Axial": vol.shape[0] // 2,
                "Coronal": vol.shape[1] // 2,
                "Sagittal": vol.shape[2] // 2,
            }
            self._levels = self._auto_limits(vol)
            self._zoom_factor = 1.0
        self._refresh_cached_images()
        self._sync_slice_controls_to_view()
        self._draw_current()

    def current_image(self) -> np.ndarray | None:
        if self._volume is None:
            return None
        view = self.current_view()
        return self._cached_images.get(view, self._image_for_view(view))

    def current_levels(self) -> tuple[float, float] | None:
        return self._levels

    def current_zoom_factor(self) -> float:
        return self._zoom_factor

    def current_view(self) -> str:
        view = self.view_combo.currentText()
        return view if view in self.VIEW_LABELS else "Axial"

    def set_levels(self, low: float, high: float) -> None:
        if self._volume is None:
            return
        low = float(low)
        high = float(high)
        if not np.isfinite(low) or not np.isfinite(high):
            return
        if high <= low:
            high = low + 1.0
        self._levels = (low, high)
        self._draw_current(emit=False)

    def reset_auto_levels(self) -> tuple[float, float] | None:
        if self._volume is None:
            return None
        self._levels = self._auto_limits(self._volume)
        self._draw_current()
        return self._levels

    def set_zoom_factor(self, zoom_factor: float) -> float:
        self._zoom_factor = min(max(float(zoom_factor), 1.0), 32.0)
        self._draw_current(emit=False)
        return self._zoom_factor

    def fit_to_view(self) -> float:
        return self.set_zoom_factor(1.0)

    def _view_changed(self, *_: object) -> None:
        if self._volume is None:
            return
        self._sync_slice_controls_to_view()
        self._draw_current()

    def _tomogram_selection_changed(self, *_: object) -> None:
        if self._block_tomogram_selection:
            return
        source_id = self.current_tomogram_source_id()
        if source_id and self.on_tomogram_selected is not None:
            self.on_tomogram_selected(source_id)

    def _slice_spin_changed(self, value: int) -> None:
        if self._block_controls or self._volume is None:
            return
        self._set_current_slice(value)

    def _slice_slider_changed(self, value: int) -> None:
        if self._block_controls or self._volume is None:
            return
        self._set_current_slice(value)

    def _set_current_slice(self, value: int) -> None:
        view = self.current_view()
        maximum = self._max_slice_for_view(view)
        value = min(max(int(value), 0), maximum)
        self._slice_indices[view] = value
        self._refresh_cached_images()
        self._block_controls = True
        self.slice_spin.setValue(value)
        self.slice_slider.setValue(value)
        self._block_controls = False
        self._draw_current()

    def _sync_slice_controls_to_view(self) -> None:
        view = self.current_view()
        maximum = self._max_slice_for_view(view)
        value = min(max(int(self._slice_indices.get(view, 0)), 0), maximum)
        self._slice_indices[view] = value
        self._set_slice_controls(value, maximum)

    def _set_slice_controls(self, value: int, maximum: int) -> None:
        self._block_controls = True
        self.slice_spin.setRange(0, maximum)
        self.slice_slider.setRange(0, maximum)
        self.slice_spin.setValue(value)
        self.slice_slider.setValue(value)
        self._block_controls = False

    def _draw_current(self, emit: bool = True) -> None:
        image = self.current_image()
        if image is None:
            self.show_message("No tomogram")
            return
        view = self.current_view()
        index = self._slice_indices[view]
        low, high = self._levels or self._auto_limits(self._volume if self._volume is not None else image)
        self.figure.clear()
        style_figure(self.figure)
        axis = self.figure.add_subplot(111)
        style_axis(axis)
        axis.imshow(image, cmap="gray", vmin=low, vmax=high, origin="upper")
        _apply_image_zoom(axis, image, self._zoom_factor)
        axis.set_title(f"{self._title} | {view} slice {index}", color=TEXT_COLOR)
        axis.axis("off")
        self.figure.tight_layout()
        self.slice_label.setText(
            f"{view} slice {index + 1}/{self._max_slice_for_view(view) + 1}    "
            f"Volume shape z/y/x: {self._volume.shape if self._volume is not None else 'unavailable'}"
        )
        self.canvas.draw_idle()
        if emit:
            self._emit_image_changed()

    def _emit_image_changed(self) -> None:
        if self.on_image_changed is not None:
            self.on_image_changed()

    def _refresh_cached_images(self) -> None:
        if self._volume is None:
            self._cached_images = {}
            return
        self._cached_images = {view: self._image_for_view(view) for view in self.VIEW_LABELS}

    def _image_for_view(self, view: str) -> np.ndarray:
        if self._volume is None:
            raise ValueError("No tomogram volume is loaded.")
        if view == "Coronal":
            return np.asarray(self._volume[:, self._slice_indices["Coronal"], :], dtype=np.float32)
        if view == "Sagittal":
            return np.asarray(self._volume[:, :, self._slice_indices["Sagittal"]], dtype=np.float32)
        return np.asarray(self._volume[self._slice_indices["Axial"]], dtype=np.float32)

    def _max_slice_for_view(self, view: str) -> int:
        if self._volume is None:
            return 0
        if view == "Coronal":
            return int(self._volume.shape[1] - 1)
        if view == "Sagittal":
            return int(self._volume.shape[2] - 1)
        return int(self._volume.shape[0] - 1)

    def _auto_limits(self, array: np.ndarray) -> tuple[float, float]:
        sample = _sample_array_for_limits(array)
        finite = sample[np.isfinite(sample)]
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


class RequestedSlicePreviewWidget(QWidget):
    """Live MBIR preview widget that only holds one requested full-resolution slice."""

    VIEW_LABELS = ("Axial", "Coronal", "Sagittal")

    def __init__(
        self,
        on_image_changed: Callable[[], None] | None = None,
        on_request_changed: Callable[[str, int], None] | None = None,
    ) -> None:
        super().__init__()
        self.on_image_changed = on_image_changed
        self.on_request_changed = on_request_changed
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        style_figure(self.figure)
        style_canvas(self.canvas)
        self.view_combo = QComboBox()
        self.view_combo.addItems(self.VIEW_LABELS)
        self.slice_spin = QSpinBox()
        self.slice_spin.setRange(1, 1)
        self.slice_label = QLabel("No MBIR preview loaded")
        self.slice_label.setWordWrap(True)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("View"))
        controls.addWidget(self.view_combo)
        controls.addWidget(QLabel("Slice"))
        controls.addWidget(self.slice_spin)
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.slice_label)
        layout.addWidget(self.canvas, 1)

        self._current_image: np.ndarray | None = None
        self._title = "MBIR Preview"
        self._levels: tuple[float, float] | None = None
        self._zoom_factor = 1.0
        self._requested_slices = {view: 0 for view in self.VIEW_LABELS}
        self._slice_counts = {view: 1 for view in self.VIEW_LABELS}
        self._displayed_view = "Axial"
        self._displayed_slice_index = 0
        self._pending_request = False
        self._block_controls = False

        self.view_combo.currentTextChanged.connect(self._view_changed)
        self.slice_spin.valueChanged.connect(self._slice_spin_changed)
        self.show_message("Waiting for first MBIR preview snapshot")

    def show_message(self, text: str) -> None:
        self._current_image = None
        self._levels = None
        self._zoom_factor = 1.0
        self.figure.clear()
        style_figure(self.figure)
        axis = self.figure.add_subplot(111)
        style_axis(axis)
        axis.text(0.5, 0.5, text, ha="center", va="center", transform=axis.transAxes, color=TEXT_COLOR)
        axis.axis("off")
        self.slice_label.setText(text)
        self.canvas.draw_idle()
        self._emit_image_changed()

    def set_volume_shape_hint(self, volume_shape: tuple[int, int, int] | None) -> None:
        if volume_shape is None or len(volume_shape) != 3 or min(int(v) for v in volume_shape) <= 0:
            return
        self._slice_counts = {
            "Axial": int(volume_shape[0]),
            "Coronal": int(volume_shape[1]),
            "Sagittal": int(volume_shape[2]),
        }
        for view in self.VIEW_LABELS:
            self._requested_slices[view] = min(
                max(int(self._requested_slices.get(view, 0)), 0),
                max(int(self._slice_counts[view]) - 1, 0),
            )
        self._sync_slice_controls()
        self._update_status_label()

    def show_slice(
        self,
        image: np.ndarray,
        title: str,
        view: str,
        slice_index: int,
        slice_counts: dict[str, int] | None = None,
    ) -> None:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Requested MBIR preview slice must be 2-D, got shape {array.shape}")
        view = self._normalize_view(view)
        previous_levels = self._levels
        previous_zoom = self._zoom_factor
        if slice_counts:
            self._slice_counts = {
                label: max(1, int(slice_counts.get(label, self._slice_counts.get(label, 1))))
                for label in self.VIEW_LABELS
            }
        self._requested_slices[view] = min(max(int(slice_index), 0), self._slice_counts.get(view, 1) - 1)
        self._set_request(view, self._requested_slices[view], emit=False)
        self._current_image = array
        self._title = title
        self._displayed_view = view
        self._displayed_slice_index = self._requested_slices[view]
        self._pending_request = False
        self._levels = previous_levels if previous_levels is not None else self._auto_limits(array)
        self._zoom_factor = previous_zoom
        self._draw_current()

    def mark_request_pending(self, view: str | None = None, slice_index: int | None = None) -> None:
        if view is not None and slice_index is not None:
            self._set_request(view, slice_index, emit=False)
        self._pending_request = True
        self._update_status_label()

    def current_image(self) -> np.ndarray | None:
        return self._current_image

    def current_levels(self) -> tuple[float, float] | None:
        return self._levels

    def current_zoom_factor(self) -> float:
        return self._zoom_factor

    def current_view(self) -> str:
        return self._normalize_view(self.view_combo.currentText())

    def current_requested_slice(self) -> int:
        return int(self._requested_slices.get(self.current_view(), 0))

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
        self._draw_current(emit=False)

    def reset_auto_levels(self) -> tuple[float, float] | None:
        if self._current_image is None:
            return None
        self._levels = self._auto_limits(self._current_image)
        self._draw_current()
        return self._levels

    def set_zoom_factor(self, zoom_factor: float) -> float:
        self._zoom_factor = min(max(float(zoom_factor), 1.0), 32.0)
        self._draw_current(emit=False)
        return self._zoom_factor

    def fit_to_view(self) -> float:
        return self.set_zoom_factor(1.0)

    def _view_changed(self, *_: object) -> None:
        if self._block_controls:
            return
        self._sync_slice_controls()
        self._pending_request = True
        self._update_status_label()
        self._emit_request_changed()

    def _slice_spin_changed(self, value: int) -> None:
        if self._block_controls:
            return
        view = self.current_view()
        self._requested_slices[view] = min(max(int(value) - 1, 0), self._slice_counts.get(view, 1) - 1)
        self._pending_request = True
        self._update_status_label()
        self._emit_request_changed()

    def _set_request(self, view: str, slice_index: int, emit: bool = True) -> None:
        view = self._normalize_view(view)
        count = max(1, int(self._slice_counts.get(view, 1)))
        slice_index = min(max(int(slice_index), 0), count - 1)
        self._requested_slices[view] = slice_index
        self._block_controls = True
        self.view_combo.setCurrentText(view)
        self.slice_spin.setRange(1, count)
        self.slice_spin.setValue(slice_index + 1)
        self._block_controls = False
        self._update_status_label()
        if emit:
            self._emit_request_changed()

    def _sync_slice_controls(self) -> None:
        view = self.current_view()
        count = max(1, int(self._slice_counts.get(view, 1)))
        value = min(max(int(self._requested_slices.get(view, 0)), 0), count - 1)
        self._requested_slices[view] = value
        self._block_controls = True
        self.slice_spin.setRange(1, count)
        self.slice_spin.setValue(value + 1)
        self._block_controls = False

    def _draw_current(self, emit: bool = True) -> None:
        if self._current_image is None:
            self.show_message("Waiting for first MBIR preview snapshot")
            return
        low, high = self._levels or self._auto_limits(self._current_image)
        self.figure.clear()
        style_figure(self.figure)
        axis = self.figure.add_subplot(111)
        style_axis(axis)
        axis.imshow(self._current_image, cmap="gray", vmin=low, vmax=high, origin="upper")
        _apply_image_zoom(axis, self._current_image, self._zoom_factor)
        axis.set_title(
            f"{self._title} | {self._displayed_view} slice {self._displayed_slice_index + 1}",
            color=TEXT_COLOR,
        )
        axis.axis("off")
        self.figure.tight_layout()
        self._update_status_label()
        self.canvas.draw_idle()
        if emit:
            self._emit_image_changed()

    def _update_status_label(self) -> None:
        requested_view = self.current_view()
        requested_slice = self._requested_slices.get(requested_view, 0)
        requested_count = max(1, int(self._slice_counts.get(requested_view, 1)))
        if self._current_image is None:
            self.slice_label.setText(
                f"Requested {requested_view} slice {requested_slice + 1}/{requested_count}. Waiting for MBIR preview."
            )
            return
        displayed_count = max(1, int(self._slice_counts.get(self._displayed_view, 1)))
        if self._pending_request and (
            requested_view != self._displayed_view or requested_slice != self._displayed_slice_index
        ):
            self.slice_label.setText(
                f"Requested {requested_view} slice {requested_slice + 1}/{requested_count}. "
                f"Waiting for next MBIR update. Showing {self._displayed_view} slice "
                f"{self._displayed_slice_index + 1}/{displayed_count}."
            )
            return
        self.slice_label.setText(
            f"{self._displayed_view} slice {self._displayed_slice_index + 1}/{displayed_count}"
        )

    def _emit_image_changed(self) -> None:
        if self.on_image_changed is not None:
            self.on_image_changed()

    def _emit_request_changed(self) -> None:
        if self.on_request_changed is not None:
            self.on_request_changed(self.current_view(), self.current_requested_slice())

    def _normalize_view(self, value: str) -> str:
        text = str(value or "").strip().lower()
        if text.startswith("cor"):
            return "Coronal"
        if text.startswith("sag"):
            return "Sagittal"
        return "Axial"

    def _auto_limits(self, array: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(array)[np.isfinite(array)]
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


def _apply_image_zoom(axis, image: np.ndarray, zoom_factor: float) -> None:
    zoom = max(1.0, float(zoom_factor))
    if zoom <= 1.0001:
        return
    rows, cols = np.asarray(image).shape[:2]
    center_col = (cols - 1) / 2.0
    center_row = (rows - 1) / 2.0
    half_width = max(0.5, cols / (2.0 * zoom))
    half_height = max(0.5, rows / (2.0 * zoom))
    axis.set_xlim(center_col - half_width, center_col + half_width)
    axis.set_ylim(center_row + half_height, center_row - half_height)


def _sample_array_for_limits(array: np.ndarray, max_values: int = 1_000_000) -> np.ndarray:
    arr = np.asarray(array)
    if arr.size <= max_values:
        return np.asarray(arr, dtype=np.float32).ravel()
    ndim = max(int(arr.ndim), 1)
    stride = max(1, int(np.ceil((float(arr.size) / float(max_values)) ** (1.0 / ndim))))
    stepped = arr[tuple(slice(None, None, stride) for _ in range(arr.ndim))]
    sample = np.asarray(stepped, dtype=np.float32).ravel()
    if sample.size > max_values:
        sample = sample[:: max(1, int(np.ceil(float(sample.size) / float(max_values))))]
    return sample


class HistogramLevelWidget(QWidget):
    def __init__(self, on_levels_changed: Callable[[float, float], None] | None = None) -> None:
        super().__init__()
        self.figure = Figure(figsize=(3.6, 2.2), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        style_figure(self.figure)
        style_canvas(self.canvas)
        self.on_levels_changed = on_levels_changed
        self.image: np.ndarray | None = None
        self.low = 0.0
        self.high = 1.0
        self._drag_target: str | None = None
        self._axis = None
        self._level_artists: list[object] = []
        self._hist_empty = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.canvas.setMinimumHeight(220)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.show_empty()

    def show_empty(self) -> None:
        self.image = None
        self._axis = None
        self._level_artists = []
        self._hist_empty = True
        self.figure.clear()
        style_figure(self.figure)
        axis = self.figure.add_subplot(111)
        style_axis(axis)
        axis.text(0.5, 0.5, "No active image", ha="center", va="center", transform=axis.transAxes, color=TEXT_COLOR)
        axis.axis("off")
        self.canvas.draw_idle()

    def set_image(self, image: np.ndarray | None, levels: tuple[float, float] | None = None) -> None:
        if image is None:
            self.show_empty()
            return
        self.image = np.asarray(image, dtype=np.float32)
        if levels is None:
            levels = self.auto_levels()
        self.low, self.high = float(levels[0]), float(levels[1])
        if self.high <= self.low:
            self.high = self.low + 1.0
        self._rebuild_histogram()

    def auto_levels(self) -> tuple[float, float]:
        if self.image is None:
            return 0.0, 1.0
        finite = self.image[np.isfinite(self.image)]
        if finite.size == 0:
            return 0.0, 1.0
        low, high = np.percentile(finite, [1, 99])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.min(finite)), float(np.max(finite))
        if high <= low:
            high = low + 1.0
        return float(low), float(high)

    def full_range_levels(self) -> tuple[float, float]:
        if self.image is None:
            return 0.0, 1.0
        finite = self.image[np.isfinite(self.image)]
        if finite.size == 0:
            return 0.0, 1.0
        low, high = float(np.min(finite)), float(np.max(finite))
        if high <= low:
            high = low + 1.0
        return low, high

    def set_levels(self, low: float, high: float, emit: bool = True) -> None:
        self.low = float(low)
        self.high = float(high)
        if self.high <= self.low:
            self.high = self.low + 1.0
        self._update_level_artists()
        if emit and self.on_levels_changed is not None:
            self.on_levels_changed(self.low, self.high)

    def _rebuild_histogram(self) -> None:
        self.figure.clear()
        style_figure(self.figure)
        axis = self.figure.add_subplot(111)
        self._axis = axis
        self._level_artists = []
        style_axis(axis, grid=True)
        if self.image is None:
            axis.text(0.5, 0.5, "No active image", ha="center", va="center", transform=axis.transAxes, color=TEXT_COLOR)
            axis.axis("off")
            self._hist_empty = True
            self.canvas.draw_idle()
            return
        finite = self.image[np.isfinite(self.image)]
        if finite.size:
            sample = _sample_array_for_limits(finite)
            axis.hist(sample.ravel(), bins=96, color=ACCENT_COLOR, alpha=0.9)
            axis.set_xlim(*self._hist_xlim(finite))
            axis.set_title("", fontsize=10, color=TEXT_COLOR)
            self._hist_empty = False
            self._update_level_artists()
        else:
            axis.text(0.5, 0.5, "No finite pixels", ha="center", va="center", transform=axis.transAxes, color=TEXT_COLOR)
            self._hist_empty = True
        axis.tick_params(labelsize=8)
        self.figure.tight_layout(pad=0.6)
        self.canvas.draw_idle()

    def _hist_xlim(self, finite: np.ndarray) -> tuple[float, float]:
        low, high = np.percentile(finite, [0.5, 99.5])
        low = min(float(low), self.low)
        high = max(float(high), self.high)
        if high <= low:
            high = low + 1.0
        pad = 0.03 * (high - low)
        return low - pad, high + pad

    def _on_press(self, event) -> None:
        if self.image is None or event.inaxes is not self._axis or event.xdata is None:
            return
        x = float(event.xdata)
        threshold = self._handle_pick_threshold_px()
        low_px, high_px = self._handle_positions_px()
        if abs(float(event.x) - low_px) <= threshold and abs(float(event.x) - high_px) <= threshold:
            self._drag_target = "low" if abs(x - self.low) <= abs(x - self.high) else "high"
        elif abs(float(event.x) - low_px) <= threshold:
            self._drag_target = "low"
        elif abs(float(event.x) - high_px) <= threshold:
            self._drag_target = "high"
        else:
            midpoint = 0.5 * (self.low + self.high)
            self._drag_target = "low" if x <= midpoint else "high"
        self._move_level(x)

    def _on_motion(self, event) -> None:
        if self._drag_target is None or event.xdata is None:
            return
        self._move_level(float(event.xdata))

    def _on_release(self, _event) -> None:
        self._drag_target = None

    def _move_level(self, value: float) -> None:
        if self._drag_target == "low":
            self.set_levels(value, self.high)
        elif self._drag_target == "high":
            self.set_levels(self.low, value)

    def _update_level_artists(self) -> None:
        if self._axis is None or self._hist_empty:
            return
        for artist in self._level_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._level_artists = []
        axis = self._axis
        band_half_width = self._handle_band_half_width()
        low_color = "#FF8A80"
        high_color = "#80D6B6"
        self._level_artists.extend(
            [
                axis.axvspan(self.low - band_half_width, self.low + band_half_width, color=low_color, alpha=0.24, zorder=3),
                axis.axvline(self.low, color=low_color, linewidth=3.0, zorder=4),
                axis.axvspan(self.high - band_half_width, self.high + band_half_width, color=high_color, alpha=0.24, zorder=3),
                axis.axvline(self.high, color=high_color, linewidth=3.0, zorder=4),
            ]
        )
        ymin, ymax = axis.get_ylim()
        marker_y = ymin + 0.93 * (ymax - ymin)
        self._level_artists.extend(
            [
                axis.scatter([self.low], [marker_y], s=90, marker="v", color=low_color, edgecolors=PANEL_BACKGROUND, linewidths=0.8, zorder=5),
                axis.scatter([self.high], [marker_y], s=90, marker="v", color=high_color, edgecolors=PANEL_BACKGROUND, linewidths=0.8, zorder=5),
            ]
        )
        axis.set_title(f"Levels {self.low:.4g} to {self.high:.4g} | drag the shaded handles", fontsize=10, color=TEXT_COLOR)
        self.canvas.draw_idle()

    def _handle_band_half_width(self) -> float:
        axis = self._axis
        if axis is None:
            return 0.5
        left, right = axis.get_xlim()
        return max(abs(right - left) * 0.012, 1e-6)

    def _handle_positions_px(self) -> tuple[float, float]:
        axis = self._axis
        if axis is None:
            return 0.0, 0.0
        low_px = float(axis.transData.transform((self.low, 0.0))[0])
        high_px = float(axis.transData.transform((self.high, 0.0))[0])
        return low_px, high_px

    def _handle_pick_threshold_px(self) -> float:
        axis = self._axis
        if axis is None:
            return 18.0
        return max(18.0, float(axis.bbox.width) * 0.03)
