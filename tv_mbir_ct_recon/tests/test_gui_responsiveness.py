from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from src.gui.plot_widgets import MetricsPlotWidget, _load_preview_image_cached, _read_csv_rows_cached
from src.gui.preview_widgets import HistogramLevelWidget, TomogramViewWidget
from src.gui.qt_compat import QApplication


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_csv_cache_invalidates_on_file_change(tmp_path) -> None:
    path = tmp_path / "scores.csv"
    path.write_text("filter,score_total\nhann,1.0\n", encoding="utf-8")

    rows1 = _read_csv_rows_cached(path)
    rows2 = _read_csv_rows_cached(path)

    assert rows1 is rows2
    assert rows1[0]["score_total"] == "1.0"

    time.sleep(0.02)
    path.write_text("filter,score_total\nhann,2.0\n", encoding="utf-8")

    rows3 = _read_csv_rows_cached(path)

    assert rows3 is not rows1
    assert rows3[0]["score_total"] == "2.0"


def test_preview_cache_invalidates_on_npy_change(tmp_path) -> None:
    npy_path = tmp_path / "preview.npy"
    np.save(npy_path, np.arange(27, dtype=np.float32).reshape(3, 3, 3))

    image1 = _load_preview_image_cached(None, npy_path)
    image2 = _load_preview_image_cached(None, npy_path)

    assert image1 is image2

    time.sleep(0.02)
    np.save(npy_path, np.full((3, 3, 3), 5.0, dtype=np.float32))

    image3 = _load_preview_image_cached(None, npy_path)

    assert image3 is not image1
    assert image3 is not None
    assert np.allclose(image3, 5.0)


def test_metrics_polling_toggle_controls_timer(monkeypatch) -> None:
    _app()

    def _noop_start(self, sync_detail: bool, request_id: int) -> None:
        self._active_refresh_request_id = int(request_id)

    monkeypatch.setattr(MetricsPlotWidget, "_start_refresh_worker", _noop_start)

    widget = MetricsPlotWidget()
    widget.set_polling_enabled(False)
    assert not widget._poll_timer.isActive()

    widget.set_polling_enabled(True)
    assert widget._poll_timer.isActive()
    widget.close()


def test_histogram_reuses_existing_image_without_rebuild(monkeypatch) -> None:
    _app()
    rebuild_calls: list[int] = []
    original = HistogramLevelWidget._rebuild_histogram

    def _wrapped(self) -> None:
        rebuild_calls.append(1)
        original(self)

    monkeypatch.setattr(HistogramLevelWidget, "_rebuild_histogram", _wrapped)

    widget = HistogramLevelWidget()
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    widget.set_image(image, (0.0, 15.0))
    widget.set_image(image, (2.0, 10.0))

    assert len(rebuild_calls) == 1
    widget.close()


def test_tomogram_slice_change_updates_only_active_view_and_reuses_artist(monkeypatch) -> None:
    _app()
    calls: list[str] = []
    original = TomogramViewWidget._image_for_view

    def _wrapped(self, view: str) -> np.ndarray:
        calls.append(view)
        return original(self, view)

    monkeypatch.setattr(TomogramViewWidget, "_image_for_view", _wrapped)

    widget = TomogramViewWidget()
    volume = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
    widget.show_volume(volume, "Tomogram")

    assert calls == ["Axial", "Coronal", "Sagittal"]

    artist = widget._image_artist
    calls.clear()
    widget._set_current_slice(1)

    assert calls == ["Axial"]

    widget.set_levels(0.0, 10.0)
    assert widget._image_artist is artist
    widget.close()
