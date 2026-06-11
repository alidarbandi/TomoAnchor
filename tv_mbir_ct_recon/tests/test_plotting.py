from __future__ import annotations

import numpy as np

from src.io_utils import temporary_output_path
from src.plotting import save_volume_preview


def test_temporary_output_path_keeps_renderable_extension() -> None:
    path = temporary_output_path("preview.png", suffix=".png.tmp")

    assert path.name.endswith(".png")
    assert not path.name.endswith(".tmp")
    assert ".tmp.png" in path.name


def test_save_volume_preview_writes_png(tmp_path) -> None:
    volume = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    output = tmp_path / "preview.png"

    saved = save_volume_preview(output, volume, "Preview")

    assert saved == output
    assert output.exists()
    assert output.stat().st_size > 0
