from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import tifffile

from src.config import AlignmentConfig, PreprocessingConfig
from src.preprocessing import compute_attenuation, flat_dark_correct, run_preprocessing


def test_fdk_compatible_flat_field_and_log_formula() -> None:
    raw = np.full((2, 4, 4), 10.0, dtype=np.float32)
    flat = np.full((4, 4), 20.0, dtype=np.float32)
    transmission = flat_dark_correct(raw, flat, dark_field=None, epsilon=1e-6)
    attenuation = compute_attenuation(transmission, clip_negative_to_zero=False)
    assert np.allclose(transmission, 0.5)
    assert np.allclose(attenuation, -np.log(0.5))


def test_dark_field_formula() -> None:
    raw = np.full((1, 2, 2), 12.0, dtype=np.float32)
    flat = np.full((2, 2), 22.0, dtype=np.float32)
    dark = np.full((2, 2), 2.0, dtype=np.float32)
    transmission = flat_dark_correct(raw, flat, dark_field=dark, epsilon=1e-6)
    assert np.allclose(transmission, 0.5)


def test_run_preprocessing_can_skip_negative_log(tmp_path) -> None:
    image = np.full((2, 2), 0.25, dtype=np.float32)
    projection_path = tmp_path / "p0.tif"
    tifffile.imwrite(str(projection_path), image)
    validation = SimpleNamespace(
        records=[SimpleNamespace(path=projection_path, filename="p0.tif")],
        x_shift_px=None,
        y_shift_px=None,
    )
    preprocessing = PreprocessingConfig(
        use_flat=False,
        use_dark=False,
        negative_log=False,
        clip_transmission=False,
        clip_negative_attenuation_to_zero=False,
    )
    alignment = AlignmentConfig(use_drift_correction=False)

    result = run_preprocessing(
        validation,
        str(tmp_path),
        preprocessing=preprocessing,
        alignment=alignment,
    )

    assert np.allclose(result.transmission_stack, image[None, :, :])
    assert np.allclose(result.attenuation_stack, image[None, :, :])
