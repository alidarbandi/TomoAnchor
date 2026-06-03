from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from src.config import AlignmentConfig, AppConfig, PreprocessingConfig
from src.fast_pipeline import _preprocess_named_set


def _write_dataset(root, raw_value: float, flat_value: float) -> tuple[str, str, str]:
    proj = root / "proj"
    flat = root / "flat"
    proj.mkdir(parents=True)
    flat.mkdir(parents=True)
    tifffile.imwrite(str(proj / "p0.tif"), np.full((2, 2), raw_value, dtype=np.float32))
    tifffile.imwrite(str(flat / "f0.tif"), np.full((2, 2), flat_value, dtype=np.float32))
    metadata = root / "metadata.csv"
    pd.DataFrame({"filename": ["p0.tif"], "angle_deg": [0.0]}).to_csv(metadata, index=False)
    return str(proj), str(flat), str(metadata)


def test_anchor_preprocessing_uses_anchor_reference_folder(tmp_path) -> None:
    main_proj, main_flat, main_meta = _write_dataset(tmp_path / "main", raw_value=10.0, flat_value=20.0)
    anchor_proj, anchor_flat, anchor_meta = _write_dataset(tmp_path / "anchor", raw_value=20.0, flat_value=80.0)
    cfg = AppConfig()
    prep = PreprocessingConfig(use_flat=True, use_dark=False, negative_log=True, clip_negative_attenuation_to_zero=False)
    align = AlignmentConfig(use_drift_correction=False)

    main_set, _, _ = _preprocess_named_set(
        "main",
        cfg,
        main_proj,
        main_flat,
        "",
        main_meta,
        cfg.metadata,
        prep,
        align,
        None,
        None,
    )
    anchor_set, _, _ = _preprocess_named_set(
        "anchor",
        cfg,
        anchor_proj,
        anchor_flat,
        "",
        anchor_meta,
        cfg.anchors.metadata,
        prep,
        align,
        None,
        None,
    )

    assert np.allclose(main_set.attenuation, -np.log(0.5))
    assert np.allclose(anchor_set.attenuation, -np.log(0.25))
