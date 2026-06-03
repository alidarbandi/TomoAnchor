from __future__ import annotations

import numpy as np

from src.config import AppConfig, load_config, save_config
from src.fast_pipeline import _load_prior_result_if_available, _resolve_run_folders


def test_fast_config_defaults_are_backward_compatible() -> None:
    cfg = AppConfig.from_dict({"input": {"output_folder": "out"}})

    assert cfg.input.output_folder == "out"
    assert cfg.anchors.enabled is False
    assert cfg.fast_recon.save_intermediate is True
    assert cfg.mbir_lite.n_sweeps == 5


def test_fast_config_round_trip(tmp_path) -> None:
    cfg = AppConfig()
    cfg.anchors.enabled = True
    cfg.anchors.input.projection_folder = "anchor/proj"
    cfg.fdk_sweep.filters = ["ram_lak", "hann"]
    cfg.prior.tv_weights = [0.005, 0.02]
    cfg.fast_recon.resume_run_folder = "output/run_1"

    path = save_config(cfg, tmp_path / "fast.yaml")
    loaded = load_config(path)

    assert loaded.anchors.enabled is True
    assert loaded.anchors.input.projection_folder == "anchor/proj"
    assert loaded.fdk_sweep.filters == ["ram_lak", "hann"]
    assert loaded.prior.tv_weights == [0.005, 0.02]
    assert loaded.fast_recon.resume_run_folder == "output/run_1"


def test_fast_resume_accepts_fast_recon_subfolders(tmp_path) -> None:
    root = tmp_path / "run_1"
    prior_folder = root / "fast_recon" / "prior"
    prior_folder.mkdir(parents=True)

    cfg = AppConfig()
    folders = _resolve_run_folders(cfg, prior_folder)

    assert folders.root == root
    assert folders.fast_recon == root / "fast_recon"
    assert folders.fast_prior == prior_folder


def test_fast_prior_loader_accepts_legacy_nested_fast_recon(tmp_path) -> None:
    root = tmp_path / "run_1"
    legacy_prior = root / "fast_recon" / "fast_recon" / "prior"
    legacy_prior.mkdir(parents=True)
    np.save(legacy_prior / "prior_selected.npy", np.ones((2, 3, 4), dtype=np.float32))
    np.save(legacy_prior / "prior_confidence.npy", np.ones((2, 3, 4), dtype=np.float32))

    result = _load_prior_result_if_available(_resolve_run_folders(AppConfig(), root))

    assert result is not None
    assert result.x_prior.shape == (2, 3, 4)
