from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import AppConfig, save_config
from .io_utils import save_image, save_stack_tiff
from .logging_utils import write_text_lines


@dataclass
class RunFolders:
    root: Path
    preprocessing: Path
    alignment: Path
    fdk: Path
    mbir: Path
    snapshots: Path
    metrics: Path
    fast_recon: Path
    fast_anchors: Path
    fast_fdk_sweep: Path
    fast_prior: Path
    fast_mbir_lite: Path
    fast_qc: Path


class OutputManager:
    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root) if str(output_root) else Path.cwd() / "output"

    def create_run(self) -> RunFolders:
        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        root = self.output_root / stamp
        folders = RunFolders(
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
        for folder in folders.__dict__.values():
            Path(folder).mkdir(parents=True, exist_ok=True)
        return folders

    def save_config(self, folders: RunFolders, config: AppConfig) -> Path:
        return save_config(config, folders.root / "config_used.yaml")

    def save_log(self, folders: RunFolders, lines: list[str]) -> Path:
        return write_text_lines(folders.root / "log.txt", lines, append=True)

    def save_diagnostics(self, folders: RunFolders, lines: list[str]) -> Path:
        return write_text_lines(folders.root / "diagnostics.txt", lines)

    def save_volume_outputs(self, folder: Path, stem: str, volume: np.ndarray, save_tif: bool = True) -> tuple[Path, Path | None]:
        npy = folder / f"{stem}.npy"
        npy.parent.mkdir(parents=True, exist_ok=True)
        temp_npy = npy.parent / f".{npy.name}.{datetime.now().strftime('%Y%m%d%H%M%S%f')}.tmp"
        with temp_npy.open("wb") as handle:
            np.save(handle, np.asarray(volume, dtype=np.float32))
        temp_npy.replace(npy)
        tif = save_stack_tiff(folder / f"{stem}.tif", volume) if save_tif else None
        return npy, tif

    def write_report(self, folders: RunFolders, lines: list[str]) -> Path:
        return write_text_lines(folders.root / "report.md", lines)
