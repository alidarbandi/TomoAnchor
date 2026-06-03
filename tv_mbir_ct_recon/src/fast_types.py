from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metadata_zeiss import MetadataValidation
from .output_manager import RunFolders


@dataclass
class ProcessedProjectionSet:
    name: str
    attenuation: np.ndarray
    angles_rad: np.ndarray
    validation: MetadataValidation
    preprocessing_report: list[str]
    weights: np.ndarray | None = None
    detector_mask: np.ndarray | None = None


@dataclass
class AnchorSplit:
    recon_indices: np.ndarray
    tune_indices: np.ndarray
    qc_indices: np.ndarray


@dataclass
class AnchorValidationData:
    projections: np.ndarray
    angles_rad: np.ndarray
    weights: np.ndarray | None = None
    detector_mask: np.ndarray | None = None
    label: str = "anchors"


@dataclass
class FDKSweepResult:
    best_volume: np.ndarray
    best_filter: str
    best_cutoff: float
    scores: list[dict[str, object]]
    recon_projections: np.ndarray
    recon_angles_rad: np.ndarray
    recon_weights: np.ndarray | None
    tune_anchor_projections: np.ndarray | None
    tune_anchor_angles_rad: np.ndarray | None
    tune_anchor_weights: np.ndarray | None


@dataclass
class PriorResult:
    x_prior: np.ndarray
    confidence: np.ndarray
    support_mask: np.ndarray
    metrics: list[dict[str, object]]
    chosen_method: str
    chosen_strength: float
    normalization: dict[str, float]


@dataclass
class MBIRLiteMetric:
    iteration: int
    objective_total: float
    data_weighted: float
    tv_term: float
    prior_anchor_term: float
    qc_anchor_residual: float
    relative_change: float
    elapsed_s: float
    solver: str = "anchored_streaming_subset_tv"

    @property
    def objective(self) -> float:
        return self.objective_total

    @property
    def data_fidelity(self) -> float:
        return self.data_weighted

    @property
    def relative_x_change(self) -> float:
        return self.relative_change

    @property
    def primal_residual(self) -> float:
        return float("nan")

    @property
    def dual_residual(self) -> float:
        return float("nan")

    @property
    def cg_residual(self) -> float:
        return float("nan")


@dataclass
class MBIRLiteResult:
    volume: np.ndarray
    metrics: list[MBIRLiteMetric]
    converged: bool
    message: str


@dataclass
class FastPipelineResult:
    run_folders: RunFolders
    main_set: ProcessedProjectionSet | None
    anchor_set: ProcessedProjectionSet | None
    anchor_split: AnchorSplit | None
    fdk_sweep_result: FDKSweepResult | None
    prior_result: PriorResult | None
    mbir_lite_result: MBIRLiteResult | None
    qc_report_path: Path | None
