from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.admm_tv_mbir import ADMMTVMBIRReconstructor
from src.cg_solver import conjugate_gradient
from src.config import AppConfig, load_config
from src.fast_pipeline import execute_fast_pipeline
from src.pipeline import execute_pipeline
from src.tigre_ops import tigre_available
from src.tv_ops import adjointness_error, gradient_3d, gradient_adjoint_3d


class IdentityOperator:
    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.volume_shape = shape
        self.projection_shape = shape

    def forward(self, x):
        return np.asarray(x, dtype=np.float32)

    def backproject(self, y):
        return np.asarray(y, dtype=np.float32)

    def normal(self, x):
        return np.asarray(x, dtype=np.float32)


def run_self_test() -> int:
    err = adjointness_error((5, 6, 7), seed=11)
    if err > 1e-5:
        raise AssertionError(f"TV adjointness error too large: {err}")
    diag = np.array([2.0, 4.0, 8.0], dtype=np.float32)
    rhs = np.array([2.0, 8.0, 24.0], dtype=np.float32)
    result = conjugate_gradient(lambda x: diag * x, rhs, max_iterations=10, tolerance=1e-7)
    if not np.allclose(result.x, np.array([1.0, 2.0, 3.0], dtype=np.float32), atol=1e-5):
        raise AssertionError(f"CG self-test failed: {result.x}")
    cfg = AppConfig()
    cfg.mbir.max_admm_iterations = 2
    cfg.mbir.inner_cg_iterations = 5
    cfg.mbir.lambda_tv = 0.01
    cfg.mbir.rho = 0.1
    phantom = np.zeros((5, 6, 7), dtype=np.float32)
    phantom[2, 2:4, 2:5] = 1.0
    solver = ADMMTVMBIRReconstructor(IdentityOperator(phantom.shape), cfg.mbir)
    recon = solver.reconstruct(phantom, x0=np.zeros_like(phantom))
    if recon.volume.shape != phantom.shape:
        raise AssertionError("ADMM self-test returned wrong shape.")
    available, message = tigre_available()
    print("Self-test passed.")
    print(f"TV adjointness relative error: {err:.3g}")
    print(f"TIGRE available: {available} ({message})")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TomoAnchor CT reconstruction for Zeiss Versa data")
    parser.add_argument("--config", type=str, help="YAML configuration file")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI")
    parser.add_argument("--preprocess-only", action="store_true", help="Run only metadata validation and preprocessing")
    parser.add_argument("--run-fdk", action="store_true", help="Run FDK reconstruction")
    parser.add_argument("--run-mbir", action="store_true", help="Run TV-MBIR reconstruction")
    parser.add_argument("--run-fast", action="store_true", help="Run the complete fast anchor-guided reconstruction workflow")
    parser.add_argument("--run-fdk-sweep", action="store_true", help="Run fast preprocessing, anchor splitting, and FDK filter sweep")
    parser.add_argument("--make-prior", action="store_true", help="Create the training-free prior from an existing fast FDK result")
    parser.add_argument("--run-mbir-lite", action="store_true", help="Run prior-anchored MBIR-lite from existing fast outputs")
    parser.add_argument("--run-fast-qc", action="store_true", help="Create the fast reconstruction QC report from existing outputs")
    parser.add_argument("--fast-run-folder", type=str, help="Existing run folder to resume for stage-only fast commands")
    parser.add_argument("--self-test", action="store_true", help="Run lightweight non-GUI checks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.self_test:
        return run_self_test()
    fast_requested = bool(args.run_fast or args.run_fdk_sweep or args.make_prior or args.run_mbir_lite or args.run_fast_qc)
    if args.gui or not (args.config or args.preprocess_only or args.run_fdk or args.run_mbir or fast_requested):
        try:
            from src.gui.main_window import run_app
        except ImportError as exc:
            print(f"Could not import GUI dependencies: {exc}", file=sys.stderr)
            return 2
        return run_app(sys.argv)
    if not args.config:
        print("--config is required for command-line reconstruction.", file=sys.stderr)
        return 2
    config = load_config(args.config)
    if fast_requested:
        result = execute_fast_pipeline(
            config,
            run_fdk_sweep=args.run_fdk_sweep,
            make_prior=args.make_prior,
            run_mbir_lite=args.run_mbir_lite,
            run_qc=args.run_fast_qc,
            run_all=args.run_fast,
            fast_run_folder=args.fast_run_folder,
        )
        print(f"Run folder: {result.run_folders.root}")
        if result.fdk_sweep_result is not None:
            print(f"Fast FDK filter: {result.fdk_sweep_result.best_filter}")
        if result.mbir_lite_result is not None:
            print(f"MBIR-lite status: {result.mbir_lite_result.message}")
        if result.qc_report_path is not None:
            print(f"QC report: {result.qc_report_path}")
        return 0
    result = execute_pipeline(
        config,
        run_fdk=args.run_fdk,
        run_mbir=args.run_mbir,
        preprocess_only=args.preprocess_only,
    )
    print(f"Run folder: {result.run_folders.root}")
    print(f"Attenuation/TIGRE input shape: {result.attenuation_shape}")
    if result.mbir_result is not None:
        print(f"MBIR status: {result.mbir_result.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
