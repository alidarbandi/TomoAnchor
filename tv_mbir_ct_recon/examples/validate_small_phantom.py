from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.config import MBIRConfig
from src.admm_tv_mbir import ADMMTVMBIRReconstructor


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


def main() -> int:
    phantom = np.zeros((16, 16, 16), dtype=np.float32)
    phantom[5:11, 5:11, 5:11] = 1.0
    config = MBIRConfig(lambda_tv=1e-3, rho=0.1, max_admm_iterations=5, inner_cg_iterations=10)
    result = ADMMTVMBIRReconstructor(IdentityOperator(phantom.shape), config).reconstruct(phantom)
    print(f"Status: {result.message}")
    print(f"Final objective: {result.metrics[-1].objective:.6g}")
    print(f"Volume shape: {result.volume.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
