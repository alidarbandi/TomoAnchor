from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent / "tv_mbir_ct_recon"
sys.path.insert(0, str(PROJECT_DIR))

from app import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
