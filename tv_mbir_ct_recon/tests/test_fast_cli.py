from __future__ import annotations

from app import parse_args


def test_fast_cli_flags_parse() -> None:
    args = parse_args(
        [
            "--config",
            "cfg.yaml",
            "--run-fdk-sweep",
            "--make-prior",
            "--run-mbir-lite",
            "--run-fast-qc",
            "--fast-run-folder",
            "output/run_1",
        ]
    )

    assert args.config == "cfg.yaml"
    assert args.run_fdk_sweep is True
    assert args.make_prior is True
    assert args.run_mbir_lite is True
    assert args.run_fast_qc is True
    assert args.fast_run_folder == "output/run_1"
