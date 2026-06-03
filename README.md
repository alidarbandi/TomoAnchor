# TomoAnchor

TomoAnchor is a Zeiss Versa cone-beam CT reconstruction application for FDK, TV-regularized MBIR, and anchor-guided MBIR-lite workflows. It is designed for sparse or low-exposure scans where you want to compare FDK candidates, build a conservative prior, run prior-anchored MBIR-lite, and inspect live metrics without leaving the GUI.

The runnable launcher is kept at the repository root, while the implementation remains under `tv_mbir_ct_recon/`.

## Highlights

- Desktop GUI plus YAML-driven command-line workflows.
- Zeiss TIFF projection/reference/dark loading with geometry CSV metadata support.
- FDK reconstruction and FDK filter sweeps with preview images and scoring tables.
- Training-free "make prior" stage with confidence and difference-map diagnostics.
- MBIR-lite reconstruction with live objective, data, TV, prior-anchor, and QC-residual metrics.
- Tomogram, MBIR preview, and metrics tabs that can preload existing run outputs.
- Zoom, pan, histogram window/level, and slice controls for comparing reconstruction results.
- GPU selection, live device monitoring, and memory estimation for MBIR-lite batch/subset settings.

## Requirements

- Python 3.10.
- Conda is recommended for installing CERN/TIGRE.
- NVIDIA GPU with CUDA for practical FDK and MBIR reconstruction.
- The CERN/TIGRE toolbox from conda, not the unrelated PyPI package named `tigre`.

## Quick Start

From the repository root:

```powershell
conda env create -f tv_mbir_ct_recon/environment.yml
conda activate tomoanchor
python mbir.py
```

You can also launch the app directly:

```powershell
python tv_mbir_ct_recon/app.py --gui
```

Run the lightweight self-test:

```powershell
python tv_mbir_ct_recon/app.py --self-test
```

Run the test suite:

```powershell
python -m pytest tv_mbir_ct_recon/tests
```

## Command-Line Examples

Run standard stages from YAML:

```powershell
python tv_mbir_ct_recon/app.py --config tv_mbir_ct_recon/examples/sample_config.yaml --preprocess-only
python tv_mbir_ct_recon/app.py --config tv_mbir_ct_recon/examples/sample_config.yaml --run-fdk
python tv_mbir_ct_recon/app.py --config tv_mbir_ct_recon/examples/sample_config.yaml --run-mbir
```

Run the complete anchor-guided workflow:

```powershell
python tv_mbir_ct_recon/app.py --config tv_mbir_ct_recon/examples/fast_recon_config.yaml --run-fast
```

Resume individual fast stages against an existing run folder:

```powershell
python tv_mbir_ct_recon/app.py --config tv_mbir_ct_recon/examples/fast_recon_config.yaml --make-prior --fast-run-folder output/run_YYYYMMDD_HHMMSS
python tv_mbir_ct_recon/app.py --config tv_mbir_ct_recon/examples/fast_recon_config.yaml --run-mbir-lite --fast-run-folder output/run_YYYYMMDD_HHMMSS
python tv_mbir_ct_recon/app.py --config tv_mbir_ct_recon/examples/fast_recon_config.yaml --run-fast-qc --fast-run-folder output/run_YYYYMMDD_HHMMSS
```

## Typical Data Layout

```text
dataset/
  projections/
    proj_000001.tif
    proj_000002.tif
  reference/
    reference_001.tif
  dark/
    dark_001.tif
  metadata/
    projection_geometry.csv
```

The metadata CSV should include a projection filename column and an angle column. Optional `angle_rad`, `x_shift_px`, and `y_shift_px` columns are supported.

## Outputs

TomoAnchor writes timestamped run folders under the configured output folder:

```text
output/run_YYYYMMDD_HHMMSS/
  preprocessing/
  alignment/
  fdk/
  mbir/
  metrics/
  fast_recon/
    anchors/
    fdk_sweep/
    prior/
    mbir_lite/
    qc/
```

Large CT datasets, reconstruction volumes, TIFF stacks, `.npy` arrays, `.npz` bundles, and local run folders are intentionally ignored by git.

## Repository Layout

```text
.
|-- mbir.py
|-- config.mbir.yaml
|-- tv_mbir_ct_recon/
|   |-- app.py
|   |-- environment.yml
|   |-- examples/
|   |-- src/
|   `-- tests/
`-- README.md
```

- `mbir.py` is the root launcher.
- `tv_mbir_ct_recon/app.py` contains the CLI and GUI entry point.
- `tv_mbir_ct_recon/src/` contains reconstruction, preprocessing, GUI, TIGRE operator, metric, and QC code.
- `tv_mbir_ct_recon/examples/` contains example YAML configurations.

More detailed technical notes are in [tv_mbir_ct_recon/README.md](tv_mbir_ct_recon/README.md).
