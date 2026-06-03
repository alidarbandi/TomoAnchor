# TomoAnchor Technical Notes

TomoAnchor is clean TV-regularized MBIR and anchor-guided MBIR-lite reconstruction software for Zeiss Versa cone-beam CT datasets exported as TIFF projection/reference images plus projection-geometry CSV metadata.

The project reuses validated FDK conventions for data loading, flat-field correction, angle handling, center-offset alignment, drift correction, and TIGRE geometry while keeping the MBIR solver modular.

## Quick Start

```powershell
cd path\to\TomoAnchor\tv_mbir_ct_recon
conda env create -f environment.yml
conda activate tomoanchor
python app.py --gui
```

Run core checks:

```powershell
python app.py --self-test
python -m pytest tests
```

Run standard stages from YAML:

```powershell
python app.py --config examples/sample_config.yaml --preprocess-only
python app.py --config examples/sample_config.yaml --run-fdk
python app.py --config examples/sample_config.yaml --run-mbir
```

Run the complete fast anchor-guided workflow:

```powershell
python app.py --config examples/fast_recon_config.yaml --run-fast
```

Run individual fast stages against a chosen run folder:

```powershell
python app.py --config examples/fast_recon_config.yaml --run-fdk-sweep
python app.py --config examples/fast_recon_config.yaml --make-prior --fast-run-folder output/run_YYYYMMDD_HHMMSS
python app.py --config examples/fast_recon_config.yaml --run-mbir-lite --fast-run-folder output/run_YYYYMMDD_HHMMSS
python app.py --config examples/fast_recon_config.yaml --run-fast-qc --fast-run-folder output/run_YYYYMMDD_HHMMSS
```

## Data Model

The default input layout matches the existing FDK workflow:

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

The CSV should include a projection filename column and an angle column. Optional `angle_rad`, `x_shift_px`, and `y_shift_px` columns are supported.

## Fast Anchor-Guided Workflow

The fast workflow is designed for sparse or low-exposure main scans that also have a separate high-exposure anchor dataset. Configure the main dataset in the existing `input`, `metadata`, `preprocessing`, and `alignment` sections. Configure the high-SNR anchors under `anchors`.

The anchor dataset has its own projection folder, flat/reference folder, optional dark folder, metadata CSV, and optional drift shift file. Anchor preprocessing runs separately so the main and anchor datasets do not share flats, darks, or drift corrections by accident.

The complete `--run-fast` stage performs:

- Main preprocessing and anchor preprocessing.
- Deterministic anchor split into reconstruction, tuning, and held-out QC views.
- FDK filter sweep scored on tuning anchors.
- Weak training-free 3-D TV prior and confidence map.
- Prior-anchored MBIR-lite with projection weights.
- Markdown, CSV, PNG, and TIFF QC outputs using held-out QC anchors.

The MBIR-lite optimization target is:

```text
0.5 ||W(Ax - y)||_2^2 + lambda_tv TV_epsilon(x)
  + 0.5 rho_prior ||sqrt(C)(x - x_prior)||_2^2
```

`x_prior` is only a conservative denoising prior, not the final answer. MBIR-lite still pulls the reconstruction back toward measured projection consistency. QC anchors are held out from FDK selection, prior selection, and final reconstruction by default.

Fast outputs are saved under:

```text
output/run_YYYYMMDD_HHMMSS/fast_recon/
  anchors/
  fdk_sweep/
  prior/
  mbir_lite/
  qc/
```

No deep learning dependencies are required or used.

## Reconstruction Model

The standard MBIR solver minimizes:

```text
0.5 * ||A x - b||_2^2 + lambda_tv * TV(x), subject to x >= 0
```

using scaled ADMM / split-Bregman:

- Matrix-free TIGRE forward/backprojection wrapper.
- Conjugate-gradient x-update.
- Isotropic 3-D TV shrinkage.
- Positivity projection.
- Objective, data term, TV term, primal/dual residual, relative change, and CG metrics.

## Large-Data Memory Strategy

The default MBIR `solver` is `auto`. For smaller volumes this uses the original ADMM/CG solver. For volumes where ADMM/CG would exceed the physical RAM budget, it switches to a low-memory Chambolle-Pock/PDHG solver. If PDHG is still too close to the RAM limit, `auto` switches again to `streaming_subset_tv`, an ordered-subset style solver that updates after each projection batch and discards the batch gradient immediately.

The default MBIR `memory_mode` is also `auto`, which currently selects full-data projection streaming. In this mode the solver keeps the same full-volume 3-D TV-MBIR objective, but evaluates TIGRE projection operations in batches:

```text
A^T(Ax - b) = sum over projection batches A_i^T(A_i x - b_i)
```

This preserves the full-data reconstruction model while avoiding one monolithic CUDA allocation for all projection views. Use:

- `projection_streaming` for the highest-quality large-data default.
- `full_gpu` only when the whole projection stack and volume comfortably fit in GPU memory.
- `distributed_full_data` for exact full-data MBIR with persistent one-GPU TIGRE workers, static angle sharding, and concurrent `Ax`, `A^T`, `A^T A`, and residual evaluation across multiple GPUs.
- `ordered_subsets` for faster, lower-memory iterative progress using interleaved angle subsets; this is useful for acceleration, but its per-iteration objective is an approximation.

For larger scans, reduce `projection_batch_size` first, for example `64`, `32`, or `16`. This is different from binning: projection pixels and reconstruction voxel size are unchanged.

The GUI includes an `Estimate Memory` button in Run Controls. It estimates host RAM and GPU VRAM from the current volume shape, projection shape, MBIR-lite batch size, subset count, selected GPUs, and TIGRE safety factors, then compares the estimate with available system memory.

## Important Conventions

- Projections are shaped `(n_angles, rows/v, cols/u)`.
- Volumes are shaped `(z, y, x)`.
- Default preprocessing is `T = I/F`, then `b = -log(T)`.
- With dark fields, preprocessing becomes `T = (I-D)/(F-D)`.
- Zeiss effective object-plane pixel size can be converted to TIGRE detector-plane sampling with `DSD/DSO`.
- Default TIGRE angle sign is inverted for compatibility with the tested FDK workflow.
- Center offset is in detector pixels after binning and maps to `geo.offDetector[1]`.
- Per-projection drift shifts are applied directly to projection images.

## Project Layout

```text
tv_mbir_ct_recon/
  app.py
  FDK_SOFTWARE_AUDIT.md
  README.md
  requirements.txt
  environment.yml
  examples/
  src/
    config.py
    io_utils.py
    metadata_zeiss.py
    reference_images.py
    preprocessing.py
    geometry_tigre.py
    alignment_center.py
    alignment_drift.py
    tigre_ops.py
    tv_ops.py
    cg_solver.py
    admm_tv_mbir.py
    pdhg_tv_mbir.py
    streaming_subset_tv.py
    fast_pipeline.py
    anchor_views.py
    projection_weights.py
    fdk_sweep.py
    denoising_prior.py
    anchored_mbir_lite.py
    fast_qc.py
    worker.py
    gui/
  tests/
```

## Notes

Do not install the unrelated PyPI package named `tigre`. Use the CERN/TIGRE toolbox, preferably from conda channel `ccpi`.
