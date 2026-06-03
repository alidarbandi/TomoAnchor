# FDK Software Audit

Source project reviewed: local `FDK_engine` codebase.

Audit date: 2026-05-13

## 1. Existing Project Architecture Summary

The existing FDK software is a compact desktop reconstruction application. The top-level `fdk-engine.py` launcher inserts `microct_tigre_gui/` into `sys.path` and calls `main.main()`.

Main folders:

- `.vscode/`: editor settings.
- `microct_tigre_gui/`: application package, environment files, example config, and entry point.
- `microct_tigre_gui/src/`: GUI, preprocessing, metadata, alignment, plotting, logging, and TIGRE FDK code.
- `microct_tigre_gui/src/assets/`: Qt icon and stylesheet assets.
- `windows_release/`: release/build artifacts.

Important files:

- `fdk-engine.py`: top-level GUI launcher.
- `microct_tigre_gui/main.py`: CLI entry point and `--self-test`.
- `microct_tigre_gui/src/gui.py`: PySide6/PyQt5 GUI, worker-thread orchestration, user controls, plotting, config save/load, and workflow glue.
- `microct_tigre_gui/src/config.py`: JSON `AppConfig` dataclass.
- `microct_tigre_gui/src/io_utils.py`: TIFF listing/loading, 2x/4x binning, stack/image output helpers.
- `microct_tigre_gui/src/metadata.py`: CSV metadata discovery, column guessing, filename/angle/shift validation, duplicate endpoint handling.
- `microct_tigre_gui/src/preprocessing.py`: averaged flat field, orientation flips, flat-field correction, attenuation conversion, truncation padding.
- `microct_tigre_gui/src/tigre_reconstruction.py`: Zeiss/TIGRE geometry setup and FDK execution.
- `microct_tigre_gui/src/center_shift_search.py`: center-offset candidate generation, FDK preview reconstruction, focus metrics.
- `microct_tigre_gui/src/shift_correction.py`: per-projection x/y shift correction and shift sanity reports.
- `microct_tigre_gui/src/plotting.py`: Qt/Matplotlib image and diagnostic plots.
- `microct_tigre_gui/src/logging_utils.py`: timestamped in-memory log and log-file writer.
- `microct_tigre_gui/src/qt_compat.py`: PySide6-first, PyQt5-fallback compatibility layer.
- `microct_tigre_gui/config_example.json` and root `config-11um.json`: saved parameter examples.
- `microct_tigre_gui/environment-tigre.yml`, `requirements.txt`, `requirements-tigre.txt`: environments/dependencies.

Tests/validation:

- No formal `tests/` folder was present.
- `microct_tigre_gui/main.py --self-test` exercises preprocessing, metadata validation, center-search metric helpers, shift correction, geometry reports, and TIGRE import detection.

## 2. Data Flow Diagram

```text
GUI path selection
  -> metadata CSV discovery/validation
  -> ProjectionRecord list with resolved TIFF paths and angles
  -> first projection read to infer raw and binned detector size
  -> reference TIFF folder read and averaged
  -> projection TIFF stack read in metadata order
  -> optional 2x/4x block-average binning during TIFF read
  -> optional vertical/horizontal orientation flips
  -> flat-field correction T = I / F
  -> optional per-projection shift on transmission stack
  -> attenuation b = -log(T)
  -> optional negative attenuation clipping
  -> optional per-projection shift on attenuation stack
  -> optional truncation correction by detector-column padding
  -> optional transpose for TIGRE
  -> TIGRE FDK with configured Zeiss geometry and angle sign
  -> TIFF stack/log/report/plot outputs
```

Projection folders are selected in `gui.py` under the "Data paths" group. Filenames are not glob-sorted for reconstruction; they are taken from the metadata CSV and resolved against the projection folder by `metadata.resolve_projection_path`. TIFF support is `.tif`, `.tiff`, `.TIF`, and `.TIFF`. Reference images are glob-sorted by `io_utils.list_tiff_files`.

Images are loaded by `tifffile.imread`, squeezed to 2-D, converted to contiguous `float32`, and optionally binned. Binning trims dimensions to a multiple of 2 or 4 and averages blocks. There is no arbitrary crop in the FDK version. Projection order follows the metadata rows, with optional duplicate 0/360 endpoint removal and optional reverse order.

## 3. Metadata Flow Summary

Metadata support is CSV-based. `metadata.find_metadata_csv` accepts either a CSV path or a folder. It prefers `projection_geometry.csv`, otherwise the first CSV in the folder or recursive subtree.

Supported/used metadata fields:

- Filename column, guessed from names such as `tiff_file`, `filename`, `file`, `projection`, etc.
- Degree angle column, guessed from names such as `angle_deg`, `theta_deg`, `rotation`, etc.
- Optional radian angle column, guessed from names such as `angle_rad`, `theta_rad`, etc. If present and numeric, it is preferred over degrees.
- Optional `x_shift_px` and `y_shift_px` columns, with several aliases.

The software does not parse Zeiss TXRM/XRM/XML/TIFF tags/JSON/YAML metadata directly. Detector/source distances, effective pixel size, voxel size, and volume size are entered in the GUI or loaded from JSON config.

Units:

- Angles are converted to `float32` radians for TIGRE.
- GUI geometry distances are in millimeters.
- Effective pixel and voxel sizes are entered in micrometers.
- Shift columns are detector image pixels after binning convention is selected/used by the loaded binned stack.
- Detector offsets are entered in detector pixels and converted to millimeters.

Optical magnification handling:

- The GUI asks for effective pixel size at object plane.
- `tigre_reconstruction.computed_detector_pixel_mm` computes TIGRE detector-plane pixel size as `effective_pixel_size_mm * DSD / DSO`.
- Code comments state that Zeiss optical/camera magnification is assumed to already be baked into the effective object-plane pixel size.

Duplicate endpoint handling:

- `validate_metadata` detects first/last angles that wrap to the same angle, such as 0 and 360 degrees.
- If enabled, the last metadata row is dropped.

## 4. Reference Image Handling Summary

The existing workflow assumes flat/reference images, not dark-current images.

- `preprocessing.compute_average_flat_field` loads all TIFFs from the reference folder, validates raw shape against the first projection before binning, applies the same binning factor, and averages in `float64` before returning `float32`.
- Flat-field correction is `T = I / F`.
- The flat image is protected by `epsilon` with `flat_safe = max(flat, epsilon)`.
- Transmission can be clipped to `[min_transmission, max_transmission]`.
- Missing references raise `FileNotFoundError`.
- Reference images are loaded from a separate folder chosen by the user, not through metadata-linked locations.
- No exposure-time normalization or dark subtraction is implemented.

For TV-MBIR, dark-field support should be added, but the default must reproduce the FDK behavior when no dark folder is supplied:

```text
b = -log(I / F)
```

or, with dark field:

```text
b = -log((I - D) / (F - D))
```

## 5. Preprocessing Pipeline Summary

Exact FDK preprocessing sequence:

1. Validate metadata and projection paths.
2. Read projection TIFFs as `float32`, with optional 1x/2x/4x binning during read.
3. Read and average reference TIFFs with the same binning.
4. Apply orientation flips to the raw stack and averaged flat field.
5. Compute transmission with `T = I / F`.
6. Optionally clip transmission.
7. Optionally apply per-projection shift correction to the transmission stack.
8. Compute attenuation line integrals with `p = -ln(T)`.
9. Optionally clamp negative attenuation values to zero.
10. Optionally apply per-projection shift correction to the attenuation stack instead.
11. Optionally apply truncation correction after `-ln`.
12. Optionally transpose attenuation with `np.transpose(stack, (0, 2, 1))` before TIGRE.

The FDK software reconstructs line-integral attenuation projections, not raw intensity and not transmission. The new MBIR solver must use the same `b = -log(T)` convention.

No bad-pixel correction, ring-artifact correction, general cropping, or dark-current subtraction exists in the FDK version.

## 6. Geometry Mapping Summary

Geometry is constructed in `tigre_reconstruction.build_tigre_geometry`.

Input dataclass: `GeometryParams`

- `effective_pixel_size_um`
- `DSO_mm`
- `DSD_mm`
- `Nx`, `Ny`, `Nz`
- `voxel_size_um`
- object offsets in mm
- detector offsets in pixels
- center-offset enable/value/method/sign

TIGRE mapping:

- `geo.DSD = DSD_mm`
- `geo.DSO = DSO_mm`
- `geo.accuracy = 0.5`
- `geo.nDetector = [detector_rows, detector_cols]`
- `geo.dDetector = [effective_pixel_mm * DSD/DSO, effective_pixel_mm * DSD/DSO]`
- `geo.sDetector = geo.nDetector * geo.dDetector`
- `geo.nVoxel = [Nz, Ny, Nx]`
- `geo.dVoxel = [voxel_size_mm, voxel_size_mm, voxel_size_mm]`
- `geo.sVoxel = geo.nVoxel * geo.dVoxel`
- `geo.offOrigin = [object_offset_z_mm, object_offset_y_mm, object_offset_x_mm]`
- `geo.offDetector = [vertical_offset_mm, horizontal_offset_mm]`

Detector stack axis convention:

- Projection stack is shaped `(n_angles, rows/v, cols/u)`.
- Reconstructed volume is treated as `(z, y, x)`.

Angle handling:

- Metadata angles are radians in normal workflow.
- The GUI default is `invert_angle_sign_for_tigre = True`, so TIGRE receives `-angles_rad`.
- Optional reverse order is available during validation.
- Duplicate 0/360 endpoint removal is enabled by default.

## 7. Center-Offset Correction Summary

Center offset is user-controlled in pixels. The GUI supports:

- Manual center-offset value.
- Enable/disable toggle.
- Correction method: `detector_offset` or `image_shift`.
- Center-shift sign convention `+1` or `-1`.
- Manual preview search over shift candidates.
- Automatic search using focus metrics.
- Fine search around selected shift.

Detector-offset method:

```text
center_offset_mm = center_shift_sign * center_offset_px * d_detector_mm
geo.offDetector[1] += center_offset_mm
```

Image-shift method:

```text
projection_stack shifted along detector column axis by center_shift_sign * center_offset_px
```

Preview search:

- Generates candidate shifts with `generate_shift_values`.
- Reconstructs a single slice or thin axial band using FDK.
- Computes gradient energy, Laplacian variance, entropy, and a combined score.
- Saves `center_offset_search_metrics.csv` when an output folder is available.

This logic is highly reusable for TV-MBIR. Version 1 should preserve detector-offset center correction and use the same sign convention.

## 8. Projection Drift Correction Summary

The FDK project implements application of per-projection shifts from metadata columns. It does not estimate drift from images.

Implementation:

- `shift_correction.apply_projection_shifts` applies SciPy `ndimage.shift` to each projection.
- Stack shape is `(num_proj, rows, cols)`.
- Formula:

```text
row_shift = y_sign * y_shift_px[i]
col_shift = x_sign * x_shift_px[i]
```

- `shift_interpretation_sign` can invert both signs.
- Interpolation order is configurable; default is 1.
- Boundary mode is configurable; default is `nearest`.
- Shifts can be applied before `-ln` on transmission, or after `-ln` on attenuation.
- Defaults from the user manual: shift correction enabled, stage `transmission`, Mode A, x sign `+1`, y sign `+1`, interpretation `+1`.
- The GUI includes a four-mode shift-convention test that reconstructs small FDK volumes for sign comparison.
- Before/after shift previews can be saved.

Version 1 TV-MBIR should apply drift shifts directly to projection images before reconstruction and document that future versions may encode per-view drift in geometry.

## 9. GUI Architecture Summary

Framework:

- `qt_compat.py` imports PySide6 first and falls back to PyQt5.
- `gui.py` uses Qt widgets and Matplotlib `FigureCanvasQTAgg`.

Main classes:

- `Worker(QObject)`: runs a callable in a `QThread`, emits `finished`, `failed`, and `log` signals.
- `MicroCTReconstructionWindow(QMainWindow)`: all GUI controls, state, plotting, validation, preprocessing, center search, shift tests, FDK run, config save/load.

Long tasks:

- `compute_flat_field`, `apply_flat_correction`, `compute_attenuation_stack`, `run_center_shift_search`, `run_reconstruction`, and `run_shift_convention_test` run through `_start_worker`.
- Buttons are disabled while a worker is active.
- Logs stream into a QTextEdit and are saved to `processing_log.txt`.

Reusable GUI pieces:

- Worker-thread pattern.
- Path and metadata panels.
- Preprocessing controls.
- Geometry controls.
- Center-offset and shift controls.
- Live logging.
- Matplotlib image display and diagnostic plots.

Items needing cleanup:

- `gui.py` is monolithic and should be split into panels/widgets for MBIR.
- No cancellation signal is provided once native TIGRE begins.
- Config is flat JSON; MBIR should use nested YAML.

## 10. Logging/Output Summary

Outputs are written directly under the selected output folder, not under timestamped run folders.

Existing outputs include:

- `averaged_flat_field.tif`
- `transmission_stack/` if enabled
- `attenuation_stack/` if enabled
- `reconstruction.tif`
- `reconstruction_stack/` if enabled
- `processing_log.txt`
- `geometry_sanity_report.txt`
- `center_offset_search_metrics.csv`
- `shift_debug_*` folders
- `shift_mode_tests/` folders with slice PNGs, JSON configs, and geometry reports

Logging:

- Timestamped GUI log lines are kept in memory and written with `write_processing_log`.
- Geometry and preprocessing reports are appended to FDK report text.

TV-MBIR should introduce timestamped run folders and subfolders for preprocessing, alignment, FDK initialization, MBIR outputs, metrics, and reports.

## 11. Performance/GPU Handling Summary

- TIGRE import is checked by `tigre_available`.
- FDK uses `tigre.algorithms.fdk`.
- GPU ID is not configurable in the FDK GUI.
- Memory estimates are shown for projection stacks and FDK volumes.
- Projection stacks and volumes are converted to `float32`.
- No CuPy integration exists.
- Native TIGRE/CUDA interruption risks are logged and shown for large jobs.

TV-MBIR needs explicit GPU ID plumbing where supported by TIGRE and optional CuPy for TV operations. Version 1 should be matrix-free and shape-safe, with NumPy fallback for TV.

## 12. Existing Known-Good Settings To Preserve

From `AppConfig`, manual, and GUI defaults:

- Effective object-plane pixel size: `11.0 um`.
- Voxel size: `11.0 um`.
- Remove duplicate 0/360 endpoint: enabled.
- Invert angle sign for TIGRE: enabled.
- Clip transmission: enabled.
- Minimum transmission: `1e-6`.
- Maximum transmission: `2.0`.
- Maximum transmission for log: `2.0`.
- Set negative attenuation to zero: enabled.
- Binning factor: `1`, allowed values 1, 2, 4.
- Projection shift correction: enabled.
- Shift stage: transmission, before `-ln`.
- Shift mode default: Mode A, row `+y`, col `+x`.
- Shift interpretation: apply shifts as correction values.
- Shift interpolation: SciPy order 1, mode `nearest`.
- FDK filter: `ram_lak`.
- Center-offset method default: detector offset.
- TIGRE volume convention: returned volume indexed `[z, y, x]`.

## 13. Reusable Components List

Directly reusable/adaptable:

- TIFF listing, reading, binning, and stack saving from `io_utils.py`.
- CSV metadata column guessing, validation, duplicate endpoint handling, and projection record model from `metadata.py`.
- Averaged flat-field computation, orientation flips, transmission clipping, `-log` conversion, and truncation correction from `preprocessing.py`.
- TIGRE geometry construction and Zeiss effective-pixel-size convention from `tigre_reconstruction.py`.
- `run_fdk` as an initialization path, with a cleaner wrapper.
- Center-offset candidate generation, detector-offset conversion, preview reconstruction, and metrics from `center_shift_search.py`.
- Projection shift application and shift sanity reporting from `shift_correction.py`.
- Qt compatibility and worker-thread pattern from `qt_compat.py`/`gui.py`.
- Matplotlib preview and plot helpers from `plotting.py`.
- Timestamped logging from `logging_utils.py`.

## 14. Components That Should Be Refactored

- Split the monolithic GUI into parameter panels, preview widgets, plot widgets, and a main window.
- Replace flat JSON config with nested YAML.
- Separate preprocessing orchestration from GUI methods so CLI, GUI, tests, and validation scripts share the same path.
- Wrap TIGRE forward/backprojection in a dedicated operator class instead of calling TIGRE directly from solvers.
- Keep ADMM/CG/TV math independent of Zeiss/TIGRE I/O.
- Add formal tests for TV adjointness, CG, preprocessing consistency, and small phantom reconstruction.
- Add timestamped run-folder management.
- Add cancellation-aware Python loops around ADMM/CG.

## 15. Risks And Ambiguities

- The existing FDK app does not parse raw Zeiss TXRM/XRM metadata; it expects extracted TIFFs and CSV metadata.
- No dark-current correction exists in the FDK path, so dark support in MBIR is new and must be validated separately.
- The default shift sign in the code is Mode A, while the user manual text also recommends testing Mode B for vertically inverted TIFFs. Preserve Mode A default but keep sign controls visible.
- The exact TIGRE Python backprojector API can vary by TIGRE build. The new wrapper should isolate this compatibility risk.
- GPU ID support depends on the local TIGRE installation.
- Detector pixel size has two possible meanings in configs: effective object-plane pixel size vs detector-plane pixel size. The MBIR config must state this clearly.
- Optional attenuation transpose exists as an escape hatch; it should be preserved for compatibility but disabled by default.
- Center-offset units are pixels after the currently selected binning. This must remain explicit.

## 16. Recommended Migration Plan Into TV-MBIR

1. Create a clean `tv_mbir_ct_recon` package with separate modules for config, I/O, metadata, references, preprocessing, geometry, alignment, TIGRE operators, TV operators, CG, ADMM, metrics, plotting, output, worker, and GUI.
2. Port the validated FDK-compatible preprocessing and geometry path first.
3. Add a `run_fdk_initialization` helper using the same TIGRE geometry, angle sign, center-offset, transpose, and shift conventions.
4. Add tests confirming TV gradient/adjoint consistency, preprocessing formula consistency, and CG correctness.
5. Implement `TigreConeBeamOperator` with `forward`, `backproject`, and `normal`.
6. Implement ADMM TV-MBIR using matrix-free CG and the TV adjoint convention verified by tests.
7. Add timestamped output folders and metrics CSV/plots.
8. Add GUI panels for dataset, preprocessing, geometry, alignment, FDK initialization, MBIR parameters, progress/logging, metrics plots, and slice previews.
9. Add validation scripts for small phantom reconstructions and FDK compatibility against the old project.
10. Optimize TV operations with CuPy when available, while keeping a NumPy fallback.
