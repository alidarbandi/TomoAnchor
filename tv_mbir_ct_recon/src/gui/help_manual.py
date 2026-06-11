from __future__ import annotations

from .qt_compat import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QTextCursor,
    QVBoxLayout,
    QWidget,
)


class HelpManualWidget(QWidget):
    """Searchable in-app user manual."""

    def __init__(self) -> None:
        super().__init__()
        self._last_query = ""

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search help")
        self.search_button = QPushButton("Search")
        self.search_status = QLabel("Type a keyword and press Search. Press Search again for the next match.")
        self.search_status.setWordWrap(True)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Keyword"))
        search_row.addWidget(self.search_edit, 1)
        search_row.addWidget(self.search_button)

        self.manual = QTextBrowser()
        self.manual.setReadOnly(True)
        self.manual.setOpenExternalLinks(False)
        self.manual.setHtml(HELP_HTML)

        layout = QVBoxLayout(self)
        layout.addLayout(search_row)
        layout.addWidget(self.search_status)
        layout.addWidget(self.manual, 1)

        self.search_button.clicked.connect(self.search_next)
        self.search_edit.returnPressed.connect(self.search_next)

    def search_next(self) -> None:
        query = self.search_edit.text().strip()
        if not query:
            self.search_status.setText("Enter a keyword to search the manual.")
            return
        if query != self._last_query:
            self.manual.moveCursor(QTextCursor.Start)
            self._last_query = query
        if self.manual.find(query):
            self.search_status.setText(f"Showing next match for: {query}")
            return
        self.manual.moveCursor(QTextCursor.Start)
        if self.manual.find(query):
            self.search_status.setText(f"Wrapped to first match for: {query}")
        else:
            self.search_status.setText(f"No matches found for: {query}")


HELP_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<style>
body {
  color: #F4F8FF;
  background-color: #273A59;
  font-family: Segoe UI, Arial, sans-serif;
  font-size: 10pt;
  line-height: 1.45;
}
h1 {
  font-size: 23pt;
  margin: 0 0 8px 0;
  color: #FFFFFF;
}
h2 {
  font-size: 17pt;
  margin-top: 28px;
  padding-bottom: 4px;
  border-bottom: 1px solid #8EAEE0;
  color: #FFFFFF;
}
h3 {
  font-size: 13pt;
  margin-top: 20px;
  color: #DDE8F7;
}
h4 {
  font-size: 11pt;
  margin-top: 14px;
  color: #FFFFFF;
}
p, li {
  margin-top: 5px;
  margin-bottom: 5px;
}
a {
  color: #A8C7F7;
}
table {
  border-collapse: collapse;
  margin: 8px 0 16px 0;
  width: 100%;
}
th {
  background-color: #354F75;
  color: #FFFFFF;
  border: 1px solid #8EAEE0;
  padding: 6px;
}
td {
  background-color: #1F2F49;
  color: #F4F8FF;
  border: 1px solid #5F78A0;
  padding: 6px;
  vertical-align: top;
}
.note {
  background-color: #1F2F49;
  border: 1px solid #8EAEE0;
  padding: 9px;
  margin: 10px 0;
}
.warn {
  background-color: #3A3148;
  border: 1px solid #FFCC80;
  padding: 9px;
  margin: 10px 0;
}
pre {
  background-color: #1F2F49;
  color: #DDE8F7;
  border: 1px solid #5F78A0;
  padding: 9px;
  white-space: pre-wrap;
}
code {
  color: #DDE8F7;
}
</style>
</head>
<body>
<h1 id="top">TomoAnchor User Manual</h1>
<p>
This help tab is the working manual for TomoAnchor, a Zeiss Versa cone-beam CT reconstruction application.
It explains the full workflow, every selectable GUI field and button, the right-side tabs, the solver and memory choices,
the new fast anchor-guided workflow, the warning dialogs, convergence graphs, troubleshooting, and frequently asked questions.
</p>

<div class="note">
Suggested starting point for most data: use flat correction, use dark correction if you have dark images, keep Negative log on,
keep Clip transmission on, keep Set negative attenuation to zero on, validate metadata before changing geometry, run FDK first,
then run MBIR with Solver set to auto and Memory mode set to auto.
For sparse or low-exposure scans with a separate high-exposure anchor set, use the Fast anchor-guided reconstruction panel:
start with Run Fast, 5 MBIR-lite sweeps, lowres FDK factor 2, tune anchors 6, QC anchors 6, and keep QC anchors held out.
</div>

<h2 id="contents">Contents</h2>
<ul>
  <li><a href="#workflow">Recommended workflow</a></li>
  <li><a href="#left-panel">Left-side control panel</a></li>
  <li><a href="#fast-recon">Fast anchor-guided reconstruction</a></li>
  <li><a href="#make-prior-guide">Make Prior guide</a></li>
  <li><a href="#tabs">Right-side tabs</a></li>
  <li><a href="#solvers">Solver selection guide</a></li>
  <li><a href="#memory">Memory mode guide</a></li>
  <li><a href="#warnings">Run warnings and how to respond</a></li>
  <li><a href="#convergence">Metrics and convergence</a></li>
  <li><a href="#tutorial">Full tutorial</a></li>
  <li><a href="#fast-tutorial">Fast anchor-guided tutorial</a></li>
  <li><a href="#yaml">Advanced YAML-only options</a></li>
  <li><a href="#troubleshooting">Troubleshooting</a></li>
  <li><a href="#faq">FAQ</a></li>
</ul>

<h2 id="workflow">Recommended Workflow</h2>
<pre>
Load folders and metadata
        |
        v
Load Metadata CSV -> Validate Metadata -> inspect Validation tab
        |
        v
Preview raw projection -> preprocess preview -> inspect corrected images
        |
        v
Check geometry, flips, transpose, center and drift alignment
        |
        v
Run FDK -> inspect Tomogram View
        |
        v
Run MBIR -> watch Progress, Metrics, Device Monitor, MBIR Preview
        |
        v
Inspect final volume, saved outputs, diagnostics, config, and plots
</pre>
<p>
FDK is the practical first reconstruction: it confirms metadata, geometry, center alignment, orientation, and GPU availability.
MBIR should be run after FDK looks physically plausible.
</p>
<h3 id="fast-workflow-summary">Fast Workflow for Anchor Data</h3>
<pre>
Main sparse/low-exposure dataset
        +
Separate high-exposure anchor dataset
        |
        v
Run Fast
        |
        v
Main preprocessing + anchor preprocessing
        |
        v
Anchor split: recon anchors + tune anchors + held-out QC anchors
        |
        v
FDK filter sweep scored on tune anchors
        |
        v
Training-free weak TV prior and confidence map
        |
        v
Prior-anchored MBIR-lite using main data and allowed anchors
        |
        v
QC report with held-out anchor residuals
</pre>
<p>
Use the fast workflow when the main scan is sparse, noisy, or low exposure, and you also acquired a smaller number of
high-SNR anchor projections. The anchor projections are not treated as training data. They are measured views used for
filter selection, prior sanity checks, statistical weighting, and independent QC.
</p>

<h2 id="left-panel">Left-Side Control Panel</h2>
<p>The left panel contains all inputs that control loading, preprocessing, geometry, alignment, initialization, MBIR, GPU use, and run control.</p>

<h3 id="dataset-input">Dataset Input</h3>
<table>
<tr><th>Field or button</th><th>What it does</th><th>Typical use</th></tr>
<tr>
<td>Projection folder</td>
<td>Folder containing the raw projection TIFF images. Filenames are matched against the metadata CSV filename column.</td>
<td>Required. Choose the folder with the scan projections.</td>
</tr>
<tr>
<td>Flat/reference folder</td>
<td>Folder containing flat-field/reference images. The app averages them and uses them as the incident-beam reference.</td>
<td>Required when Use flat correction is on. Recommended on by default for normal CT.</td>
</tr>
<tr>
<td>Dark folder</td>
<td>Folder containing dark-current images. If provided and Use dark correction is on, the average dark image is subtracted from projections and flats.</td>
<td>Use when your acquisition includes dark images. If the folder is blank, the app logs that no dark field is used.</td>
</tr>
<tr>
<td>Metadata CSV</td>
<td>Path to the metadata CSV, or a folder containing one. If a folder is selected, the app prefers projection_geometry.csv, then the first CSV it finds.</td>
<td>Required. This CSV supplies filenames, angles, and optionally drift shifts.</td>
</tr>
<tr>
<td>Output folder</td>
<td>Parent folder for reconstruction run folders. Each run writes a timestamped output folder with config, logs, diagnostics, FDK and/or MBIR outputs.</td>
<td>Required before running preprocessing, FDK, or MBIR from the pipeline.</td>
</tr>
<tr>
<td>Browse buttons</td>
<td>Open a folder or file chooser for the adjacent path field.</td>
<td>Use these to avoid path typos.</td>
</tr>
</table>

<h3 id="metadata">Metadata</h3>
<table>
<tr><th>Field or button</th><th>What it does</th><th>Guidance</th></tr>
<tr>
<td>Filename column</td>
<td>CSV column that contains projection TIFF filenames. The app auto-guesses common names such as tiff_file, filename, file, image, projection, or output_name.</td>
<td>Must point to the actual projection filename column. Relative filenames are resolved under Projection folder.</td>
</tr>
<tr>
<td>Angle column deg</td>
<td>CSV column containing projection angle values in degrees.</td>
<td>Required unless a valid radian column is selected. The degree column is also used for display and validation.</td>
</tr>
<tr>
<td>Angle column rad</td>
<td>Optional CSV column containing projection angles in radians. If selected and valid, it is used as the actual reconstruction angle source.</td>
<td>Use if your metadata has trusted radian angles. Otherwise leave at "No radian column / use degrees".</td>
</tr>
<tr>
<td>X shift column</td>
<td>Optional per-projection horizontal drift shift in pixels.</td>
<td>Select if the metadata contains measured x drift. Required for drift correction/export.</td>
</tr>
<tr>
<td>Y shift column</td>
<td>Optional per-projection vertical drift shift in pixels.</td>
<td>Select if the metadata contains measured y drift. Required for drift correction/export.</td>
</tr>
<tr>
<td>Remove duplicate 0/360 endpoint</td>
<td>When the first and last angles are the same physical view, this removes the final row after validation.</td>
<td>Recommended on by default. Many CT scans include both 0 and 360 degrees; keeping both overweights one view.</td>
</tr>
<tr>
<td>Reverse angle order</td>
<td>Reverses the validated filename/angle pairs.</td>
<td>Use only when the scan order is physically reversed relative to what TIGRE expects. If FDK looks mirrored or inconsistent, test this along with angle sign.</td>
</tr>
<tr>
<td>Load Metadata CSV</td>
<td>Reads the CSV, fills the column selectors, and logs row/column information.</td>
<td>First metadata action after selecting paths.</td>
</tr>
<tr>
<td>Choose Metadata Folder</td>
<td>Lets you select a folder and then loads a CSV from that folder.</td>
<td>Useful for Zeiss metadata folders where projection_geometry.csv is inside a metadata directory.</td>
</tr>
<tr>
<td>Projection preview index</td>
<td>Selects which projection is displayed when previewing raw or preprocessed images.</td>
<td>After validation the range is set to the number of validated projections.</td>
</tr>
<tr>
<td>Validate Metadata</td>
<td>Checks required columns, converts angles, verifies projection files exist, detects duplicate endpoints, fills detector rows/columns from the first image, and populates the Validation tab.</td>
<td>Run this before preprocessing, alignment, FDK, or MBIR. If validation fails, fix paths or column selections first.</td>
</tr>
<tr>
<td>Preview Raw Projection</td>
<td>Loads the selected projection with current binning/crop and displays it in the Preview tab.</td>
<td>Use before preprocessing to check that the image loads and orientation/crop choices make sense.</td>
</tr>
</table>

<h3 id="preprocessing">Preprocessing</h3>
<p>
Preprocessing converts raw intensity images into attenuation line integrals suitable for FDK and MBIR.
</p>
<pre>
Raw I, flat F, dark D
        |
        v
Transmission = (I - D) / max(F - D, epsilon)
        |
        v
Attenuation b = -log(Transmission)
</pre>
<table>
<tr><th>Field or button</th><th>What it does</th><th>Guidance</th></tr>
<tr>
<td>Use flat correction</td>
<td>Divides each projection by the averaged flat/reference image. With dark correction, it uses (I-D)/(F-D).</td>
<td>Recommended on by default. Turn off only for already corrected projection data.</td>
</tr>
<tr>
<td>Use dark correction if folder exists</td>
<td>Subtracts the averaged dark image from raw and flat images if a dark folder is supplied.</td>
<td>Recommended on when dark images exist. Leaving it on with a blank dark folder is safe; the app logs that no dark field was used.</td>
</tr>
<tr>
<td>Negative log</td>
<td>Converts transmission to attenuation using b = -log(transmission). This creates line-integral data for CT reconstruction.</td>
<td>It should be on by default for normal absorption CT. Turn it off only if the input stack is already attenuation/log data; the current GUI workflow expects attenuation for FDK/MBIR.</td>
</tr>
<tr>
<td>Epsilon</td>
<td>Small positive floor used to avoid division by zero and log(0).</td>
<td>Default 1e-6 is a good start. Increase only if dead pixels or zero flats cause extreme values; too large can bias attenuation low.</td>
</tr>
<tr>
<td>Clip transmission</td>
<td>Limits corrected transmission to the min/max range before the log transform.</td>
<td>Recommended on. It prevents zeros, hot pixels, or noisy flat correction from creating huge attenuation spikes.</td>
</tr>
<tr>
<td>Transmission min</td>
<td>Lower transmission bound used when Clip transmission is on.</td>
<td>Default 1e-6. Higher values reduce extreme attenuation but can suppress real high-absorption features.</td>
</tr>
<tr>
<td>Transmission max</td>
<td>Upper transmission bound used when Clip transmission is on and as the max log input in the config.</td>
<td>Default 2.0 allows mild transmission greater than 1 from noise or flat mismatch. Lower it if negative attenuation artifacts dominate.</td>
</tr>
<tr>
<td>Set negative attenuation to zero</td>
<td>After -log, any attenuation below zero is clamped to zero.</td>
<td>Recommended on for standard absorption CT. Negative values usually mean noise, overcorrection, or bright-field mismatch.</td>
</tr>
<tr>
<td>Binning Y</td>
<td>Detector-row binning during image loading. A value of 2 averages 2 rows into 1.</td>
<td>Use for faster tests or lower-memory reconstructions. It reduces detector resolution before crop and reconstruction.</td>
</tr>
<tr>
<td>Binning X</td>
<td>Detector-column binning during image loading.</td>
<td>Same guidance as Binning Y. Keep X and Y equal unless you intentionally want anisotropic detector sampling.</td>
</tr>
<tr>
<td>Crop top, bottom, left, right</td>
<td>Pixels removed after binning from each projection edge.</td>
<td>Use to remove borders, detector artifacts, or reduce field of view. Remember that cropping changes detector rows/columns after validation.</td>
</tr>
<tr>
<td>Flip horizontal</td>
<td>Mirrors projections left-right before flat correction output is used.</td>
<td>Use when FDK orientation is horizontally mirrored or the center/drift sign appears reversed.</td>
</tr>
<tr>
<td>Flip vertical</td>
<td>Mirrors projections top-bottom.</td>
<td>Use when reconstructed z orientation or projection view is vertically flipped.</td>
</tr>
<tr>
<td>Transpose attenuation for TIGRE</td>
<td>Sends TIGRE projection input as [angles, columns, rows] instead of [angles, rows, columns].</td>
<td>Use only if FDK geometry appears transposed or rows/columns are swapped. Validate with a small FDK first.</td>
</tr>
<tr>
<td>Preview image</td>
<td>Selects what image type Preview Selected Image will show: Raw projection, Averaged flat field, Flat-field corrected projection, or Attenuation projection.</td>
<td>Use these views to diagnose flat-field correction and log conversion before reconstructing.</td>
</tr>
<tr>
<td>Compute Preprocessed Preview Stack</td>
<td>Runs preprocessing for the validated scan and stores raw, flat, transmission, and attenuation previews in memory.</td>
<td>Do this before center search or detailed preview checks. It is also a good low-risk test of data loading.</td>
</tr>
<tr>
<td>Preview Selected Image</td>
<td>Displays the selected Preview image type at the Projection preview index.</td>
<td>Use repeatedly while changing preview index, levels, flips, crop, or binning.</td>
</tr>
</table>

<h3 id="image-display">Image Display</h3>
<table>
<tr><th>Control</th><th>What it does</th><th>Guidance</th></tr>
<tr>
<td>Histogram level widget</td>
<td>Shows the intensity histogram for the active image. Drag the low and high markers to change display contrast. Turn on Log histogram when you want to inspect low-count tails more clearly.</td>
<td>This affects viewing only; it does not change reconstruction data. The log/linear histogram choice stays active until you change it back.</td>
</tr>
<tr>
<td>Auto 1-99%</td>
<td>Sets display limits to the 1st and 99th percentile of the active image or volume.</td>
<td>Best default for inspecting projections and volumes without being dominated by outliers.</td>
</tr>
<tr>
<td>Full Range</td>
<td>Sets display limits to the full finite min/max range.</td>
<td>Use to check if extreme outliers are present.</td>
</tr>
<tr>
<td>Zoom In, Zoom Out, Fit</td>
<td>Changes the displayed image zoom. Fit resets to the full image.</td>
<td>Viewing only. The zoom label reports the active image zoom percentage.</td>
</tr>
</table>

<h3 id="geometry">Geometry</h3>
<p>
Geometry tells TIGRE how the cone-beam scan was acquired. Incorrect geometry is the most common cause of blurry,
stretched, doubled, or displaced reconstructions.
</p>
<table>
<tr><th>Field</th><th>What it does</th><th>Guidance</th></tr>
<tr>
<td>DSD mm</td>
<td>Source-to-detector distance in millimeters.</td>
<td>Required. Must be larger than DSO.</td>
</tr>
<tr>
<td>DSO mm</td>
<td>Source-to-object/rotation-center distance in millimeters.</td>
<td>Required. Used with DSD and effective pixel size to compute detector pixel size at the detector.</td>
</tr>
<tr>
<td>Effective pixel um</td>
<td>Effective object-space pixel size in micrometers. The app converts it to detector pixel size using DSD/DSO.</td>
<td>Use the Zeiss effective pixel size for the scan. It should match the voxel size if reconstructing at native sampling.</td>
</tr>
<tr>
<td>Voxel z mm, Voxel y mm, Voxel x mm</td>
<td>Voxel spacing of the reconstruction grid in millimeters.</td>
<td>Often set to effective_pixel_um / 1000 for isotropic voxels. Larger values reduce resolution and memory.</td>
</tr>
<tr>
<td>Detector rows, Detector columns</td>
<td>Projection image shape after binning/crop. These fields are read-only and filled during validation.</td>
<td>If these are zero, validate metadata. If they look swapped, inspect Transpose attenuation for TIGRE.</td>
</tr>
<tr>
<td>Nz, Ny, Nx</td>
<td>Reconstruction volume voxel counts in z, y, and x.</td>
<td>Required for FDK/MBIR. Full detector-sized runs can be very large; use smaller grids for debug runs.</td>
</tr>
<tr>
<td>Detector offset v px</td>
<td>Vertical detector offset in detector pixels.</td>
<td>Use if scanner metadata gives detector vertical offset. Usually 0 unless known.</td>
</tr>
<tr>
<td>Detector offset u px</td>
<td>Horizontal detector offset in detector pixels.</td>
<td>Use if scanner metadata gives detector horizontal offset. Center correction adds to this in detector-offset mode.</td>
</tr>
<tr>
<td>Invert angle sign for TIGRE</td>
<td>Multiplies reconstruction angles by -1 before passing them to TIGRE.</td>
<td>Recommended on by default for this workflow. If FDK looks mirrored or rotates the wrong way, compare on/off with a small FDK.</td>
</tr>
</table>

<h3 id="alignment">Alignment</h3>
<table>
<tr><th>Field or button</th><th>What it does</th><th>Guidance</th></tr>
<tr>
<td>Use center correction</td>
<td>Applies Center offset px as a horizontal detector-center correction.</td>
<td>Recommended on after choosing a center offset. Turn off only for diagnosis.</td>
</tr>
<tr>
<td>Center offset px</td>
<td>Horizontal center-of-rotation offset in detector pixels.</td>
<td>Small errors blur FDK and MBIR. Use the center search tools, then inspect FDK.</td>
</tr>
<tr>
<td>Center sign</td>
<td>Sign convention for converting center offset pixels into detector offset direction.</td>
<td>Usually +1. If changing center offset improves in the opposite direction, try -1.</td>
</tr>
<tr>
<td>Search start px, Search end px, Search step px</td>
<td>Defines candidate center shifts for center search.</td>
<td>Start coarse, for example -10 to 10 px by 1 px, then use fine search around the best shift.</td>
</tr>
<tr>
<td>Automatic metric</td>
<td>Chooses which image sharpness metric is used to recommend a center shift.</td>
<td>Combined score is the default recommendation. Always inspect the preview images; metrics are a guide, not an absolute truth.</td>
</tr>
<tr>
<td>Combined score</td>
<td>Weighted score using gradient energy, laplacian variance, and entropy.</td>
<td>Default. It favors sharp, structured reconstructions without excessive disorder.</td>
</tr>
<tr>
<td>Gradient energy</td>
<td>Measures edge strength in the center-search preview.</td>
<td>Higher can mean sharper edges, but it can also favor noise.</td>
</tr>
<tr>
<td>Laplacian variance</td>
<td>Measures high-frequency sharpness.</td>
<td>Higher often means better focus. Be cautious if noise dominates.</td>
</tr>
<tr>
<td>Entropy</td>
<td>Measures image disorder. For recommendation, lower entropy is preferred.</td>
<td>Useful when the correct center produces a cleaner, less smeared slice.</td>
</tr>
<tr>
<td>Run Manual Preview Search</td>
<td>Reconstructs one-slice FDK previews for each shift but does not compute automatic metric recommendation.</td>
<td>Use when you want visual comparison only.</td>
</tr>
<tr>
<td>Run Automatic Search</td>
<td>Reconstructs preview shifts, computes metrics, updates the Metrics tab, and recommends a best shift.</td>
<td>Recommended starting method for center correction.</td>
</tr>
<tr>
<td>Run Fine Search Around Selected Shift</td>
<td>Creates a narrower search range around the currently selected shift with a smaller step and runs it.</td>
<td>Use after a coarse search to refine the center.</td>
</tr>
<tr>
<td>Preview selector</td>
<td>Slider through center-shift preview reconstructions.</td>
<td>Move through previews and inspect sharpness and symmetry.</td>
</tr>
<tr>
<td>Apply Selected Shift</td>
<td>Copies the currently selected preview shift into Center offset px.</td>
<td>Use after visual inspection.</td>
</tr>
<tr>
<td>Use Automatic Best Shift</td>
<td>Copies the automatic metric recommendation into Center offset px.</td>
<td>Use when the metric recommendation matches the visual best preview.</td>
</tr>
<tr>
<td>Use drift correction</td>
<td>Applies per-projection x/y shifts from metadata if valid shift columns exist.</td>
<td>Use when stage/sample drift has been measured. If shift columns are missing, the app logs that drift correction could not be applied.</td>
</tr>
<tr>
<td>Drift stage</td>
<td>Chooses whether drift shifts are applied to transmission images before -log or to attenuation images after -log.</td>
<td>transmission is often safer for physical intensity correction. attenuation can be useful if shifts were derived after log conversion.</td>
</tr>
<tr>
<td>Save Drift-Corrected Transmission TIFFs</td>
<td>Exports flat-field corrected, drift-corrected transmission TIFFs and a manifest CSV.</td>
<td>Use to inspect or archive drift correction. Requires flat folder and valid X/Y shift columns.</td>
</tr>
<tr>
<td>X shift sign, Y shift sign</td>
<td>Choose +1 or -1 sign convention for metadata drift shifts.</td>
<td>If drift correction worsens motion, try reversing the relevant sign and preview again.</td>
</tr>
</table>

<h3 id="fdk-init">FDK Initialization</h3>
<table>
<tr><th>Field</th><th>What it does</th><th>Guidance</th></tr>
<tr>
<td>Mode: fdk</td>
<td>Runs TIGRE FDK and uses the resulting volume as the MBIR starting point.</td>
<td>Best quality starting point when memory allows. For large streaming MBIR, implicit FDK may be skipped if unsafe.</td>
</tr>
<tr>
<td>Mode: existing_fdk</td>
<td>Loads an existing .npy FDK volume and resizes it if the MBIR debug grid differs.</td>
<td>Recommended for large full-resolution MBIR after you have already saved an FDK initialization. You can also use a checkpointed standard MBIR <code>mbir_final_volume.npy</code> here to continue from the latest saved state.</td>
</tr>
<tr>
<td>Mode: zeros</td>
<td>Starts MBIR from an all-zero volume.</td>
<td>Memory robust but may need more iterations and can converge slower.</td>
</tr>
<tr>
<td>Mode: constant</td>
<td>Starts MBIR from a uniform volume equal to Constant value.</td>
<td>Use only for controlled experiments or if a known background value helps.</td>
</tr>
<tr>
<td>Existing FDK volume .npy</td>
<td>Path to an existing FDK numpy volume for existing_fdk mode.</td>
<td>Use the saved fdk_initial_volume.npy from a previous run, or point it to a checkpointed standard MBIR <code>mbir_final_volume.npy</code>.</td>
</tr>
<tr>
<td>Constant value</td>
<td>Uniform starting value for constant mode.</td>
<td>Usually 0.0 unless you have a reason to initialize differently.</td>
</tr>
</table>

<h3 id="mbir-controls">MBIR Solver Controls</h3>
<table>
<tr><th>Field</th><th>What it does</th><th>Guidance</th></tr>
<tr>
<td>lambda TV</td>
<td>TV regularization weight. Larger values enforce smoother volumes and stronger edge-preserving denoising.</td>
<td>Default 0.001. Increase if noisy or streaky; decrease if edges are oversmoothed or fine features disappear.</td>
</tr>
<tr>
<td>rho</td>
<td>ADMM penalty parameter coupling the image gradient and TV shrinkage variables.</td>
<td>Mainly affects ADMM/CG speed and balance, not the final objective. Default 0.05. If ADMM residuals are unbalanced, adjust cautiously.</td>
</tr>
<tr>
<td>ADMM iterations</td>
<td>Maximum outer MBIR iterations. This is also used as the iteration limit for PDHG and Subset-TV.</td>
<td>Use 5-10 for tests, 20-50 for production starts, more only if metrics still improve.</td>
</tr>
<tr>
<td>Inner CG iterations</td>
<td>Maximum conjugate-gradient iterations inside each ADMM iteration.</td>
<td>ADMM only. Increase if CG residual stays high and objective does not improve; reduce for faster rough tests.</td>
</tr>
<tr>
<td>Primal tolerance</td>
<td>ADMM stopping threshold for the gradient-splitting consistency residual.</td>
<td>ADMM only. Default 1e-4. Lower is stricter and slower.</td>
</tr>
<tr>
<td>Dual tolerance</td>
<td>ADMM stopping threshold for dual residual.</td>
<td>ADMM only. Default 1e-4. Use together with primal tolerance.</td>
</tr>
<tr>
<td>Relative x-change tol.</td>
<td>Stopping threshold for relative change in the reconstruction volume between iterations.</td>
<td>Applies to all solvers. If the volume no longer changes meaningfully, the run can stop.</td>
</tr>
<tr>
<td>CG tolerance</td>
<td>Relative residual target for inner CG solves.</td>
<td>ADMM only. Lower values solve each ADMM subproblem more accurately but take longer.</td>
</tr>
<tr>
<td>Positivity</td>
<td>Clamps reconstructed attenuation values to be nonnegative after solver updates.</td>
<td>Recommended on for absorption CT. Turn off only for special signed reconstruction experiments.</td>
</tr>
<tr>
<td>TV epsilon</td>
<td>Small smoothing value inside isotropic TV norm.</td>
<td>Default 1e-8. Increase only if numerical instability appears in TV updates.</td>
</tr>
<tr>
<td>Solver</td>
<td>Chooses auto, admm, pdhg_low_memory, or streaming_subset_tv.</td>
<td>See the solver guide below. Recommended default is auto.</td>
</tr>
<tr>
<td>MBIR extra binning Y, MBIR extra binning X</td>
<td>Extra MBIR-only detector/volume downsampling beyond preprocessing binning.</td>
<td>Use for debug MBIR runs. Existing FDK .npy initialization is resized to the effective MBIR grid if needed.</td>
</tr>
<tr>
<td>Use every Nth projection</td>
<td>Uses a projection stride for MBIR debug runs.</td>
<td>1 uses all views. 2 uses every other view. Larger stride is faster but less complete and can create angular undersampling artifacts.</td>
</tr>
<tr>
<td>Memory mode</td>
<td>Chooses how TIGRE projection/backprojection operations are scheduled.</td>
<td>See the memory guide below. Recommended default is auto.</td>
</tr>
<tr>
<td>Projection batch size</td>
<td>Number of projection views processed per TIGRE batch in projection_streaming and ordered_subsets modes.</td>
<td>Default 64. Use the largest value that fits GPU memory. The app may cap or reduce it automatically for safety.</td>
</tr>
<tr>
<td>Ordered subsets</td>
<td>Number of interleaved projection subsets used by ordered_subsets memory mode.</td>
<td>Use values like 2, 4, or 8 for experiments. More subsets can speed early changes but makes each iteration less exact.</td>
</tr>
<tr>
<td>PDHG TV dual dtype</td>
<td>Storage dtype for PDHG TV dual arrays: float16 or float32.</td>
<td>float16 saves RAM and is the default. Use float32 if you have enough RAM and suspect precision-related instability.</td>
</tr>
<tr>
<td>Subset-TV step safety</td>
<td>Safety multiplier for streaming_subset_tv step size.</td>
<td>Default 0.7. Lower it if objective or previews oscillate; raise only cautiously if convergence is very slow and stable.</td>
</tr>
<tr>
<td>Use TIGRE GPU acceleration</td>
<td>Enables TIGRE GPU projection/backprojection and FDK.</td>
<td>Recommended on. CPU fallback is mainly for diagnostics and may be very slow or unavailable for realistic sizes.</td>
</tr>
<tr>
<td>GPU selection</td>
<td>Controls selected GPU IDs. Use auto, all, single, or a comma-separated list such as 0,1.</td>
<td>auto is recommended. Use all for multi-GPU distributed mode. Use a specific list to avoid a busy GPU.</td>
</tr>
<tr>
<td>Primary GPU ID</td>
<td>GPU ID used when GPU selection resolves to a single GPU.</td>
<td>Usually 0 on a one-GPU workstation.</td>
</tr>
</table>

<h2 id="fast-recon">Fast Anchor-Guided Reconstruction</h2>
<p>
The Fast anchor-guided reconstruction panel adds a training-free workflow for difficult samples where the main dataset is
sparse or low exposure, and a separate high-exposure anchor dataset is available. The workflow keeps the current FDK/MBIR
pipeline intact and writes new outputs under a separate fast_recon folder inside each run.
</p>
<div class="note">
Key idea: the prior is not the final reconstruction. The app creates a conservative denoising prior from the best FDK,
then MBIR-lite pulls the final volume back toward measured projection consistency.
</div>

<h3 id="make-prior-guide">Make Prior: What It Does</h3>
<p>
Make Prior is a training-free prior-building stage between the FDK sweep and MBIR-lite. It does not create the final
reconstruction. It starts from the selected full-resolution FDK volume, builds a support mask, robustly normalizes the
volume, generates one denoised candidate for each configured TV weight, scores those candidates against tune-anchor
projection agreement and image-preservation checks, then saves one selected prior plus a voxelwise confidence map.
MBIR-lite uses that saved prior as a soft guide while still fitting measured projection data.
</p>
<pre>
best FDK from fast_recon/fdk_sweep/fdk_best.npy
        |
        v
robust normalization + support mask + FDK noise estimate
        |
        v
TV-denoise each Prior TV weight candidate
        |
        v
score each candidate with tune anchors, edge retention, and correction size
        |
        v
select accepted candidate with lowest background noise
        |
        v
save prior_selected, prior_confidence, prior_difference, prior_metrics.csv
</pre>
<p>
Default behavior is one denoising run per entry in <code>prior.tv_weights</code>. With the default list
<code>[0.005, 0.01, 0.02, 0.04]</code> and <code>prior.tv_iterations: 50</code>, Make Prior performs
4 denoising candidate runs, each with 50 TV iterations, then chooses the safest acceptable result. If no candidate
passes the conservative acceptance checks, the weakest TV candidate is used as a fallback.
</p>
<p>
The prior is intentionally conservative. A good prior removes obvious FDK noise and streak texture but should not erase
small real features. If the prior image looks cleaner but visibly loses fine structure, remove stronger TV weights or
lower the later MBIR-lite rho prior.
</p>
<div class="note">
Prior tuning parameters are stored in YAML under the <code>prior</code>, <code>prior_scoring</code>,
<code>confidence</code>, and sometimes <code>mbir_lite</code> sections. In this project you can edit
<code>config.mbir.yaml</code> in the project root, or start from the example templates
<code>tv_mbir_ct_recon/examples/fast_recon_config.yaml</code> and
<code>tv_mbir_ct_recon/examples/sample_config.yaml</code>. Every run also saves the exact active settings to
<code>output/run_YYYYMMDD_HHMMSS/config_used.yaml</code>, and Make Prior writes a stage-specific copy to
<code>output/run_YYYYMMDD_HHMMSS/fast_recon/prior/prior_config_used.yaml</code>.
</div>

<h3 id="make-prior-parameters">Make Prior Parameters</h3>
<table>
<tr><th>Parameter</th><th>What it does</th><th>How to adjust it</th></tr>
<tr>
<td>Prior TV weights</td>
<td>List of TV denoising strengths tested as separate prior candidates.</td>
<td>Default 0.005, 0.01, 0.02, 0.04. Add or keep smaller values when fine detail is at risk. Add larger values only when FDK is visibly noisy or streaky. If strong candidates keep winning but erase features, remove the largest weights first.</td>
</tr>
<tr>
<td>prior.tv_iterations</td>
<td>Number of iterations used by the TV denoiser for each candidate.</td>
<td>Default 50. Increase only if a chosen TV weight is not converging enough to clean the prior. Reduce if runtime is too long or if the same TV weights look stronger than expected. Usually tune TV weights before tuning iterations.</td>
</tr>
<tr>
<td>prior.tv_epsilon</td>
<td>Small smoothing constant used in TV gradient calculations.</td>
<td>Default 1.0e-4. Usually leave unchanged. Change only for solver experimentation or numerical stability issues.</td>
</tr>
<tr>
<td>prior.slab_depth and prior.slab_overlap</td>
<td>Process large volumes in overlapping z slabs during TV denoising.</td>
<td>Default slab depth 96 and overlap 12. Lower slab depth to reduce RAM use. Increase overlap if you suspect slab boundary artifacts. If memory allows, prefer enough overlap to blend boundaries smoothly.</td>
</tr>
<tr>
<td>prior.normalize_low_percentile and normalize_high_percentile</td>
<td>Robust intensity percentiles used to normalize the FDK before candidate scoring.</td>
<td>Defaults 0.5 and 99.5. Usually leave unchanged. Adjust only if outliers or clipping make the normalized prior scale unstable across datasets.</td>
</tr>
<tr>
<td>prior.clip_min and clip_max</td>
<td>Intensity bounds applied after normalization.</td>
<td>Defaults -0.1 and 1.2. Tighten only if extreme outliers dominate denoising or confidence estimation. Widen only if true sample intensities are being clipped.</td>
</tr>
<tr>
<td>prior.support_gaussian_sigma_voxels and support_dilation_voxels</td>
<td>Build the object support mask used for background-noise and correction measurements.</td>
<td>Increase dilation if outer sample edges are being treated as background. Reduce dilation if too much empty area is included as sample. Keep support broad enough to cover the object but not so broad that background dominates scoring.</td>
</tr>
<tr>
<td>prior.noise_highpass_sigma_voxels</td>
<td>Gaussian scale used to estimate normalized FDK noise by high-pass MAD.</td>
<td>Default 1.5 voxels. Usually leave unchanged. Adjust only if the noise diagnostic is clearly measuring structure instead of noise.</td>
</tr>
<tr>
<td>prior_scoring.max_anchor_residual_ratio</td>
<td>Maximum allowed tune-anchor projection residual relative to baseline FDK.</td>
<td>Default 1.10. Lower it to be stricter and reject candidates that drift away from measured anchor data. Raise it slightly only if all reasonable candidates are failing and the prior is still too noisy.</td>
</tr>
<tr>
<td>prior_scoring.min_edge_retention</td>
<td>Minimum allowed edge-gradient retention on the strongest FDK edges.</td>
<td>Default 0.85. Raise it to protect fine edges more strongly. Lower it only when anchors support more smoothing and you accept some edge softening.</td>
</tr>
<tr>
<td>prior_scoring.max_correction_fraction</td>
<td>Maximum allowed total change from FDK inside the support mask.</td>
<td>Default 0.15. Lower it to keep the prior closer to FDK. Raise it only when the FDK is poor enough that a larger cleanup is genuinely needed.</td>
</tr>
<tr>
<td>prior_scoring.choose_lowest_background_noise_among_valid</td>
<td>Chooses the lowest background-noise candidate among candidates that pass all conservative checks.</td>
<td>Default true. Usually leave enabled. It helps choose the cleanest acceptable candidate after the safety checks have already filtered the risky ones.</td>
</tr>
<tr>
<td>confidence.C_min and C_max</td>
<td>Clamp the confidence map used by MBIR-lite.</td>
<td>Defaults 0.05 and 1.0. Raise C_min only if MBIR-lite is ignoring a good prior too much. Lower C_min if you want more freedom to move away from the prior. Reduce C_max if the prior is too dominant in confident regions.</td>
</tr>
<tr>
<td>confidence.tau_gradient_loss and blur_sigma_voxels</td>
<td>Control how confidence drops near places where the prior removed edges, and how smoothly confidence changes spatially.</td>
<td>Defaults 0.35 and 1.5. Lower tau_gradient_loss to be more suspicious of lost edges and reduce prior trust near them. Raise it if confidence is being reduced too aggressively. Increase blur to smooth the confidence map; decrease it if confidence needs to stay more local.</td>
</tr>
<tr>
<td>MBIR-lite rho prior</td>
<td>Not part of Make Prior itself, but controls how strongly MBIR-lite follows the saved confidence-weighted prior.</td>
<td>Default 0.05. Lower it if the final reconstruction follows an oversmoothed prior too closely. Raise it only when the prior is trustworthy and the projection data are too sparse or noisy to stabilize the result on their own.</td>
</tr>
</table>

<h3 id="make-prior-outputs">Make Prior Outputs</h3>
<table>
<tr><th>Output</th><th>Meaning</th><th>How to inspect it</th></tr>
<tr>
<td>prior_selected.npy / .tif / prior_preview.png</td>
<td>The selected prior in original FDK intensity units.</td>
<td>Compare with FDK and final MBIR-lite. It should be cleaner than FDK but not considered final truth.</td>
</tr>
<tr>
<td>prior_confidence.npy / .tif / prior_confidence_preview.png</td>
<td>Voxelwise confidence from 0 to 1, based on how much the prior changed the FDK and whether it reduced edge gradients.</td>
<td>Bright/high confidence means the prior is trusted more. Dark/low confidence means MBIR-lite relies more on measured data.</td>
</tr>
<tr>
<td>prior_difference.npy / .tif / prior_difference_preview.png</td>
<td>Selected prior minus original FDK.</td>
<td>Use this to see where the prior changed the volume. Large structured differences at real features are a warning sign.</td>
</tr>
<tr>
<td>support_mask.npy</td>
<td>Mask of voxels treated as sample/support for scoring.</td>
<td>Useful when diagnosing background-noise or correction-fraction behavior.</td>
</tr>
<tr>
<td>prior_metrics.csv</td>
<td>One row per TV candidate, written live as candidates finish.</td>
<td>The Metrics tab reads this file continuously while Make Prior runs.</td>
</tr>
</table>

<h3 id="fast-panel-fields">Fast Panel Fields</h3>
<table>
<tr><th>Field or option</th><th>What it controls</th><th>Default and guidance</th></tr>
<tr>
<td>Enable fast anchor workflow</td>
<td>Marks the config as using the fast reconstruction workflow.</td>
<td>Turn on when using the Fast panel or saving a fast YAML config.</td>
</tr>
<tr>
<td>Use high-exposure anchor dataset</td>
<td>Enables the separate anchor dataset input. Anchor projections have their own folders, metadata, references, and drift correction.</td>
<td>On for normal fast workflow. Leave off only for internal tests or when running a prior from an existing FDK without anchors.</td>
</tr>
<tr>
<td>Anchor projections</td>
<td>Folder containing high-exposure anchor projection TIFF images.</td>
<td>Required when anchors are enabled. Do not point this to the main projection folder unless the anchor dataset is intentionally identical.</td>
</tr>
<tr>
<td>Anchor flat/reference</td>
<td>Folder containing flats/references for the anchor projections.</td>
<td>Use the anchor acquisition references. The app preprocesses anchors separately and does not reuse main flats.</td>
</tr>
<tr>
<td>Anchor dark</td>
<td>Optional folder containing dark images for the anchor dataset.</td>
<td>Use if the anchor acquisition has dark images. Blank is allowed.</td>
</tr>
<tr>
<td>Anchor metadata CSV</td>
<td>Metadata CSV for anchor filenames, angles, and optional drift shifts.</td>
<td>Required when anchors are enabled. It can use the same filename/angle/shift column conventions as the main metadata.</td>
</tr>
<tr>
<td>Anchor drift file</td>
<td>Optional CSV containing anchor drift shifts. Accepted X columns include x_shift_px, shift_x_px, x_shift, shift_x, u_shift_px, col_shift_px. Accepted Y columns include y_shift_px, shift_y_px, y_shift, shift_y, v_shift_px, row_shift_px.</td>
<td>If both metadata shift columns and this file exist, the drift file wins and the log says so.</td>
</tr>
<tr>
<td>Inherit main preprocessing</td>
<td>Uses the main preprocessing settings for anchors: flat/dark/log, epsilon, clipping, crop, binning, flips, transpose, and truncation settings.</td>
<td>Default on. Keep on unless the anchor data needs different crop/binning/orientation.</td>
</tr>
<tr>
<td>Inherit main alignment</td>
<td>Uses the main center and drift alignment conventions for anchors, while still allowing an anchor drift file override.</td>
<td>Default on. Keep on when both acquisitions share the same scanner orientation and shift convention.</td>
</tr>
<tr>
<td>Main exposure s</td>
<td>Optional exposure time for the main projections. Used to estimate scalar projection weights.</td>
<td>Blank by default. If known, enter seconds. Weight scales approximately with sqrt(exposure).</td>
</tr>
<tr>
<td>Anchor exposure s</td>
<td>Optional exposure time for anchor projections. Used to estimate scalar anchor weights.</td>
<td>Blank by default. If known, enter seconds. Anchor weights are still capped by Max weight ratio.</td>
</tr>
<tr>
<td>Anchor weight multiplier</td>
<td>Extra multiplier applied to anchor weights when exposure/count data are not enough or not supplied.</td>
<td>Default 1.0. Try 1.0 first. Increase cautiously only if anchor views should carry more statistical confidence.</td>
</tr>
<tr>
<td>Max weight ratio</td>
<td>Caps projection weights after normalizing median main weight to 1.0.</td>
<td>Default 10.0. This prevents anchors from numerically dominating MBIR-lite.</td>
</tr>
<tr>
<td>Recon anchor count (0 auto)</td>
<td>Number of anchor views used as reconstruction data. Zero means use all anchors not assigned to tune or QC.</td>
<td>Default 0 auto. Use explicit counts when you want a fixed reconstruction/tune/QC split.</td>
</tr>
<tr>
<td>Tune anchor count</td>
<td>Number of anchor views held out for FDK filter scoring and prior scoring.</td>
<td>Default 6. Increase if you have many anchors and want more robust filter/prior selection.</td>
</tr>
<tr>
<td>QC anchor count</td>
<td>Number of anchor views held out for independent QC residuals.</td>
<td>Default 6. Keep at least a few QC anchors when possible. QC anchors are not used for selection or final reconstruction by default.</td>
</tr>
<tr>
<td>Use tune anchors in MBIR-lite</td>
<td>Includes tune anchors in the final MBIR-lite data fidelity after they have been used for filter/prior selection.</td>
<td>Default on. Turn off only if you want tuning anchors to remain fully excluded from the final fit.</td>
</tr>
<tr>
<td>Use QC anchors in MBIR-lite</td>
<td>Includes QC anchors in the final MBIR-lite data fidelity.</td>
<td>Default off. Leave off for independent QC. Turn on only for a final production run after QC has been evaluated.</td>
</tr>
<tr>
<td>FDK lowres factor</td>
<td>Downsampling factor for the FDK filter sweep. The best filter is then rerun at full resolution.</td>
<td>Default 2. If detector or volume dimensions are not divisible, the app logs a fallback to factor 1.</td>
</tr>
<tr>
<td>FDK filters</td>
<td>Multi-select TIGRE filter list tested in the sweep.</td>
<td>Default ram_lak, shepp_logan, cosine, hann, hamming. Use the checkboxes to include or exclude filters. TIGRE cutoff sweep is not available in this environment, so cutoff remains 1.0.</td>
</tr>
<tr>
<td>FDK filter choice</td>
<td>Chooses whether the full-resolution FDK and later prior/MBIR-lite stages use the automatic score winner or a specific user-selected filter.</td>
<td>Use Auto for the lower-is-better score winner. Select a named filter if visual inspection shows that the score winner oversmooths or misses important detail.</td>
</tr>
<tr>
<td>Prior TV weights</td>
<td>Comma-separated weak TV denoising strengths tested for the training-free prior.</td>
<td>Default 0.005, 0.01, 0.02, 0.04. Start here. If the prior is too smooth, remove larger values.</td>
</tr>
<tr>
<td>MBIR-lite sweeps</td>
<td>Number of fast prior-anchored ordered-subset sweeps.</td>
<td>Default 5. This is intentionally much smaller than full MBIR. Try 3 for quick tests, 5 for default, 8-10 if residuals still improve.</td>
</tr>
<tr>
<td>MBIR-lite batch size</td>
<td>Projection batch size for MBIR-lite forward/backprojection operations.</td>
<td>Default 32. Lower to 16 or 8 if VRAM is tight. Raise only if memory is comfortable.</td>
</tr>
<tr>
<td>MBIR-lite subsets</td>
<td>Number of ordered subsets used by MBIR-lite.</td>
<td>Default 8. Try 4 for more stable but slower updates; 8 is the default speed/stability balance.</td>
</tr>
<tr>
<td>MBIR-lite start volume</td>
<td>Chooses whether MBIR-lite starts fresh from the saved best FDK or warm-starts from an existing MBIR-lite final volume in the Resume run folder.</td>
<td>Use Fresh from FDK when comparing parameters reproducibly. Use Resume previous MBIR-lite final when continuing refinement in the same run. Warm-start requires an existing fast_recon/mbir_lite/mbir_lite_final.npy.</td>
</tr>
</table>

<h3 id="mbir-lite-batch-vs-subsets">MBIR-lite Batch Size vs Subsets</h3>
<p>
MBIR-lite batch size and MBIR-lite subsets control different layers of the same projection update.
Subsets decide which projection views are active in a solver sweep. Batch size decides how many of those active
views are sent through TIGRE at one time.
</p>
<pre>
Subsets = how many projection views are used for this solver sweep
Batch size = how many active views are processed on the GPU at once
</pre>
<table>
<tr><th>Setting</th><th>What it changes</th><th>Main tradeoff</th></tr>
<tr>
<td>MBIR-lite subsets</td>
<td>Splits the full projection set into ordered groups. With 8 subsets, sweep 1 uses roughly every 8th view, sweep 2 uses the next interleaved group, and so on.</td>
<td>More subsets make each sweep faster but noisier and less exact. Fewer subsets use more views per sweep and are usually more stable, but slower.</td>
</tr>
<tr>
<td>MBIR-lite batch size</td>
<td>Splits the currently active subset into smaller TIGRE forward/backprojection chunks.</td>
<td>Smaller batches use less GPU memory and are safer. Larger batches can be faster if VRAM is comfortable.</td>
</tr>
</table>
<p>
Example for 200 projections with MBIR-lite subsets 8 and batch size 16:
</p>
<pre>
200 total projections / 8 subsets = about 25 active views per sweep

Each sweep processes about 25 active views:
  batch 1: 16 views
  batch 2:  9 views

So each MBIR-lite sweep uses about 2 TIGRE projection/backprojection batches.

Across 8 sweeps, the solver cycles through all 8 subsets once,
so all 200 projections have been touched once in subset form.
</pre>
<table>
<tr><th>Symptom or goal</th><th>First adjustment</th><th>Why</th></tr>
<tr>
<td>GPU memory runs out or TIGRE fails during projection/backprojection</td>
<td>Lower batch size: 32 -> 16 -> 8 -> 4.</td>
<td>This reduces how many views are held in GPU working memory at once.</td>
</tr>
<tr>
<td>Metrics bounce strongly or the reconstruction looks unstable</td>
<td>Lower subsets: 8 -> 4.</td>
<td>Each sweep uses more projection views, giving a steadier update.</td>
</tr>
<tr>
<td>Run is stable but slow and GPU memory is available</td>
<td>Increase batch size cautiously.</td>
<td>This can reduce overhead by processing more active views per TIGRE call.</td>
</tr>
</table>

<table>
<tr><th>Field or option</th><th>What it controls</th><th>Default and guidance</th></tr>
<tr>
<td>MBIR-lite lambda TV</td>
<td>TV regularization weight used during MBIR-lite.</td>
<td>Default 1.0e-4, lower than full MBIR. Increase if final output is noisy; decrease if edges are oversmoothed.</td>
</tr>
<tr>
<td>MBIR-lite rho prior</td>
<td>Strength of the confidence-weighted pull toward the prior.</td>
<td>Default 0.05. Increase if the final result ignores a trustworthy prior; decrease if it follows the prior too strongly.</td>
</tr>
<tr>
<td>MBIR-lite start volume</td>
<td>Controls whether MBIR-lite starts from the saved best FDK or from an existing MBIR-lite final volume in the selected Resume run folder.</td>
<td>Fresh from FDK is best for reproducible reruns. Resume previous MBIR-lite final is best for continued refinement after a prior run already looks good.</td>
</tr>
<tr>
<td>Resume run folder</td>
<td>Existing timestamped run folder for stage-only commands such as Make Prior, Run MBIR-lite, or QC Report.</td>
<td>Required for stage-only fast runs unless all prerequisites are being created in the same run.</td>
</tr>
<tr>
<td>Save reusable fast intermediates</td>
<td>Saves preprocessed main/anchor arrays, angles, and weights for stage resume.</td>
<td>Default on. Turn off only if disk space is a bigger concern than rerun speed.</td>
</tr>
<tr>
<td>Make QC report after Run Fast</td>
<td>Runs the QC report automatically at the end of Run Fast.</td>
<td>Default on. Recommended because the held-out residuals are the main sanity check.</td>
</tr>
</table>

<h3 id="fast-buttons">Fast Buttons</h3>
<table>
<tr><th>Button</th><th>What it runs</th><th>When to use it</th></tr>
<tr>
<td>Run Fast</td>
<td>Runs the complete workflow: main preprocessing, anchor preprocessing, split, FDK sweep, prior, MBIR-lite, and QC report.</td>
<td>Use for the normal fast workflow after main/anchor paths and geometry are set.</td>
</tr>
<tr>
<td>FDK Sweep</td>
<td>Runs preprocessing, anchor splitting, low-resolution FDK filter sweep, and full-resolution best FDK output.</td>
<td>Use when you want to inspect FDK filter scores before making the prior or final reconstruction.</td>
</tr>
<tr>
<td>Make Prior</td>
<td>Creates the training-free TV prior, confidence map, prior difference maps, and prior metrics from an existing best FDK.</td>
<td>Use after FDK Sweep. Set Resume run folder to the run containing fast_recon/fdk_sweep/fdk_best.npy.</td>
</tr>
<tr>
<td>Run MBIR-lite</td>
<td>Runs the fast prior-anchored solver using main projections, recon anchors, tune anchors if enabled, and not QC anchors by default.</td>
<td>Use after Make Prior. Set Resume run folder if running this as a separate stage. Choose MBIR-lite start volume first: Fresh from FDK for reproducible reruns, or Resume previous MBIR-lite final to continue refining an existing run.</td>
</tr>
<tr>
<td>QC Report</td>
<td>Creates Markdown, CSV, PNG preview panel, and held-out anchor residual TIFFs from existing fast outputs.</td>
<td>Use after MBIR-lite, or rerun when you want to regenerate QC after changing report-related outputs.</td>
</tr>
</table>

<h3 id="fast-data-flow">What the Fast Workflow Saves</h3>
<pre>
output/run_YYYYMMDD_HHMMSS/fast_recon/
  anchors/
    anchor_split.csv
    anchor_split_report.txt
  fdk_sweep/
    fdk_best.npy
    fdk_best.tif
    fdk_best_preview.png
    scores.csv
    best_filter.yaml
    candidate_previews/
  prior/
    prior_selected.npy
    prior_selected.tif
    prior_confidence.npy
    prior_confidence.tif
    prior_difference.npy
    prior_metrics.csv
    prior_preview.png
  mbir_lite/
    mbir_lite_final.npy
    mbir_lite_final.tif
    mbir_lite_preview.png
    final_minus_fdk.npy
    final_minus_prior.npy
    metrics.csv
  qc/
    report.md
    qc_metrics.csv
    preview_panel.png
    residuals/
</pre>

<h3 id="fast-suggestions">Suggested Fast Starting Values</h3>
<table>
<tr><th>Situation</th><th>Suggested values</th><th>What to inspect</th></tr>
<tr>
<td>First test on a new sample</td>
<td>FDK lowres factor 2, tune anchors 6, QC anchors 6, MBIR-lite sweeps 3-5, batch size 16 or 32, subsets 8.</td>
<td>scores.csv, best_filter.yaml, prior_difference_preview.png, qc/report.md.</td>
</tr>
<tr>
<td>Prior looks too smooth</td>
<td>Remove larger Prior TV weights, lower MBIR-lite rho prior, or lower MBIR-lite lambda TV.</td>
<td>prior_difference_preview.png, final_minus_prior.npy, edge visibility in Tomogram View.</td>
</tr>
<tr>
<td>Final remains noisy</td>
<td>Increase MBIR-lite sweeps to 8-10, increase lambda TV slightly, or increase rho prior slightly.</td>
<td>MBIR-lite metrics.csv and QC residuals. Avoid improving appearance while worsening QC residuals sharply.</td>
</tr>
<tr>
<td>GPU memory is tight</td>
<td>Lower MBIR-lite batch size to 16, 8, or 4. Keep FDK lowres factor 2 if divisible.</td>
<td>Progress / Log and Device Monitor VRAM occupancy.</td>
</tr>
<tr>
<td>Many anchors are available</td>
<td>Increase tune and QC counts, keep QC held out, use auto recon count for the remainder.</td>
<td>Stable FDK scores and meaningful held-out QC residuals.</td>
</tr>
</table>

<h3 id="run-controls">Run Controls</h3>
<table>
<tr><th>Button</th><th>What it does</th><th>Guidance</th></tr>
<tr>
<td>Preprocess Only</td>
<td>Runs metadata validation and preprocessing, writes run outputs, and stops before FDK/MBIR.</td>
<td>Use to validate data loading, flat/dark/log conversion, crop, flips, and drift correction.</td>
</tr>
<tr>
<td>Run FDK</td>
<td>Runs the full preprocessing and FDK reconstruction pipeline.</td>
<td>Run before MBIR. FDK is the best quick diagnostic for geometry and alignment.</td>
</tr>
<tr>
<td>Run MBIR</td>
<td>Runs preprocessing, initialization, and TV-MBIR with the selected solver and memory mode.</td>
<td>Use after FDK looks good. Read any warning dialog carefully before continuing.</td>
</tr>
<tr>
<td>Cancel</td>
<td>Requests cancellation. The current operation stops at the next safe cancellation checkpoint.</td>
<td>Cancellation is cooperative; a TIGRE call may need to finish its current batch first.</td>
</tr>
<tr>
<td>Estimate Memory</td>
<td>Calculates a fresh MBIR-lite RAM and GPU-memory estimate from the current detector, volume, projection count, batch size, subset count, anchor split, and GPU selection.</td>
<td>Use before Run Fast or Run MBIR-lite. Adjust MBIR-lite batch size, subsets, volume size, or GPU selection, then press it again to recalculate.</td>
</tr>
<tr>
<td>Preload Existing Results</td>
<td>Scans Resume run folder and recent run folders under Output folder, then loads any saved Metrics, Tomogram View sources, and MBIR Preview slice without starting a reconstruction.</td>
<td>Use after loading a config or browsing to an existing output folder. It is display-only and does not change saved data or interrupt live updates from a future run.</td>
</tr>
<tr>
<td>Save Config</td>
<td>Saves all current GUI settings to a YAML config.</td>
<td>Use before production runs and when sharing reproducible settings.</td>
</tr>
<tr>
<td>Load Config</td>
<td>Loads a YAML config into the GUI.</td>
<td>Use to restore previous runs or load config-47.yaml/config.mbir.yaml templates.</td>
</tr>
</table>

<h2 id="tabs">Right-Side Tabs</h2>
<table>
<tr><th>Tab</th><th>What it shows</th><th>How to use it</th></tr>
<tr>
<td>Help</td>
<td>This searchable manual.</td>
<td>Use the Keyword box and Search button. Repeated Search presses jump to the next match and wrap back to the top.</td>
</tr>
<tr>
<td>Progress / Log</td>
<td>Live MBIR progress labels, a progress bar, and detailed text log.</td>
<td>Watch current iteration, CG progress, elapsed/ETA, objective, data residual, relative change, and detailed pipeline messages.</td>
</tr>
<tr>
<td>Device Monitor</td>
<td>NVIDIA GPU table plus live compute utilization and VRAM occupancy plots.</td>
<td>Use during FDK/MBIR to see whether the selected GPU is active and whether VRAM is near full.</td>
</tr>
<tr>
<td>Validation</td>
<td>Preview table of validated metadata rows: index, filename, angle deg, angle rad, x shift, y shift, file exists.</td>
<td>Check that filenames exist and angles/shifts look sane. It shows edge rows when the dataset is large.</td>
</tr>
<tr>
<td>Metrics</td>
<td>MBIR convergence graphs or center-search metric graph.</td>
<td>Use after automatic center search or during/after MBIR to evaluate convergence and performance.</td>
</tr>
<tr>
<td>Preview</td>
<td>Raw, flat, transmission, attenuation, or center-search preview images.</td>
<td>Use with Image display controls to inspect preprocessing and center alignment.</td>
</tr>
<tr>
<td>Tomogram View</td>
<td>Final or intermediate reconstructed 3-D volume browser with Axial, Coronal, Sagittal views and slice controls.</td>
<td>Use after FDK or final MBIR to inspect slices through the volume.</td>
</tr>
<tr>
<td>MBIR Preview</td>
<td>Requested full-resolution MBIR slice during reconstruction and final MBIR result.</td>
<td>Select view and slice before or during the run. The displayed slice updates on the next MBIR progress event.</td>
</tr>
<tr>
<td>Fast outputs in existing tabs</td>
<td>Fast FDK, selected prior, and final MBIR-lite volumes are shown in the same Preview, Tomogram View, Metrics, and MBIR Preview tabs.</td>
<td>After Run Fast, the final MBIR-lite volume appears like a normal reconstruction. The log also prints the QC report path.</td>
</tr>
</table>

<h3 id="progress-log">Progress / Log Fields</h3>
<table>
<tr><th>Field</th><th>Meaning</th></tr>
<tr><td>Iteration</td><td>Current solver iteration and maximum iteration count.</td></tr>
<tr><td>Solver</td><td>ADMM CG status, or PDHG/Subset-TV update description. For ADMM it reports inner CG residual.</td></tr>
<tr><td>Time</td><td>Elapsed runtime and estimated time remaining when enough iteration data exists.</td></tr>
<tr><td>Objective</td><td>Current MBIR objective: data fidelity plus TV penalty.</td></tr>
<tr><td>Data residual</td><td>Square root of 2 times data fidelity, equivalent to ||Ax-b||.</td></tr>
<tr><td>Relative change</td><td>Relative change in the volume from the previous iteration.</td></tr>
</table>

<h2 id="solvers">Solver Selection Guide</h2>
<table>
<tr><th>Solver</th><th>What it does</th><th>When to choose it</th><th>Parameters to tune</th></tr>
<tr>
<td>auto</td>
<td>Starts from ADMM/CG when memory is safe. If ADMM arrays exceed RAM budget or available RAM is too low, it selects pdhg_low_memory. If PDHG is still too large, it selects streaming_subset_tv.</td>
<td>Recommended default. It protects large 1024-cubed style runs from obvious memory failures.</td>
<td>lambda TV, ADMM iterations, relative x-change tolerance, memory mode, projection batch size. The app reports the effective solver in warnings and logs.</td>
</tr>
<tr>
<td>admm</td>
<td>Solves TV-MBIR with ADMM outer iterations and CG inner solves. It uses full-volume x/d/u variables and can be memory heavy.</td>
<td>Best for smaller or moderate data where RAM is sufficient and you want the most exact local solver behavior.</td>
<td>lambda TV, rho, ADMM iterations, inner CG iterations, primal tolerance, dual tolerance, CG tolerance, memory mode.</td>
</tr>
<tr>
<td>pdhg_low_memory</td>
<td>Uses primal-dual hybrid gradient updates. It avoids the ADMM/CG volume pile-up and keeps fewer large arrays in RAM.</td>
<td>Large volumes where ADMM is too memory heavy, but you still want full-data objective evaluations.</td>
<td>lambda TV, max iterations, relative x-change tolerance, PDHG TV dual dtype, YAML-only pdhg_step_safety and pdhg_power_iterations.</td>
</tr>
<tr>
<td>streaming_subset_tv</td>
<td>Streams one projection batch, applies an immediate TV-regularized image update, and discards the batch gradient.</td>
<td>Very large data or memory-constrained machines. Use when ADMM and PDHG are too large or unstable in memory.</td>
<td>lambda TV, max iterations, relative x-change tolerance, projection batch size, Subset-TV step safety, YAML-only subset_tv_power_iterations.</td>
</tr>
<tr>
<td>anchored_streaming_subset_tv / MBIR-lite</td>
<td>Fast workflow solver that adds a confidence-weighted prior term to the streaming subset TV update.</td>
<td>Used by Run Fast and Run MBIR-lite. It is designed for quick refinement after the FDK sweep and training-free prior.</td>
<td>MBIR-lite sweeps, batch size, subsets, lambda TV, rho prior, projection weights, and QC anchor residuals.</td>
</tr>
</table>

<h3 id="choosing-solver">How to Choose a Solver for Your Dataset</h3>
<ol>
<li>Run FDK first. If FDK is wrong, MBIR will refine the wrong geometry.</li>
<li>For first MBIR, keep Solver on auto. Read the warning dialog to see the effective solver.</li>
<li>If the warning says ADMM fits comfortably, ADMM is a good production choice.</li>
<li>If ADMM is too large, use auto or pdhg_low_memory. Keep PDHG dual dtype at float16 for large volumes.</li>
<li>If PDHG is still too large or available RAM is low, use streaming_subset_tv and reduce projection batch size.</li>
<li>For quick parameter tuning, use MBIR extra binning or Use every Nth projection before committing to full resolution.</li>
</ol>

<h2 id="memory">Memory Mode Guide</h2>
<table>
<tr><th>Memory mode</th><th>What it does</th><th>Recommended use</th></tr>
<tr>
<td>auto</td>
<td>Uses projection_streaming internally. The app also caps/reduces projection batch size when estimates indicate risk.</td>
<td>Recommended default. It is usually safer than a monolithic full-stack TIGRE call.</td>
</tr>
<tr>
<td>full_gpu</td>
<td>Calls TIGRE with the full projection stack for forward/backprojection.</td>
<td>Use for small or moderate reconstructions that clearly fit in GPU and system memory. Fastest simple path when it fits.</td>
</tr>
<tr>
<td>distributed_full_data</td>
<td>Splits exact full-data TIGRE operations across multiple selected GPUs using shared CPU buffers.</td>
<td>Use on multi-GPU workstations when you want exact full-data ADMM/PDHG operations and have enough host RAM. Falls back when fewer than two GPUs or not enough RAM are available.</td>
</tr>
<tr>
<td>projection_streaming</td>
<td>Processes projection views in batches and accumulates backprojections or residuals.</td>
<td>Best general large-data mode. Tune Projection batch size to the largest value that fits VRAM.</td>
</tr>
<tr>
<td>ordered_subsets</td>
<td>Uses one interleaved subset of projections per ADMM iteration and scales the subset contribution toward the full data term.</td>
<td>Experimental acceleration/memory option for ADMM. It can be noisier per iteration. PDHG and Subset-TV use projection_streaming internally instead.</td>
</tr>
</table>
<p>
Projection batch size is the most practical memory knob. If a CUDA allocation fails, reduce it. If GPU utilization is low and VRAM has headroom,
increase it. Batch size 1 is safest but can be slow.
</p>

<h2 id="warnings">Run Warnings and How to Respond</h2>
<p>When Run MBIR is pressed, the app estimates projection memory, volume memory, CPU working arrays, GPU working set, selected GPUs, and effective solver/memory mode.</p>
<table>
<tr><th>Warning or dialog</th><th>Meaning</th><th>Recommended response</th></tr>
<tr>
<td>Memory estimate unavailable: Validate metadata/geometry before running MBIR.</td>
<td>The app does not know detector shape or projection count yet.</td>
<td>Run Validate Metadata first.</td>
</tr>
<tr>
<td>Memory estimate unavailable: Set positive Nz, Ny, and Nx before running MBIR.</td>
<td>The volume grid is incomplete.</td>
<td>Set all reconstruction voxel counts to positive values.</td>
</tr>
<tr>
<td>Large MBIR memory request</td>
<td>The requested run is risky but not automatically blocked.</td>
<td>Read the estimates. Cancel if unsure. First fixes: Solver auto, projection_streaming, smaller batch, fewer voxels, crop, debug binning, projection stride, or existing_fdk.</td>
</tr>
<tr>
<td>MBIR memory request cannot run safely</td>
<td>A critical memory estimate says the run is likely to exceed safe RAM/GPU limits.</td>
<td>The run is blocked. Reduce volume/projection count, use streaming_subset_tv, close memory-heavy applications, or use a larger machine.</td>
</tr>
<tr>
<td>Auto solver will use low-memory PDHG</td>
<td>ADMM/CG is estimated too large for RAM, so auto selected PDHG.</td>
<td>Usually OK. Expect ADMM primal/dual residual graphs to be replaced by PDHG data/TV graphs.</td>
</tr>
<tr>
<td>Auto solver will use streaming subset-TV</td>
<td>ADMM and PDHG are estimated too large, so auto selected the smallest-footprint solver.</td>
<td>Usually OK for huge data. Expect more approximate/noisy iteration behavior. Use more iterations and inspect previews.</td>
</tr>
<tr>
<td>ADMM/CG CPU working arrays exceed safe physical RAM budget</td>
<td>ADMM needs too many full-volume arrays for installed RAM.</td>
<td>Use Solver auto or pdhg_low_memory, or reduce volume voxels.</td>
</tr>
<tr>
<td>Currently available system RAM is too low for TIGRE Atb transient allocation</td>
<td>Even if VRAM looks available, TIGRE also needs host RAM and page-locked buffers.</td>
<td>Close other applications, reduce volume, reduce projection count, or use a lower-memory solver.</td>
</tr>
<tr>
<td>Estimated TIGRE GPU working set is close to or above available free GPU memory</td>
<td>The selected batch/full call may not fit in VRAM.</td>
<td>Close GPU applications, reduce projection batch size, reduce volume, or choose a GPU with more free VRAM.</td>
</tr>
<tr>
<td>Even one projection per TIGRE batch is estimated to exceed this GPU's total memory</td>
<td>The requested volume is too large for that GPU even with maximum streaming.</td>
<td>Reduce volume dimensions or crop the reconstruction region.</td>
</tr>
<tr>
<td>Projection streaming batch size will be capped or reduced automatically</td>
<td>The app will lower batch size to avoid accidentally making a full-stack call or exceeding estimates.</td>
<td>Accept it unless you have measured that a larger batch is safe.</td>
</tr>
<tr>
<td>Projection batch size 1 is memory-safe but can underutilize the GPU</td>
<td>You are using the safest but slowest batch size.</td>
<td>Increase batch size if VRAM has room and you want better throughput.</td>
</tr>
<tr>
<td>FDK initialization requires a monolithic TIGRE allocation</td>
<td>FDK init may not fit even though MBIR projection streaming would.</td>
<td>Use existing_fdk from a saved .npy file, or zeros initialization for a memory-robust run.</td>
</tr>
<tr>
<td>Selected GPU memory could not be queried with nvidia-smi</td>
<td>The app cannot estimate current VRAM availability.</td>
<td>Install/repair NVIDIA driver tools or proceed conservatively with smaller batches.</td>
</tr>
<tr>
<td>Distributed full-data mode requested, but fewer than two GPUs are selected</td>
<td>distributed_full_data needs multiple GPUs.</td>
<td>Use GPU selection all or 0,1 on a multi-GPU machine; otherwise use full_gpu or projection_streaming.</td>
</tr>
<tr>
<td>Large center-offset search</td>
<td>The center search will reconstruct many preview slices.</td>
<td>Use a coarser step first, then fine search around the best shift.</td>
</tr>
<tr>
<td>Folder contains TIFFs when exporting drift-corrected transmission</td>
<td>The selected output folder already has TIFF files.</td>
<td>Continue only if overwriting files with the export prefix is acceptable.</td>
</tr>
<tr>
<td>Busy</td>
<td>A processing or reconstruction task is already running.</td>
<td>Wait or press Cancel and let the current safe checkpoint finish.</td>
</tr>
</table>

<h2 id="convergence">Metrics and Convergence</h2>
<h3 id="metric-graphs">MBIR Metrics Graphs</h3>
<p>The Metrics tab shows six graphs during MBIR. Log-scaled plots are expected to trend downward or flatten.</p>
<table>
<tr><th>Graph</th><th>Meaning</th><th>How to evaluate it</th></tr>
<tr>
<td>Objective</td>
<td>Data fidelity plus lambda TV penalty.</td>
<td>Should generally decrease or stabilize. Ordered subsets and streaming updates can be noisier; look at the trend.</td>
</tr>
<tr>
<td>Full-data residual ||Ax-b||</td>
<td>Mismatch between forward projection of current volume and measured attenuation data.</td>
<td>Should decrease or flatten. If it increases strongly, check geometry, center, angle sign, lambda TV, and step safety.</td>
</tr>
<tr>
<td>Relative volume change</td>
<td>How much x changes from one iteration to the next.</td>
<td>Convergence is indicated when this becomes small and stays small, especially below Relative x-change tol.</td>
</tr>
<tr>
<td>ADMM primal residual</td>
<td>ADMM consistency between image gradient and TV shrinkage variable.</td>
<td>ADMM only. It should move toward Primal tolerance.</td>
</tr>
<tr>
<td>ADMM dual residual</td>
<td>ADMM dual-variable change scaled by rho.</td>
<td>ADMM only. It should move toward Dual tolerance. If primal and dual are very unbalanced, rho may need tuning.</td>
</tr>
<tr>
<td>CG residual</td>
<td>Relative residual of the inner conjugate-gradient solve.</td>
<td>ADMM only. If high at every iteration, increase Inner CG iterations or use a stricter CG tolerance.</td>
</tr>
<tr>
<td>Data fidelity term</td>
<td>0.5 ||Ax-b||^2 for PDHG/Subset-TV display.</td>
<td>Should decrease or stabilize. If it fights the TV term, tune lambda TV.</td>
</tr>
<tr>
<td>TV penalty term</td>
<td>lambda times total variation of the volume.</td>
<td>Can increase or decrease depending on the update. Too high lambda may oversmooth; too low lambda may leave streaks/noise.</td>
</tr>
<tr>
<td>Iteration time</td>
<td>Seconds spent per iteration.</td>
<td>Use to compare memory modes, batch sizes, and GPU selection.</td>
</tr>
</table>
<pre>
Good convergence pattern:
Objective          \________
Data residual      \_______
Relative change       \____
Preview slices: visible changes early, then subtle refinement

Warning pattern:
Objective          /\/\/\/\
Residual           rising
Preview slices: strong oscillation or growing artifacts
</pre>

<h3 id="fast-metrics">Fast Workflow Metrics Tab</h3>
<p>
For the fast workflow, the Metrics tab has a category menu. It updates from files while the run progresses, so a table or
plot can appear as soon as its CSV or preview image is written. Image displays in this tab use the same left-panel histogram,
zoom, and pan controls as the other image tabs.
</p>
<table>
<tr><th>Metrics category</th><th>What is displayed</th><th>How to read it</th></tr>
<tr>
<td>FDK sweep</td>
<td>A filter selector, candidate preview images for that filter, and the scores.csv table.</td>
<td>The highlighted row is the filter used for full-resolution FDK. score_total is a cost, so lower is better, but visual sharpness and feature preservation should still be inspected.</td>
</tr>
<tr>
<td>Make prior</td>
<td>A display selector for Selected prior, Confidence, Difference, Support mask, Selected prior normalized, and FDK normalized; a candidate-metrics graph; and prior_metrics.csv table.</td>
<td>The highlighted table row is the selected candidate. The graph helps show which TV strength reduced background noise while preserving edges and anchor consistency.</td>
</tr>
<tr>
<td>MBIR</td>
<td>Live iteration graphs and a metrics table from the current run or metrics/metrics.csv for standard full MBIR.</td>
<td>Use this for standard MBIR objective, residual, TV, relative-change, and solver-specific plots such as ADMM residuals and CG behavior when available.</td>
</tr>
<tr>
<td>MBIR-lite</td>
<td>Live iteration graphs and a metrics table from the current run or fast_recon/mbir_lite/metrics.csv.</td>
<td>Look for objective/data terms and relative_change to decrease or flatten. QC residual should not worsen sharply if QC anchors are available.</td>
</tr>
<tr>
<td>QC report</td>
<td>QC preview panel, residual bar plot, and qc_metrics.csv table.</td>
<td>Held-out QC anchors are the independent check. Lower residual is better. A visually nicer image is suspect if held-out residuals become much worse.</td>
</tr>
</table>

<h3 id="qc-report-reading">How to Read QC Report</h3>
<p>
The QC report compares three volumes on anchor projections: the selected FDK, the selected Prior, and the final MBIR-lite
result. The table and bar graph report normalized projection residuals, so lower values are better.
</p>
<table>
<tr><th>What to compare</th><th>What it means</th><th>How to interpret it</th></tr>
<tr>
<td>Final vs FDK on tune anchors</td>
<td>Shows whether MBIR-lite improved projection consistency on the anchor views used for tuning.</td>
<td>If Final is clearly lower than FDK, the fast workflow is improving data fit rather than just changing appearance.</td>
</tr>
<tr>
<td>Final vs FDK on QC anchors</td>
<td>Shows whether the improvement also holds on independent held-out anchor views.</td>
<td>This is the most important comparison. If Final improves on tune anchors but becomes much worse on QC anchors, treat the run as suspicious or over-tuned.</td>
</tr>
<tr>
<td>Prior vs FDK</td>
<td>Shows how much the conservative prior alone helped before MBIR-lite refinement.</td>
<td>A small improvement is normal. The prior is meant to be cautious, not the final answer.</td>
</tr>
<tr>
<td>QC bar annotation</td>
<td>Percent change of the QC residual relative to the matching tune residual for the same volume.</td>
<td>Values near 0 percent mean held-out QC behaves similarly to tune. Positive values mean QC residual is lower than tune. Negative values mean QC is worse than tune.</td>
</tr>
<tr>
<td>Preview panel difference images</td>
<td>Prior - FDK, Final - Prior, and Final - FDK show where each stage changed the reconstruction.</td>
<td>Sparse or structure-following differences are expected. Large broad changes with worse QC residuals can indicate oversmoothing, bias, or geometry problems.</td>
</tr>
</table>
<p>
The preview panel tiles are auto-scaled individually for readability, so compare patterns and structure more than raw
brightness between tiles.
</p>

<h3 id="make-prior-metrics">Make Prior Metrics Graph and Table</h3>
<p>
The Make Prior graph plots candidate metrics versus TV strength. Each point is one candidate from Prior TV weights.
The table shows the same values and is updated live after each candidate finishes. Candidate scoring can take time because
the app forward-projects the candidate onto tune anchors and compares it with measured anchor projections.
</p>
<table>
<tr><th>Graph or table field</th><th>How it is evaluated</th><th>What to prefer</th></tr>
<tr>
<td>background_noise</td>
<td>MAD-based noise estimate measured in background voxels outside the support mask. If no outside support exists, a high-pass residual is used.</td>
<td>Lower is cleaner, but only after the candidate passes edge, correction, and anchor-residual checks. Very low noise with poor edges can mean oversmoothing.</td>
</tr>
<tr>
<td>edge_retention</td>
<td>Median gradient magnitude in the candidate divided by the FDK gradient magnitude on the strongest FDK edges.</td>
<td>Closer to 1 means edges are preserved. Values below min_edge_retention fail the conservative check.</td>
</tr>
<tr>
<td>correction_fraction</td>
<td>Total absolute change from FDK inside the support mask, divided by total absolute FDK signal inside the support mask.</td>
<td>Lower means the prior changed the FDK conservatively. Values above max_correction_fraction fail.</td>
</tr>
<tr>
<td>anchor_residual_ratio</td>
<td>Tune-anchor projection residual for the candidate divided by the baseline FDK tune-anchor residual.</td>
<td>Values near or below 1 mean the prior remains at least as projection-consistent as FDK on tune anchors. Values above max_anchor_residual_ratio fail.</td>
</tr>
<tr>
<td>accepted</td>
<td>True only if anchor_residual_ratio, edge_retention, and correction_fraction all pass their thresholds.</td>
<td>The selected candidate normally comes from accepted rows. If no row is accepted, the weakest TV strength is selected as a fallback.</td>
</tr>
<tr>
<td>strength</td>
<td>The TV weight that generated the candidate.</td>
<td>Use the weakest value that removes enough noise without losing fine structure. A single value such as 0.005 is valid when prior experience supports it.</td>
</tr>
<tr>
<td>chosen_by_fallback</td>
<td>Appears when no candidate passed all conservative checks and the app chose the weakest TV candidate.</td>
<td>Treat this as a warning to inspect images and thresholds before trusting MBIR-lite strongly.</td>
</tr>
</table>

<h3 id="mbir-lite-metrics">MBIR-lite Metrics Graphs</h3>
<table>
<tr><th>Graph</th><th>Meaning</th><th>How to evaluate it</th></tr>
<tr>
<td>objective_total</td>
<td>Total fast objective, including weighted data fit, TV term, and prior-anchor term.</td>
<td>Should generally decrease or stabilize. Ordered subsets can make it bumpy.</td>
</tr>
<tr>
<td>data_weighted</td>
<td>Weighted projection-data mismatch for the current MBIR-lite volume.</td>
<td>Should trend down or flatten. If it rises strongly, check geometry, weights, batch/subset settings, and prior strength.</td>
</tr>
<tr>
<td>tv_term</td>
<td>TV regularization contribution during MBIR-lite.</td>
<td>Higher values indicate more texture/edge variation. If the final is noisy, lambda TV may be too low; if it is smooth, lambda TV may be too high.</td>
</tr>
<tr>
<td>prior_anchor_term</td>
<td>Confidence-weighted penalty pulling the volume toward the selected prior.</td>
<td>Should not dominate the reconstruction. If the final looks too much like an oversmoothed prior, reduce rho prior.</td>
</tr>
<tr>
<td>qc_anchor_residual</td>
<td>Residual on held-out QC anchors when available.</td>
<td>This is the independent check. It should decrease or at least not worsen sharply compared with FDK/prior QC residuals.</td>
</tr>
<tr>
<td>relative_change</td>
<td>How much the volume changes from one sweep to the next.</td>
<td>Smaller values mean the run is settling. If it remains large, use more sweeps or safer step settings.</td>
</tr>
<tr>
<td>elapsed_s</td>
<td>Elapsed runtime in seconds.</td>
<td>Use it to compare batch size, subsets, and GPU settings.</td>
</tr>
</table>

<h3 id="center-metrics">Center-Search Metrics Graph</h3>
<p>
After Run Automatic Search, the Metrics tab plots selected center-shift metric versus shift. A vertical line marks the currently selected preview,
and a highlighted point marks the automatic recommendation. The best shift should also look sharp and symmetric in the Preview tab.
</p>

<h2 id="tutorial">Full Tutorial</h2>
<ol>
<li>Select Projection folder, Flat/reference folder, optional Dark folder, Metadata CSV or folder, and Output folder.</li>
<li>Click Load Metadata CSV. Confirm the filename, angle, optional radian, and shift columns.</li>
<li>Click Validate Metadata. Check the Validation tab for missing files, angle range, and shift values. Detector rows/columns should fill automatically.</li>
<li>Click Preview Raw Projection. Use Projection preview index to inspect early, middle, and late views.</li>
<li>Keep Use flat correction on. Keep Negative log on for raw intensity data. Keep Clip transmission and Set negative attenuation to zero on for a first pass.</li>
<li>Set binning/crop if needed. For first production geometry, keep binning 1 and crop only obvious bad borders. For fast tests, use binning 2 or 4.</li>
<li>Click Compute Preprocessed Preview Stack. Inspect Raw, Averaged flat field, Flat-field corrected projection, and Attenuation projection.</li>
<li>Adjust Image display levels with Auto 1-99%, Full Range, histogram handles, and zoom. These are display-only.</li>
<li>Enter geometry: DSD, DSO, effective pixel size, voxel sizes, and Nz/Ny/Nx. Start with native isotropic voxel size if memory allows.</li>
<li>Run Automatic Search for center offset with a coarse range. Inspect previews and the metric curve. Apply the automatic or selected best shift.</li>
<li>If needed, run Fine Search Around Selected Shift and apply the refined center.</li>
<li>If metadata has drift shifts, enable Use drift correction, choose drift stage, and confirm X/Y shift signs with preprocessing preview or exported drift-corrected transmission TIFFs.</li>
<li>Run FDK. Inspect Tomogram View in axial, coronal, and sagittal orientations. If the volume is mirrored, transposed, blurred, or doubled, fix geometry/alignment before MBIR.</li>
<li>For MBIR, start with Solver auto, Memory mode auto, lambda TV 0.001, Positivity on, and a modest iteration count such as 10 for a trial.</li>
<li>For a large full-resolution run, use existing_fdk if you already have a saved FDK .npy. This avoids rerunning monolithic FDK inside MBIR.</li>
<li>Click Run MBIR. Read the memory warning dialog. If it recommends PDHG or streaming subset-TV, that is usually protective rather than an error.</li>
<li>During MBIR, watch Progress / Log, Metrics, Device Monitor, and MBIR Preview. Use MBIR Preview view/slice controls to request the slice you care about.</li>
<li>Evaluate convergence with Objective, Data residual, Relative volume change, ADMM residuals if applicable, and visual preview stability.</li>
<li>Inspect the final Tomogram View and MBIR Preview. The output run folder contains saved config, diagnostics, logs, FDK/MBIR volumes, metrics, and plots according to logging settings.</li>
</ol>

<h2 id="fast-tutorial">Fast Anchor-Guided Tutorial</h2>
<ol>
<li>Load the main sparse or low-exposure dataset in Dataset input, Metadata, Preprocessing, Geometry, and Alignment exactly as you would for a normal FDK/MBIR run.</li>
<li>Run Validate Metadata for the main dataset. Confirm detector rows/columns, filename matching, angle range, and optional drift columns.</li>
<li>Use Preview Raw Projection and Compute Preprocessed Preview Stack to confirm the main projections, flats, darks, negative log, crop, flips, and drift settings.</li>
<li>Enter geometry and center alignment. For a new sample, run a normal FDK first if practical. The fast workflow assumes the geometry conventions are already plausible.</li>
<li>Open Fast anchor-guided reconstruction. Turn on Enable fast anchor workflow and Use high-exposure anchor dataset.</li>
<li>Select Anchor projections, Anchor flat/reference, optional Anchor dark, and Anchor metadata CSV. Use Anchor drift file only if you have a separate drift CSV for anchors.</li>
<li>Keep Inherit main preprocessing and Inherit main alignment on for the first attempt. This keeps binning, crop, orientation, center, and drift conventions consistent.</li>
<li>If exposure times are known, enter Main exposure s and Anchor exposure s. Otherwise keep them blank and leave Anchor weight multiplier at 1.0.</li>
<li>Use default split values first: Recon anchor count 0 auto, Tune anchor count 6, QC anchor count 6, Use tune anchors in MBIR-lite on, Use QC anchors in MBIR-lite off.</li>
<li>Use default FDK sweep values first: lowres factor 2 and filters ram_lak, shepp_logan, cosine, hann, hamming.</li>
<li>Use default prior values first: Prior TV weights 0.005, 0.01, 0.02, 0.04. These are intentionally weak because the prior should not erase real structure.</li>
<li>Use default MBIR-lite values first: 5 sweeps, batch size 32, subsets 8, lambda TV 1.0e-4, rho prior 0.05.</li>
<li>Click Run Fast. Watch Progress / Log. Long TIGRE FDK or forward projection calls may not update the GUI until the current call returns.</li>
<li>After the run, inspect fast_recon/fdk_sweep/scores.csv and best_filter.yaml. The selected filter should make physical sense and should not be chosen only because it over-smoothed the volume.</li>
<li>Inspect fast_recon/prior/prior_preview.png, prior_difference_preview.png, and prior_confidence_preview.png. The prior should remove noise conservatively while keeping edges.</li>
<li>Inspect fast_recon/mbir_lite/mbir_lite_preview.png and the final volume in Tomogram View. Compare final_minus_fdk.npy and final_minus_prior.npy when diagnosing changes.</li>
<li>Open fast_recon/qc/report.md. Compare FDK, prior, and final residuals on tune anchors and QC anchors. The QC anchors are the independent held-out check.</li>
<li>If the final looks too prior-like, reduce rho prior or remove stronger Prior TV weights. If it remains noisy, increase sweeps or lambda TV slightly. Change one thing at a time.</li>
<li>For stage-by-stage work, run FDK Sweep first, then set Resume run folder to that run and click Make Prior, Run MBIR-lite, or QC Report.</li>
</ol>

<h2 id="yaml">Advanced YAML-Only Options</h2>
<p>
Some config fields are not exposed in the left-side GUI but can be saved, loaded, and edited in YAML. Most users do not need these for routine operation.
</p>
<table>
<tr><th>Config key</th><th>Meaning</th><th>Guidance</th></tr>
<tr><td>preprocessing.max_transmission_for_log</td><td>Upper clip used before -log.</td><td>Usually match Transmission max.</td></tr>
<tr><td>preprocessing.truncation_correction</td><td>Enables symmetric detector extension for truncated projections.</td><td>Experimental. Use only when object extends beyond detector horizontally.</td></tr>
<tr><td>preprocessing.truncation_extension_fraction</td><td>Fraction of detector columns added to each side for truncation correction.</td><td>Default 0.1. Larger values add more padding.</td></tr>
<tr><td>geometry.detector_pixel_size_mm</td><td>Explicit detector pixel size vertical/horizontal in mm.</td><td>If both are set, this overrides effective pixel derived from DSD/DSO.</td></tr>
<tr><td>geometry.origin_offset_mm</td><td>Volume origin offset z/y/x in mm.</td><td>Use only with known scanner geometry offsets.</td></tr>
<tr><td>geometry.angle_units, angle_start, angle_end, endpoint_included, angle_direction, angle_file</td><td>Generate angles from config instead of metadata in lower-level workflows.</td><td>The GUI pipeline normally uses validated metadata angles.</td></tr>
<tr><td>geometry.fdk_filter</td><td>TIGRE FDK filter name.</td><td>Default ram_lak. Change only for deliberate FDK experiments.</td></tr>
<tr><td>alignment.center_offset_method</td><td>detector_offset or image_shift.</td><td>GUI center correction uses detector_offset.</td></tr>
<tr><td>alignment.shift_interpretation_sign</td><td>Global sign multiplier for drift shifts.</td><td>Advanced sign-convention correction.</td></tr>
<tr><td>alignment.shift_interpolation_order</td><td>Spline interpolation order for drift shifting.</td><td>Default 1 is linear and robust.</td></tr>
<tr><td>alignment.shift_boundary_mode</td><td>Boundary fill mode for drift shifting.</td><td>Default nearest avoids introducing zeros at edges.</td></tr>
<tr><td>mbir.tv_type</td><td>TV model label.</td><td>Currently isotropic_3d is the implemented mode.</td></tr>
<tr><td>mbir.save_every</td><td>Iteration interval for standard MBIR checkpoint saves and live preview refresh.</td><td>Default 5. Every save updates mbir/mbir_final_volume.npy and the current MBIR metrics/preview files so the run can be monitored and restarted from the latest checkpoint if needed.</td></tr>
<tr><td>mbir.pdhg_power_iterations</td><td>Power iterations used to estimate PDHG operator norm.</td><td>More iterations improve step estimate but add startup time.</td></tr>
<tr><td>mbir.pdhg_step_safety</td><td>PDHG step-size safety factor.</td><td>Default 0.95. Lower if PDHG oscillates.</td></tr>
<tr><td>mbir.pdhg_theta</td><td>PDHG extrapolation parameter.</td><td>Default 1.0. Change only for solver experiments.</td></tr>
<tr><td>mbir.pdhg_chunk_slices</td><td>Number of z slices processed per TV chunk.</td><td>Lower reduces temporary RAM; higher may be faster.</td></tr>
<tr><td>mbir.subset_tv_power_iterations</td><td>Power iterations for Subset-TV data step estimate.</td><td>More iterations improve estimate but add startup time.</td></tr>
<tr><td>mbir.use_weights, weight_mode, weight_file, photon_count_based_weights, flat_intensity_based_weights</td><td>Future or experimental weighting controls.</td><td>Leave off unless the implementation you are using explicitly supports your weighting path.</td></tr>
<tr><td>anchors.enabled</td><td>Enables the separate anchor dataset for the fast workflow.</td><td>Default false in the sample config; true in fast_recon_config.yaml.</td></tr>
<tr><td>anchors.input.projection_folder, flat_folder, dark_folder, metadata_path</td><td>Anchor-specific projection, reference, dark, and metadata inputs.</td><td>Use anchor acquisition files, not the main dataset files.</td></tr>
<tr><td>anchors.preprocessing.inherit_main</td><td>Copies main preprocessing settings for anchors.</td><td>Default true. If false, the anchor preprocessing subfields control anchor binning, crop, flips, log, and clipping.</td></tr>
<tr><td>anchors.alignment.inherit_main</td><td>Copies main alignment settings for anchors.</td><td>Default true. Anchor drift_shift_file can still override metadata shifts.</td></tr>
<tr><td>anchors.exposure.main_exposure_s, anchor_exposure_s, anchor_weight_multiplier, max_weight_ratio</td><td>Controls scalar statistical projection weights.</td><td>Weights are normalized so median main weight is 1.0, and anchors are capped by max_weight_ratio.</td></tr>
<tr><td>anchors.split.mode, recon_count, tune_count, qc_count, random_seed</td><td>Controls deterministic anchor splitting.</td><td>Default interleaved, tune 6, QC 6, recon auto. QC anchors are held out by default.</td></tr>
<tr><td>anchors.split.use_tune_anchors_in_final</td><td>Includes tune anchors in final MBIR-lite.</td><td>Default true. They are still used for FDK/prior scoring first.</td></tr>
<tr><td>anchors.split.use_qc_anchors_in_final</td><td>Includes QC anchors in final MBIR-lite.</td><td>Default false. Leave false for independent QC.</td></tr>
<tr><td>anchors.merge.duplicate_angle_tolerance_deg, combine_duplicate_angles, sort_by_angle</td><td>Controls near-duplicate angle averaging and final angle ordering.</td><td>Default tolerance 0.01 degrees, combine true, sort true.</td></tr>
<tr><td>fdk_sweep.lowres_factor, filters, cutoffs</td><td>Controls FDK candidate sweep.</td><td>Default lowres factor 2. TIGRE cutoff support is not available here, so only cutoff 1.0 is used.</td></tr>
<tr><td>fdk_sweep.metric_anchor_weight, metric_tv_weight, metric_negative_weight</td><td>Weights robust-normalized scoring terms for FDK selection.</td><td>Defaults 1.0, 0.10, 0.10. Anchor residual is the primary score.</td></tr>
<tr><td>prior.tv_weights, tv_iterations, slab_depth, slab_overlap</td><td>Controls training-free TV prior candidates and slab processing.</td><td>Defaults 0.005-0.04, 50 iterations, slab depth 96, overlap 12.</td></tr>
<tr><td>prior_scoring.max_anchor_residual_ratio, min_edge_retention, max_correction_fraction</td><td>Conservative acceptance limits for prior candidates.</td><td>Defaults 1.10, 0.85, 0.15. If none pass, the weakest TV prior is used.</td></tr>
<tr><td>confidence.C_min, C_max, tau_gradient_loss, blur_sigma_voxels</td><td>Controls confidence map range and smoothing.</td><td>Defaults 0.05, 1.0, 0.35, 1.5 voxels.</td></tr>
<tr><td>mbir_lite.n_sweeps, projection_batch_size, ordered_subset_count</td><td>Fast solver iteration, batching, and subset controls.</td><td>Defaults 5, 32, 8.</td></tr>
<tr><td>mbir_lite.start_mode, lambda_tv, rho_prior, tv_epsilon, positivity</td><td>Fast solver initialization, regularization, and positivity controls.</td><td>Defaults fresh_from_fdk, 1.0e-4, 0.05, 1.0e-4, true. Use resume_previous_final only when the run folder already contains mbir_lite_final.npy.</td></tr>
<tr><td>mbir_lite.use_projection_weights, stop_on_qc_anchor_residual</td><td>Controls statistical weights and QC-aware stopping hooks.</td><td>Projection weights default true. QC residual is reported in metrics when QC anchors exist.</td></tr>
<tr><td>fast_recon.enabled, output_subfolder, save_intermediate, make_qc_report, resume_run_folder</td><td>Top-level fast workflow behavior and resume path.</td><td>Defaults false, fast_recon, true, true, null. Stage-only commands need resume_run_folder or --fast-run-folder.</td></tr>
<tr><td>gpu.use_cupy_for_tv_ops</td><td>Allows CuPy TV operations when arrays are CuPy arrays.</td><td>CuPy is optional and not required for the current GUI path.</td></tr>
<tr><td>logging.verbose</td><td>Controls detailed logging in config.</td><td>Recommended true for reconstruction diagnostics.</td></tr>
<tr><td>logging.save_plots</td><td>Controls saving plot outputs.</td><td>Recommended true for record keeping.</td></tr>
<tr><td>logging.save_intermediate</td><td>Controls saving intermediate outputs.</td><td>Recommended true when disk space allows.</td></tr>
</table>

<h2 id="troubleshooting">Troubleshooting</h2>
<table>
<tr><th>Symptom</th><th>Likely cause</th><th>Fix</th></tr>
<tr>
<td>GUI does not launch and says PySide6 is missing.</td>
<td>Qt GUI dependency is not installed in the active Python environment.</td>
<td>Install PySide6 in the same Python used to run mbir.py: python -m pip install PySide6.</td>
</tr>
<tr>
<td>Self-test says TIGRE available: False or No module named tigre.</td>
<td>CERN/TIGRE is not installed in the active environment.</td>
<td>Install CERN/TIGRE with conda from ccpi, for example conda install -c ccpi tigre=2.6. Do not install the unrelated PyPI package named tigre.</td>
</tr>
<tr>
<td>conda is not recognized in PowerShell, but Python comes from Anaconda.</td>
<td>Anaconda Scripts or condabin folder is not on PATH.</td>
<td>Use the full path to conda.exe or open Anaconda Prompt. A common Windows location is AppData/Local/anaconda3/Scripts/conda.exe.</td>
</tr>
<tr>
<td>nvidia-smi not found or Device Monitor unavailable.</td>
<td>NVIDIA driver tools are missing from PATH or no NVIDIA GPU is available.</td>
<td>Install/update NVIDIA drivers or add nvidia-smi to PATH. Reconstruction may still run, but memory estimates will be less informed.</td>
</tr>
<tr>
<td>CUDA cudaMalloc or out-of-memory error.</td>
<td>GPU memory, host RAM, or page-locked buffers are insufficient for the requested operation.</td>
<td>Use Memory mode auto/projection_streaming, reduce Projection batch size, reduce Nz/Ny/Nx, crop, use MBIR extra binning, close apps, or choose a larger GPU.</td>
</tr>
<tr>
<td>FDK works, MBIR fails at initialization.</td>
<td>Implicit FDK initialization may require a large monolithic allocation.</td>
<td>Use existing_fdk with a saved fdk_initial_volume.npy or zeros initialization.</td>
</tr>
<tr>
<td>Validation reports missing projection files.</td>
<td>Projection folder or filename column does not match the metadata.</td>
<td>Check Projection folder, Filename column, and whether filenames include subfolders or extensions.</td>
</tr>
<tr>
<td>Duplicate endpoint warning.</td>
<td>First and last angles are the same physical view, commonly 0 and 360 degrees.</td>
<td>Keep Remove duplicate 0/360 endpoint enabled for standard scans.</td>
</tr>
<tr>
<td>Reconstruction is mirrored, upside down, or transposed.</td>
<td>Orientation, angle sign, or TIGRE row/column mapping is wrong.</td>
<td>Try Flip horizontal, Flip vertical, Invert angle sign for TIGRE, and Transpose attenuation for TIGRE on a small FDK test.</td>
</tr>
<tr>
<td>FDK is blurry or doubled around edges.</td>
<td>Center offset, DSD/DSO, voxel size, or angle order/sign is wrong.</td>
<td>Run center search, verify geometry, and compare reverse angle order/invert angle sign with small FDK tests.</td>
</tr>
<tr>
<td>Drift correction makes images worse.</td>
<td>Shift signs or drift stage do not match how shifts were measured.</td>
<td>Try reversing X shift sign or Y shift sign, and compare transmission versus attenuation drift stage.</td>
</tr>
<tr>
<td>MBIR looks oversmoothed.</td>
<td>lambda TV is too high or too many iterations enforce strong smoothing.</td>
<td>Lower lambda TV and compare FDK/MBIR previews.</td>
</tr>
<tr>
<td>MBIR remains noisy or streaky.</td>
<td>lambda TV is too low, geometry/alignment is off, or too few iterations were run.</td>
<td>First verify FDK geometry and center, then increase lambda TV or iterations.</td>
</tr>
<tr>
<td>Objective or residual oscillates.</td>
<td>Step size too aggressive, ordered subsets too strong, or geometry mismatch.</td>
<td>For Subset-TV lower Subset-TV step safety. For PDHG lower pdhg_step_safety in YAML. Reduce ordered subsets or use projection_streaming.</td>
</tr>
<tr>
<td>Run Fast says main and anchor detector shapes differ.</td>
<td>Main and anchor preprocessing produced different row/column shapes after binning, crop, flips, transpose, or truncation.</td>
<td>Use matching preprocessing shape settings. The fast workflow does not silently resample anchors in this version.</td>
</tr>
<tr>
<td>Make Prior or Run MBIR-lite says a run folder is required.</td>
<td>Stage-only fast commands need to know which previous run contains fdk_best.npy, prior_selected.npy, and saved intermediates.</td>
<td>Set Resume run folder in the GUI or pass --fast-run-folder on the CLI.</td>
</tr>
<tr>
<td>FDK sweep logs that cutoff sweep is unavailable.</td>
<td>The installed TIGRE FDK supports filter names but not custom cutoff values.</td>
<td>This is expected. The app keeps cutoff 1.0 and sweeps filter names.</td>
</tr>
<tr>
<td>Lowres FDK sweep falls back to factor 1.</td>
<td>Detector or volume dimensions are not cleanly divisible by the requested lowres factor.</td>
<td>Use dimensions divisible by 2, or keep factor 1 for safety.</td>
</tr>
<tr>
<td>QC residual improves for the prior but final image looks too smooth.</td>
<td>The prior or rho_prior may be too strong, or larger TV prior weights may have erased edges.</td>
<td>Lower rho_prior, lower MBIR-lite lambda TV, or remove the largest Prior TV weights.</td>
</tr>
<tr>
<td>Final MBIR-lite residual is worse than FDK on held-out QC anchors.</td>
<td>Weights, anchor split, geometry, prior strength, or MBIR-lite step settings may be off.</td>
<td>Verify FDK geometry first, keep QC anchors held out, reduce rho_prior, reduce subsets to 4, or lower batch step risk by reducing batch size.</td>
</tr>
<tr>
<td>Git status shows config-47.yaml modified after pulling.</td>
<td>Line-ending normalization mismatch with .gitattributes.</td>
<td>This does not affect launching or reconstruction. It is a repository housekeeping issue, not an app setting.</td>
</tr>
</table>

<h2 id="faq">FAQ</h2>
<h3>Should Negative log be on?</h3>
<p>Yes for normal raw absorption CT intensities. Reconstruction expects line integrals b = -log(I/I0). Turn it off only if your input data is already attenuation/log converted.</p>

<h3>Should I run FDK before MBIR?</h3>
<p>Yes. FDK is the fastest way to validate geometry, center, angle sign, orientation, and GPU availability. MBIR should refine a plausible reconstruction, not diagnose basic geometry from scratch.</p>

<h3>What is the best default solver?</h3>
<p>Use auto. It chooses ADMM when memory is safe, PDHG when ADMM is too large, and streaming subset-TV when the dataset is very large.</p>

<h3>What is the best default memory mode?</h3>
<p>Use auto. In this app, auto uses projection streaming and memory-aware batch tuning, which is safer for large cone-beam CT than full_gpu.</p>

<h3>When should I use distributed_full_data?</h3>
<p>Use it only on a multi-GPU workstation when exact full-data MBIR operations are desired and host RAM is sufficient. If fewer than two GPUs are selected or RAM is too low, the app falls back.</p>

<h3>How do I choose lambda TV?</h3>
<p>Start at 0.001. Increase if the MBIR result is noisy or streaky. Decrease if edges and fine structures are oversmoothed. Tune on a binned or projection-strided debug run first.</p>

<h3>How many iterations should I run?</h3>
<p>Use 5-10 for fast tests, 20-50 for serious first runs, and more only when metrics and previews still improve. Stop when relative change is low and images are stable.</p>

<h3>Why can CUDA fail when Task Manager shows VRAM available?</h3>
<p>TIGRE also needs host RAM and page-locked host buffers. Fragmentation, other GPU applications, and transient Atb allocations can trigger cudaMalloc failures even when VRAM does not appear full.</p>

<h3>What should I inspect after preprocessing?</h3>
<p>Raw projection should load correctly. Averaged flat should be smooth. Transmission should be mostly positive and physically plausible. Attenuation should have object signal without extreme speckle spikes.</p>

<h3>What if center search metric and my eyes disagree?</h3>
<p>Trust the reconstruction physics and visual result. The metric is a guide. Use fine search near both candidates and compare FDK quality.</p>

<h3>Does the histogram or zoom change my saved data?</h3>
<p>No. Image display controls change only how the active image is shown in the GUI.</p>

<h3>Do I need CuPy?</h3>
<p>No for the current GUI workflow. TIGRE handles GPU projection/backprojection. CuPy is optional for array operations and is not required to launch or pass tests.</p>

<h3>Where are outputs saved?</h3>
<p>Inside a timestamped run folder under Output folder. The run folder includes the saved config, diagnostics, logs, and reconstruction outputs produced by the selected run mode.</p>

<h3>What is an anchor projection?</h3>
<p>An anchor projection is a high-SNR measured projection acquired separately from the main sparse or low-exposure scan. It has its own reference images and metadata. It is used for scoring, weighting, and QC, not neural training.</p>

<h3>Are anchors training data?</h3>
<p>No. The fast workflow is training-free. It does not add PyTorch, TensorFlow, pretrained networks, or per-sample neural training.</p>

<h3>Why are QC anchors held out?</h3>
<p>QC anchors give an independent projection-consistency check. By default they are not used for FDK filter selection, prior selection, or final MBIR-lite reconstruction.</p>

<h3>Should I ever turn on Use QC anchors in MBIR-lite?</h3>
<p>Only after you have already evaluated held-out QC residuals and want a final fit that uses every available measured anchor. For normal evaluation, leave it off.</p>

<h3>What should I do if I only have a few anchors?</h3>
<p>Keep at least one QC anchor if possible. If there are too few anchors for the default 6 tune and 6 QC counts, reduce both counts and keep the split deterministic.</p>

<h3>Why does the fast prior use weak TV?</h3>
<p>The prior is meant to be conservative. It should reduce obvious FDK noise while preserving structure. MBIR-lite then enforces measured projection consistency.</p>

<h3>How do I resume a fast run?</h3>
<p>Set Resume run folder to the timestamped run folder, then click Make Prior, Run MBIR-lite, or QC Report. From CLI, use --fast-run-folder with the same folder.</p>

<h3>What CLI commands run the fast workflow?</h3>
<pre>
python tv_mbir_ct_recon/app.py --config tv_mbir_ct_recon/examples/fast_recon_config.yaml --run-fast
python tv_mbir_ct_recon/app.py --config tv_mbir_ct_recon/examples/fast_recon_config.yaml --run-fdk-sweep
python tv_mbir_ct_recon/app.py --config tv_mbir_ct_recon/examples/fast_recon_config.yaml --make-prior --fast-run-folder output/run_YYYYMMDD_HHMMSS
python tv_mbir_ct_recon/app.py --config tv_mbir_ct_recon/examples/fast_recon_config.yaml --run-mbir-lite --fast-run-folder output/run_YYYYMMDD_HHMMSS
python tv_mbir_ct_recon/app.py --config tv_mbir_ct_recon/examples/fast_recon_config.yaml --run-fast-qc --fast-run-folder output/run_YYYYMMDD_HHMMSS
</pre>

<p><a href="#top">Back to top</a></p>
</body>
</html>
"""
