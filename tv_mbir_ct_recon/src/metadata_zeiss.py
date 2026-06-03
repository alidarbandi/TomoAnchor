from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


FILENAME_HINTS = ("filename", "file", "tiff", "tif", "image", "projection", "tiff_file", "output_name")
ANGLE_HINTS = ("angle_deg", "theta_deg", "angle", "theta", "rotation", "view_angle")
ANGLE_RAD_HINTS = ("angle_rad", "theta_rad", "radian", "radians", "angle_radian", "theta_radian")
X_SHIFT_HINTS = ("x_shift_px", "shift_x_px", "x_shift", "shift_x", "u_shift_px", "col_shift_px")
Y_SHIFT_HINTS = ("y_shift_px", "shift_y_px", "y_shift", "shift_y", "v_shift_px", "row_shift_px")


@dataclass(frozen=True)
class ProjectionRecord:
    index: int
    filename: str
    path: Path
    angle_deg: float
    angle_rad: float
    x_shift_px: float | None
    y_shift_px: float | None
    exists: bool


@dataclass
class MetadataValidation:
    records: list[ProjectionRecord]
    missing_files: list[str]
    duplicate_filenames: list[str]
    duplicate_endpoint_detected: bool
    duplicate_endpoint_removed: bool
    angle_direction: str
    angle_min_deg: float
    angle_max_deg: float
    angle_min_rad: float
    angle_max_rad: float
    angles_rad: np.ndarray
    angle_input_source: str
    x_shift_px: np.ndarray | None
    y_shift_px: np.ndarray | None
    x_shift_column_found: bool
    y_shift_column_found: bool
    x_shift_column: str
    y_shift_column: str
    table_rows: list[dict[str, str]]
    warnings: list[str]


def find_metadata_csv(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Metadata path does not exist: {candidate}")
    preferred = candidate / "projection_geometry.csv"
    if preferred.exists():
        return preferred
    csv_files = sorted(candidate.glob("*.csv"))
    if not csv_files:
        csv_files = sorted(candidate.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in metadata folder: {candidate}")
    return csv_files[0]


def load_metadata_csv(path: str | Path) -> pd.DataFrame:
    csv_path = find_metadata_csv(path)
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"Metadata CSV is empty: {csv_path}")
    return frame


def suggest_column(columns: Sequence[str], hints: Sequence[str]) -> str:
    normalized = [(column, column.lower().strip()) for column in columns]
    for hint in hints:
        for original, lower in normalized:
            if lower == hint:
                return original
    for hint in hints:
        for original, lower in normalized:
            if hint in lower:
                return original
    return columns[0] if columns else ""


def suggest_optional_column(columns: Sequence[str], hints: Sequence[str]) -> str:
    normalized = [(column, column.lower().strip()) for column in columns]
    for hint in hints:
        for original, lower in normalized:
            if lower == hint:
                return original
    for hint in hints:
        for original, lower in normalized:
            if hint in lower:
                return original
    return ""


def resolve_projection_path(projection_folder: str | Path, filename: str) -> Path:
    folder = Path(projection_folder)
    raw = Path(str(filename).strip())
    direct = folder / raw
    if direct.exists():
        return direct
    basename = folder / raw.name
    if basename.exists():
        return basename
    return direct


def validate_metadata(
    frame: pd.DataFrame,
    projection_folder: str | Path,
    filename_column: str,
    angle_column: str,
    angle_rad_column: str | None = None,
    x_shift_column: str | None = None,
    y_shift_column: str | None = None,
    remove_duplicate_endpoint: bool = True,
    reverse_angle_order: bool = False,
    duplicate_tolerance_deg: float = 1e-3,
) -> MetadataValidation:
    if not filename_column:
        filename_column = suggest_column([str(c) for c in frame.columns], FILENAME_HINTS)
    if not angle_column:
        angle_column = suggest_column([str(c) for c in frame.columns], ANGLE_HINTS)
    if filename_column not in frame.columns:
        raise ValueError(f"Filename column not found: {filename_column}")
    if angle_column not in frame.columns:
        raise ValueError(f"Angle column not found: {angle_column}")
    projection_dir = Path(projection_folder)
    if not projection_dir.exists():
        raise FileNotFoundError(f"Projection folder does not exist: {projection_dir}")

    warnings: list[str] = []
    filenames = frame[filename_column].astype(str).str.strip()
    if filenames.eq("").any():
        bad = filenames.index[filenames.eq("")].tolist()[:10]
        raise ValueError(f"Filename column contains empty values at row indexes: {bad}")

    filenames_list = filenames.tolist()
    duplicate_filenames = sorted(name for name in set(filenames_list) if filenames_list.count(name) > 1)
    deg_values = pd.to_numeric(frame[angle_column], errors="coerce")
    if deg_values.isna().any():
        bad = deg_values.index[deg_values.isna()].tolist()[:10]
        raise ValueError(f"Degree angle column contains empty or non-numeric values at row indexes: {bad}")
    degree_column_values = deg_values.astype(np.float32).to_numpy()

    angles_rad: np.ndarray | None = None
    angles_deg: np.ndarray | None = None
    angle_input_source = f"degree column '{angle_column}' converted to radians"
    rad_column = (angle_rad_column or "").strip()
    if rad_column:
        if rad_column not in frame.columns:
            warnings.append(f"Radian angle column '{rad_column}' was not found; falling back to degree column.")
        else:
            rad_values = pd.to_numeric(frame[rad_column], errors="coerce")
            if rad_values.notna().any():
                if rad_values.isna().any():
                    bad = rad_values.index[rad_values.isna()].tolist()[:10]
                    raise ValueError(f"Radian angle column contains empty or non-numeric values at row indexes: {bad}")
                angles_rad = rad_values.astype(np.float32).to_numpy()
                angles_deg = degree_column_values
                angle_input_source = f"radian column '{rad_column}'"
            else:
                warnings.append(f"Radian angle column '{rad_column}' is empty; falling back to degree column.")
    if angles_rad is None or angles_deg is None:
        angles_deg = degree_column_values
        angles_rad = np.deg2rad(angles_deg).astype(np.float32)

    x_shift_values, x_shift_found = _optional_numeric_column(frame, x_shift_column, "X shift", warnings)
    y_shift_values, y_shift_found = _optional_numeric_column(frame, y_shift_column, "Y shift", warnings)

    work = pd.DataFrame(
        {
            "filename": filenames.to_numpy(),
            "angle_deg": angles_deg,
            "angle_rad": angles_rad,
            "x_shift_px": x_shift_values if x_shift_values is not None else np.full(len(frame), np.nan, dtype=np.float32),
            "y_shift_px": y_shift_values if y_shift_values is not None else np.full(len(frame), np.nan, dtype=np.float32),
        },
        index=frame.index,
    )

    duplicate_endpoint_detected = _duplicate_endpoint_angle(np.rad2deg(angles_rad.astype(np.float64)), duplicate_tolerance_deg)
    duplicate_endpoint_removed = False
    if duplicate_endpoint_detected:
        warnings.append("First and last projection angles appear to be duplicate endpoints.")
        if remove_duplicate_endpoint:
            work = work.iloc[:-1].copy()
            duplicate_endpoint_removed = True
            warnings.append("Removed the last metadata row because duplicate endpoint removal is enabled.")

    records: list[ProjectionRecord] = []
    missing: list[str] = []
    for index, row in enumerate(work.itertuples(index=False), start=0):
        filename = str(row.filename).strip()
        x_shift_px = None if math.isnan(float(row.x_shift_px)) else float(row.x_shift_px)
        y_shift_px = None if math.isnan(float(row.y_shift_px)) else float(row.y_shift_px)
        path = resolve_projection_path(projection_dir, filename)
        exists = path.exists()
        if not exists:
            missing.append(filename)
        records.append(
            ProjectionRecord(
                index=index,
                filename=filename,
                path=path,
                angle_deg=float(row.angle_deg),
                angle_rad=float(row.angle_rad),
                x_shift_px=x_shift_px,
                y_shift_px=y_shift_px,
                exists=exists,
            )
        )

    if reverse_angle_order:
        records = list(reversed(records))
        warnings.append("Reversed projection-angle order while preserving filename/angle pairs.")

    angles_deg_array = np.asarray([record.angle_deg for record in records], dtype=np.float32)
    angles_rad_array = np.asarray([record.angle_rad for record in records], dtype=np.float32)
    return MetadataValidation(
        records=records,
        missing_files=missing,
        duplicate_filenames=duplicate_filenames,
        duplicate_endpoint_detected=duplicate_endpoint_detected,
        duplicate_endpoint_removed=duplicate_endpoint_removed,
        angle_direction=describe_angle_direction(np.rad2deg(angles_rad_array.astype(np.float64))),
        angle_min_deg=float(np.min(angles_deg_array)) if angles_deg_array.size else math.nan,
        angle_max_deg=float(np.max(angles_deg_array)) if angles_deg_array.size else math.nan,
        angle_min_rad=float(np.min(angles_rad_array)) if angles_rad_array.size else math.nan,
        angle_max_rad=float(np.max(angles_rad_array)) if angles_rad_array.size else math.nan,
        angles_rad=angles_rad_array,
        angle_input_source=angle_input_source,
        x_shift_px=_records_shift_array(records, "x_shift_px"),
        y_shift_px=_records_shift_array(records, "y_shift_px"),
        x_shift_column_found=x_shift_found,
        y_shift_column_found=y_shift_found,
        x_shift_column=(x_shift_column or "").strip(),
        y_shift_column=(y_shift_column or "").strip(),
        table_rows=validation_preview_rows(records),
        warnings=warnings,
    )


def validation_preview_rows(records: Sequence[ProjectionRecord], edge_count: int = 5) -> list[dict[str, str]]:
    chosen = list(records) if len(records) <= edge_count * 2 else list(records[:edge_count]) + list(records[-edge_count:])
    return [
        {
            "index": str(record.index),
            "filename": record.filename,
            "angle_deg": f"{record.angle_deg:.8g}",
            "angle_rad": f"{record.angle_rad:.10g}",
            "x_shift_px": _format_optional_float(record.x_shift_px),
            "y_shift_px": _format_optional_float(record.y_shift_px),
            "exists": "yes" if record.exists else "no",
        }
        for record in chosen
    ]


def describe_angle_direction(angles_deg: np.ndarray) -> str:
    if angles_deg.size < 2:
        return "single angle"
    diffs = np.diff(angles_deg.astype(np.float64))
    positive = np.count_nonzero(diffs > 0)
    negative = np.count_nonzero(diffs < 0)
    if positive and not negative:
        return "ascending"
    if negative and not positive:
        return "descending"
    if not positive and not negative:
        return "constant"
    return "mixed/non-monotonic"


def _duplicate_endpoint_angle(angles_deg: np.ndarray, tolerance_deg: float) -> bool:
    if angles_deg.size < 2:
        return False
    first = float(angles_deg[0])
    last = float(angles_deg[-1])
    wrapped_distance = abs(((last - first + 180.0) % 360.0) - 180.0)
    return wrapped_distance <= tolerance_deg or abs(abs(last - first) - 360.0) <= tolerance_deg


def _optional_numeric_column(
    frame: pd.DataFrame,
    column: str | None,
    label: str,
    warnings: list[str],
) -> tuple[np.ndarray | None, bool]:
    clean_column = (column or "").strip()
    if not clean_column:
        return None, False
    if clean_column not in frame.columns:
        warnings.append(f"{label} column '{clean_column}' was not found.")
        return None, False
    values = pd.to_numeric(frame[clean_column], errors="coerce")
    if not values.notna().any():
        warnings.append(f"{label} column '{clean_column}' is empty.")
        return None, True
    if values.isna().any():
        bad = values.index[values.isna()].tolist()[:10]
        raise ValueError(f"{label} column contains empty or non-numeric values at row indexes: {bad}")
    return values.astype(np.float32).to_numpy(), True


def _records_shift_array(records: Sequence[ProjectionRecord], attribute: str) -> np.ndarray | None:
    values = [getattr(record, attribute) for record in records]
    if not values or any(value is None for value in values):
        return None
    return np.asarray(values, dtype=np.float32)


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.8g}"
