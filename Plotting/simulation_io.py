"""Read the beam state belonging to each reaction event.

New truth files carry this state themselves. Legacy files need the original
preparation file; when available, run metadata verifies its content identity.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

BEAM_COLUMNS = ["Ebeam_MeV", "x_mm", "y_mm", "z_mm", "dirx", "diry", "dirz", "depth_um"]
TRUTH_BEAM_COLUMNS = {
    "beam_E_MeV": "Ebeam_MeV", "beam_x_mm": "x_mm", "beam_y_mm": "y_mm",
    "beam_z_mm": "z_mm", "beam_dirx": "dirx", "beam_diry": "diry",
    "beam_dirz": "dirz", "depth_um": "depth_um",
}


def _fingerprint(path: Path) -> tuple[int, str]:
    value, size = 14695981039346656037, 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            size += len(chunk)
            for byte in chunk:
                value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return size, f"{value:016x}"


def read_event_beam(output_dir: Path, state: str) -> pd.DataFrame:
    truth_path = output_dir / f"{state}_truth_kinematics.csv"
    columns = set(pd.read_csv(truth_path, nrows=0).columns)
    if set(TRUTH_BEAM_COLUMNS).issubset(columns):
        result = pd.read_csv(truth_path, usecols=["eventID", *TRUTH_BEAM_COLUMNS])
        result = result.rename(columns=TRUTH_BEAM_COLUMNS)
    elif set(TRUTH_BEAM_COLUMNS) & columns:
        raise ValueError(f"Incomplete per-event beam state in {truth_path}")
    else:
        beam_path = output_dir / "beam.dat"
        metadata_path = output_dir / f"{state}_metadata.json"
        identity = None
        if metadata_path.exists():
            identity = (json.loads(metadata_path.read_text()).get("inputs") or {}).get("beam")
        if identity:
            beam_path = Path(identity["path"])
            if _fingerprint(beam_path) != (identity["size_bytes"], identity["fnv1a64"]):
                raise ValueError(f"Beam file no longer matches {metadata_path}: {beam_path}")
        else:
            warnings.warn(
                f"Legacy truth file {truth_path.name} has no beam provenance; "
                "using beam.dat by row order. Verify it is the original preparation file.",
                RuntimeWarning, stacklevel=2,
            )
        result = pd.read_csv(beam_path, comment="#", sep=r"\s+", names=BEAM_COLUMNS)
        result["eventID"] = np.arange(len(result), dtype=np.int64)
    if result["eventID"].duplicated().any():
        raise ValueError(f"Duplicate event IDs in beam state for {state}")
    if not np.isfinite(result[BEAM_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite beam state for {state}")
    return result
