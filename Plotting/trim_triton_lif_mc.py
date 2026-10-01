#!/usr/bin/env python3
"""Run a TRIM Monte Carlo check of triton target loss by LiF depth slice.

The previous SRIM check uses only a stopping-power table and a straight-line
CSDA propagation.  This script uses TRIM.DAT to start tritons inside the LiF
target with event-like truth energies, depths, and directions, then reads the
TRIM backscatter/transmission output energies.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from simulation_io import read_event_beam
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401


DEFAULT_STATE = "11336_1minus"
DEFAULT_LIF_THICKNESS_UM = 1.9
DEFAULT_SRIM_ROOT = Path("/Users/mikeqiu/Downloads/SRIM-2013-Std")
DEFAULT_WINEPREFIX = Path(
    "/Users/mikeqiu/GEANT4/S2223_Sim/output/srim_lif/wineprefix",
)


@dataclass
class SliceResult:
    index: int
    percent_lo: int
    percent_hi: int
    depth_lo_um: float
    depth_hi_um: float
    input_df: pd.DataFrame
    exit_df: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TRIM MC for triton energy loss in LiF 5%% depth slices.",
    )
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--srim-root",
        type=Path,
        default=DEFAULT_SRIM_ROOT,
        help="Existing SRIM installation used as the template copy.",
    )
    parser.add_argument(
        "--trim-work-root",
        type=Path,
        default=None,
        help="Workspace SRIM/TRIM copy. Defaults to output/trim_lif/SRIM-2013-Std.",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--wineprefix", type=Path, default=DEFAULT_WINEPREFIX)
    parser.add_argument("--wine-exe", default="wine")
    parser.add_argument("--lif-thickness-um", type=float, default=DEFAULT_LIF_THICKNESS_UM)
    parser.add_argument(
        "--events-per-slice",
        type=int,
        default=300,
        help="Maximum sampled S3-gated Geant4 events per 5%% depth slice.",
    )
    parser.add_argument("--random-seed", type=int, default=2223)
    parser.add_argument(
        "--cascade-mode",
        type=int,
        default=4,
        choices=[4, 5],
        help="TRIM.DAT ion mode: 4 is faster, 5 requests full recoil cascades.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=300,
        help="Per-slice TRIM timeout.",
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=20,
        help="Limit number of 5%% slices, useful for smoke tests.",
    )
    parser.add_argument("--hist-bin-width-keV", type=float, default=1.0)
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Only prepare inputs and stop before launching TRIM.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip TRIM execution and rebuild CSV/plots from saved slice output files.",
    )
    parser.add_argument(
        "--one-ion-per-trim-run",
        action="store_true",
        help="Run each sampled triton as a separate one-ion TRIM job.",
    )
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def write_crlf(path: Path, text: str) -> None:
    path.write_text(text.replace("\n", "\r\n"), encoding="latin-1")


def ensure_trim_workspace(src_root: Path, work_root: Path) -> None:
    if (work_root / "TRIM.exe").exists() and (work_root / "Data").exists():
        return
    if not src_root.exists():
        raise FileNotFoundError(src_root)
    work_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_root, work_root, dirs_exist_ok=True)


def load_events(project_root: Path, state: str) -> pd.DataFrame:
    output_dir = project_root / "output"
    beam = read_event_beam(output_dir, state)

    truth = pd.read_csv(
        output_dir / f"{state}_truth_kinematics.csv",
        usecols=["eventID", "triton_E_MeV", "triton_theta_deg", "triton_phi_deg"],
    ).rename(
        columns={
            "triton_E_MeV": "triton_E_truth_MeV",
            "triton_theta_deg": "triton_theta_truth_deg",
            "triton_phi_deg": "triton_phi_truth_deg",
        },
    )
    exit_df = pd.read_csv(
        output_dir / f"{state}_exit_kinematics.csv",
        usecols=[
            "eventID",
            "hasMg26Exit",
            "hasTritonExit",
            "triton_E_MeV",
            "triton_S3_ringID",
        ],
    ).rename(columns={"triton_E_MeV": "triton_E_exit_MeV"})

    df = exit_df.merge(truth, on="eventID", how="inner", validate="one_to_one").merge(
        beam[["eventID", "depth_um"]],
        on="eventID",
        how="left",
        validate="one_to_one",
    )
    ring = pd.to_numeric(df["triton_S3_ringID"], errors="coerce")
    mask = (
        (df["hasTritonExit"] == 1)
        & (df["hasMg26Exit"] == 1)
        & ring.notna()
        & df["triton_E_truth_MeV"].notna()
        & df["triton_E_exit_MeV"].notna()
        & df["triton_theta_truth_deg"].notna()
        & df["triton_phi_truth_deg"].notna()
        & df["depth_um"].notna()
    )
    df = df.loc[mask].copy()
    df["geant4_dE_target_keV"] = (
        df["triton_E_truth_MeV"] - df["triton_E_exit_MeV"]
    ) * 1000.0
    df["depth_fraction"] = df["depth_um"] / DEFAULT_LIF_THICKNESS_UM

    theta = np.deg2rad(df["triton_theta_truth_deg"].to_numpy(dtype=float))
    phi = np.deg2rad(df["triton_phi_truth_deg"].to_numpy(dtype=float))
    df["trim_cos_x"] = np.cos(theta)
    df["trim_cos_y"] = np.sin(theta) * np.cos(phi)
    df["trim_cos_z"] = np.sin(theta) * np.sin(phi)
    return df


def trim_in_text(
    n_ions: int,
    max_energy_keV: float,
    cascade_mode: int,
    target_width_angstrom: float,
) -> str:
    return f"""==> SRIM-2012.01 This file controls TRIM Calculations.
Ion: Z1 ,  M1,  Energy (keV), Angle,Number,Bragg Corr,AutoSave Number.
     1   3.016049   {max_energy_keV:10.3f}       0   {n_ions:5d}        1     {max(n_ions, 1000):5d}
Cascades(1=No;2=Full;3=Sputt;4-5=Ions;6-7=Neutrons), Random Number Seed, Reminders
                      {cascade_mode:d}                                   0       0
Diskfiles (0=no,1=yes): Ranges, Backscatt, Transmit, Sputtered, Collisions(1=Ion;2=Ion+Recoils), Special EXYZ.txt file
                          0       1           1       0               0                               0
Target material : Number of Elements & Layers
"Triton in LiF from internal depth       "       2               1
PlotType (0-5); Plot Depths: Xmin, Xmax(Ang.) [=0 0 for Viewing Full Target]
       5                         0            {target_width_angstrom:.0f}
Target Elements:    Z   Mass(amu)
Atom 1 = Li =        3   6.941
Atom 2 = F =         9   18.998403
Layer   Layer Name /               Width Density     Li(3)    F(9)
Numb.   Description                (Ang) (g/cm3)    Stoich  Stoich
 1      "LiF"           {target_width_angstrom:.0f}  2.635       1       1
0  Target layer phases (0=Solid, 1=Gas)
0 
Target Compound Corrections (Bragg)
 1  
Individual target atom displacement energies (eV)
      25      25
Individual target atom lattice binding energies (eV)
       3       3
Individual target atom surface binding energies (eV)
       2       2
Stopping Power Version (1=2011, 0=2011)
 0 
"""


def trim_dat_text(sample: pd.DataFrame, description: str) -> str:
    lines = [
        "TRIM with tritons starting inside LiF",
        "Generated from Geant4 truth triton energy, angle, and target depth",
        "Depth X is in Angstrom and X increases from the upstream face into LiF",
        "Negative Cos(X) means the triton exits through the upstream face",
        "One row is one sampled S3-gated Geant4 event",
        "Ion atomic number is 1 and mass is set in TRIM.IN to triton mass",
        "Energy column is eV",
        description[:78],
        "Event  Atom  Energy(eV) Depth(A) Y(A) Z(A) CosX CosY CosZ",
        "Data rows follow",
    ]
    for row in sample.itertuples(index=False):
        name = f"T{int(row.trim_ion_number) % 10000:04d}"
        e_eV = float(row.triton_E_truth_MeV) * 1.0e6
        x_ang = float(row.depth_um) * 1.0e4
        lines.append(
            f"{name:<5s}  1  {e_eV:.8E}  {x_ang:.8E}  0  0  "
            f"{float(row.trim_cos_x):+.8f}  {float(row.trim_cos_y):+.8f}  "
            f"{float(row.trim_cos_z):+.8f}"
        )
    return "\n".join(lines) + "\n"


def select_slice_sample(
    df: pd.DataFrame,
    idx: int,
    lif_thickness_um: float,
    events_per_slice: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float, float]:
    edges = np.linspace(0.0, lif_thickness_um, 21)
    lo = float(edges[idx])
    hi = float(edges[idx + 1])
    if idx == 19:
        mask = (df["depth_um"] >= lo) & (df["depth_um"] <= hi)
    else:
        mask = (df["depth_um"] >= lo) & (df["depth_um"] < hi)
    group = df.loc[mask].copy()
    if len(group) > events_per_slice:
        seed = int(rng.integers(0, 2**32 - 1))
        group = group.sample(n=events_per_slice, random_state=seed)
    group = group.sort_values("eventID").reset_index(drop=True)
    group["trim_ion_number"] = np.arange(1, len(group) + 1, dtype=np.int64)
    return group, lo, hi


def remove_old_trim_outputs(trim_root: Path) -> None:
    names = [
        "BACKSCAT.TXT",
        "TRANSMIT.TXT",
        "RANGE.txt",
        "RANGE.TXT",
        "TRIMOUT.txt",
        "TRIMOUT.TXT",
        "COLLISON.txt",
        "COLLISON.TXT",
        "SPUTTER.txt",
        "SPUTTER.TXT",
    ]
    for name in names:
        path = trim_root / name
        if path.exists():
            path.unlink()
    srim_outputs = trim_root / "SRIM Outputs"
    srim_outputs.mkdir(parents=True, exist_ok=True)
    for name in [
        "BACKSCAT.txt",
        "BACKSCAT.TXT",
        "TRANSMIT.txt",
        "TRANSMIT.TXT",
        "TRIMOUT.txt",
        "TRIMOUT.TXT",
        "RANGE.txt",
        "RANGE.TXT",
        "COLLISON.txt",
        "COLLISON.TXT",
        "SPUTTER.txt",
        "SPUTTER.TXT",
    ]:
        path = srim_outputs / name
        if path.exists():
            path.unlink()


def run_trim(trim_root: Path, wineprefix: Path, wine_exe: str, timeout_s: int) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["WINEPREFIX"] = str(wineprefix)
    try:
        return subprocess.run(
            [wine_exe, "TRIM.exe"],
            cwd=trim_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stderr = f"{stderr}\nTRIM timed out after {timeout_s} seconds.\n"
        return subprocess.CompletedProcess([wine_exe, "TRIM.exe"], 124, stdout, stderr)


def parse_trim_exit_file(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "exit_tag",
                "trim_ion_number",
                "exit_atom_z",
                "trim_exit_E_eV",
                "exit_x_A",
                "exit_y_A",
                "exit_z_A",
                "exit_cos_x",
                "exit_cos_y",
                "exit_cos_z",
            ],
        )
    pattern = re.compile(r"^\s*([BT])\s+(.+)$")
    for line in path.read_text(errors="ignore").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        rest = re.sub(r"([+-])\s+([0-9]+E[+-][0-9]+)", r"\1.\2", match.group(2))
        parts = rest.split()
        if len(parts) < 9:
            continue
        try:
            rows.append(
                {
                    "exit_tag": match.group(1),
                    "trim_ion_number": int(float(parts[0])),
                    "exit_atom_z": int(float(parts[1])),
                    "trim_exit_E_eV": float(parts[2]),
                    "exit_x_A": float(parts[3]),
                    "exit_y_A": float(parts[4]),
                    "exit_z_A": float(parts[5]),
                    "exit_cos_x": float(parts[6]),
                    "exit_cos_y": float(parts[7]),
                    "exit_cos_z": float(parts[8]),
                },
            )
        except ValueError:
            continue
    return pd.DataFrame(rows)


def archive_trim_outputs(trim_root: Path, slice_dir: Path, input_df: pd.DataFrame) -> None:
    slice_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "TRIM.IN",
        "TRIM.DAT",
        "TRIMAUTO",
        "BACKSCAT.TXT",
        "TRANSMIT.TXT",
        "TRIMOUT.TXT",
        "TRIMOUT.txt",
    ]:
        src = trim_root / name
        if src.exists():
            shutil.copy2(src, slice_dir / name)
    srim_outputs = trim_root / "SRIM Outputs"
    for name in [
        "BACKSCAT.txt",
        "BACKSCAT.TXT",
        "TRANSMIT.txt",
        "TRANSMIT.TXT",
        "TRIMOUT.txt",
        "TRIMOUT.TXT",
    ]:
        src = srim_outputs / name
        if src.exists():
            shutil.copy2(src, slice_dir / name)
    input_df.to_csv(slice_dir / "trim_input_events.csv", index=False)


def find_slice_file(slice_dir: Path, *names: str) -> Path:
    for name in names:
        path = slice_dir / name
        if path.exists():
            return path
    return slice_dir / names[0]


def collect_slice(
    slice_dir: Path,
    idx: int,
    depth_lo_um: float,
    depth_hi_um: float,
) -> SliceResult:
    input_df = pd.read_csv(slice_dir / "trim_input_events.csv")
    back = parse_trim_exit_file(find_slice_file(slice_dir, "BACKSCAT.TXT", "BACKSCAT.txt"))
    trans = parse_trim_exit_file(find_slice_file(slice_dir, "TRANSMIT.TXT", "TRANSMIT.txt"))
    exits = pd.concat([back, trans], ignore_index=True)
    if not exits.empty:
        exits = exits[exits["exit_atom_z"] == 1].copy()
        exits = exits.drop_duplicates(subset=["trim_ion_number"], keep="first")
    merged = input_df.merge(exits, on="trim_ion_number", how="left")
    merged["trim_initial_E_eV"] = merged["triton_E_truth_MeV"] * 1.0e6
    merged["trim_dE_target_keV"] = (
        merged["trim_initial_E_eV"] - merged["trim_exit_E_eV"]
    ) / 1000.0
    merged["trim_minus_geant4_keV"] = (
        merged["trim_dE_target_keV"] - merged["geant4_dE_target_keV"]
    )
    percent_lo = idx * 5
    percent_hi = percent_lo + 5
    merged["slice_index"] = idx
    merged["depth_percent_lo"] = percent_lo
    merged["depth_percent_hi"] = percent_hi
    merged["depth_lo_um"] = depth_lo_um
    merged["depth_hi_um"] = depth_hi_um
    return SliceResult(idx, percent_lo, percent_hi, depth_lo_um, depth_hi_um, input_df, merged)


def parse_current_trim_outputs(trim_root: Path) -> pd.DataFrame:
    srim_outputs = trim_root / "SRIM Outputs"
    back = parse_trim_exit_file(find_slice_file(srim_outputs, "BACKSCAT.TXT", "BACKSCAT.txt"))
    trans = parse_trim_exit_file(find_slice_file(srim_outputs, "TRANSMIT.TXT", "TRANSMIT.txt"))
    exits = pd.concat([back, trans], ignore_index=True)
    if not exits.empty:
        exits = exits[exits["exit_atom_z"] == 1].copy()
        exits = exits.drop_duplicates(subset=["trim_ion_number"], keep="first")
    return exits


def add_trim_loss_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trim_initial_E_eV"] = out["triton_E_truth_MeV"] * 1.0e6
    out["trim_dE_target_keV"] = (
        out["trim_initial_E_eV"] - out["trim_exit_E_eV"]
    ) / 1000.0
    out["trim_minus_geant4_keV"] = (
        out["trim_dE_target_keV"] - out["geant4_dE_target_keV"]
    )
    return out


def run_slice_as_one_ion_jobs(
    sample: pd.DataFrame,
    slice_dir: Path,
    trim_work_root: Path,
    args: argparse.Namespace,
    description: str,
    target_width_angstrom: float,
    percent_lo: int,
    percent_hi: int,
) -> pd.DataFrame:
    slice_dir.mkdir(parents=True, exist_ok=True)
    event_root = slice_dir / "event_runs"
    event_root.mkdir(parents=True, exist_ok=True)
    sample.to_csv(slice_dir / "trim_input_events.csv", index=False)
    rows: list[dict[str, float | int | str]] = []
    exit_columns = [
        "exit_tag",
        "exit_atom_z",
        "trim_exit_E_eV",
        "exit_x_A",
        "exit_y_A",
        "exit_z_A",
        "exit_cos_x",
        "exit_cos_y",
        "exit_cos_z",
    ]
    for event_idx, original in enumerate(sample.to_dict(orient="records"), start=1):
        original_ion_number = int(original["trim_ion_number"])
        one = pd.DataFrame([original])
        one["trim_ion_number"] = 1
        remove_old_trim_outputs(trim_work_root)
        one_description = f"{description} event {event_idx:04d}/{len(sample):04d}"
        write_crlf(
            trim_work_root / "TRIM.IN",
            trim_in_text(
                1,
                float(one["triton_E_truth_MeV"].iloc[0]) * 1000.0,
                args.cascade_mode,
                target_width_angstrom,
            ),
        )
        write_crlf(trim_work_root / "TRIM.DAT", trim_dat_text(one, one_description))
        write_crlf(trim_work_root / "TRIMAUTO", "1\n\n")
        completed = run_trim(
            trim_work_root,
            args.wineprefix.resolve(),
            args.wine_exe,
            args.timeout_s,
        )
        event_dir = event_root / f"ion_{original_ion_number:04d}"
        event_dir.mkdir(parents=True, exist_ok=True)
        (event_dir / "wine_stdout.log").write_text(completed.stdout, errors="ignore")
        (event_dir / "wine_stderr.log").write_text(completed.stderr, errors="ignore")
        archive_trim_outputs(trim_work_root, event_dir, one)
        exits = parse_current_trim_outputs(trim_work_root) if completed.returncode == 0 else pd.DataFrame()
        row = dict(original)
        row["trim_ion_number"] = original_ion_number
        row["trim_returncode"] = int(completed.returncode)
        if not exits.empty:
            exit_row = exits.iloc[0].to_dict()
            for col in exit_columns:
                row[col] = exit_row.get(col, np.nan)
        else:
            for col in exit_columns:
                row[col] = np.nan
        rows.append(row)
        print(
            f"slice {percent_lo:02d}-{percent_hi:03d}% event "
            f"{event_idx:03d}/{len(sample):03d}: "
            f"return={completed.returncode} exit={pd.notna(row['trim_exit_E_eV'])}",
            flush=True,
        )
    merged = add_trim_loss_columns(pd.DataFrame(rows))
    merged["slice_index"] = percent_lo // 5
    merged["depth_percent_lo"] = percent_lo
    merged["depth_percent_hi"] = percent_hi
    return merged


def summarize_results(results: list[SliceResult], state: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        exit_df = result.exit_df
        finite_trim = exit_df["trim_dE_target_keV"].to_numpy(dtype=float)
        finite_trim = finite_trim[np.isfinite(finite_trim)]
        finite_g4 = exit_df["geant4_dE_target_keV"].to_numpy(dtype=float)
        finite_g4 = finite_g4[np.isfinite(finite_g4)]
        finite_diff = exit_df["trim_minus_geant4_keV"].to_numpy(dtype=float)
        finite_diff = finite_diff[np.isfinite(finite_diff)]
        quantities = [
            ("geant4_sample_truth_minus_exit", finite_g4),
            ("trim_mc_truth_minus_exit", finite_trim),
            ("trim_mc_minus_geant4_sample", finite_diff),
        ]
        n_input = len(exit_df)
        n_back = int((exit_df["exit_tag"] == "B").sum()) if "exit_tag" in exit_df else 0
        n_trans = int((exit_df["exit_tag"] == "T").sum()) if "exit_tag" in exit_df else 0
        n_exit = int(np.isfinite(exit_df["trim_dE_target_keV"]).sum())
        for quantity, values in quantities:
            if len(values) > 0:
                q = np.percentile(values, [16.0, 50.0, 84.0])
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                median = float(q[1])
                p16 = float(q[0])
                p84 = float(q[2])
            else:
                mean = std = median = p16 = p84 = float("nan")
            rows.append(
                {
                    "state": state,
                    "slice": f"{result.percent_lo:02d}_{result.percent_hi:03d}pct",
                    "depth_min_um": result.depth_lo_um,
                    "depth_max_um": result.depth_hi_um,
                    "quantity": quantity,
                    "N_input": n_input,
                    "N_exit": n_exit,
                    "N_backscatter": n_back,
                    "N_transmit": n_trans,
                    "exit_fraction": n_exit / n_input if n_input else float("nan"),
                    "mean_keV": mean,
                    "std_keV": std,
                    "median_keV": median,
                    "p16_keV": p16,
                    "p84_keV": p84,
                },
            )
    return pd.DataFrame(rows)


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    return [float(x) for x in np.linspace(lo, hi, n)]


def draw_step_hist(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    values: np.ndarray,
    bins: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    counts, edges = np.histogram(values, bins=bins)
    px0, py0, px1, py1 = plot

    def mx(xx: float) -> int:
        return int(px0 + (xx - xlim[0]) / (xlim[1] - xlim[0]) * (px1 - px0))

    def my(yy: float) -> int:
        return int(py1 - (yy - ylim[0]) / (ylim[1] - ylim[0]) * (py1 - py0))

    baseline = my(0.0)
    pts = [(mx(float(edges[0])), baseline)]
    for idx, count in enumerate(counts):
        pts.append((mx(float(edges[idx])), my(float(count))))
        pts.append((mx(float(edges[idx + 1])), my(float(count))))
    pts.append((mx(float(edges[-1])), baseline))
    draw.line(pts, fill=color, width=3)


def draw_hist_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    values: np.ndarray,
    color: tuple[int, int, int],
    bins: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    fonts: dict[str, ImageFont.ImageFont],
    show_xlabel: bool,
) -> None:
    left, top, right, bottom = box
    px0, py0, px1, py1 = (left + 54, top + 48, right - 14, bottom - 44)
    text = (35, 39, 47)
    grid = (228, 232, 236)
    draw.text((left, top), title, font=fonts["title"], fill=text)
    draw.text((px0, top + 23), "Counts", font=fonts["small"], fill=(82, 91, 105))
    draw.rectangle((px0, py0, px1, py1), outline=(180, 186, 194), width=1)

    for x in [0.0, 47.0, 94.0, 141.0, 188.0, 235.0]:
        px = int(px0 + (x - xlim[0]) / (xlim[1] - xlim[0]) * (px1 - px0))
        draw.line([(px, py0), (px, py1)], fill=grid, width=1)
        if show_xlabel:
            label = f"{x:.0f}"
            draw.text(
                (px - int(draw.textlength(label, font=fonts["tick"]) / 2), py1 + 6),
                label,
                font=fonts["tick"],
                fill=text,
            )

    for y in [ylim[0], 0.5 * (ylim[0] + ylim[1]), ylim[1]]:
        py = int(py1 - (y - ylim[0]) / (ylim[1] - ylim[0]) * (py1 - py0))
        draw.line([(px0, py), (px1, py)], fill=grid, width=1)
        label = f"{y:.0f}"
        draw.text(
            (px0 - 8 - int(draw.textlength(label, font=fonts["tick"])), py - 7),
            label,
            font=fonts["tick"],
            fill=text,
        )

    draw.line([(px0, py1), (px1, py1)], fill=text, width=2)
    draw.line([(px0, py0), (px0, py1)], fill=text, width=2)
    draw_step_hist(draw, (px0, py0, px1, py1), values, bins, xlim, ylim, color)

    if len(values) > 1:
        label = (
            f"N={len(values):,}, mean={np.mean(values):.1f} keV, "
            f"std={np.std(values, ddof=1):.1f} keV"
        )
    else:
        label = f"N={len(values):,}"
    draw.line([(px0 + 10, py0 + 17), (px0 + 42, py0 + 17)], fill=color, width=5)
    draw.text((px0 + 50, py0 + 7), label, font=fonts["small"], fill=text)

    if show_xlabel:
        xlabel = "TRIM MC triton target energy loss [keV]"
        draw.text(
            ((px0 + px1) // 2 - int(draw.textlength(xlabel, font=fonts["axis"]) / 2), bottom - 22),
            xlabel,
            font=fonts["axis"],
            fill=text,
        )


def build_hist_plot(
    results: list[SliceResult],
    state: str,
    out_png: Path,
    bin_width_keV: float,
    lif_thickness_um: float,
    cascade_mode: int,
) -> None:
    fonts = {
        "title": load_font(17, bold=True),
        "axis": load_font(14),
        "tick": load_font(12),
        "small": load_font(13),
        "suptitle": load_font(30, bold=True),
        "subtitle": load_font(20),
    }
    canvas = Image.new("RGB", (2300, 3000), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (52, 34),
        f"{state}: TRIM Monte Carlo triton loss for every 5% of LiF depth",
        font=fonts["suptitle"],
        fill=(20, 24, 31),
    )
    draw.text(
        (52, 75),
        (
            f"Each slice is {0.05 * lif_thickness_um:.3f} um thick; raw counts, "
            f"{bin_width_keV:g} keV bins, common x/y scales, TRIM.DAT ion mode {cascade_mode}"
        ),
        font=fonts["subtitle"],
        fill=(82, 91, 105),
    )
    colors = [
        (37, 99, 235),
        (20, 130, 190),
        (15, 118, 110),
        (65, 150, 80),
        (185, 145, 45),
        (217, 119, 6),
        (210, 85, 50),
        (220, 38, 38),
    ]
    bins = np.arange(0.0, 235.0 + bin_width_keV, bin_width_keV)
    ymax = 0.0
    values_by_slice: list[np.ndarray] = []
    for result in results:
        values = result.exit_df["trim_dE_target_keV"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        values_by_slice.append(values)
        counts, _ = np.histogram(values, bins=bins)
        ymax = max(ymax, float(np.max(counts)) if len(counts) else 0.0)
    ylim = (0.0, max(1.0, ymax * 1.2))
    xlim = (0.0, 235.0)

    grid_left, grid_top = 52, 136
    panel_w, panel_h = 430, 520
    gap_x, gap_y = 26, 32
    for idx, result in enumerate(results):
        row = idx // 5
        col = idx % 5
        left = grid_left + col * (panel_w + gap_x)
        top = grid_top + row * (panel_h + gap_y)
        title = f"{result.percent_lo:02d}-{result.percent_hi:03d}% depth"
        draw_hist_panel(
            draw,
            (left, top, left + panel_w, top + panel_h),
            title,
            values_by_slice[idx],
            colors[idx % len(colors)],
            bins,
            xlim,
            ylim,
            fonts,
            show_xlabel=(row == 3),
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    canvas.save(out_png.with_suffix(".pdf"))


def draw_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    title: str,
    fonts: dict[str, ImageFont.ImageFont],
    y_note: str,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    plot = (left + 76, top + 54, right - 24, bottom - 58)
    px0, py0, px1, py1 = plot
    draw.text((left, top), title, font=fonts["title"], fill=(20, 24, 31))
    draw.text((px0, top + 27), y_note, font=fonts["small"], fill=(82, 91, 105))
    draw.rectangle(plot, outline=(180, 186, 194), width=1)
    for x in nice_ticks(*xlim, 6):
        px = int(px0 + (x - xlim[0]) / (xlim[1] - xlim[0]) * (px1 - px0))
        draw.line([(px, py0), (px, py1)], fill=(228, 232, 236), width=1)
        label = f"{x:.0f}"
        draw.text(
            (px - int(draw.textlength(label, font=fonts["tick"]) / 2), py1 + 8),
            label,
            font=fonts["tick"],
            fill=(35, 39, 47),
        )
    for y in nice_ticks(*ylim, 5):
        py = int(py1 - (y - ylim[0]) / (ylim[1] - ylim[0]) * (py1 - py0))
        draw.line([(px0, py), (px1, py)], fill=(228, 232, 236), width=1)
        label = f"{y:.0f}" if abs(y) >= 10 else f"{y:.1f}"
        draw.text(
            (px0 - 10 - int(draw.textlength(label, font=fonts["tick"])), py - 8),
            label,
            font=fonts["tick"],
            fill=(35, 39, 47),
        )
    draw.line([(px0, py1), (px1, py1)], fill=(35, 39, 47), width=2)
    draw.line([(px0, py0), (px0, py1)], fill=(35, 39, 47), width=2)
    draw.text(
        ((px0 + px1) // 2 - int(draw.textlength(xlabel, font=fonts["axis"]) / 2), bottom - 34),
        xlabel,
        font=fonts["axis"],
        fill=(35, 39, 47),
    )
    return plot


def draw_line(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    x: np.ndarray,
    y: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    fonts: dict[str, ImageFont.ImageFont],
    legend_at: tuple[int, int],
) -> None:
    px0, py0, px1, py1 = plot

    def mx(xx: float) -> int:
        return int(px0 + (xx - xlim[0]) / (xlim[1] - xlim[0]) * (px1 - px0))

    def my(yy: float) -> int:
        return int(py1 - (yy - ylim[0]) / (ylim[1] - ylim[0]) * (py1 - py0))

    good = np.isfinite(x) & np.isfinite(y)
    pts = [(mx(float(xx)), my(float(yy))) for xx, yy in zip(x[good], y[good])]
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=4)
    for px, py in pts:
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
    lx, ly = legend_at
    draw.line([(lx, ly + 10), (lx + 36, ly + 10)], fill=color, width=5)
    draw.text((lx + 46, ly), label, font=fonts["small"], fill=(35, 39, 47))


def build_trend_plot(summary: pd.DataFrame, state: str, out_png: Path, cascade_mode: int) -> None:
    fonts = {
        "title": load_font(24, bold=True),
        "axis": load_font(20),
        "tick": load_font(16),
        "small": load_font(17),
        "suptitle": load_font(30, bold=True),
        "subtitle": load_font(20),
    }
    canvas = Image.new("RGB", (1800, 1320), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (48, 32),
        f"{state}: TRIM Monte Carlo target-loss trend by LiF depth",
        font=fonts["suptitle"],
        fill=(20, 24, 31),
    )
    draw.text(
        (48, 72),
        (
            f"TRIM.DAT ion mode {cascade_mode}; input tritons sampled from S3-gated "
            "Geant4 truth by fixed 5% depth slices."
        ),
        font=fonts["subtitle"],
        fill=(82, 91, 105),
    )
    g4 = summary[summary["quantity"] == "geant4_sample_truth_minus_exit"].reset_index(drop=True)
    trim = summary[summary["quantity"] == "trim_mc_truth_minus_exit"].reset_index(drop=True)
    diff = summary[summary["quantity"] == "trim_mc_minus_geant4_sample"].reset_index(drop=True)
    x = np.arange(len(trim), dtype=float) * 5.0 + 2.5

    y_max = max(235.0, float(np.nanmax([g4["mean_keV"].max(), trim["mean_keV"].max()])) * 1.12)
    plot = draw_axes(
        draw,
        (48, 130, 1776, 610),
        (0.0, 100.0),
        (0.0, y_max),
        "reaction depth percentile [%]",
        "Mean target energy loss",
        fonts,
        "Mean loss [keV]",
    )
    draw_line(
        draw,
        plot,
        (0.0, 100.0),
        (0.0, y_max),
        x,
        g4["mean_keV"].to_numpy(float),
        (37, 99, 235),
        "Geant4 sample",
        fonts,
        (plot[0] + 12, plot[1] + 12),
    )
    draw_line(
        draw,
        plot,
        (0.0, 100.0),
        (0.0, y_max),
        x,
        trim["mean_keV"].to_numpy(float),
        (220, 38, 38),
        "TRIM MC",
        fonts,
        (plot[0] + 12, plot[1] + 42),
    )

    std_max = max(20.0, float(np.nanmax([g4["std_keV"].max(), trim["std_keV"].max()])) * 1.25)
    plot = draw_axes(
        draw,
        (48, 700, 884, 1220),
        (0.0, 100.0),
        (0.0, std_max),
        "reaction depth percentile [%]",
        "Distribution width",
        fonts,
        "Std. dev. [keV]",
    )
    draw_line(
        draw,
        plot,
        (0.0, 100.0),
        (0.0, std_max),
        x,
        g4["std_keV"].to_numpy(float),
        (37, 99, 235),
        "Geant4 sample",
        fonts,
        (plot[0] + 12, plot[1] + 12),
    )
    draw_line(
        draw,
        plot,
        (0.0, 100.0),
        (0.0, std_max),
        x,
        trim["std_keV"].to_numpy(float),
        (220, 38, 38),
        "TRIM MC",
        fonts,
        (plot[0] + 12, plot[1] + 42),
    )

    max_abs = max(10.0, float(np.nanmax(np.abs(diff["mean_keV"].to_numpy(float)))) * 1.3)
    plot = draw_axes(
        draw,
        (940, 700, 1776, 1220),
        (0.0, 100.0),
        (-max_abs, max_abs),
        "reaction depth percentile [%]",
        "TRIM - Geant4 mean loss",
        fonts,
        "Difference [keV]",
    )
    draw_line(
        draw,
        plot,
        (0.0, 100.0),
        (-max_abs, max_abs),
        x,
        diff["mean_keV"].to_numpy(float),
        (15, 118, 110),
        "mean difference",
        fonts,
        (plot[0] + 12, plot[1] + 12),
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    canvas.save(out_png.with_suffix(".pdf"))


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    outdir = args.outdir or (project_root / "output" / "trim_lif")
    trim_work_root = args.trim_work_root or (outdir / "SRIM-2013-Std")
    trim_work_root = trim_work_root.resolve()
    outdir = outdir.resolve()
    slice_output_root = outdir / "slice_outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    ensure_trim_workspace(args.srim_root.resolve(), trim_work_root)
    df = load_events(project_root, args.state)
    # This TRIM coordinate setup is for tritons exiting back through the upstream
    # face, so remove any accidental forward-going entries from the gated sample.
    df = df[df["trim_cos_x"] < 0.0].copy()

    rng = np.random.default_rng(args.random_seed)
    target_width_angstrom = args.lif_thickness_um * 1.0e4 + 100.0
    results: list[SliceResult] = []

    for idx in range(min(args.max_slices, 20)):
        sample, depth_lo_um, depth_hi_um = select_slice_sample(
            df,
            idx,
            args.lif_thickness_um,
            args.events_per_slice,
            rng,
        )
        percent_lo = idx * 5
        percent_hi = percent_lo + 5
        slice_dir = slice_output_root / f"slice_{percent_lo:02d}_{percent_hi:03d}pct"
        if not args.reuse_existing:
            if sample.empty:
                continue
            description = (
                f"{args.state} triton LiF slice {percent_lo:02d}-{percent_hi:03d}% "
                f"depth {depth_lo_um:.4f}-{depth_hi_um:.4f} um"
            )
            if args.one_ion_per_trim_run and not args.no_run:
                merged = run_slice_as_one_ion_jobs(
                    sample,
                    slice_dir,
                    trim_work_root,
                    args,
                    description,
                    target_width_angstrom,
                    percent_lo,
                    percent_hi,
                )
                merged["depth_lo_um"] = depth_lo_um
                merged["depth_hi_um"] = depth_hi_um
                merged.to_csv(slice_dir / "trim_matched_events.csv", index=False)
                result = SliceResult(idx, percent_lo, percent_hi, depth_lo_um, depth_hi_um, sample, merged)
                results.append(result)
                n_exit = int(np.isfinite(result.exit_df["trim_dE_target_keV"]).sum())
                print(
                    f"slice {percent_lo:02d}-{percent_hi:03d}%: "
                    f"input={len(result.exit_df)} exit={n_exit}",
                    flush=True,
                )
                continue

            remove_old_trim_outputs(trim_work_root)
            write_crlf(
                trim_work_root / "TRIM.IN",
                trim_in_text(
                    len(sample),
                    float(sample["triton_E_truth_MeV"].max()) * 1000.0,
                    args.cascade_mode,
                    target_width_angstrom,
                ),
            )
            write_crlf(trim_work_root / "TRIM.DAT", trim_dat_text(sample, description))
            write_crlf(trim_work_root / "TRIMAUTO", "1\n\n")

            if args.no_run:
                archive_trim_outputs(trim_work_root, slice_dir, sample)
                continue

            completed = run_trim(
                trim_work_root,
                args.wineprefix.resolve(),
                args.wine_exe,
                args.timeout_s,
            )
            slice_dir.mkdir(parents=True, exist_ok=True)
            (slice_dir / "wine_stdout.log").write_text(completed.stdout, errors="ignore")
            (slice_dir / "wine_stderr.log").write_text(completed.stderr, errors="ignore")
            if completed.returncode != 0:
                archive_trim_outputs(trim_work_root, slice_dir, sample)
                raise RuntimeError(
                    f"TRIM failed for slice {percent_lo:02d}-{percent_hi:03d}% "
                    f"with return code {completed.returncode}. See {slice_dir}.",
                )
            archive_trim_outputs(trim_work_root, slice_dir, sample)

        result = collect_slice(slice_dir, idx, depth_lo_um, depth_hi_um)
        result.exit_df.to_csv(slice_dir / "trim_matched_events.csv", index=False)
        results.append(result)
        n_exit = int(np.isfinite(result.exit_df["trim_dE_target_keV"]).sum())
        print(
            f"slice {percent_lo:02d}-{percent_hi:03d}%: "
            f"input={len(result.exit_df)} exit={n_exit}",
            flush=True,
        )

    if args.no_run:
        print(f"Prepared TRIM inputs under {slice_output_root}")
        return
    if not results:
        raise RuntimeError("No TRIM slice results were collected.")

    all_events = pd.concat([item.exit_df for item in results], ignore_index=True)
    summary = summarize_results(results, args.state)
    all_events.to_csv(outdir / f"{args.state}_trim_mc_5percent_events.csv", index=False)
    summary.to_csv(outdir / f"{args.state}_trim_mc_5percent_summary.csv", index=False)
    build_hist_plot(
        results,
        args.state,
        outdir / f"{args.state}_trim_mc_lif_loss_5percent_segments.png",
        args.hist_bin_width_keV,
        args.lif_thickness_um,
        args.cascade_mode,
    )
    build_trend_plot(
        summary,
        args.state,
        outdir / f"{args.state}_trim_mc_5percent_trends.png",
        args.cascade_mode,
    )
    print(f"Wrote {outdir / f'{args.state}_trim_mc_5percent_summary.csv'}")
    print(f"Wrote {outdir / f'{args.state}_trim_mc_lif_loss_5percent_segments.png'}")
    print(f"Wrote {outdir / f'{args.state}_trim_mc_5percent_trends.png'}")


if __name__ == "__main__":
    main()
