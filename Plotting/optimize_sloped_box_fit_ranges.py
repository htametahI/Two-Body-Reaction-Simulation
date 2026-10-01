#!/usr/bin/env python3
"""Optimize reconstructed-Ex fit ranges for sloped smooth-box templates.

The notebook fits each state/ring group in a fixed window centered on the
histogram mode.  This script scans nearby fixed-width and quantile-based
windows, fits the same normalized sloped smooth-box PDF, and records the best
range under conservative retained-fraction constraints.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-s2223")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib-s2223-xdg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import erf
from scipy.stats import kstest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
DEFAULT_OUTDIR = OUT / "optimized_sloped_box_fit_ranges"

EX_COL = "Ex_calc_reco_center_ringtheta_MeV"
FIT_PLOT_BIN_WIDTH_MEV = 0.01

# Same constants used in output/Plot.ipynb for the reconstructed Ex plots.
MASS_22NE_MEV = 20484.845566
MASS_7LI_MEV = 6535.365833
MASS_TRITON_MEV = 2809.432119
MASS_26MG_MEV = 24202.632149

S3_INNER_RADIUS_MM = 11.0
S3_OUTER_RADIUS_MM = 35.0
S3_RING_COUNT = 24
S3_DISTANCE_MM = 31.0

RING_GROUPS = {
    "all_rings": (1, 24),
    "rings_01_04": (1, 4),
    "rings_05_08": (5, 8),
    "rings_09_12": (9, 12),
    "rings_13_15": (13, 15),
    "rings_19_24": (19, 24),
}


@dataclass(frozen=True)
class CandidateRange:
    lo: float
    hi: float
    strategy: str


def fit_plot_bins(lo: float, hi: float) -> np.ndarray:
    start = np.floor(lo / FIT_PLOT_BIN_WIDTH_MEV) * FIT_PLOT_BIN_WIDTH_MEV
    stop = np.ceil(hi / FIT_PLOT_BIN_WIDTH_MEV) * FIT_PLOT_BIN_WIDTH_MEV
    bins = np.arange(
        start,
        stop + 0.5 * FIT_PLOT_BIN_WIDTH_MEV,
        FIT_PLOT_BIN_WIDTH_MEV,
    )
    if len(bins) < 2:
        return np.array([lo, hi], dtype=float)
    return bins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan and optimize fit ranges for sloped smooth-box state templates.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
        help="Simulation project root.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory for scan CSVs and plots.",
    )
    parser.add_argument(
        "--state",
        action="append",
        default=None,
        help="State to process. May be passed more than once. Defaults to all states.",
    )
    parser.add_argument(
        "--ring-group",
        choices=sorted(RING_GROUPS),
        default="rings_01_04",
        help="Ring group to optimize. Defaults to rings_01_04, used by the MCMC templates.",
    )
    parser.add_argument(
        "--all-ring-groups",
        action="store_true",
        help="Process all defined ring groups instead of only --ring-group.",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=1000,
        help="Minimum events required inside an optimized fit window.",
    )
    parser.add_argument(
        "--min-retained-fraction",
        type=float,
        default=0.80,
        help="Minimum fraction of group events retained by an optimized window.",
    )
    parser.add_argument(
        "--min-edge-sigma-MeV",
        type=float,
        default=0.01,
        help="Minimum allowed fitted smooth-edge width for optimized windows.",
    )
    parser.add_argument(
        "--min-width-MeV",
        type=float,
        default=0.45,
        help="Minimum optimized fit-window width.",
    )
    parser.add_argument(
        "--max-width-MeV",
        type=float,
        default=2.20,
        help="Maximum optimized fit-window width.",
    )
    parser.add_argument(
        "--n-grid-scan",
        type=int,
        default=2500,
        help="Integration grid size used for scan evaluations.",
    )
    parser.add_argument(
        "--n-grid-final",
        type=int,
        default=8000,
        help="Integration grid size used for final/baseline evaluations and plots.",
    )
    parser.add_argument(
        "--n-bins-scan",
        type=int,
        default=160,
        help="Bins used for the fast binned likelihood during range scans.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write only CSV/TXT summaries.",
    )
    return parser.parse_args()


def discover_states(project_root: Path) -> list[str]:
    output_dir = project_root / "output"
    states: list[str] = []
    for truth_path in output_dir.glob("*_truth_kinematics.csv"):
        state = truth_path.name.removesuffix("_truth_kinematics.csv")
        if state.startswith("check_") or state == "reaction":
            continue
        if (output_dir / f"{state}_exit_kinematics.csv").exists():
            states.append(state)
    return sorted(states, key=state_sort_key)


def state_sort_key(state: str) -> tuple[float, str]:
    try:
        return (float(state.split("_", 1)[0]), state)
    except ValueError:
        return (float("inf"), state)


def state_label(state: str) -> str:
    parts = state.split("_", 1)
    if len(parts) == 1:
        return state
    energy = f"{float(parts[0]) / 1000.0:.3f}"
    spin_parity = parts[1]
    if spin_parity.endswith("plus"):
        spin = spin_parity.removesuffix("plus")
        parity = "+"
    elif spin_parity.endswith("minus"):
        spin = spin_parity.removesuffix("minus")
        parity = "-"
    else:
        return rf"$E_x = {energy}\,\mathrm{{MeV}},\ J^\pi = {spin_parity}$"
    return rf"$E_x = {energy}\,\mathrm{{MeV}},\ J^\pi = {spin}^{parity}$"


def calc_excitation(
    beam_energy_mev: np.ndarray,
    triton_energy_mev: np.ndarray,
    triton_theta_lab_deg: np.ndarray,
) -> np.ndarray:
    beam_energy_mev = np.asarray(beam_energy_mev, dtype=float)
    triton_energy_mev = np.asarray(triton_energy_mev, dtype=float)
    triton_theta_lab_deg = np.asarray(triton_theta_lab_deg, dtype=float)

    val1 = (
        MASS_22NE_MEV**2
        + MASS_7LI_MEV**2
        + MASS_TRITON_MEV**2
        + 2.0 * MASS_22NE_MEV * MASS_7LI_MEV
        + 2.0 * MASS_7LI_MEV * beam_energy_mev
    )
    val2 = -2.0 * (MASS_TRITON_MEV + triton_energy_mev) * (
        MASS_22NE_MEV + beam_energy_mev + MASS_7LI_MEV
    )
    val3 = (
        2.0
        * np.cos(np.deg2rad(triton_theta_lab_deg))
        * np.sqrt(beam_energy_mev**2 + 2.0 * MASS_22NE_MEV * beam_energy_mev)
        * np.sqrt(
            triton_energy_mev**2 + 2.0 * MASS_TRITON_MEV * triton_energy_mev
        )
    )
    return np.sqrt(np.maximum(val1 + val2 + val3, 0.0)) - MASS_26MG_MEV


def s3_ring_center_theta_deg(ring_id: np.ndarray) -> np.ndarray:
    ring_zero_based = np.asarray(ring_id, dtype=float) - 1.0
    ring_width = (S3_OUTER_RADIUS_MM - S3_INNER_RADIUS_MM) / S3_RING_COUNT
    r_center = S3_INNER_RADIUS_MM + (ring_zero_based + 0.5) * ring_width
    theta_rad = np.pi - np.arctan2(r_center, S3_DISTANCE_MM)
    return np.rad2deg(theta_rad)


def load_reconstructed_ex(project_root: Path, state: str) -> pd.DataFrame:
    exit_path = project_root / "output" / f"{state}_exit_kinematics.csv"
    usecols = [
        "eventID",
        "Ex_MeV",
        "hasMg26Exit",
        "hasTritonExit",
        "triton_S3_ringID",
        "triton_E_reco_center_MeV",
        "beam_E_center_MeV",
    ]
    df = pd.read_csv(exit_path, usecols=usecols)
    ring = pd.to_numeric(df["triton_S3_ringID"], errors="coerce")
    mask = (
        (df["hasTritonExit"] == 1)
        & (df["hasMg26Exit"] == 1)
        & ring.notna()
        & ring.between(1, S3_RING_COUNT)
        & df["triton_E_reco_center_MeV"].notna()
        & df["beam_E_center_MeV"].notna()
    )
    out = df.loc[mask].copy()
    out["ring"] = ring.loc[mask].astype(int)
    out["triton_theta_ring_center_deg"] = s3_ring_center_theta_deg(out["ring"])
    out[EX_COL] = calc_excitation(
        out["beam_E_center_MeV"].to_numpy(float),
        out["triton_E_reco_center_MeV"].to_numpy(float),
        out["triton_theta_ring_center_deg"].to_numpy(float),
    )
    return out[np.isfinite(out[EX_COL])].copy()


def sloped_smooth_box_shape(
    energy: np.ndarray,
    e_left: float,
    e_right: float,
    sigma_left: float,
    sigma_right: float,
    slope: float,
) -> np.ndarray:
    energy = np.asarray(energy, dtype=float)
    center = 0.5 * (e_left + e_right)
    left_edge = 0.5 * (
        1.0 + erf((energy - e_left) / (np.sqrt(2.0) * sigma_left))
    )
    right_edge = 0.5 * (
        1.0 - erf((energy - e_right) / (np.sqrt(2.0) * sigma_right))
    )
    plateau = 1.0 + slope * (energy - center)
    return plateau * left_edge * right_edge


def initial_edge_guesses(x: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    counts, edges = np.histogram(x, bins=200, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    half_max = 0.5 * counts.max()
    above = centers[counts > half_max]
    if len(above) >= 2:
        return float(above[0]), float(above[-1])
    q25, q75 = np.percentile(x, [25.0, 75.0])
    return float(q25), float(q75)


def fit_binned_scan(
    values: np.ndarray,
    fit_range: tuple[float, float],
    n_bins: int,
) -> dict[str, object] | None:
    lo, hi = fit_range
    x = values[(values >= lo) & (values <= hi)]
    if len(x) < 20:
        return None

    counts, edges = np.histogram(x, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    e_left_guess, e_right_guess = initial_edge_guesses(x, lo, hi)
    sigma_guess = 0.04 * (hi - lo)
    width_range = hi - lo

    def nll(e_left, e_right, sigma_left, sigma_right, slope):
        if sigma_left <= 0.0 or sigma_right <= 0.0 or e_right <= e_left:
            return 1e50
        shape = sloped_smooth_box_shape(
            centers, e_left, e_right, sigma_left, sigma_right, slope
        )
        if np.any(shape < 0.0):
            return 1e50
        weighted_shape = shape * widths
        norm = float(np.sum(weighted_shape))
        if not np.isfinite(norm) or norm <= 0.0:
            return 1e50
        probabilities = weighted_shape / norm
        if np.any((probabilities <= 0.0) & (counts > 0)):
            return 1e50
        return -float(np.sum(counts[counts > 0] * np.log(probabilities[counts > 0])))

    minuit = Minuit(
        nll,
        e_left=e_left_guess,
        e_right=e_right_guess,
        sigma_left=sigma_guess,
        sigma_right=sigma_guess,
        slope=0.0,
    )
    minuit.limits["e_left"] = (lo, hi)
    minuit.limits["e_right"] = (lo, hi)
    minuit.limits["sigma_left"] = (1e-6, None)
    minuit.limits["sigma_right"] = (1e-6, None)
    minuit.limits["slope"] = (-0.9 / width_range, 0.9 / width_range)
    minuit.errordef = Minuit.LIKELIHOOD
    minuit.migrad()
    if not minuit.valid:
        minuit.simplex()
        minuit.migrad()
    if not minuit.valid:
        return None
    return {"x": x, "minuit": minuit}


def fit_unbinned_final(
    values: np.ndarray,
    fit_range: tuple[float, float],
    n_grid: int,
) -> dict[str, object] | None:
    lo, hi = fit_range
    x = values[(values >= lo) & (values <= hi)]
    if len(x) < 20:
        return None

    grid = np.linspace(lo, hi, n_grid)
    e_left_guess, e_right_guess = initial_edge_guesses(x, lo, hi)
    sigma_guess = 0.04 * (hi - lo)
    width_range = hi - lo

    def nll(e_left, e_right, sigma_left, sigma_right, slope):
        if sigma_left <= 0.0 or sigma_right <= 0.0 or e_right <= e_left:
            return 1e50
        shape_x = sloped_smooth_box_shape(
            x, e_left, e_right, sigma_left, sigma_right, slope
        )
        shape_grid = sloped_smooth_box_shape(
            grid, e_left, e_right, sigma_left, sigma_right, slope
        )
        if np.any(shape_x <= 0.0) or np.any(shape_grid < 0.0):
            return 1e50
        norm = trapezoid(shape_grid, grid)
        if not np.isfinite(norm) or norm <= 0.0:
            return 1e50
        return -float(np.sum(np.log(shape_x / norm)))

    minuit = Minuit(
        nll,
        e_left=e_left_guess,
        e_right=e_right_guess,
        sigma_left=sigma_guess,
        sigma_right=sigma_guess,
        slope=0.0,
    )
    minuit.limits["e_left"] = (lo, hi)
    minuit.limits["e_right"] = (lo, hi)
    minuit.limits["sigma_left"] = (1e-6, None)
    minuit.limits["sigma_right"] = (1e-6, None)
    minuit.limits["slope"] = (-0.9 / width_range, 0.9 / width_range)
    minuit.errordef = Minuit.LIKELIHOOD
    minuit.migrad()
    if not minuit.valid:
        minuit.simplex()
        minuit.migrad()
    minuit.hesse()
    if not minuit.valid:
        return None
    return evaluate_fit(x, grid, minuit)


def evaluate_fit(x: np.ndarray, grid: np.ndarray, minuit: Minuit) -> dict[str, object]:
    shape_grid = sloped_smooth_box_shape(
        grid,
        minuit.values["e_left"],
        minuit.values["e_right"],
        minuit.values["sigma_left"],
        minuit.values["sigma_right"],
        minuit.values["slope"],
    )
    norm = trapezoid(shape_grid, grid)
    pdf_grid = shape_grid / norm
    cdf_grid = cumulative_trapezoid(pdf_grid, grid, initial=0.0)
    cdf_grid /= cdf_grid[-1]

    def model_cdf(values):
        return np.interp(values, grid, cdf_grid, left=0.0, right=1.0)

    ks_d, ks_p = kstest(x, model_cdf)
    fwhm, fwhm_left, fwhm_right = fwhm_from_curve(grid, pdf_grid)
    mean_fit = trapezoid(grid * pdf_grid, grid)
    return {
        "x": x,
        "grid": grid,
        "pdf_grid": pdf_grid,
        "cdf_grid": cdf_grid,
        "minuit": minuit,
        "ks_D": float(ks_d),
        "ks_p": float(ks_p),
        "mean_fit_MeV": float(mean_fit),
        "fwhm_MeV": float(fwhm),
        "fwhm_left_MeV": float(fwhm_left),
        "fwhm_right_MeV": float(fwhm_right),
    }


def fwhm_from_curve(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    half_max = 0.5 * np.nanmax(y)
    above = np.flatnonzero(y >= half_max)
    if len(above) == 0:
        return float("nan"), float("nan"), float("nan")
    left_idx = int(above[0])
    right_idx = int(above[-1])
    x_left = interpolate_crossing(x, y, left_idx - 1, left_idx, half_max)
    x_right = interpolate_crossing(x, y, right_idx, right_idx + 1, half_max)
    return max(0.0, x_right - x_left), x_left, x_right


def interpolate_crossing(
    x: np.ndarray,
    y: np.ndarray,
    i0: int,
    i1: int,
    target: float,
) -> float:
    if i0 < 0:
        return float(x[0])
    if i1 >= len(x):
        return float(x[-1])
    x0, x1 = float(x[i0]), float(x[i1])
    y0, y1 = float(y[i0]), float(y[i1])
    if y1 == y0:
        return 0.5 * (x0 + x1)
    frac = np.clip((target - y0) / (y1 - y0), 0.0, 1.0)
    return x0 + frac * (x1 - x0)


def baseline_range(values: np.ndarray) -> CandidateRange:
    counts, edges = np.histogram(values, bins=600)
    peak_idx = int(np.argmax(counts))
    peak_center = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])
    return CandidateRange(float(peak_center - 1.0), float(peak_center + 1.0), "baseline_peak_pm_1MeV")


def candidate_ranges(values: np.ndarray, baseline: CandidateRange) -> list[CandidateRange]:
    ranges: dict[tuple[float, float], CandidateRange] = {}

    def add(lo: float, hi: float, strategy: str) -> None:
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return
        key = (round(float(lo), 5), round(float(hi), 5))
        ranges[key] = CandidateRange(key[0], key[1], strategy)

    center0 = 0.5 * (baseline.lo + baseline.hi)
    for half_width in np.arange(0.55, 1.251, 0.10):
        for offset in np.arange(-0.20, 0.201, 0.10):
            center = center0 + offset
            add(center - half_width, center + half_width, f"mode_grid_hw_{half_width:.2f}_off_{offset:+.2f}")

    percentile_pairs = [
        (0.5, 99.5),
        (1.0, 99.0),
        (2.0, 98.0),
        (3.0, 97.0),
        (5.0, 95.0),
        (7.5, 92.5),
        (10.0, 90.0),
        (1.0, 97.0),
        (2.0, 97.5),
        (2.5, 98.5),
        (5.0, 97.5),
        (7.5, 95.0),
        (10.0, 95.0),
    ]
    for qlo, qhi in percentile_pairs:
        lo, hi = np.percentile(values, [qlo, qhi])
        add(float(lo), float(hi), f"quantile_{qlo:g}_{qhi:g}")

    return list(ranges.values())


def summarize_fit(
    state: str,
    ring_group: str,
    ring_min: int,
    ring_max: int,
    n_events_group: int,
    input_ex: float,
    candidate: CandidateRange,
    fit: dict[str, object],
) -> dict[str, object]:
    minuit: Minuit = fit["minuit"]  # type: ignore[assignment]
    x = fit["x"]  # type: ignore[assignment]
    return {
        "state": state,
        "ring_group": ring_group,
        "ring_min": ring_min,
        "ring_max": ring_max,
        "n_events_group": n_events_group,
        "input_Ex_MeV": input_ex,
        "fit_lo_MeV": candidate.lo,
        "fit_hi_MeV": candidate.hi,
        "fit_width_MeV": candidate.hi - candidate.lo,
        "strategy": candidate.strategy,
        "n_events_fit": len(x),
        "retained_fraction": len(x) / n_events_group,
        "E0_MeV": 0.5 * (minuit.values["e_left"] + minuit.values["e_right"]),
        "mean_fit_MeV": fit["mean_fit_MeV"],
        "box_width_MeV": minuit.values["e_right"] - minuit.values["e_left"],
        "fwhm_MeV": fit["fwhm_MeV"],
        "fwhm_left_MeV": fit["fwhm_left_MeV"],
        "fwhm_right_MeV": fit["fwhm_right_MeV"],
        "E_L_MeV": minuit.values["e_left"],
        "E_R_MeV": minuit.values["e_right"],
        "sigma_L_MeV": minuit.values["sigma_left"],
        "sigma_R_MeV": minuit.values["sigma_right"],
        "slope_MeV_inv": minuit.values["slope"],
        "ks_D": fit["ks_D"],
        "ks_p": fit["ks_p"],
        "minuit_valid": bool(minuit.valid),
        "minuit_fval": float(minuit.fval),
    }


def row_passes_constraints(
    row: dict[str, object],
    min_events: int,
    min_retained_fraction: float,
    min_edge_sigma: float,
    min_width: float,
    max_width: float,
) -> bool:
    return (
        bool(row["minuit_valid"])
        and int(row["n_events_fit"]) >= min_events
        and float(row["retained_fraction"]) >= min_retained_fraction
        and float(row["sigma_L_MeV"]) >= min_edge_sigma
        and float(row["sigma_R_MeV"]) >= min_edge_sigma
        and min_width <= float(row["fit_width_MeV"]) <= max_width
        and float(row["fit_lo_MeV"]) <= float(row["input_Ex_MeV"]) <= float(row["fit_hi_MeV"])
    )


def optimize_group(
    state: str,
    state_df: pd.DataFrame,
    ring_group: str,
    args: argparse.Namespace,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    ring_min, ring_max = RING_GROUPS[ring_group]
    group_df = state_df[state_df["ring"].between(ring_min, ring_max)]
    values = group_df[EX_COL].to_numpy(float)
    values = values[np.isfinite(values)]
    if len(values) < args.min_events:
        return None, []

    input_ex = float(group_df["Ex_MeV"].dropna().iloc[0])
    base_candidate = baseline_range(values)
    all_candidates = [base_candidate] + candidate_ranges(values, base_candidate)

    scan_rows: list[dict[str, object]] = []
    for candidate in all_candidates:
        if candidate.hi - candidate.lo < args.min_width_MeV:
            continue
        if candidate.hi - candidate.lo > args.max_width_MeV:
            continue
        x = values[(values >= candidate.lo) & (values <= candidate.hi)]
        if len(x) < args.min_events:
            continue
        if len(x) / len(values) < args.min_retained_fraction and candidate.strategy != base_candidate.strategy:
            continue
        if not (candidate.lo <= input_ex <= candidate.hi) and candidate.strategy != base_candidate.strategy:
            continue
        scan_fit = fit_binned_scan(
            values,
            (candidate.lo, candidate.hi),
            args.n_bins_scan,
        )
        if scan_fit is None:
            continue
        minuit: Minuit = scan_fit["minuit"]  # type: ignore[assignment]
        x_scan = scan_fit["x"]  # type: ignore[assignment]
        grid = np.linspace(candidate.lo, candidate.hi, args.n_grid_scan)
        evaluated = evaluate_fit(x_scan, grid, minuit)
        row = summarize_fit(
            state,
            ring_group,
            ring_min,
            ring_max,
            len(values),
            input_ex,
            candidate,
            evaluated,
        )
        row["is_baseline"] = candidate.strategy == base_candidate.strategy
        row["passes_constraints"] = row_passes_constraints(
            row,
            args.min_events,
            args.min_retained_fraction,
            args.min_edge_sigma_MeV,
            args.min_width_MeV,
            args.max_width_MeV,
        )
        scan_rows.append(row)

    if not scan_rows:
        return None, []

    baseline_rows = [row for row in scan_rows if bool(row["is_baseline"])]
    valid_rows = [row for row in scan_rows if bool(row["passes_constraints"])]
    if not valid_rows:
        valid_rows = baseline_rows or scan_rows

    best_scan = sorted(
        valid_rows,
        key=lambda row: (
            -float(row["ks_p"]),
            -float(row["retained_fraction"]),
            abs(float(row["mean_fit_MeV"]) - float(row["input_Ex_MeV"])),
        ),
    )[0]

    best_candidate = CandidateRange(
        float(best_scan["fit_lo_MeV"]),
        float(best_scan["fit_hi_MeV"]),
        str(best_scan["strategy"]),
    )
    best_final = fit_unbinned_final(
        values,
        (best_candidate.lo, best_candidate.hi),
        args.n_grid_final,
    )
    if best_final is None:
        return None, scan_rows

    best_row = summarize_fit(
        state,
        ring_group,
        ring_min,
        ring_max,
        len(values),
        input_ex,
        best_candidate,
        best_final,
    )

    baseline_final = fit_unbinned_final(
        values,
        (base_candidate.lo, base_candidate.hi),
        args.n_grid_final,
    )
    if baseline_final is not None:
        baseline_row = summarize_fit(
            state,
            ring_group,
            ring_min,
            ring_max,
            len(values),
            input_ex,
            base_candidate,
            baseline_final,
        )
        for key in [
            "fit_lo_MeV",
            "fit_hi_MeV",
            "fit_width_MeV",
            "n_events_fit",
            "retained_fraction",
            "E0_MeV",
            "box_width_MeV",
            "fwhm_MeV",
            "ks_D",
            "ks_p",
            "minuit_valid",
        ]:
            best_row[f"baseline_{key}"] = baseline_row[key]
        best_row["ks_p_improvement_factor"] = (
            float(best_row["ks_p"]) / max(float(baseline_row["ks_p"]), 1e-300)
        )
    return best_row, scan_rows


def plot_fit(row: dict[str, object], values: np.ndarray, outdir: Path) -> tuple[Path, Path]:
    lo = float(row["fit_lo_MeV"])
    hi = float(row["fit_hi_MeV"])
    x = values[(values >= lo) & (values <= hi)]
    minuit_like = {
        "e_left": float(row["E_L_MeV"]),
        "e_right": float(row["E_R_MeV"]),
        "sigma_left": float(row["sigma_L_MeV"]),
        "sigma_right": float(row["sigma_R_MeV"]),
        "slope": float(row["slope_MeV_inv"]),
    }
    grid = np.linspace(lo, hi, 8000)
    shape = sloped_smooth_box_shape(
        grid,
        minuit_like["e_left"],
        minuit_like["e_right"],
        minuit_like["sigma_left"],
        minuit_like["sigma_right"],
        minuit_like["slope"],
    )
    pdf = shape / trapezoid(shape, grid)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    counts, edges, _ = ax.hist(
        x,
        bins=fit_plot_bins(lo, hi),
        histtype="stepfilled",
        facecolor="#fe6bc2",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.65,
        label="Simulated data",
    )
    bin_width = edges[1] - edges[0]
    ax.plot(grid, len(x) * bin_width * pdf, color="black", lw=2.0, label="Fit")
    ax.axvline(
        float(row["input_Ex_MeV"]),
        color="0.45",
        linestyle="--",
        lw=1.4,
        label=r"Input $E_x$",
    )
    if "baseline_fit_lo_MeV" in row:
        ax.axvspan(
            float(row["baseline_fit_lo_MeV"]),
            float(row["baseline_fit_hi_MeV"]),
            color="tab:blue",
            alpha=0.08,
            label="baseline window",
        )
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"$E_x^\mathrm{calc}$ [MeV]")
    ax.set_ylabel("Counts")
    ax.set_title(f"{state_label(str(row['state']))}: {row['ring_group']} optimized sloped-box fit")
    text = (
        rf"range = {lo:.4f}-{hi:.4f} MeV"
        "\n"
        rf"retained = {100.0 * float(row['retained_fraction']):.1f}%"
        "\n"
        rf"$p_{{KS}}$ = {float(row['ks_p']):.3g}"
        "\n"
        rf"$E_L$ = {float(row['E_L_MeV']):.4f} MeV"
        "\n"
        rf"$E_R$ = {float(row['E_R_MeV']):.4f} MeV"
        "\n"
        rf"FWHM = {float(row['fwhm_MeV']):.4f} MeV"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.85},
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    plot_stem = outdir / f"{row['state']}_{row['ring_group']}_optimized_sloped_box_fit"
    png_path = plot_stem.with_suffix(".png")
    pdf_path = plot_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    outdir = args.outdir
    if not outdir.is_absolute():
        outdir = project_root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    states = args.state or discover_states(project_root)
    ring_groups = sorted(RING_GROUPS) if args.all_ring_groups else [args.ring_group]

    best_rows: list[dict[str, object]] = []
    scan_rows_all: list[dict[str, object]] = []
    for idx, state in enumerate(states, start=1):
        print(f"[{idx}/{len(states)}] Loading {state}")
        state_df = load_reconstructed_ex(project_root, state)
        for ring_group in ring_groups:
            print(f"  optimizing {ring_group}")
            best_row, scan_rows = optimize_group(state, state_df, ring_group, args)
            scan_rows_all.extend(scan_rows)
            if best_row is None:
                print(f"  {ring_group}: no valid fit")
                continue
            if not args.no_plots:
                ring_min, ring_max = RING_GROUPS[ring_group]
                group_values = state_df.loc[
                    state_df["ring"].between(ring_min, ring_max),
                    EX_COL,
                ].to_numpy(float)
                png_path, pdf_path = plot_fit(best_row, group_values, outdir)
                best_row["plot_file_png"] = str(png_path)
                best_row["plot_file_pdf"] = str(pdf_path)
            best_rows.append(best_row)
            base_p = best_row.get("baseline_ks_p", float("nan"))
            print(
                "  best "
                f"{best_row['fit_lo_MeV']:.4f}-{best_row['fit_hi_MeV']:.4f} MeV, "
                f"p={best_row['ks_p']:.4g}; baseline p={base_p:.4g}"
            )

    best_df = pd.DataFrame(best_rows)
    scan_df = pd.DataFrame(scan_rows_all)
    best_csv = outdir / "optimized_sloped_box_fit_ranges.csv"
    scan_csv = outdir / "all_scanned_sloped_box_fit_ranges.csv"
    summary_txt = outdir / "optimized_sloped_box_fit_ranges_summary.txt"
    best_df.to_csv(best_csv, index=False)
    scan_df.to_csv(scan_csv, index=False)

    summary_lines = [
        f"project_root: {project_root}",
        f"states: {', '.join(states)}",
        f"ring_groups: {', '.join(ring_groups)}",
        f"min_events: {args.min_events}",
        f"min_retained_fraction: {args.min_retained_fraction}",
        f"min_edge_sigma_MeV: {args.min_edge_sigma_MeV}",
        f"fit_width_bounds_MeV: {args.min_width_MeV} {args.max_width_MeV}",
        "",
    ]
    if not best_df.empty:
        cols = [
            "state",
            "ring_group",
            "n_events_group",
            "baseline_ks_p",
            "ks_p",
            "baseline_fit_lo_MeV",
            "baseline_fit_hi_MeV",
            "fit_lo_MeV",
            "fit_hi_MeV",
            "retained_fraction",
            "strategy",
        ]
        available = [col for col in cols if col in best_df.columns]
        summary_lines.append(best_df[available].to_string(index=False))
    summary_txt.write_text("\n".join(summary_lines) + "\n")

    print(f"wrote {best_csv}")
    print(f"wrote {scan_csv}")
    print(f"wrote {summary_txt}")


if __name__ == "__main__":
    main()
