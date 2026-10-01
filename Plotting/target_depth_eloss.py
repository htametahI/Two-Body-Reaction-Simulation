#!/usr/bin/env python3
"""Compare triton target energy loss by reaction depth.

New truth CSVs store the sampled beam state and LiF reaction depth alongside
the generated triton energy. Legacy results use the original beam file.
The eventID join supplies depth slices for S3-gated target energy-loss plots.
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
from simulation_io import read_event_beam
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401


LIF_THICKNESS_UM = 1.9
DEFAULT_STATE = "11336_1minus"
ENERGY_LOSS_BIN_WIDTH_KEV = 0.5
RESIDUAL_BIN_WIDTH_KEV = 1.0


@dataclass
class Series:
    name: str
    values: np.ndarray
    color: tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot target energy loss split by LiF reaction depth.",
    )
    parser.add_argument(
        "--state",
        default=DEFAULT_STATE,
        help="State name used in output/<state>_{truth,exit}_kinematics.csv.",
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Run over all states with matching truth/exit kinematics CSVs.",
    )
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="Only regenerate the 5%-slice overlay plot.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Simulation project root.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Directory for plots and summary CSV.",
    )
    parser.add_argument(
        "--surface-fraction",
        type=float,
        default=0.05,
        help="Fraction of events used for each near-surface extreme.",
    )
    parser.add_argument(
        "--lif-thickness-um",
        type=float,
        default=LIF_THICKNESS_UM,
        help="LiF thickness used for first/second-half split.",
    )
    return parser.parse_args()


def discover_states(project_root: Path) -> list[str]:
    output_dir = project_root / "output"
    states: list[str] = []
    for truth_path in output_dir.glob("*_truth_kinematics.csv"):
        state = truth_path.name.removesuffix("_truth_kinematics.csv")
        if state.startswith("check_") or state == "reaction":
            continue
        exit_path = output_dir / f"{state}_exit_kinematics.csv"
        if exit_path.exists():
            states.append(state)
    return sorted(states, key=state_sort_key)


def state_sort_key(state: str) -> tuple[float, str]:
    try:
        return (float(state.split("_", 1)[0]), state)
    except ValueError:
        return (float("inf"), state)


def state_latex_label(state: str) -> str:
    parts = state.split("_", 1)
    energy_label = parts[0]
    if energy_label.isdigit():
        energy_mev = int(energy_label) / 1000.0
        energy = f"{energy_mev:.3f}\\,\\mathrm{{MeV}}"
    else:
        energy = energy_label.replace("_", r"\_")

    jpi = ""
    if len(parts) > 1:
        spin = parts[1]
        if spin.endswith("plus"):
            jpi = spin.removesuffix("plus") + "^{+}"
        elif spin.endswith("minus"):
            jpi = spin.removesuffix("minus") + "^{-}"
        else:
            jpi = spin.replace("_", r"\_")

    if jpi:
        return rf"$E_x={energy},\ J^\pi={jpi}$"
    return rf"$E_x={energy}$"


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


def read_joined(project_root: Path, state: str, lif_thickness_um: float) -> pd.DataFrame:
    output_dir = project_root / "output"
    truth_path = output_dir / f"{state}_truth_kinematics.csv"
    exit_path = output_dir / f"{state}_exit_kinematics.csv"

    for path in [truth_path, exit_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    beam = read_event_beam(output_dir, state)

    truth = pd.read_csv(
        truth_path,
        usecols=["eventID", "triton_E_MeV", "triton_theta_deg"],
    ).rename(
        columns={
            "triton_E_MeV": "triton_E_truth_MeV",
            "triton_theta_deg": "triton_theta_truth_deg",
        },
    )

    exit_df = pd.read_csv(
        exit_path,
        usecols=[
            "eventID",
            "hasMg26Exit",
            "hasTritonExit",
            "triton_E_MeV",
            "triton_theta_deg",
            "triton_S3_ringID",
            "triton_S3_edep_smeared_MeV",
            "triton_E_reco_center_MeV",
            "beam_E_center_MeV",
        ],
    ).rename(
        columns={
            "triton_E_MeV": "triton_E_exit_MeV",
            "triton_theta_deg": "triton_theta_exit_deg",
        },
    )

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
        & df["triton_E_exit_MeV"].notna()
        & df["triton_E_truth_MeV"].notna()
        & df["depth_um"].notna()
    )
    df = df.loc[mask].copy()

    df["triton_dE_target_keV"] = (
        df["triton_E_truth_MeV"] - df["triton_E_exit_MeV"]
    ) * 1000.0
    df["center_reco_minus_truth_keV"] = (
        df["triton_E_reco_center_MeV"] - df["triton_E_truth_MeV"]
    ) * 1000.0
    df["depth_fraction"] = df["depth_um"] / lif_thickness_um
    return df


def finite(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def finite_concat(arrays: list[np.ndarray]) -> np.ndarray:
    finite_arrays = []
    for values in arrays:
        x = finite(values)
        if len(x) > 0:
            finite_arrays.append(x)
    if not finite_arrays:
        return np.array([], dtype=float)
    return np.concatenate(finite_arrays)


def value_range(values: np.ndarray, fallback: tuple[float, float]) -> tuple[float, float]:
    x = finite(values)
    if len(x) == 0:
        return fallback
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi > lo:
        return lo, hi
    pad = 0.5
    return lo - pad, hi + pad


def visible_value_range(
    values: np.ndarray,
    width: float,
    fallback: tuple[float, float],
    lower_percentile: float = 0.05,
    upper_percentile: float = 99.95,
) -> tuple[float, float]:
    x = finite(values)
    if len(x) == 0:
        return fallback
    exact_lo, exact_hi = value_range(x, fallback)
    if len(x) < 1000:
        return exact_lo, exact_hi

    p_lo, p_hi = np.percentile(x, [lower_percentile, upper_percentile])
    central_span = max(float(p_hi - p_lo), width)
    outlier_gap = max(10.0 * width, 0.25 * central_span)

    lo = exact_lo
    hi = exact_hi
    if float(p_lo - exact_lo) > outlier_gap:
        lo = float(p_lo - 2.0 * width)
    if float(exact_hi - p_hi) > outlier_gap:
        hi = float(p_hi + 2.0 * width)
    if hi <= lo:
        hi = lo + width
    return lo, hi


def bins_for_values(
    values: np.ndarray,
    width: float,
    fallback: tuple[float, float],
    visible_range: bool = False,
) -> tuple[np.ndarray, tuple[float, float]]:
    if visible_range:
        lo, hi = visible_value_range(values, width, fallback)
    else:
        lo, hi = value_range(values, fallback)
    if hi <= lo:
        hi = lo + width
    bins = np.arange(lo, hi, width)
    if len(bins) == 0 or bins[0] != lo:
        bins = np.insert(bins, 0, lo)
    if bins[-1] < hi:
        bins = np.append(bins, hi)
    if len(bins) < 2:
        bins = np.array([lo, hi], dtype=float)
    return bins, (lo, hi)


def histogram_fwhm(values: np.ndarray, bins: np.ndarray) -> float:
    x = finite(values)
    if len(x) == 0 or len(bins) < 2:
        return float("nan")

    counts, edges = np.histogram(x, bins=bins)
    if len(counts) == 0:
        return float("nan")
    peak = float(np.max(counts))
    if peak <= 0.0:
        return float("nan")

    half_max = 0.5 * peak
    above = np.flatnonzero(counts >= half_max)
    if len(above) == 0:
        return float("nan")

    centers = 0.5 * (edges[:-1] + edges[1:])
    left_idx = int(above[0])
    right_idx = int(above[-1])

    if left_idx == 0:
        left = float(edges[0])
    else:
        x0 = float(centers[left_idx - 1])
        x1 = float(centers[left_idx])
        y0 = float(counts[left_idx - 1])
        y1 = float(counts[left_idx])
        left = interpolate_half_max_crossing(x0, y0, x1, y1, half_max)

    if right_idx == len(counts) - 1:
        right = float(edges[-1])
    else:
        x0 = float(centers[right_idx])
        x1 = float(centers[right_idx + 1])
        y0 = float(counts[right_idx])
        y1 = float(counts[right_idx + 1])
        right = interpolate_half_max_crossing(x0, y0, x1, y1, half_max)

    return max(0.0, right - left)


def interpolate_half_max_crossing(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    half_max: float,
) -> float:
    if y1 == y0:
        return 0.5 * (x0 + x1)
    fraction = (half_max - y0) / (y1 - y0)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return x0 + fraction * (x1 - x0)


def count_stats_label(values: np.ndarray, value_suffix: str = "keV") -> str:
    values = finite(values)
    if len(values) == 0:
        return "N=0"
    std = np.std(values, ddof=1) if len(values) > 1 else float("nan")
    return (
        f"N={len(values):,}, mean={np.mean(values):.1f} {value_suffix}, "
        f"std={std:.1f} {value_suffix}"
    )


def stats_row(
    state: str,
    group: str,
    quantity: str,
    values: np.ndarray,
    depth_values: np.ndarray,
) -> dict[str, float | int | str]:
    x = finite(values)
    d = finite(depth_values)
    depth_min = float(np.min(d)) if len(d) else float("nan")
    depth_max = float(np.max(d)) if len(d) else float("nan")
    if len(x) == 0:
        return {
            "state": state,
            "group": group,
            "quantity": quantity,
            "N": 0,
            "depth_min_um": depth_min,
            "depth_max_um": depth_max,
            "mean_keV": float("nan"),
            "std_keV": float("nan"),
            "median_keV": float("nan"),
            "p02_5_keV": float("nan"),
            "p16_keV": float("nan"),
            "p84_keV": float("nan"),
            "p97_5_keV": float("nan"),
            "iqr_keV": float("nan"),
        }
    q = np.percentile(x, [2.5, 16.0, 50.0, 84.0, 97.5])
    return {
        "state": state,
        "group": group,
        "quantity": quantity,
        "N": int(len(x)),
        "depth_min_um": depth_min,
        "depth_max_um": depth_max,
        "mean_keV": float(np.mean(x)),
        "std_keV": float(np.std(x, ddof=1)),
        "median_keV": float(q[2]),
        "p02_5_keV": float(q[0]),
        "p16_keV": float(q[1]),
        "p84_keV": float(q[3]),
        "p97_5_keV": float(q[4]),
        "iqr_keV": float(np.percentile(x, 75.0) - np.percentile(x, 25.0)),
    }


def summary_table(
    df: pd.DataFrame,
    state: str,
    lif_thickness_um: float,
    surface_fraction: float,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    half_depth = 0.5 * lif_thickness_um
    left_cut = df["depth_um"].quantile(surface_fraction)
    right_cut = df["depth_um"].quantile(1.0 - surface_fraction)
    masks = {
        f"first_half_depth_lt_{half_depth:.3f}um": df["depth_um"] < half_depth,
        f"second_half_depth_ge_{half_depth:.3f}um": df["depth_um"] >= half_depth,
        f"left_surface_lowest_{surface_fraction:.0%}": df["depth_um"] <= left_cut,
        f"right_surface_highest_{surface_fraction:.0%}": df["depth_um"] >= right_cut,
    }
    rows: list[dict[str, float | int | str]] = []
    for group, mask in masks.items():
        depth_values = df.loc[mask, "depth_um"].to_numpy()
        rows.append(
            stats_row(
                state,
                group,
                "triton_truth_minus_target_exit",
                df.loc[mask, "triton_dE_target_keV"].to_numpy(),
                depth_values,
            ),
        )
        rows.append(
            stats_row(
                state,
                group,
                "triton_center_reco_minus_truth",
                df.loc[mask, "center_reco_minus_truth_keV"].to_numpy(),
                depth_values,
            ),
        )
    return pd.DataFrame(rows), masks


def equal_thickness_segment_masks(
    df: pd.DataFrame,
    lif_thickness_um: float,
    n_segments: int = 4,
) -> dict[str, pd.Series]:
    edges = np.linspace(0.0, lif_thickness_um, n_segments + 1)
    masks: dict[str, pd.Series] = {}
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        if idx == n_segments:
            mask = (df["depth_um"] >= lo) & (df["depth_um"] <= hi)
        else:
            mask = (df["depth_um"] >= lo) & (df["depth_um"] < hi)
        masks[f"segment_{idx}_{lo:.3f}_to_{hi:.3f}um"] = mask
    return masks


def percent_depth_segment_masks(
    df: pd.DataFrame,
    lif_thickness_um: float,
    percent_step: int = 5,
) -> dict[str, pd.Series]:
    if percent_step <= 0 or 100 % percent_step != 0:
        raise ValueError("percent_step must be a positive divisor of 100")
    n_segments = 100 // percent_step
    edges = np.linspace(0.0, lif_thickness_um, n_segments + 1)
    masks: dict[str, pd.Series] = {}
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        p_lo = idx * percent_step
        p_hi = (idx + 1) * percent_step
        if idx == n_segments - 1:
            mask = (df["depth_um"] >= lo) & (df["depth_um"] <= hi)
        else:
            mask = (df["depth_um"] >= lo) & (df["depth_um"] < hi)
        masks[f"depth_{p_lo:02d}_{p_hi:03d}pct_{lo:.3f}_to_{hi:.3f}um"] = mask
    return masks


def segment_summary_table(
    df: pd.DataFrame,
    state: str,
    masks: dict[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group, mask in masks.items():
        depth_values = df.loc[mask, "depth_um"].to_numpy()
        rows.append(
            stats_row(
                state,
                group,
                "triton_truth_minus_target_exit",
                df.loc[mask, "triton_dE_target_keV"].to_numpy(),
                depth_values,
            ),
        )
        rows.append(
            stats_row(
                state,
                group,
                "triton_center_reco_minus_truth",
                df.loc[mask, "center_reco_minus_truth_keV"].to_numpy(),
                depth_values,
            ),
        )
    return pd.DataFrame(rows)


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    return [float(x) for x in np.linspace(lo, hi, n)]


def draw_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    title: str,
    fonts: dict[str, ImageFont.ImageFont],
    y_note: str = "Counts",
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    plot = (left + 70, top + 52, right - 24, bottom - 58)
    px0, py0, px1, py1 = plot
    grid = (228, 232, 236)
    axis = (35, 39, 47)
    text = (35, 39, 47)

    draw.rectangle(plot, outline=(180, 186, 194), width=1)
    draw.text((left, top), title, font=fonts["title"], fill=text)
    draw.text((px0, top + 25), y_note, font=fonts["small"], fill=(82, 91, 105))
    draw.text(
        ((px0 + px1) // 2 - int(draw.textlength(xlabel, font=fonts["axis"]) / 2), bottom - 35),
        xlabel,
        font=fonts["axis"],
        fill=text,
    )

    for x in nice_ticks(*xlim, n=6):
        t = (x - xlim[0]) / (xlim[1] - xlim[0])
        px = int(px0 + t * (px1 - px0))
        draw.line([(px, py0), (px, py1)], fill=grid, width=1)
        label = f"{x:.0f}" if abs(x) >= 10 else f"{x:.1f}"
        draw.text(
            (px - int(draw.textlength(label, font=fonts["tick"]) / 2), py1 + 8),
            label,
            font=fonts["tick"],
            fill=text,
        )

    for y in nice_ticks(*ylim, n=5):
        t = (y - ylim[0]) / (ylim[1] - ylim[0])
        py = int(py1 - t * (py1 - py0))
        draw.line([(px0, py), (px1, py)], fill=grid, width=1)
        label = f"{y:.0f}" if y >= 10.0 else f"{y:.1f}"
        draw.text(
            (px0 - 10 - int(draw.textlength(label, font=fonts["tick"])), py - 8),
            label,
            font=fonts["tick"],
            fill=text,
        )

    draw.line([(px0, py1), (px1, py1)], fill=axis, width=2)
    draw.line([(px0, py0), (px0, py1)], fill=axis, width=2)
    return plot


def draw_step_hist(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    values: np.ndarray,
    bins: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    color: tuple[int, int, int],
    width: int = 3,
) -> None:
    px0, py0, px1, py1 = plot
    counts, edges = np.histogram(values, bins=bins)

    def mx(x: float) -> int:
        return int(px0 + (x - xlim[0]) / (xlim[1] - xlim[0]) * (px1 - px0))

    def my(y: float) -> int:
        return int(py1 - (y - ylim[0]) / (ylim[1] - ylim[0]) * (py1 - py0))

    baseline = my(0.0)
    pts: list[tuple[int, int]] = [(mx(edges[0]), baseline)]
    for i, y in enumerate(counts):
        pts.append((mx(edges[i]), my(float(y))))
        pts.append((mx(edges[i + 1]), my(float(y))))
    pts.append((mx(edges[-1]), baseline))
    draw.line(pts, fill=color, width=width, joint="curve")


def draw_step_counts(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    counts: np.ndarray,
    edges: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    color: tuple[int, int, int],
    width: int = 4,
) -> None:
    px0, py0, px1, py1 = plot

    def mx(x: float) -> int:
        return int(px0 + (x - xlim[0]) / (xlim[1] - xlim[0]) * (px1 - px0))

    def my(y: float) -> int:
        return int(py1 - (y - ylim[0]) / (ylim[1] - ylim[0]) * (py1 - py0))

    baseline = my(0.0)
    pts: list[tuple[int, int]] = [(mx(edges[0]), baseline)]
    for i, y in enumerate(counts):
        pts.append((mx(edges[i]), my(float(y))))
        pts.append((mx(edges[i + 1]), my(float(y))))
    pts.append((mx(edges[-1]), baseline))
    draw.line(pts, fill=color, width=width, joint="curve")


def hist_ylim(series: list[Series], bins: np.ndarray) -> tuple[float, float]:
    ymax = 0.0
    bin_width = bins[1] - bins[0]
    for item in series:
        counts, _ = np.histogram(item.values, bins=bins)
        ymax = max(ymax, float(np.max(counts)))
    return 0.0, ymax * 1.18 if ymax > 0.0 else 1.0


def draw_legend(
    draw: ImageDraw.ImageDraw,
    at: tuple[int, int],
    series: list[Series],
    fonts: dict[str, ImageFont.ImageFont],
    value_suffix: str = "keV",
) -> None:
    x, y = at
    for item in series:
        label = f"{item.name}: {count_stats_label(item.values, value_suffix)}"
        draw.line([(x, y + 10), (x + 36, y + 10)], fill=item.color, width=5)
        draw.text((x + 46, y), label, font=fonts["small"], fill=(35, 39, 47))
        y += 24


def draw_hist_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    xlabel: str,
    series: list[Series],
    bins: np.ndarray,
    xlim: tuple[float, float],
    fonts: dict[str, ImageFont.ImageFont],
    ylim: tuple[float, float] | None = None,
) -> None:
    if ylim is None:
        ylim = hist_ylim(series, bins)
    plot = draw_axes(draw, box, xlim, ylim, xlabel, title, fonts)
    for item in series:
        draw_step_hist(draw, plot, item.values, bins, xlim, ylim, item.color)
    draw_legend(draw, (plot[0] + 12, plot[1] + 12), series, fonts)


def draw_quantile_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    df: pd.DataFrame,
    lif_thickness_um: float,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    xlim = (0.0, lif_thickness_um)
    ylim = value_range(finite(df["triton_dE_target_keV"]), (0.0, 235.0))
    plot = draw_axes(
        draw,
        box,
        xlim,
        ylim,
        "reaction depth in LiF [um]",
        "Energy-loss band vs. depth",
        fonts,
        y_note="Target energy loss [keV]",
    )
    px0, py0, px1, py1 = plot

    def mx(x: float) -> int:
        return int(px0 + (x - xlim[0]) / (xlim[1] - xlim[0]) * (px1 - px0))

    def my(y: float) -> int:
        return int(py1 - (y - ylim[0]) / (ylim[1] - ylim[0]) * (py1 - py0))

    bins = np.linspace(0.0, lif_thickness_um, 25)
    xs: list[float] = []
    q16: list[float] = []
    q50: list[float] = []
    q84: list[float] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        vals = finite(
            df.loc[
                (df["depth_um"] >= lo) & (df["depth_um"] < hi),
                "triton_dE_target_keV",
            ],
        )
        if len(vals) < 20:
            continue
        xs.append(0.5 * (lo + hi))
        q16.append(float(np.percentile(vals, 16.0)))
        q50.append(float(np.percentile(vals, 50.0)))
        q84.append(float(np.percentile(vals, 84.0)))

    upper = [(mx(x), my(y)) for x, y in zip(xs, q84)]
    lower = [(mx(x), my(y)) for x, y in zip(reversed(xs), reversed(q16))]
    draw.polygon(upper + lower, fill=(210, 233, 238))
    draw.line([(mx(x), my(y)) for x, y in zip(xs, q84)], fill=(76, 160, 175), width=2)
    draw.line([(mx(x), my(y)) for x, y in zip(xs, q16)], fill=(76, 160, 175), width=2)
    draw.line([(mx(x), my(y)) for x, y in zip(xs, q50)], fill=(17, 94, 114), width=4)

    center_x = mx(0.5 * lif_thickness_um)
    draw.line([(center_x, py0), (center_x, py1)], fill=(115, 115, 115), width=2)
    draw.text(
        (center_x + 6, py0 + 8),
        "target center",
        font=fonts["small"],
        fill=(82, 91, 105),
    )
    draw.rectangle((px0 + 12, py0 + 12, px0 + 38, py0 + 28), fill=(210, 233, 238))
    draw.text(
        (px0 + 46, py0 + 8),
        "16-84% band, line = median",
        font=fonts["small"],
        fill=(35, 39, 47),
    )


def build_plot(
    df: pd.DataFrame,
    state: str,
    masks: dict[str, pd.Series],
    lif_thickness_um: float,
    surface_fraction: float,
    out_png: Path,
) -> None:
    fonts = {
        "title": load_font(24, bold=True),
        "axis": load_font(20),
        "tick": load_font(16),
        "small": load_font(17),
        "suptitle": load_font(30, bold=True),
    }

    canvas = Image.new("RGB", (1800, 1260), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (48, 32),
        f"{state}: S3-gated triton target energy loss by reaction depth",
        font=fonts["suptitle"],
        fill=(20, 24, 31),
    )
    draw.text(
        (48, 72),
        (
            f"LiF thickness = {lif_thickness_um:.2f} um; "
            f"near-surface gates = lowest/highest {surface_fraction:.0%} in depth"
        ),
        font=fonts["axis"],
        fill=(82, 91, 105),
    )

    blue = (37, 99, 235)
    red = (220, 38, 38)
    teal = (15, 118, 110)
    orange = (217, 119, 6)

    half_keys = list(masks.keys())[:2]
    surface_keys = list(masks.keys())[2:]
    half_series = [
        Series("first half", finite(df.loc[masks[half_keys[0]], "triton_dE_target_keV"]), blue),
        Series("second half", finite(df.loc[masks[half_keys[1]], "triton_dE_target_keV"]), red),
    ]
    surface_series = [
        Series(
            f"left surface ({surface_fraction:.0%})",
            finite(df.loc[masks[surface_keys[0]], "triton_dE_target_keV"]),
            teal,
        ),
        Series(
            f"right surface ({surface_fraction:.0%})",
            finite(df.loc[masks[surface_keys[1]], "triton_dE_target_keV"]),
            orange,
        ),
    ]
    residual_series = [
        Series(
            "first half",
            finite(df.loc[masks[half_keys[0]], "center_reco_minus_truth_keV"]),
            blue,
        ),
        Series(
            "second half",
            finite(df.loc[masks[half_keys[1]], "center_reco_minus_truth_keV"]),
            red,
        ),
    ]

    dE_bins, dE_xlim = bins_for_values(
        finite(df["triton_dE_target_keV"]),
        ENERGY_LOSS_BIN_WIDTH_KEV,
        (0.0, 235.0),
    )
    residual_bins, residual_xlim = bins_for_values(
        finite(df["center_reco_minus_truth_keV"]),
        RESIDUAL_BIN_WIDTH_KEV,
        (-170.0, 170.0),
    )
    panels = [
        (48, 130, 884, 610),
        (940, 130, 1776, 610),
        (48, 700, 884, 1180),
        (940, 700, 1776, 1180),
    ]
    draw_hist_panel(
        draw,
        panels[0],
        "First half vs. second half",
        "truth triton E - target-exit E [keV]",
        half_series,
        dE_bins,
        dE_xlim,
        fonts,
    )
    draw_hist_panel(
        draw,
        panels[1],
        "Near-surface extremes",
        "truth triton E - target-exit E [keV]",
        surface_series,
        dE_bins,
        dE_xlim,
        fonts,
    )
    draw_quantile_panel(draw, panels[2], df, lif_thickness_um, fonts)
    draw_hist_panel(
        draw,
        panels[3],
        "Center-depth correction residual",
        "reconstructed-center triton E - truth E [keV]",
        residual_series,
        residual_bins,
        residual_xlim,
        fonts,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    canvas.save(out_png.with_suffix(".pdf"))


def build_four_segment_plot(
    df: pd.DataFrame,
    state: str,
    masks: dict[str, pd.Series],
    lif_thickness_um: float,
    out_png: Path,
) -> None:
    fonts = {
        "title": load_font(24, bold=True),
        "axis": load_font(20),
        "tick": load_font(16),
        "small": load_font(17),
        "suptitle": load_font(30, bold=True),
    }

    canvas = Image.new("RGB", (1800, 1260), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (48, 32),
        f"{state}: triton target energy loss in four equal LiF depth segments",
        font=fonts["suptitle"],
        fill=(20, 24, 31),
    )
    draw.text(
        (48, 72),
        (
            f"Each segment is {lif_thickness_um / 4.0:.3f} um thick; "
            "histograms are S3-gated raw counts with common x/y scales"
        ),
        font=fonts["axis"],
        fill=(82, 91, 105),
    )

    colors = [
        (37, 99, 235),
        (15, 118, 110),
        (217, 119, 6),
        (220, 38, 38),
    ]
    edges = np.linspace(0.0, lif_thickness_um, len(masks) + 1)
    series: list[Series] = []
    for idx, ((_, mask), color) in enumerate(zip(masks.items(), colors), start=1):
        lo = edges[idx - 1]
        hi = edges[idx]
        label = f"segment {idx}: {lo:.3f}-{hi:.3f} um"
        series.append(Series(label, finite(df.loc[mask, "triton_dE_target_keV"]), color))

    bins, xlim = bins_for_values(
        finite_concat([item.values for item in series]),
        ENERGY_LOSS_BIN_WIDTH_KEV,
        (0.0, 235.0),
    )
    common_ylim = hist_ylim(series, bins)
    panels = [
        (48, 130, 884, 610),
        (940, 130, 1776, 610),
        (48, 700, 884, 1180),
        (940, 700, 1776, 1180),
    ]

    for panel, item in zip(panels, series):
        draw_hist_panel(
            draw,
            panel,
            item.name,
            "truth triton E - target-exit E [keV]",
            [item],
            bins,
            xlim,
            fonts,
            ylim=common_ylim,
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    canvas.save(out_png.with_suffix(".pdf"))


def draw_compact_hist_panel(
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
    grid = (228, 232, 236)
    axis = (35, 39, 47)
    text = (35, 39, 47)

    draw.text((left, top), title, font=fonts["title"], fill=text)
    draw.text((px0, top + 23), "Counts", font=fonts["small"], fill=(82, 91, 105))
    draw.rectangle((px0, py0, px1, py1), outline=(180, 186, 194), width=1)

    for x in nice_ticks(*xlim, n=6):
        t = (x - xlim[0]) / (xlim[1] - xlim[0])
        px = int(px0 + t * (px1 - px0))
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
        t = (y - ylim[0]) / (ylim[1] - ylim[0])
        py = int(py1 - t * (py1 - py0))
        draw.line([(px0, py), (px1, py)], fill=grid, width=1)
        label = f"{y:.0f}"
        draw.text(
            (px0 - 8 - int(draw.textlength(label, font=fonts["tick"])), py - 7),
            label,
            font=fonts["tick"],
            fill=text,
        )

    draw.line([(px0, py1), (px1, py1)], fill=axis, width=2)
    draw.line([(px0, py0), (px0, py1)], fill=axis, width=2)
    draw_step_hist(draw, (px0, py0, px1, py1), values, bins, xlim, ylim, color, width=3)

    label = count_stats_label(values)
    draw.line([(px0 + 10, py0 + 17), (px0 + 42, py0 + 17)], fill=color, width=5)
    draw.text((px0 + 50, py0 + 7), label, font=fonts["small"], fill=text)

    if show_xlabel:
        xlabel = "truth triton E - target-exit E [keV]"
        draw.text(
            ((px0 + px1) // 2 - int(draw.textlength(xlabel, font=fonts["axis"]) / 2), bottom - 22),
            xlabel,
            font=fonts["axis"],
            fill=text,
        )


def build_five_percent_segment_plot(
    df: pd.DataFrame,
    state: str,
    masks: dict[str, pd.Series],
    lif_thickness_um: float,
    out_png: Path,
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
        f"{state}: triton target energy loss for fixed LiF depth slices",
        font=fonts["suptitle"],
        fill=(20, 24, 31),
    )
    draw.text(
        (52, 75),
        (
            f"Each slice is {0.05 * lif_thickness_um:.3f} um thick; "
            "S3-gated raw counts, common x/y scales"
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
    depth_edges = np.linspace(0.0, lif_thickness_um, len(masks) + 1)
    series: list[Series] = []
    for idx, (_, mask) in enumerate(masks.items()):
        lo = depth_edges[idx]
        hi = depth_edges[idx + 1]
        label = f"{lo:.3f}-{hi:.3f} um"
        series.append(
            Series(
                label,
                finite(df.loc[mask, "triton_dE_target_keV"]),
                colors[idx % len(colors)],
            ),
        )

    bins, xlim = bins_for_values(
        finite_concat([item.values for item in series]),
        ENERGY_LOSS_BIN_WIDTH_KEV,
        (0.0, 235.0),
    )
    common_ylim = hist_ylim(series, bins)
    left_margin = 52
    top_margin = 126
    panel_w = 540
    panel_h = 548
    gap_x = 20
    gap_y = 20

    for idx, item in enumerate(series):
        row = idx // 4
        col = idx % 4
        left = left_margin + col * (panel_w + gap_x)
        top = top_margin + row * (panel_h + gap_y)
        right = left + panel_w
        bottom = top + panel_h
        show_xlabel = row == 4
        draw_compact_hist_panel(
            draw,
            (left, top, right, bottom),
            item.name,
            item.values,
            item.color,
            bins,
            xlim,
            common_ylim,
            fonts,
            show_xlabel,
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    canvas.save(out_png.with_suffix(".pdf"))


def build_five_percent_sum_plot(
    df: pd.DataFrame,
    state: str,
    masks: dict[str, pd.Series],
    lif_thickness_um: float,
    out_png: Path,
) -> pd.DataFrame:
    fonts = {
        "title": load_font(28, bold=True),
        "axis": load_font(20),
        "tick": load_font(16),
        "small": load_font(17),
        "suptitle": load_font(30, bold=True),
        "subtitle": load_font(20),
    }
    values_by_slice: list[np.ndarray] = []
    for mask in masks.values():
        values = finite(df.loc[mask, "triton_dE_target_keV"])
        values_by_slice.append(values)
    all_values = np.concatenate(values_by_slice) if values_by_slice else np.array([])

    bin_width = ENERGY_LOSS_BIN_WIDTH_KEV
    bins, xlim = bins_for_values(all_values, bin_width, (0.0, 235.0))
    summed_counts = np.zeros(len(bins) - 1, dtype=int)
    for values in values_by_slice:
        counts, _ = np.histogram(values, bins=bins)
        summed_counts += counts

    ylim = (0.0, float(np.max(summed_counts)) * 1.18 if np.max(summed_counts) > 0 else 1.0)

    canvas = Image.new("RGB", (1800, 1040), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (48, 32),
        f"{state}: summed triton target-loss distribution",
        font=fonts["suptitle"],
        fill=(20, 24, 31),
    )
    draw.text(
        (48, 72),
        (
            f"Sum of the 20 fixed 5% LiF-depth histograms; "
            f"each slice is {0.05 * lif_thickness_um:.3f} um thick; raw counts"
        ),
        font=fonts["subtitle"],
        fill=(82, 91, 105),
    )
    plot = draw_axes(
        draw,
        (64, 145, 1748, 920),
        xlim,
        ylim,
        "truth triton E - target-exit E [keV]",
        "Bin-by-bin sum of all 5% depth-slice distributions",
        fonts,
    )
    draw_step_counts(draw, plot, summed_counts, bins, xlim, ylim, (37, 99, 235))

    label = count_stats_label(all_values)
    draw.line([(plot[0] + 12, plot[1] + 18), (plot[0] + 50, plot[1] + 18)], fill=(37, 99, 235), width=6)
    draw.text((plot[0] + 62, plot[1] + 8), label, font=fonts["small"], fill=(35, 39, 47))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    canvas.save(out_png.with_suffix(".pdf"))

    hist = pd.DataFrame(
        {
            "bin_low_keV": bins[:-1],
            "bin_high_keV": bins[1:],
            "summed_counts": summed_counts,
        },
    )
    return hist


def build_five_percent_overlay_plot(
    df: pd.DataFrame,
    state: str,
    masks: dict[str, pd.Series],
    lif_thickness_um: float,
    out_png: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(masks)))
    depth_edges = np.linspace(0.0, lif_thickness_um, len(masks) + 1)
    series: list[Series] = []
    for idx, (_, mask) in enumerate(masks.items()):
        lo = depth_edges[idx]
        hi = depth_edges[idx + 1]
        label = f"{lo:.3f}-{hi:.3f} um"
        series.append(
            Series(
                label,
                finite(df.loc[mask, "triton_dE_target_keV"]),
                tuple((np.asarray(colors[idx][:3]) * 255).astype(int)),
            ),
        )

    all_values = finite_concat([item.values for item in series])
    bin_width = ENERGY_LOSS_BIN_WIDTH_KEV
    bins, xlim = bins_for_values(
        all_values,
        bin_width,
        (0.0, 235.0),
        visible_range=True,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.5, 8.0), dpi=200)
    for idx, item in enumerate(series):
        fwhm = histogram_fwhm(item.values, bins)
        fwhm_label = "n/a" if not np.isfinite(fwhm) else f"{fwhm:.1f}"
        ax.hist(
            item.values,
            bins=bins,
            histtype="step",
            linewidth=1.45,
            color=colors[idx],
            label=f"{item.name}, FWHM={fwhm_label} keV",
        )

    ax.set_title(
        f"State {state_latex_label(state)} triton target energy loss at different depths",
        fontsize=17,
        pad=14,
    )
    ax.set_xlabel("Energy loss (monte carlo) [keV]", fontsize=13)
    ax.set_ylabel("Counts", fontsize=13)
    ax.set_xlim(*xlim)
    ax.grid(True, color="#d7dce2", linewidth=0.8)
    ax.tick_params(axis="both", labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#424955")

    legend = ax.legend(
        title="Depth in LiF [um], FWHM",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8.0,
        title_fontsize=9.5,
        frameon=True,
        ncol=1,
    )
    legend.get_frame().set_edgecolor("#c6ccd3")
    legend.get_frame().set_linewidth(0.8)
    fig.subplots_adjust(right=0.73, left=0.07, bottom=0.11, top=0.90)
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def draw_trend_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    y_note: str,
    xlabel: str,
    x: np.ndarray,
    y: np.ndarray,
    color: tuple[int, int, int],
    fonts: dict[str, ImageFont.ImageFont],
    ylim: tuple[float, float],
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> None:
    xlim = (0.0, 100.0)
    plot = draw_axes(draw, box, xlim, ylim, xlabel, title, fonts, y_note=y_note)
    px0, py0, px1, py1 = plot

    def mx(xx: float) -> int:
        return int(px0 + (xx - xlim[0]) / (xlim[1] - xlim[0]) * (px1 - px0))

    def my(yy: float) -> int:
        return int(py1 - (yy - ylim[0]) / (ylim[1] - ylim[0]) * (py1 - py0))

    if lower is not None and upper is not None:
        band_valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(lower)
            & np.isfinite(upper)
        )
        x_band = x[band_valid]
        lower_band = lower[band_valid]
        upper_band = upper[band_valid]
        if len(x_band) >= 2:
            upper_pts = [
                (mx(float(xx)), my(float(yy))) for xx, yy in zip(x_band, upper_band)
            ]
            lower_pts = [
                (mx(float(xx)), my(float(yy)))
                for xx, yy in zip(x_band[::-1], lower_band[::-1])
            ]
            draw.polygon(upper_pts + lower_pts, fill=(210, 233, 238))
            draw.line(upper_pts, fill=(76, 160, 175), width=2)
            draw.line(
                [(mx(float(xx)), my(float(yy))) for xx, yy in zip(x_band, lower_band)],
                fill=(76, 160, 175),
                width=2,
            )
        draw.rectangle((px0 + 12, py0 + 12, px0 + 38, py0 + 28), fill=(210, 233, 238))
        draw.text(
            (px0 + 46, py0 + 8),
            "16-84% band",
            font=fonts["small"],
            fill=(35, 39, 47),
        )

    valid = np.isfinite(x) & np.isfinite(y)
    pts = [(mx(float(xx)), my(float(yy))) for xx, yy in zip(x[valid], y[valid])]
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=4)
    radius = 5
    for px, py in pts:
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)


def build_five_percent_trend_plot(
    summary: pd.DataFrame,
    state: str,
    out_png: Path,
    energy_loss_ylim: tuple[float, float],
) -> None:
    loss = summary[summary["quantity"] == "triton_truth_minus_target_exit"].copy()
    loss = loss.reset_index(drop=True)
    x = np.arange(len(loss), dtype=float) * 5.0 + 2.5
    mean = loss["mean_keV"].to_numpy(dtype=float)
    std = loss["std_keV"].to_numpy(dtype=float)
    p16 = loss["p16_keV"].to_numpy(dtype=float)
    p84 = loss["p84_keV"].to_numpy(dtype=float)
    finite_std = std[np.isfinite(std)]
    std_ylim_max = max(25.0, float(np.max(finite_std)) * 1.25) if len(finite_std) else 25.0

    fonts = {
        "title": load_font(24, bold=True),
        "axis": load_font(20),
        "tick": load_font(16),
        "small": load_font(17),
        "suptitle": load_font(30, bold=True),
        "subtitle": load_font(20),
    }
    canvas = Image.new("RGB", (1800, 1250), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (48, 32),
        f"{state}: 5% depth-slice energy-loss trends",
        font=fonts["suptitle"],
        fill=(20, 24, 31),
    )
    draw.text(
        (48, 72),
        "Points are slice centers; slices are 0-5%, 5-10%, ... 95-100% of LiF thickness.",
        font=fonts["subtitle"],
        fill=(82, 91, 105),
    )

    draw_trend_panel(
        draw,
        (48, 130, 1776, 640),
        "Mean target loss and central 68% band",
        "Target energy loss [keV]",
        "reaction depth percentile [%]",
        x,
        mean,
        (17, 94, 114),
        fonts,
        energy_loss_ylim,
        lower=p16,
        upper=p84,
    )
    draw_trend_panel(
        draw,
        (48, 735, 1776, 1165),
        "Width of each 5% slice",
        "Standard deviation [keV]",
        "reaction depth percentile [%]",
        x,
        std,
        (220, 38, 38),
        fonts,
        (0.0, std_ylim_max),
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    canvas.save(out_png.with_suffix(".pdf"))


def run_state(
    args: argparse.Namespace,
    state: str,
    project_root: Path,
    outdir: Path,
) -> None:
    df = read_joined(project_root, state, args.lif_thickness_um)
    five_percent_masks = percent_depth_segment_masks(df, args.lif_thickness_um, 5)
    five_percent_overlay_path = (
        outdir / f"{state}_triton_target_eloss_5percent_overlay_fine.png"
    )

    if args.overlay_only:
        build_five_percent_overlay_plot(
            df,
            state,
            five_percent_masks,
            args.lif_thickness_um,
            five_percent_overlay_path,
        )
        print(f"{state}: input S3-gated events: {len(df):,}")
        print(f"Saved {five_percent_overlay_path}")
        print(f"Saved {five_percent_overlay_path.with_suffix('.pdf')}")
        return

    summary, masks = summary_table(
        df,
        state,
        args.lif_thickness_um,
        args.surface_fraction,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / f"{state}_triton_target_eloss_by_depth_summary.csv"
    plot_path = outdir / f"{state}_triton_target_eloss_by_depth.png"
    segment_masks = equal_thickness_segment_masks(df, args.lif_thickness_um, 4)
    segment_summary = segment_summary_table(df, state, segment_masks)
    segment_summary_path = (
        outdir / f"{state}_triton_target_eloss_four_segments_summary.csv"
    )
    segment_plot_path = outdir / f"{state}_triton_target_eloss_four_segments.png"
    five_percent_summary = segment_summary_table(df, state, five_percent_masks)
    five_percent_summary_path = (
        outdir / f"{state}_triton_target_eloss_5percent_segments_summary.csv"
    )
    five_percent_plot_path = (
        outdir / f"{state}_triton_target_eloss_5percent_segments.png"
    )
    five_percent_trend_path = (
        outdir / f"{state}_triton_target_eloss_5percent_trends.png"
    )
    five_percent_sum_path = (
        outdir / f"{state}_triton_target_eloss_5percent_sum.png"
    )
    five_percent_sum_hist_path = (
        outdir / f"{state}_triton_target_eloss_5percent_sum_histogram.csv"
    )
    summary.to_csv(summary_path, index=False)
    segment_summary.to_csv(segment_summary_path, index=False)
    five_percent_summary.to_csv(five_percent_summary_path, index=False)
    build_plot(
        df,
        state,
        masks,
        args.lif_thickness_um,
        args.surface_fraction,
        plot_path,
    )
    build_four_segment_plot(
        df,
        state,
        segment_masks,
        args.lif_thickness_um,
        segment_plot_path,
    )
    build_five_percent_segment_plot(
        df,
        state,
        five_percent_masks,
        args.lif_thickness_um,
        five_percent_plot_path,
    )
    build_five_percent_trend_plot(
        five_percent_summary,
        state,
        five_percent_trend_path,
        value_range(finite(df["triton_dE_target_keV"]), (0.0, 235.0)),
    )
    five_percent_sum_hist = build_five_percent_sum_plot(
        df,
        state,
        five_percent_masks,
        args.lif_thickness_um,
        five_percent_sum_path,
    )
    five_percent_sum_hist.to_csv(five_percent_sum_hist_path, index=False)
    build_five_percent_overlay_plot(
        df,
        state,
        five_percent_masks,
        args.lif_thickness_um,
        five_percent_overlay_path,
    )

    print(f"Input S3-gated events: {len(df):,}")
    print(f"Saved {plot_path}")
    print(f"Saved {plot_path.with_suffix('.pdf')}")
    print(f"Saved {summary_path}")
    print(f"Saved {segment_plot_path}")
    print(f"Saved {segment_plot_path.with_suffix('.pdf')}")
    print(f"Saved {segment_summary_path}")
    print(f"Saved {five_percent_plot_path}")
    print(f"Saved {five_percent_plot_path.with_suffix('.pdf')}")
    print(f"Saved {five_percent_trend_path}")
    print(f"Saved {five_percent_trend_path.with_suffix('.pdf')}")
    print(f"Saved {five_percent_summary_path}")
    print(f"Saved {five_percent_sum_path}")
    print(f"Saved {five_percent_sum_path.with_suffix('.pdf')}")
    print(f"Saved {five_percent_sum_hist_path}")
    print(f"Saved {five_percent_overlay_path}")
    print(f"Saved {five_percent_overlay_path.with_suffix('.pdf')}")
    cols = ["group", "quantity", "N", "mean_keV", "std_keV", "p16_keV", "p84_keV"]
    print(summary[cols].to_string(index=False))
    print(segment_summary[cols].to_string(index=False))
    loss_rows = five_percent_summary[
        five_percent_summary["quantity"] == "triton_truth_minus_target_exit"
    ]
    print(loss_rows[cols].to_string(index=False))


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    if args.outdir is not None:
        outdir = args.outdir
    elif args.overlay_only:
        outdir = project_root / "output" / "target_depth_slice"
    else:
        outdir = project_root / "output" / "target_depth_eloss"

    if not 0.0 < args.surface_fraction < 0.5:
        raise ValueError("--surface-fraction must be between 0 and 0.5")

    states = discover_states(project_root) if args.all_states else [args.state]
    if not states:
        raise FileNotFoundError("No states with matching truth/exit kinematics CSVs found.")

    outdir.mkdir(parents=True, exist_ok=True)
    for idx, state in enumerate(states, start=1):
        if args.all_states:
            print(f"[{idx}/{len(states)}] Processing {state}")
        run_state(args, state, project_root, outdir)


if __name__ == "__main__":
    main()
