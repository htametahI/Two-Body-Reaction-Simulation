#!/usr/bin/env python3
"""Independent SRIM-based triton energy-loss check for LiF target depth.

This script consumes a SRIM/SRModule stopping table for tritons in LiF, then
uses the simulation event kinematics only for initial triton energy, direction,
and reaction depth.  The energy propagation itself is a straight-line CSDA
calculation based on SRIM stopping powers.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from simulation_io import read_event_beam
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401


DEFAULT_STATE = "11336_1minus"
DEFAULT_LIF_THICKNESS_UM = 1.9


@dataclass
class Series:
    name: str
    values: np.ndarray
    color: tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Geant4 target loss with an independent SRIM CSDA calculation.",
    )
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--srim-table",
        type=Path,
        default=None,
        help="SRIM/SRModule table. Defaults to output/srim_lif/SR_Module/Triton_in_LiF_SRIM2013.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--lif-thickness-um",
        type=float,
        default=DEFAULT_LIF_THICKNESS_UM,
    )
    parser.add_argument(
        "--slice-bin-width-keV",
        type=float,
        default=1.0,
        help="Bin width for the SRIM 5%-slice histogram grid.",
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


def parse_energy_keV(value: str, unit: str) -> float:
    scale = {"eV": 1.0e-3, "keV": 1.0, "MeV": 1.0e3, "GeV": 1.0e6}
    return float(value) * scale[unit]


def parse_srim_table(path: Path) -> pd.DataFrame:
    rows: list[tuple[float, float, float]] = []
    pattern = re.compile(
        r"^\s*([0-9.]+)\s+(eV|keV|MeV|GeV)\s+"
        r"([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+",
    )
    for line in path.read_text(errors="ignore").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        energy_keV = parse_energy_keV(match.group(1), match.group(2))
        elec_keV_um = float(match.group(3))
        nuc_keV_um = float(match.group(4))
        rows.append((energy_keV, elec_keV_um, nuc_keV_um))

    if len(rows) < 5:
        raise ValueError(f"Could not parse enough SRIM rows from {path}")

    table = pd.DataFrame(
        rows,
        columns=["energy_keV", "elec_keV_per_um", "nuc_keV_per_um"],
    )
    table = table.drop_duplicates(subset=["energy_keV"]).sort_values("energy_keV")
    table["total_keV_per_um"] = (
        table["elec_keV_per_um"] + table["nuc_keV_per_um"]
    )
    return table


def make_range_interpolator(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    e_table = table["energy_keV"].to_numpy(dtype=float)
    s_table = table["total_keV_per_um"].to_numpy(dtype=float)
    e_grid = np.unique(
        np.r_[
            np.linspace(1.0, 20.0, 800),
            np.linspace(20.0, 300.0, 2200),
            np.linspace(300.0, 10000.0, 5000),
        ],
    )
    log_s = np.interp(np.log(e_grid), np.log(e_table), np.log(s_table))
    stopping = np.exp(log_s)
    inv_stopping = 1.0 / stopping
    d_e = np.diff(e_grid)
    # CSDA range R(E) = integral_0^E dE / S(E), in um.
    range_um = np.r_[0.0, np.cumsum(0.5 * (inv_stopping[:-1] + inv_stopping[1:]) * d_e)]
    return e_grid, range_um


def energy_after_path(
    e_in_keV: np.ndarray,
    path_um: np.ndarray,
    e_grid: np.ndarray,
    range_um: np.ndarray,
) -> np.ndarray:
    r_in = np.interp(e_in_keV, e_grid, range_um)
    r_out = np.maximum(0.0, r_in - path_um)
    return np.interp(r_out, range_um, e_grid)


def load_events(project_root: Path, state: str) -> pd.DataFrame:
    output_dir = project_root / "output"
    beam = read_event_beam(output_dir, state)

    truth = pd.read_csv(
        output_dir / f"{state}_truth_kinematics.csv",
        usecols=["eventID", "triton_E_MeV", "triton_theta_deg"],
    ).rename(
        columns={
            "triton_E_MeV": "triton_E_truth_MeV",
            "triton_theta_deg": "triton_theta_truth_deg",
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
        & df["depth_um"].notna()
    )
    df = df.loc[mask].copy()
    df["geant4_dE_target_keV"] = (
        df["triton_E_truth_MeV"] - df["triton_E_exit_MeV"]
    ) * 1000.0
    return df


def add_srim_loss(df: pd.DataFrame, e_grid: np.ndarray, range_um: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    cos_theta = np.abs(np.cos(np.deg2rad(out["triton_theta_truth_deg"].to_numpy())))
    cos_theta = np.maximum(cos_theta, 1.0e-6)
    path_um = out["depth_um"].to_numpy(dtype=float) / cos_theta
    e_in_keV = out["triton_E_truth_MeV"].to_numpy(dtype=float) * 1000.0
    e_out_keV = energy_after_path(e_in_keV, path_um, e_grid, range_um)
    out["srim_path_um"] = path_um
    out["srim_E_exit_keV"] = e_out_keV
    out["srim_dE_target_keV"] = e_in_keV - e_out_keV
    out["srim_minus_geant4_keV"] = (
        out["srim_dE_target_keV"] - out["geant4_dE_target_keV"]
    )
    return out


def five_percent_summary(
    df: pd.DataFrame,
    state: str,
    lif_thickness_um: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    edges = np.linspace(0.0, lif_thickness_um, 21)
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if idx == 19:
            mask = (df["depth_um"] >= lo) & (df["depth_um"] <= hi)
        else:
            mask = (df["depth_um"] >= lo) & (df["depth_um"] < hi)
        p_lo = idx * 5
        p_hi = p_lo + 5
        group = f"depth_{p_lo:02d}_{p_hi:03d}pct"
        for quantity, col in [
            ("geant4_truth_minus_exit", "geant4_dE_target_keV"),
            ("srim_csda", "srim_dE_target_keV"),
            ("srim_minus_geant4", "srim_minus_geant4_keV"),
        ]:
            vals = df.loc[mask, col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            q = np.percentile(vals, [16.0, 50.0, 84.0])
            rows.append(
                {
                    "state": state,
                    "group": group,
                    "depth_min_um": float(lo),
                    "depth_max_um": float(hi),
                    "quantity": quantity,
                    "N": int(len(vals)),
                    "mean_keV": float(np.mean(vals)),
                    "std_keV": float(np.std(vals, ddof=1)),
                    "median_keV": float(q[1]),
                    "p16_keV": float(q[0]),
                    "p84_keV": float(q[2]),
                },
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

    pts = [(mx(float(xx)), my(float(yy))) for xx, yy in zip(x, y)]
    draw.line(pts, fill=color, width=4)
    for px, py in pts:
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
    lx, ly = legend_at
    draw.line([(lx, ly + 10), (lx + 36, ly + 10)], fill=color, width=5)
    draw.text((lx + 46, ly), label, font=fonts["small"], fill=(35, 39, 47))


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


def build_comparison_plot(
    summary: pd.DataFrame,
    df: pd.DataFrame,
    state: str,
    out_png: Path,
) -> None:
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
        f"{state}: independent SRIM CSDA check for triton loss in LiF",
        font=fonts["suptitle"],
        fill=(20, 24, 31),
    )
    draw.text(
        (48, 72),
        "SRIM uses Triton in LiF stopping powers; path = reaction depth / |cos(theta_truth)|.",
        font=fonts["subtitle"],
        fill=(82, 91, 105),
    )

    loss_g4 = summary[summary["quantity"] == "geant4_truth_minus_exit"].reset_index(drop=True)
    loss_srim = summary[summary["quantity"] == "srim_csda"].reset_index(drop=True)
    diff = summary[summary["quantity"] == "srim_minus_geant4"].reset_index(drop=True)
    x = np.arange(len(loss_g4), dtype=float) * 5.0 + 2.5

    plot = draw_axes(
        draw,
        (48, 130, 1776, 600),
        (0.0, 100.0),
        (0.0, 235.0),
        "reaction depth percentile [%]",
        "Mean target energy loss by 5% depth slice",
        fonts,
        "Mean loss [keV]",
    )
    draw_line(
        draw,
        plot,
        (0.0, 100.0),
        (0.0, 235.0),
        x,
        loss_g4["mean_keV"].to_numpy(float),
        (37, 99, 235),
        "Geant4 truth - target exit",
        fonts,
        (plot[0] + 12, plot[1] + 12),
    )
    draw_line(
        draw,
        plot,
        (0.0, 100.0),
        (0.0, 235.0),
        x,
        loss_srim["mean_keV"].to_numpy(float),
        (220, 38, 38),
        "SRIM CSDA",
        fonts,
        (plot[0] + 12, plot[1] + 40),
    )

    max_abs = max(10.0, float(np.max(np.abs(diff["mean_keV"].to_numpy(float)))) * 1.3)
    plot = draw_axes(
        draw,
        (48, 690, 884, 1220),
        (0.0, 100.0),
        (-max_abs, max_abs),
        "reaction depth percentile [%]",
        "SRIM - Geant4 mean loss",
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

    bins = np.linspace(-35.0, 35.0, 100)
    residual = df["srim_minus_geant4_keV"].to_numpy(dtype=float)
    residual = residual[np.isfinite(residual)]
    counts, _ = np.histogram(residual, bins=bins)
    ylim = (0.0, float(np.max(counts)) * 1.18)
    plot = draw_axes(
        draw,
        (940, 690, 1776, 1220),
        (-35.0, 35.0),
        ylim,
        "SRIM loss - Geant4 loss [keV]",
        "Event-by-event residual",
        fonts,
        "Counts",
    )
    draw_step_hist(draw, plot, residual, bins, (-35.0, 35.0), ylim, (217, 119, 6))
    label = (
        f"N={len(residual):,}, mean={np.mean(residual):.2f} keV, "
        f"std={np.std(residual, ddof=1):.2f} keV"
    )
    draw.text((plot[0] + 12, plot[1] + 12), label, font=fonts["small"], fill=(35, 39, 47))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    canvas.save(out_png.with_suffix(".pdf"))


def draw_compact_srim_hist_panel(
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

    label = (
        f"N={len(values):,}, mean={np.mean(values):.1f} keV, "
        f"std={np.std(values, ddof=1):.1f} keV"
    )
    draw.line([(px0 + 10, py0 + 17), (px0 + 42, py0 + 17)], fill=color, width=5)
    draw.text((px0 + 50, py0 + 7), label, font=fonts["small"], fill=text)

    if show_xlabel:
        xlabel = "SRIM triton target energy loss [keV]"
        draw.text(
            ((px0 + px1) // 2 - int(draw.textlength(xlabel, font=fonts["axis"]) / 2), bottom - 22),
            xlabel,
            font=fonts["axis"],
            fill=text,
        )


def build_srim_five_percent_hist_plot(
    df: pd.DataFrame,
    state: str,
    lif_thickness_um: float,
    out_png: Path,
    bin_width_keV: float,
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
        f"{state}: SRIM CSDA triton loss for every 5% of LiF depth",
        font=fonts["suptitle"],
        fill=(20, 24, 31),
    )
    draw.text(
        (52, 75),
        (
            f"Each slice is {0.05 * lif_thickness_um:.3f} um thick; "
            f"S3-gated raw counts, {bin_width_keV:g} keV bins, common x/y scales"
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
    edges = np.linspace(0.0, lif_thickness_um, 21)
    slices: list[Series] = []
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if idx == 19:
            mask = (df["depth_um"] >= lo) & (df["depth_um"] <= hi)
        else:
            mask = (df["depth_um"] >= lo) & (df["depth_um"] < hi)
        values = df.loc[mask, "srim_dE_target_keV"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        slices.append(
            Series(
                f"{idx * 5:02d}-{(idx + 1) * 5:03d}% depth",
                values,
                colors[idx % len(colors)],
            ),
        )

    ymax = 0.0
    for item in slices:
        counts, _ = np.histogram(item.values, bins=bins)
        ymax = max(ymax, float(np.max(counts)))
    common_ylim = (0.0, ymax * 1.18 if ymax > 0.0 else 1.0)

    left_margin = 52
    top_margin = 126
    panel_w = 540
    panel_h = 548
    gap_x = 20
    gap_y = 20

    for idx, item in enumerate(slices):
        row = idx // 4
        col = idx % 4
        left = left_margin + col * (panel_w + gap_x)
        top = top_margin + row * (panel_h + gap_y)
        draw_compact_srim_hist_panel(
            draw,
            (left, top, left + panel_w, top + panel_h),
            item.name,
            item.values,
            item.color,
            bins,
            (0.0, 235.0),
            common_ylim,
            fonts,
            show_xlabel=row == 4,
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    canvas.save(out_png.with_suffix(".pdf"))


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    outdir = args.outdir or (project_root / "output" / "srim_lif")
    srim_table = args.srim_table or (
        project_root / "output" / "srim_lif" / "SR_Module" / "Triton_in_LiF_SRIM2013"
    )

    table = parse_srim_table(srim_table)
    e_grid, range_um = make_range_interpolator(table)
    events = load_events(project_root, args.state)
    events = add_srim_loss(events, e_grid, range_um)
    summary = five_percent_summary(events, args.state, args.lif_thickness_um)

    outdir.mkdir(parents=True, exist_ok=True)
    event_path = outdir / f"{args.state}_srim_vs_geant4_event_sample.csv"
    summary_path = outdir / f"{args.state}_srim_vs_geant4_5percent_summary.csv"
    plot_path = outdir / f"{args.state}_srim_vs_geant4_lif_loss.png"
    srim_slice_plot_path = outdir / f"{args.state}_srim_lif_loss_5percent_segments.png"

    columns = [
        "eventID",
        "depth_um",
        "triton_theta_truth_deg",
        "triton_E_truth_MeV",
        "geant4_dE_target_keV",
        "srim_path_um",
        "srim_dE_target_keV",
        "srim_minus_geant4_keV",
    ]
    events[columns].to_csv(event_path, index=False)
    summary.to_csv(summary_path, index=False)
    build_comparison_plot(summary, events, args.state, plot_path)
    build_srim_five_percent_hist_plot(
        events,
        args.state,
        args.lif_thickness_um,
        srim_slice_plot_path,
        args.slice_bin_width_keV,
    )

    print(f"Parsed SRIM table: {srim_table}")
    print(f"S3-gated events: {len(events):,}")
    print(f"Saved {event_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {plot_path}")
    print(f"Saved {plot_path.with_suffix('.pdf')}")
    print(f"Saved {srim_slice_plot_path}")
    print(f"Saved {srim_slice_plot_path.with_suffix('.pdf')}")

    cols = ["group", "quantity", "N", "mean_keV", "std_keV", "p16_keV", "p84_keV"]
    print(summary[summary["quantity"] != "srim_minus_geant4"][cols].to_string(index=False))
    print(summary[summary["quantity"] == "srim_minus_geant4"][cols].to_string(index=False))


if __name__ == "__main__":
    main()
