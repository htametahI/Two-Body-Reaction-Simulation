#!/usr/bin/env python3
"""Overlay all-depth target-exit triton loss with fixed depth-slice losses."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-s2223")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib-s2223-xdg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from target_depth_eloss import (
    ENERGY_LOSS_BIN_WIDTH_KEV,
    LIF_THICKNESS_UM,
    bins_for_values,
    discover_states,
    finite,
    finite_concat,
    histogram_fwhm,
    percent_depth_segment_masks,
    read_joined,
)


TITLE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 14
NOTE_FONTSIZE = 12
LEGEND_FONTSIZE = 8.2
LEGEND_TITLE_FONTSIZE = 10.2


def apply_latex_font_style() -> None:
    """Use Matplotlib's bundled Computer Modern fonts without requiring LaTeX."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10", "Computer Modern Roman", "Latin Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "text.usetex": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the full target-exit triton energy-loss distribution on top "
            "of the fixed LiF depth-slice distributions."
        ),
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Optional single state. If omitted, all available states are processed.",
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
        help="Output directory for plots and summary CSV.",
    )
    parser.add_argument(
        "--percent-step",
        type=int,
        default=5,
        help="Depth-slice width in percent of the LiF target thickness.",
    )
    parser.add_argument(
        "--bin-width-keV",
        type=float,
        default=ENERGY_LOSS_BIN_WIDTH_KEV,
        help="Histogram bin width in keV.",
    )
    parser.add_argument(
        "--lif-thickness-um",
        type=float,
        default=LIF_THICKNESS_UM,
        help="LiF target thickness used to define fixed depth slices.",
    )
    parser.add_argument(
        "--raw-counts",
        action="store_true",
        help="Use raw counts instead of independently normalized histograms.",
    )
    return parser.parse_args()


def stats_row(
    state: str,
    group: str,
    values: np.ndarray,
    bin_width: float,
    depth_low_um: float,
    depth_high_um: float,
) -> dict[str, float | int | str]:
    x = finite(values)
    if len(x) == 0:
        return {
            "state": state,
            "group": group,
            "depth_low_um": depth_low_um,
            "depth_high_um": depth_high_um,
            "N": 0,
            "mean_keV": float("nan"),
            "std_keV": float("nan"),
            "median_keV": float("nan"),
            "p16_keV": float("nan"),
            "p84_keV": float("nan"),
            "fwhm_keV": float("nan"),
        }

    bins, _ = bins_for_values(
        x,
        bin_width,
        (0.0, 235.0),
        visible_range=True,
    )
    return {
        "state": state,
        "group": group,
        "depth_low_um": depth_low_um,
        "depth_high_um": depth_high_um,
        "N": int(len(x)),
        "mean_keV": float(np.mean(x)),
        "std_keV": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        "median_keV": float(np.median(x)),
        "p16_keV": float(np.percentile(x, 16.0)),
        "p84_keV": float(np.percentile(x, 84.0)),
        "fwhm_keV": float(histogram_fwhm(x, bins)),
    }


def finite_fwhm_label(values: np.ndarray, bins: np.ndarray) -> str:
    fwhm = histogram_fwhm(values, bins)
    if np.isfinite(fwhm):
        return f"{fwhm:.1f}"
    return "n/a"


def state_title(state: str) -> str:
    parts = state.split("_", 1)
    energy_label = parts[0]
    if energy_label.isdigit():
        energy = f"{int(energy_label) / 1000.0:.3f}"
    else:
        energy = energy_label.replace("_", r"\_")

    title = rf"State $E_x = {energy}$ MeV"
    if len(parts) > 1:
        spin = parts[1]
        if spin.endswith("plus"):
            jpi = spin.removesuffix("plus") + r"^{+}"
        elif spin.endswith("minus"):
            jpi = spin.removesuffix("minus") + r"^{-}"
        else:
            jpi = spin.replace("_", r"\_")
        title += rf", $J^\pi = {jpi}$"
    return title + ", Energy loss distributions"


def plot_state_overlay(
    df: pd.DataFrame,
    state: str,
    lif_thickness_um: float,
    percent_step: int,
    bin_width_keV: float,
    raw_counts: bool,
) -> tuple[plt.Figure, list[dict[str, float | int | str]]]:
    masks = percent_depth_segment_masks(df, lif_thickness_um, percent_step)
    depth_edges = np.linspace(0.0, lif_thickness_um, len(masks) + 1)
    slice_values: list[np.ndarray] = [
        finite(df.loc[mask, "triton_dE_target_keV"]) for mask in masks.values()
    ]
    overall_values = finite(df["triton_dE_target_keV"])
    all_values = finite_concat([overall_values, *slice_values])
    bins, xlim = bins_for_values(
        all_values,
        bin_width_keV,
        (0.0, 235.0),
        visible_range=True,
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    apply_latex_font_style()
    fig, ax = plt.subplots(figsize=(17.0, 9.8), dpi=200)
    density = not raw_counts
    ylabel = "Counts" if raw_counts else "Normalized probability density [1/keV]"

    ax.hist(
        overall_values,
        bins=bins,
        histtype="stepfilled",
        color="#111827",
        alpha=0.10,
        density=density,
    )
    ax.hist(
        overall_values,
        bins=bins,
        histtype="step",
        color="#111827",
        linewidth=2.6,
        density=density,
        label=(
            f"overall 0.000-{lif_thickness_um:.3f} um, "
            f"FWHM={finite_fwhm_label(overall_values, bins)} keV"
        ),
    )

    empty_slices = sum(1 for values in slice_values if len(values) == 0)
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(slice_values)))
    for idx, values in enumerate(slice_values):
        if len(values) == 0:
            continue
        lo = depth_edges[idx]
        hi = depth_edges[idx + 1]
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.15,
            color=colors[idx],
            alpha=0.90,
            density=density,
            label=(
                f"{lo:.3f}-{hi:.3f} um, "
                f"FWHM={finite_fwhm_label(values, bins)} keV"
            ),
        )

    ax.set_title(state_title(state), fontsize=TITLE_FONTSIZE, pad=16)
    ax.set_xlabel(r"$E_{\mathrm{loss}}$ [keV]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlim(*xlim)
    ax.grid(True, color="#d7dce2", linewidth=0.8)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_color("#424955")

    note = (
        "S3-gated events; histograms normalized independently"
        if density
        else "S3-gated events; raw counts"
    )
    if empty_slices:
        note += f"; {empty_slices} empty slice(s) omitted"
    ax.text(
        0.01,
        0.985,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=NOTE_FONTSIZE,
        color="#475569",
    )

    legend = ax.legend(
        title="Depth in LiF [um], FWHM",
        bbox_to_anchor=(0.995, 0.995),
        loc="upper right",
        borderaxespad=0.25,
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        frameon=True,
        framealpha=0.94,
        handlelength=1.75,
        handletextpad=0.5,
        labelspacing=0.26,
        borderpad=0.45,
        ncol=1,
    )
    legend.get_frame().set_edgecolor("#c6ccd3")
    legend.get_frame().set_linewidth(0.8)
    fig.subplots_adjust(right=0.985, left=0.085, bottom=0.125, top=0.885)

    rows = [
        stats_row(
            state,
            "overall_0_100pct",
            overall_values,
            bin_width_keV,
            0.0,
            lif_thickness_um,
        ),
    ]
    for idx, values in enumerate(slice_values):
        rows.append(
            stats_row(
                state,
                f"slice_{idx * percent_step:02d}_{(idx + 1) * percent_step:03d}pct",
                values,
                bin_width_keV,
                float(depth_edges[idx]),
                float(depth_edges[idx + 1]),
            ),
        )
    return fig, rows


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    outdir = args.outdir or (
        project_root / "output" / "target_exit_eloss_overlay_all_states"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    states = [args.state] if args.state else discover_states(project_root)
    if not states:
        raise FileNotFoundError("No states with matching truth/exit kinematics CSVs found.")

    mode_suffix = "raw_counts" if args.raw_counts else "normalized"
    multi_pdf = (
        outdir
        / f"all_states_triton_target_eloss_{args.percent_step}percent_slices_with_overall_{mode_suffix}.pdf"
    )
    summary_rows: list[dict[str, float | int | str]] = []

    with PdfPages(multi_pdf) as pdf:
        for idx, state in enumerate(states, start=1):
            print(f"[{idx}/{len(states)}] Processing {state}")
            df = read_joined(project_root, state, args.lif_thickness_um)
            fig, rows = plot_state_overlay(
                df,
                state,
                args.lif_thickness_um,
                args.percent_step,
                args.bin_width_keV,
                args.raw_counts,
            )
            out_png = (
                outdir
                / f"{state}_triton_target_eloss_{args.percent_step}percent_slices_with_overall_{mode_suffix}.png"
            )
            fig.savefig(out_png)
            fig.savefig(out_png.with_suffix(".pdf"))
            pdf.savefig(fig)
            plt.close(fig)
            summary_rows.extend(rows)
            print(f"Saved {out_png}")

    summary_path = (
        outdir
        / f"all_states_triton_target_eloss_{args.percent_step}percent_slices_with_overall_{mode_suffix}_summary.csv"
    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved {multi_pdf}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
