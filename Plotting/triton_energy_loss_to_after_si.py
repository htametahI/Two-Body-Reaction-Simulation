#!/usr/bin/env python3
"""Plot triton energy-loss distributions from reaction to the S3 measurement.

This is the standalone version of the ``triton_energy_loss_checks`` notebook
cell.  It keeps the existing output filenames while also writing fresh PNG
copies alongside the PDFs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-s2223")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib-s2223-xdg")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from target_depth_eloss import discover_states


AFTER_SI_CANDIDATES = [
    "triton_E_after_Si_MeV",
    "triton_E_after_Si_unsmeared_MeV",
    "triton_S3_edep_smeared_MeV",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update per-state triton energy-loss-to-after-Si plots.",
    )
    parser.add_argument(
        "--state",
        action="append",
        default=None,
        help="State to process. May be passed more than once. Defaults to all states.",
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
        help="Output directory. Defaults to output/triton_energy_loss_checks.",
    )
    parser.add_argument(
        "--xmax-percentile",
        type=float,
        default=99.5,
        help="Percentile used for the histogram upper x-limit.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=250,
        help="Number of histogram bins.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 11,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "legend.frameon": False,
        },
    )


def prettify(ax: plt.Axes) -> None:
    ax.minorticks_on()
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.20)
    ax.tick_params(which="both", top=False, right=False)


def select_after_si_column(exit_path: Path) -> str | None:
    columns = set(pd.read_csv(exit_path, nrows=0).columns)
    for column in AFTER_SI_CANDIDATES:
        if column in columns:
            return column
    return None


def energy_symbol(column: str) -> str:
    if column == "triton_S3_edep_smeared_MeV":
        return r"E_\mathrm{S3\ meas}"
    return r"E_\mathrm{after\ Si}"


def state_title(state: str) -> str:
    parts = state.split("_", 1)
    energy_label = parts[0]
    if energy_label.isdigit():
        energy = f"{int(energy_label) / 1000.0:.3f}"
    else:
        energy = energy_label.replace("_", r"\_")

    title = rf"$E_x = {energy}$ MeV"
    if len(parts) > 1:
        spin = parts[1]
        if spin.endswith("plus"):
            jpi = spin.removesuffix("plus") + r"^{+}"
        elif spin.endswith("minus"):
            jpi = spin.removesuffix("minus") + r"^{-}"
        else:
            jpi = spin.replace("_", r"\_")
        title += rf", $J^\pi = {jpi}$"
    return title + ", Triton Energy Loss Distributions"


def read_plot_data(project_root: Path, state: str) -> tuple[pd.DataFrame, str]:
    output_dir = project_root / "output"
    truth_path = output_dir / f"{state}_truth_kinematics.csv"
    exit_path = output_dir / f"{state}_exit_kinematics.csv"

    for path in [truth_path, exit_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    after_si_col = select_after_si_column(exit_path)
    if after_si_col is None:
        raise ValueError(
            f"{state}: no after-Si energy column found. "
            f"Tried: {', '.join(AFTER_SI_CANDIDATES)}",
        )

    truth_df = pd.read_csv(
        truth_path,
        usecols=["eventID", "triton_E_MeV"],
    ).rename(columns={"triton_E_MeV": "triton_E_truth_MeV"})
    exit_df = pd.read_csv(
        exit_path,
        usecols=[
            "eventID",
            "hasTritonExit",
            "triton_E_MeV",
            "triton_S3_ringID",
            after_si_col,
        ],
    )

    merged = exit_df.merge(
        truth_df,
        on="eventID",
        how="left",
        validate="many_to_one",
    )
    ring = pd.to_numeric(merged["triton_S3_ringID"], errors="coerce")
    mask = (
        (merged["hasTritonExit"] == 1)
        & ring.notna()
        & (ring >= 1)
        & merged["triton_E_truth_MeV"].notna()
        & merged["triton_E_MeV"].notna()
        & merged[after_si_col].notna()
    )
    plot_df = merged.loc[mask].copy()
    if len(plot_df) == 0:
        raise ValueError(f"{state}: no valid S3 triton events")

    e_reaction = plot_df["triton_E_truth_MeV"]
    e_target_exit = plot_df["triton_E_MeV"]
    e_after_si = plot_df[after_si_col]

    plot_df["dE_target_MeV"] = e_reaction - e_target_exit
    plot_df["dE_Si_MeV"] = e_target_exit - e_after_si
    plot_df["dE_reaction_to_after_Si_MeV"] = e_reaction - e_after_si
    return plot_df, after_si_col


def save_plot(
    fig: plt.Figure,
    output_base: Path,
) -> tuple[Path, Path]:
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    return png_path, pdf_path


def plot_state(
    project_root: Path,
    state: str,
    outdir: Path,
    bins_count: int,
    xmax_percentile: float,
) -> tuple[Path, Path, int, str]:
    plot_df, after_si_col = read_plot_data(project_root, state)
    e_after_symbol = energy_symbol(after_si_col)

    xmax = np.nanpercentile(
        plot_df["dE_reaction_to_after_Si_MeV"],
        xmax_percentile,
    )
    bins = np.linspace(0.0, xmax, bins_count)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        plot_df["dE_target_MeV"],
        bins=bins,
        histtype="step",
        linewidth=1.8,
        label=r"Target loss: $E_\mathrm{rxn} - E_\mathrm{target\ exit}$",
    )
    ax.hist(
        plot_df["dE_Si_MeV"],
        bins=bins,
        histtype="step",
        linewidth=1.8,
        label=rf"Si/S3 loss: $E_\mathrm{{target\ exit}} - {e_after_symbol}$",
    )
    ax.hist(
        plot_df["dE_reaction_to_after_Si_MeV"],
        bins=bins,
        histtype="stepfilled",
        alpha=0.35,
        color="#FE019A",
        edgecolor="black",
        linewidth=0.8,
        label=rf"Total: $E_\mathrm{{rxn}} - {e_after_symbol}$",
    )

    ax.set_xlabel(r"$\Delta E_t$ [MeV]")
    ax.set_ylabel("Counts")
    ax.set_title(state_title(state))
    ax.legend(fontsize=8)
    prettify(ax)

    outdir.mkdir(parents=True, exist_ok=True)
    output_base = outdir / f"{state}_triton_energy_loss_to_after_Si"
    png_path, pdf_path = save_plot(fig, output_base)
    plt.close(fig)
    return png_path, pdf_path, len(plot_df), after_si_col


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    project_root = args.project_root.resolve()
    outdir = args.outdir or (project_root / "output" / "triton_energy_loss_checks")
    states = args.state or discover_states(project_root)
    if not states:
        raise FileNotFoundError("No states with matching truth/exit kinematics CSVs found.")

    for state in states:
        png_path, pdf_path, count, column = plot_state(
            project_root,
            state,
            outdir,
            args.bins,
            args.xmax_percentile,
        )
        print(f"{state}: saved {png_path} and {pdf_path}; N={count:,}; energy={column}")


if __name__ == "__main__":
    main()
