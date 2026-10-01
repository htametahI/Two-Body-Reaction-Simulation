#!/usr/bin/env python3
"""Plot per-state triton energy-loss correction distributions."""

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

from target_depth_eloss import (
    ENERGY_LOSS_BIN_WIDTH_KEV,
    LIF_THICKNESS_UM,
    bins_for_values,
    discover_states,
    finite,
    histogram_fwhm,
    read_joined,
    state_latex_label,
)


BEAM_INITIAL_ENERGY_MEV = 66.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot energy-loss correction distributions for each state.",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Optional single state. If omitted, all states are processed.",
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
        help="Output directory for plots and summary CSVs.",
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
        help="LiF thickness used when joining event depth information.",
    )
    return parser.parse_args()


def add_correction_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["triton_applied_center_correction_keV"] = (
        out["triton_E_reco_center_MeV"] - out["triton_S3_edep_smeared_MeV"]
    ) * 1000.0
    out["beam_loss_to_center_keV"] = (
        BEAM_INITIAL_ENERGY_MEV - out["beam_E_center_MeV"]
    ) * 1000.0
    return out


def stats_for_quantity(
    state: str,
    quantity: str,
    values: np.ndarray,
    bin_width: float,
) -> dict[str, float | int | str]:
    x = finite(values)
    if len(x) == 0:
        return {
            "state": state,
            "quantity": quantity,
            "N": 0,
            "mean_keV": float("nan"),
            "std_keV": float("nan"),
            "median_keV": float("nan"),
            "p16_keV": float("nan"),
            "p84_keV": float("nan"),
            "fwhm_keV": float("nan"),
        }

    bins, _ = bins_for_values(x, bin_width, (0.0, 250.0), visible_range=True)
    return {
        "state": state,
        "quantity": quantity,
        "N": int(len(x)),
        "mean_keV": float(np.mean(x)),
        "std_keV": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        "median_keV": float(np.median(x)),
        "p16_keV": float(np.percentile(x, 16.0)),
        "p84_keV": float(np.percentile(x, 84.0)),
        "fwhm_keV": float(histogram_fwhm(x, bins)),
    }


def stats_label(values: np.ndarray, bins: np.ndarray) -> str:
    x = finite(values)
    if len(x) == 0:
        return "N=0"
    return f"N={len(x):,}, mean={np.mean(x):.1f} keV"


def plot_state(
    df: pd.DataFrame,
    state: str,
    out_png: Path,
    bin_width: float,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    panels = [
        (
            "Applied center-depth triton correction",
            "triton_applied_center_correction_keV",
            r"$E_{t,\mathrm{reco\ center}} - E_{t,\mathrm{S3\ smeared}}$ [keV]",
            "#2563eb",
        ),
        (
            "True triton target loss to exit",
            "triton_dE_target_keV",
            r"$E_{t,\mathrm{truth}} - E_{t,\mathrm{target\ exit}}$ [keV]",
            "#dc2626",
        ),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), dpi=200)
    for ax, (title, column, xlabel, color) in zip(axes, panels):
        values = finite(df[column])
        bins, xlim = bins_for_values(
            values,
            bin_width,
            (0.0, 250.0),
            visible_range=True,
        )
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.5,
            color=color,
            label=stats_label(values, bins),
        )
        if len(values) > 0:
            ax.axvline(np.mean(values), color=color, linestyle="--", linewidth=1.1)
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Counts", fontsize=11)
        ax.set_xlim(*xlim)
        ax.grid(True, color="#d7dce2", linewidth=0.8)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(loc="upper right", fontsize=9, frameon=True)
        for spine in ax.spines.values():
            spine.set_color("#424955")

    fig.suptitle(
        f"State {state_latex_label(state)} energy-loss corrections",
        fontsize=16,
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def run_state(
    project_root: Path,
    outdir: Path,
    state: str,
    bin_width: float,
    lif_thickness_um: float,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    df = add_correction_columns(read_joined(project_root, state, lif_thickness_um))
    out_png = outdir / f"{state}_energy_loss_correction_distribution.png"
    plot_state(df, state, out_png, bin_width)

    quantity_rows = [
        stats_for_quantity(
            state,
            "applied_triton_center_correction",
            df["triton_applied_center_correction_keV"].to_numpy(),
            bin_width,
        ),
        stats_for_quantity(
            state,
            "true_triton_target_loss_to_exit",
            df["triton_dE_target_keV"].to_numpy(),
            bin_width,
        ),
        stats_for_quantity(
            state,
            "center_reco_minus_truth",
            df["center_reco_minus_truth_keV"].to_numpy(),
            bin_width,
        ),
        stats_for_quantity(
            state,
            "beam_loss_to_center",
            df["beam_loss_to_center_keV"].to_numpy(),
            bin_width,
        ),
    ]

    mean_row = {
        "state": state,
        "N": int(len(df)),
        "mean_applied_triton_center_correction_keV": float(
            np.mean(finite(df["triton_applied_center_correction_keV"])),
        ),
        "std_applied_triton_center_correction_keV": float(
            np.std(finite(df["triton_applied_center_correction_keV"]), ddof=1),
        ),
        "mean_true_triton_target_loss_to_exit_keV": float(
            np.mean(finite(df["triton_dE_target_keV"])),
        ),
        "mean_center_reco_minus_truth_keV": float(
            np.mean(finite(df["center_reco_minus_truth_keV"])),
        ),
        "mean_beam_loss_to_center_keV": float(
            np.mean(finite(df["beam_loss_to_center_keV"])),
        ),
    }
    print(
        f"{state}: mean applied triton correction = "
        f"{mean_row['mean_applied_triton_center_correction_keV']:.3f} keV; "
        f"plot = {out_png}"
    )
    return quantity_rows, mean_row


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    outdir = args.outdir or (project_root / "output" / "state_energy_loss")
    outdir.mkdir(parents=True, exist_ok=True)

    states = [args.state] if args.state else discover_states(project_root)
    if not states:
        raise FileNotFoundError("No states with matching truth/exit kinematics CSVs found.")

    quantity_rows: list[dict[str, float | int | str]] = []
    mean_rows: list[dict[str, float | int | str]] = []
    for idx, state in enumerate(states, start=1):
        print(f"[{idx}/{len(states)}] Processing {state}")
        q_rows, mean_row = run_state(
            project_root,
            outdir,
            state,
            args.bin_width_keV,
            args.lif_thickness_um,
        )
        quantity_rows.extend(q_rows)
        mean_rows.append(mean_row)

    quantity_summary = pd.DataFrame(quantity_rows)
    mean_summary = pd.DataFrame(mean_rows)
    quantity_path = outdir / "state_energy_loss_quantity_summary.csv"
    mean_path = outdir / "mean_energy_loss_corrections_by_state.csv"
    quantity_summary.to_csv(quantity_path, index=False)
    mean_summary.to_csv(mean_path, index=False)
    print(f"Saved {quantity_path}")
    print(f"Saved {mean_path}")


if __name__ == "__main__":
    main()
