#!/usr/bin/env python3
"""Estimate reconstructed excitation-energy width budgets from event CSVs."""

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

from target_depth_eloss import bins_for_values, discover_states, histogram_fwhm, state_latex_label


U_MEV = 931.49410242
ELECTRON_MASS_MEV = 0.51099895000

# Atomic masses from AME/NIST-style neutral atom masses. Electrons cancel in the
# reaction Q value, but the kinematics use nuclear kinetic energies, so subtract
# Z electron masses for a close nuclear-mass approximation.
MASS_22NE = 21.991385114 * U_MEV - 10 * ELECTRON_MASS_MEV
MASS_7LI = 7.0160034366 * U_MEV - 3 * ELECTRON_MASS_MEV
MASS_26MG = 25.982592968 * U_MEV - 12 * ELECTRON_MASS_MEV
MASS_TRITON = 3.01604928199 * U_MEV - ELECTRON_MASS_MEV

S3_INNER_RADIUS_MM = 11.0
S3_OUTER_RADIUS_MM = 35.0
S3_DISTANCE_FROM_TARGET_CENTER_MM = 31.0
S3_RING_COUNT = 24

FULL_COMPONENT = "full_current_reconstruction"


@dataclass(frozen=True)
class WidthRow:
    state: str
    component: str
    n: int
    mean_keV: float
    std_keV: float
    fwhm_keV: float
    p16_keV: float
    median_keV: float
    p84_keV: float
    sigma68_keV: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate excitation-energy width contributions for selected states.",
    )
    parser.add_argument(
        "--states",
        nargs="*",
        default=["3588_0plus", "12345_0plus"],
        help="States to process. Defaults to the 3.588 and 12.345 MeV states.",
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Process every state with matching truth/exit kinematics files.",
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
        help="Output directory for plots and CSV summaries.",
    )
    parser.add_argument(
        "--bin-width-keV",
        type=float,
        default=5.0,
        help="Histogram bin width for excitation residual plots.",
    )
    return parser.parse_args()


def direction_from_angles(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(theta_deg.astype(float))
    phi = np.deg2rad(phi_deg.astype(float))
    return np.column_stack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ),
    )


def s3_ring_center_theta_deg(ring_id: np.ndarray) -> np.ndarray:
    ring_zero_based = ring_id.astype(float) - 1.0
    ring_width = (S3_OUTER_RADIUS_MM - S3_INNER_RADIUS_MM) / S3_RING_COUNT
    r_center = S3_INNER_RADIUS_MM + (ring_zero_based + 0.5) * ring_width
    theta = np.pi - np.arctan2(r_center, S3_DISTANCE_FROM_TARGET_CENTER_MM)
    return np.rad2deg(theta)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1)
    out = vectors.copy()
    valid = norm > 0.0
    out[valid] = out[valid] / norm[valid, None]
    out[~valid] = np.array([0.0, 0.0, 1.0])
    return out


def excitation_from_triton(
    triton_energy_mev: np.ndarray,
    triton_dir: np.ndarray,
    beam_energy_mev: np.ndarray,
    beam_dir: np.ndarray,
) -> np.ndarray:
    et = triton_energy_mev.astype(float)
    eb = beam_energy_mev.astype(float)
    beam_dir = normalize_vectors(beam_dir.astype(float))
    triton_dir = normalize_vectors(triton_dir.astype(float))

    beam_total_e = MASS_22NE + eb
    beam_p_mag = np.sqrt(np.maximum(beam_total_e * beam_total_e - MASS_22NE**2, 0.0))
    beam_p = beam_dir * beam_p_mag[:, None]

    triton_total_e = MASS_TRITON + et
    triton_p_mag = np.sqrt(
        np.maximum(triton_total_e * triton_total_e - MASS_TRITON**2, 0.0),
    )
    triton_p = triton_dir * triton_p_mag[:, None]

    missing_e = beam_total_e + MASS_7LI - triton_total_e
    missing_p = beam_p - triton_p
    missing_m2 = missing_e * missing_e - np.sum(missing_p * missing_p, axis=1)
    missing_m = np.sqrt(np.maximum(missing_m2, 0.0))
    return missing_m - MASS_26MG


def load_state(project_root: Path, state: str) -> pd.DataFrame:
    output_dir = project_root / "output"
    beam = read_event_beam(output_dir, state)

    truth = pd.read_csv(
        output_dir / f"{state}_truth_kinematics.csv",
        usecols=[
            "eventID",
            "Ex_MeV",
            "triton_E_MeV",
            "triton_theta_deg",
            "triton_phi_deg",
        ],
    ).rename(
        columns={
            "Ex_MeV": "Ex_truth_MeV",
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
            "triton_theta_deg",
            "triton_phi_deg",
            "triton_S3_ringID",
            "triton_S3_edep_smeared_MeV",
            "triton_E_reco_center_MeV",
            "beam_E_center_MeV",
        ],
    ).rename(
        columns={
            "triton_E_MeV": "triton_E_exit_MeV",
            "triton_theta_deg": "triton_theta_exit_deg",
            "triton_phi_deg": "triton_phi_exit_deg",
        },
    )

    df = exit_df.merge(truth, on="eventID", how="inner", validate="one_to_one").merge(
        beam[["eventID", "Ebeam_MeV", "dirx", "diry", "dirz", "depth_um"]],
        on="eventID",
        how="left",
        validate="one_to_one",
    )
    ring = pd.to_numeric(df["triton_S3_ringID"], errors="coerce")
    mask = (
        (df["hasTritonExit"] == 1)
        & (df["hasMg26Exit"] == 1)
        & ring.notna()
        & df["triton_E_reco_center_MeV"].notna()
        & df["beam_E_center_MeV"].notna()
        & df["triton_E_truth_MeV"].notna()
        & df["triton_theta_truth_deg"].notna()
        & df["triton_phi_truth_deg"].notna()
        & df["Ebeam_MeV"].notna()
    )
    return df.loc[mask].copy()


def component_residuals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    truth_dir = direction_from_angles(
        df["triton_theta_truth_deg"].to_numpy(float),
        df["triton_phi_truth_deg"].to_numpy(float),
    )
    ring_theta = s3_ring_center_theta_deg(df["triton_S3_ringID"].to_numpy(float))
    ring_dir = direction_from_angles(
        ring_theta,
        df["triton_phi_truth_deg"].to_numpy(float),
    )
    beam_dir = df[["dirx", "diry", "dirz"]].to_numpy(float)
    beam_z_dir = np.tile(np.array([0.0, 0.0, 1.0]), (len(df), 1))

    e_truth = df["triton_E_truth_MeV"].to_numpy(float)
    e_exit = df["triton_E_exit_MeV"].to_numpy(float)
    e_s3 = df["triton_S3_edep_smeared_MeV"].to_numpy(float)
    e_reco = df["triton_E_reco_center_MeV"].to_numpy(float)
    beam_reaction = df["Ebeam_MeV"].to_numpy(float)
    beam_center = df["beam_E_center_MeV"].to_numpy(float)

    ex_ref = excitation_from_triton(e_truth, truth_dir, beam_reaction, beam_dir)

    components = {
        "truth_reference": ex_ref - ex_ref,
        "target_exit_energy_uncorrected": (
            excitation_from_triton(e_exit, truth_dir, beam_reaction, beam_dir) - ex_ref
        ),
        "s3_smeared_energy_uncorrected": (
            excitation_from_triton(e_s3, truth_dir, beam_reaction, beam_dir) - ex_ref
        ),
        "reco_energy_only": (
            excitation_from_triton(e_reco, truth_dir, beam_reaction, beam_dir) - ex_ref
        ),
        "ring_angle_only": (
            excitation_from_triton(e_truth, ring_dir, beam_reaction, beam_dir) - ex_ref
        ),
        "beam_center_energy_only": (
            excitation_from_triton(e_truth, truth_dir, beam_center, beam_dir) - ex_ref
        ),
        "beam_axis_assumption_only": (
            excitation_from_triton(e_truth, truth_dir, beam_reaction, beam_z_dir) - ex_ref
        ),
        FULL_COMPONENT: (
            excitation_from_triton(e_reco, ring_dir, beam_center, beam_z_dir) - ex_ref
        ),
    }
    return {key: 1000.0 * value for key, value in components.items()}


def width_stats(state: str, component: str, values_keV: np.ndarray, bin_width: float) -> WidthRow:
    x = values_keV[np.isfinite(values_keV)]
    if len(x) == 0:
        return WidthRow(
            state,
            component,
            0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    bins, _ = bins_for_values(x, bin_width, (-500.0, 500.0), visible_range=True)
    p16, p50, p84 = np.percentile(x, [16.0, 50.0, 84.0])
    return WidthRow(
        state=state,
        component=component,
        n=int(len(x)),
        mean_keV=float(np.mean(x)),
        std_keV=float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        fwhm_keV=float(histogram_fwhm(x, bins)),
        p16_keV=float(p16),
        median_keV=float(p50),
        p84_keV=float(p84),
        sigma68_keV=float(0.5 * (p84 - p16)),
    )


def plot_state(
    state: str,
    residuals: dict[str, np.ndarray],
    rows: list[WidthRow],
    out_png: Path,
    bin_width: float,
) -> None:
    labels = {
        "reco_energy_only": "Reco triton energy only",
        "ring_angle_only": "Ring angle only",
        "beam_center_energy_only": "Beam center energy only",
        "beam_axis_assumption_only": "Beam direction ignored",
        FULL_COMPONENT: "Full current reconstruction",
    }
    colors = {
        "reco_energy_only": "#2563eb",
        "ring_angle_only": "#16a34a",
        "beam_center_energy_only": "#d97706",
        "beam_axis_assumption_only": "#7c3aed",
        FULL_COMPONENT: "#dc2626",
    }
    row_by_component = {row.component: row for row in rows}

    all_plot_values = np.concatenate([residuals[key] for key in labels])
    bins, xlim = bins_for_values(
        all_plot_values[np.isfinite(all_plot_values)],
        bin_width,
        (-500.0, 500.0),
        visible_range=True,
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=200)
    for key, label in labels.items():
        row = row_by_component[key]
        ax.hist(
            residuals[key],
            bins=bins,
            histtype="step",
            linewidth=1.5 if key != FULL_COMPONENT else 2.2,
            color=colors[key],
            label=f"{label}: sigma68={row.sigma68_keV:.1f} keV",
        )

    ax.axvline(0.0, color="#424955", linewidth=1.0, linestyle=":")
    ax.set_title(
        f"State {state_latex_label(state)} excitation-energy width budget",
        fontsize=16,
        pad=12,
    )
    ax.set_xlabel(r"$E_{x,\mathrm{component}} - E_{x,\mathrm{truthlike}}$ [keV]")
    ax.set_ylabel("Counts")
    ax.set_xlim(*xlim)
    ax.grid(True, color="#d7dce2", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    for spine in ax.spines.values():
        spine.set_color("#424955")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def process_state(
    project_root: Path,
    outdir: Path,
    state: str,
    bin_width: float,
) -> list[WidthRow]:
    df = load_state(project_root, state)
    residuals = component_residuals(df)
    rows = [
        width_stats(state, component, values, bin_width)
        for component, values in residuals.items()
    ]
    plot_state(
        state,
        residuals,
        rows,
        outdir / f"{state}_excitation_width_budget.png",
        bin_width,
    )
    full = next(row for row in rows if row.component == FULL_COMPONENT)
    print(
        f"{state}: full sigma68={full.sigma68_keV:.2f} keV, "
        f"std={full.std_keV:.2f} keV, FWHM={full.fwhm_keV:.2f} keV",
    )
    return rows


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    outdir = args.outdir or (project_root / "output" / "excitation_width_budget")
    outdir.mkdir(parents=True, exist_ok=True)
    states = discover_states(project_root) if args.all_states else args.states

    rows: list[WidthRow] = []
    for idx, state in enumerate(states, start=1):
        print(f"[{idx}/{len(states)}] Processing {state}")
        rows.extend(process_state(project_root, outdir, state, args.bin_width_keV))

    summary = pd.DataFrame([row.__dict__ for row in rows])
    summary_path = outdir / "excitation_width_budget_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
