#!/usr/bin/env python3
"""Preliminary differential cross sections from ring-chunk MCMC yields."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
DEFAULT_FIT_DIR = OUT / "MCMC_full_excitation_range_inputHist" / "fit_outputs"
DEFAULT_OUTDIR = OUT / "preliminary_differential_cross_sections"

MASS_22NE_MEV = 20484.845566
MASS_7LI_MEV = 6535.365833
MASS_TRITON_MEV = 2809.432119
MASS_26MG_MEV = 24202.632149
AVOGADRO = 6.02214076e23
MOLAR_MASS_7LIF_G_MOL = 7.0160034366 + 18.99840316273
MOLAR_MASS_7LI_G_MOL = 7.0160034366

S3_INNER_RADIUS_MM = 11.0
S3_OUTER_RADIUS_MM = 35.0
S3_RING_COUNT = 24
S3_DISTANCE_MM = 31.0

RING_GROUPS = {
    "RR0": (1, 4),
    "RR1": (5, 8),
    "RR2": (9, 12),
    "RR3": (13, 15),
    "RR4": (19, 24),
}


mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RR0-RR4 MCMC yields into a preliminary dσ/dΩ estimate "
            "using assumed beam normalization and geometric CM solid angles."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--beam-particles", type=float, default=1.0e12)
    parser.add_argument("--beam-energy-MeV", type=float, default=None)
    parser.add_argument("--beam-file", type=Path, default=OUT / "beam.dat")
    parser.add_argument("--target-areal-density-ug-cm2", type=float, default=500.0)
    parser.add_argument(
        "--target-material",
        choices=("7LiF", "7Li"),
        default="7LiF",
        help=(
            "Interpret the target mass thickness as either 7LiF compound or "
            "pure 7Li areal mass. 7LiF gives one 7Li nucleus per molecule."
        ),
    )
    return parser.parse_args()


def beam_energy_from_file(path: Path) -> float:
    values = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=["E_MeV", "x_mm", "y_mm", "z_mm", "dirx", "diry", "dirz", "depth_um"],
        usecols=["E_MeV"],
    )["E_MeV"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError(f"No beam energies found in {path}")
    return float(np.mean(values))


def target_nuclei_per_cm2(
    target_areal_density_ug_cm2: float,
    target_material: str,
) -> float:
    areal_g_cm2 = target_areal_density_ug_cm2 * 1.0e-6
    if target_material == "7LiF":
        return areal_g_cm2 / MOLAR_MASS_7LIF_G_MOL * AVOGADRO
    if target_material == "7Li":
        return areal_g_cm2 / MOLAR_MASS_7LI_G_MOL * AVOGADRO
    raise ValueError(f"Unsupported target material {target_material!r}")


def ring_radius_edge_mm(ring_edge_id: float) -> float:
    ring_width = (S3_OUTER_RADIUS_MM - S3_INNER_RADIUS_MM) / S3_RING_COUNT
    return S3_INNER_RADIUS_MM + ring_edge_id * ring_width


def theta_lab_from_radius_deg(radius_mm: float) -> float:
    return math.degrees(math.pi - math.atan2(radius_mm, S3_DISTANCE_MM))


def theta_lab_bounds_for_group(ring_lo: int, ring_hi: int) -> tuple[float, float]:
    inner_radius = ring_radius_edge_mm(ring_lo - 1)
    outer_radius = ring_radius_edge_mm(ring_hi)
    theta_high = theta_lab_from_radius_deg(inner_radius)
    theta_low = theta_lab_from_radius_deg(outer_radius)
    return theta_low, theta_high


def theta_lab_center_for_group(ring_lo: int, ring_hi: int) -> float:
    center_radius = 0.5 * (
        ring_radius_edge_mm(ring_lo - 1) + ring_radius_edge_mm(ring_hi)
    )
    return theta_lab_from_radius_deg(center_radius)


def theta_triton_lab_from_forward_cm_deg(
    theta_forward_cm_deg: float,
    excitation_MeV: float,
    beam_energy_MeV: float,
) -> float:
    theta = math.radians(theta_forward_cm_deg)
    beam_total_energy = MASS_22NE_MEV + beam_energy_MeV
    beam_p = math.sqrt(max(beam_total_energy**2 - MASS_22NE_MEV**2, 0.0))
    total_lab_energy = beam_total_energy + MASS_7LI_MEV
    beta = beam_p / total_lab_energy
    gamma = 1.0 / math.sqrt(1.0 - beta**2)
    s = (
        MASS_22NE_MEV**2
        + MASS_7LI_MEV**2
        + 2.0 * MASS_7LI_MEV * beam_total_energy
    )
    sqrt_s = math.sqrt(s)
    mass_residual = MASS_26MG_MEV + excitation_MeV
    lam = (s - (mass_residual + MASS_TRITON_MEV) ** 2) * (
        s - (mass_residual - MASS_TRITON_MEV) ** 2
    )
    if lam <= 0.0:
        return float("nan")
    p_cm = math.sqrt(lam) / (2.0 * sqrt_s)
    e_triton_cm = (s + MASS_TRITON_MEV**2 - mass_residual**2) / (2.0 * sqrt_s)
    # The plotted forward CM angle is the residual angle. The triton is opposite
    # the residual in the CM frame.
    pz_lab = gamma * (-p_cm * math.cos(theta) + beta * e_triton_cm)
    pt_lab = p_cm * math.sin(theta)
    return math.degrees(math.atan2(pt_lab, pz_lab))


def theta_forward_cm_from_triton_lab_deg(
    theta_lab_deg: float,
    excitation_MeV: float,
    beam_energy_MeV: float,
) -> float:
    result = minimize_scalar(
        lambda theta_cm: (
            theta_triton_lab_from_forward_cm_deg(
                theta_cm,
                excitation_MeV,
                beam_energy_MeV,
            )
            - theta_lab_deg
        )
        ** 2,
        bounds=(0.0, 80.0),
        method="bounded",
    )
    return float(result.x)


def cm_geometry_for_group(
    label: str,
    excitation_MeV: float,
    beam_energy_MeV: float,
) -> dict[str, float]:
    ring_lo, ring_hi = RING_GROUPS[label]
    theta_lab_low, theta_lab_high = theta_lab_bounds_for_group(ring_lo, ring_hi)
    theta_lab_center = theta_lab_center_for_group(ring_lo, ring_hi)
    theta_cm_edges = [
        theta_forward_cm_from_triton_lab_deg(theta_lab_low, excitation_MeV, beam_energy_MeV),
        theta_forward_cm_from_triton_lab_deg(theta_lab_high, excitation_MeV, beam_energy_MeV),
    ]
    theta_cm_low, theta_cm_high = sorted(theta_cm_edges)
    theta_cm_center = theta_forward_cm_from_triton_lab_deg(
        theta_lab_center,
        excitation_MeV,
        beam_energy_MeV,
    )
    delta_omega_cm = 2.0 * math.pi * abs(
        math.cos(math.radians(theta_cm_low)) - math.cos(math.radians(theta_cm_high))
    )
    delta_omega_lab = 2.0 * math.pi * abs(
        math.cos(math.radians(theta_lab_low)) - math.cos(math.radians(theta_lab_high))
    )
    return {
        "ring_lo": ring_lo,
        "ring_hi": ring_hi,
        "theta_lab_low_deg": theta_lab_low,
        "theta_lab_high_deg": theta_lab_high,
        "theta_lab_center_deg": theta_lab_center,
        "theta_cm_low_deg": theta_cm_low,
        "theta_cm_high_deg": theta_cm_high,
        "theta_cm_center_deg": theta_cm_center,
        "delta_omega_lab_sr": delta_omega_lab,
        "delta_omega_cm_sr": delta_omega_cm,
    }


def load_ring_yields(fit_dir: Path) -> pd.DataFrame:
    rows = []
    for label in RING_GROUPS:
        path = fit_dir / label / "MCMC_full_range_fit_summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df = df[df["input_Ex_MeV"].notna()].copy()
        df["ring_group"] = label
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def build_cross_section_table(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, float]]:
    beam_energy = (
        float(args.beam_energy_MeV)
        if args.beam_energy_MeV is not None
        else beam_energy_from_file(args.beam_file)
    )
    target_areal = target_nuclei_per_cm2(
        args.target_areal_density_ug_cm2,
        args.target_material,
    )
    yields = load_ring_yields(args.fit_dir)
    rows = []
    for row in yields.itertuples(index=False):
        ex = float(row.input_Ex_MeV)
        geom = cm_geometry_for_group(str(row.ring_group), ex, beam_energy)
        denom = args.beam_particles * target_areal * geom["delta_omega_cm_sr"]
        scale = 1.0e27 / denom
        dsdo = float(row.MCMC_N_median) * scale
        rows.append(
            {
                "component": row.component,
                "input_Ex_MeV": ex,
                "Jpi": row.Jpi,
                "ring_group": row.ring_group,
                **geom,
                "beam_energy_MeV": beam_energy,
                "assumed_beam_particles": args.beam_particles,
                "target_material": args.target_material,
                "target_areal_density_ug_cm2": args.target_areal_density_ug_cm2,
                "target_nuclei_per_cm2": target_areal,
                "yield_MCMC_N": float(row.MCMC_N_median),
                "yield_minus": float(row.MCMC_minus),
                "yield_plus": float(row.MCMC_plus),
                "dsigma_domega_mb_sr": dsdo,
                "dsigma_domega_minus_mb_sr": float(row.MCMC_minus) * scale,
                "dsigma_domega_plus_mb_sr": float(row.MCMC_plus) * scale,
            }
        )
    meta = {
        "beam_energy_MeV": beam_energy,
        "target_nuclei_per_cm2": target_areal,
    }
    return pd.DataFrame(rows), meta


def component_label(component: str, ex: float, jpi: str) -> str:
    return rf"{ex:.3f} MeV, $J^\pi={jpi}$"


def component_filename(component: str, ex: float) -> str:
    ex_key = f"{ex:.3f}".replace(".", "p")
    return f"Ex_{ex_key}_MeV_{component}"


def plot_all_states(df: pd.DataFrame, outdir: Path, *, logy: bool) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    components = (
        df[["component", "input_Ex_MeV", "Jpi"]]
        .drop_duplicates()
        .sort_values("input_Ex_MeV")
    )
    cmap = mpl.colormaps["turbo"].resampled(max(2, len(components)))
    for color_index, comp in enumerate(components.itertuples(index=False)):
        sub = df[df["component"] == comp.component].sort_values("theta_cm_center_deg")
        ax.errorbar(
            sub["theta_cm_center_deg"],
            sub["dsigma_domega_mb_sr"],
            yerr=np.vstack(
                [
                    sub["dsigma_domega_minus_mb_sr"],
                    sub["dsigma_domega_plus_mb_sr"],
                ]
            ),
            marker="o",
            markersize=3.2,
            lw=1.0,
            capsize=2.0,
            color=cmap(color_index),
            label=component_label(comp.component, comp.input_Ex_MeV, str(comp.Jpi)),
        )
    ax.set_xlabel(r"$\theta_{\mathrm{cm}}$ [deg]")
    ax.set_ylabel(r"$d\sigma/d\Omega_{\mathrm{cm}}$ [mb/sr]")
    ax.set_title("Preliminary differential cross sections")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.28)
    ax.legend(
        fontsize=6.4,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        handlelength=1.8,
        labelspacing=0.45,
    )
    fig.subplots_adjust(right=0.72)
    suffix = "logy" if logy else "linear"
    path = outdir / f"preliminary_differential_cross_sections_all_states_{suffix}.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def plot_individual_state_plots(df: pd.DataFrame, outdir: Path) -> list[Path]:
    components = (
        df[["component", "input_Ex_MeV", "Jpi"]]
        .drop_duplicates()
        .sort_values("input_Ex_MeV")
    )
    plot_dir = outdir / "individual_state_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for comp in components.itertuples(index=False):
        sub = df[df["component"] == comp.component].sort_values("theta_cm_center_deg")
        xerr = np.vstack(
            [
                sub["theta_cm_center_deg"] - sub["theta_cm_low_deg"],
                sub["theta_cm_high_deg"] - sub["theta_cm_center_deg"],
            ]
        )
        yerr = np.vstack(
            [
                sub["dsigma_domega_minus_mb_sr"],
                sub["dsigma_domega_plus_mb_sr"],
            ]
        )

        fig, ax = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)
        ax.errorbar(
            sub["theta_cm_center_deg"],
            sub["dsigma_domega_mb_sr"],
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=4.4,
            capsize=2.2,
            elinewidth=1.0,
            color="#1f4e79",
            ecolor="#1f4e79",
        )
        ax.plot(
            sub["theta_cm_center_deg"],
            sub["dsigma_domega_mb_sr"],
            lw=1.0,
            color="#1f4e79",
            alpha=0.75,
        )
        for row in sub.itertuples(index=False):
            ax.annotate(
                str(row.ring_group),
                (row.theta_cm_center_deg, row.dsigma_domega_mb_sr),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8.5,
            )
        ax.set_xlabel(r"$\theta_{\mathrm{cm}}$ [deg]")
        ax.set_ylabel(r"$d\sigma/d\Omega_{\mathrm{cm}}$ [mb/sr]")
        ax.set_title(component_label(comp.component, comp.input_Ex_MeV, str(comp.Jpi)))
        y_low = np.asarray(sub["dsigma_domega_mb_sr"] - sub["dsigma_domega_minus_mb_sr"])
        y_high = np.asarray(sub["dsigma_domega_mb_sr"] + sub["dsigma_domega_plus_mb_sr"])
        ymin = max(0.0, float(np.nanmin(y_low)) * 0.85)
        ymax = float(np.nanmax(y_high)) * 1.18 if len(y_high) else 1.0
        if ymax > ymin:
            ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.25)

        path = plot_dir / f"{component_filename(str(comp.component), float(comp.input_Ex_MeV))}.png"
        fig.savefig(path)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        written.append(path)
    return written


def plot_state_panels(df: pd.DataFrame, outdir: Path) -> Path:
    components = (
        df[["component", "input_Ex_MeV", "Jpi"]]
        .drop_duplicates()
        .sort_values("input_Ex_MeV")
    )
    path = outdir / "preliminary_differential_cross_sections_by_state.pdf"
    with PdfPages(path) as pdf:
        for start in range(0, len(components), 6):
            page = components.iloc[start : start + 6]
            fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=False)
            for ax in axes.flat:
                ax.set_visible(False)
            for ax, comp in zip(axes.flat, page.itertuples(index=False)):
                ax.set_visible(True)
                sub = df[df["component"] == comp.component].sort_values(
                    "theta_cm_center_deg"
                )
                ax.errorbar(
                    sub["theta_cm_center_deg"],
                    sub["dsigma_domega_mb_sr"],
                    yerr=np.vstack(
                        [
                            sub["dsigma_domega_minus_mb_sr"],
                            sub["dsigma_domega_plus_mb_sr"],
                        ]
                    ),
                    marker="o",
                    markersize=3.0,
                    lw=1.0,
                    capsize=2.0,
                    color="#1f77b4",
                )
                ax.set_title(component_label(comp.component, comp.input_Ex_MeV, str(comp.Jpi)), fontsize=9)
                ax.set_xlabel(r"$\theta_{\mathrm{cm}}$ [deg]")
                ax.set_ylabel(r"mb/sr")
                ax.grid(True, alpha=0.25)
            fig.suptitle("Preliminary differential cross sections by state", y=0.995)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    if args.beam_particles <= 0.0:
        raise ValueError("--beam-particles must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    table, meta = build_cross_section_table(args)
    table_path = args.outdir / "preliminary_differential_cross_sections.csv"
    table.to_csv(table_path, index=False)
    summary_path = args.outdir / "preliminary_differential_cross_sections_assumptions.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"fit_dir: {args.fit_dir}",
                f"assumed_beam_particles: {args.beam_particles:.8g}",
                f"beam_energy_MeV: {meta['beam_energy_MeV']:.8g}",
                f"target_material: {args.target_material}",
                f"target_areal_density_ug_cm2: {args.target_areal_density_ug_cm2:.8g}",
                f"target_nuclei_per_cm2: {meta['target_nuclei_per_cm2']:.8g}",
                "efficiency_correction: none; assumed epsilon=1",
                "solid_angle: analytic geometric CM solid angle from S3 ring lab edges",
                "normalization: dsigma/domega_mb_sr = 1e27 * yield / "
                "(beam_particles * target_nuclei_per_cm2 * delta_omega_cm_sr)",
            ]
        )
        + "\n"
    )
    linear = plot_all_states(table, args.outdir, logy=False)
    logy = plot_all_states(table, args.outdir, logy=True)
    panels = plot_state_panels(table, args.outdir)
    individual = plot_individual_state_plots(table, args.outdir)
    print(f"wrote {table_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {linear}")
    print(f"wrote {linear.with_suffix('.pdf')}")
    print(f"wrote {logy}")
    print(f"wrote {logy.with_suffix('.pdf')}")
    print(f"wrote {panels}")
    print(f"wrote {len(individual)} individual state plot pairs to {args.outdir / 'individual_state_plots'}")


if __name__ == "__main__":
    main()
