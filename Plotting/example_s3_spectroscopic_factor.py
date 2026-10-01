#!/usr/bin/env python3
"""Whole-S3 spectroscopic-factor scale example for one fitted state."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
AVOGADRO = 6.02214076e23
MOLAR_MASS_7LIF_G_MOL = 7.0160034366 + 18.99840316273
MOLAR_MASS_7LI_G_MOL = 7.0160034366


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Example extraction of a whole-S3 DWBA scale factor from one fitted "
            "state yield, one DWBA fort file, and one Geant4 exit-kinematics CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--component", default="10159_0plus")
    parser.add_argument("--dwba-file", type=Path, default=ROOT / "input/10159_0plus/fort.202")
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=OUT / "10159_0plus_truth_kinematics.csv",
        help="Used only to count the generated forced-reaction events.",
    )
    parser.add_argument(
        "--exit-csv",
        type=Path,
        default=OUT / "10159_0plus_exit_kinematics.csv",
        help="Used to count events with a triton S3 ring hit.",
    )
    parser.add_argument(
        "--fit-summary",
        type=Path,
        default=OUT
        / "MCMC_full_excitation_range_inputHist/fit_outputs/all/MCMC_full_range_fit_summary.csv",
    )
    parser.add_argument("--beam-particles", type=float, default=1.0e12)
    parser.add_argument("--target-areal-density-ug-cm2", type=float, default=500.0)
    parser.add_argument("--target-material", choices=("7LiF", "7Li"), default="7LiF")
    parser.add_argument(
        "--data-efficiency",
        type=float,
        default=1.0,
        help=(
            "Correction factor already applied to the measured yield denominator. "
            "Use 1 for the present preliminary uncorrected example."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=OUT / "spectroscopic_factor_examples",
    )
    return parser.parse_args()


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


def load_dwba_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    theta_deg = []
    sigma_mb_sr = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("@")
            or "END" in stripped
        ):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            theta = float(parts[0])
            sigma = float(parts[1])
        except ValueError:
            continue
        theta_deg.append(theta)
        sigma_mb_sr.append(sigma)

    theta = np.asarray(theta_deg, dtype=float)
    sigma = np.asarray(sigma_mb_sr, dtype=float)
    valid = np.isfinite(theta) & np.isfinite(sigma)
    theta = theta[valid]
    sigma = np.clip(sigma[valid], 0.0, None)
    if len(theta) < 2:
        raise ValueError(f"DWBA file has too few numeric rows: {path}")
    order = np.argsort(theta)
    return theta[order], sigma[order]


def integrate_dwba_total_mb(theta_deg: np.ndarray, sigma_mb_sr: np.ndarray) -> float:
    theta_rad = np.deg2rad(theta_deg)
    integrand = sigma_mb_sr * np.sin(theta_rad)
    return float(2.0 * math.pi * np.trapezoid(integrand, theta_rad))


def load_fit_yield(path: Path, component: str) -> dict[str, object]:
    df = pd.read_csv(path)
    rows = df[df["component"].astype(str) == component]
    if rows.empty:
        raise ValueError(f"Component {component!r} not found in {path}")
    if len(rows) > 1:
        raise ValueError(f"Component {component!r} appears more than once in {path}")
    row = rows.iloc[0]
    return {
        "component": component,
        "input_Ex_MeV": float(row["input_Ex_MeV"]),
        "Jpi": str(row["Jpi"]),
        "yield": float(row["MCMC_N_median"]),
        "yield_minus": float(row["MCMC_minus"]),
        "yield_plus": float(row["MCMC_plus"]),
    }


def count_s3_acceptance(truth_csv: Path, exit_csv: Path) -> dict[str, float | int]:
    truth = pd.read_csv(truth_csv, usecols=["eventID"])
    exits = pd.read_csv(exit_csv, usecols=["eventID", "triton_S3_ringID"])
    generated_event_ids = set(truth["eventID"].astype(int)).union(
        set(exits["eventID"].astype(int))
    )
    n_generated = len(generated_event_ids)
    if n_generated <= 0:
        raise ValueError(f"No generated events found in {truth_csv} or {exit_csv}")
    ring = pd.to_numeric(exits["triton_S3_ringID"], errors="coerce")
    accepted = ring.between(1, 24, inclusive="both")
    n_s3 = int(accepted.sum())
    fraction = n_s3 / n_generated
    fraction_unc = math.sqrt(fraction * (1.0 - fraction) / n_generated)
    return {
        "generated_events": n_generated,
        "truth_rows": int(len(truth)),
        "exit_rows": int(len(exits)),
        "s3_accepted_events": n_s3,
        "s3_acceptance_fraction": fraction,
        "s3_acceptance_fraction_unc": fraction_unc,
    }


def main() -> None:
    args = parse_args()
    if args.beam_particles <= 0.0:
        raise ValueError("--beam-particles must be positive")
    if args.data_efficiency <= 0.0:
        raise ValueError("--data-efficiency must be positive")

    fit = load_fit_yield(args.fit_summary, args.component)
    theta_deg, sigma_mb_sr = load_dwba_curve(args.dwba_file)
    sigma_total_mb = integrate_dwba_total_mb(theta_deg, sigma_mb_sr)
    acceptance = count_s3_acceptance(args.truth_csv, args.exit_csv)
    target = target_nuclei_per_cm2(
        args.target_areal_density_ug_cm2,
        args.target_material,
    )

    s3_fraction = float(acceptance["s3_acceptance_fraction"])
    s3_fraction_unc = float(acceptance["s3_acceptance_fraction_unc"])
    expected_for_unit = (
        args.beam_particles * target * sigma_total_mb * 1.0e-27 * s3_fraction
    )
    expected_unc_mc = (
        args.beam_particles * target * sigma_total_mb * 1.0e-27 * s3_fraction_unc
    )

    corrected_yield = float(fit["yield"]) / args.data_efficiency
    corrected_yield_minus = float(fit["yield_minus"]) / args.data_efficiency
    corrected_yield_plus = float(fit["yield_plus"]) / args.data_efficiency
    scale = corrected_yield / expected_for_unit
    rel_mc = expected_unc_mc / expected_for_unit if expected_for_unit > 0.0 else np.nan
    scale_minus = scale * math.sqrt((corrected_yield_minus / corrected_yield) ** 2 + rel_mc**2)
    scale_plus = scale * math.sqrt((corrected_yield_plus / corrected_yield) ** 2 + rel_mc**2)

    result = {
        **fit,
        **acceptance,
        "dwba_file": str(args.dwba_file),
        "dwba_theta_min_deg": float(theta_deg.min()),
        "dwba_theta_max_deg": float(theta_deg.max()),
        "dwba_total_mb": sigma_total_mb,
        "assumed_beam_particles": args.beam_particles,
        "target_material": args.target_material,
        "target_areal_density_ug_cm2": args.target_areal_density_ug_cm2,
        "target_nuclei_per_cm2": target,
        "data_efficiency": args.data_efficiency,
        "corrected_data_yield": corrected_yield,
        "corrected_data_yield_minus": corrected_yield_minus,
        "corrected_data_yield_plus": corrected_yield_plus,
        "expected_s3_yield_for_dwba_unit_strength": expected_for_unit,
        "expected_s3_yield_mc_unc": expected_unc_mc,
        "dwba_scale_factor": scale,
        "dwba_scale_factor_minus": scale_minus,
        "dwba_scale_factor_plus": scale_plus,
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.component}_s3_spectroscopic_factor_example"
    csv_path = args.outdir / f"{stem}.csv"
    txt_path = args.outdir / f"{stem}.txt"
    pd.DataFrame([result]).to_csv(csv_path, index=False)

    lines = [
        f"component: {args.component}",
        f"Ex_MeV: {fit['input_Ex_MeV']:.6g}",
        f"Jpi: {fit['Jpi']}",
        f"fitted_data_yield: {fit['yield']:.6g} -{fit['yield_minus']:.6g} +{fit['yield_plus']:.6g}",
        f"data_efficiency: {args.data_efficiency:.6g}",
        f"corrected_data_yield: {corrected_yield:.6g} -{corrected_yield_minus:.6g} +{corrected_yield_plus:.6g}",
        f"dwba_file: {args.dwba_file}",
        f"dwba_integral_total_mb: {sigma_total_mb:.8g}",
        (
            "s3_acceptance_fraction: "
            f"{s3_fraction:.8g} +/- {s3_fraction_unc:.3g} "
            f"({acceptance['s3_accepted_events']}/{acceptance['generated_events']})"
        ),
        f"assumed_beam_particles: {args.beam_particles:.8g}",
        f"target_material: {args.target_material}",
        f"target_areal_density_ug_cm2: {args.target_areal_density_ug_cm2:.8g}",
        f"target_nuclei_per_cm2: {target:.8g}",
        f"expected_s3_yield_for_dwba_unit_strength: {expected_for_unit:.8g} +/- {expected_unc_mc:.3g}",
        f"dwba_scale_factor: {scale:.8g} -{scale_minus:.3g} +{scale_plus:.3g}",
        "",
        "Interpretation:",
        "  The scale factor is a spectroscopic factor only if the DWBA file is normalized to unit strength.",
        "  The example keeps data_efficiency=1, so replace that with your real correction before using it as final.",
    ]
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {txt_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
