#!/usr/bin/env python3
"""Calculate alpha spectroscopic factors for fitted S2223 excitation states."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import uproot
except ImportError:  # pragma: no cover - only needed for --tof-root
    uproot = None


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[0]
OUT = ROOT / "output"
DEFAULT_FIT_SUMMARY = (
    OUT
    / "MCMC_full_excitation_range_inputHist"
    / "fit_outputs"
    / "all"
    / "MCMC_full_range_fit_summary.csv"
)
DEFAULT_RUNS_FILE = PROJECT_ROOT / "TC_code" / "runs_good_nomcur.txt"
DEFAULT_TOF_ROOT = PROJECT_ROOT / "G4EMMA" / "UserDir" / "Data" / "Plots_runs_good_nomcur_emma.root"

E_CHARGE_C = 1.602176634e-19
AVOGADRO = 6.02214076e23
MOLAR_MASS_7LIF_G_MOL = 7.0160034366 + 18.99840316273
MOLAR_MASS_7LI_G_MOL = 7.0160034366


@dataclass(frozen=True)
class ChargeBlock:
    label: str
    charge_c: float
    beam_particles: float
    target_areal_density_ug_cm2: float
    target_thickness_um: float
    target_thickness_unc_um: float
    target_nuclei_cm2: float
    target_nuclei_unc_cm2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract alpha spectroscopic factors by combining fitted excitation "
            "yields, DWBA fort.202 curves, Geant4 S3 acceptance, run-list charge, "
            "target areal densities, and detector/TOF efficiencies."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fit-summary", type=Path, default=DEFAULT_FIT_SUMMARY)
    parser.add_argument("--runs-file", type=Path, default=DEFAULT_RUNS_FILE)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input")
    parser.add_argument("--sim-output-dir", type=Path, default=OUT)
    parser.add_argument("--astro-range", type=float, nargs=2, default=(10.915, 11.364))
    parser.add_argument("--beam-charge-state", type=float, default=4.0)
    parser.add_argument("--s3-efficiency", type=float, default=0.886)
    parser.add_argument("--pgac-ic-efficiency", type=float, default=0.86)
    parser.add_argument(
        "--pgac-ic-efficiency-unc-plus",
        type=float,
        default=0.0,
        help="Absolute upper uncertainty on the PGAC+IC efficiency.",
    )
    parser.add_argument(
        "--pgac-ic-efficiency-unc-minus",
        type=float,
        default=0.036 * 0.83,
        help="Absolute lower uncertainty on the PGAC+IC efficiency.",
    )
    parser.add_argument(
        "--tof-efficiency",
        type=float,
        default=None,
        help="Override the TIGRESS-EMMA timing efficiency. If omitted, evaluate from --tof-root.",
    )
    parser.add_argument(
        "--tof-root",
        type=Path,
        default=DEFAULT_TOF_ROOT,
        help="ROOT file containing hist1D/hTigEmTOF_S3Pix1 for the TOF correction.",
    )
    parser.add_argument("--tof-hist", default="hist1D/hTigEmTOF_S3Pix1")
    parser.add_argument(
        "--tof-main-window",
        type=float,
        nargs=2,
        default=(800.0, 1000.0),
        help="Main TIGRESS-EMMA timing gate.",
    )
    parser.add_argument(
        "--tof-second-window",
        type=float,
        nargs=2,
        default=(1200.0, 1400.0),
        help="Delayed TIGRESS-EMMA timing peak window used for the yield-loss correction.",
    )
    parser.add_argument(
        "--tof-bg-left-window",
        type=float,
        nargs=2,
        default=(600.0, 800.0),
        help="Left sideband window for the TOF background estimate.",
    )
    parser.add_argument(
        "--tof-bg-right-window",
        type=float,
        nargs=2,
        default=(1400.0, 1600.0),
        help="Right sideband window for the TOF background estimate.",
    )
    parser.add_argument(
        "--target-material",
        choices=("7LiF", "7Li"),
        default="7LiF",
        help="Interpret mass thickness as LiF compound or pure 7Li.",
    )
    parser.add_argument(
        "--target-thickness-reference-um",
        type=float,
        default=1.9,
        help="Reference LiF thickness used to define the absolute thickness uncertainty.",
    )
    parser.add_argument(
        "--target-thickness-rel-unc",
        type=float,
        default=0.05,
        help="Relative uncertainty on the reference LiF thickness.",
    )
    parser.add_argument(
        "--default-target-areal-density-ug-cm2",
        type=float,
        default=500.0,
        help="Fallback only for run-list blocks whose target number cannot be identified.",
    )
    parser.add_argument(
        "--target5-areal-density-ug-cm2",
        type=float,
        default=491.0,
        help="LiF #5 areal density from TC_code/SortCodeIntoBranches.cxx.",
    )
    parser.add_argument(
        "--target6-areal-density-ug-cm2",
        type=float,
        default=507.0,
        help="LiF #6 areal density from TC_code/SortCodeIntoBranches.cxx.",
    )
    parser.add_argument(
        "--target5-thickness-um",
        type=float,
        default=1.863,
        help="LiF #5 physical thickness from TC_code/SortCodeIntoBranches.cxx.",
    )
    parser.add_argument(
        "--target6-thickness-um",
        type=float,
        default=1.924,
        help="LiF #6 physical thickness from TC_code/SortCodeIntoBranches.cxx.",
    )
    parser.add_argument(
        "--default-target-thickness-um",
        type=float,
        default=1.9,
        help="Fallback target thickness for run-list blocks whose target number cannot be identified.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=OUT / "alpha_spectroscopic_factors",
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


def classify_target_label(comment: str) -> str | None:
    text = comment.lower()
    if "lif" not in text:
        return None
    match = re.search(r"lif\s*([56])", text)
    if match:
        return f"LiF{match.group(1)}"
    return None


def target_density_for_label(args: argparse.Namespace, label: str | None) -> float:
    if label == "LiF5":
        return float(args.target5_areal_density_ug_cm2)
    if label == "LiF6":
        return float(args.target6_areal_density_ug_cm2)
    return float(args.default_target_areal_density_ug_cm2)


def target_thickness_for_label(args: argparse.Namespace, label: str | None) -> float:
    if label == "LiF5":
        return float(args.target5_thickness_um)
    if label == "LiF6":
        return float(args.target6_thickness_um)
    return float(args.default_target_thickness_um)


def load_charge_blocks(args: argparse.Namespace) -> list[ChargeBlock]:
    if args.beam_charge_state <= 0.0:
        raise ValueError("--beam-charge-state must be positive")

    charge_by_label: dict[str, float] = {}
    current_label: str | None = None
    for raw in args.runs_file.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            new_label = classify_target_label(line)
            if new_label is not None:
                current_label = new_label
            continue

        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            nominal_current_pa = float(parts[1])
            runtime_s = float(parts[2])
        except ValueError:
            continue
        label = current_label or "unknown"
        charge_by_label[label] = charge_by_label.get(label, 0.0) + nominal_current_pa * 1.0e-12 * runtime_s

    blocks: list[ChargeBlock] = []
    target_thickness_unc_um = (
        float(args.target_thickness_reference_um) * float(args.target_thickness_rel_unc)
    )
    for label, charge_c in sorted(charge_by_label.items()):
        target_areal = target_density_for_label(args, label)
        target_thickness_um = target_thickness_for_label(args, label)
        if target_thickness_um <= 0.0:
            raise ValueError(f"Target thickness must be positive for {label}")
        target_nuclei = target_nuclei_per_cm2(target_areal, args.target_material)
        target_nuclei_unc = target_nuclei * target_thickness_unc_um / target_thickness_um
        beam_particles = charge_c / (args.beam_charge_state * E_CHARGE_C)
        blocks.append(
            ChargeBlock(
                label=label,
                charge_c=charge_c,
                beam_particles=beam_particles,
                target_areal_density_ug_cm2=target_areal,
                target_thickness_um=target_thickness_um,
                target_thickness_unc_um=target_thickness_unc_um,
                target_nuclei_cm2=target_nuclei,
                target_nuclei_unc_cm2=target_nuclei_unc,
            )
        )
    if not blocks:
        raise ValueError(f"No run charge rows found in {args.runs_file}")
    return blocks


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


def root_hist_integral_centers(path: Path, hist_name: str, window: tuple[float, float]) -> float:
    if uproot is None:
        raise RuntimeError("uproot is required to evaluate --tof-root")
    with uproot.open(path) as root_file:
        hist = root_file[hist_name]
        values, edges = hist.to_numpy(flow=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    lo, hi = window
    return float(values[(centers >= lo) & (centers < hi)].sum())


def evaluate_tof_efficiency(args: argparse.Namespace) -> dict[str, float | str]:
    if args.tof_efficiency is not None:
        if not (0.0 < args.tof_efficiency <= 1.0):
            raise ValueError("--tof-efficiency must be in (0, 1]")
        return {
            "tof_source": "manual",
            "tof_hist": "",
            "tof_main_window_ns": "",
            "tof_second_window_ns": "",
            "tof_bg_left_window_ns": "",
            "tof_bg_right_window_ns": "",
            "tof_main_counts": math.nan,
            "tof_second_counts": math.nan,
            "tof_bg_left_counts": math.nan,
            "tof_bg_right_counts": math.nan,
            "tof_bg_counts_per_peak_window": math.nan,
            "tof_bg_counts_per_peak_window_unc": math.nan,
            "tof_main_real_counts": math.nan,
            "tof_second_real_counts": math.nan,
            "tof_efficiency": float(args.tof_efficiency),
            "tof_efficiency_unc": 0.0,
            "tof_correction_factor": 1.0 / float(args.tof_efficiency),
        }

    main_window = (float(args.tof_main_window[0]), float(args.tof_main_window[1]))
    second_window = (float(args.tof_second_window[0]), float(args.tof_second_window[1]))
    bg_left_window = (float(args.tof_bg_left_window[0]), float(args.tof_bg_left_window[1]))
    bg_right_window = (float(args.tof_bg_right_window[0]), float(args.tof_bg_right_window[1]))
    main_counts = root_hist_integral_centers(args.tof_root, args.tof_hist, main_window)
    second_counts = root_hist_integral_centers(args.tof_root, args.tof_hist, second_window)
    bg_left_counts = root_hist_integral_centers(args.tof_root, args.tof_hist, bg_left_window)
    bg_right_counts = root_hist_integral_centers(args.tof_root, args.tof_hist, bg_right_window)
    bg_counts = 0.5 * (bg_left_counts + bg_right_counts)
    bg_var = 0.25 * (bg_left_counts + bg_right_counts)
    main_real = main_counts - bg_counts
    second_real = second_counts - bg_counts
    if main_real <= 0.0:
        raise ValueError(f"TOF main window has no counts in {args.tof_root}:{args.tof_hist}")
    if second_real < 0.0:
        raise ValueError(f"TOF second window is below the sideband background in {args.tof_root}:{args.tof_hist}")
    total_real = main_real + second_real
    efficiency = main_real / total_real
    main_var = main_counts + bg_var
    second_var = second_counts + bg_var
    covariance = bg_var
    d_eff_d_main = second_real / (total_real * total_real)
    d_eff_d_second = -main_real / (total_real * total_real)
    efficiency_var = (
        d_eff_d_main * d_eff_d_main * main_var
        + d_eff_d_second * d_eff_d_second * second_var
        + 2.0 * d_eff_d_main * d_eff_d_second * covariance
    )
    efficiency_unc = math.sqrt(max(0.0, efficiency_var))
    return {
        "tof_source": str(args.tof_root),
        "tof_hist": args.tof_hist,
        "tof_main_window_ns": f"{main_window[0]:g}-{main_window[1]:g}",
        "tof_second_window_ns": f"{second_window[0]:g}-{second_window[1]:g}",
        "tof_bg_left_window_ns": f"{bg_left_window[0]:g}-{bg_left_window[1]:g}",
        "tof_bg_right_window_ns": f"{bg_right_window[0]:g}-{bg_right_window[1]:g}",
        "tof_main_counts": main_counts,
        "tof_second_counts": second_counts,
        "tof_bg_left_counts": bg_left_counts,
        "tof_bg_right_counts": bg_right_counts,
        "tof_bg_counts_per_peak_window": bg_counts,
        "tof_bg_counts_per_peak_window_unc": math.sqrt(bg_var),
        "tof_main_real_counts": main_real,
        "tof_second_real_counts": second_real,
        "tof_efficiency": efficiency,
        "tof_efficiency_unc": efficiency_unc,
        "tof_correction_factor": 1.0 / efficiency,
    }


def select_astro_rows(fit_summary: Path, astro_range: tuple[float, float]) -> pd.DataFrame:
    df = pd.read_csv(fit_summary)
    lo, hi = astro_range
    selected = df[(df["input_Ex_MeV"] >= lo) & (df["input_Ex_MeV"] <= hi)].copy()
    return selected.sort_values(["input_Ex_MeV", "component"]).reset_index(drop=True)


def safe_float(row: pd.Series, column: str) -> float:
    value = row.get(column, math.nan)
    return float(value) if pd.notna(value) else math.nan


def inverse_quantity_relative_uncertainties(
    value: float,
    unc_plus: float,
    unc_minus: float,
) -> tuple[float, float]:
    """Return relative SF minus/plus contributions for a denominator quantity."""
    rel_minus = 0.0
    rel_plus = 0.0
    if unc_plus > 0.0:
        rel_minus = 1.0 - value / (value + unc_plus)
    if unc_minus > 0.0:
        if value <= unc_minus:
            raise ValueError("Denominator uncertainty reaches zero or below")
        rel_plus = value / (value - unc_minus) - 1.0
    return rel_minus, rel_plus


def calculate_state(
    row: pd.Series,
    args: argparse.Namespace,
    charge_blocks: list[ChargeBlock],
    tof: dict[str, float | str],
) -> dict[str, object]:
    component = str(row["component"])
    dwba_file = args.input_dir / component / "fort.202"
    truth_csv = args.sim_output_dir / f"{component}_truth_kinematics.csv"
    exit_csv = args.sim_output_dir / f"{component}_exit_kinematics.csv"
    missing = [str(path) for path in (dwba_file, truth_csv, exit_csv) if not path.exists()]
    if missing:
        return {
            "component": component,
            "input_Ex_MeV": safe_float(row, "input_Ex_MeV"),
            "Jpi": row.get("Jpi", ""),
            "status": "skipped_missing_inputs",
            "missing_inputs": "; ".join(missing),
            "MCMC_N_median": safe_float(row, "MCMC_N_median"),
            "MCMC_minus": safe_float(row, "MCMC_minus"),
            "MCMC_plus": safe_float(row, "MCMC_plus"),
        }

    theta_deg, sigma_mb_sr = load_dwba_curve(dwba_file)
    sigma_total_mb = integrate_dwba_total_mb(theta_deg, sigma_mb_sr)
    acceptance = count_s3_acceptance(truth_csv, exit_csv)
    s3_acceptance = float(acceptance["s3_acceptance_fraction"])
    s3_acceptance_unc = float(acceptance["s3_acceptance_fraction_unc"])

    exposure_sum = sum(block.beam_particles * block.target_nuclei_cm2 for block in charge_blocks)
    exposure_sum_target_unc = sum(
        block.beam_particles * block.target_nuclei_unc_cm2 for block in charge_blocks
    )
    total_beam_particles = sum(block.beam_particles for block in charge_blocks)
    total_charge_c = sum(block.charge_c for block in charge_blocks)
    effective_target_nuclei = exposure_sum / total_beam_particles
    expected_for_unit = exposure_sum * sigma_total_mb * 1.0e-27 * s3_acceptance
    expected_unc_mc = exposure_sum * sigma_total_mb * 1.0e-27 * s3_acceptance_unc

    data_efficiency = (
        float(args.s3_efficiency)
        * float(args.pgac_ic_efficiency)
        * float(tof["tof_efficiency"])
    )
    fitted_yield = safe_float(row, "MCMC_N_median")
    fitted_minus = safe_float(row, "MCMC_minus")
    fitted_plus = safe_float(row, "MCMC_plus")
    corrected_yield = fitted_yield / data_efficiency
    corrected_minus = fitted_minus / data_efficiency
    corrected_plus = fitted_plus / data_efficiency
    spectroscopic_factor = corrected_yield / expected_for_unit
    yield_rel_minus = fitted_minus / fitted_yield
    yield_rel_plus = fitted_plus / fitted_yield
    acceptance_rel_minus, acceptance_rel_plus = inverse_quantity_relative_uncertainties(
        s3_acceptance,
        s3_acceptance_unc,
        s3_acceptance_unc,
    )
    tof_rel_minus, tof_rel_plus = inverse_quantity_relative_uncertainties(
        float(tof["tof_efficiency"]),
        float(tof["tof_efficiency_unc"]),
        float(tof["tof_efficiency_unc"]),
    )
    pgac_rel_minus, pgac_rel_plus = inverse_quantity_relative_uncertainties(
        float(args.pgac_ic_efficiency),
        float(args.pgac_ic_efficiency_unc_plus),
        float(args.pgac_ic_efficiency_unc_minus),
    )
    target_rel_minus = exposure_sum_target_unc / exposure_sum
    target_rel_plus = exposure_sum_target_unc / exposure_sum
    sf_rel_minus = math.sqrt(
        yield_rel_minus**2
        + acceptance_rel_minus**2
        + tof_rel_minus**2
        + pgac_rel_minus**2
        + target_rel_minus**2
    )
    sf_rel_plus = math.sqrt(
        yield_rel_plus**2
        + acceptance_rel_plus**2
        + tof_rel_plus**2
        + pgac_rel_plus**2
        + target_rel_plus**2
    )
    sf_minus = spectroscopic_factor * sf_rel_minus
    sf_plus = spectroscopic_factor * sf_rel_plus

    return {
        "component": component,
        "input_Ex_MeV": safe_float(row, "input_Ex_MeV"),
        "Jpi": row.get("Jpi", ""),
        "status": "calculated",
        "MCMC_N_median": fitted_yield,
        "MCMC_minus": fitted_minus,
        "MCMC_plus": fitted_plus,
        "s3_efficiency": float(args.s3_efficiency),
        "pgac_ic_efficiency": float(args.pgac_ic_efficiency),
        "pgac_ic_efficiency_unc_plus": float(args.pgac_ic_efficiency_unc_plus),
        "pgac_ic_efficiency_unc_minus": float(args.pgac_ic_efficiency_unc_minus),
        **tof,
        "total_data_efficiency": data_efficiency,
        "corrected_yield": corrected_yield,
        "corrected_yield_minus": corrected_minus,
        "corrected_yield_plus": corrected_plus,
        "total_charge_C": total_charge_c,
        "beam_charge_state": float(args.beam_charge_state),
        "beam_particles": total_beam_particles,
        "target_material": args.target_material,
        "target_thickness_reference_um": float(args.target_thickness_reference_um),
        "target_thickness_rel_unc": float(args.target_thickness_rel_unc),
        "target_thickness_unc_um": (
            float(args.target_thickness_reference_um) * float(args.target_thickness_rel_unc)
        ),
        "effective_target_nuclei_per_cm2": effective_target_nuclei,
        "beam_target_exposure_sum": exposure_sum,
        "beam_target_exposure_target_unc": exposure_sum_target_unc,
        "dwba_file": str(dwba_file),
        "dwba_total_mb": sigma_total_mb,
        "dwba_theta_min_deg": float(theta_deg.min()),
        "dwba_theta_max_deg": float(theta_deg.max()),
        **acceptance,
        "expected_s3_yield_for_unit_strength": expected_for_unit,
        "expected_s3_yield_mc_unc": expected_unc_mc,
        "alpha_spectroscopic_factor": spectroscopic_factor,
        "alpha_spectroscopic_factor_minus": sf_minus,
        "alpha_spectroscopic_factor_plus": sf_plus,
        "sf_rel_unc_yield_minus": yield_rel_minus,
        "sf_rel_unc_yield_plus": yield_rel_plus,
        "sf_rel_unc_s3_acceptance_minus": acceptance_rel_minus,
        "sf_rel_unc_s3_acceptance_plus": acceptance_rel_plus,
        "sf_rel_unc_tof_minus": tof_rel_minus,
        "sf_rel_unc_tof_plus": tof_rel_plus,
        "sf_rel_unc_pgac_ic_minus": pgac_rel_minus,
        "sf_rel_unc_pgac_ic_plus": pgac_rel_plus,
        "sf_rel_unc_target_minus": target_rel_minus,
        "sf_rel_unc_target_plus": target_rel_plus,
        "sf_rel_unc_total_minus": sf_rel_minus,
        "sf_rel_unc_total_plus": sf_rel_plus,
        "missing_inputs": "",
    }


def write_summary(
    path: Path,
    results: pd.DataFrame,
    charge_blocks: list[ChargeBlock],
    args: argparse.Namespace,
    tof: dict[str, float | str],
) -> None:
    calculated = results[results["status"] == "calculated"]
    skipped = results[results["status"] != "calculated"]
    total_charge = sum(block.charge_c for block in charge_blocks)
    total_beam = sum(block.beam_particles for block in charge_blocks)
    exposure_sum = sum(block.beam_particles * block.target_nuclei_cm2 for block in charge_blocks)
    exposure_sum_target_unc = sum(
        block.beam_particles * block.target_nuclei_unc_cm2 for block in charge_blocks
    )
    effective_target = exposure_sum / total_beam

    lines = [
        "Alpha spectroscopic-factor normalization",
        "=========================================",
        f"fit_summary: {args.fit_summary}",
        f"astro_range_MeV: {args.astro_range[0]:.6g}-{args.astro_range[1]:.6g}",
        f"runs_file: {args.runs_file}",
        f"total_charge_C: {total_charge:.12g}",
        f"beam_charge_state: {args.beam_charge_state:.6g}",
        f"beam_particles: {total_beam:.12g}",
        f"target_material: {args.target_material}",
        f"target_thickness_reference_um: {args.target_thickness_reference_um:.8g}",
        f"target_thickness_rel_unc: {args.target_thickness_rel_unc:.8g}",
        f"target_thickness_unc_um: {args.target_thickness_reference_um * args.target_thickness_rel_unc:.8g}",
        f"effective_target_nuclei_per_cm2: {effective_target:.12g}",
        f"beam_target_exposure_target_rel_unc: {exposure_sum_target_unc / exposure_sum:.8g}",
        "",
        "Charge blocks:",
    ]
    for block in charge_blocks:
        lines.append(
            "  "
            f"{block.label}: charge_C={block.charge_c:.12g}, "
            f"beam_particles={block.beam_particles:.12g}, "
            f"target_ug_cm2={block.target_areal_density_ug_cm2:.6g}, "
            f"target_thickness_um={block.target_thickness_um:.6g}, "
            f"target_thickness_unc_um={block.target_thickness_unc_um:.6g}, "
            f"target_nuclei_cm2={block.target_nuclei_cm2:.12g}"
        )

    lines.extend(
        [
            "",
            "Efficiencies:",
            f"  S3: {args.s3_efficiency:.8g}",
            (
                f"  PGAC_IC: {args.pgac_ic_efficiency:.8g} "
                f"+{args.pgac_ic_efficiency_unc_plus:.6g} "
                f"-{args.pgac_ic_efficiency_unc_minus:.6g}"
            ),
            (
                f"  TOF_efficiency: {float(tof['tof_efficiency']):.8g} "
                f"+/- {float(tof['tof_efficiency_unc']):.3g}"
            ),
            f"  TOF_correction_factor: {float(tof['tof_correction_factor']):.8g}",
            f"  TOF_source: {tof['tof_source']}",
            f"  TOF_hist: {tof['tof_hist']}",
            (
                f"  TOF_main_window_ns: {tof['tof_main_window_ns']} "
                f"raw={float(tof['tof_main_counts']):.8g} "
                f"bg_subtracted={float(tof['tof_main_real_counts']):.8g}"
            ),
            (
                f"  TOF_second_window_ns: {tof['tof_second_window_ns']} "
                f"raw={float(tof['tof_second_counts']):.8g} "
                f"bg_subtracted={float(tof['tof_second_real_counts']):.8g}"
            ),
            (
                f"  TOF_background_windows_ns: {tof['tof_bg_left_window_ns']} and "
                f"{tof['tof_bg_right_window_ns']}; average={float(tof['tof_bg_counts_per_peak_window']):.8g} "
                f"+/- {float(tof['tof_bg_counts_per_peak_window_unc']):.3g}"
            ),
            f"  total_data_efficiency: {args.s3_efficiency * args.pgac_ic_efficiency * float(tof['tof_efficiency']):.8g}",
            "",
            "Calculated states:",
        ]
    )
    if calculated.empty:
        lines.append("  none")
    else:
        for _, row in calculated.iterrows():
            lines.append(
                "  "
                f"{row['component']} Ex={row['input_Ex_MeV']:.6g} MeV Jpi={row['Jpi']}: "
                f"S_alpha={row['alpha_spectroscopic_factor']:.8g} "
                f"-{row['alpha_spectroscopic_factor_minus']:.3g} "
                f"+{row['alpha_spectroscopic_factor_plus']:.3g}; "
                f"yield={row['MCMC_N_median']:.6g} "
                f"-{row['MCMC_minus']:.3g} +{row['MCMC_plus']:.3g}; "
                f"DWBA_total_mb={row['dwba_total_mb']:.8g}; "
                f"S3_acceptance={row['s3_acceptance_fraction']:.8g}; "
                f"rel_unc(yield,S3,TOF,PGAC,target,total)="
                f"-({row['sf_rel_unc_yield_minus']:.3g},"
                f"{row['sf_rel_unc_s3_acceptance_minus']:.3g},"
                f"{row['sf_rel_unc_tof_minus']:.3g},"
                f"{row['sf_rel_unc_pgac_ic_minus']:.3g},"
                f"{row['sf_rel_unc_target_minus']:.3g},"
                f"{row['sf_rel_unc_total_minus']:.3g}) "
                f"+({row['sf_rel_unc_yield_plus']:.3g},"
                f"{row['sf_rel_unc_s3_acceptance_plus']:.3g},"
                f"{row['sf_rel_unc_tof_plus']:.3g},"
                f"{row['sf_rel_unc_pgac_ic_plus']:.3g},"
                f"{row['sf_rel_unc_target_plus']:.3g},"
                f"{row['sf_rel_unc_total_plus']:.3g})"
            )

    lines.append("")
    lines.append("Skipped states:")
    if skipped.empty:
        lines.append("  none")
    else:
        for _, row in skipped.iterrows():
            lines.append(
                "  "
                f"{row['component']} Ex={row['input_Ex_MeV']:.6g} MeV Jpi={row['Jpi']}: "
                f"{row['status']} ({row['missing_inputs']})"
            )

    lines.extend(
        [
            "",
            "Notes:",
            "  S_alpha assumes each DWBA fort.202 curve is normalized to unit alpha strength.",
            "  The TOF correction treats the sideband-subtracted delayed TIGRESS-EMMA peak as lost yield from the fitted main-peak spectrum.",
            "  The expected yield uses the whole-S3 Geant4 triton acceptance; the supplied PGAC+IC value is applied as the recoil-side efficiency.",
            "  Uncertainties include the MCMC yield interval, Geant4 binomial S3-acceptance counting uncertainty, TOF counting uncertainty, supplied PGAC+IC efficiency uncertainty, and physical LiF target-thickness uncertainty.",
            "  No uncertainty was supplied for charge integration, S3 efficiency, or DWBA normalization.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.s3_efficiency <= 1.0):
        raise ValueError("--s3-efficiency must be in (0, 1]")
    if not (0.0 < args.pgac_ic_efficiency <= 1.0):
        raise ValueError("--pgac-ic-efficiency must be in (0, 1]")
    if args.pgac_ic_efficiency_unc_plus < 0.0 or args.pgac_ic_efficiency_unc_minus < 0.0:
        raise ValueError("PGAC+IC efficiency uncertainties must be non-negative")
    if args.pgac_ic_efficiency <= args.pgac_ic_efficiency_unc_minus:
        raise ValueError("PGAC+IC lower uncertainty reaches zero efficiency")
    if args.target_thickness_reference_um <= 0.0:
        raise ValueError("--target-thickness-reference-um must be positive")
    if args.target_thickness_rel_unc < 0.0:
        raise ValueError("--target-thickness-rel-unc must be non-negative")

    args.outdir.mkdir(parents=True, exist_ok=True)
    charge_blocks = load_charge_blocks(args)
    tof = evaluate_tof_efficiency(args)
    selected = select_astro_rows(args.fit_summary, tuple(args.astro_range))
    if selected.empty:
        raise ValueError(
            f"No fit-summary components found in astro range {args.astro_range[0]}-{args.astro_range[1]} MeV"
        )

    results = pd.DataFrame(
        [calculate_state(row, args, charge_blocks, tof) for _, row in selected.iterrows()]
    )
    results_path = args.outdir / "alpha_spectroscopic_factors.csv"
    summary_path = args.outdir / "alpha_spectroscopic_factors_summary.txt"
    charge_path = args.outdir / "charge_blocks.csv"
    results.to_csv(results_path, index=False)
    pd.DataFrame([block.__dict__ for block in charge_blocks]).to_csv(charge_path, index=False)
    write_summary(summary_path, results, charge_blocks, args, tof)

    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {charge_path}")
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
