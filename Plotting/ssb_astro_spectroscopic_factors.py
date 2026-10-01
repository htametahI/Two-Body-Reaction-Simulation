#!/usr/bin/env python3
"""Evaluate astrophysical-state alpha spectroscopic factors with SSB luminosity."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_spectroscopic_factors import (
    count_s3_acceptance,
    integrate_dwba_total_mb,
    inverse_quantity_relative_uncertainties,
    load_dwba_curve,
)


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
OUTPUT_ROOT = ROOT / "output"

DEFAULT_FIT_SUMMARY = (
    OUTPUT_ROOT
    / "MCMC_full_excitation_range_inputHist"
    / "fit_outputs"
    / "all"
    / "MCMC_full_range_fit_summary.csv"
)
DEFAULT_LUMINOSITY = (
    PROJECT_ROOT / "TC_code" / "data" / "ssb-luminosity-46-58MeV-summary.csv"
)
DEFAULT_EMMA = ROOT / "input" / "emma_transmission_efficiency.csv"
DEFAULT_LITERATURE = ROOT / "input" / "literature_alpha_spectroscopic_factors.csv"
DEFAULT_SELECTED_YIELDS = ROOT / "input" / "astro_state_selected_mcmc_yields.csv"
DEFAULT_OUTPUT_CSV = OUTPUT_ROOT / "ssb_spectroscopic_factors_astro_livetime_corrected.csv"
DEFAULT_SPIN_COMPARISON_CSV = (
    OUTPUT_ROOT / "ssb_11172_spin_hypothesis_comparison_livetime_corrected.csv"
)
DEFAULT_OUTPUT_PDF = (
    OUTPUT_ROOT / "pdf" / "ssb-spectroscopic-factors-astro-livetime-corrected.pdf"
)
DEFAULT_OUTPUT_LINEAR_PDF = (
    OUTPUT_ROOT / "pdf" / "ssb-spectroscopic-factors-astro-livetime-corrected-linear.pdf"
)
DEFAULT_EMMA_ACCEPTED = 32_524_147
DEFAULT_EMMA_PRESENTED = 32_979_454

S3_EFFICIENCY = 0.886
PGAC_IC_EFFICIENCY = 0.83
PGAC_IC_UNC_PLUS = 0.0
PGAC_IC_UNC_MINUS = 0.036  # Absolute: 83.0% +0.0/-3.6 percentage points.
TOF_EFFICIENCY = 0.8959915140687807
TOF_EFFICIENCY_UNC = 0.001213022026786503


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use SSB integrated luminosity, fitted state yields, DWBA calculations, "
            "S3 acceptance, EMMA transmission, and live time to extract alpha "
            "spectroscopic factors."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fit-summary", type=Path, default=DEFAULT_FIT_SUMMARY)
    parser.add_argument("--luminosity-summary", type=Path, default=DEFAULT_LUMINOSITY)
    parser.add_argument("--emma-efficiency", type=Path, default=DEFAULT_EMMA)
    parser.add_argument(
        "--emma-accepted", type=int, default=DEFAULT_EMMA_ACCEPTED,
        help="Accepted scaler counts for the selected dataset (a subset of presented).",
    )
    parser.add_argument(
        "--emma-presented", type=int, default=DEFAULT_EMMA_PRESENTED,
        help="Presented scaler counts; A/P is applied as a common live fraction.",
    )
    parser.add_argument(
        "--literature-values",
        type=Path,
        default=DEFAULT_LITERATURE,
        help="Published spectroscopic factors and upper limits to overlay.",
    )
    parser.add_argument(
        "--selected-yields",
        type=Path,
        default=DEFAULT_SELECTED_YIELDS,
        help=(
            "Selected astrophysical-state MCMC medians and intervals. These values "
            "override the corresponding rows in --fit-summary."
        ),
    )
    parser.add_argument(
        "--q12-fraction",
        type=float,
        default=0.138,
        help="Fraction of recoils exiting the target foil in charge state 12+.",
    )
    parser.add_argument(
        "--q12-relative-uncertainty",
        type=float,
        default=0.15,
        help="Relative uncertainty on the 12+ recoil charge-state fraction.",
    )
    parser.add_argument(
        "--dwba-relative-uncertainty",
        type=float,
        default=0.10,
        help="Relative model uncertainty assigned to the DWBA cross section.",
    )
    parser.add_argument(
        "--fit-model-relative-uncertainty",
        type=float,
        default=0.05,
        help="Relative uncertainty assigned to excitation-spectrum fit-model choices.",
    )
    parser.add_argument(
        "--interpolation-relative-uncertainty",
        type=float,
        default=0.05,
        help="Relative uncertainty applied only to interpolated state inputs.",
    )
    parser.add_argument(
        "--s3-efficiency-relative-uncertainty",
        type=float,
        default=0.05,
        help="Relative uncertainty assigned to the S3 intrinsic efficiency.",
    )
    parser.add_argument(
        "--ssb-background-relative-uncertainty",
        type=float,
        default=0.05,
        help="Relative luminosity uncertainty assigned to SSB peak background.",
    )
    parser.add_argument(
        "--luminosity-systematics",
        choices=("correlated", "independent"),
        default="correlated",
        help=(
            "How to combine the target-period systematic uncertainties. Correlated is "
            "the conservative choice for common current and target calibration scales."
        ),
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--spin-comparison-csv", type=Path, default=DEFAULT_SPIN_COMPARISON_CSV
    )
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument(
        "--output-linear-pdf", type=Path, default=DEFAULT_OUTPUT_LINEAR_PDF
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Calculate tabular results without regenerating the comparison PDF.",
    )
    parser.add_argument(
        "--reuse-results-csv",
        action="store_true",
        help=(
            "Update existing --output-csv and --spin-comparison-csv and render plots "
            "without rereading simulation events. Matching live-time corrections "
            "are applied only once."
        ),
    )
    return parser.parse_args()


def read_luminosity(path: Path, systematic_mode: str) -> dict[str, float]:
    frame = pd.read_csv(path)
    luminosities = frame["integrated_luminosity_ub-1"].to_numpy(float)
    stat = frame["statistical_uncertainty_ub-1"].to_numpy(float)
    syst = frame["systematic_uncertainty_ub-1"].to_numpy(float)
    if systematic_mode == "correlated":
        combined_syst = float(syst.sum())
    else:
        combined_syst = float(np.hypot.reduce(syst))
    return {
        "luminosity_ub_inv": float(luminosities.sum()),
        "luminosity_stat_ub_inv": float(np.hypot.reduce(stat)),
        "luminosity_syst_ub_inv": combined_syst,
        "luminosity_syst_independent_ub_inv": float(np.hypot.reduce(syst)),
        "luminosity_syst_correlated_ub_inv": float(syst.sum()),
        "n_target_periods": int(len(frame)),
    }


def emma_value(frame: pd.DataFrame, ex_keV: int) -> tuple[float, float]:
    selected = frame.loc[frame["Ex_keV"] == ex_keV]
    if len(selected) != 1:
        raise ValueError(f"Expected one EMMA efficiency row for Ex={ex_keV} keV")
    row = selected.iloc[0]
    return float(row["efficiency"]), float(row["uncertainty"])


def interpolate_pair(
    energy_mev: float,
    low_energy_mev: float,
    low_value: float,
    high_energy_mev: float,
    high_value: float,
) -> tuple[float, float]:
    weight = (energy_mev - low_energy_mev) / (high_energy_mev - low_energy_mev)
    value = low_value + weight * (high_value - low_value)
    return float(value), float(weight)


def exact_dwba_and_acceptance(component: str) -> dict[str, float | int | str]:
    dwba = ROOT / "input" / component / "fort.202"
    truth = OUTPUT_ROOT / f"{component}_truth_kinematics.csv"
    exits = OUTPUT_ROOT / f"{component}_exit_kinematics.csv"
    theta, sigma = load_dwba_curve(dwba)
    acceptance = count_s3_acceptance(truth, exits)
    return {
        "dwba_total_mb": integrate_dwba_total_mb(theta, sigma),
        "s3_acceptance": float(acceptance["s3_acceptance_fraction"]),
        "s3_acceptance_unc": float(acceptance["s3_acceptance_fraction_unc"]),
        "generated_events": int(acceptance["generated_events"]),
        "dwba_method": f"exact: {component}",
        "s3_acceptance_method": f"exact: {component}",
    }


def interpolated_state_inputs(
    energy_mev: float,
    low_component: str,
    low_energy: float,
    high_component: str,
    high_energy: float,
) -> dict[str, float | int | str]:
    low_theta, low_sigma = load_dwba_curve(ROOT / "input" / low_component / "fort.202")
    high_theta, high_sigma = load_dwba_curve(ROOT / "input" / high_component / "fort.202")
    if not np.array_equal(low_theta, high_theta):
        high_sigma = np.interp(low_theta, high_theta, high_sigma)
    weight = (energy_mev - low_energy) / (high_energy - low_energy)
    sigma = low_sigma + weight * (high_sigma - low_sigma)

    low_acceptance = exact_dwba_and_acceptance(low_component)
    high_acceptance = exact_dwba_and_acceptance(high_component)
    acceptance, _ = interpolate_pair(
        energy_mev,
        low_energy,
        float(low_acceptance["s3_acceptance"]),
        high_energy,
        float(high_acceptance["s3_acceptance"]),
    )
    acceptance_unc = math.hypot(
        (1.0 - weight) * float(low_acceptance["s3_acceptance_unc"]),
        weight * float(high_acceptance["s3_acceptance_unc"]),
    )

    return {
        "dwba_total_mb": integrate_dwba_total_mb(low_theta, sigma),
        "s3_acceptance": acceptance,
        "s3_acceptance_unc": acceptance_unc,
        "generated_events": int(
            min(low_acceptance["generated_events"], high_acceptance["generated_events"])
        ),
        "dwba_method": (
            f"linear {low_component} to {high_component}; weight={weight:.9f}"
        ),
        "s3_acceptance_method": (
            f"linear {low_component} to {high_component}; weight={weight:.9f}"
        ),
        "interpolation_weight": float(weight),
    }


def interpolated_2plus_inputs(energy_mev: float) -> dict[str, float | int | str]:
    return interpolated_state_inputs(
        energy_mev,
        "10824_2plus",
        10.824,
        "11361_2plus",
        11.361,
    )


def denominator_relative_uncertainties(
    value: float, uncertainty: float
) -> tuple[float, float]:
    return inverse_quantity_relative_uncertainties(
        value=value,
        unc_plus=uncertainty,
        unc_minus=uncertainty,
    )


def calculate_state(
    fit_row: pd.Series,
    state_inputs: dict[str, float | int | str],
    luminosity: dict[str, float],
    emma_efficiency: float,
    emma_uncertainty: float,
    emma_method: str,
    q12_fraction: float,
    q12_relative_uncertainty: float,
    dwba_relative_uncertainty: float,
    fit_model_relative_uncertainty: float,
    interpolation_relative_uncertainty: float,
    s3_efficiency_relative_uncertainty: float,
    ssb_background_relative_uncertainty: float,
) -> dict[str, float | int | str]:
    fitted_yield = float(fit_row["MCMC_N_median"])
    fitted_minus = float(fit_row["MCMC_minus"])
    fitted_plus = float(fit_row["MCMC_plus"])
    s3_acceptance = float(state_inputs["s3_acceptance"])
    s3_acceptance_unc = float(state_inputs["s3_acceptance_unc"])
    sigma_total_mb = float(state_inputs["dwba_total_mb"])
    luminosity_value = luminosity["luminosity_ub_inv"]

    data_efficiency = (
        S3_EFFICIENCY
        * PGAC_IC_EFFICIENCY
        * TOF_EFFICIENCY
        * q12_fraction
        * emma_efficiency
    )
    unit_strength_observed_yield = (
        luminosity_value
        * 1000.0
        * sigma_total_mb
        * s3_acceptance
        * data_efficiency
    )
    spectroscopic_factor = fitted_yield / unit_strength_observed_yield

    yield_rel_minus = fitted_minus / fitted_yield
    yield_rel_plus = fitted_plus / fitted_yield
    luminosity_stat_rel = luminosity["luminosity_stat_ub_inv"] / luminosity_value
    luminosity_syst_rel = luminosity["luminosity_syst_ub_inv"] / luminosity_value
    lum_stat_minus, lum_stat_plus = denominator_relative_uncertainties(
        luminosity_value, luminosity["luminosity_stat_ub_inv"]
    )
    lum_syst_minus, lum_syst_plus = denominator_relative_uncertainties(
        luminosity_value, luminosity["luminosity_syst_ub_inv"]
    )
    emma_minus, emma_plus = denominator_relative_uncertainties(
        emma_efficiency, emma_uncertainty
    )
    q12_uncertainty = q12_fraction * q12_relative_uncertainty
    q12_minus, q12_plus = denominator_relative_uncertainties(
        q12_fraction, q12_uncertainty
    )
    dwba_minus, dwba_plus = denominator_relative_uncertainties(
        sigma_total_mb, sigma_total_mb * dwba_relative_uncertainty
    )
    fit_model_minus = fit_model_relative_uncertainty
    fit_model_plus = fit_model_relative_uncertainty
    interpolation_applied = "linear" in str(state_inputs["dwba_method"])
    interpolation_relative = (
        interpolation_relative_uncertainty if interpolation_applied else 0.0
    )
    interpolation_minus, interpolation_plus = denominator_relative_uncertainties(
        1.0, interpolation_relative
    )
    s3_efficiency_minus, s3_efficiency_plus = denominator_relative_uncertainties(
        1.0, s3_efficiency_relative_uncertainty
    )
    ssb_background_minus, ssb_background_plus = denominator_relative_uncertainties(
        1.0, ssb_background_relative_uncertainty
    )
    acceptance_minus, acceptance_plus = denominator_relative_uncertainties(
        s3_acceptance, s3_acceptance_unc
    )
    tof_minus, tof_plus = denominator_relative_uncertainties(
        TOF_EFFICIENCY, TOF_EFFICIENCY_UNC
    )
    pgac_minus, pgac_plus = inverse_quantity_relative_uncertainties(
        PGAC_IC_EFFICIENCY,
        PGAC_IC_UNC_PLUS,
        PGAC_IC_UNC_MINUS,
    )
    total_rel_minus = math.sqrt(
        yield_rel_minus**2
        + lum_stat_minus**2
        + lum_syst_minus**2
        + emma_minus**2
        + q12_minus**2
        + dwba_minus**2
        + acceptance_minus**2
        + tof_minus**2
        + pgac_minus**2
        + fit_model_minus**2
        + interpolation_minus**2
        + s3_efficiency_minus**2
        + ssb_background_minus**2
    )
    total_rel_plus = math.sqrt(
        yield_rel_plus**2
        + lum_stat_plus**2
        + lum_syst_plus**2
        + emma_plus**2
        + q12_plus**2
        + dwba_plus**2
        + acceptance_plus**2
        + tof_plus**2
        + pgac_plus**2
        + fit_model_plus**2
        + interpolation_plus**2
        + s3_efficiency_plus**2
        + ssb_background_plus**2
    )

    return {
        "component": str(fit_row["component"]),
        "Ex_MeV": float(fit_row["input_Ex_MeV"]),
        "Jpi": str(fit_row["Jpi"]),
        "MCMC_yield": fitted_yield,
        "MCMC_yield_minus": fitted_minus,
        "MCMC_yield_plus": fitted_plus,
        "luminosity_ub-1": luminosity_value,
        "luminosity_stat_ub-1": luminosity["luminosity_stat_ub_inv"],
        "luminosity_syst_ub-1": luminosity["luminosity_syst_ub_inv"],
        "luminosity_stat_relative": luminosity_stat_rel,
        "luminosity_syst_relative": luminosity_syst_rel,
        "dwba_total_mb": sigma_total_mb,
        "dwba_method": str(state_inputs["dwba_method"]),
        "dwba_relative_uncertainty": dwba_relative_uncertainty,
        "fit_model_relative_uncertainty": fit_model_relative_uncertainty,
        "interpolation_relative_uncertainty": interpolation_relative,
        "s3_efficiency_relative_uncertainty": s3_efficiency_relative_uncertainty,
        "ssb_background_relative_uncertainty": ssb_background_relative_uncertainty,
        "s3_acceptance": s3_acceptance,
        "s3_acceptance_unc": s3_acceptance_unc,
        "s3_acceptance_method": str(state_inputs["s3_acceptance_method"]),
        "emma_transmission": emma_efficiency,
        "emma_transmission_unc": emma_uncertainty,
        "emma_method": emma_method,
        "q12_fraction": q12_fraction,
        "q12_fraction_unc": q12_uncertainty,
        "q12_relative_uncertainty": q12_relative_uncertainty,
        "s3_efficiency": S3_EFFICIENCY,
        "pgac_ic_efficiency": PGAC_IC_EFFICIENCY,
        "tof_efficiency": TOF_EFFICIENCY,
        "total_observed_efficiency": data_efficiency,
        "unit_strength_observed_yield": unit_strength_observed_yield,
        "alpha_spectroscopic_factor": spectroscopic_factor,
        "alpha_spectroscopic_factor_fit_minus": spectroscopic_factor * yield_rel_minus,
        "alpha_spectroscopic_factor_fit_plus": spectroscopic_factor * yield_rel_plus,
        "alpha_spectroscopic_factor_total_minus": spectroscopic_factor * total_rel_minus,
        "alpha_spectroscopic_factor_total_plus": spectroscopic_factor * total_rel_plus,
        "total_relative_uncertainty_minus": total_rel_minus,
        "total_relative_uncertainty_plus": total_rel_plus,
        "dwba_model_uncertainty_included": True,
    }


def select_fit_row(frame: pd.DataFrame, component: str) -> pd.Series:
    selected = frame.loc[frame["component"] == component]
    if len(selected) != 1:
        raise ValueError(f"Expected one fitted row for {component}")
    return selected.iloc[0]


def apply_selected_yields(fits: pd.DataFrame, path: Path) -> pd.DataFrame:
    selected = pd.read_csv(path)
    required = {
        "component",
        "MCMC_N_median",
        "MCMC_minus",
        "MCMC_plus",
    }
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(
            f"Selected-yields file is missing columns: {sorted(missing)}"
        )

    updated = fits.copy()
    for row in selected.itertuples(index=False):
        mask = updated["component"] == row.component
        if int(mask.sum()) != 1:
            raise ValueError(
                f"Expected one fitted row for selected component {row.component}"
            )
        updated.loc[
            mask, ["MCMC_N_median", "MCMC_minus", "MCMC_plus"]
        ] = [row.MCMC_N_median, row.MCMC_minus, row.MCMC_plus]
    return updated


def add_dwba_uncertainty_to_existing_results(
    results: pd.DataFrame, relative_uncertainty: float
) -> pd.DataFrame:
    updated = results.copy()
    if "dwba_model_uncertainty_included" in updated.columns:
        included = updated["dwba_model_uncertainty_included"].astype(str).str.lower()
        if included.eq("true").all():
            existing = pd.to_numeric(
                updated.get("dwba_relative_uncertainty"), errors="coerce"
            )
            if existing.notna().all() and np.allclose(existing, relative_uncertainty):
                return updated
            raise ValueError(
                "Results CSV already includes a different DWBA model uncertainty; "
                "rerun without --reuse-results-csv."
            )

    dwba_minus, dwba_plus = denominator_relative_uncertainties(
        1.0, relative_uncertainty
    )
    central = updated["alpha_spectroscopic_factor"].to_numpy(float)
    old_minus = updated["total_relative_uncertainty_minus"].to_numpy(float)
    old_plus = updated["total_relative_uncertainty_plus"].to_numpy(float)
    new_minus = np.sqrt(old_minus**2 + dwba_minus**2)
    new_plus = np.sqrt(old_plus**2 + dwba_plus**2)
    updated["total_relative_uncertainty_minus"] = new_minus
    updated["total_relative_uncertainty_plus"] = new_plus
    updated["alpha_spectroscopic_factor_total_minus"] = central * new_minus
    updated["alpha_spectroscopic_factor_total_plus"] = central * new_plus
    updated["dwba_relative_uncertainty"] = relative_uncertainty
    updated["dwba_model_uncertainty_included"] = True
    return updated


def apply_emma_live_time_correction(
    results: pd.DataFrame, accepted: int, presented: int
) -> pd.DataFrame:
    """Correct a shared efficiency, retaining the existing asymmetric error budget.

    The binomial counting uncertainty is a common normalization uncertainty,
    fully correlated between states. Counts must represent the yield dataset.
    This does not estimate systematic effects from missing scaler coverage.
    """
    if not 0 < accepted <= presented:
        raise ValueError("EMMA counts must satisfy 0 < accepted <= presented")
    updated = results.copy()
    if updated.empty:
        raise ValueError("Cannot correct an empty results table")
    marker = "emma_livetime_correction_applied"
    if marker in updated:
        included = updated[marker].astype(str).str.lower()
        if included.eq("true").all():
            same_counts = all(
                column in updated and pd.to_numeric(updated[column]).eq(value).all()
                for column, value in (
                    ("emma_livetime_accepted", accepted),
                    ("emma_livetime_presented", presented),
                )
            )
            if same_counts:
                return updated
            raise ValueError(
                "Results already include a different or undocumented live-time "
                "correction; rerun without --reuse-results-csv."
            )
        if not included.eq("false").all():
            raise ValueError("Inconsistent live-time correction markers in results")

    live = accepted / presented
    live_unc = math.sqrt(live * (1.0 - live) / presented)
    live_minus, live_plus = denominator_relative_uncertainties(live, live_unc)
    updated["alpha_spectroscopic_factor_before_livetime"] = updated[
        "alpha_spectroscopic_factor"
    ]
    for column in ("total_observed_efficiency", "unit_strength_observed_yield"):
        updated[column] = updated[column] * live
    for column in (
        "alpha_spectroscopic_factor",
        "alpha_spectroscopic_factor_fit_minus",
        "alpha_spectroscopic_factor_fit_plus",
    ):
        updated[column] = updated[column] / live
    for side, contribution in (("minus", live_minus), ("plus", live_plus)):
        relative = f"total_relative_uncertainty_{side}"
        updated[relative] = np.hypot(updated[relative], contribution)
        updated[f"alpha_spectroscopic_factor_total_{side}"] = (
            updated["alpha_spectroscopic_factor"] * updated[relative]
        )
        updated[f"emma_livetime_relative_uncertainty_{side}"] = contribution
    updated["emma_livetime_accepted"] = accepted
    updated["emma_livetime_presented"] = presented
    updated["emma_live_fraction"] = live
    updated["emma_live_fraction_stat_unc"] = live_unc
    updated["emma_livetime_correction_factor"] = 1.0 / live
    updated["emma_livetime_uncertainty_correlation"] = "common across states"
    updated[marker] = True
    return updated


def plot_results(
    results: pd.DataFrame,
    spin_comparison: pd.DataFrame,
    literature: pd.DataFrame,
    output_path: Path,
    log_scale: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 15,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.97 if log_scale else 0.92)
    if not log_scale and "emma_live_fraction" in results:
        live = float(results.iloc[0]["emma_live_fraction"])
        fig.text(
            0.55, 0.965, f"EMMA live-time corrected: live fraction = {100 * live:.4f}%",
            ha="center", va="center", fontsize=11,
        )
    def state_position(energy_mev: float) -> float:
        return 0.0 if energy_mev < 11.0 else 1.0

    this_work_color = "#0072B2"
    for row in results.itertuples():
        x = state_position(float(row.Ex_MeV))
        ax.errorbar(
            x,
            float(row.alpha_spectroscopic_factor),
            yerr=np.asarray(
                [
                    [float(row.alpha_spectroscopic_factor_total_minus)],
                    [float(row.alpha_spectroscopic_factor_total_plus)],
                ]
            ),
            fmt="o",
            ms=8,
            mfc=this_work_color,
            mec="black",
            mew=0.7,
            color=this_work_color,
            ecolor=this_work_color,
            elinewidth=1.8,
            capsize=4,
            capthick=1.8,
            zorder=5,
        )

    style = {
        "Talwar et al.": ("#D55E00", "s"),
        "Ugalde et al. / Longland et al.": ("#009E73", "v"),
        "Ota et al.": ("#CC79A7", "v"),
    }
    for row in literature.itertuples():
        color, marker = style[str(row.study)]
        x = state_position(float(row.Ex_MeV))
        y = float(row.S_alpha)
        is_upper_limit = str(row.is_upper_limit).strip().lower() == "true"
        if is_upper_limit:
            ax.scatter(
                x,
                y,
                marker=marker,
                s=72,
                facecolors="white",
                edgecolors=color,
                linewidths=1.6,
                zorder=5,
            )
            ax.annotate(
                "",
                xy=(x, y / 1.85),
                xytext=(x, y / 1.04),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
                zorder=4,
            )
        else:
            ax.scatter(
                x,
                y,
                marker=marker,
                s=72,
                facecolors=color,
                edgecolors="black",
                linewidths=0.7,
                zorder=5,
            )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=this_work_color,
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=8,
            label="This work",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="#D55E00",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=8,
            label="Talwar et al. (2016)",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#009E73",
            markeredgewidth=1.6,
            markersize=8,
            label="Ugalde et al. (2007)",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#CC79A7",
            markeredgewidth=1.6,
            markersize=8,
            label="Ota et al. (2020)",
        ),
    ]

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(0.003, 2.0)
    else:
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.7)
        ax.grid(axis="y", which="minor", color="#EEEEEE", linewidth=0.45)
    ax.set_xlim(-0.48, 1.48)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([r"$10.949\;(1^-)$", r"$11.17\;(2^+)$"])
    fig.text(0.55, 0.055, r"$E_x\ \mathrm{(MeV)}$", ha="center", va="center")
    ax.set_ylabel(r"$S_{\alpha}$")
    ax.legend(handles=handles, loc="upper left", frameon=False, handletextpad=0.7)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    luminosity = read_luminosity(args.luminosity_summary, args.luminosity_systematics)
    if not 0.0 < args.q12_fraction <= 1.0:
        raise ValueError("--q12-fraction must be in (0, 1]")
    if args.q12_relative_uncertainty < 0.0:
        raise ValueError("--q12-relative-uncertainty must be non-negative")
    if not 0.0 <= args.dwba_relative_uncertainty < 1.0:
        raise ValueError("--dwba-relative-uncertainty must be in [0, 1)")
    for option, value in (
        ("--fit-model-relative-uncertainty", args.fit_model_relative_uncertainty),
        ("--interpolation-relative-uncertainty", args.interpolation_relative_uncertainty),
        ("--s3-efficiency-relative-uncertainty", args.s3_efficiency_relative_uncertainty),
        ("--ssb-background-relative-uncertainty", args.ssb_background_relative_uncertainty),
    ):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{option} must be in [0, 1)")
    if args.reuse_results_csv:
        results = add_dwba_uncertainty_to_existing_results(
            pd.read_csv(args.output_csv), args.dwba_relative_uncertainty
        )
        spin_comparison = add_dwba_uncertainty_to_existing_results(
            pd.read_csv(args.spin_comparison_csv), args.dwba_relative_uncertainty
        )
    else:
        fits = apply_selected_yields(
            pd.read_csv(args.fit_summary), args.selected_yields
        )
        emma = pd.read_csv(args.emma_efficiency)
        exact_emma, exact_emma_unc = emma_value(emma, 10949)
        low_emma, low_emma_unc = emma_value(emma, 10824)
        high_emma, high_emma_unc = emma_value(emma, 11361)
        interp_emma, weight = interpolate_pair(
            11.1717, 10.824, low_emma, 11.361, high_emma
        )
        interp_emma_unc = (1.0 - weight) * low_emma_unc + weight * high_emma_unc

        high_1minus_emma, high_1minus_emma_unc = emma_value(emma, 11183)
        interp_1minus_emma, weight_1minus = interpolate_pair(
            11.1717, 10.949, exact_emma, 11.183, high_1minus_emma
        )
        interp_1minus_emma_unc = (
            (1.0 - weight_1minus) * exact_emma_unc
            + weight_1minus * high_1minus_emma_unc
        )

        records = [
            calculate_state(
                select_fit_row(fits, "10949_1minus"),
                exact_dwba_and_acceptance("10949_1minus"),
                luminosity,
                exact_emma,
                exact_emma_unc,
                "exact 10.949-MeV value",
                args.q12_fraction,
                args.q12_relative_uncertainty,
                args.dwba_relative_uncertainty,
                args.fit_model_relative_uncertainty,
                args.interpolation_relative_uncertainty,
                args.s3_efficiency_relative_uncertainty,
                args.ssb_background_relative_uncertainty,
            ),
            calculate_state(
                select_fit_row(fits, "11172_2plus"),
                interpolated_2plus_inputs(11.1717),
                luminosity,
                interp_emma,
                interp_emma_unc,
                (
                    "linear 10.824- to 11.361-MeV 2+ interpolation; "
                    f"weight={weight:.9f}"
                ),
                args.q12_fraction,
                args.q12_relative_uncertainty,
                args.dwba_relative_uncertainty,
                args.fit_model_relative_uncertainty,
                args.interpolation_relative_uncertainty,
                args.s3_efficiency_relative_uncertainty,
                args.ssb_background_relative_uncertainty,
            ),
        ]
        alternative_fit_row = select_fit_row(fits, "11172_2plus").copy()
        alternative_fit_row["component"] = "11172_1minus_hypothesis"
        alternative_fit_row["Jpi"] = "1-"
        alternative_1minus = calculate_state(
            alternative_fit_row,
            interpolated_state_inputs(
                11.1717,
                "10949_1minus",
                10.949,
                "11183_1minus",
                11.183,
            ),
            luminosity,
            interp_1minus_emma,
            interp_1minus_emma_unc,
            (
                "linear 10.949- to 11.183-MeV 1- interpolation; "
                f"weight={weight_1minus:.9f}"
            ),
            args.q12_fraction,
            args.q12_relative_uncertainty,
            args.dwba_relative_uncertainty,
            args.fit_model_relative_uncertainty,
            args.interpolation_relative_uncertainty,
            args.s3_efficiency_relative_uncertainty,
            args.ssb_background_relative_uncertainty,
        )
        results = pd.DataFrame.from_records(records)
        spin_comparison = pd.DataFrame.from_records([records[1], alternative_1minus])
    results = apply_emma_live_time_correction(
        results, args.emma_accepted, args.emma_presented
    )
    spin_comparison = apply_emma_live_time_correction(
        spin_comparison, args.emma_accepted, args.emma_presented
    )
    baseline = float(spin_comparison.iloc[0]["alpha_spectroscopic_factor"])
    spin_comparison["ratio_to_2plus"] = (
        spin_comparison["alpha_spectroscopic_factor"] / baseline
    )
    spin_comparison["percent_change_from_2plus"] = (
        spin_comparison["ratio_to_2plus"] - 1.0
    ) * 100.0
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    args.spin_comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    spin_comparison.to_csv(args.spin_comparison_csv, index=False)
    if not args.skip_plot:
        plot_results(
            results,
            spin_comparison,
            pd.read_csv(args.literature_values),
            args.output_pdf,
            log_scale=True,
        )
        plot_results(
            results,
            spin_comparison,
            pd.read_csv(args.literature_values),
            args.output_linear_pdf,
            log_scale=False,
        )

    print(
        f"EMMA live fraction: {results.iloc[0]['emma_live_fraction']:.10f} +/- "
        f"{results.iloc[0]['emma_live_fraction_stat_unc']:.10f} (binomial stat); "
        f"S_alpha correction: {results.iloc[0]['emma_livetime_correction_factor']:.10f}"
    )
    print(
        "Combined SSB luminosity: "
        f"{luminosity['luminosity_ub_inv']:.6f} +/- "
        f"{luminosity['luminosity_stat_ub_inv']:.6f} stat +/- "
        f"{luminosity['luminosity_syst_ub_inv']:.6f} syst ub^-1"
    )
    for row in results.itertuples():
        print(
            f"Ex={row.Ex_MeV:.4f} MeV ({row.Jpi}): "
            f"S_alpha={row.alpha_spectroscopic_factor:.8f} "
            f"-{row.alpha_spectroscopic_factor_total_minus:.8f} "
            f"+{row.alpha_spectroscopic_factor_total_plus:.8f}"
        )
    print("11.1717-MeV spin-hypothesis comparison:")
    for row in spin_comparison.itertuples():
        print(
            f"  Jpi={row.Jpi}: S_alpha={row.alpha_spectroscopic_factor:.8f} "
            f"-{row.alpha_spectroscopic_factor_total_minus:.8f} "
            f"+{row.alpha_spectroscopic_factor_total_plus:.8f}; "
            f"ratio to 2+={row.ratio_to_2plus:.6f}"
        )
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved spin comparison: {args.spin_comparison_csv}")
    if not args.skip_plot:
        print(f"Saved PDF: {args.output_pdf}")
        print(f"Saved linear PDF: {args.output_linear_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
