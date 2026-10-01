#!/usr/bin/env python3
"""Run the slide-state full-range inputHist fits for all rings and ring chunks.

This driver writes one figure per histogram:
MCMC_full_range_spectrum_individual_states_poisson_errors with a label prefix.
The publication-ready PDF figures are written to the top-level output directory,
and the detailed CSV/TXT fit outputs are kept in a fit_outputs subdirectory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
DEFAULT_DATA = OUT / "inputHist.root"
DEFAULT_STATES = OUT / "MCMC_full_excitation_range" / "pptx_snapshot_states.csv"
DEFAULT_FIT_RESULTS = OUT / "Exc_sloped_box_fit" / "sloped_box_fit_results.csv"
DEFAULT_RING_CHUNK_FIT_RESULTS = (
    OUT / "Exc_by_ring_chunks" / "ring_chunk_sloped_box_fit_results.csv"
)
DEFAULT_OUTDIR = OUT / "MCMC_full_excitation_range_inputHist"
DEFAULT_FIT_LO_MEV = 2.4
DEFAULT_FIT_HI_MEV = 13.5
DEFAULT_FUSION_PROTON_BANDWIDTH_MEV = 0.10
DEFAULT_FUSION_PROTON_ENERGY_SHIFT_MEV = -0.40
DEFAULT_LOW_ENERGY_FLOOR_HI_MEV = 0.50
DEFAULT_LOW_ENERGY_FLOOR_MAX_FRACTION = 0.05
DEFAULT_COMPONENT_ENERGY_SHIFTS = ("6745_2plus:0.06",)
DEFAULT_COMPONENT_LEFT_EDGE_SHIFTS = ()
DEFAULT_COMPONENT_RIGHT_EDGE_SHIFTS = ("6256_0plus:-0.05",)
DEFAULT_COMPONENT_SIGMA_SCALES = ()
DEFAULT_FORCE_INTERPOLATE_COMPONENTS = ("10573_1minus",)
DEFAULT_RING_CHUNK_GLOBAL_ENERGY_SHIFTS_MEV = {
    "RR0": -0.02,
    "RR1": -0.02,
    "RR2": 0.02,
    "RR3": 0.08,
    "RR4": -0.06,
}
DEFAULT_RING_CHUNK_GLOBAL_SIGMA_SCALES = {}
DEFAULT_OPTIMIZED_RING_CHUNK_TEMPLATE_LABELS = ("RR3",)

INCLUSIVE_HISTOGRAMS = (
    ("all", "hExcE_PGACL_S3Rec1", "all_rings"),
    ("RR0", "hExcE_RR0_PGACL", "rings_01_04"),
    ("RR1", "hExcE_RR1_PGACL", "rings_05_08"),
    ("RR2", "hExcE_RR2_PGACL", "rings_09_12"),
    ("RR3", "hExcE_RR3_PGACL", "rings_13_15"),
    ("RR4", "hExcE_RR4_PGACL", "rings_19_24"),
)
GAMMA_GATES = ("1003", "1130", "1808", "2524")
GAMMA_GATED_HISTOGRAMS = tuple(
    (f"g{gate}_all", f"hExcE_{gate}_PGACL_Recoil", "all_rings")
    for gate in GAMMA_GATES
) + tuple(
    (f"g{gate}_{label}", f"hExcE_{label}_{gate}_PGACL", template_ring_group)
    for gate in GAMMA_GATES
    for label, _, template_ring_group in INCLUSIVE_HISTOGRAMS
    if label != "all"
)
HISTOGRAMS = INCLUSIVE_HISTOGRAMS + GAMMA_GATED_HISTOGRAMS

SUMMARY_PATTERNS = {
    "fit_range_MeV": re.compile(r"^fit_range_MeV:\s+(.+)$"),
    "n_data_in_fit_range": re.compile(r"^n_data_in_fit_range:\s+(\d+)"),
    "bin_width_MeV": re.compile(r"^bin_width_MeV:\s+(.+)$"),
    "n_bins": re.compile(r"^n_bins:\s+(\d+)"),
    "chi2_ge5": re.compile(r"^ pearson_expected_ge_5_chi2:\s+(.+)$"),
    "ndf_ge5": re.compile(r"^ pearson_expected_ge_5_ndf:\s+(.+)$"),
    "chi2_ndf_ge5": re.compile(r"^ pearson_expected_ge_5_chi2_ndf:\s+(.+)$"),
    "p_ge5": re.compile(r"^ pearson_expected_ge_5_p_value:\s+(.+)$"),
    "poisson_deviance": re.compile(r"^ poisson_deviance:\s+(.+)$"),
    "poisson_ndf": re.compile(r"^ poisson_deviance_ndf:\s+(.+)$"),
    "poisson_deviance_ndf": re.compile(r"^ poisson_deviance_per_ndf:\s+(.+)$"),
    "poisson_p": re.compile(r"^ poisson_deviance_p_value:\s+(.+)$"),
    "ks_D": re.compile(r"^ ks_D:\s+(.+)$"),
    "ks_p_naive": re.compile(r"^ ks_p_naive_fixed_cdf:\s+(.+)$"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit output/inputHist.root for all rings and RR0-RR4, writing only "
            "the individual-state Poisson-error residual spectrum plot for each."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--states-csv", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--fit-results-csv", type=Path, default=DEFAULT_FIT_RESULTS)
    parser.add_argument(
        "--histogram-set",
        choices=("inclusive", "gamma", "all"),
        default="inclusive",
        help=(
            "Histogram group to fit. 'inclusive' is the standard six spectra, "
            "'gamma' fits the 1003/1130/1808/2524 gamma-gated spectra, and "
            "'all' runs both groups."
        ),
    )
    parser.add_argument(
        "--ring-chunk-fit-results-csv",
        type=Path,
        default=DEFAULT_RING_CHUNK_FIT_RESULTS,
        help="Sloped-box parameter CSV used for RR0-RR4 when chunk templates are enabled.",
    )
    parser.add_argument(
        "--use-ring-chunk-templates",
        action="store_true",
        help=(
            "Use ring-chunk sloped-box templates for RR0-RR4. This is opt-in "
            "because the current chunk template table worsens several chunks."
        ),
    )
    parser.add_argument(
        "--no-ring-chunk-templates",
        action="store_true",
        help=(
            "Disable ring-chunk templates for all chunks, including the optimized "
            "per-chunk default."
        ),
    )
    parser.add_argument(
        "--no-optimized-ring-chunk-settings",
        action="store_true",
        help=(
            "Disable the default per-chunk global energy shifts, sigma scales, "
            "and optimized chunk-template selection."
        ),
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--fit-output-subdir", default="fit_outputs")
    parser.add_argument(
        "--fit-lo-MeV",
        type=float,
        default=DEFAULT_FIT_LO_MEV,
        help="Lower fit boundary in MeV.",
    )
    parser.add_argument(
        "--fit-hi-MeV",
        type=float,
        default=DEFAULT_FIT_HI_MEV,
        help="Upper fit boundary in MeV.",
    )
    parser.add_argument("--bin-width-MeV", type=float, default=0.10)
    parser.add_argument("--burnin-steps", type=int, default=1000)
    parser.add_argument("--production-steps", type=int, default=3000)
    parser.add_argument("--nwalkers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--fusion-proton-bandwidth-MeV",
        type=float,
        default=DEFAULT_FUSION_PROTON_BANDWIDTH_MEV,
        help="KDE bandwidth for the low-energy fusion-proton contaminant template.",
    )
    parser.add_argument(
        "--fusion-proton-energy-shift-MeV",
        type=float,
        default=DEFAULT_FUSION_PROTON_ENERGY_SHIFT_MEV,
        help=(
            "Shift applied to the low-energy fusion-proton contaminant KDE. "
            "Negative values move the contaminant template to lower excitation energy."
        ),
    )
    parser.add_argument(
        "--low-energy-floor-hi-MeV",
        type=float,
        default=DEFAULT_LOW_ENERGY_FLOOR_HI_MEV,
        help=(
            "Upper edge of the bounded low-energy floor background. The floor "
            "is passed to the fitter only when --fit-lo-MeV is below this value."
        ),
    )
    parser.add_argument(
        "--low-energy-floor-max-fraction",
        type=float,
        default=DEFAULT_LOW_ENERGY_FLOOR_MAX_FRACTION,
        help="Maximum fraction of fit counts assigned to the low-energy floor.",
    )
    parser.add_argument(
        "--component-energy-shift",
        action="append",
        default=list(DEFAULT_COMPONENT_ENERGY_SHIFTS),
        metavar="COMPONENT:SHIFT_MEV",
        help="Shift a named sloped-box component. Repeat for multiple components.",
    )
    parser.add_argument(
        "--component-left-edge-shift",
        action="append",
        default=list(DEFAULT_COMPONENT_LEFT_EDGE_SHIFTS),
        metavar="COMPONENT:SHIFT_MEV",
        help="Shift only E_L for a named sloped-box component.",
    )
    parser.add_argument(
        "--component-right-edge-shift",
        action="append",
        default=list(DEFAULT_COMPONENT_RIGHT_EDGE_SHIFTS),
        metavar="COMPONENT:SHIFT_MEV",
        help="Shift only E_R for a named sloped-box component.",
    )
    parser.add_argument(
        "--component-sigma-scale",
        action="append",
        default=list(DEFAULT_COMPONENT_SIGMA_SCALES),
        metavar="COMPONENT:SCALE",
        help="Scale sigma_L/sigma_R for a named sloped-box component.",
    )
    parser.add_argument(
        "--force-interpolate-component",
        action="append",
        default=list(DEFAULT_FORCE_INTERPOLATE_COMPONENTS),
        metavar="COMPONENT",
        help=(
            "Force a named component to use interpolated/extrapolated template "
            "parameters, even when a fitted template exists at the same energy."
        ),
    )
    parser.add_argument(
        "--per-fit-global-energy-shift",
        action="append",
        default=[],
        metavar="FIT:SHIFT_MEV",
        help=(
            "Override the global state-template energy shift for one fit label "
            "(for example RR2:0.02)."
        ),
    )
    parser.add_argument(
        "--per-fit-global-sigma-scale",
        action="append",
        default=[],
        metavar="FIT:SCALE",
        help=(
            "Override the global state-template sigma scale for one fit label "
            "(for example RR4:1.2)."
        ),
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=[label for label, _, _ in HISTOGRAMS],
        default=None,
        help="Run only the selected fit label. Repeat for multiple labels.",
    )
    return parser.parse_args()


def base_fit_label(label: str) -> str:
    parts = label.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("g"):
        return parts[1]
    return label


def parse_label_float_options(
    tokens: list[str],
    option_name: str,
    *,
    positive: bool = False,
) -> dict[str, float]:
    valid_labels = {label for label, _, _ in HISTOGRAMS} | {
        label for label, _, _ in INCLUSIVE_HISTOGRAMS
    }
    values: dict[str, float] = {}
    for token in tokens:
        label, sep, raw_value = token.partition(":")
        label = label.strip()
        if not sep or label not in valid_labels:
            raise ValueError(
                f"{option_name} must have the form FIT:VALUE where FIT is one of "
                + ", ".join(sorted(valid_labels))
            )
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid {option_name} value {raw_value!r}") from exc
        if positive and value <= 0.0:
            raise ValueError(f"{option_name} value must be positive for {label}")
        values[label] = value
    return values


def parse_goodness(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    in_mcmc_section = False
    for line in path.read_text().splitlines():
        if line.startswith("Goodness of fit using MCMC"):
            in_mcmc_section = True
            continue
        if line.startswith("Goodness of fit using Minuit"):
            in_mcmc_section = False
        for key, pattern in SUMMARY_PATTERNS.items():
            if key not in {
                "fit_range_MeV",
                "n_data_in_fit_range",
                "bin_width_MeV",
                "n_bins",
            } and not in_mcmc_section:
                continue
            match = pattern.match(line)
            if not match:
                continue
            raw = match.group(1).strip()
            if key in {"n_data_in_fit_range", "n_bins"}:
                values[key] = int(raw)
            elif key == "fit_range_MeV":
                values[key] = raw
            else:
                values[key] = float(raw)
    return values


def run_fit(
    args: argparse.Namespace,
    label: str,
    hist_name: str,
    template_ring_group: str,
    per_fit_global_energy_shifts: dict[str, float],
    per_fit_global_sigma_scales: dict[str, float],
) -> dict[str, object]:
    fit_base = args.outdir / args.fit_output_subdir
    fit_outdir = fit_base / label
    plot_path = (
        args.outdir
        / f"{label}_MCMC_full_range_spectrum_individual_states_poisson_errors.pdf"
    )
    corner_path = args.outdir / f"{label}_MCMC_full_range_astrophysical_region_corner.pdf"
    above10_corner_path = args.outdir / f"{label}_MCMC_full_range_above10MeV_corner.pdf"
    fit_outdir.mkdir(parents=True, exist_ok=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    optimized_template_labels = (
        set()
        if args.no_optimized_ring_chunk_settings
        else set(DEFAULT_OPTIMIZED_RING_CHUNK_TEMPLATE_LABELS)
    )
    base_label = base_fit_label(label)
    use_chunk_templates = (
        base_label != "all"
        and (args.use_ring_chunk_templates or base_label in optimized_template_labels)
        and not args.no_ring_chunk_templates
    )
    fit_results_csv = (
        args.ring_chunk_fit_results_csv if use_chunk_templates else args.fit_results_csv
    )
    command = [
        sys.executable,
        str(ROOT / "Plotting" / "mcmc_full_excitation_fit.py"),
        "--data",
        str(args.data),
        "--root-hist",
        hist_name,
        "--states-csv",
        str(args.states_csv),
        "--fit-results-csv",
        str(fit_results_csv),
        "--template-ring-group",
        template_ring_group if use_chunk_templates else "all_rings",
        "--outdir",
        str(fit_outdir),
        "--fit-lo-MeV",
        f"{args.fit_lo_MeV:g}",
        "--fit-hi-MeV",
        f"{args.fit_hi_MeV:g}",
        "--bin-width-MeV",
        f"{args.bin_width_MeV:g}",
        "--burnin-steps",
        str(args.burnin_steps),
        "--production-steps",
        str(args.production_steps),
        "--seed",
        str(args.seed),
        "--plot-selection",
        "individual-poisson-only",
        "--spectrum-plot-output",
        str(plot_path),
        "--astrophysical-corner-output",
        str(corner_path),
        "--above10-corner-output",
        str(above10_corner_path),
        "--fusion-proton-bandwidth-MeV",
        f"{args.fusion_proton_bandwidth_MeV:g}",
        "--fusion-proton-energy-shift-MeV",
        f"{args.fusion_proton_energy_shift_MeV:g}",
        "--global-energy-shift-MeV",
        f"{per_fit_global_energy_shifts.get(label, per_fit_global_energy_shifts.get(base_label, 0.0)):g}",
        "--global-sigma-scale",
        f"{per_fit_global_sigma_scales.get(label, per_fit_global_sigma_scales.get(base_label, 1.0)):g}",
    ]
    if args.fit_lo_MeV < args.low_energy_floor_hi_MeV:
        command.extend(
            [
                "--low-energy-floor-hi-MeV",
                f"{args.low_energy_floor_hi_MeV:g}",
                "--low-energy-floor-max-fraction",
                f"{args.low_energy_floor_max_fraction:g}",
            ]
        )
    if use_chunk_templates:
        command.extend(
            [
                "--fallback-fit-results-csv",
                str(args.fit_results_csv),
                "--fallback-template-ring-group",
                "all_rings",
            ]
        )
    for token in args.component_energy_shift:
        command.extend(["--component-energy-shift", token])
    for token in args.component_left_edge_shift:
        command.extend(["--component-left-edge-shift", token])
    for token in args.component_right_edge_shift:
        command.extend(["--component-right-edge-shift", token])
    for token in args.component_sigma_scale:
        command.extend(["--component-sigma-scale", token])
    for token in args.force_interpolate_component:
        command.extend(["--force-interpolate-component", token])
    if args.nwalkers is not None:
        command.extend(["--nwalkers", str(args.nwalkers)])

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-s2223")
    env.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib-s2223-xdg")
    subprocess.run(command, check=True, cwd=ROOT, env=env)

    metrics = parse_goodness(fit_outdir / "MCMC_full_range_goodness_of_fit.txt")
    return {
        "fit": label,
        "root_histogram": hist_name,
        "fit_results_csv_used": str(fit_results_csv),
        "template_ring_group": template_ring_group if use_chunk_templates else "all_rings",
        "global_energy_shift_MeV": per_fit_global_energy_shifts.get(
            label,
            per_fit_global_energy_shifts.get(base_label, 0.0),
        ),
        "global_sigma_scale": per_fit_global_sigma_scales.get(
            label,
            per_fit_global_sigma_scales.get(base_label, 1.0),
        ),
        "plot_file": str(plot_path),
        "corner_plot_file": str(corner_path),
        "above10_corner_plot_file": str(above10_corner_path),
        **metrics,
    }


def main() -> None:
    args = parse_args()
    custom_global_energy_shifts = parse_label_float_options(
        args.per_fit_global_energy_shift,
        "--per-fit-global-energy-shift",
    )
    custom_global_sigma_scales = parse_label_float_options(
        args.per_fit_global_sigma_scale,
        "--per-fit-global-sigma-scale",
        positive=True,
    )
    per_fit_global_energy_shifts = (
        {}
        if args.no_optimized_ring_chunk_settings
        else dict(DEFAULT_RING_CHUNK_GLOBAL_ENERGY_SHIFTS_MEV)
    )
    per_fit_global_sigma_scales = (
        {}
        if args.no_optimized_ring_chunk_settings
        else dict(DEFAULT_RING_CHUNK_GLOBAL_SIGMA_SCALES)
    )
    per_fit_global_energy_shifts.update(custom_global_energy_shifts)
    per_fit_global_sigma_scales.update(custom_global_sigma_scales)
    selected = set(args.only) if args.only else None
    histogram_source = {
        "inclusive": INCLUSIVE_HISTOGRAMS,
        "gamma": GAMMA_GATED_HISTOGRAMS,
        "all": HISTOGRAMS,
    }[args.histogram_set]
    histograms = [
        (label, hist_name, template_ring_group)
        for label, hist_name, template_ring_group in histogram_source
        if selected is None or label in selected
    ]
    rows = [
        run_fit(
            args,
            label,
            hist_name,
            template_ring_group,
            per_fit_global_energy_shifts,
            per_fit_global_sigma_scales,
        )
        for label, hist_name, template_ring_group in histograms
    ]
    summary = pd.DataFrame(rows)
    summary_path = args.outdir / "inputHist_fit_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
