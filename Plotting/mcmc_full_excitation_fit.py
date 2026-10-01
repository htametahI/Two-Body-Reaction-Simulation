#!/usr/bin/env python3
"""MCMC fit of the full excitation-energy spectrum using spreadsheet states.

The state templates are sloped smooth boxes.  Existing all-ring sloped-box
parameters are used directly when a spreadsheet state matches a fitted state;
otherwise the template parameters are linearly interpolated in excitation
energy from the existing fits.  States outside the fitted-template energy range
are linearly extrapolated from the nearest two fitted points and flagged in the
component CSV.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-s2223")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib-s2223-xdg")
logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)

import emcee
from iminuit import Minuit
import matplotlib as mpl

mpl.use("Agg")
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.special import gammaln
from scipy.stats import chi2 as chi2_dist
from scipy.stats import kstwo

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mcmc_fit_diagnostics import (  # noqa: E402
    FUSION_COMPONENT,
    bin_probabilities,
    component_pdf,
    load_sample_template_values,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
DEFAULT_DATA = OUT / "unbinned_all_rings_combined_PGACL_repeated.npz"
DEFAULT_STATE_CSV = OUT / "MCMC_full_excitation_range" / "pptx_snapshot_states.csv"
DEFAULT_FIT_RESULTS = OUT / "Exc_sloped_box_fit" / "sloped_box_fit_results.csv"
DEFAULT_OUTDIR = OUT / "MCMC_full_excitation_range"
DEFAULT_FIT_LO_MEV = None
LOW_PROTON_TEMPLATE_RANGE = (0.9, 6.5)
LOW_ENERGY_FLOOR_NAME = "low_energy_floor_bkg"
ASTROPHYSICAL_INSET_RANGE_MEV = (10.915, 11.364)
ASTROPHYSICAL_HIGHLIGHT_FACE = "#f7b6cf"
ASTROPHYSICAL_HIGHLIGHT_EDGE = "#bf3f73"
ASTROPHYSICAL_ZOOM_BOX_FACE = "#d9e6ef"
ASTROPHYSICAL_ZOOM_BOX_EDGE = "#315f7d"
ASTROPHYSICAL_ZOOM_BOX_FACE_ALPHA = 0.26

PARAM_COLS = (
    "E_L_MeV",
    "E_R_MeV",
    "sigma_L_MeV",
    "sigma_R_MeV",
    "slope_MeV_inv",
)
PARAM_TO_COMPONENT_KEY = {
    "E_L_MeV": "E_L",
    "E_R_MeV": "E_R",
    "sigma_L_MeV": "sigma_L",
    "sigma_R_MeV": "sigma_R",
    "slope_MeV_inv": "slope",
}
FUSION_HIGHLIGHT_COLOR = "crimson"
FUSION_HIGHLIGHT_COLORS = {
    "fusion_proton_low_bkg": "purple",
}


mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.width": 1.05,
        "ytick.major.width": 1.05,
        "xtick.minor.width": 0.9,
        "ytick.minor.width": 0.9,
        "legend.frameon": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the excitation-energy spectrum with MCMC using all states from "
            "the spreadsheet and sloped-box templates from existing fits."
        )
    )
    parser.add_argument("--states-csv", type=Path, default=DEFAULT_STATE_CSV)
    parser.add_argument("--fit-results-csv", type=Path, default=DEFAULT_FIT_RESULTS)
    parser.add_argument(
        "--template-ring-group",
        default="all_rings",
        help="Ring group to use from the sloped-box fit results. Defaults to all_rings.",
    )
    parser.add_argument(
        "--fallback-fit-results-csv",
        type=Path,
        default=None,
        help=(
            "Optional fallback sloped-box parameter CSV used when interpolated "
            "parameters from --fit-results-csv are invalid."
        ),
    )
    parser.add_argument(
        "--fallback-template-ring-group",
        default="all_rings",
        help="Ring group to read from --fallback-fit-results-csv.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--root-hist",
        default=None,
        help=(
            "Histogram name to read when --data points to a ROOT file. If omitted, "
            "the first TH1-like object in the file is used."
        ),
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--ring-selection",
        default="all",
        help='Pandas query for selected rings, or "all". Defaults to all S3 rings.',
    )
    parser.add_argument("--bin-width-MeV", type=float, default=0.10)
    parser.add_argument(
        "--fit-lo-MeV",
        type=float,
        default=DEFAULT_FIT_LO_MEV,
        help="Lower fit boundary in MeV. Defaults to the automatic full-range lower boundary.",
    )
    parser.add_argument("--fit-hi-MeV", type=float, default=None)
    parser.add_argument(
        "--support-nsigma",
        type=float,
        default=6.0,
        help="Automatic fit limits use EL/ER +/- this many edge sigmas.",
    )
    parser.add_argument("--grid-size", type=int, default=12000)
    parser.add_argument("--burnin-steps", type=int, default=1000)
    parser.add_argument("--production-steps", type=int, default=3000)
    parser.add_argument("--nwalkers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--max-total-yield-factor",
        type=float,
        default=2.5,
        help="Each component yield is bounded by this factor times n_data.",
    )
    parser.add_argument(
        "--match-tolerance-keV",
        type=float,
        default=3.0,
        help="Energy tolerance for using existing template parameters directly.",
    )
    parser.add_argument(
        "--force-interpolate-component",
        action="append",
        default=[],
        metavar="COMPONENT",
        help=(
            "Force a named component to use interpolated/extrapolated template "
            "parameters, even when a fitted template has the same excitation energy. "
            "Repeat for multiple components."
        ),
    )
    parser.add_argument(
        "--no-posterior-draws-plot",
        action="store_true",
        help="Skip the posterior draw overlay plot.",
    )
    parser.add_argument(
        "--plot-selection",
        choices=("all", "individual-poisson-only"),
        default="all",
        help=(
            "Select which diagnostic plots to write. 'individual-poisson-only' "
            "writes only MCMC_full_range_spectrum_individual_states_poisson_errors."
        ),
    )
    parser.add_argument(
        "--plot-format",
        dest="plot_formats",
        action="append",
        choices=("pdf", "png"),
        default=None,
        help=(
            "Plot format for standard outdir plots. Repeat to write multiple "
            "formats. Defaults to both pdf and png."
        ),
    )
    parser.add_argument(
        "--spectrum-plot-output",
        type=Path,
        default=None,
        help=(
            "Exact output path for the selected spectrum plot. Intended for "
            "--plot-selection individual-poisson-only so wrapper scripts can "
            "collect one named plot per histogram."
        ),
    )
    parser.add_argument(
        "--plot-uncertainty-scale",
        type=float,
        default=1.0,
        help=(
            "Plot-only common scale applied to Poisson count intervals. Pearson "
            "residuals and the displayed chi-square are divided by the same scale. "
            "This does not alter the likelihood or fitted yields."
        ),
    )
    parser.add_argument(
        "--likelihood-uncertainty-scale",
        type=float,
        default=1.0,
        help=(
            "Temper the binned Poisson log likelihood by this scale squared. "
            "Values above one broaden the sampled yield posterior while retaining "
            "the positive-yield prior and correlations."
        ),
    )
    parser.add_argument(
        "--astrophysical-corner-output",
        type=Path,
        default=None,
        help=(
            "Exact output path for a corner plot of MCMC yields for sloped-box "
            "states with input excitation energy in the astrophysical window."
        ),
    )
    parser.add_argument(
        "--above10-corner-output",
        type=Path,
        default=None,
        help=(
            "Exact output path for a corner plot of MCMC yields for sloped-box "
            "states with input excitation energy above 10 MeV."
        ),
    )
    parser.add_argument(
        "--fusion-proton-bandwidth-MeV",
        type=float,
        default=0.10,
        help="KDE bandwidth for the low-energy fusion-proton contaminant template.",
    )
    parser.add_argument(
        "--fusion-proton-energy-shift-MeV",
        type=float,
        default=0.0,
        help=(
            "Shift applied to the low-energy fusion-proton contaminant KDE. "
            "Negative values move the contaminant template to lower excitation energy."
        ),
    )
    parser.add_argument(
        "--component-energy-shift",
        action="append",
        default=[],
        metavar="COMPONENT:SHIFT_MEV",
        help=(
            "Shift a named sloped-box component by adding SHIFT_MEV to E_L and "
            "E_R. Repeat for multiple components."
        ),
    )
    parser.add_argument(
        "--component-left-edge-shift",
        action="append",
        default=[],
        metavar="COMPONENT:SHIFT_MEV",
        help=(
            "Shift only E_L for a named sloped-box component. "
            "Repeat for multiple components."
        ),
    )
    parser.add_argument(
        "--component-right-edge-shift",
        action="append",
        default=[],
        metavar="COMPONENT:SHIFT_MEV",
        help=(
            "Shift only E_R for a named sloped-box component. "
            "Repeat for multiple components."
        ),
    )
    parser.add_argument(
        "--component-sigma-scale",
        action="append",
        default=[],
        metavar="COMPONENT:SCALE",
        help=(
            "Scale sigma_L and sigma_R for a named sloped-box component. "
            "Repeat for multiple components."
        ),
    )
    parser.add_argument(
        "--global-energy-shift-MeV",
        type=float,
        default=0.0,
        help=(
            "Shift all sloped-box state templates by this energy after any "
            "component-specific shifts."
        ),
    )
    parser.add_argument(
        "--global-sigma-scale",
        type=float,
        default=1.0,
        help=(
            "Scale all sloped-box state template sigmas after any "
            "component-specific sigma scales."
        ),
    )
    parser.add_argument(
        "--low-energy-floor-hi-MeV",
        type=float,
        default=None,
        help=(
            "Optionally add a bounded flat background from the lower fit "
            "edge to this energy. This is useful for full-range fits that "
            "include data below the low-energy contaminant template support."
        ),
    )
    parser.add_argument(
        "--low-energy-floor-max-fraction",
        type=float,
        default=0.05,
        help="Maximum fraction of fit-range counts assigned to the low-energy floor.",
    )
    parser.add_argument(
        "--no-fusion-proton-contaminants",
        action="store_true",
        help=(
            "Disable the PACE4 low-energy proton sample-KDE contaminant template. Kept as a "
            "backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--no-fusion-contaminants",
        action="store_true",
        help="Disable the PACE4 low-energy proton sample-KDE contaminant template.",
    )
    return parser.parse_args()


def parse_component_energy_shifts(
    tokens: list[str],
    option_name: str = "--component-energy-shift",
) -> dict[str, float]:
    shifts: dict[str, float] = {}
    for token in tokens:
        name, sep, raw_shift = token.partition(":")
        if not sep or not name.strip():
            raise ValueError(
                f"{option_name} must have the form COMPONENT:SHIFT_MEV"
            )
        try:
            shift = float(raw_shift)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {option_name} shift {raw_shift!r} for {name!r}"
            ) from exc
        shifts[name.strip()] = shift
    return shifts


def parse_component_scales(tokens: list[str], option_name: str) -> dict[str, float]:
    scales: dict[str, float] = {}
    for token in tokens:
        name, sep, raw_scale = token.partition(":")
        if not sep or not name.strip():
            raise ValueError(f"{option_name} must have the form COMPONENT:SCALE")
        try:
            scale = float(raw_scale)
        except ValueError as exc:
            raise ValueError(f"Invalid {option_name} scale {raw_scale!r}") from exc
        if scale <= 0.0:
            raise ValueError(f"{option_name} scale must be positive for {name!r}")
        scales[name.strip()] = scale
    return scales


def apply_component_energy_shifts(
    components: list[dict[str, object]],
    component_df: pd.DataFrame,
    shifts: dict[str, float],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    if not shifts:
        return components, component_df

    component_df = component_df.copy()
    component_df["energy_shift_MeV"] = 0.0
    component_by_name = {str(component["name"]): component for component in components}
    unknown = sorted(set(shifts) - set(component_by_name))
    if unknown:
        raise ValueError(
            "Cannot shift unknown component(s): " + ", ".join(unknown)
        )

    for name, shift in shifts.items():
        component = component_by_name[name]
        if component.get("kind", "sloped_box") != "sloped_box":
            raise ValueError(f"--component-energy-shift only supports sloped boxes: {name}")
        component["E_L"] = float(component["E_L"]) + shift
        component["E_R"] = float(component["E_R"]) + shift
        component["energy_shift_MeV"] = shift
        mask = component_df["component"] == name
        component_df.loc[mask, ["E_L_MeV", "E_R_MeV"]] = [
            component["E_L"],
            component["E_R"],
        ]
        component_df.loc[mask, "energy_shift_MeV"] = shift
        component_df.loc[mask, "template_source"] = (
            component_df.loc[mask, "template_source"].astype(str) + "+shifted"
        )
    return components, component_df


def apply_component_edge_shifts(
    components: list[dict[str, object]],
    component_df: pd.DataFrame,
    left_shifts: dict[str, float],
    right_shifts: dict[str, float],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    if not left_shifts and not right_shifts:
        return components, component_df

    component_df = component_df.copy()
    if "E_L_shift_MeV" not in component_df:
        component_df["E_L_shift_MeV"] = 0.0
    if "E_R_shift_MeV" not in component_df:
        component_df["E_R_shift_MeV"] = 0.0

    component_by_name = {str(component["name"]): component for component in components}
    unknown = sorted((set(left_shifts) | set(right_shifts)) - set(component_by_name))
    if unknown:
        raise ValueError(
            "Cannot edge-shift unknown component(s): " + ", ".join(unknown)
        )

    for name in sorted(set(left_shifts) | set(right_shifts)):
        component = component_by_name[name]
        if component.get("kind", "sloped_box") != "sloped_box":
            raise ValueError(
                f"--component-left/right-edge-shift only supports sloped boxes: {name}"
            )
        left_shift = float(left_shifts.get(name, 0.0))
        right_shift = float(right_shifts.get(name, 0.0))
        component["E_L"] = float(component["E_L"]) + left_shift
        component["E_R"] = float(component["E_R"]) + right_shift
        component["E_L_shift_MeV"] = left_shift
        component["E_R_shift_MeV"] = right_shift
        if float(component["E_R"]) <= float(component["E_L"]):
            raise ValueError(
                f"Edge shifts make {name} invalid: E_R <= E_L "
                f"({component['E_R']:g} <= {component['E_L']:g})"
            )
        mask = component_df["component"] == name
        component_df.loc[mask, ["E_L_MeV", "E_R_MeV"]] = [
            component["E_L"],
            component["E_R"],
        ]
        component_df.loc[mask, "E_L_shift_MeV"] = left_shift
        component_df.loc[mask, "E_R_shift_MeV"] = right_shift
        component_df.loc[mask, "template_source"] = (
            component_df.loc[mask, "template_source"].astype(str) + "+edge_shifted"
        )
    return components, component_df


def apply_component_sigma_scales(
    components: list[dict[str, object]],
    component_df: pd.DataFrame,
    scales: dict[str, float],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    if not scales:
        return components, component_df

    component_df = component_df.copy()
    if "sigma_scale" not in component_df:
        component_df["sigma_scale"] = 1.0
    component_by_name = {str(component["name"]): component for component in components}
    unknown = sorted(set(scales) - set(component_by_name))
    if unknown:
        raise ValueError(
            "Cannot scale sigma for unknown component(s): " + ", ".join(unknown)
        )

    for name, scale in scales.items():
        component = component_by_name[name]
        if component.get("kind", "sloped_box") != "sloped_box":
            raise ValueError(f"--component-sigma-scale only supports sloped boxes: {name}")
        component["sigma_L"] = float(component["sigma_L"]) * scale
        component["sigma_R"] = float(component["sigma_R"]) * scale
        component["sigma_scale"] = scale
        mask = component_df["component"] == name
        component_df.loc[mask, ["sigma_L_MeV", "sigma_R_MeV"]] = [
            component["sigma_L"],
            component["sigma_R"],
        ]
        component_df.loc[mask, "sigma_scale"] = scale
        component_df.loc[mask, "template_source"] = (
            component_df.loc[mask, "template_source"].astype(str) + "+sigma_scaled"
        )
    return components, component_df


def apply_global_component_adjustments(
    components: list[dict[str, object]],
    component_df: pd.DataFrame,
    *,
    energy_shift_MeV: float = 0.0,
    sigma_scale: float = 1.0,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    if sigma_scale <= 0.0:
        raise ValueError("--global-sigma-scale must be positive")
    if energy_shift_MeV == 0.0 and sigma_scale == 1.0:
        return components, component_df

    component_df = component_df.copy()
    if "global_energy_shift_MeV" not in component_df:
        component_df["global_energy_shift_MeV"] = 0.0
    if "global_sigma_scale" not in component_df:
        component_df["global_sigma_scale"] = 1.0

    for component in components:
        if component.get("kind", "sloped_box") != "sloped_box":
            continue
        name = str(component["name"])
        component["E_L"] = float(component["E_L"]) + energy_shift_MeV
        component["E_R"] = float(component["E_R"]) + energy_shift_MeV
        component["sigma_L"] = float(component["sigma_L"]) * sigma_scale
        component["sigma_R"] = float(component["sigma_R"]) * sigma_scale
        mask = component_df["component"] == name
        component_df.loc[mask, ["E_L_MeV", "E_R_MeV"]] = [
            component["E_L"],
            component["E_R"],
        ]
        component_df.loc[mask, ["sigma_L_MeV", "sigma_R_MeV"]] = [
            component["sigma_L"],
            component["sigma_R"],
        ]
        component_df.loc[mask, "global_energy_shift_MeV"] = energy_shift_MeV
        component_df.loc[mask, "global_sigma_scale"] = sigma_scale
        component_df.loc[mask, "template_source"] = (
            component_df.loc[mask, "template_source"].astype(str)
            + "+global_adjusted"
        )
    return components, component_df


def jp_slug(jp: str) -> str:
    cleaned = jp.strip()
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace("->", "_to_")
    cleaned = cleaned.replace("+", "plus").replace("-", "minus")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def state_name(ex_mev: float, jp: str) -> str:
    return f"{int(round(1000.0 * ex_mev))}_{jp_slug(jp)}"


def latex_jp(jp: str) -> str:
    pieces = []
    for piece in re.split(r"\s*,\s*", jp.strip()):
        match = re.fullmatch(r"(?P<J>\d+)\s*(?P<parity>[+-])", piece)
        if match:
            pieces.append(rf"{match.group('J')}^{{{match.group('parity')}}}")
        elif piece:
            pieces.append(piece)
    return ", ".join(pieces) if pieces else jp


def component_label(component: dict[str, object], with_yield: bool = False) -> str:
    if is_fusion_component(component):
        ejectile = str(component.get("ejectile", "fusion"))
        if ejectile == "proton":
            return "proton contaminants"
        if ejectile == "alpha":
            label = r"$\alpha$-fusion contaminant"
        else:
            label = "fusion contaminant"
        if with_yield:
            label += "\n" + rf"$N={float(component['yield']):.0f}$"
        return label

    if component.get("kind", "sloped_box") != "sloped_box":
        label = str(component.get("name", "non-state component")).replace("_", " ")
        if with_yield:
            label += "\n" + rf"$N={float(component['yield']):.0f}$"
        return label

    ex = float(component["input_Ex_MeV"])
    jp = str(component["Jpi"])
    label = rf"${ex:.3f}\,\mathrm{{MeV}},\ J^\pi={latex_jp(jp)}$"
    if with_yield:
        label += "\n" + rf"$N={float(component['yield']):.0f}$"
    return label


def is_fusion_component(component: dict[str, object]) -> bool:
    return component.get("role") == "fusion_background" or str(
        component.get("name", "")
    ).startswith("fusion_")


def fusion_component_color(component: dict[str, object]) -> str:
    return FUSION_HIGHLIGHT_COLORS.get(str(component.get("name")), FUSION_HIGHLIGHT_COLOR)


def parse_first_float(value: object) -> float:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value))
    if not match:
        raise ValueError(f"Could not parse numeric value from {value!r}")
    return float(match.group(0))


def maybe_float(value: object) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return parse_first_float(text)
    except ValueError:
        return None


def normalized_header(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("26mg", "")
    text = text.replace("mg", "")
    text = text.replace("[%]", "percent")
    text = text.replace("[mev]", "mev")
    text = text.replace("[kev]", "kev")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def read_state_table(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        raw_rows = [
            raw
            for raw in csv.reader(handle)
            if raw and raw[0].strip() and not raw[0].strip().startswith("#")
        ]
    if not raw_rows:
        raise ValueError(f"No states found in {path}")

    first_is_data = maybe_float(raw_rows[0][0]) is not None
    if first_is_data:
        header: list[str] | None = None
        data_rows = raw_rows
        ex_idx = 0
        jpi_idx = 1
    else:
        header = [normalized_header(value) for value in raw_rows[0]]
        data_rows = raw_rows[1:]
        ex_candidates = {
            "ex_mev",
            "input_ex_mev",
            "excitation_mev",
            "ex",
            "ex_energy_mev",
        }
        jpi_candidates = {"jpi", "j_pi", "jp", "spin_parity"}
        ex_idx = next(
            (idx for idx, name in enumerate(header) if name in ex_candidates),
            0,
        )
        jpi_idx = next(
            (idx for idx, name in enumerate(header) if name in jpi_candidates),
            1,
        )

    for raw in data_rows:
        if len(raw) <= max(ex_idx, jpi_idx):
            continue
        ex_value = maybe_float(raw[ex_idx])
        if ex_value is None:
            continue
        jp = raw[jpi_idx].strip()
        if not jp:
            raise ValueError(f"Missing J/pi entry for Ex={ex_value:g} MeV in {path}")
        row: dict[str, object] = {
            "input_Ex_MeV": ex_value,
            "Jpi": jp,
            "component": state_name(ex_value, jp),
        }
        if header is not None:
            for idx, name in enumerate(header):
                if idx >= len(raw) or idx in {ex_idx, jpi_idx} or not name:
                    continue
                numeric_value = maybe_float(raw[idx])
                row[name] = numeric_value if numeric_value is not None else raw[idx].strip()
        rows.append(row)

    if not rows:
        raise ValueError(f"No states found in {path}")
    states = pd.DataFrame(rows).sort_values("input_Ex_MeV").reset_index(drop=True)
    if states["component"].duplicated().any():
        duplicated = ", ".join(states.loc[states["component"].duplicated(), "component"])
        raise ValueError(f"Duplicate component names after parsing: {duplicated}")
    return states


def load_existing_fits(path: Path, ring_group: str = "all_rings") -> pd.DataFrame:
    fits = pd.read_csv(path)
    required = {"state", "ring_group", "input_Ex_MeV", "minuit_valid", *PARAM_COLS}
    missing = sorted(required - set(fits.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    fits = fits[
        (fits["ring_group"] == ring_group) & (fits["minuit_valid"].astype(bool))
    ].copy()
    if fits.empty:
        raise ValueError(f"No valid {ring_group} sloped-box fits found in {path}")
    fits = fits.sort_values("input_Ex_MeV").reset_index(drop=True)
    if fits["input_Ex_MeV"].duplicated().any():
        duplicated = fits.loc[fits["input_Ex_MeV"].duplicated(), "input_Ex_MeV"]
        raise ValueError(f"Duplicate fitted excitation energies: {duplicated.tolist()}")
    return fits


def linear_interp_or_extrapolate(
    x: float, xp: np.ndarray, yp: np.ndarray
) -> tuple[float, str, str]:
    if len(xp) < 2:
        raise ValueError("Need at least two fitted points for interpolation")
    idx = int(np.argmin(np.abs(xp - x)))
    if np.isclose(x, xp[idx], atol=1e-12):
        return float(yp[idx]), "direct", f"{idx}"
    if x < xp[0]:
        slope = (yp[1] - yp[0]) / (xp[1] - xp[0])
        return float(yp[0] + slope * (x - xp[0])), "extrapolated_low", "0,1"
    if x > xp[-1]:
        slope = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2])
        last = len(xp) - 1
        return float(yp[-1] + slope * (x - xp[-1])), "extrapolated_high", f"{last-1},{last}"
    value = float(np.interp(x, xp, yp))
    right = int(np.searchsorted(xp, x))
    return value, "interpolated", f"{right-1},{right}"


def component_row_from_fits(
    state: pd.Series,
    fits: pd.DataFrame,
    match_tolerance_keV: float,
    force_interpolate_components: set[str] | None = None,
) -> dict[str, object]:
    fit_ex = fits["input_Ex_MeV"].to_numpy(dtype=float)
    match_tolerance_mev = match_tolerance_keV / 1000.0
    ex = float(state["input_Ex_MeV"])
    component_name = str(state["component"])
    force_interpolate = component_name in (force_interpolate_components or set())
    nearest_idx = int(np.argmin(np.abs(fit_ex - ex)))
    nearest_delta = abs(float(fit_ex[nearest_idx]) - ex)
    row: dict[str, object] = {
        "component": component_name,
        "input_Ex_MeV": ex,
        "Jpi": state["Jpi"],
    }

    if nearest_delta <= match_tolerance_mev and not force_interpolate:
        fit_row = fits.iloc[nearest_idx]
        source = "direct"
        source_indices = str(nearest_idx)
        source_states = str(fit_row["state"])
        row["matched_fit_Ex_MeV"] = float(fit_row["input_Ex_MeV"])
        for col in PARAM_COLS:
            row[col] = float(fit_row[col])
    else:
        interp_fits = fits
        interp_fit_ex = fit_ex
        if force_interpolate and nearest_delta <= match_tolerance_mev:
            keep_mask = np.ones(len(fits), dtype=bool)
            keep_mask[nearest_idx] = False
            interp_fits = fits.loc[keep_mask].reset_index(drop=True)
            if len(interp_fits) < 2:
                raise ValueError(
                    f"Cannot force interpolation for {component_name}: fewer than "
                    "two fitted templates remain after excluding the direct match"
                )
            interp_fit_ex = interp_fits["input_Ex_MeV"].to_numpy(dtype=float)
        source_flags: set[str] = set()
        source_index_tokens: set[str] = set()
        row["matched_fit_Ex_MeV"] = np.nan
        for col in PARAM_COLS:
            value, source, source_indices = linear_interp_or_extrapolate(
                ex,
                interp_fit_ex,
                interp_fits[col].to_numpy(dtype=float),
            )
            row[col] = value
            source_flags.add(source)
            source_index_tokens.update(source_indices.split(","))
        source = (
            "extrapolated_low"
            if "extrapolated_low" in source_flags
            else "extrapolated_high"
            if "extrapolated_high" in source_flags
            else "interpolated"
        )
        ordered_indices = sorted(int(token) for token in source_index_tokens)
        source_indices = ",".join(str(i) for i in ordered_indices)
        source_states = ",".join(
            str(interp_fits.iloc[idx]["state"]) for idx in ordered_indices
        )
        if force_interpolate:
            source = f"{source}+forced"

    row["template_source"] = source
    row["source_fit_indices"] = source_indices
    row["source_fit_states"] = source_states
    return row


def component_row_invalid_reason(row: dict[str, object]) -> str | None:
    values = {col: float(row[col]) for col in PARAM_COLS}
    for col, value in values.items():
        if not np.isfinite(value):
            return f"{col} is non-finite ({value})"
    if values["sigma_L_MeV"] <= 0.0 or values["sigma_R_MeV"] <= 0.0:
        return (
            f"non-positive sigma_L/sigma_R="
            f"{values['sigma_L_MeV']}, {values['sigma_R_MeV']}"
        )
    if values["E_R_MeV"] <= values["E_L_MeV"]:
        return f"E_R <= E_L ({values['E_R_MeV']} <= {values['E_L_MeV']})"
    return None


def build_components(
    states: pd.DataFrame,
    fits: pd.DataFrame,
    match_tolerance_keV: float,
    fallback_fits: pd.DataFrame | None = None,
    force_interpolate_components: set[str] | None = None,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    components: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []

    for _, state in states.iterrows():
        row = component_row_from_fits(
            state,
            fits,
            match_tolerance_keV,
            force_interpolate_components=force_interpolate_components,
        )
        invalid_reason = component_row_invalid_reason(row)
        if invalid_reason is not None and fallback_fits is not None:
            primary_source = str(row["template_source"])
            primary_states = str(row["source_fit_states"])
            row = component_row_from_fits(
                state,
                fallback_fits,
                match_tolerance_keV,
                force_interpolate_components=force_interpolate_components,
            )
            row["template_source"] = (
                f"{row['template_source']}+fallback_from_invalid_{primary_source}"
            )
            row["fallback_reason"] = invalid_reason
            row["fallback_replaced_source_fit_states"] = primary_states
            invalid_reason = component_row_invalid_reason(row)
        if invalid_reason is not None:
            raise ValueError(
                f"Invalid template parameters for {row['component']}: {invalid_reason}"
            )

        component = {
            "name": row["component"],
            "input_Ex_MeV": row["input_Ex_MeV"],
            "Jpi": row["Jpi"],
        }
        if "height_all" in state and pd.notna(state["height_all"]):
            component["initial_weight"] = float(state["height_all"])
            row["height_all"] = float(state["height_all"])
        for col, key in PARAM_TO_COMPONENT_KEY.items():
            component[key] = float(row[col])
        components.append(component)

        rows.append(row)

    component_df = pd.DataFrame(rows)
    return components, component_df


def fusion_contaminant_components(
    bandwidth_MeV: float = 0.10,
    energy_shift_MeV: float = 0.0,
) -> list[dict[str, object]]:
    if bandwidth_MeV <= 0.0:
        raise ValueError("--fusion-proton-bandwidth-MeV must be positive")
    proton = dict(FUSION_COMPONENT)
    proton.update(
        {
            "name": "fusion_proton_low_bkg",
            "ejectile": "proton",
            "role": "fusion_background",
            "energy_region": "low",
            "sample_range": LOW_PROTON_TEMPLATE_RANGE,
            "bandwidth": float(bandwidth_MeV),
            "energy_shift_MeV": float(energy_shift_MeV),
        }
    )
    return [proton]


def fusion_contaminant_component_row(component: dict[str, object]) -> dict[str, object]:
    source = Path(component["source"]).expanduser()
    ejectile = str(component.get("ejectile", "fusion"))
    return {
        "component": component["name"],
        "input_Ex_MeV": np.nan,
        "Jpi": f"{ejectile} fusion",
        "matched_fit_Ex_MeV": np.nan,
        "E_L_MeV": np.nan,
        "E_R_MeV": np.nan,
        "sigma_L_MeV": np.nan,
        "sigma_R_MeV": np.nan,
        "slope_MeV_inv": np.nan,
        "height_all": np.nan,
        "template_source": component.get("kind", "sample_kde"),
        "source_fit_indices": "",
        "source_fit_states": str(source),
        "sample_column": component["column"],
        "sample_range_MeV": (
            f"{component['sample_range'][0]:g}-{component['sample_range'][1]:g}"
        ),
        "energy_region": component.get("energy_region", ""),
        "bandwidth_MeV": float(component["bandwidth"]),
        "energy_shift_MeV": float(component.get("energy_shift_MeV", 0.0)),
        "max_fraction": float(component["max_fraction"]),
        "n_source_events": len(load_sample_template_values(component)),
    }


def append_fusion_contaminants(
    components: list[dict[str, object]],
    component_df: pd.DataFrame,
    *,
    bandwidth_MeV: float = 0.10,
    energy_shift_MeV: float = 0.0,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    contaminants = fusion_contaminant_components(
        bandwidth_MeV=bandwidth_MeV,
        energy_shift_MeV=energy_shift_MeV,
    )
    existing_names = {existing["name"] for existing in components}
    duplicate_names = sorted(
        str(component["name"])
        for component in contaminants
        if component["name"] in existing_names
    )
    if duplicate_names:
        raise ValueError(f"Duplicate component names: {', '.join(duplicate_names)}")
    components = [*components, *contaminants]
    contaminant_df = pd.DataFrame(
        [fusion_contaminant_component_row(component) for component in contaminants]
    )
    component_df = pd.concat([component_df, contaminant_df], ignore_index=True)
    return components, component_df


def low_energy_floor_component(
    fit_lo_MeV: float,
    floor_hi_MeV: float,
    max_fraction: float,
) -> dict[str, object]:
    if floor_hi_MeV <= fit_lo_MeV:
        raise ValueError(
            "--low-energy-floor-hi-MeV must be above the lower fit boundary"
        )
    if max_fraction <= 0.0:
        raise ValueError("--low-energy-floor-max-fraction must be positive")
    return {
        "name": LOW_ENERGY_FLOOR_NAME,
        "kind": "bounded_flat",
        "role": "low_energy_background",
        "E_L": float(fit_lo_MeV),
        "E_R": float(floor_hi_MeV),
        "max_fraction": float(max_fraction),
    }


def low_energy_floor_component_row(component: dict[str, object]) -> dict[str, object]:
    return {
        "component": component["name"],
        "input_Ex_MeV": np.nan,
        "Jpi": "low-energy background",
        "matched_fit_Ex_MeV": np.nan,
        "E_L_MeV": float(component["E_L"]),
        "E_R_MeV": float(component["E_R"]),
        "sigma_L_MeV": np.nan,
        "sigma_R_MeV": np.nan,
        "slope_MeV_inv": np.nan,
        "height_all": np.nan,
        "template_source": component["kind"],
        "source_fit_indices": "",
        "source_fit_states": "",
        "energy_region": "low_floor",
        "max_fraction": float(component["max_fraction"]),
    }


def append_low_energy_floor(
    components: list[dict[str, object]],
    component_df: pd.DataFrame,
    *,
    fit_lo_MeV: float,
    floor_hi_MeV: float,
    max_fraction: float,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    if LOW_ENERGY_FLOOR_NAME in {component["name"] for component in components}:
        raise ValueError(f"Duplicate component name: {LOW_ENERGY_FLOOR_NAME}")
    floor = low_energy_floor_component(fit_lo_MeV, floor_hi_MeV, max_fraction)
    floor_df = pd.DataFrame([low_energy_floor_component_row(floor)])
    return [*components, floor], pd.concat([component_df, floor_df], ignore_index=True)


def find_root_histogram(root_file, hist_name: str | None):
    keys = root_file.keys()
    if hist_name:
        matches = [
            key
            for key in keys
            if key == hist_name or key.split(";", 1)[0] == hist_name
        ]
        if not matches:
            raise ValueError(
                f"ROOT histogram {hist_name!r} not found. Available keys: "
                + ", ".join(keys)
            )
        return matches[0], root_file[matches[0]]

    for key in keys:
        candidate = root_file[key]
        if hasattr(candidate, "to_numpy") and str(candidate.classname).startswith("TH1"):
            return key, candidate
    raise ValueError("No TH1-like histogram found in ROOT file")


def load_root_histogram_data(path: Path, hist_name: str | None) -> pd.DataFrame:
    try:
        import uproot
    except ImportError as exc:
        raise ImportError(
            "Reading ROOT histogram input requires uproot in this Python environment"
        ) from exc

    with uproot.open(path) as root_file:
        selected_key, hist = find_root_histogram(root_file, hist_name)
        counts, edges = hist.to_numpy()

    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if len(edges) != len(counts) + 1:
        raise ValueError(f"{path}:{selected_key} has inconsistent bin edges/counts")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError(f"{path}:{selected_key} contains invalid bin counts")
    rounded_counts = np.rint(counts).astype(int)
    if not np.allclose(counts, rounded_counts, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"{path}:{selected_key} has non-integer bin contents; binned ROOT "
            "inputs currently require event-count histograms"
        )

    centers = 0.5 * (edges[:-1] + edges[1:])
    excitation = np.repeat(centers, rounded_counts)
    df = pd.DataFrame({"excitation_MeV": excitation})
    df["source_file"] = str(path)
    df["source_hist"] = selected_key.split(";", 1)[0]
    return df


def load_data(path: Path, ring_selection: str, root_hist: str | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".root":
        if ring_selection.strip().lower() != "all":
            raise ValueError("--ring-selection is not supported for ROOT histogram input")
        return load_root_histogram_data(path, root_hist)

    npz = np.load(path, allow_pickle=True)
    df = pd.DataFrame({key: npz[key] for key in npz.files})
    if "source_file" in df:
        df["source_file"] = df["source_file"].astype(str)
    if "excitation_MeV" not in df:
        raise ValueError(f"{path} does not contain excitation_MeV")
    if ring_selection.strip().lower() != "all":
        df = df.query(ring_selection).copy()
    return df


def component_support(
    component: dict[str, object],
    nsigma: float,
) -> tuple[float, float] | None:
    if component.get("kind", "sloped_box") == "sloped_box":
        return (
            float(component["E_L"]) - nsigma * float(component["sigma_L"]),
            float(component["E_R"]) + nsigma * float(component["sigma_R"]),
        )
    if component.get("kind") == "bounded_flat":
        return float(component["E_L"]), float(component["E_R"])
    if "sample_range" in component:
        sample_lo, sample_hi = component["sample_range"]
        energy_shift = float(component.get("energy_shift_MeV", 0.0))
        return float(sample_lo) + energy_shift, float(sample_hi) + energy_shift
    return None


def automatic_fit_range(
    data: np.ndarray,
    components: list[dict[str, object]],
    nsigma: float,
    bin_width: float,
) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        raise ValueError("No finite excitation energies available")
    supports = [
        support for support in (component_support(c, nsigma) for c in components)
        if support is not None
    ]
    if not supports:
        raise ValueError("No finite component supports available")
    support_lo = min(lo for lo, _ in supports)
    support_hi = max(hi for _, hi in supports)
    lo = max(float(np.min(finite)), support_lo)
    hi = min(float(np.max(finite)), support_hi)
    lo = math.floor(lo / bin_width) * bin_width
    hi = math.ceil(hi / bin_width) * bin_width
    if hi <= lo:
        raise ValueError(f"Invalid automatic fit range {lo:g}-{hi:g} MeV")
    return lo, hi


def make_bins(fit_range: tuple[float, float], bin_width: float) -> np.ndarray:
    lo, hi = fit_range
    n_bins = int(math.ceil((hi - lo) / bin_width))
    bins = lo + bin_width * np.arange(n_bins + 1, dtype=float)
    if bins[-1] < hi:
        bins = np.append(bins, hi)
    else:
        bins[-1] = hi
    return bins


def fit_binned_amplitudes(
    observed: np.ndarray,
    bin_probs: np.ndarray,
    upper_bounds: np.ndarray,
    initial_weights: np.ndarray | None = None,
) -> Minuit:
    n_comp = bin_probs.shape[0]
    par_names = [f"N{i}" for i in range(n_comp)]

    def nll_array(values: np.ndarray) -> float:
        yields = np.asarray(values, dtype=float)
        if np.any(yields < 0.0) or np.any(yields > upper_bounds):
            return 1e100
        expected = yields @ bin_probs
        if np.any((expected <= 0.0) & (observed > 0)):
            return 1e100
        valid = expected > 0.0
        return float(np.sum(expected[valid] - observed[valid] * np.log(expected[valid])))

    def nll(*values: float) -> float:
        return nll_array(np.asarray(values, dtype=float))

    total = float(np.sum(observed))
    if initial_weights is not None and np.any(initial_weights > 0.0):
        weights = np.clip(np.asarray(initial_weights, dtype=float), 0.0, None)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        starts = total * weights / np.sum(weights)
        starts = np.clip(starts, 1.0, None)
    else:
        starts = np.full(n_comp, max(total / n_comp, 1.0), dtype=float)
    starts = np.minimum(starts, 0.5 * upper_bounds)
    minuit = Minuit(nll, *starts, name=par_names)
    minuit.errordef = Minuit.LIKELIHOOD
    for idx, name in enumerate(par_names):
        minuit.limits[name] = (0.0, float(upper_bounds[idx]))
    minuit.migrad()
    if not minuit.valid:
        minuit.simplex()
        minuit.migrad()
    minuit.hesse()
    return minuit


def component_upper_bounds(
    components: list[dict[str, object]],
    n_data: int,
    max_total_yield_factor: float,
) -> np.ndarray:
    default_upper = max_total_yield_factor * n_data
    bounds = np.full(len(components), default_upper, dtype=float)
    for idx, component in enumerate(components):
        if "max_fraction" in component:
            bounds[idx] = min(bounds[idx], float(component["max_fraction"]) * n_data)
    return bounds


def log_probability_binned(
    log_yields: np.ndarray,
    observed: np.ndarray,
    bin_probs: np.ndarray,
    upper_bounds: np.ndarray,
    uncertainty_scale: float = 1.0,
) -> float:
    if not np.all(np.isfinite(log_yields)):
        return -np.inf
    yields = np.exp(log_yields)
    if np.any(yields <= 0.0) or np.any(yields > upper_bounds):
        return -np.inf
    expected = yields @ bin_probs
    if np.any((expected <= 0.0) & (observed > 0)) or not np.all(np.isfinite(expected)):
        return -np.inf
    valid = expected > 0.0
    log_like = -float(np.sum(expected[valid]))
    log_like += float(np.sum(observed[valid] * np.log(expected[valid])))
    return log_like / uncertainty_scale**2 + float(np.sum(log_yields))


def pearson_chi2(
    observed: np.ndarray,
    expected: np.ndarray,
    n_parameters: int,
    min_expected: float = 0.0,
) -> dict[str, float | int]:
    valid = expected > min_expected
    chi2 = float(np.sum((observed[valid] - expected[valid]) ** 2 / expected[valid]))
    ndf = int(np.count_nonzero(valid) - n_parameters)
    return {
        "chi2": chi2,
        "ndf": ndf,
        "chi2_ndf": chi2 / ndf if ndf > 0 else np.nan,
        "p_value": float(chi2_dist.sf(chi2, ndf)) if ndf > 0 else np.nan,
        "min_expected": float(min_expected),
        "model_bins_used": int(np.count_nonzero(valid)),
    }


def poisson_deviance(
    observed: np.ndarray,
    expected: np.ndarray,
    n_parameters: int,
) -> dict[str, float | int]:
    valid = expected > 0.0
    obs = observed[valid].astype(float)
    exp = expected[valid].astype(float)
    terms = np.empty_like(obs, dtype=float)
    positive = obs > 0.0
    terms[positive] = (
        obs[positive] * np.log(obs[positive] / exp[positive])
        - (obs[positive] - exp[positive])
    )
    terms[~positive] = exp[~positive]
    deviance = float(2.0 * np.sum(terms))
    ndf = int(np.count_nonzero(valid) - n_parameters)
    return {
        "deviance": deviance,
        "ndf": ndf,
        "deviance_ndf": deviance / ndf if ndf > 0 else np.nan,
        "p_value": float(chi2_dist.sf(deviance, ndf)) if ndf > 0 else np.nan,
    }


def multinomial_log_likelihood(observed: np.ndarray, expected: np.ndarray) -> float:
    total_expected = float(np.sum(expected))
    if total_expected <= 0.0:
        return -np.inf
    probs = expected / total_expected
    valid = (observed > 0) & (probs > 0.0)
    if np.any((observed > 0) & (probs <= 0.0)):
        return -np.inf
    n = int(np.sum(observed))
    return float(gammaln(n + 1) - np.sum(gammaln(observed + 1)) + np.sum(observed[valid] * np.log(probs[valid])))


def mixture_cdf_on_grid(yields: np.ndarray, grid: np.ndarray, pdfs_grid: np.ndarray) -> np.ndarray:
    weights = np.asarray(yields, dtype=float)
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("Mixture yields must be non-negative with positive total")
    weights = weights / np.sum(weights)
    pdf_grid = weights @ pdfs_grid
    cdf_grid = cumulative_trapezoid(pdf_grid, grid, initial=0.0)
    cdf_grid /= cdf_grid[-1]
    return cdf_grid


def ks_statistic(data: np.ndarray, grid: np.ndarray, cdf_grid: np.ndarray) -> tuple[float, float, float]:
    x = np.sort(np.asarray(data, dtype=float))
    n = len(x)
    fitted_cdf = np.interp(x, grid, cdf_grid, left=0.0, right=1.0)
    empirical_hi = np.arange(1, n + 1) / n
    empirical_lo = np.arange(0, n) / n
    d_plus = float(np.max(empirical_hi - fitted_cdf))
    d_minus = float(np.max(fitted_cdf - empirical_lo))
    return float(max(d_plus, d_minus)), d_plus, d_minus


def poisson_count_errors(
    counts: np.ndarray,
    confidence: float = 0.682689492,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact central Garwood errors for Poisson-distributed counts."""
    observed = np.asarray(counts, dtype=float)
    alpha = 1.0 - confidence
    lower = np.zeros_like(observed)
    positive = observed > 0.0
    lower[positive] = 0.5 * chi2_dist.ppf(alpha / 2.0, 2.0 * observed[positive])
    upper = 0.5 * chi2_dist.ppf(
        1.0 - alpha / 2.0,
        2.0 * (observed + 1.0),
    )
    return observed - lower, upper - observed


def pearson_residuals(
    observed: np.ndarray,
    expected: np.ndarray,
    min_expected: float = 0.0,
) -> np.ndarray:
    residuals = np.full_like(np.asarray(observed, dtype=float), np.nan)
    valid = expected > min_expected
    residuals[valid] = (observed[valid] - expected[valid]) / np.sqrt(expected[valid])
    return residuals


def highlight_astrophysical_region(
    ax: plt.Axes,
    *,
    face_alpha: float,
    edge_alpha: float = 0.80,
    edge_zorder: float = 6.0,
) -> None:
    """Mark the astrophysical energy window using the zoom-box style."""
    x_lo, x_hi = ASTROPHYSICAL_INSET_RANGE_MEV
    ax.axvspan(
        x_lo,
        x_hi,
        facecolor=ASTROPHYSICAL_ZOOM_BOX_FACE,
        alpha=face_alpha,
        linewidth=0.0,
        zorder=0,
    )
    ax.axvline(
        x_lo,
        color=ASTROPHYSICAL_ZOOM_BOX_EDGE,
        alpha=edge_alpha,
        lw=0.9,
        zorder=edge_zorder,
    )
    ax.axvline(
        x_hi,
        color=ASTROPHYSICAL_ZOOM_BOX_EDGE,
        alpha=edge_alpha,
        lw=0.9,
        zorder=edge_zorder,
    )


def add_astrophysical_region_inset(
    ax: plt.Axes,
    *,
    bins: np.ndarray,
    observed: np.ndarray,
    bin_centers: np.ndarray,
    lower_error: np.ndarray,
    upper_error: np.ndarray,
    bin_width: float,
    grid: np.ndarray,
    component_curves: list[tuple[np.ndarray, object, float, float]],
    total_fine: np.ndarray,
    font_scale: float = 1.0,
) -> None:
    """Add a zoomed spectrum inset for the astrophysical energy window."""
    x_lo, x_hi = ASTROPHYSICAL_INSET_RANGE_MEV
    grid_window = (grid >= x_lo) & (grid <= x_hi)
    point_window = (bin_centers >= x_lo) & (bin_centers <= x_hi)
    if not np.any(grid_window) or not np.any(point_window):
        return

    inset = ax.inset_axes([0.055, 0.50, 0.31, 0.45])
    inset.set_facecolor("#f7fbfd")
    inset.stairs(
        observed,
        bins,
        fill=True,
        baseline=0,
        facecolor="0.90",
        edgecolor="0.35",
        linewidth=0.55,
        alpha=0.86,
        zorder=1,
    )

    y_candidates = [
        np.nanmax(total_fine[grid_window]),
        np.nanmax(observed[point_window] + upper_error[point_window]),
    ]
    y_top = max(1.0, float(np.nanmax(y_candidates))) * 1.18
    component_threshold = max(0.2, 0.01 * y_top)
    for counts_fine, color, linewidth, alpha in component_curves:
        counts_window = counts_fine[grid_window]
        if counts_window.size == 0 or np.nanmax(counts_window) < component_threshold:
            continue
        inset.plot(
            grid[grid_window],
            counts_window,
            lw=max(0.65, linewidth * 0.75),
            alpha=min(0.95, max(0.50, alpha)),
            color=color,
            zorder=2,
        )

    inset.plot(grid[grid_window], total_fine[grid_window], color="black", lw=1.7, zorder=4)
    inset.errorbar(
        bin_centers[point_window],
        observed[point_window],
        yerr=np.vstack((lower_error[point_window], upper_error[point_window])),
        fmt="o",
        markersize=2.0,
        color="black",
        ecolor="black",
        elinewidth=0.65,
        capsize=1.2,
        capthick=0.65,
        zorder=5,
    )
    inset.set_xlim(x_lo, x_hi)
    inset.set_ylim(0.0, y_top)
    inset.set_title(
        r"$10.915 \leq E_x \leq 11.364$ MeV",
        fontsize=8.2 * font_scale,
        pad=2.0 * font_scale,
    )
    inset.set_xlabel(r"$E_x$ [MeV]", fontsize=6.8 * font_scale, labelpad=0.5 * font_scale)
    inset.set_ylabel(
        rf"Counts / ${1000.0 * bin_width:.0f}\,\mathrm{{keV}}$",
        fontsize=6.8 * font_scale,
        labelpad=1.0 * font_scale,
    )
    inset.set_xticks([10.92, 11.14, 11.36])
    inset.yaxis.set_major_locator(MaxNLocator(nbins=4))
    inset.tick_params(
        axis="both",
        which="major",
        labelsize=6.2 * font_scale,
        width=0.75 * font_scale,
        length=2.8 * font_scale,
    )
    inset.tick_params(
        axis="both",
        which="minor",
        width=0.55 * font_scale,
        length=1.6 * font_scale,
    )
    inset.minorticks_on()
    for spine in inset.spines.values():
        spine.set_color(ASTROPHYSICAL_ZOOM_BOX_EDGE)
        spine.set_linewidth(1.15)
    zoom_patch, connector_1, connector_2 = mark_inset(
        ax,
        inset,
        loc1=1,
        loc2=4,
        fc=ASTROPHYSICAL_ZOOM_BOX_FACE,
        ec=ASTROPHYSICAL_ZOOM_BOX_EDGE,
        lw=1.05,
    )
    zoom_patch.set_zorder(6)
    zoom_patch.set_facecolor(
        mpl.colors.to_rgba(
            ASTROPHYSICAL_ZOOM_BOX_FACE,
            ASTROPHYSICAL_ZOOM_BOX_FACE_ALPHA,
        )
    )
    zoom_patch.set_edgecolor(ASTROPHYSICAL_ZOOM_BOX_EDGE)
    for connector in (connector_1, connector_2):
        connector.set_zorder(6)
        connector.set_linewidth(1.05)
        connector.set_alpha(0.92)
        connector.set_edgecolor(ASTROPHYSICAL_ZOOM_BOX_EDGE)


def plot_spectrum(
    outdir: Path,
    x: np.ndarray,
    bins: np.ndarray,
    components: list[dict[str, object]],
    yields: np.ndarray,
    grid: np.ndarray,
    pdfs_grid: np.ndarray,
    chi2_info: dict[str, float | int],
    *,
    individual_states: bool = False,
    poisson_errorbars: bool = False,
    uncertainty_scale: float = 1.0,
    plot_formats: tuple[str, ...] = ("pdf", "png"),
    output_path: Path | None = None,
) -> list[Path]:
    if uncertainty_scale <= 0.0:
        raise ValueError("uncertainty_scale must be positive")
    is_full_s3_output = (
        output_path is not None
        and output_path.name.startswith("all_")
        and individual_states
        and poisson_errorbars
    )
    main_label_fontsize = 16.0 if is_full_s3_output else 10.0
    residual_label_fontsize = 15.0 if is_full_s3_output else 10.0
    tick_labelsize = 13.0 if is_full_s3_output else None
    legend_fontsize = 15.0 if is_full_s3_output else 6.4
    inset_font_scale = 1.55 if is_full_s3_output else 1.0
    error_marker_size = 3.4 if is_full_s3_output else 2.7
    error_line_width = 1.05 if is_full_s3_output else 0.85
    cap_size = 2.0 if is_full_s3_output else 1.6
    cap_thick = 1.05 if is_full_s3_output else 0.85

    observed, _ = np.histogram(x, bins=bins)
    bin_width = bins[1] - bins[0]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    lower_error, upper_error = poisson_count_errors(observed)
    lower_error *= uncertainty_scale
    upper_error *= uncertainty_scale
    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(25.0, 13.5)
        if is_full_s3_output
        else (14.6, 8.1)
        if individual_states
        else (10.8, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
    )
    ax.stairs(
        observed,
        bins,
        fill=True,
        baseline=0,
        facecolor="0.86",
        edgecolor="0.25",
        linewidth=0.75,
        alpha=0.70,
        label="_nolegend_" if poisson_errorbars else "data",
    )
    if poisson_errorbars:
        ax.errorbar(
            bin_centers,
            observed,
            yerr=np.vstack((lower_error, upper_error)),
            fmt="o",
            markersize=error_marker_size,
            color="black",
            ecolor="black",
            elinewidth=error_line_width,
            capsize=cap_size,
            capthick=cap_thick,
            label="data",
            zorder=7,
        )

    cmap = plt.get_cmap("tab20")
    state_components = [
        component
        for component in components
        if component.get("kind", "sloped_box") == "sloped_box"
    ]
    state_cmap = mpl.colormaps["turbo"].resampled(max(2, len(state_components)))
    state_index = 0
    total_fine = np.zeros_like(grid)
    component_curves: list[tuple[np.ndarray, object, float, float]] = []
    state_components_labelled = False
    for idx, component in enumerate(components):
        counts_fine = yields[idx] * pdfs_grid[idx] * bin_width
        total_fine += counts_fine
        if is_fusion_component(component):
            color = fusion_component_color(component)
            ax.fill_between(
                grid,
                0.0,
                counts_fine,
                color=color,
                alpha=0.24,
                linewidth=0.0,
                zorder=3,
            )
            ax.plot(
                grid,
                counts_fine,
                lw=2.2,
                alpha=0.95,
                color=color,
                label=component_label(component),
                zorder=4,
            )
            component_curves.append((counts_fine, color, 2.2, 0.95))
            continue

        if component.get("kind", "sloped_box") == "sloped_box":
            if individual_states:
                label = component_label(component)
                color = state_cmap(state_index)
                linewidth = 1.25
                alpha = 0.86
            else:
                label = "state components" if not state_components_labelled else "_nolegend_"
                color = cmap(idx % cmap.N)
                linewidth = 0.9
                alpha = 0.58
            state_components_labelled = True
            state_index += 1
        else:
            label = component_label(component)
            color = cmap(idx % cmap.N)
            linewidth = 0.9
            alpha = 0.58
        ax.plot(
            grid,
            counts_fine,
            lw=linewidth,
            alpha=alpha,
            color=color,
            label=label,
        )
        component_curves.append((counts_fine, color, linewidth, alpha))
    ax.plot(grid, total_fine, color="black", lw=2.6, label=r"MCMC median total")
    add_astrophysical_region_inset(
        ax,
        bins=bins,
        observed=observed,
        bin_centers=bin_centers,
        lower_error=lower_error,
        upper_error=upper_error,
        bin_width=bin_width,
        grid=grid,
        component_curves=component_curves,
        total_fine=total_fine,
        font_scale=inset_font_scale,
    )
    ax.set_ylabel(
        rf"Counts / ${1000.0 * bin_width:.0f}\,\mathrm{{keV}}$",
        fontsize=main_label_fontsize,
    )
    chi2_label = (
        rf"$\chi^2/\mathrm{{dof}}={float(chi2_info['chi2_ndf']):.2f}$"
        rf", $p={float(chi2_info['p_value']):.2f}$"
    )
    ax.set_title("")
    chi2_handle = Line2D([], [], linestyle="none", label=chi2_label)
    if individual_states:
        handles, labels = ax.get_legend_handles_labels()
        handles = [chi2_handle] + handles
        labels = [chi2_label] + labels
        ax.legend(
            handles,
            labels,
            fontsize=legend_fontsize,
            ncol=1,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            handlelength=2.0,
            labelspacing=0.35 if is_full_s3_output else 0.45,
        )
        fig.subplots_adjust(right=0.76 if is_full_s3_output else 0.73)
    else:
        handles, labels = ax.get_legend_handles_labels()
        handles = [chi2_handle] + handles
        labels = [chi2_label] + labels
        ax.legend(handles, labels, fontsize=9, loc="upper left")

    bin_probs = bin_probabilities(grid, pdfs_grid, bins)
    expected = yields @ bin_probs
    min_expected = float(chi2_info.get("min_expected", 0.0))
    residuals = pearson_residuals(
        observed,
        expected,
        min_expected=min_expected,
    )
    valid_residuals = expected > min_expected
    astro_residuals = (
        valid_residuals
        & (bin_centers >= ASTROPHYSICAL_INSET_RANGE_MEV[0])
        & (bin_centers <= ASTROPHYSICAL_INSET_RANGE_MEV[1])
    )
    residual_yerr = np.vstack(
        (
            lower_error[valid_residuals]
            / (uncertainty_scale * np.sqrt(expected[valid_residuals])),
            upper_error[valid_residuals]
            / (uncertainty_scale * np.sqrt(expected[valid_residuals])),
        )
    )
    highlight_astrophysical_region(
        rax,
        face_alpha=ASTROPHYSICAL_ZOOM_BOX_FACE_ALPHA,
        edge_alpha=0.85,
        edge_zorder=5.0,
    )
    rax.axhline(0.0, color="black", lw=1.0)
    rax.axhline(2.0, color="tab:blue", linestyle=":", lw=1.0)
    rax.axhline(-2.0, color="tab:blue", linestyle=":", lw=1.0)
    rax.errorbar(
        bin_centers[valid_residuals],
        residuals[valid_residuals],
        yerr=residual_yerr,
        fmt="o",
        markersize=error_marker_size,
        color="black",
        ecolor="black",
        elinewidth=error_line_width,
        capsize=cap_size,
        capthick=cap_thick,
        markeredgewidth=0.55,
        zorder=3,
    )
    if np.any(astro_residuals):
        astro_yerr = np.vstack(
            (
                lower_error[astro_residuals]
                / (uncertainty_scale * np.sqrt(expected[astro_residuals])),
                upper_error[astro_residuals]
                / (uncertainty_scale * np.sqrt(expected[astro_residuals])),
            )
        )
        rax.errorbar(
            bin_centers[astro_residuals],
            residuals[astro_residuals],
            yerr=astro_yerr,
            fmt="o",
            markersize=4.3,
            markerfacecolor=ASTROPHYSICAL_ZOOM_BOX_FACE,
            markeredgecolor=ASTROPHYSICAL_ZOOM_BOX_EDGE,
            markeredgewidth=0.95,
            ecolor=ASTROPHYSICAL_ZOOM_BOX_EDGE,
            elinewidth=1.25,
            capsize=2.2,
            capthick=1.25,
            zorder=8,
        )
    rax.set_xlabel(r"$E_x\ \mathrm{[MeV]}$", fontsize=main_label_fontsize)
    rax.set_ylabel(
        r"$(N_{\mathrm{data}}-N_{\mathrm{fit}})/\sqrt{N_{\mathrm{fit}}}$",
        fontsize=residual_label_fontsize,
    )
    if tick_labelsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
        rax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
    if individual_states and poisson_errorbars:
        output_stem = "MCMC_full_range_spectrum_individual_states_poisson_errors"
    elif individual_states:
        output_stem = "MCMC_full_range_spectrum_individual_states"
    elif poisson_errorbars:
        output_stem = "MCMC_full_range_spectrum_with_residuals_poisson_errors"
    else:
        output_stem = "MCMC_full_range_spectrum_with_residuals"
    written_paths: list[Path] = []
    if output_path is not None:
        if not output_path.suffix:
            raise ValueError("--spectrum-plot-output must include a file suffix")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        written_paths.append(output_path)
    else:
        for fmt in plot_formats:
            path = outdir / f"{output_stem}.{fmt}"
            fig.savefig(path)
            written_paths.append(path)
    plt.close(fig)
    return written_paths


def plot_component_lines(
    outdir: Path,
    bins: np.ndarray,
    components: list[dict[str, object]],
    yields: np.ndarray,
    grid: np.ndarray,
    pdfs_grid: np.ndarray,
    plot_formats: tuple[str, ...] = ("pdf", "png"),
) -> list[Path]:
    bin_width = bins[1] - bins[0]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    total = np.zeros_like(grid)
    for idx, component in enumerate(components):
        counts = yields[idx] * pdfs_grid[idx] * bin_width
        total += counts
        if is_fusion_component(component):
            color = fusion_component_color(component)
            lw = 2.4
            alpha = 0.95
        else:
            color = cmap(idx % cmap.N)
            lw = 1.2
            alpha = 1.0
        ax.plot(
            grid,
            counts,
            lw=lw,
            alpha=alpha,
            color=color,
            label=component_label({**component, "yield": yields[idx]}, with_yield=True),
        )
    ax.plot(grid, total, color="black", lw=2.5, label="total")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$E_x\ \mathrm{[MeV]}$")
    ax.set_ylabel(rf"Counts / ${1000.0 * bin_width:.0f}\,\mathrm{{keV}}$")
    ax.set_title("Fitted State Components")
    ax.grid(True, alpha=0.28)
    ax.legend(
        fontsize=6.6,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        handlelength=1.8,
    )
    written_paths: list[Path] = []
    for fmt in plot_formats:
        path = outdir / f"MCMC_full_range_component_lines.{fmt}"
        fig.savefig(path)
        written_paths.append(path)
    plt.close(fig)
    return written_paths


def plot_posterior_draws(
    outdir: Path,
    x: np.ndarray,
    bins: np.ndarray,
    samples_yields: np.ndarray,
    median_yields: np.ndarray,
    grid: np.ndarray,
    pdfs_grid: np.ndarray,
    plot_formats: tuple[str, ...] = ("pdf", "png"),
) -> list[Path]:
    observed, _ = np.histogram(x, bins=bins)
    bin_width = bins[1] - bins[0]
    plot_grid = np.linspace(bins[0], bins[-1], 900)
    pdfs_plot = np.vstack(
        [np.interp(plot_grid, grid, pdf_grid) for pdf_grid in pdfs_grid]
    )
    median_counts = median_yields @ pdfs_plot * bin_width

    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    ax.stairs(
        observed,
        bins,
        fill=True,
        baseline=0,
        facecolor="0.84",
        edgecolor="black",
        linewidth=0.9,
        alpha=0.48,
        label=r"$N_{\mathrm{data}}$",
    )
    first_chunk = True
    for start in range(0, len(samples_yields), 500):
        sample_chunk = samples_yields[start : start + 500]
        counts = sample_chunk @ pdfs_plot * bin_width
        x_segments = np.broadcast_to(plot_grid, counts.shape)
        segments = np.stack((x_segments, counts), axis=-1)
        collection = LineCollection(
            segments,
            colors=(0.12, 0.32, 0.70, 0.012),
            linewidths=0.28,
            rasterized=True,
            label=(
                rf"{len(samples_yields)} retained walker draws"
                if first_chunk
                else "_nolegend_"
            ),
        )
        ax.add_collection(collection)
        first_chunk = False
    ax.plot(plot_grid, median_counts, color="black", lw=2.4, label="posterior median")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$E_x\ \mathrm{[MeV]}$")
    ax.set_ylabel(rf"Counts / ${1000.0 * bin_width:.0f}\,\mathrm{{keV}}$")
    ax.set_title("All Retained MCMC Walker Draws")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=9)
    written_paths: list[Path] = []
    for fmt in plot_formats:
        path = outdir / f"MCMC_full_range_posterior_draws.{fmt}"
        fig.savefig(path)
        written_paths.append(path)
    plt.close(fig)
    return written_paths


def plot_low_proton_walker_draws(
    outdir: Path,
    bins: np.ndarray,
    components: list[dict[str, object]],
    samples_yields: np.ndarray,
    median_yields: np.ndarray,
    grid: np.ndarray,
    pdfs_grid: np.ndarray,
    plot_formats: tuple[str, ...] = ("pdf", "png"),
) -> list[Path]:
    low_indices = [
        idx
        for idx, component in enumerate(components)
        if component.get("name") == "fusion_proton_low_bkg"
    ]
    if not low_indices:
        return []
    idx = low_indices[0]
    component = components[idx]
    color = fusion_component_color(component)
    bin_width = bins[1] - bins[0]
    plot_grid = np.linspace(bins[0], bins[-1], 900)
    pdf_plot = np.interp(plot_grid, grid, pdfs_grid[idx])
    median_counts = median_yields[idx] * pdf_plot * bin_width

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    first_chunk = True
    for start in range(0, len(samples_yields), 500):
        sample_chunk = samples_yields[start : start + 500, idx]
        counts = sample_chunk[:, None] * pdf_plot[None, :] * bin_width
        x_segments = np.broadcast_to(plot_grid, counts.shape)
        segments = np.stack((x_segments, counts), axis=-1)
        collection = LineCollection(
            segments,
            colors=(0.45, 0.0, 0.50, 0.018),
            linewidths=0.28,
            rasterized=True,
            label=(
                rf"{len(samples_yields)} retained walker draws"
                if first_chunk
                else "_nolegend_"
            ),
        )
        ax.add_collection(collection)
        first_chunk = False

    ax.fill_between(plot_grid, 0.0, median_counts, color=color, alpha=0.20)
    ax.plot(
        plot_grid,
        median_counts,
        color=color,
        lw=2.6,
        label=component_label({**component, "yield": median_yields[idx]}, with_yield=True),
    )
    ax.set_xlim(*LOW_PROTON_TEMPLATE_RANGE)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$E_x\ \mathrm{[MeV]}$")
    ax.set_ylabel(rf"Counts / ${1000.0 * bin_width:.0f}\,\mathrm{{keV}}$")
    ax.set_title(r"Low-$E_x$ Proton Contaminant Walker Draws")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=9)
    written_paths: list[Path] = []
    for fmt in plot_formats:
        path = outdir / f"MCMC_full_range_low_proton_walker_draws.{fmt}"
        fig.savefig(path)
        written_paths.append(path)
    plt.close(fig)
    return written_paths


def corner_axis_label(component: dict[str, object], *, compact: bool = False) -> str:
    ex = float(component["input_Ex_MeV"])
    return rf"${ex:.3f}\,\mathrm{{MeV}}$"


def corner_yield_title(median: float, lower: float, upper: float) -> str:
    return rf"${median:.1f}^{{+{upper:.1f}}}_{{-{lower:.1f}}}$"


def plot_selected_yield_corner(
    outdir: Path,
    components: list[dict[str, object]],
    samples_yields: np.ndarray,
    q50: np.ndarray,
    yield_minus: np.ndarray,
    yield_plus: np.ndarray,
    selected_indices: list[int],
    title: str,
    default_stem: str,
    output_arg_name: str,
    *,
    plot_formats: tuple[str, ...] = ("pdf",),
    output_path: Path | None = None,
) -> list[Path]:
    if not selected_indices:
        return []

    selected_samples = samples_yields[:, selected_indices]
    selected_components = [components[idx] for idx in selected_indices]
    compact_labels = len(selected_indices) > 3
    labels = [
        corner_axis_label(component, compact=compact_labels)
        for component in selected_components
    ]
    yield_titles = [
        corner_yield_title(
            q50[idx],
            yield_minus[idx],
            yield_plus[idx],
        )
        for idx in selected_indices
    ]
    try:
        import corner
    except ImportError as exc:
        raise ImportError(
            "The selected-state corner plot requires the corner package."
        ) from exc

    max_corner_samples = 12_000
    if len(selected_indices) > 3 and selected_samples.shape[0] > max_corner_samples:
        rng = np.random.default_rng(20260810)
        keep = np.sort(
            rng.choice(
                selected_samples.shape[0],
                size=max_corner_samples,
                replace=False,
            )
        )
        selected_samples = selected_samples[keep]

    if len(selected_indices) == 1:
        fig, ax = plt.subplots(figsize=(5.7, 4.6))
        values = selected_samples[:, 0]
        ax.hist(values, bins=48, histtype="stepfilled", color="0.72", edgecolor="0.15")
        median = q50[selected_indices[0]]
        ax.axvline(median, color="black", lw=1.5)
        ax.axvline(
            median - yield_minus[selected_indices[0]],
            color=ASTROPHYSICAL_HIGHLIGHT_EDGE,
            lw=1.1,
            linestyle="--",
        )
        ax.axvline(
            median + yield_plus[selected_indices[0]],
            color=ASTROPHYSICAL_HIGHLIGHT_EDGE,
            lw=1.1,
            linestyle="--",
        )
        ax.set_xlabel(labels[0])
        ax.set_ylabel("Posterior samples")
        ax.set_title(yield_titles[0], fontsize=28.0, pad=8.0)
        ax.xaxis.label.set_size(25.0)
        ax.yaxis.label.set_size(22.0)
        ax.tick_params(axis="both", which="major", labelsize=16.0)
    else:
        title_fontsize = 25.0 if compact_labels else 26.0
        label_fontsize = 28.0 if compact_labels else 29.0
        tick_fontsize = 15.0 if compact_labels else 16.0
        fig = corner.corner(
            selected_samples,
            labels=labels,
            titles=yield_titles,
            quantiles=(0.16, 0.50, 0.84),
            show_titles=True,
            title_fmt=None,
            title_kwargs={"fontsize": title_fontsize, "loc": "center"},
            label_kwargs={"fontsize": label_fontsize},
            color="black",
            hist_kwargs={"color": "0.25", "linewidth": 1.15},
            contour_kwargs={"linewidths": 1.0},
            levels=(1.0 - np.exp(-0.5), 1.0 - np.exp(-2.0)),
            plot_datapoints=False,
            fill_contours=True,
            smooth=1.0,
            smooth1d=1.0,
            truths=q50[selected_indices],
            truth_color=ASTROPHYSICAL_HIGHLIGHT_EDGE,
        )
        for corner_ax in fig.get_axes():
            corner_ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
            corner_ax.xaxis.labelpad = 10.0 if compact_labels else 14.0
            corner_ax.yaxis.labelpad = 10.0 if compact_labels else 14.0
        side_inches = max(8.2, 3.35 * len(selected_indices))
        fig.set_size_inches(side_inches, side_inches)
    fig.subplots_adjust(top=0.96)

    written_paths: list[Path] = []
    if output_path is not None:
        if not output_path.suffix:
            raise ValueError(f"{output_arg_name} must include a file suffix")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        written_paths.append(output_path)
    else:
        for fmt in plot_formats:
            path = outdir / f"{default_stem}.{fmt}"
            fig.savefig(path)
            written_paths.append(path)
    plt.close(fig)
    return written_paths


def plot_astrophysical_region_corner(
    outdir: Path,
    components: list[dict[str, object]],
    samples_yields: np.ndarray,
    q50: np.ndarray,
    yield_minus: np.ndarray,
    yield_plus: np.ndarray,
    *,
    plot_formats: tuple[str, ...] = ("pdf",),
    output_path: Path | None = None,
) -> list[Path]:
    selected_indices = [
        idx
        for idx, component in enumerate(components)
        if component.get("kind", "sloped_box") == "sloped_box"
        and "input_Ex_MeV" in component
        and ASTROPHYSICAL_INSET_RANGE_MEV[0]
        <= float(component["input_Ex_MeV"])
        <= ASTROPHYSICAL_INSET_RANGE_MEV[1]
    ]
    return plot_selected_yield_corner(
        outdir,
        components,
        samples_yields,
        q50,
        yield_minus,
        yield_plus,
        selected_indices,
        r"Astrophysical Region Yield Correlations, $10.915 \leq E_x \leq 11.364$ MeV",
        "MCMC_full_range_astrophysical_region_corner",
        "--astrophysical-corner-output",
        plot_formats=plot_formats,
        output_path=output_path,
    )


def plot_above10_region_corner(
    outdir: Path,
    components: list[dict[str, object]],
    samples_yields: np.ndarray,
    q50: np.ndarray,
    yield_minus: np.ndarray,
    yield_plus: np.ndarray,
    *,
    plot_formats: tuple[str, ...] = ("pdf",),
    output_path: Path | None = None,
) -> list[Path]:
    excluded_components = {"10573_1minus", "12110_0plus"}
    selected_indices = [
        idx
        for idx, component in enumerate(components)
        if component.get("kind", "sloped_box") == "sloped_box"
        and "input_Ex_MeV" in component
        and float(component["input_Ex_MeV"]) >= 10.0
        and str(component["name"]) not in excluded_components
    ]
    return plot_selected_yield_corner(
        outdir,
        components,
        samples_yields,
        q50,
        yield_minus,
        yield_plus,
        selected_indices,
        r"Above-10 MeV Yield Correlations, $E_x \geq 10.0$ MeV",
        "MCMC_full_range_above10MeV_corner",
        "--above10-corner-output",
        plot_formats=plot_formats,
        output_path=output_path,
    )


def main() -> None:
    args = parse_args()
    if args.bin_width_MeV <= 0:
        raise ValueError("--bin-width-MeV must be positive")
    if args.fusion_proton_bandwidth_MeV <= 0:
        raise ValueError("--fusion-proton-bandwidth-MeV must be positive")
    if args.spectrum_plot_output is not None and args.plot_selection != "individual-poisson-only":
        raise ValueError(
            "--spectrum-plot-output is only supported with "
            "--plot-selection individual-poisson-only"
        )
    plot_formats = tuple(args.plot_formats or ("pdf", "png"))
    component_energy_shifts = parse_component_energy_shifts(
        args.component_energy_shift
    )
    component_left_edge_shifts = parse_component_energy_shifts(
        args.component_left_edge_shift,
        "--component-left-edge-shift",
    )
    component_right_edge_shifts = parse_component_energy_shifts(
        args.component_right_edge_shift,
        "--component-right-edge-shift",
    )
    component_sigma_scales = parse_component_scales(
        args.component_sigma_scale,
        "--component-sigma-scale",
    )
    force_interpolate_components = {
        name.strip() for name in args.force_interpolate_component if name.strip()
    }
    if args.global_sigma_scale <= 0.0:
        raise ValueError("--global-sigma-scale must be positive")
    if args.plot_uncertainty_scale <= 0.0:
        raise ValueError("--plot-uncertainty-scale must be positive")
    if args.likelihood_uncertainty_scale <= 0.0:
        raise ValueError("--likelihood-uncertainty-scale must be positive")
    if (
        args.low_energy_floor_hi_MeV is not None
        and args.low_energy_floor_max_fraction <= 0.0
    ):
        raise ValueError("--low-energy-floor-max-fraction must be positive")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    states = read_state_table(args.states_csv)
    fits = load_existing_fits(args.fit_results_csv, args.template_ring_group)
    fallback_fits = (
        load_existing_fits(
            args.fallback_fit_results_csv,
            args.fallback_template_ring_group,
        )
        if args.fallback_fit_results_csv is not None
        else None
    )
    components, component_df = build_components(
        states,
        fits,
        match_tolerance_keV=args.match_tolerance_keV,
        fallback_fits=fallback_fits,
        force_interpolate_components=force_interpolate_components,
    )
    components, component_df = apply_component_energy_shifts(
        components,
        component_df,
        component_energy_shifts,
    )
    components, component_df = apply_component_edge_shifts(
        components,
        component_df,
        component_left_edge_shifts,
        component_right_edge_shifts,
    )
    components, component_df = apply_component_sigma_scales(
        components,
        component_df,
        component_sigma_scales,
    )
    components, component_df = apply_global_component_adjustments(
        components,
        component_df,
        energy_shift_MeV=float(args.global_energy_shift_MeV),
        sigma_scale=float(args.global_sigma_scale),
    )
    if not (args.no_fusion_contaminants or args.no_fusion_proton_contaminants):
        components, component_df = append_fusion_contaminants(
            components,
            component_df,
            bandwidth_MeV=args.fusion_proton_bandwidth_MeV,
            energy_shift_MeV=args.fusion_proton_energy_shift_MeV,
        )

    data_df = load_data(args.data, args.ring_selection, args.root_hist)
    all_ex = data_df["excitation_MeV"].to_numpy(dtype=float)
    all_ex = all_ex[np.isfinite(all_ex)]
    auto_lo, auto_hi = automatic_fit_range(
        all_ex,
        components,
        nsigma=args.support_nsigma,
        bin_width=args.bin_width_MeV,
    )
    fit_lo = auto_lo if args.fit_lo_MeV is None else float(args.fit_lo_MeV)
    fit_hi = auto_hi if args.fit_hi_MeV is None else float(args.fit_hi_MeV)
    if fit_hi <= fit_lo:
        raise ValueError(f"Invalid fit range {fit_lo:g}-{fit_hi:g} MeV")
    if (
        args.low_energy_floor_hi_MeV is not None
        and args.low_energy_floor_hi_MeV > fit_lo
    ):
        floor_hi = min(float(args.low_energy_floor_hi_MeV), fit_hi)
        if floor_hi > fit_lo:
            components, component_df = append_low_energy_floor(
                components,
                component_df,
                fit_lo_MeV=fit_lo,
                floor_hi_MeV=floor_hi,
                max_fraction=args.low_energy_floor_max_fraction,
            )
    component_df.to_csv(outdir / "MCMC_full_range_components.csv", index=False)

    x = all_ex[(all_ex >= fit_lo) & (all_ex <= fit_hi)]
    if len(x) == 0:
        raise ValueError(f"No data in fit range {fit_lo:g}-{fit_hi:g} MeV")
    bins = make_bins((fit_lo, fit_hi), args.bin_width_MeV)
    observed, _ = np.histogram(x, bins=bins)
    grid = np.linspace(fit_lo, fit_hi, args.grid_size)
    _, pdfs_grid = component_pdf(x, grid, components)
    bin_probs = bin_probabilities(grid, pdfs_grid, bins)
    prob_sums = bin_probs.sum(axis=1)
    if not np.allclose(prob_sums, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"Component bin probabilities do not sum to one: {prob_sums}")
    unsupported_bins = np.where((bin_probs.sum(axis=0) <= 0.0) & (observed > 0))[0]
    if len(unsupported_bins) > 0:
        examples = ", ".join(
            f"{bins[idx]:.3g}-{bins[idx + 1]:.3g} MeV: {int(observed[idx])}"
            for idx in unsupported_bins[:8]
        )
        raise ValueError(
            "Observed bins have zero model support. Add a low-energy floor "
            "component or extend a contaminant template. First unsupported bins: "
            + examples
        )

    n_comp = len(components)
    upper_bounds = component_upper_bounds(
        components,
        len(x),
        args.max_total_yield_factor,
    )
    initial_weights = np.array(
        [float(component.get("initial_weight", np.nan)) for component in components],
        dtype=float,
    )
    minuit = fit_binned_amplitudes(
        observed,
        bin_probs,
        upper_bounds,
        initial_weights=initial_weights,
    )
    minuit_yields = np.array([minuit.values[f"N{i}"] for i in range(n_comp)])

    nwalkers = args.nwalkers or max(64, 4 * n_comp)
    if nwalkers < 2 * n_comp:
        raise ValueError("--nwalkers must be at least 2 * n_components")
    rng = np.random.default_rng(args.seed)
    start_yields = np.clip(minuit_yields, 1e-3, 0.90 * upper_bounds)
    p0 = np.log(start_yields) + 2e-3 * rng.normal(size=(nwalkers, n_comp))
    p0 = np.minimum(p0, np.log(0.95 * upper_bounds))

    np.random.seed(args.seed)
    sampler = emcee.EnsembleSampler(
        nwalkers,
        n_comp,
        log_probability_binned,
        args=(observed, bin_probs, upper_bounds, args.likelihood_uncertainty_scale),
    )
    state = sampler.run_mcmc(
        p0,
        args.burnin_steps,
        progress=False,
        skip_initial_state_check=True,
    )
    sampler.reset()
    sampler.run_mcmc(state, args.production_steps, progress=False)

    try:
        tau = sampler.get_autocorr_time()
        if not np.all(np.isfinite(tau)):
            raise emcee.autocorr.AutocorrError(tau)
        thin = max(1, int(0.5 * np.max(tau)))
        tau_text = ", ".join(f"{value:.2f}" for value in tau)
    except emcee.autocorr.AutocorrError:
        thin = max(1, args.production_steps // 300)
        tau_text = f"not stable; thin={thin} used"

    samples_log = sampler.get_chain(flat=True, thin=thin)
    samples_yields = np.exp(samples_log)
    q16, q50, q84 = np.percentile(samples_yields, [16, 50, 84], axis=0)
    yield_minus = q50 - q16
    yield_plus = q84 - q50

    expected_mcmc = q50 @ bin_probs
    expected_minuit = minuit_yields @ bin_probs
    chi2_mcmc = pearson_chi2(observed, expected_mcmc, n_comp)
    chi2_mcmc_exp1 = pearson_chi2(observed, expected_mcmc, n_comp, min_expected=1.0)
    chi2_mcmc_exp5 = pearson_chi2(observed, expected_mcmc, n_comp, min_expected=5.0)
    plot_chi2_mcmc_exp5 = dict(chi2_mcmc_exp5)
    plot_chi2_mcmc_exp5["chi2"] = (
        float(chi2_mcmc_exp5["chi2"]) / args.plot_uncertainty_scale**2
    )
    plot_chi2_mcmc_exp5["chi2_ndf"] = (
        float(plot_chi2_mcmc_exp5["chi2"]) / int(plot_chi2_mcmc_exp5["ndf"])
    )
    plot_chi2_mcmc_exp5["p_value"] = float(
        chi2_dist.sf(
            float(plot_chi2_mcmc_exp5["chi2"]),
            int(plot_chi2_mcmc_exp5["ndf"]),
        )
    )
    chi2_minuit = pearson_chi2(observed, expected_minuit, n_comp)
    chi2_minuit_exp5 = pearson_chi2(
        observed, expected_minuit, n_comp, min_expected=5.0
    )
    deviance_mcmc = poisson_deviance(observed, expected_mcmc, n_comp)
    deviance_minuit = poisson_deviance(observed, expected_minuit, n_comp)
    cdf_mcmc = mixture_cdf_on_grid(q50, grid, pdfs_grid)
    ks_d, ks_d_plus, ks_d_minus = ks_statistic(x, grid, cdf_mcmc)
    ks_p_naive = float(kstwo.sf(ks_d, len(x)))
    ks_info = {
        "D": ks_d,
        "D_plus": ks_d_plus,
        "D_minus": ks_d_minus,
        "p_value": ks_p_naive,
    }

    log_likelihood_mcmc = multinomial_log_likelihood(observed, expected_mcmc)
    aic = 2.0 * n_comp - 2.0 * log_likelihood_mcmc
    bic = math.log(len(x)) * n_comp - 2.0 * log_likelihood_mcmc

    summary_df = component_df.copy()
    summary_df["Minuit_N"] = minuit_yields
    summary_df["MCMC_N_median"] = q50
    summary_df["MCMC_minus"] = yield_minus
    summary_df["MCMC_plus"] = yield_plus
    outdir.mkdir(parents=True, exist_ok=True)
    component_df.to_csv(outdir / "MCMC_full_range_components.csv", index=False)
    summary_df.to_csv(outdir / "MCMC_full_range_fit_summary.csv", index=False)

    written_plots: list[Path] = []
    written_plots.extend(
        plot_astrophysical_region_corner(
            outdir,
            components,
            samples_yields,
            q50,
            yield_minus,
            yield_plus,
            plot_formats=plot_formats,
            output_path=args.astrophysical_corner_output,
        )
    )
    written_plots.extend(
        plot_above10_region_corner(
            outdir,
            components,
            samples_yields,
            q50,
            yield_minus,
            yield_plus,
            plot_formats=plot_formats,
            output_path=args.above10_corner_output,
        )
    )
    if args.plot_selection == "individual-poisson-only":
        written_plots.extend(
            plot_spectrum(
                outdir,
                x,
                bins,
                components,
                q50,
                grid,
                pdfs_grid,
                plot_chi2_mcmc_exp5,
                individual_states=True,
                poisson_errorbars=True,
                uncertainty_scale=args.plot_uncertainty_scale,
                plot_formats=plot_formats,
                output_path=args.spectrum_plot_output,
            )
        )
    else:
        written_plots.extend(
            plot_spectrum(
                outdir,
                x,
                bins,
                components,
                q50,
                grid,
                pdfs_grid,
                plot_chi2_mcmc_exp5,
                uncertainty_scale=args.plot_uncertainty_scale,
                plot_formats=plot_formats,
            )
        )
        written_plots.extend(
            plot_spectrum(
                outdir,
                x,
                bins,
                components,
                q50,
                grid,
                pdfs_grid,
                plot_chi2_mcmc_exp5,
                poisson_errorbars=True,
                uncertainty_scale=args.plot_uncertainty_scale,
                plot_formats=plot_formats,
            )
        )
        written_plots.extend(
            plot_spectrum(
                outdir,
                x,
                bins,
                components,
                q50,
                grid,
                pdfs_grid,
                plot_chi2_mcmc_exp5,
                individual_states=True,
                uncertainty_scale=args.plot_uncertainty_scale,
                plot_formats=plot_formats,
            )
        )
        written_plots.extend(
            plot_spectrum(
                outdir,
                x,
                bins,
                components,
                q50,
                grid,
                pdfs_grid,
                plot_chi2_mcmc_exp5,
                individual_states=True,
                poisson_errorbars=True,
                uncertainty_scale=args.plot_uncertainty_scale,
                plot_formats=plot_formats,
            )
        )
        written_plots.extend(
            plot_component_lines(
                outdir,
                bins,
                components,
                q50,
                grid,
                pdfs_grid,
                plot_formats=plot_formats,
            )
        )
        if not args.no_posterior_draws_plot:
            written_plots.extend(
                plot_posterior_draws(
                    outdir,
                    x,
                    bins,
                    samples_yields,
                    q50,
                    grid,
                    pdfs_grid,
                    plot_formats=plot_formats,
                )
            )
            written_plots.extend(
                plot_low_proton_walker_draws(
                    outdir,
                    bins,
                    components,
                    samples_yields,
                    q50,
                    grid,
                    pdfs_grid,
                    plot_formats=plot_formats,
                )
            )

    source_counts = summary_df["template_source"].value_counts().to_dict()
    fusion_components = [component for component in components if is_fusion_component(component)]
    source_hist_values = (
        sorted(data_df["source_hist"].dropna().astype(str).unique().tolist())
        if "source_hist" in data_df
        else []
    )
    fusion_template_lines = []
    for component in fusion_components:
        fusion_template_lines.append(
            "fusion_template: "
            f"{component['name']}, "
            f"ejectile={component.get('ejectile', 'unknown')}, "
            f"region={component.get('energy_region', 'unknown')}, "
            f"source={component['source']}, "
            f"column={component['column']}, "
            "sample_range_MeV="
            f"{component['sample_range'][0]} {component['sample_range'][1]}, "
            f"bandwidth_MeV={component['bandwidth']}, "
            f"energy_shift_MeV={component.get('energy_shift_MeV', 0.0)}, "
            f"max_fraction={component['max_fraction']}, "
            f"n_source_events={len(load_sample_template_values(component))}"
        )
    if not fusion_template_lines:
        fusion_template_lines.append("fusion_templates: disabled")
    low_energy_floor_lines = []
    for component in components:
        if component.get("name") != LOW_ENERGY_FLOOR_NAME:
            continue
        low_energy_floor_lines.append(
            "low_energy_floor: "
            f"range_MeV={component['E_L']} {component['E_R']}, "
            f"max_fraction={component['max_fraction']}"
        )
    if not low_energy_floor_lines:
        low_energy_floor_lines.append("low_energy_floor: disabled")
    lines = [
        f"data_file: {args.data}",
        *([f"root_histogram: {source_hist_values[0]}"] if source_hist_values else []),
        f"states_csv: {args.states_csv}",
        f"fit_results_csv: {args.fit_results_csv}",
        f"template_ring_group: {args.template_ring_group}",
        f"fallback_fit_results_csv: {args.fallback_fit_results_csv or 'none'}",
        f"fallback_template_ring_group: "
        f"{args.fallback_template_ring_group if args.fallback_fit_results_csv is not None else 'none'}",
        f"ring_selection: {args.ring_selection}",
        "component_energy_shifts_MeV: "
        + (
            ", ".join(
                f"{name}={shift:g}"
                for name, shift in sorted(component_energy_shifts.items())
            )
            if component_energy_shifts
            else "none"
        ),
        "component_left_edge_shifts_MeV: "
        + (
            ", ".join(
                f"{name}={shift:g}"
                for name, shift in sorted(component_left_edge_shifts.items())
            )
            if component_left_edge_shifts
            else "none"
        ),
        "component_right_edge_shifts_MeV: "
        + (
            ", ".join(
                f"{name}={shift:g}"
                for name, shift in sorted(component_right_edge_shifts.items())
            )
            if component_right_edge_shifts
            else "none"
        ),
        "component_sigma_scales: "
        + (
            ", ".join(
                f"{name}={scale:g}"
                for name, scale in sorted(component_sigma_scales.items())
            )
            if component_sigma_scales
            else "none"
        ),
        "force_interpolate_components: "
        + (
            ", ".join(sorted(force_interpolate_components))
            if force_interpolate_components
            else "none"
        ),
        f"global_energy_shift_MeV: {args.global_energy_shift_MeV:g}",
        f"global_sigma_scale: {args.global_sigma_scale:g}",
        f"fit_range_MeV: {fit_lo:.6f} {fit_hi:.6f}",
        f"automatic_fit_range_MeV: {auto_lo:.6f} {auto_hi:.6f}",
        f"support_nsigma: {args.support_nsigma:g}",
        f"n_data_selected_before_fit_range: {len(all_ex)}",
        f"n_data_in_fit_range: {len(x)}",
        f"n_components: {n_comp}",
        "template_source_counts: "
        + ", ".join(f"{key}={value}" for key, value in sorted(source_counts.items())),
        *fusion_template_lines,
        *low_energy_floor_lines,
        f"bin_width_MeV: {args.bin_width_MeV:g}",
        f"n_bins: {len(bins) - 1}",
        f"nonzero_model_bins: {chi2_mcmc['model_bins_used']}",
        f"plot_uncertainty_scale: {args.plot_uncertainty_scale:.9f}",
        f"likelihood_uncertainty_scale: {args.likelihood_uncertainty_scale:.9f}",
        f"plot_pearson_expected_ge_5_chi2: {plot_chi2_mcmc_exp5['chi2']:.6f}",
        f"plot_pearson_expected_ge_5_ndf: {plot_chi2_mcmc_exp5['ndf']}",
        f"plot_pearson_expected_ge_5_chi2_ndf: {plot_chi2_mcmc_exp5['chi2_ndf']:.6f}",
        f"plot_pearson_expected_ge_5_p_value: {plot_chi2_mcmc_exp5['p_value']:.6g}",
        f"minuit_valid: {bool(minuit.valid)}",
        f"minuit_fval: {float(minuit.fval):.6f}",
        f"nwalkers: {nwalkers}",
        f"burnin_steps: {args.burnin_steps}",
        f"production_steps: {args.production_steps}",
        f"thin: {thin}",
        f"posterior_draws_used_for_median: {len(samples_yields)}",
        f"acceptance_fraction_mean: {float(np.mean(sampler.acceptance_fraction)):.6f}",
        f"autocorr_time: {tau_text}",
        "",
        "Goodness of fit using MCMC posterior median yields:",
        f" pearson_all_bins_chi2: {chi2_mcmc['chi2']:.6f}",
        f" pearson_all_bins_ndf: {chi2_mcmc['ndf']}",
        f" pearson_all_bins_chi2_ndf: {chi2_mcmc['chi2_ndf']:.6f}",
        f" pearson_all_bins_p_value: {chi2_mcmc['p_value']:.6g}",
        f" pearson_expected_ge_1_bins_used: {chi2_mcmc_exp1['model_bins_used']}",
        f" pearson_expected_ge_1_chi2: {chi2_mcmc_exp1['chi2']:.6f}",
        f" pearson_expected_ge_1_ndf: {chi2_mcmc_exp1['ndf']}",
        f" pearson_expected_ge_1_chi2_ndf: {chi2_mcmc_exp1['chi2_ndf']:.6f}",
        f" pearson_expected_ge_1_p_value: {chi2_mcmc_exp1['p_value']:.6g}",
        f" pearson_expected_ge_5_bins_used: {chi2_mcmc_exp5['model_bins_used']}",
        f" pearson_expected_ge_5_chi2: {chi2_mcmc_exp5['chi2']:.6f}",
        f" pearson_expected_ge_5_ndf: {chi2_mcmc_exp5['ndf']}",
        f" pearson_expected_ge_5_chi2_ndf: {chi2_mcmc_exp5['chi2_ndf']:.6f}",
        f" pearson_expected_ge_5_p_value: {chi2_mcmc_exp5['p_value']:.6g}",
        f" poisson_deviance: {deviance_mcmc['deviance']:.6f}",
        f" poisson_deviance_ndf: {deviance_mcmc['ndf']}",
        f" poisson_deviance_per_ndf: {deviance_mcmc['deviance_ndf']:.6f}",
        f" poisson_deviance_p_value: {deviance_mcmc['p_value']:.6g}",
        f" ks_D: {ks_d:.6f}",
        f" ks_D_plus: {ks_d_plus:.6f}",
        f" ks_D_minus: {ks_d_minus:.6f}",
        f" ks_p_naive_fixed_cdf: {ks_p_naive:.6g}",
        f" multinomial_log_likelihood: {log_likelihood_mcmc:.6f}",
        f" AIC: {aic:.6f}",
        f" BIC: {bic:.6f}",
        "",
        "Goodness of fit using Minuit maximum-likelihood yields:",
        f" pearson_all_bins_chi2: {chi2_minuit['chi2']:.6f}",
        f" pearson_all_bins_ndf: {chi2_minuit['ndf']}",
        f" pearson_all_bins_chi2_ndf: {chi2_minuit['chi2_ndf']:.6f}",
        f" pearson_all_bins_p_value: {chi2_minuit['p_value']:.6g}",
        f" pearson_expected_ge_5_bins_used: {chi2_minuit_exp5['model_bins_used']}",
        f" pearson_expected_ge_5_chi2: {chi2_minuit_exp5['chi2']:.6f}",
        f" pearson_expected_ge_5_ndf: {chi2_minuit_exp5['ndf']}",
        f" pearson_expected_ge_5_chi2_ndf: {chi2_minuit_exp5['chi2_ndf']:.6f}",
        f" pearson_expected_ge_5_p_value: {chi2_minuit_exp5['p_value']:.6g}",
        f" poisson_deviance: {deviance_minuit['deviance']:.6f}",
        f" poisson_deviance_ndf: {deviance_minuit['ndf']}",
        f" poisson_deviance_per_ndf: {deviance_minuit['deviance_ndf']:.6f}",
        f" poisson_deviance_p_value: {deviance_minuit['p_value']:.6g}",
        "",
        summary_df[
            [
                col
                for col in [
                    "component",
                    "input_Ex_MeV",
                    "Jpi",
                    "height_all",
                    "template_source",
                    "source_fit_states",
                    "Minuit_N",
                    "MCMC_N_median",
                    "MCMC_minus",
                    "MCMC_plus",
                ]
                if col in summary_df.columns
            ]
        ].to_string(index=False),
    ]
    summary_txt = outdir / "MCMC_full_range_goodness_of_fit.txt"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_txt.write_text("\n".join(lines) + "\n")

    print(f"wrote {outdir / 'MCMC_full_range_components.csv'}")
    print(f"wrote {outdir / 'MCMC_full_range_fit_summary.csv'}")
    print(f"wrote {summary_txt}")
    for path in written_plots:
        print(f"wrote {path}")
    print(
        "chi2_expected_ge_5_ndf_posterior_median="
        f"{chi2_mcmc_exp5['chi2']:.6f}/{chi2_mcmc_exp5['ndf']}="
        f"{chi2_mcmc_exp5['chi2_ndf']:.6f}; p={chi2_mcmc_exp5['p_value']:.6g}"
    )
    print(
        "poisson_deviance_ndf_posterior_median="
        f"{deviance_mcmc['deviance']:.6f}/{deviance_mcmc['ndf']}="
        f"{deviance_mcmc['deviance_ndf']:.6f}; p={deviance_mcmc['p_value']:.6g}"
    )
    print(f"ks_D={ks_d:.6f}; p_naive={ks_p_naive:.6g}")


if __name__ == "__main__":
    main()
