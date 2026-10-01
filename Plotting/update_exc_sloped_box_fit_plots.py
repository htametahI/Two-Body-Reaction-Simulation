#!/usr/bin/env python3
"""Regenerate output/Exc_sloped_box_fit plots with optimized all-ring ranges."""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-s2223")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib-s2223-xdg")
logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from optimize_sloped_box_fit_ranges import (
    EX_COL,
    baseline_range,
    candidate_ranges,
    discover_states,
    evaluate_fit,
    fit_binned_scan,
    fit_unbinned_final,
    load_reconstructed_ex,
    optimize_group,
    sloped_smooth_box_shape,
    state_label,
)


matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,
        "figure.dpi": 300,
        "savefig.dpi": 200,
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
    }
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
PLOT_DIR = OUT / "Exc_sloped_box_fit"
OPTIMIZED_CSV = OUT / "optimized_sloped_box_fit_ranges" / "optimized_sloped_box_fit_ranges.csv"
LOCAL_OPTIMIZED_CSV = PLOT_DIR / "optimized_all_ring_fit_ranges.csv"
FIT_RESULTS_CSV = PLOT_DIR / "sloped_box_fit_results.csv"
MIN_EDGE_SIGMA_MEV = 0.01
FIT_PLOT_BIN_WIDTH_MEV = 0.01


def fit_plot_bins(lo: float, hi: float) -> np.ndarray:
    start = np.floor(lo / FIT_PLOT_BIN_WIDTH_MEV) * FIT_PLOT_BIN_WIDTH_MEV
    stop = np.ceil(hi / FIT_PLOT_BIN_WIDTH_MEV) * FIT_PLOT_BIN_WIDTH_MEV
    bins = np.arange(
        start,
        stop + 0.5 * FIT_PLOT_BIN_WIDTH_MEV,
        FIT_PLOT_BIN_WIDTH_MEV,
    )
    if len(bins) < 2:
        return np.array([lo, hi], dtype=float)
    return bins


def best_binned_fallback(
    values: np.ndarray,
    input_ex: float,
    *,
    min_events: int = 1000,
    min_retained_fraction: float = 0.80,
) -> tuple[object, dict[str, object]] | None:
    base_candidate = baseline_range(values)
    candidates = [base_candidate] + candidate_ranges(values, base_candidate)
    fits: list[tuple[float, float, float, object, dict[str, object]]] = []

    for candidate in candidates:
        if not (candidate.lo <= input_ex <= candidate.hi):
            continue
        x = values[(values >= candidate.lo) & (values <= candidate.hi)]
        retained_fraction = len(x) / len(values)
        if len(x) < min_events or retained_fraction < min_retained_fraction:
            continue
        scan_fit = fit_binned_scan(values, (candidate.lo, candidate.hi), n_bins=160)
        if scan_fit is None:
            continue

        grid = np.linspace(candidate.lo, candidate.hi, 8000)
        fit = evaluate_fit(scan_fit["x"], grid, scan_fit["minuit"])
        fits.append(
            (
                float(fit["ks_p"]),
                retained_fraction,
                -abs(float(fit["mean_fit_MeV"]) - input_ex),
                candidate,
                fit,
            )
        )

    if not fits:
        return None
    _, _, _, candidate, fit = sorted(fits, reverse=True, key=lambda row: row[:3])[0]
    return candidate, fit


def source_mtime(states: list[str]) -> float:
    mtimes: list[float] = []
    for state in states:
        for suffix in ("truth_kinematics", "exit_kinematics"):
            path = OUT / f"{state}_{suffix}.csv"
            if path.exists():
                mtimes.append(path.stat().st_mtime)
    return max(mtimes, default=0.0)


def load_current_optimized_ranges(
    states: list[str],
) -> tuple[dict[str, pd.Series], Path | None]:
    latest_source = source_mtime(states)
    candidates = [LOCAL_OPTIMIZED_CSV, OPTIMIZED_CSV]
    existing = sorted(
        [path for path in candidates if path.exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in existing:
        if path.stat().st_mtime < latest_source:
            continue
        optimized = pd.read_csv(path)
        if "ring_group" not in optimized.columns:
            continue
        optimized = optimized[optimized["ring_group"] == "all_rings"].copy()
        if optimized.empty:
            continue
        bad_edge = (
            (optimized["sigma_L_MeV"] < MIN_EDGE_SIGMA_MEV)
            | (optimized["sigma_R_MeV"] < MIN_EDGE_SIGMA_MEV)
        )
        if bad_edge.any():
            bad_states = ", ".join(optimized.loc[bad_edge, "state"].astype(str))
            print(
                f"Ignoring {path}: edge sigma below {MIN_EDGE_SIGMA_MEV:g} MeV "
                f"for {bad_states}"
            )
            continue
        return {str(row["state"]): row for _, row in optimized.iterrows()}, path
    return {}, None


def compute_optimized_all_ring_ranges(states: list[str]) -> dict[str, pd.Series]:
    print("Optimized all-ring ranges are missing or stale; recomputing.")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        min_events=1000,
        min_retained_fraction=0.80,
        min_edge_sigma_MeV=MIN_EDGE_SIGMA_MEV,
        min_width_MeV=0.45,
        max_width_MeV=2.20,
        n_grid_scan=2500,
        n_grid_final=8000,
        n_bins_scan=160,
    )
    rows: list[dict[str, object]] = []
    for state in states:
        state_df = load_reconstructed_ex(ROOT, state)
        best_row, _ = optimize_group(state, state_df, "all_rings", args)
        if best_row is None:
            print(f"  {state}: no optimized all-ring range; using fallback fit")
            continue
        rows.append(best_row)
        print(
            f"  {state}: optimized {best_row['fit_lo_MeV']:.4f}-"
            f"{best_row['fit_hi_MeV']:.4f} MeV, p={best_row['ks_p']:.3g}"
        )

    optimized = pd.DataFrame(rows)
    optimized.to_csv(LOCAL_OPTIMIZED_CSV, index=False)
    print(f"wrote {LOCAL_OPTIMIZED_CSV}")
    return {str(row["state"]): row for _, row in optimized.iterrows()}


def optimized_all_ring_ranges(states: list[str]) -> dict[str, pd.Series]:
    ranges, source = load_current_optimized_ranges(states)
    if ranges:
        print(f"Using optimized all-ring ranges from {source}")
        return ranges
    return compute_optimized_all_ring_ranges(states)


def display_range(values: np.ndarray, fit_lo: float, fit_hi: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return fit_lo, fit_hi
    tail_lo, tail_hi = np.percentile(finite, [0.05, 99.95])
    lo = min(float(tail_lo), fit_lo)
    hi = max(float(tail_hi), fit_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return fit_lo, fit_hi
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def plot_fit(state: str, row: dict[str, object], values: np.ndarray) -> None:
    lo = float(row["fit_lo_MeV"])
    hi = float(row["fit_hi_MeV"])
    x = values[(values >= lo) & (values <= hi)]
    if "plot_lo_MeV" in row and "plot_hi_MeV" in row:
        plot_lo = float(row["plot_lo_MeV"])
        plot_hi = float(row["plot_hi_MeV"])
    else:
        plot_lo, plot_hi = display_range(values, lo, hi)
    plot_x = values[(values >= plot_lo) & (values <= plot_hi)]
    grid = np.linspace(plot_lo, plot_hi, 10000)
    shape = sloped_smooth_box_shape(
        grid,
        float(row["E_L_MeV"]),
        float(row["E_R_MeV"]),
        float(row["sigma_L_MeV"]),
        float(row["sigma_R_MeV"]),
        float(row["slope_MeV_inv"]),
    )
    pdf = shape / trapezoid(shape, grid)

    fig, ax = plt.subplots(figsize=(8, 5))
    counts, edges, _ = ax.hist(
        plot_x,
        bins=fit_plot_bins(plot_lo, plot_hi),
        histtype="stepfilled",
        alpha=0.55,
        color="#FE019A",
        edgecolor="black",
        linewidth=0.6,
        label="Simulated Data",
    )
    bin_width = edges[1] - edges[0]
    fit_counts = len(plot_x) * bin_width * pdf
    ax.plot(grid, fit_counts, color="black", linewidth=2, label="Fit")

    half_height_counts = 0.5 * np.max(fit_counts)
    ax.hlines(
        half_height_counts,
        float(row["fwhm_left_MeV"]),
        float(row["fwhm_right_MeV"]),
        color="gray",
        linestyle=":",
        linewidth=2,
        label="FWHM",
    )
    ax.axvline(
        float(row["input_Ex_MeV"]),
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=r"Input $E_x$",
    )
    ax.text(
        0.05,
        0.98,
        rf"$E_0$ = {float(row['E0_MeV']):.4f} MeV"
        "\n"
        rf"$\langle E_x\rangle$ = {float(row['mean_fit_MeV']):.4f} MeV"
        "\n"
        rf"$W$ = {float(row['box_width_MeV']):.4f} MeV"
        "\n"
        rf"$\mathrm{{FWHM}}$ = {float(row['fwhm_MeV']):.4f} MeV"
        "\n"
        rf"$E_L$ = {float(row['E_L_MeV']):.4f} MeV"
        "\n"
        rf"$E_R$ = {float(row['E_R_MeV']):.4f} MeV"
        "\n"
        rf"$\sigma_L$ = {float(row['sigma_L_MeV']):.4f} MeV"
        "\n"
        rf"$\sigma_R$ = {float(row['sigma_R_MeV']):.4f} MeV"
        "\n"
        rf"$k$ = {float(row['slope_MeV_inv']):.4f} MeV$^{{-1}}$"
        "\n"
        rf"$p$ = {float(row['ks_p']):.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    ax.set_xlim(plot_lo, plot_hi)
    ax.set_xlabel(r"$E_x^\mathrm{calc}$ [MeV]")
    ax.set_ylabel("Counts")
    ax.set_title(rf"{state_label(state)}: Maximum Likelihood Fit")
    ax.legend(fontsize=8)
    ax.minorticks_on()
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.20)
    ax.tick_params(which="both", top=False, right=False)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = PLOT_DIR / f"{state}fit.png"
    pdf_path = PLOT_DIR / f"{state}fit.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    states = discover_states(ROOT)
    ranges = optimized_all_ring_ranges(states)
    rows: list[dict[str, object]] = []

    for state in states:
        df = load_reconstructed_ex(ROOT, state)
        values = df[EX_COL].to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values) < 20:
            print(f"Skipping {state}: not enough reconstructed Ex values")
            continue
        input_ex = float(df["Ex_MeV"].dropna().iloc[0])
        fit_method = "unbinned_likelihood"

        if state in ranges:
            range_row = ranges[state]
            fit_range = (float(range_row["fit_lo_MeV"]), float(range_row["fit_hi_MeV"]))
            strategy = str(range_row["strategy"])
        else:
            candidate = baseline_range(values)
            fit_range = (candidate.lo, candidate.hi)
            strategy = "fallback_peak_pm_1MeV"

        fit = fit_unbinned_final(values, fit_range, n_grid=8000)
        if fit is None and strategy != "fallback_peak_pm_1MeV":
            candidate = baseline_range(values)
            fit_range = (candidate.lo, candidate.hi)
            strategy = "fallback_peak_pm_1MeV"
            fit = fit_unbinned_final(values, fit_range, n_grid=8000)
        if fit is None:
            binned_fallback = best_binned_fallback(values, input_ex)
            if binned_fallback is None:
                print(f"Skipping {state}: fit failed")
                continue
            candidate, fit = binned_fallback
            fit_range = (float(candidate.lo), float(candidate.hi))
            strategy = f"binned_fallback_{candidate.strategy}"
            fit_method = "binned_likelihood"

        minuit = fit["minuit"]
        x = fit["x"]
        plot_lo, plot_hi = display_range(values, fit_range[0], fit_range[1])
        row = {
            "state": state,
            "ring_group": "all_rings",
            "n_events_group": len(values),
            "input_Ex_MeV": input_ex,
            "fit_lo_MeV": fit_range[0],
            "fit_hi_MeV": fit_range[1],
            "fit_width_MeV": fit_range[1] - fit_range[0],
            "plot_lo_MeV": plot_lo,
            "plot_hi_MeV": plot_hi,
            "strategy": strategy,
            "fit_method": fit_method,
            "n_events_fit": len(x),
            "retained_fraction": len(x) / len(values),
            "E0_MeV": 0.5 * (minuit.values["e_left"] + minuit.values["e_right"]),
            "mean_fit_MeV": fit["mean_fit_MeV"],
            "box_width_MeV": minuit.values["e_right"] - minuit.values["e_left"],
            "fwhm_MeV": fit["fwhm_MeV"],
            "fwhm_left_MeV": fit["fwhm_left_MeV"],
            "fwhm_right_MeV": fit["fwhm_right_MeV"],
            "E_L_MeV": minuit.values["e_left"],
            "E_R_MeV": minuit.values["e_right"],
            "sigma_L_MeV": minuit.values["sigma_left"],
            "sigma_R_MeV": minuit.values["sigma_right"],
            "slope_MeV_inv": minuit.values["slope"],
            "ks_D": fit["ks_D"],
            "ks_p": fit["ks_p"],
            "minuit_valid": bool(minuit.valid),
            "minuit_fval": float(minuit.fval),
            "plot_file_png": str(PLOT_DIR / f"{state}fit.png"),
            "plot_file_pdf": str(PLOT_DIR / f"{state}fit.pdf"),
        }
        plot_fit(state, row, values)
        rows.append(row)
        print(
            f"{state}: wrote {PLOT_DIR / f'{state}fit.pdf'} and "
            f"{PLOT_DIR / f'{state}fit.png'}; p={float(row['ks_p']):.3g}"
        )

    pd.DataFrame(rows).to_csv(FIT_RESULTS_CSV, index=False)
    print(f"wrote {FIT_RESULTS_CSV}")


if __name__ == "__main__":
    main()
