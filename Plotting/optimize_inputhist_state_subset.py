#!/usr/bin/env python3
"""Greedy state-subset scan for the inputHist full-range fits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mcmc_full_excitation_fit as fit  # noqa: E402
from mcmc_fit_diagnostics import bin_probabilities, component_pdf  # noqa: E402
import run_inputhist_full_range_fits as runner  # noqa: E402


@dataclass(frozen=True)
class FitContext:
    label: str
    root_hist: str
    components: list[dict[str, object]]
    names: list[str]
    observed: np.ndarray
    bin_probs: np.ndarray
    initial_weights: np.ndarray
    upper_bounds: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Greedily remove states from the inputHist full-range model and keep "
            "removals that improve an aggregate goodness metric."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=runner.DEFAULT_DATA)
    parser.add_argument("--states-csv", type=Path, default=runner.DEFAULT_STATES)
    parser.add_argument("--fit-results-csv", type=Path, default=runner.DEFAULT_FIT_RESULTS)
    parser.add_argument(
        "--ring-chunk-fit-results-csv",
        type=Path,
        default=runner.DEFAULT_RING_CHUNK_FIT_RESULTS,
    )
    parser.add_argument(
        "--output-states-csv",
        type=Path,
        default=(
            runner.OUT
            / "MCMC_full_excitation_range"
            / "pptx_snapshot_states_optimized.csv"
        ),
    )
    parser.add_argument(
        "--report-txt",
        type=Path,
        default=(
            runner.OUT
            / "MCMC_full_excitation_range"
            / "state_subset_scan_report.txt"
        ),
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=(
            runner.OUT
            / "MCMC_full_excitation_range"
            / "state_subset_scan_history.csv"
        ),
    )
    parser.add_argument(
        "--metric",
        choices=("pearson_ge5", "poisson_deviance"),
        default="pearson_ge5",
        help="Aggregate reduced metric to minimize.",
    )
    parser.add_argument(
        "--fit-label",
        action="append",
        choices=[label for label, _, _ in runner.INCLUSIVE_HISTOGRAMS],
        default=None,
        help=(
            "Inclusive fit label included in the aggregate objective. Repeat for "
            "multiple labels; omitted means all, RR0, RR1, RR2, RR3, and RR4."
        ),
    )
    parser.add_argument("--fit-lo-MeV", type=float, default=runner.DEFAULT_FIT_LO_MEV)
    parser.add_argument("--fit-hi-MeV", type=float, default=runner.DEFAULT_FIT_HI_MEV)
    parser.add_argument("--bin-width-MeV", type=float, default=0.10)
    parser.add_argument("--grid-size", type=int, default=12000)
    parser.add_argument("--match-tolerance-keV", type=float, default=3.0)
    parser.add_argument("--max-total-yield-factor", type=float, default=2.5)
    parser.add_argument(
        "--fusion-proton-bandwidth-MeV",
        type=float,
        default=runner.DEFAULT_FUSION_PROTON_BANDWIDTH_MEV,
    )
    parser.add_argument(
        "--fusion-proton-energy-shift-MeV",
        type=float,
        default=runner.DEFAULT_FUSION_PROTON_ENERGY_SHIFT_MEV,
    )
    parser.add_argument(
        "--component-energy-shift",
        action="append",
        default=list(runner.DEFAULT_COMPONENT_ENERGY_SHIFTS),
        metavar="COMPONENT:SHIFT_MEV",
    )
    parser.add_argument(
        "--component-left-edge-shift",
        action="append",
        default=list(runner.DEFAULT_COMPONENT_LEFT_EDGE_SHIFTS),
        metavar="COMPONENT:SHIFT_MEV",
    )
    parser.add_argument(
        "--component-right-edge-shift",
        action="append",
        default=list(runner.DEFAULT_COMPONENT_RIGHT_EDGE_SHIFTS),
        metavar="COMPONENT:SHIFT_MEV",
    )
    parser.add_argument(
        "--component-sigma-scale",
        action="append",
        default=list(runner.DEFAULT_COMPONENT_SIGMA_SCALES),
        metavar="COMPONENT:SCALE",
    )
    parser.add_argument(
        "--force-interpolate-component",
        action="append",
        default=list(runner.DEFAULT_FORCE_INTERPOLATE_COMPONENTS),
        metavar="COMPONENT",
    )
    parser.add_argument(
        "--keep-state",
        action="append",
        default=[],
        metavar="COMPONENT",
        help="State component that the greedy removal scan is not allowed to remove.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Minimum aggregate reduced-metric improvement required to accept a removal.",
    )
    return parser.parse_args()


def prepare_contexts(
    args: argparse.Namespace,
    states: pd.DataFrame,
) -> list[FitContext]:
    component_energy_shifts = fit.parse_component_energy_shifts(
        args.component_energy_shift,
        "--component-energy-shift",
    )
    component_left_edge_shifts = fit.parse_component_energy_shifts(
        args.component_left_edge_shift,
        "--component-left-edge-shift",
    )
    component_right_edge_shifts = fit.parse_component_energy_shifts(
        args.component_right_edge_shift,
        "--component-right-edge-shift",
    )
    component_sigma_scales = fit.parse_component_scales(
        args.component_sigma_scale,
        "--component-sigma-scale",
    )
    force_interpolate_components = {
        name.strip() for name in args.force_interpolate_component if name.strip()
    }
    optimized_template_labels = set(runner.DEFAULT_OPTIMIZED_RING_CHUNK_TEMPLATE_LABELS)
    selected_labels = set(args.fit_label or [])
    histogram_specs = [
        spec
        for spec in runner.INCLUSIVE_HISTOGRAMS
        if not selected_labels or spec[0] in selected_labels
    ]
    per_fit_global_energy_shifts = dict(runner.DEFAULT_RING_CHUNK_GLOBAL_ENERGY_SHIFTS_MEV)
    per_fit_global_sigma_scales = dict(runner.DEFAULT_RING_CHUNK_GLOBAL_SIGMA_SCALES)

    contexts: list[FitContext] = []
    for label, root_hist, template_ring_group in histogram_specs:
        base_label = runner.base_fit_label(label)
        use_chunk_templates = (
            base_label != "all" and base_label in optimized_template_labels
        )
        fit_results_csv = (
            args.ring_chunk_fit_results_csv if use_chunk_templates else args.fit_results_csv
        )
        primary_group = template_ring_group if use_chunk_templates else "all_rings"
        fits = fit.load_existing_fits(fit_results_csv, primary_group)
        fallback_fits = (
            fit.load_existing_fits(args.fit_results_csv, "all_rings")
            if use_chunk_templates
            else None
        )
        components, component_df = fit.build_components(
            states,
            fits,
            match_tolerance_keV=args.match_tolerance_keV,
            fallback_fits=fallback_fits,
            force_interpolate_components=force_interpolate_components,
        )
        components, component_df = fit.apply_component_energy_shifts(
            components,
            component_df,
            component_energy_shifts,
        )
        components, component_df = fit.apply_component_edge_shifts(
            components,
            component_df,
            component_left_edge_shifts,
            component_right_edge_shifts,
        )
        components, component_df = fit.apply_component_sigma_scales(
            components,
            component_df,
            component_sigma_scales,
        )
        components, component_df = fit.apply_global_component_adjustments(
            components,
            component_df,
            energy_shift_MeV=float(
                per_fit_global_energy_shifts.get(
                    label,
                    per_fit_global_energy_shifts.get(base_label, 0.0),
                )
            ),
            sigma_scale=float(
                per_fit_global_sigma_scales.get(
                    label,
                    per_fit_global_sigma_scales.get(base_label, 1.0),
                )
            ),
        )
        components, component_df = fit.append_fusion_contaminants(
            components,
            component_df,
            bandwidth_MeV=args.fusion_proton_bandwidth_MeV,
            energy_shift_MeV=args.fusion_proton_energy_shift_MeV,
        )
        data_df = fit.load_data(args.data, "all", root_hist)
        all_ex = data_df["excitation_MeV"].to_numpy(dtype=float)
        all_ex = all_ex[np.isfinite(all_ex)]
        x = all_ex[(all_ex >= args.fit_lo_MeV) & (all_ex <= args.fit_hi_MeV)]
        if len(x) == 0:
            raise ValueError(f"No data in fit range for {label}:{root_hist}")
        bins = fit.make_bins((args.fit_lo_MeV, args.fit_hi_MeV), args.bin_width_MeV)
        observed, _ = np.histogram(x, bins=bins)
        grid = np.linspace(args.fit_lo_MeV, args.fit_hi_MeV, args.grid_size)
        _, pdfs_grid = component_pdf(x, grid, components)
        bin_probs = bin_probabilities(grid, pdfs_grid, bins)
        initial_weights = np.asarray(
            [
                float(component.get("initial_weight", np.nan))
                if np.isfinite(float(component.get("initial_weight", np.nan)))
                else np.nan
                for component in components
            ],
            dtype=float,
        )
        upper_bounds = fit.component_upper_bounds(
            components,
            len(x),
            args.max_total_yield_factor,
        )
        contexts.append(
            FitContext(
                label=label,
                root_hist=root_hist,
                components=components,
                names=[str(component["name"]) for component in components],
                observed=observed,
                bin_probs=bin_probs,
                initial_weights=initial_weights,
                upper_bounds=upper_bounds,
            )
        )
    return contexts


def fit_context_subset(
    context: FitContext,
    active_states: set[str],
) -> dict[str, float | int | str]:
    indices = [
        idx
        for idx, component in enumerate(context.components)
        if component.get("kind", "sloped_box") != "sloped_box"
        or str(component["name"]) in active_states
    ]
    observed = context.observed
    bin_probs = context.bin_probs[indices]
    upper_bounds = context.upper_bounds[indices]
    initial = context.initial_weights[indices]
    minuit = fit.fit_binned_amplitudes(
        observed,
        bin_probs,
        upper_bounds,
        initial_weights=initial,
    )
    yields = np.asarray([minuit.values[f"N{i}"] for i in range(len(indices))])
    expected = yields @ bin_probs
    pearson = fit.pearson_chi2(observed, expected, len(indices), min_expected=5.0)
    poisson = fit.poisson_deviance(observed, expected, len(indices))
    return {
        "label": context.label,
        "pearson_chi2": float(pearson["chi2"]),
        "pearson_ndf": int(pearson["ndf"]),
        "pearson_chi2_ndf": float(pearson["chi2_ndf"]),
        "poisson_deviance": float(poisson["deviance"]),
        "poisson_ndf": int(poisson["ndf"]),
        "poisson_deviance_ndf": float(poisson["deviance_ndf"]),
        "n_components": len(indices),
        "minuit_valid": bool(minuit.valid),
    }


def evaluate_subset(
    contexts: list[FitContext],
    active_states: set[str],
    metric: str,
) -> tuple[float, dict[str, float | int], list[dict[str, float | int | str]]]:
    rows = [fit_context_subset(context, active_states) for context in contexts]
    pearson_chi2 = float(sum(float(row["pearson_chi2"]) for row in rows))
    pearson_ndf = int(sum(int(row["pearson_ndf"]) for row in rows))
    poisson_deviance = float(sum(float(row["poisson_deviance"]) for row in rows))
    poisson_ndf = int(sum(int(row["poisson_ndf"]) for row in rows))
    aggregate = {
        "pearson_chi2": pearson_chi2,
        "pearson_ndf": pearson_ndf,
        "pearson_chi2_ndf": pearson_chi2 / pearson_ndf,
        "poisson_deviance": poisson_deviance,
        "poisson_ndf": poisson_ndf,
        "poisson_deviance_ndf": poisson_deviance / poisson_ndf,
        "n_states": len(active_states),
    }
    score_key = (
        "pearson_chi2_ndf"
        if metric == "pearson_ge5"
        else "poisson_deviance_ndf"
    )
    return float(aggregate[score_key]), aggregate, rows


def summary_row(
    step: int,
    action: str,
    score: float,
    aggregate: dict[str, float | int],
) -> dict[str, float | int | str]:
    return {
        "step": step,
        "action": action,
        "score": score,
        **aggregate,
    }


def main() -> None:
    args = parse_args()
    raw_states = pd.read_csv(args.states_csv)
    states = fit.read_state_table(args.states_csv)
    contexts = prepare_contexts(args, states)
    active_states = set(states["component"].astype(str))
    state_order = list(states["component"].astype(str))
    keep_states = {name.strip() for name in args.keep_state if name.strip()}
    unknown_keep_states = sorted(keep_states - active_states)
    if unknown_keep_states:
        raise ValueError("--keep-state names not found: " + ", ".join(unknown_keep_states))
    score, aggregate, per_fit_rows = evaluate_subset(contexts, active_states, args.metric)
    history = [summary_row(0, "start_all_states", score, aggregate)]
    print(
        f"start: score={score:.6g}, states={len(active_states)}, "
        f"pearson_ge5={aggregate['pearson_chi2_ndf']:.6g}, "
        f"poisson={aggregate['poisson_deviance_ndf']:.6g}",
        flush=True,
    )

    removed: list[str] = []
    step = 0
    while len(active_states) > 1:
        trials: list[
            tuple[str, float, dict[str, float | int], list[dict[str, float | int | str]]]
        ] = []
        for candidate in state_order:
            if candidate not in active_states:
                continue
            if candidate in keep_states:
                continue
            trial_states = active_states - {candidate}
            trial_score, trial_aggregate, trial_rows = evaluate_subset(
                contexts,
                trial_states,
                args.metric,
            )
            trials.append((candidate, trial_score, trial_aggregate, trial_rows))
        candidate, best_score, best_aggregate, best_rows = min(
            trials,
            key=lambda item: item[1],
        )
        improvement = score - best_score
        if improvement <= args.tolerance:
            break
        active_states.remove(candidate)
        removed.append(candidate)
        step += 1
        score = best_score
        aggregate = best_aggregate
        per_fit_rows = best_rows
        history.append(summary_row(step, f"remove {candidate}", score, aggregate))
        print(
            f"step {step}: removed {candidate}, score={score:.6g}, "
            f"improvement={improvement:.6g}, states={len(active_states)}",
            flush=True,
        )

    # One add-back pass catches cases where an early greedy removal becomes useful again.
    for candidate in list(removed):
        trial_states = active_states | {candidate}
        trial_score, trial_aggregate, trial_rows = evaluate_subset(
            contexts,
            trial_states,
            args.metric,
        )
        improvement = score - trial_score
        if improvement <= args.tolerance:
            continue
        active_states.add(candidate)
        removed.remove(candidate)
        step += 1
        score = trial_score
        aggregate = trial_aggregate
        per_fit_rows = trial_rows
        history.append(summary_row(step, f"add_back {candidate}", score, aggregate))
        print(
            f"step {step}: added back {candidate}, score={score:.6g}, "
            f"improvement={improvement:.6g}, states={len(active_states)}",
            flush=True,
        )

    keep_mask = states["component"].astype(str).isin(active_states)
    kept_raw = raw_states.loc[keep_mask, ["Ex_MeV", "Jpi", "BR_to_1809_percent", "Ref"]]
    args.output_states_csv.parent.mkdir(parents=True, exist_ok=True)
    kept_raw.to_csv(args.output_states_csv, index=False)
    pd.DataFrame(history).to_csv(args.history_csv, index=False)
    per_fit_df = pd.DataFrame(per_fit_rows)

    kept_rows = states.loc[keep_mask, ["component", "input_Ex_MeV", "Jpi"]]
    removed_rows = states.loc[~keep_mask, ["component", "input_Ex_MeV", "Jpi"]]
    lines = [
        f"metric: {args.metric}",
        f"fit_labels: {', '.join(context.label for context in contexts)}",
        f"fit_range_MeV: {args.fit_lo_MeV:g} {args.fit_hi_MeV:g}",
        f"bin_width_MeV: {args.bin_width_MeV:g}",
        "kept_fixed_states: "
        + (", ".join(sorted(keep_states)) if keep_states else "none"),
        f"final_score: {score:.8g}",
        f"aggregate_pearson_expected_ge5_chi2_ndf: {aggregate['pearson_chi2_ndf']:.8g}",
        f"aggregate_poisson_deviance_ndf: {aggregate['poisson_deviance_ndf']:.8g}",
        f"n_states_kept: {len(active_states)}",
        f"n_states_removed: {len(removed_rows)}",
        "",
        "kept_states:",
        *(
            f"  {row.component}: {row.input_Ex_MeV:.6g} MeV, Jpi={row.Jpi}"
            for row in kept_rows.itertuples(index=False)
        ),
        "",
        "removed_states:",
        *(
            f"  {row.component}: {row.input_Ex_MeV:.6g} MeV, Jpi={row.Jpi}"
            for row in removed_rows.itertuples(index=False)
        ),
        "",
        "per_fit_final_metrics:",
        per_fit_df.to_string(index=False),
    ]
    args.report_txt.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output_states_csv}", flush=True)
    print(f"wrote {args.history_csv}", flush=True)
    print(f"wrote {args.report_txt}", flush=True)


if __name__ == "__main__":
    main()
