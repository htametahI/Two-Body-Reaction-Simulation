#!/usr/bin/env python3
"""Apply a post-fit Pearson overdispersion scale to MCMC yield intervals."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from scipy.stats import chi2 as chi2_dist


ROOT = Path(__file__).resolve().parents[1]
FIT_ROOT = ROOT / "output" / "MCMC_full_excitation_range_inputHist"
DEFAULT_FIT_SUMMARY = FIT_ROOT / "fit_outputs" / "all" / "MCMC_full_range_fit_summary.csv"
DEFAULT_GOODNESS = FIT_ROOT / "fit_outputs" / "all" / "MCMC_full_range_goodness_of_fit.txt"
DEFAULT_SELECTED_YIELDS = ROOT / "input" / "astro_state_selected_mcmc_yields.csv"
DEFAULT_OUTPUT_FIT_SUMMARY = (
    FIT_ROOT / "fit_outputs" / "all" / "MCMC_full_range_fit_summary_p050.csv"
)
DEFAULT_OUTPUT_SELECTED_YIELDS = ROOT / "input" / "astro_state_selected_mcmc_yields_p050.csv"
DEFAULT_OUTPUT_REPORT = FIT_ROOT / "fit_outputs" / "all" / "MCMC_uncertainty_inflation_p050.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inflate MCMC yield interval widths by the common Pearson-error scale "
            "needed to reach a requested goodness-of-fit p-value."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fit-summary", type=Path, default=DEFAULT_FIT_SUMMARY)
    parser.add_argument("--goodness", type=Path, default=DEFAULT_GOODNESS)
    parser.add_argument("--selected-yields", type=Path, default=DEFAULT_SELECTED_YIELDS)
    parser.add_argument("--target-p-value", type=float, default=0.5)
    parser.add_argument("--output-fit-summary", type=Path, default=DEFAULT_OUTPUT_FIT_SUMMARY)
    parser.add_argument(
        "--output-selected-yields", type=Path, default=DEFAULT_OUTPUT_SELECTED_YIELDS
    )
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    return parser.parse_args()


def goodness_value(text: str, key: str) -> float:
    match = re.search(rf"^ {re.escape(key)}:\s+(.+)$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not find {key!r} in goodness-of-fit report")
    return float(match.group(1))


def inflation_scale(goodness_path: Path, target_p_value: float) -> dict[str, float]:
    if not 0.0 < target_p_value < 1.0:
        raise ValueError("--target-p-value must be between zero and one")
    text = goodness_path.read_text(encoding="utf-8")
    observed_chi2 = goodness_value(text, "pearson_expected_ge_5_chi2")
    ndf = int(goodness_value(text, "pearson_expected_ge_5_ndf"))
    if observed_chi2 <= 0.0 or ndf <= 0:
        raise ValueError("Pearson chi-square and degrees of freedom must be positive")
    target_chi2 = float(chi2_dist.isf(target_p_value, ndf))
    scale = math.sqrt(observed_chi2 / target_chi2)
    return {
        "observed_chi2": observed_chi2,
        "ndf": float(ndf),
        "original_p_value": float(chi2_dist.sf(observed_chi2, ndf)),
        "target_p_value": target_p_value,
        "target_chi2": target_chi2,
        "scale": scale,
        "adjusted_chi2": observed_chi2 / scale**2,
    }


def inflate_table(path: Path, scale: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"MCMC_minus", "MCMC_plus"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    frame["MCMC_minus_raw"] = frame["MCMC_minus"]
    frame["MCMC_plus_raw"] = frame["MCMC_plus"]
    frame["MCMC_uncertainty_scale"] = scale
    frame["MCMC_minus"] = frame["MCMC_minus"] * scale
    frame["MCMC_plus"] = frame["MCMC_plus"] * scale
    return frame


def main() -> int:
    args = parse_args()
    info = inflation_scale(args.goodness, args.target_p_value)
    scale = info["scale"]

    fit_summary = inflate_table(args.fit_summary, scale)
    selected_yields = inflate_table(args.selected_yields, scale)
    for path, frame in (
        (args.output_fit_summary, fit_summary),
        (args.output_selected_yields, selected_yields),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    adjusted_p = float(chi2_dist.sf(info["adjusted_chi2"], int(info["ndf"])))
    report = "\n".join(
        [
            "method: post-fit common Pearson overdispersion scale",
            "pearson_bin_selection: expected counts >= 5",
            f"source_goodness_file: {args.goodness}",
            f"observed_chi2: {info['observed_chi2']:.9f}",
            f"ndf: {int(info['ndf'])}",
            f"original_p_value: {info['original_p_value']:.9g}",
            f"target_p_value: {info['target_p_value']:.9g}",
            f"target_chi2: {info['target_chi2']:.9f}",
            f"uncertainty_scale: {scale:.9f}",
            f"adjusted_chi2: {info['adjusted_chi2']:.9f}",
            f"adjusted_p_value: {adjusted_p:.9g}",
            "medians_changed: false",
            "interpretation: pragmatic post-fit overdispersion correction; not a rerun of the Poisson MCMC",
            "",
        ]
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(report, encoding="utf-8")

    print(report, end="")
    print(f"saved fit summary: {args.output_fit_summary}")
    print(f"saved selected yields: {args.output_selected_yields}")
    print(f"saved report: {args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
