from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mcmc_fit_diagnostics import (
    COMPONENTS,
    DATA,
    FIT_RANGE,
    OUT,
    RING_SELECTION,
    SUMMARY_CSV,
    binned_chi2,
    fit_amplitudes,
    select_components_for_range,
)


SCAN_CSV = OUT / "chi2_binning_scan.csv"
SCAN_TXT = OUT / "chi2_binning_scan_summary.txt"
SCAN_PDF = OUT / "chi2_binning_scan.pdf"
SCAN_PNG = OUT / "chi2_binning_scan.png"
BEST_PDF = OUT / "MCMC_fit_spectrum_best_binning.pdf"
BEST_PNG = OUT / "MCMC_fit_spectrum_best_binning.png"


def make_shifted_bins(lo, hi, width, offset_fraction):
    offset = offset_fraction * width
    internal = np.arange(lo + offset, hi, width)
    internal = internal[(internal > lo) & (internal < hi)]
    bins = np.concatenate(([lo], internal, [hi]))
    bins = np.unique(np.round(bins, 12))
    return bins


def plot_scan(scan_df):
    best = scan_df.loc[scan_df["chi2_ndf"].idxmin()]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plot_df = scan_df.sort_values("target_width_MeV")
    for offset_fraction, group in plot_df.groupby("offset_fraction"):
        group = group.sort_values("target_width_MeV")
        ax.plot(
            1000.0 * group["target_width_MeV"],
            group["chi2_ndf"],
            marker="o",
            ms=3,
            lw=1,
            alpha=0.75,
            label=f"offset={offset_fraction:.2f} bin",
        )
    ax.scatter(
        [1000.0 * best["target_width_MeV"]],
        [best["chi2_ndf"]],
        color="red",
        zorder=5,
        label=(
            "best: "
            f"{1000.0 * best['target_width_MeV']:.0f} keV, "
            f"offset={best['offset_fraction']:.2f}"
        ),
    )
    ax.set_xlabel("Target bin width [keV]")
    ax.set_ylabel(r"$\chi^2/\mathrm{ndf}$")
    ax.set_title("Post-fit Pearson chi2 Sensitivity to Binning")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(SCAN_PDF)
    fig.savefig(SCAN_PNG)
    plt.close(fig)


def plot_best_spectrum(x, fit_result, active_components, yields, best, yield_source):
    bins = make_shifted_bins(
        FIT_RANGE[0],
        FIT_RANGE[1],
        best["target_width_MeV"],
        best["offset_fraction"],
    )
    info = binned_chi2(
        yields,
        x,
        fit_result["grid"],
        fit_result["pdfs_grid"],
        len(active_components),
        bins,
    )
    observed = info["observed"]
    expected = info["expected"]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    effective_width = np.median(np.diff(bins))
    grid = fit_result["grid"]
    pdfs_grid = fit_result["pdfs_grid"]

    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    ax.stairs(
        observed,
        bins,
        baseline=0,
        fill=True,
        facecolor="0.80",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.55,
        label="Data",
    )

    total_fine = np.zeros_like(grid, dtype=float)
    for i, c in enumerate(active_components):
        comp_counts_fine = yields[i] * pdfs_grid[i] * effective_width
        total_fine += comp_counts_fine
        label = c["name"].replace("_", " ")
        ax.plot(grid, comp_counts_fine, lw=1.2, alpha=0.8, label=label)
    ax.plot(grid, total_fine, color="black", lw=2.0, label=f"{yield_source} sum")
    ax.set_ylabel(f"Counts / {effective_width:.3f} MeV")
    ax.set_title(
        rf"Best Scanned Binning, $\chi^2/\mathrm{{ndf}} = "
        rf"{info['chi2']:.1f}/{info['ndf']} = {info['chi2_ndf']:.2f}$"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    valid = expected > 0
    resid = np.zeros_like(observed, dtype=float)
    resid[valid] = (observed[valid] - expected[valid]) / np.sqrt(expected[valid])
    rax.axhline(0, color="black", lw=1)
    rax.stairs(resid, bins, color="black", linewidth=1.2)
    rax.set_xlabel(r"$E_x$ [MeV]")
    rax.set_ylabel(r"$(D-M)/\sqrt{M}$")
    rax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(BEST_PDF)
    fig.savefig(BEST_PNG)
    plt.close(fig)


def main():
    npz = np.load(DATA, allow_pickle=True)
    df = pd.DataFrame({key: npz[key] for key in npz.files})
    rr0 = df.query(RING_SELECTION)

    allowed_state_suffixes = ("_1minus", "_0plus", "_2plus")
    active_components = select_components_for_range(
        [
            c
            for c in COMPONENTS
            if c["name"].endswith(allowed_state_suffixes)
        ],
        FIT_RANGE,
        nsigma=6.0,
    )
    fit_result = fit_amplitudes(rr0["excitation_MeV"], active_components, FIT_RANGE)
    x = fit_result["x"]
    minuit = fit_result["minuit"]
    minuit_yields = np.array(
        [minuit.values[f"N{i}"] for i in range(len(active_components))],
        dtype=float,
    )
    labels = [c["name"] for c in active_components]
    yields = minuit_yields
    yield_source = "Minuit"
    if SUMMARY_CSV.exists():
        summary_df = pd.read_csv(SUMMARY_CSV)
        if list(summary_df["component"]) == labels:
            yields = summary_df["MCMC_N_median"].to_numpy(dtype=float)
            yield_source = "MCMC median"

    rows = []
    lo, hi = FIT_RANGE
    for width_keV in range(50, 351, 10):
        width = width_keV / 1000.0
        for offset_fraction in np.linspace(0.0, 0.9, 10):
            bins = make_shifted_bins(lo, hi, width, offset_fraction)
            if len(bins) < len(active_components) + 3:
                continue
            info = binned_chi2(
                yields,
                x,
                fit_result["grid"],
                fit_result["pdfs_grid"],
                len(active_components),
                bins,
            )
            min_expected = float(np.min(info["expected"][info["expected"] > 0]))
            rows.append(
                {
                    "target_width_MeV": width,
                    "target_width_keV": width_keV,
                    "offset_fraction": float(offset_fraction),
                    "n_bins": len(bins) - 1,
                    "ndf": info["ndf"],
                    "chi2": info["chi2"],
                    "chi2_ndf": info["chi2_ndf"],
                    "p_value": info["p_value"],
                    "min_expected": min_expected,
                    "max_expected": float(np.max(info["expected"])),
                }
            )

    scan_df = pd.DataFrame(rows)
    scan_df = scan_df.sort_values(["chi2_ndf", "chi2"], ascending=[True, True])
    scan_df.to_csv(SCAN_CSV, index=False)
    plot_scan(scan_df)
    plot_best_spectrum(x, fit_result, active_components, yields, scan_df.iloc[0], yield_source)

    conservative = scan_df[scan_df["min_expected"] >= 5.0]
    summary_lines = [
        f"fit_range_MeV: {FIT_RANGE[0]} {FIT_RANGE[1]}",
        f"selection: {RING_SELECTION}",
        "scan: target bin widths 50-350 keV in 10 keV steps; offsets 0.0-0.9 bin",
        f"yield_source: {yield_source}",
        "optimization target: minimum Pearson chi2/ndf for fixed fitted yields",
        "",
        "best_overall:",
        scan_df.head(1).to_string(index=False),
        "",
        "best_with_min_expected_ge_5:",
        conservative.head(1).to_string(index=False),
        "",
        "best_10_overall:",
        scan_df.head(10).to_string(index=False),
        "",
        "best_10_with_min_expected_ge_5:",
        conservative.head(10).to_string(index=False),
    ]
    SCAN_TXT.write_text("\n".join(summary_lines) + "\n")

    print(f"wrote {SCAN_CSV}")
    print(f"wrote {SCAN_TXT}")
    print(f"wrote {SCAN_PDF}")
    print(f"wrote {SCAN_PNG}")
    print(f"wrote {BEST_PDF}")
    print(f"wrote {BEST_PNG}")
    print(scan_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
