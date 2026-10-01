from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mcmc_fit_diagnostics import (
    COMPONENTS,
    DATA,
    OUT,
    RING_SELECTION,
    binned_chi2,
    fit_amplitudes,
    select_components_for_range,
)


BIN_WIDTH_MEV = 0.10
LOWER_GRID = np.arange(10.50, 11.01, 0.05)
UPPER_GRID = np.arange(11.70, 12.31, 0.05)
MIN_WIDTH_MEV = 1.00
MIN_NDF = 10
MIN_EVENTS = 250
MIN_EXPECTED_CONSERVATIVE = 5.0

SCAN_CSV = OUT / "fit_range_chi2_scan_100keV.csv"
SCAN_TXT = OUT / "fit_range_chi2_scan_100keV_summary.txt"
SCAN_PDF = OUT / "fit_range_chi2_scan_100keV.pdf"
SCAN_PNG = OUT / "fit_range_chi2_scan_100keV.png"


def bins_for_range(lo, hi):
    return np.arange(lo, hi + 0.5 * BIN_WIDTH_MEV, BIN_WIDTH_MEV)


def load_ring_data():
    npz = np.load(DATA, allow_pickle=True)
    df = pd.DataFrame({key: npz[key] for key in npz.files})
    return df.query(RING_SELECTION)


def active_components_for(fit_range):
    allowed_state_suffixes = ("_1minus", "_0plus", "_2plus")
    return select_components_for_range(
        [
            c
            for c in COMPONENTS
            if c["name"].endswith(allowed_state_suffixes)
        ],
        fit_range,
        nsigma=6.0,
    )


def scan_ranges():
    df = load_ring_data()
    rows = []

    for lo in LOWER_GRID:
        for hi in UPPER_GRID:
            lo = float(np.round(lo, 2))
            hi = float(np.round(hi, 2))
            if hi - lo < MIN_WIDTH_MEV:
                continue

            fit_range = (lo, hi)
            components = active_components_for(fit_range)
            if not components:
                continue

            try:
                fit_result = fit_amplitudes(
                    df["excitation_MeV"],
                    components,
                    fit_range=fit_range,
                )
            except Exception as exc:
                rows.append(
                    {
                        "fit_lo_MeV": lo,
                        "fit_hi_MeV": hi,
                        "fit_width_MeV": hi - lo,
                        "status": f"failed: {exc}",
                    }
                )
                continue

            x = fit_result["x"]
            if len(x) < MIN_EVENTS:
                continue

            minuit = fit_result["minuit"]
            yields = np.array(
                [minuit.values[f"N{i}"] for i in range(len(components))],
                dtype=float,
            )
            bins = bins_for_range(lo, hi)
            info = binned_chi2(
                yields,
                x,
                fit_result["grid"],
                fit_result["pdfs_grid"],
                len(components),
                bins,
            )
            expected_positive = info["expected"][info["expected"] > 0]
            if info["ndf"] < MIN_NDF:
                continue

            rows.append(
                {
                    "fit_lo_MeV": lo,
                    "fit_hi_MeV": hi,
                    "fit_width_MeV": hi - lo,
                    "n_events": len(x),
                    "n_components": len(components),
                    "components": ",".join(c["name"] for c in components),
                    "n_bins": len(bins) - 1,
                    "ndf": info["ndf"],
                    "chi2": info["chi2"],
                    "chi2_ndf": info["chi2_ndf"],
                    "p_value": info["p_value"],
                    "min_expected": float(np.min(expected_positive)),
                    "max_expected": float(np.max(info["expected"])),
                    "minuit_valid": bool(minuit.valid),
                    "status": "ok",
                }
            )

    return pd.DataFrame(rows)


def plot_scan(scan_df):
    ok = scan_df[scan_df["status"] == "ok"].copy()
    best = ok.loc[ok["chi2_ndf"].idxmin()]

    pivot = ok.pivot(index="fit_lo_MeV", columns="fit_hi_MeV", values="chi2_ndf")
    xvals = pivot.columns.to_numpy(dtype=float)
    yvals = pivot.index.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    mesh = ax.pcolormesh(
        xvals,
        yvals,
        pivot.to_numpy(),
        shading="nearest",
        cmap="viridis_r",
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$\chi^2/\nu$")
    ax.scatter(
        [best["fit_hi_MeV"]],
        [best["fit_lo_MeV"]],
        color="red",
        edgecolor="white",
        s=80,
        label=(
            rf"best: ${best['fit_lo_MeV']:.2f}$-"
            rf"${best['fit_hi_MeV']:.2f}\,\mathrm{{MeV}}$"
        ),
    )
    ax.set_xlabel(r"Upper fit limit $E_{\mathrm{hi}}\ \mathrm{[MeV]}$")
    ax.set_ylabel(r"Lower fit limit $E_{\mathrm{lo}}\ \mathrm{[MeV]}$")
    ax.set_title(r"Fit-range scan with $100\,\mathrm{keV}$ bins")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(SCAN_PDF)
    fig.savefig(SCAN_PNG)
    plt.close(fig)


def main():
    scan_df = scan_ranges()
    scan_df = scan_df.sort_values(["chi2_ndf", "chi2"], ascending=[True, True])
    scan_df.to_csv(SCAN_CSV, index=False)
    plot_scan(scan_df)

    ok = scan_df[scan_df["status"] == "ok"]
    conservative = ok[ok["min_expected"] >= MIN_EXPECTED_CONSERVATIVE]
    lines = [
        f"data_file: {DATA}",
        f"selection: {RING_SELECTION}",
        f"bin_width_MeV: {BIN_WIDTH_MEV}",
        f"lower_grid: {LOWER_GRID[0]:.2f} to {LOWER_GRID[-1]:.2f} MeV",
        f"upper_grid: {UPPER_GRID[0]:.2f} to {UPPER_GRID[-1]:.2f} MeV",
        f"min_width_MeV: {MIN_WIDTH_MEV}",
        f"min_ndf: {MIN_NDF}",
        f"min_events: {MIN_EVENTS}",
        "",
        "best_overall:",
        ok.head(1).to_string(index=False),
        "",
        f"best_with_min_expected_ge_{MIN_EXPECTED_CONSERVATIVE:g}:",
        conservative.head(1).to_string(index=False),
        "",
        "best_15_overall:",
        ok.head(15).to_string(index=False),
    ]
    SCAN_TXT.write_text("\n".join(lines) + "\n")

    print(f"wrote {SCAN_CSV}")
    print(f"wrote {SCAN_TXT}")
    print(f"wrote {SCAN_PDF}")
    print(f"wrote {SCAN_PNG}")
    print(ok.head(15).to_string(index=False))
    if not conservative.empty:
        print("\nbest conservative:")
        print(conservative.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
