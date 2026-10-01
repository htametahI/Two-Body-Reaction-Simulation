import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.stats import kstwo

from mcmc_fit_diagnostics import (
    BIN_WIDTH_CHI2_MEV,
    BKG_COMPONENT,
    COMPONENTS,
    DATA,
    FIT_RANGE,
    FUSION_COMPONENT,
    OUT,
    RING_SELECTION,
    binned_chi2,
    fit_amplitudes,
)


STATE_FIT_CSV = OUT / "Exc_by_ring_chunks" / "ring_chunk_sloped_box_fit_results.csv"
SUMMARY_TXT = OUT / "component_model_comparison_summary.txt"
SUMMARY_CSV = OUT / "component_model_comparison_summary.csv"


def load_ring_data():
    npz = np.load(DATA, allow_pickle=True)
    df = pd.DataFrame({key: npz[key] for key in npz.files})
    if "source_file" in df:
        df["source_file"] = df["source_file"].astype(str)
    return df.query(RING_SELECTION)


def component_from_row(row):
    return {
        "name": row["state"],
        "E_L": float(row["E_L_MeV"]),
        "E_R": float(row["E_R_MeV"]),
        "sigma_L": float(row["sigma_L_MeV"]),
        "sigma_R": float(row["sigma_R_MeV"]),
        "slope": float(row["slope_MeV_inv"]),
    }


def csv_components(states):
    fits = pd.read_csv(STATE_FIT_CSV)
    rows = fits[
        (fits["ring_group"] == "rings_01_04")
        & (fits["state"].isin(states))
        & (fits["minuit_valid"].astype(bool))
    ].copy()
    found = set(rows["state"])
    missing = sorted(set(states) - found)
    if missing:
        raise ValueError(f"Missing valid fitted templates for: {', '.join(missing)}")
    rows = rows.set_index("state").loc[states].reset_index()
    return [component_from_row(row) for _, row in rows.iterrows()]


def mixture_cdf_on_grid(yields, grid, pdfs_grid):
    weights = np.asarray(yields, dtype=float)
    weights = weights / np.sum(weights)
    pdf_grid = weights @ pdfs_grid
    cdf_grid = cumulative_trapezoid(pdf_grid, grid, initial=0.0)
    cdf_grid /= cdf_grid[-1]
    return cdf_grid


def ks_statistic(data, grid, cdf_grid):
    x = np.sort(np.asarray(data, dtype=float))
    n = len(x)
    fitted_cdf = np.interp(x, grid, cdf_grid, left=0.0, right=1.0)
    empirical_hi = np.arange(1, n + 1) / n
    empirical_lo = np.arange(0, n) / n
    d_plus = np.max(empirical_hi - fitted_cdf)
    d_minus = np.max(fitted_cdf - empirical_lo)
    return float(max(d_plus, d_minus))


def evaluate_model(name, components, data):
    fit_result = fit_amplitudes(data["excitation_MeV"], components, FIT_RANGE)
    minuit = fit_result["minuit"]
    yields = np.array(
        [minuit.values[f"N{i}"] for i in range(len(components))], dtype=float
    )
    bins = np.arange(
        FIT_RANGE[0],
        FIT_RANGE[1] + 0.5 * BIN_WIDTH_CHI2_MEV,
        BIN_WIDTH_CHI2_MEV,
    )
    chi2_info = binned_chi2(
        yields,
        fit_result["x"],
        fit_result["grid"],
        fit_result["pdfs_grid"],
        len(components),
        bins,
    )
    cdf_grid = mixture_cdf_on_grid(
        yields, fit_result["grid"], fit_result["pdfs_grid"]
    )
    ks_d = ks_statistic(fit_result["x"], fit_result["grid"], cdf_grid)
    return {
        "model": name,
        "n_events": len(fit_result["x"]),
        "n_components": len(components),
        "n_bins": len(bins) - 1,
        "ndf": chi2_info["ndf"],
        "chi2": chi2_info["chi2"],
        "chi2_ndf": chi2_info["chi2_ndf"],
        "chi2_p_value": chi2_info["p_value"],
        "ks_D": ks_d,
        "ks_p_naive": float(kstwo.sf(ks_d, len(fit_result["x"]))),
        "minuit_valid": bool(minuit.valid),
        "components": ",".join(c["name"] for c in components),
    }


def main():
    data = load_ring_data()

    def exp_bkg(tau):
        component = dict(BKG_COMPONENT)
        component["tau"] = tau
        component["name"] = f"unresolved_tail_bkg_tau_{tau:g}"
        component["origin"] = FIT_RANGE[0]
        return component

    def flat_bkg():
        return {
            "name": "unresolved_flat_bkg",
            "kind": "flat_pull",
            "max_fraction": BKG_COMPONENT["max_fraction"],
        }

    variants = {
        "selected_states_plus_subthreshold_states": COMPONENTS,
        "selected_states_plus_subthreshold_states_fusion_proton_template": COMPONENTS
        + [FUSION_COMPONENT],
        "selected_states_plus_subthreshold_states_flat_bkg": COMPONENTS
        + [flat_bkg()],
        "selected_states_plus_subthreshold_states_exp_bkg_tau_0p4": COMPONENTS
        + [exp_bkg(0.4)],
        "selected_states_plus_subthreshold_states_exp_bkg_tau_0p8": COMPONENTS
        + [exp_bkg(0.8)],
        "selected_states_plus_subthreshold_states_exp_bkg_tau_1p2": COMPONENTS
        + [exp_bkg(1.2)],
    }

    rows = [evaluate_model(name, components, data) for name, components in variants.items()]
    summary = pd.DataFrame(rows).sort_values("chi2_ndf")
    summary.to_csv(SUMMARY_CSV, index=False)

    lines = [
        f"data_file: {DATA}",
        f"selection: {RING_SELECTION}",
        f"fit_range_MeV: {FIT_RANGE[0]} {FIT_RANGE[1]}",
        f"bin_width_chi2_MeV: {BIN_WIDTH_CHI2_MEV}",
        "",
        summary.to_string(index=False),
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n")

    print(f"wrote {SUMMARY_CSV}")
    print(f"wrote {SUMMARY_TXT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
