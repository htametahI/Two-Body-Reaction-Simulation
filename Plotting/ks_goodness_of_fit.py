import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.stats import kstwo

from mcmc_fit_diagnostics import (
    BKG_COMPONENT,
    COMPONENTS,
    DATA,
    FIT_RANGE,
    FUSION_COMPONENT,
    INCLUDE_FUSION_PROTON_TEMPLATE,
    INCLUDE_SMOOTH_BKG,
    OUT,
    RING_SELECTION,
    fit_amplitudes,
    select_components_for_range,
)


SUMMARY_CSV = OUT / "MCMC_fit_summary.csv"
KS_SUMMARY = OUT / "ks_goodness_of_fit_summary.txt"
BOOTSTRAP_SAMPLES = 1000
RNG_SEED = 24680


def load_fit_data():
    npz = np.load(DATA, allow_pickle=True)
    df = pd.DataFrame({key: npz[key] for key in npz.files})
    if "source_file" in df:
        df["source_file"] = df["source_file"].astype(str)
    rr0 = df.query(RING_SELECTION)

    allowed_state_suffixes = ("_1minus", "_0plus", "_2plus")
    candidate_components = [
        c for c in COMPONENTS if c["name"].endswith(allowed_state_suffixes)
    ]
    if INCLUDE_FUSION_PROTON_TEMPLATE:
        candidate_components.append(FUSION_COMPONENT)
    if INCLUDE_SMOOTH_BKG:
        candidate_components.append(BKG_COMPONENT)
    components = select_components_for_range(
        candidate_components,
        FIT_RANGE,
        nsigma=6.0,
    )
    fit_result = fit_amplitudes(
        rr0["excitation_MeV"], components, fit_range=FIT_RANGE
    )
    return fit_result


def mixture_cdf_on_grid(yields, grid, pdfs_grid):
    weights = np.asarray(yields, dtype=float)
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Mixture yields must be non-negative with positive total")
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
    d = max(d_plus, d_minus)
    return float(d), float(d_plus), float(d_minus)


def sample_from_mixture(n_events, grid, cdf_grid, rng):
    u = rng.random(n_events)
    return np.interp(u, cdf_grid, grid)


def fit_ks_for_data(data, components):
    fit_result = fit_amplitudes(data, components, fit_range=FIT_RANGE)
    minuit = fit_result["minuit"]
    yields = np.array(
        [minuit.values[f"N{i}"] for i in range(len(components))], dtype=float
    )
    cdf_grid = mixture_cdf_on_grid(
        yields, fit_result["grid"], fit_result["pdfs_grid"]
    )
    d, d_plus, d_minus = ks_statistic(fit_result["x"], fit_result["grid"], cdf_grid)
    return d, d_plus, d_minus, yields, fit_result


def bootstrap_p_value(observed_d, n_events, components, grid, cdf_grid):
    rng = np.random.default_rng(RNG_SEED)
    bootstrap_d = np.empty(BOOTSTRAP_SAMPLES, dtype=float)

    for i in range(BOOTSTRAP_SAMPLES):
        sample = sample_from_mixture(n_events, grid, cdf_grid, rng)
        bootstrap_d[i], _, _, _, _ = fit_ks_for_data(sample, components)

    p_value = (np.count_nonzero(bootstrap_d >= observed_d) + 1) / (
        BOOTSTRAP_SAMPLES + 1
    )
    return float(p_value), bootstrap_d


def main():
    fit_result = load_fit_data()
    data = fit_result["x"]
    grid = fit_result["grid"]
    pdfs_grid = fit_result["pdfs_grid"]
    components = fit_result["components"]
    labels = [c["name"] for c in components]

    minuit = fit_result["minuit"]
    minuit_yields = np.array(
        [minuit.values[f"N{i}"] for i in range(len(components))], dtype=float
    )
    minuit_cdf = mixture_cdf_on_grid(minuit_yields, grid, pdfs_grid)
    minuit_d, minuit_d_plus, minuit_d_minus = ks_statistic(data, grid, minuit_cdf)
    minuit_p_naive = float(kstwo.sf(minuit_d, len(data)))

    summary = pd.read_csv(SUMMARY_CSV)
    mcmc_yields = summary["MCMC_N_median"].to_numpy(dtype=float)
    mcmc_cdf = mixture_cdf_on_grid(mcmc_yields, grid, pdfs_grid)
    mcmc_d, mcmc_d_plus, mcmc_d_minus = ks_statistic(data, grid, mcmc_cdf)
    mcmc_p_naive = float(kstwo.sf(mcmc_d, len(data)))

    bootstrap_p, bootstrap_d = bootstrap_p_value(
        minuit_d, len(data), components, grid, minuit_cdf
    )

    lines = [
        f"data_file: {DATA}",
        f"selection: {RING_SELECTION}",
        f"fit_range_MeV: {FIT_RANGE[0]} {FIT_RANGE[1]}",
        f"n_data_in_fit_range: {len(data)}",
        "active_components: " + ", ".join(labels),
        "",
        "KS test against fitted CDF using the plotted MCMC median yields:",
        f" ks_D: {mcmc_d:.6f}",
        f" ks_D_plus: {mcmc_d_plus:.6f}",
        f" ks_D_minus: {mcmc_d_minus:.6f}",
        f" ks_p_naive_fixed_cdf: {mcmc_p_naive:.6g}",
        "",
        "KS test against fitted CDF using Minuit maximum-likelihood yields:",
        f" ks_D: {minuit_d:.6f}",
        f" ks_D_plus: {minuit_d_plus:.6f}",
        f" ks_D_minus: {minuit_d_minus:.6f}",
        f" ks_p_naive_fixed_cdf: {minuit_p_naive:.6g}",
        "",
        "Parametric bootstrap for fitted-parameter KS p-value:",
        f" bootstrap_samples: {BOOTSTRAP_SAMPLES}",
        f" rng_seed: {RNG_SEED}",
        f" ks_p_bootstrap_refit_minuit: {bootstrap_p:.6g}",
        f" bootstrap_D_median: {np.median(bootstrap_d):.6f}",
        f" bootstrap_D_16: {np.percentile(bootstrap_d, 16):.6f}",
        f" bootstrap_D_84: {np.percentile(bootstrap_d, 84):.6f}",
    ]
    KS_SUMMARY.write_text("\n".join(lines) + "\n")

    print(f"wrote {KS_SUMMARY}")
    print(f"fit_range_MeV={FIT_RANGE[0]}-{FIT_RANGE[1]}")
    print(f"n_data={len(data)}")
    print(f"ks_mcmc_median={mcmc_d:.6f}; p_naive={mcmc_p_naive:.6g}")
    print(f"ks_minuit={minuit_d:.6f}; p_naive={minuit_p_naive:.6g}")
    print(
        "ks_bootstrap_refit_minuit="
        f"{bootstrap_p:.6g}; samples={BOOTSTRAP_SAMPLES}"
    )


if __name__ == "__main__":
    main()
