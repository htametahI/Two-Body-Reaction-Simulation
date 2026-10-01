from pathlib import Path
import re

import corner
import emcee
from iminuit import Minuit
import matplotlib as mpl

mpl.use("Agg")
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import erf
from scipy.stats import chi2 as chi2_dist


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
DATA = OUT / "unbinned_all_rings_combined_PGACL_repeated.npz"

FIT_RANGE = (10.1, 12.3)
RING_SELECTION = "0 <= ring <= 3"
BIN_WIDTH_CHI2_MEV = 0.10
INCLUDE_SMOOTH_BKG = False
BKG_COMPONENT = {
    "name": "unresolved_tail_bkg",
    "kind": "exp_tail",
    "tau": 0.4,
    "origin": FIT_RANGE[0],
    "max_fraction": 0.20,
}
INCLUDE_FUSION_PROTON_TEMPLATE = True
PACE4_PROTON_EX_CSV = Path(
    "/Users/mikeqiu/GEANT4/Simulations/output/protons_excitation_as_triton.csv"
)
FUSION_TEMPLATE_RANGE = (10.0, 12.5)
FUSION_COMPONENT = {
    "name": "fusion_proton_bkg",
    "kind": "sample_kde",
    "role": "background",
    "source": PACE4_PROTON_EX_CSV,
    "column": "Ex_energy_loss_corrected_MeV",
    "sample_range": FUSION_TEMPLATE_RANGE,
    "bandwidth": 0.10,
    "max_fraction": 0.20,
}

CORNER_PDF = OUT / "MCMC_parameter_constraints_correlations.pdf"
CORNER_PNG = OUT / "MCMC_parameter_constraints_correlations.png"
SPECTRUM_PDF = OUT / "MCMC_fit_spectrum_with_residuals.pdf"
SPECTRUM_PNG = OUT / "MCMC_fit_spectrum_with_residuals.png"
POSTERIOR_DRAWS_PDF = OUT / "MCMC_posterior_draws_with_median.pdf"
POSTERIOR_DRAWS_PNG = OUT / "MCMC_posterior_draws_with_median.png"
FUSION_TEMPLATE_PDF = OUT / "fusion_proton_template_mockup.pdf"
FUSION_TEMPLATE_PNG = OUT / "fusion_proton_template_mockup.png"
SUMMARY_TXT = OUT / "MCMC_fit_summary.txt"
SUMMARY_CSV = OUT / "MCMC_fit_summary.csv"


mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
    }
)


COMPONENTS = [
    {
        "name": "10159_0plus",
        "role": "background",
        "E_L": 9.95088973815644,
        "E_R": 10.079665899444144,
        "sigma_L": 0.08498713437279559,
        "sigma_R": 0.08075024010985754,
        "slope": -0.4305412760684126,
    },
    {
        "name": "10573_0plus",
        "E_L": 10.1539,
        "E_R": 11.0130,
        "sigma_L": 0.0891,
        "sigma_R": 0.0981,
        "slope": -0.2472,
    },
    {
        "name": "10824_2plus",
        "E_L": 10.3924,
        "E_R": 11.2878,
        "sigma_L": 0.0843,
        "sigma_R": 0.0975,
        "slope": -0.3139,
    },
    {
        "name": "11183_1minus",
        "E_L": 10.7313,
        "E_R": 11.6623,
        "sigma_L": 0.0824,
        "sigma_R": 0.1015,
        "slope": -0.2374,
    },
    {
        "name": "11328_1minus",
        "E_L": 10.8652,
        "E_R": 11.8197,
        "sigma_L": 0.0792,
        "sigma_R": 0.1084,
        "slope": -0.2552,
    },
    {
        "name": "11830_1minus",
        "E_L": 11.37,
        "E_R": 12.32,
        "sigma_L": 0.09,
        "sigma_R": 0.10,
        "slope": -0.3504,
    },
    {
        "name": "12345_0plus",
        "E_L": 11.8233,
        "E_R": 12.9256,
        "sigma_L": 0.0893,
        "sigma_R": 0.1046,
        "slope": -0.2991,
    },
]

_SAMPLE_VALUES_CACHE = {}
_SAMPLE_DENSITY_CACHE = {}


def model(E, EL, ER, sigmaL, sigmaR, k):
    E = np.asarray(E, dtype=float)
    Ec = (EL + ER) / 2
    left = 0.5 * (1.0 + erf((E - EL) / (np.sqrt(2.0) * sigmaL)))
    right = 0.5 * (1.0 - erf((E - ER) / (np.sqrt(2.0) * sigmaR)))
    line = 1.0 + k * (E - Ec)
    return line * left * right


def component_shape(E, component):
    kind = component.get("kind", "sloped_box")
    if kind == "sloped_box":
        return model(
            E,
            component["E_L"],
            component["E_R"],
            component["sigma_L"],
            component["sigma_R"],
            component["slope"],
        )
    if kind == "flat_pull":
        return np.ones_like(np.asarray(E, dtype=float), dtype=float)
    if kind == "bounded_flat":
        values = np.asarray(E, dtype=float)
        lo = float(component["E_L"])
        hi = float(component["E_R"])
        if hi <= lo:
            raise ValueError("bounded_flat component requires E_R > E_L")
        return ((values >= lo) & (values <= hi)).astype(float)
    if kind == "exp_tail":
        values = np.asarray(E, dtype=float)
        tau = float(component["tau"])
        if tau <= 0:
            raise ValueError("exp_tail component requires tau > 0")
        origin = float(component.get("origin", np.min(values)))
        return np.exp(-(values - origin) / tau)
    if kind == "sample_kde":
        support, density, _ = sample_template_support(component)
        energy_shift = float(component.get("energy_shift_MeV", 0.0))
        return np.interp(
            np.asarray(E, dtype=float) - energy_shift,
            support,
            density,
            left=0.0,
            right=0.0,
        )
    raise ValueError(f"Unknown component kind {kind!r}")


def load_sample_template_values(component):
    source = Path(component["source"]).expanduser()
    column = component["column"]
    sample_lo, sample_hi = component.get("sample_range", FIT_RANGE)
    sample_lo = float(sample_lo)
    sample_hi = float(sample_hi)
    key = (str(source), column, sample_lo, sample_hi)
    if key not in _SAMPLE_VALUES_CACHE:
        values = pd.read_csv(source, usecols=[column])[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        values = values[(values >= sample_lo) & (values <= sample_hi)]
        if len(values) == 0:
            raise ValueError(
                f"{component['name']} has no finite {column} values in "
                f"{sample_lo:g}-{sample_hi:g} MeV from {source}"
            )
        _SAMPLE_VALUES_CACHE[key] = values
    return _SAMPLE_VALUES_CACHE[key]


def sample_template_support(component):
    values = load_sample_template_values(component)
    sample_lo, sample_hi = component.get("sample_range", FIT_RANGE)
    sample_lo = float(sample_lo)
    sample_hi = float(sample_hi)
    bandwidth = float(component["bandwidth"])
    if bandwidth <= 0:
        raise ValueError("sample_kde component requires bandwidth > 0")

    key = (
        str(Path(component["source"]).expanduser()),
        component["column"],
        sample_lo,
        sample_hi,
        bandwidth,
        int(component.get("kde_grid_size", 4001)),
    )
    if key in _SAMPLE_DENSITY_CACHE:
        return _SAMPLE_DENSITY_CACHE[key]

    support = np.linspace(sample_lo, sample_hi, key[-1])
    density = np.zeros_like(support)
    norm = len(values) * bandwidth * np.sqrt(2.0 * np.pi)
    for start in range(0, len(values), 5000):
        chunk = values[start : start + 5000]
        z = (support[:, None] - chunk[None, :]) / bandwidth
        density += np.exp(-0.5 * z * z).sum(axis=1)
    density /= norm
    _SAMPLE_DENSITY_CACHE[key] = (support, density, len(values))
    return _SAMPLE_DENSITY_CACHE[key]


def select_components_for_range(components, fit_range, nsigma=6.0):
    lo, hi = fit_range
    selected = []
    for c in components:
        if c.get("kind", "sloped_box") != "sloped_box":
            selected.append(c)
            continue
        support_lo = c["E_L"] - nsigma * c["sigma_L"]
        support_hi = c["E_R"] + nsigma * c["sigma_R"]
        if support_hi >= lo and support_lo <= hi:
            selected.append(c)
    return selected


def component_pdf(x, grid, components):
    pdfs = []
    pdfs_grid = []
    for c in components:
        shape = component_shape(x, c)
        shape_grid = component_shape(grid, c)
        norm = trapezoid(shape_grid, grid)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"{c['name']} has invalid normalization {norm}")
        pdfs.append(shape / norm)
        pdfs_grid.append(shape_grid / norm)
    return np.asarray(pdfs), np.asarray(pdfs_grid)


def fit_amplitudes(ex_values, components, fit_range, gridsize=8000):
    lo, hi = fit_range
    x = np.asarray(ex_values, dtype=float)
    x = x[np.isfinite(x)]
    x = x[(x >= lo) & (x <= hi)]
    grid = np.linspace(lo, hi, gridsize)
    pdfs_x, pdfs_grid = component_pdf(x, grid, components)
    n_comp = len(components)
    par_names = [f"N{i}" for i in range(n_comp)]
    upper_bounds = np.full(n_comp, np.inf, dtype=float)
    for i, component in enumerate(components):
        if "max_fraction" in component:
            upper_bounds[i] = float(component["max_fraction"]) * len(x)

    def nll_array(N):
        N = np.asarray(N)
        if np.any(N < 0):
            return 1e100
        if np.any(N > upper_bounds):
            return 1e100
        intensity_x = np.sum(N[:, None] * pdfs_x, axis=0)
        expected_events = np.sum(N)
        if not np.all(np.isfinite(intensity_x)) or np.any(intensity_x <= 0):
            return 1e100
        return expected_events - np.sum(np.log(intensity_x))

    def nll(*N):
        return nll_array(N)

    start = np.full(n_comp, len(x) / n_comp, dtype=float)
    m = Minuit(nll, *start, name=par_names)
    m.errordef = Minuit.LIKELIHOOD
    for i, name in enumerate(par_names):
        upper = None if np.isinf(upper_bounds[i]) else upper_bounds[i]
        m.limits[name] = (0, upper)
    m.migrad()
    m.hesse()
    return {
        "minuit": m,
        "x": x,
        "grid": grid,
        "pdfs_x": pdfs_x,
        "pdfs_grid": pdfs_grid,
        "components": components,
        "fit_range": fit_range,
        "upper_bounds": upper_bounds,
    }


def log_probability(logN, pdfs_x, n_data, upper_bounds=None):
    if not np.all(np.isfinite(logN)):
        return -np.inf
    N = np.exp(logN)
    if np.any(N <= 0) or np.any(N > 20 * n_data):
        return -np.inf
    if upper_bounds is not None and np.any(N > upper_bounds):
        return -np.inf
    intensity = N @ pdfs_x
    if np.any(intensity <= 0) or not np.all(np.isfinite(intensity)):
        return -np.inf
    log_like = -np.sum(N) + np.sum(np.log(intensity))
    log_prior = np.sum(logN)
    return log_like + log_prior


def latex_spin_parity(token):
    match = re.fullmatch(r"(?P<J>\d+)(?P<parity>plus|minus)", token)
    if not match:
        return token.replace("plus", "+").replace("minus", "-")
    sign = "+" if match.group("parity") == "plus" else "-"
    return rf"{match.group('J')}^{{{sign}}}"


def latex_component_label(name, yield_value=None, yield_minus=None, yield_plus=None):
    if name == "fusion_proton_bkg":
        label = r"$N_{p,\mathrm{fusion}}$"
        if yield_value is not None:
            if yield_minus is not None and yield_plus is not None:
                label += (
                    "\n"
                    + rf"$N = {yield_value:.0f}^{{+{yield_plus:.0f}}}_{{-{yield_minus:.0f}}}$"
                )
            else:
                label += "\n" + rf"$N = {yield_value:.0f}$"
        return label

    if name == "unresolved_tail_bkg":
        label = r"$N_{\mathrm{bkg}}$"
        if yield_value is not None:
            if yield_minus is not None and yield_plus is not None:
                label += (
                    "\n"
                    + rf"$N = {yield_value:.0f}^{{+{yield_plus:.0f}}}_{{-{yield_minus:.0f}}}$"
                )
            else:
                label += "\n" + rf"$N = {yield_value:.0f}$"
        return label

    match = re.fullmatch(r"(?P<energy>\d+)(?:_(?P<jp>.+))?", name)
    if not match:
        return name.replace("_", "\n")
    energy_mev = int(match.group("energy")) / 1000.0
    jp = match.group("jp")
    if jp:
        label = rf"${energy_mev:.2f}\,\mathrm{{MeV}},\ J^\pi = {latex_spin_parity(jp)}$"
    else:
        label = rf"${energy_mev:.2f}\,\mathrm{{MeV}}$"
    if yield_value is not None:
        if yield_minus is not None and yield_plus is not None:
            label += "\n" + rf"$N = {yield_value:.0f}^{{+{yield_plus:.0f}}}_{{-{yield_minus:.0f}}}$"
        else:
            label += "\n" + rf"$N = {yield_value:.0f}$"
    return label


def padded_corner_ranges(samples, truths=None, pad_fraction=0.35, min_pad_fraction=0.08):
    ranges = []
    for i in range(samples.shape[1]):
        values = samples[:, i]
        values = values[np.isfinite(values)]
        lo = np.min(values)
        hi = np.max(values)
        if truths is not None and np.isfinite(truths[i]):
            lo = min(lo, truths[i])
            hi = max(hi, truths[i])
        width = hi - lo
        if width <= 0:
            width = max(abs(hi), 1.0)
        scale = max(abs(hi), abs(lo), 1.0)
        pad = max(pad_fraction * width, min_pad_fraction * scale)
        ranges.append((lo - pad, hi + pad))
    return ranges


def bin_probabilities(grid, pdfs_grid, bins):
    probs = []
    for pdf_grid in pdfs_grid:
        cdf = cumulative_trapezoid(pdf_grid, grid, initial=0.0)
        cdf /= cdf[-1]
        probs.append(np.diff(np.interp(bins, grid, cdf)))
    return np.asarray(probs)


def binned_chi2(yields, data, grid, pdfs_grid, n_parameters, bins):
    observed, _ = np.histogram(data, bins=bins)
    bin_probs = bin_probabilities(grid, pdfs_grid, bins)
    expected = yields @ bin_probs
    valid = expected > 0
    chi2 = float(np.sum((observed[valid] - expected[valid]) ** 2 / expected[valid]))
    ndf = int(np.count_nonzero(valid) - n_parameters)
    return {
        "observed": observed,
        "expected": expected,
        "chi2": chi2,
        "ndf": ndf,
        "chi2_ndf": chi2 / ndf,
        "p_value": float(chi2_dist.sf(chi2, ndf)),
        "bin_probs": bin_probs,
    }


def make_corner_plot(samples_N, labels, minuit_N, q50, yield_minus, yield_plus):
    n_comp = len(labels)
    corner_labels = [latex_component_label(label) for label in labels]
    corner_titles = [
        rf"$N = {q50[i]:.0f}^{{+{yield_plus[i]:.0f}}}_{{-{yield_minus[i]:.0f}}}$"
        for i in range(n_comp)
    ]
    corner_ranges = padded_corner_ranges(samples_N, truths=minuit_N)

    fig = plt.figure(figsize=(max(9, 3.2 * n_comp), max(9, 3.2 * n_comp)))
    fig = corner.corner(
        samples_N,
        labels=corner_labels,
        truths=minuit_N,
        bins=40,
        range=corner_ranges,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=False,
        use_math_text=True,
        fig=fig,
        label_kwargs={"fontsize": 20},
    )
    axes = np.array(fig.get_axes()).reshape((n_comp, n_comp))
    for i in range(n_comp):
        axes[i, i].set_title(corner_titles[i], fontsize=20)
    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=16)
    fig.subplots_adjust(
        left=0.17, bottom=0.17, right=0.98, top=0.96, wspace=0.08, hspace=0.08
    )
    fig.savefig(CORNER_PDF)
    fig.savefig(CORNER_PNG)
    plt.close(fig)


def make_spectrum_plot(
    x, bins, active_components, q50, chi2_info, grid, pdfs_grid, bin_probs
):
    observed = chi2_info["observed"]
    expected = chi2_info["expected"]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]

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
        label=r"$N_{\mathrm{data}}$",
    )

    total_fine = np.zeros_like(grid, dtype=float)
    for i, c in enumerate(active_components):
        comp_counts_fine = q50[i] * pdfs_grid[i] * bin_width
        total_fine += comp_counts_fine
        ax.plot(
            grid,
            comp_counts_fine,
            lw=1.2,
            alpha=0.8,
            label=latex_component_label(c["name"]),
        )
    ax.plot(
        grid,
        total_fine,
        color="black",
        lw=2.0,
        label=r"$N_{\mathrm{fit}}$ (MCMC median)",
    )
    ax.set_ylabel(rf"Counts / ${1000.0 * bin_width:.0f}\,\mathrm{{keV}}$")
    ax.set_title(
        rf"MCMC Fit, $\chi^2/\mathrm{{dof}} = {chi2_info['chi2_ndf']:.2f}$"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    valid = expected > 0
    resid = np.zeros_like(observed, dtype=float)
    resid[valid] = (observed[valid] - expected[valid]) / np.sqrt(expected[valid])
    rax.axhline(0, color="black", lw=1)
    rax.stairs(resid, bins, color="black", linewidth=1.2)
    rax.set_xlabel(r"$E_x\ \mathrm{[MeV]}$")
    rax.set_ylabel("Residual")
    rax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SPECTRUM_PDF)
    fig.savefig(SPECTRUM_PNG)
    plt.close(fig)


def make_posterior_draws_plot(x, bins, samples_N, q50, chi2_info, grid, pdfs_grid):
    observed, _ = np.histogram(x, bins=bins)
    bin_width = bins[1] - bins[0]
    plot_grid = np.linspace(bins[0], bins[-1], 700)
    pdfs_plot = np.vstack(
        [np.interp(plot_grid, grid, pdf_grid) for pdf_grid in pdfs_grid]
    )
    median_counts = q50 @ pdfs_plot * bin_width

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.stairs(
        observed,
        bins,
        baseline=0,
        fill=True,
        facecolor="0.84",
        edgecolor="black",
        linewidth=1.1,
        alpha=0.45,
        label=r"$N_{\mathrm{data}}$",
    )

    x_segments = np.broadcast_to(plot_grid, (min(500, len(samples_N)), len(plot_grid)))
    first = True
    for start in range(0, len(samples_N), 500):
        sample_chunk = samples_N[start : start + 500]
        counts = sample_chunk @ pdfs_plot * bin_width
        if len(sample_chunk) != len(x_segments):
            x_segments = np.broadcast_to(plot_grid, counts.shape)
        segments = np.stack((x_segments, counts), axis=-1)
        collection = LineCollection(
            segments,
            colors=(0.12, 0.32, 0.70, 0.025),
            linewidths=0.35,
            rasterized=True,
            label=(
                rf"{len(samples_N)} retained MCMC draws"
                if first
                else "_nolegend_"
            ),
        )
        ax.add_collection(collection)
        first = False

    ax.plot(
        plot_grid,
        median_counts,
        color="black",
        lw=2.4,
        label=r"posterior median",
    )
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$E_x\ \mathrm{[MeV]}$")
    ax.set_ylabel(rf"Counts / ${1000.0 * bin_width:.0f}\,\mathrm{{keV}}$")
    ax.set_title(
        rf"MCMC Draws, median $\chi^2/\mathrm{{dof}} = {chi2_info['chi2_ndf']:.2f}$"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(POSTERIOR_DRAWS_PDF)
    fig.savefig(POSTERIOR_DRAWS_PNG)
    plt.close(fig)


def make_fusion_template_plot(component):
    values = load_sample_template_values(component)
    support, density, _ = sample_template_support(component)
    sample_lo, sample_hi = component["sample_range"]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bins = np.arange(sample_lo, sample_hi + BIN_WIDTH_CHI2_MEV, BIN_WIDTH_CHI2_MEV)
    ax.hist(
        values,
        bins=bins,
        density=True,
        histtype="stepfilled",
        facecolor="0.80",
        edgecolor="black",
        alpha=0.55,
        label=r"PACE4 $p$ sample",
    )
    ax.plot(
        support,
        density,
        color="tab:red",
        lw=2.0,
        label=rf"KDE, $\sigma = {component['bandwidth']:.2f}\,\mathrm{{MeV}}$",
    )
    ax.axvspan(
        FIT_RANGE[0],
        FIT_RANGE[1],
        color="tab:blue",
        alpha=0.10,
        label=r"fit range",
    )
    ax.set_xlabel(r"$E_x\ \mathrm{[MeV]}$")
    ax.set_ylabel(r"Normalized density")
    ax.set_title(r"PACE4 Proton-Fusion Template")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FUSION_TEMPLATE_PDF)
    fig.savefig(FUSION_TEMPLATE_PNG)
    plt.close(fig)


def main():
    npz = np.load(DATA, allow_pickle=True)
    df = pd.DataFrame({key: npz[key] for key in npz.files})
    df["source_file"] = df["source_file"].astype(str)
    rr0 = df.query(RING_SELECTION)

    allowed_state_suffixes = ("_1minus", "_0plus", "_2plus")
    spin_filtered_components = [
        c for c in COMPONENTS if c["name"].endswith(allowed_state_suffixes)
    ]
    candidate_components = spin_filtered_components.copy()
    if INCLUDE_FUSION_PROTON_TEMPLATE:
        candidate_components.append(FUSION_COMPONENT)
    if INCLUDE_SMOOTH_BKG:
        candidate_components.append(BKG_COMPONENT)
    active_components = select_components_for_range(
        candidate_components, FIT_RANGE, nsigma=6.0
    )

    fit_result = fit_amplitudes(
        rr0["excitation_MeV"], active_components, fit_range=FIT_RANGE
    )
    m = fit_result["minuit"]
    x = fit_result["x"]
    grid = fit_result["grid"]
    pdfs_x = fit_result["pdfs_x"]
    pdfs_grid = fit_result["pdfs_grid"]
    labels = [c["name"] for c in active_components]
    n_comp = len(labels)
    minuit_N = np.array([m.values[f"N{i}"] for i in range(n_comp)], dtype=float)
    upper_bounds = fit_result["upper_bounds"]

    nwalkers = max(32, 4 * n_comp)
    rng = np.random.default_rng(12345)
    start_N = np.clip(minuit_N, 1e-6, None)
    finite_upper = np.isfinite(upper_bounds)
    start_N[finite_upper] = np.minimum(
        start_N[finite_upper], 0.98 * upper_bounds[finite_upper]
    )
    start_logN = np.log(start_N)
    p0 = start_logN + 1e-3 * rng.normal(size=(nwalkers, n_comp))
    for i, upper in enumerate(upper_bounds):
        if np.isfinite(upper):
            p0[:, i] = np.minimum(p0[:, i], np.log(0.98 * upper))

    np.random.seed(12345)
    sampler = emcee.EnsembleSampler(
        nwalkers, n_comp, log_probability, args=(pdfs_x, len(x), upper_bounds)
    )
    state = sampler.run_mcmc(p0, 1000, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, 5000, progress=False)

    try:
        tau = sampler.get_autocorr_time()
        thin = max(1, int(0.5 * np.max(tau)))
        tau_text = ", ".join(f"{t:.2f}" for t in tau)
    except emcee.autocorr.AutocorrError:
        thin = 10
        tau_text = "not stable; thin=10 used"

    samples_logN = sampler.get_chain(flat=True, thin=thin)
    samples_N = np.exp(samples_logN)
    q16, q50, q84 = np.percentile(samples_N, [16, 50, 84], axis=0)
    yield_minus = q50 - q16
    yield_plus = q84 - q50

    make_corner_plot(samples_N, labels, minuit_N, q50, yield_minus, yield_plus)

    bins = np.arange(
        FIT_RANGE[0],
        FIT_RANGE[1] + 0.5 * BIN_WIDTH_CHI2_MEV,
        BIN_WIDTH_CHI2_MEV,
    )
    chi2_mcmc = binned_chi2(q50, x, grid, pdfs_grid, n_comp, bins)
    chi2_minuit = binned_chi2(minuit_N, x, grid, pdfs_grid, n_comp, bins)
    make_spectrum_plot(
        x,
        bins,
        active_components,
        q50,
        chi2_mcmc,
        grid,
        pdfs_grid,
        chi2_mcmc["bin_probs"],
    )
    make_posterior_draws_plot(
        x, bins, samples_N, q50, chi2_mcmc, grid, pdfs_grid
    )
    if INCLUDE_FUSION_PROTON_TEMPLATE:
        make_fusion_template_plot(FUSION_COMPONENT)

    summary_df = pd.DataFrame(
        {
            "component": labels,
            "Minuit_N": minuit_N,
            "MCMC_N_median": q50,
            "MCMC_minus": yield_minus,
            "MCMC_plus": yield_plus,
        }
    )
    summary_df.to_csv(SUMMARY_CSV, index=False)
    non_state_bkg_indices = [
        i
        for i, component in enumerate(active_components)
        if component.get("kind", "sloped_box") != "sloped_box"
    ]
    bkg_indices = [
        i
        for i, component in enumerate(active_components)
        if component.get("role") == "background"
        or component.get("kind", "sloped_box") != "sloped_box"
    ]
    non_state_bkg_fraction_mcmc = (
        float(np.sum(q50[non_state_bkg_indices]) / len(x))
        if non_state_bkg_indices
        else 0.0
    )
    non_state_bkg_fraction_minuit = (
        float(np.sum(minuit_N[non_state_bkg_indices]) / len(x))
        if non_state_bkg_indices
        else 0.0
    )
    bkg_fraction_mcmc = (
        float(np.sum(q50[bkg_indices]) / len(x)) if bkg_indices else 0.0
    )
    bkg_fraction_minuit = (
        float(np.sum(minuit_N[bkg_indices]) / len(x)) if bkg_indices else 0.0
    )
    fusion_template_count = (
        len(load_sample_template_values(FUSION_COMPONENT))
        if INCLUDE_FUSION_PROTON_TEMPLATE
        else 0
    )

    lines = [
        f"data_file: {DATA}",
        f"selection: {RING_SELECTION}",
        f"fit_range_MeV: {FIT_RANGE[0]} {FIT_RANGE[1]}",
        f"n_data_in_fit_range: {len(x)}",
        "active_components: " + ", ".join(labels),
        (
            "smooth_background_component: "
            + (
                f"{BKG_COMPONENT['name']}, kind={BKG_COMPONENT['kind']}, "
                f"tau_MeV={BKG_COMPONENT['tau']}, "
                f"max_fraction={BKG_COMPONENT['max_fraction']}"
                if INCLUDE_SMOOTH_BKG
                else "disabled"
            )
        ),
        (
            "fusion_proton_template: "
            + (
                f"{FUSION_COMPONENT['name']}, "
                f"source={FUSION_COMPONENT['source']}, "
                f"column={FUSION_COMPONENT['column']}, "
                "sample_range_MeV="
                f"{FUSION_COMPONENT['sample_range'][0]} "
                f"{FUSION_COMPONENT['sample_range'][1]}, "
                f"bandwidth_MeV={FUSION_COMPONENT['bandwidth']}, "
                f"max_fraction={FUSION_COMPONENT['max_fraction']}, "
                f"n_source_events={fusion_template_count}"
                if INCLUDE_FUSION_PROTON_TEMPLATE
                else "disabled"
            )
        ),
        f"nwalkers: {nwalkers}",
        "burnin_steps: 1000",
        "production_steps: 5000",
        f"thin: {thin}",
        f"posterior_draws_used_for_median: {len(samples_N)}",
        f"acceptance_fraction_mean: {np.mean(sampler.acceptance_fraction):.6f}",
        f"autocorr_time: {tau_text}",
        f"bin_width_chi2_MeV: {BIN_WIDTH_CHI2_MEV}",
        f"bins_for_chi2: {len(bins) - 1}",
        f"nonzero_model_bins: {np.count_nonzero(chi2_mcmc['expected'] > 0)}",
        f"n_parameters: {n_comp}",
        f"non_state_background_fraction_posterior_median: {non_state_bkg_fraction_mcmc:.6f}",
        f"non_state_background_fraction_minuit: {non_state_bkg_fraction_minuit:.6f}",
        f"total_background_fraction_posterior_median: {bkg_fraction_mcmc:.6f}",
        f"total_background_fraction_minuit: {bkg_fraction_minuit:.6f}",
        f"chi2_posterior_median: {chi2_mcmc['chi2']:.6f}",
        f"ndf_posterior_median: {chi2_mcmc['ndf']}",
        f"chi2_ndf_posterior_median: {chi2_mcmc['chi2_ndf']:.6f}",
        f"chi2_p_value_posterior_median: {chi2_mcmc['p_value']:.6g}",
        f"chi2_minuit: {chi2_minuit['chi2']:.6f}",
        f"ndf_minuit: {chi2_minuit['ndf']}",
        f"chi2_ndf_minuit: {chi2_minuit['chi2_ndf']:.6f}",
        f"chi2_p_value_minuit: {chi2_minuit['p_value']:.6g}",
        "",
        summary_df.to_string(index=False),
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n")

    print(f"wrote {CORNER_PDF}")
    print(f"wrote {CORNER_PNG}")
    print(f"wrote {SPECTRUM_PDF}")
    print(f"wrote {POSTERIOR_DRAWS_PDF}")
    print(f"wrote {POSTERIOR_DRAWS_PNG}")
    if INCLUDE_FUSION_PROTON_TEMPLATE:
        print(f"wrote {FUSION_TEMPLATE_PDF}")
        print(f"wrote {FUSION_TEMPLATE_PNG}")
    print(f"wrote {SUMMARY_TXT}")
    print(f"wrote {SUMMARY_CSV}")
    print(f"n_data={len(x)}")
    print("components=" + ",".join(labels))
    print(
        "chi2_ndf_posterior_median="
        f"{chi2_mcmc['chi2']:.6f}/{chi2_mcmc['ndf']}="
        f"{chi2_mcmc['chi2_ndf']:.6f}; p={chi2_mcmc['p_value']:.6g}"
    )
    print(
        "chi2_ndf_minuit="
        f"{chi2_minuit['chi2']:.6f}/{chi2_minuit['ndf']}="
        f"{chi2_minuit['chi2_ndf']:.6f}; p={chi2_minuit['p_value']:.6g}"
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
