#!/usr/bin/env python3
"""Plot every TH1 histogram in output/inputHist.root."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-s2223")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib-s2223-xdg")

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "output" / "inputHist.root"
DEFAULT_OUTDIR = ROOT / "output" / "inputHist_histogram_plots"
HISTOGRAM_COLOR = "#41049D"


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
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all TH1 histograms stored in inputHist.root.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--x-min-MeV", type=float, default=None)
    parser.add_argument("--x-max-MeV", type=float, default=None)
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Also write log-y PNGs and a log-y overview beside the default linear plots.",
    )
    return parser.parse_args()


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("_")
    return cleaned or "histogram"


def histogram_label(name: str) -> str:
    if name == "hExcE_PGACL_S3Rec1":
        return "Excitation Energy, All Rings"
    match = re.fullmatch(r"hExcE_(RR\d)_PGACL", name)
    if match:
        return f"Inclusive {match.group(1)}"
    match = re.fullmatch(r"hExcE_(\d+)_PGACL_Recoil", name)
    if match:
        energy_label = "1809" if match.group(1) == "1808" else match.group(1)
        return rf"$E_\gamma={energy_label}$ keV, All Rings"
    match = re.fullmatch(r"hExcE_(RR\d)_(\d+)_PGACL", name)
    if match:
        energy_label = "1809" if match.group(2) == "1808" else match.group(2)
        return rf"$E_\gamma={energy_label}$ keV, {match.group(1)}"
    return name


def load_histograms(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with uproot.open(path) as root_file:
        for key in root_file.keys():
            obj = root_file[key]
            if not hasattr(obj, "to_numpy") or not str(obj.classname).startswith("TH1"):
                continue
            counts, edges = obj.to_numpy()
            counts = np.asarray(counts, dtype=float)
            edges = np.asarray(edges, dtype=float)
            name = key.split(";", 1)[0]
            rows.append(
                {
                    "name": name,
                    "counts": counts,
                    "edges": edges,
                    "entries": float(np.sum(counts)),
                    "nonzero_bins": int(np.count_nonzero(counts)),
                    "x_min": float(edges[0]),
                    "x_max": float(edges[-1]),
                    "bin_width": float(np.median(np.diff(edges))),
                }
            )
    return rows


def select_range(
    counts: np.ndarray,
    edges: np.ndarray,
    x_min: float | None,
    x_max: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    lo = edges[0] if x_min is None else x_min
    hi = edges[-1] if x_max is None else x_max
    mask = (edges[:-1] >= lo) & (edges[1:] <= hi)
    if not np.any(mask):
        return counts, edges
    idx = np.where(mask)[0]
    return counts[idx], edges[idx[0] : idx[-1] + 2]


def plot_one(
    item: dict[str, object],
    path: Path,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    log_y: bool = False,
) -> None:
    counts, edges = select_range(
        np.asarray(item["counts"], dtype=float),
        np.asarray(item["edges"], dtype=float),
        x_min,
        x_max,
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.stairs(counts, edges, fill=True, baseline=0.0, color=HISTOGRAM_COLOR, alpha=0.42)
    ax.stairs(counts, edges, color=HISTOGRAM_COLOR, linewidth=1.0)
    ax.set_xlim(edges[0], edges[-1])
    if log_y:
        positive = counts[counts > 0.0]
        ax.set_yscale("log")
        ax.set_ylim(0.8, max(2.0, float(np.max(positive)) * 1.5) if len(positive) else 2.0)
    else:
        ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$E_x$ [MeV]")
    ax.set_ylabel("Counts")
    ax.set_title(histogram_label(str(item["name"])))
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_overview(
    histograms: list[dict[str, object]],
    path: Path,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    log_y: bool = False,
) -> None:
    n_hist = len(histograms)
    n_cols = 3
    n_rows = math.ceil(n_hist / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13.6, max(3.0, 2.1 * n_rows)),
        sharex=True,
    )
    flat_axes = np.atleast_1d(axes).ravel()
    for ax, item in zip(flat_axes, histograms):
        counts, edges = select_range(
            np.asarray(item["counts"], dtype=float),
            np.asarray(item["edges"], dtype=float),
            x_min,
            x_max,
        )
        ax.stairs(counts, edges, color=HISTOGRAM_COLOR, linewidth=0.8)
        ax.fill_between(
            np.repeat(edges, 2)[1:-1],
            np.repeat(counts, 2),
            step=None,
            color=HISTOGRAM_COLOR,
            alpha=0.20,
        )
        ax.set_xlim(edges[0], edges[-1])
        if log_y:
            positive = counts[counts > 0.0]
            ax.set_yscale("log")
            ax.set_ylim(
                0.8,
                max(2.0, float(np.max(positive)) * 1.5) if len(positive) else 2.0,
            )
        else:
            ax.set_ylim(bottom=0.0)
        ax.set_title(histogram_label(str(item["name"])), fontsize=8.5)
        ax.grid(True, alpha=0.20)
    for ax in flat_axes[len(histograms) :]:
        ax.axis("off")
    for ax in flat_axes[-n_cols:]:
        ax.set_xlabel(r"$E_x$ [MeV]")
    fig.supylabel("Counts", x=0.006)
    fig.suptitle(
        "inputHist.root Histograms" + (" (log y)" if log_y else ""),
        y=0.998,
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_pdf(
    histograms: list[dict[str, object]],
    path: Path,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for item in histograms:
            counts, edges = select_range(
                np.asarray(item["counts"], dtype=float),
                np.asarray(item["edges"], dtype=float),
                x_min,
                x_max,
            )
            fig, ax = plt.subplots(figsize=(9.2, 4.8))
            ax.stairs(counts, edges, fill=True, baseline=0.0, color=HISTOGRAM_COLOR, alpha=0.42)
            ax.stairs(counts, edges, color=HISTOGRAM_COLOR, linewidth=1.0)
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(bottom=0.0)
            ax.set_xlabel(r"$E_x$ [MeV]")
            ax.set_ylabel("Counts")
            ax.set_title(histogram_label(str(item["name"])))
            ax.grid(True, alpha=0.25)
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    histograms = load_histograms(args.input)
    if not histograms:
        raise ValueError(f"No TH1 histograms found in {args.input}")

    outdir = args.outdir
    individual_dir = outdir / "individual"
    individual_pdf_dir = outdir / "individual_pdf"
    for item in histograms:
        stem = safe_filename(str(item["name"]))
        plot_one(
            item,
            individual_dir / f"{stem}.png",
            x_min=args.x_min_MeV,
            x_max=args.x_max_MeV,
        )
        plot_one(
            item,
            individual_pdf_dir / f"{stem}.pdf",
            x_min=args.x_min_MeV,
            x_max=args.x_max_MeV,
        )
        if args.log_y:
            plot_one(
                item,
                individual_dir / f"{stem}_logy.png",
                x_min=args.x_min_MeV,
                x_max=args.x_max_MeV,
                log_y=True,
            )

    plot_overview(
        histograms,
        outdir / "inputHist_histograms_overview_linear.png",
        x_min=args.x_min_MeV,
        x_max=args.x_max_MeV,
    )
    if args.log_y:
        plot_overview(
            histograms,
            outdir / "inputHist_histograms_overview_logy.png",
            x_min=args.x_min_MeV,
            x_max=args.x_max_MeV,
            log_y=True,
        )
    write_pdf(
        histograms,
        outdir / "inputHist_histograms_all.pdf",
        x_min=args.x_min_MeV,
        x_max=args.x_max_MeV,
    )

    summary = pd.DataFrame(
        [
            {
                "histogram": item["name"],
                "entries": item["entries"],
                "nonzero_bins": item["nonzero_bins"],
                "x_min": item["x_min"],
                "x_max": item["x_max"],
                "bin_width": item["bin_width"],
                "plot_file": str(individual_dir / f"{safe_filename(str(item['name']))}.png"),
                "pdf_file": str(individual_pdf_dir / f"{safe_filename(str(item['name']))}.pdf"),
            }
            for item in histograms
        ]
    )
    summary_path = outdir / "inputHist_histogram_summary.csv"
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"wrote {len(histograms)} individual histogram plots to {individual_dir}")
    print(f"wrote {len(histograms)} individual histogram PDFs to {individual_pdf_dir}")
    print(f"wrote {outdir / 'inputHist_histograms_overview_linear.png'}")
    if args.log_y:
        print(f"wrote {outdir / 'inputHist_histograms_overview_logy.png'}")
    print(f"wrote {outdir / 'inputHist_histograms_all.pdf'}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
