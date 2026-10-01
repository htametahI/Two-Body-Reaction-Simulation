#!/usr/bin/env python3
"""Plot selected transmission angular distributions on a shared scale."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle


TRANSMISSION_DIR = Path(__file__).resolve().parents[1] / "output" / "transmission"
STATES = (
    ("3.588 MeV (0+)", TRANSMISSION_DIR / "3588_0plus.csv"),
    ("10.949 MeV (1-)", TRANSMISSION_DIR / "10949_1minus.csv"),
)
OUTPUT = TRANSMISSION_DIR / "theta_phi_3588_10949.png"
VIEW_LIMIT_DEG = 10.0
ACCEPTANCE_DEG = 4.2
N_BINS = 240


def main() -> None:
    edges = np.linspace(-VIEW_LIMIT_DEG, VIEW_LIMIT_DEG, N_BINS + 1)
    histograms = []
    event_counts = []
    acceptance_fractions = []

    for _, path in STATES:
        angles = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2))
        angles = angles[np.isfinite(angles).all(axis=1)]
        histogram, _, _ = np.histogram2d(
            angles[:, 0], angles[:, 1], bins=(edges, edges)
        )
        histograms.append(histogram.T)
        event_counts.append(len(angles))
        radius = np.hypot(angles[:, 0], angles[:, 1])
        acceptance_fractions.append(np.count_nonzero(radius <= ACCEPTANCE_DEG) / len(radius))

    vmax = max(float(histogram.max()) for histogram in histograms)
    fig, axes = plt.subplots(
        1, 2, figsize=(12.4, 5.7), sharex=True, sharey=True, constrained_layout=True
    )

    mesh = None
    for ax, (label, _), histogram, count, acceptance_fraction in zip(
        axes, STATES, histograms, event_counts, acceptance_fractions
    ):
        mesh = ax.pcolormesh(
            edges,
            edges,
            histogram,
            cmap="plasma",
            norm=LogNorm(vmin=1, vmax=vmax),
            shading="auto",
            rasterized=True,
        )
        ax.add_patch(
            Circle(
                (0, 0),
                ACCEPTANCE_DEG,
                fill=False,
                edgecolor="black",
                linewidth=2.8,
                linestyle=(0, (4, 3)),
                alpha=0.55,
                zorder=4,
            )
        )
        ax.add_patch(
            Circle(
                (0, 0),
                ACCEPTANCE_DEG,
                fill=False,
                edgecolor="white",
                linewidth=1.6,
                linestyle=(0, (4, 3)),
                zorder=5,
            )
        )
        ax.axhline(0, color="white", linewidth=0.5, alpha=0.35)
        ax.axvline(0, color="white", linewidth=0.5, alpha=0.35)
        ax.set_title(f"{label}\n{count:,} total S3-gated events", fontsize=12)
        ax.text(
            0.03,
            0.96,
            f"Within 4.2 deg: {acceptance_fraction:.1%}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 4},
            zorder=6,
        )
        ax.set_xlabel(r"$\theta_{spec}$ (deg)")
        ax.set_xlim(-VIEW_LIMIT_DEG, VIEW_LIMIT_DEG)
        ax.set_ylim(-VIEW_LIMIT_DEG, VIEW_LIMIT_DEG)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks((-10, -5, 0, 5, 10))
        ax.set_yticks((-10, -5, 0, 5, 10))

    axes[0].set_ylabel(r"$\phi_{spec}$ (deg)")
    colorbar = fig.colorbar(mesh, ax=axes, shrink=0.88, pad=0.025)
    colorbar.set_ticks([])
    colorbar.ax.tick_params(which="both", length=0)
    fig.suptitle("Recoil Distribution", fontsize=15)
    fig.text(
        0.5,
        0.015,
        "Dashed circle: 4.2 deg angular acceptance",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    print(
        f"Wrote {OUTPUT} with event counts {event_counts} "
        f"and acceptance fractions {acceptance_fractions}"
    )


if __name__ == "__main__":
    main()
