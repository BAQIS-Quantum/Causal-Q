#!/usr/bin/env python3
"""Plot the NN4 pair-ranking sensitivity to the endpoint-CMI neighbor count."""

import argparse
from pathlib import Path

import numpy as np

from plot_flux_robustness import errorbar_series, format_axis, plt, read_json, save_figure


COUNTS = (5000, 20000)
NEIGHBOR_COUNTS = (3, 6, 10, 20, 30)
PANELS = (
    ("average_precision", "(a) Support ranking", "Average precision"),
    ("spearman_nonzero", "(b) Coupled pairs", r"Spearman $r_s$"),
    ("spearman_all", "(c) All pairs", r"Spearman $r_s$"),
)
STYLES = (("#795787", "D", "--", -0.58), ("#19775F", "o", "-", 0.58))


def read_rankings(path):
    groups = read_json(path)["groups"]
    if sorted(group["sample_count"] for group in groups) != list(COUNTS):
        raise ValueError("Expected one group for each of N=5000 and N=20000")

    rankings = {}
    expected_ks = {str(k) for k in NEIGHBOR_COUNTS}
    for group in groups:
        n = group["sample_count"]
        by_k = group["by_k"]
        if by_k.keys() != expected_ks:
            raise ValueError(f"Incomplete k_CMI sweep for N={n}")
        for metric, _, _ in PANELS:
            rows = np.array([by_k[str(k)][metric]["values"] for k in NEIGHBOR_COUNTS], dtype=float)
            lower = 0 if metric == "average_precision" else -1
            if (rows.shape != (len(NEIGHBOR_COUNTS), 5) or not np.isfinite(rows).all()
                    or rows.min() < lower or rows.max() > 1):
                raise ValueError(f"Invalid {metric} values for N={n}")
            rankings[n, metric] = rows
    return rankings


def plot_rankings(rankings):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.4, 4.5))
    fig.subplots_adjust(left=0.092, right=0.983, bottom=0.343, top=0.85, wspace=0.46)
    positions = np.asarray(NEIGHBOR_COUNTS)
    spearman = np.stack([rankings[n, metric] for n in COUNTS
                         for metric in ("spearman_nonzero", "spearman_all")])
    spearman_means = spearman.mean(axis=-1)
    spearman_sd = spearman.std(axis=-1, ddof=1)
    bottom = min(-0.45, spearman.min() - 0.05, (spearman_means - spearman_sd).min() - 0.05)
    top = max(1.025, (spearman_means + spearman_sd).max() + 0.025)
    for axis, (metric, title, ylabel) in zip(axes, PANELS, strict=True):
        for n, (color, marker, style, offset) in zip(COUNTS, STYLES, strict=True):
            errorbar_series(axis, positions + offset, rankings[n, metric], color=color,
                            marker=marker, linestyle=style, label=f"$N={n:,}$".replace(",", "{,}"),
                            jitter=0.52)
        axis.set_xlim(1.35, 31.65)
        axis.set_xticks(positions)
        axis.set_xlabel(r"Neighbor count $k_\mathrm{CMI}$")
        axis.set_ylabel(ylabel)
        if metric == "average_precision":
            axis.set_ylim(0, 1.025)
            axis.set_yticks(np.linspace(0, 1, 6))
        else:
            axis.set_ylim(bottom, top)
            axis.set_yticks([tick for tick in np.linspace(-1, 1, 9) if bottom <= tick <= top])
            axis.axhline(0, color="#8B9298", linestyle=":", linewidth=0.8)
        format_axis(axis, title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.97),
               ncol=2, frameon=False, columnspacing=2)
    caption = (
        r"$\mathbf{Fig.\ S2.}$ Sensitivity of sparse-case pair rankings to $k_\mathrm{CMI}$ in Eq. (4). "
        "The scan uses"
        "\n"
        "independent fluxes and noiseless frequency responses. The two sample counts use nested"
        "\n"
        r"prefixes of each of five independent datasets, with the same rows for every $k_\mathrm{CMI}$. Average"
        "\n"
        "precision (AP) evaluates the ranking of 24 coupled pairs among all 120 unordered pairs."
        "\n"
        r"Spearman $r_s$ compares scores with $|h_{ij}|$ for 24 coupled pairs in (b) and all 120 pairs in (c)."
        "\n"
        "Small symbols show individual datasets. Large symbols and bars show their mean and sample"
        "\n"
        "standard deviation. Symbols are shifted slightly along the horizontal axis for visibility."
    )
    fig.text(0.025, 0.023, caption, ha="left", va="bottom", fontsize=8, linespacing=1.22,
             gid="supplement-caption")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="JSON ranking summary for the k_CMI sweep")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--appendix-figures", action="store_true",
                        help="Save the plot as a caption-free appendix PDF")
    args = parser.parse_args()
    rankings = read_rankings(args.summary)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9.2, "axes.titleweight": "bold", "axes.linewidth": 0.65,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "legend.fontsize": 7.4,
        "legend.handlelength": 1.65, "pdf.fonttype": 42,
    }):
        save_figure(plot_rankings(rankings), args.output_directory, "flux_k_sensitivity",
                    appendix_bounds=(0, 82, 461, 319) if args.appendix_figures else None)


if __name__ == "__main__":
    main()
