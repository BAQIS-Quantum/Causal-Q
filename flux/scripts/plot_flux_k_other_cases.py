#!/usr/bin/env python3
"""Plot the endpoint-CMI neighbor-count scan for the other flux cases."""

import argparse
from pathlib import Path

from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_flux_k_sensitivity import NEIGHBOR_COUNTS
from plot_flux_robustness import errorbar_series, format_axis, plt, read_json, save_figure


CASE_COUNTS = {"four": (20_000, 6, 6), "nn8": (40_000, 120, 42), "dense": (40_000, 120, 120)}
METRICS = ("average_precision", "spearman_nonzero", "spearman_all")


def read_rankings(path):
    summary = read_json(path)
    groups = summary["groups"]
    counts = {g["case"]: (g["sample_count"], g["pair_count"], g["coupled_pair_count"]) for g in groups}
    if len(groups) != len(CASE_COUNTS) or counts != CASE_COUNTS:
        raise ValueError("Expected the four-qubit, NN8 and dense benchmark cases")
    if summary["rho"] != 0 or summary["noise_sd_ghz"] != 0 or len(set(summary["sample_seeds"])) != 5:
        raise ValueError("Expected five independent-flux, noiseless datasets per case")

    rankings = {}
    expected_ks = {str(k) for k in NEIGHBOR_COUNTS}
    for group in groups:
        case = group["case"]
        by_k = group["by_k"]
        if by_k.keys() != expected_ks or group["repeats"] != list(range(5)):
            raise ValueError(f"Incomplete k_CMI sweep for {case}")
        for metric in METRICS:
            summaries = [by_k[str(k)][metric] for k in NEIGHBOR_COUNTS]
            if case != "nn8" and metric == "average_precision":
                if any(entry is not None for entry in summaries):
                    raise ValueError(f"Support ranking is degenerate for fully coupled case {case}")
                continue
            rows = np.array([entry["values"] for entry in summaries], dtype=float)
            lower = 0 if metric == "average_precision" else -1
            if (rows.shape != (len(NEIGHBOR_COUNTS), 5) or not np.isfinite(rows).all()
                    or rows.min() < lower or rows.max() > 1):
                raise ValueError(f"Invalid {metric} values for {case}")
            rankings[case, metric] = rows
        if case != "nn8" and not np.array_equal(rankings[case, "spearman_nonzero"], rankings[case, "spearman_all"]):
            raise ValueError(f"All and coupled pairs must coincide for {case}")
    return rankings


def plot_rankings(rankings):
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 4.5))
    fig.subplots_adjust(left=0.097, right=0.983, bottom=0.375, top=0.841, wspace=0.49)
    positions = np.asarray(NEIGHBOR_COUNTS)
    panels = (
        ("(a) Moderate support", "Average precision", (
            ("nn8", "average_precision", "#795787", "D", "-", 0.0, "Moderate"),
        )),
        ("(b) Moderate pair ranks", r"Spearman $r_s$", (
            ("nn8", "spearman_nonzero", "#176793", "o", "-", -0.58, "Coupled"),
            ("nn8", "spearman_all", "#AD462C", "s", "--", 0.58, "All"),
        )),
        ("(c) Fully coupled", r"Spearman $r_s$", (
            ("four", "spearman_all", "#19775F", "^", "-", -0.58, "Four qubits"),
            ("dense", "spearman_all", "#686B76", "o", "--", 0.58, "Dense"),
        )),
    )
    for axis, (title, ylabel, series) in zip(axes, panels, strict=True):
        data = []
        for case, metric, color, marker, linestyle, offset, label in series:
            rows = rankings[case, metric]
            data.append(rows)
            errorbar_series(axis, positions + offset, rows, color=color, marker=marker, label=label,
                            linestyle=linestyle, jitter=0.52)
        observed = np.stack(data)
        means = observed.mean(axis=-1)
        deviations = observed.std(axis=-1, ddof=1)
        minimum = min(observed.min(), (means - deviations).min())
        maximum = max(observed.max(), (means + deviations).max())
        margin = max(0.004, 0.09 * (maximum - minimum))
        bottom, top = minimum - margin, maximum + margin
        axis.set_ylim(bottom, top)
        metric_minimum = 0 if ylabel == "Average precision" else -1
        ticks = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(bottom, top).round(12)
        axis.set_yticks([tick for tick in ticks if max(metric_minimum, bottom) <= tick <= min(1, top)])
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.set_xlim(1.35, 31.65)
        axis.set_xticks(positions)
        axis.set_xlabel(r"Neighbor count $k_\mathrm{CMI}$")
        axis.set_ylabel(ylabel)
        format_axis(axis, title)
        if len(series) > 1:
            axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2,
                        frameon=False, columnspacing=1.0, borderaxespad=0)

    caption = (
        r"$\mathbf{Fig.\ S3.}$ Sensitivity of pair rankings to $k_\mathrm{CMI}$ in Eq. (4) under independent fluxes and clean"
        "\n"
        r"frequencies. Each of five datasets per case uses the same rows for all $k_\mathrm{CMI}$ ($N=20{,}000$ for"
        "\n"
        r"four qubits and $N=40{,}000$ for moderate and dense). In the moderate case, 42 of 120 pairs"
        "\n"
        r"are coupled. In (c), only all-pair $r_s$ is shown, with 6 coupled pairs for four qubits and 120 for dense."
        "\n"
        r"Spearman correlations compare $M_{ij}$ with $|h_{ij}|$. Vertical scales differ by panel. Small markers"
        "\n"
        "denote datasets. Large markers and error bars give the mean and sample standard deviation."
        "\n"
        "Points are offset horizontally for clarity."
    )
    fig.text(0.025, 0.023, caption, ha="left", va="bottom", fontsize=8, linespacing=1.22,
             gid="supplement-caption")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
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
        save_figure(plot_rankings(rankings), args.output_directory, "flux_k_other_cases",
                    appendix_bounds=(0, 92, 461, 322) if args.appendix_figures else None)


if __name__ == "__main__":
    main()
