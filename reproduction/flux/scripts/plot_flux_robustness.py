#!/usr/bin/env python3
"""Plot the fixed-device flux robustness results as a four-panel figure."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
import numpy as np


SAMPLE_COUNTS = (1000, 5000, 10000, 20000)
SCREEN_CONDITIONS = ("iid_clean", "rho03_clean", "iid_noise")
MATCHED_CONDITIONS = ("iid", "rho03")
RHO_CONDITIONS = {"iid": 0.0, "rho005": 0.05, "rho010": 0.1, "rho020": 0.2, "rho03": 0.3}
MATCHED_SAMPLE_COUNT = 20000
WEAK_PAIR_COUNT = 8
SCORE_NEIGHBOR_COUNT = 30
PARAMETER_STYLE = {3: ("#AD462C", "s"), 30: ("#176793", "o")}
SELECTED_MI_STYLE = {3: PARAMETER_STYLE[3], 130: ("#454382", "D")}
GRAY = "#686B76"
GREEN = "#19775F"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def score_cells(root, case, condition, n, repeats):
    directory = root / "scores" / case / condition
    return [read_json(directory / f"repeat_{repeat:02d}_n{n}.json")
            for repeat in range(repeats)]


def errorbar_series(axis, positions, rows, *, color, marker, label, linestyle="-", jitter=0.0):
    """Plot individual datasets and the group mean and sample standard deviation."""
    means, deviations = [], []
    for position, values in zip(positions, rows):
        finite = np.array([v for v in values if v is not None], dtype=float)
        means.append(float(finite.mean()) if len(finite) else np.nan)
        deviations.append(float(finite.std(ddof=1)) if len(finite) > 1 else 0.0)
        if jitter:
            offsets = np.linspace(-jitter, jitter, len(finite))
            positions_jittered = position * np.exp(offsets) if axis.get_xscale() == "log" else position + offsets
            axis.scatter(positions_jittered, finite, color=color, marker=marker,
                         s=10, alpha=0.26, linewidths=0, zorder=3)
    return axis.errorbar(positions, means, yerr=deviations, color=color, marker=marker, label=label,
                         linestyle=linestyle, linewidth=1.1, markersize=3.8, capsize=2.0, zorder=4)


def sample_axis(axis):
    axis.set_xscale("log")
    axis.set_xticks(SAMPLE_COUNTS, ["1k", "5k", "10k", "20k"])
    axis.set_xlim(840, 23500)
    axis.minorticks_off()
    axis.set_xlabel("Number of samples $N$ (log scale)")


def format_axis(axis, title):
    axis.set_title(title, loc="left", pad=6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", linewidth=0.45, color="#dadddf")
    axis.set_axisbelow(True)
    axis.tick_params(length=3, width=0.6)


def plot_rank_panel(axis, root, repeats):
    sample_axis(axis)
    by_count = [score_cells(root, "nn4", "iid", n, repeats) for n in SAMPLE_COUNTS]
    handles = []
    for k, (color, marker) in PARAMETER_STYLE.items():
        rows = [[cell["by_k"][str(k)]["average_precision"] for cell in cells] for cells in by_count]
        handles.append(errorbar_series(axis, SAMPLE_COUNTS, rows, color=color, marker=marker,
                                       label=rf"$M_{{ij}},\ k_{{\mathrm{{CMI}}}}={k}$", jitter=0.052))
    rows = [[cell["dcor"]["average_precision"] for cell in cells] for cells in by_count]
    handles.append(errorbar_series(axis, SAMPLE_COUNTS, rows, color=GRAY, marker="^",
                                   label="Distance correlation", jitter=0.052))
    random_ap = read_json(root / "scores/summary.json")["nn4_expected_ap_for_random_strict_ranking"]
    handles.append(axis.axhline(random_ap, color="#8B9298", linestyle=":", linewidth=0.85,
                                label="Random ranking"))
    axis.set_ylim(0, 1.14)
    axis.set_ylabel("Average precision")
    axis.legend(handles=handles, loc="upper left", ncol=2, frameon=False, columnspacing=1.0,
                borderaxespad=0.35, labelspacing=0.35)
    format_axis(axis, "(a) Sample efficiency")


def mean_sd(values):
    return {"mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def selected_scores(cell, pairs):
    scores = np.asarray(cell["by_k"][str(SCORE_NEIGHBOR_COUNT)]["scores_bits"])
    pair_indices = {tuple(pair): i for i, pair in enumerate(cell["pairs"])}
    expected = len(cell["pairs"])
    if (scores.shape != (expected,) or len(pair_indices) != expected
            or any(tuple(pair) not in pair_indices for pair in pairs)):
        raise ValueError(f"Invalid pair-score mapping for {cell['case']} repeat {cell['repeat']}")
    selected = scores[[pair_indices[tuple(pair)] for pair in pairs]]
    if not np.isfinite(selected).all():
        raise ValueError(f"Non-finite pair scores for {cell['case']} repeat {cell['repeat']}")
    return selected


def summarize_matched_weak_pairs(root, repeats):
    """Match the fixed weak NN4 pairs and summarize over independent datasets."""
    pairs = score_cells(root, "nn4", "iid", MATCHED_SAMPLE_COUNT, 1)[0]["weak_pairs"]
    if len(pairs) != WEAK_PAIR_COUNT or len(set(map(tuple, pairs))) != WEAK_PAIR_COUNT:
        raise ValueError("Expected eight distinct weak NN4 pairs")

    groups = {}
    sample_keys = None
    for condition in MATCHED_CONDITIONS:
        nn4 = score_cells(root, "nn4", condition, MATCHED_SAMPLE_COUNT, repeats)
        null = score_cells(root, "null", condition, MATCHED_SAMPLE_COUNT, repeats)
        records = []
        for repeat, (nn4_cell, null_cell) in enumerate(zip(nn4, null, strict=True)):
            seed = nn4_cell["sample_seed"]
            sample_hash = nn4_cell["provenance"]["input_sha256"]
            if (nn4_cell["repeat"] != repeat or null_cell["repeat"] != repeat
                    or nn4_cell["weak_pairs"] != pairs
                    or null_cell["sample_seed"] != seed
                    or null_cell["provenance"]["input_sha256"] != sample_hash):
                raise ValueError(f"Unmatched pair or sample provenance for {condition} repeat {repeat}")
            null_scores = selected_scores(null_cell, pairs)
            nn4_scores = selected_scores(nn4_cell, pairs)
            records.append({
                "repeat": repeat,
                "sample_seed": seed,
                "sample_sha256": sample_hash,
                "null_scores_bits": null_scores.tolist(),
                "nn4_scores_bits": nn4_scores.tolist(),
                "null_mean_bits": float(null_scores.mean()),
                "nn4_mean_bits": float(nn4_scores.mean()),
                "paired_mean_bits": float((nn4_scores - null_scores).mean()),
                "ranges_overlap": bool(max(null_scores.min(), nn4_scores.min())
                                       <= min(null_scores.max(), nn4_scores.max())),
            })
        keys = [(row["sample_seed"], row["sample_sha256"]) for row in records]
        if len(set(keys)) != repeats or (sample_keys is not None and keys != sample_keys):
            raise ValueError(f"Unmatched or duplicated dataset provenance for {condition}")
        sample_keys = keys
        groups[condition] = {
            "null": mean_sd([row["null_mean_bits"] for row in records]),
            "nn4": mean_sd([row["nn4_mean_bits"] for row in records]),
            "paired_delta": mean_sd([row["paired_mean_bits"] for row in records]),
            "overlapping_ranges": sum(row["ranges_overlap"] for row in records),
            "repeats": records,
        }
    return {"schema": "causal-q-matched-weak-pair-scores-v1", "selected_pairs": pairs,
            "sample_count": MATCHED_SAMPLE_COUNT, "k": SCORE_NEIGHBOR_COUNT, "conditions": groups}


def load_rho_sweep(root, matched):
    sweep = read_json(root / "rho_sweep/summary.json")
    if any(matched[key] != sweep[key] for key in ("selected_pairs", "sample_count", "k")):
        raise ValueError("Rho sweep and archived endpoints must use the same pairs and estimator")
    conditions = sweep["conditions"]
    if conditions.keys() != RHO_CONDITIONS.keys():
        raise ValueError("Rho sweep must include all five prescribed correlation settings")
    for condition in MATCHED_CONDITIONS:
        archived = matched["conditions"][condition]["repeats"]
        endpoint = conditions[condition]["repeats"]
        for previous, current in zip(archived, endpoint, strict=True):
            keys = ("sample_seed", "sample_sha256", "null_scores_bits", "nn4_scores_bits")
            if any(previous[key] != current[key] for key in keys):
                raise ValueError(f"Rho sweep endpoint differs from archived scores for {condition}")
    expected_seeds = [record["sample_seed"] for record in conditions["iid"]["repeats"]]
    for condition, rho in RHO_CONDITIONS.items():
        group = conditions[condition]
        records = group["repeats"]
        if group["rho"] != rho or [record["sample_seed"] for record in records] != expected_seeds:
            raise ValueError(f"Rho sweep has mismatched correlation or sample seeds for {condition}")
        for record in records:
            null_scores = np.asarray(record["null_scores_bits"])
            nn4_scores = np.asarray(record["nn4_scores_bits"])
            for scores, name in ((null_scores, "null"), (nn4_scores, "nn4")):
                if (scores.shape != (WEAK_PAIR_COUNT,) or not np.isfinite(scores).all()
                        or not np.isclose(scores.mean(), record[f"{name}_mean_bits"], rtol=0, atol=1e-14)):
                    raise ValueError(f"Rho sweep contains malformed {name} scores for {condition}")
            if not np.isclose((nn4_scores - null_scores).mean(), record["paired_mean_bits"],
                              rtol=0, atol=1e-14):
                raise ValueError(f"Rho sweep has an inconsistent paired gap for {condition}")
    return sweep


def load_selected_sweep(root):
    sweep = read_json(root / "rho_sweep/summary.json")
    conditions = sweep["conditions"]
    records = [group["repeats"] for group in conditions.values()]
    if (sweep["k"] != 3 or sweep["sample_count"] != MATCHED_SAMPLE_COUNT
            or list(conditions) != list(RHO_CONDITIONS) or len(sweep["selected_pairs"]) != WEAK_PAIR_COUNT
            or any([r["repeat"] for r in rows] != list(range(5)) for rows in records)):
        raise ValueError("Selected rho sweep must use the matched five datasets and k_CMI=3")
    return sweep


def plot_delta_inset(axis, positions, groups):
    inset = axis.inset_axes([0.105, 0.57, 0.37, 0.23])
    rows = [[1000 * row["paired_mean_bits"] for row in group["repeats"]] for group in groups]
    color = "#54456C"
    for rho, values in zip(positions, rows):
        inset.scatter(rho + np.linspace(-0.006, 0.006, len(values)), values, s=6.5, color=color,
                      alpha=0.39, linewidths=0, zorder=3)
    summary = [mean_sd(values) for values in rows]
    means = np.asarray([point["mean"] for point in summary])
    sd = np.asarray([point["sd"] or 0 for point in summary])
    inset.errorbar(positions, means, yerr=sd, fmt="o", color=color,
                   markersize=2.6, capsize=1.8, elinewidth=0.78, zorder=4)
    inset.axhline(0, color="#858991", linewidth=0.7, linestyle=":")
    inset.set_xlim(-0.022, 0.322)
    low = min(0, min(map(min, rows)), min(means - sd))
    high = max(max(map(max, rows)), max(means + sd))
    padding = 0.045 * (high - low)
    inset.set_ylim(low - padding, high + padding)
    inset.set_xticks((0.0, 0.1, 0.2, 0.3), ("0", "0.1", "0.2", "0.3"))
    inset.yaxis.set_major_locator(MaxNLocator(nbins=4))
    inset.tick_params(axis="both", labelsize=6, length=2, pad=1.4)
    inset.set_title(r"$\Delta M_{ij}$ ($10^{-3}$ bits)", fontsize=6.5, loc="left", pad=2.5)
    for spine in inset.spines.values():
        spine.set_color("#9DA1A8")
        spine.set_linewidth(0.55)


def plot_rho_sweep_panel(axis, sweep):
    positions = np.array(list(RHO_CONDITIONS.values()))
    groups = [sweep["conditions"][name] for name in RHO_CONDITIONS]
    models = (("null", GRAY, "o", "--", "Uncoupled control", -1),
              ("nn4", GREEN, "s", "-", "Weak links (sparse)", 1))
    for model, color, marker, style, _, sign in models:
        rows = [[row[f"{model}_mean_bits"] for row in group["repeats"]] for group in groups]
        summary = [mean_sd(values) for values in rows]
        means = [point["mean"] for point in summary]
        sd = [point["sd"] or 0 for point in summary]
        axis.plot(positions, means, color=color, linestyle=style, linewidth=1.15, zorder=3)
        for rho, values in zip(positions, rows):
            offsets = sign * np.linspace(0.004, 0.012, len(values))
            axis.scatter(rho + offsets, values, color=color, marker=marker, s=8,
                         alpha=0.36, linewidths=0, zorder=4)
        axis.errorbar(positions + sign * 0.002, means, yerr=sd, fmt=marker, color=color,
                      markersize=3.5, capsize=1.9, elinewidth=0.85, zorder=5)
    axis.set_xlim(-0.026, 0.326)
    values = [row[f"{model}_mean_bits"] for group in groups for row in group["repeats"]
              for model, *_ in models]
    low, high = min(0, min(values)), max(0, max(values))
    span = high - low
    axis.set_ylim(low - 0.10 * span, high + 0.17 * span)
    axis.set_xticks(positions, ("0", "0.05", "0.1", "0.2", "0.3"))
    axis.set_xlabel(r"Flux correlation $\rho$")
    axis.set_ylabel(r"Mean pair score $M_{ij}$ (bits)")
    axis.axhline(0, color="#8B9298", linestyle=":", linewidth=0.8)
    handles = [Line2D([], [], color=color, marker=marker, linestyle=style, markersize=3.6, label=label)
               for _, color, marker, style, label, _ in models]
    axis.legend(handles=handles, loc="upper left", ncol=2, frameon=False,
                columnspacing=1.1, borderaxespad=0.35)
    format_axis(axis, "(b) Flux-correlation sweep")
    plot_delta_inset(axis, positions, groups)


def plot_screen_panel(axis, root, repeats):
    records = read_json(root / "screen" / "summary.json")["cells"]
    conditions = [[row for row in records if row["condition"] == name] for name in SCREEN_CONDITIONS]
    for name, rows in zip(SCREEN_CONDITIONS, conditions):
        if {row["repeat"] for row in rows} != set(range(repeats)) or len(rows) != repeats:
            raise ValueError(f"Incomplete or duplicated screening repeats for {name}")
    x = np.arange(len(conditions))
    for field, offset, color, marker in (("precision", -0.13, "#2D4853", "o"),
                                         ("recall", 0.13, "#795787", "D")):
        rows = [[row[field] for row in cells] for cells in conditions]
        errorbar_series(axis, x + offset, rows, color=color, marker=marker,
                        label=field.capitalize(), linestyle="none", jitter=0.065)
    axis.set_ylim(0, 1.055)
    axis.set_xlim(-0.55, 2.55)
    axis.set_xticks(x, ("Independent\nClean", r"Correlated ($\rho=0.3$)" + "\nClean",
                        "Independent\n0.5 MHz noise"))
    axis.set_ylabel("Precision / recall")
    axis.legend(loc="lower right", ncol=2, frameon=False)
    format_axis(axis, "(c) Directed screening")


def plot_self_mi_panel(axis, root, repeats, title, parameter_style=PARAMETER_STYLE):
    records = read_json(root / "diagnostics" / "summary.json")["cells"]
    sample_axis(axis)
    for k, (color, marker) in parameter_style.items():
        for noise, noise_label, style in ((0.0, "clean", "-"), (0.0005, "0.5 MHz", "--")):
            rows = [[row["mean_bits"] for row in records if
                     (row["sample_count"], row["noise_ghz"], row["k"]) == (n, noise, k)]
                    for n in SAMPLE_COUNTS]
            if any(len(values) != repeats for values in rows):
                raise ValueError(f"Incomplete self-MI repeats at noise={noise:g} GHz, k_MI={k}")
            series = errorbar_series(axis, SAMPLE_COUNTS, rows, color=color, marker=marker,
                                     label=rf"$k_\mathrm{{MI}}={k}$, {noise_label}", linestyle=style)
            if noise and 130 in parameter_style:
                series.lines[0].set_markerfacecolor("white")
                series.lines[0].set_markersize(2.9)
    axis.set_ylabel(r"Local $\widehat I(\Phi_i;f_i^{\mathrm{obs}})$ (bits)")
    axis.set_ylim(0.8 if 130 in parameter_style else 2.65, 12.25)
    axis.legend(loc="upper left", ncol=2, frameon=False, columnspacing=0.9)
    format_axis(axis, title)


def plot_robustness(root, repeats, sweep=None, selected=None):
    if sweep is None:
        sweep = load_rho_sweep(root, summarize_matched_weak_pairs(root, repeats))
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.85), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.07, w_pad=0.07, hspace=0.09, wspace=0.07)
    rank, background, screen, self_mi = axes.flat
    plot_rank_panel(rank, root, repeats)
    plot_rho_sweep_panel(background, sweep)
    plot_screen_panel(screen, selected or root, repeats)
    plot_self_mi_panel(self_mi, selected or root, repeats, "(d) Local MI diagnostic",
                       SELECTED_MI_STYLE if selected else PARAMETER_STYLE)
    return fig


def plot_weak_pair_distribution(matched, repeats, comparison=None):
    sweeps = (matched, comparison) if comparison else (matched,)
    height = 2.0 + 0.50 * repeats
    fig, axes = plt.subplots(repeats, len(sweeps), figsize=(6.9 if comparison else 6.4, height),
                             sharex="col", squeeze=False)
    fig.subplots_adjust(left=0.155, right=0.975, bottom=1.25 / height,
                        top=1 - (0.68 if comparison else 0.52) / height, hspace=0.19, wspace=0.18)
    models = (("null", "o", GRAY, 0.73, r"Uncoupled control ($h=0$)"),
              ("nn4", "s", GREEN, 0.27, "Weak pairs (sparse)"))
    shared_fill = "#F2E4BF"
    offsets = np.resize((0.0, 0.11, -0.11, 0.05, -0.05), WEAK_PAIR_COUNT)
    for column, sweep in enumerate(sweeps):
        column_axes = axes[:, column]
        records = sweep["conditions"]["rho03"]["repeats"]
        minimum, maximum = np.inf, -np.inf
        for axis, record in zip(column_axes, records, strict=True):
            bounds = []
            for model, marker, color, row, _ in models:
                scores = np.sort(record[f"{model}_scores_bits"])
                lower, upper = float(scores[0]), float(scores[-1])
                bounds.append((lower, upper))
                minimum, maximum = min(minimum, lower), max(maximum, upper)
                axis.hlines(row, lower, upper, color=color, alpha=0.70, linewidth=1.0, zorder=2)
                axis.scatter(scores, row + offsets, marker=marker, color=color, s=12.5,
                             edgecolors="white", linewidths=0.25, zorder=3)
            shared_start = max(low for low, _ in bounds)
            shared_end = min(high for _, high in bounds)
            if shared_start <= shared_end:
                axis.axvspan(shared_start, shared_end, color=shared_fill, linewidth=0, zorder=0)
            if column == 0:
                axis.text(-0.036, 0.5, f"Dataset {record['repeat'] + 1}", ha="right", va="center",
                          transform=axis.transAxes)
            axis.set_ylim(0, 1)
            axis.set_yticks([])
            axis.spines[["left", "right", "top"]].set_visible(False)
            axis.spines["bottom"].set_color("#CDD0D4")
            axis.tick_params(axis="x", length=2.8, width=0.6)
        if comparison:
            column_axes[0].set_title(rf"$k_\mathrm{{CMI}}={sweep['k']}$")
        column_axes[-1].set_xlim(minimum - 0.0007, maximum + 0.0007)
        column_axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=5 if comparison else 6))
        column_axes[-1].set_xlabel(r"Pair score $M_{ij}$ (bits)")
        for axis in column_axes[:-1]:
            axis.tick_params(axis="x", bottom=False)
    legend = [Line2D([], [], color=color, marker=marker, markersize=4, linewidth=1, label=label)
              for _, marker, color, _, label in models]
    legend.append(Patch(facecolor=shared_fill, edgecolor="none", label="Shared score range"))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.565, 0.97), ncol=3,
               frameon=False, columnspacing=1.7)
    protocol = (rf"Both panels use clean responses at $N=20{{,}}000$ and show the same {repeats} independent datasets."
                if comparison else
                rf"Clean responses use $N=20{{,}}000$ and $k_\mathrm{{CMI}}={matched['k']}$ in each of {repeats} datasets.")
    caption = (
        r"$\mathbf{Fig.\ S1.}$ Raw scores at $\rho=0.30$ for the eight weakest pairs in the sparse case."
        "\n"
        "The uncoupled control uses the same pair indices. Both models use the same flux rows."
        "\n"
        + protocol +
        "\n"
        "Dots show individual pair scores with vertical offsets. Lines span the observed ranges."
        "\n"
        "Shading marks their intersection." + (" The panels use different horizontal scales." if comparison else "")
    )
    fig.text(0.025, 0.023, caption, ha="left", va="bottom", fontsize=8, linespacing=1.25,
             gid="supplement-caption")
    return fig


def plot_self_mi(root, repeats, parameter_style=PARAMETER_STYLE):
    fig, axis = plt.subplots(figsize=(3.38, 2.28), layout="constrained")
    plot_self_mi_panel(axis, root, repeats, "", parameter_style)
    return fig


def save_figure(fig, directory, filename, *, appendix_bounds=None):
    if appendix_bounds is None:
        fig.savefig(directory / f"{filename}.pdf")
        fig.savefig(directory / f"{filename}.png", dpi=240)
    else:
        for artist in tuple(fig.texts):
            if artist.get_gid() == "supplement-caption":
                artist.remove()
        # The artwork bounds are specified in PDF points. Matplotlib takes inches.
        bbox = Bbox.from_extents(*(coordinate / 72 for coordinate in appendix_bounds))
        fig.savefig(directory / f"{filename}_appendix.pdf", bbox_inches=bbox)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Experiment result directory")
    parser.add_argument("--selected", type=Path, help="Selected configuration results (130/3)")
    parser.add_argument("--reference-cmi-k", type=int, choices=(3, 30),
                        help="Pair-score neighbor count for the flux-correlation sweep")
    parser.add_argument("--figure-dir", type=Path, help="Output directory (default: OUT/figures_rho_sweep)")
    parser.add_argument("--repeats", default=5, type=int)
    parser.add_argument("--only", choices=("robustness", "self-mi", "weak-pair-distribution"))
    parser.add_argument("--appendix-figures", action="store_true",
                        help="Save the weak-pair figure as a caption-free appendix PDF")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    reference_k = args.reference_cmi_k or (3 if args.selected else 30)
    if reference_k == 3 and not args.selected:
        parser.error("--reference-cmi-k 3 requires --selected")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9.2, "axes.titleweight": "bold", "axes.linewidth": 0.65,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "legend.fontsize": 7.0,
        "legend.handlelength": 1.65, "pdf.fonttype": 42,
    })
    variant = "figures_rho_sweep_k30" if args.selected and reference_k == 30 else "figures_rho_sweep"
    directory = args.figure_dir or (args.selected or args.out) / variant
    directory.mkdir(parents=True, exist_ok=True)
    if args.only == "self-mi":
        save_figure(plot_self_mi(args.selected or args.out, args.repeats,
                                SELECTED_MI_STYLE if args.selected else PARAMETER_STYLE), directory, "flux_self_mi")
    else:
        if reference_k == 30:
            matched = summarize_matched_weak_pairs(args.out, args.repeats)
        else:
            matched = load_selected_sweep(args.selected)
        if args.only in (None, "robustness"):
            sweep = load_rho_sweep(args.out, matched) if reference_k == 30 else matched
            save_figure(plot_robustness(args.out, args.repeats, sweep, args.selected), directory, "flux_robustness")
        if args.only in (None, "weak-pair-distribution"):
            comparison = None
            if args.selected:
                comparison = (load_selected_sweep(args.selected) if reference_k == 30
                              else summarize_matched_weak_pairs(args.out, args.repeats))
            if comparison:
                for key in ("selected_pairs", "sample_count"):
                    if matched[key] != comparison[key]:
                        raise ValueError("Supplemental comparisons must use the same pairs and sample count")
                sample_keys = ("repeat", "sample_seed", "sample_sha256")
                for new, archived in zip(matched["conditions"]["rho03"]["repeats"],
                                         comparison["conditions"]["rho03"]["repeats"], strict=True):
                    if any(new[key] != archived[key] for key in sample_keys):
                        raise ValueError("Supplemental comparisons must use matched datasets")
            save_figure(plot_weak_pair_distribution(matched, args.repeats, comparison), directory,
                        "flux_weak_pair_distribution",
                        appendix_bounds=(0, 61, 497, 319) if args.appendix_figures else None)
            (directory / "matched_weak_pairs.json").write_text(
                json.dumps(matched, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
