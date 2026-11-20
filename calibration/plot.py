"""Render Fig. 5 from conflict graphs and the mean best-score trajectories."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASES = ("uncoupled", "chain", "dense")
METHODS = {
    "uncompressed": ("Uncompressed", "#606a77", 2.2, "-"),
    "original": ("Graph support", "#246eb2", 1.5, "-"),
    "audited": ("Audited support", "#169779", 1.5, "--"),
    "shuffled": ("Shuffled support", "#bd7750", 1.5, "--"),
}
CASE_NAMES = ("Uncoupled", "Segmented chain", "Dense")
PALETTE = ("#347eac", "#cd8b38", "#659d7b", "#956ba9", "#b86876", "#7395a2", "#93935a", "#8e7460")


def plot(cases, budget, directory):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6), gridspec_kw={"height_ratios": [1, 1.05]})
    theta = np.pi / 2 + np.arange(8) * 2 * np.pi / 8
    xy = np.column_stack((np.cos(theta), np.sin(theta)))
    for column, (case, name) in enumerate(zip(CASES, CASE_NAMES)):
        data = cases[case]
        structure = data["structure"]
        ax = axes[0, column]
        for i, j in np.argwhere(np.triu(data["conflicts"], k=1)):
            ax.plot(xy[[i, j], 0], xy[[i, j], 1], color="#bbc2ca", lw=0.8, zorder=1)
        for group, color in zip(structure["groups"], PALETTE):
            ax.scatter(
                xy[group, 0],
                xy[group, 1],
                s=190,
                c=color,
                edgecolors="white",
                linewidths=0.9,
                zorder=2,
            )
        for site, (x, y) in enumerate(xy):
            ax.text(
                x,
                y,
                str(site),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
                zorder=3,
            )
        ax.set(aspect="equal", xlim=(-1.35, 1.35), ylim=(-1.28, 1.28))
        ax.axis("off")
        ax.set_title(f"{name} case", pad=2)
        ax.text(
            0.5,
            -0.06,
            f"Mean reach: {structure['mean_reach']:g}\nTask batches: {structure['minimum_task_batches']}",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            linespacing=1.3,
        )
        ax.text(
            -0.08,
            1.06,
            f"({chr(97 + column)})",
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,
        )

        ax = axes[1, column]
        for method, (label, color, width, style) in METHODS.items():
            ax.plot(
                np.arange(1, budget + 1),
                data["curves"][method],
                label=label,
                color=color,
                lw=width,
                ls=style,
            )
        ax.set(xlim=(1, budget), xlabel="Paid vector queries")
        ax.set_xticks([1, budget // 4, budget // 2, 3 * budget // 4, budget])
        ax.grid(alpha=0.18, lw=0.5)
        ax.set_title(f"{name} model", pad=5)
        ax.text(
            -0.16,
            1.06,
            f"({chr(100 + column)})",
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,
        )
        if column == 0:
            ax.set_ylabel(r"Mean best four-probe $\widetilde F$")

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=7,
        columnspacing=1,
        handlelength=2,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.955, bottom=0.13, hspace=0.70, wspace=0.32)
    fig.savefig(directory / "figure5.pdf", dpi=250, metadata={"CreationDate": None})
    plt.close(fig)
