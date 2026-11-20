"""Figure 6. Best mean single-qubit Haar fidelity by query."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator


CASES = (("uncoupled", "Uncoupled"), ("chain", "Segmented chain"), ("dense", "Dense"))
PLANS = (("planned_ascending", "Ascending"), ("planned_permuted", "Permuted"))
METHODS = {
    "individual": {"label": "Individual differences", "color": "#596570", "lw": 1.8},
    "graph": {"label": "Graph support", "color": "#2064a0", "lw": 1.35, "ls": (0, (3.3, 2.0))},
}
THRESHOLD = 0.995
BUDGET = 256


def plot(output, case_curves):
    queries = np.arange(1, BUDGET + 1)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(7.2, 3.38))
    for j, (case, title) in enumerate(CASES):
        curves = case_curves[case]
        axes[0, j].set_title(title, pad=6)
        for i, (plan, _) in enumerate(PLANS):
            ax = axes[i, j]
            ax.set_xscale("log", base=2)
            ax.set(xlim=(1, BUDGET), ylim=(0.970, 1.0016))
            ax.xaxis.set_major_locator(FixedLocator([1, 4, 16, 64, 256]))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_locator(FixedLocator([0.970, 0.980, 0.990, 1.000]))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.tick_params(length=2.2, width=0.6, direction="out", pad=2)
            ax.grid(axis="y", color="#e0e4e8", lw=0.5)
            ax.axhline(
                THRESHOLD,
                color="#76584a",
                lw=0.8,
                ls=(0, (1, 2)),
                label="0.995 threshold",
                zorder=1.5,
            )
            for k, (method, style) in enumerate(METHODS.items()):
                ax.step(queries, curves[plan, method], where="post", zorder=3 + k, **style)
            panel = chr(ord("a") + 3 * i + j)
            ax.text(
                0.042,
                0.94,
                f"({panel})",
                transform=ax.transAxes,
                va="top",
                fontweight="bold",
                fontsize=8,
            )

    fig.subplots_adjust(left=0.082, right=0.947, top=0.9, bottom=0.21, hspace=0.27, wspace=0.12)
    for i, (_, name) in enumerate(PLANS):
        bounds = axes[i, -1].get_position()
        fig.text(0.978, (bounds.y0 + bounds.y1) / 2, name, va="center", ha="center", rotation=-90)
    fig.supxlabel("Full-vector queries", x=0.515, y=0.09, fontsize=8)
    fig.supylabel("Best mean Haar fidelity", x=0.021, y=0.555, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [1, 2, 0]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower center",
        ncol=3,
        frameon=False,
        handlelength=2.8,
        bbox_to_anchor=(0.515, 0.008),
    )
    fig.savefig(output, metadata={"CreationDate": None})
    plt.close(fig)
