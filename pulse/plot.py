"""Render the eight panels of Fig. 4 from graphs, MI matrices and couplings."""
import numpy as np

_NETWORK_TITLES = {
    "uncoupled": "Uncoupled",
    "chain": "Sparse (segmented chain)",
    "dense": "Dense (fully connected)",
}
_ROWS = ("Phi", "f", "omega_d", "F", "Omega")
_COLORS = ("#82cfe3", "#8ee991", "#ffb200", "#f1e58a", "#f08080")
_BLUE = "#165dc2"


def _box(fig, bounds):
    from matplotlib.patches import FancyBboxPatch
    x, y, width, height = bounds
    fig.add_artist(FancyBboxPatch(
        (x, y), width, height, transform=fig.transFigure,
        boxstyle="round,pad=0.004,rounding_size=0.014", facecolor="none",
        edgecolor=".15", linewidth=.7, zorder=-1,
    ))


def _schematic(fig):
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    ax = fig.add_axes((.041, .79, .48, .115))
    for i in range(7):
        for shift in (-.022, .022):
            ax.plot((i, i + 1), (shift, shift), color=".12", lw=.6, zorder=1)
    for i, j, radius in ((0, 3, .28), (0, 4, -.27), (1, 5, .28),
                          (2, 6, -.28), (3, 7, .28), (4, 7, -.28)):
        ax.add_patch(FancyArrowPatch(
            (i, 0), (j, 0), arrowstyle="-", connectionstyle=f"arc3,rad={radius}",
            color=_BLUE, lw=.6, shrinkA=5.9, shrinkB=5.9, zorder=1,
        ))
    ax.scatter(range(8), np.zeros(8), s=140, c="#247db5", linewidths=0, zorder=3)
    for i in range(8):
        ax.text(i, 0, f"Q{i}", ha="center", va="center", fontsize=6.8, zorder=4)
    ax.set(xlim=(-.25, 7.25), ylim=(-.62, .62))
    ax.axis("off")
    legend = [Line2D([], [], marker="o", markersize=5.8, lw=0, color="#247db5"),
              Line2D([], [], lw=.85, color=".12"),
              Line2D([], [], lw=.6, color=_BLUE)]
    fig.legend(legend, ("Qubit", "Coupler", "Long-range interaction"),
               loc="center", bbox_to_anchor=(.285, .76), frameon=False,
               ncol=3, columnspacing=1.0, handletextpad=.35,
               handlelength=1.35, fontsize=6.8)
    fig.text(.027, .938, "(a)", fontsize=9.0, va="top")


def _network(fig, case, title, panel, y):
    from matplotlib.patches import FancyArrowPatch

    edges = {tuple(edge) for edge in case["edges"]}
    ax = fig.add_axes((.606, y, .363, .231))
    # Offset alternating rows to keep vertical dependencies visible.
    shifts = (.12, -.12, .12, -.12, .12)
    positions = {f"{prefix}_{i}": (i + shifts[row], 4 - row)
                 for row, prefix in enumerate(_ROWS) for i in range(8)}
    for source, target in sorted(edges):
        reverse = (target, source) in edges
        ax.add_patch(FancyArrowPatch(
            positions[source], positions[target], arrowstyle="->", mutation_scale=5.7,
            shrinkA=5.9, shrinkB=5.9, connectionstyle=f"arc3,rad={.12 if reverse else 0}",
            color=".08", lw=.48, alpha=.85, zorder=1,
        ))
    for row, (prefix, color) in enumerate(zip(_ROWS, _COLORS)):
        xx = np.arange(8) + shifts[row]
        ax.scatter(xx, np.full(8, 4 - row), s=98, c=color,
                   edgecolors=".12", linewidths=.48, zorder=4)
        for i, x in enumerate(xx):
            ax.text(x, 4 - row, str(i), ha="center", va="center",
                    fontsize=6.8, zorder=5)
    ax.set(xlim=(-.48, 7.48), ylim=(-.31, 4.31))
    ax.axis("off")
    fig.text(.579, y + .237, f"({panel}) {title}", fontsize=8.2, va="bottom")


def _variable_legend(fig):
    from matplotlib.lines import Line2D

    labels = (r"$\Phi$", r"$f$", r"$\omega_d$", r"$\widetilde F$", r"$\Omega$")
    handles = [Line2D([], [], marker="o", markersize=5.0, lw=0,
                      markerfacecolor=color, markeredgecolor=".1", markeredgewidth=.45)
               for color in _COLORS]
    legend = fig.legend(handles, labels, loc="center", bbox_to_anchor=(.786, .94),
                        frameon=False, ncol=5, fontsize=6.8, handlelength=.7,
                        handletextpad=.3, columnspacing=.9, borderaxespad=0)
    legend.get_texts()[3].set_color(_BLUE)


def _heatmap(fig, case, panel, statistic, x, y):
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    mi = np.asarray(case["mi_bits"])
    coupling = np.asarray(case["coupling"])
    ax = fig.add_axes((x, y, .17, .233333333333))
    if statistic == "mi":
        values = 100 * mi
        lower = min(0., float(values.min()))
        upper = max(float(values.max()), 1e-12)
        label = r"MI ($10^{-2}$ bits)"
    else:
        values = coupling
        lower, upper = 0., max(float(values.max()), 1e-12)
        label = r"$h_{ij}$ (code units)"
    img = ax.imshow(values, origin="lower", cmap="viridis", interpolation="nearest",
                    vmin=lower, vmax=upper, aspect="auto")
    ax.set_xticks((0, 2, 4, 6))
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6.8, length=2.1, width=.45, pad=1.7)
    for spine in ax.spines.values():
        spine.set_linewidth(.45)
    color_ax = fig.add_axes((x - .025, y, .009, .233333333333))
    bar = fig.colorbar(img, cax=color_ax)
    bar.ax.yaxis.set_ticks_position("left")
    bar.ax.yaxis.set_label_position("left")
    bar.ax.tick_params(labelsize=6.8, length=2., width=.45, pad=1.6)
    bar.locator = MaxNLocator(nbins=4)
    if statistic != "mi":
        bar.formatter = FormatStrFormatter("%.3f")
    bar.update_ticks()
    bar.set_label(label, fontsize=6.8, labelpad=3, color=_BLUE)
    bar.outline.set_linewidth(.45)
    fig.text(x - .065, y + .241, f"({panel})", fontsize=9.0, va="bottom")


def draw(cases, out_dir):
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    with rc_context({"font.family": "DejaVu Sans", "font.size": 6.8,
                     "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
                     "ps.fonttype": 42, "axes.unicode_minus": True}):
        fig = Figure(figsize=(7., 5.1), facecolor="white")
        FigureCanvasAgg(fig)
        _box(fig, (.014, .733, .535, .225))
        _box(fig, (.014, .025, .535, .69))
        _box(fig, (.569, .025, .415, .933))
        fig.text(.5, .986, "Pulse-level dependencies", ha="center", va="top", fontsize=9., color=_BLUE)
        _schematic(fig)
        _variable_legend(fig)
        for (name, title), panel, y in zip(_NETWORK_TITLES.items(), "bcd", (.677, .376, .075)):
            _network(fig, cases[name], title, panel, y)
        for name, letters, y in (("chain", "ef", .423), ("dense", "gh", .073)):
            _heatmap(fig, cases[name], letters[0], "mi", .102, y)
            _heatmap(fig, cases[name], letters[1], "coupling", .372, y)
        paths = {ext: out_dir / f"Fig4.{ext}" for ext in ("pdf", "png")}
        for ext, path in paths.items():
            fig.savefig(path, format=ext, dpi=300, metadata={"Creator": "Causal-Q", **({"CreationDate": None} if ext == "pdf" else {})})
