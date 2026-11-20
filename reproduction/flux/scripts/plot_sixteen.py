"""Figure 3. Flux dependencies on three sixteen-qubit coupling graphs."""
from pathlib import Path
import numpy as np

# Colors are anchored in GHz for the model couplings, and bits for marginal MI.
STYLES = {
    "nn4": ("Sparse", [0, .010, .014, .018, .020], .554),
    "nn8": ("Moderate", [0, .009, .014, .018, .020], .524),
    "dense": ("Dense", [0, .004, .006, .010, .020], .2065),
}

def draw_figure3(records, summaries, output):
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    colors = ['white', 'cornflowerblue', 'royalblue', 'mediumpurple', 'darkblue']
    mi_cmap = LinearSegmentedColormap.from_list('display_mi', colors, N=256)
    fig = Figure(figsize=(15.5, 12.5))
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(3, 3, left=0.062, right=0.985, bottom=0.069, top=0.95, wspace=0.37, hspace=0.3)
    coupling_label = '$h_{ij}$ (MHz)'
    (label_size, tick_size) = (15, 13)
    positions = {i: (i % 4, -(i // 4)) for i in range(16)}
    labels = 'abcdefghi'
    for row, (case, (label, h_anchors, mi_max)) in enumerate(STYLES.items()):
        record = records[case]
        coupling = np.asarray(record["coupling_hij"])
        mij = np.asarray(record["mij_bits"])
        mi = np.asarray(record["mi_bits"])
        row_label = f"{label} | {record['sample_count']:,} samples"
        for col in range(3):
            ax = fig.add_subplot(grid[row, col])
            ax.text(-0.15, 1.03, f'({labels[row * 3 + col]})', transform=ax.transAxes, fontsize=18, fontweight='bold')
            if col == 2:
                ii = np.triu_indices(16, 1)
                ax.scatter(np.abs(coupling[ii]) * 1000, mij[ii], s=17, alpha=0.8)
                rho = summaries[case]["coupled_spearman"]
                annotation = f'$r_s = {rho:.3f}$'
                score_k = 30
                annotation += '\n' + f'$k_\\mathrm{{CMI}} = {score_k}$'
                ax.text(0.06, 0.92, annotation, transform=ax.transAxes, fontsize=label_size, va='top')
                ax.set_xlabel(coupling_label, fontsize=label_size)
                ax.set_ylabel('$M_{ij}$ (bits)', fontsize=label_size)
                ax.xaxis.set_major_locator(MultipleLocator(2.5))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
                ax.tick_params(labelsize=tick_size)
                ax.grid(alpha=0.2)
                continue
            if col == 0:
                cmap = LinearSegmentedColormap.from_list('couplings', list(zip(np.asarray(h_anchors) / 0.02, colors)), N=256)
                norm = Normalize(0, 0.02)
                for i in range(16):
                    for j in range(i + 1, 16):
                        if coupling[i, j] != 0:
                            ax.plot(*zip(positions[i], positions[j]), c=cmap(norm(coupling[i, j])), lw=1.7, zorder=1)
            else:
                edges = record['displayed_edges']
                cmap, norm = mi_cmap, Normalize(0, mi_max)
                all_edges = set(map(tuple, record['inferred_edges']))
                for (i, j) in edges:
                    rad = 0.15 if (j, i) in all_edges else 0.0
                    ax.add_patch(FancyArrowPatch(positions[i], positions[j], arrowstyle='->', mutation_scale=15, connectionstyle=f'arc3,rad={rad}', shrinkA=9, shrinkB=9, color=cmap(norm(mi[i, j])), linewidth=1.8, zorder=1))
            xy = np.array(list(positions.values()))
            ax.scatter(xy[:, 0], xy[:, 1], s=225, c='lightblue', edgecolors='black' if col == 0 else 'none', lw=0.7, zorder=3)
            for (i, pos) in positions.items():
                ax.text(*pos, str(i), ha='center', va='center', fontsize=13, zorder=4)
            ax.set(xlim=(-0.28, 3.28), ylim=(-3.28, 0.28), aspect='equal')
            ax.axis('off')
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, location='left', fraction=0.044, pad=0.14)
            cb.ax.tick_params(labelsize=12)
            if col == 0:
                cb.set_label(coupling_label, fontsize=14)
                cb.formatter = FuncFormatter(lambda x, _: f'{x * 1000:.1f}')
                cb.update_ticks()
            else:
                cb.set_label('MI (bits)', fontsize=14)
                ax.set_title(row_label, fontsize=14, pad=12)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with rc_context({'pdf.fonttype': 42}):
        for suffix in ('pdf', 'png'):
            fig.savefig(output / f'figure3_recomputed.{suffix}', dpi=170)
