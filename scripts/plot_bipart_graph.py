#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize, LinearSegmentedColormap, PowerNorm

import ipywidgets as widgets
from IPython.display import display, clear_output


# =========================
#   Global Figure (use display(fig), not display(fig.canvas))
# =========================
fig = plt.figure(figsize=(10, 8))


# =========================
#   Locate latest mi_results file
# =========================
def find_latest_mi_file():
    cand = sorted(glob.glob("mi_results_*.npz")) + sorted(glob.glob("mi_results_*.json"))
    if not cand:
        raise FileNotFoundError("No mi_results_*.npz or mi_results_*.json in current directory")
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]


def load_mi_results(path):
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        mi_results = []
        for i in range(len(z["mi"])):
            mi_results.append({
                "src_idx": int(z["src_idx"][i]),
                "dst_idx": int(z["dst_idx"][i]),
                "MI": float(z["mi"][i]),
            })
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        return mi_results, meta
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("mi_results", []), payload.get("meta", {})
    else:
        raise ValueError("Expected an .npz or .json file")


def infer_num_qubits(mi_results):
    if not mi_results:
        return 0
    return max(
        max(int(it["src_idx"]) for it in mi_results),
        max(int(it["dst_idx"]) for it in mi_results),
    ) + 1


# =========================
#   Adjustable piecewise colormap
# =========================
def _make_strictly_increasing(arr, eps=1e-9):
    arr = list(arr)
    for i in range(1, len(arr)):
        if arr[i] <= arr[i - 1]:
            arr[i] = arr[i - 1] + eps
    return arr


def get_custom_palette(vmin, vmin_mid, mid, mid_vmax, vmax):
    """
    Piecewise colormap: small MI -> light, large MI -> dark.
    vmin_mid/mid/mid_vmax control transition speed.
    """
    colors = ["white", "cornflowerblue", "royalblue", "mediumpurple", "darkblue"]

    vmin, vmin_mid, mid, mid_vmax, vmax = _make_strictly_increasing(
        [vmin, vmin_mid, mid, mid_vmax, vmax], eps=1e-9
    )
    denom = (vmax - vmin) if (vmax - vmin) != 0 else 1e-12

    p1 = (vmin_mid - vmin) / denom
    p2 = (mid - vmin) / denom
    p3 = (mid_vmax - vmin) / denom

    positions = _make_strictly_increasing([0.0, p1, p2, p3, 1.0], eps=1e-6)
    return LinearSegmentedColormap.from_list("custom_mi", list(zip(positions, colors)))


# =========================
#   Draw bipartite graph (small MI -> light color)
# =========================
def plot_bipartite(mi_results,
                   threshold=0.01,
                   vmin=0.0, vmin_mid=0.1, mid=0.44, mid_vmax=0.66, vmax=1.0,
                   color_gamma=1.0,
                   node_phi=2000, node_f=2000,
                   font_size=25, edge_width=4.0, arrowsize=18,
                   x_left=-1.5, x_right=1.5, y_gap=1.2):
    fig.clf()
    ax = fig.add_subplot(111)

    n = infer_num_qubits(mi_results)
    if n == 0:
        ax.text(0.5, 0.5, "mi_results empty", ha="center", va="center")
        ax.axis("off")
        return

    flux_nodes = [f"flux{i}" for i in range(n)]
    freq_nodes = [f"omega_q{i}" for i in range(n)]

    # layout with padding
    pos = {f"flux{i}": (x_left,  i * y_gap) for i in range(n)}
    pos.update({f"omega_q{i}": (x_right, i * y_gap) for i in range(n)})

    G = nx.DiGraph()
    G.add_nodes_from(flux_nodes)
    G.add_nodes_from(freq_nodes)

    edges, weights = [], []
    for it in mi_results:
        si = int(it["src_idx"])
        di = int(it["dst_idx"])
        mi = float(it["MI"])
        if mi > threshold:
            u = f"flux{si}"
            v = f"omega_q{di}"
            G.add_edge(u, v, weight=mi)
            edges.append((u, v))
            weights.append(mi)

    # draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=flux_nodes,
                           node_size=node_phi, node_color="lightblue", edgecolors="black")
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=freq_nodes,
                           node_size=node_f, node_color="lightgreen", edgecolors="black")

    # latex labels
    labels = {f"flux{i}": rf"$\Phi_{{{i}}}$" for i in range(n)}
    labels.update({f"omega_q{i}": rf"$f_{{{i}}}$" for i in range(n)})
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=font_size)

    if len(edges) == 0:
        ax.set_title(f"No edges above threshold={threshold}", fontsize=font_size)
        ax.axis("off")
        fig.tight_layout()
        return

    weights = np.array(weights, dtype=float)

    # Ensure a nonzero color range.
    if vmax == vmin:
        vmax = vmin + 1e-12

    if color_gamma == 1.0:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        # gamma>1 makes colors look more uniform (less contrast for small MI)
        norm = PowerNorm(gamma=color_gamma, vmin=vmin, vmax=vmax)
    cmap = get_custom_palette(vmin, vmin_mid, mid, mid_vmax, vmax)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=edges,
        edge_color=weights,
        edge_cmap=cmap,
        edge_vmin=norm.vmin,
        edge_vmax=norm.vmax,
        width=edge_width,
        arrows=True, arrowstyle="-|>", arrowsize=arrowsize,
        min_source_margin=20, min_target_margin=20
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="MI")
    cbar.ax.tick_params(labelsize=25)
    cbar.set_label("MI", fontsize=25)

    ax.axis("off")
    fig.tight_layout()


# =========================
#   Interactive plot controls
# =========================
# 1) load
# Set MI_PATH to a file or use None to load the latest result.
# MI_PATH = None
# MI_PATH = "mi_results_20260104_230204.npz"
MI_PATH = "mi_results_20260108_232543.npz"
mi_path = MI_PATH or find_latest_mi_file()
mi_results, meta = load_mi_results(mi_path)
print("[LOAD]", mi_path, "edges=", len(mi_results), "meta=", meta)

mi_vals = np.array([float(x["MI"]) for x in mi_results], dtype=float) if mi_results else np.array([1.0])
mi_max = float(mi_vals.max()) if mi_vals.size else 1.0

# 2) threshold and color sliders
thr_slider = widgets.FloatSlider(value=0.01, min=0.0, max=max(0.1, mi_max),
                                 step=max(0.1, mi_max)/200, description="thr")

vmin_slider = widgets.FloatSlider(value=0.0, min=0.0, max=max(1.0, mi_max),
                                  step=max(1.0, mi_max)/200, description="vmin")
vmin_mid_slider = widgets.FloatSlider(value=min(0.1, mi_max), min=0.0, max=max(1.0, mi_max),
                                      step=max(1.0, mi_max)/200, description="vmin_mid")
mid_slider = widgets.FloatSlider(value=min(0.44, mi_max), min=0.0, max=max(1.0, mi_max),
                                 step=max(1.0, mi_max)/200, description="mid")
mid_vmax_slider = widgets.FloatSlider(value=min(0.66, mi_max), min=0.0, max=max(1.0, mi_max),
                                      step=max(1.0, mi_max)/200, description="mid_vmax")
vmax_slider = widgets.FloatSlider(value=mi_max, min=0.0, max=max(1.0, mi_max),
                                  step=max(1.0, mi_max)/200, description="vmax")

# 3) figure output
out = widgets.Output()
display(out)

def _wrap(thr, vmin, vmin_mid, mid, mid_vmax, vmax):
    with out:
        clear_output(wait=True)
        plot_bipartite(
            mi_results,
            threshold=thr,
            vmin=vmin, vmin_mid=vmin_mid, mid=mid, mid_vmax=mid_vmax, vmax=vmax,
        )
        display(fig)  # Render the figure in the output widget.

widgets.interact(
    _wrap,
    thr=thr_slider,
    vmin=vmin_slider,
    vmin_mid=vmin_mid_slider,
    mid=mid_slider,
    mid_vmax=mid_vmax_slider,
    vmax=vmax_slider,
)

# 4) save the current figure
save_button = widgets.Button(description="Save PDF")

def on_save_clicked(b):
    with out:
        fig.canvas.draw()
        plt.pause(0.05)
        out_path = "bipartite_mi.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        print("[SAVE]", out_path)

save_button.on_click(on_save_clicked)
display(save_button)
