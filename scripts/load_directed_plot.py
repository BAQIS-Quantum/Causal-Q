#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from npeet import entropy_estimators as ee
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr
import os
import time
import math


# ====== Widgets (same style as analysis_v5.py) =====
import ipywidgets as widgets
from IPython.display import display


# =========================
#   Parse variable names & fetch columns
# =========================
def parse_var(varname: str):
    # "Phi_3" -> ("Phi", 3) ; "omega_2" -> ("omega", 2)
    p, idx = varname.rsplit("_", 1)
    return p, int(idx)

def get_1d_from_data(data: dict, varname: str, var_map: dict):
    """
    Given names like \"Phi_3\" or \"omega_2\", pull the corresponding column from data
    and return shape (S,1). var_map maps prefix to matrix key, e.g. {\"Phi\":\"flux\"}.
    data[key] must be shape (S, N).
    """
    prefix, i = parse_var(varname)
    if prefix not in var_map:
        raise KeyError(f"prefix {prefix} missing in var_map (e.g. {prefix}: <data_key>)")
    key = var_map[prefix]
    if key not in data:
        raise KeyError(f"data missing key={key}, available keys={list(data.keys())[:50]} ...")
    mat = np.asarray(data[key])
    if mat.ndim != 2:
        raise ValueError(f"data[{key}] is not 2D, shape={mat.shape}")
    if i < 0 or i >= mat.shape[1]:
        raise IndexError(f"{varname} index {i} exceeds data[{key}] columns {mat.shape[1]}")
    return mat[:, i].reshape(-1, 1)  # (S,1)


# =========================
#   有向图 P_total 的加载
# =========================
def load_P_json(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return {k: set(v) for k, v in obj.items()}

# =========================
#   只对图中边计算 MI
# =========================
def compute_flux_f_mi_on_edges(P_total, data, var_map, knn=3):
    """
    返回 mi_results: list[dict]，每条边一个 MI
    dict: {src, dst, src_idx, dst_idx, MI}
    """
    mi_results = []
    for src, dsts in P_total.items():
        # NOTE example : src="Phi_0"，dsts={"omega_0","omega_1"}
        for dst in dsts:
            x = get_1d_from_data(data, src, var_map)
            y = get_1d_from_data(data, dst, var_map)
            mi = float(ee.mi(x, y, k=knn))
            _, si = parse_var(src)
            _, di = parse_var(dst)
            mi_results.append({
                "src": src, "dst": dst,
                "src_idx": si, "dst_idx": di,
                "MI": mi
            })
    return mi_results

def save_mi_results_json(mi_results, path="mi_results.json", meta=None):
    """
    保存为 JSON：便于查看、跨语言
    meta: dict 可选，写入一些信息（knn、npz_path、P_json_path、width/length 等）
    """
    payload = {
        "meta": meta or {},
        "mi_results": mi_results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] mi_results -> {path} (edges={len(mi_results)})")

def load_mi_results_json(path="mi_results.json"):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("mi_results", []), payload.get("meta", {})

# def save_mi_results_npz(mi_results, path="mi_results.npz", meta=None):
#     """
#     保存为 NPZ：加载快、体积小
#     """
#     src_idx = np.array([int(x["src_idx"]) for x in mi_results], dtype=int)
#     dst_idx = np.array([int(x["dst_idx"]) for x in mi_results], dtype=int)
#     mi      = np.array([float(x["MI"]) for x in mi_results], dtype=float)
#     src     = np.array([x["src"] for x in mi_results], dtype=object)
#     dst     = np.array([x["dst"] for x in mi_results], dtype=object)

#     np.savez_compressed(
#         path,
#         src_idx=src_idx,
#         dst_idx=dst_idx,
#         mi=mi,
#         src=src,
#         dst=dst,
#         meta=json.dumps(meta or {}, ensure_ascii=False),
#     )
    # print(f"[SAVE] mi_results -> {path} (edges={len(mi_results)})")

def save_mi_results_npz(mi_results,M_phi_omega=None,path="mi_results.npz", meta=None):
    """
    保存为 NPZ：加载快、体积小
    可选把 M_ij 一起存进去（key="Mij"）
    """
    src_idx = np.array([int(x["src_idx"]) for x in mi_results], dtype=int)
    dst_idx = np.array([int(x["dst_idx"]) for x in mi_results], dtype=int)
    mi      = np.array([float(x["MI"]) for x in mi_results], dtype=float)
    src     = np.array([x["src"] for x in mi_results], dtype=object)
    dst     = np.array([x["dst"] for x in mi_results], dtype=object)

    payload = dict(
        src_idx=src_idx,
        dst_idx=dst_idx,
        mi=mi,
        src=src,
        dst=dst,
        meta=json.dumps(meta or {}, ensure_ascii=False),
    )

    if M_phi_omega is not None:
        payload["Mij_phi_omega"] = np.asarray(M_phi_omega, dtype=float)
    np.savez_compressed(path, **payload)
    print(f"[SAVE] mi_results -> {path} (edges={len(mi_results)})"
          + (f", Mij_phi_omega={payload['Mij_phi_omega'].shape}" if "Mij_phi_omega" in payload else ""))

def load_mi_results_npz(path="mi_results.npz"):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}

    mi_results = []
    for i in range(len(z["mi"])):
        mi_results.append({
            "src": str(z["src"][i]),
            "dst": str(z["dst"][i]),
            "src_idx": int(z["src_idx"][i]),
            "dst_idx": int(z["dst_idx"][i]),
            "MI": float(z["mi"][i]),
        })

    M = z["Mij_phi_omega"] if "Mij_phi_omega" in z.files else None
    return mi_results, M,meta

# def load_mi_results_npz(path="mi_results.npz"):
#     z = np.load(path, allow_pickle=True)
#     meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
#     mi_results = []
#     for i in range(len(z["mi"])):
#         mi_results.append({
#             "src": str(z["src"][i]),
#             "dst": str(z["dst"][i]),
#             "src_idx": int(z["src_idx"][i]),
#             "dst_idx": int(z["dst_idx"][i]),
#             "MI": float(z["mi"][i]),
#         })
#     return mi_results, meta

# =========================
#   颜色映射（保持你原逻辑 + 加安全兜底）
# =========================
def _make_strictly_increasing(arr, eps=1e-9):
    arr = list(arr)
    for i in range(1, len(arr)):
        if arr[i] <= arr[i - 1]:
            arr[i] = arr[i - 1] + eps
    return arr


def get_custom_palette(vmin, vmin_mid, mid, mid_vmax, vmax):
    """
    按 analysis_v5.py 的分段逻辑生成 colormap，但做了“位置单调”兜底，避免报错。
    :contentReference[oaicite:1]{index=1}
    """
    colors = ["lightgray", "cornflowerblue", "royalblue", "darkblue", "mediumpurple"]

    # 兜底：避免除零或区间反转导致 positions 不递增
    vmin, vmin_mid, mid, mid_vmax, vmax = _make_strictly_increasing([vmin, vmin_mid, mid, mid_vmax, vmax])

    vmin_mid_normalized = (vmin_mid - vmin) / (mid - vmin)
    mid_vmax_normalized = (mid_vmax - mid) / (vmax - mid)

    positions = [
        0.0,
        np.clip(vmin_mid_normalized * 0.6, 0.0, 1.0),
        np.clip(vmin_mid_normalized * 0.8, 0.0, 1.0),
        np.clip(mid_vmax_normalized,       0.0, 1.0),
        1.0,
    ]
    # 再兜底一次：强制递增
    positions = _make_strictly_increasing(positions, eps=1e-6)

    return LinearSegmentedColormap.from_list("custom", list(zip(positions, colors)))

# 更平滑的连续色图（保持原来的浅->深蓝紫风格）
USE_SMOOTH_CMAP = True
SMOOTH_CMAP_COLORS = ["white", "cornflowerblue", "royalblue", "mediumpurple", "darkblue"]

def get_smooth_palette():
    return LinearSegmentedColormap.from_list("custom_smooth", SMOOTH_CMAP_COLORS, N=256)

# =========================
#   网格布局绘图：你的原逻辑不变
#   只把默认参数改成 analysis_v5.py 那套：
#     node_size=400, font_size=18, arrowsize=20, edge_width=3.0
# =========================
def plot_grid_with_mi(mi_results, width=None, length=None,
                      threshold=0.01,
                      vmin=0, vmin_mid=0.25, mid=0.5, mid_vmax=0.75, vmax=1.0,
                      node_size=400, font_size=18, arrowsize=20, edge_width=3.0,
                      RAD_BIDIR=0.12, RAD_SINGLE=0.0,
                      source_margin=12, target_margin=12,
                      save_path=None):

    plt.clf()
    ax = plt.gca()

    # ---------- 1) 从 mi_results 自动推断需要的节点数 ----------
    if len(mi_results) == 0:
        # 没边时也至少画个空图
        N = (width or 1) * (length or 1)
    else:
        max_idx = 0
        for it in mi_results:
            max_idx = max(max_idx, int(it["src_idx"]), int(it["dst_idx"]))
        N = max_idx + 1

    # ---------- 2) width/length 兜底：如果没给或给错，就自动生成 ----------
    if width is None or length is None or width * length < N:
        width = int(math.ceil(math.sqrt(N)))
        length = int(math.ceil(N / width))

    # grid pos: i -> (col, -row)，保证 0..N-1 都有 pos
    pos = {i: (divmod(i, width)[1], -divmod(i, width)[0]) for i in range(N)}

    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    # ---- 先统计双向边 ----
    edge_set = set()
    for it in mi_results:
        si = int(it["src_idx"])
        di = int(it["dst_idx"])
        if si != di:
            edge_set.add((si, di))

    bidir_pairs = {
        tuple(sorted((u, v)))
        for (u, v) in edge_set
        if (v, u) in edge_set
    }

    edges = []
    edge_colors = []
    connectionstyles = []

    for item in mi_results:
        si = int(item["src_idx"])
        di = int(item["dst_idx"])
        if si == di:
            continue

        mi = float(item["MI"])
        if mi <= threshold:
            continue

        pair = tuple(sorted((si, di)))
        rad = RAD_BIDIR if pair in bidir_pairs else RAD_SINGLE

        edges.append((si, di))
        edge_colors.append(mi)
        connectionstyles.append(f"arc3,rad={rad}")
        G.add_edge(si, di)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = get_smooth_palette() if USE_SMOOTH_CMAP else get_custom_palette(vmin, vmin_mid, mid, mid_vmax, vmax)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color="lightblue")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size)

    for i, (u, v) in enumerate(edges):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            edge_color=[edge_colors[i]],
            edge_cmap=cmap,
            edge_vmin=vmin, edge_vmax=vmax,
            arrowsize=arrowsize,
            width=edge_width,
            connectionstyle=connectionstyles[i],
            arrowstyle="->",
            ax=ax,
            min_source_margin=source_margin,
            min_target_margin=target_margin,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="MI")
    cbar.set_label("MI", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.tick_params(labelsize=16)


    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    fig = ax.figure
    fig.canvas.draw_idle()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")


def compute_average_flux_f_cmi(data, flux_key="flux", omega_key="omega_q_clean", knn=3):
    """
    计算对称矩阵 M (N,N):
      M_ij = 1/2 * [ CMI(flux_i, flux_j | omega_q_i) + CMI(flux_i, flux_j | omega_q_j) ]
    其中 omega_q_i 指 omega_key 矩阵的第 i 列 (S,1)。

    返回:
      M: (N,N) numpy array, 对称, 对角为0
    """
    flux  = np.asarray(data[flux_key])      # (S, N)
    omega = np.asarray(data[omega_key])     # (S, N)
    if flux.ndim != 2 or omega.ndim != 2:
        raise ValueError(f"flux/omega must be 2D, got {flux.shape}, {omega.shape}")
    if flux.shape[0] != omega.shape[0] or flux.shape[1] != omega.shape[1]:
        raise ValueError(f"flux and omega must have same shape (S,N), got {flux.shape}, {omega.shape}")

    S, N = flux.shape
    M = np.zeros((N, N), dtype=float)

    # 只算上三角，再对称化
    for i in range(N):
        Xi = flux[:, [i]]  # (S,1)
        Zi = omega[:, [i]] # (S,1)
        for j in range(i + 1, N):
            Xj = flux[:, [j]]
            Zj = omega[:, [j]]

            # NPEET: ee.cmi(X, Y, Z, k=knn)
            cmi_i = float(ee.cmi(Xi, Xj, Zi, k=knn))
            cmi_j = float(ee.cmi(Xi, Xj, Zj, k=knn))

            mij = 0.5 * (cmi_i + cmi_j)
            M[i, j] = mij
            M[j, i] = mij

    return M

def scatter_hij_vs_Mij(hij, M, use_abs_hij=True, title="scatter(hij, Mij)",save_path=None):
    hij = np.asarray(hij)
    M   = np.asarray(M)
    if hij.shape != M.shape or hij.ndim != 2 or hij.shape[0] != hij.shape[1]:
        raise ValueError(f"hij and M must be same square shape, got hij={hij.shape}, M={M.shape}")

    N = hij.shape[0]
    iu = np.triu_indices(N, k=1)   # i<j

    x = hij[iu].astype(float)
    y = M[iu].astype(float)
    if use_abs_hij:
        x = np.abs(x)
    x = x * 1000.0
    # 相关系数时去掉 hij 为 0 的点，作图仍保留
    mask = x != 0
    x_corr = x[mask]
    y_corr = y[mask]

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=28, alpha=0.8)

    r_s, _ = spearmanr(x_corr, y_corr)
    r_p = np.corrcoef(x_corr, y_corr)[0, 1]

    plt.xlabel(r"$h_{ij}~(\mathrm{MHz})$" if use_abs_hij else r"$h_{ij}~(\mathrm{MHz})$", fontsize=18)
    plt.ylabel(r"$M_{ij}$", fontsize=18)
    # plt.title(title, fontsize=18)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
    ax.text(0.02, 0.98, f"$r_s = {r_s:.3f}$\n$r = {r_p:.3f}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def corr_hij_Mij(hij, M, use_abs_hij=True):
    hij = np.asarray(hij)
    M = np.asarray(M)
    if hij.shape != M.shape or hij.ndim != 2:
        raise ValueError("hij 和 M 需要同形状的方阵")

    N = hij.shape[0]
    iu = np.triu_indices(N, k=1)
    x = hij[iu].astype(float)
    y = M[iu].astype(float)
    if use_abs_hij:
        x = np.abs(x)

    # Pearson 相关系数
    r = np.corrcoef(x, y)[0, 1]
    return r
            
# =========================
#              main
# =========================
# ✅ 参照 analysis_v5.py：全局 Figure + display(fig.canvas) + widgets.interact
# :contentReference[oaicite:2]{index=2}
fig = plt.figure(figsize=(8, 8))

if __name__ == "__main__":
    # 1) 载入图（P1）
    # P_total = load_P_json("P1_4*4_graph.json")
    # P_total = load_P_json("P1_2*2_graph.json")
    # P_total = load_P_json("P1_4*4_nn8_graph.json")
    P_total = load_P_json("P1_4*4_dense_graph.json")
    # 2) 载入数据
    # npz_path = "fullinterac_dataset_gpu_4_4_withoutF.npz"
    # npz_path = "fullinterac_dataset_gpu_2_2_withoutF.npz"
    # npz_path = "fullinterac_dataset_gpu_4_4_nn8_withoutF.npz"
    npz_path = "fullinterac_dataset_gpu_4_4_dense_withoutF.npz"
    z = np.load(npz_path, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    print("[INFO] data keys (first 30):", list(data.keys())[:30])

    # 3) 映射：Phi_? 用 data["flux"]，omega_? 用 data["omega_q_clean"]
    var_map = {
        "Phi": "flux",
        "omega": "omega_q_clean",
    }

    # 4) 只对图中的边计算 MI
    mi_results = compute_flux_f_mi_on_edges(P_total, data, var_map, knn=3)
    print("[INFO] #edges:", len(mi_results))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    meta = {
        "P_json": "P1_graph.json",
        "npz_path": npz_path,
        "var_map": var_map,
        "knn": 3,
        "width": 2,
        "length": 2,
    }
    # save_mi_results_json(mi_results, path=f"mi_results_{stamp}.json", meta=meta)
    # save_mi_results_npz(mi_results, path=f"mi_results_{stamp}.npz", meta=meta)

    if len(mi_results) > 0:
        mi_vals = np.array([x["MI"] for x in mi_results], dtype=float)
        mi_max = float(mi_vals.max())
    else:
        mi_max = 1.0

    # ===== 交互控件（照 analysis_v5.py 的写法）=====
    # 颜色断点 slider：范围按 MI 最大值自动设置（analysis_v5.py 是 0~1，这里更通用）
    vmin_slider = widgets.FloatSlider(value=0.0, min=0.0, max=max(1.0, mi_max), step=max(1.0, mi_max)/200, description="vmin")
    vmin_mid_slider = widgets.FloatSlider(value=0.1, min=0.0, max=max(1.0, mi_max), step=max(1.0, mi_max)/200, description="vmin_mid")
    mid_slider  = widgets.FloatSlider(value=0.44, min=0.0, max=max(1.0, mi_max), step=max(1.0, mi_max)/200, description="mid")
    mid_vmax_slider = widgets.FloatSlider(value=0.66, min=0.0, max=max(1.0, mi_max), step=max(1.0, mi_max)/200, description="mid_vmax")
    vmax_slider = widgets.FloatSlider(value=min(1.0, mi_max) if mi_max <= 1.0 else mi_max, min=0.0, max=max(1.0, mi_max), step=max(1.0, mi_max)/200, description="vmax")

    # 你要“点大小/线粗细”可控（analysis_v5.py 里是固定 node_size=400, font=18, width=3.0；这里默认同它）
    node_size_slider = widgets.IntSlider(value=400, min=100, max=2000, step=50, description="node")
    font_size_slider = widgets.IntSlider(value=18, min=8, max=30, step=1, description="font")
    edge_width_slider = widgets.FloatSlider(value=3.0, min=0.5, max=10.0, step=0.1, description="lw")
    arrowsize_slider = widgets.IntSlider(value=20, min=6, max=40, step=1, description="arrow")

    # 其他常用的也给你（可留可删）
    threshold_slider = widgets.FloatSlider(value=0.01, min=0.0, max=max(0.1, mi_max), step=max(0.1, mi_max)/200, description="thr")
    rad_bidir_slider = widgets.FloatSlider(value=0.15, min=0.0, max=1.0, step=0.01, description="rad2")
    rad_single_slider = widgets.FloatSlider(value=0.0, min=-1.0, max=1.0, step=0.01, description="rad1")
    src_margin_slider = widgets.IntSlider(value=12, min=0, max=40, step=1, description="m_src")
    tgt_margin_slider = widgets.IntSlider(value=12, min=0, max=40, step=1, description="m_tgt")

    def _plot_wrapper(vmin, vmin_mid, mid, mid_vmax, vmax,
                      node_size, font_size, edge_width, arrowsize,
                      threshold, RAD_BIDIR, RAD_SINGLE, source_margin, target_margin):
        # 保持你原 plot_grid_with_mi 的逻辑：只把参数从 slider 注入
        plot_grid_with_mi(
            mi_results,
            width=2, length=2,
            threshold=threshold,
            vmin=vmin, vmin_mid=vmin_mid, mid=mid, mid_vmax=mid_vmax, vmax=vmax,
            node_size=node_size, font_size=font_size,
            arrowsize=arrowsize, edge_width=edge_width,
            RAD_BIDIR=RAD_BIDIR, RAD_SINGLE=RAD_SINGLE,
            source_margin=source_margin, target_margin=target_margin,
            save_path=f"grid_mi_{stamp}.pdf"
        )

    widgets.interact(
        _plot_wrapper,
        vmin=vmin_slider,
        vmin_mid=vmin_mid_slider,
        mid=mid_slider,
        mid_vmax=mid_vmax_slider,
        vmax=vmax_slider,
        node_size=node_size_slider,
        font_size=font_size_slider,
        edge_width=edge_width_slider,
        arrowsize=arrowsize_slider,
        threshold=threshold_slider,
        RAD_BIDIR=rad_bidir_slider,
        RAD_SINGLE=rad_single_slider,
        source_margin=src_margin_slider,
        target_margin=tgt_margin_slider,
    )

    display(fig.canvas)

    M = compute_average_flux_f_cmi(data,knn=30)

    save_mi_results_npz(mi_results, M,path=f"mi_results_4*4_dense_{stamp}.npz", meta=meta)

    scatter_hij_vs_Mij(data['hij'],M,save_path='scatter_4*4_nn8_mij_hij.pdf')
