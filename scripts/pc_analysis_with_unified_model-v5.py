#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
layered_pc_analysis.py

Layered causal discovery:
    Layer 1: Phi_i (flux_i)  ->  omega_q_j
    Layer 2: omega_q_i       ->  F_k
    Layer 3: Omega_i         ->  F_k
    Layer 4: omega_d_i       ->  F_k   (if 'omega_d' exists in data)

Use includeF/run_layers to run only layer 1 or the full stack.
"""

import numpy as np
from itertools import combinations
import networkx as nx
from npeet import entropy_estimators as ee
from matplotlib import pyplot as plt

import json


# ============================================
#        互信息 / 条件互信息 工具函数
# ============================================

def MI(x, y):
    """
    I(x;y)，x,y 一维向量
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    return ee.mi(x, y)


def CMI_xy_z(x, y, Z):
    """
    I(x;y|Z)，Z:(n,d_z)
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    Z = np.asarray(Z)
    return ee.cmi(x, y, Z)


def independence_test(x, y, Z=None,
                      alpha=1e-3,
                      use_perm=False,
                      n_perm=30,
                      rng=None,k=10):
    """
    互信息/条件互信息独立性检验：
      - Z is None: 用 MI(x,y)
      - Z 不为空: 用 CMI(x,y|Z)
    use_perm=True 时，使用 permutation baseline 的 ΔI = I - E[I_perm]
    返回: (indep: bool, score: float)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)

    if Z is None:
        score = ee.mi(x, y,k=k) #Warning
        indep = (score <= alpha)
        return indep, score

    Z = np.asarray(Z)
    if not use_perm:
        val = ee.cmi(x, y, Z)
        indep = (val <= alpha)
        return indep, val

    # 带 permutation baseline 的版本
    base = ee.cmi(x, y, Z)
    perml = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        perml.append(ee.cmi(x, y_perm, Z))
    perml = np.asarray(perml)
    diff = base - perml.mean()
    indep = (diff <= alpha)
    return indep, diff


# ============================================
#          通用 PC 算法（连续变量 + MI/CMI）
# ============================================

def pc_generic(data_matrix,
               var_names,
               candidate_pairs=None,
               exogenous=None,
               alpha=1e-3,
               use_perm=False,
               n_perm=30,
               max_cond_set=2,
               verbose=False):
    """
    一个完全通用的 PC 实现：

    参数
    ----
    data_matrix : (n_samples, n_vars)
    var_names   : 长度为 p 的变量名列表，对应到 data_matrix 的列
    candidate_pairs : 可选，允许存在无向边的节点对集合
        - 类型：set of frozenset({a,b})
        - 如果为 None，则默认全连通图（完全图）
    exogenous   : 可选，外生变量集合（禁止有指向它们的箭头）
    alpha       : 独立性阈值，score <= alpha 视为独立
    use_perm    : 条件 MI 时是否用 permutation baseline
    n_perm      : permutation 次数
    max_cond_set: 条件集合最大阶数

    返回
    ----
    P : dict，P[u] 是 u -> child 的集合（有向边）
    G : dict，无向骨架图，G[u] 是与 u 相邻的节点集合（不含方向）
    sep_sets : dict，分离集合，键为 frozenset({a,b})，值为分离它们的条件集 S
    """
    X = np.asarray(data_matrix)
    n, p = X.shape
    assert len(var_names) == p
    idx = {v: i for i, v in enumerate(var_names)}

    # 允许边
    if candidate_pairs is None:
        allow = set()
        for i in range(p):
            for j in range(i + 1, p):
                allow.add(frozenset({var_names[i], var_names[j]}))
    else:
        allow = set(candidate_pairs)

    exogenous = set(exogenous) if exogenous is not None else set()

    # 初始化无向骨架图 G
    G = {v: set() for v in var_names}
    for e in allow:
        a, b = tuple(e)
        G[a].add(b)
        G[b].add(a)

    if verbose:
        print("[PC] initial skeleton:")
        for v in var_names:
            print(v, ":", sorted(G[v]))

    # 裁剪骨架: 逐阶增加条件集合大小
    sep_sets = {}
    L = 0
    changed = True
    rng = np.random.default_rng(0)

    while changed and L <= max_cond_set:
        changed = False
        # 为了遍历每条边一次，用一个 snapshot
        edges_snapshot = [
            (a, b)
            for a in var_names
            for b in list(G[a])
            if idx[a] < idx[b]
        ]

        if verbose:
            print(f"[PC] L = {L}, #edges = {len(edges_snapshot)}")

        for (a, b) in edges_snapshot:
            neighbors = [x for x in G[a] if x != b]
            if len(neighbors) < L:
                continue

            x = X[:, idx[a]]
            y = X[:, idx[b]]

            for S in combinations(neighbors, L):
                if L == 0:
                    Z = None
                else:
                    Z = np.column_stack([X[:, idx[s]] for s in S])

                indep, score = independence_test(
                    x, y, Z,
                    alpha=alpha,
                    use_perm=use_perm,
                    n_perm=n_perm,
                    rng=rng,
                    k = 30
                )

                if indep:
                    # 删除边 a—b
                    G[a].discard(b)
                    G[b].discard(a)
                    sep_sets[frozenset({a, b})] = set(S)
                    changed = True
                    # 测试
                    if (a, b) == ("Phi_0", "omega_4") or (a, b) == ("omega_4", "Phi_0"):
                        print(f"[PC-debug] removed Phi_0—omega_4 with S={S}, L={L}, score={score:.6g}")
                    if verbose:
                        print(f"[PC-prune] remove {a}—{b}, L={L}, score={score:.3e}, S={S}")
                    break
        L += 1

    if verbose:
        print("[PC] pruned skeleton:")
        for v in var_names:
            print(v, ":", sorted(G[v]))

    # ============================================
    #              定向阶段
    # ============================================

    P = {v: set() for v in var_names}  # P[u] = {children}

    # 先用外生性：把 exogenous -> 邻居 的方向固定
    for v in exogenous:
        if v not in G:
            continue
        for nei in G[v]:
            P[v].add(nei)

    # V 结构：x - z - y 且 x,y 不邻接，z 不在 sep(x,y) ⇒ x->z<-y
    for z in var_names:
        z_nei = list(G[z])
        for i in range(len(z_nei)):
            for j in range(i + 1, len(z_nei)):
                x, y = z_nei[i], z_nei[j]
                # x,y 邻接则不是 V 结构
                if (y in G[x]) or (x in G[y]):
                    continue
                Sxy = sep_sets.get(frozenset({x, y}), set())
                if z not in Sxy:
                    # 避免指向外生变量
                    if z not in exogenous:
                        P[x].add(z)
                        P[y].add(z)

    # Meek 规则 R1：x->y 且 y—z 无向，且 x,z 不邻接 ⇒ y->z
    changed = True
    while changed:
        changed = False
        for y in var_names:
            undirected_nei = [
                z for z in G[y]
                if (z not in P[y]) and (y not in P[z])
            ]
            if not undirected_nei:
                continue

            parents_y = [x for x in var_names if y in P[x]]
            if not parents_y:
                continue

            for z in undirected_nei:
                # 检查是否存在 x->y，且 x,z 不邻接
                for x in parents_y:
                    if (z not in G[x]) and (x not in G[z]):
                        if z not in exogenous:
                            P[y].add(z)
                            changed = True
                            break
                if changed:
                    break

    # 最后再保险：清理所有指向外生变量的箭头
    for v in exogenous:
        for u in var_names:
            if v in P[u]:
                P[u].discard(v)

    return P, G, sep_sets


def print_directed(P, title="Directed edges"):
    print("====", title, "====")
    edges = []
    for u in sorted(P.keys()):
        for v in sorted(P[u]):
            edges.append((u, v))
    for (u, v) in edges:
        print(f"{u} -> {v}")
    if not edges:
        print("(no directed edges)")


# ============================================
#            分层因果：各层封装
# ============================================

def layer_phi_to_omega(data,
                       alpha=1e-3,
                       use_perm=False,
                       n_perm=30,
                       max_cond_set=2,
                       verbose=False):
    """
    第 1 层：Phi_i (flux_i) -> omega_q_j
    数据字段：
      - data["flux"]    : (samples, N)
      - data["omega_q"] : (samples, N)
    """
    flux = np.asarray(data["flux"])
    omega_q = np.asarray(data["omega_q_clean"])
    n_samples, N = flux.shape

    X = np.concatenate([flux, omega_q], axis=1)
    var_names = [f"Phi_{i}" for i in range(N)] + [f"omega_{i}" for i in range(N)]

    # 只允许 Phi_i -- omega_j 的边
    candidate_pairs = set()
    for i in range(N):
        a = f"Phi_{i}"
        for j in range(N):
            b = f"omega_{j}"
            candidate_pairs.add(frozenset({a, b}))

    exogenous = [f"Phi_{i}" for i in range(N)]

    P, G, sep = pc_generic(
        X,
        var_names,
        candidate_pairs=candidate_pairs,
        exogenous=exogenous,
        alpha=alpha,
        use_perm=use_perm,
        n_perm=n_perm,
        max_cond_set=max_cond_set,
        verbose=verbose,
    )
    return P, G, sep

def layer_omegaq_to_omegad(data,
                          alpha=1e-3, 
                          use_perm=False,
                          n_perm=30,
                          max_cond_set=2,
                          verbose=False):
    """
    NOTE:
    第 2 层：omega_q_i -> omega_d_k, omega_q_i 是通过dyson计算出来的
    数据字段：
      - data["omega_q"] : (samples, N)
      - data["omega_d"] : (samples, N)
    """
    omega_q = np.asarray(data["omega_q_clean"])
    omega_d = np.asarray(data["omega_d_noisy"])
    n_samples, N = omega_q.shape

    X = np.concatenate([omega_q, omega_d], axis=1)
    var_names = [f"omega_q_{i}" for i in range(N)] + [f"omega_d_{k}" for k in range(N)]

    candidate_pairs = set()
    for i in range(N):
        a = f"omega_q_{i}"
        for k in range(N):
            b = f"omega_d_{k}"
            candidate_pairs.add(frozenset({a, b}))

    exogenous = [f"omega_q_{i}" for i in range(N)]

    P, G, sep = pc_generic(
        X,
        var_names,
        candidate_pairs=candidate_pairs,
        exogenous=exogenous,
        alpha=alpha,
        use_perm=use_perm,
        n_perm=n_perm,
        max_cond_set=max_cond_set,
        verbose=verbose,
    )
    return P, G, sep

def layer_omega_to_F(data,
                     alpha=1e-3,
                     use_perm=False,
                     n_perm=30,
                     max_cond_set=2,
                     verbose=False):
    """
    NOTE:
    第 2 层：omega_q_i -> F_k,omega_q_i 是通过dyson计算出来的
    数据字段：
      - data["omega_q"] : (samples, N)
      - data["F"]       : (samples, N)
    """
    omega_q = np.asarray(data["omega_q_clean"])
    F = np.asarray(data["F"])
    n_samples, N = omega_q.shape

    X = np.concatenate([omega_q, F], axis=1)
    var_names = [f"omega_{i}" for i in range(N)] + [f"F_{k}" for k in range(N)]

    candidate_pairs = set()
    for i in range(N):
        a = f"omega_{i}"
        for k in range(N):
            b = f"F_{k}"
            candidate_pairs.add(frozenset({a, b}))

    exogenous = [f"omega_{i}" for i in range(N)]

    P, G, sep = pc_generic(
        X,
        var_names,
        candidate_pairs=candidate_pairs,
        exogenous=exogenous,
        alpha=alpha,
        use_perm=use_perm,
        n_perm=n_perm,
        max_cond_set=max_cond_set,
        verbose=verbose,
    )
    return P, G, sep


def layer_Omega_to_F(data,
                     alpha=1e-3,
                     use_perm=False,
                     n_perm=30,
                     max_cond_set=2,
                     verbose=False):
    """
    第 3 层：Omega_i -> F_k
    数据字段：
      - data["Omega_real"] : (samples, N)
      - data["F"]          : (samples, N)
    """
    Omega = np.asarray(data["Omega_real"])
    F = np.asarray(data["F"])
    n_samples, N = Omega.shape

    X = np.concatenate([Omega, F], axis=1)
    var_names = [f"Omega_{i}" for i in range(N)] + [f"F_{k}" for k in range(N)]

    candidate_pairs = set()
    for i in range(N):
        a = f"Omega_{i}"
        for k in range(N):
            b = f"F_{k}"
            candidate_pairs.add(frozenset({a, b}))

    exogenous = [f"Omega_{i}" for i in range(N)]

    P, G, sep = pc_generic(
        X,
        var_names,
        candidate_pairs=candidate_pairs,
        exogenous=exogenous,
        alpha=alpha,
        use_perm=use_perm,
        n_perm=n_perm,
        max_cond_set=max_cond_set,
        verbose=verbose,
    )
    return P, G, sep


def layer_omega_d_to_F(data,
                       alpha=1e-3,
                       use_perm=False,
                       n_perm=30,
                       max_cond_set=2,
                       verbose=False):
    """
    第 4 层：omega_d_i -> F_k

    这里假设数据中存在：
      - data["omega_d"] : (samples, N)  驱动频率
      - data["F"]       : (samples, N)

    如果没有 'omega_d' 这个 key，会自动跳过（返回空图）。
    """

    omega_d = np.asarray(data["omega_d_noisy"])
    F = np.asarray(data["F"])
    n_samples, N = omega_d.shape

    X = np.concatenate([omega_d, F], axis=1)
    var_names = [f"omega_d_{i}" for i in range(N)] + [f"F_{k}" for k in range(N)]

    candidate_pairs = set()
    for i in range(N):
        a = f"omega_d_{i}"
        for k in range(N):
            b = f"F_{k}"
            candidate_pairs.add(frozenset({a, b}))

    exogenous = [f"omega_d_{i}" for i in range(N)]

    P, G, sep = pc_generic(
        X,
        var_names,
        candidate_pairs=candidate_pairs,
        exogenous=exogenous,
        alpha=alpha,
        use_perm=use_perm,
        n_perm=n_perm,
        max_cond_set=max_cond_set,
        verbose=verbose,
    )
    return P, G, sep


# ============================================
#           组合多层图的工具函数
# ============================================

def merge_graphs(*Ps):
    """
    把多层得到的 P 字典合并成一个总的有向图。
    """
    P_total = {}
    for P in Ps:
        if not P:
            continue
        for u, children in P.items():
            if not children:
                continue
            if u not in P_total:
                P_total[u] = set()
            P_total[u].update(children)
    return P_total

def merge_skeletons(*Gs):
    """
    把多层得到的无向骨架 G 合并成一个总骨架（无向邻接表 dict）。
    每个 G 的格式：G[u] = set(neighbors)
    """
    G_total = {}
    for G in Gs:
        if not G:
            continue
        for u, neis in G.items():
            if u not in G_total:
                G_total[u] = set()
            for v in neis:
                G_total[u].add(v)
                if v not in G_total:
                    G_total[v] = set()
                G_total[v].add(u)
    return G_total


def run_pc_pipeline(
        data,
        includeF=True,                 # 兼容旧用法
        run_layers=None,               # 新增：比如 ("P1",) 或 ("P1","P2")
        merge_mode="directed",         # "directed" 或 "skeleton"
        alpha1=1e-3,
        alpha2=1e-3,
        alpha3=1e-3,
        alpha4=1e-3,
        use_perm=False,
        n_perm=30,
        max_cond_set=2,
        verbose=True
    ):
    """
    run_layers 可选值：P1,P2,P3,P4
      P1: Phi -> omega_q
      P2: omega_q -> omega_d   （注意：你代码里 Layer2 目前是这个）
      P3: Omega -> F
      P4: omega_d -> F

    merge_mode:
      - "directed": 返回合并后的有向边 P_total
      - "skeleton": 返回合并后的无向骨架 G_total
    """

    # 兼容旧 includeF：如果用户不传 run_layers，则按 includeF 决定
    if run_layers is None:
        run_layers = ("P1",) if not includeF else ("P1", "P2", "P3", "P4")
    run_layers = tuple(run_layers)

    P_layers = {}
    G_layers = {}

    # ---------- P1 ----------
    if "P1" in run_layers:
        print("\n=== Layer 1: Phi -> omega_q ===")
        P1, G1, sep1 = layer_phi_to_omega(
            data,
            alpha=alpha1,
            use_perm=use_perm,
            n_perm=n_perm,
            max_cond_set=max_cond_set,
            verbose=verbose,
        )
        P_layers["P1"] = P1
        G_layers["P1"] = G1
        print_directed(P1, title="Layer 1: Phi -> omega_q")

    # ---------- P2 ----------
    if "P2" in run_layers:
        print("\n=== Layer 2: omega_q -> omega_d ===")
        P2, G2, sep2 = layer_omegaq_to_omegad(
            data,
            alpha=alpha2,
            use_perm=use_perm,
            n_perm=n_perm,
            max_cond_set=max_cond_set,
            verbose=verbose,
        )
        P_layers["P2"] = P2
        G_layers["P2"] = G2
        print_directed(P2, title="Layer 2: omega_q -> omega_d")

    # ---------- P3 ----------
    if "P3" in run_layers:
        print("\n=== Layer 3: Omega -> F ===")
        P3, G3, sep3 = layer_Omega_to_F(
            data,
            alpha=alpha3,
            use_perm=use_perm,
            n_perm=n_perm,
            max_cond_set=max_cond_set,
            verbose=verbose,
        )
        P_layers["P3"] = P3
        G_layers["P3"] = G3
        print_directed(P3, title="Layer 3: Omega -> F")

    # ---------- P4 ----------
    if "P4" in run_layers:
        print("\n=== Layer 4: omega_d -> F ===")
        P4, G4, sep4 = layer_omega_d_to_F(
            data,
            alpha=alpha4,
            use_perm=use_perm,
            n_perm=n_perm,
            max_cond_set=max_cond_set,
            verbose=verbose,
        )
        P_layers["P4"] = P4
        G_layers["P4"] = G4
        print_directed(P4, title="Layer 4: omega_d -> F")

    # ---------- 合并 ----------
    if merge_mode == "skeleton":
        G_total = merge_skeletons(*(G_layers[k] for k in run_layers if k in G_layers))
        print("\n=== Final merged skeleton (undirected) ===")
        return G_total, {"P_layers": P_layers, "G_layers": G_layers}

    # 默认：合并有向图
    P_total = merge_graphs(*(P_layers[k] for k in run_layers if k in P_layers))
    print("\n=== Final merged causal graph ===")
    print_directed(P_total, title=f"Merged graph {run_layers}")
    return P_total, {"P_layers": P_layers, "G_layers": G_layers}


import re

def _idx_from_name(name):
    m = re.search(r'(\d+)$', name)
    return int(m.group(1)) if m else 0

def plot_layered_causal_graph(
        P_total,
        x_gap=1.5,
        y_gap=1.5,
        dx_Phi=0.0,
        dx_omega=0.0,
        dx_omegad=0.7,
        dx_F=0.5,
        dx_Omega=0.8,
        node_radius=0.22,
        shrink_pix=12
    ):
    """
    论文级分层 DAG，确保节点是圆形且比例一致，不会出现白图。
    """

    # -------- 收集节点 --------
    nodes = set(P_total.keys())
    for ch in P_total.values():
        nodes |= set(ch)

    # -------- 每类变量颜色 --------
    type_color = {
        "Phi": "skyblue",
        "omega": "lightgreen",
        "omega_d": "orange",
        "F": "khaki",
        "Omega": "lightcoral",
    }

    # -------- 层次结构 --------
    row_of_type = {
        "Phi": 0,
        "omega": 1,      # ⭐ omega 行放在 omega_d 上面
        "omega_d": 2,
        "F": 3,
        "Omega": 4,
    }

    def _node_type(n):
        if n.startswith("Phi"):
            return "Phi"
        if n.startswith("omega_d"):
            return "omega_d"
        if n.startswith("Omega_"):
            return "Omega"
        if n.startswith("F_"):
            return "F"
        if n.startswith("omega"):
            return "omega"   # ω_q
        return "other"

    # -------- 计算节点坐标 --------
    positions = {}
    max_idx = max(_idx_from_name(n) for n in nodes)

    for n in nodes:
        idx = _idx_from_name(n)
        t = _node_type(n)

        row = row_of_type[t]
        x = idx * x_gap
        y = -row * y_gap

        if t == "Phi":
            x += dx_Phi
        elif t == "omega":
            x += dx_omega
        elif t == "omega_d":
            x += dx_omegad
        elif t == "F":
            x += dx_F
        elif t == "Omega":
            x += dx_Omega

        positions[n] = (x, y)

    # -------- 新增：计算坐标范围 → 自动适配图像大小 --------
    xs = np.array([p[0] for p in positions.values()])
    ys = np.array([p[1] for p in positions.values()])
    xmin, xmax = xs.min() - 2, xs.max() + 2
    ymin, ymax = ys.min() - 2, ys.max() + 2

    width = max(12, (xmax - xmin) * 0.8)   # 动态宽度，最多屏幕安全范围
    height = max(6, (ymax - ymin) * 0.6)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")

    # -------- 画节点 --------
    for n, (x, y) in positions.items():
        t = _node_type(n)
        col = type_color.get(t, "white")

        circ = plt.Circle((x, y), node_radius, fc=col, ec="black")
        ax.add_patch(circ)

        ax.text(x, y, str(_idx_from_name(n)),
                ha="center", va="center", fontsize=11)

    # -------- 画箭头（从圆周到圆周）--------
    for src, children in P_total.items():
        x1, y1 = positions[src]
        for tgt in children:
            x2, y2 = positions[tgt]
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=1.3,
                    shrinkA=shrink_pix,
                    shrinkB=shrink_pix
                )
            )

    # -------- 图例 --------
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], marker='o', linestyle='None',
                             markerfacecolor=c, markeredgecolor="black",
                             markersize=8)
               for c in type_color.values()]

    # 使用 LaTeX 格式为图例标签
    labels = [r"$\Phi$", r"$f$", r"$\omega_d$", r"$F$", r"$\Omega$"]

    ax.legend(handles, labels, title="Variable type", loc="upper right")

    # -------- 设置坐标范围（核心修复点）--------
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # -------- 防止裁切 --------
    plt.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05)

    # -------- 强制比例一致，使节点为正圆形 --------
    ax.set_aspect('equal', adjustable='box')

    plt.show()

    return fig

# 假设你已经运行过因果推断过程并获得了 P_total
# 假设 P_total 已经被生成，你可以使用以下代码来保存为 GraphML 格式
# save_graph_as_graphml(P_total)

# 假设你已经运行过因果推断过程并获得了 P_total
# 假设 P_total 已经被生成，你可以使用以下代码来绘制因果图
# plot_causal_graph(P_total)

def plot_causal_graph(P_total):
    """
    绘制整体因果图
    P_total: dict，合并后的因果关系图（每个节点指向它的子节点）
    """
    G = nx.DiGraph()

    # 定义每种类型的变量及其对应颜色
    node_types = {
        "Phi": "skyblue",
        "omega_q": "lightgreen",
        "Omega": "lightcoral",
        "omega_d": "orange",
        "F": "lightyellow"
    }

    # 添加节点和边
    for u, children in P_total.items():
        for v in children:
            G.add_edge(u, v)

    # 画图：布局策略
    plt.figure(figsize=(12, 8))

    # 使用 spring_layout，增大 k 值可以增加节点之间的间距，避免重叠
    pos = nx.spring_layout(G, seed=42, k=1.5)  # 你可以试着调整 k 的值来减少重叠

    # 节点颜色和类型
    node_color = []
    for node in G.nodes():
        if "Phi" in node:
            node_color.append(node_types["Phi"])
        elif "omega_q" in node:
            node_color.append(node_types["omega_q"])
        elif "Omega" in node:
            node_color.append(node_types["Omega"])
        elif "omega_d" in node:
            node_color.append(node_types["omega_d"])
        else:
            node_color.append(node_types["F"])

    # 修改标签为比特数字
    labels = {node: ''.join(filter(str.isdigit, node)) for node in G.nodes()}  # 只保留数字

    # 绘制图，设置较小的节点
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=1000, node_color=node_color, font_size=10, font_weight='bold', edge_color='gray', arrows=True, edgecolors="black", width=1.5)

    # 添加图例
    handles = []
    labels_legend = []
    for node_type, color in node_types.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
        labels_legend.append(node_type)

    plt.legend(handles, labels_legend, title="Variable Types", loc="upper right", fontsize=12)

    # 添加标题
    plt.title("Causal Graph", fontsize=16)
    plt.show()

# 假设你已经运行过因果推断过程并获得了 P_total
# 假设 P_total 已经被生成，你可以使用以下代码来绘制因果图
# plot_causal_graph(P_total)

def plot_causal_graph(P_total):
    """
    绘制整体因果图
    P_total: dict，合并后的因果关系图（每个节点指向它的子节点）
    """
    G = nx.DiGraph()

    # 定义每种类型的变量及其对应颜色
    node_types = {
        "Phi": "skyblue",
        "omega_q": "lightgreen",
        "Omega": "lightcoral",
        "omega_d": "orange",
        "F": "lightyellow"
    }

    # 添加节点和边
    for u, children in P_total.items():
        for v in children:
            G.add_edge(u, v)

    # 画图：布局策略
    plt.figure(figsize=(12, 8))

    # 使用 spring_layout，增大 k 值可以增加节点之间的间距，避免重叠
    pos = nx.spring_layout(G, seed=42, k=1.5)  # 你可以试着调整 k 的值来减少重叠

    # 节点颜色和类型
    node_color = []
    for node in G.nodes():
        if "Phi" in node:
            node_color.append(node_types["Phi"])
        elif "omega_q" in node:
            node_color.append(node_types["omega_q"])
        elif "Omega" in node:
            node_color.append(node_types["Omega"])
        elif "omega_d" in node:
            node_color.append(node_types["omega_d"])
        else:
            node_color.append(node_types["F"])

    # 修改标签为比特数字
    labels = {node: ''.join(filter(str.isdigit, node)) for node in G.nodes()}  # 只保留数字

    # 绘制图，设置较小的节点
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=1000, node_color=node_color, font_size=10, font_weight='bold', edge_color='gray', arrows=True, edgecolors="black", width=1.5)

    # 添加图例
    handles = []
    labels_legend = []
    for node_type, color in node_types.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
        labels_legend.append(node_type)

    plt.legend(handles, labels_legend, title="Variable Types", loc="upper right", fontsize=12)

    # 添加标题
    plt.title("Causal Graph", fontsize=16)
    plt.show()


def save_P_json(P_total, path):
    """P_total: dict[str, set[str]] or dict[str, list[str]]"""
    obj = {k: sorted(list(v)) for k, v in P_total.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
# ============================================
#                   Demo 入口
# ============================================

if __name__ == "__main__":
    # 你可以在这里改成自己的数据文件
    # npz_path = "demo_dataset_3_3_gpu.npz" 
    # npz_path = "demo_dataset_gpu_1_10_chain_withF.npz"
    # npz_path = "nointerac_dataset_gpu_1_10_chain_withF.npz"
    # npz_path = "tttnointerac_dataset_gpu_1_10_chain_withF.npz"
    # npz_path = "tttnointerac_dataset_gpu_1_5_chain_withF.npz"
    # npz_path = "tttnointerac_dataset_gpu_1_5_chain_withF-2.npz"
    # npz_path = "nointerac_dataset_gpu_1_10_chain_withF.npz" #图a
    # npz_path = "nointerac_dataset_gpu_1_5_chain_withF.npz"
    # npz_path = "interac_dataset_gpu_1_8_chain_withF.npz"
    # npz_path = "interac_dataset_gpu_1_8_chain_withF-v2.npz" # 图b
    # npz_path = "fullinterac_dataset_gpu_2_4_dense_withF.npz" # 图c
    # npz_path = "fullinterac_dataset_gpu_4_4_nn8_withoutF.npz" # 用来做2*2 的论文图
    # npz_path = "fullinterac_dataset_gpu_2_2_withoutF.npz"  
    # npz_path = "fullinterac_dataset_gpu_4_4_nn8_withoutF.npz"
    npz_path = "fullinterac_dataset_gpu_4_4_dense_withoutF.npz"
    # includeF = False 只做 Phi -> omega_q
    # includeF = True  做全部分层 (omega_q->F, Omega->F, omega_d->F)
    # includeF = True

    with np.load(npz_path, allow_pickle=True) as f:
        data = {k: f[k] for k in f.files}

    # 这里 alpha 建议先设得比较小（甚至 0.0）看看骨架情况，再慢慢调大
    P_total, pack = run_pc_pipeline(
        data,
        run_layers=("P1",),
        merge_mode="directed",
        alpha1=0.00,
        max_cond_set=2,
        use_perm=False,
    )
    plot_causal_graph(P_total)
    fig = plot_layered_causal_graph(P_total)

    plot_causal_graph(P_total)
    # save_P_json(P_total, "P1_4*4_nn4_graph.json")
    # save_P_json(P_total, "P1_4*4_nn8_graph.json")
    save_P_json(P_total, "P1_4*4_dense_graph.json")
    # fig=plot_layered_causal_graph(
    #     P_total,
    #     x_gap=1.5,
    #     y_gap=1.5,
    #     dx_Phi=0.0,
    #     dx_omega=-0.5,
    #     dx_omegad=0.0,
    #     dx_F=-0.5,
    #     dx_Omega=-0.0,
    #     node_radius=0.22,
    #     shrink_pix=12
    # )
    # # fig.savefig("layered_causal_graph2.svg")
