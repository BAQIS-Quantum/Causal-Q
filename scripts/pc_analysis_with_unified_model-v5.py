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
# Mutual information and conditional mutual information
# ============================================

def MI(x, y):
    """
    Estimate I(x;y) for two one-dimensional samples.
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    return ee.mi(x, y)


def CMI_xy_z(x, y, Z):
    """
    Estimate I(x;y|Z) for Z with shape (n_samples, n_covariates).
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
    Compare an information estimate with the independence threshold alpha.
      Use MI when Z is absent and CMI otherwise.
    For CMI, use_perm subtracts the mean from permutations of y.
    Return the independence decision and the score.
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

    # Center the CMI estimate on the permutation mean.
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
# PC-style search for continuous variables
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
    Estimate a skeleton and orient edges with exogeneity constraints, colliders, and Meek rule R1.

    Parameters
    ----------
    data_matrix : (n_samples, n_vars)
    var_names   : Variable names in column order.
    candidate_pairs : Optional set of permitted undirected edges.
        Each edge is a frozenset of its two endpoints.
        The default allows all pairs.
    exogenous   : Optional set of nodes with no incoming arrows.
    alpha       : Estimates at or below this threshold indicate independence.
    use_perm    : Subtract the permutation baseline when computing CMI.
    n_perm      : Number of permutations.
    max_cond_set: Maximum conditioning-set size.

    Returns
    -------
    P : dict mapping each node to its children.
    G : dict mapping each node to its neighbors in the undirected skeleton.
    sep_sets : dict mapping each removed pair to its separating set.
    """
    X = np.asarray(data_matrix)
    n, p = X.shape
    assert len(var_names) == p
    idx = {v: i for i, v in enumerate(var_names)}

    # Candidate edges
    if candidate_pairs is None:
        allow = set()
        for i in range(p):
            for j in range(i + 1, p):
                allow.add(frozenset({var_names[i], var_names[j]}))
    else:
        allow = set(candidate_pairs)

    exogenous = set(exogenous) if exogenous is not None else set()

    # Initialize the undirected skeleton.
    G = {v: set() for v in var_names}
    for e in allow:
        a, b = tuple(e)
        G[a].add(b)
        G[b].add(a)

    if verbose:
        print("[PC] initial skeleton:")
        for v in var_names:
            print(v, ":", sorted(G[v]))

    # Prune edges in order of increasing conditioning-set size.
    sep_sets = {}
    L = 0
    changed = True
    rng = np.random.default_rng(0)

    while changed and L <= max_cond_set:
        changed = False
        # Visit each edge once at this order.
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
                    G[a].discard(b)
                    G[b].discard(a)
                    sep_sets[frozenset({a, b})] = set(S)
                    changed = True
                    # Trace removal of this pair.
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
    # Edge orientation
    # ============================================

    P = {v: set() for v in var_names}  # P[u] = {children}

    # Orient edges outward from exogenous nodes.
    for v in exogenous:
        if v not in G:
            continue
        for nei in G[v]:
            P[v].add(nei)

    # Orient unshielded triples x-z-y as colliders when z is absent from the separating set.
    for z in var_names:
        z_nei = list(G[z])
        for i in range(len(z_nei)):
            for j in range(i + 1, len(z_nei)):
                x, y = z_nei[i], z_nei[j]
                # Adjacent endpoints shield the triple.
                if (y in G[x]) or (x in G[y]):
                    continue
                Sxy = sep_sets.get(frozenset({x, y}), set())
                if z not in Sxy:
                    if z not in exogenous:
                        P[x].add(z)
                        P[y].add(z)

    # Meek rule R1: orient y-z as y->z when x->y and x is not adjacent to z.
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
                for x in parents_y:
                    if (z not in G[x]) and (x not in G[z]):
                        if z not in exogenous:
                            P[y].add(z)
                            changed = True
                            break
                if changed:
                    break

    # Remove any remaining arrows into exogenous nodes.
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
# Layer-specific analyses
# ============================================

def layer_phi_to_omega(data,
                       alpha=1e-3,
                       use_perm=False,
                       n_perm=30,
                       max_cond_set=2,
                       verbose=False):
    """
    Layer 1: Phi_i (flux_i) -> omega_q_j.
    Input arrays:
      - data["flux"]    : (samples, N)
      - data["omega_q_clean"] : (samples, N)
    """
    flux = np.asarray(data["flux"])
    omega_q = np.asarray(data["omega_q_clean"])
    n_samples, N = flux.shape

    X = np.concatenate([flux, omega_q], axis=1)
    var_names = [f"Phi_{i}" for i in range(N)] + [f"omega_{i}" for i in range(N)]

    # Allow only flux-frequency edges.
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
    Layer 2: Dyson-derived qubit frequencies -> drive frequencies.
    Input arrays:
      - data["omega_q_clean"] : (samples, N)
      - data["omega_d_noisy"] : (samples, N)
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
    Alternate layer 2: Dyson-derived qubit frequencies -> F_k.
    Input arrays:
      - data["omega_q_clean"] : (samples, N)
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
    Layer 3: Omega_i -> F_k.
    Input arrays:
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
    Layer 4: omega_d_i -> F_k.

    Input arrays:
      - data["omega_d_noisy"] : (samples, N), drive frequency
      - data["F"]       : (samples, N)

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
# Merge layer graphs
# ============================================

def merge_graphs(*Ps):
    """
    Merge directed child sets across layers.
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
    Merge the undirected skeletons into one adjacency table.
    Each G maps a node to its set of neighbors.
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
        includeF=True,                 # Used when run_layers is omitted.
        run_layers=None,               # For example, ("P1",) or ("P1", "P2").
        merge_mode="directed",         # "directed" or "skeleton"
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
    Available layers:
      P1: Phi -> omega_q
      P2: omega_q -> omega_d
      P3: Omega -> F
      P4: omega_d -> F

    merge_mode:
      - "directed" returns the merged directed graph P_total.
      - "skeleton" returns the merged undirected graph G_total.
    """

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

    # Merge the selected layers.
    if merge_mode == "skeleton":
        G_total = merge_skeletons(*(G_layers[k] for k in run_layers if k in G_layers))
        print("\n=== Final merged skeleton (undirected) ===")
        return G_total, {"P_layers": P_layers, "G_layers": G_layers}

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
    Plot the layered graph with circular nodes and equal axis scaling.
    """

    # Collect all endpoints.
    nodes = set(P_total.keys())
    for ch in P_total.values():
        nodes |= set(ch)

    # Colors by variable type
    type_color = {
        "Phi": "skyblue",
        "omega": "lightgreen",
        "omega_d": "orange",
        "F": "khaki",
        "Omega": "lightcoral",
    }

    # Layer order
    row_of_type = {
        "Phi": 0,
        "omega": 1,      # Place qubit frequencies above drive frequencies.
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

    # Node coordinates
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

    # Size the figure to fit all nodes.
    xs = np.array([p[0] for p in positions.values()])
    ys = np.array([p[1] for p in positions.values()])
    xmin, xmax = xs.min() - 2, xs.max() + 2
    ymin, ymax = ys.min() - 2, ys.max() + 2

    width = max(12, (xmax - xmin) * 0.8)
    height = max(6, (ymax - ymin) * 0.6)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")

    # Draw nodes.
    for n, (x, y) in positions.items():
        t = _node_type(n)
        col = type_color.get(t, "white")

        circ = plt.Circle((x, y), node_radius, fc=col, ec="black")
        ax.add_patch(circ)

        ax.text(x, y, str(_idx_from_name(n)),
                ha="center", va="center", fontsize=11)

    # Draw arrows between node boundaries.
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

    # Legend
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], marker='o', linestyle='None',
                             markerfacecolor=c, markeredgecolor="black",
                             markersize=8)
               for c in type_color.values()]

    labels = [r"$\Phi$", r"$f$", r"$\omega_d$", r"$F$", r"$\Omega$"]

    ax.legend(handles, labels, title="Variable type", loc="upper right")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Leave room for edge labels.
    plt.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05)

    # Draw nodes as circles.
    ax.set_aspect('equal', adjustable='box')

    plt.show()

    return fig

# To export the merged graph:
# save_graph_as_graphml(P_total)

# To plot the merged graph:
# plot_causal_graph(P_total)

def plot_causal_graph(P_total):
    """
    Plot the merged graph.
    P_total maps each node to its children.
    """
    G = nx.DiGraph()

    # Color by variable type.
    node_types = {
        "Phi": "skyblue",
        "omega_q": "lightgreen",
        "Omega": "lightcoral",
        "omega_d": "orange",
        "F": "lightyellow"
    }

    for u, children in P_total.items():
        for v in children:
            G.add_edge(u, v)

    plt.figure(figsize=(12, 8))

    # Larger k increases the preferred node spacing.
    pos = nx.spring_layout(G, seed=42, k=1.5)

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

    # Label each node with its qubit index.
    labels = {node: ''.join(filter(str.isdigit, node)) for node in G.nodes()}

    nx.draw(G, pos, with_labels=True, labels=labels, node_size=1000, node_color=node_color, font_size=10, font_weight='bold', edge_color='gray', arrows=True, edgecolors="black", width=1.5)

    handles = []
    labels_legend = []
    for node_type, color in node_types.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
        labels_legend.append(node_type)

    plt.legend(handles, labels_legend, title="Variable Types", loc="upper right", fontsize=12)

    plt.title("Causal Graph", fontsize=16)
    plt.show()

# To plot the merged graph:
# plot_causal_graph(P_total)

def plot_causal_graph(P_total):
    """
    Plot the merged graph.
    P_total maps each node to its children.
    """
    G = nx.DiGraph()

    # Color by variable type.
    node_types = {
        "Phi": "skyblue",
        "omega_q": "lightgreen",
        "Omega": "lightcoral",
        "omega_d": "orange",
        "F": "lightyellow"
    }

    for u, children in P_total.items():
        for v in children:
            G.add_edge(u, v)

    plt.figure(figsize=(12, 8))

    # Larger k increases the preferred node spacing.
    pos = nx.spring_layout(G, seed=42, k=1.5)

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

    # Label each node with its qubit index.
    labels = {node: ''.join(filter(str.isdigit, node)) for node in G.nodes()}

    nx.draw(G, pos, with_labels=True, labels=labels, node_size=1000, node_color=node_color, font_size=10, font_weight='bold', edge_color='gray', arrows=True, edgecolors="black", width=1.5)

    handles = []
    labels_legend = []
    for node_type, color in node_types.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
        labels_legend.append(node_type)

    plt.legend(handles, labels_legend, title="Variable Types", loc="upper right", fontsize=12)

    plt.title("Causal Graph", fontsize=16)
    plt.show()


def save_P_json(P_total, path):
    """P_total: dict[str, set[str]] or dict[str, list[str]]"""
    obj = {k: sorted(list(v)) for k, v in P_total.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
# ============================================
# Demo
# ============================================

if __name__ == "__main__":
    # Select the input dataset.
    # npz_path = "demo_dataset_3_3_gpu.npz" 
    # npz_path = "demo_dataset_gpu_1_10_chain_withF.npz"
    # npz_path = "nointerac_dataset_gpu_1_10_chain_withF.npz"
    # npz_path = "tttnointerac_dataset_gpu_1_10_chain_withF.npz"
    # npz_path = "tttnointerac_dataset_gpu_1_5_chain_withF.npz"
    # npz_path = "tttnointerac_dataset_gpu_1_5_chain_withF-2.npz"
    # npz_path = "nointerac_dataset_gpu_1_10_chain_withF.npz"  # Panel (a)
    # npz_path = "nointerac_dataset_gpu_1_5_chain_withF.npz"
    # npz_path = "interac_dataset_gpu_1_8_chain_withF.npz"
    # npz_path = "interac_dataset_gpu_1_8_chain_withF-v2.npz"  # Panel (b)
    # npz_path = "fullinterac_dataset_gpu_2_4_dense_withF.npz"  # Panel (c)
    # npz_path = "fullinterac_dataset_gpu_4_4_nn8_withoutF.npz"  # 2-by-2 panel layout
    # npz_path = "fullinterac_dataset_gpu_2_2_withoutF.npz"  
    # npz_path = "fullinterac_dataset_gpu_4_4_nn8_withoutF.npz"
    npz_path = "fullinterac_dataset_gpu_4_4_dense_withoutF.npz"
    # includeF=False selects only the flux-frequency layer.
    # includeF=True selects all four layers.
    # includeF = True

    with np.load(npz_path, allow_pickle=True) as f:
        data = {k: f[k] for k in f.files}

    # A small alpha, including zero, retains weak edges.
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
