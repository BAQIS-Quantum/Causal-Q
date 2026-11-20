"""Project the layered graph to task reachability and color the control conflicts."""

from itertools import combinations

import numpy as np


def color_support(mask, method="original"):
    if method == "uncompressed":
        return [[i] for i in range(24)]
    conflict = mask.astype(int) @ mask.astype(int).T > 0
    np.fill_diagonal(conflict, False)
    colors = np.full(24, -1, int)
    for _ in range(24):
        candidates = np.flatnonzero(colors < 0)
        node = max(
            map(int, candidates),
            key=lambda i: (len(set(colors[conflict[i]]) - {-1}), int(conflict[i].sum()), -i),
        )
        forbidden = set(colors[conflict[node]])
        color = next(c for c in range(24) if c not in forbidden)
        colors[node] = color
    groups = [
        np.flatnonzero(colors == c).astype(int).tolist() for c in range(int(colors.max()) + 1)
    ]
    assert all(not np.any(mask[group].sum(axis=0) > 1) for group in groups)
    return groups


def parse_site(node):
    return int(node.rsplit("_", 1)[1])


def task_support(edges):
    adjacency = {}
    for source, target in edges:
        adjacency.setdefault(source, []).append(target)
    support = np.zeros((8, 8), dtype=bool)
    for site in range(8):
        pending = [f"{control}_{site}" for control in ("Phi", "Omega", "omega_d")]
        seen = set(pending)
        while pending:
            for target in adjacency.get(pending.pop(), ()):
                if target.startswith("F_"):
                    support[site, parse_site(target)] = True
                elif target not in seen:
                    seen.add(target)
                    pending.append(target)
    return support


def task_structure(mask):
    overlap = mask.astype(int) @ mask.astype(int).T > 0
    np.fill_diagonal(overlap, False)
    remaining = set(range(8))
    groups = []
    while remaining:
        root = min(remaining)
        group = [root]
        remaining.remove(root)
        for site in group:
            reachable = remaining.intersection(map(int, np.flatnonzero(overlap[site])))
            remaining.difference_update(reachable)
            group.extend(sorted(reachable))
        groups.append(sorted(group))
    clique = max(
        len(vertices)
        for size in range(1, 9)
        for vertices in combinations(range(8), size)
        if all(overlap[i, j] for i, j in combinations(vertices, 2))
    )
    color_groups = color_support(np.tile(mask, (3, 1)))
    # Any task clique of size k forces 3k control colors. If coloring
    # doesn't reach this lower bound, do not present it as exact.
    n_colors = len(color_groups)
    batch_count = None
    order = sorted(range(8), key=lambda i: (-overlap[i].sum(), i))

    def can_color(position, labels, limit):
        if position == len(order):
            return True
        i = order[position]
        forbidden = {labels[j] for j in np.flatnonzero(overlap[i]) if j in labels}
        for color in range(min(limit, max(labels.values(), default=-1) + 2)):
            if color not in forbidden:
                labels[i] = color
                if can_color(position + 1, labels, limit):
                    return True
                del labels[i]
        return False

    for limit in range(clique, 9):
        if can_color(0, {}, limit):
            batch_count = limit
            break
    return {
        "task_target_pairs": int(mask.sum()),
        "mean_reach": float(mask.sum() / 8),
        "groups": groups,
        "mean_group_size": float(sum(len(g) ** 2 for g in groups) / 8),
        "task_conflicts": int(overlap.sum() // 2),
        "minimum_task_batches": batch_count,
        "control_colors_used": n_colors,
        "control_coloring_certified_optimal_by_task_clique": n_colors == 3 * clique,
        "queries_per_function_gradient": 1 + n_colors,
    }


SEEDS = range(2026111201, 2026111207)


def variants(audited, physical_groups):
    components = {site: k for k, sites in enumerate(physical_groups) for site in sites}
    within = np.array([[components[i] == components[j] for j in range(8)] for i in range(8)])
    result = {"audited": audited, "individual": np.ones((8, 8), dtype=bool)}
    if not audited[0, 1]:
        raise ValueError("Expected task 0 to reach target 1 before the targeted deletion")
    missing = audited.copy()
    missing[0, 1] = False
    result["minus_0_to_1"] = missing
    for name, allowed in (
        ("add", ~within & ~audited),
        ("delete", audited & ~np.eye(8, dtype=bool)),
    ):
        eligible = np.argwhere(allowed)
        for seed in SEEDS:
            chosen = eligible[np.random.default_rng(seed).choice(len(eligible), 3, replace=False)]
            mask = audited.copy()
            mask[chosen[:, 0], chosen[:, 1]] = name == "add"
            result[f"{name}_{seed}"] = mask
    return result
