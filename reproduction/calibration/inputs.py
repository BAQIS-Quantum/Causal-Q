"""Fixed operating points, layered graph inputs and response-audit masks."""

import json
from pathlib import Path

import numpy as np

from graphs import task_support

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GRAPHS = HERE.parent / "pulse" / "data"
CASES = ("uncoupled", "chain", "dense")
METHODS = {
    "uncompressed": "Individual",
    "original": "Graph",
    "audited": "Audited",
    "shuffled": "Shuffled",
}


def workpoints():
    return json.loads((DATA / "workpoints.json").read_text())


def graph(case, directory=GRAPHS):
    edges = json.loads((directory / f"{case}_graph.json").read_text())["edges"]
    return edges, task_support(edges)


def audited_masks(records, support):
    baseline = records["primary_baseline"][:8, None, None, None, :]
    change = np.abs(records["primary_scores"][:8] - baseline)
    return support | (change > 0.001).any(axis=(0, 3))


def physical_groups(h):
    h = np.asarray(h)
    remaining = set(range(len(h)))
    groups = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        group = [root]
        for site in group:
            reachable = remaining.intersection(np.flatnonzero(h[site]))
            group.extend(sorted(reachable))
            remaining.difference_update(reachable)
        groups.append(sorted(group))
    return groups
