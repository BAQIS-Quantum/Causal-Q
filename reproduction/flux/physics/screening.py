"""Layered dependency screen with immediate deletion and sorted traversal."""
from itertools import combinations

import numpy as np

from .information import _load_estimator, _seeded_estimation


def screen_layer(source, target, source_prefix, target_prefix, threshold,
                 mi_k=130, cmi_k=3, seed=43, max_cond_set=2):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 2 or source.shape != target.shape or len(source) <= max(mi_k, cmi_k):
        raise ValueError("Layer inputs must share shape and have more samples than k")

    width = source.shape[1]
    neighbors = [set(range(width)) for _ in range(width)]
    estimator = _load_estimator()
    records = []
    with _seeded_estimation(seed):
        for order in range(max_cond_set + 1):
            changed = False
            candidates = [(i, j) for i in range(width) for j in sorted(neighbors[i])]
            for i, j in candidates:
                available = sorted(neighbors[i] - {j})
                x, y = source[:, [i]], target[:, [j]]
                for condition in combinations(available, order):
                    if condition:
                        score = estimator.cmi(x, y, target[:, condition], k=cmi_k, base=2)
                    else:
                        score = estimator.mi(x, y, k=mi_k, base=2)
                    score = float(score)
                    if not np.isfinite(score):
                        raise ValueError("Nonfinite MI/CMI estimate")
                    removed = score <= threshold
                    records.append({"source": i, "target": j, "condition_target_indices": list(condition),
                                    "bits": score, "removed": removed})
                    if removed:
                        neighbors[i].remove(j)
                        changed = True
                        break
            if not changed:
                break

    edges = [[f"{source_prefix}_{i}", f"{target_prefix}_{j}"]
             for i in range(width) for j in sorted(neighbors[i])]
    return {
        "source_prefix": source_prefix, "target_prefix": target_prefix,
        "threshold_bits": threshold, "mi_k": mi_k, "cmi_k": cmi_k,
        "seed": seed, "max_cond_set": max_cond_set, "edges": edges, "tests": records,
        "traversal": "Sorted source, target and conditioning indices with immediate deletion",
    }
