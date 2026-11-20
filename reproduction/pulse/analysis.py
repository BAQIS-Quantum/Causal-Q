"""Figure 4 marginal information maps and deterministic layered screening."""

import numpy as np
from npeet import entropy_estimators as ee

from flux.physics.counts import accelerate_marginal_counts
from flux.physics.information import _seeded_estimation, sha256_file
from flux.physics.screening import screen_layer


LAYERS = (
    ("flux", "omega_q_clean", "Phi", "f", 0.0),
    ("omega_q_clean", "omega_d_noisy", "f", "omega_d", 0.006),
    ("Omega_real", "F", "Omega", "F", 0.003),
    ("omega_d_noisy", "F", "omega_d", "F", 0.025),
)
MI_K = 130
CMI_K = 3
MAX_ORDER = 2


def heatmap(drive, fidelity):
    matrix = np.zeros((drive.shape[1], drive.shape[1]))
    with _seeded_estimation(43):
        for i, j in np.ndindex(matrix.shape):
            if i != j:
                matrix[i, j] = ee.mi(drive[:, i, None], fidelity[:, j, None], k=MI_K, base=2)
    return matrix


def compute_case(path, name):
    with np.load(path, allow_pickle=False) as data, accelerate_marginal_counts(ee):
        layers = []
        for seed, (a, b, src, dst, cutoff) in enumerate(LAYERS, start=43):
            result = screen_layer(data[a], data[b], src, dst, cutoff, mi_k=MI_K,
                                  cmi_k=CMI_K, seed=seed, max_cond_set=MAX_ORDER)
            layers.append({key: val for key, val in result.items()
                           if key not in ("tests", "scope", "traversal")})
        matrix = None if name == "uncoupled" else heatmap(data["omega_d_noisy"], data["F"])
    dataset_hash = sha256_file(path)
    edges = sorted(edge for layer in layers for edge in layer["edges"])
    graph = {"dataset_sha256": dataset_hash,
             "nodes": [f"{p}_{i}" for p in ("Phi", "f", "omega_d", "F", "Omega") for i in range(8)],
             "layers": layers, "edges": edges, "edge_count": len(edges)}
    mi = (None if matrix is None else
          {"dataset_sha256": dataset_hash, "seed": 43, "k": MI_K, "mi_bits": matrix.tolist()})
    return graph, mi
