"""Evaluate signed perturbations in isolation and in proposed batches."""

import numpy as np
from policy import ACTIONS, N_SITES, TOLERANCE, partition_masks, summarize_partitions

SIGNS = (-1, 1)
STEPS = {"amplitude": 0.005, "drive": 0.002, "flux": 0.005}
CONTROL_KEYS = {"amplitude": "amplitude_actual", "drive": "drive", "flux": "flux"}
ACTION_CONTROLS = dict(zip(ACTIONS, (("amplitude",), ("drive",), ("flux",), tuple(STEPS))))


def vector_probe(model, point, action, perturbation, target):
    controls = {
        name: np.asarray(point[CONTROL_KEYS[name]]) + STEPS[name] * perturbation
        for name in ACTION_CONTROLS[action]
    }
    return model.evaluate(point, targets=target, **controls)


def recheck_point(model, point):
    target = model.targets(point)
    baseline = model.evaluate(point, targets=target)
    scores = np.empty((len(ACTIONS), N_SITES, len(SIGNS), N_SITES))
    for ai, action in enumerate(ACTIONS):
        for site in range(N_SITES):
            for si, sign in enumerate(SIGNS):
                perturbation = np.zeros(N_SITES)
                perturbation[site] = sign
                scores[ai, site, si] = vector_probe(model, point, action, perturbation, target)
    return {"baseline": baseline.tolist(), "scores": scores.tolist()}


def schedule_point(model, point, recheck, policies, signs):
    target = model.targets(point)
    partitions = {}
    for name, mask in policies.items():
        overlap = mask.astype(int) @ mask.T > 0
        np.fill_diagonal(overlap, False)
        partitions[name] = partition_masks(overlap)
    required = sorted({batch for parts in partitions.values() for part in parts for batch in part})
    members = {mask: np.array([i for i in range(N_SITES) if mask >> i & 1]) for mask in required}
    isolated = np.asarray(recheck["scores"])[ACTIONS.index("joint_task")]
    cache = {}
    records = []
    for sign_vector in signs:
        risks = np.zeros(1 << N_SITES)
        diagonal = np.array(
            [isolated[i, SIGNS.index(sign), i] for i, sign in enumerate(sign_vector)]
        )
        for mask, sites in members.items():
            if len(sites) == 1:
                continue
            perturbation = np.zeros(N_SITES, int)
            perturbation[sites] = sign_vector[sites]
            key = tuple(perturbation)
            if key not in cache:
                cache[key] = vector_probe(model, point, "joint_task", perturbation, target)
            risks[mask] = np.max(np.abs(cache[key][sites] - diagonal[sites]))
        records.append(
            {
                name: summarize_partitions(parts, risks, TOLERANCE)
                for name, parts in partitions.items()
            }
        )
    return {"records": records, "additional_vector_queries": len(cache)}
