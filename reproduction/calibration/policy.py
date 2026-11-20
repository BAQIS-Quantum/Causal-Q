"""Recheck coverage and equal-weight summaries over minimum batch schedules."""

from itertools import combinations

import numpy as np

TOLERANCE = 0.001
N_SITES = 8
ACTIONS = ("amplitude", "carrier", "flux_fixed_carrier", "joint_task")


def partition_masks(a):
    """Enumerate all minimum color partitions; no favourable tie is selected."""
    n = len(a)
    lower = max(
        k
        for k in range(1, n + 1)
        if any(all(a[i, j] for i, j in combinations(c, 2)) for c in combinations(range(n), k))
    )
    for k in range(lower, n + 1):
        result = []

        def visit(i, groups):
            if i == n:
                if len(groups) == k:
                    result.append([sum(1 << j for j in group) for group in groups])
                return
            for group in groups:
                if all(not a[i, j] for j in group):
                    group.append(i)
                    visit(i + 1, groups)
                    group.pop()
            if len(groups) < k:
                groups.append([i])
                visit(i + 1, groups)
                groups.pop()

        visit(0, [])
        if result:
            return result
    raise RuntimeError("No partition")


def summarize_partitions(partitions, risk, tolerance):
    rows = []
    for part in partitions:
        failed = [mask for mask in part if risk[mask] > tolerance]
        rows.append(
            (
                len(part),
                max(risk[mask] for mask in part),
                not failed,
                len(part) + sum(mask.bit_count() for mask in failed),
                len(failed),
            )
        )
    a = np.array(rows, float)
    return {
        "planned_batches": float(a[:, 0].mean()),
        "max_target_contamination": float(a[:, 1].mean()),
        "all_batches_pass": float(a[:, 2].mean()),
        "discard_then_serial_retry_batches": float(a[:, 3].mean()),
        "unsafe_batches": float(a[:, 4].mean()),
        "partitions_averaged": len(partitions),
    }


def check_policy(mask, affected, change):
    affected, retained = np.broadcast_arrays(affected, mask[:, None, :])
    missed = affected & ~retained
    return {
        "mean_jobs": float(mask.sum(1).mean()),
        "no_miss_event_fraction": float(np.mean(~missed.any(axis=-1))),
        "pooled_recall": (
            float((affected & retained).sum() / affected.sum()) if affected.any() else 1.0
        ),
        "mean_max_missed_change": float(np.where(retained, 0, change).max(axis=-1).mean()),
        "events": affected.size // N_SITES,
    }


def random_policy(mask, affected, change):
    # Enumerate every same-size set that retains the task's own target.
    rows = []
    for site, count in enumerate(mask.sum(1)):
        candidates = list(combinations([i for i in range(N_SITES) if i != site], count - 1))
        selected = np.zeros((len(candidates), N_SITES), bool)
        selected[:, site] = True
        for i, others in enumerate(candidates):
            selected[i, list(others)] = True
        trial = check_policy(
            selected, affected[:, site : site + 1, :, :], change[:, site : site + 1, :, :]
        )
        rows.append(trial)
    # Pool recall over affected targets, and other means over equal-size site samples.
    keys = ("mean_jobs", "no_miss_event_fraction", "mean_max_missed_change")
    truth_counts = affected.sum(axis=(0, 2, 3))
    result = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    result["pooled_recall"] = (
        float(np.average([row["pooled_recall"] for row in rows], weights=truth_counts))
        if truth_counts.sum()
        else 1.0
    )
    result["events"] = affected.size // N_SITES
    return result
