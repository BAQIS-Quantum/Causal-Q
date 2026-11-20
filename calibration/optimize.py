"""Bounded L-BFGS-B with charged individual or colored finite-difference queries."""

import numpy as np
from scipy.optimize import minimize
from graphs import color_support

H = 1e-4
SCALES = np.repeat([0.015, 0.006, 0.015], 8)


class BudgetReached(Exception):
    pass


def physical_controls(point, z):
    z = np.asarray(z, float)
    if z.shape != (24,) or not np.isfinite(z).all() or abs(z).max() > 1 + 1e-9:
        raise ValueError("Invalid normalized 24D command")
    return {
        "amplitude": np.asarray(point["amplitude_actual"]) + SCALES[:8] * z[:8],
        "drive": np.asarray(point["drive"]) + SCALES[8:16] * z[8:16],
        "flux": np.asarray(point["flux"]) + SCALES[16:] * z[16:],
    }


def finite_difference(z, query, mask, groups, h=H):
    """query(z,role,group) is counted by caller; return mean loss,grad,J."""
    z = np.asarray(z, float)
    base = np.asarray(query(z, "base", None))
    jac = np.zeros((24, 8))
    steps = np.where(z + h <= 1, h, -h)
    for group_index, group in enumerate(groups):
        trial = z.copy()
        trial[group] += steps[group]
        changed = np.asarray(query(trial, "finite_difference", group_index)) - base
        for i in group:
            jac[i, mask[i]] = changed[mask[i]] / steps[i]
    return float(np.mean(1 - base)), -jac.mean(axis=1), jac


def optimize(response, point, mask, method="original", budget=256):
    """Collect every paid query, including line searches and partial gradients."""
    groups = color_support(mask, method)
    trace = []
    best_loss = float("inf")
    best_z = np.zeros(24)

    def query(z, role, group):
        nonlocal best_loss, best_z
        if len(trace) >= budget:
            raise BudgetReached
        scores = response.evaluate(point, **physical_controls(point, z))
        loss = float(np.mean(1 - scores))
        if loss < best_loss:
            best_loss, best_z = loss, z.copy()
        trace.append({"z": z.copy(), "F": scores, "loss": loss})
        return scores

    def loss_and_gradient(z):
        loss, grad, _ = finite_difference(z, query, mask, groups)
        return loss, grad

    try:
        while len(trace) < budget:
            minimize(
                loss_and_gradient,
                best_z.copy(),
                method="L-BFGS-B",
                jac=True,
                bounds=[(-1.0, 1.0)] * 24,
                options={"ftol": 1e-11, "gtol": 1e-6, "maxiter": 100, "maxls": 12, "maxcor": 10},
            )
    except BudgetReached:
        pass
    return {"trace": trace, "best_z": best_z}


def optimize_block(response, point, groups, budget=256):
    command = np.zeros(8)
    initial_amp = np.asarray(point["amplitude_actual"])
    trace = []

    def evaluate(trial):
        if len(trace) >= budget:
            raise BudgetReached
        scores = response.evaluate(point, amplitude=initial_amp + trial)
        trace.append(scores)
        return float(np.mean(1 - scores))

    value = evaluate(command)
    while len(trace) < budget:
        for group in groups:
            trials = []

            def block_objective(values):
                trial = command.copy()
                trial[group] = values
                loss = evaluate(trial)
                trials.append((loss, trial))
                return loss

            try:
                minimize(
                    block_objective,
                    command[group],
                    method="L-BFGS-B",
                    bounds=[(-0.015, 0.015)] * len(group),
                    options={"eps": 1e-6, "ftol": 1e-11, "gtol": 1e-6, "maxiter": 25, "maxls": 15},
                )
            except BudgetReached:
                pass
            if trials:
                candidate, trial = min(trials, key=lambda row: row[0])
                if candidate < value:
                    command, value = trial, candidate
            if len(trace) >= budget:
                break
    return trace


def mask_for(method, graph_support, audited_support, point_id):
    if method == "uncompressed":
        return np.ones((24, 8), dtype=bool)
    if method == "shuffled":
        permutation = np.random.default_rng(202611066810 + point_id).permutation(8)
        return np.tile(graph_support[permutation], (3, 1))
    return np.tile(audited_support if method == "audited" else graph_support, (3, 1))
