"""Joint L-BFGS-B and carrier-only queries in the fixed-target protocol."""

import numpy as np
from scipy.optimize import minimize

SCALES = np.repeat([0.02, 0.08], 8)
STEP = 1e-4
BUDGET = 256


class BudgetReached(Exception):
    pass


def joint(response, support=None, groups=None):
    settings, scores, four_probe = [], [], []
    best_z = np.zeros(16)
    best_loss = np.inf

    def query(z):
        nonlocal best_loss, best_z
        if len(scores) == BUDGET:
            raise BudgetReached
        corrections = z * SCALES
        four, haar = response.evaluate(
            response.initial_amplitude + corrections[:8], response.initial_drive + corrections[8:]
        )
        loss = 1 - np.mean(haar)
        if loss < best_loss - 1e-14:
            best_loss, best_z = loss, z.copy()
        settings.append(z.copy())
        four_probe.append(four)
        scores.append(haar)
        return loss

    def grouped_gradient(z):
        loss = query(z)
        baseline = scores[-1]
        steps = np.where(z + STEP <= 1, STEP, -STEP)
        gradient = np.empty(len(z))
        for group in groups:
            trial = z.copy()
            trial[group] += steps[group]
            query(trial)
            for i in group:
                diff = (scores[-1] - baseline)[support[i]].sum() / len(baseline)
                gradient[i] = -diff / (trial[i] - z[i])
        return loss, gradient

    grouped = support is not None
    query(best_z)
    try:
        while len(scores) < BUDGET:
            minimize(
                grouped_gradient if grouped else query,
                best_z.copy(),
                jac=True if grouped else None,
                method="L-BFGS-B",
                bounds=[(-1.0, 1.0)] * 16,
                options={"ftol": 1e-13, "gtol": 1e-7, "eps": STEP, "maxiter": 100, "maxls": 20},
            )
    except BudgetReached:
        pass
    return np.array(scores), np.array(settings), np.array(four_probe)


def carrier(response):
    drive = response.initial_drive.copy()

    def evaluate(nu):
        return response.evaluate(response.initial_amplitude, nu)[1]

    scores = [evaluate(drive)]
    best_mean = scores[0].mean()
    for step in (0.04, 0.01, 0.0025):
        for i in range(8):
            baseline = drive.copy()
            for factor in (-1.0, -0.5, 0.5, 1.0):
                trial = baseline.copy()
                trial[i] = np.clip(
                    baseline[i] + step * factor,
                    response.initial_drive[i] - 0.05,
                    response.initial_drive[i] + 0.05,
                )
                haar = evaluate(trial)
                scores.append(haar)
                if haar.mean() > best_mean + 1e-14:
                    drive, best_mean = trial, haar.mean()
    return np.array(scores)
