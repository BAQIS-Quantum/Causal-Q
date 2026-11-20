"""Fixed X(pi/2) pulse evolution in the operating point's reference frame."""

import numpy as np
from scipy.integrate import solve_ivp

from pulse.model import components
from pulse.selected_pulse import SelectedPulse, _rhs, channel_metrics

X90 = (np.eye(2) - 1j * np.array([[0, 1], [1, 0]])) / np.sqrt(2)
DURATION = 4.0  # ns


def propagate(model, bare, drive, amplitude, *, tight):
    max_step, rtol, atol = (0.025, 1e-11, 1e-13) if tight else (0.1, 1e-9, 1e-11)
    if not len(model.pg):
        columns = model.propagate(
            bare, drive, amplitude, DURATION, max_step_ns=max_step, rtol=rtol, atol=atol
        )["columns"]
    else:
        rates = 2 * np.pi * (drive[model.pi] - drive[model.pj])
        if np.max(abs(rates)):
            max_step = min(max_step, 2 * np.pi / (16 * np.max(abs(rates))))
        shape = model.initial.shape
        diagonal = model.z @ (bare - drive) / 2

        def derivative(t, state):
            return _rhs(
                t,
                state.reshape(shape),
                diagonal,
                amplitude,
                model.masks,
                model.pi,
                model.pj,
                model.pg,
                rates,
            ).ravel()

        solution = solve_ivp(
            derivative,
            (0.0, DURATION),
            model.initial.ravel(),
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        columns = solution.y[:, -1].reshape(shape)
    phase = np.exp(-1j * np.pi * DURATION * (model.z @ (drive - bare)))
    return phase[:, None] * columns


class GateResponse:
    def __init__(self, h, point):
        self.bare = np.array(point["bare"])
        self.initial_amplitude = 0.0625 + np.array(point["amplitude_error"])
        self.initial_drive = self.bare + np.array(point["carrier_error"])
        self.parts = [
            (np.array(group), SelectedPulse(h[np.ix_(group, group)])) for group in components(h)
        ]

    def evaluate(self, amplitude, drive, tight=False):
        four, haar = np.empty(8), np.empty(8)
        for sites, model in self.parts:
            columns = propagate(
                model, self.bare[sites], drive[sites], amplitude[sites], tight=tight
            )
            targets = np.repeat(X90[None, :, :], len(sites), axis=0)
            four[sites], haar[sites], _ = channel_metrics(columns, targets, len(sites))
        return four, haar
