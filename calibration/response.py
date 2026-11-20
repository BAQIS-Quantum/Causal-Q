"""Time-dependent pulse evolution with the nominal target and initial frame held fixed."""

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pulse.selected_pulse import SelectedPulse, channel_metrics, target_unitaries

from pulse.model import SolverConfig, bare_response, components, coupling_matrix, generate_rows


class TimeDependentResponse:
    def __init__(self, case, h, solver=SolverConfig()):
        self.case = case
        self.solver = solver
        self.h = np.asarray(h)
        self.components = components(self.h)
        self.models = [
            (np.asarray(sites), SelectedPulse(self.h[np.ix_(sites, sites)]))
            for sites in self.components
        ]

    def targets(self, point):
        return target_unitaries(
            np.asarray(point["q"]) - np.asarray(point["bare"]),
            point["amplitude_nominal"],
            point["duration"] / (2 * np.pi),
        )

    def score(self, point, *, flux=None, amplitude=None, drive=None, targets=None):
        """Return the four-probe and Haar scores relative to the same target."""
        bare = np.asarray(point["bare"]) if flux is None else bare_response(np.asarray(flux))
        reference = np.asarray(point["drive"])
        carrier = reference if drive is None else np.asarray(drive)
        amplitude = np.asarray(point["amplitude_actual"] if amplitude is None else amplitude)
        target = self.targets(point) if targets is None else np.asarray(targets)
        duration = point["duration"] / (2 * np.pi)
        four, haar = np.empty(len(self.h)), np.empty(len(self.h))
        for sites, model in self.models:
            result = model.propagate(
                bare[sites],
                carrier[sites],
                amplitude[sites],
                duration,
                max_step_ns=self.solver.max_step_internal,
                rtol=self.solver.rtol,
                atol=self.solver.atol,
            )
            # The two rotating frames coincide at t=0. At the endpoint,
            # undo the candidate frame and enter the initial carrier frame.
            phase = np.exp(-1j * np.pi * duration * (model.z @ (carrier[sites] - reference[sites])))
            columns = phase[:, None] * result["columns"]
            four[sites], haar[sites], _ = channel_metrics(columns, target[sites], model.n)
        if not np.isfinite(four).all() or np.min(four) < -1e-8 or np.max(four) > 1 + 1e-8:
            raise RuntimeError("Invalid four-probe score from the time-dependent response")
        return four, haar

    def evaluate(self, point, *, flux=None, amplitude=None, drive=None, targets=None):
        return self.score(point, flux=flux, amplitude=amplitude, drive=drive, targets=targets)[0]


def generate_points(case):
    """Draw the baseline, independent validation, and optimization cohorts."""
    h = coupling_matrix(case, seed=43)
    points = {"h": h}
    fields = {
        "flux": "flux",
        "bare": "omega_d_clean",
        "q": "omega_q_clean",
        "drive": "omega_d_noisy",
        "amplitude_actual": "Omega_real",
        "amplitude_nominal": "Omega_ideal",
        "duration": "Tg",
    }
    for cohort, count, seed in [
        ("primary", 16, 2026110601),
        ("validation", 16, 2026110603),
        ("optimization", 8, 2026110605),
    ]:
        rows = generate_rows(h, np.random.default_rng(seed), count)
        points[cohort] = [
            dict(point_id=i, **{field: rows[key][i] for field, key in fields.items()})
            for i in range(count)
        ]
    return points
