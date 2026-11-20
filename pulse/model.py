"""Control ensembles, scalar frequencies and the four-probe pulse observable."""

from dataclasses import dataclass

import numpy as np

from flux.physics import coupling as static
from flux.physics.coupling import bare_response, frequency_batch
from .selected_pulse import SelectedPulse, channel_metrics, target_unitaries

SAMPLES = {"uncoupled": 40000, "chain": 20000, "dense": 40000}


@dataclass(frozen=True)
class SolverConfig:
    max_step_internal: float = 1.0
    rtol: float = 1e-9
    atol: float = 1e-11


def coupling_matrix(case, seed=43):
    topology = {"uncoupled": "nointeraction", "chain": "broken_chain", "dense": "dense"}[case]
    return static.coupling_matrix({"rows": 1, "columns": 8, "topology": topology, "seed": seed})


def initialize_rng(seed=43):
    return static.initialize_rng({"rows": 1, "columns": 8, "seed": seed})


def components(h):
    remaining = set(range(len(h)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, group = [start], [start]
        while todo:
            i = todo.pop()
            for j in sorted(remaining):
                if h[i, j] != 0:
                    remaining.remove(j)
                    todo.append(j)
                    group.append(j)
        groups.append(sorted(group))
    return groups


def generate_rows(h, rng, count):
    """Sample the eight-qubit control and frequency fields in row-major draw order."""
    flux = np.empty((count, 8))
    nominal = np.empty_like(flux)
    amplitude = np.empty_like(flux)
    carrier_error = np.empty_like(flux)
    duration = np.empty(count)
    for row in range(count):
        flux[row] = rng.normal(0., .25, 8)
        nominal[row] = rng.normal((.06 + .08)/2, (.08 - .06)/6, 8)
        amplitude[row] = nominal[row] + rng.normal(0., .005, 8)
        carrier_error[row] = rng.normal(0., .02, 8)
        duration[row] = rng.uniform(120., 120.)
    bare = bare_response(flux)
    frequency = frequency_batch(bare, h)
    return {"flux": flux, "omega_d_clean": bare, "omega_q_clean": frequency,
            "omega_d_noisy": bare + carrier_error, "omega_q_noisy": frequency.copy(),
            "Omega_ideal": nominal, "Omega_real": amplitude, "phi": np.zeros_like(flux),
            "Tg": duration}


def unified_pulse_batch(data, h, solver=SolverConfig()):
    if not np.all(data["phi"] == 0):
        raise ValueError("This declared protocol requires all phases to be zero")
    count, n = data["flux"].shape
    result = {"F": np.empty((count, n)),
              "gram_residual": np.zeros(count), "channel_tp_residual": np.zeros(count),
              "integration_nfev": np.zeros(count, dtype=np.int64),
              "integration_steps": np.zeros(count, dtype=np.int64),
              "integration_rejections": np.zeros(count, dtype=np.int64)}
    for sites in components(h):
        sites = np.asarray(sites, dtype=int)
        model = SelectedPulse(h[np.ix_(sites, sites)])
        q, bare = data["omega_q_clean"][:, sites], data["omega_d_clean"][:, sites]
        drive, amplitude = data["omega_d_noisy"][:, sites], data["Omega_real"][:, sites]
        internal_duration = data["Tg"] / (2*np.pi)
        references = target_unitaries(q-bare, data["Omega_ideal"][:, sites], internal_duration[:, None])
        for row in range(count):
            propagated = model.propagate(bare[row], drive[row], amplitude[row], internal_duration[row],
                max_step_ns=solver.max_step_internal, rtol=solver.rtol, atol=solver.atol)
            four, _, tp = channel_metrics(propagated["columns"], references[row], len(sites))
            result["F"][row, sites] = four
            result["gram_residual"][row] = max(result["gram_residual"][row], propagated["gram_residual"])
            result["channel_tp_residual"][row] = max(result["channel_tp_residual"][row], tp)
            for target, source in [("integration_nfev", "nfev"), ("integration_steps", "accepted_steps"),
                                   ("integration_rejections", "rejected_steps")]:
                result[target][row] += propagated[source]
    if not np.isfinite(result["F"]).all() or np.min(result["F"]) < -1e-8 or np.max(result["F"]) > 1+1e-8:
        raise ValueError("Invalid four-probe output")
    return result
