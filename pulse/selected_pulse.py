"""Selected-column integration of the manuscript's Pauli-Z RWA convention.

All 2**n output amplitudes are retained. Only the vacuum and the n single-bit
input columns are needed for the induced one-qubit channels with spectators
initially |0>. This is not a truncation to the single-excitation subspace.
The engine uses cyclic frequency and time in reciprocal units. Fig. 4 passes
T_code/(2*pi) as its time coordinate. The fixed-gate benchmark uses GHz and ns.
The Hamiltonian is
  K(t) = sum (f0-nu) Z/2 + sum Omega X/2
       + sum g/2 [cos(2*pi*(nu_i-nu_j)*t)(XX+YY)
                   + sin(2*pi*(nu_i-nu_j)*t)(XY-YX)].
The target unitaries and probe states define the score for each induced channel.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.integrate._ivp.dop853_coefficients import A, B, C, E3, E5


@njit(cache=True)
def _rhs(t, state, diagonal, amplitude, masks, pair_i, pair_j, pair_g, rates):
    dim, count = state.shape
    result = np.empty_like(state)
    for row in range(dim):
        for col in range(count):
            result[row, col] = diagonal[row] * state[row, col]
    for i in range(len(masks)):
        mask = masks[i]
        value = amplitude[i] / 2
        for row in range(dim):
            for col in range(count):
                result[row, col] += value * state[row ^ mask, col]
    for p in range(len(pair_g)):
        mi, mj = masks[pair_i[p]], masks[pair_j[p]]
        positive = pair_g[p] * np.exp(1j * rates[p] * t)
        negative = np.conj(positive)
        for row in range(dim):
            bi, bj = bool(row & mi), bool(row & mj)
            if bi == bj:
                continue
            value = negative if bi else positive
            source = row ^ mi ^ mj
            for col in range(count):
                result[row, col] += value * state[source, col]
    return (-2j * np.pi) * result


@njit(cache=True)
def _integrate(initial, diagonal, amplitude, masks, pi, pj, pg, rates,
               duration, max_step, rtol, atol, coeff_a, coeff_b, coeff_c,
               coeff_e3, coeff_e5):
    """DOP853 with the SciPy tableau and embedded error controller, in nopython."""
    state = initial.copy()
    stages = np.empty((13, state.shape[0], state.shape[1]), np.complex128)
    stages[0] = _rhs(0., state, diagonal, amplitude, masks, pi, pj, pg, rates)
    time, step, nfev, accepted, rejected = 0., max_step, 1, 0, 0
    was_rejected = False
    while time < duration:
        step = min(step, max_step, duration - time)
        if step < 1e-12:
            raise RuntimeError("DOP853 step underflow")
        for s in range(1, 12):
            trial = state.copy()
            for j in range(s):
                if coeff_a[s, j] != 0:
                    trial += (step * coeff_a[s, j]) * stages[j]
            stages[s] = _rhs(time + coeff_c[s] * step, trial,
                             diagonal, amplitude, masks, pi, pj, pg, rates)
        candidate = state.copy()
        for j in range(12):
            if coeff_b[j] != 0:
                candidate += (step * coeff_b[j]) * stages[j]
        stages[12] = _rhs(time + step, candidate, diagonal, amplitude,
                          masks, pi, pj, pg, rates)
        nfev += 12
        e5_norm, e3_norm = 0., 0.
        for row in range(state.shape[0]):
            for col in range(state.shape[1]):
                e5, e3 = 0j, 0j
                for j in range(13):
                    e5 += coeff_e5[j] * stages[j, row, col]
                    e3 += coeff_e3[j] * stages[j, row, col]
                scale = atol + rtol * max(abs(state[row, col]), abs(candidate[row, col]))
                e5_norm += abs(e5 / scale) ** 2
                e3_norm += abs(e3 / scale) ** 2
        denominator = e5_norm + .01 * e3_norm
        error = 0. if denominator == 0 else step * e5_norm / np.sqrt(denominator * state.size)
        if error < 1:
            state = candidate
            time += step
            stages[0] = stages[12]
            factor = 10. if error == 0 else min(10., .9 * error ** (-1. / 8.))
            if was_rejected:
                factor = min(1., factor)
            step *= factor
            was_rejected = False
            accepted += 1
        else:
            step *= max(.2, .9 * error ** (-1. / 8.))
            was_rejected = True
            rejected += 1
    return state, nfev, accepted, rejected


def target_unitaries(detuning, amplitude, duration_ns):
    """exp[-i*2*pi*T*(detuning*Z+amplitude*X)/2], site by site."""
    detuning, amplitude = np.broadcast_arrays(np.asarray(detuning, float), np.asarray(amplitude, float))
    radius = np.hypot(detuning, amplitude)
    angle = np.pi * duration_ns * radius
    scale = np.where(radius == 0, np.pi * duration_ns, np.sin(angle) / np.where(radius == 0, 1, radius))
    target = np.zeros(detuning.shape + (2, 2), complex)
    target[..., 0, 0] = np.cos(angle) - 1j * scale * detuning
    target[..., 1, 1] = np.cos(angle) + 1j * scale * detuning
    target[..., 0, 1] = target[..., 1, 0] = -1j * scale * amplitude
    return target


def channel_metrics(columns, targets, n):
    """Four-probe score, Haar fidelity and trace-preservation residual for the specified targets."""
    columns, targets = np.asarray(columns), np.asarray(targets)
    if columns.shape != (2 ** n, n + 1) or targets.shape != (n, 2, 2):
        raise ValueError("Invalid selected-column or target shape")
    probes = np.array([[1, 0], [0, 1], [1, 1], [1, 1j]], complex)
    probes[2:] /= np.sqrt(2)
    basis = np.arange(2 ** n)
    four, haar, tp = [], [], []
    for i in range(n):
        mask = 1 << (n - i - 1)
        rows = basis[(basis & mask) == 0]
        kraus = np.stack((columns[rows][:, [0, i + 1]],
                          columns[rows | mask][:, [0, i + 1]]), axis=1)
        complete = np.einsum("eji,ejk->ik", kraus.conj(), kraus)
        tp.append(float(np.linalg.norm(complete - np.eye(2), ord=2)))
        expected = np.einsum("ij,pj->pi", targets[i], probes)
        actual = np.einsum("eij,pj->pei", kraus, probes)
        overlaps = np.einsum("pi,pei->pe", expected.conj(), actual)
        four.append(float(np.mean(np.sum(np.abs(overlaps) ** 2, axis=1))))
        traces = np.einsum("ij,eij->e", targets[i].conj(), kraus)
        haar.append(float((np.sum(np.abs(traces) ** 2) + 2) / 6))
    if max(tp) > 1e-6:
        raise RuntimeError("Induced channel trace-preservation residual exceeds 1e-6")
    return np.array(four), np.array(haar), max(tp)


@dataclass
class SelectedPulse:
    coupling_ghz: np.ndarray

    def __post_init__(self):
        self.coupling_ghz = np.asarray(self.coupling_ghz, float)
        self.n = len(self.coupling_ghz)
        if self.coupling_ghz.shape != (self.n, self.n) or not np.isfinite(self.coupling_ghz).all():
            raise ValueError("Coupling must be a finite square matrix")
        if not np.allclose(self.coupling_ghz, self.coupling_ghz.T, rtol=0, atol=1e-14) or np.any(np.diag(self.coupling_ghz)):
            raise ValueError("Coupling must be symmetric with zero diagonal")
        self.masks = (1 << np.arange(self.n - 1, -1, -1)).astype(np.int64)
        self.basis = np.arange(1 << self.n)
        self.z = 1 - 2 * ((self.basis[:, None] & self.masks) != 0)
        self.pi, self.pj = np.where(np.triu(self.coupling_ghz, 1) != 0)
        self.pg = self.coupling_ghz[self.pi, self.pj].copy()
        self.initial = np.eye(1 << self.n, dtype=complex)[:, np.r_[0, self.masks]].copy()

    def propagate(self, bare, drive, amplitude, duration_ns, *, max_step_ns=.1,
                  rtol=1e-9, atol=1e-11):
        arrays = [np.asarray(value, float) for value in (bare, drive, amplitude)]
        if any(a.shape != (self.n,) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("Inputs must be finite site vectors")
        bare, drive, amplitude = arrays
        if not np.isfinite(duration_ns) or duration_ns <= 0:
            raise ValueError("Duration must be finite and positive")
        if len(self.pg) == 0:
            local = target_unitaries(bare - drive, amplitude, duration_ns)
            columns = []
            for index in np.r_[0, self.masks]:
                column = np.ones(1, complex)
                for i, mask in enumerate(self.masks):
                    column = np.kron(column, local[i, :, int(bool(index & mask))])
                columns.append(column)
            output = np.stack(columns, axis=1)
            nfev, accepted, rejected = 0, 0, 0
        else:
            rates = 2 * np.pi * (drive[self.pi] - drive[self.pj])
            max_rate = np.max(np.abs(rates))
            step = min(max_step_ns, 2 * np.pi / (16 * max_rate)) if max_rate else max_step_ns
            output, nfev, accepted, rejected = _integrate(
                self.initial, self.z @ (bare - drive) / 2, amplitude, self.masks,
                self.pi, self.pj, self.pg, rates, duration_ns, step, rtol, atol,
                A[:12, :12].copy(), B.copy(), C[:12].copy(), E3.copy(), E5.copy())
        residual = float(np.linalg.norm(output.conj().T @ output - np.eye(self.n + 1), ord=2))
        if residual > 1e-6:
            raise RuntimeError("Selected-column Gram residual exceeds 1e-6")
        return {"columns": output, "gram_residual": residual, "nfev": int(nfev),
                "accepted_steps": int(accepted), "rejected_steps": int(rejected)}
