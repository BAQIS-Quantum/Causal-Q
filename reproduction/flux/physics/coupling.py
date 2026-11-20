"""Static flux models in GHz and the seed-43 coupling matrices."""
import numpy as np
from scipy.optimize import linear_sum_assignment

TOPOLOGY = {"figure2_flux": (2, "explicit_four"), "figure3_nn4": (4, "nn4"),
            "figure3_nn8": (4, "nn8"), "figure3_dense": (4, "dense")}

def case_config(case, samples, seed=43):
    side, topology = TOPOLOGY[case]
    return {"columns": side, "rows": side, "topology": topology, "samples": samples, "seed": seed}

def bare_response(flux):
    return 7.0 * np.sqrt(np.cos(flux)**2 + 0.35**2 * np.sin(flux)**2) - 0.22


def coupling_matrix(config):
    n = config["columns"] * config["rows"]
    h = np.zeros((n, n)); rng = np.random.default_rng(config["seed"])
    mode = config["topology"]
    if mode == "explicit_four":
        for (i, j), value in zip([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)], [.001,.020,.014,.016,.008,.018]):
            h[i,j] = h[j,i] = value
    elif mode == "dense":
        a = rng.uniform(0, .02, (n,n)); h = (a + a.T) / 2; np.fill_diagonal(h, 0)
    elif mode == "broken_chain":
        for i, value in enumerate(rng.uniform(0, .03, n-1)):
            if i not in (2, 5):
                h[i,i+1] = h[i+1,i] = value
    elif mode in ("nn4", "nn8"):
        columns, rows = config["columns"], config["rows"]
        for i in range(n):
            r,c = divmod(i, columns)
            offsets = [(-1,0),(1,0),(0,-1),(0,1)]
            if mode == "nn8": offsets += [(-1,-1),(-1,1),(1,-1),(1,1)]
            for dr,dc in offsets:
                if 0 <= r+dr < rows and 0 <= c+dc < columns:
                    j = (r+dr)*columns+c+dc
                    if h[i,j] == 0:
                        h[i,j] = h[j,i] = rng.uniform(0, .02)
    elif mode != "nointeraction":
        raise ValueError("Unsupported topology")
    return h


def initialize_rng(config):
    """Preserve constructor draws for the unused resonator field."""
    rng = np.random.default_rng(config["seed"])
    n = config["columns"] * config["rows"]
    rng.choice([-1,1], size=n); rng.uniform(.05,.1,n)
    return rng


def frequency_batch(bare, h):
    """Independent row-wise stopping, vectorized without branch changes."""
    n = len(h); mask = ~np.eye(n,dtype=bool); h2 = h*h
    denominator = bare[:,:,None] - bare[:,None,:]
    initial_denominator = np.where(mask & (denominator != 0), denominator, np.inf)
    f = bare + np.sum(h2[None,:,:] / initial_denominator, axis=2)
    active = np.ones(len(bare), dtype=bool)
    for _ in range(50):
        if not active.any(): break
        b, current = bare[active], f[active]
        denominator = np.where(mask, current[:,:,None] - b[:,None,:], np.inf)
        sigma = np.sum(h2 / denominator, axis=2)
        derivative = -np.sum(h2 / denominator**2, axis=2)
        updated = current - .7 * (current - b - sigma) / (1 - derivative)
        indices = np.flatnonzero(active)
        f[indices] = updated
        active[indices[np.max(abs(updated-current),axis=1) < 1e-10]] = False
    if not np.isfinite(f).all():
        raise ValueError("Dyson iteration produced nonfinite values; no pole repair applied")
    return f


def exact_spectrum(bare, coupling):
    # Site labels maximize total eigenvector probability; no claim of localization.
    count, n = bare.shape
    assigned = np.empty_like(bare)
    sorted_modes = np.empty_like(bare)
    weight = np.empty_like(bare)
    indices = np.empty((count, n), dtype=int)
    residual_max = 0.
    for row in range(count):
        matrix = np.diag(bare[row]) + coupling
        values, vectors = np.linalg.eigh(matrix)
        sites, modes = linear_sum_assignment(-abs(vectors)**2)
        permutation = np.empty(n, dtype=int)
        permutation[sites] = modes
        assigned[row] = values[permutation]
        sorted_modes[row] = values
        weight[row] = abs(vectors[np.arange(n), permutation])**2
        indices[row] = permutation
        residual_max = max(residual_max, float(np.max(abs(matrix @ vectors - vectors * values))))
    return assigned, sorted_modes, weight, indices, residual_max
