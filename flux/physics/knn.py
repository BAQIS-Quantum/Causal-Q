"""Joint nearest-neighbor queries for the endpoint-CMI sweeps."""
import json
from pathlib import Path
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def random_ranking_ap(positives, pairs):
    """Expected AP for a uniformly random strict ranking with fixed support."""
    other_positive_fraction = (positives - 1) / (pairs - 1)
    harmonic = sum(1 / rank for rank in range(1, pairs + 1))
    return other_positive_fraction + (1 - other_positive_fraction) * harmonic / pairs


def cmi_many_k(x, y, z, ks, rng, *, marginal_counts=None):
    """Exact NPEET continuous CMI formula, sharing one joint neighbor query.

    Each k uses the same positive 1e-10 x/y jitter. Passing a RandomState seeded
    once per matrix preserves the estimator draw order for
    every k separately. z is not jittered, matching NPEET's implementation.
    A custom marginal counter receives xz, yz, z, and the exact query radii.
    """
    x, y, z = (np.asarray(v, dtype=float).reshape(len(v), -1) for v in (x, y, z))
    ks = np.asarray(ks, dtype=int)
    if (len(x) != len(y) or len(y) != len(z) or len(x) <= max(ks)
            or np.any(ks < 1) or not all(np.isfinite(v).all() for v in (x, y, z))):
        raise ValueError("Finite equally sized arrays and 1 <= k < sample count are required")
    x = x + 1e-10 * rng.random_sample(x.shape)
    y = y + 1e-10 * rng.random_sample(y.shape)
    xyz, xz, yz = np.c_[x, y, z], np.c_[x, z], np.c_[y, z]
    joint = KDTree(xyz, metric="chebyshev")
    distances = joint.query(xyz, k=int(max(ks)) + 1)[0][:, ks]
    if marginal_counts is None:
        txz, tyz, tz = (KDTree(v, metric="chebyshev") for v in (xz, yz, z))

        def marginal_counts(xz, yz, z, radius):
            return (txz.query_radius(xz, radius, count_only=True),
                    tyz.query_radius(yz, radius, count_only=True),
                    tz.query_radius(z, radius, count_only=True))

    output = []
    for column, k in enumerate(ks):
        radius = distances[:, column] - 1e-15
        nxz, nyz, nz = marginal_counts(xz, yz, z, radius)
        output.append(float((digamma(k) - digamma(nxz).mean()
                             - digamma(nyz).mean() + digamma(nz).mean()) / np.log(2)))
    return np.asarray(output)
