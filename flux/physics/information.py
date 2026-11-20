"""Nearest-neighbor MI and endpoint-conditioned flux pair scores, in bits."""
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
import hashlib
from pathlib import Path
import random
import numpy as np
from npeet import entropy_estimators

_RNG_LOCK = RLock()

def _load_estimator():
    return entropy_estimators


@dataclass(frozen=True)
class InformationConfig:
    mi_k: int = 3
    cmi_k: int = 30
    seed: int = 43

    def __post_init__(self):
        for name in ("mi_k", "cmi_k", "seed"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            object.__setattr__(self, name, int(value))
        if self.seed >= 2 ** 32:
            raise ValueError("seed must be less than 2**32")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _seeded_estimation(seed):
    # Seed once per statistic matrix, not once per pair. This matches the
    # sampling convention for both CMI and display MI.
    with _RNG_LOCK:
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            np.random.seed(seed)
            random.seed(seed)
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def _finite_score(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("NPEET returned a non-finite information estimate")
    return value


def compute_information_matrices(flux, frequency, config=InformationConfig(), *, mij_only=False):
    """Return full Mij and optional full directed MI arrays, both in bits.

    Mij is averaged CMI between two controls conditioned on each clean response.
    MI[i,j] is I(flux_i;omega_q_clean_j). No PC edge selection is performed.
    """
    estimator = _load_estimator()
    n = flux.shape[1]
    mij = np.zeros((n, n), dtype=float)
    components = []
    with _seeded_estimation(config.seed):
        for i in range(n):
            for j in range(i + 1, n):
                given_i = _finite_score(estimator.cmi(flux[:, [i]], flux[:, [j]], frequency[:, [i]],
                                                     k=config.cmi_k, base=2))
                given_j = _finite_score(estimator.cmi(flux[:, [i]], flux[:, [j]], frequency[:, [j]],
                                                     k=config.cmi_k, base=2))
                mij[i, j] = mij[j, i] = (given_i + given_j) / 2
                components.append({"i": i, "j": j, "cmi_given_frequency_i_bits": given_i,
                                   "cmi_given_frequency_j_bits": given_j})
    mi = None
    if not mij_only:
        mi = np.empty((n, n), dtype=float)
        with _seeded_estimation(config.seed):
            for i in range(n):
                for j in range(n):
                    mi[i, j] = _finite_score(estimator.mi(flux[:, [i]], frequency[:, [j]],
                                                        k=config.mi_k, base=2))
    return mij, mi, components
