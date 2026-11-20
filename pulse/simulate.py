"""Generate the seeded three eight-qubit pulse ensembles, including all four-probe scores."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from itertools import repeat
from pathlib import Path

import numpy as np

from .model import SAMPLES, SolverConfig, coupling_matrix, generate_rows, initialize_rng, unified_pulse_batch


def evolve(controls, h):
    return {**controls, **unified_pulse_batch(controls, h)}


def metadata(case, n):
    return {"configuration": {"case": "figure4_" + case, "samples": n, "seed": 43,
                              "columns": 8, "rows": 1, "duration_code": 120.0,
                              "flux_std": .25, "nominal_amplitude_mean": .07,
                              "nominal_amplitude_std": (.08 - .06) / 6,
                              "carrier_error_std": .02, "amplitude_error_std": .005},
            "solver": asdict(SolverConfig())}


def simulate(case, path, workers=1, count=None):
    h = coupling_matrix(case)
    rng = initialize_rng()
    n = SAMPLES[case] if count is None else count
    controls = (generate_rows(h, rng, min(128, n-start)) for start in range(0, n, 128))
    if workers == 1:
        chunks = list(map(evolve, controls, repeat(h)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(evolve, controls, repeat(h)))
    arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays, hij=h, grid=np.array([8, 1], dtype=np.int64),
                        metadata_json=np.array(json.dumps(metadata(case, n))))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", choices=tuple(SAMPLES), action="append")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for case in args.case or SAMPLES:
        print(f"Generating {SAMPLES[case]} samples for {case}", flush=True)
        simulate(case, args.out / f"{case}.npz", args.workers)


if __name__ == "__main__":
    main()
