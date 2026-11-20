#!/usr/bin/env python3
"""Evaluate own-flux marginal MI in the matched Fig. 3 uncoupled control."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics.knn import atomic_json
from physics.information import _load_estimator, _seeded_estimation, sha256_file


SAMPLE_COUNTS = (1000, 5000, 10000, 20000)
NEIGHBOR_COUNTS = (3, 130)
NOISE_LEVELS_GHZ = (0.0, 0.0005)
ESTIMATOR_SEED = 43


def evaluate_repeat(sample_path, neighbors=NEIGHBOR_COUNTS):
    repeat = int(sample_path.stem.removeprefix("repeat_"))
    with np.load(sample_path, allow_pickle=False) as data:
        flux = data["flux_iid"][:max(SAMPLE_COUNTS)]
        clean = data["null_iid_clean"]
        noise = data["noise_iid"]
    expected_shape = (max(SAMPLE_COUNTS), 16)
    if any(array.shape != expected_shape or not np.isfinite(array).all()
           for array in (flux, clean, noise)):
        raise ValueError(f"Expected finite arrays with shape {expected_shape} in {sample_path}")

    estimator = _load_estimator()
    cells = []
    for sigma in NOISE_LEVELS_GHZ:
        observed = clean + sigma * noise
        for n, k in product(SAMPLE_COUNTS, neighbors):
            with _seeded_estimation(ESTIMATOR_SEED):
                by_site = [
                    float(estimator.mi(flux[:n, site], observed[:n, site], k=k, base=2))
                    for site in range(flux.shape[1])
                ]
            cells.append({
                "repeat": repeat,
                "sample_count": n,
                "noise_ghz": sigma,
                "k": k,
                "mi_bits_by_site": by_site,
                "mean_bits": float(np.mean(by_site)),
            })
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Experiment root containing samples/")
    parser.add_argument("--diagnostic-dir", type=Path, help="Output directory for this diagnostic")
    parser.add_argument("--neighbors", type=int, nargs="+", default=NEIGHBOR_COUNTS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 2 or args.workers < 1:
        parser.error("At least two repeats and one worker are required")
    if len(set(args.neighbors)) != len(args.neighbors) or not all(0 < k < min(SAMPLE_COUNTS) for k in args.neighbors):
        parser.error("Neighbor counts must be distinct and below every sample count")
    if tuple(args.neighbors) != NEIGHBOR_COUNTS and args.diagnostic_dir is None:
        parser.error("Use --diagnostic-dir to keep the archived diagnostic separate")

    source_paths = [args.out / "samples" / f"repeat_{i:02d}.npz" for i in range(args.repeats)]
    cells = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_repeat, path, args.neighbors) for path in source_paths]
        for path, future in zip(source_paths, futures, strict=True):
            rows = future.result()
            cells.extend(rows)
            print(f"Completed {path.name}", flush=True)
    groups = []
    for n, k, sigma in product(SAMPLE_COUNTS, args.neighbors, NOISE_LEVELS_GHZ):
        values = [cell["mean_bits"] for cell in cells if
                  (cell["sample_count"], cell["k"], cell["noise_ghz"]) == (n, k, sigma)]
        groups.append({
            "sample_count": n,
            "noise_ghz": sigma,
            "k": k,
            "mean_bits": float(np.mean(values)),
            "sd_bits": float(np.std(values, ddof=1)),
        })
    destination = args.diagnostic_dir or args.out / "diagnostics"
    atomic_json(destination / "summary.json", {
        "schema": "causal-q-flux-robustness-self-mi-v1",
        "estimator": "NPEET marginal MI (base 2 and pinned tie jitter)",
        "estimator_seed": ESTIMATOR_SEED,
        "sources": {str(path.relative_to(args.out)): sha256_file(path) for path in source_paths},
        "cells": cells,
        "groups": groups,
    })


if __name__ == "__main__":
    main()
