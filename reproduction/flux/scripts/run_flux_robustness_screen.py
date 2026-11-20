"""Repeat the Figure 3 flux-to-frequency screen on matched NN4 samples."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

# The jobs parallelize across datasets, rather than within a neighbor query.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physics.knn import atomic_json
from physics.information import _load_estimator, sha256_file
from physics.screening import screen_layer


N_SAMPLES = 20_000
N_REPEATS = 5
N_QUBITS = 16
CONDITIONS = {
    "iid_clean": {"flux": "flux_iid", "frequency": "nn4_iid_clean", "rho": 0., "sigma_ghz": 0.},
    "rho03_clean": {"flux": "flux_rho03", "frequency": "nn4_rho03_clean", "rho": .3, "sigma_ghz": 0.},
    "iid_noise": {"flux": "flux_iid", "frequency": "nn4_iid_clean", "rho": 0., "sigma_ghz": .0005},
}
SCREEN = {"threshold": 0., "mi_k": 130, "cmi_k": 3, "seed": 43, "max_cond_set": 2}
ACCELERATION_SOURCE = ROOT / "physics" / "counts.py"


def score_edges(edges, support):
    offsite = [(i, j) for i, j in edges if i != j]
    positives = int(np.count_nonzero(support))
    candidates = len(support) * (len(support) - 1)
    tp = sum(bool(support[i, j]) for i, j in offsite)
    fp = len(offsite) - tp
    return {
        "candidate_count": candidates, "physical_positive": positives,
        "retained": len(offsite), "own_site_retained": len(edges) - len(offsite),
        "tp": tp, "fp": fp, "fn": positives - tp, "tn": candidates - positives - fp,
        "precision": tp / len(offsite) if offsite else None,
        "recall": tp / positives,
    }


def run_cell(path, repeat, condition, protocol):
    start = time.perf_counter()
    scenario = CONDITIONS[condition]
    with np.load(path, allow_pickle=False) as data:
        support = data["h_nn4"] != 0
        flux = data[scenario["flux"]][:N_SAMPLES]
        frequency = data[scenario["frequency"]][:N_SAMPLES]
        if scenario["sigma_ghz"]:
            frequency = frequency + scenario["sigma_ghz"] * data["noise_iid"][:N_SAMPLES]
    if (support.shape != (N_QUBITS, N_QUBITS) or np.any(support.diagonal())
            or not np.array_equal(support, support.T) or int(support.sum()) != 48
            or flux.shape != (N_SAMPLES, N_QUBITS) or frequency.shape != flux.shape
            or not np.isfinite(flux).all() or not np.isfinite(frequency).all()):
        raise ValueError(f"Malformed NN4 samples in {path}")
    counts_context = nullcontext()
    if "marginal_acceleration" in protocol:
        from physics.counts import accelerate_marginal_counts

        counts_context = accelerate_marginal_counts(_load_estimator())
    with counts_context:
        graph = screen_layer(flux, frequency, "Phi", "f", **protocol["alg1"])
    edges = [(int(a.split("_")[1]), int(b.split("_")[1])) for a, b in graph["edges"]]
    result = {
        "repeat": repeat, "condition": condition, "protocol": protocol,
        "test_count": len(graph["tests"]), "elapsed_seconds": time.perf_counter() - start,
        **score_edges(edges, support),
    }
    return result, graph


def aggregate(cells):
    result = {}
    for condition in CONDITIONS:
        group = [cell for cell in cells if cell["condition"] == condition]
        metrics = {}
        for metric in ("precision", "recall", "tp", "fp", "fn", "tn", "retained"):
            values = [cell[metric] for cell in group if cell[metric] is not None]
            metrics[metric] = {
                "n": len(values), "mean": float(np.mean(values)) if values else None,
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
        result[condition] = metrics
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mi-k", type=int, default=SCREEN["mi_k"])
    parser.add_argument("--cmi-k", type=int, default=SCREEN["cmi_k"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--accelerate-marginal-counts", action="store_true",
                        help="Use exact interval counts for 1D and 2D NPEET marginal spaces")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not (0 < args.mi_k < N_SAMPLES and 0 < args.cmi_k < N_SAMPLES):
        parser.error("--mi-k and --cmi-k must be positive and below the sample count")
    args.out.mkdir(parents=True, exist_ok=True)
    samples = [args.input_dir / f"repeat_{repeat:02d}.npz" for repeat in range(N_REPEATS)]
    hashes = [sha256_file(path) for path in samples]
    estimator = _load_estimator()
    settings = {
        "sample_count": N_SAMPLES,
        "alg1": {**SCREEN, "mi_k": args.mi_k, "cmi_k": args.cmi_k},
        "conditions": CONDITIONS,
        "screen_source_sha256": sha256_file(ROOT / "physics" / "screening.py"),
        "estimator_source_sha256": sha256_file(estimator.__file__),
        "dependencies": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "NPEET")},
    }
    provenance = {**settings, "driver_source_sha256": sha256_file(__file__)}
    if args.accelerate_marginal_counts:
        provenance["marginal_acceleration"] = {
            "source_sha256": sha256_file(ACCELERATION_SOURCE), "numba": version("numba"),
        }
    cells = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {}
        for repeat, (path, sample_sha) in enumerate(zip(samples, hashes)):
            for condition in CONDITIONS:
                stem = f"repeat_{repeat:02d}_{condition}"
                checkpoint = args.out / "cells" / f"{stem}.json"
                trace = args.out / "traces" / f"{stem}.json"
                protocol = {**provenance, "sample_sha256": sample_sha}
                if checkpoint.exists():
                    result = json.loads(checkpoint.read_text(encoding="utf-8"))
                    prior = result["protocol"]
                    accelerated = prior.get("marginal_acceleration")
                    if (any(prior.get(key) != value for key, value in settings.items())
                            or prior.get("sample_sha256") != sample_sha
                            or (accelerated and accelerated["source_sha256"] != sha256_file(ACCELERATION_SOURCE))
                            or result["repeat"] != repeat
                            or result["condition"] != condition or result["trace_sha256"] != sha256_file(trace)):
                        raise ValueError(f"Checkpoint no longer matches protocol: {checkpoint}")
                    cells.append(result)
                    continue
                future = executor.submit(run_cell, path, repeat, condition, protocol)
                pending[future] = (checkpoint, trace)
        for future in as_completed(pending):
            checkpoint, trace = pending[future]
            result, graph = future.result()
            atomic_json(trace, graph)
            result["trace_sha256"] = sha256_file(trace)
            atomic_json(checkpoint, result)
            cells.append(result)
            print(f"{checkpoint.stem}: TP={result['tp']} FP={result['fp']} "
                  f"tests={result['test_count']} ({result['elapsed_seconds']:.1f}s)", flush=True)
    cells.sort(key=lambda result: (result["repeat"], result["condition"]))
    atomic_json(args.out / "summary.json", {
        "schema": "causal-q-flux-robustness-screen-v1", "protocol": provenance,
        "cells": cells, "aggregate": aggregate(cells),
    })


if __name__ == "__main__":
    main()
