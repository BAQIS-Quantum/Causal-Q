#!/usr/bin/env python3
"""Sweep flux-input correlation for the same eight weak NN4 pairs and null controls."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from flux_robustness_scores import (ESTIMATOR_SEED, WEAK_PAIRS, endpoint_scores,
                                    exact_marginal_counts, score_statistics)
from generate_flux_robustness import (CASES, DEVICE_SEED, PHASE_STD, SEEDS, bare_response,
                                      case_config, coupling_matrix, solve_rows)
from physics import counts as flux_robustness_acceleration, knn as flux_detection_scan
from physics.information import sha256_file
import numpy as np


RHO_LEVELS = {"iid": 0.0, "rho005": 0.05, "rho010": 0.1, "rho020": 0.2, "rho03": 0.3}
ARCHIVED_CONDITIONS = ("iid", "rho03")
N = 20_000
K = 30


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def flux_for_rho(z, shared, rho):
    return PHASE_STD * (np.sqrt(1 - rho) * z + np.sqrt(rho) * shared)


def selected_endpoint_scores(flux, frequency, all_pairs, selected, k=K):
    rng = np.random.RandomState(ESTIMATOR_SEED)
    positions = {pair: index for index, pair in enumerate(selected)}
    scores = np.empty(len(selected))
    for pair in all_pairs:
        if pair not in positions:
            # Two x/y jitter arrays for each of the two conditioning terms.
            rng.random_sample(4 * len(flux))
            continue
        i, j = pair
        x, y = flux[:, [i]], flux[:, [j]]
        given_i = flux_detection_scan.cmi_many_k(x, y, frequency[:, [i]], (k,), rng,
                                                 marginal_counts=exact_marginal_counts)[0]
        given_j = flux_detection_scan.cmi_many_k(x, y, frequency[:, [j]], (k,), rng,
                                                 marginal_counts=exact_marginal_counts)[0]
        scores[positions[pair]] = (given_i + given_j) / 2
    return scores


def compare_models(sample_key, null, nn4):
    return {
        **sample_key,
        "null_scores_bits": null.tolist(),
        "nn4_scores_bits": nn4.tolist(),
        "null_mean_bits": float(null.mean()),
        "nn4_mean_bits": float(nn4.mean()),
        "paired_mean_bits": float((nn4 - null).mean()),
        "ranges_overlap": bool(max(null.min(), nn4.min()) <= min(null.max(), nn4.max())),
    }


def run_repeat(root, output_directory, repeat, manifest_record, source_hashes, selected, k):
    start = time.perf_counter()
    stem = f"repeat_{repeat:02d}"
    path = root / "samples" / f"{stem}.npz"
    sample_key = {"repeat": repeat, "sample_seed": SEEDS[repeat], "sample_sha256": sha256_file(path)}
    if (manifest_record["file"] != path.name or manifest_record["sha256"] != sample_key["sample_sha256"]
            or manifest_record["sample_seed"] != sample_key["sample_seed"]):
        raise ValueError(f"Sample does not match the frozen manifest: {path}")
    archive = {(cond, model): root / "scores" / model / cond / f"{stem}_n{N}.json"
               for cond in ARCHIVED_CONDITIONS for model in ("null", "nn4")}
    provenance = {**sample_key, "rho_levels": RHO_LEVELS, "sample_count": N, "k": k,
                  "selected_pairs": [list(pair) for pair in selected],
                  "source_sha256": source_hashes,
                  "archived_score_sha256": {f"{cond}/{model}": sha256_file(file)
                                            for (cond, model), file in archive.items()}}
    output = output_directory / "cells" / f"{stem}.json"
    if output.exists():
        cell = read_json(output)
        if cell["provenance"] != provenance:
            raise ValueError(f"Checkpoint uses different inputs or settings: {output}")
        return cell

    name, _, width = CASES["nn4"]
    h = coupling_matrix(case_config(name, N, seed=DEVICE_SEED))
    triangle = np.triu_indices(width, 1)
    all_pairs = list(zip(*triangle))
    selected_indices = [all_pairs.index(pair) for pair in selected]
    coupling = np.abs(h[triangle])
    nonzero = np.flatnonzero(coupling)
    weak_indices = nonzero[np.argsort(coupling[nonzero], kind="stable")[:WEAK_PAIRS]]
    if [all_pairs[i] for i in weak_indices] != selected:
        raise ValueError("The archived pairs are no longer the weakest eight fixed NN4 pairs")

    rng = np.random.default_rng(sample_key["sample_seed"])
    z = rng.standard_normal((max(case[1] for case in CASES.values()), width))
    shared = rng.standard_normal((N, 1))
    phases = {condition: flux_for_rho(z[:N], shared, rho) for condition, rho in RHO_LEVELS.items()}
    records = {}
    with np.load(path, allow_pickle=False) as samples:
        if int(samples["sample_seed"]) != sample_key["sample_seed"]:
            raise ValueError(f"Unexpected sample seed: {path}")
        np.testing.assert_array_equal(samples["h_nn4"], h)
        np.testing.assert_array_equal(samples["flux_iid"], PHASE_STD * z)
        np.testing.assert_array_equal(samples["flux_rho03"], phases["rho03"])
        for condition, flux in phases.items():
            null_frequency = bare_response(flux)
            nn4_frequency = solve_rows(null_frequency, h)
            scores, endpoint_metrics = [], {}
            for model, frequency in (("null", null_frequency), ("nn4", nn4_frequency)):
                if condition not in ARCHIVED_CONDITIONS:
                    values = selected_endpoint_scores(flux, frequency, all_pairs, selected, k)
                else:
                    np.testing.assert_array_equal(samples[f"{model}_{condition}_clean"], frequency)
                    archived = read_json(archive[condition, model])
                    if (archived["sample_seed"] != sample_key["sample_seed"]
                            or archived["provenance"]["input_sha256"] != sample_key["sample_sha256"]
                            or archived["provenance"]["estimator_seed"] != ESTIMATOR_SEED
                            or archived["sample_count"] != N or archived["repeat"] != repeat
                            or archived["pairs"] != [list(pair) for pair in all_pairs]):
                        raise ValueError(f"Unmatched archived score: {archive[condition, model]}")
                    if str(k) in archived["by_k"]:
                        metrics = archived["by_k"][str(k)]
                        values = np.array(metrics["scores_bits"])[selected_indices]
                        np.testing.assert_array_equal(
                            selected_endpoint_scores(flux, frequency, all_pairs, selected, k), values
                        )
                    else:
                        full = endpoint_scores(flux, frequency, all_pairs, (k,), accelerated=True)[0]
                        labels = coupling if model == "nn4" else np.zeros_like(coupling)
                        metrics = score_statistics(full, labels, selected_indices if model == "nn4" else [])
                        values = full[selected_indices]
                    endpoint_metrics[model] = metrics
                scores.append(values)
            records[condition] = compare_models(sample_key, *scores)
            if endpoint_metrics:
                records[condition]["all_pairs"] = endpoint_metrics
    cell = {"provenance": provenance, "conditions": records, "elapsed_seconds": time.perf_counter() - start}
    flux_detection_scan.atomic_json(output, cell)
    return cell


def distribution(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--out", type=Path, help="Output directory for this rho sweep")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if not 0 < args.k < N:
        parser.error("--k must be positive and below the sample count")
    root = args.run_directory.resolve()
    output_directory = (args.out or root / "rho_sweep").resolve()
    if output_directory == root / "rho_sweep" and args.k != K:
        parser.error("Use --out to keep the archived k=30 sweep separate")
    manifest = read_json(root / "samples" / "manifest.json")
    archived = read_json(root / "scores" / "nn4" / "iid" / f"repeat_00_n{N}.json")
    selected = list(map(tuple, archived["weak_pairs"]))
    if len(set(selected)) != WEAK_PAIRS or len(selected) != WEAK_PAIRS:
        raise ValueError("Expected eight distinct weak NN4 pairs")
    script = Path(__file__).resolve()
    sources = (script, script.with_name("generate_flux_robustness.py"),
               script.with_name("flux_robustness_scores.py"), Path(flux_detection_scan.__file__),
               Path(flux_robustness_acceleration.__file__))
    source_hashes = manifest["source_sha256"] | {p.name: sha256_file(p) for p in sources}
    cells = []
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        tasks = [workers.submit(run_repeat, root, output_directory, repeat, row, source_hashes, selected, args.k)
                 for repeat, row in enumerate(manifest["datasets"])]
        for finished in as_completed(tasks):
            cell = finished.result()
            cells.append(cell)
            print(f"{len(cells)}/{len(tasks)} repeat={cell['provenance']['repeat']} "
                  f"({cell['elapsed_seconds']:.1f} s)", flush=True)
    cells.sort(key=lambda cell: cell["provenance"]["repeat"])
    if [cell["provenance"]["sample_seed"] for cell in cells] != list(SEEDS):
        raise ValueError("The rho sweep must use the original five seeds")
    groups = {}
    for condition, rho in RHO_LEVELS.items():
        records = [cell["conditions"][condition] for cell in cells]
        groups[condition] = {
            "rho": rho,
            "null": distribution([r["null_mean_bits"] for r in records]),
            "nn4": distribution([r["nn4_mean_bits"] for r in records]),
            "paired_delta": distribution([r["paired_mean_bits"] for r in records]),
            "overlapping_ranges": sum(r["ranges_overlap"] for r in records),
            "repeats": records,
        }
    verified_endpoints = sum(
        str(args.k) in read_json(root / "scores" / model / cond / f"repeat_{r:02d}_n{N}.json")["by_k"]
        for r in range(len(SEEDS)) for model in ("null", "nn4") for cond in ARCHIVED_CONDITIONS
    )
    flux_detection_scan.atomic_json(output_directory / "summary.json", {
        "schema": "causal-q-matched-weak-pair-rho-sweep-v1", "selected_pairs": selected,
        "sample_count": N, "k": args.k, "conditions": groups,
        "validation": {"archived_conditions": list(ARCHIVED_CONDITIONS),
                       "selected_scores_equal_bitwise": verified_endpoints * WEAK_PAIRS,
                       "archived_control_and_frequency_matrices_equal_bitwise": True},
    })


if __name__ == "__main__":
    main()
