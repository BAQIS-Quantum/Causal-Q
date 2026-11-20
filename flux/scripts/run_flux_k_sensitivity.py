#!/usr/bin/env python3
"""Scan NN4 endpoint-CMI neighbor counts on the archived clean flux rows."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from flux_robustness_scores import (
    ESTIMATOR_SEED, endpoint_scores, flux_detection_scan, mean_sd,
    score_statistics, sha256_file,
)
from physics import counts as flux_robustness_acceleration
import numpy as np


SAMPLE_COUNTS = (5_000, 20_000)
REPEATS = tuple(range(5))
ARCHIVED_K = (3, 30)
NEW_K = (6, 10, 20)
METRICS = ('average_precision', 'spearman_nonzero', 'spearman_all')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run_cell(root, repeat, count, source_hashes):
    start = time.perf_counter()
    stem = f'repeat_{repeat:02d}'
    sample_path = root / 'samples' / f'{stem}.npz'
    endpoint_path = root / 'scores' / 'nn4' / 'iid' / f'{stem}_n{count}.json'
    output_path = root / 'k_sensitivity' / 'cells' / f'{stem}_n{count}.json'
    provenance = {
        'source_sha256': source_hashes, 'input_sha256': sha256_file(sample_path),
        'endpoint_cell_sha256': sha256_file(endpoint_path),
        'repeat': repeat, 'sample_count': count, 'estimator_seed': ESTIMATOR_SEED,
        'archived_k': list(ARCHIVED_K), 'new_k': list(NEW_K),
        'marginal_counter': 'ordered_sweep',
    }
    if output_path.is_file():
        cell = read_json(output_path)
        if cell['provenance'] != provenance:
            raise ValueError(f'Existing result uses different inputs or settings: {output_path}')
        return cell

    archived = read_json(endpoint_path)
    with np.load(sample_path, allow_pickle=False) as samples:
        flux = samples['flux_iid'][:count]
        frequency = samples['nn4_iid_clean'][:count]
        network = samples['h_nn4']
        seed = int(samples['sample_seed'])
    indices = np.triu_indices(16, 1)
    pairs = np.column_stack(indices)
    coupling = np.abs(network[indices])
    if (flux.shape != (count, 16) or frequency.shape != flux.shape
            or np.count_nonzero(coupling) != 24 or archived['sample_seed'] != seed
            or archived['provenance']['input_sha256'] != provenance['input_sha256']
            or archived['provenance']['estimator_seed'] != ESTIMATOR_SEED
            or (archived['case'], archived['condition']) != ('nn4', 'iid')
            or archived['sample_count'] != count or archived['repeat'] != repeat
            or archived['pairs'] != pairs.tolist() or archived['coupling_strengths'] != coupling.tolist()):
        raise ValueError(f'Archived endpoint data do not match {sample_path} at N={count}')

    archived_scores = np.array([archived['by_k'][str(k)]['scores_bits'] for k in ARCHIVED_K])
    new_scores = endpoint_scores(flux, frequency, pairs, NEW_K, accelerated=True)

    by_k = {}
    for k, scores in zip(ARCHIVED_K + NEW_K, np.vstack((archived_scores, new_scores))):
        statistics = score_statistics(scores, coupling, [])
        metrics = {key: statistics[key] for key in METRICS}
        by_k[str(k)] = {**metrics, 'scores_bits': statistics['scores_bits']}
    cell = {
        'schema': 'causal-q-flux-k-sensitivity-cell-v1', 'provenance': provenance,
        'sample_seed': seed, 'pairs': pairs.tolist(), 'coupling_strengths': coupling.tolist(),
        'by_k': by_k, 'elapsed_seconds': time.perf_counter() - start,
    }
    flux_detection_scan.atomic_json(output_path, cell)
    return cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    sources = [Path(__file__), Path(flux_detection_scan.__file__),
               Path(flux_robustness_acceleration.__file__), Path(__file__).with_name('flux_robustness_scores.py')]
    source_hashes = {path.name: sha256_file(path) for path in sources}
    cells = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(run_cell, root, repeat, count, source_hashes): (count, repeat)
                 for count in reversed(SAMPLE_COUNTS) for repeat in REPEATS}
        for finished in as_completed(tasks):
            count, repeat = tasks[finished]
            cell = finished.result()
            cells[count, repeat] = cell
            print(f"{len(cells)}/{len(tasks)} N={count} repeat={repeat} ({cell['elapsed_seconds']:.1f} s)",
                  flush=True)

    seeds = [cells[SAMPLE_COUNTS[0], repeat]['sample_seed'] for repeat in REPEATS]
    if len(set(seeds)) != len(REPEATS):
        raise ValueError('Repeats must use distinct sample seeds')
    reference_couplings = cells[SAMPLE_COUNTS[0], REPEATS[0]]['coupling_strengths']
    for (count, repeat), cell in cells.items():
        if cell['sample_seed'] != seeds[repeat] or cell['coupling_strengths'] != reference_couplings:
            raise ValueError(f'Inconsistent seed or coupling matrix at N={count}, repeat={repeat}')
    ks = sorted(ARCHIVED_K + NEW_K)
    groups = []
    for count in SAMPLE_COUNTS:
        by_k = {str(k): {metric: mean_sd([cells[count, r]['by_k'][str(k)][metric] for r in REPEATS])
                         for metric in METRICS} for k in ks}
        groups.append({'sample_count': count, 'repeats': list(REPEATS), 'by_k': by_k})
    flux_detection_scan.atomic_json(root / 'k_sensitivity' / 'summary.json', {
        'schema': 'causal-q-flux-k-sensitivity-summary-v1', 'case': 'nn4',
        'rho': 0.0, 'noise_sd_ghz': 0.0, 'sample_seeds': seeds, 'k': ks, 'groups': groups,
    })


if __name__ == '__main__':
    main()
