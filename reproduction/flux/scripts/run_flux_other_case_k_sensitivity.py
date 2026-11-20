#!/usr/bin/env python3
"""Scan endpoint-CMI neighbor counts for the fixed four-qubit, NN8, and dense cases."""

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


CASES = {
    'four': (20_000, 4, 6),
    'nn8': (40_000, 16, 42),
    'dense': (40_000, 16, 120),
}
REPEATS = tuple(range(5))
NEW_K = (3, 6, 10, 20)
KS = NEW_K + (30,)
METRICS = ('average_precision', 'spearman_nonzero', 'spearman_all')


def run_cell(root, case, repeat, source_hashes):
    start = time.perf_counter()
    count, width, coupled_count = CASES[case]
    stem = f'repeat_{repeat:02d}'
    sample_path = root / 'samples' / f'{stem}.npz'
    endpoint_path = root / 'scores' / case / 'iid' / f'{stem}_n{count}.json'
    output_path = root / 'k_sensitivity_other_cases' / 'cells' / case / f'{stem}_n{count}.json'
    provenance = {
        'source_sha256': source_hashes, 'input_sha256': sha256_file(sample_path),
        'endpoint_cell_sha256': sha256_file(endpoint_path),
        'case': case, 'repeat': repeat, 'sample_count': count,
        'estimator_seed': ESTIMATOR_SEED, 'k': list(KS), 'archived_k': [30],
        'marginal_counter': 'ordered_sweep',
    }
    if output_path.is_file():
        cell = json.loads(output_path.read_text(encoding='utf-8'))
        if cell['provenance'] != provenance:
            raise ValueError(f'Existing result uses different inputs or settings: {output_path}')
        return cell

    archived = json.loads(endpoint_path.read_text(encoding='utf-8'))
    with np.load(sample_path, allow_pickle=False) as samples:
        flux = samples['flux_iid'][:count, :width]
        frequency = samples[f'{case}_iid_clean'][:count]
        network = samples[f'h_{case}']
        seed = int(samples['sample_seed'])
    indices = np.triu_indices(width, 1)
    pairs = np.column_stack(indices)
    coupling = np.abs(network[indices])
    if (flux.shape != (count, width) or frequency.shape != flux.shape
            or network.shape != (width, width) or np.count_nonzero(coupling) != coupled_count
            or archived['sample_seed'] != seed
            or archived['provenance']['input_sha256'] != provenance['input_sha256']
            or archived['provenance']['estimator_seed'] != ESTIMATOR_SEED
            or (archived['case'], archived['condition']) != (case, 'iid')
            or archived['sample_count'] != count or archived['repeat'] != repeat
            or archived['pairs'] != pairs.tolist() or archived['coupling_strengths'] != coupling.tolist()):
        raise ValueError(f'Archived endpoint data do not match {sample_path} ({case}, N={count})')

    archived_scores = np.array(archived['by_k']['30']['scores_bits'])
    new_scores = endpoint_scores(flux, frequency, pairs, NEW_K, accelerated=True)

    by_k = {}
    for k, scores in zip(KS, np.vstack((new_scores, archived_scores)), strict=True):
        statistics = score_statistics(scores, coupling, [])
        metrics = {key: statistics[key] for key in METRICS}
        by_k[str(k)] = {**metrics, 'scores_bits': statistics['scores_bits']}
    cell = {
        'schema': 'causal-q-flux-other-case-k-sensitivity-cell-v1', 'provenance': provenance,
        'sample_seed': seed, 'pairs': pairs.tolist(), 'coupling_strengths': coupling.tolist(),
        'by_k': by_k, 'elapsed_seconds': time.perf_counter() - start,
    }
    flux_detection_scan.atomic_json(output_path, cell)
    return cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    sources = [Path(__file__), Path(flux_detection_scan.__file__),
               Path(flux_robustness_acceleration.__file__), Path(__file__).with_name('flux_robustness_scores.py')]
    source_hashes = {path.name: sha256_file(path) for path in sources}
    cells = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(run_cell, root, case, repeat, source_hashes): (case, repeat)
                 for case in ('nn8', 'dense', 'four') for repeat in REPEATS}
        for finished in as_completed(tasks):
            case, repeat = tasks[finished]
            cell = finished.result()
            cells[case, repeat] = cell
            print(f"{len(cells)}/{len(tasks)} {case} repeat={repeat} ({cell['elapsed_seconds']:.1f} s)",
                  flush=True)

    seeds = [cells['four', repeat]['sample_seed'] for repeat in REPEATS]
    if len(set(seeds)) != len(REPEATS):
        raise ValueError('Repeats must use distinct sample seeds')
    groups = []
    for case, (count, width, coupled_count) in CASES.items():
        rows = [cells[case, repeat] for repeat in REPEATS]
        if ([cell['sample_seed'] for cell in rows] != seeds
                or any(cell['coupling_strengths'] != rows[0]['coupling_strengths'] for cell in rows)):
            raise ValueError(f'Inconsistent seed or coupling matrix for {case}')
        by_k = {str(k): {metric: mean_sd([row['by_k'][str(k)][metric] for row in rows])
                         for metric in METRICS} for k in KS}
        groups.append({'case': case, 'sample_count': count, 'repeats': list(REPEATS),
                       'pair_count': width * (width - 1) // 2, 'coupled_pair_count': coupled_count, 'by_k': by_k})
    flux_detection_scan.atomic_json(root / 'k_sensitivity_other_cases' / 'summary.json', {
        'schema': 'causal-q-flux-other-case-k-sensitivity-summary-v1',
        'rho': 0.0, 'noise_sd_ghz': 0.0, 'sample_seeds': seeds, 'k': list(KS), 'groups': groups,
    })


if __name__ == '__main__':
    main()
