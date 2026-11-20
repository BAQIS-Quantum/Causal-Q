#!/usr/bin/env python3
"""Compare clean and noisy NN4 endpoint-CMI rankings on the same five datasets."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from flux_robustness_scores import (ESTIMATOR_SEED, endpoint_scores, flux_detection_scan,
                                   mean_sd, score_statistics, sha256_file)
from physics import counts as flux_robustness_acceleration
import numpy as np


N = 20_000
REPEATS = tuple(range(5))
KS = (3, 30)
SIGMA_GHZ = 0.0005
METRICS = ('average_precision', 'spearman_nonzero', 'spearman_all')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run_cell(root, repeat, source_hashes):
    start = time.perf_counter()
    stem = f'repeat_{repeat:02d}'
    sample_path = root / 'samples' / f'{stem}.npz'
    clean_path = root / 'scores' / 'nn4' / 'iid' / f'{stem}_n{N}.json'
    output_path = root / 'noisy_ranking' / 'cells' / f'{stem}.json'
    input_hash = sha256_file(sample_path)
    provenance = {
        'source_sha256': source_hashes,
        'input_sha256': input_hash,
        'clean_cell_sha256': sha256_file(clean_path),
        'repeat': repeat,
        'sample_count': N,
        'estimator_seed': ESTIMATOR_SEED,
        'k': list(KS),
        'noise_sd_ghz': SIGMA_GHZ,
        'marginal_counter': 'ordered_sweep',
    }
    if output_path.exists():
        cell = read_json(output_path)
        if cell['provenance'] != provenance:
            raise ValueError(f'Existing result uses different inputs or settings: {output_path}')
        return cell

    clean = read_json(clean_path)
    with np.load(sample_path, allow_pickle=False) as samples:
        flux = samples['flux_iid'][:N]
        clean_frequency = samples['nn4_iid_clean'][:N]
        frequency = clean_frequency + SIGMA_GHZ * samples['noise_iid'][:N]
        network = samples['h_nn4']
        seed = int(samples['sample_seed'])
    indices = np.triu_indices(16, 1)
    pairs = np.column_stack(indices)
    coupling = np.abs(network[indices])
    if (flux.shape != (N, 16) or frequency.shape != flux.shape
            or np.count_nonzero(coupling) != 24 or clean['sample_seed'] != seed
            or clean['provenance']['input_sha256'] != input_hash
            or clean['provenance']['estimator_seed'] != ESTIMATOR_SEED
            or clean['sample_count'] != N or clean['repeat'] != repeat
            or clean['pairs'] != pairs.tolist() or clean['coupling_strengths'] != coupling.tolist()):
        raise ValueError(f'Archived clean data do not match the noisy inputs for repeat {repeat}')

    clean_scores = np.array([clean['by_k'][str(k)]['scores_bits'] for k in KS])
    noisy_scores = endpoint_scores(flux, frequency, pairs, KS, accelerated=True)

    by_k = {}
    for index, k in enumerate(KS):
        clean_metrics = score_statistics(clean_scores[index], coupling, [])
        noisy_metrics = score_statistics(noisy_scores[index], coupling, [])
        by_k[str(k)] = {}
        for metric in METRICS:
            old = clean_metrics[metric]
            new = noisy_metrics[metric]
            by_k[str(k)][metric] = {'clean': old, 'noisy': new, 'noisy_minus_clean': new - old}
        by_k[str(k)]['noisy_scores_bits'] = noisy_scores[index].tolist()
    cell = {
        'schema': 'causal-q-flux-noisy-ranking-cell-v1',
        'provenance': provenance,
        'sample_seed': seed,
        'pairs': pairs.tolist(),
        'coupling_strengths': coupling.tolist(),
        'by_k': by_k,
        'elapsed_seconds': time.perf_counter() - start,
    }
    flux_detection_scan.atomic_json(output_path, cell)
    return cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    screen = read_json(root / 'screen_settings' / 'screen' / 'summary.json')
    condition = screen['protocol']['conditions']['iid_noise']
    if condition != {'flux': 'flux_iid', 'frequency': 'nn4_iid_clean', 'rho': 0., 'sigma_ghz': SIGMA_GHZ}:
        raise ValueError('The archived screen uses a different observation-noise condition')
    sources = [Path(__file__), Path(flux_detection_scan.__file__),
               Path(flux_robustness_acceleration.__file__), Path(__file__).with_name('flux_robustness_scores.py')]
    source_hashes = {p.name: sha256_file(p) for p in sources}
    cells = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(run_cell, root, repeat, source_hashes) for repeat in REPEATS]
        for finished in as_completed(tasks):
            cell = finished.result()
            cells.append(cell)
            print(f"{len(cells)}/{len(tasks)} repeat={cell['provenance']['repeat']} "
                  f"({cell['elapsed_seconds']:.1f} s)", flush=True)
    cells.sort(key=lambda cell: cell['provenance']['repeat'])
    if len({cell['sample_seed'] for cell in cells}) != len(REPEATS):
        raise ValueError('Repeats must use distinct sample seeds')
    by_k = {
        str(k): {metric: {scenario: mean_sd([cell['by_k'][str(k)][metric][scenario] for cell in cells])
                          for scenario in ('clean', 'noisy', 'noisy_minus_clean')}
                 for metric in METRICS}
        for k in KS
    }
    flux_detection_scan.atomic_json(root / 'noisy_ranking' / 'summary.json', {
        'schema': 'causal-q-flux-noisy-ranking-summary-v1', 'sample_count': N,
        'rho': 0., 'noise_sd_ghz': SIGMA_GHZ, 'sample_seeds': [cell['sample_seed'] for cell in cells],
        'by_k': by_k,
    })


if __name__ == '__main__':
    main()
