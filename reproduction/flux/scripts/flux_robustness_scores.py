"""Score independent samples from the fixed Figure 2/3 flux models."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from sklearn.neighbors import KDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physics import knn as flux_detection_scan
from physics.information import sha256_file


ESTIMATOR_SEED = 43
WEAK_PAIRS = 8
PROTOCOL = (
    ('nn4', 'iid', 1000, (3, 30)),
    ('nn4', 'iid', 5000, (3, 30)),
    ('nn4', 'iid', 10000, (3, 30)),
    ('nn4', 'iid', 20000, (3, 30)),
    ('null', 'iid', 20000, (30,)),
    ('null', 'rho03', 20000, (30,)),
    ('nn4', 'rho03', 20000, (30,)),
    ('four', 'iid', 20000, (30,)),
    ('nn8', 'iid', 40000, (30,)),
    ('dense', 'iid', 40000, (30,)),
)
SCALARS = ('average_precision', 'spearman_all', 'spearman_nonzero', 'raw_mean_bits',
           'raw_p90_bits', 'weak_mean_bits')


def exact_marginal_counts(xz, yz, z, radius):
    from physics.counts import count_intervals, count_rectangles

    if radius.min() < 0:
        return tuple(KDTree(points, metric='chebyshev').query_radius(points, radius, count_only=True)
                     for points in (xz, yz, z))
    return count_rectangles(xz, radius), count_rectangles(yz, radius), count_intervals(z[:, 0], radius)


def endpoint_scores(flux, frequency, pairs, ks, *, accelerated=False):
    scores = np.empty((len(ks), len(pairs)))
    rng = np.random.RandomState(ESTIMATOR_SEED)
    counts = exact_marginal_counts if accelerated else None
    for index, (i, j) in enumerate(pairs):
        x, y = flux[:, [i]], flux[:, [j]]
        given_i = flux_detection_scan.cmi_many_k(x, y, frequency[:, [i]], ks, rng, marginal_counts=counts)
        given_j = flux_detection_scan.cmi_many_k(x, y, frequency[:, [j]], ks, rng, marginal_counts=counts)
        scores[:, index] = (given_i + given_j) / 2
    return scores


def score_statistics(scores, coupling, weak):
    hits = coupling != 0
    varying_all = len(np.unique(coupling)) >= 2 and len(np.unique(scores)) >= 2
    varying_hits = len(np.unique(coupling[hits])) >= 2 and len(np.unique(scores[hits])) >= 2
    return {
        'average_precision': float(average_precision_score(hits, scores)) if hits.any() and not hits.all() else None,
        'spearman_all': float(spearmanr(coupling, scores).statistic) if varying_all else None,
        'spearman_nonzero': float(spearmanr(coupling[hits], scores[hits]).statistic) if varying_hits else None,
        'raw_mean_bits': float(np.mean(scores)),
        'raw_p90_bits': float(np.percentile(scores, 90)),
        'weak_mean_bits': float(np.mean(scores[weak])) if len(weak) else None,
        'scores_bits': scores.tolist(),
    }


def distance_scores(flux, frequency, pairs, labels):
    import dcor

    scores = [
        0.5 * (dcor.distance_correlation(flux[:, i], frequency[:, j], method='avl')
               + dcor.distance_correlation(flux[:, j], frequency[:, i], method='avl'))
        for i, j in pairs
    ]
    return {'average_precision': float(average_precision_score(labels, scores)), 'scores': scores}


def run_cell(samples_path, output_path, repeat, specification, source_hashes, accelerated):
    case, condition, count, ks = specification
    provenance = {'source_sha256': source_hashes, 'input_sha256': sha256_file(samples_path),
                  'repeat': repeat, 'case': case, 'condition': condition, 'sample_count': count,
                  'k': list(ks), 'estimator_seed': ESTIMATOR_SEED}
    provenance['marginal_counter'] = 'ordered_sweep' if accelerated else 'kdtree'
    if output_path.is_file():
        cell = json.loads(output_path.read_text(encoding='utf-8'))
        if cell['provenance'] != provenance:
            raise ValueError(f'Existing cell uses different source, input or settings: {output_path}')
        return cell

    start = time.perf_counter()
    with np.load(samples_path, allow_pickle=False) as dataset:
        frequency = dataset[f'{case}_{condition}_clean'][:count]
        width = frequency.shape[1]
        flux = dataset['flux_iid' if condition == 'iid' else 'flux_rho03'][:count, :width]
        coupling_matrix = dataset[f'h_{case}'] if case != 'null' else np.zeros((width, width))
        sample_seed = int(dataset['sample_seed'])
    if (len(flux) != count or flux.shape != frequency.shape or coupling_matrix.shape != (width, width)
            or not all(np.isfinite(values).all() for values in (flux, frequency, coupling_matrix))):
        raise ValueError(f'Invalid sample inputs: {samples_path} ({specification})')

    indices = np.triu_indices(width, 1)
    pairs = np.column_stack(indices)
    coupling = np.abs(coupling_matrix[indices])
    coupled = np.flatnonzero(coupling)
    weak = coupled[np.argsort(coupling[coupled], kind='stable')[:WEAK_PAIRS]] if case == 'nn4' else []
    matrix = endpoint_scores(flux, frequency, pairs, ks, accelerated=accelerated)
    by_k = {str(k): score_statistics(score, coupling, weak) for k, score in zip(ks, matrix)}
    baseline = distance_scores(flux, frequency, pairs, coupling != 0) if case == 'nn4' and condition == 'iid' else None
    cell = {
        'schema': 'causal-q-flux-robustness-score-cell-v1', 'provenance': provenance,
        'sample_seed': sample_seed, 'repeat': repeat, 'case': case, 'condition': condition,
        'sample_count': count, 'pairs': pairs.tolist(), 'coupling_strengths': coupling.tolist(),
        'weak_pairs': pairs[weak].tolist(), 'by_k': by_k, 'dcor': baseline,
        'elapsed_seconds': time.perf_counter() - start,
    }
    flux_detection_scan.atomic_json(output_path, cell)
    return cell


def mean_sd(values):
    if any(value is None for value in values):
        return None
    return {'values': values, 'mean': float(np.mean(values)),
            'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None}


def summarize(cells):
    groups = []
    for case, condition, count, ks in PROTOCOL:
        matched = sorted((cell for cell in cells if (cell['case'], cell['condition'], cell['sample_count'])
                          == (case, condition, count)), key=lambda cell: cell['repeat'])
        if not matched:
            continue
        k_metrics = {
            str(k): {key: mean_sd([cell['by_k'][str(k)][key] for cell in matched]) for key in SCALARS}
            for k in ks
        }
        baseline = mean_sd([cell['dcor']['average_precision'] for cell in matched]) if matched[0]['dcor'] else None
        groups.append({'case': case, 'condition': condition, 'sample_count': count,
                       'repeats': [cell['repeat'] for cell in matched], 'by_k': k_metrics, 'dcor_ap': baseline})
    n, p = 120, 24
    random_ap = flux_detection_scan.random_ranking_ap(p, n)
    return {'schema': 'causal-q-flux-robustness-score-summary-v1', 'estimator_seed': ESTIMATOR_SEED,
            'nn4_positive_prevalence': p / n, 'nn4_expected_ap_for_random_strict_ranking': random_ap,
            'groups': groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--repeats', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--accelerated', action='store_true', help='Use exact sort-and-sweep marginal counts')
    args = parser.parse_args()
    if len(set(args.repeats)) != len(args.repeats) or any(repeat < 0 for repeat in args.repeats):
        parser.error('Repeats must be distinct nonnegative integers')
    root = args.run_directory.resolve()
    inputs = {repeat: root / 'samples' / f'repeat_{repeat:02d}.npz' for repeat in args.repeats}
    for path in inputs.values():
        if not path.is_file():
            parser.error(f'Missing samples: {path}')
    sources = [Path(__file__), Path(flux_detection_scan.__file__)]
    if args.accelerated:
        from physics import counts as flux_robustness_acceleration

        sources.append(Path(flux_robustness_acceleration.__file__))
    source_hashes = {path.name: sha256_file(path) for path in sources}
    cells = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = []
        for specification in PROTOCOL:
            case, condition, count, _ = specification
            for repeat, samples_path in inputs.items():
                output_path = root / 'scores' / case / condition / f'repeat_{repeat:02d}_n{count}.json'
                tasks.append(pool.submit(run_cell, samples_path, output_path, repeat, specification,
                                         source_hashes, args.accelerated))
        for finished in as_completed(tasks):
            cell = finished.result()
            cells.append(cell)
            print(f"{len(cells)}/{len(tasks)} {cell['case']} {cell['condition']} n={cell['sample_count']} "
                  f"repeat={cell['repeat']} ({cell['elapsed_seconds']:.1f} s)", flush=True)
    flux_detection_scan.atomic_json(root / 'scores' / 'summary.json', summarize(cells))


if __name__ == '__main__':
    main()
