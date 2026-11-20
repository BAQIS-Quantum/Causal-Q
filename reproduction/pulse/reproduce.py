"""Reproduce Figure 4 and its numerical statistics from the eight-qubit ensembles."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from flux.physics.information import sha256_file
from .analysis import CMI_K, MI_K, compute_case
from .model import SAMPLES
from .plot import draw

DATA = Path(__file__).resolve().parent / "data"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, obj):
    path.write_text(json.dumps(obj, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def summarize(path, name, graph, mi):
    with np.load(path, allow_pickle=False) as data:
        bare, fidelity, h = data["omega_d_clean"], data["F"], data["hij"]
        if fidelity.shape != (SAMPLES[name], 8):
            raise ValueError(f"Unexpected Fig. 4 ensemble dimensions for {name}: {fidelity.shape}")
        i, j = np.nonzero(np.triu(h, 1))
        gap = np.abs(bare[:, i] - bare[:, j])
        near_resonance = np.any(gap <= np.abs(h[i, j]), axis=1)
        stats = {"samples": len(bare), "edges": graph["edges"],
                 "edge_count": len(graph["edges"]), "coupling": h.tolist(),
                 "mean_four_probe_score": float(fidelity.mean()),
                 "near_resonance_fraction": float(near_resonance.mean())}
    if mi:
        matrix = np.asarray(mi["mi_bits"])
        off_site = ~np.eye(8, dtype=bool)
        stats.update(mi_bits=mi["mi_bits"], min_bits=float(matrix.min()),
                     max_bits=float(matrix.max()),
                     spearman_off_site=float(spearmanr(matrix[off_site], np.abs(h[off_site])).statistic))
    return stats


def export_numbers(path, cases):
    """Format the UnifiedPulse macros used by the manuscript."""
    numbers = {"MIK": MI_K, "CMIK": CMI_K}
    for case, stats in cases.items():
        prefix = case.capitalize()
        numbers[prefix + "N"] = format(stats["samples"], ",").replace(",", "{,}")
        if "mi_bits" in stats:
            numbers[prefix + "K"] = MI_K
        numbers[prefix + "MeanF"] = format(stats["mean_four_probe_score"], ".4f")
        if "mi_bits" in stats:
            numbers[prefix + "Min"] = format(stats["min_bits"], ".6f")
            numbers[prefix + "Max"] = format(stats["max_bits"], ".6f")
        numbers[prefix + "Edges"] = stats["edge_count"]
        if "mi_bits" in stats:
            numbers[prefix + "Rho"] = format(stats["spearman_off_site"], ".3f")
    lines = [rf"\newcommand{{\UnifiedPulse{name}}}{{{value}}}" for name, value in numbers.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(item):
    name, path, recompute = item
    if recompute:
        graph, mi = compute_case(path, name)
    else:
        graph = read(DATA / f"{name}_graph.json")
        mi = None if name == "uncoupled" else read(DATA / f"{name}_mi.json")
        dataset_hash = sha256_file(path)
        if graph["dataset_sha256"] != dataset_hash or (mi and mi["dataset_sha256"] != dataset_hash):
            raise ValueError(f"Cached Fig. 4 statistics do not match {path}. Use --recompute.")
    return name, graph, mi, summarize(path, name, graph, mi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA, help="Directory with three ensemble NPZ files")
    parser.add_argument("--recompute", action="store_true", help="Re-estimate all four layers and MI maps")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [(case, args.data / f"{case}.npz", args.recompute) for case in SAMPLES]
    if args.workers == 1:
        results = list(map(analyze, tasks))
    else:
        with ProcessPoolExecutor(max_workers=min(len(tasks), args.workers)) as pool:
            results = list(pool.map(analyze, tasks))
    cases = {}
    for name, graph, mi, stats in results:
        cases[name] = stats
        if args.recompute:
            save(args.out / f"{name}_graph.json", graph)
            if mi:
                save(args.out / f"{name}_mi.json", mi)
        print(f"{name}: {stats['samples']} samples, {stats['edge_count']} edges, "
              f"mean score {stats['mean_four_probe_score']:.6f}")
    save(args.out / "summary.json", {"mi_k": MI_K, "cmi_k": CMI_K, "cases": cases})
    export_numbers(args.out / "unified_pulse_numbers.tex", cases)
    draw(cases, args.out)


if __name__ == "__main__":
    main()
