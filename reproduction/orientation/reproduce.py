"""Appendix D. Joint PC-stable, conservative collider orientation and Table VIII."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import lru_cache
from itertools import combinations, repeat
import json
from pathlib import Path

import numpy as np

from .rules import orient_pdag

HERE = Path(__file__).resolve().parent
FLUX = HERE.parent / "flux/data/reference"
PULSE = HERE.parent / "pulse/data"
CASES = {
    "four": 4,
    "nn4": 16,
    "nn8": 16,
    "dense": 16,
    "pulse_uncoupled": 8,
    "pulse_chain": 8,
    "pulse_dense": 8,
}
FIELDS = (
    ("flux", "Phi"),
    ("omega_q_clean", "f"),
    ("omega_d_noisy", "omega_d"),
    ("Omega_real", "Omega"),
    ("F", "F"),
)
COLUMNS = (
    "screened",
    "collider_agrees",
    "collider_opposes",
    "unoriented",
    "absent",
)


def estimator(case, fields, from_samples, flux_dir, pulse_dir):
    if not from_samples:
        with (HERE / "data" / f"{case}.csv").open() as stream:
            scores = {
                (
                    int(r["i"]),
                    int(r["j"]),
                    tuple(int(i) for i in (r["condition_1"], r["condition_2"]) if i),
                ): float(r["bits"])
                for r in csv.DictReader(stream)
            }
        return lambda i, j, cond: scores[i, j, cond]

    from npeet import entropy_estimators as ee

    name = case.removeprefix("pulse_")
    folder = pulse_dir if case.startswith("pulse_") else flux_dir
    with np.load(folder / f"{name}.npz", allow_pickle=False) as data:
        values = np.column_stack([data[field] for field, _ in fields])
    values = (values - values.mean(axis=0)) / values.std(axis=0)

    @lru_cache(maxsize=None)
    def score(i, j, cond):
        np.random.seed(43)  # NPEET tie-breaking jitter is reseeded for each joint CI query.
        x, y = values[:, [i]], values[:, [j]]
        if cond:
            return float(ee.cmi(x, y, values[:, cond], k=10, base=2))
        return float(ee.mi(x, y, k=10, base=2))

    return score


def discover(n, score):
    edges = set(combinations(range(n), 2))
    separators = {}

    def neighborhoods():
        neighbors = [set() for _ in range(n)]
        for a, b in edges:
            neighbors[a].add(b)
            neighbors[b].add(a)
        return neighbors

    def subsets(a, b, order):
        return sorted(
            set(combinations(sorted(adj[a] - {b}), order))
            | set(combinations(sorted(adj[b] - {a}), order))
        )

    for order in range(3):
        adj = neighborhoods()
        for a, b in sorted(edges):
            for cond in subsets(a, b, order):
                if score(a, b, cond) <= 0.02:
                    edges.remove((a, b))
                    separators[a, b] = {cond}
                    break
    adj = neighborhoods()
    triples = [
        (a, center, b)
        for center, nodes in enumerate(adj)
        for a, b in combinations(sorted(nodes), 2)
        if (a, b) not in edges
    ]
    for a, b in sorted({(a, b) for a, _, b in triples}):
        for order in range(3):
            for cond in subsets(a, b, order):
                if score(a, b, cond) <= 0.02:
                    separators[a, b].add(cond)
    ambiguous = [
        (a, center, b)
        for a, center, b in triples
        if len({center in cond for cond in separators[a, b]}) == 2
    ]
    representative = {
        pair: min(sets, key=lambda s: (len(s), s)) for pair, sets in separators.items()
    }
    graph = orient_pdag(
        n, sorted(edges), representative, ambiguous_triples=ambiguous, apply_meek=False
    )
    return edges, graph, ambiguous


def analyze(case, from_samples, flux_dir=FLUX, pulse_dir=PULSE, graph_dir=PULSE):
    size = CASES[case]
    pulse = case.startswith("pulse_")
    fields = FIELDS if pulse else FIELDS[:2]
    names = [f"{label}_{i}" for _, label in fields for i in range(size)]
    score = estimator(case, fields, from_samples, flux_dir, pulse_dir)
    context = nullcontext()
    if from_samples:
        from npeet import entropy_estimators
        from flux.physics.counts import accelerate_marginal_counts

        context = accelerate_marginal_counts(entropy_estimators)
    with context:
        skeleton, graph, ambiguous = discover(len(names), score)
    if pulse:
        data = json.loads((graph_dir / f"{case[6:]}_graph.json").read_text())
        screen = [tuple(edge) for edge in data["edges"]]
        displayed = set()
    else:
        data = json.loads((flux_dir / f"{case}.json").read_text())
        screen = [(f"Phi_{i}", f"f_{j}") for i, j in data["inferred_edges"]]
        displayed = {(f"Phi_{i}", f"f_{j}") for i, j in data["displayed_edges"]}
    directions = {
        (names[r["source"]], names[r["target"]]): r["rule"] for r in graph["orientation_log"]
    }
    adjacencies = {frozenset((names[a], names[b])) for a, b in skeleton}
    counts = dict.fromkeys(COLUMNS, 0)
    records = []
    for arrow in sorted(screen):
        if arrow in directions:
            tag = "collider_agrees"
        elif arrow[::-1] in directions:
            tag = "collider_opposes"
        else:
            tag = "unoriented" if frozenset(arrow) in adjacencies else "absent"
        counts[tag] += 1
        records.append({"edge": arrow, "status": tag})
    counts["screened"] = len(screen)
    report = {
        "edge_comparison": records,
        "joint_arrows": [[a, b, r] for (a, b), r in directions.items()],
        "ambiguous_triples": [[names[a], names[c], names[b]] for a, c, b in ambiguous],
    }
    if not pulse:
        report["displayed_collider_agreements"] = sum(
            directions.get(edge) == "collider" for edge in displayed
        )
    if case == "nn8":
        report["example_bits"] = {
            "MI(Phi7,Phi11)": score(7, 11, ()),
            "CMI(Phi7,Phi11|f11)": score(7, 11, (size + 11,)),
        }
    if case == "pulse_chain":
        report["CMI(Omega1,F0|F1)_bits"] = score(
            names.index("Omega_1"), names.index("F_0"), (names.index("F_1"),)
        )
    if case == "pulse_uncoupled":
        report["ambiguous_drive_amplitude_pairs"] = []
        for i in range(size):
            drive, amp, fidelity = (
                names.index(f"{prefix}_{i}") for prefix in ("omega_d", "Omega", "F")
            )
            report["ambiguous_drive_amplitude_pairs"].append(
                {
                    "qubit": i,
                    "mi_bits": score(drive, amp, ()),
                    "cmi_given_F_bits": score(drive, amp, (fidelity,)),
                }
            )
    return counts, report


def write_tex(counts, out):
    labels = (
        "Four qubits",
        "Sparse flux",
        "Moderate flux",
        "Dense flux",
        "Uncoupled pulse",
        "Segmented-chain pulse",
        "Dense pulse",
    )
    lines = [r"\newcommand{\OrientationComparisonRows}{%"]
    for label, row in zip(labels, counts):
        lines.append(" & ".join([label, *(str(row[key]) for key in COLUMNS)]) + r"\\%")
    lines.append("}")
    macros = {"Screened": "screened", "VAgree": "collider_agrees", "VOppose": "collider_opposes"}
    for group, rows in (("Flux", counts[:4]), ("Pulse", counts[4:])):
        for macro, key in macros.items():
            value = sum(row[key] for row in rows)
            lines.append(rf"\newcommand{{\Orientation{group}{macro}}}{{{value}}}")
    (out / "orientation_comparison_numbers.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--from-samples",
        action="store_true",
        help="Re-estimate all MI and CMI scores from the raw observations.",
    )
    parser.add_argument(
        "--flux-dir", type=Path, default=FLUX, help="Flux samples and screened edges"
    )
    parser.add_argument("--pulse-dir", type=Path, default=PULSE, help="Pulse sample files")
    parser.add_argument("--graphs", type=Path, default=PULSE, help="Pulse screened-edge JSON files")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    if not args.from_samples and (
        args.flux_dir != FLUX or args.pulse_dir != PULSE or args.graphs != PULSE
    ):
        parser.error("Custom datasets or screens require --from-samples")
    args.out.mkdir(parents=True, exist_ok=True)
    cases = list(CASES)
    inputs = (
        cases,
        repeat(args.from_samples),
        repeat(args.flux_dir),
        repeat(args.pulse_dir),
        repeat(args.graphs),
    )
    if args.workers == 1:
        results = list(map(analyze, *inputs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(analyze, *inputs))
    with (args.out / "table_VIII.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case", *COLUMNS))
        writer.writeheader()
        for case, (counts, details) in zip(cases, results):
            writer.writerow({"case": case, **counts})
            (args.out / f"{case}.json").write_text(json.dumps(details, indent=2) + "\n")
            print(case, counts, flush=True)
    write_tex([counts for counts, _ in results], args.out)


if __name__ == "__main__":
    main()
