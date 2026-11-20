"""Figure 6 and Table IV, from query vectors or fresh simulated evaluations."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from calibration.graphs import color_support, task_support
from .plot import CASES, METHODS, PLANS, THRESHOLD, plot

DATA = Path(__file__).resolve().parent / "data"
PULSE = DATA.parents[1] / "pulse/data"


def operating_points():
    from pulse.model import bare_response, coupling_matrix, generate_rows

    rows = generate_rows(coupling_matrix("dense"), np.random.default_rng(2026110607), 8)
    permuted = np.random.default_rng(20261106701).permutation(8)
    points = {}
    for plan, row, order in (
        ("planned_ascending", 0, np.arange(8)),
        ("planned_permuted", 7, permuted),
    ):
        frequency = np.arange(3.0, 6.51, 0.5)[order]
        flux = np.arccos(np.sqrt((((frequency + 0.22) / 7) ** 2 - 0.35**2) / (1 - 0.35**2)))
        points[plan] = {
            "bare": bare_response(flux),
            "carrier_error": rows["omega_d_noisy"][row] - rows["omega_d_clean"][row],
            "amplitude_error": rows["Omega_real"][row] - rows["Omega_ideal"][row],
        }
    return points


def supports(case, graph_dir):
    edges = json.loads((graph_dir / f"{case}_graph.json").read_text())["edges"]
    task = task_support(edges)
    groups = [[i for i in group if i < 16] for group in color_support(np.tile(task, (3, 1)))]
    return np.tile(task, (2, 1)), [group for group in groups if group]


def simulate(out, pulse_dir, graph_dir):
    from .dynamics import GateResponse
    from .optimize import joint, carrier

    points = operating_points()
    for case, _ in CASES:
        with np.load(pulse_dir / f"{case}.npz", allow_pickle=False) as samples:
            h = samples["hij"]
        support, groups = supports(case, graph_dir)
        scores, settings = [], []
        table_scores, table_settings, table_four, table_carrier = [], [], [], []
        for plan, _ in PLANS:
            print(case, plan, flush=True)
            response = GateResponse(h, points[plan])
            individual, z, four = joint(response)
            grouped, zg, _ = joint(response, support, groups)
            scores.append([individual, grouped])
            settings.append([z, zg])
            if case == "dense":
                table_scores.append(individual)
                table_settings.append(z)
                table_four.append(four)
                table_carrier.append(carrier(response))
        np.savez_compressed(out / f"{case}.npz", scores=scores, settings=settings)
        if case == "dense":
            np.savez_compressed(
                out / "joint.npz", scores=table_scores, settings=table_settings, four=table_four
            )
            np.savez_compressed(out / "carrier.npz", scores=table_carrier)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Rerun the optimizer and pulse evolution for every query.",
    )
    parser.add_argument("--pulse-dir", type=Path, default=PULSE)
    parser.add_argument("--graphs", type=Path, default=PULSE)
    args = parser.parse_args()
    if not args.recompute and (args.graphs != PULSE or args.pulse_dir != PULSE):
        parser.error("Custom inputs require --recompute")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    records = DATA
    if args.recompute:
        records = out / "queries"
        records.mkdir(exist_ok=True)
        simulate(records, args.pulse_dir, args.graphs)

    curves = {}
    with (out / "figure6.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "case",
                "plan",
                "method",
                "queries_per_gradient",
                "first_query_to_0.995",
                "terminal_mean_haar",
            )
        )
        for case, _ in CASES:
            _, groups = supports(case, args.graphs)
            with np.load(records / f"{case}.npz") as data:
                scores = data["scores"]
            if scores.shape != (2, 2, 256, 8):
                raise ValueError(f"Incomplete query vectors for {case}")
            curves[case] = {}
            for p, (plan, _) in enumerate(PLANS):
                for m, method in enumerate(METHODS):
                    curve = np.maximum.accumulate(scores[p, m].mean(axis=1))
                    hit = np.flatnonzero(curve >= THRESHOLD)
                    writer.writerow(
                        (
                            case,
                            plan,
                            method,
                            17 if m == 0 else 1 + len(groups),
                            int(hit[0]) + 1 if hit.size else None,
                            curve[-1],
                        )
                    )
                    curves[case][plan, method] = curve
    plot(out / "figure6.pdf", curves)

    from .dynamics import GateResponse
    from .optimize import SCALES

    points = operating_points()
    with np.load(records / "carrier.npz") as carrier, np.load(
        records / "joint.npz"
    ) as joint, np.load(args.pulse_dir / "dense.npz") as samples, (out / "table_IV.csv").open(
        "w"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("plan", "initial", "carrier", "joint", "minimum", "tight_solver_max_difference")
        )
        for i, (plan, _) in enumerate(PLANS):
            scores = joint["scores"][i]
            best = np.argmin(np.mean(1 - scores, axis=1))
            response = GateResponse(samples["hij"], points[plan])
            corrections = SCALES * joint["settings"][i, best]
            tight_four, tight_haar = response.evaluate(
                response.initial_amplitude + corrections[:8],
                response.initial_drive + corrections[8:],
                tight=True,
            )
            difference = max(
                np.max(abs(tight_haar - scores[best])),
                np.max(abs(tight_four - joint["four"][i, best])),
            )
            writer.writerow(
                (
                    plan,
                    scores[0].mean(),
                    carrier["scores"][i].mean(axis=1).max(),
                    scores[best].mean(),
                    scores[best].min(),
                    difference,
                )
            )


if __name__ == "__main__":
    main()
