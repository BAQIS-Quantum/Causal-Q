"""Rebuild Fig. 5, Tables I–III and Appendix B using the saved query records."""

import argparse
import csv
from pathlib import Path

import numpy as np

from graphs import color_support, task_structure, variants
from inputs import CASES, DATA, GRAPHS, METHODS, audited_masks, graph, physical_groups, workpoints
from optimize import mask_for
from plot import plot
from policy import check_policy, random_policy


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def best_curves(scores):
    losses = (1 - scores).mean(axis=-1)
    return 1 - np.minimum.accumulate(losses, axis=-1)


def coverage(records, stage, action, mask):
    change = np.abs(
        records[f"{stage}_scores"][:, action] - records[f"{stage}_baseline"][:, None, None, :]
    )
    affected = change > 0.001
    return check_policy(mask, affected, change), random_policy(mask, affected, change)


def analyze(output, data=DATA, graphs=GRAPHS):
    output.mkdir(parents=True, exist_ok=True)
    points = workpoints()
    numbers = {}

    def number(key, value, digits=0):
        numbers[key] = f"{value:.{digits}f}"

    rows1, rows2, rows3, panels = [], [], [], {}
    for case in CASES:
        tag = case.capitalize()
        edges, support = graph(case, graphs)
        structure = task_structure(support)
        rows1.append(
            dict(
                case=case,
                layered_edges=len(edges),
                task_target_pairs=structure["task_target_pairs"],
                mean_reach=structure["mean_reach"],
                mean_group_size=structure["mean_group_size"],
                largest_group=max(map(len, structure["groups"])),
                batches=structure["minimum_task_batches"],
                control_colors=structure["control_colors_used"],
                queries=structure["queries_per_function_gradient"],
            )
        )
        with np.load(data / f"{case}_policy.npz") as records:
            augmented = audited_masks(records, support)
            physical = np.zeros_like(support)
            for group in physical_groups(points[case]["h"]):
                physical[np.ix_(group, group)] = True
            policies = [("primary", "graph", support, "Primary")]
            if case == "chain":
                policies.extend(
                    ("validation", name, mask, "Validation" + name.capitalize())
                    for name, mask in [
                        ("graph", support),
                        ("audited", augmented[-1]),
                        ("components", physical),
                    ]
                )
                number(
                    "ChainAuditedCarrierCoverage",
                    100
                    * coverage(records, "validation", 1, augmented[1])[0]["no_miss_event_fraction"],
                    2,
                )
                for key, label in [
                    ("graph", "Graph"),
                    ("sitewise", "Sitewise"),
                    ("all_joint", "Joint"),
                ]:
                    loss = 1 - best_curves(records[f"block_{key}"])
                    number("ChainBlock" + label + "Auc", (loss / loss[:, :1]).mean(), 4)
            for stage, policy, mask, suffix in policies:
                stats, random = coverage(records, stage, 3, mask)
                batches, passed, retry = records[f"{stage}_schedule_{policy}"].mean(axis=(0, 1))
                prefix = tag + suffix
                cov = 100 * stats["no_miss_event_fraction"]
                number(prefix + "List", stats["mean_jobs"], 3)
                number(prefix + "Coverage", cov, 2)
                number(prefix + "Batches", batches)
                number(prefix + "AllPass", 100 * passed, 2)
                number(prefix + "Retry", retry, 3)
                random_percent = 100 * random["no_miss_event_fraction"]
                if stage == "primary":
                    number(prefix + "Random", random_percent, 2)
                if policy != "components":
                    rows2.append(
                        dict(
                            case=case,
                            cohort=stage,
                            support=policy,
                            mean_list=stats["mean_jobs"],
                            coverage_pct=cov,
                            random_pct=random_percent if stage == "primary" else "",
                            planned_batches=batches,
                            all_pass_pct=100 * passed,
                            batches_with_retries=retry,
                        )
                    )
        with np.load(data / f"{case}_optimization.npz") as records:
            curves = {key: best_curves(records[key + "_F"]) for key in METHODS}
            result = {"case": case, "initial": curves["original"][:, 0].mean()}
            number(tag + "Initial", result["initial"], 6)
            for key, label in METHODS.items():
                traces = curves[key]
                masks = records[key + "_mask"]
                expected = [mask_for(key, support, augmented[-1], i) for i in range(8)]
                if not np.array_equal(expected, masks):
                    raise ValueError(f"Query records use a different task support in {case}/{key}")
                result[label.lower()] = traces[:, -1].mean()
                number(tag + label + "Final", result[label.lower()], 6)
                reached = traces - traces[:, :1] >= 0.05
                number(tag + label + "Hit", reached.any(axis=1).sum())
                if reached.any(axis=1).all():
                    number(tag + label + "HitMean", (1 + reached.argmax(axis=1)).mean(), 3)
                number(
                    tag + label + "Win",
                    np.sum(traces[:, -1] - curves["uncompressed"][:, -1] > 1e-9),
                )
                counts = {1 + len(color_support(mask, key)) for mask in masks}
                if len(counts) != 1:
                    raise ValueError("Probe count varies between paired operating points")
                number(tag + label + "Queries", counts.pop())
            initial_haar, terminal_haar = records["audited_haar"].mean(axis=(0, 2))
            result["audited_haar"] = terminal_haar
            number(tag + "InitialHaar", initial_haar, 6)
            number(tag + "AuditedHaar", terminal_haar, 6)
            rows3.append(result)
        conflicts = support.astype(int) @ support.T > 0
        np.fill_diagonal(conflicts, False)
        panels[case] = dict(
            structure=structure,
            conflicts=conflicts,
            curves={k: v.mean(axis=0) for k, v in curves.items()},
        )
        if case == "chain":
            chain_masks = variants(augmented[-1], physical_groups(points[case]["h"]))
    with np.load(data / "support_errors.npz") as records, np.load(
        data / "initial_gradients.npz"
    ) as gradient:
        reference = records["individual"][:, -1]
        rows_b = []
        for name, mask in chain_masks.items():
            final = records[name][:, -1]
            rows_b.append(
                dict(
                    perturbation=name,
                    queries=1 + len(color_support(np.tile(mask, (3, 1)))),
                    terminal=final.mean(),
                    wins=int(np.sum(final - reference > 1e-9)),
                )
            )
        for mode in ("add", "delete"):
            rows = [row for row in rows_b if row["perturbation"].startswith(mode + "_")]
            costs = [row["queries"] for row in rows]
            low, high = min(costs), max(costs)
            numbers["Robust" + mode.capitalize() + "Queries"] = (
                str(low) if low == high else str(low) + r"\text{--}" + str(high)
            )
            number(
                "Robust" + mode.capitalize() + "Better",
                sum(row["terminal"] > reference.mean() for row in rows),
            )
        full = gradient["individual"]
        diff = np.linalg.norm(full - gradient["minus_0_to_1"], axis=1)
        number("RobustOmittedError", 100 * np.mean(diff / np.linalg.norm(full, axis=1)), 2)
        for mask, label in [("audited", "Audited"), ("minus_0_to_1", "Omitted")]:
            number("Robust" + label + "Wins", np.sum(records[mask][:, -1] - reference > 1e-9))
    table(output / "table_1.csv", rows1)
    table(output / "table_2.csv", sorted(rows2, key=lambda row: row["cohort"] != "primary"))
    table(output / "table_3.csv", rows3)
    table(output / "appendix_b.csv", rows_b)
    table(
        output / "figure5_curves.csv",
        [
            {
                "query": q + 1,
                **{
                    case + "_" + METHODS[key].lower(): panels[case]["curves"][key][q]
                    for case in CASES
                    for key in METHODS
                },
            }
            for q in range(256)
        ],
    )
    (output / "fig5_numbers.tex").write_text(
        "".join("\\newcommand{\\GC" + key + "}{" + value + "}\n" for key, value in numbers.items())
    )
    plot(panels, 256, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/calibration",
    )
    parser.add_argument(
        "--records", type=Path, default=DATA, help="directory of simulated query records"
    )
    parser.add_argument(
        "--graphs", type=Path, default=GRAPHS, help="Fig. 4 inferred edge-list directory"
    )
    args = parser.parse_args()
    analyze(args.output, args.records, args.graphs)
