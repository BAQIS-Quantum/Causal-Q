"""Generate calibration query records from the operating points and pulse equations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import repeat
import os
from pathlib import Path

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np

from graphs import color_support, task_structure, variants
from inputs import CASES, DATA, GRAPHS, METHODS, audited_masks, graph, physical_groups
from optimize import finite_difference, mask_for, optimize, optimize_block, physical_controls
from probes import recheck_point, schedule_point
from response import TimeDependentResponse, generate_points


def write(path, arrays):
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)
    print(f"Saved {path}", flush=True)


def block_scores(model, point, amplitudes):
    """Physically remeasure each setting along the published block-search path."""
    return np.array([model.evaluate(point, amplitude=trial) for trial in amplitudes])


def policy_records(pool, case, points, support, repeat_block_search=False):
    model = TimeDependentResponse(case, points["h"])
    groups = task_structure(support)["groups"]
    masks = {"graph": support, "components": np.zeros_like(support)}
    for group in model.components:
        masks["components"][np.ix_(group, group)] = True
    signs = np.vstack(
        [
            np.ones(8, int),
            -np.ones(8, int),
            np.random.default_rng(20261106901).choice([-1, 1], size=(6, 8)),
        ]
    )
    arrays = {}
    for stage in ("primary", "validation"):
        cohort = points[stage]
        isolated = list(pool.map(recheck_point, repeat(model), cohort))
        for field in ("baseline", "scores"):
            arrays[f"{stage}_{field}"] = np.array([r[field] for r in isolated])
        if stage == "validation":
            masks["audited"] = audited_masks(arrays, support)[-1]
        schedule = list(
            pool.map(
                schedule_point,
                repeat(model),
                cohort[:4],
                isolated[:4],
                repeat(masks),
                repeat(signs),
            )
        )
        fields = ("planned_batches", "all_batches_pass", "discard_then_serial_retry_batches")
        for name in masks:
            arrays[f"{stage}_schedule_{name}"] = np.array(
                [
                    [[row[name][key] for key in fields] for row in point["records"]]
                    for point in schedule
                ]
            )
    if case == "chain":
        recorded_groups = task_structure(graph(case)[1])["groups"]
        for method, blocks in [
            ("graph", groups),
            ("all_joint", [list(range(8))]),
            ("sitewise", [[i] for i in range(8)]),
        ]:
            if repeat_block_search or (method == "graph" and blocks != recorded_groups):
                runs = pool.map(
                    optimize_block, repeat(model), points["primary"][:4], repeat(blocks)
                )
            else:
                with np.load(DATA / "block_amplitudes.npz") as commands:
                    runs = pool.map(
                        block_scores, repeat(model), points["primary"][:4], commands[method]
                    )
            arrays["block_" + method] = np.array(list(runs))
    return arrays


def optimization_point(model, point, support, audited):
    result = {}
    for name in METHODS:
        mask = mask_for(name, support, audited, point["point_id"])
        run = optimize(model, point, mask, name)
        trace = run["trace"]
        best = physical_controls(point, np.asarray(run["best_z"]))
        score, haar = model.score(point, **best)
        tight_model = TimeDependentResponse(
            model.case,
            model.h,
            replace(
                model.solver,
                rtol=1e-11,
                atol=1e-13,
                max_step_internal=model.solver.max_step_internal / 2,
            ),
        )
        tight_score, tight_haar = tight_model.score(point, **best)
        if max(np.max(abs(score - tight_score)), np.max(abs(haar - tight_haar))) > 1e-7:
            raise ArithmeticError("Endpoint integration accuracy outside protocol tolerance")
        _, initial_haar = model.score(point, **physical_controls(point, np.zeros(24)))
        for key in ("F", "z"):
            result[f"{name}_{key}"] = np.array([r[key] for r in trace])
        result[name + "_haar"] = [initial_haar, haar]
        result[name + "_mask"] = mask
    return result


def gradient(model, point, mask):
    support = np.tile(mask, (3, 1))

    def evaluate(z, role, group):
        return model.evaluate(point, **physical_controls(point, z))

    return finite_difference(np.zeros(24), evaluate, support, color_support(support))[1]


def support_curve(model, point, mask):
    run = optimize(model, point, np.tile(mask, (3, 1)))
    return 1 - np.minimum.accumulate([r["loss"] for r in run["trace"]])


def generate(out, cases, study, workers, graphs=GRAPHS, repeat_block_search=False):
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for case in cases:
            points = generate_points(case)
            _, support = graph(case, graphs)
            policy_file = out / f"{case}_policy.npz"
            if study in ("all", "policy"):
                write(policy_file, policy_records(pool, case, points, support, repeat_block_search))
            if study == "policy":
                continue
            model = TimeDependentResponse(case, points["h"])
            # A single requested downstream study can use the distributed response audit.
            with np.load(
                policy_file if policy_file.exists() else DATA / policy_file.name
            ) as records:
                audited = audited_masks(records, support)[-1]
            cohort = points["optimization"]
            if study in ("all", "optimization"):
                runs = list(
                    pool.map(
                        optimization_point, repeat(model), cohort, repeat(support), repeat(audited)
                    )
                )
                write(
                    out / f"{case}_optimization.npz",
                    {key: np.asarray([r[key] for r in runs]) for key in runs[0]},
                )
            if case == "chain" and study in ("all", "support-errors", "gradients"):
                masks = variants(audited, physical_groups(points["h"]))
                gradients = {
                    key: list(pool.map(gradient, repeat(model), cohort, repeat(masks[key])))
                    for key in ("individual", "minus_0_to_1")
                }
                write(out / "initial_gradients.npz", gradients)
                if study != "gradients":
                    curves = {
                        key: list(pool.map(support_curve, repeat(model), cohort, repeat(mask)))
                        for key, mask in masks.items()
                    }
                    write(out / "support_errors.npz", curves)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument(
        "--study",
        choices=("all", "policy", "optimization", "support-errors", "gradients"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--graphs", type=Path, default=GRAPHS, help="Fig. 4 inferred edge-list directory"
    )
    parser.add_argument(
        "--repeat-block-search",
        action="store_true",
        help="Rerun the three amplitude-only searches with L-BFGS-B. By default, "
        "reintegrate their recorded trial amplitudes to reproduce the reported AUCs.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    selected = args.case or CASES
    if args.study in ("support-errors", "gradients"):
        if args.case and args.case != ["chain"]:
            parser.error("Appendix B uses the segmented chain")
        selected = ["chain"]
    generate(args.out, selected, args.study, args.workers, args.graphs, args.repeat_block_search)
