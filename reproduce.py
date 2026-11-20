"""Reproduce the numerical figures and tables of Causal-Q."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("results"), help="Directory for generated results"
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Generate the observations and run all estimators, propagations and optimizations",
    )
    parser.add_argument(
        "--workers",
        type=int,
        choices=range(1, 5),
        default=4,
        help="Worker processes for parallel simulation and estimation",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if out == ROOT or any(
        out.is_relative_to(ROOT / name)
        for name in ("flux", "pulse", "orientation", "calibration", "fixed_gate", "vendor")
    ):
        parser.error("Choose an output directory separate from the source and input files")
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"

    def run(label, *command):
        print(f"\n{label}", flush=True)
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, env=env, check=True)

    workers = str(args.workers)
    fresh = args.from_scratch
    flux = out / "flux"
    pulse = out / "pulse"
    calibration = out / "calibration"
    samples = out / "pulse_samples"
    queries = out / "calibration_queries"
    graphs = ["--graphs", pulse] if fresh else []
    run(
        "Flux dependencies and robustness",
        "flux/run.py",
        "--out",
        flux,
        *(["--full", "--workers", workers] if fresh else []),
    )
    if fresh:
        run(
            "Eight-qubit pulse ensembles",
            "-m",
            "pulse.simulate",
            "--out",
            samples,
            "--workers",
            workers,
        )
    run(
        "Pulse dependencies",
        "-m",
        "pulse.reproduce",
        "--out",
        pulse,
        *(["--data", samples, "--recompute", "--workers", workers] if fresh else []),
    )
    if fresh:
        run(
            "Calibration queries",
            "calibration/simulate.py",
            "--out",
            queries,
            "--workers",
            workers,
            *graphs,
        )
    run(
        "Graph-based calibration",
        "calibration/run.py",
        "--out",
        calibration,
        *(["--records", queries] if fresh else []),
        *graphs,
    )
    run(
        "Fixed-target gate calibration",
        "-m",
        "fixed_gate.reproduce",
        "--out",
        out / "fixed_gate",
        *(["--recompute", "--pulse-dir", samples] if fresh else []),
        *graphs,
    )
    run(
        "Joint orientation",
        "-m",
        "orientation.reproduce",
        "--out",
        out / "orientation",
        *(
            [
                "--workers",
                workers,
                "--from-samples",
                "--flux-dir",
                flux / "reference",
                "--pulse-dir",
                samples,
            ]
            if fresh
            else []
        ),
        *graphs,
    )
    print(f"\nResults written to {out}", flush=True)


if __name__ == "__main__":
    main()
