#!/usr/bin/env python3
"""Generate paired flux robustness datasets for the fixed Fig. 2 and 3 models."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics.coupling import case_config, coupling_matrix, frequency_batch
from physics.information import sha256_file
from physics.coupling import bare_response


CASES = {
    "four": ("figure2_flux", 20_000, 4),
    "nn4": ("figure3_nn4", 20_000, 16),
    "nn8": ("figure3_nn8", 40_000, 16),
    "dense": ("figure3_dense", 40_000, 16),
}
SEEDS = tuple(range(2026111800, 2026111805))
DEVICE_SEED = 43
PHASE_STD = 0.25
RHO = 0.3
NOISE_GHZ = 0.0005


def solve_rows(bare, coupling):
    # The solver stops per row, so chunking bounds memory without changing its results.
    chunks = (bare[start:start + 1000] for start in range(0, len(bare), 1000))
    return np.concatenate([frequency_batch(chunk, coupling) for chunk in chunks])


def generate(seed, couplings):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((40_000, 16))
    common = rng.standard_normal((20_000, 1))
    flux_iid = PHASE_STD * z
    flux_correlated = PHASE_STD * (np.sqrt(1 - RHO) * z[:20_000] + np.sqrt(RHO) * common)
    bare_iid = bare_response(flux_iid)
    bare_correlated = bare_response(flux_correlated)
    noise = rng.standard_normal((20_000, 16))
    data = {
        "sample_seed": np.asarray(seed),
        "flux_iid": flux_iid,
        "flux_rho03": flux_correlated,
        "noise_iid": noise,
        "null_iid_clean": bare_iid[:20_000],
        "null_rho03_clean": bare_correlated,
        "nn4_rho03_clean": solve_rows(bare_correlated, couplings["nn4"]),
    }
    for key, (_, samples, sites) in CASES.items():
        data[f"h_{key}"] = couplings[key]
        data[f"{key}_iid_clean"] = solve_rows(bare_iid[:samples, :sites], couplings[key])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Directory for paired sample files")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    couplings = {
        key: coupling_matrix(case_config(case, samples, seed=DEVICE_SEED))
        for key, (case, samples, _) in CASES.items()
    }
    datasets = []
    for repeat, seed in enumerate(SEEDS):
        path = args.out / f"repeat_{repeat:02d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as existing:
                if int(existing["sample_seed"]) != seed or any(
                    not np.array_equal(existing[f"h_{key}"], matrix) for key, matrix in couplings.items()
                ):
                    raise ValueError(f"Existing dataset does not match the fixed protocol: {path}")
        else:
            data = generate(seed, couplings)
            staged = args.out / f".repeat_{repeat:02d}.npz"
            np.savez(staged, **data)
            staged.replace(path)
        datasets.append({"file": path.name, "sample_seed": seed, "sha256": sha256_file(path)})
        print(f"Ready {path.name} (sample seed {seed})", flush=True)

    manifest = {
        "schema": "causal-q-flux-robustness-samples-v1",
        "device_seed": DEVICE_SEED,
        "phase_std": PHASE_STD,
        "rho_correlated": RHO,
        "frequency_noise_std_ghz": NOISE_GHZ,
        "frequency_solver": "Fig. 2 and 3 scalar self-consistent approximation",
        "null_response": "All couplings zero, dressed frequency equals bare response",
        "cases": CASES,
        "numpy_version": np.__version__,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (Path(__file__).resolve(), ROOT / "physics/coupling.py")
        },
        "datasets": datasets,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
