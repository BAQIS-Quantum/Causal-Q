"""Reference-device samples, information scores and figure/table summaries."""
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from physics.coupling import bare_response, case_config, coupling_matrix, exact_spectrum, frequency_batch, initialize_rng
from physics.knn import atomic_json, random_ranking_ap

CASES = {"four": ("figure2_flux", 20000), "nn4": ("figure3_nn4", 20000),
         "nn8": ("figure3_nn8", 40000), "dense": ("figure3_dense", 40000)}
LABELS = {"four": "Four", "nn4": "NNFour", "nn8": "NNEight", "dense": "Dense"}
MODELS = {"scalar": "Scalar", "exact_bare": "ExactBare", "exact_transmon": "ExactTransmon"}
SUBSET_SEEDS = {"nn4": 20261018, "nn8": 20261019}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def generate_samples(name):
    case, n = CASES[name]
    config = case_config(case, n, seed=43)
    h = coupling_matrix(config)
    width = len(h)
    rng = initialize_rng(config)
    flux = np.empty((n, width))
    for row in range(n):
        flux[row] = rng.normal(0, 0.25, width)
        # The joint control sampler also draws amplitude, amplitude error,
        # carrier error and duration on every row. Preserve its RNG sequence.
        rng.normal(0.07, 0.02 / 6, width)
        rng.normal(0, 0.005, width)
        rng.normal(0, 0.2, width)
        rng.uniform(120.0, 120.0)
    bare = bare_response(flux)
    dressed = np.concatenate([frequency_batch(bare[i:i + 1000], h) for i in range(0, n, 1000)])
    return {"flux": flux, "hij": h, "omega_d_clean": bare, "omega_q_clean": dressed}


def compute(name, directory):
    from physics.counts import accelerate_marginal_counts
    from physics.information import compute_information_matrices, _load_estimator
    from physics.screening import screen_layer
    from scripts.flux_robustness_scores import distance_scores

    samples = generate_samples(name)
    x, y, h = (samples[k] for k in ("flux", "omega_q_clean", "hij"))
    np.savez_compressed(directory / f"{name}.npz", **samples)
    pairs = np.triu_indices(len(h), 1)
    with accelerate_marginal_counts(_load_estimator()):
        m, mi, _ = compute_information_matrices(x, y)
        graph = screen_layer(x, y, "Phi", "f", threshold=0, mi_k=130, cmi_k=3, seed=43)
        edges = sorted([int(s.split("_")[1]), int(t.split("_")[1])] for s, t in graph["edges"])
        displayed = [e for e in edges if mi[tuple(e)] > (0.010 if name == "four" else 0.)]
        record = dict(sample_count=len(x), coupling_hij=h.tolist(), mi_bits=mi.tolist(),
                      mij_bits=m.tolist(), inferred_edges=edges,
                      displayed_edges=[e for e in displayed if e[0] != e[1]])
        if name == "four":
            record["bipartite_displayed_edges"] = displayed
        if name in SUBSET_SEEDS:
            seed = SUBSET_SEEDS[name]
            indices = np.sort(np.random.default_rng(seed).choice(len(x), 5000, replace=False))
            subset_m, _, _ = compute_information_matrices(x[indices], y[indices], mij_only=True)
            dcor = distance_scores(x, y, list(zip(*pairs)), h[pairs] != 0)
            record.update(subset_seed=seed, subset_mij_bits=subset_m[pairs].tolist(),
                          full_dcor_scores=dcor["scores"])
        models = {}
        if name in ("four", "nn4"):
            base = samples["omega_d_clean"]
            transmon = 7 * (np.cos(x)**2 + 0.35**2 * np.sin(x)**2)**0.25 - 0.22
            models["scalar"] = m.tolist()
            for key, bare in (("exact_bare", base), ("exact_transmon", transmon)):
                exact = exact_spectrum(bare, h)[0]
                matrix, _, _ = compute_information_matrices(x, exact, mij_only=True)
                models[key] = matrix.tolist()
    atomic_json(directory / f"{name}.json", record)
    return CASES[name][0], models


def reference_numbers(reference, repeats):
    records, summaries, numbers = {}, {}, {
        "FluxFigureThreeDisplayMIK": "3", "FluxFigureTwoDisplayMIK": "3",
        "FluxFigureTwoCMIK": "30", "FluxSupportBenchmarkCMIK": "30"}
    models = read(reference / "model_matrices.json")
    model_numbers = {"FluxModelCheckCMIK": "30"}
    for name, (case, samples) in CASES.items():
        row = records[name] = read(reference / f"{name}.json")
        h = np.asarray(row["coupling_hij"])
        pairs = np.triu_indices(len(h), 1)
        strength = np.abs(h[pairs])
        coupled = strength != 0
        def ranks(scores):
            return float(spearmanr(scores[coupled], strength[coupled]).statistic), float(spearmanr(scores, strength).statistic)
        scores = np.array(row["mij_bits"])[pairs]
        rho, all_rho = ranks(scores)
        label = LABELS[name]
        prefix = "FluxReference" + label
        numbers[prefix + "N"] = f"{samples:,}".replace(",", "{,}")
        numbers[prefix + "Rho"] = f"{rho:.3f}"
        offsite = [(i,j) for i,j in row["inferred_edges"] if i != j]
        tp = sum(bool(h[i,j]) for i,j in offsite)
        metrics = summaries[name] = dict(nonzero_pairs=int(coupled.sum()), sample_count=samples,
            coupled_spearman=rho, all_pair_spearman=all_rho, displayed=len(row["displayed_edges"]),
            screened_cross_qubit=len(offsite), true_positive=tp, direct_links=int(2*coupled.sum()),
            recall=tp/(2*int(coupled.sum())), precision=tp/len(offsite))
        if name != "four":
            numbers.update({prefix + "AllRho": f"{all_rho:.3f}", prefix + "TP": str(tp),
                            prefix + "Displayed": str(metrics["displayed"]), prefix + "Recall": f"{metrics['recall']:.3f}",
                            "FluxFigureThree" + label + "CMIK": "30"})
        if name in SUBSET_SEEDS:
            subset = np.array(row["subset_mij_bits"])
            metrics.update(expected_ap_random_ranking=random_ranking_ap(int(coupled.sum()), len(coupled)),
                           ap=float(average_precision_score(coupled, scores)),
                           dcor_ap=float(average_precision_score(coupled, row["full_dcor_scores"])),
                           subset_ap=float(average_precision_score(coupled, subset)),
                           subset_rho=ranks(subset)[0])
            for suffix, field in (("AP","ap"), ("DcorAP","dcor_ap"), ("SubsetAP","subset_ap"),
                                  ("SubsetRho","subset_rho"), ("Precision","precision")):
                numbers[prefix + suffix] = f"{metrics[field]:.3f}"
        group = next(g for g in repeats if g["case"]==name and g["sample_count"]==samples and g["condition"]=="iid")
        prefix = "FluxFigureTwo" if name=="four" else "FluxFigureThree" + label
        for suffix, field in (("RhoRepeat","spearman_nonzero"), ("AllRhoRepeat","spearman_all")):
            values = group["by_k"]["30"][field]["values"]
            numbers[prefix + suffix] = rf"{mean(values):.3f}\pm{stdev(values):.3f}"
        if name in ("four", "nn4"):
            for key, model in MODELS.items():
                partial, all_pairs = ranks(np.asarray(models[case][key])[pairs])
                prefix = "FluxModelCheck" + model
                if name == "four":
                    model_numbers[prefix + "Four"] = f"{all_pairs:.4f}"
                else:
                    model_numbers.update({prefix + "NNFourNonzero": f"{partial:.4f}",
                                          prefix + "NNFourAll": f"{all_pairs:.4f}"})
    return records, summaries, numbers, model_numbers
