"""Format the reported flux robustness statistics for the paper."""
from hashlib import sha256
import json
import math
from statistics import mean

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(metric, digits, *, signed=False):
    mean, sd = metric["mean"], metric["sd"]
    if mean is None or sd is None or not (math.isfinite(mean) and math.isfinite(sd)):
        raise ValueError(f"Expected a finite mean and sample SD: {metric}")
    sign = "+" if signed else ""
    return rf"{mean:{sign}.{digits}f}\pm{sd:.{digits}f}"


def validate_selected(results, selected, screen, own, matched, reference_k):
    setup = screen["protocol"]["alg1"]
    if (setup["mi_k"], setup["cmi_k"], setup["threshold"]) != (130, 3, 0.0):
        raise ValueError("Expected k_MI=130 and k_CMI=3 with the zero-bit flux cutoff")
    if matched["k"] != reference_k:
        raise ValueError(f"Expected k_CMI={reference_k} for the flux-correlation sweep")
    if {group["k"] for group in own["groups"]} != {3, 130}:
        raise ValueError("Expected k_MI=3 and 130 in the local MI comparison")
    manifest = load(results / "samples/manifest.json")["datasets"]
    for group in matched["conditions"].values():
        if [r["sample_sha256"] for r in group["repeats"]] != [r["sha256"] for r in manifest]:
            raise ValueError("Rho sweep uses different samples")
    for cell in screen["cells"]:
        trace = selected / "screen/traces" / f"repeat_{cell['repeat']:02d}_{cell['condition']}.json"
        if (cell["protocol"]["alg1"] != setup
                or cell["protocol"]["sample_sha256"] != manifest[cell["repeat"]]["sha256"]
                or cell["trace_sha256"] != sha256(trace.read_bytes()).hexdigest()):
            raise ValueError(f"Screening protocol, sample or trace mismatch: {trace}")
    if own["sources"] != {f"samples/{row['file']}": row["sha256"] for row in manifest}:
        raise ValueError("Local MI estimates use different samples")


def export_numbers(results, selected):
    reference_k = 30
    scores = load(results / "scores/summary.json")
    active = selected
    screen = load(active / "screen/summary.json")
    own = load(active / "diagnostics/summary.json")
    sweep_root = results
    matched = load(sweep_root / "rho_sweep/summary.json")
    noisy = load(results / "noisy_ranking/summary.json")
    groups = {(g["case"], g["condition"], g["sample_count"]): g for g in scores["groups"]}
    numbers = {}
    validate_selected(results, selected, screen, own, matched, reference_k)
    numbers["FluxScreenMIK"] = str(screen["protocol"]["alg1"]["mi_k"])
    numbers["FluxScreenCMIK"] = str(screen["protocol"]["alg1"]["cmi_k"])
    numbers["FluxPairReferenceCMIK"] = str(matched["k"])

    for case, label, samples in (
        ("four", "Four", 20000), ("nn4", "NNFour", 20000),
        ("nn8", "NNEight", 40000), ("dense", "Dense", 40000),
    ):
        group = groups[case, "iid", samples]
        if group["repeats"] != list(range(5)):
            raise ValueError(f"Expected five independent samples for {case}")
        metrics = group["by_k"][str(reference_k)]
        numbers[f"FluxRho{label}Nonzero"] = mean_sd(metrics["spearman_nonzero"], 3)
        numbers[f"FluxRho{label}All"] = mean_sd(metrics["spearman_all"], 3)

    for samples, label in ((5000, "Five"), (20000, "Twenty")):
        group = groups["nn4", "iid", samples]
        for k, k_label in (("3", "Three"), ("30", "Thirty")):
            metric = group["by_k"][k]["average_precision"]
            numbers[f"FluxAp{label}K{k_label}"] = mean_sd(metric, 3)

    sensitivity_path = results / "k_sensitivity/summary.json"
    if sensitivity_path.is_file():
        scan = {g["sample_count"]: g for g in load(sensitivity_path)["groups"]}
        numbers["FluxScanFiveTenAp"] = mean_sd(scan[5000]["by_k"]["10"]["average_precision"], 3)
        for k, label in (("6", "Six"), ("10", "Ten")):
            numbers[f"FluxScanTwenty{label}Nonzero"] = mean_sd(scan[20000]["by_k"][k]["spearman_nonzero"], 3)

    other_scan_path = results / "k_sensitivity_other_cases/summary.json"
    if other_scan_path.is_file():
        other_scan = {g["case"]: g["by_k"] for g in load(other_scan_path)["groups"]}
        nn8 = other_scan["nn8"]
        numbers["FluxScanNNEightTenAp"] = mean_sd(nn8["10"]["average_precision"], 3)
        numbers["FluxScanNNEightThreeNonzero"] = mean_sd(nn8["3"]["spearman_nonzero"], 3)
        numbers["FluxScanNNEightThirtyNonzero"] = mean_sd(nn8["30"]["spearman_nonzero"], 3)
        dense_means = [row["spearman_all"]["mean"] for row in other_scan["dense"].values()]
        numbers["FluxScanDenseMin"] = f"{min(dense_means):.3f}"
        numbers["FluxScanDenseMax"] = f"{max(dense_means):.3f}"

    matched_conditions = matched["conditions"]
    if (sorted(group["rho"] for group in matched_conditions.values()) != [0, 0.05, 0.10, 0.20, 0.30]
            or any(len(group["repeats"]) != 5 for group in matched_conditions.values())):
        raise ValueError("Fig. 7(b) expects five rho settings with five paired datasets each")
    for condition, label in (("iid", "Iid"), ("rho03", "Corr")):
        stat = groups["null", condition, 20000]["by_k"]["30"]["raw_mean_bits"]
        numbers[f"FluxNull{label}"] = mean_sd(stat, 5)
    for condition, label in (("iid", "Iid"), ("rho005", "Twentieth"),
                             ("rho010", "Tenth"), ("rho03", "Corr")):
        numbers[f"FluxMatchedNull{label}"] = mean_sd(matched_conditions[condition]["null"], 5)
    paired_means = [condition["paired_delta"]["mean"] for condition in matched_conditions.values()]
    numbers["FluxMatchedDeltaMin"] = f"{min(paired_means):.5f}"
    numbers["FluxMatchedDeltaMax"] = f"{max(paired_means):.5f}"
    correlated = matched_conditions["rho03"]
    numbers["FluxWeakCorr"] = mean_sd(correlated["nn4"], 5)
    numbers["FluxMatchedDeltaCorr"] = mean_sd(correlated["paired_delta"], 5)
    numbers["FluxMatchedOverlapCorr"] = str(correlated["overlapping_ranges"])
    comparison = load(selected / "rho_sweep/summary.json")
    archived_corr = comparison["conditions"]["rho03"]
    comparison_k = 3
    if (comparison["k"] != comparison_k or comparison["selected_pairs"] != matched["selected_pairs"]
            or comparison["sample_count"] != matched["sample_count"]
            or [row["sample_sha256"] for row in archived_corr["repeats"]]
               != [row["sample_sha256"] for row in correlated["repeats"]]):
        raise ValueError("The two overlap estimates must use matched samples and pairs")
    numbers["FluxWeakIid"] = mean_sd(matched_conditions["iid"]["nn4"], 5)
    numbers["FluxPairComparisonCMIK"] = str(comparison_k)
    numbers["FluxMatchedOverlapCorrComparison"] = str(archived_corr["overlapping_ranges"])
    numbers["FluxMatchedDeltaCorrComparison"] = mean_sd(archived_corr["paired_delta"], 5)
    correlated_ap = groups["nn4", "rho03", 20000]["by_k"]["30"]["average_precision"]
    numbers["FluxApCorr"] = mean_sd(correlated_ap, 3)

    noise_deltas = []
    for k, k_label in (("3", "Three"), ("30", "Thirty")):
        for metric, metric_label in (("average_precision", "Ap"), ("spearman_nonzero", "Nonzero"),
                                     ("spearman_all", "All")):
            for condition, label in (("clean", "Clean"), ("noisy", "Noisy"),
                                     ("noisy_minus_clean", "Delta")):
                result = noisy["by_k"][k][metric][condition]
                if len(result["values"]) != 5:
                    raise ValueError(f"Expected five paired ranking runs for k={k}, {metric}, {condition}")
                delta = condition == "noisy_minus_clean"
                if delta:
                    noise_deltas.append(abs(result["mean"]))
                numbers[f"FluxNoise{k_label}{label}{metric_label}"] = mean_sd(
                    result, 4 if delta else 3, signed=delta
                )
    numbers["FluxNoiseMaxShift"] = f"{max(noise_deltas):.4f}"

    for condition, label in (("iid_clean", "Iid"), ("rho03_clean", "Corr"), ("iid_noise", "Noisy")):
        for metric, metric_label in (("precision", "P"), ("recall", "R")):
            entry = screen["aggregate"][condition][metric]
            if entry["n"] != 5:
                raise ValueError(f"Expected five defined {metric} values for {condition}")
            numbers[f"FluxScreen{label}{metric_label}"] = mean_sd(entry, 3)
        numbers[f"FluxScreen{label}FP"] = f"{screen['aggregate'][condition]['fp']['mean']:.1f}"
        if condition != "iid_noise":
            removed = []
            for repeat in range(5):
                trace = load(active / "screen/traces" / f"repeat_{repeat:02d}_{condition}.json")
                removed.append(sum(test["removed"] and test["source"] != test["target"]
                                   and not test["condition_target_indices"] for test in trace["tests"]))
            numbers[f"FluxScreen{label}MarginalDrops"] = f"{mean(removed):.1f}"

    for group in own["groups"]:
        sample_label = {1000: "One", 20000: "Twenty"}.get(group["sample_count"])
        if sample_label and (group["k"] in (3, 130)):
            noise_label = "Clean" if group["noise_ghz"] == 0.0 else "Noisy"
            k_label = "Three" if group["k"] == 3 else "OneThirty"
            numbers[f"FluxSelfMI{sample_label}{noise_label}K{k_label}"] = mean_sd(
                {"mean": group["mean_bits"], "sd": group["sd_bits"]}, 3
            )

    return "% Generated from the frozen flux robustness results.\n" + "".join(
        rf"\newcommand{{\{name}}}{{{value}}}" + "\n" for name, value in numbers.items()
    )
