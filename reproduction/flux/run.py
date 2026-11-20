"""Regenerate the flux figures and numerical tables of the manuscript."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
from physics.knn import atomic_json
from reference import CASES, compute, read, reference_numbers


def script(name, *args):
    subprocess.run([sys.executable, str(ROOT / "scripts" / (name + ".py")), *map(str, args)], check=True)


def recalculate_robustness(root, workers):
    selected = root / "screen_settings"
    jobs = ["--workers", str(workers)]
    script("generate_flux_robustness", "--out", root / "samples")
    script("flux_robustness_scores", root, "--accelerated", *jobs)
    script("run_flux_robustness_screen", "--input-dir", root / "samples", "--out", selected / "screen",
           "--mi-k", 130, "--cmi-k", 3, "--accelerate-marginal-counts", *jobs)
    script("run_flux_noisy_ranking", root, *jobs)
    script("run_flux_rho_scan", root, *jobs)
    script("run_flux_rho_scan", root, "--k", 3, "--out", selected / "rho_sweep", *jobs)
    script("run_flux_k_sensitivity", root, *jobs)
    script("run_flux_other_case_k_sensitivity", root, *jobs)
    script("flux_robustness_self_mi", "--out", root, "--neighbors", 3, 130,
           "--diagnostic-dir", selected / "diagnostics", *jobs)


def render(records, summaries, robust, output):
    from scripts.plot_four import update_artwork
    from scripts.plot_sixteen import draw_figure3

    output.mkdir(parents=True, exist_ok=True)
    four = update_artwork(DATA / "four_layout.svg", records["four"])
    svg = output / "figure2.svg"
    four.write(svg, encoding="utf-8", xml_declaration=True)
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "--format=pdf1.5", "--output", str(output/"figure2.pdf"), str(svg)], check=True)
    draw_figure3(records, summaries, output)
    script("plot_flux_robustness", "--out", robust, "--selected", robust / "screen_settings",
           "--figure-dir", output, "--reference-cmi-k", 30, "--appendix-figures")
    for name, scan in (("plot_flux_k_sensitivity", "k_sensitivity"),
                       ("plot_flux_k_other_cases", "k_sensitivity_other_cases")):
        script(name, robust / scan / "summary.json", output, "--appendix-figures")

    # The supplement presents the three appendix plots with embedded captions.
    with TemporaryDirectory(dir=output) as tmp:
        pages = Path(tmp)
        script("plot_flux_robustness", "--out", robust, "--selected", robust / "screen_settings",
               "--figure-dir", pages, "--reference-cmi-k", 30, "--only", "weak-pair-distribution")
        for plotter, folder in (("plot_flux_k_sensitivity", "k_sensitivity"),
                                ("plot_flux_k_other_cases", "k_sensitivity_other_cases")):
            script(plotter, robust / folder / "summary.json", pages)
        components = [pages / (name + ".pdf") for name in
                      ("flux_weak_pair_distribution", "flux_k_sensitivity", "flux_k_other_cases")]
        if shutil.which("pdfunite"):
            target = pages / "supplement_flux_robustness.pdf"
            subprocess.run(["pdfunite", *map(str, components), str(target)], check=True)
            target.replace(output / target.name)
        else:
            for number, page in enumerate(components, start=1):
                shutil.copyfile(page, output / f"supplement_S{number}.pdf")


def write_tables(output, model_numbers, repeats, noisy):
    with (output / "table_V_frequency_models.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frequency_model", "four_all_r_s", "sparse_coupled_r_s", "sparse_all_r_s"])
        for label, name in (("Scalar", "scalar"), ("ExactBare", "exact_bare"), ("ExactTransmon", "exact_transmon")):
            writer.writerow([name, *[model_numbers["FluxModelCheck" + label + field]
                                     for field in ("Four", "NNFourNonzero", "NNFourAll")]])
    with (output / "table_VI_flux_rank_correlations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "samples", "coupled_pairs", "coupled_r_s_mean", "coupled_r_s_sd",
                         "all_r_s_mean", "all_r_s_sd"])
        for name, pairs in zip(CASES, (6, 24, 42, 120)):
            count = CASES[name][1]
            group = next(g for g in repeats if (g["case"], g["condition"], g["sample_count"]) == (name,"iid",count))
            scores = group["by_k"]["30"]
            writer.writerow([name, count, pairs, *[scores[m][stat] for m in ("spearman_nonzero","spearman_all")
                                                   for stat in ("mean","sd")]])
    with (output / "table_VII_paired_frequency_noise.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["k_CMI", "metric", "clean_mean", "clean_sd", "noisy_mean", "noisy_sd",
                         "paired_delta_mean", "paired_delta_sd"])
        for k in ("3", "30"):
            for metric in ("average_precision", "spearman_nonzero", "spearman_all"):
                row = noisy["by_k"][k][metric]
                writer.writerow([k, metric, *[row[condition][stat] for condition in
                                               ("clean", "noisy", "noisy_minus_clean") for stat in ("mean","sd")]])


def write_macros(path, numbers):
    path.write_text("".join(rf"\newcommand{{\{name}}}{{{value}}}" + "\n" for name,value in numbers.items()),
                    encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--full", action="store_true", help="Regenerate both reference samples and all robustness estimates")
    p.add_argument("--workers", type=int, choices=range(1,5), default=4)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    out = args.out.resolve()
    if out == DATA or DATA in out.parents:
        p.error("Choose an output directory outside the packaged input data.")
    out.mkdir(parents=True, exist_ok=True)
    reference = DATA / "reference"
    robust = DATA / "robustness"
    if args.full:
        reference, robust = out / "reference", out / "robustness"
        reference.mkdir(exist_ok=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(compute, name, reference) for name in CASES]
            matrices = dict(job.result() for job in jobs)
        atomic_json(reference / "model_matrices.json", {k:v for k,v in matrices.items() if v})
        recalculate_robustness(robust, args.workers)
    from scripts.robustness_numbers import export_numbers
    repeats = read(robust / "scores" / "summary.json")["groups"]
    records, summaries, numbers, model = reference_numbers(reference, repeats)
    atomic_json(out / "reference_metrics.json", summaries)
    write_macros(out / "flux_reference_numbers.tex", numbers)
    write_macros(out / "model_sensitivity_numbers.tex", model)
    write_tables(out, model, repeats, read(robust / "noisy_ranking/summary.json"))
    (out / "flux_robustness_numbers.tex").write_text(export_numbers(robust, robust/"screen_settings"),
                                                    encoding="utf-8")
    if not args.no_plots:
        render(records, summaries, robust, out / "figures")


if __name__ == "__main__":
    main()
