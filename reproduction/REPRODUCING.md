# Reproducing the paper

Run the commands from the directory containing `reproduce.py`. In a clone of the main branch, this is `reproduction/`. The reference environment uses CPython 3.10.14 on CPU, NumPy 1.26.4, and SciPy 1.14.1.

## Install and run

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --out results
```

On Windows, activate the environment with `.venv\Scripts\activate`. The [pinned requirements](requirements.txt) install the bundled NPEET 1.0.1 wheel from [upstream commit 8b0d948](https://github.com/gregversteeg/NPEET/tree/8b0d9485423f74e5eb199324cf362765596538d3). For all PDF exports, install `rsvg-convert` (librsvg) and `pdfunite` (Poppler). The numerical results can be generated with Python alone.

The default run rebuilds the figures and tables from the supplied samples, dependence estimates, and calibration query records. Figure 1 is a workflow diagram. To regenerate the samples from seeds, rerun the estimators, and integrate the pulse dynamics, use a separate directory and one to four workers.

```sh
python reproduce.py --from-scratch --out results-full --workers 4
```

## Inputs and results

| Paper | Inputs | Results |
| --- | --- | --- |
| Figs. 2–3 and Table V (Appendix A) | [Flux samples and couplings](flux/data/reference/) | [Figures](results/flux/figures/), [ranking metrics](results/flux/reference_metrics.json), [frequency models](results/flux/table_V_frequency_models.csv) |
| Fig. 4 | [Pulse samples](pulse/data/) | [Pulse graphs](results/pulse/Fig4.pdf), [counts by topology](results/pulse/summary.json) |
| Fig. 5, Tables I–III, Appendix B | [Calibration query records](calibration/data/) | [Query curves, tables, and support perturbations](results/calibration/) |
| Fig. 6 and Table IV | [Fixed-gate records](fixed_gate/data/) | [Query curves and Haar fidelities](results/fixed_gate/) |
| Figs. 7–10, Tables VI–VII (Appendix C), Figs. S1–S3 | [Robustness samples and scores](flux/data/robustness/) | [Figures](results/flux/figures/), [rank correlations](results/flux/table_VI_flux_rank_correlations.csv), [paired noise comparisons](results/flux/table_VII_paired_frequency_noise.csv) |
| Table VIII (Appendix D) | [Unrounded MI and CMI estimates](orientation/data/) | [Collider counts](results/orientation/table_VIII.csv) |

## Numerical settings

Flux screening uses 130 neighbors for MI and 3 for CMI. Flux plots use 3 neighbors for displayed MI and 30 for endpoint-conditioned pair scores. The pulse heatmaps use 130 neighbors. The Appendix D collider diagnostic uses 10 neighbors, a 0.02-bit cutoff, and conditioning sets of up to two variables.

In the full run, the regenerated Fig. 4 graphs feed Figs. 5–6 and Appendix D. For the segmented-chain amplitude-block comparison, the code reintegrates the published trial amplitudes and recovers the single-qubit-block loss AUC of 0.6920. An independent search on the reference machine gave 0.6833. To repeat that search, run

```sh
python calibration/simulate.py --out independent --case chain --study policy --repeat-block-search
```

For Table IV, independent fixed-gate optimization reproduced the two mean Haar fidelities, 0.999470 and 0.999673, to six decimal places. The last CSV column checks the archived endpoints with tighter integration tolerances. The change can vary at machine precision and was at most $3.4\times 10^{-16}$ in the paper.
