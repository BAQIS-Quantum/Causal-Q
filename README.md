# Causal-Q (self-contained)

All working copies of legacy scripts are under `CausCal-Q/scripts/`; original repo files stay untouched. Environment: `conda activate calibration-env`.

## Dependencies

```
pip install numpy scipy torch matplotlib networkx npeet tqdm ipywidgets
```

## Common commands (run from repo root)

```
# 1) Generate data
python CausCal-Q/cli.py gen-data --width 4 --length 4 --samples 40000 \
  --coupling dense --out CausCal-Q/fullinterac_dataset_gpu_4_4_dense_withoutF.npz --compute-F

# 2) Layer-1 PC
python CausCal-Q/cli.py pc --npz CausCal-Q/fullinterac_dataset_gpu_4_4_dense_withoutF.npz \
  --alpha1 0.0 --out CausCal-Q/P1_graph.json --fig CausCal-Q/layered_graph.pdf

# 3) MI on P1 edges
python CausCal-Q/cli.py mi --npz CausCal-Q/fullinterac_dataset_gpu_4_4_dense_withoutF.npz \
  --graph CausCal-Q/P1_graph.json --out CausCal-Q/mi_results.npz \
  --heatmap phi_to_omega_heatmap.pdf --scatter hij_vs_mi_scatter.pdf

# 4) Bipartite causal plot
python CausCal-Q/cli.py bipart --mi CausCal-Q/mi_results.npz --out CausCal-Q/bipartite_mi.pdf

# 5) Coupling heatmap
python CausCal-Q/cli.py hij --npz CausCal-Q/fullinterac_dataset_gpu_4_4_dense_withoutF.npz \
  --out CausCal-Q/hij_heatmap.pdf
```

## Pipeline mapping

1. `gen-data` → `scripts/unified_calibration_model_v4_GPU.py`
2. `pc` → `scripts/pc_analysis_with_unified_model-v5.py` (defaults to P1 only)
3. `mi` → `scripts/load_directed_plot.py` (MI on P1 edges)
4. `bipart` → `scripts/plot_bipart_graph.py`
5. `hij` / `scatter` → data/MI visualizations

To add more PC layers or extra plots, extend `CausCal-Q/causal_q/pipeline.py`.
