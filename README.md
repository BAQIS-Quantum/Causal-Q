# Causal-Q

Code and data for *A Model-Free, Data-Driven Causal Graph Framework for Characterization and Calibration of Superconducting Quantum Chips*.

The [paper reproduction package](reproduction/) contains the synthetic inputs, estimators, calibration query records, and scripts for Figures 2–10, Tables I–VIII, and supplemental Figures S1–S3. [Version v2026.11.20](https://github.com/BAQIS-Quantum/Causal-Q/tree/v2026.11.20) fixes the files used for the revision. [Download the versioned ZIP](https://github.com/BAQIS-Quantum/Causal-Q/archive/refs/tags/v2026.11.20.zip).

From a checkout of this repository, open the `reproduction/` directory. The versioned ZIP opens directly into the reproduction package. In a Python 3.10 environment, run

```sh
python -m pip install -r requirements.txt
python reproduce.py --out results
```

The [reproduction guide](reproduction/README.md) lists the system tools for PDF export, the command for regeneration from seeds, and the numerical sensitivity of the segmented-chain optimization. The [initial implementation](https://github.com/BAQIS-Quantum/Causal-Q/tree/c7c13fcda80cdf8a480bcb3bd5e8c2ed3fcd1c4e) is also available.
