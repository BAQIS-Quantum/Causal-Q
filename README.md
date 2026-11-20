# Causal-Q

Changing one control on a superconducting quantum chip can affect several qubits. Causal-Q maps dependencies between control settings, qubit frequencies, and gate performance. It uses mutual information and conditional mutual information to screen for links in calibration data. The order of the calibration steps sets the direction of each link.

The repository includes two simulation benchmarks. The first uses flux and frequency data to rank residual couplings between qubits. The second uses pulse-response graphs to suggest which qubits to recheck after a control change and which controls to probe together during gate calibration.

These benchmarks accompany the paper *A Model-Free, Data-Driven Causal Graph Framework for Characterization and Calibration of Superconducting Quantum Chips*.

## Installation

With a Python 3.10 environment active, run

```sh
git clone https://github.com/BAQIS-Quantum/Causal-Q.git
cd Causal-Q/reproduction
python -m pip install -r requirements.txt
```

If you download the versioned ZIP, run the last command in the extracted folder.

[Reproduction guide](reproduction/REPRODUCING.md) · [Download code and data](https://github.com/BAQIS-Quantum/Causal-Q/archive/refs/tags/v2026.11.20.1.zip)
