# Ma–Li overlay and reproduction notes

The upstream source is cloned at commit 32c392a. The overlay contains only our
changes and inputs: Qiskit compatibility changes, offline graph-data builder,
fake Osaka/Kyoto properties, manifests and the two experimental runners.

The local CPU environment used Python 3.10.21 with Torch 1.13.1, Qiskit 0.44.2,
PyG 2.5.2, NumPy 1.23.5, SciPy 1.11.4, pandas 2.1.4 and scikit-learn 1.3.0.
The full adaptation used a distinct RTX 5070 Ti environment with Torch
2.7.1+cu128 and PyG 2.6.1. Batch size is 32 because the paper's batch 128
exceeded local 16-GiB VRAM.

The fake backend property files are versioned offline snapshots. They permit
reproduction of our proxy adaptation and do not reconstruct the historical
calibration state at the time of the original 340 QPU observations. The
seed-1234 from-scratch control is mandatory when interpreting transfer results;
see RESULTS.md and artifacts/mali/fold10_qwalk_outlier.json.
