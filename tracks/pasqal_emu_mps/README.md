# Pasqal EMU-MPS GPU track

This is a separate neutral-atom emulator track. It does not pool its runtime
target with Qiskit Aer, Ma–Li, Qonductor, or QPU service-runtime targets.

The benchmark records end-to-end wall time and EMU-MPS per-step statistics:
bond dimension, MPS memory footprint, RSS and step duration. The local GPU
smoke uses the RTX 5070 Ti and is intended as a reproducible starting point,
not as a universal simulator-runtime claim.

The isolated environment can reuse the CUDA-enabled PyTorch installation from
`Quantum-Execution-Time-Prediction/.venv-mali-gpu` through `PYTHONPATH`:

```bash
PYTHONPATH=/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/lib/python3.10/site-packages \
  .venv-emu-mps-gpu/bin/python tracks/pasqal_emu_mps/scripts/run_gpu_frontier.py \
  --output artifacts/pasqal_emu_mps/gpu_frontier.csv
```

The output CSV and `.environment.json` are the reproducibility artifacts for
the run. EMU-MPS is public at <https://github.com/pasqal-io/emulators>.

## Fidelity-based bond-dimension pilot

The post-run `max_bond_dimension_observed` is not a pre-run prediction. To
make the target operational, `scripts/run_threshold_sweep.py` first computes
a high-χ reference state and then sweeps the allowed rungs
`1, 2, 4, ..., 256`. The label is `required_max_bond_dim`: the smallest rung
whose final-state fidelity against that reference reaches the configured
target (0.99 in the checked-in run). This is the closest local analogue of
the QCRE threshold-classification approach, while retaining EMU-MPS-specific
semantics.

Reproduce the 12-label pilot with:

```bash
PYTHONPATH=/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/lib/python3.10/site-packages \
  .venv-emu-mps-gpu/bin/python tracks/pasqal_emu_mps/scripts/run_threshold_sweep.py \
  --output artifacts/pasqal_emu_mps/threshold_sweep.csv

PYTHONPATH=/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/lib/python3.10/site-packages \
  .venv-emu-mps-gpu/bin/python tracks/pasqal_emu_mps/scripts/fit_threshold_predictor.py \
  artifacts/pasqal_emu_mps/threshold_sweep_labels.csv \
  --output-prefix artifacts/pasqal_emu_mps/threshold_predictor
```

The checked-in labels are:

| family | N=8 | N=12 | N=16 |
|---|---:|---:|---:|
| weak | 2 | 2 | 4 |
| strong | 8 | 16 | 32 |
| detuned | 8 | 16 | 16 |
| two_stage | 8 | 16 | 32 |

The small random-forest classifier is deliberately reported as a feasibility
pilot, not as a universal estimator. Size-held-out evaluation reaches 25.0%
exact rung accuracy and 100.0% within one rung; family-held-out evaluation
reaches 50.0% exact and 75.0% within one rung. The supplied family is an
input feature, so this is not yet an unknown-family end-to-end predictor.
There are only 12 labels, and all references are generated in the same local
EMU-MPS configuration. The labels and all per-rung fidelity measurements are
in `artifacts/pasqal_emu_mps/threshold_sweep_labels.csv` and
`artifacts/pasqal_emu_mps/threshold_sweep.csv`.
