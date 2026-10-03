# Maestro component adaptation on local QCSim

Status: completed measurement and out-of-fold comparison. This is an
adaptation of Maestro's component model, not a reproduction of its closed
simulator selector and not a reproduction of the paper's reported results.

## What was measured

- One exact-hash panel: 150 QASM identities, widths 2–9, five frozen C44 folds.
- QCSim CPU Statevector, FP64, 1,000 shots. The target is the engine-reported
  `result.time_taken` clock; process startup and Python result conversion are
  retained as separate host clocks and are not substituted for this target.
- 143 panel labels observed; seven repeated-measurement circuits are
  unavailable. All 150 remain in `panel_targets.csv` and in assigned coverage.
- The 143 observed hashes contain 112 full-register circuits, 23 terminal
  partial-measurement circuits and 8 circuits with wider classical registers.
  Original ordered qubit-to-classical-bit maps and widths are in the plan,
  attempt ledger and target table.
- Full phase: 3,930 attempt identities across 112 calibration cells and 150
  panel hashes. 150 primary calibration attempts reference matching pilot
  measurements; the immutable full ledger contains 3,825 successful and 105
  explicitly unsupported terminal rows. Pilot includes 300 measured calls
  across primary and thread-environment diagnostic arms.

The native binding serializes both simulator and method enums as integer `0`
in the per-attempt fields. Before accepting a timed result, the worker compared
those returned values against the explicitly configured pinned
`SimulatorType.QCSim.value` and `SimulationType.Statevector.value`; both matched.
The run manifest records their readable names. Thus `0` is the binding's enum
encoding, not an unknown simulator selection.

The seven unavailable QASMs each repeat measurement of the same qubits into a
second classical register. They are retained as unavailable because this
contract accepts injective terminal measurement maps and does not discard a
readout layer.

## Results

All learned/common baselines are evaluated on the same 143 observed QCSim
labels and frozen outer folds. MAE is absolute error in seconds; convert to
microseconds for readability.

| Method | OOF rows | MAE (µs) | R² |
|---|---:|---:|---:|
| Outer-train median | 143 | 442.64 | −0.0422 |
| Grouped Ridge | 143 | 155.20 | 0.9231 |
| Source-DAG graph adaptation (CUDA) | 143 | 642.58 | −0.0094 |
| Maestro component adaptation | 13 | 25.34 | −2.9826 |

The component result has only 13 scored rows: the fixed decomposition produced
a valid width coefficient only at n=2. At n=3–9 the added-CX increment was
negative, unresolved or unstable under the frozen identification rule. Its
small MAE is therefore a low-coverage result, not a win over methods evaluated
on 143 rows. The component sampling fit uses zero-state full-register controls;
its prediction does not model measured-bit count, output width or outcome
support. Its partial/wider-register scores are out-of-layout extrapolation at
n=2 and are diagnostic only; those layouts had no matching calibration cells.
No panel target was used to fit or repair its coefficients.

Measurement-group residuals are in `measurement_group_metrics/`. For grouped
Ridge, MAE is 182.45 µs on 112 full-register rows, 57.38 µs on 23 partial rows,
and 54.95 µs on 8 wider-register rows. These are descriptive small-panel
subgroups; they do not establish that partial measurement is inherently
easier. The CUDA source-DAG graph adaptation has MAE 705.92, 428.91 and
370.23 µs respectively on those same groups.

Graph training used CUDA, seeds 42/1234/31415, five outer folds and the frozen
500-epoch protocol. Runtime labels came only from this QCSim run; Aer, CUDA-Q,
Maestro historical, and QPU labels were not imported.
`evaluation/training_environment.json` records the GPU process and contemporaneous
GPU utilization/memory samples, Python/Torch/CUDA versions, GPU model, and the
hash of the imported graph helper. The run manifest's OpenBLAS 28-thread
fingerprint was captured in the controller process. Each timed child received
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `OMP_THREAD_LIMIT=1`; the
controller fingerprint is not a measurement of the child's BLAS pool. QCSim's
timed code is native C++ and its OpenMP limit is pinned separately.

## Reproduction on the measured host

The plan pins the 150 exact hashes, folds, source file locations, seed
registry, protocol and calibration cells. The source QASM corpus, Maestro
checkout/build and Python environments are outside this publish candidate;
this package is not a clean-clone installation recipe. The existing protocol
contains the complete host pins.

From the candidate root, with the original source workspaces available:

```bash
PYTHONPATH=. /home/server/Documents/.venv-qonductor/bin/python \
  benchmark_v1/scripts/prepare_maestro_cpu_component.py \
  --source-dir /home/server/Documents/Quantum-Execution-Time-Prediction/data/quantum_circuits \
  --output-dir work/reproduction-plan

PYTHONPATH=. /home/server/Documents/.venv-qonductor/bin/python \
  benchmark_v1/scripts/run_maestro_cpu_component_v1.py \
  --action pilot10 --plan work/reproduction-plan/plan.json \
  --output-dir work/reproduction-pilot

PYTHONPATH=. /home/server/Documents/.venv-qonductor/bin/python \
  benchmark_v1/scripts/run_maestro_cpu_component_v1.py \
  --action full --plan work/reproduction-plan/plan.json \
  --output-dir work/reproduction-full \
  --pilot-report work/reproduction-pilot/pilot_report.json

PYTHONPATH=. /home/server/Documents/.venv-qonductor/bin/python \
  benchmark_v1/scripts/run_maestro_cpu_component_v1.py \
  --action aggregate --plan work/reproduction-plan/plan.json \
  --output-dir work/reproduction-full

PYTHONPATH=. /home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python \
  benchmark_v1/scripts/evaluate_maestro_cpu_component_v1.py \
  --panel-targets work/reproduction-full/panel_targets.csv \
  --calibration-model work/reproduction-full/calibration_model.json \
  --output-dir work/reproduction-evaluation

PYTHONPATH=. /home/server/Documents/.venv-qonductor/bin/python \
  benchmark_v1/scripts/summarize_maestro_measurement_groups.py \
  --panel-targets work/reproduction-full/panel_targets.csv \
  --oof-predictions work/reproduction-evaluation/oof_predictions.csv \
  --output-dir work/reproduction-measurement-groups
```

Only one QCSim timing worker is used. The training environment was Python
3.10.21, Torch 2.7.1+cu128, PyG 2.6.1, CUDA available, NVIDIA RTX 5070 Ti.
The host measurement environment was Python 3.10.21, Qiskit 2.5.2, NumPy
2.2.6, SciPy 1.15.3, qoro-maestro 0.3.1, and the pinned native Maestro/QCSim
libraries. The full test suite after implementation changes passed 217 tests
and 4 subtests. No Git commit or push was made.

## Files

- `plan.json`, `protocol.json`, `static_preflight.json`: frozen inputs and
  preflight receipt.
- `pilot/`: all paired pilot attempts, gate report and run manifest.
- `full/`: append-only full attempt ledger, run manifest, calibration model,
  and one target row per exact QASM hash.
- `evaluation/`: OOF predictions, method metrics, component predictions and
  source-hash manifest.
- `measurement_group_metrics/`: reproducible residual table by measurement
  layout, with its input/output hash receipt.
- `SHA256SUMS`: checksums for every packaged file except the checksum list.
