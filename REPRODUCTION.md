# Reproduction protocol

## Guarantee and boundary

A fresh clone verifies all committed reports and machine-readable fixtures
without credentials. It can clone fixed upstream revisions, apply our overlays
and rerun public-artifact or local-proxy analyses. A live IBM campaign and the
historical Ma–Li calibration state are outside this guarantee.

```bash
git clone <this-repository-url>
cd quantum-runtime-estimator-replications
python3 scripts/verify_artifacts.py
bash scripts/bootstrap_upstreams.sh work
```

The ignored work/ directory holds fresh upstream clones. Bootstrap checks out
the pinned revision before applying any overlay.

## Qonductor

The public-artifact path requires no IBM credential. Create the environment
listed in tracks/qonductor/requirements-reproduction.txt, then run:

```bash
cd work/qonductor
PYTHONPATH=. python experiments/reproduce_public_results.py
PYTHONPATH=. python experiments/fake_scheduler_smoke.py
```

Compare outputs with artifacts/qonductor/. This recomputes metrics from
author-generated predictions and runs a fake-backend smoke test; it does not
retrain Qonductor or contact IBM.

The resource-estimator CSV is not safely joinable to the circuit/job database;
run the credential-free audit before attempting any row-level comparison:

```bash
python experiments/qonductor_mapping_audit.py --qonductor-root work/qonductor
```

It should report 100 resource rows, no stable join key, 7,449 database jobs
and 166,093 archived circuits. Do not attach those rows by position or nearest
runtime.

## Ma–Li

Bootstrap overlays the local DAG/preprocessing fixes, fake properties, manifests
and runners onto the pinned public checkout. The full paired adaptation is:

```bash
cd work/mali
python data_preparation/build_training_data_from_logs.py \
  --workers 8 --checkpoint-dir /tmp/mali_checkpoints_full
python data_preparation/build_training_data_from_logs.py \
  --devices osaka kyoto --backend-properties-dir data/fake_backend_properties \
  --workers 8 --output data/osaka_kyoto_fake_standardization.npy \
  --manifest data/osaka_kyoto_fake_manifest.csv
python experiments/replicate_mali_dag.py
python experiments/adapt_mali_real_qpu.py \
  --data data/osaka_kyoto_fake_standardization.npy \
  --manifest data/osaka_kyoto_fake_manifest.csv \
  --pretrained experiments/results/full_best.pt \
  --init pretrained --properties-kind fake --folds 10 --epochs 500 \
  --batch-size 32 --seed 1234 --device cuda --cpu-threads 8 \
  --output-dir experiments/results/adaptation_fake_full
python experiments/adapt_mali_real_qpu.py \
  --data data/osaka_kyoto_fake_standardization.npy \
  --manifest data/osaka_kyoto_fake_manifest.csv \
  --init random --properties-kind fake --folds 10 --epochs 500 \
  --batch-size 32 --seed 1234 --device cuda --cpu-threads 8 \
  --output-dir experiments/results/adaptation_fake_full_from_scratch
```

The expected aggregate JSON and fold-10 fixture are committed. Checkpoints are
derived, large and unnecessary to audit the reported predictions.

## Azizov et al. independent reproduction

The new track is deliberately separated from the four older target semantics.
It uses the pinned Ma–Li QASM pool as the public source circuit pool, but its
target is local Qiskit Aer noisy-simulator execution after transpilation.
The paper's code and raw runtime table are not public, so the result is not
advertised as an exact replication.

The credential-free P0 path is:

```bash
python tracks/azizov_independent/scripts/build_manifest.py \
  --mali-root work/mali \
  --output work/azizov_independent/source_manifest.csv
python tracks/azizov_independent/scripts/run_p0_smoke.py \
  --mali-root work/mali \
  --manifest work/azizov_independent/source_manifest.csv \
  --output artifacts/azizov_independent/p0_smoke.csv
python tracks/azizov_independent/scripts/summarize_p0.py \
  --input artifacts/azizov_independent/p0_smoke.csv \
  --ablation artifacts/azizov_independent/p0_feature_ablation.json \
  --output artifacts/azizov_independent/P0_REPORT.md
```

The runner records Qiskit versions, fake-backend class, seeds and the timing
protocol. It uses 1,024 shots, optimization levels 0--3 and one timed
`AerSimulator.run(...).result()` after an untimed warm-up. The committed P0
baseline is fitted with a circuit-ID grouped split:

```bash
python tracks/azizov_independent/scripts/train_p0_baselines.py \
  --input artifacts/azizov_independent/p0_smoke.csv \
  --output artifacts/azizov_independent/p0_baselines.json
python tracks/azizov_independent/scripts/compare_feature_blocks.py \
  --input artifacts/azizov_independent/p0_smoke.csv \
  --output artifacts/azizov_independent/p0_feature_ablation.json
```

P1 (all unique circuits, explicit 900-second censoring) and P2 (backend/family
held-out and transpiler-seed sensitivity) are intentionally separate commands
to prevent a smoke run from being mistaken for the paper-scale experiment.

To reproduce the completed local-feasible subset evaluation from the committed
rows (without rerunning Aer), run:

```bash
python tracks/azizov_independent/scripts/evaluate_subset_splits.py \
  --input artifacts/azizov_independent/feasible_q9_runtime.csv \
  --output-json artifacts/azizov_independent/FEASIBLE_Q9_SPLIT_EVALUATION.json \
  --output-md artifacts/azizov_independent/FEASIBLE_Q9_SPLIT_EVALUATION.md
```

This evaluates random-row, grouped-circuit, family-held-out and both
fake-backend-held-out splits. The grouped-circuit split is the primary local
generalization estimate; random-row results are retained as an optimistic
diagnostic.

For the separate gate-aware proxy against the recorded Ma–Li labels, use a
Qiskit 1.4.1 environment containing NumPy and the IBM Runtime fake provider:

```bash
python experiments/mali_qcre_proxy.py --mali-root work/mali
```

This transpiles all 340 Osaka/Kyoto QASM rows (seed 1234, optimization level
1) and writes `artifacts/validation/mali_qcre_proxy/`. To re-check the
statistics without repeating transpilation, pass the committed feature CSV:

```bash
python experiments/mali_qcre_proxy.py --mali-root work/mali \
  --features-csv artifacts/validation/mali_qcre_proxy/mali_qcre_proxy_features.csv
```

The resulting weighted path is a current FakeBackend target proxy. It is not a
recovered historical IBM calibration snapshot and is not interchangeable with
the Ma–Li observed `time_taken` target.

Run the final leakage/ablation pass (no transpilation required):

```bash
python experiments/mali_qcre_validation.py --mali-root work/mali
```

This groups repeated logical circuits by QASM SHA-256 and uses the same five
folds for every feature set.

To reproduce the direct source-specific estimator evaluation from the committed
feature table (no transpilation or IBM credential required), create an isolated
environment and install the pinned evaluation dependencies:

```bash
python3 -m venv .venv-mali-direct
.venv-mali-direct/bin/pip install -r experiments/requirements-mali-direct-estimator.txt
.venv-mali-direct/bin/python experiments/run_mali_direct_estimator.py
```

It writes grouped-logical-circuit, paired-backend diagnostic, strict
backend-plus-unseen-circuit and family-held-out OOF predictions to
`artifacts/validation/mali_direct_estimator/`. The strict split trains on the
other backend *and* other logical-QASM folds; it is the relevant transfer
check. The artifact consumes the committed current-FakeBackend feature CSV and
therefore reproduces our proxy evaluation, not the unavailable historical
Ma–Li transpilation/calibration state.

For a clean rerun that avoids overwriting the committed baseline, select an
alternate output directory:

```bash
.venv-mali-direct/bin/python experiments/run_mali_direct_estimator.py \
  --output-dir work/mali_direct_estimator_rerun
```

For transpiler-seed sensitivity, rerun the 340 rows with three seeds (the
committed run used Qiskit 1.4.1, optimization level 1):

```bash
python experiments/mali_qcre_seed_sensitivity.py --mali-root work/mali \
  --seeds 1234,2025,31415 --optimization-level 1
```

The seed run takes substantially longer than the report-only validation since
each seed repeats topology mapping and routing.

Once the seed-row CSV exists, report generation can be checked without
retranspiling:

```bash
python experiments/mali_qcre_seed_sensitivity.py --mali-root work/mali \
  --rows-csv artifacts/validation/mali_qcre_seed_sensitivity/mali_qcre_seed_sensitivity_rows.csv \
  --seeds 1234,2025,31415 --optimization-level 1
```

## CDAA/QCRE

Use the environment notes in tracks/cdaa_qcre/README.md. Bundled artifact
analysis is credential-free. The bootstrap supplies the six archived
instruction-duration snapshots used by the proxy; they are not live calibration
queries. From the capsule root, materialize the independent QASM paths with:

```bash
cd work/cdaa_independent/0_compilation
# New environment: fake-backend Qiskit and TKET topology proxies
python offline_qiskit_compile.py
python offline_qiskit_translate.py
python offline_tket_compile.py
python offline_tket_translate.py
# Old environment: historical SQGM/SABRE artifact driver (no IBM credential)
mkdir -p qasm/compiled
python sqgm/main.py sqgm/exp-in/exp_eagle.json
python sqgm/main.py sqgm/exp-in/exp_heron.json
python offline_translate.py
python independent_all_metrics.py
```

The old driver may need the `requirements-sqgm.txt` environment. SQGM/SABRE
use the historical artifact driver; offline Qiskit/TKET uses
FakeSherbrooke/FakeMarrakesh topology proxies and must retain that label.
Compare output metrics with artifacts/cdaa_qcre/.

## cuTensorNet

This track requires an NVIDIA GPU and CUDA 13 compatible environment:

```bash
python3 -m venv .venv-cutensornet
.venv-cutensornet/bin/pip install \
  'cuquantum-python-cu13==26.6.0' 'cupy-cuda13x[ctk]'
tracks/cutensornet/code/run_cutensornet_runtime_benchmark.sh \
  --families ghz,hea,qaoa_cycle,random_brickwork,qft \
  --qubits 16,20,24,28,30 --warmups 2 --repeats 5 \
  --output-dir run-output/cutensornet/current_grid
```

Compare target columns separately: cutensornet_runtime_est_s,
first_contract_gpu_s, contract_gpu_median_s and end_to_end_first_s are not
interchangeable. A different GPU/software stack produces a new measurement,
not a failed reproduction.

## Matched dense-statevector runtime matrix

This local supplementary track measures one statevector implementation rather
than comparing frameworks: allocate a fresh state, apply a materialized gate
schedule, and calculate `<Z_0>`. It therefore supplies a complete CPU/GPU ×
complex64/complex128 matrix without conflating a framework change with a
device change. It excludes sampling and compilation/transpilation.

Use a CUDA-enabled PyTorch environment. The recorded run used Python 3.13,
PyTorch `2.11.0+cu128` and CUDA 12.8; use the matching wheel index below, or
an equivalent PyTorch wheel for a different review machine:

```bash
ROOT=/path/to/quantum-runtime-estimator-replications
OUT="$ROOT/artifacts/simulator_runtime_v1/torch_dense_statevector_v1"

python3 -m venv "$ROOT/.venv-torch-matrix"
"$ROOT/.venv-torch-matrix/bin/python" -m pip install 'torch==2.11.0' \
  --index-url https://download.pytorch.org/whl/cu128
PY="$ROOT/.venv-torch-matrix/bin/python"

python3 -m venv "$ROOT/.venv-runtime-eval"
"$ROOT/.venv-runtime-eval/bin/python" -m pip install \
  -r "$ROOT/experiments/simulator_runtime_v1/requirements-evaluation.txt"
EVAL_PY="$ROOT/.venv-runtime-eval/bin/python"

"$PY" "$ROOT/experiments/simulator_runtime_v1/run_torch_statevector_matrix.py" \
  --output-dir "$OUT"
"$EVAL_PY" "$ROOT/experiments/simulator_runtime_v1/evaluate_torch_statevector_matrix.py" \
  --input "$OUT/records.csv" --output-dir "$OUT/evaluation"
"$EVAL_PY" "$ROOT/experiments/simulator_runtime_v1/summarize_torch_statevector_matrix.py" \
  --run-dir "$OUT"
```

The runner is append-only and resumes by `record_id`. The checked-in 96-row
result is a local RTX 5070 Ti/16-CPU-thread measurement, not an external
simulator or cross-GPU claim. See `experiments/simulator_runtime_v1/README.md`
for the exact target semantics and output files.

### Structural v2 extension

The v2 follow-up adds seeded random topologies and a q26/q28 GPU-memory
frontier under the same target contract. Its clean-clone commands, environment,
split definitions and expected checks are kept in
[`experiments/simulator_runtime_v2/README.md`](experiments/simulator_runtime_v2/README.md).
The committed final artifact is
`artifacts/simulator_runtime_v2/dense_statevector_structural_v2_20260920/`.
Run the standard verifier after materializing a clone; it checks raw coverage,
source provenance, OOM retention, estimator metrics and the q28 envelope
boundary without rerunning GPU timing.

## CUDA-Q matched simulator pilot

CUDA-Q is installed in a separate Python 3.13 environment. The pilot records
`qpp-cpu` FP64, NVIDIA FP32 and NVIDIA FP64 `cudaq.sample` wall-clock labels.
It does not contact a QPU or cloud service. First-call and warm-call timings
are separate targets.

```bash
ROOT=/path/to/quantum-runtime-estimator-replications
CUDAQ_PY=/path/to/Capstone-Project/.venv-cudaq313/bin/python
EVAL_PY=/path/to/Capstone-Project/.venv/bin/python
OUT="$ROOT/artifacts/cudaq_runtime/<run>"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 "$CUDAQ_PY" \
  "$ROOT/experiments/cudaq_runtime/run_cudaq_matrix.py" \
  --targets gpu_fp32,gpu_fp64 \
  --families ghz,hea,qaoa_cycle,random_brickwork \
  --widths 16,20,24,28 --shots 32 --warmups 1 --repeats 3 \
  --output "$OUT/gpu_matrix.csv"

CUDA_VISIBLE_DEVICES="" "$CUDAQ_PY" \
  "$ROOT/experiments/cudaq_runtime/run_cudaq_matrix.py" \
  --targets cpu_fp64 \
  --families ghz,hea,qaoa_cycle,random_brickwork \
  --widths 16,20,22 --shots 32 --warmups 1 --repeats 3 \
  --output "$OUT/cpu_matrix.csv"

python3 "$ROOT/experiments/cudaq_runtime/merge_cudaq_matrix.py" \
  --inputs "$OUT/cpu_matrix.csv" "$OUT/gpu_matrix.csv" \
  --output "$OUT/cudaq_matrix.csv"
"$EVAL_PY" "$ROOT/experiments/cudaq_runtime/evaluate_cudaq_matrix.py" \
  --input "$OUT/cudaq_matrix.csv" --output-dir "$OUT/evaluation"
```

The checked-in run is
`artifacts/cudaq_runtime/cudaq_matrix_20260921/`; its report records the
44-row coverage, target semantics, raw timing table and small-corpus split
diagnostics. Do not append CUDA-Q rows to the PyTorch, Aer or cuTensorNet
tables without preserving the target contract.

### CUDA-Q MPS extension

The MPS experiment freezes one bond-cap/cutoff configuration per process and
records both `cudaq.get_state` and `cudaq.sample` timings. It also reads the
returned tensor extents to record post-run bond/storage diagnostics. Recreate
the checked-in 48-row matrix with:

```bash
ROOT=/path/to/quantum-runtime-estimator-replications
CUDAQ_PY=/path/to/Capstone-Project/.venv-cudaq313/bin/python
OUT="$ROOT/artifacts/cudaq_runtime/<run>"
mkdir -p "$OUT"
for cap in 2 4 8 16; do
  CUDA_VISIBLE_DEVICES=0 "$CUDAQ_PY" \
    "$ROOT/experiments/cudaq_runtime/run_cudaq_mps_matrix.py" \
    --families ghz,hea,qaoa_cycle,random_brickwork \
    --widths 16,20,24 --max-bond "$cap" --abs-cutoff 1e-12 \
    --precision fp64 --shots 32 --warmups 1 --repeats 2 \
    --reference-width-max 16 \
    --output "$OUT/mps_cap${cap}.csv"
done
python3 "$ROOT/experiments/cudaq_runtime/merge_cudaq_matrix.py" \
  --inputs "$OUT/mps_cap2.csv" "$OUT/mps_cap4.csv" "$OUT/mps_cap8.csv" "$OUT/mps_cap16.csv" \
  --output "$OUT/mps_matrix.csv"
/path/to/Capstone-Project/.venv/bin/python \
  "$ROOT/experiments/cudaq_runtime/evaluate_cudaq_mps_matrix.py" \
  --input "$OUT/mps_matrix.csv" --output-dir "$OUT/evaluation"
```

The static evaluator excludes `observed_max_bond`, tensor storage and fidelity
from its features. These are post-run diagnostics or labels, not information
available to a pre-run estimator.
