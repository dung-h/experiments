# Cross-track validation experiments

Ma–Li and Azizov compiled-transfer scripts in this directory belong to
**Cluster 1** in [`docs/FINDINGS_CLUSTERS.md`](../docs/FINDINGS_CLUSTERS.md).
They share the Osaka/Kyoto `result.time_taken` labels. They do not pool
Quantum Rings or cuTensorNet clocks.

Family-aware paper scripts are *not* here; they are vendored under
[`tracks/quantum_rings/family_aware_paper/experiments/`](../tracks/quantum_rings/family_aware_paper/experiments/).

## Qonductor mapping audit

```bash
python experiments/qonductor_mapping_audit.py --qonductor-root work/qonductor
```

The audit checks whether the 100-row public resource-estimator CSV has a stable
join to the archived circuit/job database. It does not guess a row mapping.

## Ma–Li / QCRE proxy

```bash
python experiments/mali_qcre_proxy.py --mali-root work/mali
```

The committed 340-row feature artifact can be re-scored without repeating the
transpilation (useful for checking report code):

```bash
python experiments/mali_qcre_proxy.py \
  --mali-root work/mali \
  --features-csv artifacts/validation/mali_qcre_proxy/mali_qcre_proxy_features.csv
```

The script transpiles each Ma–Li logical QASM with the current FakeOsaka or
FakeKyoto target, reconstructs a backend-target weighted critical path, and
compares that proxy to the upstream observed `time_taken`. It reports
Pearson/Spearman association and five-fold one-dimensional log calibration for
logical depth, two-qubit depth, physical depth and the weighted path. This is
`OUR_PROXY`: the historical physical circuits and calibration snapshots are not
claimed to be recovered. Qiskit's ALAP scheduling pass is intentionally not
used because it is not robust for some large Ma–Li QASM inputs under Qiskit
1.4; the duration calculation is explicit and reproducible in the script.

## Final Ma–Li validation pass

Run the short validation pass on the committed feature CSV:

```bash
python experiments/mali_qcre_validation.py --mali-root work/mali
```

It groups repeated logical circuits by QASM SHA-256, applies one fixed grouped
five-fold split to all features and ablations, checks duration coverage, and
fits the optional `1024 * weighted_path` affine calibration. The seed-sensitivity
run re-transpiles the same 340 rows with three seeds:

```bash
python experiments/mali_qcre_seed_sensitivity.py --mali-root work/mali \
  --seeds 1234,2025,31415 --optimization-level 1
```

The committed seed-row CSV can be re-scored without retranspiling:

```bash
python experiments/mali_qcre_seed_sensitivity.py --mali-root work/mali \
  --rows-csv artifacts/validation/mali_qcre_seed_sensitivity/mali_qcre_seed_sensitivity_rows.csv \
  --seeds 1234,2025,31415 --optimization-level 1
```

Both reports are diagnostic extensions of the proxy result; they do not turn a
FakeBackend reconstruction into historical QPU calibration data.

## Ma–Li direct source-specific estimator

The direct evaluation consumes the committed 340-row proxy feature table. It
does not retranspile or submit a QPU job. Create an isolated recent Python
environment, then install the pinned evaluation dependencies:

```bash
python3 -m venv .venv-mali-direct
.venv-mali-direct/bin/pip install -r experiments/requirements-mali-direct-estimator.txt
```

Run the evaluator with that interpreter:

```bash
.venv-mali-direct/bin/python experiments/run_mali_direct_estimator.py
```

It predicts `log1p(result.time_taken seconds)` from target-specific compiled
structure and the weighted-path proxy. Its primary split is five-fold by logical
QASM SHA-256. It also records a deliberately weaker paired-backend diagnostic,
a strict backend-plus-unseen-logical-circuit 2 × 5 split, and leave-one-family
out. The latter two must be read as ten correlated outer partitions / two
backend directions, not as broad device cross-validation.

The output directory contains a report, all model/split aggregate metrics,
per-fold metrics, strict per-backend metrics, all outer-fold predictions, and
input provenance:

```text
artifacts/validation/mali_direct_estimator/
```

No estimator is fitted and saved for deployment by this script. The output is
an evaluation artifact: a model/selection protocol must be frozen before
fitting an operational source-specific model.

## Ma–Li saved-model graph intervention audit

The graph intervention audit tests what the two saved 10-fold models rely on
at inference time. Run it with the Ma–Li GPU environment because it requires
the upstream PyTorch/PyG model definition and local checkpoints:

```bash
/path/to/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python \
  experiments/mali_graph_intervention_audit.py \
  --mali-root /path/to/Quantum-Execution-Time-Prediction
```

It evaluates the exact held-out rows after neutralizing one standardized node
feature block, removing/reversing/rewiring dependency edges, or leaving the
input unchanged. These are test-time, out-of-distribution interventions on
already-trained checkpoints, not retrained feature ablations. The canonical
interpretation and raw out-of-fold predictions are under
`artifacts/mali/graph_intervention_audit_20260921/`.

## Ma–Li deep physical-DAG protocol

The leakage-safe split manifest is generated before any model fitting. It uses
the SHA-256 digest of raw QASM bytes as circuit identity, and connects filename
families through identical QASM hashes before family holdout:

```bash
python experiments/mali_split_audit.py \
  --mali-root /path/to/Quantum-Execution-Time-Prediction \
  --output-dir artifacts/validation/mali_deep_protocol_v1/split_audit
```

The compact physical corpus uses the frozen proxy environment (Qiskit 1.4.1
and `qiskit-ibm-runtime` 0.36.1). It stores each unique `(QASM hash, backend)`
native-operation DAG once under `work/` and keeps the 340 observed labels in a
separate artifact manifest. The generated NPZ cache is intentionally ignored
by Git and can be resumed after interruption:

```bash
/path/to/.venv-cdaa-new/bin/python \
  experiments/build_mali_physical_dag.py \
  --mali-root /path/to/Quantum-Execution-Time-Prediction \
  --output-dir work/mali_physical_dag_v1 \
  --artifact-dir artifacts/validation/mali_physical_dag_v1
```

Use `--limit 2` first to check the environment. The builder is a
`CURRENT_FAKE_SNAPSHOT_PROXY`: target durations and layouts are reconstructed
from the present FakeOsaka/FakeKyoto package, not recovered historical IBM
calibration. The protocol and model ladder are documented in
`docs/MA_LI_DEEP_RUNTIME_ESTIMATOR_PROTOCOL_2026-09-21.md`.

Before loading graph tensors, run the static gate using only the generated
manifest and graph summaries:

```bash
python experiments/evaluate_mali_physical_baselines.py \
  --artifact-dir artifacts/validation/mali_physical_dag_v1 \
  --split-manifest artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv
```

This compares the one-feature QCRE path, compiled features, and compact
physical node/edge summaries on identical grouped and strict splits. It is a
screening baseline; it is not a graph-neural result.

The richer static descriptor table used by the evaluator is regenerated with:

```bash
python experiments/build_mali_graph_summary_features.py \
  --artifact-dir artifacts/validation/mali_physical_dag_v1 \
  --graph-dir work/mali_physical_dag_v1/graphs
```

It adds layer, native-opcode, duration-distribution and criticality summaries;
these are still pre-run static features and remain separate from the observed
runtime label.

The first ordered-DAG residual pilot is reproducible with the CUDA/PyG
environment after the NPZ cache exists:

```bash
/path/to/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python \
  experiments/train_mali_layer_dag_rnn.py \
  --artifact-dir artifacts/validation/mali_physical_dag_v1 \
  --split-manifest artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv \
  --graph-dir work/mali_physical_dag_v1/graphs --device cuda
```

This is deliberately labelled a coarsened-DAG GRU pilot: it is not a full
native-operation GNN and its report must be read alongside the static baseline.

The next diagnostic pass is deterministic and CPU-only. It attributes the rich
baseline gain to interpretable feature blocks and audits QWalk as a tail/OOD
case under the same outer folds:

```bash
python experiments/evaluate_mali_feature_ablation.py \
  --artifact-dir artifacts/validation/mali_physical_dag_v1 \
  --split-manifest artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv

python experiments/mali_qwalk_forensics.py \
  --artifact-dir artifacts/validation/mali_physical_dag_v1 \
  --split-manifest artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv
```

The ablation adds layer, timing, edge and opcode blocks one at a time; all
preprocessing is train-only. The QWalk report gives feature percentiles,
standardized distance to the training support, nearest training row and OOF
errors. Since QWalk has two rows for one QASM hash, these files are a tail
diagnostic, not an independent score.

The critical-subgraph follow-up extracts high-criticality bottleneck summaries
and tests whether they add signal beyond the rich aggregate descriptors:

```bash
python experiments/build_mali_critical_subgraph_features.py \
  --artifact-dir artifacts/validation/mali_physical_dag_v1 \
  --graph-dir work/mali_physical_dag_v1/graphs
python experiments/evaluate_mali_critical_subgraph.py \
  --artifact-dir artifacts/validation/mali_physical_dag_v1 \
  --split-manifest artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv
```

The uncertainty pass audits the 40 duplicated `(QASM hash, backend)` cells and
builds split-conformal log-runtime intervals with a held-out calibration group:

```bash
python experiments/mali_uncertainty_calibration.py \
  --artifact-dir artifacts/validation/mali_physical_dag_v1 \
  --split-manifest artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv
```

The resulting interval coverage is empirical for this Ma--Li dataset and proxy;
it is not a hardware-wide confidence guarantee.

Before fitting a larger GNN, reproduce the topology control. It holds node
features, folds, residual target and model capacity fixed while changing only
the binned adjacency. The committed result is a CPU grouped-QASM screen because
the shared GPU was occupied during the run; it must not be relabelled as a full
strict-backend validation.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /path/to/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python \
  experiments/train_mali_topology_controls.py \
  --device cpu --evaluation grouped --epochs 15 --patience 4 \
  --seeds 1234 --output-name topology_controls_grouped_cpu_screen

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /path/to/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python \
  experiments/train_mali_topology_controls.py \
  --device cpu --evaluation grouped --epochs 15 --patience 4 \
  --seeds 2025 31415 --output-name topology_controls_grouped_cpu_seeds_2025_31415

python experiments/evaluate_mali_topology_controls.py \
  --input-dir artifacts/validation/mali_physical_dag_v1/topology_controls_grouped_cpu_screen \
  --input-dir artifacts/validation/mali_physical_dag_v1/topology_controls_grouped_cpu_seeds_2025_31415
```

The three-seed grouped result does not support the coarsened true-DAG model:
its seed-mean log-R² is `0.7382`, below node-only `0.7666`; paired QASM-group
bootstrap gives only `2.9%` probability that true DAG improves log-R². True
edges remain better than reversed/shuffled edges, so orientation has some
signal, but not enough to offset message-passing loss at this representation.

The strict transfer screen explicitly removes every real-QPU QASM hash from the
simulator pretraining data. It is deliberately a shared logical-feature MLP
screen, not a replacement for the original Graph Transformer. Run it with the
Ma--Li environment, then aggregate the three-seed OOF predictions by raw-QASM
group:

```bash
/path/to/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python \
  experiments/mali_strict_transfer_logical_screen.py \
  --mali-root /path/to/Quantum-Execution-Time-Prediction
python experiments/evaluate_mali_strict_transfer_screen.py \
  --mali-root /path/to/Quantum-Execution-Time-Prediction
```

For the duplicate/OOD follow-up, use the same physical-DAG corpus and split
manifest. Aggregation and inverse-noise weights use outer-training labels only;
each test row remains a recorded observed target. Support-stratified conformal
intervals are a sparse-calibration diagnostic rather than conditional-coverage
guarantees.

```bash
python experiments/mali_ood_and_duplicate_validation.py \
  --artifact-dir artifacts/validation/mali_physical_dag_v1 \
  --split-manifest artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv
```

Read the resulting `REPORT.md` files in `mali_strict_transfer_logical_v1/`,
`duplicate_aware_validation/`, and `ood_uncertainty_validation/` before using a
point estimate without an OOD flag.

## Ma–Li family-OOD and compiled transfer (Cluster 1)

These scripts stay on the Osaka/Kyoto `result.time_taken` labels. They do not
use Quantum Rings or cuTensorNet clocks. Current FakeOsaka/FakeKyoto
snapshots are not historical job-day calibration.

```bash
python experiments/mali_family_feature_space_audit.py --workers 12
python experiments/evaluate_mali_unseen_family_transfer.py
python experiments/evaluate_mali_simulator_assisted_family_transfer.py
python experiments/evaluate_mali_azizov_compiled_transfer.py
python experiments/evaluate_mali_azizov_transpiled_dag_transfer.py
```

Reports:

- `artifacts/validation/mali_family_ood_v1/`
- `artifacts/validation/mali_azizov_compiled_transfer_v1/`
- `artifacts/validation/mali_azizov_transpiled_dag_v1/`

Ignore `mali_azizov_transpiled_dag_v1/qpu_scratch_pilot/` (`softplus` decode).
Coarsened tensors under that tree are rebuildable and are not committed.

## Public calibration snapshot variability

The temporal audit compares the frozen local fake-provider pair with a public
timestamped calibration series from the DAQEC-Benchmark (42 timestamps per
backend over 14 days, three replicate rows per timestamp):

```bash
python experiments/mali_snapshot_variability.py \
  --download \
  --properties-dir /path/to/Quantum-Execution-Time-Prediction/data/fake_backend_properties \
  --output-dir artifacts/validation/mali_snapshot_variability
```

For offline reruns, download the DAQEC `drift_characterization.csv` once and
replace `--download` with `--source-csv /path/to/drift_characterization.csv`.
The resulting report separates spatial variation inside one frozen snapshot
from temporal variation across public calibration timestamps. The DAQEC series
is a 2025 aggregate calibration source, not the missing per-qubit historical
Ma–Li tensor.

## Matched dense-statevector simulator matrix

`simulator_runtime_v1` is a local actual-runtime label set, separate from the
Aer, CUDA-Q and cuTensorNet targets. It uses the same PyTorch dense-statevector
implementation for CPU/GPU and complex64/complex128, so device and precision
are not confounded with framework choice. Its target is prepared execution
with a fresh statevector and `<Z_0>` evaluation; construction, gate
materialization, transpilation and sampling remain separate or not applicable.

The completed 96-row baseline and its report are under
[`artifacts/simulator_runtime_v1/torch_dense_statevector_v1/`](../artifacts/simulator_runtime_v1/torch_dense_statevector_v1/).
Use the scoped [track README](simulator_runtime_v1/README.md) to regenerate the
matrix, OOF baseline evaluation and report.

## CUDA-Q matched simulator pilot

The CUDA-Q pilot is a separate target contract from the PyTorch dense matrix,
Aer, cuTensorNet and QPU tracks. It records CPU QPP FP64, NVIDIA GPU FP32 and
NVIDIA GPU FP64 `cudaq.sample` timings. First-call and warm-call labels are
separate because first use may include lazy compilation or backend
initialisation.

Use the Python 3.13 CUDA-Q environment (the CUDA-Q wheel is not installed in
the repository's default interpreter):

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/.venv-cudaq313/bin/python \
  experiments/cudaq_runtime/run_cudaq_matrix.py \
  --targets gpu_fp32,gpu_fp64 \
  --families ghz,hea,qaoa_cycle,random_brickwork \
  --widths 16,20,24,28 --shots 32 --warmups 1 --repeats 3 \
  --output artifacts/cudaq_runtime/<run>/gpu_matrix.csv

CUDA_VISIBLE_DEVICES="" /path/to/.venv-cudaq313/bin/python \
  experiments/cudaq_runtime/run_cudaq_matrix.py \
  --targets cpu_fp64 --widths 16,20,22 --shots 32 --warmups 1 --repeats 3 \
  --output artifacts/cudaq_runtime/<run>/cpu_matrix.csv
```

Merge the two CSVs while preserving `target_name`, then evaluate both target
contracts:

```bash
/path/to/.venv/bin/python experiments/cudaq_runtime/evaluate_cudaq_matrix.py \
  --input artifacts/cudaq_runtime/<run>/cudaq_matrix.csv \
  --output-dir artifacts/cudaq_runtime/<run>/evaluation
```

The current run and interpretation are in
[`artifacts/cudaq_runtime/cudaq_matrix_20260921/REPORT.md`](../artifacts/cudaq_runtime/cudaq_matrix_20260921/REPORT.md).
The run is intentionally a small estimator diagnostic; it does not claim
cross-GPU generalisation or a universal simulator runtime model.

### CUDA-Q MPS runtime/feasibility pilot

The MPS target has a different contract from CUDA-Q dense statevector. Run one
process per configured bond cap so that the pre-run `CUDAQ_MPS_MAX_BOND`,
absolute cutoff and SVD algorithm are frozen before target initialisation:

```bash
CUDAQ_PY=/path/to/.venv-cudaq313/bin/python
for cap in 2 4 8 16; do
  CUDA_VISIBLE_DEVICES=0 "$CUDAQ_PY" \
    experiments/cudaq_runtime/run_cudaq_mps_matrix.py \
    --families ghz,hea,qaoa_cycle,random_brickwork \
    --widths 16,20,24 --max-bond "$cap" --abs-cutoff 1e-12 \
    --precision fp64 --shots 32 --warmups 1 --repeats 2 \
    --reference-width-max 16 \
    --output "artifacts/cudaq_runtime/<run>/mps_cap${cap}.csv"
done
python experiments/cudaq_runtime/merge_cudaq_matrix.py \
  --inputs artifacts/cudaq_runtime/<run>/mps_cap{2,4,8,16}.csv \
  --output artifacts/cudaq_runtime/<run>/mps_matrix.csv
```

Evaluate only pre-run features with:

```bash
/path/to/.venv/bin/python experiments/cudaq_runtime/evaluate_cudaq_mps_matrix.py \
  --input artifacts/cudaq_runtime/<run>/mps_matrix.csv \
  --output-dir artifacts/cudaq_runtime/<run>/evaluation
```

`observed_max_bond`, tensor storage and dense-reference fidelity are retained
as post-run diagnostics/labels and are excluded from the static feature block.
The checked-in 48-row run is documented in
[`artifacts/cudaq_runtime/cudaq_mps_20260921/REPORT.md`](../artifacts/cudaq_runtime/cudaq_mps_20260921/REPORT.md).

## Quantum Rings / iQuHACK public-artifact audit

The committed result is the 18 September 2026 evaluation of Spirit Sprinters
and SoftLocked on the public 144-row table. Re-read
[`artifacts/quantum_rings/REPORT.md`](../artifacts/quantum_rings/REPORT.md)
before comparing scores. The contest labels do not include a simulation
machine profile.

```bash
python3 scripts/verify_artifacts.py
bash tracks/quantum_rings/clone_upstreams.sh work
```

Do not pool these clocks with Ma–Li, cuTensorNet, or dense-statevector rows.
