> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Native input contracts

QPU: in-memory saved logical QASM bytes và raw snapshot properties bytes.
Parse QASM → original 51 globals; Graph thêm nominal T1/T2 decoding, pruning,
compact DAG và 178-wide nodes → frozen outer-fold transforms → H2D → 3 seeds
→ host median. MLP không cần DAG/backend properties: không bắt trả Graph cost.
Không transpile: compiled circuits không là native input của frozen models.
Qonductor input vẫn qualified recipe, không exact historical logical recovery.

Dense: in-memory source QASM bytes và fixed shots=32 → QASM parse → exact
RCCX lowering bằng original `_lower_circuit` → seven globals; Graph thêm DAG
→ original train-only transforms/tensor encoding → H2D → 3 seeds → expm1,
zero clipping, host median. Qiskit 0.44.2 không có qasm2.dumps; pipeline không
gọi serialization-only phần normalize wrapper (unused by model). So exact
consumed globals/DAG against original frozen features; raw source hash pinned.
Không compiler, CUDA-Q, simulator hoặc fresh transpilation được gọi.

Median: frozen fold scalar is sufficient native input; circuit parsing không
cần thiết. Native CPU dispatch đo riêng, không CUDA synchronization wrapper.

Offline/excluded B2: source retrieval/disk I/O, model selection/fold routing,
training/calibration, imports/checkpoint load, metadata acquisition. Setup
separate process/bundle observation; không standalone model cold-start claim.
DAG/feature caches and prebuilt device tensors never enter timed raw pipeline.
Maestro/native analytical B2 and joint-MPS remain unmeasured in this bounded
comparison; saved coefficients alone are not a full raw-pipeline qualification.
