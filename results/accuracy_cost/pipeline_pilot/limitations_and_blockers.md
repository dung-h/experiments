> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Limitations and blockers

- Strong classical comparator remains BLOCKED. Scoped scan: QPU regression
  40 files; dense classical11/precision; fixed-MPS28; family-MPS40;
  Maestro evaluation4; Azizov322. No candidate fitted serialization files.
  Producing code discards fitted estimators after predicting; settings/seeds,
  scaler summaries and predictions do not recover trees, SVs or coefficients.
  Existing median scalars/affine coefficients/Maestro knots/neural states
  were already recoverable: zero newly recovered competitive models.
- QPU Graph:38/45 raw reconstructions fail exact sparse-array equality. All
  45 globals/MLP predictions pass. Do not change node ordering/calibration
  or cache saved graph to relabel as full raw B2. Single-request diagnostic
  below documents discrepant fields; it does not diagnose all38 cases.
- Dense: FP32 three circuits and FP64 one circuit fail both model paths at
  RCCX lowering. Parser/environment definition includesu1/u2 where frozen
  normalizer requirescx/h/t/tdg. All30/context attempted; no cherry-picking
  27/29 passing subsets. Context-wide gate blocks both neural methods.
- QPack5 selected raw QASM unavailable at declared locations: no generator
  execution or invented QASM to bypass limitation. This does not change
  qualified dataset or reconstruction labels.
- Historical compilation/backend acquisition, exact historical recovery,
  standalone cold starts, batch amortization, cross-hardware scaling,
  planner cost, and strongest classical full-pipeline frontier remain open.
- Shared model-bundle RAM/VRAM peaks and stage-end allocation snapshots are
  not marginal per-model memory costs. Desktop background activity retained;
  hardware is not exclusive. No target simulation was run for break-even.
- Native Qiskit compiled-QASM timeout root cause is still unresolved. Logical
  QPU MLP parsing is a different input path and cannot clear that historical
  timeout. Maestro/MPS/native analytical/TN B2 NOT EXECUTED in Phase2.

## Single-request graph reconciliation

Per-observation private QPU example omitted from this publication view; original evidence is retained locally.