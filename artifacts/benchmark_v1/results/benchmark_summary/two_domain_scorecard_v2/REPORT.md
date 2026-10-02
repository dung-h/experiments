# Two-domain benchmark reader tables v2

Built from pinned archived real-QPU aggregation and local simulator artifacts.
This is a reader surface, not a flat leaderboard: QPU source labels, scheduled
outputs, and local simulator clocks remain separate.

## Real-QPU domain

The canonical population is **8,767 archived circuit observations**. The C4
attempt ledger has 24 method variants × 8,767 assigned rows (210,408 attempts).
Read `qpu/archived_method_coverage.csv` for assigned/predicted/unavailable
counts, `qpu/archived_method_source_metrics.csv` for source-local errors and
the 46-workflow QPack diagnostic, and `qpu/archived_method_pairwise_comparisons.csv`
for paired source-local contrasts. There is no cross-source pooled QPU score.
The three unified learned-adaptation summaries and their compatible-row pairs
are separate in `qpu/unified_learned_*.csv`.

## Local-simulator domain

The fixed panel has 204 members / 191 exact QASM hashes (162 core, 42 frontier,
q=2–16). Engine, precision, first/warm clock, approximation quality and
failure states stay separate in the five copied simulator tables. The C3a
local Aer graph adaptation is reported separately from the Azizov GNN: on 162
core OOF rows / 150 hashes its raw-derived MAE is 0.387654 s,
MedAE 0.036025 s, log1p-MAE 0.091407,
R² 0.155835; q10–16 are excluded. It is an unregistered
supplementary architecture adaptation, not original Ma–Li or Azizov.

Maestro has **no score in this pack**. The previous v3 preparation failure is
preserved; the separately governed partial run was not started at this build.
This is not a measured common-panel prediction failure.

## Scope limits

No cross-domain or cross-clock ranking is emitted. `evaluation_target_clock`
and `method_output_clock` remain distinct in the source tables. QPack replay
circuits are reconstruction-qualified; they do not change the archived labels.
The package is a local review artifact, not a public-release approval. Root
license/citation and external redistribution-rights decisions remain open.
