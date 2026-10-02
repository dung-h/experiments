# Two-domain benchmark reader tables v3

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

Maestro has **no score in this pack**. The S9 synthetic calibration pilot is
terminal `pilot_gate_failed`: 10 completed cells, 240 API calls (150 timed),
9 SV stability failures and 1 MPS stability pass. Of 432 assigned calibration
cells, 33 are quarantined and 389 eligible cells remain unstarted. The
204-member / 408-candidate-attempt panel was not executed. See
`simulator/maestro_pilot_summary.csv` for independently recomputed medians/MADs.
These process-isolated reported timings are not persistent-process warm
timings. No predictor score or physical launch-overhead coefficient is inferred.
The copied simulator availability/appendix tables are historical snapshots;
the manifest and pilot summary carry the latest Maestro state.

## Scope limits

No cross-domain or cross-clock ranking is emitted. `evaluation_target_clock`
and `method_output_clock` remain distinct in the source tables. QPack replay
circuits are reconstruction-qualified; they do not change the archived labels.
The package is a local review artifact, not a public-release approval. Root
license/citation and external redistribution-rights decisions remain open.


## Required reconstruction and input disclosures

The evaluated learned graph uses a logical/physical mixture: Ma–Li and QPack
logical representations, Qonductor exact submitted physical QASM. It did not
execute S67's promised target-compiled Ma–Li/QPack inputs. Graph has seven
global features versus polynomial's five; performance cannot isolate graph
structure as its cause. QPack analytical replay uses six reconstructed
structures with representative rz(0.3)/rx(0.2), not recovered submitted circuits
or optimizer angles. Workflow holdout is not unseen-template holdout. Snapshot
calibrations are nominal backend-matched, not row-day historical calibration.
24 evaluated QPU variants are not 24 independent published approaches.

`method_scope.csv` covers the original approach list, including unevaluated
routes. `qpu/source_native_scope.csv` gives scoped historical pointers without
promoting new source-native accuracy. Source-native 340/4,482 remain separate,
not unified 8,767-row evaluations. `scientific_coverage=PARTIAL`; build validation
PASS means the evidence was assembled correctly, not full matrix completion,
clean-environment reproduction, slide approval or public-release permission.
