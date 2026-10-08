# Data lineage and feature dictionary

The real-QPU dataset contains **4,515 observations**, in 165 conservative groups:
Ma–Li 340 source logical inputs, Qonductor 230 archive-supported recipe
adaptations and QPack 3,945 structural reconstructions. The
[dataset report](data_preprocessing.md) defines selection, row identity,
label units and reconstruction limits. No reconstruction creates a runtime label.

All methods share fresh outer test assignments. Backend and shots remain
context fields; a circuit hash is not an observation ID. No group crosses
outer or inner partitions. Six QPack structural groups contain many repeated
workflow/context observations.

## Global input: 51 ordered positions

`global_00`–`global_43` are exact OpenQASM operation-name counts, in this
order. Missing names count as zero; names are not pooled into broad gate
families. In particular, literal `swap` means the `swap` operation, not a
proxy for all native two-qubit operations.

```text
00 u3       01 u2        02 u1        03 cx        04 id
05 u0       06 u         07 p         08 x         09 y
10 z        11 h         12 s         13 sdg       14 t
15 tdg      16 rx        17 ry        18 rz        19 sx
20 sxdg     21 cz        22 cy        23 swap      24 ch
25 ccx      26 cswap     27 crx       28 cry       29 crz
30 cu1      31 cp        32 cu3       33 csx       34 cu
35 rxx      36 rzz       37 rccx      38 rc3x      39 c3x
40 c3sqrtx  41 c4x       42 xx_plus_yy 43 ecr
```

| Position | Feature | Definition / caveat |
| ---: | --- | --- |
| 44 | `num_qubits` | Allocated quantum register width from Qiskit's `num_qubits`; not active width and not total quantum-plus-classical width |
| 45 | `depth` | Pinned upstream `qc.depth()` semantics before the graph-specific barrier/final-measurement removal |
| 46 | `program_communication` | SupermarQ-compatible connectivity score `C / (n × (n − 1))` |
| 47 | `critical_depth` | `0` when there are no nonlocal gates; otherwise multi-qubit depth divided by nonlocal gate count |
| 48 | `entanglement_ratio` | Nonlocal gate count divided by non-barrier/non-measure gate count |
| 49 | `parallelism` | `(G / D − 1) / (n − 1)` under the pinned upstream definitions |
| 50 | `liveness` | Sum of operation arities divided by `(D × n)` under the pinned upstream definitions |

For positions 46–50, `n` is allocated width; `G` excludes barriers and
measurements; `D` is depth excluding barriers and measurements; `M` is the
pinned nonlocal-gate count; `D_multi` is multi-qubit depth; `A` is the sum of
quantum argument counts; and `C` sums distinct neighbors separately for each
wire. Each undirected two-qubit neighbor pair contributes two to this sum.
Undefined or out-of-range values are unavailable, not
clipped or filled with invented epsilons.

The extractor uses pinned Ma–Li helper functions with a documented public-API
compatibility port for loose OpenQASM 3 wire indices. The raw 51-position
schema is retained. The upstream positive-column-sum mask is computed on the
gradient-fit population only. Its retained positions and statistics must be
recorded separately for each fresh fold. This completed run retains 40 globals
per fold, with 167 variable node positions among the 178 supplied positions.
There are no missing raw global features. Means and
sample standard deviations also use gradient-fit inputs; constant fields are
zeroed. No validation/test inputs or held-out labels enter these transforms.

## Directed graph input: 178 positions per node

| Node-feature positions | Width | Meaning |
| --- | ---: | --- |
| 0–45 | 46 | `in`/`out` node markers followed by 44 operation-name one-hot positions in the exact global gate order above |
| 46–172 | 127 | Compact first-encounter used-wire index encoding; these are not raw physical-qubit IDs |
| 173–176 | 4 | `T1(first)`, `T2(first)`, `T1(second)`, `T2(second)` for the first two operands |
| 177 | 1 | Node enumeration index after pruning |

The DAG uses directed upstream dependency edges. Multiple wire dependencies
between the same node pair collapse to a single directed edge. Barriers,
final measurements and idle wire in/out pairs are removed; mid-circuit
measurement, dynamic control flow, unsupported classical operations and
unknown gates are rejected rather than flattened into a false static graph.
The 127-wire encoding uses first-encounter compact remapping, not physical
wire number. The upstream encoder's padding quirks are preserved: one-qubit
operations and named higher-arity gates do not receive extra wire/T1/T2
slots, while two-qubit operations use both operands.

T1/T2 values are taken from the nominal snapshot mapped to logical wire
index and converted to microseconds; they are calibration context, not synthetic measurements and not
proof of historical job-day calibration. The graph input does **not** encode
numerical gate-error or gate-duration values. A configuration flag for gate
error is not equivalent to the model consuming those errors.

All 178 node positions are passed to the graph model. Constant positions are
zeroed using the gradient-fit inputs only, with each fresh fold's mask recorded.
The matched MLP receives the same transformed global schema, not the graph.
Neither model appends shots, source ID or backend one-hot features. Shots are
part of the dataset and analytical/polynomial method inputs where declared;
they are not silently added to the Ma–Li-style 51-feature input.

## Qonductor-style compiled features

The ordered regression input is `swap`, `depth`, `num_qubits`, `shots`,
`circuit_count`. Here `swap` preserves the author's CX/CZ/ECR counter; it is
not the literal operation count at global position 23. Depth is operand-stack
depth and width is touched/active width in the submitted or nominal compiled
view. Shots are archived context and circuit count is one. No source ID,
backend one-hot or source-specific missing mask is appended. A valid missing
operation name counts as zero; unavailable feature support is never zero-filled.

## Signatures and split diagnostics

The support audit hashes the exact transformed model tensors: float32 global
vector; float32 graph node matrix plus int64 directed ordered edge index; and
their combined identity. Schema, dtype, shape and little-endian array bytes
are framed into the digest. Signed zero is canonicalized; nonfinite values
fail. IDs, filenames, targets and runtime are not included. Therefore an
exact signature match means identical declared tensor inputs under that
fold's saved transform, not identical original circuits or equal quantum
behavior. An unseen signature is not by itself a family/backend/topology
out-of-distribution claim.

Each neural transform is fit on inner folds 1–3 within outer-train. Inner
fold 0 selects the epoch but supplies neither transform statistics nor gradient
updates; there is no subsequent full outer-train neural refit. Support diagnostics must be recomputed using the new fold transforms. No
exact-match or unseen-structure result is presumed before that check completes.

Rebuildable machine-readable definitions and evidence:

- [Frozen reporting protocol](../protocol/reporting.json)
- Panel inputs and split manifest (not bundled)
- Saved per-fold feature masks (aggregate counts in [dataset profile](../results/real_qpu/dataset_profile.json); per-row tensors not bundled)
- Per-row exact-input support (not bundled)
- Hash-pinned replay manifest (not bundled)
