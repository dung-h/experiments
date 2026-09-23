# Family-OOD protocol for Ma--Li

**Finding recorded:** 2026-09-22.

This folder splits questions that were previously collapsed into one
family-held-out number on the 340 Osaka/Kyoto rows. Experiment A and C
are unchanged. Historical B stays blocked. B2 uses existing labels.
Live-QPU B1 and external-join B3 were removed: they need credentials or
a trusted QASM-runtime join that this workspace does not have.

| Experiment | Question | Status |
|---|---|---|
| A | Do the 12 MQT families absent from the hardware table occupy a different feature region from the 10 selected families? | Logical audit complete; physical FakeOsaka/FakeKyoto proxy in `feature_space_audit/` |
| B-historical | Can a simulator-pretrained model predict Osaka/Kyoto `result.time_taken` for the 12 missing families? | Blocked: no labels; machines retired |
| B2 | If the family is seen on the simulator but unseen on QPU, does simulator pretraining help on the 10 hardware families? | Complete on the same 9 components as C |
| C | If a family is unseen in both simulator pretraining and QPU training, can compiled features still predict its Osaka/Kyoto runtime? | Complete on the 10 hardware families / 9 connected components |

Target semantics are not pooled. Hardware labels remain Ma--Li observed
`result.time_taken` at 1024 shots. Simulator labels remain FakeWashington /
FakeSherbrooke `time_taken`. Physical compiled features are
`CURRENT_FAKE_SNAPSHOT_PROXY` (Qiskit 1.4.1, IBM Runtime 0.36.1, FakeOsaka
1.0.16 / FakeKyoto 1.2.29, snapshot 2024-02-28), not recovered job-day
calibration.

Hold-out is by family connected component, not filename. `realamprandom`
and `twolocalrandom` share 22 QASM hashes and are one component.
`qwalk-noancilla` is two rows and is reported, not headlined.

Detailed tables:

- [feature_space_audit/REPORT.md](feature_space_audit/REPORT.md)
- [unseen_family_transfer/REPORT.md](unseen_family_transfer/REPORT.md)
- [simulator_assisted_transfer/REPORT.md](simulator_assisted_transfer/REPORT.md)

## Experiment A

Local MQT Bench extract: 1,510 independent-Qiskit QASMs, 22 families.
Hardware table uses 170 unique QASM hashes from 10 families. The other
12 families (549 circuits) have no Osaka/Kyoto labels here.

Logical structure already separates the two groups. Selected-family
circuits are dense and highly two-qubit; missing-family circuits are
sparse state-preparation / oracle / small variational instances.

| Feature | Median selected (n=961) | Median missing (n=549) | Missing / selected |
|---|---:|---:|---:|
| Logical depth | 253 | 64 | 0.25 |
| Two-qubit count | 4,606 | 81 | 0.018 |
| Interaction density | 1.00 | 0.034 | 0.034 |
| Mean interaction degree | 71 | 1.98 | 0.028 |

Standardized log1p centroid distance is 4.46. Leave-one-family nearest
selected/missing-centroid assignment is 20/22 (0.909). The two
exceptions are informative rather than noise:

- `ae` is labelled missing but sits nearer the selected centroid. It is
  structurally closer to QPE than to GHZ/DJ.
- `qwalk-noancilla` is labelled selected but sits nearer the missing
  centroid. That is the same family that later fails as a runtime tail.

The gap is not only a width mix. Inside the 91--127 qubit bucket the
two-qubit-count and interaction-density distributions do not overlap
(KS = 1.0): selected medians are 11,389 two-qubit gates and density 1.0,
missing medians are 114.5 and 0.018. At 2--16 qubits the same comparison
is milder (density 1.0 vs 0.5). Large-width missing families are GHZ,
W-state, GraphState and DJ; they remain sparse while selected families
remain dense.

Even inside the 10 selected families, the 170 hardware hashes are not a
uniform sample. Hardware-sampled QASMs are wider and deeper than the
unsampled members of those families (median width 91 vs 70, median gate
count 7,194 vs 3,871). Interaction density stays 1.0 in both, so the
bias is size, not topology.

The current FakeOsaka/FakeKyoto proxy (3,020 successful transpilations,
zero errors) widens the same split. Median physical depth is 17,040
versus 275; median native two-qubit count 27,130 versus 182; median
routing-depth ratio 62.2 versus 7.6. Standardized centroid distance
rises from 4.46 (logical) to 5.87 (compiled). Leave-one-family
nearest-group accuracy stays 20/22, with the same two exceptions: `ae`
and `qwalk-noancilla`. Recorded SWAP count is zero on both groups
because the FakeBackend native set decomposes routing into ECR/CX
rather than leaving `swap` in the circuit.

This is a sampling and representation finding. It is not a runtime score
for the 12 missing families.

## Experiment B

Historical B remains blocked. Scoring simulator-to-QPU transfer on GHZ,
W-state, GraphState, DJ, AE, VQE, QAOA, Grover and the remaining
walk/portfolio variants still requires observed Osaka/Kyoto labels.
Those labels are not here. The machines are retired. FakeOsaka/FakeKyoto
remain a compiled-structure proxy, not `result.time_taken`.

B2 is the unblocked sibling on existing data. It keeps Experiment C's
QPU splits and changes only whether the held family stays in
FakeWashington/Sherbrooke pretraining. Scratch T0–T3 match C to
`0.000` log-R². Seeing the family on the simulator does not produce a
hardware-unseen estimator:

- logical T0 `incl` is worse than `excl` (log-R² `-0.377` vs `-0.282`,
  excluding QWalk);
- compiled-proxy T2 is the best sim arm (`0.248` vs `0.225`), still far
  behind scratch T1 (`0.859` / `0.624 s`);
- that `0.25` is ansatz-driven (`realamp+twolocal`, `su2`); QFT/QPE/QNN
  stay large-negative.

Live IBM jobs for the 12 missing families, and any attempt to invent those
labels from QPack/Qonductor/IonQ, were removed from this capsule on
2026-09-22. They are not experiments that ran here.

Details:

- [simulator_assisted_transfer/REPORT.md](simulator_assisted_transfer/REPORT.md)

## Experiment C

For each of the 9 hardware components, the component is removed from
QPU training and from the 3,020-row simulator table, then scored on its
observed Osaka/Kyoto rows. Simulator pretraining uses logical T0 only;
Washington/Sherbrooke compilation is not treated as Osaka/Kyoto
structure. An affine map is then fit on the remaining QPU families.

Pooled out-of-family predictions, excluding the two QWalk rows:

| Model | log-R² | MAE |
|---|---:|---:|
| scratch T0, logical | -1.283 | 1.259 s |
| scratch T1, compiled depth/width/2q | 0.859 | 0.624 s |
| scratch T2, T1 + weighted path | 0.860 | 0.612 s |
| scratch T3, rich static summary | 0.893 | 0.546 s |
| sim T0 + affine QPU | -0.282 | 1.696 s |

Scratch T3 on all 9 components, including QWalk, is log-R² 0.8700 and
matches the earlier family-component rich-summary gate. The new
information is the ladder around that number:

1. Logical T0 does not transfer across families. QNN and `random` collapse
   (log-R² -8.05 and -88.7). Ansatz-like families still look learnable
   from logical counts, which is why a pooled logical score is easy to
   misread.
2. Compiled T1 restores family-OOD. The jump from T0 to T1 is the
   result; T2's extra duration path is small on this split.
3. T3 is the best pooled score but is not uniformly best:
   `random` prefers T1/T2 (log-R² 0.86 / 0.89) over T3 (0.08).
4. Simulator logical pretraining plus affine calibration does not beat
   QPU-scratch T0 on most families and is far behind compiled scratch.
   It only reduces error on the QWalk tail (MAE 5.00 s vs 8.28 s) and on
   `random` relative to logical scratch, not relative to compiled scratch.

The 12 missing families remain unscored. Experiment A says most of them
do not live in the selected-family region; Experiment C says that even
inside the selected region, logical features are not the transferable
object. Compiled physical structure is.

## Reproduction

```bash
.venv-cdaa-new/bin/python experiments/mali_family_feature_space_audit.py --workers 12
.venv-cdaa-new/bin/python experiments/evaluate_mali_unseen_family_transfer.py
.venv-cdaa-new/bin/python experiments/evaluate_mali_simulator_assisted_family_transfer.py
```

`--logical-only` skips FakeBackend transpilation. Physical rows resume
from `feature_space_audit/physical_proxy_features.csv`. Ma--Li QASM and
simulator labels are read from the local
`Quantum-Execution-Time-Prediction` checkout.
