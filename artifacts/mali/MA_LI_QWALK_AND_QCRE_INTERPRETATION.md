# Ma–Li QWalk failure and Ma–Li/QCRE proxy interpretation

## Purpose

This note preserves the interpretation to use in the next report. It keeps
the two experiments separate:

1. **Ma–Li transfer experiment:** a DAG Graph Transformer pretrained on fake
   simulator data and adapted to the 340 recorded Osaka/Kyoto labels.
2. **Ma–Li/QCRE proxy validation:** a separate, compact audit of logical and
   post-transpile features against the same 340 labels.

The second experiment is not the Ma–Li GNN and is not a claim that the current
FakeOsaka/FakeKyoto snapshot recovers the historical IBM calibration state.

## 1. QWalk fold-10 failure

The circuit is `qwalk-noancilla_indep_qiskit_9` from the public Ma–Li clone at
commit `32c392a6ece276f1ff046d4e30052d0571ff6dc6`. The exact QASM hash,
source rows, split and predictions are preserved in
[`fold10_qwalk_outlier.json`](fold10_qwalk_outlier.json).

### Circuit structure

| Property | Value |
|---|---:|
| Logical qubits / classical bits | 9 / 9 |
| Approximate operation count | 6,200 |
| Logical depth | 6,068 |
| Logical two-qubit depth | 5,916 |
| `cx` / `cu1` count | 3,030 / 2,910 |
| `u2` / `p` / `x` / `h` count | 86 / 78 / 16 / 27 |
| `ccx` count | 6 |
| SupermarQ critical-depth ratio | 0.99495 |
| SupermarQ entanglement ratio | 0.96074 |
| SupermarQ parallelism | 0.00251 |

The circuit is therefore a **depth and serial-dependency outlier**, not a
width outlier. It has repeated controlled-phase/CNOT walk structure and very
little parallelism. Its large graph is a plausible stress case for a model
whose simulator pretraining has learned a high-cost prior for deep circuits.

### Fold-10 numbers

| Quantity | Osaka test | Kyoto training copy |
|---|---:|---:|
| Recorded `result.time_taken` | 13.6962 s | 13.3764 s |
| Pretrained prediction | 36.0312 s | 13.3550 s |
| From-scratch prediction | 12.8724 s | 13.4610 s |

The pretrained Osaka error is `+22.335 s` (about `2.63×` the observed
runtime). It contributes `498.85/508.59 = 98.1%` of fold-10 squared error and
produces fold R² `-1.8991`. Excluding this fold, mean R² is `0.8690` for the
pretrained run and `0.8799` from scratch; the aggregate ten-fold values remain
`0.5922 ± 0.8317` and `0.8855 ± 0.0461`, respectively.

The Kyoto copy is in fine-tuning training while the Osaka copy is held out.
This is the paper-style cross-backend transfer setting, not the stricter
QASM-hash-grouped split used by the proxy audit.

## 2. How the Ma–Li metadata enters the model

The DAG encoder stores a 178-dimensional node vector:

| Node feature block | Dimensions |
|---|---:|
| Input/output indicators and 44 gate types | 46 |
| 127-qubit index one-hot | 127 |
| T1/T2 slots for the operation's qubit(s) | 4 |
| Gate/node order index | 1 |
| **Total** | **178** |

The graph edges encode circuit dependencies. Global features contain the 44
gate counts, qubit count, logical depth and five SupermarQ statistics
(`program_communication`, `critical_depth`, `entanglement_ratio`,
`parallelism`, `liveness`). The raw global vector is about 51-dimensional;
the local artifact removes all-zero columns and stores 40 dimensions.

Inference is:

```text
QASM → dependency DAG + node metadata
     → 3 TransformerConv layers + ReLU
     → global mean pooling

global circuit features → 64-wide MLP

graph embedding + global embedding
     → 512 → 512 → 128 → 1
     → runtime prediction in seconds
```

The pretrained checkpoint was learned from FakeWashington/FakeSherbrooke
simulator labels, then fine-tuned on the 340 Osaka/Kyoto labels. The target in
the public data is the upstream recorded `result.time_taken` for
`execute(shots=1024)`; this replication did not submit new QPU jobs.

## 3. Why the QWalk metadata causes the pretrained failure

The two QWalk graph instances have identical QASM, topology and global
features. In the local fake snapshots, the four T1/T2 node columns differ:

| Backend | Standardized T1/T2 node-column means |
|---|---|
| Osaka | `[0.802, 0.809, 1.069, 1.403]` |
| Kyoto | `[-0.112, -0.104, -0.696, -0.753]` |

The Osaka and Kyoto raw T1/T2 values also differ substantially on the first
few qubits. Since these values are attached to many operation nodes and then
propagated through graph message passing, a calibration shift can change the
graph embedding even when the logical circuit is identical.

The direct intervention is the strongest local diagnostic:

| Input/model | Kyoto | Osaka |
|---|---:|---:|
| Pretrained, full T1/T2 | 13.355 s | 36.031 s |
| Pretrained, T1/T2 replaced by standardized mean | 19.402 s | 19.402 s |
| From scratch, full T1/T2 | 13.461 s | 12.872 s |
| From scratch, T1/T2 replaced by mean | 13.213 s | 13.213 s |

This supports the following interpretation, without claiming causality for T1
or T2 themselves:

1. The simulator prior is badly scaled for this QWalk: simulator labels are
   approximately 42.43 s and 73.24 s, while the Osaka observed label is
   13.70 s.
2. The reconstructed fake calibration snapshots do not necessarily match the
   historical properties used when the real labels were collected.
3. Simulator and real-device feature populations were standardized separately,
   so the pretrained weights may see a different input scale during adaptation.
4. A 34-row held-out fold lets one extreme circuit dominate R².

The correct claim is therefore: **the current simulator checkpoint can impose
a harmful, calibration-conditioned prior on an extreme real-device circuit**.
It is not evidence that T1/T2 are intrinsically reliable real-QPU runtime
predictors. Exact attribution would require the historical processed tensor
and the specific calibration snapshot used by the authors. Public candidate
snapshots exist, but none has yet been matched to that tensor or to the
collection timestamp of the 340 Ma–Li labels.

### Published snapshot candidates

There are more calibration sources than the single fake-provider pair used in
the local run:

| Source | What is public | Limitation for exact Ma–Li recovery |
|---|---|---|
| Qiskit fake provider | One frozen `FakeOsaka`/`FakeKyoto` property JSON pair used locally; both are dated 2024-02-28 | Simulated snapshot, not necessarily the calibration used for the real labels |
| [IBM Quantum API](https://quantum.cloud.ibm.com/docs/en/guides/qpu-information) | `backend.properties(datetime=...)` supports historical properties retrieval | Requires IBM access and a date query; the paper's queried dates are not recorded in the public Ma–Li package |
| [Qwork public experiment page](https://stevetipp.github.io/Qwork.github.io/experiment19.html) | References a Kyoto calibration CSV dated 2024-03-07 | Reference in an experiment page, not a verified complete archive of the Ma–Li tensor |
| [Provenova hardware corpus](https://provenova.net/hardware) | Lists IBM raw calibration captures for Osaka and Kyoto dated 2024-06-01 | Candidate capture; provenance and per-qubit alignment to Ma–Li still need verification |
| [DAQEC-Benchmark Zenodo dataset](https://zenodo.org/records/17881116) | 14 days, 42 day-backend clusters, including Osaka and Kyoto, with calibration summaries | Primarily aggregate `cal_t1_mean`/`cal_t2_mean`; not the full 127-qubit node tensor needed by Ma–Li |

The defensible wording is therefore **“the exact historical snapshot used by
Ma–Li is not identified or bundled with the authors' public package”**, not
“no Osaka/Kyoto snapshots are publicly available.”

## 4. Ma–Li/QCRE proxy validation

The proxy audit uses the 340 existing Osaka/Kyoto labels but a different
inference path. It transpiles each QASM with Qiskit 1.4.1 `FakeOsaka` or
`FakeKyoto`, optimization level 1 and `seed_transpiler=1234`, then computes:

- logical depth and logical two-qubit depth;
- physical depth and physical two-qubit depth;
- physical gate counts and routing/SWAP diagnostics;
- a backend-target weighted critical path from native gate durations.

ALAP scheduling was not used because the current Qiskit environment fails on
some large QASM inputs; the script uses an explicit topological critical-path
calculation instead. Duration coverage was complete: no unsupported or
zero-duration rows.

The audit splits by logical-QASM SHA-256: 340 rows become 170 logical groups,
and 134 hashes occurring on both devices stay in one fold. It fits calibration
only on training groups using a log-linear model:

```text
log(1 + observed_time) = α + β log(1 + feature)
predicted_time = exp(predicted_log_time) - 1
```

For multivariate ablations, the same transformation is applied to physical
depth, physical two-qubit depth and weighted path. This is a calibrated proxy,
not the Ma–Li Graph Transformer.

### Grouped results

| Feature | MAE (s) | RMSE (s) | R² |
|---|---:|---:|---:|
| Logical depth | 1.4867 | 1.9410 | -0.0346 |
| Logical two-qubit depth | 1.4105 | 1.8353 | 0.0750 |
| Physical depth | 0.6763 | 0.8788 | 0.7879 |
| Physical two-qubit depth | 0.7062 | 0.9090 | 0.7731 |
| Weighted critical path | 0.8276 | 1.0976 | 0.6692 |

Incremental ablation under the same groups:

| Feature set | R² |
|---|---:|
| Physical depth | 0.7879 |
| + physical two-qubit depth | 0.7906 |
| + weighted path | 0.7910 |
| + both additions | 0.7946 |

Multiplying the path by `shots=1024` and fitting an out-of-sample affine map
gives R² `0.7536`. The raw weighted path median is `0.00510 s`, versus an
observed-label median of `8.637 s`, so it is a gate-critical-path quantity,
not a complete job runtime. Its scale gap reflects shots and execution/runtime
overhead not represented by the native-gate critical path.

The two leave-one-backend-out directions for the weighted proxy are:

- Osaka → Kyoto: R² `0.6705`;
- Kyoto → Osaka: R² `0.6798`.

Transpiler seed sensitivity is modest under the same grouped folds:

- physical depth mean R² `0.7847`, range `0.7764–0.7897`;
- physical two-qubit depth mean R² `0.7720`;
- weighted path mean R² `0.6628`, range `0.6437–0.6756`.

## 5. Reporting interpretation

Keep these conclusions separate:

- **Ma–Li:** DAG topology plus metadata can predict runtime, but simulator
  pretraining can hurt under calibration/domain shift. The QWalk failure is a
  concrete, retrievable example.
- **QCRE proxy:** compiled physical structure is substantially more predictive
  than logical depth on these labels. Native duration adds signal, but only as
  a calibrated secondary feature.
- **Overall:** neither result supports a universal runtime estimator. The
  next estimator should retain source/backend semantics and report grouped,
  backend-held-out and family-held-out performance.

## Retrieval checklist for the next report

1. Start with this file for the QWalk explanation.
2. Use [`fold10_qwalk_outlier.json`](fold10_qwalk_outlier.json) for exact
   circuit, hash, split and prediction values.
3. Use [`MA_LI_TRANSFER_BASELINE_COMPARISON.md`](MA_LI_TRANSFER_BASELINE_COMPARISON.md)
   for the matched pretrained/from-scratch comparison.
4. Use
   [`../validation/mali_qcre_final/MALI_QCRE_FINAL_VALIDATION_REPORT.md`](../validation/mali_qcre_final/MALI_QCRE_FINAL_VALIDATION_REPORT.md)
   for grouped proxy metrics, ablation and duration audit.
5. Do not describe the fake-snapshot adaptation as recovery of the paper's
   historical `total_ibm_standardization.npy` or IBM calibration snapshot.
