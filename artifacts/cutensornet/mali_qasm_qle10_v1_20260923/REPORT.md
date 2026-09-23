# Ma–Li MQT Bench QASM on the cuTensorNet contraction estimator

Finding recorded: 2026-09-23.

This is a local RTX 5070 Ti scalar-contraction experiment. It is not a QPU
runtime result, not Osaka/Kyoto `result.time_taken`, and not a replacement of
the synthetic 70% 25-row / 55-row grids or the 90% QFT width frontier.

## Purpose

The earlier cuTensorNet tables did **not** use the Ma–Li circuit pool. They
built synthetic tensor networks for five generated families:

```text
GHZ, HEA, QAOA-cycle, random brickwork, QFT
```

Those families are named after common algorithms, but the tensors were
constructed by the local runner, not loaded from MQT Bench OpenQASM.

The question here is whether `RUNTIME_EST` after `TIME_TUNED` still over-predicts
cheap kernels when the input is the public Ma–Li MQT Bench QASM pool
(`*_indep_qiskit_*.qasm`, 22 families, 1,510 files), rather than those
synthetic graphs.

The clock remains the same local GPU target as the other cuTensorNet capsules:

```text
T_sim_contract = warm CUDA-event median of a zero-bitstring amplitude
```

That is `contract_gpu_median_s`. It is not Ma–Li hardware `result.time_taken`.

## Why the full 1,510-circuit pool was not contracted

The Ma–Li pool is dense only to about 10 qubits, then jumps to widths 30–127.
Exact scalar contraction of the whole pool on one 16 GB GPU is not feasible.
The runner therefore selected a documented executable subset and recorded every
skip. Skipped files remain in the pool; they were not deleted.

Selection rule:

```text
all 22 families
2 ≤ n_qubits ≤ 10
approximate gate count ≤ 2,500
```

| Pool | Count |
| --- | ---: |
| Public Ma–Li MQT Bench QASM | 1,510 |
| Selected and timed | 176 |
| Skipped, width outside 2–10 | 1,331 |
| Skipped, gate count above 2,500 | 3 |
| Failed contractions | 0 |

The three width-eligible circuits deferred by the gate cap are:

| Circuit | Qubits | Approximate gates |
| --- | ---: | ---: |
| `grover-noancilla_indep_qiskit_8` | 8 | 3,144 |
| `qwalk-noancilla_indep_qiskit_8` | 8 | 3,123 |
| `qwalk-noancilla_indep_qiskit_9` | 9 | 6,189 |

Do not read 176 as “the 1,510-circuit Ma–Li dataset”. It is the subset that
this GPU and this exact-contraction target can currently finish.

## Setup

| Item | Value |
| --- | --- |
| Source pool | Ma–Li `data/quantum_circuits`, MQT Bench v1.0.0 `*_indep_qiskit_*.qasm` |
| Families present in the timed subset | all 22 |
| Precision | complex64 |
| Workspace | `memory_limit="90%"` |
| Optimizer | `TIME_TUNED`, seed 137, architecture 12 |
| Optimizer samples | 16 default; 8 if gates ≥ 800 |
| Timing | two warm-ups, five CUDA-event repeats |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16.61 GB, compute capability 12.0 |
| Software | cuquantum-python-cu13 26.6.0, Qiskit 2.5.2 |

Four circuits used eight optimizer samples:
`qwalk-noancilla_6`, `grover-v-chain_9`, `grover-noancilla_7`,
`qwalk-noancilla_7`.

## Headline

Across 176 successful rows, median warm actual / `RUNTIME_EST` is **0.231**
(mean 0.270, minimum 0.190, maximum 2.227). 127 rows sit between 0.20 and
0.30. That is the same cheap-kernel over-prediction seen on the 70% 25-row
synthetic grid (median 0.196). It is **not** the 90% QFT q36–q44
under-prediction of 4.19–5.11×.

```text
small Ma–Li QASM, q≤10:  actual ≈ 0.23 × RUNTIME_EST
large synthetic QFT, q36–q44: actual ≈ 4–5 × RUNTIME_EST
```

`RUNTIME_EST` is therefore still a plan-cost proxy. One median ratio cannot be
moved between these two regimes.

## Families

Every Ma–Li family appears in the timed subset. Counts differ because some
families have fewer files at q≤10, and because three heavy Grover/QWalk files
were deferred.

| Family | n | qubits | median actual/EST | min | max | median warm |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| qwalk-v-chain | 4 | 3–9 | 0.194 | 0.190 | 0.208 | 0.92 ms |
| qwalk-noancilla | 5 | 3–7 | 0.199 | 0.190 | 0.438 | 1.06 ms |
| qnn | 9 | 2–10 | 0.203 | 0.193 | 0.278 | 0.25 ms |
| portfolioqaoa | 8 | 3–10 | 0.205 | 0.195 | 0.258 | 0.19 ms |
| random | 9 | 2–10 | 0.206 | 0.194 | 0.283 | 0.20 ms |
| portfoliovqe | 8 | 3–10 | 0.207 | 0.196 | 0.253 | 0.19 ms |
| ae | 9 | 2–10 | 0.215 | 0.198 | 0.391 | 0.11 ms |
| realamprandom | 9 | 2–10 | 0.224 | 0.200 | 0.335 | 0.17 ms |
| qftentangled | 9 | 2–10 | 0.225 | 0.200 | 0.451 | 0.09 ms |
| su2random | 9 | 2–10 | 0.226 | 0.201 | 0.327 | 0.16 ms |
| qaoa | 8 | 3–10 | 0.229 | 0.205 | 0.272 | 0.10 ms |
| twolocalrandom | 9 | 2–10 | 0.229 | 0.201 | 0.323 | 0.17 ms |
| qpeexact | 9 | 2–10 | 0.232 | 0.201 | 0.488 | 0.09 ms |
| vqe | 8 | 3–10 | 0.234 | 0.208 | 0.274 | 0.10 ms |
| qpeinexact | 9 | 2–10 | 0.237 | 0.209 | 0.473 | 0.09 ms |
| wstate | 9 | 2–10 | 0.238 | 0.215 | 0.438 | 0.08 ms |
| grover-noancilla | 6 | 2–7 | 0.238 | 0.194 | 0.562 | 0.28 ms |
| qft | 9 | 2–10 | 0.252 | 0.209 | 0.678 | 0.08 ms |
| dj | 8 | 2–10 | 0.256 | 0.225 | 0.459 | 0.06 ms |
| graphstate | 7 | 3–9 | 0.282 | 0.234 | 0.373 | 0.06 ms |
| ghz | 9 | 2–10 | 0.322 | 0.248 | 0.653 | 0.05 ms |
| grover-v-chain | 6 | 2–9 | 0.430 | 0.200 | 2.227 | 0.59 ms |

Ma–Li `qft` in this table is q2–q10. Those rows behave like the cheap 70% QFT
q16–q24 kernels, not like the 90% QFT q36–q44 frontier.

Width itself moves the ratio toward the 0.20–0.23 band as soon as the kernel
is no longer tiny:

| Qubits | n | median actual/EST |
| ---: | ---: | ---: |
| 2 | 15 | 0.451 |
| 3 | 22 | 0.275 |
| 4 | 21 | 0.241 |
| 5 | 22 | 0.235 |
| 6 | 20 | 0.231 |
| 7 | 22 | 0.215 |
| 8 | 17 | 0.211 |
| 9 | 20 | 0.215 |
| 10 | 17 | 0.210 |

q=2 inflates the ratio because both the estimate and the warm time sit at tens
of microseconds. That is not a family effect.

## Two circuits that leave the cheap-kernel band

Almost every family still over-predicts. Two Ma–Li QASM files leave that
pattern without becoming the large-QFT regime.

### `grover-v-chain_indep_qiskit_9` — the only under-prediction

| Quantity | Value |
| --- | --- |
| Qubits / gates | 9 / 871 |
| Slices | 1 |
| Largest intermediate | 2.684e8 complex64 (~2 GiB) |
| FLOPs | 42.77 B |
| `RUNTIME_EST` | 23.497 ms |
| Warm actual | 52.319 ms |
| First contraction | 52.688 ms |
| actual / estimate | **2.227** |

This is the only row with actual/estimate > 1. First and warm times agree, so
the residual is not Python launch. The intermediate is much larger than the
typical q≤10 plan (often 8–16 k elements) and much smaller than 90% QFT-36
(~8 GiB). The sign of the error has flipped, but the magnitude is still closer
to QFT-32 (1.92×) than to QFT-36–44 (4–5×).

The rest of `grover-v-chain` is mixed, not uniformly reversed:

| Circuit | q | gates | intermediate | actual/EST |
| --- | ---: | ---: | ---: | ---: |
| grover-v-chain_2 | 2 | 2 | 2 | 0.618 |
| grover-v-chain_3 | 3 | 13 | 8 | 0.275 |
| grover-v-chain_4 | 4 | 44 | 32 | 0.205 |
| grover-v-chain_5 | 5 | 125 | 262,144 | 0.584 |
| grover-v-chain_7 | 7 | 310 | 512 | 0.200 |
| grover-v-chain_9 | 9 | 871 | 268,435,456 | 2.227 |

### `qwalk-noancilla_indep_qiskit_6` — the only sliced plan

| Quantity | Value |
| --- | --- |
| Qubits / gates | 6 / 831 |
| Slices | **32** |
| Largest intermediate | 4.194e6 complex64 (32 MiB) |
| `RUNTIME_EST` | 0.313 s |
| Warm actual | 0.137 s |
| actual / estimate | 0.438 |

This is the longest warm contraction in the 176-row table, and the only plan
that sliced. The estimator still over-predicts. Slicing therefore does not, by
itself, reproduce the 90% QFT under-prediction.

`qwalk-noancilla_7` is heavier in gates (1,593) but returned to one slice, a
16,384-element intermediate, warm 3.99 ms versus estimate 16.90 ms (0.236).
Its cost is path search: 108.7 s of optimizer time versus 4 ms of warm
contraction. End-to-end first-call latency is 108.8 s. That column is not an
NVIDIA estimate and must not be compared with `RUNTIME_EST`.

## Interpretation

1. The previous synthetic estimator did **not** already cover the Ma–Li MQT
   families. It covered five generated graphs. This run puts all 22 public
   Ma–Li families onto the same GPU contraction clock, at widths the machine
   can finish.
2. For this q≤10 executable subset, `RUNTIME_EST` remains conservative by
   about 4×. Family identity does not break that pattern except for
   `grover-v-chain_9`.
3. The 90% large-QFT under-prediction is a width/intermediate regime, not a
   workspace-policy label. The present table also used 90% and still
   over-predicted, because the Ma–Li files that survived the screen are small.
4. Path-search wall time can dominate end-to-end latency (`qwalk-noancilla_7`)
   even when the warm contraction is milliseconds. A pre-run estimator that
   quotes only `RUNTIME_EST` is silent about that cost.

For a runtime estimator the usable form is unchanged:

```text
RUNTIME_EST
+ workspace policy
+ plan statistics (slices, FLOPs, largest intermediate)
+ family/width regime
→ calibrate separately on cheap q≤10 QASM versus large unsliced/sliced QFT
→ predict warm contraction time
```

Do not apply the 176-row median 0.231 to the 90% QFT frontier, and do not
apply the QFT-44 factor 5.11 to these Ma–Li QASM rows.

## Reproduction

The QASM files are not redistributed. Clone the pinned Ma–Li repository, then
point the runner at `data/quantum_circuits`.

```bash
# from the capsule root, after scripts/bootstrap_upstreams.sh
# or an equivalent clone of mooselab/Quantum-Execution-Time-Prediction
# at commit 32c392a6ece276f1ff046d4e30052d0571ff6dc6
export CUTENSORNET_PYTHON=/path/to/python   # cuquantum-python-cu13==26.6.0, cupy-cuda13x, qiskit==2.5.2
tracks/cutensornet/code/with_cutensornet_env.sh   tracks/cutensornet/code/run_cutensornet_mali_qasm.py   --qasm-dir work/mali/data/quantum_circuits   --min-qubits 2 --max-qubits 10 --max-gates 2500   --memory-limit 90% --optimizer-samples 16 --heavy-optimizer-samples 8   --heavy-gate-threshold 800 --optimizer-seed 137   --optimizer-cost-function TIME_TUNED   --warmups 2 --repeats 5   --output-dir run-output/cutensornet/mali_qasm_qle10
```

The runner resumes from an existing CSV. A different GPU, CUDA, cuTensorNet
version, thread count or workspace policy is a new measurement, not a failed
copy of these seconds.

## Raw files

- `cutensornet_mali_qasm.csv` — 176 timed rows.
- `skipped.csv` — 1,334 documented skips.
- `selection.json` — pool size, selection rule, family list.
- `family_summary.csv` — per-family medians.
- `headline.json` — packaged checks.
- `run_summary.json` — 176/176 ok, 0 failed, 207.9 s selected-loop wall time
  after resume.
- `run.log` — per-circuit progress.

The contraction runner is `tracks/cutensornet/code/run_cutensornet_mali_qasm.py`.

## What this does not claim

- It does not contract all 1,510 Ma–Li circuits.
- It does not predict Osaka/Kyoto `result.time_taken`.
- It does not replace the 70% synthetic grids or the 90% QFT frontier.
- It does not make `RUNTIME_EST` a universal simulator runtime estimator.
- It does not transfer these seconds onto another GPU.
