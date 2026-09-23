# Azizov-style compiled simulator pretraining → Ma–Li QPU transfer

**Finding recorded:** 2026-09-22.

This experiment tests the protocol requested after the Ma–Li / Azizov
comparison: transpile first, predict simulator runtime from compiled
features, then transfer that mapping onto real-QPU labels that also use
compiled inputs.

It is not Ma–Li's source-DAG Graph Transformer. It is not Azizov's
transpilation-aware GNN, and it does not re-measure Azizov's Aer `T_exec`
table. Simulator labels remain the public Washington/Sherbrooke
`time_taken` CSVs. Compile targets are current FakeWashingtonV2 and
FakeSherbrooke snapshots. QPU inputs are current FakeOsaka and FakeKyoto
compiled features from the family-OOD proxy table. The two clocks are
never pooled.

## Protocol

Stage 1 transpiles every Ma–Li QASM (`n=1510`) to FakeWashingtonV2 and
FakeSherbrooke (`optimization_level=1`, `seed_transpiler=1234`). Ridge
fits log1p simulator `time_taken` from:

- T0: logical width/depth/2q depth/2q count/gate count
- T1: physical depth/2q depth/2q count/gate count + logical width
- T2: T1 plus duration-weighted critical path and routing-depth ratio

A Sherbrooke dummy is included for in-domain simulator models only. It is
not applied on Osaka/Kyoto.

Stage 2 keeps the Stage-1 Ridge weights frozen, scores FakeOsaka/FakeKyoto
compiled features, then fits an affine head `T_qpu ≈ α + β · T_sim` on the
QPU training fold. Scratch compiled Ridge on the 340 Osaka/Kyoto
`result.time_taken` rows is the ceiling.

Splits:

- grouped QASM, simulator in-domain
- same-circuit `incl`: test QASM remains in simulator pretraining (Ma–Li analogue)
- same-circuit `excl`: that QASM is removed from simulator pretraining
- family hold-out: B2 analogue with device-matched simulator compilation

Compile coverage: 3020/3020 rows succeeded in 701 s. Unique QASM hashes are
1402 because some Ma–Li files share content, mainly `realamprandom` /
`twolocalrandom`. All 170 hardware QASMs are present on both Washington and
Sherbrooke.

Current fake snapshots are not historical job-day calibrations.
FakeWashingtonV2 `dt` is `2.22e-10` s; FakeOsaka/FakeKyoto `dt` is `5e-10` s.

## Simulator in-domain

Washington/Sherbrooke `time_taken` is a heavy-tailed simulator clock:
median 0.83 s, mean 1.31 s, p99 9.8 s, max 73.2 s
(`qwalk-noancilla_indep_qiskit_9` on Sherbrooke). QPU labels on the 340
rows are much tighter: median 8.64 s, max 13.70 s.

| Model | log-R² / MAE | MedAE |
|---|---|---|
| logical T0 | 0.2719 / 0.614 s (n=3020) | 0.262 s |
| compiled T1 | 0.3127 / 0.583 s (n=3020) | 0.219 s |
| compiled T2 | 0.4348 / 0.500 s (n=3020) | 0.167 s |

Compiled features do help on this clock. T2 vs T0 is +0.16 log-R². Rank
correlation with simulator labels is 0.67 for logical depth, 0.73 for
physical depth, and 0.76 for physical 2q count / weighted path. Linear R²
on raw seconds stays weak (0.12) because a few outliers dominate.

If labels above 10 s are ignored, T2 log-R² rises to 0.58. The gain is
therefore a bulk-circuit result, not a QWalk result. Per-backend the
fit is uneven: T2 log-R² is 0.03 on Washington and 0.56 on Sherbrooke.
Ansatz-like families (`random`, `realamprandom`, `su2random`, `qnn`) are
the only simulator families with log-R² above 0.5. GHZ, QAOA, AE and
QWalk remain near zero or negative.

This is the Azizov-style representation result on Ma–Li's simulator
clock, not a reproduction of Azizov's `T_exec` numbers.

## Same-circuit transfer onto 340 QPU rows

`incl` keeps the test QASM in simulator pretraining. `excl` removes it.
Scratch never uses simulator labels. An Osaka dummy is used only in QPU
scratch, not in the transferred mapping.

| Model | All families log-R² / MAE | Excluding QWalk |
|---|---|---|
| QPU scratch T0 | 0.7198 / 0.642 s (n=340) | 0.8728 / 0.586 s (n=338) |
| QPU scratch T1 | 0.8625 / 0.551 s (n=340) | 0.8996 / 0.516 s (n=338) |
| QPU scratch T2 | 0.8745 / 0.527 s (n=340) | 0.9082 / 0.493 s (n=338) |
| Sim→QPU T0 incl | -0.0127 / 1.468 s (n=340) | 0.0290 / 1.430 s (n=338) |
| Sim→QPU T1 incl | 0.1587 / 1.499 s (n=340) | 0.1508 / 1.475 s (n=338) |
| Sim→QPU T2 incl | 0.3533 / 1.290 s (n=340) | 0.3882 / 1.216 s (n=338) |
| Sim→QPU T1 excl | 0.1564 / 1.502 s (n=340) | 0.1478 / 1.479 s (n=338) |
| Sim→QPU T2 excl | 0.3493 / 1.286 s (n=340) | 0.3814 / 1.215 s (n=338) |

Scratch compiled QPU remains the ceiling. Transfer T2 beats transfer T0,
so compiled structure is the transferable piece, but it does not beat a
model trained directly on hardware labels. `incl` and `excl` differ by
about 0.004 log-R². Seeing the same QASM on Washington/Sherbrooke does
not help Osaka/Kyoto once compiled features are already on both sides.

Same-circuit T2 transfer by family:

| Family | n | Sim→QPU T2 incl | QPU scratch T1 |
|---|---:|---|---|
| qft | 46 | -1.959 / 1.474 s | 0.450 / 0.619 s |
| qftentangled | 47 | -1.904 / 1.505 s | 0.442 / 0.613 s |
| qnn | 26 | -10.082 / 2.654 s | 0.365 / 0.375 s |
| qpeexact | 35 | -1.696 / 1.286 s | 0.362 / 0.597 s |
| qpeinexact | 36 | -2.225 / 1.246 s | 0.350 / 0.508 s |
| qwalk_noancilla | 2 | -3711 / 13.874 s | -3003 / 6.577 s |
| random | 30 | 0.225 / 1.045 s | 0.907 / 0.310 s |
| realamprandom+twolocalrandom | 80 | 0.316 / 0.662 s | 0.679 / 0.467 s |
| su2random | 38 | 0.246 / 0.770 s | 0.581 / 0.562 s |

The pooled transfer number is carried by ansatz families. QFT/QPE/QNN
remain large-negative after affine calibration. Osaka and Kyoto transfer
T2 are almost identical (log-R² 0.35 / 0.35).

On the QPU side, physical 2q count Spearman is 0.85 and physical depth is
0.82; logical depth is only 0.24. The hardware clock is aligned with
compiled size. The simulator clock is only partially aligned, then
distorted by outliers. Affine calibration cannot repair that mismatch.

## Family-unseen QPU, family-seen simulator

This is the B2 question with device-matched Washington/Sherbrooke
compilation rather than the earlier FakeOsaka/FakeKyoto descriptor proxy.

| Model | All 9 components | Excluding QWalk |
|---|---|---|
| QPU scratch T1 | 0.8565 / 0.576 s (n=340) | 0.8920 / 0.541 s (n=338) |
| Sim T1 incl | -0.1799 / 1.766 s (n=340) | -0.1951 / 1.743 s (n=338) |
| Sim T2 incl | 0.1151 / 1.521 s (n=340) | 0.1434 / 1.449 s (n=338) |
| Sim T2 excl | 0.0509 / 1.557 s (n=340) | 0.0712 / 1.492 s (n=338) |

Keeping the family in simulator pretraining helps a little (T2 incl 0.12
vs excl 0.05) and beats the earlier mismatched-proxy B2 T2 incl of 0.22
only in the same-circuit setting, not here. It still does not approach
scratch T1. Per-component, only `realamprandom+twolocalrandom` and
`su2random` are weakly positive; QFT/QPE/QNN/random stay negative.

## Comparison with earlier runs

| Setting | log-R² / MAE |
|---|---|
| This run, QPU scratch compiled T1 | 0.862 / 0.551 s |
| This run, sim compiled T2 → affine QPU, same-circuit | 0.353 / 1.290 s |
| This run, sim compiled T2 → affine QPU, family-unseen | 0.115 / 1.521 s |
| B2 mismatched proxy, sim T2 incl, family-unseen | 0.219 / 1.434 s |
| Experiment C, sim logical T0 + affine QPU | -0.273 / 1.716 s |
| Ma–Li paper, Graph Transformer pretrained then fine-tuned | ~0.59 reported |
| Ma–Li paper, Graph Transformer trained on QPU only | ~0.89 reported |

Device-matched compiled pretraining is better than logical T0 transfer
and better than same-circuit T0. It is not better than training on the
340 hardware rows. The Ma–Li paper gap between pretrained 0.59 and
scratch 0.89 is the same qualitative result, now recovered with compiled
Ridge instead of a source-DAG GNN.

## What this can and cannot claim

- Compiled representation helps simulator `time_taken` relative to logical
  T0, and it is the only sim→QPU mapping that produces a positive pooled
  log-R².
- Frozen simulator Ridge plus an affine QPU head does not beat QPU-scratch
  compiled features. For a runtime estimator, the hardware labels already
  contain the useful mapping.
- Same-circuit leakage into simulator pretraining is irrelevant here:
  `incl` ≈ `excl`.
- This does not train Azizov's GNN, does not reproduce Azizov's Aer
  `T_exec`, and does not collect new QPU jobs.
- Current fake snapshots are not the historical Washington/Sherbrooke or
  Osaka/Kyoto calibrations used when those CSVs were written.
