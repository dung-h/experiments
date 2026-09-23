# Unseen-family transfer on the 10 Ma--Li hardware families

**Finding recorded:** 2026-09-22.

This is Experiment C. A family connected component is removed from
both simulator pretraining and QPU training, then scored on that
component's observed Osaka/Kyoto `result.time_taken` rows. The 12 MQT
families without hardware labels are not in this table.

`realamprandom` and `twolocalrandom` share 22 QASM hashes, so they are
held out as one connected component. `qwalk-noancilla` has two rows
and is reported, not used as a headline R².

Simulator pretraining is logical T0 only. FakeWashington/Sherbrooke
compilation is a different device snapshot from FakeOsaka/FakeKyoto,
so compiled features are not transferred from the simulator table.

## Pooled out-of-family predictions

| Model | All 9 components log-R² / MAE | Excluding QWalk log-R² / MAE |
|---|---:|---:|
| `scratch_T0` | -1.3082 / 1.3002 s | -1.2825 / 1.2589 s |
| `scratch_T1` | 0.8125 / 0.6619 s | 0.8589 / 0.6236 s |
| `scratch_T2` | 0.8153 / 0.6503 s | 0.8603 / 0.6123 s |
| `scratch_T3` | 0.8700 / 0.5753 s | 0.8928 / 0.5458 s |
| `scratch_qcre` | 0.6207 / 0.9157 s | 0.6168 / 0.9011 s |
| `scratch_compiled` | 0.8020 / 0.7228 s | 0.8091 / 0.7034 s |
| `scratch_physical_graph` | 0.8351 / 0.6136 s | 0.8750 / 0.5771 s |
| `sim_T0_affine_qpu` | -0.2725 / 1.7155 s | -0.2822 / 1.6961 s |

## Per-component scores

| Component | n_test | scratch T0 | scratch T1 | scratch T3 | sim T0 + affine QPU |
|---|---:|---:|---:|---:|---:|
| `qft` | 46 | 0.4215 / 0.612 s | 0.2796 / 0.718 s | 0.5501 / 0.546 s | -1.1128 / 1.215 s |
| `qftentangled` | 47 | 0.5356 / 0.582 s | 0.3121 / 0.695 s | 0.5072 / 0.599 s | -0.7874 / 1.148 s |
| `qnn` | 26 | -8.0459 / 2.360 s | 0.0818 / 0.432 s | 0.4286 / 0.336 s | -26.5362 / 4.887 s |
| `qpeexact` | 35 | 0.4429 / 0.540 s | 0.3115 / 0.618 s | 0.3982 / 0.560 s | -4.7530 / 1.855 s |
| `qpeinexact` | 36 | 0.3992 / 0.519 s | 0.2742 / 0.529 s | 0.4734 / 0.427 s | -5.7948 / 1.802 s |
| `qwalk_noancilla` | 2 | -5875.3413 / 8.282 s | -3771.7369 / 7.140 s | -1915.8411 / 5.549 s | -1464.6734 / 4.996 s |
| `random` | 30 | -88.6676 / 7.454 s | 0.8611 / 0.398 s | 0.0804 / 1.103 s | -6.0342 / 2.791 s |
| `realamprandom+twolocalrandom` | 80 | 0.7554 / 0.401 s | 0.3971 / 0.693 s | 0.6907 / 0.459 s | -0.5402 / 1.028 s |
| `su2random` | 38 | 0.7118 / 0.405 s | 0.4575 / 0.681 s | 0.6572 / 0.466 s | -0.4886 / 1.069 s |

## Reading the comparison

- `scratch_T*` trains only on other hardware families. That is the
  fully unseen-family QPU baseline.
- `sim_T0_affine_qpu` first fits a logical estimator on simulator
  rows whose family names are not the held component, then fits an
  affine map on the remaining QPU families. If this does not beat
  `scratch_T0`, simulator pretraining is not helping family-OOD
  transfer under this representation.
- T1 compiled depth/width/2q, T2 adds the duration-weighted path,
  and T3 is the rich static summary. These blocks are alternative
  QPU-scratch representations, not nested on top of T0, and they
  are not simulator-to-QPU compiled transfer.

Experiment B, simulator-to-QPU transfer onto the 12 missing
families, remains blocked: those families have no observed QPU
labels, and the original Osaka/Kyoto machines are retired.
