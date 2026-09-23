# Simulator-assisted transfer onto hardware-unseen families

**Finding recorded:** 2026-09-22.

This is Experiment B2. It uses the 340 Osaka/Kyoto `result.time_taken`
rows and the same 9 family connected components as Experiment C.
The target family stays out of QPU training. The only protocol change
is whether that family remains in FakeWashington/Sherbrooke
pretraining.

Historical Experiment B, observed runtime for the 12 MQT families
missing from the hardware table, remains blocked. This table does not
score GHZ, Grover, QAOA or the other missing families on a QPU.

`excl` matches Experiment C: the held family is removed from simulator
pretraining. `incl` is B2: the held family stays in simulator
pretraining. Scratch T0–T3 never use simulator labels.

Compiled-proxy simulator models predict Washington/Sherbrooke
`time_taken` from CURRENT_FAKE_SNAPSHOT_PROXY FakeOsaka/FakeKyoto
descriptors averaged over the two backends. That X is a circuit
descriptor, not a matched-device compilation of the simulator runs.

## Pooled out-of-family predictions

| Model | Target family in sim? | All 9 components log-R² / MAE | Excluding QWalk log-R² / MAE |
|---|---|---:|---:|
| `scratch_T0` | unused | -1.3082 / 1.300 s | -1.2825 / 1.259 s |
| `scratch_T1` | unused | 0.8125 / 0.662 s | 0.8589 / 0.624 s |
| `scratch_T2` | unused | 0.8153 / 0.650 s | 0.8603 / 0.612 s |
| `scratch_T3` | unused | 0.8700 / 0.575 s | 0.8928 / 0.546 s |
| `sim_T0_excl_affine_qpu` | no (Experiment C) | -0.2725 / 1.716 s | -0.2822 / 1.696 s |
| `sim_T0_incl_affine_qpu` | yes (B2) | -0.3821 / 1.790 s | -0.3775 / 1.763 s |
| `sim_T1proxy_excl_affine_qpu` | no | -0.1626 / 1.747 s | -0.1601 / 1.700 s |
| `sim_T1proxy_incl_affine_qpu` | yes (B2) | -0.1109 / 1.696 s | -0.1057 / 1.646 s |
| `sim_T2proxy_excl_affine_qpu` | no | 0.2019 / 1.445 s | 0.2248 / 1.381 s |
| `sim_T2proxy_incl_affine_qpu` | yes (B2) | 0.2188 / 1.434 s | 0.2483 / 1.363 s |
| `sim_T0T1proxy_excl_affine_qpu` | no | -0.2399 / 1.756 s | -0.2669 / 1.756 s |
| `sim_T0T1proxy_incl_affine_qpu` | yes (B2) | -0.1542 / 1.677 s | -0.1799 / 1.679 s |

## Does seeing the family on the simulator help?

Delta is B2 (`incl`) minus Experiment C (`excl`), excluding QWalk.
Positive log-R² delta, or negative MAE delta, means keeping the
family in simulator pretraining improved hardware-unseen transfer.

| Representation | excl log-R² | incl log-R² | Δ log-R² | excl MAE | incl MAE | Δ MAE |
|---|---:|---:|---:|---:|---:|---:|
| logical T0 | -0.2822 | -0.3775 | -0.0953 | 1.696 s | 1.763 s | 0.067 s |
| compiled proxy T1 | -0.1601 | -0.1057 | 0.0544 | 1.700 s | 1.646 s | -0.054 s |
| compiled proxy T2 | 0.2248 | 0.2483 | 0.0235 | 1.381 s | 1.363 s | -0.019 s |
| logical T0 + compiled proxy T1 | -0.2669 | -0.1799 | 0.0870 | 1.756 s | 1.679 s | -0.077 s |

## Per-component log-R² / MAE

| Component | n | scratch T1 | sim T0 incl | sim T2proxy excl | sim T2proxy incl |
|---|---:|---:|---:|---:|---:|
| `qft` | 46 | 0.2796 / 0.718 s | -1.1502 / 1.225 s | -2.1344 / 1.503 s | -2.0976 / 1.494 s |
| `qftentangled` | 47 | 0.3121 / 0.695 s | -0.7915 / 1.150 s | -2.6767 / 1.659 s | -2.6750 / 1.659 s |
| `qnn` | 26 | 0.0818 / 0.432 s | -26.1641 / 4.841 s | -13.6607 / 3.143 s | -13.0589 / 3.060 s |
| `qpeexact` | 35 | 0.3115 / 0.618 s | -4.7869 / 1.860 s | -2.1800 / 1.387 s | -2.0615 / 1.364 s |
| `qpeinexact` | 36 | 0.2742 / 0.529 s | -5.8839 / 1.815 s | -2.7124 / 1.306 s | -2.6469 / 1.292 s |
| `qwalk_noancilla` | 2 | -3771.7369 / 7.140 s | -2720.0597 / 6.347 s | -3075.2323 / 12.218 s | -3533.7886 / 13.419 s |
| `random` | 30 | 0.8611 / 0.398 s | -10.6028 / 3.530 s | -0.3341 / 1.537 s | -0.4365 / 1.600 s |
| `realamprandom+twolocalrandom` | 80 | 0.3971 / 0.693 s | -0.5634 / 1.034 s | 0.0919 / 0.815 s | 0.1815 / 0.766 s |
| `su2random` | 38 | 0.4575 / 0.681 s | -0.4889 / 1.069 s | 0.1215 / 0.819 s | 0.1492 / 0.809 s |

## What this can and cannot claim

- Scratch T0–T3 were recomputed on Experiment C's splits. The largest absolute log-R² difference versus the frozen C metrics file is 0.00000000.
- A positive `incl − excl` delta would mean simulator-seen family structure helps after affine calibration on other QPU families.
- Beating scratch T0 is not enough. The relevant ceiling on this split is compiled scratch T1/T3.
- Pooled compiled-proxy T2 log-R² of about 0.25 is family-heterogeneous. Ansatz components can be weakly positive; QFT/QPE/QNN remain large-negative. Do not read the pooled number as a universal simulator-to-QPU estimator.
- Compiled-proxy simulator models are not device-matched pretraining. Washington/Sherbrooke `time_taken` is the label; FakeOsaka/FakeKyoto structure is only a circuit descriptor.
- This still does not score the 12 missing families on real QPU. Historical Osaka/Kyoto labels for those families do not exist here; live collection was removed from the capsule.

