# Azizov-style compiled simulator pretraining → Ma–Li QPU transfer

**Finding recorded:** 2026-09-22.

Stage 1 predicts FakeWashington/FakeSherbrooke `time_taken` from current
FakeWashingtonV2/FakeSherbrooke compiled features. Stage 2 keeps that
Ridge mapping, then fits an affine head on Osaka/Kyoto compiled features
against the 340 hardware `result.time_taken` labels. The two clocks stay
separate.

## Coverage

- Simulator compiled rows used: 1060
- Unique simulator QASMs: 488
- QPU rows: 340
- Priority QPU QASMs present on both W/S: 170

## Simulator in-domain (grouped QASM, Washington/Sherbrooke labels)

| Model | log-R² / MAE |
|---|---|
| logical T0 | 0.0410 / 0.953 s (n=1060) |
| compiled T1 | 0.0360 / 0.917 s (n=1060) |
| compiled T2 | 0.0392 / 0.913 s (n=1060) |

## Same-circuit transfer onto 340 QPU rows

`incl` keeps the test QASM in simulator pretraining (Ma–Li analogue).
`excl` removes it. Scratch never uses simulator labels.

| Model | All families log-R² / MAE | Excluding QWalk |
|---|---|---|
| QPU scratch T0 | 0.7203 / 0.638 s (n=340) | 0.8736 / 0.581 s (n=338) |
| QPU scratch T1 | 0.8630 / 0.550 s (n=340) | 0.9001 / 0.514 s (n=338) |
| QPU scratch T2 | 0.8752 / 0.524 s (n=340) | 0.9089 / 0.490 s (n=338) |
| Sim→QPU T0 incl | 0.0389 / 1.509 s (n=340) | 0.0501 / 1.455 s (n=338) |
| Sim→QPU T1 incl | 0.2830 / 1.408 s (n=340) | 0.3026 / 1.350 s (n=338) |
| Sim→QPU T2 incl | 0.2824 / 1.279 s (n=340) | 0.3130 / 1.208 s (n=338) |
| Sim→QPU T1 excl | 0.2624 / 1.430 s (n=340) | 0.2597 / 1.401 s (n=338) |
| Sim→QPU T2 excl | 0.2551 / 1.295 s (n=340) | 0.2830 / 1.226 s (n=338) |

## Family-unseen QPU, family-seen simulator

| Model | All 9 components | Excluding QWalk |
|---|---|---|
| QPU scratch T1 | 0.8572 / 0.573 s (n=340) | 0.8928 / 0.538 s (n=338) |
| Sim T1 incl | -0.0294 / 1.644 s (n=340) | -0.0135 / 1.583 s (n=338) |
| Sim T2 incl | 0.0939 / 1.460 s (n=340) | 0.1176 / 1.392 s (n=338) |
| Sim T2 excl | -0.1115 / 1.515 s (n=340) | -0.1152 / 1.476 s (n=338) |

## What this can and cannot claim

- A win for `incl` over scratch would mean compiled simulator pretraining helps QPU estimation.
- A win for compiled T1/T2 over T0 on simulator labels is the Azizov-style representation result, on Ma–Li's simulator clock, not Aer `T_exec` re-measured here.
- Current FakeWashingtonV2/FakeSherbrooke/FakeOsaka/FakeKyoto snapshots are not historical job-day calibrations.
- This does not train the paper GNN, and it does not collect new QPU jobs.
