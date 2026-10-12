> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Generalization decomposition

Authority: the completed 21-case frozen RQ4 grid, not the older10/12-case
snapshots. The primary neural estimator is the protocol-defined per-row median
of predictions from seeds42,1234,31415. Saved predictions alone were reduced
and scored here. CPU selectors were chosen on inner validation, never outer test.

`generalization_decomposition.csv` has 42 case/model rows; the 1,071-row
`generalization_field_support.csv` gives51 raw-field support diagnostics per
case. `generalization_predictive_skill.csv` compares saved CPU/neural estimates
with case-matched references. All units are seconds unless explicitly ratios.

## Family transfer

The nine cases have14–27 independent conservative test groups. There are no
exact consumed-input collisions in either neural model. Group embargo removes
known structural components; family holdout is not merely a new observation ID.

| Family | Rows / groups | Graph / MLP MAE | Graph / MLP skill vs train global median | Directional diagnostic |
| --- | ---: | ---: | ---: | --- |
| qft | 46 /26 | .563 /.513 | .774 /.794 | Graph near-zero mean signed error; MLP tends low |
| qftentangled | 54 /27 | .743 /.552 | .711 /.785 | Graph tends high |
| qnn | 26 /14 | 36.272 /50.764 | −12.646 /−18.099 | Both overpredict all 26 |
| qpeexact | 40 /23 | 2.202 /.604 | .281 /.803 | Graph underpredicts all 40 |
| qpeinexact | 36 /22 | 1.575 /.713 | .505 /.776 | Graph low in 34; MLP high in 33 |
| random | 30 /16 | 145.708 /188.815 | −39.168 /−51.051 | Both overpredict all 30 |
| realamprandom | 40 /22 | .636 /.952 | .432 /.151 | MLP underpredicts all 40 |
| su2random | 38 /21 | .589 /1.465 | .496 /−.254 | MLP underpredicts all 38 |
| twolocalrandom | 40 /22 | 1.194 /1.259 | .145 /.098 | Graph underpredicts all 40 |

Positive skill relative to a weak global median need not beat a stronger
classical predictor. The saved CPU selector has MAE .434s on QNN and 1.431s on
Random; it also beats both neural models on realamprandom, su2random and
twolocalrandom. The complete-grid paired intervals remain the uncertainty
authority; the point skill ratios here have no separately constructed CI.

### QNN and Random are systematic failures, not isolated bad rows

QNN actual runtime is3.107–7.376s. Graph MedAE/RMSE are35.841/36.587s;
MLP50.357/51.284s. R² is−2197.298/−4318.143. Maximum absolute errors are
44.797/63.458s. Random actual runtime is7.981–12.751s, MedAE148.522/198.602s,
RMSE146.956/190.551s, R²−13158.490/−22124.320 and max180.805/230.097s.
All-row positive residuals and large MedAE rule out a single outlier explanation.
Graph wins their paired relative comparison but neither is practically useful.

QNN retains 40 globals and loses no test-only global field. Its maximum
absolute train-standardized feature value is596.84, with median per-row maximum
476.78. Random retains 22 and drops18 fields active only at test; maximum
train-standardized value is1452.36, median1183.99. These are **per-feature
standardized range diagnostics**, not nearest-neighbor distances or causal
attributions. Constant train fields do not receive invented finite z-scores;
out-of-range support is recorded separately. Every QNN/Random test row is
outside at least one raw train range. Random also loses18 node fields in the
saved node-mask audit. No controlled saved experiment establishes that masking
alone caused either failure.

Holdout/parent-same-row MAE ratios are included as descriptive degradation:
Graph70.94× for QNN and 263.06× for Random; MLP138.41× and 460.85×. They are not
causal holdout penalties: train populations differ, and parent retained-fit
versus RQ4 full-refit procedures differ too.

## Backend-context transfer

All ten cases have disjoint train/test observation IDs. All Graph complete
consumed-input collision counts are zero. MLP omits backend, nominal T1/T2 and
shots, so equal logical-feature vectors can recur across backend observations.
That is a representation collision, not observation-ID leakage.

| Backend | Test rows / groups | Exact MLP fit collisions | Interpretation |
| --- | ---: | ---: | --- |
| Belem | 24 /3 | 24 | Known features; small support |
| Jakarta | 408 /8 | 403 | Mostly repeated features; source mix |
| Kyoto | 148 /117 | 148 | Known logical structures, new backend context; Ma–Li only |
| Lagos | 368 /5 | 368 | Known features; QPack-dominated |
| Lima | 41 /4 | 36 | Mostly repeated features |
| Manila | 367 /8 | 360 | Mostly repeated features; source mix |
| Nairobi | 1565 /9 | 1062 | Repeated and new support; QPack-dominated |
| Osaka | 192 /148 | 149 | 149 known-group rows and 43 new-group rows; Ma–Li only |
| Perth | 351 /9 | 336 | Mostly repeated features; source mix |
| Quito | 1051 /7 | 1051 | 1009 QPack +42 Qonductor observations; known features |

See the full-grid `reader_tables/input_support.csv`, `source_metrics.csv` and
`method_metrics.csv` for known/new-group and source-conditioned scores, rather
than interpreting a pooled case as pure hardware change. In particular,
Kyoto has 117 distinct collided MLP feature vectors:148 observations are not
148 independent circuit structures. Graph can see changed nominal calibration,
but zero complete-tensor collisions does not prove learned hardware adaptation.
Nominal-index T1/T2 is not historical physical placement/calibration recovery.

MLP wins9/10 backend MAE comparisons, but can still have negative practical
skill: on Jakarta both lose to the global median; on Manila MLP loses to it;
on Nairobi Graph loses to it. Quito Graph skill is only .005 against global
median. Only Kyoto/Osaka have at least10 test groups among backend cases under
the frozen uncertainty threshold. Device generation/topology, source, workload,
shots and source runtime boundaries are not independently varied. No causal
generation-transfer claim is identified.

## Size extrapolation

QPack train_through_4 tests1,321 observations/3 structures; train_through_5
tests498/2. Graph/MLP MAE is .727/.641 and .474/.789s. Skill vs train median is
.140/.241 and .453/.090. Both underpredict all 498 rows in the second probe,
whose R² is−4.685/−13.060. Low absolute error and positive median skill do not
make a successful variance-explaining predictor. There is no bootstrap CI below
ten groups. These are partial structural-size probes, not a scalable QPU law.

## Shots: NOT ANSWERED for systematic real-QPU generalization

No saved systematic QPU shot-held-out/extrapolation study was identified in the
protocols or experiment/report inventory. Existing diversity is:

- Ma–Li340: shots1024 only.
- QPack3945: shots4096 only.
- Qonductor230: shots1000(116 rows),4000(82),8000(14),12000(12),16000(3),20000(3).

Per-backend/family counts are in `shot_diversity.csv`. Qonductor's variation
is not a balanced factorial experiment; shots correlate with source/workload.
Graph/MLP lack a shots input. Regression has one; deterministic analytical
shot scaling assumes repeat costs rather than validating them empirically.

A **separate saved CUDA-Q microbenchmark** varied shots at fixedq20/RX16;
`artifacts/cudaq_runtime/launch_scaling_v1_20260922/analysis/shot_sweep.csv`
documents API sampling cost. It is not QPU evidence or a prediction shot holdout.
Its zero-gate intercept is API/state/sampling overhead, not isolated physical
CUDA kernel-launch latency.

## Uncertainty and applicability

Use the existing10,000-replicate seed 42 structural-group bootstrap for paired
MAE deltas; intervals are conditional on fitted predictions, not retraining
or independent hardware repeats. No interval below ten shared groups. Seed
spread is not independent test support. Do not pool6,684 overlapping test-case
assignments as6,684 independent observations: they cover4,515 unique IDs.
Keep family/backend/size estimands separate and retain QPack's six-structure
dominance. Twolocalrandom remains unauthorized-at-execution, subsequently
independently admitted; interpretation does not rewrite that execution history.

The raw eleven zero fields are irrelevant to the current learned fit; other
test-only fields and ranges expose genuine representation support limitations.
Unseen-shots, topology-only benefit and isolated generation transfer remain
unanswered without invalidating the completed fixed-grid comparisons.
