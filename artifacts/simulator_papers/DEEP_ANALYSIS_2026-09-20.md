# Deep analysis of the simulator-estimator validation pass

**Date:** 20 September 2026
**Method:** descriptive analysis of existing raw rows and out-of-fold outputs.
No model was retrained for this report.  The mechanisms below are
evidence-consistent explanations, not causal identification.

## VQCSim: high score is credible for this contract, but mainly tests a smooth small generator grid

The base and mirror circuits have a median time ratio of
**1.014x**. Their mean
absolute timing change is **1.72%**
(maximum **5.21%**).  The
primary circuit-group split therefore did not obtain its score merely by
splitting a base/mirror pair across train and test.

However, the corpus still contains only five generator families and nine
ordered widths.  Its circuit-group HGB residual is largest for
`vqe_two_local` (MAE 101.297 ms) and at
width 24 (MAE 326.607 ms).
The strong grouped score is thus best interpreted as interpolation over smooth
family/size trajectories under fixed hardware and prepared-inference timing.
The lower width-held-out result in the main report is the more relevant warning
for a new width; unseen circuit templates, sampling and CPU paths remain out
of scope.

## Noisy Aer: conformal coverage fails because backend shift violates the exchangeability assumption

| Held-out backend | N | Coverage | Median actual/predicted | Median interval width (s) | Median log radius |
|---|---:|---:|---:|---:|---:|
| FakeSherbrooke | 596 | 27.9% | 0.686x | 0.292 | 0.089 |
| FakeWashingtonV2 | 596 | 15.6% | 1.318x | 0.166 | 0.058 |


The conformal residual distribution is estimated from the training backend.
After a fake-backend holdout, compiled gate/routing/noise context changes in a
way that the current feature representation and residual radius do not cover.
The result is not evidence that conformal prediction is inherently invalid;
it is evidence that the current calibration is **not domain-conditional**.
Backend-specific calibration, a richer backend representation, or an explicit
shift detector is required before an interval can be shown to a user.

## PPS: `delta` controls a representation transition rather than a smooth depth effect

Across the controlled grid, the median propagation time by declared delta is:

| Delta | Median seconds | Minimum seconds | Maximum seconds |
|---:|---:|---:|---:|
| 1e-03 | 0.5764 | 0.0098 | 0.9560 |
| 5e-03 | 0.0346 | 0.0030 | 0.0484 |
| 1e-02 | 0.0132 | 0.0031 | 0.0420 |


The post-run number of retained Pauli terms is deliberately excluded from the
predictor, yet it changes abruptly once truncation becomes less aggressive.
That gives a plausible mechanism for the weak delta-held-out behavior: three
control levels do not identify a stable response curve, and the same logical
circuit can cross a representation-complexity boundary.  The largest HGB
delta-held-out mean absolute log error is at delta `1e-03`
(2.433).  A useful next model needs
more declared control levels and a model class that can express monotone,
piecewise behavior; it must still avoid post-run representation leakage.

## Plan ranking: FLOPs explain much of this feasible set, not all GPU time

The weakest within-network Spearman correlation between log FLOPs and log warm
runtime occurs for `ghz` at
8 qubits
(0.107).  The largest
family-level regret is `ghz`
(1.164x).  Six of seven explicit random plans
are infeasible under the declared 50% VRAM limit; the remaining failure is
unsupported by the cuTensorNet path API.

FLOP count does not encode peak workspace, slicing, memory traffic, kernel
choice, or launch overhead.  More importantly, the 80% selection result is
conditional on the feasible candidate subset.  A deployable ranker must first
model feasibility under a memory budget, then rank among feasible plans; it
cannot learn from a table where infeasible plans are silently removed.

## EMU-MPS: the target is quantized and the current family signal is constructed

| Width | Labels | Minimum threshold | Median threshold | Maximum threshold |
|---:|---:|---:|---:|---:|
| 8 | 8 | 2 | 8.0 | 8 |
| 12 | 8 | 2 | 16.0 | 16 |
| 16 | 8 | 4 | 32.0 | 32 |


Exact rung accuracy is low partly because the label space doubles at each step;
an error of one rung is already a 2x bond-dimension difference.  The 100%
family-classifier result has no generalization meaning here: pulse duration,
segment count, amplitudes and detuning deterministically identify the eight
constructed families.  The residual correction ties the shared model because
the available data are dominated by a monotone width/intensity pattern and
every held-out width still has every family in training.  An unseen-family
claim needs a non-constructed circuit corpus and a family split that removes
the entire family from training.

## Precision: random row splits share numerical regimes that family holdout removes

| Family | Pairs | Safe complex64 rate | Median relative difference | Maximum relative difference | Median 128/64 time |
|---|---:|---:|---:|---:|---:|
| ghz | 8 | 100.0% | 1.711e-08 | 1.711e-08 | 2.838x |
| hea | 12 | 50.0% | 5.300e-07 | 2.245e-06 | 3.407x |
| qaoa_cycle | 12 | 0.0% | 1.663e-06 | 5.471e-06 | 3.383x |
| qft | 8 | 75.0% | 3.576e-07 | 5.364e-07 | 4.001x |
| random_brickwork | 12 | 33.3% | 5.265e-07 | 1.372e-06 | 3.450x |


At the declared `5e-7` scalar-relative-difference tolerance, family-held-out
evaluation chooses complex64 only
3 times, and
2 of those
choices are unsafe.  The label is sensitive to circuit family, depth and
contraction path, while the corpus has only two seeds and one scalar output
per pair.  Random folds leak those regimes across train/test; family holdout
exposes their absence.  Moreover, scalar agreement is not state fidelity, so
even a better classifier would need a separate numerical-accuracy validation.

## Engineering implications

1. Keep estimator, feasibility, confidence interval and numerical-precision
   decisions as separate heads with separate labels.
2. Make configuration/domain holdout the release gate. Random-row scores are
   descriptive only.
3. Version backend, precision, memory budget, candidate generator and timing
   inclusion policy with every label.  These are causal runtime inputs, not
   incidental provenance.
4. Collect more independent circuit seeds and configurations before increasing
   model complexity. The current bottleneck is experimental support, not the
   absence of another regressor.
