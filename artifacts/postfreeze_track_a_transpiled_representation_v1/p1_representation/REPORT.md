# P1 representation results

## Facts

The frozen Track-A panel supplied 4,515 observations and 165 group-disjoint
identities.  P0 established complete physical-graph and nominal-duration-proxy
coverage, so every S0/P0/P1/H0/H1 comparison uses the identical 4,515 held-out
OOF rows.  Ridge is the preregistered primary learner; HGB is one bounded
secondary robustness learner.  All hyperparameter selection was inside each
outer training partition.

| Learner | View | Group-balanced MAE (s) | Pooled MAE (s) | Raw R² | log1p R² |
|---|---|---:|---:|---:|---:|
| ridge | S0 | 0.853614 | 1.153642 | -0.053790 | -0.006479 |
| ridge | P0 | 1.189556 | 0.981639 | 0.341224 | 0.367539 |
| ridge | P1 | 0.777248 | 0.958265 | 0.412883 | 0.404247 |
| ridge | H0 | 0.647324 | 0.894772 | 0.391948 | 0.368819 |
| ridge | H1 | 0.657647 | 0.914899 | 0.366661 | 0.359970 |
| hgb | S0 | 0.630509 | 0.859529 | 0.497342 | 0.475706 |
| hgb | P0 | 0.627485 | 0.864478 | 0.452223 | 0.393129 |
| hgb | P1 | 0.630099 | 0.961332 | 0.300500 | 0.233613 |
| hgb | H0 | 0.648004 | 0.854972 | 0.490842 | 0.445461 |
| hgb | H1 | 0.611596 | 0.886080 | 0.392884 | 0.359352 |


Negative deltas below favor the left view.  Intervals are 10,000 paired
structural-group bootstrap draws (seed 42), conditional on these saved fitted
models; they do not add retraining uncertainty.

| Learner | Comparison | Δ group-balanced MAE (s) | 95% CI |
|---|---|---:|---|
| ridge | Transpiled-static − Source-static | +0.335942 | [+0.160810, +0.510825] |
| ridge | Timing − Transpiled-static | -0.412308 | [-0.562020, -0.271797] |
| ridge | Hybrid − Source-static | -0.206289 | [-0.404378, -0.062162] |
| ridge | Hybrid − Transpiled-static | -0.542231 | [-0.739623, -0.371255] |
| ridge | Hybrid+timing − Hybrid | +0.010323 | [-0.054420, +0.097901] |
| hgb | Transpiled-static − Source-static | -0.003024 | [-0.070772, +0.055686] |
| hgb | Timing − Transpiled-static | +0.002614 | [-0.025257, +0.030085] |
| hgb | Hybrid − Source-static | +0.017494 | [-0.021167, +0.056868] |
| hgb | Hybrid − Transpiled-static | +0.020519 | [-0.037120, +0.085484] |
| hgb | Hybrid+timing − Hybrid | -0.036408 | [-0.071769, -0.004819] |


## Interpretation

The primary Ridge comparison does **not** support H1: Transpiled-static alone
is worse than Source-static.  It does show an H3-style improvement when the
saved nominal duration proxy is added to physical static descriptors, and an
H2-style Ridge complementarity result for H0.  Those directions are not
replicated by the bounded HGB robustness analysis: its relevant intervals all
cross zero except H1−H0, which is a hybrid timing effect rather than a
standalone transpiled-static gain.  The source slices are heterogeneous and
the Qonductor slice is adverse for the Ridge hybrid, so neither the Ridge-only
timing nor hybrid result warrants a broad representation claim.

P2 is therefore **not started**.  The preregistered P1 gate requires a
favorable physical/transpiled representation result not explained by a single
source, no severe tail issue, and a feasible bounded graph design.  Direct
Transpiled-static evidence fails that gate; a topology model would be a rescue
search rather than attribution of an established representation benefit.

## Verdict

**A. No benchmark-scale support for transpiled representation.** This is a
bounded negative result for the declared static representation and timing proxy
under the heterogeneous Track-A target contract.  It neither contradicts the
historical 340-row pilot nor says that exact physical representation is
universally unhelpful.

## Limits

- Qonductor has exact submitted physical structure, while Ma–Li/QPack physical
  views are frozen reconstruction/nominal target views.
- The duration feature is a saved nominal single-shot proxy, not historical
  calibration timing; source target boundaries remain heterogeneous.
- 3,945 QPack rows represent only six groups; group balancing addresses row
  multiplicity but cannot create independent circuits.
- This study contains no physical-DAG message passing and no QPU coupling graph.
