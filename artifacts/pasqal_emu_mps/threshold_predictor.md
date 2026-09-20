# EMU-MPS required bond-dimension threshold predictor

This is a small QCRE-style threshold-classification pilot.
The label is the smallest max-bond-dimension rung reaching final-state fidelity 0.99 against a high-χ reference.
The supplied family label is an input feature; therefore this is not yet an end-to-end unknown-family predictor.

| Split | N | Exact | Within ±1 rung | Mean absolute rung error |
|---|---:|---:|---:|---:|
| size-held-out | 12 | 25.0% | 100.0% | 0.750 |
| family-held-out | 12 | 50.0% | 75.0% | 0.917 |

The sample has only 12 labelled sequence configurations and four hand-designed pulse families.
These metrics are feasibility evidence, not a generalization claim. The next run must add independent random/algorithmic families, vary dt/precision and infer family from raw pulse/circuit features.
