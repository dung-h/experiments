# Public MQT family-classifier pretraining

Recorded: 2026-09-18

## Purpose

The paper describes a family classifier pretrained on a separate set of
approximately 200 MQT Bench circuits, but the exact circuits and weights are
not bundled with the public Quantum Rings repository. To remove the previous
fold-local replacement, this experiment generates an independent public MQT
set, trains a classifier once, freezes it, and only then evaluates the
Quantum Rings public circuits.

The result is a reproducible public approximation. It is not a claim that the
authors' private training set or weights have been recovered.

## Pretraining data and protocol

| Item | Value |
|---|---|
| Source | MQT Bench public Python package |
| Package versions | `mqt.bench 2.2.3`, `qiskit 2.5.2` |
| Circuits | 200 QASM files, 20 per family |
| Families | DJ, GHZ, W-State, Graph State, Grover, QFT, QFT Entangled, QPE, QNN, VQE |
| Size range | 5–24 qubits |
| Generation | `alg` level, random parameters enabled, seed 0 (per-circuit seed recorded) |
| Features | 28 circuit features from the replication feature extractor |
| Split | stratified circuit-level 80/20, seed 0 |
| Evaluation leakage check | 0 duplicate QASM hashes against the 36 evaluation circuits |

The complete QASM set and provenance are in
`data/mqt_family_pretraining/`; the generation command is documented in its
README.

## Classifier result

The frozen MLP uses two hidden layers `(128, 64)`, ReLU/Adam, standardisation,
early stopping and a 160-example training split. Held-out accuracy is
**97.5% (39/40)**. The Random Forest reference reaches **90.0% (36/40)**.
There is one MLP error in the held-out split: a DJ circuit is predicted as
Graph State. The exact row-level predictions are in
`results/mqt_family_classifier/heldout_predictions.csv`.

This 97.5% is numerically the same as the accuracy reported in the paper, but
the matching number alone does not establish exact replication: the generated
circuits, seeds, feature implementation and model weights are not the
authors' private artifact.

## Frozen-classifier integration

`experiments/replicate_with_mqt_family.py` loads the saved MLP without fitting
on evaluation circuits. It maps the repository's labels to the ten-family
vocabulary and marks unsupported labels (for example Shor, QAOA and
TwoLocalRandom) as `UNKNOWN`. It then trains the family-aware predictor only
inside circuit-level 5-fold evaluation, comparing the public MQT predictions
with an oracle mapped family label.

| Target | Family input | Exact threshold | Within ±1 rung | R²(log runtime) | Median relative error |
|---|---|---:|---:|---:|---:|
| 0.75 | MQT-predicted | 56.94% | 81.94% | −0.198 | 63.8% |
| 0.75 | mapped oracle | 58.33% | 74.31% | −0.317 | 79.3% |
| 0.99 | MQT-predicted | 52.78% | 78.47% | −0.010 | 64.0% |
| 0.99 | mapped oracle | 57.64% | 69.44% | −0.842 | 65.8% |

The family classifier's evaluation accuracy is 33.33% over all 36 circuits,
46.15% among the 26 circuits whose labels belong to the ten-family vocabulary;
known-family coverage is 72.22%. These figures are not classifier test
accuracy. They show that MQT pretraining is now independent of evaluation,
but the generated MQT distribution does not transfer cleanly to every
repository family and the runtime model remains unstable on this tiny,
heterogeneous public set.

## Interpretation

This closes one reproducibility gap: the family input is no longer trained on
the 36 evaluation circuits. It does not close the protocol gaps around the
0.75 versus 0.99 threshold target, missing paper-protocol forward timings,
Quantum Rings SDK execution, private holdout truth, or the authors' exact MQT
pretraining artifact. The defensible conclusion is therefore:

> A public MQT-pretrained family classifier can be built reproducibly and can
> reach 97.5% on its own held-out generated set, but that accuracy does not
> by itself yield reliable cross-distribution runtime prediction on the
> Quantum Rings public evaluation circuits.
