# Quantum Runtime Estimation Benchmark

Comparison of runtime prediction methods on archived real-QPU observations
and simulator measurements collected on a local workstation.

This repository contains the study protocol, preprocessing and analysis code,
aggregate dataset descriptions and local simulator measurements. It does not
redistribute the source real-QPU datasets or their row-level derived inputs.

## Start here

1. [Dataset construction](docs/data_preprocessing.md): source joins, units,
   circuit reconstruction, selection and missing data.
2. [Feature dictionary](docs/data_and_feature_dictionary.md): ordered inputs,
   extraction rules and fit-only transformations.
3. [Training and measurement](docs/training_and_measurement.md): folds, seeds,
   hardware, shots, clocks and quality thresholds.
4. [Results](docs/results.md) and [paper comparison](docs/paper_comparison.md).

| Track | Evaluation population | Results |
| --- | --- | --- |
| Real QPU | **4,515 observations**: Ma–Li 340 + Qonductor 230 + QPack 3,945; 165 groups | [Table](results/real_qpu/method_comparison.md) · [CSV](results/real_qpu/method_comparison.csv) · [Same-row comparisons](results/real_qpu/shared_row_comparisons.csv) |
| Simulator | 204 circuit members / 191 QASM hashes through 16 qubits; predictor core of 150 hashes at 2–9 qubits | [Table](results/simulator/method_comparison.md) · [CSV](results/simulator/method_comparison.csv) |

QPU methods use the same observation panel and outer folds, with declared
method-specific inputs. Simulator methods are compared within the same engine,
precision, execution clock and quality context. Scores across these contexts
are not a single leaderboard. Unified implementations are identified as
adaptations, not paper-exact reproductions.

## Contents

- `data/simulator/`: locally measured times, failures, quality records, features,
  frozen partitions and saved predictions. Source QASM and serialized source
  DAGs are not included.
- `results/`: both method tables, source profiles, coverage, paired comparisons and figures.
- `methods/`: project extraction, reconstruction, fitting and analysis code.
- `protocol/`: recorded method and execution settings.
- `docs/`: dataset preparation, experiment procedure, findings and limitations.
- `provenance/`: upstream revisions, publication selection and file hashes.

The real-QPU source labels, per-observation features, split membership and
predictions remain outside this repository. Their origins and processing are
documented; current aggregate QPU scores come from completed fresh fits and
independent saved-evidence checks. See [data availability](docs/distribution.md).

## Check this checkout

```bash
python3 scripts/verify.py
python3 -B scripts/check_qpu_results.py
python3 -B scripts/check_simulator_results.py
python3 -B -m unittest discover -s tests
```

These checks verify published files, reporting scope, QPU aggregate consistency
and saved simulator scores. They do not
retrain models, regenerate QPU scores or run simulators. Some retained runners
still use the original experiment paths and need external inputs and a matching
environment. See [reproduction scope](docs/reproduction.md) before running them.

Qonductor logical inputs are archive-supported recipes; QPack inputs are
representative-angle structural reconstructions. They are not exact recovered
historical logical circuits. The dataset report states the evidence and limits.

Historical IDs and paths inside saved implementation records identify the producing run.
They are provenance, not additional release directories or downloadable inputs.
