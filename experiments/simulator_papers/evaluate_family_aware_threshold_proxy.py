#!/usr/bin/env python3
"""Run a nested known-family evaluation for the EMU-MPS threshold pilot.

The experiment mirrors the *structure* of a family-aware threshold predictor:
a classifier derives a known family from raw execution/circuit metadata, then
a shared numeric predictor is corrected by a family residual.  It is not a
reproduction of arXiv:2606.11620: the corpus is 12 local neutral-atom pulse
sequences, not the authors' QASM simulator dataset.

The only supported split is leave-one-size-out.  Every held-out size still has
examples of all four families in training, so this measures known-family size
transfer. It explicitly does not make a claim for unseen-family transfer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = [
    "n_qubits", "rows", "cols", "dt_ns", "precision", "target_fidelity",
    "pulse_duration_ns", "pulse_segments", "amplitude_max", "amplitude_mean",
    "detuning_max_abs", "detuning_mean",
]


def rung(label: int) -> int:
    return int(round(math.log2(label)))


def to_threshold(value: float, train_targets: np.ndarray) -> int:
    rungs = sorted(set(int(item) for item in train_targets))
    # Return the selected *rung*, not its bond-dimension value. Conversion to
    # 2**rung belongs only in the serialized threshold field.
    return min(rungs, key=lambda candidate: abs(candidate - value))


def metrics(rows: list[dict[str, object]], method: str) -> dict[str, float | int]:
    subset = [row for row in rows if row["method"] == method]
    errors = [int(row["absolute_rung_error"]) for row in subset]
    return {
        "n": len(subset),
        "exact_accuracy": sum(error == 0 for error in errors) / len(errors),
        "within_one_rung_accuracy": sum(error <= 1 for error in errors) / len(errors),
        "mean_absolute_rung_error": sum(errors) / len(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "labels", nargs="+", type=Path,
        help="One or more threshold-label CSVs with the same local EMU-MPS target semantics.",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(path) for path in args.labels], ignore_index=True)
    frame = frame[frame["required_max_bond_dim"] != "*"].copy()
    if "reference_censored" in frame:
        frame = frame[frame["reference_censored"].astype(str).str.lower() != "true"].copy()
    if len(frame) < 8:
        raise SystemExit("Need at least eight uncensored threshold labels")
    frame["required_max_bond_dim"] = frame["required_max_bond_dim"].astype(int)
    frame["actual_rung"] = frame["required_max_bond_dim"].map(rung)
    if frame["n_qubits"].nunique() < 3 or frame["family"].nunique() < 2:
        raise SystemExit("Need at least three sizes and two known families")

    predictions: list[dict[str, object]] = []
    for held_size in sorted(frame["n_qubits"].unique()):
        train = frame[frame["n_qubits"] != held_size].copy()
        test = frame[frame["n_qubits"] == held_size].copy()
        x_train, x_test = train[NUMERIC], test[NUMERIC]
        y_train = train["actual_rung"].to_numpy(dtype=int)
        y_test = test["actual_rung"].to_numpy(dtype=int)

        family_classifier = Pipeline([
            ("scale", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=300, random_state=args.seed)),
        ])
        family_classifier.fit(x_train, train["family"])
        inferred_family = family_classifier.predict(x_test)

        # Shared predictor: raw numeric context only.
        shared = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
        shared.fit(x_train, y_train)
        shared_train = shared.predict(x_train)
        shared_test = shared.predict(x_test)

        # Family residual is fitted only on the size-training partition.
        residuals = pd.DataFrame({
            "family": train["family"].to_numpy(),
            "residual": y_train - shared_train,
        }).groupby("family")["residual"].median().to_dict()
        global_residual = float(np.median(y_train - shared_train))

        for index, (_, item) in enumerate(test.iterrows()):
            actual = int(y_test[index])
            base_rung = to_threshold(float(shared_test[index]), y_train)
            residual_prediction = float(shared_test[index]) + float(residuals.get(inferred_family[index], global_residual))
            residual_rung = to_threshold(residual_prediction, y_train)
            for method, predicted in (("shared_numeric", base_rung), ("inferred_family_residual", residual_rung)):
                predictions.append({
                    "split": "leave_one_size_out_known_family",
                    "held_out_n_qubits": int(held_size),
                    "method": method,
                    "actual_family": item["family"],
                    "inferred_family": inferred_family[index],
                    "family_inference_correct": bool(inferred_family[index] == item["family"]),
                    "actual_threshold": int(2 ** actual),
                    "predicted_threshold": int(2 ** predicted),
                    "actual_rung": actual,
                    "predicted_rung": predicted,
                    "absolute_rung_error": abs(actual - predicted),
                })

    results = {method: metrics(predictions, method) for method in ("shared_numeric", "inferred_family_residual")}
    family_predictions = [row for row in predictions if row["method"] == "shared_numeric"]
    results["family_classifier"] = {
        "n": len(family_predictions),
        "accuracy": accuracy_score(
            [row["actual_family"] for row in family_predictions],
            [row["inferred_family"] for row in family_predictions],
        ),
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    summary = {
        "target_semantics": "smallest EMU-MPS max_bond_dim rung achieving final-state fidelity 0.99 against a local reference",
        "split": "leave-one-size-out with every family represented in training",
        "n_labels": len(frame),
        "features": NUMERIC,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "results": results,
        "limitation": (
            "Family labels have deterministic pulse-metadata signatures in this local constructed pulse corpus. "
            "This does not test unseen-family generalization or replicate the QASM-based paper corpus."
        ),
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    args.output_prefix.with_suffix(".md").write_text(f"""# Family-aware EMU-MPS threshold proxy

## Scope

- Conceptual reference: family-aware residual threshold prediction in
  arXiv:2606.11620.
- Target: smallest allowed EMU-MPS `max_bond_dim` rung attaining fidelity 0.99
  against a local high-χ reference; it is **not** wall-clock runtime.
- Corpus: **{len(frame)}** uncensored local pulse sequences across **{frame['family'].nunique()}** known pulse families,
  at sizes 8, 12 and 16.
- Split: leave one size out. The held-out split has every family in train;
  this is known-family size transfer only.

| Method | N | Exact rung | Within ±1 rung | Mean abs. rung error |
|---|---:|---:|---:|---:|
| Shared numeric | {results['shared_numeric']['n']} | {100 * results['shared_numeric']['exact_accuracy']:.1f}% | {100 * results['shared_numeric']['within_one_rung_accuracy']:.1f}% | {results['shared_numeric']['mean_absolute_rung_error']:.3f} |
| Inferred-family residual | {results['inferred_family_residual']['n']} | {100 * results['inferred_family_residual']['exact_accuracy']:.1f}% | {100 * results['inferred_family_residual']['within_one_rung_accuracy']:.1f}% | {results['inferred_family_residual']['mean_absolute_rung_error']:.3f} |

The classifier inferred the family with **{100 * results['family_classifier']['accuracy']:.1f}%** accuracy in the same nested folds. Since the families are deliberately constructed with visibly distinct pulse metadata, that value is a data-property check, not a machine-learning claim.

## Boundary

The result cannot be compared numerically with the paper's QASM runtime and
approximation-threshold results. It merely checks whether the proposed
family-classification → residual-correction pattern adds value to the existing
local EMU-MPS threshold pilot without using the true test family label.
""")
    print(args.output_prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
