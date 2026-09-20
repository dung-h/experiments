#!/usr/bin/env python3
"""Fit a small QCRE-style bond-dimension threshold classifier.

This is a pilot diagnostic on EMU-MPS labels.  The family label is supplied as
an input feature here; a production predictor must infer it from the sequence
or use a separate family classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERIC = [
    "n_qubits",
    "rows",
    "cols",
    "dt_ns",
    "precision",
    "target_fidelity",
    "pulse_duration_ns",
    "pulse_segments",
    "amplitude_max",
    "amplitude_mean",
    "detuning_max_abs",
    "detuning_mean",
]
CATEGORICAL = ["family"]


def make_model(seed: int = 1234) -> Pipeline:
    preprocessor = ColumnTransformer(
        [("numeric", "passthrough", NUMERIC),
         ("family", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)]
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=seed,
    )
    return Pipeline([("features", preprocessor), ("classifier", classifier)])


def rung(value: object) -> int:
    return int(np.log2(int(value)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    with args.labels.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["required_max_bond_dim"] != "*"]
    if len(rows) < 8:
        raise ValueError("The pilot needs at least eight uncensored labels.")

    x = [{key: (float(row[key]) if key in NUMERIC else row[key]) for key in NUMERIC + CATEGORICAL} for row in rows]
    y = np.array([rung(row["required_max_bond_dim"]) for row in rows])
    sizes = np.array([int(row["n_qubits"]) for row in rows])
    families = np.array([row["family"] for row in rows])
    predictions: list[dict[str, object]] = []

    def evaluate(group_name: str, group_values: np.ndarray) -> None:
        for held_out in sorted(set(group_values)):
            train = group_values != held_out
            test = ~train
            model = make_model()
            # ColumnTransformer needs a tabular 2-D input; a list of dicts is
            # interpreted as a one-dimensional object array by sklearn.
            train_frame = pd.DataFrame([x[i] for i in range(len(rows)) if train[i]])
            test_frame = pd.DataFrame([x[i] for i in range(len(rows)) if test[i]])
            model.fit(train_frame, y[train])
            predicted = model.predict(test_frame)
            test_indices = np.flatnonzero(test)
            for index, prediction in zip(test_indices, predicted):
                predictions.append(
                    {
                        "split": group_name,
                        "held_out": str(held_out),
                        "family": rows[index]["family"],
                        "n_qubits": rows[index]["n_qubits"],
                        "actual_threshold": int(2 ** y[index]),
                        "predicted_threshold": int(2 ** int(prediction)),
                        "actual_rung": int(y[index]),
                        "predicted_rung": int(prediction),
                        "absolute_rung_error": int(abs(prediction - y[index])),
                    }
                )

    # Size-held-out tests extrapolate to a new system size.
    evaluate("size", sizes)
    # Family-held-out tests check transfer to a new pulse family.
    evaluate("family", families)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_prefix.with_suffix(".csv")
    with predictions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    metrics = {}
    for split in ("size", "family"):
        subset = [row for row in predictions if row["split"] == split]
        metrics[split] = {
            "n": len(subset),
            "exact_accuracy": sum(row["absolute_rung_error"] == 0 for row in subset) / len(subset),
            "within_one_rung_accuracy": sum(row["absolute_rung_error"] <= 1 for row in subset) / len(subset),
            "mean_absolute_rung_error": sum(row["absolute_rung_error"] for row in subset) / len(subset),
        }
    report_path = args.output_prefix.with_suffix(".md")
    lines = [
        "# EMU-MPS required bond-dimension threshold predictor",
        "",
        "This is a small QCRE-style threshold-classification pilot.",
        "The label is the smallest max-bond-dimension rung reaching final-state fidelity 0.99 against a high-χ reference.",
        "The supplied family label is an input feature; therefore this is not yet an end-to-end unknown-family predictor.",
        "",
        "| Split | N | Exact | Within ±1 rung | Mean absolute rung error |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("size", "family"):
        m = metrics[split]
        lines.append(
            f"| {split}-held-out | {m['n']} | {100*m['exact_accuracy']:.1f}% | "
            f"{100*m['within_one_rung_accuracy']:.1f}% | {m['mean_absolute_rung_error']:.3f} |"
        )
    lines += [
        "",
        "The sample has only 12 labelled sequence configurations and four hand-designed pulse families.",
        "These metrics are feasibility evidence, not a generalization claim. The next run must add independent random/algorithmic families, vary dt/precision and infer family from raw pulse/circuit features.",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps({"n_labels": len(rows), "metrics": metrics, "features": NUMERIC + CATEGORICAL}, indent=2) + "\n"
    )
    print(report_path)
    print(predictions_path)


if __name__ == "__main__":
    main()
