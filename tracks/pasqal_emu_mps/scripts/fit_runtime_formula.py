#!/usr/bin/env python3
"""Calibrate the documented EMU-MPS runtime scaling on local GPU runs.

The fit is diagnostic only.  The current pilot has three points and uses the
observed maximum bond dimension, which is not available before a target run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 3:
        raise ValueError("At least three observations are needed for diagnostics.")

    n = np.array([float(row["n_qubits"]) for row in rows])
    chi = np.array([float(row["max_bond_dimension_observed"]) for row in rows])
    target = np.array([float(row["sum_step_seconds"]) for row in rows])
    x = np.column_stack((n**2 * chi**3, n**3 * chi**2))
    coefficients, _, rank, singular_values = np.linalg.lstsq(x, target, rcond=None)
    prediction = x @ coefficients
    residual = prediction - target
    condition_number = float(singular_values[0] / singular_values[-1])

    output_rows = []
    for row, estimate, error in zip(rows, prediction, residual):
        output_rows.append(
            {
                **row,
                "formula_predicted_step_sum_seconds": float(estimate),
                "formula_error_seconds": float(error),
                "formula_relative_error": float(error / float(row["sum_step_seconds"])),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output.with_suffix(".csv")
    with predictions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    report_path = args.output.with_suffix(".md")
    warning = (
        "The fitted alpha is negative and the design matrix is ill-conditioned; "
        "this is not a stable predictive estimator yet."
        if coefficients[0] < 0 or coefficients[1] < 0 or condition_number > 20
        else "Both coefficients are non-negative and the pilot fit is numerically stable."
    )
    lines = [
        "# EMU-MPS runtime formula calibration",
        "",
        "Target: `sum_step_seconds` (sum of EMU-MPS per-step durations).",
        "",
        "The documented diagnostic form is `T ≈ α N²χ³ + β N³χ²`.",
        "This run uses observed maximum `χ`, so it is post-run calibration, not a pre-run estimate.",
        "",
        f"- alpha: `{coefficients[0]:.12g}`",
        f"- beta: `{coefficients[1]:.12g}`",
        f"- matrix rank: `{rank}`",
        f"- condition number: `{condition_number:.3f}`",
        f"- diagnostic: {warning}",
        "",
        "| case | actual step sum (s) | predicted (s) | relative error |",
        "|---|---:|---:|---:|",
    ]
    for row in output_rows:
        lines.append(
            f"| {row['case']} | {float(row['sum_step_seconds']):.3f} | "
            f"{float(row['formula_predicted_step_sum_seconds']):.3f} | "
            f"{100 * float(row['formula_relative_error']):.1f}% |"
        )
    lines += [
        "",
        "The pilot is too small and `N` and `χ` are correlated, so the two terms "
        "cannot yet be separated reliably. More runs varying `max_bond_dim`, "
        "pulse duration and sequence family are required before using this as a "
        "pre-run predictor.",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "formula": "T = alpha*N^2*chi^3 + beta*N^3*chi^2",
                "target": "sum_step_seconds",
                "chi_source": "max_bond_dimension_observed",
                "alpha": float(coefficients[0]),
                "beta": float(coefficients[1]),
                "condition_number": condition_number,
                "n_observations": len(rows),
            },
            indent=2,
        )
        + "\n"
    )
    print(report_path)
    print(predictions_path)


if __name__ == "__main__":
    main()
