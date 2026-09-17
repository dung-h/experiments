#!/usr/bin/env python3
"""Leakage and stability checks for the Ma--Li/QCRE proxy artifact.

This pass never transpiles a circuit.  It consumes the committed feature CSV,
groups Osaka/Kyoto copies by logical-QASM SHA-256, and applies one fixed
grouped five-fold split to every feature and ablation.  The resulting scores
are therefore out-of-sample with respect to repeated logical circuits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

try:
    from mali_qcre_proxy import backend_metadata, held_out_group_calibration, regression_metrics
except ImportError:  # pragma: no cover - allows direct import from another cwd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mali_qcre_proxy import backend_metadata, held_out_group_calibration, regression_metrics


ROOT = Path(__file__).resolve().parents[1]
WEIGHTED_FEATURE = "qcre_weighted_critical_path_seconds"
SINGLE_FEATURES = (
    "logical_depth",
    "logical_two_qubit_depth",
    "physical_depth",
    "physical_two_qubit_depth",
    WEIGHTED_FEATURE,
)
ABLATIONS = {
    "physical_depth": ("physical_depth",),
    "physical_depth_plus_physical_2q_depth": (
        "physical_depth",
        "physical_two_qubit_depth",
    ),
    "physical_depth_plus_weighted_path": (
        "physical_depth",
        WEIGHTED_FEATURE,
    ),
    "physical_depth_plus_physical_2q_depth_plus_weighted_path": (
        "physical_depth",
        "physical_two_qubit_depth",
        WEIGHTED_FEATURE,
    ),
}


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {path}")
    for row in rows:
        row.setdefault("shots", "1024")
    return rows


def add_logical_hashes(rows: list[dict[str, object]], mali_root: Path) -> None:
    for row in rows:
        path = mali_root / str(row["qasm_path"])
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            digest = str(row.get("qasm_sha256", ""))
        if not digest:
            raise FileNotFoundError(f"QASM unavailable for {row['circuit']}: {path}")
        row["logical_circuit_hash"] = digest


def make_group_folds(rows: list[dict[str, object]], seed: int, folds: int) -> tuple[list[int], dict[str, object]]:
    groups = sorted({str(row["logical_circuit_hash"]) for row in rows})
    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(groups)[rng.permutation(len(groups))])
    assignment: dict[str, int] = {}
    for fold, bucket in enumerate(np.array_split(np.asarray(shuffled, dtype=object), folds)):
        for group in bucket:
            assignment[str(group)] = fold
    row_folds = [assignment[str(row["logical_circuit_hash"])] for row in rows]
    duplicate_groups = {
        group: sum(str(row["logical_circuit_hash"]) == group for row in rows)
        for group in groups
        if sum(str(row["logical_circuit_hash"]) == group for row in rows) > 1
    }
    size_counts: dict[str, int] = {}
    for size in duplicate_groups.values():
        size_counts[str(size)] = size_counts.get(str(size), 0) + 1
    metadata = {
        "strategy": "grouped_5fold_by_logical_qasm_sha256",
        "seed": seed,
        "folds": folds,
        "n_rows": len(rows),
        "n_unique_logical_circuits": len(groups),
        "duplicate_logical_groups": len(duplicate_groups),
        "duplicate_group_size_counts": size_counts,
        "fold_row_counts": [row_folds.count(fold) for fold in range(folds)],
        "fold_group_counts": [
            sum(assignment[group] == fold for group in groups) for fold in range(folds)
        ],
    }
    return row_folds, metadata


def design(rows: list[dict[str, object]], indices: list[int], features: tuple[str, ...]) -> np.ndarray:
    values = np.asarray(
        [[float(row[feature]) for feature in features] for row in rows], dtype=float
    )
    return np.column_stack((np.ones(len(indices)), np.log1p(np.maximum(values[indices], 0))))


def grouped_log_calibration(
    rows: list[dict[str, object]], row_folds: list[int], features: tuple[str, ...], folds: int
) -> tuple[np.ndarray, dict[str, object]]:
    target = np.asarray([float(row["target_time_taken_seconds"]) for row in rows])
    prediction = np.empty(len(rows), dtype=float)
    fold_results: list[dict[str, object]] = []
    for fold in range(folds):
        test = np.asarray([index for index, value in enumerate(row_folds) if value == fold], dtype=int)
        train = np.asarray([index for index, value in enumerate(row_folds) if value != fold], dtype=int)
        coefficients, *_ = np.linalg.lstsq(
            design(rows, train, features), np.log1p(target[train]), rcond=None
        )
        prediction[test] = np.expm1(design(rows, test, features) @ coefficients)
        fold_results.append(
            {
                "fold": fold + 1,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_train_groups": len({str(rows[index]["logical_circuit_hash"]) for index in train}),
                "n_test_groups": len({str(rows[index]["logical_circuit_hash"]) for index in test}),
                "metrics": regression_metrics(target[test], prediction[test]),
            }
        )
    return prediction, {
        "features": list(features),
        "folds": fold_results,
        "aggregate": regression_metrics(target, prediction),
        "aggregate_medae_seconds": float(
            statistics.median(abs(float(actual) - float(estimate)) for actual, estimate in zip(target, prediction))
        ),
    }


def grouped_median_baseline(
    rows: list[dict[str, object]], row_folds: list[int], folds: int
) -> dict[str, object]:
    target = np.asarray([float(row["target_time_taken_seconds"]) for row in rows])
    prediction = np.empty(len(rows), dtype=float)
    fold_results: list[dict[str, object]] = []
    for fold in range(folds):
        test = np.asarray([index for index, value in enumerate(row_folds) if value == fold], dtype=int)
        train = np.asarray([index for index, value in enumerate(row_folds) if value != fold], dtype=int)
        prediction[test] = float(np.median(target[train]))
        fold_results.append(
            {"fold": fold + 1, "train_median_seconds": float(np.median(target[train])),
             "metrics": regression_metrics(target[test], prediction[test])}
        )
    return {
        "aggregate": regression_metrics(target, prediction),
        "aggregate_medae_seconds": float(
            statistics.median(abs(float(actual) - float(estimate)) for actual, estimate in zip(target, prediction))
        ),
        "folds": fold_results,
    }


def grouped_affine_calibration(
    rows: list[dict[str, object]], row_folds: list[int], folds: int
) -> dict[str, object]:
    """Fit original-scale T_obs ~= alpha + beta * (shots * weighted path)."""
    target = np.asarray([float(row["target_time_taken_seconds"]) for row in rows])
    scaled = np.asarray(
        [float(row["shots"]) * float(row[WEIGHTED_FEATURE]) for row in rows]
    )
    prediction = np.empty(len(rows), dtype=float)
    fold_results: list[dict[str, object]] = []
    for fold in range(folds):
        test = np.asarray([index for index, value in enumerate(row_folds) if value == fold], dtype=int)
        train = np.asarray([index for index, value in enumerate(row_folds) if value != fold], dtype=int)
        x_train = np.column_stack((np.ones(len(train)), scaled[train]))
        coefficients, *_ = np.linalg.lstsq(x_train, target[train], rcond=None)
        x_test = np.column_stack((np.ones(len(test)), scaled[test]))
        prediction[test] = np.maximum(x_test @ coefficients, 0.0)
        fold_results.append(
            {
                "fold": fold + 1,
                "alpha_seconds": float(coefficients[0]),
                "beta": float(coefficients[1]),
                "metrics": regression_metrics(target[test], prediction[test]),
            }
        )
    return {
        "formula": "T_obs_seconds = alpha + beta * (shots * weighted_path_seconds)",
        "shots_values": sorted({int(float(row["shots"])) for row in rows}),
        "folds": fold_results,
        "aggregate": regression_metrics(target, prediction),
        "aggregate_medae_seconds": float(
            statistics.median(abs(float(actual) - float(estimate)) for actual, estimate in zip(target, prediction))
        ),
    }


def duration_audit(rows: list[dict[str, object]]) -> dict[str, object]:
    unsupported = sorted(
        {
            value
            for row in rows
            for value in str(row.get("qcre_unsupported_duration_ops", "")).split(",")
            if value
        }
    )
    dt_values = np.asarray([float(row["qcre_weighted_critical_path_dt"]) for row in rows])
    seconds_values = np.asarray([float(row[WEIGHTED_FEATURE]) for row in rows])
    return {
        "rows": len(rows),
        "unsupported_duration_ops": unsupported,
        "rows_with_unsupported_duration_ops": sum(bool(row.get("qcre_unsupported_duration_ops")) for row in rows),
        "negative_or_nonfinite_dt": int(np.sum(~np.isfinite(dt_values) | (dt_values < 0))),
        "negative_or_nonfinite_seconds": int(np.sum(~np.isfinite(seconds_values) | (seconds_values < 0))),
        "zero_duration_rows": int(np.sum(dt_values == 0)),
        "measurement_reset_delay_policy": (
            "native target durations are used for gates, measurement and reset; "
            "barrier/directive instructions are skipped; delay uses its own duration; "
            "unknown operations are listed in qcre_unsupported_duration_ops"
        ),
        "coverage_status": "complete" if not unsupported and np.all(dt_values > 0) else "review_required",
    }


def per_backend(
    rows: list[dict[str, object]], prediction: np.ndarray
) -> dict[str, object]:
    result: dict[str, object] = {}
    for backend in sorted({str(row["backend_label"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if str(row["backend_label"]) == backend]
        target = np.asarray([float(rows[index]["target_time_taken_seconds"]) for index in indices])
        estimate = prediction[indices]
        result[backend] = {
            "n": len(indices),
            "target_median_seconds": float(np.median(target)),
            "prediction_median_seconds": float(np.median(estimate)),
            "metrics": regression_metrics(target, estimate),
            "medae_seconds": float(statistics.median(abs(target - estimate))),
        }
    return result


def write_grouped_rows(path: Path, rows: list[dict[str, object]], row_folds: list[int], predictions: dict[str, np.ndarray]) -> None:
    output = []
    for index, row in enumerate(rows):
        enriched = dict(row)
        enriched["logical_circuit_hash"] = row["logical_circuit_hash"]
        enriched["grouped_fold"] = row_folds[index] + 1
        for name, values in predictions.items():
            enriched[f"grouped_pred_{name}_seconds"] = float(values[index])
        output.append(enriched)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def report(summary: dict[str, object]) -> str:
    split = summary["split"]
    duration = summary["duration_audit"]
    lines = [
        "# Ma–Li / QCRE final validation pass",
        "",
        "This report validates the committed `OUR_PROXY` feature artifact; it does",
        "not claim historical calibration recovery or a universal QPU estimator.",
        "",
        f"Rows: **{split['n_rows']}**; unique logical QASM hashes: **{split['n_unique_logical_circuits']}**; "
        f"grouped folds: **{split['folds']}**, seed **{split['seed']}**.",
        "",
        "## Split and duration integrity",
        "",
        f"Repeated Osaka/Kyoto copies are grouped by SHA-256 (`{split['duplicate_logical_groups']}` "
        "duplicate logical groups), so no logical circuit appears in both train and test.",
        f"Duration coverage: **{duration['coverage_status']}**; unsupported operation rows: "
        f"**{duration['rows_with_unsupported_duration_ops']}**; zero-duration rows: "
        f"**{duration['zero_duration_rows']}**.",
        f"Policy: {duration['measurement_reset_delay_policy']}",
        "",
        "## Grouped out-of-sample single-feature calibration",
        "",
        "All rows below use the same grouped folds and fit log1p(target) only on",
        "the training groups of each fold.",
        "",
        "| Feature | MAE (s) | MedAE (s) | RMSE (s) | R² | log-R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feature, result in summary["single_feature_grouped_cv"].items():
        metric = result["aggregate"]
        lines.append(
            f"| `{feature}` | {metric['mae_seconds']:.4f} | {result['aggregate_medae_seconds']:.4f} | "
            f"{metric['rmse_seconds']:.4f} | {metric['r2_seconds']:.4f} | "
            f"{metric['r2_log1p_seconds']:.4f} |"
        )
    median = summary["grouped_median_baseline"]["aggregate"]
    lines.extend(
        [
            "",
            f"Grouped median baseline: MAE {median['mae_seconds']:.4f} s, "
            f"MedAE {summary['grouped_median_baseline']['aggregate_medae_seconds']:.4f} s, "
            f"R² {median['r2_seconds']:.4f}.",
            "",
            "## Incremental ablation (same grouped folds)",
            "",
            "| Feature set | MAE (s) | MedAE (s) | RMSE (s) | R² |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, result in summary["ablation_grouped_cv"].items():
        metric = result["aggregate"]
        lines.append(
            f"| `{name}` | {metric['mae_seconds']:.4f} | {result['aggregate_medae_seconds']:.4f} | "
            f"{metric['rmse_seconds']:.4f} | {metric['r2_seconds']:.4f} |"
        )
    affine = summary["shots_scaled_affine_grouped_cv"]
    lines.extend(
        [
            "",
            "## Shots-scaled affine calibration",
            "",
            "The upstream execution helper uses `shots=1024`; this test fits only",
            "the training groups of each fold:",
            "`T_obs = alpha + beta * (1024 * weighted_path)`.",
            "",
            f"MAE {affine['aggregate']['mae_seconds']:.4f} s; MedAE "
            f"{affine['aggregate_medae_seconds']:.4f} s; RMSE "
            f"{affine['aggregate']['rmse_seconds']:.4f} s; R² "
            f"{affine['aggregate']['r2_seconds']:.4f}.",
            "",
            "## Per-backend grouped out-of-sample result",
            "",
            "The weighted-path prediction below is produced by the same grouped",
            "calibration, then sliced by backend for diagnostics.",
            "",
            "| Backend | n | Target median (s) | Prediction median (s) | MAE (s) | MedAE (s) | R² |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for backend, result in summary["per_backend_weighted_path"].items():
        metric = result["metrics"]
        lines.append(
            f"| `{backend}` | {result['n']} | {result['target_median_seconds']:.4f} | "
            f"{result['prediction_median_seconds']:.4f} | {metric['mae_seconds']:.4f} | "
            f"{result['medae_seconds']:.4f} | {metric['r2_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Leave-one-backend-out: the two transfer directions",
            "",
            "These are the two domain-transfer directions, each trained on one",
            "backend and evaluated on the other. They are not five iid folds.",
            "",
            "| Train backend | Held-out backend | n | MAE (s) | RMSE (s) | R² |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for held_out, result in summary["leave_one_backend_out_weighted_path"].items():
        metric = result["metrics"]
        train_backend = ", ".join(result["train_groups"])
        lines.append(
            f"| `{train_backend}` | `{held_out}` | {result['n']} | "
            f"{metric['mae_seconds']:.4f} | {metric['rmse_seconds']:.4f} | "
            f"{metric['r2_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reading the result",
            "",
            "The grouped scores are the defensible numbers for this artifact. If",
            "weighted duration fails to improve the physical-depth baseline, the",
            "claim should remain that compiled physical structure carries the",
            "transferable signal; duration is then a calibrated proxy rather than",
            "an independently validated hardware clock.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=ROOT / "artifacts" / "validation" / "mali_qcre_proxy" / "mali_qcre_proxy_features.csv",
    )
    parser.add_argument("--mali-root", type=Path, default=ROOT.parent / "Quantum-Execution-Time-Prediction")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "validation" / "mali_qcre_final"
    )
    args = parser.parse_args()
    rows = load_rows(args.features_csv)
    add_logical_hashes(rows, args.mali_root.expanduser().resolve())
    row_folds, split = make_group_folds(rows, args.seed, args.folds)
    duration = duration_audit(rows)

    single: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}
    for feature in SINGLE_FEATURES:
        prediction, result = grouped_log_calibration(rows, row_folds, (feature,), args.folds)
        single[feature] = result
        predictions[feature] = prediction

    ablation: dict[str, object] = {}
    for name, features in ABLATIONS.items():
        prediction, result = grouped_log_calibration(rows, row_folds, features, args.folds)
        ablation[name] = result
        predictions[name] = prediction

    median_result = grouped_median_baseline(rows, row_folds, args.folds)
    affine = grouped_affine_calibration(rows, row_folds, args.folds)
    weighted_prediction = predictions[WEIGHTED_FEATURE]
    summary: dict[str, object] = {
        "provenance_class": "OUR_VALIDATION",
        "target_semantics": rows[0]["target_semantics"],
        "split": split,
        "duration_audit": duration,
        "grouped_median_baseline": median_result,
        "single_feature_grouped_cv": single,
        "ablation_grouped_cv": ablation,
        "shots_scaled_affine_grouped_cv": affine,
        "per_backend_weighted_path": per_backend(rows, weighted_prediction),
        "leave_one_backend_out_weighted_path": held_out_group_calibration(
            rows, WEIGHTED_FEATURE, "backend_label"
        ),
        "protocol": {
            "same_folds_for_all_features": True,
            "calibration": "log1p(target) ~ 1 + log1p(feature), fit on train groups only",
            "transpile": "already materialized in the committed proxy CSV; no transpilation in this pass",
        },
        "software_and_backend_metadata": backend_metadata(),
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mali_qcre_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "MALI_QCRE_FINAL_VALIDATION_REPORT.md").write_text(
        report(summary), encoding="utf-8"
    )
    write_grouped_rows(output_dir / "mali_qcre_grouped_validation_rows.csv", rows, row_folds, predictions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
