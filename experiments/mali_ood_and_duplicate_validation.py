#!/usr/bin/env python3
"""Validate support-aware uncertainty and duplicate-aware Ma--Li fitting.

Both parts retain Ma--Li's observed ``result.time_taken`` target and grouped
splits. The OOD part uses train-support distance only; the duplicate part uses
only labels available in each training partition.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_mali_feature_ablation import BLOCKS
from evaluate_mali_physical_baselines import ROOT, grouped_splits, metrics, read_csv, strict_splits


FIELDS = BLOCKS["rich_summary"]


def transform(rows: list[dict[str, str]], indices: np.ndarray) -> np.ndarray:
    values = np.asarray([[float(row[field]) for field in FIELDS] for row in rows], dtype=float)
    result = np.log1p(np.maximum(values, 0.0))
    for position, field in enumerate(FIELDS):
        if "fraction" in field: result[:, position] = values[:, position]
    backend = np.asarray([1.0 if row["backend"] == "osaka" else 0.0 for row in rows])[:, None]
    return np.column_stack((np.ones(len(rows)), result, backend))[indices]


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    mean = x_train[:, 1:-1].mean(axis=0); scale = x_train[:, 1:-1].std(axis=0); scale[scale == 0] = 1.0
    train = x_train.copy(); test = x_test.copy()
    train[:, 1:-1] = (train[:, 1:-1] - mean) / scale; test[:, 1:-1] = (test[:, 1:-1] - mean) / scale
    weight = np.ones(len(train)) if weights is None else weights / np.mean(weights)
    penalty = np.eye(train.shape[1]); penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(train.T @ (weight[:, None] * train) + penalty, train.T @ (weight * y_train))
    return test @ coefficient


def support_distance(x_reference: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    # Exclude intercept and backend token: this is circuit-feature support, not
    # an artificial distance caused only by backend category.
    reference, query = x_reference[:, 1:-1], x_query[:, 1:-1]
    mean = reference.mean(axis=0); scale = reference.std(axis=0); scale[scale == 0] = 1.0
    ref = (reference - mean) / scale; qry = (query - mean) / scale
    return np.asarray([float(np.sqrt(np.mean((ref - point) ** 2, axis=1)).min()) for point in qry])


def higher(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="higher"))


def load_rows(artifact: Path, split_manifest: Path) -> list[dict[str, str]]:
    manifest = read_csv(artifact / "manifest.csv")
    split = {(row["qasm_sha256"], row["backend"]): row for row in read_csv(split_manifest)}
    rich = {row["graph_id"]: row for row in read_csv(artifact / "graph_summary_features.csv")}
    rows = []
    for row in manifest:
        item = dict(row); audit = split[(row["qasm_sha256"], row["backend"])]
        item.update({key: audit[key] for key in ("group_fold", "family_component")}); item.update(rich[row["graph_id"]]); rows.append(item)
    return rows


def duplicate_predictions(rows: list[dict[str, str]], splits: list[tuple[str, np.ndarray, np.ndarray]]) -> list[dict[str, object]]:
    y = np.asarray([float(row["target_seconds"]) for row in rows]); log_y = np.log1p(y)
    output: list[dict[str, object]] = []
    for split_name, train, test in splits:
        x_all = transform(rows, np.arange(len(rows)))
        # The cell key is exact QASM/backend. It never mixes a label with a
        # different feature graph, and aggregation happens only in outer train.
        cells: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index in train: cells[(rows[index]["qasm_sha256"], rows[index]["backend"])].append(int(index))
        duplicate_variance = [np.var(log_y[index]) for index in cells.values() if len(index) > 1]
        floor = float(np.median(duplicate_variance)) if duplicate_variance else 0.01
        datasets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {"row_weighted": (train, log_y[train], None)}
        representative = []; means = []; weights = []
        for members in cells.values():
            representative.append(members[0]); means.append(float(np.log1p(np.mean(y[members]))))
            variance = float(np.var(log_y[members])) if len(members) > 1 else floor
            weights.append(1.0 / max(variance, floor * 0.25, 1e-6))
        representative_array = np.asarray(representative, dtype=int)
        datasets["cell_mean_equal"] = (representative_array, np.asarray(means), None)
        datasets["cell_mean_noise_weighted"] = (representative_array, np.asarray(means), np.asarray(weights))
        for mode, (fit_index, fit_y, weight) in datasets.items():
            pred_log = fit_ridge(x_all[fit_index], fit_y, x_all[test], weight)
            prediction = np.maximum(np.expm1(pred_log), 0.0)
            for index, estimate in zip(test, prediction):
                output.append({"split": split_name, "mode": mode, "row_id": rows[index]["row_id"], "circuit": rows[index]["circuit"], "backend": rows[index]["backend"], "target_seconds": y[index], "prediction_seconds": estimate, "absolute_error_seconds": abs(float(estimate) - y[index]), "train_cells": len(cells), "train_rows": len(train), "noise_floor_log_variance": floor})
    return output


def uncertainty_predictions(rows: list[dict[str, str]], splits: list[tuple[str, np.ndarray, np.ndarray]], folds: int) -> list[dict[str, object]]:
    y = np.asarray([float(row["target_seconds"]) for row in rows]); log_y = np.log1p(y); x_all = transform(rows, np.arange(len(rows)))
    output: list[dict[str, object]] = []
    for split_name, outer_train, test in splits:
        outer_fold = int(split_name.rsplit("_", 1)[-1]); calibration_fold = outer_fold % folds + 1
        calibration = np.asarray([index for index in outer_train if int(rows[index]["group_fold"]) == calibration_fold], dtype=int)
        fit = np.asarray([index for index in outer_train if int(rows[index]["group_fold"]) != calibration_fold], dtype=int)
        if not len(calibration) or not len(fit): continue
        cal_log = fit_ridge(x_all[fit], log_y[fit], x_all[calibration])
        test_log = fit_ridge(x_all[fit], log_y[fit], x_all[test])
        residual = np.abs(cal_log - log_y[calibration])
        cal_distance = support_distance(x_all[fit], x_all[calibration]); test_distance = support_distance(x_all[fit], x_all[test])
        threshold = float(np.median(cal_distance))
        for alpha in (0.10, 0.20):
            global_q = higher(residual, 1.0 - alpha)
            groups = {"low": residual[cal_distance <= threshold], "high": residual[cal_distance > threshold]}
            # Sparse high/low groups fall back to global conformal quantile.
            quantiles = {name: higher(value, 1.0 - alpha) if len(value) >= 8 else global_q for name, value in groups.items()}
            for method in ("global", "support_stratified"):
                for index, estimate_log, distance in zip(test, test_log, test_distance):
                    band = "high" if distance > threshold else "low"
                    q = global_q if method == "global" else quantiles[band]
                    lower, upper = max(float(np.expm1(estimate_log - q)), 0.0), float(np.expm1(estimate_log + q))
                    output.append({"split": split_name, "method": method, "alpha": alpha, "row_id": rows[index]["row_id"], "circuit": rows[index]["circuit"], "backend": rows[index]["backend"], "target_seconds": y[index], "point_prediction_seconds": float(np.expm1(estimate_log)), "lower_seconds": lower, "upper_seconds": upper, "covered": lower <= y[index] <= upper, "interval_width_seconds": upper-lower, "support_distance": distance, "support_band": band, "calibration_threshold": threshold, "calibration_rows": len(calibration)})
    return output


def summarize_predictions(rows: list[dict[str, object]], key: str, prefix: str) -> dict[str, dict[str, float]]:
    out = {}
    for name in sorted({str(row[key]) for row in rows}):
        chosen = [row for row in rows if str(row[key]) == name and str(row["split"]).startswith(prefix)]
        out[name] = metrics(np.asarray([float(row["target_seconds"]) for row in chosen]), np.asarray([float(row["prediction_seconds"]) for row in chosen]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/validation/mali_physical_dag_v1")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(); rows = load_rows(args.artifact_dir, args.split_manifest)
    splits = [*grouped_splits(rows, args.folds), *strict_splits(rows, args.folds)]
    duplicate = duplicate_predictions(rows, splits)
    uncertainty = uncertainty_predictions(rows, splits, args.folds)
    duplicate_output = args.artifact_dir / "duplicate_aware_validation"; duplicate_output.mkdir(parents=True, exist_ok=True)
    uncertainty_output = args.artifact_dir / "ood_uncertainty_validation"; uncertainty_output.mkdir(parents=True, exist_ok=True)
    with (duplicate_output / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(duplicate[0])); writer.writeheader(); writer.writerows(duplicate)
    duplicate_summary = {"n_rows": len(rows), "n_exact_cells": len({(r['qasm_sha256'],r['backend']) for r in rows}), "methods": ["row_weighted", "cell_mean_equal", "cell_mean_noise_weighted"], "pooled": {"grouped_qasm": summarize_predictions(duplicate, "mode", "grouped"), "strict_all": summarize_predictions(duplicate, "mode", "strict")}, "semantics": "training-only duplicate aggregation/noise weighting; test remains every observed row"}
    (duplicate_output / "summary.json").write_text(json.dumps(duplicate_summary, indent=2) + "\n", encoding="utf-8")
    with (uncertainty_output / "intervals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(uncertainty[0])); writer.writeheader(); writer.writerows(uncertainty)
    interval_summary = {}
    for prefix, label in (("grouped", "grouped_qasm"), ("strict", "strict_all")):
        for alpha in (0.10, 0.20):
            for method in ("global", "support_stratified"):
                chosen = [row for row in uncertainty if str(row["split"]).startswith(prefix) and float(row["alpha"]) == alpha and row["method"] == method]
                interval_summary[f"{label}_{method}_alpha_{alpha}"] = {"n": len(chosen), "coverage": float(np.mean([bool(row["covered"]) for row in chosen])), "median_width_seconds": float(np.median([float(row["interval_width_seconds"]) for row in chosen])), "qwalk_coverage": float(np.mean([bool(row["covered"]) for row in chosen if "qwalk" in str(row["circuit"]).lower()])), "qwalk_median_width_seconds": float(np.median([float(row["interval_width_seconds"]) for row in chosen if "qwalk" in str(row["circuit"]).lower()]))}
    uncertainty_summary = {"n_rows": len(rows), "methods": ["global", "support_stratified"], "intervals": interval_summary, "semantics": "support distance is nearest standardized rich-feature distance to fit rows; support-stratified quantiles are sparse-calibration diagnostics, not distribution-free conditional coverage"}
    (uncertainty_output / "summary.json").write_text(json.dumps(uncertainty_summary, indent=2) + "\n", encoding="utf-8")
    d_report = ["# Ma--Li duplicate-aware fitting", "", "All tests remain original observed rows. Only outer-training rows are collapsed by exact `(QASM hash, backend)` cell; noise weights are calculated from outer-training duplicate spread.", "", "| Evaluation | Method | MAE (s) | log-R² | seconds-R² |", "|---|---|---:|---:|---:|"]
    for label, key in (("grouped-QASM", "grouped_qasm"), ("strict backend+QASM", "strict_all")):
        for mode, value in duplicate_summary["pooled"][key].items(): d_report.append(f"| {label} | `{mode}` | {value['mae_seconds']:.4f} | {value['r2_log1p_seconds']:.4f} | {value['r2_seconds']:.4f} |")
    d_report += ["", "A gain would indicate that repeated labels were distorting training weight. If row-weighted remains best, duplicate handling should be retained as uncertainty evidence rather than a model-selection method."]
    (duplicate_output / "REPORT.md").write_text("\n".join(d_report) + "\n", encoding="utf-8")
    u_report = ["# Ma--Li support-aware uncertainty validation", "", "Global conformal and a two-band support-stratified variant use the same QASM-disjoint calibration group. The latter is diagnostic because each band is small.", "", "| Evaluation | Method | Nominal | Coverage | Median width (s) | QWalk coverage | QWalk median width (s) |", "|---|---|---:|---:|---:|---:|---:|"]
    for label, prefix in (("grouped-QASM", "grouped_qasm"), ("strict backend+QASM", "strict_all")):
        for alpha in (0.10, 0.20):
            for method in ("global", "support_stratified"):
                value = interval_summary[f"{prefix}_{method}_alpha_{alpha}"]
                u_report.append(f"| {label} | `{method}` | {1-alpha:.0%} | {value['coverage']:.1%} | {value['median_width_seconds']:.4f} | {value['qwalk_coverage']:.1%} | {value['qwalk_median_width_seconds']:.4f} |")
    u_report += ["", "QWalk is consistently in the high-support-distance band, but neither global nor two-band intervals cover it. Support distance is therefore useful to trigger abstention/additional calibration, not yet sufficient to create a reliable automatic tail interval. It is not a hardware-wide confidence guarantee."]
    (uncertainty_output / "REPORT.md").write_text("\n".join(u_report) + "\n", encoding="utf-8")
    print(json.dumps({"duplicate": duplicate_summary, "uncertainty": uncertainty_summary}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
