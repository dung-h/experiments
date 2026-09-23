#!/usr/bin/env python3
"""Estimate label noise and leakage-safe prediction intervals for Ma--Li.

Intervals use a simple split-conformal calibration on grouped-QASM folds. They
are not a distributional claim about IBM hardware; they quantify uncertainty
under the observed dataset/proxy protocol and make duplicate-label variation
explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_mali_feature_ablation import BLOCKS, fit_predict
from evaluate_mali_physical_baselines import ROOT, grouped_splits, metrics, read_csv, strict_splits


def quantile_higher(values: np.ndarray, probability: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(values, probability, method="higher"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/validation/mali_physical_dag_v1")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    manifest = read_csv(args.artifact_dir / "manifest.csv")
    audit = {(r["qasm_sha256"], r["backend"]): r for r in read_csv(args.split_manifest)}
    rich = {r["graph_id"]: r for r in read_csv(args.artifact_dir / "graph_summary_features.csv")}
    rows = []
    for row in manifest:
        item = dict(row)
        split_row = audit[(row["qasm_sha256"], row["backend"])]
        item.update({key: split_row[key] for key in ("group_fold", "family_component")})
        item.update(rich[row["graph_id"]]); rows.append(item)
    target = np.asarray([float(r["target_seconds"]) for r in rows])

    # Dataset noise floor: repeated labels for the exact same QASM/backend cell.
    duplicate_groups: dict[tuple[str, str], list[tuple[dict[str, str], float]]] = defaultdict(list)
    for row in rows:
        duplicate_groups[(row["qasm_sha256"], row["backend"])].append((row, float(row["target_seconds"])))
    pair_rows = []
    for (qasm_hash, backend), values in sorted(duplicate_groups.items()):
        if len(values) < 2:
            continue
        for left, right in zip(values[::2], values[1::2]):
            a, b = left[1], right[1]
            pair_rows.append({
                "qasm_sha256": qasm_hash, "backend": backend,
                "row_id_a": left[0]["row_id"], "row_id_b": right[0]["row_id"],
                "target_a_seconds": a, "target_b_seconds": b,
                "absolute_difference_seconds": abs(a - b),
                "log1p_difference": abs(np.log1p(a) - np.log1p(b)),
                "relative_difference_to_mean": abs(a - b) / max((a + b) / 2.0, 1e-12),
            })

    # Split-conformal intervals. We reserve one group fold inside each outer
    # training set for calibration; test rows never affect fit or quantiles.
    interval_rows = []
    for split_name, outer_train, test in [*grouped_splits(rows, args.folds), *strict_splits(rows, args.folds)]:
        fold = int(split_name.rsplit("_", 1)[-1])
        calibration_fold = fold % args.folds + 1
        held_backend = None
        if split_name.startswith("strict_"):
            held_backend = split_name.split("_")[1]
        allowed = np.ones(len(rows), dtype=bool)
        if held_backend:
            allowed &= np.asarray([r["backend"] != held_backend for r in rows])
        allowed &= np.asarray([int(r["group_fold"]) != fold for r in rows])
        calibration = np.asarray([i for i, r in enumerate(rows) if allowed[i] and int(r["group_fold"]) == calibration_fold], dtype=int)
        fit = np.asarray([i for i, r in enumerate(rows) if allowed[i] and int(r["group_fold"]) != calibration_fold], dtype=int)
        if len(fit) == 0 or len(calibration) == 0:
            continue
        fit_prediction = fit_predict(rows, fit, calibration, BLOCKS["rich_summary"])
        calibration_residual = np.abs(np.log1p(np.maximum(fit_prediction, 0.0)) - np.log1p(target[calibration]))
        test_prediction = fit_predict(rows, fit, test, BLOCKS["rich_summary"])
        test_log_prediction = np.log1p(np.maximum(test_prediction, 0.0))
        for alpha in (0.10, 0.20):
            q = quantile_higher(calibration_residual, 1.0 - alpha)
            lower = np.maximum(np.expm1(test_log_prediction - q), 0.0)
            upper = np.expm1(test_log_prediction + q)
            for index, pred, lo, hi in zip(test, test_prediction, lower, upper):
                interval_rows.append({
                    "split": split_name, "alpha": alpha, "row_id": rows[index]["row_id"],
                    "backend": rows[index]["backend"], "target_seconds": target[index],
                    "point_prediction_seconds": pred, "lower_seconds": lo, "upper_seconds": hi,
                    "covered": bool(lo <= target[index] <= hi), "interval_width_seconds": hi - lo,
                    "calibration_rows": len(calibration), "fit_rows": len(fit), "q_log": q,
                })

    def interval_summary(prefix: str, alpha: float) -> dict[str, float]:
        chosen = [r for r in interval_rows if str(r["split"]).startswith(prefix) and float(r["alpha"]) == alpha]
        widths = np.asarray([float(r["interval_width_seconds"]) for r in chosen])
        return {
            "n": len(chosen), "coverage": float(np.mean([bool(r["covered"]) for r in chosen])) if chosen else float("nan"),
            "median_width_seconds": float(np.median(widths)) if len(widths) else float("nan"),
            "p90_width_seconds": float(np.percentile(widths, 90)) if len(widths) else float("nan"),
        }

    duplicate_diffs = np.asarray([float(r["absolute_difference_seconds"]) for r in pair_rows])
    duplicate_log_diffs = np.asarray([float(r["log1p_difference"]) for r in pair_rows])
    oof_path = args.artifact_dir / "baseline_evaluation/oof_predictions.csv"
    oof = read_csv(oof_path)
    rich_grouped = [r for r in oof if r["model"] == "rich_summary" and r["split"].startswith("grouped")]
    rich_strict = [r for r in oof if r["model"] == "rich_summary" and r["split"].startswith("strict")]
    summary = {
        "duplicate_cells": len(pair_rows),
        "duplicate_rows": 2 * len(pair_rows),
        "duplicate_label_noise": {
            "absolute_diff_median_seconds": float(np.median(duplicate_diffs)),
            "absolute_diff_p90_seconds": float(np.percentile(duplicate_diffs, 90)),
            "absolute_diff_max_seconds": float(np.max(duplicate_diffs)),
            "log1p_diff_median": float(np.median(duplicate_log_diffs)),
            "pair_mean_rmse_floor_seconds": float(np.sqrt(np.mean((duplicate_diffs / 2.0) ** 2))),
        },
        "rich_static_oof_error": {
            "grouped_mae_seconds": float(np.mean([float(r["absolute_error_seconds"]) for r in rich_grouped])),
            "grouped_abs_error_p90_seconds": float(np.percentile([float(r["absolute_error_seconds"]) for r in rich_grouped], 90)),
            "strict_mae_seconds": float(np.mean([float(r["absolute_error_seconds"]) for r in rich_strict])),
            "strict_abs_error_p90_seconds": float(np.percentile([float(r["absolute_error_seconds"]) for r in rich_strict], 90)),
        },
        "conformal_intervals": {
            "grouped_alpha_0.10": interval_summary("grouped", 0.10),
            "grouped_alpha_0.20": interval_summary("grouped", 0.20),
            "strict_alpha_0.10": interval_summary("strict", 0.10),
            "strict_alpha_0.20": interval_summary("strict", 0.20),
        },
        "semantics": "observed Ma-Li result.time_taken; intervals calibrated on held-out QASM groups, not a hardware-wide confidence guarantee",
    }
    output = args.artifact_dir / "uncertainty_calibration"; output.mkdir(parents=True, exist_ok=True)
    with (output / "duplicate_label_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0])); writer.writeheader(); writer.writerows(pair_rows)
    with (output / "conformal_intervals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(interval_rows[0])); writer.writeheader(); writer.writerows(interval_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Ma--Li label noise and uncertainty calibration", "",
        "The duplicate audit groups rows by exact raw-QASM SHA-256 and backend.",
        f"It finds {len(pair_rows)} duplicated cells ({2 * len(pair_rows)} rows).",
        "The duplicate spread is a lower-bound warning for prediction precision:",
        f"median absolute difference {summary['duplicate_label_noise']['absolute_diff_median_seconds']:.4f} s,",
        f"P90 {summary['duplicate_label_noise']['absolute_diff_p90_seconds']:.4f} s,",
        f"maximum {summary['duplicate_label_noise']['absolute_diff_max_seconds']:.4f} s.", "",
        "Intervals use split-conformal calibration in log1p(seconds). The calibration",
        "groups are disjoint from both the fit and test QASM groups. Coverage is an",
        "empirical property of this dataset/proxy protocol, not an IBM hardware-wide",
        "confidence guarantee.", "",
        "| Evaluation | Target coverage | Empirical coverage | Median width (s) | P90 width (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key, alpha in (("Grouped-QASM", "grouped", 0.10), ("Grouped-QASM", "grouped", 0.20), ("Strict backend+QASM", "strict", 0.10), ("Strict backend+QASM", "strict", 0.20)):
        value = summary["conformal_intervals"][f"{key}_alpha_{alpha:.2f}"]
        report.append(f"| {label} | {1-alpha:.0%} | {value['coverage']:.1%} | {value['median_width_seconds']:.4f} | {value['p90_width_seconds']:.4f} |")
    report += ["", "The interval artifact is intended for a runtime-estimator deployment gate: return a point estimate plus an uncertainty band, and flag QWalk-like support gaps rather than silently presenting a precise scalar.", ""]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
