#!/usr/bin/env python3
"""Evaluate whether critical-subgraph summaries add to the rich static model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_mali_feature_ablation import BLOCKS, fit_predict
from evaluate_mali_physical_baselines import ROOT, family_splits, grouped_splits, metrics, read_csv, strict_splits


CRITICAL_FIELDS = (
    "critical_p90_dt", "critical_p95_dt", "critical_p99_dt",
    "critical_p90_node_fraction", "critical_p95_node_fraction", "critical_p99_node_fraction",
    "critical_p90_duration_fraction", "critical_p95_duration_fraction", "critical_p99_duration_fraction",
    "critical_p90_mean_dt", "critical_p95_mean_dt", "critical_p99_mean_dt",
    "critical_p90_edge_fraction", "critical_p95_edge_fraction", "critical_p99_edge_fraction",
    "critical_top001_duration_fraction", "critical_top01_duration_fraction", "critical_top05_duration_fraction",
    "critical_top001_mean_dt", "critical_top01_mean_dt", "critical_top05_mean_dt",
    "criticality_duration_weighted_sum_dt", "criticality_mean_dt", "criticality_max_dt",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/validation/mali_physical_dag_v1")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    manifest = read_csv(args.artifact_dir / "manifest.csv")
    split = {(r["qasm_sha256"], r["backend"]): r for r in read_csv(args.split_manifest)}
    rich = {r["graph_id"]: r for r in read_csv(args.artifact_dir / "graph_summary_features.csv")}
    critical = {r["graph_id"]: r for r in read_csv(args.artifact_dir / "critical_subgraph_features.csv")}
    rows = []
    for row in manifest:
        item = dict(row)
        audit = split[(row["qasm_sha256"], row["backend"])]
        item.update({k: audit[k] for k in ("group_fold", "family_component")})
        item.update(rich[row["graph_id"]]); item.update(critical[row["graph_id"]])
        rows.append(item)

    fields = {"rich_summary": BLOCKS["rich_summary"],
              "rich_plus_critical": BLOCKS["rich_summary"] + CRITICAL_FIELDS}
    splits = [*grouped_splits(rows, args.folds), *strict_splits(rows, args.folds), *family_splits(rows)]
    target = np.asarray([float(r["target_seconds"]) for r in rows])
    metric_rows = []; pred_rows = []
    for split_name, train, test in splits:
        for model, model_fields in fields.items():
            prediction = fit_predict(rows, train, test, model_fields)
            metric_rows.append({"split": split_name, "model": model, "n_train": len(train), "n_test": len(test), **metrics(target[test], prediction)})
            for idx, estimate in zip(test, prediction):
                pred_rows.append({"split": split_name, "model": model, "row_id": rows[idx]["row_id"],
                                  "circuit": rows[idx]["circuit"], "backend": rows[idx]["backend"],
                                  "target_seconds": target[idx], "prediction_seconds": estimate,
                                  "absolute_error_seconds": abs(float(estimate) - target[idx])})

    def pooled(prefix: str) -> dict[str, dict[str, float]]:
        result = {}
        for model in fields:
            chosen = [r for r in pred_rows if r["model"] == model and str(r["split"]).startswith(prefix)]
            result[model] = metrics(np.asarray([float(r["target_seconds"]) for r in chosen]), np.asarray([float(r["prediction_seconds"]) for r in chosen]))
        return result

    output = args.artifact_dir / "critical_subgraph_evaluation"; output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0])); writer.writeheader(); writer.writerows(metric_rows)
    with (output / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pred_rows[0])); writer.writeheader(); writer.writerows(pred_rows)
    summary = {"n_rows": len(rows), "features": list(CRITICAL_FIELDS), "pooled_metrics": {"grouped_qasm": pooled("grouped"), "strict_all": pooled("strict"), "family_all": pooled("family")}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = ["# Ma--Li critical-subgraph evaluation", "", "The `rich_plus_critical` model adds high-criticality tail and induced-subgraph summaries to the same rich static descriptor baseline. All preprocessing and fits are train-only on the existing folds.", "", "| Evaluation | Model | MAE (s) | log-R² | seconds-R² |", "|---|---|---:|---:|---:|"]
    for evaluation, key in (("grouped-QASM", "grouped_qasm"), ("strict backend+QASM", "strict_all"), ("family held-out", "family_all")):
        for model, value in summary["pooled_metrics"][key].items():
            report.append(f"| {evaluation} | `{model}` | {value['mae_seconds']:.4f} | {value['r2_log1p_seconds']:.4f} | {value['r2_seconds']:.4f} |")
    report += ["", "A gain over `rich_summary` must be stable across grouped, strict and family splits before a critical-subgraph model is preferred. If it is absent, the bottleneck descriptors are redundant with existing aggregate timing features."]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
