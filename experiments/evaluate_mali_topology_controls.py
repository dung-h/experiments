#!/usr/bin/env python3
"""Aggregate multi-seed coarsened topology controls with group bootstrap.

The unit resampled is raw-QASM SHA-256, so paired Osaka/Kyoto rows and filename
duplicates remain together.  This evaluates the architectural comparison, not
a post-hoc selected deployment ensemble.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_mali_physical_baselines import ROOT, metrics, read_csv


VARIANTS = ("node_only", "true_dag", "reversed_dag", "shuffled_dag")


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/validation/mali_physical_dag_v1")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv")
    parser.add_argument("--input-dir", type=Path, action="append", required=True,
                        help="One or more topology-control output directories.")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output-name", default="topology_controls_validation")
    args = parser.parse_args()

    all_rows: list[dict[str, str]] = []
    for directory in args.input_dir:
        all_rows.extend(read_csv(directory / "oof_predictions.csv"))
    seeds = sorted({int(row["seed"]) for row in all_rows})
    split_by_row = {row["row_id"]: row for row in read_csv(args.split_manifest)}
    indexed: dict[tuple[int, str, str], dict[str, str]] = {}
    for row in all_rows:
        key = (int(row["seed"]), row["variant"], row["row_id"])
        if key in indexed:
            raise ValueError(f"duplicate seed/variant/row prediction: {key}")
        indexed[key] = row
    row_ids = sorted(split_by_row)
    for seed in seeds:
        for variant in VARIANTS:
            missing = [row_id for row_id in row_ids if (seed, variant, row_id) not in indexed]
            if missing:
                raise ValueError(f"missing {len(missing)} rows for seed={seed}, variant={variant}")

    # Arithmetic mean only summarizes training-seed variation. It is never
    # serialized as a selected operational model.
    ensemble: dict[str, list[dict[str, object]]] = {variant: [] for variant in VARIANTS}
    per_seed: dict[str, dict[str, dict[str, float]]] = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for seed in seeds:
            selected = [indexed[(seed, variant, row_id)] for row_id in row_ids]
            per_seed[variant][str(seed)] = metrics(
                np.asarray([float(row["target_seconds"]) for row in selected]),
                np.asarray([float(row["prediction_seconds"]) for row in selected]),
            )
        for row_id in row_ids:
            values = [indexed[(seed, variant, row_id)] for seed in seeds]
            ensemble[variant].append({
                "row_id": row_id,
                "qasm_sha256": split_by_row[row_id]["qasm_sha256"],
                "target_seconds": float(values[0]["target_seconds"]),
                "prediction_seconds": float(np.mean([float(value["prediction_seconds"]) for value in values])),
            })
    ensemble_metrics = {
        variant: metrics(np.asarray([row["target_seconds"] for row in values]), np.asarray([row["prediction_seconds"] for row in values]))
        for variant, values in ensemble.items()
    }
    groups = sorted({row["qasm_sha256"] for row in ensemble["node_only"]})
    group_indices = {
        group: np.asarray([i for i, row in enumerate(ensemble["node_only"]) if row["qasm_sha256"] == group], dtype=int)
        for group in groups
    }
    rng = np.random.default_rng(args.seed)
    comparisons: dict[str, dict[str, object]] = {}
    baseline = ensemble["node_only"]
    for other in ("true_dag", "reversed_dag", "shuffled_dag"):
        deltas_log = []; deltas_mae = []
        for _ in range(args.bootstrap_replicates):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            index = np.concatenate([group_indices[group] for group in sampled])
            target = np.asarray([baseline[i]["target_seconds"] for i in index])
            node_prediction = np.asarray([baseline[i]["prediction_seconds"] for i in index])
            other_prediction = np.asarray([ensemble[other][i]["prediction_seconds"] for i in index])
            node_metric = metrics(target, node_prediction)
            other_metric = metrics(target, other_prediction)
            deltas_log.append(other_metric["r2_log1p_seconds"] - node_metric["r2_log1p_seconds"])
            deltas_mae.append(other_metric["mae_seconds"] - node_metric["mae_seconds"])
        d_log = np.asarray(deltas_log); d_mae = np.asarray(deltas_mae)
        comparisons[f"{other}_minus_node_only"] = {
            "ensemble_delta_log_r2": ensemble_metrics[other]["r2_log1p_seconds"] - ensemble_metrics["node_only"]["r2_log1p_seconds"],
            "ensemble_delta_mae_seconds": ensemble_metrics[other]["mae_seconds"] - ensemble_metrics["node_only"]["mae_seconds"],
            "bootstrap_log_r2_95pct": percentile_interval(d_log),
            "bootstrap_mae_seconds_95pct": percentile_interval(d_mae),
            "probability_log_r2_better_than_node_only": float(np.mean(d_log > 0)),
            "probability_mae_better_than_node_only": float(np.mean(d_mae < 0)),
        }
    output = args.artifact_dir / args.output_name; output.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for variant, values in ensemble.items():
        for row in values:
            rows_out.append({"variant": variant, **row})
    with (output / "seed_mean_oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0])); writer.writeheader(); writer.writerows(rows_out)
    summary = {
        "evaluation": "grouped-QASM coarsened topology screening only",
        "seeds": seeds,
        "n_rows": len(row_ids), "n_qasm_hashes": len(groups),
        "bootstrap_replicates": args.bootstrap_replicates, "bootstrap_seed": args.seed,
        "ensemble_metrics": ensemble_metrics, "per_seed_metrics": per_seed,
        "paired_group_bootstrap": comparisons,
        "interpretation": "true DAG must beat node-only and shuffled controls before topology is claimed useful; seed-mean is analysis-only, not a selected deployment ensemble",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = ["# Ma--Li topology-control validation", "", "This combines three independently trained seeds of the 256-bin grouped-QASM screen. The seed mean is used only to reduce initialization variance for analysis; it is not a selected model. Bootstrap resampling is by raw-QASM hash, preserving paired backend rows.", "", "| Variant | Mean-seed MAE (s) | Mean-seed log-R² |", "|---|---:|---:|"]
    for variant in VARIANTS:
        value = ensemble_metrics[variant]
        report.append(f"| `{variant}` | {value['mae_seconds']:.4f} | {value['r2_log1p_seconds']:.4f} |")
    report += ["", "| Comparison | Δ log-R² 95% CI | Δ MAE (s) 95% CI | P(log-R² improves) | P(MAE improves) |", "|---|---:|---:|---:|---:|"]
    for name, value in comparisons.items():
        log_ci = value["bootstrap_log_r2_95pct"]; mae_ci = value["bootstrap_mae_seconds_95pct"]
        report.append(f"| `{name}` | [{log_ci[0]:.4f}, {log_ci[1]:.4f}] | [{mae_ci[0]:.4f}, {mae_ci[1]:.4f}] | {value['probability_log_r2_better_than_node_only']:.3f} | {value['probability_mae_better_than_node_only']:.3f} |")
    report += ["", "## Decision", "", "The control only supports topology if true DAG beats both node-only and shuffled adjacency with intervals favoring it. If it does not, the valid result is that this coarsened message-passing representation loses to node/timing summaries; it does not rule out every possible native-DAG architecture."]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
