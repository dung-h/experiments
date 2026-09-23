#!/usr/bin/env python3
"""Measure which compact physical feature blocks add signal on Ma--Li.

The evaluator intentionally uses the same grouped-QASM, strict backend and
family-component folds as the physical-DAG baseline gate.  It fits a small
ridge model on log1p(seconds), with all preprocessing learned on the training
part of each split.  The purpose is attribution of the static baseline gain,
not a final universal estimator.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_mali_physical_baselines import (
    ROOT,
    family_splits,
    grouped_splits,
    metrics,
    read_csv,
    strict_splits,
)


BLOCKS: dict[str, tuple[str, ...]] = {
    "qcre": ("qcre_critical_path_seconds",),
    "compiled": (
        "qcre_critical_path_seconds", "compiled_depth", "active_physical_width",
    ),
    "graph_size": (
        "qcre_critical_path_seconds", "compiled_depth", "active_physical_width",
        "n_nodes", "n_edges",
    ),
    "layer": (
        "qcre_critical_path_seconds", "compiled_depth", "active_physical_width",
        "n_nodes", "n_edges", "rich_n_layers", "rich_layer_nodes_mean",
        "rich_layer_nodes_p90", "rich_layer_nodes_max",
    ),
    "timing": (
        "qcre_critical_path_seconds", "compiled_depth", "active_physical_width",
        "n_nodes", "n_edges", "rich_zero_duration_fraction",
        "rich_duration_sum_dt", "rich_duration_max_dt",
        "rich_duration_nonzero_mean_dt", "rich_duration_nonzero_p90_dt",
        "rich_forward_max_dt", "rich_forward_p90_dt", "rich_reverse_p90_dt",
        "rich_criticality_p90_dt",
    ),
    "edge": (
        "qcre_critical_path_seconds", "compiled_depth", "active_physical_width",
        "n_nodes", "n_edges", "rich_quantum_edge_fraction",
    ),
    "opcode": (
        "qcre_critical_path_seconds", "compiled_depth", "active_physical_width",
        "n_nodes", "n_edges", *(f"rich_opcode_{i}_count" for i in range(14)),
    ),
    "rich_summary": (
        "rich_n_nodes", "rich_n_edges", "rich_n_layers", "rich_active_width",
        "rich_zero_duration_fraction", "rich_duration_sum_dt",
        "rich_duration_max_dt", "rich_duration_nonzero_mean_dt",
        "rich_duration_nonzero_p90_dt", "rich_forward_max_dt",
        "rich_forward_p90_dt", "rich_reverse_p90_dt", "rich_criticality_p90_dt",
        "rich_layer_nodes_mean", "rich_layer_nodes_p90", "rich_layer_nodes_max",
        "rich_quantum_edge_fraction",
        *(f"rich_opcode_{i}_count" for i in range(14)),
    ),
}


def design(rows: list[dict[str, str]], indices: np.ndarray, fields: tuple[str, ...]) -> np.ndarray:
    values = np.asarray([[float(row[field]) for field in fields] for row in rows], dtype=float)
    # Log counts/times, but retain fractions as-is.  This avoids letting the
    # 300k-node graphs dominate the small ridge fit.
    transformed = np.log1p(np.maximum(values, 0.0))
    fractions = {i for i, field in enumerate(fields) if "fraction" in field}
    for i in fractions:
        transformed[:, i] = values[:, i]
    backend = np.asarray([1.0 if row["backend"] == "osaka" else 0.0 for row in rows])[:, None]
    return np.column_stack((np.ones(len(rows)), transformed, backend))[indices]


def fit_predict(rows: list[dict[str, str]], train: np.ndarray, test: np.ndarray,
                fields: tuple[str, ...]) -> np.ndarray:
    target = np.asarray([float(row["target_seconds"]) for row in rows])
    x_train = design(rows, train, fields)
    x_test = design(rows, test, fields)
    if x_train.shape[1] > 2:
        mean = x_train[:, 1:-1].mean(axis=0)
        scale = x_train[:, 1:-1].std(axis=0)
        scale[scale == 0] = 1.0
        x_train[:, 1:-1] = (x_train[:, 1:-1] - mean) / scale
        x_test[:, 1:-1] = (x_test[:, 1:-1] - mean) / scale
    penalty = np.eye(x_train.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(
        x_train.T @ x_train + penalty,
        x_train.T @ np.log1p(target[train]),
    )
    return np.expm1(x_test @ coefficient)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path,
                        default=ROOT / "artifacts/validation/mali_physical_dag_v1")
    parser.add_argument("--split-manifest", type=Path,
                        default=ROOT / "artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    manifest = read_csv(args.artifact_dir / "manifest.csv")
    split_rows = read_csv(args.split_manifest)
    split_by_key = {(r["qasm_sha256"], r["backend"]): r for r in split_rows}
    rich = {r["graph_id"]: r for r in read_csv(args.artifact_dir / "graph_summary_features.csv")}
    rows = []
    for row in manifest:
        extra = split_by_key[(row["qasm_sha256"], row["backend"])]
        merged = dict(row)
        merged.update({k: extra[k] for k in ("group_fold", "family_component")})
        merged.update(rich[row["graph_id"]])
        rows.append(merged)

    splits = [*grouped_splits(rows, args.folds), *strict_splits(rows, args.folds), *family_splits(rows)]
    target = np.asarray([float(row["target_seconds"]) for row in rows])
    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for split_name, train, test in splits:
        for block, fields in BLOCKS.items():
            prediction = fit_predict(rows, train, test, fields)
            result = metrics(target[test], prediction)
            metrics_rows.append({"split": split_name, "block": block,
                                 "n_train": len(train), "n_test": len(test), **result})
            for index, estimate in zip(test, prediction):
                prediction_rows.append({
                    "split": split_name, "block": block, "row_id": rows[index]["row_id"],
                    "circuit": rows[index]["circuit"], "backend": rows[index]["backend"],
                    "target_seconds": target[index], "prediction_seconds": estimate,
                    "absolute_error_seconds": abs(float(estimate) - target[index]),
                })

    def pool(prefix: str) -> dict[str, dict[str, float]]:
        out = {}
        for block in BLOCKS:
            chosen = [r for r in prediction_rows if r["block"] == block and str(r["split"]).startswith(prefix)]
            out[block] = metrics(np.asarray([float(r["target_seconds"]) for r in chosen]),
                                 np.asarray([float(r["prediction_seconds"]) for r in chosen]))
        return out

    output = args.artifact_dir / "feature_ablation"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0]))
        writer.writeheader(); writer.writerows(metrics_rows)
    with (output / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader(); writer.writerows(prediction_rows)
    summary = {
        "n_rows": len(rows), "n_qasm_hashes": len({r["qasm_sha256"] for r in rows}),
        "feature_blocks": {key: list(value) for key, value in BLOCKS.items()},
        "target_semantics": "Ma-Li observed result.time_taken seconds; 1024 shots; queue excluded",
        "preprocessing": "log1p non-fraction features; train-only standardization; ridge on log1p target; backend indicator",
        "pooled_metrics": {
            "grouped_qasm": pool("grouped"), "strict_all": pool("strict"), "family_all": pool("family")
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Ma--Li compact feature ablation",
        "",
        "This table attributes the improvement over the one-feature QCRE proxy.",
        "Every block uses the same outer folds, train-only preprocessing and",
        "observed `result.time_taken` target. It is an ablation, not evidence that",
        "the current FakeBackend timing is historical hardware truth.",
        "",
        "| Evaluation | Block | MAE (s) | log-R² | seconds-R² |",
        "|---|---|---:|---:|---:|",
    ]
    for evaluation, key in (("grouped-QASM", "grouped_qasm"), ("strict backend+QASM", "strict_all"), ("family held-out", "family_all")):
        for block, value in summary["pooled_metrics"][key].items():
            report.append(f"| {evaluation} | `{block}` | {value['mae_seconds']:.4f} | {value['r2_log1p_seconds']:.4f} | {value['r2_seconds']:.4f} |")
    report += [
        "", "## Reading the result", "",
        "`compiled` isolates depth and active width; `graph_size` adds native node/edge counts;",
        "`layer`, `timing`, `edge` and `opcode` add one interpretable block at a time.",
        "`rich_summary` is the all-descriptor reference. A gain is meaningful only if it",
        "survives grouped and strict splits; family-held-out values are a stress test.",
        "QWalk is inspected separately because it has only two rows.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
