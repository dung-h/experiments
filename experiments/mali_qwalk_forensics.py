#!/usr/bin/env python3
"""Diagnose the two-row Ma--Li QWalk tail without treating it as a score.

For each grouped/strict held-out fold, compare QWalk's compact physical
descriptors with the training distribution and report OOF errors from the
existing baselines plus the ordered-DAG pilot.  The output is deliberately
diagnostic: two rows cannot establish an independent generalisation metric.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_mali_physical_baselines import ROOT, read_csv


FEATURES = (
    "qcre_critical_path_seconds", "compiled_depth", "active_physical_width",
    "n_nodes", "n_edges", "rich_n_layers", "rich_duration_sum_dt",
    "rich_forward_max_dt", "rich_criticality_p90_dt", "rich_layer_nodes_max",
    "rich_quantum_edge_fraction",
)


def percentile(value: float, train: np.ndarray) -> float:
    if len(train) == 0:
        return float("nan")
    return float(100.0 * np.mean(train <= value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path,
                        default=ROOT / "artifacts/validation/mali_physical_dag_v1")
    parser.add_argument("--split-manifest", type=Path,
                        default=ROOT / "artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv")
    args = parser.parse_args()

    manifest = read_csv(args.artifact_dir / "manifest.csv")
    split_rows = read_csv(args.split_manifest)
    splits = {(r["qasm_sha256"], r["backend"]): r for r in split_rows}
    rich = {r["graph_id"]: r for r in read_csv(args.artifact_dir / "graph_summary_features.csv")}
    rows = []
    for row in manifest:
        merged = dict(row)
        merged.update({k: splits[(row["qasm_sha256"], row["backend"])][k] for k in ("group_fold", "family_component")})
        merged.update(rich[row["graph_id"]])
        rows.append(merged)

    prediction_paths = {
        "qcre": args.artifact_dir / "baseline_evaluation/oof_predictions.csv",
        "compiled": args.artifact_dir / "baseline_evaluation/oof_predictions.csv",
        "physical_graph": args.artifact_dir / "baseline_evaluation/oof_predictions.csv",
        "rich_summary": args.artifact_dir / "baseline_evaluation/oof_predictions.csv",
        "ordered_dag_gru": args.artifact_dir / "layer_dag_rnn/oof_predictions.csv",
    }
    pred_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for model, path in prediction_paths.items():
        if not path.exists():
            continue
        for item in read_csv(path):
            source_model = item.get("model", model)
            if model != "ordered_dag_gru" and source_model != model:
                continue
            pred_by_key[(item["split"], source_model, item["row_id"])] = item

    qwalk = [r for r in rows if "qwalk" in r["circuit"].lower()]
    if len(qwalk) != 2:
        raise ValueError(f"expected two QWalk rows, found {len(qwalk)}")
    output_rows = []
    comparison_rows = []
    for qrow in qwalk:
        q_index = rows.index(qrow)
        for mode in ("grouped", "strict"):
            if mode == "grouped":
                split_names = [f"grouped_qasm_fold_{qrow['group_fold']}"]
                train_mask = lambda r: r["group_fold"] != qrow["group_fold"]
            else:
                split_names = [f"strict_{qrow['backend']}_fold_{qrow['group_fold']}"]
                train_mask = lambda r: r["backend"] != qrow["backend"] and r["group_fold"] != qrow["group_fold"]
            train = [r for r in rows if train_mask(r)]
            for feature in FEATURES:
                qvalue = float(qrow[feature])
                train_values = np.asarray([float(r[feature]) for r in train])
                positive = np.maximum(train_values, 0.0)
                qlog = np.log1p(max(qvalue, 0.0))
                logs = np.log1p(positive)
                scale = float(np.std(logs)) or 1.0
                comparison_rows.append({
                    "row_id": qrow["row_id"], "backend": qrow["backend"], "mode": mode,
                    "feature": feature, "qwalk_value": qvalue,
                    "train_median": float(np.median(train_values)),
                    "train_p05": float(np.percentile(train_values, 5)),
                    "train_p95": float(np.percentile(train_values, 95)),
                    "qwalk_percentile": percentile(qvalue, train_values),
                    "log_distance_from_train_mean": float(abs(qlog - np.mean(logs)) / scale),
                })
            # Nearest training row in the same compact feature space. This is a
            # heuristic OOD diagnostic, not a learned uncertainty estimate.
            qvec = np.log1p(np.maximum(np.asarray([float(qrow[f]) for f in FEATURES]), 0.0))
            matrix = np.log1p(np.maximum(np.asarray([[float(r[f]) for f in FEATURES] for r in train]), 0.0))
            mean = matrix.mean(axis=0); std = matrix.std(axis=0); std[std == 0] = 1.0
            distances = np.sqrt(np.mean(((matrix - mean) / std - (qvec - mean) / std) ** 2, axis=1))
            nearest = train[int(np.argmin(distances))]
            split_name = split_names[0]
            output_rows.append({
                "row_id": qrow["row_id"], "backend": qrow["backend"], "mode": mode,
                "split": split_name, "target_seconds": qrow["target_seconds"],
                "nearest_train_row_id": nearest["row_id"],
                "nearest_train_circuit": nearest["circuit"],
                "nearest_train_backend": nearest["backend"],
                "nearest_standardized_rms_distance": float(np.min(distances)),
                "train_size": len(train),
            })
            for model in ("qcre", "compiled", "physical_graph", "rich_summary", "ordered_dag_gru"):
                lookup_model = "ordered_dag_gru" if model == "ordered_dag_gru" else model
                item = pred_by_key.get((split_name, lookup_model, qrow["row_id"]))
                if item:
                    output_rows[-1][f"{model}_prediction_seconds"] = item["prediction_seconds"]
                    output_rows[-1][f"{model}_absolute_error_seconds"] = item["absolute_error_seconds"]

    output = args.artifact_dir / "qwalk_forensics"
    output.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0])
    with (output / "qwalk_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(output_rows)
    fields = list(comparison_rows[0])
    with (output / "qwalk_feature_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(comparison_rows)

    summary = {
        "n_qwalk_rows": len(qwalk), "qasm_hashes": sorted({r["qasm_sha256"] for r in qwalk}),
        "qwalk_rows": [r["row_id"] for r in qwalk],
        "interpretation": "tail diagnostic only; two rows are not an independent test set",
        "features_compared": list(FEATURES),
        "outputs": {"rows": "qwalk_rows.csv", "feature_comparison": "qwalk_feature_comparison.csv"},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Ma--Li QWalk tail forensics",
        "",
        "QWalk has two observed rows: one Osaka and one Kyoto measurement of the",
        "same raw-QASM hash. It is therefore retained as a stress case, not scored",
        "as a standalone R² or treated as an iid circuit family.",
        "",
        "`qwalk_rows.csv` records the nearest training row in standardized compact",
        "physical-feature space and all available out-of-fold predictions.\n",
        "`qwalk_feature_comparison.csv` gives train p05/median/p95, percentile and",
        "standardized log-distance for the features most relevant to the tail.",
        "",
        "A high percentile or nearest-neighbour distance indicates that the model",
        "is extrapolating. Lower error from `rich_summary` or the ordered-DAG pilot",
        "would show feature coverage helps, but would not prove topology causality.",
        "If QWalk remains far outside train support, the correct remedy is explicit",
        "OOD uncertainty/coverage handling or more representative training data,",
        "not silently pooling the two rows into the headline metric.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
