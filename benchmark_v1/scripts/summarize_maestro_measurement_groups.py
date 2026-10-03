#!/usr/bin/env python3
"""Summarize OOF absolute error by the frozen terminal-measurement strata."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

PREDICTIONS = {
    "maestro_style_cpu_sv_component_adaptation": "pred_maestro_style_cpu_sv_component_adaptation_seconds",
    "outer_train_median": "pred_outer_train_median_seconds",
    "nested_grouped_ridge": "pred_nested_grouped_ridge_seconds",
    "source_dag_graph_adaptation_cuda": "pred_source_dag_graph_adaptation_cuda_seconds",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stratum(target: dict[str, str], prediction: dict[str, str]) -> str:
    if target["measurement_map_injective"].lower() != "true":
        return "repeated_measurement"
    n = int(prediction["normalized_width"])
    m = int(target["measured_qubit_count"])
    c = int(target["declared_classical_width"])
    if m == n and c == n:
        return "full_register"
    if m < n and c == m and int(target["measured_classical_extent"]) == m:
        return "partial_measurement"
    if m == n and c > n:
        return "wider_classical_register"
    return "other"


def summarize(target_path: Path, prediction_path: Path) -> list[dict[str, object]]:
    targets = read_csv(target_path)
    predictions = read_csv(prediction_path)
    by_hash = {r["source_qasm_sha256"]: r for r in targets}
    if len(by_hash) != len(targets) or len({r["source_qasm_sha256"] for r in predictions}) != len(predictions):
        raise ValueError("duplicate exact-QASM hash in targets or OOF predictions")
    if set(by_hash) != {r["source_qasm_sha256"] for r in predictions} or len(by_hash) != 150:
        raise ValueError("targets and predictions must contain the same frozen 150 hashes")
    groups: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = {}
    for pred in predictions:
        target = by_hash[pred["source_qasm_sha256"]]
        groups.setdefault(stratum(target, pred), []).append((target, pred))

    rows: list[dict[str, object]] = []
    for group in ("full_register", "partial_measurement", "wider_classical_register", "repeated_measurement", "other"):
        pair = groups.get(group, [])
        if not pair:
            continue
        observed = [(t, p) for t, p in pair if t["target_status"] == "runtime_observed"]
        for method, column in PREDICTIONS.items():
            scored = []
            for target, pred in observed:
                raw = pred.get(column, "").strip()
                if raw:
                    y, estimate = float(target["runtime_seconds"]), float(raw)
                    if not math.isfinite(y) or not math.isfinite(estimate):
                        raise ValueError("nonfinite target or prediction")
                    scored.append((y, estimate))
            errors = [abs(y - estimate) for y, estimate in scored]
            ordered = sorted(errors)
            rows.append({
                "measurement_group": group,
                "method_id": method,
                "assigned_hashes": len(pair),
                "observed_target_hashes": len(observed),
                "predictions_on_observed": len(scored),
                "coverage_of_assigned": len(scored) / len(pair),
                "mae_seconds": statistics.mean(errors) if errors else "",
                "medae_seconds": statistics.median(errors) if errors else "",
                "p90_abs_error_seconds": ordered[math.ceil(.90 * len(ordered)) - 1] if errors else "",
                "p99_abs_error_seconds": ordered[math.ceil(.99 * len(ordered)) - 1] if errors else "",
                "max_abs_error_seconds": max(errors) if errors else "",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-targets", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows = summarize(args.panel_targets, args.oof_predictions)
    args.output_dir.mkdir(parents=True)
    table = args.output_dir / "measurement_group_metrics.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = {
        "artifact_id": "maestro-terminal-measurement-group-metrics-v1",
        "status": "complete",
        "panel_targets_sha256": sha(args.panel_targets),
        "oof_predictions_sha256": sha(args.oof_predictions),
        "summarizer_sha256": sha(Path(__file__).resolve()),
        "measurement_group_metrics_sha256": sha(table),
        "groups": ["full_register", "partial_measurement", "wider_classical_register", "repeated_measurement"],
        "error_scope": "only runtime_observed targets with an available OOF prediction",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "metric_rows": len(rows), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
