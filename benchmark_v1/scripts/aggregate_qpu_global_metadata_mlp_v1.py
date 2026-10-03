#!/usr/bin/env python3
"""Validate and aggregate the frozen five-fold QPU metadata-MLP OOF run.

The paired primary comparison is restricted to the 8,766 IDs also scored by
Graph V3-large. Source slices and a source-balanced diagnostic are emitted
separately; the single row4477 prediction is retained as supplementary only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
OUTER = ROOT / "artifacts/benchmark_v2/real_qpu/unified_outer_split_v2.csv"
GRAPH = ROOT / "artifacts/benchmark_v3/real_qpu/unified_graph_v3_large_oof_20260930/predictions.csv"
PROTOCOL = ROOT / "benchmark_v1/protocol/qpu_global_metadata_mlp_baseline_v1.json"
SEED_REGISTRY = ROOT / "benchmark_v1/registry/seed_registry.json"
METHOD_ID = "qpu_global_metadata_mlp_baseline_v1"
GRAPH_ID = "mali_graph_architecture_v3_large"
ROW4477 = "qonductor_single_circuit_ibm|row4477"
EXPECTED_FOLD_IDS = {0: 1778, 1: 2206, 2: 1547, 3: 1467, 4: 1769}
BOOTSTRAP_REPLICATES = 10_000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def metric(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    if not actual:
        return {"n": 0, "mae_seconds": "", "medae_seconds": "", "log1p_mae": "",
                "r2_seconds": "", "p90_abs_error_seconds": "", "p99_abs_error_seconds": "",
                "max_abs_error_seconds": ""}
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    e = np.abs(a - p)
    denom = float(np.square(a - a.mean()).sum())
    r2 = "" if denom == 0 else 1.0 - float(np.square(a - p).sum()) / denom
    return {"n": len(a), "mae_seconds": float(e.mean()), "medae_seconds": float(np.median(e)),
            "log1p_mae": float(np.mean(np.abs(np.log1p(np.maximum(a, 0)) - np.log1p(np.maximum(p, 0))))),
            "r2_seconds": r2, "p90_abs_error_seconds": float(np.quantile(e, 0.90)),
            "p99_abs_error_seconds": float(np.quantile(e, 0.99)),
            "max_abs_error_seconds": float(e.max())}


def cluster_id(row: dict[str, str]) -> str:
    source = row["source_id"]
    if source == "mali_real_qpu":
        value = row.get("circuit_group_hash", "")
        kind = "exact_logical_qasm"
    elif source == "qonductor_single_circuit_ibm":
        value = row.get("qasm_bytes_sha256", "")
        kind = "exact_submitted_qasm"
    elif source == "qpack_mcp":
        value = row.get("workflow_group_hash", "")
        kind = "optimizer_workflow"
    else:
        raise ValueError(f"unrecognized source_id={source!r}")
    if not value:
        raise ValueError(f"missing bootstrap cluster identity for {row['canonical_row_id']}")
    return f"{source}|{kind}|{value}"


def bootstrap_delta(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Cluster bootstrap candidate-minus-graph row-weighted MAE difference."""
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        groups[row["cluster_id"]].append((row["candidate_abs_error"], row["graph_abs_error"]))
    keys = sorted(groups)
    if not keys:
        return {"bootstrap_clusters": 0, "observed_delta_seconds": "",
                "bootstrap_mean_delta_seconds": "", "bootstrap_ci_low_seconds": "",
                "bootstrap_ci_high_seconds": ""}
    observed = mean(a - b for values in groups.values() for a, b in values)
    rng = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    candidate_sums = np.asarray([sum(value[0] for value in groups[key]) for key in keys], dtype=np.float64)
    graph_sums = np.asarray([sum(value[1] for value in groups[key]) for key in keys], dtype=np.float64)
    row_counts = np.asarray([len(groups[key]) for key in keys], dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(keys), size=len(keys))
        row_count = float(row_counts[sampled].sum())
        deltas[index] = candidate_sums[sampled].sum() / row_count - graph_sums[sampled].sum() / row_count
    return {"bootstrap_clusters": len(keys), "observed_delta_seconds": observed,
            "bootstrap_mean_delta_seconds": float(deltas.mean()),
            "bootstrap_ci_low_seconds": float(np.quantile(deltas, 0.025)),
            "bootstrap_ci_high_seconds": float(np.quantile(deltas, 0.975))}


def validate_fold(path: Path, fold: int, canonical: dict[str, dict[str, str]],
                  outer: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    manifest_path = path / "run_manifest.json"
    predictions_path = path / "predictions.csv"
    attempts_path = path / "attempts.csv"
    summary_path = path / "summary.json"
    required = (manifest_path, predictions_path, attempts_path, summary_path)
    if any(not file.is_file() for file in required):
        raise ValueError(f"fold {fold} output incomplete: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("fit_completed") is not True:
        raise ValueError(f"fold {fold} training manifest is not PASS")
    if (manifest.get("method_id") != METHOD_ID or manifest.get("outer_fold") != fold
            or manifest.get("cuda_smoke") != "PASS"
            or not manifest.get("compute_lock", {}).get("shared_timing_and_host_gpu_lease_held_for_cuda_operations")):
        raise ValueError(f"fold {fold} method/CUDA/lock receipt mismatch")
    for file in (predictions_path, attempts_path, summary_path):
        if manifest.get("outputs", {}).get(file.name) != sha(file):
            raise ValueError(f"fold {fold} output hash mismatch: {file.name}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("outer_fold", -1)) != fold or summary.get("group_overlap_count") != 0:
        raise ValueError(f"fold {fold} split/leakage summary invalid")
    rows = read_csv(predictions_path)
    actual = {row["canonical_observation_id"]: row for row in rows}
    attempt_rows = read_csv(attempts_path)
    attempts = {row["canonical_observation_id"]: row for row in attempt_rows}
    expected = {cid for cid, row in outer.items() if int(row["outer_fold"]) == fold}
    if (len(rows) != EXPECTED_FOLD_IDS[fold] or set(actual) != expected
            or len(attempt_rows) != len(expected) or set(attempts) != expected):
        raise ValueError(f"fold {fold} ID envelope mismatch: {len(actual)} != {len(expected)}")
    checkpoint_dir = path / "checkpoints"
    expected_checkpoints = {f"seed_{seed}_epoch_{epoch:04d}.pt"
                            for seed in (42, 1234, 31415) for epoch in range(50, 501, 50)}
    checkpoints = {file.name: file for file in checkpoint_dir.glob("seed_*_epoch_*.pt")}
    if set(checkpoints) != expected_checkpoints:
        raise ValueError(f"fold {fold} checkpoint set is incomplete")
    for name, file in checkpoints.items():
        if manifest.get("outputs", {}).get(str(file.relative_to(path))) != sha(file):
            raise ValueError(f"fold {fold} checkpoint hash mismatch: {name}")
    for cid, row in actual.items():
        target = float(canonical[cid]["target_seconds"])
        if float(row["actual_seconds"]) != target:
            raise ValueError(f"fold {fold} exact target label mismatch: {cid}")
        values = [float(row[f"predicted_seed_{seed}_seconds"]) for seed in (42, 1234, 31415)]
        prediction = float(row["predicted_seconds"])
        if not all(math.isfinite(value) and value >= 0 for value in [prediction, *values]):
            raise ValueError(f"fold {fold} non-finite/negative prediction: {cid}")
        if not math.isclose(prediction, median(values), rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"fold {fold} does not use the registered three-seed median: {cid}")
        if row["source_id"] != canonical[cid]["source_id"] or int(row["outer_fold"]) != fold:
            raise ValueError(f"fold {fold} identity metadata mismatch: {cid}")
        if attempts[cid]["status"] != ("predicted_supplementary" if cid == ROW4477 else "predicted"):
            raise ValueError(f"fold {fold} terminal status mismatch: {cid}")
    return actual


def bootstrap_seed(registry: dict[str, Any], label: str) -> int:
    material = (f"{registry['root_seed']}|{registry['streams']['bootstrap']}|"
                f"qpu-global-metadata-mlp-v1|{label}|0").encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def aggregate(fold0: Path, run_dir: Path, output: Path,
              canonical_path: Path = CANONICAL, outer_path: Path = OUTER,
              graph_path: Path = GRAPH) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    canonical_rows, outer_rows = read_csv(canonical_path), read_csv(outer_path)
    canonical = {row["canonical_row_id"]: row for row in canonical_rows}
    outer = {row["canonical_observation_id"]: row for row in outer_rows}
    if len(canonical) != 8767 or set(canonical) != set(outer):
        raise ValueError("canonical/outer source identity must be exactly 8,767 rows")
    if sum(int(row["source_id"] == "mali_real_qpu") for row in canonical.values()) != 340:
        raise ValueError("unexpected Ma-Li source count")
    if sum(int(row["source_id"] == "qonductor_single_circuit_ibm") for row in canonical.values()) != 4482:
        raise ValueError("unexpected Qonductor source count")
    if sum(int(row["source_id"] == "qpack_mcp") for row in canonical.values()) != 3945:
        raise ValueError("unexpected QPack source count")

    selected: dict[int, Path] = {0: fold0}
    selected.update({fold: run_dir / f"fold_{fold}" for fold in range(1, 5)})
    mlp: dict[str, dict[str, str]] = {}
    for fold, folder in selected.items():
        for cid, row in validate_fold(folder, fold, canonical, outer).items():
            if cid in mlp:
                raise ValueError(f"duplicate OOF prediction across folds: {cid}")
            mlp[cid] = row
    if set(mlp) != set(canonical):
        raise ValueError(f"MLP OOF coverage mismatch: {len(mlp)}/8767")
    assigned_primary = set(mlp) - {ROW4477}
    if len(assigned_primary) != 8766 or ROW4477 not in mlp:
        raise ValueError("row4477 supplementary/8,766 primary paired policy changed")

    graph_rows = read_csv(graph_path)
    graph = {row["canonical_observation_id"]: row for row in graph_rows}
    if len(graph) != 8766 or ROW4477 in graph:
        raise ValueError("Graph V3-large comparison envelope is not the frozen 8,766-row panel")
    for cid, row in graph.items():
        if cid not in canonical or int(row["outer_fold"]) != int(outer[cid]["outer_fold"]):
            raise ValueError(f"Graph V3-large identity/fold mismatch: {cid}")
        if not math.isclose(float(row["actual_seconds"]), float(canonical[cid]["target_seconds"]),
                            rel_tol=1e-6, abs_tol=1e-5):
            raise ValueError(f"Graph V3-large archived target mismatch: {cid}")
    paired = sorted(assigned_primary & set(graph))
    if len(paired) != 8766 or set(graph) != assigned_primary:
        raise ValueError("MLP and Graph V3-large shared row set differs from 8,766")
    pairs: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for cid in sorted(mlp):
        base, mlp_row = canonical[cid], mlp[cid]
        candidate = float(mlp_row["predicted_seconds"])
        actual = float(base["target_seconds"])
        graph_row = graph.get(cid)
        graph_prediction = float(graph_row["predicted_seconds"]) if graph_row else ""
        record = {"canonical_observation_id": cid, "source_id": base["source_id"],
                  "outer_fold": int(outer[cid]["outer_fold"]), "actual_seconds": actual,
                  "mlp_predicted_seconds": candidate,
                  "mlp_absolute_error_seconds": abs(candidate - actual),
                  "comparison_scope": "primary_paired" if cid != ROW4477 else "supplementary_single_case",
                  "graph_v3_large_predicted_seconds": graph_prediction,
                  "graph_v3_large_absolute_error_seconds": abs(graph_prediction - actual) if graph_row else "",
                  "cluster_id": cluster_id(base)}
        predictions.append(record)
        if graph_row:
            pairs.append({"canonical_observation_id": cid, "source_id": base["source_id"],
                          "cluster_id": cluster_id(base), "actual_seconds": actual,
                          "candidate_prediction": candidate, "graph_prediction": float(graph_prediction),
                          "candidate_abs_error": abs(candidate - actual),
                          "graph_abs_error": abs(float(graph_prediction) - actual)})

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    registry = json.loads(SEED_REGISTRY.read_text(encoding="utf-8"))
    all_metrics: list[dict[str, Any]] = []
    source_metrics: list[dict[str, Any]] = []
    for method_id, source_field in ((METHOD_ID, "mlp_predicted_seconds"),
                                    (GRAPH_ID, "graph_v3_large_predicted_seconds")):
        applicable = [row for row in predictions if row[source_field] != "" and row["comparison_scope"] == "primary_paired"]
        score = metric([float(row["actual_seconds"]) for row in applicable],
                       [float(row[source_field]) for row in applicable])
        all_metrics.append({"method_id": method_id, "reader_label": protocol["reader_label"] if method_id == METHOD_ID else GRAPH_ID,
                            "fidelity_class": protocol["fidelity_class"] if method_id == METHOD_ID else "adaptation",
                            "evaluation_target_clock": protocol["target_and_split"]["evaluation_target_clock"],
                            "method_output_clock": protocol["target_and_split"]["method_output_clock"] if method_id == METHOD_ID else "predicted_archived_observed_service_execution_time_seconds",
                            "claim_boundary": protocol["purpose"] if method_id == METHOD_ID else "Unified Ma-Li-style graph architecture adaptation; not the paper's native data/model split",
                            "assigned_primary_n": 8766, "predicted_primary_n": len(applicable),
                            "coverage_primary": len(applicable) / 8766, "metric_scope": "row-weighted diagnostic across three heterogeneous archived sources; source slices are primary interpretation",
                            **score})
        for source in sorted({row["source_id"] for row in canonical.values()}):
            part = [row for row in applicable if row["source_id"] == source]
            source_base = [row for row in canonical.values() if row["source_id"] == source and row["canonical_row_id"] != ROW4477]
            metrics = metric([float(row["actual_seconds"]) for row in part],
                             [float(row[source_field]) for row in part])
            source_metrics.append({"method_id": method_id, "source_id": source,
                                   "evaluation_target_clock": protocol["target_and_split"]["evaluation_target_clock"],
                                   "method_output_clock": protocol["target_and_split"]["method_output_clock"] if method_id == METHOD_ID else "predicted_archived_observed_service_execution_time_seconds",
                                   "assigned_primary_n": len(source_base), "predicted_n": len(part),
                                   "coverage": len(part) / len(source_base) if source_base else 0,
                                   "metric_scope": "source-stratified; QPack also clustered by workflow",
                                   **metrics})

    pair_rows: list[dict[str, Any]] = []
    for source in ["all_primary_rows", *sorted({row["source_id"] for row in pairs})]:
        part = pairs if source == "all_primary_rows" else [row for row in pairs if row["source_id"] == source]
        cand = metric([float(row["actual_seconds"]) for row in part],
                      [float(row["candidate_prediction"]) for row in part])
        ref = metric([float(row["actual_seconds"]) for row in part],
                     [float(row["graph_prediction"]) for row in part])
        boot = bootstrap_delta(part, bootstrap_seed(registry, source))
        row = {"candidate_method_id": METHOD_ID, "reference_method_id": GRAPH_ID,
               "source_id": source, "comparison_scope": "paired_same_heldout_rows",
               "evaluation_target_clock": protocol["target_and_split"]["evaluation_target_clock"],
               "candidate_output_clock": protocol["target_and_split"]["method_output_clock"],
               "reference_output_clock": "predicted_archived_observed_service_execution_time_seconds",
               "n_shared": len(part), "coverage_status": "symmetric",
               "candidate_mae_seconds": cand["mae_seconds"], "reference_mae_seconds": ref["mae_seconds"],
               "observed_delta_candidate_minus_reference_seconds": boot["observed_delta_seconds"],
               "bootstrap_mean_delta_seconds": boot["bootstrap_mean_delta_seconds"],
               "bootstrap_ci_low_seconds": boot["bootstrap_ci_low_seconds"],
               "bootstrap_ci_high_seconds": boot["bootstrap_ci_high_seconds"],
               "bootstrap_replicates": BOOTSTRAP_REPLICATES,
               "bootstrap_clusters": boot["bootstrap_clusters"],
               "claim_boundary": "pooled row-weighted diagnostic only" if source == "all_primary_rows" else "source-local paired comparison"}
        if source == "all_primary_rows":
            row["source_balanced_macro_candidate_mae_diagnostic"] = mean(
                float(item["mae_seconds"]) for item in source_metrics
                if item["method_id"] == METHOD_ID and item["mae_seconds"] != "")
            row["source_balanced_macro_reference_mae_diagnostic"] = mean(
                float(item["mae_seconds"]) for item in source_metrics
                if item["method_id"] == GRAPH_ID and item["mae_seconds"] != "")
        pair_rows.append(row)

    supplementary = mlp[ROW4477]
    supplemental = {"canonical_observation_id": ROW4477,
                    "actual_seconds": float(canonical[ROW4477]["target_seconds"]),
                    "predicted_seconds": float(supplementary["predicted_seconds"]),
                    "absolute_error_seconds": abs(float(supplementary["predicted_seconds"]) - float(canonical[ROW4477]["target_seconds"])),
                    "scope": "one supplementary retrospective metadata-only case; excluded from all primary MLP-versus-graph metrics"}

    output.mkdir(parents=True)
    write_csv(output / "oof_predictions.csv", predictions)
    write_csv(output / "method_metrics.csv", all_metrics)
    write_csv(output / "source_metrics.csv", source_metrics)
    write_csv(output / "paired_comparison.csv", pair_rows)
    (output / "row4477_supplementary.json").write_text(json.dumps(supplemental, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    inputs = [canonical_path, outer_path, graph_path, PROTOCOL, SEED_REGISTRY, Path(__file__).resolve(),
              *(path / "run_manifest.json" for path in selected.values()),
              *(path / "predictions.csv" for path in selected.values()),
              *(path / "attempts.csv" for path in selected.values())]
    result = {"artifact_id": "qpu-global-metadata-mlp-oof-v1", "status": "PASS",
              "assigned_n": 8767, "primary_paired_n": 8766, "supplementary_row4477_n": 1,
              "source_counts": {source: sum(row["source_id"] == source for row in canonical.values())
                                for source in sorted({row["source_id"] for row in canonical.values()})},
              "fold_directories": {str(fold): str(path) for fold, path in selected.items()},
              "primary_claim": "V3-large-matched global-metadata-only MLP versus Graph V3-large on identical 8,766 IDs; source-stratified metrics and paired grouped bootstrap; pooled row-weighted result is diagnostic only",
              "input_hashes": {str(path.resolve()): sha(path) for path in inputs},
              "output_hashes": {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["manifest_sha256"] = sha(output / "manifest.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold0-attempt", type=Path, required=True,
                        help="The independently validated fold-0 attempt; allows the rejected initial attempt to remain immutable")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing validated fold_1 through fold_4")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, default=CANONICAL)
    parser.add_argument("--outer", type=Path, default=OUTER)
    parser.add_argument("--graph", type=Path, default=GRAPH)
    args = parser.parse_args()
    result = aggregate(args.fold0_attempt.resolve(), args.run_dir.resolve(), args.output_dir.resolve(),
                       args.canonical.resolve(), args.outer.resolve(), args.graph.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
