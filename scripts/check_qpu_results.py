#!/usr/bin/env python3
"""Check published QPU aggregates, not the excluded per-observation evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"mali_real_qpu": 340, "qonductor_single_circuit_ibm": 230,
           "qpack_mcp": 3945}
TARGET = "archived_observed_service_execution_time"


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-11):
        raise ValueError(f"{label}: {actual} != {expected}")


def digest(value):
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid successful-row digest")


def check_metric(row, assigned=4515):
    n = int(row["predicted_rows"])
    if int(row["assigned_rows"]) != assigned or not 0 <= n <= assigned:
        raise ValueError("assigned/scored counts disagree")
    close(row["coverage"], n / assigned, "coverage")
    if n:
        digest(row["scored_ids_sha256"])
        for key in ("mae_seconds", "medae_seconds", "rmse_seconds", "r2_seconds",
                    "log1p_mae", "p90_absolute_error_seconds",
                    "p99_absolute_error_seconds", "max_absolute_error_seconds"):
            value = float(row[key])
            if not math.isfinite(value) or (key != "r2_seconds" and value < 0):
                raise ValueError(f"invalid {key}")
        if not 0 <= int(row["negative_prediction_count"]) <= n:
            raise ValueError("invalid negative-prediction count")
    elif any(row[key] for key in ("mae_seconds", "rmse_seconds", "r2_seconds")):
        raise ValueError("unavailable method was scored")


def check_pair(pair, methods):
    left, right = pair["left_method"], pair["right_method"]
    if left not in methods or right not in methods or left == right:
        raise ValueError("unknown/identical paired methods")
    n, groups = int(pair["n"]), int(pair["groups"])
    if not 0 < groups <= 165 or not groups <= n <= min(
            int(methods[left]["predicted_rows"]), int(methods[right]["predicted_rows"])):
        raise ValueError("paired counts disagree")
    digest(pair["ids_sha256"])
    close(pair["observed_mae_delta_seconds"],
          float(pair["left_mae_seconds_on_shared_rows"]) -
          float(pair["right_mae_seconds_on_shared_rows"]), "observed paired delta")
    if int(pair["bootstrap_seed"]) != 42 or int(pair["bootstrap_replicates"]) != 10000:
        raise ValueError("bootstrap settings disagree")
    if float(pair["bootstrap_ci_low_seconds"]) > float(pair["bootstrap_ci_high_seconds"]):
        raise ValueError("reversed interval")
    for side, method in (("left", left), ("right", right)):
        if n == int(methods[method]["predicted_rows"]):
            if pair["ids_sha256"] != methods[method]["scored_ids_sha256"]:
                raise ValueError("successful-row identity disagrees")
            close(pair[f"{side}_mae_seconds_on_shared_rows"], methods[method]["mae_seconds"], "same-row MAE")
            close(pair[f"{side}_r2_on_shared_rows"], methods[method]["r2_seconds"], "same-row R2")


def check(root=ROOT):
    root = root.resolve()
    table = root / "results/real_qpu"
    metrics = rows(table / "method_comparison.csv")
    methods = {row["method_id"]: row for row in metrics}
    if len(metrics) != 32 or len(methods) != 32:
        raise ValueError("expected 32 distinct variants")
    for method, row in methods.items():
        check_metric(row)
        if (row["evaluation_target_clock"] != TARGET or not row["method_output_clock"]
                or not row["reader_label"] or not row["claim_boundary"]
                or row["fidelity_class"] not in {"adaptation", "baseline", "analytical_proxy", "unavailable"}):
            raise ValueError("method identity/clock/qualification missing")
        if method == "scholten_original_paper": expected = 0
        elif method.startswith("scholten_unified_nominal_throughput"): expected = 3399
        elif method == "hyb_nominal_r2_log_cost_ridge_v1": expected = 4367
        elif method.startswith("hyb_nominal_r2_"): expected = 4175
        else: expected = 4515
        if int(row["predicted_rows"]) != expected:
            raise ValueError(f"support counts disagree: {method}")
    pairs = rows(table / "shared_row_comparisons.csv")
    keys = {frozenset((r["left_method"], r["right_method"])) for r in pairs}
    if len(pairs) != 465 or len(keys) != 465:
        raise ValueError("expected all 465 distinct successful-method pairs")
    for pair in pairs:
        check_pair(pair, methods)
    failures = Counter()
    for row in rows(table / "failure_counts.csv"):
        if row["method_id"] not in methods or row["source_id"] not in SOURCES or not row["reason"]:
            raise ValueError("invalid failure ledger")
        failures[row["method_id"]] += int(row["rows"])
    for method, row in methods.items():
        if failures[method] + int(row["predicted_rows"]) != 4515:
            raise ValueError(f"failure accounting disagrees: {method}")
    stratified = rows(table / "source_and_backend_metrics.csv")
    for method, overall in methods.items():
        for kind in ("source_id", "backend", "logical_input_tier"):
            slices = [r for r in stratified if r["method_id"] == method and r["slice_type"] == kind]
            if sum(int(r["assigned_rows"]) for r in slices) != 4515 or sum(
                    int(r["predicted_rows"]) for r in slices) != int(overall["predicted_rows"]):
                raise ValueError(f"slice accounting disagrees: {method}/{kind}")
            for row in slices:
                check_metric(row, int(row["assigned_rows"]))
        slices = [r for r in stratified if r["method_id"] == method and r["slice_type"] == "source_id"]
        if {r["slice_value"]: int(r["assigned_rows"]) for r in slices} != SOURCES:
            raise ValueError("source slices disagree")
        scored = [r for r in slices if int(r["predicted_rows"])]
        if scored:
            close(overall["mae_seconds"], sum(float(r["mae_seconds"]) * int(r["predicted_rows"])
                  for r in scored) / sum(int(r["predicted_rows"]) for r in scored), "pooled MAE")
        if len(scored) == 3:
            close(overall["source_balanced_mae_seconds"], sum(float(r["mae_seconds"])
                  for r in scored) / 3, "source-balanced MAE")
        elif overall["source_balanced_mae_seconds"]:
            raise ValueError("incomplete sources reported as a three-source macro")
    seeds = rows(table / "seed_metrics.csv")
    keys = {(r["method_id"], int(r["seed"]), r["source_slice"]) for r in seeds}
    expected = {(m, s, c) for m in ("mali_logical_graph", "mali_global_mlp")
                for s in (42, 1234, 31415) for c in ("all_sources", *SOURCES)}
    if len(seeds) != 24 or keys != expected:
        raise ValueError("seed registry coverage disagrees")
    for row in seeds:
        check_metric(row, 4515 if row["source_slice"] == "all_sources" else SOURCES[row["source_slice"]])
    profile = json.loads((table / "dataset_profile.json").read_text())
    if profile["rows"] != 4515 or profile["groups"] != 165 or profile["source_counts"] != SOURCES:
        raise ValueError("dataset profile disagrees")
    receipt = json.loads((root / "provenance/real_qpu.json").read_text())
    if receipt["private_validation"]["status"] != "PASS" or receipt["private_validation"]["neural_cells"] != 30:
        raise ValueError("private validation receipt disagrees")
    for entry in receipt["aggregate_source_files"] + receipt["producing_code"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != entry["source_sha256"]:
            raise ValueError(f"source hash mismatch: {entry['path']}")
    return {"variants": len(metrics), "paired_comparisons": len(pairs), "seed_rows": len(seeds)}


if __name__ == "__main__":
    result = check()
    print(f"Checked {result['variants']} QPU aggregate variants, {result['paired_comparisons']} pairs and {result['seed_rows']} seed rows; no per-row score recomputation")
