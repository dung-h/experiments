#!/usr/bin/env python3
"""Join frozen archived-QPU predictions and build source-aware diagnostics.

This is a derived aggregation only: no fitting, inference or experiment runs.
Every method is checked against canonical labels and the frozen outer fold.
The three source datasets have different target semantics, hardware, shots and
dependence structures. The full envelope is retained, but errors and pairs are
reported within source; this script does not emit a pooled cross-source score.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
OUTER = ROOT / "artifacts/benchmark_v2/real_qpu/unified_outer_split_v2.csv"
LEARNED = ROOT / "artifacts/benchmark_v3/real_qpu/unified_learned_methods_oof_v4r2/method_attempts.csv"
LARGE = ROOT / "artifacts/benchmark_v3/real_qpu/unified_graph_v3_large_oof_20260930/predictions.csv"
LARGE_ATTEMPTS = ROOT / "artifacts/benchmark_v3/real_qpu/unified_graph_v3_large_oof_20260930/method_attempts.csv"
ANALYTICAL = ROOT / "artifacts/benchmark_v3/real_qpu/archived_analytical_wave_v3_20260930/attempts.csv"
FIDELITY = ROOT / "benchmark_v1/registry/method_fidelity_registry_v2.json"
SEED_REGISTRY = ROOT / "benchmark_v1/registry/seed_registry.json"
TARGET_CLOCK = "archived_observed_service_execution_time"
PRIMARY_METHODS = (
    "mali_graph_architecture_v3",
    "mali_graph_architecture_v3_large",
    "unified_polynomial_v3",
    "qcre_snapshot_critical_path",
    "qcre_shot_scaled_schedule_seconds",
    "qiskit_estimate_duration_snapshot",
    "qiskit_shot_scaled_schedule_seconds",
    "hyb_hanas_one_circuit_effective_cost_adaptation",
    "hyb_hanas_shot_linear_effective_seconds",
    "scholten_nominal_single_circuit_adaptation",
)
def pair_references(available_methods: set[str]) -> tuple[tuple[str, str], ...]:
    base = [(method, "unified_polynomial_v3") for method in PRIMARY_METHODS if method != "unified_polynomial_v3"]
    calibrated = sorted(
        method for method in available_methods
        if method.endswith("_outer_train_affine_seconds") or method.endswith("_outer_train_log_affine_seconds")
    )
    base.extend((method, "unified_polynomial_v3") for method in calibrated)
    base.append(("mali_graph_architecture_v3_large", "mali_graph_architecture_v3"))
    return tuple(base)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    fields = list(fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity_hash(ids: list[str]) -> str:
    payload = "\n".join(sorted(ids)) + f"\ncount={len(ids)}\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def bootstrap_seed(label: str, registry: dict[str, object]) -> int:
    stream = registry["streams"]["bootstrap"]
    material = f"{registry['root_seed']}|{stream}|unified-qpu-method-comparisons-v2|{label}|0"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def metric(actual: list[float], predicted: list[float]) -> dict[str, object]:
    if not actual:
        return {"n": 0, "mae_seconds": "", "medae_seconds": "", "log1p_mae": "", "r2_seconds": "",
                "r2_status": "not_scored_no_rows", "p90_abs_error_seconds": "",
                "p99_abs_error_seconds": "", "max_abs_error_seconds": ""}
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    error = np.abs(a - p)
    denom = float(np.square(a - a.mean()).sum())
    with np.errstate(over="ignore", invalid="ignore"):
        sse = float(np.square(a - p).sum())
    if not denom:
        r2, r2_status = "", "undefined_constant_actual"
    elif not math.isfinite(sse):
        r2, r2_status = "", "score_overflow"
    else:
        r2, r2_status = 1.0 - sse / denom, "finite_score"
    return {
        "n": len(a),
        "mae_seconds": float(error.mean()),
        "medae_seconds": float(np.median(error)),
        "log1p_mae": float(np.mean(np.abs(np.log1p(a) - np.log1p(p)))),
        "r2_seconds": r2,
        "r2_status": r2_status,
        "p90_abs_error_seconds": float(np.quantile(error, .9)),
        "p99_abs_error_seconds": float(np.quantile(error, .99)),
        "max_abs_error_seconds": float(error.max()),
    }


def cluster_id(row: dict[str, str]) -> tuple[str, str]:
    source = row["source_id"]
    if source == "mali_real_qpu":
        value = row["circuit_group_hash"]
        kind = "exact_logical_qasm"
    elif source == "qonductor_single_circuit_ibm":
        value = row["qasm_bytes_sha256"]
        kind = "exact_submitted_qasm"
    elif source == "qpack_mcp":
        value = row["workflow_group_hash"]
        kind = "optimizer_workflow"
    else:
        raise ValueError(f"unknown source: {source}")
    if not value:
        raise ValueError(f"missing cluster identity for {row['canonical_row_id']}")
    return kind, value


def grouped_bootstrap(
    paired: list[tuple[str, float, float, float, str]], seed: int, replicates: int = 1000
) -> dict[str, object]:
    """Return a cluster-bootstrap interval for candidate-minus-reference MAE."""
    groups: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for key, candidate_error, reference_error, candidate_win, _source in paired:
        groups[key].append((candidate_error, reference_error, candidate_win))
    keys = sorted(groups)
    counts = np.asarray([len(groups[key]) for key in keys], dtype=np.int64)
    cand = np.asarray([sum(item[0] for item in groups[key]) for key in keys], dtype=float)
    ref = np.asarray([sum(item[1] for item in groups[key]) for key in keys], dtype=float)
    wins = np.asarray([sum(item[2] for item in groups[key]) for key in keys], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(keys), size=(replicates, len(keys)))
    denom = counts[sampled].sum(axis=1)
    delta = (cand[sampled].sum(axis=1) - ref[sampled].sum(axis=1)) / denom
    win_rate = wins[sampled].sum(axis=1) / denom
    return {
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "bootstrap_clusters": len(keys),
        "bootstrap_pooled_delta_ci_low": float(np.quantile(delta, .025)),
        "bootstrap_pooled_delta_ci_high": float(np.quantile(delta, .975)),
        "bootstrap_pooled_win_rate_ci_low": float(np.quantile(win_rate, .025)),
        "bootstrap_pooled_win_rate_ci_high": float(np.quantile(win_rate, .975)),
        "bootstrap_interpretation": "row-weighted source-local diagnostic",
    }


def load_predictions(
    canonical: dict[str, dict[str, str]],
    outer: dict[str, dict[str, str]],
    analytical_path: Path,
) -> dict[str, dict[str, dict[str, object]]]:
    predictions: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)

    def add(method: str, cid: str, fold: str, actual: str, pred: str, status: str, reason: str, output_clock: str) -> None:
        if cid not in canonical or cid not in outer:
            raise ValueError(f"unknown observation ID {cid} ({method})")
        if int(fold) != int(outer[cid]["outer_fold"]):
            raise ValueError(f"outer-fold mismatch: {method} {cid}")
        if not math.isclose(float(actual), float(canonical[cid]["target_seconds"]), rel_tol=1e-6, abs_tol=1e-5):
            raise ValueError(f"target mismatch: {method} {cid}")
        if cid in predictions[method]:
            raise ValueError(f"duplicate prediction/attempt: {method} {cid}")
        value: float | None = None
        normalized = "predicted"
        if status in {"prediction_produced", "predicted", "ok"}:
            if pred in ("", None):
                normalized = "missing_prediction_value"
                reason = reason or "success status without a numeric prediction"
            else:
                try:
                    value = float(pred)
                except (TypeError, ValueError):
                    normalized = "invalid_prediction_value"
                    reason = reason or "prediction is not numeric"
                    value = None
                if value is not None and not math.isfinite(value):
                    normalized = "nonfinite_prediction"
                    value = None
                elif value is not None and value < 0:
                    normalized = "negative_runtime_prediction"
                    reason = reason or "runtime prediction is negative seconds"
                    value = None
        else:
            normalized = status or "unavailable"
        predictions[method][cid] = {
            "prediction_seconds": value,
            "raw_prediction_seconds": "" if pred is None else pred,
            "status": normalized,
            "reason": reason,
            "method_output_clock": output_clock,
        }

    for row in read_csv(LEARNED):
        add(row["method_id"], row["canonical_observation_id"], row["outer_fold"], row["actual_seconds"],
            row["predicted_seconds"], row["attempt_status"], row["no_prediction_reason"], TARGET_CLOCK)

    for row in read_csv(LARGE_ATTEMPTS):
        add(row["method_id"], row["canonical_observation_id"], row["outer_fold"], row["actual_seconds"],
            row["predicted_seconds"], row["status"], row["reason"], TARGET_CLOCK)

    analytical = read_csv(analytical_path)
    for row in analytical:
        if row["method_id"] not in PRIMARY_METHODS and "_outer_train_" not in row["method_id"]:
            continue
        add(row["method_id"], row["canonical_row_id"], row["outer_fold"], row["observed_target_seconds"],
            row["prediction_seconds"], row["status"], row["terminal_reason"], row["method_output_clock"])
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytical-attempts", type=Path, default=ANALYTICAL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hyb-status", choices=("historical", "repaired"), default="historical")
    parser.add_argument("--hyb-repair-manifest", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output_dir}")
    canonical_rows = read_csv(CANONICAL)
    canonical = {row["canonical_row_id"]: row for row in canonical_rows}
    outer_rows = read_csv(OUTER)
    outer = {row["canonical_observation_id"]: row for row in outer_rows}
    if (len(canonical_rows) != 8767 or len(canonical) != 8767
            or len(outer_rows) != 8767 or len(outer) != 8767 or set(canonical) != set(outer)):
        raise ValueError("canonical ledger and outer split identities are not exactly the same 8,767 rows")
    expected_source_counts = {
        "mali_real_qpu": 340,
        "qonductor_single_circuit_ibm": 4482,
        "qpack_mcp": 3945,
    }
    if dict(Counter(row["source_id"] for row in canonical.values())) != expected_source_counts:
        raise ValueError("canonical source denominators differ from the frozen 340/4,482/3,945 contract")
    cluster_folds: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cid, base in canonical.items():
        cluster_folds[cluster_id(base)].add(outer[cid]["outer_fold"])
    leaking_clusters = [key for key, folds in cluster_folds.items() if len(folds) != 1]
    if leaking_clusters:
        raise ValueError(f"frozen outer split leaks {len(leaking_clusters)} bootstrap clusters")
    fidelity = json.loads(FIDELITY.read_text(encoding="utf-8"))
    seed_registry = json.loads(SEED_REGISTRY.read_text(encoding="utf-8"))
    registry = {row["method_id"]: row for row in fidelity["methods"]}

    def card_fields(method: str, prefix: str = "") -> dict[str, object]:
        card = registry.get(method, {})
        return {
            f"{prefix}reader_label": card.get("reader_label", method),
            f"{prefix}fidelity_class": card.get("fidelity_class", "unregistered"),
            f"{prefix}claim_boundary": card.get("claim_boundary", "method card absent"),
            f"{prefix}ranking_eligibility": card.get("ranking_eligibility", "not_declared"),
            f"{prefix}equivalence_group": card.get("equivalence_group", "none_declared"),
        }

    repair_manifest = None
    if args.hyb_status == "repaired":
        if args.hyb_repair_manifest is None:
            raise ValueError("hyb-status=repaired requires --hyb-repair-manifest")
        repair_manifest = json.loads(args.hyb_repair_manifest.read_text(encoding="utf-8"))
        expected_merged_path = (ROOT / repair_manifest["merged_for_c4_path"]).resolve()
        if args.analytical_attempts.resolve() != expected_merged_path:
            raise ValueError("analytical attempts path does not match the PASS repair manifest")
        if (repair_manifest.get("status") != "PASS"
                or repair_manifest.get("canonical_n") != 8767
                or repair_manifest.get("canonical_identity_hash") != identity_hash(list(canonical))
                or repair_manifest.get("merged_attempts") != 21 * 8767
                or repair_manifest.get("merged_rows_per_method") != 8767
                or not repair_manifest.get("non_hyb_values_preserved")
                or repair_manifest.get("original_artifacts_mutated")):
            raise ValueError("C2 repair manifest does not satisfy the frozen C4 provenance gate")
    predictions = load_predictions(canonical, outer, args.analytical_attempts)
    historical_analytical_methods = {row["method_id"] for row in read_csv(ANALYTICAL)}
    expected_methods = (
        historical_analytical_methods
        | {row["method_id"] for row in read_csv(LEARNED)}
        | {row["method_id"] for row in read_csv(LARGE_ATTEMPTS)}
    )
    if set(predictions) != expected_methods:
        missing = sorted(expected_methods - set(predictions))
        unexpected = sorted(set(predictions) - expected_methods)
        raise ValueError(f"method coverage differs from frozen C4 set; missing={missing}; unexpected={unexpected}")
    for method, rows in predictions.items():
        if set(rows) != set(canonical):
            raise ValueError(f"method envelope mismatch {method}: {len(rows)} != 8767")

    attempt_fields = ["canonical_row_id", "source_id", "outer_fold", "method_id", "fidelity_class", "reader_label",
                      "claim_boundary", "evaluation_target_clock", "method_output_clock", "status", "terminal_reason",
                      "actual_seconds", "raw_prediction_seconds", "prediction_seconds", "absolute_error_seconds",
                      "cluster_kind", "cluster_id"]
    attempts: list[dict[str, object]] = []
    for method, rows in sorted(predictions.items()):
        card = registry.get(method, {})
        for cid in sorted(canonical):
            base = canonical[cid]
            row = rows[cid]
            pred = row["prediction_seconds"]
            attempts.append({
                "canonical_row_id": cid, "source_id": base["source_id"], "outer_fold": outer[cid]["outer_fold"],
                "method_id": method, "fidelity_class": card.get("fidelity_class", "unregistered"),
                "reader_label": card.get("reader_label", method), "claim_boundary": card.get("claim_boundary", "method card absent"),
                "evaluation_target_clock": TARGET_CLOCK, "method_output_clock": row["method_output_clock"],
                "status": row["status"], "terminal_reason": row["reason"],
                "actual_seconds": base["target_seconds"], "raw_prediction_seconds": row["raw_prediction_seconds"],
                "prediction_seconds": "" if pred is None else pred,
                "absolute_error_seconds": "" if pred is None else abs(float(base["target_seconds"]) - pred),
                "cluster_kind": cluster_id(base)[0], "cluster_id": cluster_id(base)[1],
            })

    by_method = {method: {row["canonical_row_id"]: row for row in attempts if row["method_id"] == method}
                 for method in predictions}
    method_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    source_maes: dict[str, dict[str, float]] = defaultdict(dict)
    for method in sorted(by_method):
        card = registry.get(method, {})
        rows = list(by_method[method].values())
        ok = [r for r in rows if r["prediction_seconds"] != ""]
        output_clocks = sorted({str(r["method_output_clock"]) for r in rows})
        method_rows.append({
            "method_id": method, "reader_label": card.get("reader_label", method),
            "fidelity_class": card.get("fidelity_class", "unregistered"),
            "claim_boundary": card.get("claim_boundary", "method card absent"),
            "evaluation_target_clock": TARGET_CLOCK, "method_output_clock": ";".join(output_clocks),
            "assigned_n": len(rows), "predicted_n": len(ok), "unavailable_n": len(rows) - len(ok),
            "coverage": len(ok) / len(rows), "row_set_hash": identity_hash([str(r["canonical_row_id"]) for r in ok]),
            "mae_seconds": "", "medae_seconds": "", "log1p_mae": "", "r2_seconds": "",
            "p90_abs_error_seconds": "", "p99_abs_error_seconds": "", "max_abs_error_seconds": "",
            "metric_scope": "coverage_only_across_heterogeneous_sources; see source_metrics.csv",
        })
        for source in sorted({str(r["source_id"]) for r in rows}):
            part = [r for r in ok if r["source_id"] == source]
            m = metric([float(r["actual_seconds"]) for r in part], [float(r["prediction_seconds"]) for r in part])
            source_bootstrap = {}
            if part:
                sample = []
                for r in part:
                    kind, key = cluster_id(canonical[str(r["canonical_row_id"])])
                    error = float(r["absolute_error_seconds"])
                    sample.append((f"{source}|{kind}|{key}", error, 0.0, 0.0, source))
                src_seed = bootstrap_seed(f"method|{method}|source|{source}", seed_registry)
                source_bootstrap = grouped_bootstrap(sample, src_seed)
            source_rows.append({"method_id": method, "source_id": source, "n": m["n"], "n_observations": m["n"],
                                **card_fields(method),
                                "assigned_n": sum(r["source_id"] == source for r in rows),
                                "coverage": len(part) / sum(r["source_id"] == source for r in rows), **m,
                                "mae_bootstrap_cluster_count": source_bootstrap.get("bootstrap_clusters", 0),
                                "mae_bootstrap_ci_low": source_bootstrap.get("bootstrap_pooled_delta_ci_low", ""),
                                "mae_bootstrap_ci_high": source_bootstrap.get("bootstrap_pooled_delta_ci_high", ""),
                                "workflow_count": "", "metric_scope": "row-weighted source slice"})
            if m["n"]:
                source_maes[method][source] = float(m["mae_seconds"])
            if source == "qpack_mcp":
                workflows: dict[str, list[float]] = defaultdict(list)
                for r in part:
                    workflows[str(r["cluster_id"])].append(float(r["absolute_error_seconds"]))
                workflow_maes = {workflow: mean(errors) for workflow, errors in workflows.items()}
                assigned_workflows = {
                    str(r["workflow_group_hash"]) for r in canonical.values()
                    if r["source_id"] == source
                }
                workflow_sample = [(workflow, value, 0.0, 0.0, source) for workflow, value in sorted(workflow_maes.items())]
                workflow_bootstrap = (
                    grouped_bootstrap(workflow_sample, bootstrap_seed(f"method|{method}|qpack_equal_workflow", seed_registry))
                    if workflow_sample else {}
                )
                source_rows.append({"method_id": method, "source_id": source, "n": len(workflows),
                                    "n_observations": len(part),
                                    **card_fields(method),
                                    "assigned_n": sum(r["source_id"] == source for r in rows),
                                    "coverage": len(part) / sum(r["source_id"] == source for r in rows),
                                    "mae_seconds": mean(workflow_maes.values()) if workflow_maes else "",
                                    "mae_bootstrap_cluster_count": len(workflows),
                                    "mae_bootstrap_ci_low": workflow_bootstrap.get("bootstrap_pooled_delta_ci_low", ""),
                                    "mae_bootstrap_ci_high": workflow_bootstrap.get("bootstrap_pooled_delta_ci_high", ""),
                                    "workflow_count": len(workflows),
                                    "workflow_assigned_n": len(assigned_workflows),
                                    "workflow_success_n": len(workflows),
                                    "workflow_coverage": len(workflows) / len(assigned_workflows) if assigned_workflows else 0,
                                    "metric_scope": "equal-weight mean over successful workflows; diagnostic"})
        present = source_maes[method]
        method_rows[-1]["source_balanced_macro_mae_diagnostic"] = mean(present.values()) if len(present) == 3 else ""
        method_rows[-1]["source_balanced_macro_sources"] = len(present)
        method_rows[-1]["source_balanced_macro_status"] = (
            "not_rankable_cross_source_target_heterogeneity" if len(present) == 3
            else "not_computed_missing_source"
        )

    # S45 forbids pooling method errors across heterogeneous data sources.
    source_pair_rows: list[dict[str, object]] = []
    all_sources = sorted({row["source_id"] for row in canonical.values()})
    for candidate, reference in pair_references(set(predictions)):
        if candidate not in by_method or reference not in by_method:
            continue
        left, right = by_method[candidate], by_method[reference]
        for source in all_sources:
            source_ids = {cid for cid, base in canonical.items() if base["source_id"] == source}
            success_left = {cid for cid in source_ids if left[cid]["prediction_seconds"] != ""}
            success_right = {cid for cid in source_ids if right[cid]["prediction_seconds"] != ""}
            shared = sorted(success_left & success_right)
            asymmetric = success_left != success_right
            actual = [float(canonical[cid]["target_seconds"]) for cid in shared]
            left_values = [float(left[cid]["prediction_seconds"]) for cid in shared]
            right_values = [float(right[cid]["prediction_seconds"]) for cid in shared]
            lm, rm = metric(actual, left_values), metric(actual, right_values)
            left_errors = [abs(y - pred) for y, pred in zip(actual, left_values)]
            right_errors = [abs(y - pred) for y, pred in zip(actual, right_values)]
            paired_delta = mean(a - b for a, b in zip(left_errors, right_errors)) if shared else ""
            grouped = []
            for cid, left_error, right_error in zip(shared, left_errors, right_errors):
                kind, cluster = cluster_id(canonical[cid])
                grouped.append((f"{source}|{kind}|{cluster}", left_error, right_error,
                                float(left_error < right_error), source))
            pair_label = f"pair|{candidate}|vs|{reference}|source|{source}"
            boot = grouped_bootstrap(grouped, bootstrap_seed(pair_label, seed_registry)) if grouped else {}
            clusters = int(boot.get("bootstrap_clusters", 0))
            row: dict[str, object] = {
                "candidate_method_id": candidate, "reference_method_id": reference,
                **card_fields(candidate, "candidate_"), **card_fields(reference, "reference_"),
                "source_id": source, "source_assigned_n": len(source_ids),
                "candidate_success_n": len(success_left), "reference_success_n": len(success_right),
                "shared_success_n": len(shared), "shared_row_set_hash": identity_hash(shared),
                "coverage_status": "coverage_asymmetric" if asymmetric else "coverage_symmetric",
                "claim_status": (
                    "descriptive_only_coverage_asymmetric" if asymmetric else
                    "micro_only_fewer_than_10_clusters" if clusters < 10 else
                    "source_local_paired_diagnostic_not_global_ranking"
                ),
                "evaluation_target_clock": TARGET_CLOCK,
                "candidate_output_clock": ";".join(sorted({str(r["method_output_clock"]) for r in left.values()})),
                "reference_output_clock": ";".join(sorted({str(r["method_output_clock"]) for r in right.values()})),
                "candidate_mae_shared": lm["mae_seconds"], "reference_mae_shared": rm["mae_seconds"],
                "candidate_medae_shared": lm["medae_seconds"], "reference_medae_shared": rm["medae_seconds"],
                "candidate_log1p_mae_shared": lm["log1p_mae"], "reference_log1p_mae_shared": rm["log1p_mae"],
                "candidate_p90_shared": lm["p90_abs_error_seconds"], "reference_p90_shared": rm["p90_abs_error_seconds"],
                "candidate_p99_shared": lm["p99_abs_error_seconds"], "reference_p99_shared": rm["p99_abs_error_seconds"],
                "candidate_max_abs_error_shared": lm["max_abs_error_seconds"],
                "reference_max_abs_error_shared": rm["max_abs_error_seconds"],
                "paired_mae_delta_candidate_minus_reference": paired_delta,
                "paired_medae_delta_candidate_minus_reference": (
                    float(lm["medae_seconds"]) - float(rm["medae_seconds"]) if shared else ""
                ),
                "paired_log1p_mae_delta_candidate_minus_reference": (
                    float(lm["log1p_mae"]) - float(rm["log1p_mae"]) if shared else ""
                ),
                "candidate_win_n": sum(a < b for a, b in zip(left_errors, right_errors)),
                "tie_n": sum(a == b for a, b in zip(left_errors, right_errors)),
                "candidate_loss_n": sum(a > b for a, b in zip(left_errors, right_errors)),
                "candidate_win_rate": sum(a < b for a, b in zip(left_errors, right_errors)) / len(shared)
                if shared else "",
                **boot,
            }
            if source == "qpack_mcp":
                assigned_workflows = {
                    str(base["workflow_group_hash"]) for base in canonical.values()
                    if base["source_id"] == source
                }
                candidate_workflows = {
                    str(canonical[cid]["workflow_group_hash"]) for cid in success_left
                }
                reference_workflows = {
                    str(canonical[cid]["workflow_group_hash"]) for cid in success_right
                }
                workflow_errors: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
                for cid, lerr, rerr in zip(shared, left_errors, right_errors):
                    workflow = str(canonical[cid]["workflow_group_hash"])
                    workflow_errors[workflow][0].append(lerr)
                    workflow_errors[workflow][1].append(rerr)
                workflow_maes = [(wid, mean(errors[0]), mean(errors[1]))
                                 for wid, errors in sorted(workflow_errors.items())]
                workflow_sample = [(wid, lmae, rmae, float(lmae < rmae), source)
                                   for wid, lmae, rmae in workflow_maes]
                workflow_boot = (
                    grouped_bootstrap(workflow_sample,
                                      bootstrap_seed(pair_label + "|equal_workflow", seed_registry))
                    if workflow_sample else {}
                )
                row.update({
                    "candidate_successful_workflow_n": len(candidate_workflows),
                    "reference_successful_workflow_n": len(reference_workflows),
                    "qpack_assigned_workflow_n": len(assigned_workflows),
                    "qpack_shared_workflow_n": len(workflow_maes),
                    "qpack_shared_workflow_coverage": len(workflow_maes) / len(assigned_workflows)
                    if assigned_workflows else 0,
                    "qpack_candidate_equal_workflow_mae": mean(x[1] for x in workflow_maes)
                    if workflow_maes else "",
                    "qpack_reference_equal_workflow_mae": mean(x[2] for x in workflow_maes)
                    if workflow_maes else "",
                    "qpack_equal_workflow_delta_candidate_minus_reference": mean(
                        x[1] - x[2] for x in workflow_maes
                    ) if workflow_maes else "",
                    "qpack_equal_workflow_bootstrap_clusters": workflow_boot.get("bootstrap_clusters", 0),
                    "qpack_equal_workflow_bootstrap_ci_low": workflow_boot.get(
                        "bootstrap_pooled_delta_ci_low", ""),
                    "qpack_equal_workflow_bootstrap_ci_high": workflow_boot.get(
                        "bootstrap_pooled_delta_ci_high", ""),
                })
            source_pair_rows.append(row)

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "method_metrics.csv", method_rows, list(method_rows[0]))
    write_csv(args.output_dir / "source_metrics.csv", source_rows, list(source_rows[0]))
    write_csv(args.output_dir / "pairwise_comparisons.csv", source_pair_rows, list(source_pair_rows[0]))
    write_csv(args.output_dir / "attempts.csv", attempts, attempt_fields)
    inputs = [CANONICAL, OUTER, LEARNED, LARGE_ATTEMPTS, args.analytical_attempts, FIDELITY, SEED_REGISTRY, Path(__file__).resolve()]
    if args.hyb_repair_manifest is not None:
        inputs.append(args.hyb_repair_manifest)
    manifest = {
        "artifact_id": "unified-qpu-method-comparisons-v2",
        "status": "PASS_WITH_HISTORICAL_HYB_PENDING" if args.hyb_status == "historical" else "PASS",
        "hyb_variant_status": args.hyb_status,
        "hyb_repair_manifest_sha256": sha(args.hyb_repair_manifest) if args.hyb_repair_manifest else "",
        "scope": "derived join and metric aggregation only; no fitting/training/inference/experiment",
        "assigned_n": len(canonical), "method_count": len(predictions),
        "pair_comparisons": len(source_pair_rows),
        "coverage_asymmetric_pairs": sum(row["coverage_status"] == "coverage_asymmetric" for row in source_pair_rows),
        "comparison_reference": "unified_polynomial_v3 for every candidate, plus V3-large versus V3; source-stratified only",
        "qpack_equal_workflow_mae_present": True,
        "cross_source_pooled_error_metrics": "not_computed",
        "source_balanced_macro": "diagnostic_not_rankable; emitted only when all three source slices have successful predictions",
        "s49_rule": "unequal source-local success sets are coverage_asymmetric and paired errors descriptive only",
        "bootstrap_stream": "20260925-bootstrap",
        "bootstrap_replicates": 1000,
        "bootstrap_seed_derivation": "sha256(root_seed|registered_bootstrap_stream|artifact_id|comparison_label|0) first 4 bytes big-endian",
        "bootstrap_percentile_rule": "numpy.quantile linear 0.025 and 0.975",
        "input_hashes": {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha(path) for path in inputs},
        "output_hashes": {path.name: sha(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "methods": len(predictions), "source_pairs": len(source_pair_rows),
                      "coverage_asymmetric_pairs": manifest["coverage_asymmetric_pairs"], "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
