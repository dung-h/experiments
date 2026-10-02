#!/usr/bin/env python3
"""Build the current two-domain reader tables from pinned benchmark outputs.

This builder creates one new semantic output directory. It never overwrites
historical reader packs, writes CURRENT, trains a model, starts CUDA, or runs a
simulator. Archived-QPU tables are copied from the hash-pinned C4 aggregation;
the separate C3a Aer graph diagnostic is recomputed from its frozen OOF rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v2"
C4 = ROOT / "artifacts/benchmark_v3/real_qpu/unified_method_comparisons_v2"
LEARNED = ROOT / "artifacts/benchmark_v3/real_qpu/unified_learned_derived_comparisons_v1"
CANONICAL = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
SIM_PACK = ROOT / "artifacts/benchmark_v3/simulator/simulator_reader_pack_v1"
AER = ROOT / "artifacts/benchmark_v1/aer_mali_graph_adaptation_v1_20261001"
PARTIAL_MAESTRO = ROOT / "benchmark_v1/execution/manifests/maestro_common_panel_completion_partial.json"
MAESTRO_V3 = ROOT / "artifacts/benchmark_v3/simulator/maestro_candidate_runtime_v3/calibration/run_manifest.json"
REGISTRY = ROOT / "benchmark_v1/registry/method_fidelity_registry_v2.json"
DEFAULT_REPORT = ROOT / "benchmark_v1/execution/manifests/report_finalization.json"
FINAL_OUT = ROOT / "artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3"
REBUILD_ROOT = ROOT / "work/repo_finalization/rebuilds"

PINS = {
    "c4_manifest": "91449edbe653017b9dd54f79d465ec136f3495c9306fa42a49c3a64e9d8d4d83",
    "canonical_observations": "920d745e03dd00e8155118d9323baea18ff865e4df2620f3a059bd5503a05ba4",
    "fidelity_registry_v2": "98f3460c8d9a0cbabb424555dd695cee573409ed65facdb60557d836f1750810",
    "learned_manifest": "60d18c8cc5fe8f3e31a2ae04fd1d88e76bb581ad337d6003bc52a51f7e2eee13",
    "simulator_pack_manifest": "07b59ac1dc7ffa16e37ae4f638f93c4331df8318a0d2277e9ba7649db4e54d91",
    "aer_training_manifest": "84de8ebb2a501ba09233261329b708371bf06be59afcda4a23f5231edd3a2082",
    "aer_overall_qa": "083dc30ae81dddd979b3b09535a36baf4f2ca621ebc4e31b8b66fd44e64b6649",
    "aer_oof_predictions": "02bacf27c3a7cecde07283eda6518eadb1e1bca79eeb14508e7a69e2bca320a6",
    "aer_core_metrics": "320cc482d6fdabfc4c214fc01c2d4c5bfcf719b19d175a09f9eaaeb6a3cab074",
    "partial_maestro_policy": "3675c80f370f36aafec0dda3aa6814d5348e5f942400166f046d74388a256f5b",
    "failed_maestro_v3": "9e60bb86d08c0db51f76d1b50cc03bde68f6b69d1a7aa0e5f34ec2089c9a30c6",
}

C4_FILES = ("method_metrics.csv", "source_metrics.csv", "pairwise_comparisons.csv")
LEARNED_FILES = ("own_success_metrics.csv", "pair_metrics.csv", "bootstrap.csv")
SIM_FILES = (
    "per_configuration_metrics.csv",
    "native_method_appendix.csv",
    "blocked_unavailable.csv",
    "availability_matrix.csv",
    "common_exact_qasm_panel.csv",
)
OUTPUTS = (
    "README.md",
    "REPORT.md",
    "qpu/archived_method_coverage.csv",
    "qpu/archived_method_source_metrics.csv",
    "qpu/archived_method_pairwise_comparisons.csv",
    "qpu/unified_learned_metrics.csv",
    "qpu/unified_learned_pairs.csv",
    "qpu/unified_learned_bootstrap.csv",
    "simulator/per_configuration_metrics.csv",
    "simulator/native_method_appendix.csv",
    "simulator/blocked_unavailable.csv",
    "simulator/availability_matrix.csv",
    "simulator/common_exact_qasm_panel.csv",
    "simulator/aer_mali_graph_c3a_diagnostic.csv",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to emit empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def require_pin(path: Path, expected: str, label: str) -> str:
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual}")
    return actual


def check_manifest_files(folder: Path, manifest: dict) -> None:
    output_map = manifest.get("output_hashes", manifest.get("outputs", {}))
    for name, expected in output_map.items():
        path = folder / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"manifest output hash mismatch: {path}")


def check_input_hashes(manifest: dict) -> None:
    for name, expected in manifest.get("input_hashes", {}).items():
        path = ROOT / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"manifest input hash mismatch: {name}")
    inputs = manifest.get("inputs", {})
    if isinstance(inputs, dict):
        for item in inputs.values():
            if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
                continue
            path = ROOT / item["path"]
            if not path.is_file() or sha(path) != item["sha256"]:
                raise ValueError(f"manifest input hash mismatch: {item['path']}")


def copy_file(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    return sha(destination)


def compute_metrics(rows: list[dict[str, str]]) -> dict[str, float | int]:
    actual = [float(row["actual_seconds"]) for row in rows]
    predicted = [float(row["predicted_seconds"]) for row in rows]
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    if not errors:
        raise ValueError("cannot compute metrics for empty prediction set")
    log_errors = [abs(math.log1p(a) - math.log1p(p)) for a, p in zip(actual, predicted)]
    center = mean(actual)
    denominator = sum((value - center) ** 2 for value in actual)
    r2 = 1.0 - sum((a - p) ** 2 for a, p in zip(actual, predicted)) / denominator if denominator else float("nan")
    ordered = sorted(errors)

    def quantile(q: float) -> float:
        position = (len(ordered) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "n": len(rows),
        "mae_seconds": mean(errors),
        "medae_seconds": median(errors),
        "mae_log1p_seconds": mean(log_errors),
        "r2_seconds": r2,
        "p90_absolute_error_seconds": quantile(0.90),
        "p99_absolute_error_seconds": quantile(0.99),
        "max_absolute_error_seconds": max(errors),
    }


def aer_diagnostic() -> tuple[list[dict[str, object]], dict[str, object]]:
    prediction_path = AER / "core_oof_predictions.csv"
    metrics_path = AER / "core_metrics.csv"
    qa_path = AER / "overall_qa.json"
    training_path = AER / "training_manifest.json"
    for name, path in (
        ("aer_training_manifest", training_path),
        ("aer_overall_qa", qa_path),
        ("aer_oof_predictions", prediction_path),
        ("aer_core_metrics", metrics_path),
    ):
        require_pin(path, PINS[name], name)
    training = read_json(training_path)
    qa = read_json(qa_path)
    if training.get("status") != "PASS" or qa.get("status") != "PASS":
        raise ValueError("C3a training/QA status is not PASS")
    predictions = read_csv(prediction_path)
    if len(predictions) != 162 or len({r["observation_id"] for r in predictions}) != 162:
        raise ValueError("C3a must retain exactly 162 unique core OOF rows")
    if len({r["source_sha256"] for r in predictions}) != 150:
        raise ValueError("C3a QASM-hash count differs from frozen QA")
    if {int(r["fold"]) for r in predictions} != set(range(5)):
        raise ValueError("C3a fold coverage mismatch")
    for row in predictions:
        actual, predicted = float(row["actual_seconds"]), float(row["predicted_seconds"])
        if actual < 0 or predicted < 0 or not math.isfinite(actual) or not math.isfinite(predicted):
            raise ValueError(f"C3a contains an invalid runtime row: {row['observation_id']}")
        if not math.isclose(abs(actual - predicted), float(row["absolute_error_seconds"]), rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"C3a absolute error mismatch: {row['observation_id']}")
    overall = compute_metrics(predictions)
    published = next(r for r in read_csv(metrics_path) if r["scope"] == "all_core_oof")
    for field, key in (
        ("mae_seconds", "mae_seconds"),
        ("medae_seconds", "medae_seconds"),
        ("mae_log1p_seconds", "mae_log1p_seconds"),
        ("r2_seconds", "r2_seconds"),
        ("p90_absolute_error_seconds", "p90_absolute_error_seconds"),
        ("p99_absolute_error_seconds", "p99_absolute_error_seconds"),
        ("max_absolute_error_seconds", "max_absolute_error_seconds"),
    ):
        published_field = "mae_log1p_seconds" if field == "mae_log1p_seconds" else field
        if not math.isclose(float(published[published_field]), float(overall[key]), rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"C3a raw-derived metric differs from pinned core metrics: {published_field}")
    row = {
        "method_id": "c3a_mali_style_aer_graph_local_diagnostic",
        "reader_label": "S71-style Ma-Li graph + global-feature adaptation on local Aer warm labels",
        "registry_method_id": "",
        "fidelity_class": "",
        "registry_note": "No method ID/card exists for this C3a diagnostic in fidelity registry v2; do not treat as a registered paper-method comparison.",
        "claim_boundary": "Local C3a architecture adaptation; not original Ma-Li, not Azizov GNN, and not a full 204-row panel method.",
        "evaluation_target_clock": "warm_execution",
        "method_output_clock": "warm_execution",
        "population": "sim_common_q16_v1_core_only",
        "unique_qasm_hashes": len({r["source_sha256"] for r in predictions}),
        **overall,
        "frontier_rows_in_accuracy": 0,
        "quality_policy": "not an approximation-quality-constrained simulator; Aer observed warm runtime labels",
        "training_manifest_sha256": PINS["aer_training_manifest"],
        "overall_qa_sha256": PINS["aer_overall_qa"],
        "prediction_sha256": PINS["aer_oof_predictions"],
        "metrics_recomputed_from_oof_rows": True,
    }
    fold_rows = []
    for fold in range(5):
        fold_set = [r for r in predictions if int(r["fold"]) == fold]
        fold_rows.append({"fold": fold, **compute_metrics(fold_set)})
    return [row], {"fold_rows": fold_rows, "status": qa["status"]}


def load_report_contract(path: Path) -> dict:
    """Validate the frozen report authority, not an experiment authorization."""
    contract = read_json(path)
    if (contract.get("schema_version") != "report_finalization_v1"
            or contract.get("status") != "LOCKED_FOR_REPORT"
            or contract.get("scientific_coverage") != "PARTIAL"
            or contract.get("experiment_or_training_authorized") is not False
            or contract.get("public_release_approved") is not False
            or contract.get("wave4_five_fold_training_authorized") is not False):
        raise ValueError("report contract status/scope/authority mismatch")
    if contract.get("final_output_path") != rel(FINAL_OUT):
        raise ValueError("report contract final output path mismatch")
    if contract.get("supported_rebuild_root") != "work/repo_finalization/rebuilds":
        raise ValueError("report contract rebuild root mismatch")
    if (contract["real_qpu"]["canonical_n"] != 8767
            or contract["real_qpu"]["outer_folds"] != 5
            or contract["real_qpu"]["inner_folds"] != 4
            or contract["real_qpu"]["groups"] != 2320
            or contract["real_qpu"]["evaluation_target_clock"] != "archived_observed_service_execution_time"
            or contract["real_qpu"]["unified_source_slices_are_same_oof_predictions"] is not True):
        raise ValueError("report QPU evaluation contract mismatch")
    if (Counter(row["domain"] for row in contract["method_scope"]) != {"real_qpu": 6, "simulator": 6}
            or any("method_output_clock" not in row or "claim_boundary" not in row for row in contract["method_scope"])):
        raise ValueError("report original approach/clock coverage mismatch")
    for name, expected in contract["source_hashes"].items():
        require_pin(ROOT / name, expected, name)
    required = [CANONICAL, REGISTRY, C4 / "manifest.json", LEARNED / "manifest.json",
                SIM_PACK / "manifest.json", AER / "core_oof_predictions.csv"]
    required += [ROOT / item["path"] for item in contract["maestro"]["files"].values()]
    required += [ROOT / item["path"] for item in contract["immutable_authorities"].values()]
    if any(rel(p) not in contract["source_hashes"] for p in required):
        raise ValueError("report contract missing a required evidence pin")
    for item in (*contract["maestro"]["files"].values(),
                 *contract["immutable_authorities"].values()):
        if contract["source_hashes"].get(item["path"]) != item["sha256"]:
            raise ValueError("report contract has conflicting evidence pins")
    canonical = read_csv(CANONICAL)
    if (len(canonical) != 8767 or len({r["canonical_row_id"] for r in canonical}) != 8767
            or dict(Counter(r["source_id"] for r in canonical)) != contract["real_qpu"]["source_counts"]):
        raise ValueError("report canonical population/source count mismatch")
    outer = read_csv(ROOT / "artifacts/benchmark_v2/real_qpu/unified_outer_split_v2.csv")
    outer_by_id = {r["canonical_observation_id"]: r for r in outer}
    if len(outer) != 8767 or set(outer_by_id) != {r["canonical_row_id"] for r in canonical}:
        raise ValueError("report outer-split canonical IDs mismatch")
    group_folds: dict[str, set[str]] = {}
    for row in outer:
        group_folds.setdefault(row["unified_leakage_group_id"], set()).add(row["outer_fold"])
    if len(group_folds) != 2320 or any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("report outer-split group leakage/count mismatch")
    if {r["outer_fold"] for r in outer} != {str(f) for f in range(5)}:
        raise ValueError("report outer fold coverage mismatch")
    inner = read_csv(ROOT / "artifacts/benchmark_v2/real_qpu/unified_inner_split_v2.csv")
    slots = {(r["canonical_observation_id"], r["outer_fold"]) for r in inner}
    expected_slots = {(rid, str(fold)) for rid in outer_by_id for fold in range(5)}
    if len(inner) != 5 * 8767 or slots != expected_slots:
        raise ValueError("report inner-split row envelope mismatch")
    inner_groups: dict[tuple[str, str], set[str]] = {}
    for row in inner:
        rid, fold = row["canonical_observation_id"], row["outer_fold"]
        base = outer_by_id[rid]
        if row["unified_leakage_group_id"] != base["unified_leakage_group_id"]:
            raise ValueError("report inner-split group identity mismatch")
        if fold == base["outer_fold"]:
            if row["inner_fold"] != "":
                raise ValueError("report outer-test label entered inner tuning")
        else:
            if row["inner_fold"] not in {"0", "1", "2", "3"}:
                raise ValueError("report inner-training fold missing/invalid")
            inner_groups.setdefault((fold, row["unified_leakage_group_id"]), set()).add(row["inner_fold"])
    if any(len(folds) != 1 for folds in inner_groups.values()):
        raise ValueError("report inner-split group leakage")
    panel = read_csv(SIM_PACK / "common_exact_qasm_panel.csv")
    if (len(panel) != 204 or len({r["panel_member_id"] for r in panel}) != 204
            or len({r["exact_qasm_sha256"] for r in panel}) != 191
            or dict(Counter(r["stratum"] for r in panel)) != {"core_q2_q9": 162, "frontier_q10_q16": 42}):
        raise ValueError("report simulator panel population/hash/stratum mismatch")
    return contract


def maestro_terminal_evidence(contract: dict) -> tuple[dict, list[dict]]:
    """Recompute pilot stability and attempt accounting without native imports."""
    cfg = contract["maestro"]
    paths = {key: ROOT / item["path"] for key, item in cfg["files"].items()}
    for key, path in paths.items():
        require_pin(path, cfg["files"][key]["sha256"], f"Maestro {key}")
    run, preflight, acceptance, fit = (read_json(paths[k]) for k in
                                     ("run", "preflight", "acceptance", "fit"))
    policy, base = read_json(paths["policy"]), read_json(paths["base_policy"])
    identity = hashlib.sha256(json.dumps(run["run_identity"], sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    if (run.get("status") != "pilot_gate_failed" or run.get("score_created") is not False
            or run.get("manifest_id") != policy["manifest_id"]
            or run.get("manifest_sha256") != sha(paths["policy"])
            or identity != run.get("run_identity_sha256")
            or preflight.get("run_identity_sha256") != identity
            or acceptance.get("run_identity_sha256") != identity
            or fit.get("run_identity_sha256") != identity
            or acceptance.get("status") != "pass"
            or run.get("acceptance_sha256") != sha(paths["acceptance"])
            or run.get("accepted_preflight_run_manifest_sha256") != sha(paths["preflight"])
            or acceptance.get("preflight_run_manifest_sha256") != sha(paths["preflight"])
            or run["raw_records"]["sha256"] != sha(paths["raw"])
            or fit.get("calibration_raw_sha256") != sha(paths["raw"])
            or run.get("fitted_calibration_sha256") != sha(paths["fit"])
            or run.get("checkpoint_sha256") != sha(paths["checkpoint"])):
        raise ValueError("Maestro terminal identity/acceptance/evidence mismatch")
    records = read_csv(paths["raw"])
    if len({r["attempt_id"] for r in records}) != len(records):
        raise ValueError("Maestro duplicate attempt ID")
    stages = Counter(r["stage"] for r in records)
    expected_stages = {run["timed_stage"]: 150, run["untimed_stage"]: 90}
    if dict(stages) != expected_stages or len(records) != run["raw_records"]["rows"]:
        raise ValueError("Maestro timed/untimed attempt denominator mismatch")
    if any(r["status"] != "ok" or r["transport_status"] != "ok"
           or r["run_identity_sha256"] != identity or r["clock_id"] != run["selected_clock_id"]
           or r["selected_clock_id"] != run["selected_clock_id"]
           or r["selected_clock_field"] != run["selected_clock_field"]
           or r["process_start_method"] != "spawn" or r["call_process_isolated"] != "True" for r in records):
        raise ValueError("Maestro attempt context/status/clock mismatch")
    checkpoints = [json.loads(line) for line in paths["checkpoint"].read_text().splitlines()]
    ck_counts = Counter(row["record_type"] for row in checkpoints)
    expected_ck = {"slot_allocated": 10368, "terminal_skip": 792,
                   "call_started": 240, "attempt_result": 240}
    if dict(ck_counts) != expected_ck or any(row["run_identity_sha256"] != identity for row in checkpoints):
        raise ValueError("Maestro checkpoint allocation/attempt accounting mismatch")
    for kind in ("call_started", "attempt_result"):
        ids = [row["attempt_id"] for row in checkpoints if row["record_type"] == kind]
        if len(ids) != len(set(ids)) or set(ids) != {r["attempt_id"] for r in records}:
            raise ValueError("Maestro checkpoint attempt IDs mismatch")
    result_rows = {row["attempt_id"]: row["row"] for row in checkpoints
                   if row["record_type"] == "attempt_result"}
    for r in records:
        ck = result_rows[r["attempt_id"]]
        for name in ("cell_id", "stage", "status", "whole_qasm_sha256", "worker_context_sha256"):
            if str(ck[name]) != r[name]:
                raise ValueError(f"Maestro raw/checkpoint row mismatch: {name}")
        for name in ("reported_time_seconds", "host_wall_seconds"):
            if r[name] and float(ck[name]) != float(r[name]):
                raise ValueError(f"Maestro raw/checkpoint time mismatch: {name}")
    cells = {r["cell_id"] for r in records}
    if cells != set(run["completed_cell_ids"]) or cells != set(run["first_ten_cell_ids"]) or len(cells) != 10:
        raise ValueError("Maestro completed/pilot cell IDs mismatch")
    analysis, measurement = base["analysis"], base["measurement"]
    if (analysis["relative_tolerance"] != 0.2 or analysis["absolute_tolerance_seconds"] != 5e-5
            or measurement["sessions"] != 3 or measurement["timed_repetitions_per_cell_per_session"] != 5
            or measurement["untimed_calls_per_cell_per_session"] != 3
            or measurement["selected_clock_id"] != run["selected_clock_id"]
            or run["planned_cells"] != 432 or run["planned_enabled_cells"] != 399
            or run["planned_quarantined_cells"] != 33 or run["completed_cells"] != 10):
        raise ValueError("Maestro original stability/repetition contract changed")
    cell_rows = []
    for cell in run["first_ten_cell_ids"]:
        part = [r for r in records if r["cell_id"] == cell]
        desc = part[0]
        config_key = f"{desc['candidate']}:{desc['chi'] or 'None'}"
        for r in part:
            if (r["worker_context_sha256"] != run["worker_context_sha256_by_config"][config_key]
                    or r["resolved_config_sha256"] != run["configuration_records"][config_key]["sha256"]
                    or r["worker_environment_sha256"] != run["worker_environment_sha256"]):
                raise ValueError("Maestro worker/config context drift")
            for name in ("candidate", "width", "chi", "operation_class", "operation_repeats", "whole_qasm_sha256"):
                if r[name] != desc[name]:
                    raise ValueError("Maestro cell descriptor/QASM drift")
        for stage, count in ((run["timed_stage"], 5), (run["untimed_stage"], 3)):
            slots = [(int(r["session"]), int(r["repetition"])) for r in part if r["stage"] == stage]
            if len(slots) != 3 * count or set(slots) != {(s, rep) for s in (1, 2, 3) for rep in range(count)}:
                raise ValueError("Maestro duplicate/missing per-cell repetition")
        medians, mads, failed = [], [], False
        for session in (1, 2, 3):
            times = [float(r["reported_time_seconds"]) for r in part
                     if r["stage"] == run["timed_stage"] and int(r["session"]) == session]
            if any(not math.isfinite(t) or t < 0 for t in times):
                raise ValueError("Maestro nonfinite/negative timed value")
            center = median(times)
            mad = median(abs(t - center) for t in times)
            medians.append(center)
            mads.append(mad)
            failed |= mad > max(5e-5, 0.2 * center)
        between = max(medians) - min(medians)
        failed |= between > max(5e-5, 0.2 * median(medians))
        state = "unavailable" if failed else "ok"
        published = fit["fit_diagnostics"]["cell_qa"][cell]
        if state != published["status"] or state != run["pilot_qa"]["10"]["cell_status"][cell]:
            raise ValueError("Maestro recomputed stability status mismatch")
        for s, (center, mad) in enumerate(zip(medians, mads), 1):
            if (not math.isclose(center, published["session_medians_seconds"][str(s)], abs_tol=1e-12)
                    or not math.isclose(mad, published["within_session_mad_seconds"][str(s)], abs_tol=1e-12)):
                raise ValueError("Maestro recomputed median/MAD mismatch")
        timed = [r for r in part if r["stage"] == run["timed_stage"]]
        cell_rows.append({"cell_id": cell, "candidate": desc["candidate"], "width": desc["width"],
                          "chi": desc["chi"], "operation_class": desc["operation_class"],
                          "operation_repeats": desc["operation_repeats"], "timed_n": len(timed),
                          "session_medians_seconds": json.dumps(medians), "session_mads_seconds": json.dumps(mads),
                          "between_session_range_seconds": between,
                          "min_reported_seconds": min(float(r["reported_time_seconds"]) for r in timed),
                          "max_reported_seconds": max(float(r["reported_time_seconds"]) for r in timed),
                          "stability_status": state, "method_output_clock": run["selected_clock_id"],
                          "score_eligible": False})
    sv_failed = sum(r["candidate"] == "statevector" and r["stability_status"] == "unavailable" for r in cell_rows)
    mps_passed = sum(r["candidate"] == "mps_fixed_chi" and r["stability_status"] == "ok" for r in cell_rows)
    if sv_failed != 9 or mps_passed != 1 or run["pilot_qa"]["10"]["status"] != "fail":
        raise ValueError("Maestro terminal pilot outcome mismatch")
    status = {"status_at_build": "pilot_gate_failed", "score": None,
              "partial_experiment_id": policy["manifest_id"], "run_identity_sha256": identity,
              "assigned_calibration_cells": 432, "eligible_calibration_cells": 399,
              "quarantined_calibration_cells": 33, "completed_cells": 10,
              "eligible_unstarted_cells": 389, "api_calls": 240, "timed_calls": 150, "untimed_calls": 90,
              "sv_stability_failed_cells": sv_failed, "mps_stability_passed_cells": mps_passed,
              "panel_assigned_members": 204, "panel_assigned_candidate_attempts": 408,
              "panel_execution_started": False, "accepted_predictor": False,
              "selected_clock_id": run["selected_clock_id"], "persistent_process_warm_claim": False,
              "stability_recomputed_from_raw": True, "checkpoint_verified": True,
              "historical_availability_tables_are_snapshots": True,
              "interpretation": "Terminal synthetic calibration pilot failure; no common-panel predictor accuracy. Cause of SV variability remains unknown."}
    if any(status.get(key) != value for key, value in cfg["expected"].items()):
        raise ValueError("Maestro report contract expected outcome mismatch")
    return status, cell_rows


def build(output: Path, *, report_manifest: Path | None = None) -> dict[str, object]:
    output = output if output.is_absolute() else ROOT / output
    contract = load_report_contract(report_manifest) if report_manifest is not None else None
    if contract is not None:
        if output.resolve() != FINAL_OUT.resolve() and not output.resolve().is_relative_to(REBUILD_ROOT.resolve()):
            raise ValueError("final reader output must be canonical v3 or a fresh supported rebuild directory")
    elif output.resolve() != DEFAULT_OUT.resolve():
        raise ValueError("v2 reader tables use the single canonical output path")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {output}")

    c4_manifest_path = C4 / "manifest.json"
    learned_manifest_path = LEARNED / "manifest.json"
    sim_manifest_path = SIM_PACK / "manifest.json"
    c4_manifest = read_json(c4_manifest_path)
    learned_manifest = read_json(learned_manifest_path)
    sim_manifest = read_json(sim_manifest_path)
    require_pin(c4_manifest_path, PINS["c4_manifest"], "C4 manifest")
    require_pin(CANONICAL, PINS["canonical_observations"], "canonical corpus")
    require_pin(REGISTRY, PINS["fidelity_registry_v2"], "fidelity registry v2")
    require_pin(learned_manifest_path, PINS["learned_manifest"], "unified learned manifest")
    require_pin(sim_manifest_path, PINS["simulator_pack_manifest"], "simulator reader pack manifest")
    if contract is None:
        require_pin(PARTIAL_MAESTRO, PINS["partial_maestro_policy"], "Maestro partial policy")
        require_pin(MAESTRO_V3, PINS["failed_maestro_v3"], "failed Maestro v3 manifest")
    if c4_manifest.get("status") != "PASS" or c4_manifest.get("method_count") != 24:
        raise ValueError("C4 is not the pinned 24-method PASS aggregation")
    if learned_manifest.get("status") != "PASS":
        raise ValueError("unified learned-QPU reader source is not PASS")
    if (
        sim_manifest.get("artifact_id") != "simulator-reader-pack-v1"
        or sim_manifest.get("panel_rows") != 204
        or sim_manifest.get("unique_qasm_sha256") != 191
        or sim_manifest.get("flat_cross_clock_rank_emitted") is not False
    ):
        raise ValueError("simulator reader source does not satisfy the pinned 204/191 contract")
    check_manifest_files(C4, c4_manifest)
    check_manifest_files(LEARNED, learned_manifest)
    check_manifest_files(SIM_PACK, sim_manifest)
    check_input_hashes(c4_manifest)
    check_input_hashes(learned_manifest)

    canonical_rows = read_csv(CANONICAL)
    if len(canonical_rows) != 8767:
        raise ValueError("canonical QPU corpus must contain 8,767 observations")
    canonical = {r["canonical_row_id"]: r for r in canonical_rows}
    if len(canonical) != len(canonical_rows):
        raise ValueError("canonical row IDs are not unique")
    attempts = read_csv(C4 / "attempts.csv")
    if len(attempts) != 24 * 8767:
        raise ValueError(f"C4 attempt envelope mismatch: {len(attempts)}")
    by_method: dict[str, set[str]] = {}
    for row in attempts:
        rid = row["canonical_row_id"]
        if rid not in canonical:
            raise ValueError(f"C4 attempt has unknown canonical row {rid}")
        base = canonical[rid]
        if row["source_id"] != base["source_id"] or not math.isclose(
            float(row["actual_seconds"]), float(base["target_seconds"]), rel_tol=0, abs_tol=1e-10
        ):
            raise ValueError(f"C4 attempt/canonical label mismatch {rid}")
        by_method.setdefault(row["method_id"], set()).add(rid)
    if len(by_method) != 24 or any(ids != set(canonical) for ids in by_method.values()):
        raise ValueError("C4 methods do not each retain the same full 8,767-row envelope")

    # Complete all scientific input checks before creating the output directory.
    aer_rows, aer_extra = aer_diagnostic()
    partial = read_json(PARTIAL_MAESTRO)
    old_v3 = read_json(MAESTRO_V3)
    terminal_status, pilot_rows = maestro_terminal_evidence(contract) if contract else (None, [])

    output.mkdir(parents=True, exist_ok=True)
    qpu = output / "qpu"
    sim = output / "simulator"
    copied: dict[str, str] = {}
    for name, target in zip(C4_FILES, (
        "archived_method_coverage.csv", "archived_method_source_metrics.csv", "archived_method_pairwise_comparisons.csv"
    )):
        copied[f"qpu/{target}"] = copy_file(C4 / name, qpu / target)
    for name, target in zip(LEARNED_FILES, (
        "unified_learned_metrics.csv", "unified_learned_pairs.csv", "unified_learned_bootstrap.csv"
    )):
        copied[f"qpu/{target}"] = copy_file(LEARNED / name, qpu / target)
    for name in SIM_FILES:
        copied[f"simulator/{name}"] = copy_file(SIM_PACK / name, sim / name)

    write_csv(sim / "aer_mali_graph_c3a_diagnostic.csv", aer_rows)
    copied["simulator/aer_mali_graph_c3a_diagnostic.csv"] = sha(sim / "aer_mali_graph_c3a_diagnostic.csv")
    write_csv(sim / "aer_mali_graph_c3a_by_fold.csv", aer_extra["fold_rows"])
    copied["simulator/aer_mali_graph_c3a_by_fold.csv"] = sha(sim / "aer_mali_graph_c3a_by_fold.csv")

    # The Maestro partial policy has not produced any new timed result at build time.
    # Carry only its governing identity and an explicit no-score status.
    maestro_status = {
        "partial_experiment_id": partial.get("manifest_id", ""),
        "partial_manifest_path": rel(PARTIAL_MAESTRO),
        "partial_manifest_sha256": sha(PARTIAL_MAESTRO),
        "status_at_build": "not_started_no_partial_run_manifest",
        "score": None,
        "failed_v3_artifact_status": old_v3.get("status", "unknown"),
        "failed_v3_is_resumed": False,
        "interpretation": "No new partial-calibration or panel timing score is included; preserve v3 failure and do not report this as an estimator failure on the common panel.",
    }
    final_outputs: list[str] = []
    if contract is not None:
        maestro_status = terminal_status
        write_csv(sim / "maestro_pilot_summary.csv", pilot_rows)
        write_csv(output / "method_scope.csv", contract["method_scope"])
        write_csv(qpu / "source_native_scope.csv", contract["source_native_appendix"])
        final_outputs += ["simulator/maestro_pilot_summary.csv", "method_scope.csv", "qpu/source_native_scope.csv"]
        for source, target in (("coverage.csv", "unified_learned_coverage.csv"),
                               ("source_stratified.csv", "unified_learned_source_metrics.csv"),
                               ("unavailable.csv", "unified_learned_unavailable.csv"),
                               ("tail_diagnostics.csv", "unified_learned_tail_diagnostics.csv")):
            name = f"qpu/{target}"
            copied[name] = copy_file(LEARNED / source, output / name)
            final_outputs.append(name)

    c3a = aer_rows[0]
    report = f"""# Two-domain benchmark reader tables v2

Built from pinned archived real-QPU aggregation and local simulator artifacts.
This is a reader surface, not a flat leaderboard: QPU source labels, scheduled
outputs, and local simulator clocks remain separate.

## Real-QPU domain

The canonical population is **8,767 archived circuit observations**. The C4
attempt ledger has 24 method variants × 8,767 assigned rows (210,408 attempts).
Read `qpu/archived_method_coverage.csv` for assigned/predicted/unavailable
counts, `qpu/archived_method_source_metrics.csv` for source-local errors and
the 46-workflow QPack diagnostic, and `qpu/archived_method_pairwise_comparisons.csv`
for paired source-local contrasts. There is no cross-source pooled QPU score.
The three unified learned-adaptation summaries and their compatible-row pairs
are separate in `qpu/unified_learned_*.csv`.

## Local-simulator domain

The fixed panel has 204 members / 191 exact QASM hashes (162 core, 42 frontier,
q=2–16). Engine, precision, first/warm clock, approximation quality and
failure states stay separate in the five copied simulator tables. The C3a
local Aer graph adaptation is reported separately from the Azizov GNN: on 162
core OOF rows / 150 hashes its raw-derived MAE is {c3a['mae_seconds']:.6f} s,
MedAE {c3a['medae_seconds']:.6f} s, log1p-MAE {c3a['mae_log1p_seconds']:.6f},
R² {c3a['r2_seconds']:.6f}; q10–16 are excluded. It is an unregistered
supplementary architecture adaptation, not original Ma–Li or Azizov.

Maestro has **no score in this pack**. The previous v3 preparation failure is
preserved; the separately governed partial run was not started at this build.
This is not a measured common-panel prediction failure.

## Scope limits

No cross-domain or cross-clock ranking is emitted. `evaluation_target_clock`
and `method_output_clock` remain distinct in the source tables. QPack replay
circuits are reconstruction-qualified; they do not change the archived labels.
The package is a local review artifact, not a public-release approval. Root
license/citation and external redistribution-rights decisions remain open.
"""
    readme = """# Two-domain benchmark tables v2

Start with `REPORT.md`, then use `qpu/` and `simulator/` as separate evidence
domains. This pack replaces neither CURRENT nor any historical reader pack.
There is intentionally no single cross-domain ranking: clocks and targets
differ. `manifest.json` pins the source/output hashes and the explicit Maestro
no-score status at build time.
"""
    if contract is not None:
        report = report.replace("reader tables v2", "reader tables v3")
        report = report.replace(
            "Maestro has **no score in this pack**. The previous v3 preparation failure is\n"
            "preserved; the separately governed partial run was not started at this build.\n"
            "This is not a measured common-panel prediction failure.",
            "Maestro has **no score in this pack**. The S9 synthetic calibration pilot is\n"
            "terminal `pilot_gate_failed`: 10 completed cells, 240 API calls (150 timed),\n"
            "9 SV stability failures and 1 MPS stability pass. Of 432 assigned calibration\n"
            "cells, 33 are quarantined and 389 eligible cells remain unstarted. The\n"
            "204-member / 408-candidate-attempt panel was not executed. See\n"
            "`simulator/maestro_pilot_summary.csv` for independently recomputed medians/MADs.\n"
            "These process-isolated reported timings are not persistent-process warm\n"
            "timings. No predictor score or physical launch-overhead coefficient is inferred.\n"
            "The copied simulator availability/appendix tables are historical snapshots;\n"
            "the manifest and pilot summary carry the latest Maestro state.")
        report += """

## Required reconstruction and input disclosures

The evaluated learned graph uses a logical/physical mixture: Ma–Li and QPack
logical representations, Qonductor exact submitted physical QASM. It did not
execute S67's promised target-compiled Ma–Li/QPack inputs. Graph has seven
global features versus polynomial's five; performance cannot isolate graph
structure as its cause. QPack analytical replay uses six reconstructed
structures with representative rz(0.3)/rx(0.2), not recovered submitted circuits
or optimizer angles. Workflow holdout is not unseen-template holdout. Snapshot
calibrations are nominal backend-matched, not row-day historical calibration.
24 evaluated QPU variants are not 24 independent published approaches.

`method_scope.csv` covers the original approach list, including unevaluated
routes. `qpu/source_native_scope.csv` gives scoped historical pointers without
promoting new source-native accuracy. Source-native 340/4,482 remain separate,
not unified 8,767-row evaluations. `scientific_coverage=PARTIAL`; build validation
PASS means the evidence was assembled correctly, not full matrix completion,
clean-environment reproduction, slide approval or public-release permission.
"""
        readme = """# Two-domain benchmark tables v3

Start with REPORT.md and method_scope.csv. QPU and simulator results use separate
targets and clocks; there is no flat leaderboard. The report contract pins
sources and immutable authorities; manifest.json pins all outputs. Latest
Maestro state is a terminal calibration pilot failure with no predictor score.
Copied simulator tables retain their historical snapshot semantics.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "README.md").write_text(readme, encoding="utf-8")

    source_paths = [
        Path(__file__).resolve(),
        C4 / "manifest.json", C4 / "attempts.csv", C4 / "method_metrics.csv",
        C4 / "source_metrics.csv", C4 / "pairwise_comparisons.csv",
        CANONICAL, REGISTRY, LEARNED / "manifest.json", SIM_PACK / "manifest.json",
        AER / "training_manifest.json", AER / "overall_qa.json",
        AER / "core_oof_predictions.csv", AER / "core_metrics.csv",
        PARTIAL_MAESTRO, MAESTRO_V3,
    ]
    source_hashes = {rel(path): sha(path) for path in source_paths}
    if contract is not None:
        source_hashes.update(contract["source_hashes"])
        source_hashes[rel(report_manifest)] = sha(report_manifest)
    output_hashes = {name: sha(output / name) for name in (*OUTPUTS, "simulator/aer_mali_graph_c3a_by_fold.csv", *final_outputs)}
    payload = {
        "artifact_id": "two-domain-benchmark-reader-tables-v2",
        "status": "BUILT_FROM_PINNED_INPUTS_MAESTRO_NOT_STARTED",
        "domains": ["archived_real_qpu", "local_simulator"],
        "real_qpu": {
            "canonical_observations": 8767,
            "c4_method_variants": len(by_method),
            "c4_assigned_attempts": len(attempts),
            "cross_source_pooled_error_metrics": False,
            "source_pair_table": "qpu/archived_method_pairwise_comparisons.csv",
        },
        "simulator": {
            "panel_members": 204,
            "unique_qasm_hashes": 191,
            "core_members": 162,
            "frontier_members": 42,
            "c3a_oof_rows": c3a["n"],
            "c3a_unique_qasm_hashes": c3a["unique_qasm_hashes"],
            "c3a_metrics_recomputed_from_raw_oof": True,
            "maestro": maestro_status,
        },
        "no_flat_cross_domain_or_cross_clock_rank": True,
        "current_json_modified": False,
        "historical_packs_modified": False,
        "training_or_timing_performed": False,
        "public_release_approved": False,
        "source_hashes": source_hashes,
        "copied_or_derived_output_hashes": copied,
        "outputs": output_hashes,
    }
    if contract is not None:
        payload.update({"artifact_id": "two-domain-benchmark-reader-tables-v3",
                        "status": "BUILT_FROM_PINNED_INPUTS_MAESTRO_PILOT_FAILED",
                        "scientific_coverage": "PARTIAL",
                        "report_contract": {"path": rel(report_manifest), "sha256": sha(report_manifest)},
                        "wave4_five_fold_training_authorized": False,
                        "historical_availability_tables_are_snapshots": True})
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-manifest", type=Path,
                        help="Frozen report authority; selects final v3 instead of historical v2")
    args = parser.parse_args()
    report_manifest = args.report_manifest
    if report_manifest is not None and not report_manifest.is_absolute():
        report_manifest = ROOT / report_manifest
    output = args.output_dir or (FINAL_OUT if report_manifest else DEFAULT_OUT)
    payload = build(output, report_manifest=report_manifest)
    print(json.dumps({"artifact_id": payload["artifact_id"], "status": payload["status"],
                      "output_dir": str(output), "c4_method_variants": payload["real_qpu"]["c4_method_variants"],
                      "c3a_rows": payload["simulator"]["c3a_oof_rows"],
                      "maestro_status": payload["simulator"]["maestro"]["status_at_build"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
