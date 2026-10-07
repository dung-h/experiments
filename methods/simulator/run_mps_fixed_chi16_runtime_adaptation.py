#!/usr/bin/env python3
"""Prepare or run a five-fold CUDA-Q MPS warm-runtime predictor adaptation.

This is a local source-DAG adaptation, not the Family-Aware, Azizov, or
original Ma--Li method.  The default action is read-only with respect to
historical inputs and writes only a small, hash-pinned derived target table.
Model fitting stays blocked until the per-method protocol authorization is
valid and simulator timing is idle under the shared compute lock. Fold 0 is an integrity-only
checkpoint: after its technical QA passes, folds 1--4 run unchanged. Metrics
never control continuation.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import functools
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "artifacts/benchmark_v1/cudaq_mps_common_q16_bridge_full_v1_20260928/cudaq_mps_common_raw.csv"
RAW_QA = ROOT / "artifacts/benchmark_v1/cudaq_mps_common_q16_bridge_qa_v1_20260928/qa.json"
RUN_MANIFEST = ROOT / "artifacts/benchmark_v1/cudaq_mps_common_q16_bridge_full_v1_20260928/run_manifest.json"
ENVIRONMENT = ROOT / "artifacts/benchmark_v1/cudaq_mps_common_q16_bridge_full_v1_20260928/environment.json"
ATTEMPTS = ROOT / "artifacts/benchmark_v1/cudaq_mps_common_q16_bridge_full_v1_20260928/attempt_records.jsonl"
BRIDGE = ROOT / "benchmark_v1/execution/manifests/cudaq_verified_bridge_v1.json"
ENV_LOCK = ROOT / "benchmark_v1/execution/manifests/simulator_environment_lock_v1.json"
SIM_GATES = ROOT / "benchmark_v1/protocol/simulator_completion_gates_v1.json"
MPS_AUTH_DECISION = ROOT / "benchmark_v1/decisions/S85_MPS_FOLD0_DIAGNOSTIC_AUTHORIZATION_20261002.md"
PANEL = ROOT / "artifacts/benchmark_v1/sim_common_q16_manifest_20260927/sim_common_q16_manifest.csv"
FOLDS = ROOT / "artifacts/benchmark_v1/c44_aer_q16_full_panel_evaluation_20260927/aer_q16_reduced_warm.csv"
STATIC_PACK = ROOT / "artifacts/benchmark_v3/simulator/azizov_common_core_gnn_v1_materialization"
TOPOLOGY = STATIC_PACK / "source_dag_topology.jsonl"
STATIC_SOURCE_HASHES = STATIC_PACK / "source_hashes.json"
S84_RUN_ROOT = ROOT / "artifacts/benchmark_v3/simulator/maestro_threadpool_diagnostic_20261002"
S84_ATTEMPT_DIR = S84_RUN_ROOT / "attempt_003"
SHARED_TIMING_LOCK = ROOT / "work/locks/maestro_qcsim_v2_cpu_timing.lock"
GPU_LEASE_PATH = Path("/tmp/qre-benchmark-gpu-lease.lock")
_COMPUTE_LOCK_OWNER_PID: int | None = None
DEFAULT_OUTPUT = ROOT / "artifacts/benchmark_v1/mps_fixed_chi16_runtime_only_adaptation_v1"
SEEDS = (42, 1234, 31415)
GLOBAL_NAMES = (
    "active_width",
    "measurement_stripped_structural_depth",
    "one_qubit_gate_count",
    "two_qubit_gate_count",
    "multi_qubit_gate_count",
    "swap_like_gate_count",
)
TARGET_ID = "cudaq_mps_fp64_bond16_warm_state_execution_seconds"
SOURCE_TARGET_ID = "cudaq_mps_state_wall_clock_quality_constrained"
RUNNER_ID = "mps-fixed-chi16-source-dag-runtime-adaptation-v1"
EXPECTED_FOLD_HASH_COUNTS = {0: 37, 1: 28, 2: 26, 3: 25, 4: 34}
MPS_METHOD_FAMILY = "cudaq_mps_fixed_chi16_runtime_only_adaptation"
MPS_METHOD_ID = "mps_fixed_bond16_warm_runtime_prediction"
AUTHORIZED_MPS_GATE_STATE = "conditionally_authorized_five_fold_oof_when_timing_idle"
AUTHORIZED_PREDICTION_METHODS = (
    "train_fold_median",
    "ridge_alpha_1",
    "graph_seed_42",
    "graph_seed_1234",
    "graph_seed_31415",
    "graph_median_three_seeds",
)
FROZEN_FOLDS = (0, 1, 2, 3, 4)
FOLD0_CHECKPOINT_RULES = {
    "technical_qa_only": True,
    "automatically_continue_folds_1_through_4": True,
    "metrics_quality_inspected": False,
    "prediction_quality_inspected": False,
    "requires_finite_predictions_for_every_assigned_hash": True,
    "requires_exact_hash_and_fold_coverage": True,
    "requires_zero_train_test_hash_overlap": True,
    "requires_frozen_cuda_environment_match": True,
    "requires_all_fixed_baselines_and_seed_outputs": True,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def s84_attempt_manifest_path() -> Path:
    """Return the immutable S84 append-only attempt_003 terminal manifest path.

    The root-level S84 manifest describes an earlier blocked preflight and is
    intentionally not authoritative for the resumed attempt.
    """
    return S84_ATTEMPT_DIR / "run_manifest.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def input_hashes() -> dict[str, str]:
    files = {
        "mps_raw_csv": RAW,
        "mps_qa_json": RAW_QA,
        "mps_run_manifest": RUN_MANIFEST,
        "mps_environment": ENVIRONMENT,
        "mps_attempt_records": ATTEMPTS,
        "cudaq_bridge_manifest": BRIDGE,
        "simulator_environment_lock": ENV_LOCK,
        "simulator_completion_gates": SIM_GATES,
        "mps_execution_authorization_decision": MPS_AUTH_DECISION,
        "panel_manifest_csv": PANEL,
        "c44_fold_source_csv": FOLDS,
        "static_source_dag_jsonl": TOPOLOGY,
        "static_source_hashes": STATIC_SOURCE_HASHES,
    }
    missing = [str(path.relative_to(ROOT)) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("required pinned inputs missing: " + ", ".join(missing))
    return {name: sha256_file(path) for name, path in files.items()}


def validate_provenance(hashes: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    run = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    qa = json.loads(RAW_QA.read_text(encoding="utf-8"))
    checks = {
        "raw_hash_matches_run_manifest": hashes["mps_raw_csv"] == run["outputs"]["raw_csv"]["sha256"],
        "environment_hash_matches_run_manifest": hashes["mps_environment"] == run["outputs"]["environment"]["sha256"],
        "attempts_hash_matches_run_manifest": hashes["mps_attempt_records"] == run["outputs"]["attempt_records"]["sha256"],
        "panel_hash_matches_run_manifest": hashes["panel_manifest_csv"] == run["panel_manifest"]["sha256"],
        "bridge_hash_matches_run_manifest": hashes["cudaq_bridge_manifest"] == run["bridge_manifest"]["sha256"],
        "environment_lock_hash_matches_run_manifest": hashes["simulator_environment_lock"] == run["environment_lock"]["sha256"],
        "raw_hash_matches_qa": hashes["mps_raw_csv"] == qa["raw_csv_sha256"],
        "panel_hash_matches_qa": hashes["panel_manifest_csv"] == qa["panel_manifest_sha256"],
        "qa_pass": qa["status"] == "PASS" and not qa["errors"],
        "measurement_is_existing_evidence_only": run["timing_performed"] is True,
        "one_gpu_worker": run["concurrency"] == "one sequential child process and one GPU timing worker",
        "fixed_mps_configuration": run["mps"] == {"abs_cutoff": 1e-10, "max_bond": 16, "precision": "fp64", "svd_algorithm": "gesvdj"},
    }
    failures = [key for key, value in checks.items() if not value]
    if failures:
        raise ValueError("MPS provenance checks failed: " + ", ".join(failures))
    if qa["raw_rows"] != 3672 or qa["cells"] != 612 or qa["timing_recomputed"] is not False:
        raise ValueError("MPS QA counts/policy differ from the pinned bridge evidence")
    panel = read_csv(PANEL)
    folds = read_csv(FOLDS)
    static_source_hashes = json.loads(STATIC_SOURCE_HASHES.read_text(encoding="utf-8"))
    static_pin_checks = {
        "static_pack_topology_sha_matches_source_hashes": hashes["static_source_dag_jsonl"] == static_source_hashes["source_dag_jsonl_sha256"],
        "static_pack_panel_sha_matches_source_hashes": hashes["panel_manifest_csv"] == static_source_hashes[str(PANEL.relative_to(ROOT))],
        "static_pack_c44_sha_matches_source_hashes": hashes["c44_fold_source_csv"] == static_source_hashes[str(FOLDS.relative_to(ROOT))],
        "static_pack_upstream_commit_pinned": bool(static_source_hashes.get("upstream_repository_commit")),
    }
    failures = [key for key, value in static_pin_checks.items() if not value]
    if failures:
        raise ValueError("static source-DAG pack pin validation failed: " + ", ".join(failures))
    return {"run_manifest": run, "qa": qa, "checks": {**checks, **static_pin_checks}}, panel, folds


def frozen_fold_map(c44_rows: list[dict[str, str]]) -> dict[str, int]:
    source: dict[str, int] = {}
    for row in c44_rows:
        if row["stratum"] != "core_q2_q9":
            continue
        digest = row["source_sha256"]
        fold = int(row["fold"])
        old = source.setdefault(digest, fold)
        if old != fold:
            raise ValueError(f"C44 exact-QASM hash crosses folds: {digest}")
    if len(source) != 150:
        raise ValueError(f"C44 frozen split must contain 150 unique core hashes, got {len(source)}")
    actual_counts = Counter(source.values())
    if dict(sorted(actual_counts.items())) != EXPECTED_FOLD_HASH_COUNTS:
        raise ValueError(f"frozen C44 unique-hash fold counts differ: {dict(sorted(actual_counts.items()))}")
    return source


def load_topology() -> dict[str, dict[str, Any]]:
    graphs: dict[str, dict[str, Any]] = {}
    with TOPOLOGY.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            digest = record["source_qasm_sha256"]
            if digest in graphs:
                raise ValueError(f"duplicate source DAG hash at JSONL line {line_no}: {digest}")
            nodes = record.get("nodes", [])
            if int(record["node_count"]) != len(nodes):
                raise ValueError(f"node count mismatch at JSONL line {line_no}")
            ids = [int(node["node_id"]) for node in nodes]
            if ids != list(range(len(nodes))):
                raise ValueError(f"non-contiguous/non-ordered DAG node IDs at JSONL line {line_no}")
            if not record.get("representation", "").startswith("source_qiskit_dag_topology_v1"):
                raise ValueError(f"unexpected source DAG representation at line {line_no}")
            graphs[digest] = record
    if len(graphs) != 150:
        raise ValueError(f"expected 150 source DAG hashes, got {len(graphs)}")
    return graphs


def graph_features(record: dict[str, Any]) -> tuple[dict[str, int], dict[str, Any]]:
    """Create a measurement-stripped source DAG; parameter values are ignored."""
    source_nodes = record["nodes"]
    num_qubits = int(record["num_qubits"])
    measurements: list[tuple[int, set[int]]] = []
    for node in source_nodes:
        name = str(node["operation"]).lower()
        qargs = {int(q) for q in node.get("qargs", [])}
        if any(q < 0 or q >= num_qubits for q in qargs):
            raise ValueError(f"qarg out of range in source DAG {record['source_qasm_sha256']}")
        node_id = int(node["node_id"])
        if name in {"measure", "measure_all"}:
            measurements.append((node_id, qargs))
    if measurements:
        first_measure_id = min(node_id for node_id, _qargs in measurements)
        later_gates = [
            str(node["operation"]).lower()
            for node in source_nodes
            if int(node["node_id"]) > first_measure_id
            and str(node["operation"]).lower() not in {"measure", "measure_all", "barrier", "snapshot"}
        ]
        if later_gates:
            raise ValueError(f"nonterminal measurement followed by quantum gate {later_gates[0]} in source DAG {record['source_qasm_sha256']}")

    nodes: list[dict[str, Any]] = []
    active: set[int] = set()
    last_by_qubit: dict[int, int] = {}
    depths: list[int] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []
    counts = Counter()
    swap_like = 0
    for original in source_nodes:
        name = str(original["operation"]).lower()
        if name in {"barrier", "snapshot", "measure", "measure_all"}:
            continue
        qargs = [int(q) for q in original.get("qargs", [])]
        active.update(qargs)
        arity = len(qargs)
        if arity == 1:
            counts["one"] += 1
        elif arity == 2:
            counts["two"] += 1
        elif arity >= 3:
            counts["multi"] += 1
        if "swap" in name:
            swap_like += 1
        node_id = len(nodes)
        for qubit in qargs:
            prior = last_by_qubit.get(qubit)
            if prior is not None:
                edge_src.append(prior)
                edge_dst.append(node_id)
        depth = 1 + max((depths[last_by_qubit[q]] for q in qargs if q in last_by_qubit), default=-1)
        for qubit in qargs:
            last_by_qubit[qubit] = node_id
        depths.append(depth)
        params = original.get("params_text", [])
        nodes.append({
            "operation": name,
            "arity": min(arity, 3),
            "parameter_present": bool(params),
            "qargs": qargs,
        })
    if not nodes:
        raise ValueError(f"no quantum-operation nodes remain after measurement stripping for {record['source_qasm_sha256']}")
    globals_ = {
        "active_width": len(active),
        "measurement_stripped_structural_depth": max(depths, default=0),
        "one_qubit_gate_count": counts["one"],
        "two_qubit_gate_count": counts["two"],
        "multi_qubit_gate_count": counts["multi"],
        "swap_like_gate_count": swap_like,
    }
    return globals_, {
        "num_qubits": num_qubits,
        "nodes": nodes,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "removed_measurements": len(measurements),
        "removed_barriers": sum(str(n["operation"]).lower() == "barrier" for n in source_nodes),
    }


def aggregate_targets(
    raw_rows: list[dict[str, str]],
    panel_rows: list[dict[str, str]],
    fold_by_hash: dict[str, int],
    *,
    expected_core_members: int = 162,
    expected_hash_groups: int = 150,
) -> list[dict[str, Any]]:
    core_panel = [row for row in panel_rows if row["stratum"] == "core_q2_q9"]
    members_by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in core_panel:
        members_by_hash[row["qasm_sha256"]].append(row)
    expected_hashes = set(members_by_hash)
    if len(core_panel) != expected_core_members or len(expected_hashes) != expected_hash_groups or expected_hashes != set(fold_by_hash):
        raise ValueError(f"core panel/frozen fold envelope is not {expected_core_members} members / {expected_hash_groups} exact hashes")
    warm = [r for r in raw_rows if r["stratum"] == "core_q2_q9" and r["observation_kind"] == "warm"]
    raw_hashes = {r["qasm_sha256"] for r in raw_rows if r["stratum"] == "core_q2_q9"}
    if raw_hashes != expected_hashes:
        raise ValueError("MPS warm raw exact hashes differ from frozen core panel")
    by_session: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    member_status: dict[str, set[str]] = defaultdict(set)
    quality_states: dict[str, set[str]] = defaultdict(set)
    for row in warm:
        digest = row["qasm_sha256"]
        if digest not in expected_hashes:
            raise ValueError(f"warm row not in core panel: {digest}")
        member_status[digest].add(row["status"])
        if row.get("fidelity") and row.get("quality_threshold"):
            state = "pass" if float(row["fidelity"]) >= float(row["quality_threshold"]) else "fail"
            quality_states[digest].add(state)
        value = row.get("t_warm_execution_s", "")
        if row["status"] in {"adapter_error", "unsupported", "timeout"} and value:
            raise ValueError(f"terminal failure has a warm-runtime value for {digest}: {row['status']}")
        if row["status"] in {"adapter_error", "unsupported", "timeout"} and row["error"] == "":
            raise ValueError(f"terminal failure is missing its recorded error for {digest}: {row['status']}")
        if value:
            seconds = float(value)
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"invalid finite warm target for {digest}")
            by_session[(digest, row["panel_member_id"], row["session_id"])].append(seconds)
    sessions_by_hash: dict[str, list[float]] = defaultdict(list)
    for (digest, _member, _session), values in by_session.items():
        if values:
            sessions_by_hash[digest].append(statistics.median(values))
    rows: list[dict[str, Any]] = []
    for digest in sorted(expected_hashes):
        values = sessions_by_hash[digest]
        families = sorted({r["family"] for r in members_by_hash[digest]})
        status = "runtime_observed" if values else "unavailable_adapter_error"
        qstates = quality_states[digest]
        if qstates == {"pass"}:
            quality = "quality_pass"
        elif qstates == {"fail"}:
            quality = "quality_failed"
        elif qstates == {"pass", "fail"}:
            quality = "quality_mixed"
        else:
            quality = "quality_unavailable"
        rows.append({
            "source_qasm_sha256": digest,
            "fold": fold_by_hash[digest],
            "target_status": status,
            "target_seconds": statistics.median(values) if values else "",
            "quality_status_separate": quality,
            "quality_is_model_input": False,
            "panel_member_count": len(members_by_hash[digest]),
            "family_alias_count_audit_only": len(families),
            "session_member_medians": len(values),
            "warm_repetitions_seen": sum(len(v) for k, v in by_session.items() if k[0] == digest),
            "raw_statuses_audit_only": ";".join(sorted(member_status[digest])),
        })
    return rows


def load_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    hashes = input_hashes()
    provenance, panel_rows, c44_rows = validate_provenance(hashes)
    raw_rows = read_csv(RAW)
    source_target_ids = {row["target_id"] for row in raw_rows if row["stratum"] == "core_q2_q9"}
    if source_target_ids != {SOURCE_TARGET_ID}:
        raise ValueError(f"unexpected historical raw target id(s): {sorted(source_target_ids)}")
    fold_by_hash = frozen_fold_map(c44_rows)
    graph_records = load_topology()
    panel_hashes = {r["qasm_sha256"] for r in panel_rows if r["stratum"] == "core_q2_q9"}
    if panel_hashes != set(graph_records):
        raise ValueError("source-DAG pack hash set does not match frozen MPS core panel")
    source_hashes = json.loads(STATIC_SOURCE_HASHES.read_text(encoding="utf-8"))
    if sorted(graph_records) != source_hashes["external_source_qasm_hashes"]:
        raise ValueError("static source-DAG exact-QASM hash list differs from source_hashes.json")
    feature_by_hash: dict[str, dict[str, Any]] = {}
    for digest, record in graph_records.items():
        globals_, graph = graph_features(record)
        feature_by_hash[digest] = {"globals": globals_, "graph": graph}
    targets = aggregate_targets(raw_rows, panel_rows, fold_by_hash)
    return {"hashes": hashes, "provenance": provenance, "panel": panel_rows, "fold_by_hash": fold_by_hash}, targets, graph_records, feature_by_hash


def validate_family_support(panel: list[dict[str, str]], fold_by_hash: dict[str, int]) -> dict[str, list[str]]:
    families_by_hash: dict[str, set[str]] = defaultdict(set)
    for row in panel:
        if row["stratum"] == "core_q2_q9":
            families_by_hash[row["qasm_sha256"]].add(row["family"])
    all_hashes = set(fold_by_hash)
    result: dict[str, list[str]] = {}
    for fold in range(5):
        test_hashes = {h for h, f in fold_by_hash.items() if f == fold}
        train_hashes = all_hashes - test_hashes
        test_families = set().union(*(families_by_hash[h] for h in test_hashes))
        train_families = set().union(*(families_by_hash[h] for h in train_hashes))
        result[str(fold)] = sorted(test_families - train_families)
    return result


def encode_node_features(node: dict[str, Any], vocab_map: dict[str, int], unk_index: int, num_qubits: int) -> list[float]:
    """Encode gate identity/arity/parameter-presence and ordered qarg roles."""
    if len(node["qargs"]) > 3:
        raise ValueError("source-DAG adaptation supports at most three ordered qargs per gate")
    gate_width = len(vocab_map) + 1
    values = [0.0] * (gate_width + 11)
    values[vocab_map.get(node["operation"], unk_index)] = 1.0
    values[gate_width + min(int(node["arity"]), 3)] = 1.0
    values[gate_width + 4] = 1.0 if node["parameter_present"] else 0.0
    denominator = float(max(1, num_qubits - 1))
    for role, qubit in enumerate(node["qargs"]):
        values[gate_width + 5 + role] = float(qubit) / denominator
        values[gate_width + 8 + role] = 1.0
    return values


def cuda_training_environment() -> dict[str, Any]:
    probe = r'''import json, torch, torch_geometric
import os, sys
print(json.dumps({
    "python_executable": sys.executable,
    "torch": str(torch.__version__),
    "torch_cuda_runtime": str(torch.version.cuda),
    "torch_geometric": str(torch_geometric.__version__),
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
}))'''
    result = subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True, text=True)
    info = json.loads(result.stdout.strip().splitlines()[-1])
    if not info["cuda_available"] or info["device_count"] < 1:
        raise RuntimeError("CUDA training environment is not available; CPU fallback is forbidden")
    freeze_bytes, freeze_tool = package_freeze_snapshot()
    info["pip_freeze_sha256"] = sha256_bytes(freeze_bytes)
    info["pip_freeze_tool"] = freeze_tool
    return info


def package_freeze_snapshot() -> tuple[bytes, str]:
    """Return a stable installed-package inventory without changing the environment."""
    pip_result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
    )
    if pip_result.returncode == 0:
        return pip_result.stdout.encode(), "python-m-pip-freeze"
    uv_path = shutil.which("uv")
    if uv_path:
        uv_result = subprocess.run(
            [uv_path, "pip", "freeze", "--python", sys.executable],
            capture_output=True,
            text=True,
        )
        if uv_result.returncode == 0:
            return uv_result.stdout.encode(), "uv-pip-freeze"
        detail = uv_result.stderr.strip().splitlines()[-1:] or ["unknown uv error"]
    else:
        detail = ["uv is not installed"]
    raise RuntimeError(
        "cannot record installed-package inventory: python -m pip freeze failed and "
        + "; ".join(detail)
    )


def validate_mps_execution_authorization(gates_document: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the one method-specific five-fold authorization matches exactly."""
    if gates_document.get("default_execution_state") != "not_authorized":
        raise ValueError("global execution default must remain not_authorized")
    matches = [gate for gate in gates_document.get("gates", []) if gate.get("method_id") == MPS_METHOD_ID]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one authorization gate for {MPS_METHOD_ID}")
    gate = matches[0]
    if gate.get("method_family") != MPS_METHOD_FAMILY:
        raise ValueError("MPS method ID is not bound to the expected method family")
    if gate.get("target_clock_id") != TARGET_ID:
        raise ValueError("MPS gate target clock does not match this runner")
    if gate.get("authorization_decision") != str(MPS_AUTH_DECISION.relative_to(ROOT)):
        raise ValueError("MPS gate does not point to the scoped authorization decision")
    if gate.get("state") != AUTHORIZED_MPS_GATE_STATE:
        raise ValueError("MPS five-fold OOF is not authorized under timing exclusion")
    authorization = gate.get("execution_authorization")
    if not isinstance(authorization, dict) or authorization.get("status") != "conditionally_authorized" or authorization.get("scope") != "fixed_split_five_fold_oof_with_fold0_technical_checkpoint":
        raise ValueError("MPS per-method execution authorization is missing or has an invalid scope")
    if authorization.get("folds") != list(FROZEN_FOLDS) or authorization.get("seeds") != list(SEEDS):
        raise ValueError("MPS authorization must cover all five frozen folds and exactly the three frozen seeds")
    if authorization.get("prediction_methods") != list(AUTHORIZED_PREDICTION_METHODS):
        raise ValueError("MPS authorization prediction-method list differs from the runner")
    if authorization.get("family_ood") is not False or authorization.get("new_simulator_timing") is not False or authorization.get("leaderboard_promotion") is not False:
        raise ValueError("MPS authorization exceeds the fixed-split five-fold OOF scope")
    if authorization.get("five_fold_oof") is not True:
        raise ValueError("MPS authorization must explicitly cover the frozen five-fold OOF run")
    if authorization.get("fold0_checkpoint_rules") != FOLD0_CHECKPOINT_RULES:
        raise ValueError("MPS fold-0 continuation rules differ from the frozen metric-blind technical QA contract")
    if authorization.get("requires_s84_complete") is not False or authorization.get("requires_exclusive_timing_lock") is not True or authorization.get("requires_host_gpu_lease") is not True or authorization.get("requires_no_live_timing_worker") is not True:
        raise ValueError("MPS authorization must require timing exclusion without S84 completion")
    basis = authorization.get("basis", "")
    if not isinstance(basis, str) or "user" not in basis.lower() or "review" not in basis.lower():
        raise ValueError("MPS authorization must record the user-request and scientific-review basis")
    return {"method_family": MPS_METHOD_FAMILY, "method_id": MPS_METHOD_ID, **authorization}


def s84_status_audit(manifest_path: Path | None = None) -> dict[str, Any]:
    """Record existing S84 status without using results/completion to permit fitting."""
    manifest_path = manifest_path or s84_attempt_manifest_path()
    if not manifest_path.is_file():
        return {"status": "not_present", "completion_required": False}
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        return {"status": "audit_unreadable", "error_class": type(exc).__name__, "completion_required": False}
    manifest_digest = sha256_bytes(manifest_bytes)
    try:
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError("S84 audit manifest is not an object")
    except (ValueError, UnicodeDecodeError) as exc:
        return {"status": "audit_invalid", "error_class": type(exc).__name__, "run_manifest_sha256": manifest_digest, "completion_required": False}
    return {
        "status": manifest.get("status", "unknown"),
        "attempt_count": manifest.get("attempt_count"),
        "scheduled_attempts": manifest.get("scheduled_attempts"),
        "run_manifest_sha256": manifest_digest,
        "completion_required": False,
    }


def live_timing_processes(proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    """Find S84 controllers and possible orphaned Python spawn workers.

    S84 creates spawn children whose argv does not name its runner. Conservatively
    block such workers too; a controller's absence alone does not prove quiescence.
    Inspection errors fail closed except for processes disappearing during scan.
    """
    active = []
    for process in proc_root.iterdir():
        if not process.name.isdecimal() or int(process.name) == os.getpid():
            continue
        try:
            command = process.joinpath("cmdline").read_bytes().split(b"\0")
            args = [value.decode("utf-8", errors="replace") for value in command if value]
            is_controller = any(Path(arg).name == "run_maestro_threadpool_intervention.py" for arg in args) and "--execute" in args
            is_spawn_worker = any("multiprocessing.spawn" in arg and "spawn_main" in arg for arg in args)
            if is_controller or is_spawn_worker:
                active.append({"pid": int(process.name), "kind": "s84_controller" if is_controller else "possible_orphan_spawn_timing_worker"})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise ValueError(f"timing process inspection unavailable for PID {process.name}") from exc
    return active


@contextlib.contextmanager
def exclusive_compute_lock(lock_path: Path | None = None, gpu_lease_path: Path | None = None):
    """Hold both Maestro timing flock and the host GPU lease throughout CUDA."""
    global _COMPUTE_LOCK_OWNER_PID
    if _COMPUTE_LOCK_OWNER_PID == os.getpid():
        yield
        return
    lock_path = lock_path or SHARED_TIMING_LOCK
    gpu_lease_path = gpu_lease_path or GPU_LEASE_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_lease_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("shared simulator timing/compute lock is held; CUDA work blocked") from exc
        try:
            with gpu_lease_path.open("a+", encoding="utf-8") as gpu_handle:
                try:
                    fcntl.flock(gpu_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ValueError("host GPU lease is held; CUDA work blocked") from exc
                try:
                    active = live_timing_processes()
                    if active:
                        raise ValueError(f"live timing controller or possible orphan worker detected: {active}")
                    _COMPUTE_LOCK_OWNER_PID = os.getpid()
                    yield
                finally:
                    _COMPUTE_LOCK_OWNER_PID = None
                    fcntl.flock(gpu_handle.fileno(), fcntl.LOCK_UN)
        finally:
            _COMPUTE_LOCK_OWNER_PID = None
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def with_compute_exclusion(function):
    @functools.wraps(function)
    def guarded(*args, **kwargs):
        with exclusive_compute_lock():
            return function(*args, **kwargs)
    return guarded


@with_compute_exclusion
def materialize(output_dir: Path) -> None:
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing adaptation output: {output_dir}")
    # The preflight performs a CUDA environment probe. Keep even that probe
    # behind the shared timing lock, held until the probe process exits.
    try:
        validate_mps_execution_authorization(json.loads(SIM_GATES.read_text(encoding="utf-8")))
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"MPS CUDA preflight remains blocked: {exc}") from exc
    _, targets, graph_records, features = load_inputs()
    # Re-read the minimum audit inputs only after their hashes were validated.
    panel = read_csv(PANEL)
    fold_by_hash = frozen_fold_map(read_csv(FOLDS))
    if len(targets) != 150 or sum(row["target_status"] == "runtime_observed" for row in targets) != 144:
        raise ValueError("MPS target materialization expected 150 hashes with 144 finite runtime labels")
    quality_counts = Counter(row["quality_status_separate"] for row in targets if row["target_status"] == "runtime_observed")
    if quality_counts != Counter({"quality_pass": 142, "quality_failed": 2}):
        raise ValueError(f"unexpected quality statuses among finite targets: {quality_counts}")
    unavailable = [row for row in targets if row["target_status"] != "runtime_observed"]
    if len(unavailable) != 6 or any(row["target_seconds"] != "" for row in unavailable):
        raise ValueError("adapter_error rows must remain six unavailable, un-imputed targets")
    family_support = validate_family_support(panel, fold_by_hash)
    if any(family_support.values()):
        raise ValueError("primary exact-QASM folds unexpectedly contain family-unseen tests")
    rows = []
    for row in targets:
        digest = row["source_qasm_sha256"]
        graph = features[digest]["graph"]
        rows.append({**row, **features[digest]["globals"], "node_count_after_strip": len(graph["nodes"]), "edge_count_after_strip": len(graph["edge_src"])})
    hashes = input_hashes()
    fold_counts = Counter(int(row["fold"]) for row in targets)
    gpu_environment = cuda_training_environment()
    output_dir.mkdir(parents=True)
    fields = list(rows[0])
    target_path = output_dir / "hash_targets_and_static_features.csv"
    write_csv(target_path, fields, rows)
    manifest = {
        "artifact_id": RUNNER_ID,
        "status": "preflight_pass_training_not_run",
        "method_claim": "fixed-config CUDA-Q MPS warm-runtime prediction; source-DAG method adaptation only",
        "not_claimed": ["Family-Aware paper reproduction", "Azizov method reproduction", "original Ma-Li reproduction", "simulator selection", "runtime at other chi/precision/engine"],
        "target": TARGET_ID,
        "source_raw_target_id": SOURCE_TARGET_ID,
        "target_boundary": "median of existing warm get_state repetitions only; build/first/result extraction/fidelity clocks excluded",
        "prediction_unit": "one exact source-QASM SHA-256 group",
        "primary_split": "C44 five-fold exact-QASM hash assignment; family known in every fold; all five folds produce OOF predictions",
        "core_accounting": {"panel_members": 162, "exact_hash_groups": 150, "finite_runtime_labels": 144, "quality_pass_finite": 142, "quality_failed_finite": 2, "adapter_error_unavailable": 6},
        "target_reduction": "median warm repetitions per member/session, then median of member/session medians per exact hash",
        "quality_policy": "quality status is separately reported, never a predictor; finite quality_failed timings remain runtime targets; adapter_error is unavailable and never imputed",
        "representation": {
            "id": "mps_fixed_chi16_measurement_stripped_source_dag_v1",
        "input": "pre-materialized source-QASM operation topology keyed by exact QASM SHA; no circuit bytes or family labels are inputs",
            "node_fields": ["gate-name one-hot (vocabulary fit on outer train)", "arity bucket", "parameter-present bit only", "normalized qarg positions"],
            "global_features": list(GLOBAL_NAMES),
            "preprocessing": "log1p globals, mean/std fit on finite outer-train hashes only; parameter values, shots, measurements, family, quality, backend, source path and hash excluded",
            "normalization": "barrier/snapshot and verified-terminal measurements removed; per-wire DAG edges rebuilt from retained ordered qargs",
        },
        "fold0_unique_hash_count": sum(int(row["fold"]) == 0 for row in targets),
        "fold0_labeled_hash_count": sum(int(row["fold"]) == 0 and row["target_status"] == "runtime_observed" for row in targets),
        "unique_hashes_by_frozen_fold": {str(k): fold_counts[k] for k in sorted(fold_counts)},
        "execution_plan": {
            "folds": list(FROZEN_FOLDS),
            "fold0_role": "metric_blind_technical_integrity_checkpoint",
            "on_fold0_technical_pass": "automatically_continue_folds_1_through_4_unchanged",
            "metrics_used_for_continuation": False,
            "leaderboard_promotion": False,
        },
        "cuda_training_environment_preflight": gpu_environment,
        "timing_exclusion_evidence": {
            "shared_lock_path": str(SHARED_TIMING_LOCK.relative_to(ROOT)),
            "host_gpu_lease_path": str(GPU_LEASE_PATH),
            "host_gpu_lease_held_for_entire_cuda_preflight": True,
            "shared_lock_held_for_entire_cuda_preflight": True,
            "no_live_timing_workers_at_lock_acquisition": True,
            "s84_status_audit": s84_status_audit(),
        },
        "source_data_provenance": {
            "upstream_repository_commit": json.loads(STATIC_SOURCE_HASHES.read_text(encoding="utf-8"))["upstream_repository_commit"],
            "upstream_license_status": "unresolved_in_original_repository; no QASM files are bundled; public redistribution review required",
            "runner_uses_bundled_static_DAG_pack": True,
        },
        "expected_methods_if_authorized": ["training-fold median", "six-feature standardized Ridge(alpha=1.0)", "three-layer TransformerConv graph adaptation"],
        "graph_configuration_if_authorized": {"seeds": list(SEEDS), "epochs": 500, "batch_size": 32, "optimizer": "Adam(lr=0.0005,weight_decay=0.0001)", "precision": "float32", "early_stopping": False, "HPO": False, "target_transform": "log1p(seconds)"},
        "family_support_by_primary_fold_unseen": family_support,
        "input_sha256": hashes,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "target_table_sha256": sha256_file(target_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_authorized": False,
    }
    write_json(output_dir / "preflight_manifest.json", manifest)
    qa = {
        "artifact_id": f"{RUNNER_ID}-preflight-qa",
        "status": "PASS",
        "input_hashes_match_provenance": True,
        "unique_hashes": len(targets),
        "finite_runtime_labels": 144,
        "quality_pass_finite": 142,
        "quality_failed_finite_but_included": 2,
        "adapter_error_unavailable_unimputed": 6,
        "fold_hash_join_complete": True,
        "hashes_cross_fold": 0,
        "family_unseen_in_primary_fold": False,
        "source_dag_records": len(graph_records),
        "measurement_stripped_depth_features": True,
        "parameter_values_used": False,
        "shots_used": False,
        "family_or_source_identity_used_as_feature": False,
        "fold0_assigned_hashes": manifest["fold0_unique_hash_count"],
        "fold0_labeled_hashes": manifest["fold0_labeled_hash_count"],
        "training_performed": False,
        "simulator_timing_performed": False,
        "target_table_sha256": manifest["target_table_sha256"],
        "target_clock_id": TARGET_ID,
    }
    write_json(output_dir / "preflight_qa.json", qa)
    print(json.dumps({"manifest": manifest, "qa": qa}, indent=2, sort_keys=True))


def _torch_graph_model(node_width: int, global_width: int):
    import torch
    from torch_geometric.nn import TransformerConv, global_mean_pool

    class GraphRuntimeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv0 = TransformerConv(node_width, 64)
            self.conv1 = TransformerConv(64, 64)
            self.conv2 = TransformerConv(64, 64)
            self.gf1 = torch.nn.Linear(global_width, 64)
            self.gf2 = torch.nn.Linear(64, 64)
            self.head1 = torch.nn.Linear(128, 512)
            self.head2 = torch.nn.Linear(512, 512)
            self.head3 = torch.nn.Linear(512, 128)
            self.head4 = torch.nn.Linear(128, 1)

        def forward(self, batch):
            x = torch.relu(self.conv0(batch.x, batch.edge_index))
            x = torch.relu(self.conv1(x, batch.edge_index))
            x = torch.relu(self.conv2(x, batch.edge_index))
            x = global_mean_pool(x, batch.batch)
            g = torch.relu(self.gf1(batch.global_features))
            g = torch.relu(self.gf2(g))
            x = torch.cat([x, g], dim=1)
            x = torch.relu(self.head1(x))
            x = torch.relu(self.head2(x))
            x = torch.relu(self.head3(x))
            return self.head4(x).reshape(-1)

    return GraphRuntimeModel


def validate_cuda_environment_match(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    """Require the training process to match the CUDA environment frozen at preflight."""
    required = (
        "python_executable", "torch", "torch_cuda_runtime", "torch_geometric",
        "cuda_available", "device_count", "device_name", "device_capability",
        "cuda_visible_devices", "pip_freeze_sha256", "pip_freeze_tool",
    )
    missing = [key for key in required if key not in expected or key not in actual]
    if missing:
        raise ValueError("CUDA environment record missing pinned field(s): " + ", ".join(missing))
    mismatches = [key for key in required if expected[key] != actual[key]]
    if mismatches:
        raise ValueError("CUDA environment differs from frozen preflight: " + ", ".join(mismatches))
    if actual["cuda_available"] is not True or int(actual["device_count"]) < 1:
        raise ValueError("CUDA training environment is unavailable; CPU fallback is forbidden")


def validate_fold_integrity(
    output_dir: Path,
    fold: int,
    fold_by_hash: dict[str, int],
    targets: dict[str, dict[str, Any]],
    expected_cuda_env: dict[str, Any],
) -> dict[str, Any]:
    """Validate only technical integrity; intentionally never reads metric values."""
    fold_dir = output_dir / f"fold_{fold}"
    prediction_path = fold_dir / f"fold{fold}_predictions.csv"
    qa_path = fold_dir / f"fold{fold}_qa.json"
    environment_path = fold_dir / "environment.json"
    for path in (prediction_path, qa_path, environment_path):
        if not path.is_file():
            raise ValueError(f"fold {fold} technical artifact is missing: {path.name}")

    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if qa.get("status") != "PASS" or qa.get("fold") != fold:
        raise ValueError(f"fold {fold} technical QA did not pass")
    if qa.get("cuda_pyg_smoke") != "PASS" or qa.get("train_test_hash_overlap") != 0:
        raise ValueError(f"fold {fold} CUDA smoke or leakage check failed")
    if qa.get("seed_set") != list(SEEDS):
        raise ValueError(f"fold {fold} did not retain the fixed seed set")
    if qa.get("prediction_sha256") != sha256_file(prediction_path):
        raise ValueError(f"fold {fold} prediction file does not match its QA hash")
    expected_methods = ["train_fold_median", "ridge_alpha_1", *[f"graph_seed_{s}" for s in SEEDS], "graph_median_three_seeds"]
    if qa.get("prediction_methods") != expected_methods:
        raise ValueError(f"fold {fold} omitted or changed a frozen baseline/prediction method")

    rows = read_csv(prediction_path)
    expected_test = {digest for digest, assigned_fold in fold_by_hash.items() if assigned_fold == fold}
    actual_hashes = [row.get("source_qasm_sha256", "") for row in rows]
    if len(actual_hashes) != len(set(actual_hashes)) or set(actual_hashes) != expected_test:
        raise ValueError(f"fold {fold} prediction rows do not exactly cover unique assigned hashes")
    if any(row.get("fold") != str(fold) for row in rows):
        raise ValueError(f"fold {fold} prediction rows contain a wrong fold ID")
    if set(targets) != set(fold_by_hash):
        raise ValueError("target table and frozen C44 hash envelope differ")
    train_hashes = {digest for digest, assigned_fold in fold_by_hash.items() if assigned_fold != fold}
    if train_hashes & expected_test:
        raise ValueError(f"fold {fold} has exact-hash train/test overlap")
    expected_train_finite = sum(targets[digest]["target_status"] == "runtime_observed" for digest in train_hashes)
    expected_finite_test = sum(targets[digest]["target_status"] == "runtime_observed" for digest in expected_test)
    if qa.get("train_finite_hashes") != expected_train_finite:
        raise ValueError(f"fold {fold} training-label count differs from frozen targets")
    if qa.get("test_assigned_hashes") != len(expected_test) or qa.get("test_finite_label_hashes") != expected_finite_test or qa.get("test_unavailable_hashes") != len(expected_test) - expected_finite_test:
        raise ValueError(f"fold {fold} target coverage counts differ from frozen hashes")
    if len(rows) != len(expected_test) or not qa.get("predictions_cover_all_assigned_test_hashes"):
        raise ValueError(f"fold {fold} prediction coverage check failed")

    prediction_columns = [f"pred_{method}_seconds" for method in expected_methods]
    for row in rows:
        digest = row["source_qasm_sha256"]
        target = targets[digest]
        if row.get("target_status") != target["target_status"]:
            raise ValueError(f"fold {fold} target status changed for {digest}")
        if row.get("quality_status_separate_audit_only") != target["quality_status_separate"]:
            raise ValueError(f"fold {fold} quality-status audit field changed for {digest}")
        raw_target = row.get("target_seconds", "")
        if target["target_status"] == "runtime_observed":
            if not raw_target or not math.isfinite(float(raw_target)) or not math.isclose(float(raw_target), float(target["target_seconds"]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"fold {fold} finite target changed for {digest}")
        elif raw_target:
            raise ValueError(f"fold {fold} unavailable target was imputed for {digest}")
        for column in prediction_columns:
            try:
                prediction = float(row[column])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"fold {fold} has a missing/non-numeric prediction in {column}") from exc
            if not math.isfinite(prediction) or prediction < 0:
                raise ValueError(f"fold {fold} has a non-finite/negative prediction in {column}")

    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    actual_cuda_env = environment.get("cuda_training_environment")
    if not isinstance(actual_cuda_env, dict):
        raise ValueError(f"fold {fold} environment.json lacks CUDA environment fingerprint")
    validate_cuda_environment_match(expected_cuda_env, actual_cuda_env)
    return {
        "fold": fold,
        "assigned_hashes": len(expected_test),
        "finite_label_hashes": expected_finite_test,
        "unavailable_hashes": len(expected_test) - expected_finite_test,
        "prediction_rows": len(rows),
        "methods": expected_methods,
        "cuda_environment": actual_cuda_env,
        "technical_qa": "PASS",
        "metrics_quality_consulted": False,
    }


def metric_bundle(actual: list[float], predicted: list[float]) -> dict[str, float | int | None]:
    import numpy as np

    if len(actual) != len(predicted) or not actual:
        return {"n": len(actual), "mae_seconds": None, "medae_seconds": None, "mae_log1p_seconds": None, "r2_seconds": None, "p90_abs_error_seconds": None, "p99_abs_error_seconds": None, "max_abs_error_seconds": None}
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    e = np.abs(y - p)
    residual = y - p
    denominator = float(np.square(y - y.mean()).sum())
    return {
        "n": int(len(y)),
        "mae_seconds": float(e.mean()),
        "medae_seconds": float(np.median(e)),
        "mae_log1p_seconds": float(np.abs(np.log1p(y) - np.log1p(np.maximum(p, 0))).mean()),
        "r2_seconds": float(1 - np.square(residual).sum() / denominator) if denominator else None,
        "p90_abs_error_seconds": float(np.quantile(e, .90)),
        "p99_abs_error_seconds": float(np.quantile(e, .99)),
        "max_abs_error_seconds": float(e.max()),
    }


@with_compute_exclusion
def fit_fold(output_dir: Path, fold: int) -> None:
    if fold not in FROZEN_FOLDS:
        raise ValueError(f"fold must be one of {FROZEN_FOLDS}, got {fold}")
    preflight = output_dir / "preflight_manifest.json"
    preflight_qa = output_dir / "preflight_qa.json"
    if not preflight.is_file() or not preflight_qa.is_file():
        raise SystemExit("run --action preflight first")
    manifest = json.loads(preflight.read_text(encoding="utf-8"))
    qa = json.loads(preflight_qa.read_text(encoding="utf-8"))
    if qa.get("status") != "PASS" or manifest.get("training_authorized") is not False:
        raise SystemExit("preflight is not PASS or training authorization boundary changed")
    if input_hashes() != manifest["input_sha256"]:
        raise SystemExit("a frozen input changed after preflight; rebuild in a new output directory")
    if manifest.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise SystemExit("runner code changed after preflight; regenerate preflight in a new output directory")
    target_table_path = output_dir / "hash_targets_and_static_features.csv"
    if not target_table_path.is_file() or sha256_file(target_table_path) != manifest.get("target_table_sha256") or qa.get("target_table_sha256") != manifest.get("target_table_sha256"):
        raise SystemExit("preflight target table is missing or changed; regenerate in a new output directory")
    gates = json.loads(SIM_GATES.read_text(encoding="utf-8"))
    try:
        authorization = validate_mps_execution_authorization(gates)
        timing_exclusion_evidence = {"shared_lock_held": True, "host_gpu_lease_held": True, "host_gpu_lease_path": str(GPU_LEASE_PATH), "no_live_timing_workers_at_lock_acquisition": True, "s84_status_audit": s84_status_audit()}
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"MPS CUDA checkpoint remains blocked: {exc}") from exc
    fold_dir = output_dir / f"fold_{fold}"
    if fold_dir.exists():
        raise SystemExit(f"refusing to overwrite existing fold artifact: {fold_dir}")
    import numpy as np
    import torch
    import torch_geometric
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
    live_cuda_environment = {
        "python_executable": sys.executable,
        "torch": str(torch.__version__),
        "torch_cuda_runtime": str(torch.version.cuda),
        "torch_geometric": str(torch_geometric.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }
    freeze_bytes, freeze_tool = package_freeze_snapshot()
    live_cuda_environment["pip_freeze_sha256"] = sha256_bytes(freeze_bytes)
    live_cuda_environment["pip_freeze_tool"] = freeze_tool
    validate_cuda_environment_match(manifest["cuda_training_environment_preflight"], live_cuda_environment)
    torch.set_num_threads(2)
    panel = read_csv(PANEL)
    raw_rows = read_csv(RAW)
    c44_rows = read_csv(FOLDS)
    fold_by_hash = frozen_fold_map(c44_rows)
    records = load_topology()
    _, target_rows, _, feature_by_hash = load_inputs()
    targets = {r["source_qasm_sha256"]: r for r in target_rows}
    all_hashes = sorted(records)
    train_hashes = [h for h in all_hashes if fold_by_hash[h] != fold and targets[h]["target_status"] == "runtime_observed"]
    test_hashes = [h for h in all_hashes if fold_by_hash[h] == fold]
    if not train_hashes or not test_hashes or set(train_hashes) & set(test_hashes):
        raise ValueError(f"fold-{fold} exact-hash train/test partition invalid")

    vocab = sorted({node["operation"] for h in train_hashes for node in feature_by_hash[h]["graph"]["nodes"]})
    vocab_map = {name: i for i, name in enumerate(vocab)}
    unk_index = len(vocab)
    node_width = len(vocab) + 1 + 11
    global_matrix = {h: np.log1p(np.asarray([feature_by_hash[h]["globals"][name] for name in GLOBAL_NAMES], dtype=np.float32)) for h in all_hashes}
    x_train = np.stack([global_matrix[h] for h in train_hashes])
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0

    def data_for(digest: str, with_y: bool):
        graph = feature_by_hash[digest]["graph"]
        nodes = graph["nodes"]
        x = torch.tensor(
            [encode_node_features(node, vocab_map, unk_index, int(graph["num_qubits"])) for node in nodes],
            dtype=torch.float32,
        )
        edges = graph["edge_src"]
        edge_index = torch.tensor([edges, graph["edge_dst"]], dtype=torch.long) if edges else torch.empty((2, 0), dtype=torch.long)
        g = ((global_matrix[digest] - mean) / std).reshape(1, -1)
        data = Data(x=x, edge_index=edge_index, global_features=torch.as_tensor(g, dtype=torch.float32))
        if with_y:
            data.y = torch.tensor(math.log1p(float(targets[digest]["target_seconds"])), dtype=torch.float32)
        return data

    model_class = _torch_graph_model(node_width, len(GLOBAL_NAMES))
    device = torch.device("cuda:0")
    # Minimal CUDA + PyG forward/backward smoke; this is not simulator timing.
    smoke = model_class().to(device)
    from torch_geometric.data import Batch
    smoke_example = data_for(train_hashes[0], True)
    smoke_example.global_features = torch.zeros((1, len(GLOBAL_NAMES)), dtype=torch.float32)
    smoke_batch = Batch.from_data_list([smoke_example]).to(device)
    out = smoke(smoke_batch)
    out.sum().backward()
    torch.cuda.synchronize(device)
    if not torch.isfinite(out).all().item():
        raise RuntimeError("CUDA/PyG smoke produced non-finite output")
    del smoke, smoke_batch, out
    torch.cuda.empty_cache()

    y_train = [float(targets[h]["target_seconds"]) for h in train_hashes]
    train_median = statistics.median(y_train)
    ridge_scaler = StandardScaler().fit(x_train)
    ridge = Ridge(alpha=1.0).fit(ridge_scaler.transform(x_train), np.log1p(np.asarray(y_train)))
    x_test = np.stack([global_matrix[h] for h in test_hashes])
    ridge_predictions = np.maximum(0.0, np.expm1(ridge.predict(ridge_scaler.transform(x_test))))
    prediction_by_method: dict[str, dict[str, float]] = {
        "train_fold_median": {h: float(train_median) for h in test_hashes},
        "ridge_alpha_1": {h: float(p) for h, p in zip(test_hashes, ridge_predictions)},
    }

    seed_predictions: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        loader = DataLoader([data_for(h, True) for h in train_hashes], batch_size=32, shuffle=True)
        for epoch in range(500):
            model.train()
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(batch)
                loss = torch.nn.functional.mse_loss(pred, batch.y.reshape(-1))
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"non-finite loss: seed={seed} epoch={epoch}")
                loss.backward()
                optimizer.step()
        model.eval()
        test_data = [data_for(h, False) for h in test_hashes]
        predicted: list[float] = []
        with torch.no_grad():
            for batch in DataLoader(test_data, batch_size=32, shuffle=False):
                batch = batch.to(device)
                predicted.extend(torch.expm1(model(batch)).clamp_min(0).cpu().tolist())
        if len(predicted) != len(test_hashes) or not all(math.isfinite(x) and x >= 0 for x in predicted):
            raise RuntimeError(f"invalid graph predictions for seed {seed}")
        seed_predictions[seed] = {h: float(p) for h, p in zip(test_hashes, predicted)}
        del model, optimizer, loader, test_data
        torch.cuda.empty_cache()
    for h in test_hashes:
        prediction_by_method.setdefault("graph_median_three_seeds", {})[h] = statistics.median(seed_predictions[s][h] for s in SEEDS)
        for seed in SEEDS:
            prediction_by_method.setdefault(f"graph_seed_{seed}", {})[h] = seed_predictions[seed][h]

    predictions: list[dict[str, Any]] = []
    for digest in test_hashes:
        target = targets[digest]
        row: dict[str, Any] = {
            "source_qasm_sha256": digest,
            "fold": fold,
            "target_status": target["target_status"],
            "target_seconds": target["target_seconds"],
            "quality_status_separate_audit_only": target["quality_status_separate"],
        }
        for method in ["train_fold_median", "ridge_alpha_1", *[f"graph_seed_{s}" for s in SEEDS], "graph_median_three_seeds"]:
            row[f"pred_{method}_seconds"] = prediction_by_method[method][digest]
        predictions.append(row)
    fold_dir.mkdir()
    pred_path = fold_dir / f"fold{fold}_predictions.csv"
    write_csv(pred_path, list(predictions[0]), predictions)
    env = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda_runtime": str(torch.version.cuda),
        "torch_geometric": str(torch_geometric.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_training_environment": live_cuda_environment,
        "cpu_threads": torch.get_num_threads(),
        "cuda_pyg_smoke": "PASS",
        "training_authorization_gate": "method_scoped_conditional_authorization_verified",
        "training_authorization_scope": authorization,
        "timing_exclusion_evidence": timing_exclusion_evidence,
        "pip_freeze_sha256": live_cuda_environment["pip_freeze_sha256"],
        "pip_freeze_tool": live_cuda_environment["pip_freeze_tool"],
    }
    write_json(fold_dir / "environment.json", env)
    write_json(fold_dir / f"fold{fold}_qa.json", {
        "status": "PASS",
        "fold": fold,
        "train_test_hash_overlap": 0,
        "train_finite_hashes": len(train_hashes),
        "test_assigned_hashes": len(test_hashes),
        "test_finite_label_hashes": sum(targets[h]["target_status"] == "runtime_observed" for h in test_hashes),
        "test_unavailable_hashes": sum(targets[h]["target_status"] != "runtime_observed" for h in test_hashes),
        "test_hashes_unique": len({r["source_qasm_sha256"] for r in predictions}) == len(predictions),
        "predictions_cover_all_assigned_test_hashes": len(predictions) == len(test_hashes),
        "prediction_methods": ["train_fold_median", "ridge_alpha_1", *[f"graph_seed_{s}" for s in SEEDS], "graph_median_three_seeds"],
        "seed_set": list(SEEDS),
        "quality_failed_targets_in_fit_and_score_when_finite": True,
        "unavailable_targets_not_imputed": all(targets[h]["target_seconds"] == "" for h in test_hashes if targets[h]["target_status"] != "runtime_observed"),
        "cuda_pyg_smoke": "PASS",
        "authorization_scope": authorization,
        "timing_exclusion_evidence": timing_exclusion_evidence,
        "prediction_sha256": sha256_file(pred_path),
        "fold0_checkpoint_rule": "technical_QA_only_no_metric_quality_gate" if fold == 0 else "not_applicable",
        "metrics_quality_consulted": False,
    })


def aggregate_five_fold_oof(
    output_dir: Path,
    fold_by_hash: dict[str, int],
    targets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Write the five-fold OOF envelope and metrics on the common finite-label rows."""
    combined: list[dict[str, str]] = []
    fold_file_hashes: dict[str, str] = {}
    for fold in FROZEN_FOLDS:
        path = output_dir / f"fold_{fold}" / f"fold{fold}_predictions.csv"
        if not path.is_file():
            raise ValueError(f"cannot aggregate: missing fold {fold} predictions")
        rows = read_csv(path)
        fold_file_hashes[str(fold)] = sha256_file(path)
        combined.extend(rows)
    hashes = [row["source_qasm_sha256"] for row in combined]
    if len(hashes) != 150 or len(set(hashes)) != 150 or set(hashes) != set(fold_by_hash):
        raise ValueError("five-fold OOF rows must cover each of the 150 frozen hashes exactly once")
    if any(int(row["fold"]) != fold_by_hash[row["source_qasm_sha256"]] for row in combined):
        raise ValueError("five-fold OOF output contains a row assigned to the wrong C44 fold")
    methods = ["train_fold_median", "ridge_alpha_1", *[f"graph_seed_{s}" for s in SEEDS], "graph_median_three_seeds"]
    finite = [row for row in combined if targets[row["source_qasm_sha256"]]["target_status"] == "runtime_observed"]
    unavailable = [row for row in combined if targets[row["source_qasm_sha256"]]["target_status"] != "runtime_observed"]
    if len(finite) != 144 or len(unavailable) != 6:
        raise ValueError(f"five-fold OOF expected 144 finite targets and 6 unavailable; got {len(finite)}/{len(unavailable)}")
    if sum(row["quality_status_separate_audit_only"] == "quality_pass" for row in finite) != 142 or sum(row["quality_status_separate_audit_only"] == "quality_failed" for row in finite) != 2:
        raise ValueError("five-fold OOF quality-status accounting differs from frozen source targets")
    metrics = []
    for method in methods:
        column = f"pred_{method}_seconds"
        values = [float(row[column]) for row in finite]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f"five-fold OOF has invalid predictions for {method}")
        metrics.append({
            "method_id": method,
            **metric_bundle(
                [float(row["target_seconds"]) for row in finite],
                values,
            ),
        })
    metrics_by_fold = []
    for fold in FROZEN_FOLDS:
        fold_finite = [row for row in finite if int(row["fold"]) == fold]
        fold_methods = []
        for method in methods:
            column = f"pred_{method}_seconds"
            fold_methods.append({
                "method_id": method,
                **metric_bundle(
                    [float(row["target_seconds"]) for row in fold_finite],
                    [float(row[column]) for row in fold_finite],
                ),
            })
        metrics_by_fold.append({"fold": fold, "finite_score_rows": len(fold_finite), "methods": fold_methods})
    combined.sort(key=lambda row: row["source_qasm_sha256"])
    pred_path = output_dir / "five_fold_oof_predictions.csv"
    summary_path = output_dir / "five_fold_oof_summary.json"
    if pred_path.exists() or summary_path.exists():
        raise SystemExit("refusing to overwrite existing five-fold aggregate artifacts")
    write_csv(pred_path, list(combined[0]), combined)
    summary = {
        "artifact_id": f"{RUNNER_ID}-five-fold-oof",
        "status": "five_fold_oof_complete_not_promoted",
        "method_claim": "fixed-configuration local MPS runtime-only method adaptation; not a Family-Aware reproduction",
        "target_clock_id": TARGET_ID,
        "split_authority": "C44 frozen exact-QASM hash folds; no fold reassignment",
        "folds": list(FROZEN_FOLDS),
        "seeds": list(SEEDS),
        "prediction_methods": methods,
        "coverage": {
            "unique_hashes": len(set(hashes)),
            "assigned_hashes": len(fold_by_hash),
            "finite_runtime_labels": len(finite),
            "quality_pass_finite": 142,
            "quality_failed_finite_retained": 2,
            "unavailable_retained_unimputed": len(unavailable),
        },
        "metrics_on_common_finite_label_rows_only": metrics,
        "metrics_by_outer_fold_after_all_folds_complete": metrics_by_fold,
        "family_ood": False,
        "family_aware_claim": False,
        "leaderboard_promotion": False,
        "fold0_checkpoint_gate": "technical_integrity_only; metric values never controlled continuation",
        "fold_prediction_sha256": fold_file_hashes,
        "oof_predictions_sha256": sha256_file(pred_path),
        "quality_status_used_as_predictor": False,
        "unavailable_target_imputation": False,
    }
    write_json(summary_path, summary)
    return summary


@with_compute_exclusion
def run_five_fold_oof(output_dir: Path) -> None:
    """Run fixed seeds/baselines for all folds; fold 0 gates only on technical QA."""
    preflight = output_dir / "preflight_manifest.json"
    preflight_qa = output_dir / "preflight_qa.json"
    if not preflight.is_file() or not preflight_qa.is_file():
        raise SystemExit("run --action preflight first while simulator timing is idle")
    manifest = json.loads(preflight.read_text(encoding="utf-8"))
    qa = json.loads(preflight_qa.read_text(encoding="utf-8"))
    if qa.get("status") != "PASS" or manifest.get("training_authorized") is not False:
        raise SystemExit("preflight is not PASS or training authorization boundary changed")
    if input_hashes() != manifest["input_sha256"]:
        raise SystemExit("a frozen input changed after preflight; build a new preflight directory")
    if manifest.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise SystemExit("runner code changed after preflight; regenerate preflight in a new output directory")
    target_table_path = output_dir / "hash_targets_and_static_features.csv"
    if not target_table_path.is_file() or sha256_file(target_table_path) != manifest.get("target_table_sha256") or qa.get("target_table_sha256") != manifest.get("target_table_sha256"):
        raise SystemExit("preflight target table is missing or changed; build a new preflight directory")
    try:
        validate_mps_execution_authorization(json.loads(SIM_GATES.read_text(encoding="utf-8")))
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"MPS five-fold OOF remains blocked: {exc}") from exc

    _, target_rows, _, _ = load_inputs()
    targets = {row["source_qasm_sha256"]: row for row in target_rows}
    fold_by_hash = frozen_fold_map(read_csv(FOLDS))
    if len(targets) != 150 or set(targets) != set(fold_by_hash):
        raise ValueError("frozen MPS target and C44 hash envelopes must match at 150 hashes")
    expected_cuda_env = manifest.get("cuda_training_environment_preflight")
    if not isinstance(expected_cuda_env, dict):
        raise ValueError("preflight is missing the frozen CUDA environment")

    fold_reports = []
    for fold in FROZEN_FOLDS:
        existing_fold = output_dir / f"fold_{fold}"
        resumed = existing_fold.exists()
        # Resume only from a fully validated frozen fold; do not overwrite or
        # partially repair any prior fold outputs.
        if not resumed:
            fit_fold(output_dir, fold)
        report = validate_fold_integrity(output_dir, fold, fold_by_hash, targets, expected_cuda_env)
        if resumed:
            print(json.dumps({"fold": fold, "status": "resume_existing_technical_qa_pass", "metrics_quality_consulted": False}, sort_keys=True))
        fold_reports.append(report)
        if fold == 0:
            # This is intentionally metric-blind: the validator does not read any
            # fold metrics, scores, ranks, or seed comparisons before continuing.
            print(json.dumps({
                "fold0_checkpoint": "TECHNICAL_QA_PASS",
                "continuation": "automatically_start_folds_1_through_4_unchanged",
                "metrics_quality_consulted": False,
                "fold0_technical_qa": report,
            }, sort_keys=True))
    summary = aggregate_five_fold_oof(output_dir, fold_by_hash, targets)
    write_json(output_dir / "five_fold_execution_receipt.json", {
        "status": "complete",
        "execution_scope": "fixed_split_five_fold_oof",
        "folds": fold_reports,
        "final_summary_sha256": sha256_file(output_dir / "five_fold_oof_summary.json"),
        "final_predictions_sha256": summary["oof_predictions_sha256"],
        "metrics_quality_used_as_continuation_gate": False,
        "timing_exclusion_evidence": {
            "shared_lock_path": str(SHARED_TIMING_LOCK.relative_to(ROOT)),
            "host_gpu_lease_path": str(GPU_LEASE_PATH),
            "host_gpu_lease_held_for_entire_five_fold_fit": True,
            "shared_lock_held_for_entire_five_fold_fit": True,
            "no_live_timing_workers_at_lock_acquisition": True,
            "s84_completion_required": False,
        },
        "leaderboard_promotion": False,
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "fit-five-fold-oof"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.action == "preflight":
        materialize(output)
    else:
        run_five_fold_oof(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
