#!/usr/bin/env python3
"""Prepare and (when separately invoked) measure a joint MPS runtime/quality panel.

No CUDA-Q import occurs for ``prepare``.  ``measure --scope pilot10`` is the
first timing action; ``measure --scope full`` is fail-closed unless the linked
pilot report passes the frozen technical gate.  There is deliberately no
training action in this runner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_mps_fixed_chi16_runtime_adaptation_v1 as s85


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "benchmark_v1/protocol/family_aware_joint_runtime_quality_mps_v1.json"
PANEL = ROOT / "artifacts/benchmark_v1/sim_common_q16_manifest_20260927/sim_common_q16_manifest.csv"
FOLDS = ROOT / "artifacts/benchmark_v1/c44_aer_q16_full_panel_evaluation_20260927/aer_q16_reduced_warm.csv"
TOPOLOGY = ROOT / "artifacts/benchmark_v3/simulator/azizov_common_core_gnn_v1_materialization/source_dag_topology.jsonl"
STATIC_HASHES = TOPOLOGY.parent / "source_hashes.json"
COMMON_MPS_BRIDGE = ROOT / "benchmark_v1/scripts/run_common_cudaq_mps.py"
CUDAQ_BRIDGE_MANIFEST = ROOT / "benchmark_v1/execution/manifests/cudaq_verified_bridge_v1.json"
DEFAULT_PLAN = ROOT / "work/family_aware_joint_mps_ladder_v1/preflight"
DEFAULT_OUTPUT = ROOT / "artifacts/benchmark_v3/simulator/family_aware_joint_mps_ladder_v1"
RUNNER_ID = "family-aware-joint-mps-runtime-quality-ladder-v1"
RUNG_CHI = (2, 4, 8, 16, 32, 64)
SESSIONS = 3
WARMUPS = 3
REPETITIONS = 5
QUALITY_THRESHOLD = 0.99
EXPECTED_FOLD_COUNTS = {0: 37, 1: 28, 2: 26, 3: 25, 4: 34}
INPUT_FEATURES = list(s85.GLOBAL_NAMES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def cudaq_distribution_versions(cudaq_module: Any) -> dict[str, str]:
    """Return installed distribution versions separately from CUDA-Q's build banner.

    CUDA-Q exposes ``__version__`` as a human-readable string that can include
    its source URL and commit, while the frozen environment pins the wheel
    version.  Comparing those two representations directly rejects a valid
    environment before any simulator work begins.
    """
    package_name = str(getattr(cudaq_module, "__name__", "cudaq")).split(".", 1)[0]
    distributions = importlib.metadata.packages_distributions().get(package_name, [])
    versions: dict[str, str] = {}
    for distribution_name in distributions:
        try:
            versions[distribution_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    if not versions:
        raise RuntimeError(f"cannot resolve installed distribution version for imported package {package_name!r}")
    return versions


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


def unitary_kernel(qasm_path: Path, expected_hash: str, expected_width: int) -> tuple[Any, str, int]:
    """Minimal local adapter matching the audited common-MPS bridge semantics."""
    from qiskit import QuantumCircuit, qasm2
    from cudaq.contrib.qiskit_convert import from_qiskit

    if sha256_file(qasm_path) != expected_hash:
        raise ValueError("QASM hash differs from immutable common-panel manifest")
    circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
    if circuit.num_qubits != expected_width:
        raise ValueError("QASM width differs from immutable common-panel manifest")
    stripped = circuit.remove_final_measurements(inplace=False)
    if int(stripped.count_ops().get("measure", 0)):
        raise ValueError("nonterminal measurement: no unitary MPS-state adapter is declared")
    digest = sha256_json({
        "adapter": "cudaq.contrib.qiskit_convert.from_qiskit",
        "mode": "mps_state_strip_terminal_measurements_only",
        "canonical_qasm": qasm2.dumps(stripped),
    })
    return from_qiskit(stripped), digest, int(circuit.count_ops().get("measure", 0))


def state_fidelity(mps_state: Any, dense_state: Any, width: int) -> float:
    """Use the same little-endian amplitude alignment as the pinned MPS bridge."""
    basis = [format(index, f"0{width}b") for index in range(2**width)]
    little_endian_basis = [bits[::-1] for bits in basis]
    mps = [complex(value) for value in mps_state.amplitudes(little_endian_basis)]
    dense = [complex(value) for value in dense_state.to_numpy()]
    expected_length = 2**width
    if len(mps) != expected_length or len(dense) != expected_length:
        raise ValueError("state amplitude vector length differs from 2**width")
    if any(not math.isfinite(value.real) or not math.isfinite(value.imag) for value in (*mps, *dense)):
        raise ValueError("state amplitudes contain non-finite values")
    mps_norm = math.sqrt(sum(abs(value) ** 2 for value in mps))
    dense_norm = math.sqrt(sum(abs(value) ** 2 for value in dense))
    if not math.isfinite(mps_norm) or not math.isfinite(dense_norm) or mps_norm == 0 or dense_norm == 0:
        raise ValueError("statevector overlap input has invalid norm")
    overlap = sum((left.conjugate() / dense_norm) * (right / mps_norm) for left, right in zip(dense, mps))
    fidelity = float(abs(overlap) ** 2)
    if not math.isfinite(fidelity) or fidelity < -1e-12 or fidelity > 1.0 + 1e-8:
        raise ValueError("normalized state overlap is non-finite or outside physical range")
    return min(1.0, max(0.0, fidelity))


def validate_overlap_controls() -> dict[str, Any]:
    """CPU-only positive/negative/basis-order and invalid-state controls."""
    class FakeMps:
        def __init__(self, values: list[complex], expected_basis: list[str] | None = None):
            self.values = values
            self.expected_basis = expected_basis

        def amplitudes(self, bitstrings: list[str]) -> list[complex]:
            if self.expected_basis is not None and bitstrings != self.expected_basis:
                raise ValueError(f"amplitude basis order mismatch: {bitstrings}")
            return self.values

    class FakeDense:
        def __init__(self, values: list[complex]):
            self.values = values

        def to_numpy(self) -> list[complex]:
            return self.values

    basis = ["00", "10", "01", "11"]
    positive = state_fidelity(FakeMps([0j, 1 + 0j, 0j, 0j], basis), FakeDense([0j, 1 + 0j, 0j, 0j]), 2)
    negative = state_fidelity(FakeMps([1 + 0j, 0j, 0j, 0j], basis), FakeDense([0j, 1 + 0j, 0j, 0j]), 2)
    invalid: dict[str, bool] = {}
    for name, state in (("zero_norm", [0j] * 4), ("nonfinite", [complex(float("nan"), 0), 0j, 0j, 0j])):
        try:
            state_fidelity(FakeMps(state, basis), FakeDense([1 + 0j, 0j, 0j, 0j]), 2)
        except ValueError:
            invalid[name] = True
        else:
            invalid[name] = False
    if abs(positive - 1.0) > 1e-12 or abs(negative) > 1e-12 or not all(invalid.values()):
        raise ValueError("state-fidelity controls failed")
    return {"positive_fidelity": positive, "negative_fidelity": negative,
            "little_endian_basis_order": basis, "invalid_state_rejected": invalid,
            "status": "PASS"}


def validate_cudaq_basis_canary(cudaq: Any, QuantumCircuit: Any, from_qiskit: Any) -> dict[str, Any]:
    """Exercise actual CUDA-Q MPS/dense amplitude ordering outside timed samples."""
    results = []
    states = []
    ir_digests = []
    for qubit, expected_index in ((0, 1), (1, 2)):
        circuit = QuantumCircuit(2)
        circuit.x(qubit)
        kernel = from_qiskit(circuit)
        ir_digests.append(sha256_json({"adapter": "cudaq.contrib.qiskit_convert.from_qiskit",
                                       "qasm": str(circuit)}))
        cudaq.set_target("tensornet-mps", option="fp64")
        mps = cudaq.get_state(kernel)
        cudaq.set_target("nvidia", option="fp64")
        dense = cudaq.get_state(kernel)
        little_endian_basis = [format(index, "02b")[::-1] for index in range(4)]
        mps_amplitudes = [complex(value) for value in mps.amplitudes(little_endian_basis)]
        dense_amplitudes = [complex(value) for value in dense.to_numpy()]
        expected = [0j] * 4
        expected[expected_index] = 1 + 0j
        if len(mps_amplitudes) != 4 or len(dense_amplitudes) != 4:
            raise ValueError("CUDA-Q canary amplitude vector is not length four")
        if any(not math.isfinite(value.real) or not math.isfinite(value.imag)
               for value in (*mps_amplitudes, *dense_amplitudes)):
            raise ValueError("CUDA-Q canary amplitudes contain non-finite values")
        if (any(abs(a - b) > 1e-12 for a, b in zip(mps_amplitudes, expected))
                or any(abs(a - b) > 1e-12 for a, b in zip(dense_amplitudes, expected))):
            raise ValueError(f"CUDA-Q amplitude basis ordering failed for X(q{qubit})")
        if abs(state_fidelity(mps, dense, 2) - 1.0) > 1e-12:
            raise ValueError(f"CUDA-Q MPS/dense positive control failed for X(q{qubit})")
        results.append({"prepared_qubit": qubit, "expected_basis_index": expected_index,
                        "positive_fidelity": state_fidelity(mps, dense, 2)})
        states.append((mps, dense))
    negative = state_fidelity(states[0][0], states[1][1], 2)
    if abs(negative) > 1e-12:
        raise ValueError("CUDA-Q orthogonal-state negative control failed")
    cudaq.set_target("tensornet-mps", option="fp64")
    return {"status": "PASS", "states": results, "negative_fidelity": negative,
            "ir_sha256": sha256_json(ir_digests), "timed_clock_included": False}


def family_components(panel_rows: list[dict[str, str]]) -> dict[str, str]:
    """Deterministic connected components for co-occurring family aliases."""
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(a: str, b: str) -> None:
        x, y = find(a), find(b)
        if x != y:
            low, high = sorted((x, y))
            parent[high] = low

    families_by_hash: dict[str, set[str]] = defaultdict(set)
    for row in panel_rows:
        if row["stratum"] != "core_q2_q9":
            continue
        digest, family = row["qasm_sha256"], row["family"].strip()
        if not digest or not family:
            raise ValueError("core panel has empty hash/family")
        families_by_hash[digest].add(family)
        find(family)
    for names in families_by_hash.values():
        ordered = sorted(names)
        for name in ordered[1:]:
            union(ordered[0], name)
    members: dict[str, list[str]] = defaultdict(list)
    for family in sorted(parent):
        members[find(family)].append(family)
    label = {name: json.dumps(sorted(names), ensure_ascii=False, separators=(",", ":"))
             for names in members.values() for name in names}
    result: dict[str, str] = {}
    for digest, names in families_by_hash.items():
        labels = {label[name] for name in names}
        if len(labels) != 1:
            raise ValueError(f"family alias components disagree for exact hash {digest}")
        result[digest] = next(iter(labels))
    return result


def deterministic_family_folds(component_by_hash: dict[str, str], fold_count: int = 5) -> dict[str, int]:
    """Balanced deterministic family-component holdout, secondary only."""
    hashes_by_component: dict[str, list[str]] = defaultdict(list)
    for digest, component in component_by_hash.items():
        hashes_by_component[component].append(digest)
    if len(hashes_by_component) < fold_count:
        raise ValueError("insufficient distinct family components for five-fold family holdout")
    groups = sorted(hashes_by_component.items(), key=lambda pair: (-len(pair[1]), pair[0]))
    loads = [0] * fold_count
    assignment: dict[str, int] = {}
    for component, hashes in groups:
        fold = min(range(fold_count), key=lambda index: (loads[index], index))
        loads[fold] += len(hashes)
        for digest in hashes:
            assignment[digest] = fold
    return assignment


def load_static_inputs() -> tuple[list[dict[str, str]], dict[str, int], dict[str, dict[str, Any]], dict[str, Any]]:
    required = (PROTOCOL, PANEL, FOLDS, TOPOLOGY, STATIC_HASHES, CUDAQ_BRIDGE_MANIFEST)
    absent = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if absent:
        raise FileNotFoundError("required source input(s) missing: " + ", ".join(absent))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "family-aware-joint-runtime-quality-mps-v1":
        raise ValueError("unexpected joint MPS protocol id")
    panel_all = read_csv(PANEL)
    core = [row for row in panel_all if row["stratum"] == "core_q2_q9"]
    by_hash: dict[str, dict[str, str]] = {}
    for row in core:
        digest = row["qasm_sha256"]
        width = int(row["width_qubits"])
        if width < 2 or width > 9:
            raise ValueError(f"core panel hash outside declared q2-q9 range: {digest}")
        if digest in by_hash:
            prior = by_hash[digest]
            if prior["width_qubits"] != row["width_qubits"]:
                raise ValueError(f"exact hash has conflicting widths: {digest}")
            prior["family_aliases"] = ",".join(sorted(set(prior["family_aliases"].split(",")) | {row["family"]}))
            continue
        by_hash[digest] = {**row, "family_aliases": row["family"]}
    if len(by_hash) != 150:
        raise ValueError(f"expected 150 unique core exact-QASM hashes; got {len(by_hash)}")
    c44_rows = read_csv(FOLDS)
    fold_by_hash = s85.frozen_fold_map(c44_rows)
    if set(fold_by_hash) != set(by_hash):
        raise ValueError("the selected exact-QASM hash set differs from frozen C44")
    topology = s85.load_topology()
    if set(topology) != set(by_hash):
        raise ValueError("static source-DAG pack does not cover exactly the core 150 hashes")
    source_hashes = json.loads(STATIC_HASHES.read_text(encoding="utf-8"))
    if sha256_file(TOPOLOGY) != source_hashes.get("source_dag_jsonl_sha256"):
        raise ValueError("source topology hash differs from its existing source_hashes pin")
    for input_row in by_hash.values():
        qasm_path = Path(input_row["source_root"]) / input_row["basename"]
        if not qasm_path.is_file() or sha256_file(qasm_path) != input_row["qasm_sha256"]:
            raise ValueError(f"source QASM absent or hash mismatch: {input_row['basename']}")
    hashes = {
        "protocol_sha256": sha256_file(PROTOCOL),
        "panel_manifest_sha256": sha256_file(PANEL),
        "c44_fold_source_sha256": sha256_file(FOLDS),
        "source_dag_topology_sha256": sha256_file(TOPOLOGY),
        "source_hashes_sha256": sha256_file(STATIC_HASHES),
        "measurement_runner_sha256": sha256_file(Path(__file__).resolve()),
        "shared_cudaq_measurement_bridge_sha256": sha256_file(COMMON_MPS_BRIDGE),
        "s85_static_feature_loader_sha256": sha256_file(Path(s85.__file__).resolve()),
        "cudaq_verified_bridge_manifest_sha256": sha256_file(CUDAQ_BRIDGE_MANIFEST),
    }
    return list(by_hash.values()), fold_by_hash, topology, {"protocol": protocol, "hashes": hashes, "family_components": family_components(core)}


def make_plan(output: Path) -> dict[str, Any]:
    panel, fold_by_hash, topology, loaded = load_static_inputs()
    protocol = loaded["protocol"]
    components = loaded["family_components"]
    overlap_controls = validate_overlap_controls()
    family_fold_by_hash = deterministic_family_folds(components)
    rows = []
    for item in sorted(panel, key=lambda row: row["qasm_sha256"]):
        features, _graph = s85.graph_features(topology[item["qasm_sha256"]])
        row = {
            "source_qasm_sha256": item["qasm_sha256"], "width_qubits": int(item["width_qubits"]),
            "c44_fold": fold_by_hash[item["qasm_sha256"]],
            "family_holdout_fold_diagnostic_only": family_fold_by_hash[item["qasm_sha256"]],
            "family_component_training_label_only": components[item["qasm_sha256"]],
            "source_path_for_preparation_not_model_input": item["source_path"],
            "panel_member": item["basename"],
            **{name: int(features[name]) for name in INPUT_FEATURES},
        }
        rows.append(row)
    folds = Counter(fold_by_hash.values())
    if dict(sorted(folds.items())) != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"C44 fold counts mismatch: {dict(sorted(folds.items()))}")
    attempts = []
    for row in rows:
        for chi in RUNG_CHI:
            for session in range(1, SESSIONS + 1):
                attempts.append({
                    "source_qasm_sha256": row["source_qasm_sha256"],
                    "width_qubits": row["width_qubits"], "c44_fold": row["c44_fold"],
                    "max_bond": chi, "session_id": f"session-{session}",
                })
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    feature_fields = ["source_qasm_sha256", "width_qubits", "c44_fold", "family_holdout_fold_diagnostic_only",
                      "family_component_training_label_only", "source_path_for_preparation_not_model_input", "panel_member", *INPUT_FEATURES]
    write_csv(output / "hash_feature_manifest.csv", feature_fields, rows)
    write_csv(output / "attempt_manifest.csv", ["source_qasm_sha256", "width_qubits", "c44_fold", "max_bond", "session_id"], attempts)
    manifest = {
        "artifact_id": "family-aware-joint-mps-ladder-preflight-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "runner_id": RUNNER_ID,
        "protocol_id": protocol["protocol_id"], "protocol_sha256": loaded["hashes"]["protocol_sha256"],
        "input_hashes": loaded["hashes"], "assigned_unique_exact_qasm_hashes": len(rows),
        "width_range_inclusive": [2, 9], "c44_unique_hash_fold_counts": {str(k): folds[k] for k in sorted(folds)},
        "family_component_count": len(set(components.values())),
        "family_holdout_hash_fold_counts_diagnostic_only": dict(sorted(Counter(family_fold_by_hash.values()).items())),
        "rungs": list(RUNG_CHI), "session_count_per_rung": SESSIONS,
        "planned_rung_configurations": len(rows) * len(RUNG_CHI),
        "planned_session_attempts": len(attempts),
        "pilot_hashes": sorted(row["source_qasm_sha256"] for row in rows)[:10],
        "pilot_session_attempts": 10 * len(RUNG_CHI) * SESSIONS,
        "planned_worker_processes": len(attempts),
        "cudaq_imported": False, "timing_performed": False, "training_performed": False,
        "model_feature_policy": "Only six static structural columns and fold-local classifier-predicted family may enter prediction. True family component column is training-label-only and diagnostic; it is not a prediction input.",
        "overlap_controls": overlap_controls,
        "files": {
            "hash_feature_manifest.csv": sha256_file(output / "hash_feature_manifest.csv"),
            "attempt_manifest.csv": sha256_file(output / "attempt_manifest.csv"),
        },
    }
    write_json(output / "preflight_manifest.json", manifest)
    return manifest


def reduce_ladder(attempt_rows: list[dict[str, Any]], expected_hashes: list[str]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in attempt_rows:
        by_cell[(row["source_qasm_sha256"], int(row["max_bond"]))].append(row)
    output: list[dict[str, Any]] = []
    expected_sessions = {f"session-{i}" for i in range(1, SESSIONS + 1)}
    for digest in expected_hashes:
        cell_values: dict[int, list[dict[str, Any]]] = {}
        technical_complete = True
        for chi in RUNG_CHI:
            rows = by_cell.get((digest, chi), [])
            if {row["session_id"] for row in rows} != expected_sessions or len(rows) != SESSIONS:
                technical_complete = False
                continue
            if any(row["status"] != "ok" or not math.isfinite(float(row["fidelity"]))
                   or row.get("basis_canary_status") != "PASS" for row in rows):
                technical_complete = False
            cell_values[chi] = rows
        ladder_ir_hashes = {row.get("ir_sha256") for rows in cell_values.values() for row in rows}
        if len(ladder_ir_hashes) != 1 or None in ladder_ir_hashes:
            technical_complete = False
        selected = None
        if technical_complete and len(cell_values) == len(RUNG_CHI):
            selected = next((chi for chi in RUNG_CHI if all(float(row["fidelity"]) >= QUALITY_THRESHOLD for row in cell_values[chi])), None)
        if not technical_complete:
            status = "incomplete_or_failed_ladder"
        elif selected is None:
            status = "quality_unattainable_on_declared_ladder"
        else:
            status = "quality_target_observed"
        runtimes = [] if selected is None else [statistics.median(float(value) for value in row["warm_seconds"]) for row in cell_values[selected]]
        output.append({
            "source_qasm_sha256": digest,
            "ladder_status": status,
            "minimum_passing_max_bond": selected if selected is not None else "",
            "quality_constrained_warm_get_state_seconds": statistics.median(runtimes) if runtimes else "",
            "quality_threshold": QUALITY_THRESHOLD,
            "selected_rung_session_count": SESSIONS if selected is not None else 0,
        })
    return output


def parse_gpu_memory() -> tuple[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=True,
    )
    first = result.stdout.strip().splitlines()[0].split(",")
    return int(first[0].strip()), int(first[1].strip())


def gpu_identity() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=15, check=True,
    )
    return result.stdout.strip().splitlines()[0]


def run_worker_process(
    command: list[str], timeout_seconds: float = 180.0,
    heartbeat: Any | None = None,
) -> tuple[str, str, dict[str, Any] | None, int]:
    """Run one isolated hash x chi x session, monitoring time and VRAM."""
    payload: dict[str, Any] | None = None
    max_used_mib = 0
    with tempfile.TemporaryFile(mode="w+t") as stdout_file, tempfile.TemporaryFile(mode="w+t") as stderr_file:
        process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, text=True)
        cell_started = time.monotonic()
        memory_limit_hit = False
        timed_out = False
        while process.poll() is None:
            if heartbeat is not None:
                heartbeat(max_used_mib)
            if time.monotonic() - cell_started >= timeout_seconds:
                timed_out = True
                process.kill()
                break
            current_free, current_used = parse_gpu_memory()
            max_used_mib = max(max_used_mib, current_used)
            if current_free < 4096 or current_used > 12288:
                memory_limit_hit = True
                process.kill()
                break
            time.sleep(0.25)
        process.wait()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout, stderr = stdout_file.read(), stderr_file.read()
    if memory_limit_hit:
        return "resource_limit", "GPU memory guard exceeded while worker active", None, max_used_mib
    if timed_out:
        return "timeout", f"hash/rung/session worker exceeded {timeout_seconds:g} seconds", None, max_used_mib
    line = next((value[len("WORKER_RESULT="):] for value in reversed(stdout.splitlines()) if value.startswith("WORKER_RESULT=")), None)
    try:
        payload = json.loads(line) if line else None
    except json.JSONDecodeError as exc:
        return "worker_error", f"malformed worker JSON: {exc}; {stderr[-600:] or stdout[-600:]}", None, max_used_mib
    if payload is None or process.returncode != 0:
        return "worker_error", stderr[-800:] or stdout[-800:], None, max_used_mib
    return str(payload.get("status", "worker_error")), str(payload.get("error", "")), payload, max_used_mib


def worker(args: argparse.Namespace) -> int:
    result: dict[str, Any] = {"status": "adapter_error", "error": "", "build_seconds": None, "session": {"session_id": args.session_id}}
    try:
        import cudaq
        cudaq_build = str(getattr(cudaq, "__version__", "unknown"))
        cudaq_versions = cudaq_distribution_versions(cudaq)
        result["cudaq_distribution_versions"] = cudaq_versions
        result["cudaq_version"] = next(iter(cudaq_versions.values())) if len(cudaq_versions) == 1 else ""
        result["cudaq_runtime_build"] = cudaq_build
        if args.expected_cudaq_version not in cudaq_versions.values():
            raise RuntimeError(
                f"CUDA-Q package version mismatch: expected {args.expected_cudaq_version}, "
                f"installed distributions are {cudaq_versions}; runtime build banner is {cudaq_build}"
            )

        if args.qiskit_site_packages:
            parser_site = Path(args.qiskit_site_packages).resolve()
            if not parser_site.is_dir():
                raise FileNotFoundError(f"Qiskit parser site-packages missing: {parser_site}")
            if str(parser_site) not in sys.path:
                sys.path.insert(0, str(parser_site))

        started = time.perf_counter()
        kernel, ir_sha, stripped_measurements = unitary_kernel(Path(args.qasm_path), args.qasm_sha256, args.width_qubits)
        import qiskit
        result["qiskit_version"] = str(qiskit.__version__)
        if result["qiskit_version"] != args.expected_qiskit_version:
            raise RuntimeError(f"Qiskit version mismatch: expected {args.expected_qiskit_version}, got {result['qiskit_version']}")
        result["build_seconds"] = time.perf_counter() - started
        result["ir_sha256"] = ir_sha
        result["terminal_measurements_stripped"] = stripped_measurements
        os.environ["CUDAQ_MPS_MAX_BOND"] = str(args.max_bond)
        os.environ["CUDAQ_MPS_ABS_CUTOFF"] = "1e-10"
        os.environ["CUDAQ_MPS_SVD_ALGO"] = "gesvdj"
        cudaq.set_target("tensornet-mps", option="fp64")
        started = time.perf_counter(); state = cudaq.get_state(kernel)
        first = time.perf_counter() - started
        for _ in range(WARMUPS):
            cudaq.get_state(kernel)
        warm = []
        for _ in range(REPETITIONS):
            started = time.perf_counter(); cudaq.get_state(kernel)
            warm.append(time.perf_counter() - started)
        result["session"].update({"first_seconds": first, "warm_seconds": warm})
        reference_started = time.perf_counter()
        cudaq.set_target("nvidia", option="fp64")
        dense = cudaq.get_state(kernel)
        result["quality_reference_seconds"] = time.perf_counter() - reference_started
        fidelity_started = time.perf_counter()
        result["session"]["fidelity"] = state_fidelity(state, dense, args.width_qubits)
        result["session"]["quality_extract_seconds"] = time.perf_counter() - fidelity_started
        canary_started = time.perf_counter()
        from qiskit import QuantumCircuit
        from cudaq.contrib.qiskit_convert import from_qiskit
        result["basis_canary"] = validate_cudaq_basis_canary(cudaq, QuantumCircuit, from_qiskit)
        result["basis_canary_seconds"] = time.perf_counter() - canary_started
        numeric = [first, *warm, result["session"]["fidelity"]]
        if len(warm) != REPETITIONS or any(not math.isfinite(float(value)) or float(value) < 0 for value in numeric):
            raise ValueError("timing/fidelity result has invalid length, non-finite, or negative values")
        result["status"] = "ok"
    except MemoryError as exc:
        result.update(status="resource_limit", error=f"MemoryError:{exc}")
    except Exception as exc:
        detail = f"{type(exc).__name__}:{exc}".splitlines()[0][:1000]
        low = detail.lower()
        status = "resource_limit" if any(token in low for token in ("out of memory", "memory allocation", "cuda error 2")) else "unsupported" if "unsupported" in low else "adapter_error"
        result.update(status=status, error=detail)
    print("WORKER_RESULT=" + json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


def read_plan(plan_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest_path = plan_dir / "preflight_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"preflight manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("runner_id") != RUNNER_ID or manifest.get("timing_performed") is not False:
        raise ValueError("invalid or already-executed preparation manifest")
    current = {
        "protocol_sha256": sha256_file(PROTOCOL),
        "panel_manifest_sha256": sha256_file(PANEL),
        "c44_fold_source_sha256": sha256_file(FOLDS),
        "source_dag_topology_sha256": sha256_file(TOPOLOGY),
        "source_hashes_sha256": sha256_file(STATIC_HASHES),
        "measurement_runner_sha256": sha256_file(Path(__file__).resolve()),
        "shared_cudaq_measurement_bridge_sha256": sha256_file(COMMON_MPS_BRIDGE),
        "s85_static_feature_loader_sha256": sha256_file(Path(s85.__file__).resolve()),
        "cudaq_verified_bridge_manifest_sha256": sha256_file(CUDAQ_BRIDGE_MANIFEST),
    }
    for name, digest in current.items():
        if manifest["input_hashes"].get(name) != digest:
            raise ValueError(f"preflight source changed after preparation: {name}")
    for name, digest in manifest.get("files", {}).items():
        if sha256_file(plan_dir / name) != digest:
            raise ValueError(f"preflight artifact changed after preparation: {name}")
    rows = read_csv(plan_dir / "hash_feature_manifest.csv")
    if len(rows) != 150 or len({row["source_qasm_sha256"] for row in rows}) != 150:
        raise ValueError("preflight feature manifest must contain 150 exact hashes")
    return manifest, rows


def validate_attempt_ledger(rows: list[dict[str, Any]]) -> set[tuple[str, int, str]]:
    """Fail closed on duplicated cells or malformed successful sessions."""
    seen: set[tuple[str, int, str]] = set()
    allowed_statuses = {"ok", "timeout", "resource_limit", "worker_error", "unsupported", "adapter_error", "technical_error"}
    for row in rows:
        key = (str(row["source_qasm_sha256"]), int(row["max_bond"]), str(row["session_id"]))
        if key in seen:
            raise ValueError(f"duplicate hash x chi x session record: {key}")
        seen.add(key)
        if row.get("attempt_terminal") is not True or row.get("status") not in allowed_statuses:
            raise ValueError(f"attempt row is nonterminal or has unknown status: {key}")
        if row["status"] == "ok":
            warm = row.get("warm_seconds")
            if not isinstance(warm, list) or len(warm) != REPETITIONS:
                raise ValueError(f"successful session lacks {REPETITIONS} warm samples: {key}")
            values = [row.get("first_seconds"), *warm, row.get("fidelity")]
            if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
                raise ValueError(f"successful session has invalid timing/fidelity values: {key}")
            if float(row["fidelity"]) > 1.0 + 1e-8:
                raise ValueError(f"successful session has nonphysical fidelity: {key}")
            if row.get("basis_canary_status") != "PASS" or not row.get("basis_canary_ir_sha256"):
                raise ValueError(f"successful session lacks a passing CUDA-Q basis-order canary: {key}")
    return seen


def classify_worker_status(status: str, expected_session_id: str, detail: dict[str, Any]) -> tuple[str, str]:
    if status == "ok" and detail.get("session_id") != expected_session_id:
        return "technical_error", f"worker session id mismatch: expected {expected_session_id}, got {detail.get('session_id')}"
    return status, ""


def capture_execution_environment(args: argparse.Namespace) -> dict[str, Any]:
    bridge = json.loads(CUDAQ_BRIDGE_MANIFEST.read_text(encoding="utf-8"))
    runtime = bridge["base_runtime"]
    parser = bridge["parser_runtime"]
    expected_python = Path(runtime["interpreter"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise ValueError(f"wrong CUDA-Q interpreter: expected {expected_python}, got {Path(sys.executable).resolve()}")
    expected_qiskit = Path(parser["site_packages"]).resolve()
    if args.qiskit_site_packages is None or args.qiskit_site_packages.resolve() != expected_qiskit:
        raise ValueError(f"wrong Qiskit parser site-packages: expected {expected_qiskit}")
    if not expected_qiskit.is_dir():
        raise FileNotFoundError(f"verified Qiskit parser site-packages missing: {expected_qiskit}")
    return {
        "python": sys.version,
        "interpreter": str(expected_python),
        "platform": platform.platform(),
        "cudaq_expected_version": runtime["cudaq"],
        "cudaq_package_lock_sha256": runtime["package_lock_sha256"],
        "qiskit_expected_version": parser["qiskit"],
        "qiskit_site_packages": str(expected_qiskit),
        "qiskit_package_lock_sha256": parser["package_lock_sha256"],
        "gpu": gpu_identity(),
    }


def validate_pilot_for_full(
    pilot: dict[str, Any], preflight_sha256: str, environment: dict[str, Any],
    attempt_rows: list[dict[str, Any]] | None = None,
    expected_pilot_hashes: list[str] | None = None,
) -> None:
    if pilot.get("status") != "PASS" or pilot.get("scope") != "pilot10":
        raise ValueError("full campaign requires status=PASS from a pilot10 report")
    if pilot.get("attempt_rows") != 180 or pilot.get("target_hashes") != 10:
        raise ValueError("pilot report must contain exactly 180 attempts over 10 hashes")
    if pilot.get("technical_error_count") != 0:
        raise ValueError("pilot report contains technical errors")
    if pilot.get("max_gpu_memory_used_mib_observed", 12289) > 12288:
        raise ValueError("pilot exceeded the frozen 12288 MiB GPU-memory cap")
    if pilot.get("projected_full_campaign_seconds", float("inf")) > 86400:
        raise ValueError("pilot projection exceeds the frozen 24-hour campaign budget")
    if pilot.get("preflight_manifest_sha256") != preflight_sha256:
        raise ValueError("pilot report belongs to a different preflight manifest")
    if pilot.get("execution_environment") != environment:
        raise ValueError("pilot and full campaign execution environments differ")
    if attempt_rows is None:
        raise ValueError("pilot full-campaign gate requires its sibling attempt ledger")
    if expected_pilot_hashes is None or pilot.get("pilot_hashes") != expected_pilot_hashes:
        raise ValueError("pilot report does not name the first ten frozen hashes in canonical order")
    seen = validate_attempt_ledger(attempt_rows)
    expected = {(digest, chi, f"session-{index}")
                for digest in pilot.get("pilot_hashes", [])
                for chi in RUNG_CHI for index in range(1, SESSIONS + 1)}
    if len(pilot.get("pilot_hashes", [])) != 10 or seen != expected or len(attempt_rows) != 180:
        raise ValueError("pilot attempt ledger is not exactly complete for its ten named hashes")
    if any(row.get("status") != "ok" for row in attempt_rows):
        raise ValueError("pilot attempt ledger contains a non-successful technical session")
    if pilot.get("attempt_ledger_sha256") != sha256_bytes(
            (json.dumps(attempt_rows, sort_keys=True, allow_nan=False) + "\n").encode()):
        raise ValueError("pilot ledger digest does not match the report")


def execute_measurement(args: argparse.Namespace) -> dict[str, Any]:
    if args.scope == "full" and not args.pilot_report:
        raise ValueError("full campaign requires a passing pilot_report.json")
    plan, hashes = read_plan(args.plan_dir)
    environment = capture_execution_environment(args)
    panel, fold_by_hash, _topology, _loaded = load_static_inputs()
    if _loaded["hashes"] != plan["input_hashes"]:
        raise ValueError("pinned source/code inputs changed after preflight")
    item_by_hash = {row["qasm_sha256"]: row for row in panel}
    hashes_by_digest = {row["source_qasm_sha256"]: row for row in hashes}
    ordered = sorted(hashes_by_digest)
    selected_hashes = ordered[:10] if args.scope == "pilot10" else ordered
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to use non-empty output without --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "attempt_records.jsonl"
    run_path = output / "run_manifest.json"
    preflight_sha = sha256_file(args.plan_dir / "preflight_manifest.json")
    if args.scope == "full":
        pilot = json.loads(args.pilot_report.read_text(encoding="utf-8"))
        ledger_path = args.pilot_report.parent / "attempt_records.jsonl"
        if not ledger_path.is_file():
            raise ValueError("pilot attempt_records.jsonl is missing beside pilot report")
        pilot_rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        validate_pilot_for_full(pilot, preflight_sha, environment, pilot_rows, ordered[:10])
    if records_path.exists() and not args.resume:
        raise FileExistsError(f"attempt log exists; use --resume only for this identical plan: {records_path}")
    if args.resume and not run_path.is_file():
        raise FileNotFoundError("--resume requires its matching run_manifest.json")
    prior_rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()] if records_path.exists() else []
    prior_completed = validate_attempt_ledger(prior_rows)
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        expected_identity = {
            "runner_id": RUNNER_ID,
            "runner_sha256": plan["input_hashes"]["measurement_runner_sha256"],
            "protocol_sha256": plan["protocol_sha256"],
            "preflight_manifest_sha256": preflight_sha,
            "scope": args.scope,
            "execution_environment": environment,
        }
        if any(previous.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("resume refuses changed runner, protocol, plan, scope, or execution environment")
        accumulated_before = float(previous.get("accumulated_wall_seconds", 0.0))
        max_gpu_used_prior = int(previous.get("max_gpu_memory_used_mib_observed", 0))
    else:
        accumulated_before = 0.0
        max_gpu_used_prior = 0
        write_json(run_path, {
            "runner_id": RUNNER_ID,
            "runner_sha256": plan["input_hashes"]["measurement_runner_sha256"],
            "scope": args.scope,
            "preflight_manifest_sha256": preflight_sha,
            "protocol_sha256": plan["protocol_sha256"],
            "execution_environment": environment,
            "status": "prepared",
            "timing_started": False,
            "accumulated_wall_seconds": accumulated_before,
            "max_gpu_memory_used_mib_observed": max_gpu_used_prior,
            "attempt_count": 0,
        })
    records_path.touch(exist_ok=True)
    expected_keys = {(digest, chi, f"session-{session}") for digest in selected_hashes for chi in RUNG_CHI for session in range(1, SESSIONS + 1)}
    if not prior_completed.issubset(expected_keys):
        raise ValueError("attempt ledger contains a hash/rung/session outside this run scope")
    completed = prior_completed if args.resume else set()
    started = time.monotonic()
    technical_errors = 0
    technical_errors += sum(row.get("status") != "ok" for row in prior_rows)
    max_gpu_used_mib = max_gpu_used_prior
    campaign_budget_hit = False
    try:
        run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
        run_manifest["status"] = "running"
        run_manifest["timing_started"] = True
        write_json(run_path, run_manifest)
        # Both locks are held over the campaign, with one fresh process per session.
        with s85.exclusive_compute_lock():
            for digest in selected_hashes:
                for chi in RUNG_CHI:
                    for session_index in range(1, SESSIONS + 1):
                        session_id = f"session-{session_index}"
                        key = (digest, chi, session_id)
                        item = item_by_hash[digest]
                        if key in completed:
                            continue
                        if accumulated_before + time.monotonic() - started > (3600 if args.scope == "pilot10" else 86400):
                            technical_errors += 1
                            campaign_budget_hit = True
                            break
                        free_mib, used_mib = parse_gpu_memory()
                        if free_mib < 4096 or used_mib > 12288:
                            status, error, payload = "resource_limit", f"GPU memory guard failed: free={free_mib} MiB used={used_mib} MiB", None
                        else:
                            max_gpu_used_mib = max(max_gpu_used_mib, used_mib)
                            item = item_by_hash[digest]
                            payload = None
                            status = "not_started"
                            error = ""
                            try:
                                command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                                           "--qasm-path", str(Path(item["source_root"]) / item["basename"]),
                                           "--qasm-sha256", digest, "--width-qubits", item["width_qubits"],
                                           "--max-bond", str(chi), "--session-id", session_id,
                                           "--expected-cudaq-version", str(environment["cudaq_expected_version"]),
                                           "--expected-qiskit-version", str(environment["qiskit_expected_version"]),
                                           "--qiskit-site-packages", str(args.qiskit_site_packages.resolve())]
                                def save_heartbeat(current_peak: int) -> None:
                                    nonlocal max_gpu_used_mib
                                    max_gpu_used_mib = max(max_gpu_used_mib, current_peak)
                                    heartbeat_manifest = json.loads(run_path.read_text(encoding="utf-8"))
                                    heartbeat_manifest.update({
                                        "accumulated_wall_seconds": accumulated_before + time.monotonic() - started,
                                        "max_gpu_memory_used_mib_observed": max_gpu_used_mib,
                                        "attempt_count": len(completed),
                                        "technical_error_count": technical_errors,
                                    })
                                    write_json(run_path, heartbeat_manifest)
                                status, error, payload, cell_peak = run_worker_process(
                                    command, timeout_seconds=180, heartbeat=save_heartbeat)
                                max_gpu_used_mib = max(max_gpu_used_mib, cell_peak)
                            except Exception as exc:
                                status, error = "worker_error", f"{type(exc).__name__}:{exc}".splitlines()[0][:800]
                        detail = (payload or {}).get("session", {})
                        row_status, session_error = classify_worker_status(status, session_id, detail)
                        if session_error:
                            error = session_error
                        row = {
                            "source_qasm_sha256": digest, "width_qubits": int(item_by_hash[digest]["width_qubits"]),
                            "c44_fold": fold_by_hash[digest], "max_bond": chi,
                            "session_id": session_id, "status": row_status,
                            "error": error, "first_seconds": detail.get("first_seconds", ""),
                            "warm_seconds": detail.get("warm_seconds", []), "fidelity": detail.get("fidelity", ""),
                            "quality_status": ("pass" if detail.get("fidelity", -1) >= QUALITY_THRESHOLD else "fail") if row_status == "ok" else "not_evaluated",
                            "quality_reference_seconds": (payload or {}).get("quality_reference_seconds", ""),
                            "quality_extract_seconds": detail.get("quality_extract_seconds", ""),
                            "basis_canary_status": (payload or {}).get("basis_canary", {}).get("status", "NOT_RUN"),
                            "basis_canary_ir_sha256": (payload or {}).get("basis_canary", {}).get("ir_sha256", ""),
                            "basis_canary_seconds": (payload or {}).get("basis_canary_seconds", ""),
                            "build_seconds": (payload or {}).get("build_seconds", ""),
                            "cudaq_version": (payload or {}).get("cudaq_version", ""),
                            "cudaq_distribution_versions": (payload or {}).get("cudaq_distribution_versions", {}),
                            "cudaq_runtime_build": (payload or {}).get("cudaq_runtime_build", ""),
                            "qiskit_version": (payload or {}).get("qiskit_version", ""),
                            "ir_sha256": (payload or {}).get("ir_sha256", ""),
                            "attempt_terminal": True,
                        }
                        try:
                            validate_attempt_ledger([row])
                        except Exception as exc:
                            row["status"] = "technical_error"
                            row["quality_status"] = "not_evaluated"
                            row["error"] = f"ledger validation: {type(exc).__name__}:{exc}"
                        with records_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                        completed.add(key)
                        if row["status"] != "ok":
                            technical_errors += 1
                        # Persist cumulative time after every terminal cell, so a hard
                        # interruption cannot reset the time cap on the next resume.
                        run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
                        run_manifest.update({
                            "accumulated_wall_seconds": accumulated_before + time.monotonic() - started,
                            "max_gpu_memory_used_mib_observed": max_gpu_used_mib,
                            "attempt_count": len(completed),
                            "technical_error_count": technical_errors,
                        })
                        write_json(run_path, run_manifest)
                    if campaign_budget_hit:
                        break
                if campaign_budget_hit:
                    break
        new_records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        validate_attempt_ledger(new_records)
    except Exception as exc:
        elapsed = time.monotonic() - started
        failure_manifest = json.loads(run_path.read_text(encoding="utf-8"))
        failure_manifest.update({"status": "failed", "terminal_error": f"{type(exc).__name__}:{exc}".splitlines()[0][:1000],
                                 "accumulated_wall_seconds": accumulated_before + elapsed,
                                 "max_gpu_memory_used_mib_observed": max_gpu_used_mib,
                                 "attempt_count": len([line for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()])})
        write_json(run_path, failure_manifest)
        write_json(output / ("pilot_report.json" if args.scope == "pilot10" else "run_summary.json"), {
            "status": "FAIL", "scope": args.scope, "preflight_manifest_sha256": preflight_sha,
            "execution_environment": environment, "attempt_count": failure_manifest["attempt_count"],
            "terminal_error": failure_manifest["terminal_error"], "timing_performed": True,
        })
        raise
    targets = reduce_ladder(new_records, selected_hashes)
    write_csv(output / "quality_constrained_targets.csv", list(targets[0]) if targets else [], targets)
    elapsed = time.monotonic() - started
    complete = all(row["ladder_status"] != "incomplete_or_failed_ladder" for row in targets)
    accumulated_after = accumulated_before + elapsed
    projected = accumulated_after / max(1, len(selected_hashes)) * 150
    pilot_pass = (args.scope == "pilot10" and complete and technical_errors == 0 and projected <= 86400 and max_gpu_used_mib <= 12288)
    summary = {
        "artifact_id": "family-aware-joint-mps-ladder-measurement-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner_id": RUNNER_ID, "scope": args.scope, "preflight_manifest_sha256": preflight_sha,
        "timing_performed": True, "training_performed": False,
        "attempt_rows": len(new_records), "attempt_count": len(new_records), "target_hashes": len(targets),
        "ladder_status_counts": dict(Counter(row["ladder_status"] for row in targets)),
        "technical_error_count": technical_errors, "elapsed_seconds": elapsed,
        "attempt_ledger_sha256": sha256_bytes((json.dumps(new_records, sort_keys=True, allow_nan=False) + "\n").encode()),
        "pilot_hashes": selected_hashes if args.scope == "pilot10" else [],
        "max_gpu_memory_used_mib_observed": max_gpu_used_mib,
        "projected_full_campaign_seconds": projected,
        "status": "PASS" if pilot_pass else "FAIL" if args.scope == "pilot10" else "PASS" if complete and technical_errors == 0 else "PARTIAL",
        "execution_environment": environment,
        "inputs": plan["input_hashes"], "rungs": list(RUNG_CHI), "sessions": SESSIONS,
        "accumulated_wall_seconds": accumulated_after,
        "cudaq_versions": sorted({str(row["cudaq_version"]) for row in new_records if row.get("cudaq_version")}),
        "cudaq_runtime_builds": sorted({str(row["cudaq_runtime_build"]) for row in new_records if row.get("cudaq_runtime_build")}),
    }
    run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    run_manifest.update({"status": summary["status"].lower(), "accumulated_wall_seconds": accumulated_after,
                         "max_gpu_memory_used_mib_observed": max_gpu_used_mib, "attempt_count": len(new_records)})
    write_json(run_path, run_manifest)
    write_json(output / "pilot_report.json" if args.scope == "pilot10" else output / "run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "measure"), default="prepare")
    parser.add_argument("--output", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--scope", choices=("pilot10", "full"), default="pilot10")
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--qiskit-site-packages", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--qasm-path", type=Path)
    parser.add_argument("--qasm-sha256")
    parser.add_argument("--width-qubits", type=int)
    parser.add_argument("--max-bond", type=int)
    parser.add_argument("--session-id")
    parser.add_argument("--expected-cudaq-version")
    parser.add_argument("--expected-qiskit-version")
    args = parser.parse_args()
    if args.worker:
        valid_sessions = {f"session-{index}" for index in range(1, SESSIONS + 1)}
        if (not args.qasm_path or not args.qasm_sha256 or not args.width_qubits or args.max_bond not in RUNG_CHI
                or args.session_id not in valid_sessions or not args.expected_cudaq_version
                or not args.expected_qiskit_version):
            parser.error("invalid internal worker arguments")
        raise SystemExit(worker(args))
    if args.action == "prepare":
        result = make_plan(args.output.resolve())
    else:
        result = execute_measurement(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
