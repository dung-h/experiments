#!/usr/bin/env python3
"""Materialize the frozen Azizov-style 51/64/51 common-core inputs.

This is an adaptation extractor, not the paper's unrecovered 36/13 schema.
It parses and transpiles only; it never constructs or runs an Aer simulator,
fits a model, or inspects GPU state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import io
import itertools
import json
import math
import numbers
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "benchmark_v1/protocol/azizov_common_core_gnn_v1.json"
DICTIONARY_PATH = ROOT / "benchmark_v1/protocol/azizov_local_feature_dictionary_v1.json"
PANEL_DIR = ROOT / "artifacts/benchmark_v1/sim_common_q16_manifest_20260927"
PANEL_CSV = PANEL_DIR / "sim_common_q16_manifest.csv"
PANEL_JSON = PANEL_DIR / "manifest.json"
LABEL_CSV = ROOT / "artifacts/benchmark_v1/c44_aer_q16_full_panel_evaluation_20260927/aer_q16_reduced_warm.csv"
OLD_STATIC_DIR = ROOT / "artifacts/benchmark_v3/simulator/azizov_common_core_gnn_v1_materialization"
OLD_STATIC_DAG = OLD_STATIC_DIR / "source_dag_topology.jsonl"
OLD_STATIC_SUMS = OLD_STATIC_DIR / "SHA256SUMS.txt"

UPSTREAM_COMMIT = "32c392a6ece276f1ff046d4e30052d0571ff6dc6"
UPSTREAM_FILE_HASHES = {
    "data_preparation/helper.py": "c34cc98df02b016ee9c94dc9ca27efdbbb7e6c6439831f09d3c92ed3a78f74ad",
    "data_preparation/utils.py": "d70db4646f0b9d3deb548d6d960711a43ad818cef344113a6fa1e9f9ae64bcb7",
    "model/transformer_model.py": "d6d250b2cb3082efc756ae9c5c269d4fe40bbdf49c743c535cddf6a0691c5683",
}
C44_SUMMARY_EXTRACTOR = ROOT / "tracks/azizov_independent/scripts/run_matrix.py"
C44_TWO_QUBIT_NAMES = {"cx", "ecr", "cz", "rzz", "swap", "iswap"}
SPLIT_PREFIX = "20260925-split|"
CONTROL_FLOW_NAMES = {"if_else", "while_loop", "for_loop", "switch_case", "if", "while", "for", "switch"}
STRUCTURAL_EXCLUDED_NAMES = {"barrier", "measure"}


class UnsupportedCircuit(ValueError):
    """A circuit cannot be represented under the frozen local graph contract."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_idempotent(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_bytes_idempotent(path, buffer.getvalue().encode("utf-8"))


def write_json_idempotent(path: Path, value: object) -> None:
    write_bytes_idempotent(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def write_bytes_idempotent(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or sha256_file(path) != sha256_bytes(data):
            raise FileExistsError(f"refusing to replace existing generated artifact with different bytes: {path}")
        return
    path.write_bytes(data)


def fold_for(source_sha256: str) -> int:
    digest = hashlib.sha256((SPLIT_PREFIX + source_sha256).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 5


def _instruction_parts(item: Any) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
    if hasattr(item, "operation") and hasattr(item, "qubits"):
        return item.operation, tuple(item.qubits), tuple(item.clbits)
    operation, qargs, cargs = item
    return operation, tuple(qargs), tuple(cargs)


def _bit_index(circuit: Any, bit: Any) -> int:
    return int(circuit.find_bit(bit).index)


def _parameter_value_over_pi(value: Any) -> float:
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        raise UnsupportedCircuit("unsupported_parameter", f"non-real or symbolic parameter: {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise UnsupportedCircuit("unsupported_parameter", f"non-finite parameter: {value!r}")
    return numeric / math.pi


def _validate_operation(operation: Any) -> str:
    name = str(getattr(operation, "name", ""))
    if name in CONTROL_FLOW_NAMES or getattr(operation, "blocks", None):
        raise UnsupportedCircuit("unsupported_dynamic_control", f"control-flow operation {name!r}")
    if getattr(operation, "condition", None) is not None:
        raise UnsupportedCircuit("unsupported_dynamic_control", f"legacy conditional operation {name!r}")
    if name == "delay":
        unit = getattr(operation, "unit", None)
        # The frozen context does not define a conversion contract for delay units.
        raise UnsupportedCircuit("unsupported_delay_unit", f"delay operation uses uncalibrated unit {unit!r}")
    return name


def _circuit_caps(circuit: Any, max_nodes: int, max_edges: int, max_q: int, max_c: int) -> None:
    if len(circuit.data) == 0:
        raise UnsupportedCircuit("empty_graph", "empty operation graph")
    if int(circuit.num_qubits) > max_q:
        raise UnsupportedCircuit("quantum_wire_cap", f"{circuit.num_qubits} quantum wires exceed cap {max_q}")
    if int(circuit.num_clbits) > max_c:
        raise UnsupportedCircuit("classical_wire_cap", f"{circuit.num_clbits} classical wires exceed cap {max_c}")
    if len(circuit.data) > max_nodes:
        raise UnsupportedCircuit("operation_node_cap", f"{len(circuit.data)} operation nodes exceed cap {max_nodes}")
    if len(circuit.data) > 0 and int(circuit.num_qubits) == 0:
        raise UnsupportedCircuit("invalid_wire_index", "operation graph has nodes but no quantum wires")


def graph_record(circuit: Any, source_sha256: str, representation: str, caps: dict[str, int]) -> dict[str, Any]:
    _circuit_caps(circuit, caps["max_operation_nodes"], caps["max_edges"], caps["max_quantum_wires"], caps["max_classical_wires"])
    node_count = len(circuit.data)
    nodes: list[dict[str, Any]] = []
    previous: dict[tuple[str, int], int] = {}
    edges: set[tuple[int, int]] = set()

    for index, item in enumerate(circuit.data):
        operation, qargs, cargs = _instruction_parts(item)
        name = _validate_operation(operation)
        q_indices = [_bit_index(circuit, bit) for bit in qargs]
        c_indices = [_bit_index(circuit, bit) for bit in cargs]
        if any(index < 0 or index >= caps["max_quantum_wires"] for index in q_indices):
            raise UnsupportedCircuit("invalid_wire_index", f"quantum wire index out of range in node {index}")
        if any(index < 0 or index >= caps["max_classical_wires"] for index in c_indices):
            raise UnsupportedCircuit("invalid_wire_index", f"classical wire index out of range in node {index}")

        params = list(getattr(operation, "params", ()))
        if len(params) > caps["max_real_parameters"]:
            raise UnsupportedCircuit("parameter_count_cap", f"node {index} has {len(params)} parameters")
        param_values = [_parameter_value_over_pi(value) for value in params]
        param_masks = [1.0] * len(param_values)
        param_values.extend([0.0] * (caps["max_real_parameters"] - len(param_values)))
        param_masks.extend([0.0] * (caps["max_real_parameters"] - len(param_masks)))

        nodes.append(
            {
                "node_index": index,
                "operation": name,
                "qargs": q_indices,
                "cargs": c_indices,
                "parameter_values_over_pi": param_values,
                "parameter_presence_masks": param_masks,
                "quantum_arity_normalized": len(q_indices) / caps["max_quantum_wires"],
                "classical_arity_normalized": len(c_indices) / caps["max_classical_wires"],
                "instruction_index_normalized": index / max(1, node_count - 1),
            }
        )

        touched = [("qubit", wire) for wire in q_indices] + [("clbit", wire) for wire in c_indices]
        for wire in touched:
            if wire in previous:
                edges.add((previous[wire], index))
            previous[wire] = index

    if len(edges) > caps["max_edges"]:
        raise UnsupportedCircuit("edge_cap", f"{len(edges)} graph edges exceed cap {caps['max_edges']}")
    edge_rows = [[source, target] for source, target in sorted(edges)]
    if any(source >= target for source, target in edges):
        raise UnsupportedCircuit("invalid_graph_order", "an edge is not directed from an earlier to a later instruction")
    return {
        "schema_version": "azizov_raw_graph_v1",
        "source_sha256": source_sha256,
        "representation": representation,
        "num_qubits": int(circuit.num_qubits),
        "num_clbits": int(circuit.num_clbits),
        "node_count": node_count,
        "edge_count": len(edge_rows),
        "nodes": nodes,
        "edges": edge_rows,
    }


def _longest_path(circuit: Any, *, multi_qubit_only: bool = False) -> tuple[int, int, int]:
    """Return (operation count, longest dependency path, summed q-arity)."""
    previous: dict[int, int] = {}
    level_by_node: dict[int, int] = {}
    count = 0
    qarity_sum = 0
    for node_index, item in enumerate(circuit.data):
        operation, qargs, _ = _instruction_parts(item)
        name = str(getattr(operation, "name", ""))
        if name in STRUCTURAL_EXCLUDED_NAMES:
            continue
        if multi_qubit_only and len(qargs) <= 1:
            continue
        q_indices = [_bit_index(circuit, bit) for bit in qargs]
        level = 1 + max((level_by_node.get(previous[q], 0) for q in q_indices if q in previous), default=0)
        level_by_node[node_index] = level
        for q in q_indices:
            previous[q] = node_index
        count += 1
        qarity_sum += len(q_indices)
    depth = max(level_by_node.values(), default=0)
    return count, depth, qarity_sum


def structural_metrics(circuit: Any) -> dict[str, float]:
    n = int(circuit.num_qubits)
    gate_count, dependency_depth, qarity_sum = _longest_path(circuit)
    multi_count, multi_depth, _ = _longest_path(circuit, multi_qubit_only=True)
    pairs: set[tuple[int, int]] = set()
    for item in circuit.data:
        operation, qargs, _ = _instruction_parts(item)
        if str(getattr(operation, "name", "")) in STRUCTURAL_EXCLUDED_NAMES:
            continue
        q_indices = [_bit_index(circuit, bit) for bit in qargs]
        for left, right in itertools.combinations(sorted(set(q_indices)), 2):
            pairs.add((left, right))

    communication = 2.0 * len(pairs) / (n * (n - 1)) if n >= 2 else 0.0
    critical_depth = multi_depth / multi_count if multi_count else 0.0
    entanglement_ratio = multi_count / gate_count if gate_count else 0.0
    parallelism = (gate_count / dependency_depth - 1.0) / (n - 1) if n >= 2 and dependency_depth else 0.0
    liveness = qarity_sum / (n * dependency_depth) if n > 0 and dependency_depth else 0.0
    values = {
        "program_communication": communication,
        "critical_depth": critical_depth,
        "entanglement_ratio": entanglement_ratio,
        "parallelism": parallelism,
        "liveness": liveness,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise UnsupportedCircuit("nonfinite_global", f"{name} is not finite")
        if value < -1e-8 or value > 1.0 + 1e-8:
            raise UnsupportedCircuit("structural_metric_out_of_range", f"{name}={value} outside [0,1]")
    return values


def global_features(circuit: Any, ordered_fields: list[str]) -> dict[str, float]:
    counts = circuit.count_ops()
    features: dict[str, float] = {}
    for field in ordered_fields:
        if field.startswith("count_"):
            gate = field.removeprefix("count_")
            features[field] = float(counts.get(gate, 0))
        elif field == "num_qubits":
            features[field] = float(circuit.num_qubits)
        elif field == "depth":
            features[field] = float(circuit.depth() or 0)
        elif field in {"program_communication", "critical_depth", "entanglement_ratio", "parallelism", "liveness"}:
            # Compute once below, retaining the dictionary's canonical order.
            continue
        else:
            raise ValueError(f"unrecognized frozen global field: {field}")
    structural = structural_metrics(circuit)
    for name, value in structural.items():
        if name in ordered_fields:
            features[name] = float(value)
    ordered = {field: float(features[field]) for field in ordered_fields}
    if len(ordered) != len(ordered_fields) or any(not math.isfinite(value) for value in ordered.values()):
        raise UnsupportedCircuit("invalid_global_vector", "global vector is incomplete or non-finite")
    return ordered


def compiled_extension(circuit: Any, ordered_fields: list[str]) -> dict[str, float]:
    counts = circuit.count_ops()
    structural = structural_metrics(circuit)
    values = {
        "compiled_depth": float(circuit.depth() or 0),
        "compiled_total_operation_count": float(len(circuit.data)),
        "compiled_two_qubit_operation_count": float(
            sum(1 for item in circuit.data if len(_instruction_parts(item)[1]) == 2 and str(getattr(_instruction_parts(item)[0], "name", "")) not in STRUCTURAL_EXCLUDED_NAMES)
        ),
        "compiled_measurement_count": float(counts.get("measure", 0)),
        "compiled_count_rz": float(counts.get("rz", 0)),
        "compiled_count_sx": float(counts.get("sx", 0)),
        "compiled_count_x": float(counts.get("x", 0)),
        "compiled_count_ecr": float(counts.get("ecr", 0)),
        "compiled_program_communication": structural["program_communication"],
        "compiled_critical_depth": structural["critical_depth"],
        "compiled_entanglement_ratio": structural["entanglement_ratio"],
        "compiled_parallelism": structural["parallelism"],
        "compiled_liveness": structural["liveness"],
    }
    ordered = {field: float(values[field]) for field in ordered_fields}
    if len(ordered) != 13 or any(not math.isfinite(value) for value in ordered.values()):
        raise UnsupportedCircuit("invalid_compiled_extension", "compiled extension is incomplete or non-finite")
    return ordered


def summary_metrics(circuit: Any) -> dict[str, int]:
    counts = circuit.count_ops()
    return {
        "physical_width": int(circuit.num_qubits),
        "physical_depth": int(circuit.depth() or 0),
        "physical_ops": int(sum(counts.values())),
        "physical_two_qubit_ops": int(sum(value for name, value in counts.items() if name in C44_TWO_QUBIT_NAMES)),
        "swap_count": int(counts.get("swap", 0)),
        "dag_node_count": int(len(circuit.data)),
    }


def operation_audit(circuit: Any, global_gate_names: list[str]) -> dict[str, Any]:
    counts = {str(name): int(value) for name, value in circuit.count_ops().items()}
    unlisted = {name: value for name, value in sorted(counts.items()) if name not in global_gate_names}
    return {"operation_counts": dict(sorted(counts.items())), "unlisted_operation_counts": unlisted}


def _run_git(upstream_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(upstream_root), *args], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def verify_upstream(upstream_root: Path) -> dict[str, Any]:
    commit = _run_git(upstream_root, "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"pinned upstream HEAD mismatch: {commit} != {UPSTREAM_COMMIT}")
    files: dict[str, str] = {}
    for relative, expected in UPSTREAM_FILE_HASHES.items():
        blob = subprocess.run(
            ["git", "-C", str(upstream_root), "show", f"{UPSTREAM_COMMIT}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        actual = sha256_bytes(blob)
        if actual != expected:
            raise ValueError(f"pinned upstream blob hash mismatch for {relative}: {actual} != {expected}")
        files[relative] = actual
    return {
        "repository": "https://github.com/mooselab/Quantum-Execution-Time-Prediction",
        "commit": commit,
        "pinned_file_sha256": files,
        "code_import_policy": "The pinned blobs are verified from Git objects; no live upstream Python module is imported.",
        "source_qasm_root": str(upstream_root / "data/quantum_circuits"),
    }


def verify_context(protocol: dict[str, Any], root: Path) -> tuple[dict[str, Any], Any]:
    local = protocol["local_cell"]
    context_path = root / local["frozen_context_source"]
    lock_path = root / local["package_lock_source"]
    actual_context_hash = sha256_file(context_path)
    actual_lock_hash = sha256_file(lock_path)
    if actual_context_hash != local["frozen_context_sha256"]:
        raise ValueError(f"frozen environment metadata hash mismatch: {actual_context_hash}")
    if actual_lock_hash != local["package_lock_sha256"]:
        raise ValueError(f"frozen package lock hash mismatch: {actual_lock_hash}")

    context = json.loads(context_path.read_text(encoding="utf-8"))
    expected = local["context"]
    if metadata.version("qiskit") != expected["qiskit_distribution"]:
        raise ValueError("qiskit distribution version does not match the frozen environment")
    if metadata.version("qiskit-terra") != expected["qiskit_terra_distribution"]:
        raise ValueError("qiskit-terra distribution version does not match the frozen environment")
    if metadata.version("qiskit-aer") != expected["qiskit_aer"]:
        raise ValueError("qiskit-aer distribution version does not match the frozen environment")
    if platform.python_version() != str(context.get("python_version", "")):
        raise ValueError("Python version does not match the captured Aer environment")

    import qiskit
    from qiskit.providers.fake_provider import FakeSherbrooke

    backend = FakeSherbrooke()
    module_version = str(getattr(qiskit, "__version__", ""))
    if module_version != expected["environment_reported_module_qiskit_version"]:
        raise ValueError(f"Qiskit module version mismatch: {module_version}")
    if str(backend.name) != "fake_sherbrooke":
        raise ValueError(f"unexpected FakeSherbrooke backend name {backend.name!r}")
    target_operations = set(str(name) for name in backend.operation_names)
    target_basis = sorted(target_operations - {"measure", "delay"})
    expected_basis = sorted(expected["historical_conf_basis_gates"])
    if target_basis != expected_basis:
        raise ValueError(f"FakeSherbrooke target basis mismatch: {target_basis} != {expected_basis}")
    if int(backend.num_qubits) != 127:
        raise ValueError(f"FakeSherbrooke width mismatch: {backend.num_qubits} != 127")
    env = {
        "status": "pass",
        "verification_scope": "Exact Python/Qiskit/Aer distributions, frozen environment metadata/lock byte hashes, FakeSherbrooke backend identity/width/basis; all-hash transpile-summary replay is recorded separately.",
        "python_version": platform.python_version(),
        "qiskit_module_version": module_version,
        "qiskit_distribution": metadata.version("qiskit"),
        "qiskit_terra_distribution": metadata.version("qiskit-terra"),
        "qiskit_aer_distribution": metadata.version("qiskit-aer"),
        "frozen_context_path": str(context_path.relative_to(root)),
        "frozen_context_sha256": actual_context_hash,
        "package_lock_path": str(lock_path.relative_to(root)),
        "package_lock_sha256": actual_lock_hash,
        "backend": {
            "class": "qiskit.providers.fake_provider.FakeSherbrooke",
            "name": str(backend.name),
            "num_qubits": int(backend.num_qubits),
            "target_operation_names": sorted(target_operations),
            "basis_gate_match": target_basis == expected_basis,
            "recorded_snapshot_sha256": expected["fake_backend_snapshot_hash"],
            "recorded_noise_model_sha256": expected["noise_model_hash"],
            "recorded_historical_conf_sha256": expected["historical_conf_sha256"],
            "snapshot_noise_hash_scope": "Existing C44 context hashes are verified as locked provenance values; no Aer simulator is run or noise-model calibration is recomputed.",
        },
        "transpilation": {
            "optimization_level": expected["optimization_level"],
            "seed_transpiler": expected["transpiler_seed"],
            "call": "transpile(qc, backend=FakeSherbrooke(), optimization_level=1, seed_transpiler=1234)",
        },
        "simulator_runtime_called": False,
        "model_fit_called": False,
        "cuda_or_gpu_state_inspected": False,
    }
    return env, backend


def load_old_source_dag_audit() -> dict[str, Any]:
    if not OLD_STATIC_DAG.is_file() or not OLD_STATIC_SUMS.is_file():
        return {"status": "not_available", "path": str(OLD_STATIC_DAG.relative_to(ROOT))}
    sums = {}
    for line in OLD_STATIC_SUMS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(None, 1)
        sums[filename.strip()] = digest
    actual = sha256_file(OLD_STATIC_DAG)
    expected = sums.get(OLD_STATIC_DAG.name)
    if actual != expected:
        raise ValueError("existing partial source DAG failed its own SHA256SUMS check")
    return {
        "status": "hash_verified_cross_check_only",
        "path": str(OLD_STATIC_DAG.relative_to(ROOT)),
        "sha256": actual,
        "use": "Read-only cross-check; the new source graph is independently reconstructed from exact-hash QASM in the frozen Terra environment.",
    }


def panel_and_label_audit(protocol: dict[str, Any], upstream_root: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    panel_rows = read_rows(PANEL_CSV)
    label_rows = read_rows(LABEL_CSV)
    panel_core = [row for row in panel_rows if row["stratum"] == "core_q2_q9"]
    labels_core = [row for row in label_rows if row["stratum"] == "core_q2_q9"]
    label_by_key = {(row["circuit_id"], row["source_sha256"]): row for row in labels_core}
    if len(panel_core) != 162 or len(labels_core) != 162 or len(label_by_key) != 162:
        raise ValueError(f"expected 162 unique core label joins; found {len(panel_core)}, {len(labels_core)}, {len(label_by_key)}")

    qasm_root = upstream_root / "data/quantum_circuits"
    member_rows: list[dict[str, Any]] = []
    members_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels_out: list[dict[str, Any]] = []
    for panel in sorted(panel_core, key=lambda row: row["panel_member_id"]):
        circuit_id = Path(panel["basename"]).stem
        digest = panel["qasm_sha256"]
        label = label_by_key.get((circuit_id, digest))
        if label is None:
            raise ValueError(f"missing exact C44 label join for {(circuit_id, digest)}")
        source_path = qasm_root / panel["basename"]
        if not source_path.is_file():
            raise FileNotFoundError(f"pinned source QASM missing: {source_path}")
        actual_qasm_sha = sha256_file(source_path)
        if actual_qasm_sha != digest or actual_qasm_sha != label["source_sha256"]:
            raise ValueError(f"source QASM hash mismatch for {panel['panel_member_id']}")
        c44_fold = int(label["fold"])
        computed_fold = fold_for(digest)
        if c44_fold != computed_fold:
            raise ValueError(f"C44 fold mismatch for {digest}: {c44_fold} != {computed_fold}")
        target = float(label["observed_seconds"])
        if not math.isfinite(target) or target < 0:
            raise ValueError(f"invalid frozen C44 target for {panel['panel_member_id']}")
        member = {
            "panel_member_id": panel["panel_member_id"],
            "circuit_id": circuit_id,
            "basename": panel["basename"],
            "family": panel["family"],
            "width_qubits": int(panel["width_qubits"]),
            "stratum": panel["stratum"],
            "source_sha256": digest,
            "source_qasm_path": str(source_path),
            "source_qasm_bytes": source_path.stat().st_size,
            "c44_fold": c44_fold,
            "computed_fold": computed_fold,
            "fold_match": True,
            "target_seconds": target,
            "target_clock": protocol["local_cell"]["evaluation_target_clock"],
            "label_row": label,
        }
        members_by_hash[digest].append(member)
        member_rows.append(
            {
                "panel_member_id": member["panel_member_id"],
                "circuit_id": circuit_id,
                "basename": panel["basename"],
                "family": panel["family"],
                "width_qubits": int(panel["width_qubits"]),
                "stratum": panel["stratum"],
                "source_sha256": digest,
                "c44_fold": c44_fold,
                "computed_fold": computed_fold,
                "fold_match": "true",
                "aliases_for_hash_count": 0,
            }
        )
        labels_out.append(
            {
                "panel_member_id": member["panel_member_id"],
                "circuit_id": circuit_id,
                "source_sha256": digest,
                "fold": c44_fold,
                "observed_seconds": target,
                "target_clock": protocol["local_cell"]["evaluation_target_clock"],
            }
        )

    if len(members_by_hash) != 150:
        raise ValueError(f"expected exactly 150 unique source hashes, found {len(members_by_hash)}")
    for digest, members in members_by_hash.items():
        folds = {int(item["c44_fold"]) for item in members}
        if len(folds) != 1:
            raise ValueError(f"exact-hash aliases cross folds for {digest}")
        summaries = {
            tuple(float(item["label_row"][name]) for name in ("physical_width", "physical_depth", "physical_ops", "physical_two_qubit_ops", "swap_count", "dag_node_count"))
            for item in members
        }
        if len(summaries) != 1:
            raise ValueError(f"C44 replay summaries disagree across aliases for {digest}")
        for row in member_rows:
            if row["source_sha256"] == digest:
                row["aliases_for_hash_count"] = len(members)
    return member_rows, members_by_hash, labels_out


def replay_audit_row(source_sha256: str, fold: int, expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"source_sha256": source_sha256, "fold": fold}
    matches = []
    for name in ("physical_width", "physical_depth", "physical_ops", "physical_two_qubit_ops", "swap_count", "dag_node_count"):
        value = int(float(expected[name]))
        replayed = int(actual[name])
        match = value == replayed
        row[f"c44_{name}"] = value
        row[f"replay_{name}"] = replayed
        row[f"{name}_match"] = "true" if match else "false"
        matches.append(match)
    row["replay_status"] = "PASS" if all(matches) else "FAIL"
    return row


def materialize_one(
    digest: str,
    members: list[dict[str, Any]],
    backend: Any,
    ordered_globals: list[str],
    ordered_extensions: list[str],
    caps: dict[str, int],
    source_old_by_hash: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from qiskit import QuantumCircuit, transpile
    from qiskit import qpy

    source_path = Path(members[0]["source_qasm_path"])
    source = QuantumCircuit.from_qasm_file(str(source_path))
    source_graph = graph_record(source, digest, "source", caps)
    source_globals = global_features(source, ordered_globals)
    compiled = transpile(
        source,
        backend=backend,
        optimization_level=1,
        seed_transpiler=1234,
    )
    compiled_graph = graph_record(compiled, digest, "transpiled", caps)
    compiled_globals = global_features(compiled, ordered_globals)
    extension = compiled_extension(compiled, ordered_extensions)
    summary = summary_metrics(compiled)
    expected_summaries = {
        name: float(members[0]["label_row"][name])
        for name in ("physical_width", "physical_depth", "physical_ops", "physical_two_qubit_ops", "swap_count", "dag_node_count")
    }
    replay = replay_audit_row(digest, int(members[0]["c44_fold"]), expected_summaries, summary)

    source_graph_again = graph_record(source, digest, "source", caps)
    if sha256_bytes(canonical_bytes(source_graph)) != sha256_bytes(canonical_bytes(source_graph_again)):
        raise UnsupportedCircuit("nondeterministic_graph_serialization", "source graph changed on repeated extraction")

    old = source_old_by_hash.get(digest)
    old_graph_match = None
    if old is not None:
        old_nodes = old.get("nodes", [])
        old_edges = {(int(edge["source_node"]), int(edge["target_node"])) for edge in old.get("edges", [])}
        new_edges = {tuple(edge) for edge in source_graph["edges"]}
        old_node_signature = [
            (str(node["operation"]), tuple(int(value) for value in node["qargs"]), tuple(int(value) for value in node["cargs"]))
            for node in old_nodes
        ]
        new_node_signature = [
            (str(node["operation"]), tuple(int(value) for value in node["qargs"]), tuple(int(value) for value in node["cargs"]))
            for node in source_graph["nodes"]
        ]
        old_graph_match = old_node_signature == new_node_signature and old_edges == new_edges

    qasm_text = compiled.qasm()
    compiled_qasm = qasm_text.encode("utf-8")
    qpy_buffer = io.BytesIO()
    qpy.dump(compiled, qpy_buffer)
    compiled_qpy = qpy_buffer.getvalue()
    reloaded = qpy.load(io.BytesIO(compiled_qpy))
    if len(reloaded) != 1 or reloaded[0] != compiled:
        raise UnsupportedCircuit("lossless_serialization_failure", "QPY round-trip did not preserve compiled circuit")

    source_ops = operation_audit(source, [field.removeprefix("count_") for field in ordered_globals if field.startswith("count_")])
    compiled_ops = operation_audit(compiled, [field.removeprefix("count_") for field in ordered_globals if field.startswith("count_")])
    return {
        "source_graph": source_graph,
        "compiled_graph": compiled_graph,
        "source_globals": source_globals,
        "compiled_globals": compiled_globals,
        "compiled_extension": extension,
        "replay_audit": replay,
        "summary_metrics": summary,
        "source_ops": source_ops,
        "compiled_ops": compiled_ops,
        "compiled_qasm": compiled_qasm,
        "compiled_qpy": compiled_qpy,
        "old_source_graph_match": old_graph_match,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/benchmark_v3/simulator/azizov_common_core_adaptation",
    )
    args = parser.parse_args()
    upstream_root = args.upstream_root.resolve()
    output_dir = args.output_dir.resolve()
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    dictionary_bytes = DICTIONARY_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    dictionary = json.loads(dictionary_bytes)
    ordered_globals = list(dictionary["ordered_global_fields"])
    ordered_extensions = list(dictionary["ordered_compiled_extension_fields"])
    dimensions = protocol["representation_contract"]["dimensions_before_train_only_constant_removal"]
    if (len(ordered_globals), len(ordered_extensions)) != (51, 13):
        raise ValueError("frozen dictionary does not contain exactly 51 source and 13 compiled extension fields")
    if len(set(ordered_globals)) != 51 or len(set(ordered_extensions)) != 13:
        raise ValueError("frozen global field names are not unique")
    if dimensions != {"source_gnn": 51, "hybrid_gnn": 64, "transpiled_gnn": 51}:
        raise ValueError(f"representation contract dimensions changed: {dimensions}")

    upstream = verify_upstream(upstream_root)
    environment, backend = verify_context(protocol, ROOT)
    member_rows, members_by_hash, label_rows = panel_and_label_audit(protocol, upstream_root)
    old_dag_audit = load_old_source_dag_audit()
    old_source_by_hash: dict[str, dict[str, Any]] = {}
    if old_dag_audit.get("status") == "hash_verified_cross_check_only":
        with OLD_STATIC_DAG.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                old_source_by_hash[str(record["source_qasm_sha256"])] = record

    caps = {key: int(value) for key, value in protocol["representation_contract"]["dag_size_policy"].items() if key.startswith("max_")}
    if caps != {
        "max_operation_nodes": 300000,
        "max_edges": 1000000,
        "max_quantum_wires": 127,
        "max_classical_wires": 127,
        "max_real_parameters": 4,
    }:
        raise ValueError(f"frozen graph support caps changed: {caps}")

    all_graph_source: list[dict[str, Any]] = []
    all_graph_transpiled: list[dict[str, Any]] = []
    feature_rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    hash_manifest_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for digest in sorted(members_by_hash):
        members = members_by_hash[digest]
        fold = int(members[0]["c44_fold"])
        member_ids = [member["panel_member_id"] for member in members]
        entry: dict[str, Any] = {
            "source_sha256": digest,
            "fold": fold,
            "member_ids": member_ids,
            "source_qasm": {
                "path": str(Path(members[0]["source_qasm_path"]).relative_to(upstream_root)),
                "bytes": int(members[0]["source_qasm_bytes"]),
                "sha256": digest,
            },
            "representation_status": {"source_gnn": "unavailable", "hybrid_gnn": "unavailable", "transpiled_gnn": "unavailable"},
            "terminal_reason": "",
        }
        try:
            materialized = materialize_one(
                digest,
                members,
                backend,
                ordered_globals,
                ordered_extensions,
                caps,
                old_source_by_hash,
            )
            if materialized["replay_audit"]["replay_status"] != "PASS":
                raise UnsupportedCircuit("compiled_replay_mismatch", "one or more C44 compiled-summary fields did not match")

            qasm_rel = f"compiled_qasm/{digest}.qasm"
            qpy_rel = f"compiled_qpy/{digest}.qpy"
            write_bytes_idempotent(output_dir / qasm_rel, materialized["compiled_qasm"])
            write_bytes_idempotent(output_dir / qpy_rel, materialized["compiled_qpy"])
            all_graph_source.append(materialized["source_graph"])
            all_graph_transpiled.append(materialized["compiled_graph"])
            feature_row: dict[str, object] = {"source_sha256": digest}
            feature_row.update({f"source__{key}": value for key, value in materialized["source_globals"].items()})
            feature_row.update({f"compiled_ext__{key}": value for key, value in materialized["compiled_extension"].items()})
            feature_row.update({f"transpiled__{key}": value for key, value in materialized["compiled_globals"].items()})
            if len([key for key in feature_row if key.startswith("source__")]) != 51:
                raise UnsupportedCircuit("invalid_global_vector", "source representation is not 51-dimensional")
            if len([key for key in feature_row if key.startswith("source__") or key.startswith("compiled_ext__")]) != 64:
                raise UnsupportedCircuit("invalid_global_vector", "hybrid representation is not 64-dimensional")
            if len([key for key in feature_row if key.startswith("transpiled__")]) != 51:
                raise UnsupportedCircuit("invalid_global_vector", "transpiled representation is not 51-dimensional")
            feature_rows.append(feature_row)
            replay_rows.append(materialized["replay_audit"])
            entry["representation_status"] = {"source_gnn": "available", "hybrid_gnn": "available", "transpiled_gnn": "available"}
            entry["terminal_reason"] = "all_three_representations_materialized_and_compiled_summary_matched"
            entry["compiled_qasm"] = {
                "path": qasm_rel,
                "bytes": len(materialized["compiled_qasm"]),
                "sha256": sha256_bytes(materialized["compiled_qasm"]),
            }
            entry["compiled_qpy"] = {
                "path": qpy_rel,
                "bytes": len(materialized["compiled_qpy"]),
                "sha256": sha256_bytes(materialized["compiled_qpy"]),
                "round_trip": "PASS",
            }
            entry["source_operation_audit"] = materialized["source_ops"]
            entry["compiled_operation_audit"] = materialized["compiled_ops"]
            entry["compiled_summary"] = materialized["summary_metrics"]
            entry["old_source_topology_cross_check"] = materialized["old_source_graph_match"]
            entry["source_graph_sha256"] = sha256_bytes(canonical_bytes(materialized["source_graph"]))
            entry["transpiled_graph_sha256"] = sha256_bytes(canonical_bytes(materialized["compiled_graph"]))
        except Exception as exc:
            reason = exc.code if isinstance(exc, UnsupportedCircuit) else f"{type(exc).__name__}: {exc}"
            entry["terminal_reason"] = reason
            errors.append({"source_sha256": digest, "error": reason})
            # If a later stage failed after preliminary output writes, do not expose features/graphs.
            all_graph_source = [row for row in all_graph_source if row["source_sha256"] != digest]
            all_graph_transpiled = [row for row in all_graph_transpiled if row["source_sha256"] != digest]
            feature_rows = [row for row in feature_rows if row["source_sha256"] != digest]
            replay_rows = [row for row in replay_rows if row["source_sha256"] != digest]
        terminal_rows.append(
            {
                "source_sha256": digest,
                "fold": fold,
                "member_ids_json": json.dumps(member_ids, separators=(",", ":")),
                "source_gnn_status": entry["representation_status"]["source_gnn"],
                "hybrid_gnn_status": entry["representation_status"]["hybrid_gnn"],
                "transpiled_gnn_status": entry["representation_status"]["transpiled_gnn"],
                "terminal_reason": entry["terminal_reason"],
            }
        )
        hash_manifest_rows.append(entry)

    member_fields = [
        "panel_member_id", "circuit_id", "basename", "family", "width_qubits", "stratum", "source_sha256",
        "c44_fold", "computed_fold", "fold_match", "aliases_for_hash_count",
    ]
    label_fields = ["panel_member_id", "circuit_id", "source_sha256", "fold", "observed_seconds", "target_clock"]
    replay_fields = ["source_sha256", "fold"]
    for name in ("physical_width", "physical_depth", "physical_ops", "physical_two_qubit_ops", "swap_count", "dag_node_count"):
        replay_fields.extend([f"c44_{name}", f"replay_{name}", f"{name}_match"])
    replay_fields.append("replay_status")
    terminal_fields = [
        "source_sha256", "fold", "member_ids_json", "source_gnn_status", "hybrid_gnn_status",
        "transpiled_gnn_status", "terminal_reason",
    ]
    feature_fields = ["source_sha256"]
    feature_fields.extend(f"source__{name}" for name in ordered_globals)
    feature_fields.extend(f"compiled_ext__{name}" for name in ordered_extensions)
    feature_fields.extend(f"transpiled__{name}" for name in ordered_globals)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_bytes_idempotent(output_dir / "feature_dictionary.json", dictionary_bytes)
    write_csv_idempotent(output_dir / "fold_assignments_and_hash_audit.csv", member_rows, member_fields)
    write_csv_idempotent(output_dir / "labels_core.csv", label_rows, label_fields)
    write_csv_idempotent(output_dir / "compiled_replay_audit.csv", replay_rows, replay_fields)
    write_csv_idempotent(output_dir / "representation_coverage_and_terminal_statuses.csv", terminal_rows, terminal_fields)
    write_csv_idempotent(output_dir / "global_features.csv", feature_rows, feature_fields)
    source_graph_text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for row in sorted(all_graph_source, key=lambda item: item["source_sha256"]))
    transpiled_graph_text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for row in sorted(all_graph_transpiled, key=lambda item: item["source_sha256"]))
    write_bytes_idempotent(output_dir / "source_graphs.jsonl", source_graph_text.encode("utf-8"))
    write_bytes_idempotent(output_dir / "transpiled_graphs.jsonl", transpiled_graph_text.encode("utf-8"))

    input_paths = {
        "benchmark_v1/protocol/azizov_common_core_gnn_v1.json": sha256_bytes(protocol_bytes),
        "benchmark_v1/protocol/azizov_local_feature_dictionary_v1.json": sha256_bytes(dictionary_bytes),
        str(PANEL_CSV.relative_to(ROOT)): sha256_file(PANEL_CSV),
        str(PANEL_JSON.relative_to(ROOT)): sha256_file(PANEL_JSON),
        str(LABEL_CSV.relative_to(ROOT)): sha256_file(LABEL_CSV),
        str(C44_SUMMARY_EXTRACTOR.relative_to(ROOT)): sha256_file(C44_SUMMARY_EXTRACTOR),
        str((ROOT / protocol["local_cell"]["frozen_context_source"]).relative_to(ROOT)): environment["frozen_context_sha256"],
        str((ROOT / protocol["local_cell"]["package_lock_source"]).relative_to(ROOT)): environment["package_lock_sha256"],
    }
    if old_dag_audit.get("status") == "hash_verified_cross_check_only":
        input_paths[old_dag_audit["path"]] = old_dag_audit["sha256"]
    output_names = [
        "feature_dictionary.json", "fold_assignments_and_hash_audit.csv", "labels_core.csv",
        "compiled_replay_audit.csv", "representation_coverage_and_terminal_statuses.csv", "global_features.csv",
        "source_graphs.jsonl", "transpiled_graphs.jsonl",
    ]
    output_names.extend(
        str(path.relative_to(output_dir))
        for path in sorted((output_dir / "compiled_qasm").glob("*.qasm"))
    )
    output_names.extend(
        str(path.relative_to(output_dir))
        for path in sorted((output_dir / "compiled_qpy").glob("*.qpy"))
    )
    outputs = {
        name: {"sha256": sha256_file(output_dir / name), "bytes": (output_dir / name).stat().st_size}
        for name in output_names
    }
    source_hashes = {
        "artifact_id": "azizov-common-core-adaptation-e2-materialization-v1-20261002",
        "inputs": input_paths,
        "upstream_commit": upstream["commit"],
        "upstream_pinned_file_sha256": upstream["pinned_file_sha256"],
        "external_source_hashes": sorted(members_by_hash),
        "outputs": outputs,
    }
    write_json_idempotent(output_dir / "source_hashes.json", source_hashes)

    status_counts = Counter(
        row["source_gnn_status"] for row in terminal_rows
    )
    fold_counts = Counter(str(row["fold"]) for row in terminal_rows)
    old_graph_crosscheck_count = sum(row.get("old_source_topology_cross_check") is True for row in hash_manifest_rows)
    artifact_status = "complete_static_materialization" if len(feature_rows) == 150 and not errors else "partial_static_materialization_with_explicit_unavailable_hashes"
    manifest = {
        "artifact_id": "azizov-common-core-adaptation-e2-materialization-v1-20261002",
        "schema_version": "1.0",
        "protocol_id": protocol["protocol_id"],
        "dictionary_id": dictionary["dictionary_id"],
        "status": artifact_status,
        "scope": "Static feature and graph materialization plus deterministic transpile-summary preflight; no model fit, Aer simulation, runtime measurement, or GPU inspection.",
        "assigned_member_count": len(member_rows),
        "unique_hash_count": len(members_by_hash),
        "available_hash_count": len(feature_rows),
        "unavailable_hash_count": len(errors),
        "representation_dimensions": dimensions,
        "representation_status_counts": dict(status_counts),
        "fold_hash_counts": dict(sorted(fold_counts.items())),
        "representations": {
            "source_gnn": {"global_columns_prefix": "source__", "global_dimension": 51, "graph_file": "source_graphs.jsonl"},
            "hybrid_gnn": {"global_columns": ["source__*", "compiled_ext__*"], "global_dimension": 64, "graph_file": "source_graphs.jsonl"},
            "transpiled_gnn": {"global_columns_prefix": "transpiled__", "global_dimension": 51, "graph_file": "transpiled_graphs.jsonl"},
        },
        "global_feature_column_order": feature_fields,
        "graph_record_schema": {
            "schema_version": "azizov_raw_graph_v1",
            "gate_vocabulary": "Not fitted/materialized in E2; E4 must construct fold-local vocabulary from training hashes only.",
            "sparse_wire_encoding": "qargs/cargs index lists represent fixed 127-wide multi-hot quantum/classical wire blocks; E4 expands them when forming node tensors.",
            "parameter_encoding": "parameter_values_over_pi and parameter_presence_masks are fixed length 4; no angle reduction.",
            "edges": "Deduplicated sorted directed pairs from previous-touch dependencies over quantum and classical wires.",
        },
        "upstream": upstream,
        "environment": environment,
        "existing_source_dag_cross_check": {**old_dag_audit, "matching_hash_count": old_graph_crosscheck_count},
        "hashes": hash_manifest_rows,
        "files": outputs,
        "input_hashes_path": "source_hashes.json",
        "terminal_statuses_path": "representation_coverage_and_terminal_statuses.csv",
        "summary_extractor": {
            "path": str(C44_SUMMARY_EXTRACTOR.relative_to(ROOT)),
            "sha256": input_paths[str(C44_SUMMARY_EXTRACTOR.relative_to(ROOT))],
            "summary_fields": ["physical_width", "physical_depth", "physical_ops", "physical_two_qubit_ops", "swap_count", "dag_node_count"],
            "two_qubit_gate_names": sorted(C44_TWO_QUBIT_NAMES),
            "result": "PASS" if len(replay_rows) == len(feature_rows) and all(row["replay_status"] == "PASS" for row in replay_rows) else "FAIL_OR_UNAVAILABLE",
        },
        "preflight": {
            "member_join": "PASS" if len(member_rows) == 162 else "FAIL",
            "unique_hashes": "PASS" if len(members_by_hash) == 150 else "FAIL",
            "folds_match_c44": "PASS" if all(row["fold_match"] == "true" for row in member_rows) else "FAIL",
            "compiled_replay": "PASS" if len(replay_rows) == len(feature_rows) and all(row["replay_status"] == "PASS" for row in replay_rows) else "PARTIAL_OR_FAIL",
            "all_representations_per_hash": "PASS" if all(row["source_gnn_status"] == row["hybrid_gnn_status"] == row["transpiled_gnn_status"] for row in terminal_rows) else "FAIL",
        },
        "errors": errors,
        "reader_claim": protocol["reader_claim"],
    }
    write_json_idempotent(output_dir / "environment_and_context_lock.json", environment)
    write_json_idempotent(output_dir / "representation_manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "status": artifact_status,
                "assigned_members": len(member_rows),
                "unique_hashes": len(members_by_hash),
                "available_hashes": len(feature_rows),
                "unavailable_hashes": len(errors),
                "compiled_replay_passes": len(replay_rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
