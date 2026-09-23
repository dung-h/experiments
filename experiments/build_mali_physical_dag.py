#!/usr/bin/env python3
"""Build a compact, resumable physical native-operation DAG corpus for Ma--Li.

The builder deliberately runs in the frozen Qiskit 1.4.1 / IBM Runtime 0.36.1
environment used by ``mali_qcre_proxy.py``.  It stores each unique
``(logical-QASM hash, backend)`` graph once and keeps observed labels in a
separate manifest.  The files are compact NumPy arrays rather than pickled
PyG objects, so the corpus can be loaded by the separate CUDA/PyG environment.

This is a current FakeBackend reconstruction and is labelled as such in every
artifact.  It does not recover the historical physical circuit used for the
original IBM observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import qiskit
import qiskit_ibm_runtime
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit_ibm_runtime.fake_provider import FakeKyoto, FakeOsaka


ROOT = Path(__file__).resolve().parents[1]
OPCODE_NAMES = (
    "barrier", "delay", "ecr", "id", "measure", "reset", "rz", "sx", "x",
    "switch_case", "for_loop", "if_else", "while_loop", "unknown",
)
OPCODE_ID = {name: index for index, name in enumerate(OPCODE_NAMES)}
UNKNOWN_OPCODE = OPCODE_ID["unknown"]


def resolve_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    for candidate in (ROOT / "work" / "mali", ROOT.parent / "Quantum-Execution-Time-Prediction"):
        if (candidate / "data" / "quantum_circuits").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Ma--Li checkout not found; pass --mali-root")


def load_labels(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for backend in ("osaka", "kyoto"):
        path = root / "data" / f"{backend}_time_taken.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for source in csv.DictReader(handle):
                circuit = source.get("quantum_circuit") or source.get("circuit_name")
                if not circuit:
                    raise ValueError(f"missing circuit name in {path}")
                qasm_path = root / "data" / "quantum_circuits" / f"{circuit}.qasm"
                digest = hashlib.sha256(qasm_path.read_bytes()).hexdigest()
                rows.append(
                    {
                        "row_id": f"{backend}:{len(rows)}",
                        "backend": backend,
                        "circuit": circuit,
                        "qasm_path": str(qasm_path.relative_to(root)),
                        "qasm_sha256": digest,
                        "target_seconds": float(source["time_taken"]),
                    }
                )
    return rows


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return str(value)


def snapshot_payload(backend: Any) -> dict[str, object]:
    configuration = backend.configuration().to_dict()
    properties = backend.properties().to_dict()
    return {
        "class": type(backend).__name__,
        "backend_version": getattr(backend, "backend_version", None),
        "qiskit_version": qiskit.__version__,
        "qiskit_ibm_runtime_version": qiskit_ibm_runtime.__version__,
        "num_qubits": int(backend.num_qubits),
        "dt_seconds": float(backend.dt),
        "basis_gates": list(configuration.get("basis_gates", [])),
        "coupling_map_edges": [list(edge) for edge in backend.coupling_map.get_edges()],
        "configuration": jsonable(configuration),
        "properties": jsonable(properties),
    }


def snapshot_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def qarg_index(circuit: QuantumCircuit, bit: Any) -> int:
    return int(circuit.find_bit(bit).index)


def duration_and_error(instruction: Any, locations: tuple[int, ...], backend: Any) -> tuple[int, float, int, str | None]:
    """Return duration in dt, error, missing-error flag and missing-op name."""
    properties = None
    try:
        properties = backend.target[instruction.name].get(locations)
        if properties is None and len(locations) == 2:
            properties = backend.target[instruction.name].get((locations[1], locations[0]))
    except (KeyError, TypeError):
        properties = None
    duration_seconds = getattr(properties, "duration", None)
    error = getattr(properties, "error", None)
    if duration_seconds is None:
        raw_duration = getattr(instruction, "duration", None)
        raw_unit = getattr(instruction, "unit", "dt")
        if raw_duration is not None:
            duration_seconds = float(raw_duration) * float(backend.dt) if raw_unit == "dt" else float(raw_duration)
        else:
            return 0, float("nan"), 1, instruction.name
    duration_dt = max(0, int(round(float(duration_seconds) / float(backend.dt))))
    if error is None or not math.isfinite(float(error)):
        return duration_dt, float("nan"), 1, None
    return duration_dt, float(error), 0, None


def node_id(node: Any) -> int | None:
    value = getattr(node, "_node_id", None)
    return int(value) if value is not None else None


def build_graph(original: QuantumCircuit, compiled: QuantumCircuit, backend: Any) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    dag = circuit_to_dag(compiled)
    nodes = list(dag.topological_op_nodes())
    node_index = {node_id(node): index for index, node in enumerate(nodes)}
    if any(value is None for value in node_index):
        raise RuntimeError("Qiskit DAG operation node has no stable _node_id")

    qbit_index = {id(bit): index for index, bit in enumerate(compiled.qubits)}
    node_opcode: list[int] = []
    node_q0: list[int] = []
    node_q1: list[int] = []
    node_duration: list[int] = []
    node_error: list[float] = []
    node_error_missing: list[int] = []
    node_duration_missing: list[int] = []
    unsupported: set[str] = set()
    unknown_opcodes: set[str] = set()
    active_qubits: set[int] = set()

    for node in nodes:
        instruction = node.op
        name = str(instruction.name)
        if name not in OPCODE_ID:
            unknown_opcodes.add(name)
        node_opcode.append(OPCODE_ID.get(name, UNKNOWN_OPCODE))
        qargs = tuple(qarg_index(compiled, bit) for bit in node.qargs)
        if qargs:
            active_qubits.update(qargs)
        node_q0.append(qargs[0] if len(qargs) >= 1 else -1)
        node_q1.append(qargs[1] if len(qargs) >= 2 else -1)
        if getattr(instruction, "_directive", False) or name == "barrier":
            duration_dt, error, error_missing, missing = 0, float("nan"), 1, None
        else:
            duration_dt, error, error_missing, missing = duration_and_error(instruction, qargs, backend)
        if missing:
            unsupported.add(missing)
        node_duration.append(duration_dt)
        node_error.append(error)
        node_error_missing.append(error_missing)
        node_duration_missing.append(1 if missing else 0)

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_wire: list[int] = []
    edge_kind: list[int] = []
    for source, target, wire in dag.edges():
        source_index = node_index.get(node_id(source))
        target_index = node_index.get(node_id(target))
        if source_index is None or target_index is None:
            continue
        wire_index = qbit_index.get(id(wire), -1)
        edge_src.append(source_index)
        edge_dst.append(target_index)
        edge_wire.append(wire_index)
        edge_kind.append(0 if wire_index >= 0 else 1)

    n_nodes = len(nodes)
    predecessors: list[list[int]] = [[] for _ in range(n_nodes)]
    successors: list[list[int]] = [[] for _ in range(n_nodes)]
    for source, target in zip(edge_src, edge_dst):
        predecessors[target].append(source)
        successors[source].append(target)

    forward = np.zeros(n_nodes, dtype=np.uint64)
    topo_layer = np.zeros(n_nodes, dtype=np.uint32)
    durations = np.asarray(node_duration, dtype=np.uint64)
    for index in range(n_nodes):
        if predecessors[index]:
            forward[index] = durations[index] + max(int(forward[item]) for item in predecessors[index])
            topo_layer[index] = 1 + max(int(topo_layer[item]) for item in predecessors[index])
        else:
            forward[index] = durations[index]
            topo_layer[index] = 0
    reverse = np.zeros(n_nodes, dtype=np.uint64)
    for index in range(n_nodes - 1, -1, -1):
        if successors[index]:
            reverse[index] = durations[index] + max(int(reverse[item]) for item in successors[index])
        else:
            reverse[index] = durations[index]
    critical_path = int(forward.max(initial=0))
    criticality = np.maximum(
        0,
        critical_path - (forward.astype(np.int64) + reverse.astype(np.int64) - durations.astype(np.int64)),
    ).astype(np.uint64)

    arrays = {
        "node_opcode": np.asarray(node_opcode, dtype=np.uint8),
        "node_q0": np.asarray(node_q0, dtype=np.int16),
        "node_q1": np.asarray(node_q1, dtype=np.int16),
        "node_duration_dt": durations.astype(np.uint64),
        "node_duration_missing": np.asarray(node_duration_missing, dtype=np.uint8),
        "node_gate_error": np.asarray(node_error, dtype=np.float32),
        "node_gate_error_missing": np.asarray(node_error_missing, dtype=np.uint8),
        "node_topo_layer": topo_layer,
        "node_forward_path_dt": forward,
        "node_reverse_path_dt": reverse,
        "node_criticality_dt": criticality,
        "edge_src": np.asarray(edge_src, dtype=np.uint32),
        "edge_dst": np.asarray(edge_dst, dtype=np.uint32),
        "edge_wire": np.asarray(edge_wire, dtype=np.int16),
        "edge_kind": np.asarray(edge_kind, dtype=np.uint8),
        "active_physical_qubits": np.asarray(sorted(active_qubits), dtype=np.int16),
        "opcode_names": np.asarray(OPCODE_NAMES),
    }
    metadata = {
        "n_nodes": n_nodes,
        "n_edges": len(edge_src),
        "active_physical_width": len(active_qubits),
        "compiled_width": int(compiled.num_qubits),
        "compiled_depth": int(compiled.depth()),
        "compiled_gate_count": int(len(compiled.data)),
        "compiled_counts": {str(name): int(count) for name, count in compiled.count_ops().items()},
        "native_two_qubit_gate_count": int(sum(
            count for name, count in compiled.count_ops().items()
            if len(tuple(next((entry.qubits for entry in compiled.data if entry.operation.name == name), ()))) >= 2
        )),
        "qcre_critical_path_dt": critical_path,
        "qcre_critical_path_seconds": critical_path * float(backend.dt),
        "zero_duration_nodes": int(np.sum(durations == 0)),
        "duration_missing_nodes": int(np.sum(np.asarray(node_duration_missing) != 0)),
        "unsupported_duration_ops": sorted(unsupported),
        "unknown_opcode_names": sorted(unknown_opcodes),
        "edge_kind_counts": {
            "quantum": int(sum(kind == 0 for kind in edge_kind)),
            "classical": int(sum(kind == 1 for kind in edge_kind)),
        },
    }
    return arrays, metadata


def graph_id(qasm_sha256: str, backend_name: str, snapshot: str, seed: int, optimization_level: int) -> str:
    value = f"{qasm_sha256}|{backend_name}|{snapshot}|qiskit={qiskit.__version__}|seed={seed}|opt={optimization_level}"
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def layout_metadata(original: QuantumCircuit, compiled: QuantumCircuit) -> dict[str, object]:
    output: dict[str, object] = {}
    layout = getattr(compiled, "layout", None)
    initial = getattr(layout, "initial_layout", None) if layout is not None else None
    mapping: dict[str, int] = {}
    if initial is not None:
        for index, qubit in enumerate(original.qubits):
            try:
                mapping[str(index)] = int(initial[qubit])
            except (KeyError, TypeError, ValueError):
                continue
    output["logical_to_initial_physical"] = mapping
    return output


def save_graph(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def load_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["metadata_json"].item()))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mali-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work" / "mali_physical_dag_v1")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "validation" / "mali_physical_dag_v1")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit label rows for a smoke run")
    args = parser.parse_args()

    mali_root = resolve_root(str(args.mali_root) if args.mali_root else None)
    labels = load_labels(mali_root)
    if args.limit is not None:
        labels = labels[: args.limit]

    backends = {"osaka": FakeOsaka(), "kyoto": FakeKyoto()}
    snapshots: dict[str, tuple[str, dict[str, object]]] = {}
    for name, backend in backends.items():
        payload = snapshot_payload(backend)
        digest = snapshot_id(payload)
        snapshots[name] = (digest, payload)
        path = args.output_dir / "backend" / f"{name}_{digest}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    graph_dir = args.output_dir / "graphs"
    graph_rows: dict[str, dict[str, object]] = {}
    build_started = time.time()
    unique_keys: dict[tuple[str, str], dict[str, object]] = {}
    for label in labels:
        unique_keys[(str(label["qasm_sha256"]), str(label["backend"]))] = label

    for position, ((digest, backend_name), label) in enumerate(sorted(unique_keys.items()), start=1):
        snapshot_digest, _ = snapshots[backend_name]
        gid = graph_id(digest, backend_name, snapshot_digest, args.seed, args.optimization_level)
        graph_path = graph_dir / f"{gid}.npz"
        started = time.time()
        if graph_path.exists():
            metadata = load_metadata(graph_path)
            status = "cached"
        else:
            qasm_path = mali_root / str(label["qasm_path"])
            original = QuantumCircuit.from_qasm_file(str(qasm_path))
            compiled = transpile(
                original,
                backend=backends[backend_name],
                optimization_level=args.optimization_level,
                seed_transpiler=args.seed,
            )
            arrays, metadata = build_graph(original, compiled, backends[backend_name])
            metadata.update(
                {
                    "graph_id": gid,
                    "backend": backend_name,
                    "qasm_sha256": digest,
                    "snapshot_id": snapshot_digest,
                    "seed_transpiler": args.seed,
                    "optimization_level": args.optimization_level,
                    "qiskit_version": qiskit.__version__,
                    "qiskit_ibm_runtime_version": qiskit_ibm_runtime.__version__,
                    "provenance_class": "CURRENT_FAKE_SNAPSHOT_PROXY",
                    "layout": layout_metadata(original, compiled),
                }
            )
            save_graph(graph_path, arrays, metadata)
            status = "built"
        metadata = dict(metadata)
        metadata.update(
            {
                "graph_id": gid,
                "backend": backend_name,
                "qasm_sha256": digest,
                "snapshot_id": snapshot_digest,
            }
        )
        graph_rows[gid] = metadata
        elapsed = time.time() - started
        print(
            f"physical-dag {position}/{len(unique_keys)} {status} {backend_name} "
            f"nodes={metadata['n_nodes']} edges={metadata['n_edges']} "
            f"active_q={metadata['active_physical_width']} elapsed={elapsed:.2f}s",
            flush=True,
        )

    manifest_rows: list[dict[str, object]] = []
    for label in labels:
        digest = str(label["qasm_sha256"])
        backend_name = str(label["backend"])
        snapshot_digest, _ = snapshots[backend_name]
        gid = graph_id(digest, backend_name, snapshot_digest, args.seed, args.optimization_level)
        record = graph_rows[gid]
        manifest_rows.append(
            {
                "row_id": label["row_id"],
                "circuit": label["circuit"],
                "backend": backend_name,
                "qasm_path": label["qasm_path"],
                "qasm_sha256": digest,
                "target_seconds": label["target_seconds"],
                "graph_id": gid,
                "snapshot_id": snapshot_digest,
                "n_nodes": record["n_nodes"],
                "n_edges": record["n_edges"],
                "active_physical_width": record["active_physical_width"],
                "compiled_depth": record["compiled_depth"],
                "qcre_critical_path_seconds": record["qcre_critical_path_seconds"],
                "provenance_class": "CURRENT_FAKE_SNAPSHOT_PROXY",
            }
        )

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.artifact_dir / "manifest.csv", manifest_rows)
    write_csv(args.artifact_dir / "graph_records.csv", [graph_rows[key] for key in sorted(graph_rows)])
    summary = {
        "generated_utc": "2026-09-21",
        "n_label_rows": len(labels),
        "n_unique_graphs": len(graph_rows),
        "n_cached_or_built": len(graph_rows),
        "output_dir": str(args.output_dir),
        "artifact_dir": str(args.artifact_dir),
        "seed_transpiler": args.seed,
        "optimization_level": args.optimization_level,
        "qiskit_version": qiskit.__version__,
        "qiskit_ibm_runtime_version": qiskit_ibm_runtime.__version__,
        "provenance_class": "CURRENT_FAKE_SNAPSHOT_PROXY",
        "target_semantics": (
            "Ma-Li Osaka/Kyoto observed result.time_taken, 1024 shots, queue excluded; "
            "target rows remain separate from reconstructed physical graphs"
        ),
        "graph_semantics": "transpiled native-operation DAG with current FakeOsaka/FakeKyoto target durations",
        "totals": {
            "nodes": int(sum(int(item["n_nodes"]) for item in graph_rows.values())),
            "edges": int(sum(int(item["n_edges"]) for item in graph_rows.values())),
            "active_qubits_max": int(max(int(item["active_physical_width"]) for item in graph_rows.values())),
            "duration_missing_nodes": int(sum(int(item["duration_missing_nodes"]) for item in graph_rows.values())),
        },
        "elapsed_seconds": time.time() - build_started,
        "resume_policy": "existing graph NPZ with matching graph_id is loaded and not rebuilt",
    }
    (args.artifact_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = (
        "# Ma--Li compact physical-DAG corpus\n\n"
        "This artifact is a `CURRENT_FAKE_SNAPSHOT_PROXY`. It transpiles the Ma--Li\n"
        "logical QASM with FakeOsaka/FakeKyoto under the frozen Qiskit protocol and\n"
        "stores one compact NPZ graph per `(QASM hash, backend)` cell. The observed\n"
        "`result.time_taken` labels are kept in `manifest.csv`; they are not treated\n"
        "as a duration generated by the fake target.\n\n"
        f"- label rows: **{summary['n_label_rows']}**\n"
        f"- unique physical graphs: **{summary['n_unique_graphs']}**\n"
        f"- native operation nodes: **{summary['totals']['nodes']}**\n"
        f"- dependency edges: **{summary['totals']['edges']}**\n"
        f"- maximum active physical width: **{summary['totals']['active_qubits_max']}**\n"
        f"- duration-missing nodes: **{summary['totals']['duration_missing_nodes']}**\n\n"
        "Each graph stores native opcode, physical qargs, target duration in dt,\n"
        "gate-error availability, topological layer, forward/reverse weighted path,\n"
        "criticality, dependency wire and dependency kind. Zero-duration `rz` is\n"
        "retained as a valid target property; it is not marked missing.\n\n"
        "The graph is suitable for a static QCRE residual or a small directed DAG\n"
        "model. It does not establish historical physical-circuit truth, and it\n"
        "must be evaluated with the grouped-QASM/strict backend splits in the\n"
        "deep protocol document.\n"
    )
    (args.artifact_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
