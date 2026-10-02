#!/usr/bin/env python3
"""Materialize read-only Azizov common-core inputs; never fits or executes a model."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
PANEL_DIR = ROOT / "artifacts/benchmark_v1/sim_common_q16_manifest_20260927"
PANEL_CSV = PANEL_DIR / "sim_common_q16_manifest.csv"
PANEL_JSON = PANEL_DIR / "manifest.json"
LABEL_CSV = ROOT / "artifacts/benchmark_v1/c44_aer_q16_full_panel_evaluation_20260927/aer_q16_reduced_warm.csv"
PROTOCOL = ROOT / "benchmark_v1/protocol/azizov_common_core_gnn_v1.json"
S82 = ROOT / "benchmark_v1/decisions/S82_UNIFIED_BENCHMARK_CONTRACT_20261002.md"
UPSTREAM = Path("/home/server/Documents/Quantum-Execution-Time-Prediction")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, data: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def upstream_blob(path: str, commit: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(UPSTREAM), "show", f"{commit}:{path}"],
        check=False,
        capture_output=True,
    )
    return sha256_bytes(result.stdout) if result.returncode == 0 else None


def bit_index(circuit: QuantumCircuit, bit: object) -> int:
    return int(circuit.find_bit(bit).index)


def parameter_text(value: object) -> str:
    try:
        return format(float(value), ".17g")
    except (TypeError, ValueError):
        return str(value)


def source_dag(path: Path, expected_hash: str) -> dict[str, object]:
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_hash:
        raise ValueError(f"source QASM hash mismatch: {path}")
    circuit = QuantumCircuit.from_qasm_file(str(path))
    dag = circuit_to_dag(circuit)
    op_nodes = list(dag.topological_op_nodes())
    index = {node: i for i, node in enumerate(op_nodes)}
    nodes = []
    for i, node in enumerate(op_nodes):
        nodes.append(
            {
                "node_id": i,
                "operation": str(node.name),
                "qargs": [bit_index(circuit, bit) for bit in node.qargs],
                "cargs": [bit_index(circuit, bit) for bit in node.cargs],
                "params_text": [parameter_text(p) for p in node.op.params],
            }
        )
    edges = set()
    for src, dst, wire in dag.edges():
        if src not in index or dst not in index:
            continue
        wire_kind = "qubit" if wire in circuit.qubits else "clbit"
        edges.add((index[src], index[dst], wire_kind, bit_index(circuit, wire)))
    edge_rows = [
        {"source_node": a, "target_node": b, "wire_kind": kind, "wire_index": wire}
        for a, b, kind, wire in sorted(edges)
    ]
    payload = {
        "source_qasm_sha256": expected_hash,
        "representation": "source_qiskit_dag_topology_v1_not_Azizov_GNN_ready",
        "num_qubits": int(circuit.num_qubits),
        "num_clbits": int(circuit.num_clbits),
        "circuit_depth": int(circuit.depth()),
        "node_count": len(nodes),
        "edge_count": len(edge_rows),
        "nodes": nodes,
        "edges": edge_rows,
    }
    return {**payload, "topology_sha256": sha256_bytes(canonical(payload))}


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    panel = rows(PANEL_CSV)
    labels = rows(LABEL_CSV)
    core = [r for r in panel if r["stratum"] == "core_q2_q9"]
    label_core = [r for r in labels if r["stratum"] == "core_q2_q9"]
    label_by_key = {(r["circuit_id"], r["source_sha256"]): r for r in label_core}
    if len(label_by_key) != len(label_core):
        raise ValueError("C44 core labels are not unique by (circuit_id, source_sha256)")
    if len(core) != 162 or len(label_core) != 162 or len(label_by_key) != 162:
        raise ValueError(f"expected 162 panel and label rows, got {len(core)} / {len(label_core)}")

    source_root = Path(json.loads(PANEL_JSON.read_text(encoding="utf-8"))["source_root"])
    manifest_rows: list[dict[str, object]] = []
    dag_by_hash: dict[str, dict[str, object]] = {}
    dag_errors: list[dict[str, str]] = []
    folds_by_hash: dict[str, set[str]] = defaultdict(set)
    for p in core:
        stem = Path(p["basename"]).stem
        key = (stem, p["qasm_sha256"])
        label = label_by_key.get(key)
        if label is None:
            raise ValueError(f"missing exact C44 label for {key}")
        if not label.get("fold", "").strip():
            raise ValueError(f"missing frozen fold for {key}")
        source = source_root / p["basename"]
        if not source.is_file():
            raise FileNotFoundError(source)
        live_hash = sha256_file(source)
        if live_hash != p["qasm_sha256"] or live_hash != label["source_sha256"]:
            raise ValueError(f"source hash does not match panel+label for {key}")
        folds_by_hash[live_hash].add(label["fold"])
        if live_hash not in dag_by_hash:
            try:
                dag = source_dag(source, live_hash)
                # Reparse once to verify deterministic serialization under the pinned environment.
                dag_again = source_dag(source, live_hash)
                if dag["topology_sha256"] != dag_again["topology_sha256"]:
                    raise ValueError("source DAG serialization changed on repeated parse")
                dag_by_hash[live_hash] = dag
            except Exception as exc:  # preserve an explicit terminal, not a fabricated DAG
                dag_errors.append({"source_qasm_sha256": live_hash, "error": f"{type(exc).__name__}: {exc}"})
        manifest_rows.append(
            {
                "panel_member_id": p["panel_member_id"],
                "circuit_id": stem,
                "basename": p["basename"],
                "family": p["family"],
                "width_qubits": p["width_qubits"],
                "stratum": p["stratum"],
                "source_path": str(source),
                "source_sha256": live_hash,
                "fold": label["fold"],
                "target_clock": "warm_execution",
                "observed_seconds": label["observed_seconds"],
                "sessions": label["sessions"],
                "raw_repetitions": label["raw_repetitions"],
                "backend_name": label["backend_name"],
                "optimization_level": label["optimization_level"],
                "shots": label["shots"],
                "source_dag_status": "materialized" if live_hash in dag_by_hash else "parse_unavailable",
                "source_dag_sha256": dag_by_hash.get(live_hash, {}).get("topology_sha256", ""),
            }
        )
    if len(manifest_rows) != 162 or len({r["panel_member_id"] for r in manifest_rows}) != 162:
        raise ValueError("panel membership is not one-to-one")
    if any(len(folds) != 1 for folds in folds_by_hash.values()):
        raise ValueError("an exact-QASM hash crosses frozen folds")
    if len(folds_by_hash) != 150:
        raise ValueError(f"expected 150 unique core QASM hashes, got {len(folds_by_hash)}")

    dag_records = [dag_by_hash[h] for h in sorted(dag_by_hash)]
    node_counts = [int(d["node_count"]) for d in dag_records]
    edge_counts = [int(d["edge_count"]) for d in dag_records]
    gate_counts = Counter(n["operation"] for d in dag_records for n in d["nodes"])
    invalid_dag_records = []
    for d in dag_records:
        for n in d["nodes"]:
            if not n["operation"] or any(q < 0 or q >= d["num_qubits"] for q in n["qargs"]):
                invalid_dag_records.append(d["source_qasm_sha256"])
            if any(c < 0 or c >= d["num_clbits"] for c in n["cargs"]):
                invalid_dag_records.append(d["source_qasm_sha256"])
        if any(e["source_node"] >= e["target_node"] for e in d["edges"]):
            invalid_dag_records.append(d["source_qasm_sha256"])
    if invalid_dag_records:
        raise ValueError(f"DAG structural QA failed for {len(set(invalid_dag_records))} records")

    fields = list(manifest_rows[0])
    write_csv(OUT / "fold_assignments_and_hash_audit.csv", manifest_rows, fields)
    with (OUT / "source_dag_topology.jsonl").open("w", encoding="utf-8") as f:
        for record in dag_records:
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")

    try:
        upstream_commit = subprocess.run(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        upstream_commit = None
    evidence = {
        "feature_dictionary_status": "blocked_incomplete_schema_do_not_fit",
        "paper_claimed_global_counts": {"source": 41, "hybrid": 54, "transpiled": 41},
        "paper_breakdown": {
            "source": "36 Ma-Li-derived fields plus five SupermarQ metrics",
            "hybrid": "the 41 source globals plus 13 post-transpilation globals",
            "transpiled": "41 globals computed from the transpiled circuit",
        },
        "five_supermarq_names_supported_by_pinned_upstream_code": [
            "program_communication", "critical_depth", "entanglement_ratio", "parallelism", "liveness"
        ],
        "unresolved_fields": [
            "exact 36-field Ma-Li-derived name-to-definition mapping",
            "exact 13 post-transpilation field names and definitions",
            "complete graph node-feature encoding used by Azizov",
            "whether pinned Ma-Li source helper's data-dependent pruning corresponds to Azizov's 36 fields",
        ],
        "evidence": {
            "paper": "https://arxiv.org/html/2609.12980v1 (feature counts and categories; no complete dictionaries recovered)",
            "pinned_upstream_repository": str(UPSTREAM),
            "pinned_upstream_commit": upstream_commit,
            "helper_py_sha256": upstream_blob("data_preparation/helper.py", upstream_commit) if upstream_commit else None,
            "utils_py_sha256": upstream_blob("data_preparation/utils.py", upstream_commit) if upstream_commit else None,
            "interpretation": "The upstream code candidate exposes 44 gate-count slots, qubit count, depth and five SupermarQ metrics before data-dependent pruning; this does not establish the paper's exact 36-field subset.",
        },
        "prohibition": "No unnamed proxy or padded vector was generated; source/transpiled/hybrid model inputs remain blocked.",
    }
    write_json(OUT / "feature_dictionary.json", evidence)

    context_path = ROOT / protocol["local_cell"]["frozen_context_source"]
    env_lock = {
        "protocol_id": protocol["protocol_id"],
        "required_frozen_context_path": str(context_path),
        "required_frozen_context_status": "present" if context_path.is_file() else "missing",
        "candidate_parse_environment_only": {
            "python": platform.python_version(),
            "qiskit": package_version("qiskit-terra") or package_version("qiskit"),
            "qiskit_aer": package_version("qiskit-aer"),
            "torch": package_version("torch"),
            "torch_geometric": package_version("torch-geometric"),
            "role": "QASM parsing and source-DAG static materialization only; not approved as the frozen timing/transpilation context",
        },
        "execution": {
            "model_fit": False,
            "cuda_smoke_or_training": False,
            "simulator_timing": False,
            "transpilation": False,
        },
        "transpiled_representation_status": "blocked_missing_historical_target_and_noise_hash; current FakeSherbrooke is not substituted",
    }
    write_json(OUT / "environment_and_context_lock.json", env_lock)

    fold_counts = Counter(r["fold"] for r in manifest_rows)
    alias_groups = sum(1 for digest in folds_by_hash if sum(r["source_sha256"] == digest for r in manifest_rows) > 1)
    audit = {
        "artifact_id": "azizov-common-core-static-materialization-v1-20261002",
        "status": "partial_static_materialization_feature_and_transpiled_inputs_blocked",
        "assigned_panel_rows": len(manifest_rows),
        "unique_exact_qasm_hashes": len(folds_by_hash),
        "same_hash_alias_groups": alias_groups,
        "frozen_fold_counts": dict(sorted(fold_counts.items())),
        "hash_leakage": {"cross_fold_exact_hash_groups": 0, "status": "PASS"},
        "label_join": {"matched_rows": len(manifest_rows), "unmatched": 0, "duplicate_join_keys": 0, "status": "PASS"},
        "source_hash_audit": {"matched_rows": len(manifest_rows), "mismatch": 0, "status": "PASS"},
        "source_dag": {
            "unique_hashes_materialized": len(dag_records),
            "parse_errors": dag_errors,
            "nodes_min_max": [min(node_counts), max(node_counts)] if node_counts else None,
            "edges_min_max": [min(edge_counts), max(edge_counts)] if edge_counts else None,
            "gate_name_counts": dict(sorted(gate_counts.items())),
            "shape_missing_constant_feature_audit": "not_run_for_41/54/41_vectors_because_evidence-backed_schema_is_incomplete",
            "structural_qa": "PASS" if not dag_errors else "PARTIAL_WITH_EXPLICIT_PARSE_UNAVAILABLE",
        },
        "representation_readiness": {
            "source_dag_topology": "materialized_topology_only_not_Azizov_GNN_ready",
            "source_41_globals": "blocked_exact_36_field_dictionary_unresolved",
            "hybrid_54_globals": "blocked_exact_13_post_transpilation_field_dictionary_unresolved",
            "transpiled_dag_and_41_globals": "blocked_frozen_target_noise_hash_missing_and_feature_dictionary_unresolved",
        },
        "run_authority": "static materialization only; model execution remains unauthorized pending S-A1 signature",
    }
    write_json(OUT / "static_qa.json", audit)

    source_files = [PROTOCOL, S82, PANEL_CSV, PANEL_JSON, LABEL_CSV]
    source_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_files}
    source_hashes["upstream_repository_commit"] = upstream_commit
    source_hashes["external_source_qasm_hashes"] = sorted(folds_by_hash)
    source_hashes["source_dag_jsonl_sha256"] = sha256_file(OUT / "source_dag_topology.jsonl")
    write_json(OUT / "source_hashes.json", source_hashes)

    readme = """# Azizov common-core static inputs (partial)

This pack freezes the 162-row `core_q2_q9` join to the existing C44 warm-Aer labels and materializes source-QASM DAG topology for the unique exact hashes. It does **not** execute a simulator, transpile circuits, fit a predictor, or authorize training.

## Contents

- `fold_assignments_and_hash_audit.csv`: 162 panel members, exact-QASM hashes, frozen folds and joined observed warm-execution labels.
- `source_dag_topology.jsonl`: 150 unique source DAGs; this is topology-only and is not claimed to be the Azizov GNN input representation.
- `feature_dictionary.json`: evidence-backed counts and explicit unresolved fields. No 41/54/41 vectors or unnamed proxies are fabricated.
- `environment_and_context_lock.json`: candidate parser environment and missing frozen historical target/noise context.
- `static_qa.json`, `source_hashes.json`: join, fold-leakage, source-hash, structural and provenance audits.
- `materialize_static_pack.py`: static-only materializer.

## Reproduce the static pack

From the repository root, with the candidate Qiskit environment activated:

```bash
python artifacts/benchmark_v3/simulator/azizov_common_core_gnn_v1_materialization/materialize_static_pack.py
```

The script reads the existing panel, C44 labels, protocol, and external source QASM files; it does not modify those inputs. The pinned frozen target/noise file named by the protocol is absent, so the transpiled representation remains blocked. The 36 Ma–Li-derived field names and exact 13 post-transpilation fields are not evidenced by the pinned sources, so all model-fit variants remain blocked pending method review.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    (OUT / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in files), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUT), "status": audit["status"], "rows": len(manifest_rows), "unique_hashes": len(dag_records), "parse_errors": len(dag_errors)}, sort_keys=True))


if __name__ == "__main__":
    main()
