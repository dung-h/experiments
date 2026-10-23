#!/usr/bin/env python3
"""Read-only P0 coverage/provenance audit for Track-A transpiled representations.

This script deliberately reads saved representation pickles one at a time.  It
does not invoke Qiskit, transpile a circuit, fit a model, or read target values
other than checking the frozen panel's declared target availability.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "artifacts/real_qpu/qualified_logical_panel/panel.csv"
REP_ROOT = ROOT / "artifacts/benchmark_v3/real_qpu/compiled_unified"
REPRESENTATIONS = REP_ROOT / "representation_rows.csv"
ANALYTICAL_ATTEMPTS = ROOT / "artifacts/real_qpu/qualified_logical_panel/analytical/attempts.csv"
OUT = ROOT / "artifacts/postfreeze_track_a_transpiled_representation_v1/p0_coverage"
PROTOCOL = ROOT / "artifacts/postfreeze_track_a_transpiled_representation_v1/PROTOCOL.md"
FORMAT = "s71-operation-dag-v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {key: None for key in ("n", "min", "median", "p90", "p95", "p99", "max")}
    ordered = sorted(values)
    def q(probability: float) -> float:
        index = (len(ordered) - 1) * probability
        low, high = int(index), min(int(index) + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return {"n": len(ordered), "min": ordered[0], "median": q(.5), "p90": q(.9),
            "p95": q(.95), "p99": q(.99), "max": ordered[-1]}


def structure_provenance(row: dict[str, str]) -> str:
    if row["compiler_route"] == "preserve_exact_submitted_physical_qasm_no_retranspile":
        return "EXACT_SUBMITTED_PHYSICAL"
    return "FROZEN_RECONSTRUCTION"


def timing_provenance(row: dict[str, str]) -> str:
    tier = row.get("snapshot_tier", "").lower()
    if "historical_exact" in tier:
        return "HISTORICAL_EXACT"
    if "timestamp" in tier:
        return "TIMESTAMP_MATCHED_PROXY"
    if tier:
        return "FROZEN_SNAPSHOT_PROXY"
    return "UNAVAILABLE"


def graph_summary(graph_path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        with graph_path.open("rb") as handle:
            graph = pickle.load(handle)
        if not isinstance(graph, dict) or graph.get("format") != FORMAT:
            return None, "unexpected_graph_format"
        node_count = int(graph["operation_count"])
        edge_count = len(graph["edge_src"])
        if edge_count != len(graph["edge_dst"]):
            return None, "edge_vector_length_mismatch"
        return {
            "dag_node_count": node_count,
            "dag_edge_count": edge_count,
            "graph_num_qubits": int(graph["num_qubits"]),
            "graph_num_clbits": int(graph["num_clbits"]),
            "graph_format": graph["format"],
        }, ""
    except Exception as exc:  # Provenance audit must record, not conceal, failures.
        return None, f"pickle_read_error:{type(exc).__name__}"


def main() -> int:
    panel = read_csv(PANEL)
    reps = read_csv(REPRESENTATIONS)
    analytical_attempts = read_csv(ANALYTICAL_ATTEMPTS)
    panel_ids = [row["canonical_observation_id"] for row in panel]
    expected_sources = {"mali_real_qpu": 340, "qonductor_single_circuit_ibm": 230, "qpack_mcp": 3945}
    source_counts = Counter(row["source_id"] for row in panel)
    if len(panel) != 4515 or len(set(panel_ids)) != 4515 or source_counts != expected_sources:
        raise RuntimeError(f"frozen_panel_invariant_failed: rows={len(panel)} unique={len(set(panel_ids))} sources={dict(source_counts)}")
    if len({row["group_id"] for row in panel}) != 165:
        raise RuntimeError("frozen_group_invariant_failed")
    if any(not row.get("target_seconds") for row in panel):
        raise RuntimeError("target_availability_invariant_failed")

    rep_by_id = {row["canonical_row_id"]: row for row in reps}
    if len(rep_by_id) != len(reps):
        raise RuntimeError("representation_id_not_unique")
    missing_representation = sorted(set(panel_ids) - set(rep_by_id))
    if missing_representation:
        raise RuntimeError("representation_join_missing:" + ",".join(missing_representation[:10]))
    raw_duration = {
        row["canonical_observation_id"]: row
        for row in analytical_attempts
        if row.get("method_id") == "qiskit_estimate_duration_snapshot__raw" and row.get("status") == "predicted"
    }
    if len(raw_duration) != len(panel_ids) or set(raw_duration) != set(panel_ids):
        raise RuntimeError(f"raw_duration_coverage_failed: expected={len(panel_ids)} got={len(raw_duration)}")

    unique_paths = {rep_by_id[oid]["graph_file"] for oid in panel_ids if rep_by_id[oid].get("graph_file")}
    graph_cache: dict[str, tuple[dict[str, Any] | None, str]] = {}
    for index, name in enumerate(sorted(unique_paths), start=1):
        graph_cache[name] = graph_summary(REP_ROOT / "graphs" / name)
        if index % 100 == 0:
            print(f"audited_graphs={index}/{len(unique_paths)}", flush=True)

    observation_rows: list[dict[str, Any]] = []
    graph_rows: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for row in panel:
        rep = rep_by_id[row["canonical_observation_id"]]
        path = rep.get("graph_file", "")
        graph, reason = graph_cache.get(path, (None, "missing_graph_path"))
        duration = raw_duration[row["canonical_observation_id"]]
        available = graph is not None and rep.get("graph_status") == "available"
        structure = structure_provenance(rep)
        timing = timing_provenance(rep)
        try:
            histogram = json.dumps(json.loads(rep.get("gate_counts_json", "{}")), sort_keys=True, separators=(",", ":"))
        except json.JSONDecodeError:
            histogram = ""
            reason = reason or "invalid_gate_counts_json"
            available = False
        output = {
            "canonical_observation_id": row["canonical_observation_id"], "source_id": row["source_id"],
            "group_id": row["group_id"], "backend": row["backend"], "shots": row["shots"],
            "target_available": "YES", "logical_input_status": row["logical_input_tier"],
            "logical_hash": row["source_qasm_sha256"],
            "compiled_input_status": "AVAILABLE" if rep.get("compiled_or_submitted_digest") else "UNAVAILABLE",
            "compiled_hash": rep.get("compiled_or_submitted_digest", ""),
            "structure_provenance": structure, "timing_provenance": timing,
            "backend_snapshot_id": rep.get("target_snapshot_id", ""), "snapshot_hash": rep.get("target_properties_sha256", ""),
            "physical_graph_key": "|".join((rep.get("compiled_or_submitted_digest", ""), rep.get("target_snapshot_id", ""), "p0-v1")),
            "active_physical_width": rep.get("active_width", ""), "physical_depth": rep.get("structural_depth", ""),
            "physical_2q_depth": "UNAVAILABLE", "native_operation_count": rep.get("operation_count", ""),
            "native_2q_operation_count": rep.get("two_qubit_count", ""), "native_gate_histogram": histogram,
            "dag_node_count": graph.get("dag_node_count", "") if graph else "",
            "dag_edge_count": graph.get("dag_edge_count", "") if graph else "",
            "duration_coverage": "AVAILABLE_NOMINAL_SINGLE_SHOT_PROXY",
            "weighted_critical_path": duration["predicted_seconds"],
            "routing_expansion_descriptors": "UNAVAILABLE_IN_P0", "terminal_status": "AVAILABLE" if available else "UNAVAILABLE",
            "failure_reason": reason or rep.get("graph_terminal_reason", ""), "graph_file": path,
            "graph_format": graph.get("graph_format", "") if graph else "",
        }
        observation_rows.append(output)
        if graph and path not in graph_rows:
            graph_rows[path] = {"graph_file": path, "physical_graph_key": output["physical_graph_key"],
                "compiled_or_submitted_digest": rep.get("compiled_or_submitted_digest", ""),
                "snapshot_id": rep.get("target_snapshot_id", ""), "source_id_examples": row["source_id"],
                "dag_node_count": graph["dag_node_count"], "dag_edge_count": graph["dag_edge_count"],
                "graph_num_qubits": graph["graph_num_qubits"], "graph_num_clbits": graph["graph_num_clbits"],
                "graph_format": graph["graph_format"], "serialized_bytes": (REP_ROOT / "graphs" / path).stat().st_size}
        if not available:
            failures.append({"canonical_observation_id": output["canonical_observation_id"], "source_id": output["source_id"],
                "group_id": output["group_id"], "failure_reason": output["failure_reason"], "graph_file": path})

    available_rows = [row for row in observation_rows if row["terminal_status"] == "AVAILABLE"]
    by_source: list[dict[str, Any]] = []
    by_backend: list[dict[str, Any]] = []
    by_provenance: list[dict[str, Any]] = []
    def summarize(rows: list[dict[str, Any]], key: str, destination: list[dict[str, Any]]) -> None:
        for value in sorted({row[key] for row in rows}):
            selected = [row for row in rows if row[key] == value]
            destination.append({key: value, "observation_rows": len(selected), "available_rows": sum(r["terminal_status"] == "AVAILABLE" for r in selected),
                "unique_groups": len({r["group_id"] for r in selected}), "unique_graph_files": len({r["graph_file"] for r in selected if r["graph_file"]}),
                "unique_physical_graph_keys": len({r["physical_graph_key"] for r in selected if r["graph_file"]})})
    summarize(observation_rows, "source_id", by_source)
    summarize(observation_rows, "backend", by_backend)
    for structure in sorted({row["structure_provenance"] for row in observation_rows}):
        for timing in sorted({row["timing_provenance"] for row in observation_rows if row["structure_provenance"] == structure}):
            selected = [row for row in observation_rows if row["structure_provenance"] == structure and row["timing_provenance"] == timing]
            by_provenance.append({"structure_provenance": structure, "timing_provenance": timing, "observation_rows": len(selected),
                "available_rows": sum(r["terminal_status"] == "AVAILABLE" for r in selected), "unique_groups": len({r["group_id"] for r in selected})})

    unique = list(graph_rows.values())
    node_stats, edge_stats = quantiles([int(r["dag_node_count"]) for r in unique]), quantiles([int(r["dag_edge_count"]) for r in unique])
    distribution = [{"measure": "dag_node_count", **node_stats, "over_50000": sum(int(r["dag_node_count"]) > 50_000 for r in unique),
        "over_100000": sum(int(r["dag_node_count"]) > 100_000 for r in unique), "over_300000": sum(int(r["dag_node_count"]) > 300_000 for r in unique)},
        {"measure": "dag_edge_count", **edge_stats, "over_50000": sum(int(r["dag_edge_count"]) > 50_000 for r in unique),
        "over_100000": sum(int(r["dag_edge_count"]) > 100_000 for r in unique), "over_300000": sum(int(r["dag_edge_count"]) > 300_000 for r in unique)}]

    ledger_fields = list(observation_rows[0])
    write_csv(OUT / "observation_coverage.csv", observation_rows, ledger_fields)
    write_csv(OUT / "unique_graphs.csv", unique, list(unique[0]) if unique else ["graph_file"])
    write_csv(OUT / "source_coverage.csv", by_source, ["source_id", "observation_rows", "available_rows", "unique_groups", "unique_graph_files", "unique_physical_graph_keys"])
    write_csv(OUT / "backend_coverage.csv", by_backend, ["backend", "observation_rows", "available_rows", "unique_groups", "unique_graph_files", "unique_physical_graph_keys"])
    write_csv(OUT / "provenance_coverage.csv", by_provenance, ["structure_provenance", "timing_provenance", "observation_rows", "available_rows", "unique_groups"])
    write_csv(OUT / "dag_size_distribution.csv", distribution, list(distribution[0]))
    write_csv(OUT / "failures.csv", failures, ["canonical_observation_id", "source_id", "group_id", "failure_reason", "graph_file"])

    def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        lines.extend("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows)
        return "\n".join(lines)

    report = f"""# P0 coverage and provenance report

## Facts

- Frozen qualified panel: **{len(panel)} observations**, **{len(set(r['group_id'] for r in panel))} groups**; source counts are {dict(sorted(source_counts.items()))}.
- The pinned compiled-unified ledger joined all {len(panel)} panel IDs.  This audit read {len(unique_paths)} distinct saved graph files once each; no circuit was transpiled or regenerated.
- Available saved physical-DAG inputs: **{len(available_rows)}/{len(observation_rows)}** observations; unique graph payloads within the qualified panel: **{len(unique)}**.  The graph-key count is **{len({r['physical_graph_key'] for r in observation_rows})}**, because the same graph payload can occur with a distinct pinned snapshot identity.
- A saved deterministic nominal single-shot duration proxy is available for all {len(raw_duration)} IDs.  It is recorded as a timing proxy, not historical execution-day gate timing or a target label.
- The graph inventory is stored in `unique_graphs.csv`; exact node/edge quantiles and large-graph counts are in `dag_size_distribution.csv`.

### Coverage by source

{markdown_table(by_source, ["source_id", "observation_rows", "available_rows", "unique_groups", "unique_graph_files", "unique_physical_graph_keys"])}

### Coverage by provenance

{markdown_table(by_provenance, ["structure_provenance", "timing_provenance", "observation_rows", "available_rows", "unique_groups"])}

### Saved physical-DAG size distribution

{markdown_table(distribution, ["measure", "n", "median", "p90", "p95", "p99", "max", "over_50000", "over_100000", "over_300000"])}

## Provenance interpretation

`EXACT_SUBMITTED_PHYSICAL` applies only to the Qonductor route which preserves archived submitted physical QASM.  `FROZEN_RECONSTRUCTION` means the saved physical view was deterministically materialized from the declared Ma–Li logical or QPack structural route; it does **not** assert recovery of the historical executed compilation.  Snapshot tiers are carried separately and are not promoted to historical calibration timing.

## Decision

P0 is an availability/provenance audit only.  It has not evaluated H1–H4, fitted a learner, extracted new timing features, or authorized P1/P2.  Any P1 comparison must use the saved feature/graph identity and an exact common-success population rather than substituting newly transpiled inputs.

## Inputs

- `{PANEL.relative_to(ROOT)}` SHA256 `{sha256(PANEL)}`
- `{REPRESENTATIONS.relative_to(ROOT)}` SHA256 `{sha256(REPRESENTATIONS)}`
- `{ANALYTICAL_ATTEMPTS.relative_to(ROOT)}` SHA256 `{sha256(ANALYTICAL_ATTEMPTS)}`
- `{PROTOCOL.relative_to(ROOT)}` SHA256 `{sha256(PROTOCOL)}`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    receipt = {"status": "PASS", "scope": "read_only_p0_coverage_provenance", "panel_rows": len(panel), "panel_groups": 165,
        "panel_sources": dict(sorted(source_counts.items())), "joined_rows": len(observation_rows), "available_rows": len(available_rows),
        "unique_graph_files": len(unique), "graph_files_examined": len(unique_paths), "failures": len(failures),
        "duration_proxy_available_rows": len(raw_duration), "inputs": {str(PANEL.relative_to(ROOT)): sha256(PANEL), str(REPRESENTATIONS.relative_to(ROOT)): sha256(REPRESENTATIONS), str(ANALYTICAL_ATTEMPTS.relative_to(ROOT)): sha256(ANALYTICAL_ATTEMPTS), str(PROTOCOL.relative_to(ROOT)): sha256(PROTOCOL)}}
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
