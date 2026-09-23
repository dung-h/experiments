#!/usr/bin/env python3
"""Extract compact graph-level descriptors from the Ma--Li NPZ corpus."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def describe(graph_path: Path) -> dict[str, object]:
    with np.load(graph_path, allow_pickle=False) as archive:
        opcode = archive["node_opcode"].astype(np.int64)
        duration = archive["node_duration_dt"].astype(np.float64)
        layer = archive["node_topo_layer"].astype(np.int64)
        forward = archive["node_forward_path_dt"].astype(np.float64)
        reverse = archive["node_reverse_path_dt"].astype(np.float64)
        criticality = archive["node_criticality_dt"].astype(np.float64)
        edge_kind = archive["edge_kind"].astype(np.int64)
        active = archive["active_physical_qubits"].astype(np.int64)

    layer_counts = np.bincount(layer) if len(layer) else np.zeros(1)
    nonzero_duration = duration[duration > 0]
    values: dict[str, object] = {
        "rich_n_nodes": len(opcode),
        "rich_n_edges": len(edge_kind),
        "rich_n_layers": len(layer_counts),
        "rich_active_width": len(active),
        "rich_zero_duration_fraction": float(np.mean(duration == 0)) if len(duration) else 0.0,
        "rich_duration_sum_dt": float(duration.sum()),
        "rich_duration_max_dt": float(duration.max(initial=0)),
        "rich_duration_nonzero_mean_dt": float(nonzero_duration.mean()) if len(nonzero_duration) else 0.0,
        "rich_duration_nonzero_p90_dt": float(np.percentile(nonzero_duration, 90)) if len(nonzero_duration) else 0.0,
        "rich_forward_max_dt": float(forward.max(initial=0)),
        "rich_forward_p90_dt": float(np.percentile(forward, 90)) if len(forward) else 0.0,
        "rich_reverse_p90_dt": float(np.percentile(reverse, 90)) if len(reverse) else 0.0,
        "rich_criticality_p90_dt": float(np.percentile(criticality, 90)) if len(criticality) else 0.0,
        "rich_layer_nodes_mean": float(layer_counts.mean()),
        "rich_layer_nodes_p90": float(np.percentile(layer_counts, 90)),
        "rich_layer_nodes_max": float(layer_counts.max(initial=0)),
        "rich_quantum_edge_fraction": float(np.mean(edge_kind == 0)) if len(edge_kind) else 0.0,
    }
    for opcode_id in range(14):
        values[f"rich_opcode_{opcode_id}_count"] = int(np.sum(opcode == opcode_id))
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "validation" / "mali_physical_dag_v1")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "work" / "mali_physical_dag_v1" / "graphs")
    args = parser.parse_args()
    manifest = read_csv(args.artifact_dir / "manifest.csv")
    graph_ids = sorted({row["graph_id"] for row in manifest})
    summaries = {}
    for position, graph_id in enumerate(graph_ids, start=1):
        summaries[graph_id] = describe(args.graph_dir / f"{graph_id}.npz")
        if position % 50 == 0 or position == len(graph_ids):
            print(f"graph-summary {position}/{len(graph_ids)}", flush=True)
    output = []
    for row in manifest:
        enriched = {"graph_id": row["graph_id"]}
        enriched.update(summaries[row["graph_id"]])
        output.append(enriched)
    out_path = args.artifact_dir / "graph_summary_features.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        # one row per graph, not one row per observed label
        writer.writerows({"graph_id": graph_id, **summaries[graph_id]} for graph_id in graph_ids)
    (args.artifact_dir / "graph_summary_features.json").write_text(
        '{\n  "n_unique_graphs": %d,\n  "feature_semantics": "train-independent compact descriptors from current FakeBackend NPZ graphs"\n}\n'
        % len(graph_ids),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
