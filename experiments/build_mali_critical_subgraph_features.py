#!/usr/bin/env python3
"""Extract compact bottleneck/critical-subgraph descriptors from Ma--Li DAGs.

The descriptors are computed before seeing the observed runtime.  They summarize
the high-criticality tail of the reconstructed physical DAG rather than feeding
the full native graph to a larger GNN.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def describe(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        opcode = archive["node_opcode"].astype(np.int64)
        duration = archive["node_duration_dt"].astype(np.float64)
        criticality = archive["node_criticality_dt"].astype(np.float64)
        edge_src = archive["edge_src"].astype(np.int64)
        edge_dst = archive["edge_dst"].astype(np.int64)

    n = len(criticality)
    if not n:
        return {"critical_n_nodes": 0}
    values: dict[str, object] = {}
    for quantile, label in ((90, "p90"), (95, "p95"), (99, "p99")):
        threshold = float(np.percentile(criticality, quantile))
        mask = criticality >= threshold
        sub_duration = duration[mask]
        values[f"critical_{label}_dt"] = threshold
        values[f"critical_{label}_node_fraction"] = float(np.mean(mask))
        values[f"critical_{label}_duration_sum_dt"] = float(sub_duration.sum())
        values[f"critical_{label}_duration_fraction"] = float(sub_duration.sum() / max(duration.sum(), 1.0))
        values[f"critical_{label}_mean_dt"] = float(sub_duration.mean()) if len(sub_duration) else 0.0
        values[f"critical_{label}_opcode_entropy"] = float(_entropy(opcode[mask])) if len(sub_duration) else 0.0
        node_mask = mask
        if len(edge_src):
            edge_mask = node_mask[edge_src] & node_mask[edge_dst]
            values[f"critical_{label}_edge_fraction"] = float(np.mean(edge_mask))
            values[f"critical_{label}_edge_count"] = int(np.sum(edge_mask))
        else:
            values[f"critical_{label}_edge_fraction"] = 0.0
            values[f"critical_{label}_edge_count"] = 0

    order = np.argsort(criticality)[::-1]
    for fraction, label in ((0.001, "top001"), (0.01, "top01"), (0.05, "top05")):
        count = max(1, int(np.ceil(n * fraction)))
        top = order[:count]
        top_duration = duration[top]
        top_criticality = criticality[top]
        values[f"critical_{label}_n_nodes"] = count
        values[f"critical_{label}_duration_fraction"] = float(top_duration.sum() / max(duration.sum(), 1.0))
        values[f"critical_{label}_mean_dt"] = float(top_duration.mean())
        values[f"critical_{label}_max_criticality_dt"] = float(top_criticality.max(initial=0.0))

    normalized = criticality / max(float(criticality.max()), 1.0)
    values["criticality_duration_weighted_sum_dt"] = float(np.sum(duration * normalized))
    values["criticality_mean_dt"] = float(criticality.mean())
    values["criticality_max_dt"] = float(criticality.max(initial=0.0))
    return values


def _entropy(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path,
                        default=ROOT / "artifacts/validation/mali_physical_dag_v1")
    parser.add_argument("--graph-dir", type=Path,
                        default=ROOT / "work/mali_physical_dag_v1/graphs")
    args = parser.parse_args()
    manifest = read_csv(args.artifact_dir / "manifest.csv")
    graph_ids = sorted({row["graph_id"] for row in manifest})
    rows = []
    for position, graph_id in enumerate(graph_ids, start=1):
        row = {"graph_id": graph_id}
        row.update(describe(args.graph_dir / f"{graph_id}.npz"))
        rows.append(row)
        if position % 50 == 0 or position == len(graph_ids):
            print(f"critical-summary {position}/{len(graph_ids)}", flush=True)
    output = args.artifact_dir / "critical_subgraph_features.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.artifact_dir / "critical_subgraph_features.json").write_text(
        '{\n  "n_unique_graphs": %d,\n  "feature_semantics": "pre-run high-criticality summaries from current FakeBackend physical DAGs"\n}\n'
        % len(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
