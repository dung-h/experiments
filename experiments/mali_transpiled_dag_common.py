#!/usr/bin/env python3
"""Shared coarsening for native transpiled DAGs (Ma-Li flow, Azizov object)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OPCODE_DIM = 14
NODE_DIM = 24  # counts + opcodes + duration/path + mean T1/T2 + n_qubits_used
GLOBAL_DIM = 8
MAX_BINS = 256


def read_qubit_t1_t2(snapshot_path: Path, n_qubits: int = 127) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    t1 = np.zeros(n_qubits, dtype=np.float64)
    t2 = np.zeros(n_qubits, dtype=np.float64)
    qubits = payload["properties"]["qubits"]
    for index, entries in enumerate(qubits):
        values = {item["name"]: float(item["value"]) for item in entries if "name" in item and "value" in item}
        t1[index] = values.get("T1", 0.0)
        t2[index] = values.get("T2", 0.0)
    return t1, t2


def load_snapshot_map(backend_dirs: list[Path]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for directory in backend_dirs:
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
            name = path.stem.split("_")[0]
            mapping[name] = path
            mapping[path.stem] = path
    return mapping


def coarsen_graph(
    graph_path: Path,
    t1: np.ndarray,
    t2: np.ndarray,
    max_bins: int = MAX_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (node_x [B, NODE_DIM], edge_index [2,E], edge_weight [E], global_x [GLOBAL_DIM])."""
    with np.load(graph_path, allow_pickle=False) as archive:
        opcode = archive["node_opcode"].astype(np.int64)
        duration = archive["node_duration_dt"].astype(np.float64)
        layer = archive["node_topo_layer"].astype(np.int64)
        forward = archive["node_forward_path_dt"].astype(np.float64)
        reverse = archive["node_reverse_path_dt"].astype(np.float64)
        criticality = archive["node_criticality_dt"].astype(np.float64)
        q0 = archive["node_q0"].astype(np.int64)
        q1 = archive["node_q1"].astype(np.int64)
        edge_src = archive["edge_src"].astype(np.int64)
        edge_dst = archive["edge_dst"].astype(np.int64)
        metadata = json.loads(str(archive["metadata_json"].item()))

    unique_layers = np.unique(layer)
    if len(unique_layers) == 0:
        nodes = np.zeros((1, NODE_DIM), dtype=np.float32)
        return nodes, np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.float32), np.zeros(GLOBAL_DIM, dtype=np.float32)

    bins = int(min(max_bins, len(unique_layers)))
    layer_to_bin = np.floor(np.arange(len(unique_layers)) * bins / len(unique_layers)).astype(np.int64)
    node_bins = layer_to_bin[np.searchsorted(unique_layers, layer)]
    counts = np.bincount(node_bins, minlength=bins).astype(np.float64)
    nodes = np.zeros((bins, NODE_DIM), dtype=np.float64)
    nodes[:, 0] = np.log1p(counts)
    for opcode_id in range(OPCODE_DIM):
        nodes[:, 1 + opcode_id] = np.log1p(
            np.bincount(node_bins, weights=(opcode == opcode_id).astype(float), minlength=bins)
        )

    def sum_feature(values: np.ndarray) -> np.ndarray:
        return np.log1p(np.bincount(node_bins, weights=np.maximum(values, 0), minlength=bins))

    def max_feature(values: np.ndarray) -> np.ndarray:
        result = np.zeros(bins, dtype=np.float64)
        np.maximum.at(result, node_bins, np.maximum(values, 0))
        return np.log1p(result)

    nodes[:, 15] = sum_feature(duration)
    nodes[:, 16] = max_feature(duration)
    nodes[:, 17] = max_feature(forward)
    nodes[:, 18] = max_feature(reverse)
    nodes[:, 19] = max_feature(criticality)

    q0_valid = q0 >= 0
    node_t1 = np.zeros(len(opcode), dtype=np.float64)
    node_t2 = np.zeros(len(opcode), dtype=np.float64)
    node_t1[q0_valid] = t1[np.clip(q0[q0_valid], 0, len(t1) - 1)]
    node_t2[q0_valid] = t2[np.clip(q0[q0_valid], 0, len(t2) - 1)]
    q1_valid = q1 >= 0
    node_t1[q1_valid] = 0.5 * node_t1[q1_valid] + 0.5 * t1[np.clip(q1[q1_valid], 0, len(t1) - 1)]
    node_t2[q1_valid] = 0.5 * node_t2[q1_valid] + 0.5 * t2[np.clip(q1[q1_valid], 0, len(t2) - 1)]
    t1_sum = np.bincount(node_bins, weights=node_t1, minlength=bins)
    t2_sum = np.bincount(node_bins, weights=node_t2, minlength=bins)
    nodes[:, 20] = t1_sum / np.maximum(counts, 1.0) * 1e6  # microseconds
    nodes[:, 21] = t2_sum / np.maximum(counts, 1.0) * 1e6
    used = ((q0 >= 0) | (q1 >= 0)).astype(float)
    nodes[:, 22] = np.log1p(np.bincount(node_bins, weights=used, minlength=bins))
    nodes[:, 23] = np.log1p(np.bincount(node_bins, minlength=bins))

    if len(edge_src):
        source = node_bins[edge_src]
        destination = node_bins[edge_dst]
        keep = source != destination
        source, destination = source[keep], destination[keep]
        if len(source):
            packed = source.astype(np.int64) * bins + destination.astype(np.int64)
            unique, multiplicity = np.unique(packed, return_counts=True)
            edges = np.vstack((unique // bins, unique % bins)).astype(np.int64)
            weights = np.log1p(multiplicity.astype(np.float64)).astype(np.float32)
        else:
            edges = np.empty((2, 0), dtype=np.int64)
            weights = np.empty(0, dtype=np.float32)
    else:
        edges = np.empty((2, 0), dtype=np.int64)
        weights = np.empty(0, dtype=np.float32)

    global_x = np.asarray(
        [
            np.log1p(float(metadata.get("n_nodes", len(opcode)))),
            np.log1p(float(metadata.get("n_edges", len(edge_src)))),
            np.log1p(float(metadata.get("active_physical_width", 0))),
            np.log1p(float(metadata.get("compiled_depth", 0))),
            np.log1p(float(metadata.get("native_two_qubit_gate_count", 0))),
            np.log1p(float(metadata.get("qcre_critical_path_dt", 0))),
            float(metadata.get("qcre_critical_path_seconds", 0.0)),
            float(metadata.get("compiled_width", 0.0)),
        ],
        dtype=np.float32,
    )
    return nodes.astype(np.float32), edges, weights, global_x
