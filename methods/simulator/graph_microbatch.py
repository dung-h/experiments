#!/usr/bin/env python3
"""Deterministic node-budget microbatch planner for Graph V3-large.

Pure Python: no torch import.  Loss scaling is
`sum(per-graph MSE) / effective_batch_size` so accumulated microbatch
gradients match the mean loss of the effective batch.
"""
from __future__ import annotations

import random
from typing import Iterable


EFFECTIVE_BATCH_SIZE = 32
MAX_NODES_PER_MICROBATCH = 250_000
LARGE_GRAPH_OPERATIONS = 1_431_974
LARGE_QASM_DIGEST = "39074c915e29ec988d891918b8859ddbd72dab277f9520c539d8205b763a3a78"
RESCUED_IDS = (
    "qonductor_single_circuit_ibm|row39",
    "qonductor_single_circuit_ibm|row40",
    "qonductor_single_circuit_ibm|row41",
)
UNAVAILABLE_ID = "qonductor_single_circuit_ibm|row4477"
UNAVAILABLE_REASON = "dynamic_control_not_representable_by_declared_flat_dag_v3"
EXPECTED_ELIGIBLE = 8766
EXPECTED_ASSIGNED = 8767
EXPECTED_TEST = {0: 1777, 1: 2206, 2: 1547, 3: 1467, 4: 1769}
EXPECTED_TRAIN = {0: 6989, 1: 6560, 2: 7219, 3: 7299, 4: 6997}
SEEDS = (42, 1234, 31415)


def epoch_permutation(n: int, seed: int, epoch: int) -> list[int]:
    """Observation order for epoch `epoch` under seed `seed`.

    The mix `seed * 1_000_003 + epoch` is the frozen permutation function.
    Resume may restart at an epoch boundary; it must not reshuffle a
    partially consumed epoch.
    """
    rng = random.Random(int(seed) * 1_000_003 + int(epoch))
    order = list(range(int(n)))
    rng.shuffle(order)
    return order


def is_solo_graph(node_count: int, max_nodes: int = MAX_NODES_PER_MICROBATCH) -> bool:
    return int(node_count) > int(max_nodes)


def pack_microbatches(
    group_ids: list[str],
    node_counts: dict[str, int],
    max_nodes: int = MAX_NODES_PER_MICROBATCH,
) -> list[list[str]]:
    """Pack one effective batch into node-budget microbatches.

    Graphs above `max_nodes` always occupy their own microbatch (batch size 1).
    Smaller graphs fill a microbatch until the next graph would exceed the
    node budget.  Group membership is preserved: packing never pulls graphs
    from a later effective batch.
    """
    microbatches: list[list[str]] = []
    current: list[str] = []
    current_nodes = 0
    for cid in group_ids:
        n = int(node_counts[cid])
        if is_solo_graph(n, max_nodes):
            if current:
                microbatches.append(current)
                current, current_nodes = [], 0
            microbatches.append([cid])
            continue
        if current and current_nodes + n > max_nodes:
            microbatches.append(current)
            current, current_nodes = [], 0
        current.append(cid)
        current_nodes += n
    if current:
        microbatches.append(current)
    return microbatches


def plan_epoch(
    train_ids: list[str],
    node_counts: dict[str, int],
    seed: int,
    epoch: int,
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE,
    max_nodes: int = MAX_NODES_PER_MICROBATCH,
) -> list[dict[str, object]]:
    order = epoch_permutation(len(train_ids), seed, epoch)
    ids = [train_ids[index] for index in order]
    planned: list[dict[str, object]] = []
    for start in range(0, len(ids), effective_batch_size):
        group = ids[start : start + effective_batch_size]
        microbatches = pack_microbatches(group, node_counts, max_nodes=max_nodes)
        planned.append(
            {
                "effective_batch_index": start // effective_batch_size,
                "ids": group,
                "size": len(group),
                "microbatches": microbatches,
                "microbatch_sizes": [len(item) for item in microbatches],
                "microbatch_nodes": [sum(int(node_counts[cid]) for cid in item) for item in microbatches],
                "contains_solo_graph": any(is_solo_graph(int(node_counts[cid]), max_nodes) for cid in group),
            }
        )
    return planned


def scaled_microbatch_loss_terms(
    per_graph_mse: Iterable[float],
    effective_batch_size: int,
) -> float:
    """Sum of per-graph MSE divided by the effective batch size (or remainder)."""
    values = list(per_graph_mse)
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")
    return float(sum(values)) / float(effective_batch_size)


def reference_mean_loss(per_graph_mse: Iterable[float]) -> float:
    values = list(per_graph_mse)
    if not values:
        raise ValueError("empty effective batch")
    return float(sum(values)) / float(len(values))
