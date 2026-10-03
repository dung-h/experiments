from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import aggregate_qpu_global_metadata_mlp_v1 as aggregate  # noqa: E402


def test_metrics_keep_tail_and_scale_fields():
    values = aggregate.metric([1.0, 2.0, 4.0], [1.0, 2.5, 8.0])
    assert values["n"] == 3
    assert values["mae_seconds"] == 1.5
    assert values["max_abs_error_seconds"] == 4.0
    assert values["p99_abs_error_seconds"] >= values["p90_abs_error_seconds"]


def test_cluster_key_preserves_source_identity():
    row = {"canonical_row_id": "qpack_mcp|r1", "source_id": "qpack_mcp",
           "workflow_group_hash": "workflow-hash"}
    assert aggregate.cluster_id(row) == "qpack_mcp|optimizer_workflow|workflow-hash"


def test_bootstrap_reports_observed_point_separately_from_replicate_mean():
    rows = [
        {"cluster_id": "a", "candidate_abs_error": 3.0, "graph_abs_error": 1.0},
        {"cluster_id": "b", "candidate_abs_error": 1.0, "graph_abs_error": 2.0},
    ]
    result = aggregate.bootstrap_delta(rows, seed=17)
    assert result["observed_delta_seconds"] == 0.5
    assert result["bootstrap_clusters"] == 2
    assert result["bootstrap_ci_low_seconds"] <= result["bootstrap_ci_high_seconds"]
    assert result["bootstrap_mean_delta_seconds"] != ""
