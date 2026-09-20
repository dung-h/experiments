#!/usr/bin/env python3
"""Regenerate the concise v2 finding report from final raw/evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    headers = [str(column) for column in frame.columns]
    values = [[f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(header), *(len(row[index]) for row in values)) for index, header in enumerate(headers)]
    render = lambda row: "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
    return "\n".join([render(headers), render(["-" * width for width in widths]), *(render(row) for row in values)])


def metric(frame: pd.DataFrame, split: str, model: str) -> pd.Series:
    result = frame[(frame["split"] == split) & (frame["model"] == model)]
    if len(result) != 1:
        raise ValueError(f"expected one aggregate row for {split}/{model}; found {len(result)}")
    return result.iloc[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    evaluation_dir = run_dir / "evaluation_final"
    raw = pd.read_csv(run_dir / "records.csv")
    corpus = pd.read_csv(run_dir / "canonical_corpus.csv")
    aggregate = pd.read_csv(evaluation_dir / "aggregate_metrics.csv")
    frontier = pd.read_csv(evaluation_dir / "resource_frontier_prediction.csv")
    selection = pd.read_csv(evaluation_dir / "context_selection_oracle_distribution.csv")
    matching = aggregate[aggregate["split"] == "circuit_group"].copy()
    matching = matching[matching["model"].isin([
        "analytical_calibrated_ridge", "analytical_hgb", "logical_hgb", "graph_hgb", "graph_hgb_with_family",
    ])].sort_values("model")
    width = aggregate[(aggregate["split"] == "width_held_out") & aggregate["model"].isin([
        "analytical_calibrated_ridge", "structural_ridge", "logical_hgb", "graph_hgb",
    ])].sort_values("model")
    seed = aggregate[(aggregate["split"] == "seed_held_out_structural") & aggregate["model"].isin([
        "analytical_calibrated_ridge", "logical_hgb", "graph_hgb",
    ])].sort_values("model")
    q28 = raw[(raw["num_qubits"] == 28) & (raw["execution_device"] == "cuda")].copy()
    q28 = q28[["family", "depth_parameter", "context_id", "status", "warm_execution_median_seconds", "statevector_bytes", "torch_cuda_peak_reserved_bytes"]].sort_values(["family", "depth_parameter", "context_id"])
    logical = metric(aggregate, "circuit_group", "logical_hgb")
    analytical = metric(aggregate, "circuit_group", "analytical_calibrated_ridge")
    graph = metric(aggregate, "circuit_group", "graph_hgb")
    logical_width = metric(aggregate, "width_held_out", "logical_hgb")
    analytical_width = metric(aggregate, "width_held_out", "analytical_calibrated_ridge")
    resource_counts = raw[raw["num_qubits"] == 28].groupby(["context_id", "status"]).size().to_dict()
    report = [
        "# Dense-statevector v2: estimator and feasibility findings",
        "",
        "**Finding recorded:** 20 September 2026.",
        "",
        "## Claim boundary",
        "",
        "This report concerns one local PyTorch dense-statevector reference kernel on the recorded RTX 5070 Ti and CPU. The target is `prepared_execute_reset_state`: allocate a new state, apply a pre-materialized direct gate schedule, then calculate `<Z_0>`. It excludes construction, materialization, transpilation, sampling, cloud service and QPU time. It must not be pooled with Aer, CUDA-Q, cuTensorNet or observed-QPU targets.",
        "",
        "The final corpus has **%d raw rows**, **%d successful duration rows**, %d logical circuits and five widths (16, 20, 24, 26, 28). It retains **%d resource-limit rows** instead of replacing them with duration labels." % (
            len(corpus), int((corpus["status"] == "ok").sum()), corpus["circuit_id"].nunique(), int((corpus["status"] == "resource_limit").sum())
        ),
        "",
        "## RQ1 — Runtime prediction for an unseen logical circuit",
        "",
        "The primary `circuit_group` split holds every context of the logical test circuit outside training. Thus CPU/GPU/precision measurements of that circuit cannot leak into training. Models predict log warm execution time from pre-execution features and known context.",
        "",
        table(matching[["model", "n_test", "mae_log", "rmse_log", "r2_log", "mae_seconds", "r2_seconds"]]),
        "",
        "The calibrated analytical Ridge uses only final-state bytes, `gate_count × 2^width`, and context. `logical_hgb` adds logical depth and one/two-qubit gate counts. It improves circuit-held-out log-R² from %.4f to %.4f and lowers log-MAE from %.4f to %.4f. This is evidence that logical structure adds value within this fixed simulator contract." % (
            analytical.r2_log, logical.r2_log, analytical.mae_log, logical.mae_log
        ),
        "",
        "The graph ablation is a negative result: `graph_hgb` reaches %.4f log-R², below `logical_hgb` %.4f. Adding the coarse family label also does not improve it. The present data therefore do **not** support a claim that interaction-graph features add incremental value beyond logical depth/gate composition." % (
            graph.r2_log, logical.r2_log
        ),
        "",
        "## RQ2 — Interpolation is not width extrapolation",
        "",
        table(width[["model", "n_test", "mae_log", "r2_log", "mae_seconds", "r2_seconds"]]),
        "",
        "When an entire width is absent from training, analytical Ridge is more robust: log-R² %.4f versus %.4f for logical HGB. A nonlinear model that performs best inside the observed width range is not automatically the safe model for a wider circuit. This is why random-row metrics are not used as the main conclusion." % (
            analytical_width.r2_log, logical_width.r2_log
        ),
        "",
        "## RQ3 — Independent random-topology seeds",
        "",
        "The GPU-only seed holdout leaves out topology seeds 17, 43 or 101 in turn at q16–q24. The two random families retain the same width/depth profile across seeds; this is a controlled test of topology variation, not a broad circuit-distribution benchmark.",
        "",
        table(seed[["model", "n_test", "mae_log", "r2_log", "mae_seconds", "r2_seconds"]]),
        "",
        "Logical HGB reaches %.4f log-R² on this matched seed split. The result confirms stability for these generated topologies, but it does not overturn the graph-ablation result: its relevant message is that the pre-execution logical workload/context contract is stable across these seeds." % metric(aggregate, "seed_held_out_structural", "logical_hgb").r2_log,
        "",
        "## RQ4 — GPU feasibility requires a peak-memory envelope",
        "",
        "A naive feasibility check uses only `statevector_bytes <= VRAM`. The direct gate kernel allocates temporary tensors, so that is a lower bound rather than a sufficient condition. The frontier envelope calibrates the maximum peak-reserved/statevector ratio through q24, then predicts q26/q28 without using their outcomes.",
        "",
        table(frontier),
        "",
        "At q28, all complex128 rows are `resource_limit` while all complex64 rows complete: %s. The raw complex128 statevector is 4 GiB and therefore passes the naive 15.47 GiB VRAM check, but the q≤24 envelope predicts 17.55 GB peak and correctly rejects it. q28 complex64 is predicted feasible and all four rows complete." % resource_counts,
        "",
        "### q28 observed rows",
        "",
        table(q28),
        "",
        "This is the strongest engineering result in the study: before attempting execution, a context-calibrated peak envelope separates the feasible q28 complex64 context from the infeasible q28 complex128 context on this kernel/GPU.",
        "",
        "## Context-selection result",
        "",
        table(selection),
        "",
        "The full four-context groups all have `cuda:complex64` as observed fastest. Consequently the 100% context-selection score in the evaluator is trivial and is **not** presented as an estimator contribution. It is a machine/kernel property for this benchmark range.",
        "",
        "## What can and cannot be claimed",
        "",
        "- Supported: a source-aware local estimator can predict warm runtime accurately inside this fixed dense-statevector contract; logical gate/depth features improve the analytical workload baseline on unseen circuits in-range; an empirically calibrated peak-memory envelope predicts the q28 precision feasibility boundary.",
        "- Not supported: universal simulator runtime prediction, cross-GPU transfer, Aer/CUDA-Q/cuTensorNet equivalence, QPU-runtime prediction, sampling/noise/transpilation timing, or incremental graph-feature benefit in this corpus.",
        "- Next test: add a second physical host/GPU and an independently implemented simulator as separately labelled domains. A domain-transfer result must be evaluated without pooling their runtimes.",
        "",
        "## Reproduce",
        "",
        "Run `experiments/simulator_runtime_v2/run_dense_statevector_v2.py` according to `experiments/simulator_runtime_v2/README.md`, then `build_corpus.py`, `evaluate_runtime_v2.py`, and this script. Raw measurements are `records.jsonl`/`records.csv`; all OOF predictions and metrics are under `evaluation_final/`.",
    ]
    target = run_dir / "REPORT.md"
    target.write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = {
        "raw_rows": int(len(corpus)),
        "successful_duration_rows": int((corpus["status"] == "ok").sum()),
        "resource_limit_rows": int((corpus["status"] == "resource_limit").sum()),
        "circuit_group": {
            "analytical_calibrated_ridge": analytical.to_dict(),
            "logical_hgb": logical.to_dict(),
            "graph_hgb": graph.to_dict(),
        },
        "width_held_out": {
            "analytical_calibrated_ridge": analytical_width.to_dict(),
            "logical_hgb": logical_width.to_dict(),
        },
        "resource_frontier": frontier.to_dict(orient="records"),
    }
    (run_dir / "results_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
