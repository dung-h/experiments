#!/usr/bin/env python3
"""Evaluate a bounded contraction-plan ranking proxy.

For every fixed circuit-derived tensor network, cuTensorNet is asked to choose
a plan with several path-search sample budgets.  ``RUNTIME_EST`` ranks those
selected plans before contraction; warm CUDA-event time supplies the outcome.

This deliberately does *not* claim to reproduce the candidate generator or
the learned ranker in arXiv:2608.05819.  It is an independent, local check of
the narrower question: does the native pre-run score select the fastest plan
from this explicitly recorded candidate set on this GPU?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

KEYS = ["family", "circuit_variant", "num_qubits", "family_depth_multiplier", "seed"]
REQUIRED = {
    *KEYS,
    "optimizer_samples",
    "optimizer_seed",
    "cutensornet_runtime_est_s",
    "contract_gpu_median_s",
    "path_optimization_s",
    "cutensornet_flop_count",
    "cutensornet_largest_intermediate_elements",
    "cutensornet_num_slices",
    "runtime_semantics",
}


def average_ranks(values: list[float]) -> list[float]:
    """Return average ranks, assigning tied values the same mean rank."""
    ranked = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    position = 0
    while position < len(ranked):
        end = position + 1
        while end < len(ranked) and ranked[end][1] == ranked[position][1]:
            end += 1
        mean_rank = (position + 1 + end) / 2.0
        for index, _ in ranked[position:end]:
            result[index] = mean_rank
        position = end
    return result


def spearman(values_a: list[float], values_b: list[float]) -> float | None:
    ranks_a, ranks_b = average_ranks(values_a), average_ranks(values_b)
    mean_a, mean_b = statistics.fmean(ranks_a), statistics.fmean(ranks_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(ranks_a, ranks_b))
    denominator_a = math.sqrt(sum((a - mean_a) ** 2 for a in ranks_a))
    denominator_b = math.sqrt(sum((b - mean_b) ** 2 for b in ranks_b))
    if denominator_a == 0 or denominator_b == 0:
        return None
    return numerator / (denominator_a * denominator_b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Input has no rows")
    missing = sorted(REQUIRED - set(rows[0]))
    if missing:
        raise SystemExit(f"Input misses columns: {missing}")
    semantics = {row["runtime_semantics"] for row in rows if row["runtime_semantics"]}
    expected_semantics = {"T_sim_contract: local GPU tensor-network scalar contraction"}
    if semantics != expected_semantics:
        raise SystemExit(f"Unexpected/mixed target semantics: {sorted(semantics)}")
    for column in [
        "optimizer_samples", "optimizer_seed", "cutensornet_runtime_est_s", "contract_gpu_median_s",
        "path_optimization_s", "cutensornet_flop_count",
        "cutensornet_largest_intermediate_elements", "cutensornet_num_slices",
    ]:
        for row in rows:
            try:
                row[column] = float(row[column])
            except ValueError as exc:
                raise SystemExit(f"Non-numeric {column}: {row[column]!r}") from exc
    for row in rows:
        if row["cutensornet_runtime_est_s"] <= 0 or row["contract_gpu_median_s"] <= 0:
            raise SystemExit("Non-positive estimate or observed contraction time")

    selections: list[dict[str, object]] = []
    correlations: list[float] = []
    distinct_plan_counts: list[int] = []
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[name]) for name in KEYS)].append(row)
    for key, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        if len({(row["optimizer_samples"], row["optimizer_seed"]) for row in group}) != len(group):
            raise SystemExit(f"Duplicate candidate (budget, seed) for circuit {key}")
        group.sort(key=lambda row: (float(row["optimizer_samples"]), float(row["optimizer_seed"])))
        estimate_choice = min(group, key=lambda row: float(row["cutensornet_runtime_est_s"]))
        oracle = min(group, key=lambda row: float(row["contract_gpu_median_s"]))
        rho = spearman(
            [float(row["cutensornet_runtime_est_s"]) for row in group],
            [float(row["contract_gpu_median_s"]) for row in group],
        )
        if rho is not None:
            correlations.append(rho)
        signatures = {
            "|".join(str(row[name]) for name in [
                "cutensornet_flop_count", "cutensornet_largest_intermediate_elements",
                "cutensornet_num_slices",
            ])
            for row in group
        }
        distinct_plan_counts.append(len(signatures))
        selection = dict(zip(KEYS, key))
        selection.update({
            "candidate_count": int(len(group)),
            "distinct_structural_plan_signatures": len(signatures),
            "estimate_selected_optimizer_samples": int(float(estimate_choice["optimizer_samples"])),
            "estimate_selected_optimizer_seed": int(float(estimate_choice["optimizer_seed"])),
            "oracle_optimizer_samples": int(float(oracle["optimizer_samples"])),
            "oracle_optimizer_seed": int(float(oracle["optimizer_seed"])),
            "estimate_selected_runtime_est_s": float(estimate_choice["cutensornet_runtime_est_s"]),
            "estimate_selected_actual_s": float(estimate_choice["contract_gpu_median_s"]),
            "oracle_actual_s": float(oracle["contract_gpu_median_s"]),
            "selection_regret_ratio": float(
                float(estimate_choice["contract_gpu_median_s"]) / float(oracle["contract_gpu_median_s"])
            ),
            "exact_fastest_selection": bool(
                estimate_choice["optimizer_samples"] == oracle["optimizer_samples"]
            ),
            "spearman_estimate_actual_within_circuit": rho,
        })
        selections.append(selection)

    if not selections:
        raise SystemExit("Need at least one circuit with two or more candidates")
    output = args.output_prefix
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(selections[0])
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selections)
    exact = sum(bool(row["exact_fastest_selection"]) for row in selections) / len(selections)
    regret = [float(row["selection_regret_ratio"]) for row in selections]
    summary = {
        "target_semantics": "T_sim_contract: local GPU tensor-network scalar contraction",
        "observation_unit": "one fixed tensor network with several selected cuTensorNet candidate plans",
        "n_candidate_rows": len(rows),
        "n_circuits": len(selections),
        "native_runtime_est_exact_fastest_rate": exact,
        "selection_regret_ratio_median": statistics.median(regret),
        "selection_regret_ratio_max": max(regret),
        "within_circuit_spearman_median": None if not correlations else statistics.median(correlations),
        "distinct_structural_plan_signatures_median": statistics.median(distinct_plan_counts),
        "limitation": (
            "Candidate plans are generated only by changing cuTensorNet sample budget; this is not the "
            "paper's candidate generator, learned ranker, dataset, or cross-GPU protocol."
        ),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    markdown = f"""# cuTensorNet contraction-plan ranking proxy

## Scope

- Paper relation: independent proxy for the *plan-selection* question in
  `Learning to Rank Tensor Network Contraction Plans for GPU-Accelerated
  Quantum Circuit Simulation` (arXiv:2608.05819).
- Target: warm CUDA-event `contract_gpu_median_s` for a scalar tensor-network
  contraction on the local RTX 5070 Ti. Path search and first contraction are
  not included in this target.
- Candidates: one cuTensorNet-selected plan for each recorded `optimizer_samples`
  budget. This is a bounded candidate source, not a reconstruction of the
  authors' search space or their learned ranker.

## Result

| Quantity | Value |
|---|---:|
| Candidate rows | {summary['n_candidate_rows']} |
| Fixed-circuit candidate sets | {summary['n_circuits']} |
| Native-estimate exact fastest selection | {100 * exact:.1f}% |
| Median actual/actual-oracle selection regret | {summary['selection_regret_ratio_median']:.3f}× |
| Worst selection regret | {summary['selection_regret_ratio_max']:.3f}× |
| Median within-circuit Spearman(estimate, actual) | {summary['within_circuit_spearman_median']!s} |
| Median distinct structural plan signatures | {summary['distinct_structural_plan_signatures_median']:.1f} |

`selection_regret_ratio` is one when the pre-run native estimate chooses a
candidate tied for fastest in the measured set. The raw candidate rows are in
the input CSV; per-circuit choices are in the sibling CSV to this report.

## Interpretation boundary

The test establishes only how the native score behaves over these local
candidate plans. It cannot establish the paper's learned-ranker results,
cross-GPU transfer, or a universal simulator runtime estimator.
"""
    output.with_suffix(".md").write_text(markdown)
    print(output.with_suffix(".json"))
    print(output.with_suffix(".csv"))
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
