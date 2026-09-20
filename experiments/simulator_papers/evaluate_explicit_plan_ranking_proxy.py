#!/usr/bin/env python3
"""Evaluate FLOP-cost candidate selection in the explicit-plan proxy corpus."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


KEYS = ("family", "circuit_variant", "num_qubits", "circuit_seed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("No successful candidate rows")
    if {row["runtime_semantics"] for row in rows} != {"T_sim_contract: local GPU tensor-network scalar contraction"}:
        raise SystemExit("Refusing mixed semantics")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in KEYS)].append(row)
    choices = []
    for key, candidates in sorted(groups.items()):
        if len(candidates) < 2:
            continue
        flop_choice = min(candidates, key=lambda row: float(row["pre_run_flop_count"]))
        oracle = min(candidates, key=lambda row: float(row["contract_gpu_median_s"]))
        native = next((row for row in candidates if row["candidate_plan"] == "native_time_tuned"), None)
        choice = dict(zip(KEYS, key))
        choice.update({
            "candidate_count": len(candidates),
            "flop_choice": flop_choice["candidate_plan"],
            "oracle_choice": oracle["candidate_plan"],
            "flop_choice_actual_s": float(flop_choice["contract_gpu_median_s"]),
            "oracle_actual_s": float(oracle["contract_gpu_median_s"]),
            "flop_selection_regret_ratio": float(flop_choice["contract_gpu_median_s"]) / float(oracle["contract_gpu_median_s"]),
            "flop_exact_fastest": flop_choice["candidate_plan"] == oracle["candidate_plan"],
            "native_actual_s": None if native is None else float(native["contract_gpu_median_s"]),
            "native_regret_ratio": None if native is None else float(native["contract_gpu_median_s"]) / float(oracle["contract_gpu_median_s"]),
            "scalar_spread": max(
                abs(complex(float(row["scalar_real"]), float(row["scalar_imag"]))) for row in candidates
            ) - min(abs(complex(float(row["scalar_real"]), float(row["scalar_imag"]))) for row in candidates),
        })
        choices.append(choice)
    if not choices:
        raise SystemExit("No complete candidate groups")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(choices[0]))
        writer.writeheader()
        writer.writerows(choices)
    regrets = [float(row["flop_selection_regret_ratio"]) for row in choices]
    native_regrets = [float(row["native_regret_ratio"]) for row in choices if row["native_regret_ratio"] is not None]
    summary = {
        "target_semantics": "T_sim_contract: local GPU tensor-network scalar contraction",
        "n_candidate_rows": len(rows),
        "n_complete_circuit_groups": len(choices),
        "flop_exact_fastest_rate": sum(bool(row["flop_exact_fastest"]) for row in choices) / len(choices),
        "flop_selection_regret_ratio_median": statistics.median(regrets),
        "flop_selection_regret_ratio_max": max(regrets),
        "native_time_tuned_regret_ratio_median": statistics.median(native_regrets),
        "max_scalar_magnitude_spread": max(float(row["scalar_spread"]) for row in choices),
        "limitation": (
            "This compares transparent local candidates and FLOP cost only. It is not the paper's "
            "candidate generation, learned ranking model, benchmark corpus or cross-GPU evaluation."
        ),
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    args.output_prefix.with_suffix(".md").write_text(f"""# Explicit contraction-plan ranking proxy

The corpus keeps one fixed scalar tensor network per group and compares named
explicit candidates, including a cuTensorNet `TIME_TUNED` plan, left/right
folds and any recorded deterministic random paths. The measured target is
warm CUDA-event contraction time only.

| Quantity | Value |
|---|---:|
| Candidate rows | {summary['n_candidate_rows']} |
| Complete circuit groups | {summary['n_complete_circuit_groups']} |
| Min-FLOP chooses fastest plan | {100 * summary['flop_exact_fastest_rate']:.1f}% |
| Min-FLOP median regret | {summary['flop_selection_regret_ratio_median']:.3f}× |
| Min-FLOP worst regret | {summary['flop_selection_regret_ratio_max']:.3f}× |
| Native TIME_TUNED median regret | {summary['native_time_tuned_regret_ratio_median']:.3f}× |
| Max scalar-output magnitude spread | {summary['max_scalar_magnitude_spread']:.3e} |

The result tests a deliberately weak analytical baseline. It cannot be used
as a result for arXiv:2608.05819 or as evidence for a learned ranker.
""")
    print(args.output_prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
