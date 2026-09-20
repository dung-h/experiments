#!/usr/bin/env python3
"""Summarize complex64 versus complex128 from the local precision proxy."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


KEYS = ("family", "circuit_variant", "family_depth_multiplier", "num_qubits", "circuit_seed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="One or more homogeneous precision-run CSV files. Complete pairs are matched only within their full circuit key.",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for input_path in args.inputs:
        with input_path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise SystemExit("No rows")
    expected = {"T_sim_contract: local GPU tensor-network scalar contraction"}
    if {row["runtime_semantics"] for row in rows} != expected:
        raise SystemExit("Refusing mixed target semantics")
    grouped: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        # Preserve compatibility with the initial pilot, which did not yet
        # serialize explicit depth metadata. Those rows all used multiplier 1.
        row.setdefault("family_depth_multiplier", "1.0")
        row.setdefault("family_depth_parameter", "0")
        grouped[tuple(row[name] for name in KEYS)][row["precision"]] = row
    pairs = []
    for key, precision_rows in sorted(grouped.items()):
        if set(precision_rows) != {"complex64", "complex128"}:
            continue
        low, reference = precision_rows["complex64"], precision_rows["complex128"]
        low_value = complex(float(low["scalar_real"]), float(low["scalar_imag"]))
        ref_value = complex(float(reference["scalar_real"]), float(reference["scalar_imag"]))
        result = dict(zip(KEYS, key))
        result.update({
            "complex64_median_s": float(low["contract_gpu_median_s"]),
            "complex128_median_s": float(reference["contract_gpu_median_s"]),
            "complex128_over_complex64_time_ratio": float(reference["contract_gpu_median_s"]) / float(low["contract_gpu_median_s"]),
            "absolute_scalar_difference": abs(low_value - ref_value),
            "relative_scalar_difference": abs(low_value - ref_value) / max(abs(ref_value), 1e-15),
        })
        pairs.append(result)
    if not pairs:
        raise SystemExit("No complete precision pairs")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    ratios = [float(row["complex128_over_complex64_time_ratio"]) for row in pairs]
    abs_errors = [float(row["absolute_scalar_difference"]) for row in pairs]
    rel_errors = [float(row["relative_scalar_difference"]) for row in pairs]
    summary = {
        "target_semantics": "T_sim_contract: local GPU tensor-network scalar contraction",
        "n_pairs": len(pairs),
        "complex128_over_complex64_time_ratio_median": statistics.median(ratios),
        "complex128_over_complex64_time_ratio_range": [min(ratios), max(ratios)],
        "absolute_scalar_difference_max": max(abs_errors),
        "relative_scalar_difference_max": max(rel_errors),
        "reference": "complex128 result from the same local cuTensorNet setup",
        "limitation": "No SGEMM emulation, exponent-statistics selector, RCS/Sycamore corpus or fidelity study is reproduced. The complex128 scalar is only an in-run numerical comparison point.",
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    report = "\n".join([
        "# cuTensorNet precision-selection proxy",
        "",
        "Each pair contracts the same local scalar tensor network once in `complex64` and once in `complex128`. Warm CUDA-event time is measured separately from path search. The `complex128` scalar is a local comparison reference.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Matched precision pairs | {summary['n_pairs']} |",
        f"| Median complex128 / complex64 time | {summary['complex128_over_complex64_time_ratio_median']:.3f}× |",
        f"| Range of time ratios | {summary['complex128_over_complex64_time_ratio_range'][0]:.3f}–{summary['complex128_over_complex64_time_ratio_range'][1]:.3f}× |",
        f"| Maximum absolute scalar difference | {summary['absolute_scalar_difference_max']:.3e} |",
        f"| Maximum relative scalar difference | {summary['relative_scalar_difference_max']:.3e} |",
        "",
        "This is a local precision trade-off measurement, not a reimplementation of the SGEMM-emulation paper's automatic selector or reported speedups.",
        "",
    ])
    args.output_prefix.with_suffix(".md").write_text(report)
    print(args.output_prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
