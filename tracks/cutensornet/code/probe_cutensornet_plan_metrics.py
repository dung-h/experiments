#!/usr/bin/env python3
"""Extract public cuTensorNet plan metrics without executing a contraction.

This diagnostic isolates the output of ``Network.contract_path``.  It does
not call ``Network.contract`` and is therefore safe to use to inspect a
previously measured large-width configuration without retaking its GPU timing.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import time

from run_cutensornet_runtime_benchmark import build_family, get_optimizer_value


ROOT = Path(__file__).resolve().parents[3]


def optional_optimizer_value(cutn: object, network: object, attribute: object, np: object) -> float | None:
    """Return an optimizer attribute when the selected plan supports it."""
    try:
        return get_optimizer_value(cutn, network, attribute, np)
    except Exception as exc:  # Some attributes, such as slicing overhead, are unavailable without slices.
        if "NOT_SUPPORTED" not in str(exc):
            raise
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", default="28,32,36,40,42,44")
    parser.add_argument("--family", default="qft")
    parser.add_argument("--memory-limit", default="90%")
    parser.add_argument("--optimizer-samples", type=int, default=32)
    parser.add_argument("--optimizer-seed", type=int, default=137)
    parser.add_argument("--optimizer-cost-function", default="TIME_TUNED")
    parser.add_argument("--circuit-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "run-output" / "cutensornet-plan-probe",
    )
    args = parser.parse_args()

    try:
        import cupy as cp
        import numpy as np
        from cuquantum import tensornet as tn
        from cuquantum.bindings import cutensornet as cutn
    except ImportError as exc:
        raise SystemExit("cuQuantum/cuPy is unavailable; use the cuTensorNet environment wrapper.") from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise SystemExit("No CUDA GPU visible to CuPy.")
    try:
        cost_function = getattr(cutn.OptimizerCost, args.optimizer_cost_function.upper())
    except AttributeError as exc:
        raise SystemExit(f"Unknown objective {args.optimizer_cost_function!r}") from exc

    compute_capability = str(cp.cuda.Device().compute_capability)
    gpu_arch = int(compute_capability[:-1]) if len(compute_capability) > 1 else int(compute_capability)
    widths = [int(value) for value in args.qubits.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for width in widths:
        print(f"Optimizing {args.family}, {width} qubits (no contraction)", flush=True)
        started = time.perf_counter()
        operands, features = build_family(cp, args.family, width, args.circuit_seed, 1.0)
        network = tn.Network(*operands, options=tn.NetworkOptions(memory_limit=args.memory_limit, blocking="auto"))
        try:
            optimizer = tn.OptimizerOptions(
                samples=args.optimizer_samples,
                seed=args.optimizer_seed,
                threads=max(1, (os.cpu_count() or 2) // 2),
                cost_function=cost_function,
                gpu_arch=gpu_arch,
            )
            path_started = time.perf_counter()
            _, optimizer_info = network.contract_path(optimizer)
            path_finished = time.perf_counter()
            info = cutn.ContractionOptimizerInfoAttribute
            rows.append({
                "source": "cutensornet",
                "observation_unit": "one circuit-derived tensor-network plan; no contraction executed",
                "family": features.family,
                "circuit_variant": features.circuit_variant,
                "num_qubits": features.num_qubits,
                "logical_depth": features.logical_depth,
                "logical_total_gate_count": features.logical_total_gate_count,
                "logical_two_qubit_gate_count": features.logical_two_qubit_gate_count,
                "precision": "complex64",
                "workspace_memory_limit": args.memory_limit,
                "optimizer_cost_function": args.optimizer_cost_function.upper(),
                "optimizer_samples": args.optimizer_samples,
                "optimizer_seed": args.optimizer_seed,
                "optimizer_gpu_arch": gpu_arch,
                "phase1_flop_count": get_optimizer_value(cutn, network, info.PHASE1_FLOP_COUNT, np),
                "flop_count_after_slicing": get_optimizer_value(cutn, network, info.FLOP_COUNT, np),
                "slicing_overhead": optional_optimizer_value(cutn, network, info.SLICING_OVERHEAD, np),
                "runtime_est_s": get_optimizer_value(cutn, network, info.RUNTIME_EST, np),
                "effective_flops_est": get_optimizer_value(cutn, network, info.EFFECTIVE_FLOPS_EST, np),
                "largest_intermediate_elements": float(optimizer_info.largest_intermediate),
                "num_slices": int(get_optimizer_value(cutn, network, info.NUM_SLICES, np)),
                "num_sliced_modes": int(get_optimizer_value(cutn, network, info.NUM_SLICED_MODES, np)),
                "network_num_slices": int(network.num_slices),
                "path_optimization_s": path_finished - path_started,
                "plan_only_wall_s": path_finished - started,
            })
        finally:
            network.free()
            cp.get_default_memory_pool().free_all_blocks()

    output_path = args.output_dir / "cutensornet_plan_metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} plan-only rows to {output_path}")


if __name__ == "__main__":
    main()
