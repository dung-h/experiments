#!/usr/bin/env python3
"""Measure complex64/complex128 cost and scalar agreement for cuTensorNet.

This is a local numerical-precision study related to arXiv:2303.08989. It
does not implement SGEMM emulation or claim that paper's RCS/Sycamore results.
The complex128 scalar is an in-run comparison reference, not exact truth.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tracks" / "cutensornet" / "code" / "run_cutensornet_runtime_benchmark.py"
FAMILIES = ("ghz", "hea", "qaoa_cycle", "random_brickwork", "qft")


def load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("precision_circuit_builder", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_one(builder: Any, cp: Any, np: Any, tn: Any, cutn: Any, family: str,
            num_qubits: int, seed: int, precision: str, warmups: int,
            repeats: int, depth_multiplier: float) -> dict[str, object]:
    operands, features = builder.build_family(
        cp, family, num_qubits, seed, depth_multiplier, precision=precision,
    )
    capability = str(cp.cuda.Device().compute_capability)
    gpu_arch = int(capability[:-1]) if len(capability) > 1 else int(capability)
    network = tn.Network(*operands, options=tn.NetworkOptions(memory_limit="70%", blocking="auto"))
    try:
        optimizer = tn.OptimizerOptions(samples=16, seed=42, cost_function=cutn.OptimizerCost.TIME_TUNED, gpu_arch=gpu_arch)
        path_started = time.perf_counter()
        _, info = network.contract_path(optimizer)
        path_s = time.perf_counter() - path_started
        flops = builder.get_optimizer_value(cutn, network, cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT, np)
        first_start, first_end = cp.cuda.Event(), cp.cuda.Event()
        first_start.record()
        result = network.contract()
        first_end.record()
        first_end.synchronize()
        first_s = cp.cuda.get_elapsed_time(first_start, first_end) / 1_000.0
        scalar = complex(cp.asnumpy(result).item())
        for _ in range(warmups):
            network.contract()
        cp.cuda.get_current_stream().synchronize()
        timings = []
        for _ in range(repeats):
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            network.contract()
            end.record()
            end.synchronize()
            timings.append(cp.cuda.get_elapsed_time(start, end) / 1_000.0)
        gpu = cp.cuda.runtime.getDeviceProperties(0)
        return {
            "status": "ok", "source": "cutensornet_precision_proxy",
            "runtime_semantics": "T_sim_contract: local GPU tensor-network scalar contraction",
            "target_name": "contract_gpu_median_s",
            "observation_unit": "one circuit-derived scalar tensor contraction at one complex precision",
            "family": features.family, "circuit_variant": features.circuit_variant,
            "family_depth_multiplier": features.family_depth_multiplier,
            "family_depth_parameter": features.family_depth_parameter,
            "num_qubits": features.num_qubits, "logical_depth": features.logical_depth,
            "logical_total_gate_count": features.logical_total_gate_count,
            "logical_two_qubit_gate_count": features.logical_two_qubit_gate_count,
            "circuit_seed": seed, "precision": precision,
            "optimizer_samples": 16, "optimizer_seed": 42,
            "path_optimization_s": path_s, "pre_run_flop_count": flops,
            "pre_run_largest_intermediate_elements": float(info.largest_intermediate),
            "pre_run_num_slices": int(network.num_slices),
            "first_contract_gpu_s": first_s, "contract_gpu_median_s": statistics.median(timings),
            "contract_gpu_min_s": min(timings), "contract_gpu_max_s": max(timings),
            "scalar_real": scalar.real, "scalar_imag": scalar.imag,
            "gpu_name": gpu["name"].decode(),
            "gpu_compute_capability": f"{capability[:-1]}.{capability[-1]}",
        }
    finally:
        network.free()
        cp.get_default_memory_pool().free_all_blocks()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--qubits", default="8,12,16,20")
    parser.add_argument(
        "--depth-multipliers", default="1",
        help=("Comma-separated depth multipliers for HEA, QAOA-cycle and random brickwork. "
              "GHZ/QFT always use 1."),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    try:
        import cupy as cp
        import numpy as np
        from cuquantum import tensornet as tn
        from cuquantum.bindings import cutensornet as cutn
    except ImportError as exc:
        raise SystemExit("Requires CUDA-enabled cuQuantum and CuPy") from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise SystemExit("No CUDA GPU visible")
    builder = load_builder()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    multipliers = [float(item) for item in args.depth_multipliers.split(",") if item]
    if not multipliers or any(item <= 0 for item in multipliers):
        raise SystemExit("--depth-multipliers must contain positive values")
    for family in filter(None, args.families.split(",")):
        for qubits in [int(item) for item in args.qubits.split(",") if item]:
            family_multipliers = (1.0,) if family in {"ghz", "qft"} else multipliers
            for depth_multiplier in family_multipliers:
                for precision in ("complex64", "complex128"):
                    print(
                        f"BEGIN family={family} qubits={qubits} depth_multiplier={depth_multiplier:g} precision={precision}",
                        flush=True,
                    )
                    try:
                        row = run_one(builder, cp, np, tn, cutn, family, qubits, args.seed, precision, args.warmups, args.repeats, depth_multiplier)
                        rows.append(row)
                        print(f"END precision={precision} median_ms={float(row['contract_gpu_median_s']) * 1e3:.3f}", flush=True)
                    except Exception as exc:
                        failures.append({"family": family, "num_qubits": qubits, "depth_multiplier": depth_multiplier, "precision": precision, "error": repr(exc)})
                        print(f"FAILED family={family} qubits={qubits} depth_multiplier={depth_multiplier:g} precision={precision}: {exc}", flush=True)
    if rows:
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    args.output.with_suffix(".failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    args.output.with_suffix(".environment.json").write_text(json.dumps({
        "python": sys.version, "cuquantum": __import__("cuquantum").__version__,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "families": args.families, "qubits": args.qubits, "depth_multipliers": multipliers,
        "precisions": ["complex64", "complex128"],
        "target_semantics": "T_sim_contract: local GPU tensor-network scalar contraction",
    }, indent=2) + "\n")
    print(f"WROTE {len(rows)} rows; failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
