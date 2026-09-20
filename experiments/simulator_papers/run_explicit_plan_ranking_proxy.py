#!/usr/bin/env python3
"""Create an auditable, small contraction-plan ranking corpus on one GPU.

Each fixed scalar tensor network is contracted with three *named* candidates:
the cuTensorNet TIME_TUNED plan, a left-fold plan, and a right-fold plan.
The latter two are deliberately simple valid einsum paths, not plans claimed
to be competitive.  Their purpose is to create a transparent candidate set
whose FLOP cost and measured contraction time can be compared before training
any learned ranker.

This is an independent proxy for arXiv:2608.05819, not an artifact-level
reproduction of its plan generator, labels, ranker or multi-GPU study.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tracks" / "cutensornet" / "code" / "run_cutensornet_runtime_benchmark.py"
FAMILIES = ("ghz", "hea", "qaoa_cycle", "random_brickwork", "qft")


def load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("cutensornet_benchmark_builder", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import circuit builder from {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def path_for(
    kind: str, operand_count: int, operand_modes: list[object] | None = None,
) -> list[tuple[int, int]] | None:
    if kind == "native_time_tuned":
        return None
    if kind == "left_fold":
        return [(0, 1)] * (operand_count - 1)
    if kind == "right_fold":
        return [(remaining - 2, remaining - 1) for remaining in range(operand_count, 1, -1)]
    if kind.startswith("random_path_"):
        if operand_modes is None or len(operand_modes) != operand_count:
            raise ValueError("A random explicit path requires one mode collection per operand")
        path_seed = int(kind.removeprefix("random_path_"))
        generator = random.Random(path_seed)
        # An arbitrary pair can be a disconnected outer product, which this
        # cuTensorNet path API can reject. Select only a pair with at least
        # one shared mode. ``numpy.einsum_path`` removes the selected pair
        # and appends their output operand, so update the active mode list in
        # exactly that order. (Treating the result as an in-place replacement
        # produces a different, often unsupported, path.)
        active = [set(modes) for modes in operand_modes]
        path: list[tuple[int, int]] = []
        while len(active) > 1:
            connected = [
                (left, right)
                for left in range(len(active))
                for right in range(left + 1, len(active))
                if active[left].intersection(active[right])
            ]
            if not connected:
                raise ValueError("Tensor network became disconnected before a complete connected path was formed")
            left, right = generator.choice(connected)
            shared = active[left].intersection(active[right])
            output_modes = active[left].union(active[right]).difference(shared)
            del active[right]
            del active[left]
            active.append(output_modes)
            path.append((left, right))
        return path
    raise ValueError(kind)


def value(builder: Any, cutn: Any, network: Any, attribute: Any, np: Any) -> float:
    return builder.get_optimizer_value(cutn, network, attribute, np)


def run_candidate(
    builder: Any, cp: Any, np: Any, tn: Any, cutn: Any,
    family: str, num_qubits: int, seed: int, repeats: int, warmups: int,
    kind: str, memory_limit: str,
) -> dict[str, object]:
    operands, features = builder.build_family(cp, family, num_qubits, seed, 1.0)
    operand_count = len(operands) // 2
    network = tn.Network(*operands, options=tn.NetworkOptions(memory_limit=memory_limit, blocking="auto"))
    try:
        compute_capability = str(cp.cuda.Device().compute_capability)
        gpu_arch = int(compute_capability[:-1]) if len(compute_capability) > 1 else int(compute_capability)
        candidate_path = path_for(kind, operand_count, list(operands[1::2]))
        if candidate_path is None:
            optimizer = tn.OptimizerOptions(
                samples=16, seed=42, cost_function=cutn.OptimizerCost.TIME_TUNED,
                gpu_arch=gpu_arch,
            )
        else:
            optimizer = tn.OptimizerOptions(
                path=candidate_path, cost_function=cutn.OptimizerCost.TIME_TUNED,
                gpu_arch=gpu_arch,
            )
        search_started = time.perf_counter()
        _, info = network.contract_path(optimizer)
        path_optimization_s = time.perf_counter() - search_started
        runtime_est = value(builder, cutn, network, cutn.ContractionOptimizerInfoAttribute.RUNTIME_EST, np)
        flop_count = value(builder, cutn, network, cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT, np)
        first_start, first_end = cp.cuda.Event(), cp.cuda.Event()
        first_start.record()
        result = network.contract()
        first_end.record()
        first_end.synchronize()
        first_s = cp.cuda.get_elapsed_time(first_start, first_end) / 1000.0
        scalar = complex(cp.asnumpy(result).item())
        for _ in range(warmups):
            network.contract()
        cp.cuda.get_current_stream().synchronize()
        timings = []
        for _ in range(repeats):
            started, ended = cp.cuda.Event(), cp.cuda.Event()
            started.record()
            network.contract()
            ended.record()
            ended.synchronize()
            timings.append(cp.cuda.get_elapsed_time(started, ended) / 1000.0)
        properties = cp.cuda.runtime.getDeviceProperties(0)
        return {
            "status": "ok",
            "source": "cutensornet",
            "runtime_semantics": "T_sim_contract: local GPU tensor-network scalar contraction",
            "target_name": "contract_gpu_median_s",
            "observation_unit": "one fixed tensor network under one explicit candidate contraction plan",
            "family": features.family,
            "circuit_variant": features.circuit_variant,
            "num_qubits": features.num_qubits,
            "logical_depth": features.logical_depth,
            "logical_total_gate_count": features.logical_total_gate_count,
            "logical_two_qubit_gate_count": features.logical_two_qubit_gate_count,
            "circuit_seed": seed,
            "candidate_plan": kind,
            "candidate_plan_source": "cuTensorNet TIME_TUNED" if kind == "native_time_tuned" else "explicit deterministic einsum path",
            "operand_count": operand_count,
            "memory_limit": memory_limit,
            "precision": "complex64",
            "path_optimization_s": path_optimization_s,
            "pre_run_runtime_est_s": runtime_est if runtime_est > 0 else "",
            "pre_run_flop_count": flop_count,
            "pre_run_largest_intermediate_elements": float(info.largest_intermediate),
            "pre_run_num_slices": int(network.num_slices),
            "first_contract_gpu_s": first_s,
            "contract_gpu_median_s": statistics.median(timings),
            "contract_gpu_min_s": min(timings),
            "contract_gpu_max_s": max(timings),
            "scalar_real": scalar.real,
            "scalar_imag": scalar.imag,
            "gpu_name": properties["name"].decode(),
            "gpu_compute_capability": f"{compute_capability[:-1]}.{compute_capability[-1]}",
        }
    finally:
        network.free()
        cp.get_default_memory_pool().free_all_blocks()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--qubits", default="8,10,12")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--memory-limit", default="70%")
    parser.add_argument(
        "--random-path-count", type=int, default=0,
        help="Add this many deterministic random explicit paths per circuit. Zero preserves the original three-plan pilot.",
    )
    args = parser.parse_args()
    try:
        import cupy as cp
        import numpy as np
        from cuquantum import tensornet as tn
        from cuquantum.bindings import cutensornet as cutn
    except ImportError as exc:
        raise SystemExit("Requires a CUDA-enabled cuQuantum/CuPy environment") from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise SystemExit("No CUDA GPU visible")
    if args.random_path_count < 0:
        raise SystemExit("random-path-count must be non-negative")
    builder = load_builder()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    candidate_kinds = ["native_time_tuned", "left_fold", "right_fold"] + [
        f"random_path_{path_seed}" for path_seed in range(args.random_path_count)
    ]
    for family in [item for item in args.families.split(",") if item]:
        for num_qubits in [int(item) for item in args.qubits.split(",") if item]:
            for kind in candidate_kinds:
                print(f"BEGIN family={family} qubits={num_qubits} candidate={kind}", flush=True)
                try:
                    row = run_candidate(
                        builder, cp, np, tn, cutn, family, num_qubits, args.seed,
                        args.repeats, args.warmups, kind, args.memory_limit,
                    )
                    rows.append(row)
                    print(
                        f"END candidate={kind} flops={float(row['pre_run_flop_count']):.3g} "
                        f"median_ms={float(row['contract_gpu_median_s']) * 1e3:.3f}",
                        flush=True,
                    )
                except Exception as exc:
                    failure = {
                        "family": family, "num_qubits": num_qubits,
                        "candidate_plan": kind, "error": repr(exc),
                    }
                    failures.append(failure)
                    print(f"FAILED {failure}", flush=True)
    if rows:
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    args.output.with_suffix(".failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    args.output.with_suffix(".environment.json").write_text(json.dumps({
        "python": sys.version,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cuquantum": __import__("cuquantum").__version__,
        "families": args.families,
        "qubits": args.qubits,
        "candidate_plans": candidate_kinds,
        "target_semantics": "T_sim_contract: local GPU tensor-network scalar contraction",
    }, indent=2) + "\n")
    print(f"WROTE {len(rows)} rows; failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
