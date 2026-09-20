#!/usr/bin/env python3
"""Benchmark cuTensorNet's pre-run contraction-time estimate against GPU time.

This is a simulator-only experiment.  It deliberately stores three different
quantities rather than calling all of them ``runtime``:

* ``cutensornet_runtime_est_s``: pre-run path-optimizer estimate;
* ``contract_gpu_median_s``: warm GPU contraction measured with CUDA events;
* ``end_to_end_first_s``: construction + path search + first contraction.

The runner creates scalar tensor-network contractions for several quantum
circuit families.  Contracting to a scalar avoids materialising a 2**n output
statevector, while retaining the contraction graph induced by every gate.

Run through ``run_cutensornet_runtime_benchmark.sh`` so CUDA 13 runtime
libraries are visible to Python before CuPy is imported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "run-output" / "cutensornet"
DEFAULT_FAMILIES = ("ghz", "hea", "qaoa_cycle", "random_brickwork", "qft")


@dataclass
class CircuitFeatures:
    family: str
    num_qubits: int
    circuit_variant: str = ""
    family_depth_multiplier: float = 1.0
    family_depth_parameter: int = 0
    logical_depth: int = 0
    logical_total_gate_count: int = 0
    logical_two_qubit_gate_count: int = 0
    logical_single_qubit_gate_count: int = 0


class TensorCircuitBuilder:
    """Build a scalar tensor-network representation while retaining features."""

    def __init__(self, cp: Any, num_qubits: int, family: str, seed: int, dtype: Any) -> None:
        self.cp = cp
        self.dtype = dtype
        self.num_qubits = num_qubits
        self.features = CircuitFeatures(family=family, num_qubits=num_qubits)
        self.items: list[Any] = []
        self.current_modes = list(range(num_qubits))
        self.qubit_depths = [0] * num_qubits
        self.next_mode = num_qubits
        self.rng = __import__("random").Random(seed)
        zero = cp.asarray([1.0, 0.0], dtype=self.dtype)
        for qubit in range(num_qubits):
            self.items.extend([zero, (qubit,)])

    def _record_gate(self, qubits: tuple[int, ...]) -> None:
        gate_depth = max(self.qubit_depths[qubit] for qubit in qubits) + 1
        for qubit in qubits:
            self.qubit_depths[qubit] = gate_depth
        self.features.logical_depth = max(self.features.logical_depth, gate_depth)
        self.features.logical_total_gate_count += 1
        if len(qubits) == 1:
            self.features.logical_single_qubit_gate_count += 1
        else:
            self.features.logical_two_qubit_gate_count += 1

    def one_qubit(self, gate: Any, qubit: int) -> None:
        self._record_gate((qubit,))
        new_mode = self.next_mode
        self.next_mode += 1
        self.items.extend([gate, (new_mode, self.current_modes[qubit])])
        self.current_modes[qubit] = new_mode

    def two_qubit(self, gate: Any, q0: int, q1: int) -> None:
        if q0 == q1:
            raise ValueError("Two-qubit gate requires distinct wires")
        self._record_gate((q0, q1))
        new0, new1 = self.next_mode, self.next_mode + 1
        self.next_mode += 2
        self.items.extend([
            gate,
            (new0, new1, self.current_modes[q0], self.current_modes[q1]),
        ])
        self.current_modes[q0], self.current_modes[q1] = new0, new1

    def finish_scalar_amplitude(self) -> tuple[list[Any], CircuitFeatures]:
        zero = self.cp.asarray([1.0, 0.0], dtype=self.dtype)
        for mode in self.current_modes:
            self.items.extend([zero, (mode,)])
        return self.items, self.features

    def h(self) -> Any:
        return self.cp.asarray([[1, 1], [1, -1]], dtype=self.dtype) / math.sqrt(2)

    def rx(self, theta: float) -> Any:
        c, s = math.cos(theta / 2), -1j * math.sin(theta / 2)
        return self.cp.asarray([[c, s], [s, c]], dtype=self.dtype)

    def ry(self, theta: float) -> Any:
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return self.cp.asarray([[c, -s], [s, c]], dtype=self.dtype)

    def rz(self, theta: float) -> Any:
        return self.cp.asarray(
            [[complex(math.cos(theta / 2), -math.sin(theta / 2)), 0],
             [0, complex(math.cos(theta / 2), math.sin(theta / 2))]],
            dtype=self.dtype,
        )

    def cx(self) -> Any:
        return self.cp.asarray(
            [[[[(1 if (out0 == in0 and out1 == (in1 ^ in0)) else 0)
                for in1 in range(2)] for in0 in range(2)]
              for out1 in range(2)] for out0 in range(2)],
            dtype=self.dtype,
        )

    def controlled_phase(self, theta: float) -> Any:
        diagonal = self.cp.asarray([1, 1, 1, complex(math.cos(theta), math.sin(theta))], dtype=self.dtype)
        return self.cp.diag(diagonal).reshape((2, 2, 2, 2))

    def rzz(self, theta: float) -> Any:
        phases = [
            complex(math.cos(-theta / 2), math.sin(-theta / 2)),
            complex(math.cos(theta / 2), math.sin(theta / 2)),
            complex(math.cos(theta / 2), math.sin(theta / 2)),
            complex(math.cos(-theta / 2), math.sin(-theta / 2)),
        ]
        return self.cp.diag(self.cp.asarray(phases, dtype=self.dtype)).reshape((2, 2, 2, 2))


def build_family(
    cp: Any, family: str, num_qubits: int, seed: int, depth_multiplier: float,
    precision: str = "complex64",
) -> tuple[list[Any], CircuitFeatures]:
    try:
        dtype = getattr(cp, precision)
    except AttributeError as exc:
        raise ValueError(f"Unsupported CuPy precision: {precision}") from exc
    circuit = TensorCircuitBuilder(cp, num_qubits, family, seed, dtype)
    circuit.features.family_depth_multiplier = depth_multiplier
    if family == "ghz":
        circuit.features.circuit_variant = "chain"
        circuit.features.family_depth_parameter = 1
        circuit.one_qubit(circuit.h(), 0)
        for qubit in range(num_qubits - 1):
            circuit.two_qubit(circuit.cx(), qubit, qubit + 1)
    elif family == "hea":
        layers = max(1, round(4 * depth_multiplier))
        circuit.features.circuit_variant = f"layers_{layers}"
        circuit.features.family_depth_parameter = layers
        for layer in range(layers):
            for qubit in range(num_qubits):
                circuit.one_qubit(circuit.ry(0.17 * (qubit + 1) * (layer + 1)), qubit)
                circuit.one_qubit(circuit.rz(0.11 * (qubit + 1) * (layer + 1)), qubit)
            for qubit in range((layer + 1) % 2, num_qubits - 1, 2):
                circuit.two_qubit(circuit.cx(), qubit, qubit + 1)
    elif family == "qaoa_cycle":
        layers = max(1, round(3 * depth_multiplier))
        circuit.features.circuit_variant = f"p_{layers}"
        circuit.features.family_depth_parameter = layers
        for qubit in range(num_qubits):
            circuit.one_qubit(circuit.h(), qubit)
        for layer in range(layers):
            gamma, beta = 0.23 * (layer + 1), 0.31 * (layer + 1)
            for qubit in range(num_qubits):
                circuit.two_qubit(circuit.rzz(gamma), qubit, (qubit + 1) % num_qubits)
            for qubit in range(num_qubits):
                circuit.one_qubit(circuit.rx(2 * beta), qubit)
    elif family == "random_brickwork":
        layers = max(1, round(8 * depth_multiplier))
        circuit.features.circuit_variant = f"layers_{layers}"
        circuit.features.family_depth_parameter = layers
        for layer in range(layers):
            for qubit in range(num_qubits):
                circuit.one_qubit(circuit.ry(circuit.rng.uniform(-math.pi, math.pi)), qubit)
                circuit.one_qubit(circuit.rz(circuit.rng.uniform(-math.pi, math.pi)), qubit)
            for qubit in range(layer % 2, num_qubits - 1, 2):
                circuit.two_qubit(circuit.cx(), qubit, qubit + 1)
    elif family == "qft":
        circuit.features.circuit_variant = "full"
        circuit.features.family_depth_parameter = 1
        for target in range(num_qubits):
            circuit.one_qubit(circuit.h(), target)
            for control in range(target + 1, num_qubits):
                circuit.two_qubit(circuit.controlled_phase(math.pi / (2 ** (control - target))), control, target)
    else:
        raise ValueError(f"Unknown family {family!r}; choose from {', '.join(DEFAULT_FAMILIES)}")
    return circuit.finish_scalar_amplitude()


def get_optimizer_value(cutn: Any, network: Any, attribute: Any, np: Any) -> float:
    dtype = cutn.contraction_optimizer_info_get_attribute_dtype(attribute)
    value = np.empty(1, dtype=dtype)
    cutn.contraction_optimizer_info_get_attribute(
        network.handle, network.optimizer_info_ptr, attribute, value.ctypes.data, value.nbytes
    )
    return float(value[0])


def benchmark_one(
    cp: Any,
    np: Any,
    tn: Any,
    cutn: Any,
    family: str,
    num_qubits: int,
    seed: int,
    depth_multiplier: float,
    warmups: int,
    repeats: int,
    memory_limit: str,
    optimizer_samples: int,
    optimizer_seed: int,
) -> dict[str, Any]:
    compute_capability = str(cp.cuda.Device().compute_capability)
    # CuPy returns values such as ``"120"`` for Blackwell compute capability
    # 12.0, while cuTensorNet expects the major architecture number (12).
    gpu_arch = int(compute_capability[:-1]) if len(compute_capability) > 1 else int(compute_capability)
    compute_capability_display = f"{compute_capability[:-1]}.{compute_capability[-1]}" if len(compute_capability) > 1 else compute_capability
    build_started = time.perf_counter()
    operands, features = build_family(cp, family, num_qubits, seed, depth_multiplier)
    build_finished = time.perf_counter()
    # ``auto`` leaves GPU execution asynchronous while keeping the high-level
    # API compatible across cuQuantum releases; CUDA events below synchronize
    # precisely around each contraction.
    options = tn.NetworkOptions(memory_limit=memory_limit, blocking="auto")
    network = tn.Network(*operands, options=options)
    try:
        optimizer = tn.OptimizerOptions(
            samples=optimizer_samples,
            seed=optimizer_seed,
            threads=max(1, (os.cpu_count() or 2) // 2),
            cost_function=cutn.OptimizerCost.TIME_TUNED,
            gpu_arch=gpu_arch,
        )
        path_started = time.perf_counter()
        _, optimizer_info = network.contract_path(optimizer)
        path_finished = time.perf_counter()
        runtime_est = get_optimizer_value(cutn, network, cutn.ContractionOptimizerInfoAttribute.RUNTIME_EST, np)
        effective_flops = get_optimizer_value(cutn, network, cutn.ContractionOptimizerInfoAttribute.EFFECTIVE_FLOPS_EST, np)
        flop_count = get_optimizer_value(cutn, network, cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT, np)
        first_start, first_end = cp.cuda.Event(), cp.cuda.Event()
        first_start.record()
        network.contract()
        first_end.record()
        first_end.synchronize()
        first_contract_s = cp.cuda.get_elapsed_time(first_start, first_end) / 1_000.0
        for _ in range(warmups):
            network.contract()
        cp.cuda.get_current_stream().synchronize()
        timings: list[float] = []
        for _ in range(repeats):
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            network.contract()
            end.record()
            end.synchronize()
            timings.append(cp.cuda.get_elapsed_time(start, end) / 1_000.0)
        end_to_end_first_s = (path_finished - build_started) + first_contract_s
        properties = cp.cuda.runtime.getDeviceProperties(0)
        return {
            "source": "cutensornet",
            "runtime_semantics": "T_sim_contract: local GPU tensor-network scalar contraction",
            "observation_unit": "one circuit-derived tensor-network contraction plan",
            "target_name": "contract_gpu_median_s",
            "family": features.family,
            "circuit_variant": features.circuit_variant,
            "family_depth_multiplier": features.family_depth_multiplier,
            "family_depth_parameter": features.family_depth_parameter,
            "num_qubits": features.num_qubits,
            "logical_depth": features.logical_depth,
            "logical_total_gate_count": features.logical_total_gate_count,
            "logical_single_qubit_gate_count": features.logical_single_qubit_gate_count,
            "logical_two_qubit_gate_count": features.logical_two_qubit_gate_count,
            "precision": "complex64",
            "optimizer_cost_function": "TIME_TUNED",
            "optimizer_samples": optimizer_samples,
            "optimizer_seed": optimizer_seed,
            "cutensornet_runtime_est_s": runtime_est,
            "cutensornet_effective_flops_est": effective_flops,
            "cutensornet_flop_count": flop_count,
            "cutensornet_largest_intermediate_elements": float(optimizer_info.largest_intermediate),
            "cutensornet_num_slices": int(network.num_slices),
            "tensor_network_build_s": build_finished - build_started,
            "path_optimization_s": path_finished - path_started,
            "first_contract_gpu_s": first_contract_s,
            "contract_gpu_median_s": statistics.median(timings),
            "contract_gpu_mean_s": statistics.fmean(timings),
            "contract_gpu_min_s": min(timings),
            "contract_gpu_max_s": max(timings),
            "contract_gpu_stdev_s": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            "end_to_end_first_s": end_to_end_first_s,
            "warmups": warmups,
            "repeats": repeats,
            "seed": seed,
            "gpu_name": properties["name"].decode(),
            "gpu_compute_capability": compute_capability_display,
            "gpu_memory_bytes": int(properties["totalGlobalMem"]),
        }
    finally:
        network.free()
        cp.get_default_memory_pool().free_all_blocks()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", default="16,20,24,28,30", help="Comma-separated widths")
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES), help="Comma-separated family names")
    parser.add_argument(
        "--depth-multipliers", default="1",
        help=("Comma-separated multipliers of each family's default depth "
              "(HEA=4 layers, QAOA=p3, brickwork=8 layers). GHZ/QFT use 1 only."),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--memory-limit", default="70%", help="cuTensorNet workspace limit")
    parser.add_argument(
        "--optimizer-samples", default="16",
        help=("Comma-separated cuTensorNet path-search sample budgets. Each budget "
              "produces one selected candidate plan for the same tensor network."),
    )
    parser.add_argument(
        "--optimizer-seeds", default="42",
        help=("Comma-separated hyperoptimizer RNG seeds. Multiple values create "
              "candidate plans for the same circuit and sample budget."),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        import cupy as cp
        import numpy as np
        from cuquantum import tensornet as tn
        from cuquantum.bindings import cutensornet as cutn
    except ImportError as exc:
        raise SystemExit("cuQuantum/cuPy is unavailable. Use run_cutensornet_runtime_benchmark.sh after setup.") from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise SystemExit("No CUDA GPU visible to cuPy.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    widths = [int(value) for value in args.qubits.split(",") if value.strip()]
    families = [value.strip() for value in args.families.split(",") if value.strip()]
    depth_multipliers = [float(value) for value in args.depth_multipliers.split(",") if value.strip()]
    optimizer_samples = [int(value) for value in args.optimizer_samples.split(",") if value.strip()]
    optimizer_seeds = [int(value) for value in args.optimizer_seeds.split(",") if value.strip()]
    if not optimizer_samples or any(value < 1 for value in optimizer_samples):
        raise SystemExit("--optimizer-samples must contain positive integers")
    if not optimizer_seeds:
        raise SystemExit("--optimizer-seeds must contain at least one integer")
    for family in families:
        for num_qubits in widths:
            multipliers = (1.0,) if family in {"ghz", "qft"} else depth_multipliers
            for depth_multiplier in multipliers:
                for sample_budget in optimizer_samples:
                    for optimizer_seed in optimizer_seeds:
                        print(
                            f"Benchmarking {family}, {num_qubits} qubits, depth multiplier "
                            f"{depth_multiplier:g}, optimizer samples {sample_budget}, "
                            f"optimizer seed {optimizer_seed}",
                            flush=True,
                        )
                        try:
                            rows.append(benchmark_one(
                                cp, np, tn, cutn, family, num_qubits, args.seed, depth_multiplier,
                                args.warmups, args.repeats, args.memory_limit, sample_budget, optimizer_seed,
                            ))
                        except Exception as exc:  # Preserve failed configurations rather than hiding them.
                            failures.append({
                                "family": family,
                                "num_qubits": num_qubits,
                                "depth_multiplier": depth_multiplier,
                                "optimizer_samples": sample_budget,
                                "optimizer_seed": optimizer_seed,
                                "error": repr(exc),
                            })
                            print(
                                f"FAILED {family}/{num_qubits}/x{depth_multiplier:g}/samples{sample_budget}/seed{optimizer_seed}: {exc}",
                                file=sys.stderr, flush=True,
                            )
    write_csv(args.output_dir / "cutensornet_runtime_benchmark.csv", rows)
    failure_path = args.output_dir / "cutensornet_runtime_benchmark_failures.json"
    if failures:
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    else:
        failure_path.unlink(missing_ok=True)
    print(f"Wrote {len(rows)} successful rows to {args.output_dir}")
    if failures:
        print(f"Recorded {len(failures)} failed configurations; inspect failures JSON.", file=sys.stderr)


if __name__ == "__main__":
    main()
