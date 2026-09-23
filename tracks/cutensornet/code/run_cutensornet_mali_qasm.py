#!/usr/bin/env python3
"""Contract Ma-Li MQT Bench QASM circuits with cuTensorNet RUNTIME_EST.

This uses the public 1,510-circuit Ma-Li QASM pool. It does *not* predict
Osaka/Kyoto result.time_taken. The target is the same local GPU scalar
contraction clock as the synthetic cuTensorNet runners:

    T_sim_contract = warm CUDA-event median of a zero-bitstring amplitude.

The 1,510-circuit pool is not fully executable as exact tensor-network
contraction on one 16 GB GPU. This runner selects a documented subset
(default: all 22 families with n_qubits <= 10) and records skips.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cutensornet_runtime_benchmark import get_optimizer_value, write_csv  # noqa: E402

CANDIDATE_QASM_DIRS = (
    ROOT / "work/mali/data/quantum_circuits",
    Path("/home/server/Documents/Quantum-Execution-Time-Prediction/data/quantum_circuits"),
)


def default_qasm_dir() -> Path:
    for candidate in CANDIDATE_QASM_DIRS:
        if candidate.is_dir():
            return candidate
    return CANDIDATE_QASM_DIRS[0]


def parse_family_width(stem: str) -> tuple[str, int]:
    family, width_text = stem.split("_indep_qiskit_")
    return family, int(width_text)


def strip_circuit(qc: Any) -> Any:
    from qiskit import QuantumCircuit

    qc.remove_final_measurements()
    cleaned = QuantumCircuit(*qc.qregs)
    skipped = 0
    for instruction in qc.data:
        if instruction.operation.name in {"barrier", "delay", "measure", "reset"}:
            skipped += 1
            continue
        cleaned.append(instruction)
    return cleaned, skipped


def list_qasm(qasm_dir: Path) -> list[Path]:
    return sorted(qasm_dir.glob("*_indep_qiskit_*.qasm"))


def select_circuits(paths: list[Path], min_qubits: int, max_qubits: int, families: set[str] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for path in paths:
        family, width = parse_family_width(path.stem)
        record = {
            "circuit_id": path.stem,
            "family": family,
            "num_qubits": width,
            "qasm_name": path.name,
        }
        if families is not None and family not in families:
            record["skip_reason"] = "family_not_requested"
            skipped.append(record)
            continue
        if width < min_qubits or width > max_qubits:
            record["skip_reason"] = f"width_outside_{min_qubits}_{max_qubits}"
            skipped.append(record)
            continue
        selected.append(record)
    return selected, skipped


def approx_gate_count(path: Path) -> int:
    count = 0
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("//", "OPENQASM", "include", "qreg", "creg", "barrier", "measure")):
            count += 1
    return count


def benchmark_qasm(
    cp: Any,
    np: Any,
    tn: Any,
    cutn: Any,
    qiskit_mod: Any,
    path: Path,
    family: str,
    num_qubits: int,
    warmups: int,
    repeats: int,
    memory_limit: str,
    optimizer_samples: int,
    optimizer_seed: int,
    optimizer_cost_function: str,
    gpu_arch: int,
) -> dict[str, Any]:
    raw = path.read_bytes()
    qc = qiskit_mod.QuantumCircuit.from_qasm_file(str(path))
    cleaned, n_stripped = strip_circuit(qc)
    if cleaned.num_qubits != num_qubits:
        raise RuntimeError(
            f"{path.name}: filename width {num_qubits} != circuit width {cleaned.num_qubits}"
        )
    n1 = n2 = n3 = 0
    for instruction in cleaned.data:
        nq = len(instruction.qubits)
        if nq >= 3:
            n3 += 1
        elif nq == 2:
            n2 += 1
        else:
            n1 += 1
    compute_capability = str(cp.cuda.Device().compute_capability)
    compute_capability_display = (
        f"{compute_capability[:-1]}.{compute_capability[-1]}"
        if len(compute_capability) > 1 else compute_capability
    )
    build_started = time.perf_counter()
    converter = tn.CircuitToEinsum(cleaned, dtype="complex64", backend="cupy")
    expression, operands = converter.amplitude("0" * cleaned.num_qubits)
    build_finished = time.perf_counter()
    options = tn.NetworkOptions(memory_limit=memory_limit, blocking="auto")
    network = tn.Network(expression, *operands, options=options)
    try:
        cost_function = getattr(cutn.OptimizerCost, optimizer_cost_function)
        optimizer = tn.OptimizerOptions(
            samples=optimizer_samples,
            seed=optimizer_seed,
            threads=max(1, (os.cpu_count() or 2) // 2),
            cost_function=cost_function,
            gpu_arch=gpu_arch,
        )
        path_started = time.perf_counter()
        _, optimizer_info = network.contract_path(optimizer)
        path_finished = time.perf_counter()
        info = cutn.ContractionOptimizerInfoAttribute
        runtime_est = get_optimizer_value(cutn, network, info.RUNTIME_EST, np)
        effective_flops = get_optimizer_value(cutn, network, info.EFFECTIVE_FLOPS_EST, np)
        flop_count = get_optimizer_value(cutn, network, info.FLOP_COUNT, np)
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
        properties = cp.cuda.runtime.getDeviceProperties(0)
        return {
            "source": "cutensornet_mali_qasm",
            "runtime_semantics": "T_sim_contract: local GPU tensor-network scalar amplitude of Ma-Li MQT QASM; not result.time_taken",
            "observation_unit": "one QASM-derived zero-bitstring amplitude contraction",
            "target_name": "contract_gpu_median_s",
            "circuit_id": path.stem,
            "qasm_name": path.name,
            "qasm_sha256": hashlib.sha256(raw).hexdigest(),
            "family": family,
            "circuit_variant": "mqt_bench_v1_indep_qiskit",
            "num_qubits": cleaned.num_qubits,
            "logical_total_gate_count": n1 + n2 + n3,
            "logical_single_qubit_gate_count": n1,
            "logical_two_qubit_gate_count": n2,
            "logical_multi_qubit_gate_count": n3,
            "n_stripped_ops": n_stripped,
            "n_tensors": len(operands),
            "precision": "complex64",
            "workspace_memory_limit": memory_limit,
            "optimizer_cost_function": optimizer_cost_function,
            "optimizer_samples": optimizer_samples,
            "optimizer_seed": optimizer_seed,
            "optimizer_gpu_arch": gpu_arch,
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
            "end_to_end_first_s": (path_finished - build_started) + first_contract_s,
            "warmups": warmups,
            "repeats": repeats,
            "gpu_name": properties["name"].decode(),
            "gpu_compute_capability": compute_capability_display,
            "gpu_memory_bytes": int(properties["totalGlobalMem"]),
            "status": "ok",
        }
    finally:
        network.free()
        cp.get_default_memory_pool().free_all_blocks()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qasm-dir", type=Path, default=default_qasm_dir())
    parser.add_argument("--min-qubits", type=int, default=2)
    parser.add_argument("--max-qubits", type=int, default=10)
    parser.add_argument("--families", default="", help="Optional comma-separated family filter")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--memory-limit", default="90%")
    parser.add_argument("--optimizer-samples", type=int, default=16)
    parser.add_argument("--heavy-optimizer-samples", type=int, default=8)
    parser.add_argument("--heavy-gate-threshold", type=int, default=800)
    parser.add_argument("--optimizer-seed", type=int, default=137)
    parser.add_argument("--optimizer-cost-function", default="TIME_TUNED")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-gates", type=int, default=2500,
                        help="Skip selected-width circuits whose stripped gate count exceeds this cap.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "run-output" / "cutensornet" / "mali_qasm_qle10")
    args = parser.parse_args()

    try:
        import cupy as cp
        import numpy as np
        from cuquantum import tensornet as tn
        from cuquantum.bindings import cutensornet as cutn
        import qiskit
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}") from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise SystemExit("No CUDA GPU visible to CuPy.")

    qasm_dir = args.qasm_dir
    if not qasm_dir.is_dir():
        raise SystemExit(f"QASM directory not found: {qasm_dir}")
    family_filter = {item.strip() for item in args.families.split(",") if item.strip()} or None
    all_paths = list_qasm(qasm_dir)
    selected, skipped = select_circuits(all_paths, args.min_qubits, args.max_qubits, family_filter)
    still_selected = []
    for item in selected:
        n_gates = approx_gate_count(qasm_dir / item["qasm_name"])
        item["n_gates_approx"] = n_gates
        if n_gates > args.max_gates:
            item["skip_reason"] = f"gate_count_{n_gates}_exceeds_{args.max_gates}"
            skipped.append(item)
        else:
            still_selected.append(item)
    selected = sorted(still_selected, key=lambda row: (row["n_gates_approx"], row["num_qubits"], row["circuit_id"]))
    for item in skipped:
        item.setdefault("n_gates_approx", None)
        item.setdefault("skip_reason", "unspecified")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection.json").write_text(json.dumps({
        "qasm_dir": str(qasm_dir),
        "pool_size": len(all_paths),
        "n_selected": len(selected),
        "n_skipped": len(skipped),
        "min_qubits": args.min_qubits,
        "max_qubits": args.max_qubits,
        "families": sorted({row["family"] for row in selected}),
        "optimizer_samples": args.optimizer_samples,
        "heavy_optimizer_samples": args.heavy_optimizer_samples,
        "heavy_gate_threshold": args.heavy_gate_threshold,
        "memory_limit": args.memory_limit,
        "optimizer_seed": args.optimizer_seed,
        "note": "Skipped circuits remain in the Ma-Li pool; they were not silently deleted.",
    }, indent=2) + "\n")
    write_csv(args.output_dir / "skipped.csv", skipped)

    compute_capability = str(cp.cuda.Device().compute_capability)
    gpu_arch = int(compute_capability[:-1]) if len(compute_capability) > 1 else int(compute_capability)
    result_path = args.output_dir / "cutensornet_mali_qasm.csv"
    failure_path = args.output_dir / "failures.json"
    done_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    if result_path.exists():
        with result_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
            done_ids = {row["circuit_id"] for row in rows}
    failures: list[dict[str, Any]] = []
    if failure_path.exists():
        failures = json.loads(failure_path.read_text())
        done_ids.update(row["circuit_id"] for row in failures if row.get("status") == "timeout")

    pending = [row for row in selected if row["circuit_id"] not in done_ids]
    print(
        f"Pool {len(all_paths)}; selected {len(selected)}; skipped {len(skipped)}; "
        f"already done {len(done_ids)}; pending {len(pending)}",
        flush=True,
    )
    started_all = time.perf_counter()
    for index, item in enumerate(pending, 1):
        path = qasm_dir / item["qasm_name"]
        n_gates = 0
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(("//", "OPENQASM", "include", "qreg", "creg", "barrier", "measure")):
                n_gates += 1
        samples = args.heavy_optimizer_samples if n_gates >= args.heavy_gate_threshold else args.optimizer_samples
        print(
            f"[{index}/{len(pending)}] {item['circuit_id']} gates~{n_gates} samples={samples}",
            flush=True,
        )
        circuit_started = time.perf_counter()
        try:
            row = benchmark_qasm(
                cp, np, tn, cutn, qiskit, path, item["family"], item["num_qubits"],
                args.warmups, args.repeats, args.memory_limit, samples,
                args.optimizer_seed, args.optimizer_cost_function.upper(), gpu_arch,
            )
            elapsed = time.perf_counter() - circuit_started
            if elapsed > args.timeout_s:
                raise TimeoutError(f"finished but exceeded timeout: {elapsed:.1f}s")
            row["wall_s"] = elapsed
            rows.append(row)
            write_csv(result_path, rows)
            ratio = float(row["contract_gpu_median_s"]) / float(row["cutensornet_runtime_est_s"]) if float(row["cutensornet_runtime_est_s"]) else float("nan")
            print(
                f"  ok warm={row['contract_gpu_median_s']:.4g}s est={row['cutensornet_runtime_est_s']:.4g}s "
                f"ratio={ratio:.3f} path={row['path_optimization_s']:.3f}s wall={elapsed:.3f}s",
                flush=True,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - circuit_started
            status = "timeout" if isinstance(exc, TimeoutError) or elapsed > args.timeout_s else "error"
            failures.append({
                "circuit_id": item["circuit_id"],
                "family": item["family"],
                "num_qubits": item["num_qubits"],
                "n_gates_approx": n_gates,
                "optimizer_samples": samples,
                "status": status,
                "error": repr(exc),
                "wall_s": elapsed,
            })
            failure_path.write_text(json.dumps(failures, indent=2) + "\n")
            print(f"  {status.upper()} {item['circuit_id']}: {exc}", file=sys.stderr, flush=True)
            # Do not leave a poisoned GPU allocator behind a failed large plan.
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
    summary = {
        "pool_size": len(all_paths),
        "n_selected": len(selected),
        "n_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "n_failed": len(failures),
        "elapsed_s": time.perf_counter() - started_all,
        "result_csv": str(result_path),
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
