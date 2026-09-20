#!/usr/bin/env python3
"""Generate source-aware dense-statevector runtime labels.

The experiment deliberately measures one narrowly defined target:

  prepared_execute_reset_state

That target allocates a fresh |0...0> state, applies a pre-materialized gate
schedule, and evaluates <Z_0>.  It excludes circuit construction, gate tensor
materialization, compiler/transpiler work, sampling and host-to-device result
transfer.  Those quantities are nevertheless recorded separately where they
exist.  CPU and CUDA measurements use the same PyTorch implementation and can
therefore form a matched device/precision matrix; they are not Aer, CUDA-Q or
cuTensorNet labels.

The output is append-only JSONL plus a rectangular CSV projection.  A row is a
single logical circuit/configuration and one execution context, never an
implicit generic ``runtime_seconds`` observation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "simulator_runtime_v1" / "torch_dense_statevector_v1"
TARGET_SEMANTICS = (
    "prepared dense-statevector execution: allocate a fresh |0...0> state, "
    "apply the materialized gate schedule, and evaluate <Z_0>; excludes "
    "schedule construction, gate materialization, compiler/transpiler and sampling"
)
COLD_SEMANTICS = (
    "first execution after this circuit's gate tensors were materialized; "
    "the Python process and CUDA context have already been initialized"
)
FAMILIES = ("ghz", "hea", "qaoa_cycle", "random_brickwork", "qft")


@dataclass(frozen=True)
class GateSpec:
    name: str
    qubits: tuple[int, ...]
    theta: float | None = None


@dataclass(frozen=True)
class CircuitSpec:
    circuit_id: str
    family: str
    num_qubits: int
    depth_parameter: int
    seed: int
    gates: tuple[GateSpec, ...]
    logical_depth: int
    logical_total_gate_count: int
    logical_two_qubit_gate_count: int
    logical_single_qubit_gate_count: int
    logical_gate_histogram: dict[str, int]


@dataclass(frozen=True)
class MaterializedGate:
    qubits: tuple[int, ...]
    matrix: torch.Tensor


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def csv_list(raw: str, cast: type = str) -> list[Any]:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def parse_contexts(raw: str) -> list[tuple[str, str]]:
    contexts: list[tuple[str, str]] = []
    for item in csv_list(raw):
        try:
            device, precision = item.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid context '{item}'; expected cpu:complex64") from exc
        device = device.strip().lower()
        precision = precision.strip().lower()
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported device '{device}'")
        if precision not in {"complex64", "complex128"}:
            raise ValueError(f"Unsupported precision '{precision}'")
        contexts.append((device, precision))
    if not contexts:
        raise ValueError("At least one context is required")
    return contexts


def _append_gate(
    gates: list[GateSpec],
    qubit_depths: list[int],
    histogram: dict[str, int],
    name: str,
    qubits: Iterable[int],
    theta: float | None = None,
) -> None:
    wires = tuple(int(qubit) for qubit in qubits)
    if not wires or len(set(wires)) != len(wires):
        raise ValueError(f"Invalid wires for {name}: {wires}")
    current_depth = max(qubit_depths[wire] for wire in wires) + 1
    for wire in wires:
        qubit_depths[wire] = current_depth
    histogram[name] = histogram.get(name, 0) + 1
    gates.append(GateSpec(name=name, qubits=wires, theta=theta))


def build_circuit_spec(family: str, num_qubits: int, depth_parameter: int, seed: int) -> CircuitSpec:
    if family not in FAMILIES:
        raise ValueError(f"Unknown family '{family}'")
    if num_qubits < 2:
        raise ValueError("At least two qubits are required")
    gates: list[GateSpec] = []
    histogram: dict[str, int] = {}
    depths = [0] * num_qubits
    rng = __import__("random").Random(seed)

    if family == "ghz":
        _append_gate(gates, depths, histogram, "h", (0,))
        for qubit in range(num_qubits - 1):
            _append_gate(gates, depths, histogram, "cx", (qubit, qubit + 1))
    elif family == "hea":
        for layer in range(depth_parameter):
            for qubit in range(num_qubits):
                _append_gate(gates, depths, histogram, "ry", (qubit,), 0.17 * (qubit + 1) * (layer + 1))
                _append_gate(gates, depths, histogram, "rz", (qubit,), 0.11 * (qubit + 1) * (layer + 1))
            for qubit in range(layer % 2, num_qubits - 1, 2):
                _append_gate(gates, depths, histogram, "cx", (qubit, qubit + 1))
    elif family == "qaoa_cycle":
        for qubit in range(num_qubits):
            _append_gate(gates, depths, histogram, "h", (qubit,))
        for layer in range(depth_parameter):
            gamma = 0.23 * (layer + 1)
            beta = 0.31 * (layer + 1)
            for qubit in range(num_qubits):
                _append_gate(gates, depths, histogram, "rzz", (qubit, (qubit + 1) % num_qubits), gamma)
            for qubit in range(num_qubits):
                _append_gate(gates, depths, histogram, "rx", (qubit,), 2.0 * beta)
    elif family == "random_brickwork":
        for layer in range(depth_parameter):
            for qubit in range(num_qubits):
                _append_gate(gates, depths, histogram, "ry", (qubit,), rng.uniform(-math.pi, math.pi))
                _append_gate(gates, depths, histogram, "rz", (qubit,), rng.uniform(-math.pi, math.pi))
            for qubit in range(layer % 2, num_qubits - 1, 2):
                _append_gate(gates, depths, histogram, "cx", (qubit, qubit + 1))
    elif family == "qft":
        for target in range(num_qubits):
            _append_gate(gates, depths, histogram, "h", (target,))
            for control in range(target + 1, num_qubits):
                theta = math.pi / (2 ** (control - target))
                _append_gate(gates, depths, histogram, "cp", (control, target), theta)
        for lower in range(num_qubits // 2):
            _append_gate(gates, depths, histogram, "swap", (lower, num_qubits - lower - 1))

    canonical = {
        "family": family,
        "num_qubits": num_qubits,
        "depth_parameter": depth_parameter,
        "seed": seed,
        "gates": [asdict(gate) for gate in gates],
    }
    circuit_id = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    two_qubit = sum(1 for gate in gates if len(gate.qubits) == 2)
    return CircuitSpec(
        circuit_id=circuit_id,
        family=family,
        num_qubits=num_qubits,
        depth_parameter=depth_parameter,
        seed=seed,
        gates=tuple(gates),
        logical_depth=max(depths),
        logical_total_gate_count=len(gates),
        logical_two_qubit_gate_count=two_qubit,
        logical_single_qubit_gate_count=len(gates) - two_qubit,
        logical_gate_histogram=histogram,
    )


def matrix_for(gate: GateSpec, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    theta = gate.theta
    if gate.name == "h":
        raw = [[1, 1], [1, -1]]
        return torch.tensor(raw, dtype=dtype, device=device) / math.sqrt(2.0)
    if gate.name == "rx":
        assert theta is not None
        c, s = math.cos(theta / 2.0), -1j * math.sin(theta / 2.0)
        return torch.tensor([[c, s], [s, c]], dtype=dtype, device=device)
    if gate.name == "ry":
        assert theta is not None
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        return torch.tensor([[c, -s], [s, c]], dtype=dtype, device=device)
    if gate.name == "rz":
        assert theta is not None
        phase = theta / 2.0
        return torch.tensor(
            [[complex(math.cos(phase), -math.sin(phase)), 0], [0, complex(math.cos(phase), math.sin(phase))]],
            dtype=dtype,
            device=device,
        )
    if gate.name == "cx":
        return torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=dtype, device=device)
    if gate.name == "cp":
        assert theta is not None
        phase = complex(math.cos(theta), math.sin(theta))
        return torch.diag(torch.tensor([1, 1, 1, phase], dtype=dtype, device=device))
    if gate.name == "rzz":
        assert theta is not None
        low = complex(math.cos(-theta / 2.0), math.sin(-theta / 2.0))
        high = complex(math.cos(theta / 2.0), math.sin(theta / 2.0))
        return torch.diag(torch.tensor([low, high, high, low], dtype=dtype, device=device))
    if gate.name == "swap":
        return torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=dtype, device=device)
    raise ValueError(f"Unsupported gate '{gate.name}'")


def materialize(spec: CircuitSpec, precision: str, device: torch.device) -> tuple[MaterializedGate, ...]:
    dtype = torch.complex64 if precision == "complex64" else torch.complex128
    return tuple(MaterializedGate(gate.qubits, matrix_for(gate, dtype, device)) for gate in spec.gates)


def apply_gate(state: torch.Tensor, gate: MaterializedGate) -> torch.Tensor:
    wire_count = len(gate.qubits)
    remaining = [axis for axis in range(state.ndim) if axis not in gate.qubits]
    permutation = list(gate.qubits) + remaining
    inverse = [0] * len(permutation)
    for index, value in enumerate(permutation):
        inverse[value] = index
    left_dim = 2 ** wire_count
    updated = gate.matrix @ state.permute(permutation).reshape(left_dim, -1)
    reshaped = updated.reshape([2] * state.ndim)
    return reshaped.permute(inverse)


def execute(program: tuple[MaterializedGate, ...], num_qubits: int, precision: str, device: torch.device) -> torch.Tensor:
    dtype = torch.complex64 if precision == "complex64" else torch.complex128
    state = torch.zeros([2] * num_qubits, dtype=dtype, device=device)
    state[(0,) * num_qubits] = 1.0
    for gate in program:
        state = apply_gate(state, gate)
    probabilities = state.reshape(2, -1).abs().square()
    return probabilities[0].sum() - probabilities[1].sum()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_once(program: tuple[MaterializedGate, ...], spec: CircuitSpec, precision: str, device: torch.device) -> tuple[float, float]:
    synchronize(device)
    started = time.perf_counter()
    value = execute(program, spec.num_qubits, precision, device)
    synchronize(device)
    elapsed = time.perf_counter() - started
    return elapsed, float(value.detach().cpu().item())


def statevector_bytes(num_qubits: int, precision: str) -> int:
    return (2 ** num_qubits) * (8 if precision == "complex64" else 16)


def make_plan(families: list[str], widths: list[int], seed: int) -> list[tuple[str, int, int, int]]:
    depth_profiles = {
        "ghz": [1],
        "hea": [2, 4],
        "qaoa_cycle": [2, 4],
        "random_brickwork": [4, 8],
        "qft": [1],
    }
    plan: list[tuple[str, int, int, int]] = []
    for family in families:
        for width in widths:
            for depth in depth_profiles[family]:
                plan.append((family, width, depth, seed))
    return plan


def to_csv(records_path: Path, output_path: Path) -> None:
    if not records_path.exists():
        return
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fields = sorted({field for row in rows for field in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flattened = {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            writer.writerow(flattened)


def load_completed(records_path: Path) -> set[str]:
    if not records_path.exists():
        return set()
    return {
        str(json.loads(line).get("record_id"))
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def environment_manifest(args: argparse.Namespace, contexts: list[tuple[str, str]]) -> dict[str, Any]:
    cuda_name = None
    cuda_total = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        cuda_name = properties.name
        cuda_total = int(properties.total_memory)
    return {
        "schema_version": "torch_dense_statevector_v1",
        "created_utc": utc_now(),
        "target_semantics": TARGET_SEMANTICS,
        "cold_semantics": COLD_SEMANTICS,
        "contexts_requested": [f"{device}:{precision}" for device, precision in contexts],
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
        "hardware": {
            "cpu_count": os.cpu_count(),
            "torch_cpu_threads": torch.get_num_threads(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_name": cuda_name,
            "cuda_total_bytes": cuda_total,
        },
        "known_boundaries": [
            "This is a PyTorch dense-statevector reference kernel, not Qiskit Aer, CUDA-Q or cuTensorNet.",
            "No transpilation, noise model, sampling or cloud/QPU timing is included.",
            "No cross-GPU conclusion is possible because this machine has one GPU.",
        ],
    }


def measure_cell(spec: CircuitSpec, device_name: str, precision: str, warmup: int, repeats: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "torch_dense_statevector_v1",
        "recorded_utc": utc_now(),
        "simulator": "torch_dense_statevector_reference",
        "runtime_semantics": TARGET_SEMANTICS,
        "cold_semantics": COLD_SEMANTICS,
        "compile_or_transpile_semantics": "not_applicable: direct gate schedule; no compiler/transpiler",
        "sampling_semantics": "not_applicable: exact statevector and <Z_0> expectation only",
        "family": spec.family,
        "circuit_id": spec.circuit_id,
        "num_qubits": spec.num_qubits,
        "depth_parameter": spec.depth_parameter,
        "seed": spec.seed,
        "logical_depth": spec.logical_depth,
        "logical_total_gate_count": spec.logical_total_gate_count,
        "logical_two_qubit_gate_count": spec.logical_two_qubit_gate_count,
        "logical_single_qubit_gate_count": spec.logical_single_qubit_gate_count,
        "logical_gate_histogram": spec.logical_gate_histogram,
        "execution_device": device_name,
        "precision": precision,
        "context_id": f"{device_name}:{precision}",
        "warmup_count": warmup,
        "repeat_count": repeats,
        "statevector_bytes": statevector_bytes(spec.num_qubits, precision),
    }
    if device_name == "cuda" and not torch.cuda.is_available():
        record.update(status="unsupported", error="torch.cuda.is_available() is False")
        return record
    device = torch.device(device_name)
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            synchronize(device)

        started = time.perf_counter()
        e2e_spec = build_circuit_spec(spec.family, spec.num_qubits, spec.depth_parameter, spec.seed)
        e2e_program = materialize(e2e_spec, precision, device)
        e2e_elapsed, e2e_expectation = run_once(e2e_program, e2e_spec, precision, device)
        synchronize(device)
        record["end_to_end_first_seconds"] = time.perf_counter() - started
        record["end_to_end_execution_component_seconds"] = e2e_elapsed
        record["end_to_end_z0_expectation"] = e2e_expectation
        del e2e_program

        preparation_started = time.perf_counter()
        program = materialize(spec, precision, device)
        synchronize(device)
        record["gate_materialization_seconds"] = time.perf_counter() - preparation_started

        cold_seconds, cold_expectation = run_once(program, spec, precision, device)
        record["cold_post_prepare_seconds"] = cold_seconds
        record["cold_z0_expectation"] = cold_expectation
        for _ in range(warmup):
            run_once(program, spec, precision, device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
        samples: list[float] = []
        expectation = cold_expectation
        for _ in range(repeats):
            elapsed, expectation = run_once(program, spec, precision, device)
            samples.append(elapsed)
        record["warm_execution_seconds_samples"] = samples
        record["warm_execution_median_seconds"] = statistics.median(samples)
        record["warm_execution_mean_seconds"] = statistics.mean(samples)
        record["warm_execution_stdev_seconds"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        record["warm_z0_expectation"] = expectation
        if device.type == "cuda":
            record["torch_cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            record["torch_cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        else:
            record["torch_cuda_peak_allocated_bytes"] = None
            record["torch_cuda_peak_reserved_bytes"] = None
        record["status"] = "ok"
        del program
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # preserve failures as experimental rows
        record.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=5),
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--widths", default="16,20,24")
    parser.add_argument(
        "--contexts",
        default="cpu:complex64,cpu:complex128,cuda:complex64,cuda:complex128",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    families = csv_list(args.families)
    unknown_families = sorted(set(families) - set(FAMILIES))
    if unknown_families:
        raise ValueError(f"Unknown families: {unknown_families}")
    widths = csv_list(args.widths, int)
    if min(widths) < 2:
        raise ValueError("Widths must be >= 2")
    contexts = parse_contexts(args.contexts)
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")
    torch.set_num_threads(max(1, args.cpu_threads))
    torch.set_num_interop_threads(1)

    plan = make_plan(families, widths, args.seed)
    jobs = [(family, width, depth, seed, device, precision) for family, width, depth, seed in plan for device, precision in contexts]
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
    print(f"planned_jobs={len(jobs)} logical_configs={len(plan)} contexts={len(contexts)}", flush=True)
    if args.dry_run:
        for job in jobs[:12]:
            print("plan", job)
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    records_path = output_dir / "records.jsonl"
    csv_path = output_dir / "records.csv"
    manifest_path.write_text(json.dumps(environment_manifest(args, contexts), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completed = load_completed(records_path) if args.resume else set()
    total = len(jobs)
    emitted = 0
    with records_path.open("a", encoding="utf-8") as handle:
        for index, (family, width, depth, seed, device, precision) in enumerate(jobs, start=1):
            spec = build_circuit_spec(family, width, depth, seed)
            record_id = f"{spec.circuit_id}:{device}:{precision}"
            if record_id in completed:
                print(f"[{index}/{total}] resume-skip {family} q={width} d={depth} {device}/{precision}", flush=True)
                continue
            print(f"[{index}/{total}] start {family} q={width} d={depth} {device}/{precision}", flush=True)
            record = measure_cell(spec, device, precision, args.warmup, args.repeats)
            record["record_id"] = record_id
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            emitted += 1
            target = record.get("warm_execution_median_seconds")
            print(f"[{index}/{total}] {record['status']} warm_median_s={target}", flush=True)
    to_csv(records_path, csv_path)
    manifest = environment_manifest(args, contexts)
    manifest.update(
        completed_utc=utc_now(),
        planned_jobs=total,
        newly_emitted_records=emitted,
        records_jsonl=str(records_path.name),
        records_csv=str(csv_path.name),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
