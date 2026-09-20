#!/usr/bin/env python3
"""Collect a structurally diverse, checkpointed dense-statevector runtime corpus.

This is a follow-up to v1, not a replacement for it.  It preserves the same
``prepared_execute_reset_state`` target but adds two random-topology families,
independent circuit seeds and pre-execution interaction-graph features.  A
row that raises CUDA out-of-memory is retained as ``resource_limit``; it is
never silently omitted or assigned a made-up runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "simulator_runtime_v1"))
import run_torch_statevector_matrix as v1  # noqa: E402
from features import static_features  # noqa: E402


ROOT = HERE.parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "simulator_runtime_v2" / "dense_statevector_structural_v2_20260920"
BASE_FAMILIES = set(v1.FAMILIES)
EXTRA_FAMILIES = {"random_matching", "random_star"}
ALL_FAMILIES = tuple((*v1.FAMILIES, "random_matching", "random_star"))
DEPTH_PROFILES = {
    "ghz": (1,),
    "hea": (2, 4),
    "qaoa_cycle": (2, 4),
    "random_brickwork": (4, 8),
    "qft": (1,),
    "random_matching": (4, 8),
    "random_star": (4, 8),
}
SCHEMA = "torch_dense_statevector_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def csv_list(raw: str, cast: type = str) -> list[Any]:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def _build_extra_spec(family: str, width: int, depth_parameter: int, seed: int) -> v1.CircuitSpec:
    """Build random-topology circuits with deterministic topology per seed."""
    gates: list[v1.GateSpec] = []
    histogram: dict[str, int] = {}
    depths = [0] * width
    rng = random.Random(seed)
    for layer in range(depth_parameter):
        for qubit in range(width):
            v1._append_gate(gates, depths, histogram, "ry", (qubit,), rng.uniform(-3.14159, 3.14159))
            v1._append_gate(gates, depths, histogram, "rz", (qubit,), rng.uniform(-3.14159, 3.14159))
        if family == "random_matching":
            wires = list(range(width))
            rng.shuffle(wires)
            for first, second in zip(wires[::2], wires[1::2]):
                v1._append_gate(gates, depths, histogram, "cx", (first, second))
        elif family == "random_star":
            hub = rng.randrange(width)
            targets = [wire for wire in range(width) if wire != hub]
            rng.shuffle(targets)
            for target in targets[: min(6, len(targets))]:
                v1._append_gate(gates, depths, histogram, "cx", (hub, target))
        else:  # pragma: no cover - guarded by caller
            raise ValueError(f"Unknown extra family: {family}")
    canonical = {
        "family": family,
        "num_qubits": width,
        "depth_parameter": depth_parameter,
        "seed": seed,
        "gates": [asdict(gate) for gate in gates],
    }
    circuit_id = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    two_qubit = sum(len(gate.qubits) == 2 for gate in gates)
    return v1.CircuitSpec(
        circuit_id=circuit_id,
        family=family,
        num_qubits=width,
        depth_parameter=depth_parameter,
        seed=seed,
        gates=tuple(gates),
        logical_depth=max(depths),
        logical_total_gate_count=len(gates),
        logical_two_qubit_gate_count=two_qubit,
        logical_single_qubit_gate_count=len(gates) - two_qubit,
        logical_gate_histogram=histogram,
    )


def build_spec(family: str, width: int, depth_parameter: int, seed: int) -> v1.CircuitSpec:
    if family in BASE_FAMILIES:
        return v1.build_circuit_spec(family, width, depth_parameter, seed)
    if family in EXTRA_FAMILIES:
        return _build_extra_spec(family, width, depth_parameter, seed)
    raise ValueError(f"Unknown family: {family}")


def make_plan(families: list[str], widths: list[int], seeds: list[int]) -> list[tuple[str, int, int, int]]:
    return [
        (family, width, depth, seed)
        for family in families
        for width in widths
        for seed in seeds
        for depth in DEPTH_PROFILES[family]
    ]


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(json.loads(line)["record_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def project_csv(jsonl_path: Path, csv_path: Path) -> None:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def device_metadata(device_name: str) -> dict[str, Any]:
    if device_name != "cuda" or not torch.cuda.is_available():
        return {"device_total_memory_bytes": None, "statevector_fits_raw_device_memory": None}
    total = int(torch.cuda.get_device_properties(0).total_memory)
    return {
        "device_total_memory_bytes": total,
        "statevector_fits_raw_device_memory": None,  # populated after static features are attached
    }


def measure(spec: v1.CircuitSpec, device_name: str, precision: str, warmup: int, repeats: int) -> dict[str, Any]:
    """Measure the v2 spec in three phases without v1's family reconstruction.

    The original runner rebuilds the logical circuit before its end-to-end
    timing.  That is correct for its five fixed families, but would turn a v2
    random topology into an error.  This local implementation keeps the same
    timing semantics while rebuilding through :func:`build_spec`.
    """
    record: dict[str, Any] = {
        "schema_version": SCHEMA,
        "recorded_utc": utc_now(),
        "simulator": "torch_dense_statevector_reference",
        "source_run_kind": "v2_structural_extension",
        "runtime_semantics": v1.TARGET_SEMANTICS,
        "cold_semantics": v1.COLD_SEMANTICS,
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
    }
    record.update(static_features(spec, precision))
    record.update(device_metadata(device_name))
    record["statevector_fits_raw_device_memory"] = (
        bool(record["statevector_bytes"] <= record["device_total_memory_bytes"])
        if record["device_total_memory_bytes"] is not None else None
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        record.update(status="unsupported", error="torch.cuda.is_available() is False")
        return record

    device = torch.device(device_name)
    program: tuple[v1.MaterializedGate, ...] | None = None
    e2e_program: tuple[v1.MaterializedGate, ...] | None = None
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            v1.synchronize(device)
        end_to_end_started = time.perf_counter()
        e2e_spec = build_spec(spec.family, spec.num_qubits, spec.depth_parameter, spec.seed)
        e2e_program = v1.materialize(e2e_spec, precision, device)
        e2e_elapsed, e2e_expectation = v1.run_once(e2e_program, e2e_spec, precision, device)
        v1.synchronize(device)
        record["end_to_end_first_seconds"] = time.perf_counter() - end_to_end_started
        record["end_to_end_execution_component_seconds"] = e2e_elapsed
        record["end_to_end_z0_expectation"] = e2e_expectation
        del e2e_program
        e2e_program = None

        preparation_started = time.perf_counter()
        program = v1.materialize(spec, precision, device)
        v1.synchronize(device)
        record["gate_materialization_seconds"] = time.perf_counter() - preparation_started
        cold_seconds, cold_expectation = v1.run_once(program, spec, precision, device)
        record["cold_post_prepare_seconds"] = cold_seconds
        record["cold_z0_expectation"] = cold_expectation
        for _ in range(warmup):
            v1.run_once(program, spec, precision, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            v1.synchronize(device)
        samples: list[float] = []
        expectation = cold_expectation
        for _ in range(repeats):
            elapsed, expectation = v1.run_once(program, spec, precision, device)
            samples.append(elapsed)
        record["warm_execution_seconds_samples"] = samples
        record["warm_execution_median_seconds"] = statistics.median(samples)
        record["warm_execution_mean_seconds"] = statistics.mean(samples)
        record["warm_execution_stdev_seconds"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        record["warm_z0_expectation"] = expectation
        record["torch_cuda_peak_allocated_bytes"] = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        )
        record["torch_cuda_peak_reserved_bytes"] = (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        )
        record["status"] = "ok"
    except Exception as exc:  # preserve failures as labels, never as a made-up runtime
        record.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=5),
        )
    finally:
        if program is not None:
            del program
        if e2e_program is not None:
            del e2e_program
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reserved = record.get("torch_cuda_peak_reserved_bytes")
    record["peak_reserved_over_statevector"] = (
        float(reserved) / float(record["statevector_bytes"])
        if reserved is not None and record["statevector_bytes"] else None
    )
    if record.get("status") == "error":
        error = str(record.get("error", "")).lower()
        if "out of memory" in error or "cuda error: memory" in error:
            record["status"] = "resource_limit"
            record["resource_limit_kind"] = "cuda_out_of_memory"
    return record


def initial_manifest(args: argparse.Namespace, contexts: list[tuple[str, str]]) -> dict[str, Any]:
    cuda_properties = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "schema_version": SCHEMA,
        "target_semantics": v1.TARGET_SEMANTICS,
        "cold_semantics": v1.COLD_SEMANTICS,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "hardware": {
            "cpu_count": os.cpu_count(),
            "torch_cpu_threads": torch.get_num_threads(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_name": cuda_properties.name if cuda_properties else None,
            "cuda_total_bytes": int(cuda_properties.total_memory) if cuda_properties else None,
        },
        "contexts_requested": [f"{device}:{precision}" for device, precision in contexts],
        "known_boundaries": [
            "The target is a PyTorch dense-statevector reference kernel, not Aer, CUDA-Q, cuTensorNet or QPU time.",
            "Runtime and resource-limit labels are separate outcomes; a failed allocation has no invented runtime.",
            "No cross-GPU conclusion is possible from this one-GPU machine.",
        ],
        "invocations": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--families", default="random_matching,random_star")
    parser.add_argument("--widths", default="16,20,24")
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--contexts", default="cpu:complex64,cpu:complex128,cuda:complex64,cuda:complex128")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    families = csv_list(args.families)
    unknown = sorted(set(families) - set(ALL_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown families: {unknown}")
    widths = csv_list(args.widths, int)
    seeds = csv_list(args.seeds, int)
    contexts = v1.parse_contexts(args.contexts)
    if not families or not widths or not seeds:
        raise ValueError("families, widths and seeds must be non-empty")
    if min(widths) < 2 or args.warmup < 0 or args.repeats < 1:
        raise ValueError("invalid width/warmup/repeats argument")
    torch.set_num_threads(max(1, args.cpu_threads))
    torch.set_num_interop_threads(1)

    plan = make_plan(families, widths, seeds)
    jobs = [(*item, device, precision) for item in plan for device, precision in contexts]
    if args.max_jobs:
        jobs = jobs[: args.max_jobs]
    print(f"planned_jobs={len(jobs)} logical_configs={len(plan)} contexts={len(contexts)}", flush=True)
    if args.dry_run:
        for job in jobs[:20]:
            print("plan", job)
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    csv_path = output_dir / "records.csv"
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else initial_manifest(args, contexts)
    completed = load_completed(records_path) if args.resume else set()
    emitted = 0
    with records_path.open("a", encoding="utf-8") as handle:
        for index, (family, width, depth, seed, device, precision) in enumerate(jobs, start=1):
            spec = build_spec(family, width, depth, seed)
            record_id = f"{spec.circuit_id}:{device}:{precision}"
            if record_id in completed:
                print(f"[{index}/{len(jobs)}] resume-skip {family} q={width} d={depth} seed={seed} {device}/{precision}", flush=True)
                continue
            print(f"[{index}/{len(jobs)}] start {family} q={width} d={depth} seed={seed} {device}/{precision}", flush=True)
            record = measure(spec, device, precision, args.warmup, args.repeats)
            record["record_id"] = record_id
            record["recorded_utc"] = utc_now()
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            emitted += 1
            print(f"[{index}/{len(jobs)}] {record['status']} warm_median_s={record.get('warm_execution_median_seconds')}", flush=True)
    project_csv(records_path, csv_path)
    manifest["invocations"].append({
        "completed_utc": utc_now(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "planned_jobs": len(jobs),
        "newly_emitted_records": emitted,
    })
    manifest["records_jsonl"] = records_path.name
    manifest["records_csv"] = csv_path.name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
