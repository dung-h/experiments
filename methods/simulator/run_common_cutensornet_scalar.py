#!/usr/bin/env python3
"""Measure cuTensorNet scalar-amplitude clocks on the frozen 204-QASM panel.

Each cell creates a new process and retains network construction, contraction
path search, NVIDIA's ``RUNTIME_EST``, first contraction, and warm
contractions as separate observations.  This is a local GPU scalar-amplitude
target, not full-state sampling and not an archived-QPU-time estimator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from gpu_lease import acquire_gpu_lease
except ModuleNotFoundError:  # imported by the contract tests via a file spec
    from benchmark_v1.scripts.gpu_lease import acquire_gpu_lease


ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = ROOT.parent
WORKSPACE_BYTES = 14_947_909_632
OPTIMIZER_SAMPLES = 32
OPTIMIZER_SEED = 137
OPTIMIZER_THREADS = 1
PLANNER_TIMEOUT_SECONDS = 1_800
CONTRACTION_TIMEOUT_SECONDS = 900


def sha_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def sha_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else DOCUMENTS / path


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"


def environment() -> dict[str, Any]:
    packages = command_output([sys.executable, "-m", "pip", "freeze", "--all"])
    result = {
        "python": sys.version, "platform": platform.platform(), "interpreter": sys.executable,
        "package_lock": packages, "package_lock_sha256": hashlib.sha256(packages.encode()).hexdigest(),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "workspace_bytes": WORKSPACE_BYTES, "workspace_fraction": 0.9,
        "optimizer": {"samples": OPTIMIZER_SAMPLES, "seed": OPTIMIZER_SEED, "threads": OPTIMIZER_THREADS, "cost_function": "TIME_TUNED"},
        "gpu_concurrency": 1,
    }
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        result["cupy"] = {"runtime_api": int(cp.cuda.runtime.runtimeGetVersion()), "driver_api": int(cp.cuda.runtime.driverGetVersion()), "name": props["name"].decode(), "vram_bytes": int(props["totalGlobalMem"])}
    except Exception as exc:
        result["cupy_error"] = f"{type(exc).__name__}:{exc}"
    return result


def strict_unitary(qasm: Path, expected_hash: str, expected_width: int) -> tuple[Any, str, int]:
    from qiskit import QuantumCircuit, qasm2

    if sha_file(qasm) != expected_hash:
        raise ValueError("QASM hash differs from immutable panel manifest")
    circuit = QuantumCircuit.from_qasm_file(str(qasm))
    if circuit.num_qubits != expected_width:
        raise ValueError("QASM width differs from immutable panel manifest")
    terminal_measurements = int(circuit.count_ops().get("measure", 0))
    unitary = circuit.remove_final_measurements(inplace=False)
    if unitary.count_ops().get("measure", 0):
        raise ValueError("nonterminal measurement: no scalar-amplitude adapter is declared")
    if unitary.count_ops().get("reset", 0):
        raise ValueError("reset: no scalar-amplitude unitary adapter is declared")
    return unitary, sha_json({
        "adapter": "cuquantum.CircuitToEinsum",
        "mode": "scalar_amplitude_strip_terminal_measurements_only",
        "canonical_qasm": qasm2.dumps(unitary),
    }), terminal_measurements


def optimizer_value(cutn: Any, network: Any, attribute: Any, np: Any) -> Any:
    """Read a scalar optimizer attribute across the cuTensorNet 2.13/2.14 API.

    ``Network.contract_path`` returns a public ``OptimizerInfo`` object, while
    scalar attributes such as ``RUNTIME_EST`` and ``EFFECTIVE_FLOPS_EST`` are
    exposed through the low-level optimizer-info handle.  The old runner used
    the removed ``network.optimizer_info`` property and silently became an
    adapter error on the fresh 26.6 environment.
    """
    dtype = cutn.contraction_optimizer_info_get_attribute_dtype(attribute)
    value = np.zeros((1,), dtype=dtype)
    cutn.contraction_optimizer_info_get_attribute(
        network.handle,
        network.optimizer_info_ptr,
        attribute,
        value.ctypes.data,
        value.dtype.itemsize,
    )
    return value.item()


def timed_contraction(callback: Any) -> Any:
    """Bound one contraction call on the Linux measurement worker.

    The parent process separately bounds plan search.  Keeping the contraction
    timeout around each call prevents a stalled CUDA call from consuming the
    entire checkpoint while preserving a terminal timeout row.
    """
    previous = signal.getsignal(signal.SIGALRM)

    def alarm(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"contraction exceeded {CONTRACTION_TIMEOUT_SECONDS}s")

    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, CONTRACTION_TIMEOUT_SECONDS)
    try:
        return callback()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def child(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "status": "ok", "error": "", "timeout_stage": "", "ir_sha256": "", "terminal_measurements_stripped": None,
        "network_build_s": None, "path_search_s": None, "runtime_est_s": None,
        "effective_flops_est": None, "flop_count": None, "largest_intermediate_elements": None,
        "num_slices": None, "first_contract_s": None, "warm_contract_s": [],
    }
    network = None
    try:
        import cupy as cp
        import numpy as np
        from cuquantum import tensornet as tn
        from cuquantum.bindings import cutensornet as cutn

        unitary, payload["ir_sha256"], payload["terminal_measurements_stripped"] = strict_unitary(Path(args.qasm_path), args.qasm_sha256, args.width_qubits)
        started = time.perf_counter()
        converter = tn.CircuitToEinsum(unitary, dtype="complex64", backend="cupy")
        expression, operands = converter.amplitude("0" * unitary.num_qubits)
        network = tn.Network(expression, *operands, options=tn.NetworkOptions(memory_limit=WORKSPACE_BYTES, blocking="auto"))
        payload["network_build_s"] = time.perf_counter() - started
        props = cp.cuda.runtime.getDeviceProperties(0)
        optimizer = tn.OptimizerOptions(samples=OPTIMIZER_SAMPLES, seed=OPTIMIZER_SEED, threads=OPTIMIZER_THREADS, cost_function=cutn.OptimizerCost.TIME_TUNED, gpu_arch=int(props["major"]) * 10 + int(props["minor"]))
        started = time.perf_counter(); _path, optimizer_info = network.contract_path(optimizer)
        payload["path_search_s"] = time.perf_counter() - started
        info = cutn.ContractionOptimizerInfoAttribute
        payload["runtime_est_s"] = optimizer_value(cutn, network, info.RUNTIME_EST, np)
        payload["effective_flops_est"] = optimizer_value(cutn, network, info.EFFECTIVE_FLOPS_EST, np)
        payload["flop_count"] = optimizer_value(cutn, network, info.FLOP_COUNT, np)
        payload["largest_intermediate_elements"] = float(optimizer_info.largest_intermediate)
        payload["num_slices"] = int(optimizer_info.num_slices)
        def first_contract() -> float:
            start, end = cp.cuda.Event(), cp.cuda.Event(); start.record(); network.contract(); end.record(); end.synchronize()
            return cp.cuda.get_elapsed_time(start, end) / 1000.0

        payload["first_contract_s"] = timed_contraction(first_contract)
        for _ in range(args.warmups):
            timed_contraction(network.contract)
        cp.cuda.get_current_stream().synchronize()
        for _ in range(args.repetitions):
            def warm_contract() -> float:
                start, end = cp.cuda.Event(), cp.cuda.Event(); start.record(); network.contract(); end.record(); end.synchronize()
                return cp.cuda.get_elapsed_time(start, end) / 1000.0

            payload["warm_contract_s"].append(timed_contraction(warm_contract))
    except MemoryError as exc:
        payload.update(status="resource_limit", error=f"MemoryError:{exc}")
    except TimeoutError as exc:
        payload.update(status="timeout", timeout_stage="contraction", error=str(exc))
    except Exception as exc:
        detail = f"{type(exc).__name__}:{exc}".splitlines()[0][:1000]
        lowered = detail.lower()
        status = "resource_limit" if any(token in lowered for token in ("out of memory", "memory allocation", "cuda error 2")) else "unsupported" if "unsupported" in lowered else "adapter_error"
        payload.update(status=status, error=detail)
    finally:
        if network is not None:
            try:
                network.free()
            except Exception:
                pass
        try:
            cp.get_default_memory_pool().free_all_blocks()  # type: ignore[name-defined]
        except Exception:
            pass
    Path(args.worker_output).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return 0


def make_rows(item: dict[str, str], session_id: str, payload: dict[str, Any], repetitions: int) -> list[dict[str, Any]]:
    base = {
        "panel_id": item["panel_id"], "panel_member_id": item["panel_member_id"], "basename": item["basename"], "qasm_sha256": item["qasm_sha256"],
        "family": item["family"], "width_qubits": item["width_qubits"], "stratum": item["stratum"],
        "engine_id": "cutensornet_scalar", "method_id": "cutensornet_scalar_local", "target_id": "cutensornet_scalar_amplitude_gpu_clock",
        "precision": "complex64", "shots": "", "session_id": session_id, "status": payload.get("status", "adapter_error"), "timeout_stage": payload.get("timeout_stage", ""), "error": payload.get("error", ""),
        "ir_sha256": payload.get("ir_sha256", ""), "terminal_measurements_stripped": payload.get("terminal_measurements_stripped"),
        "workspace_bytes": WORKSPACE_BYTES, "optimizer_samples": OPTIMIZER_SAMPLES, "optimizer_seed": OPTIMIZER_SEED, "optimizer_threads": OPTIMIZER_THREADS,
        "t_network_build_s": payload.get("network_build_s"), "t_path_search_s": payload.get("path_search_s"), "runtime_est_s": payload.get("runtime_est_s"),
        "effective_flops_est": payload.get("effective_flops_est"), "flop_count": payload.get("flop_count"), "largest_intermediate_elements": payload.get("largest_intermediate_elements"), "num_slices": payload.get("num_slices"),
    }
    first = dict(base); first.update({"observation_kind": "first", "repetition": 0, "t_first_contract_s": payload.get("first_contract_s"), "t_warm_contract_s": "", "t_end_to_end_s": ""})
    rows = [first]
    values = payload.get("warm_contract_s", [])
    for repetition in range(repetitions):
        warm = dict(base); warm.update({"observation_kind": "warm", "repetition": repetition, "t_first_contract_s": "", "t_warm_contract_s": values[repetition] if repetition < len(values) else "", "t_end_to_end_s": ""})
        rows.append(warm)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--qasm-path", type=Path); parser.add_argument("--qasm-sha256"); parser.add_argument("--width-qubits", type=int); parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--warmups", type=int, default=3); parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--panel-manifest", type=Path); parser.add_argument("--environment-lock", type=Path, default=ROOT / "benchmark_v1/execution/manifests/simulator_environment_lock_v1.json")
    parser.add_argument("--output-dir", type=Path); parser.add_argument("--sessions", type=int, default=3); parser.add_argument("--cell-timeout-seconds", type=int, default=PLANNER_TIMEOUT_SECONDS); parser.add_argument("--limit-circuits", type=int); parser.add_argument("--selection-manifest", type=Path, help="immutable JSON list of exact panel_member_id values for a checkpoint")
    parser.add_argument("--resume", action="store_true", help="resume only from this runner's append-only attempt JSONL")
    args = parser.parse_args()
    if args.worker:
        if any(value is None for value in (args.qasm_path, args.qasm_sha256, args.width_qubits, args.worker_output)):
            raise SystemExit("worker requires QASM identity, width and output")
        raise SystemExit(child(args))
    acquire_gpu_lease("run_common_cutensornet_scalar")
    if args.panel_manifest is None or args.output_dir is None:
        raise SystemExit("parent requires --panel-manifest and --output-dir")
    if (args.sessions, args.warmups, args.repetitions) != (3, 3, 5):
        raise SystemExit("sessions=3, warmups=3 and repetitions=5 are frozen by S38")
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite output directory: {output}")
    with args.panel_manifest.open(newline="", encoding="utf-8") as handle:
        panel = sorted(csv.DictReader(handle), key=lambda row: row["basename"])
    if len(panel) != 204:
        raise SystemExit(f"expected 204 immutable panel rows, got {len(panel)}")
    if args.limit_circuits is not None and args.selection_manifest is not None:
        raise SystemExit("use either --limit-circuits or --selection-manifest, not both")
    selection_record: dict[str, Any] | None = None
    if args.selection_manifest is not None:
        selection_path = args.selection_manifest.resolve()
        selection_record = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_record.get("panel_manifest_sha256") != sha_file(args.panel_manifest.resolve()):
            raise SystemExit("checkpoint selection was created for a different panel manifest")
        selected_ids = selection_record.get("selected_panel_member_ids")
        if not isinstance(selected_ids, list) or not selected_ids or len(selected_ids) != len(set(selected_ids)):
            raise SystemExit("selection manifest must contain a non-empty unique selected_panel_member_ids list")
        panel_by_id = {row["panel_member_id"]: row for row in panel}
        missing = sorted(set(selected_ids) - set(panel_by_id))
        if missing:
            raise SystemExit(f"selection manifest contains panel members absent from frozen panel: {missing}")
        selected = [panel_by_id[member_id] for member_id in selected_ids]
    else:
        selected = panel if args.limit_circuits is None else panel[:args.limit_circuits]
    if not selected:
        raise SystemExit("selected panel is empty")
    attempts_path = output / "attempt_records.jsonl"
    if args.resume:
        if not attempts_path.is_file():
            raise SystemExit("--resume requires an existing append-only attempt_records.jsonl")
    else:
        output.mkdir(parents=True); attempts_path.touch()
    workers = output / "worker_payloads"; workers.mkdir(exist_ok=True)
    environment_path = output / "environment.json"
    if not args.resume:
        environment_path.write_text(json.dumps(environment(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    existing_cells: set[tuple[str, str]] = set()
    for line in attempts_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            previous = json.loads(line)
            if previous.get("observation_kind") == "first":
                existing_cells.add((str(previous["panel_member_id"]), str(previous["session_id"])))
    for item in selected:
        for session in range(1, args.sessions + 1):
            session_id = f"session-{session}"
            if (item["panel_member_id"], session_id) in existing_cells:
                continue
            payload_path = workers / f"{item['basename']}.complex64.{session_id}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--qasm-path", str(source_path(item["source_path"])), "--qasm-sha256", item["qasm_sha256"], "--width-qubits", item["width_qubits"], "--warmups", str(args.warmups), "--repetitions", str(args.repetitions), "--worker-output", str(payload_path)]
            try:
                process = subprocess.run(command, text=True, capture_output=True, timeout=args.cell_timeout_seconds, check=False)
                payload = json.loads(payload_path.read_text()) if payload_path.exists() else {"status": "adapter_error", "error": f"child exit={process.returncode}; {process.stderr[-600:]}"}
            except subprocess.TimeoutExpired:
                payload = {"status": "planner_timeout", "timeout_stage": "planner", "error": f"cell exceeded {args.cell_timeout_seconds}s"}
            rows = make_rows(item, session_id, payload, args.repetitions)
            with attempts_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
    records = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit("no completed attempt records")
    raw_path = output / "cutensornet_scalar_common_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    manifest = {
        "artifact_id": "cutensornet-scalar-common-panel-v1-20260928", "created_utc": datetime.now(timezone.utc).isoformat(), "timing_performed": True,
        "panel_rows_frozen": len(panel), "panel_rows_attempted": len(selected), "checkpoint_only": args.limit_circuits is not None or args.selection_manifest is not None, "sessions": 3, "warmups": 3, "warm_repetitions": 5,
        "workspace": {"bytes": WORKSPACE_BYTES, "fraction_of_detected_vram": 0.9}, "optimizer": {"cost_function": "TIME_TUNED", "samples": OPTIMIZER_SAMPLES, "seed": OPTIMIZER_SEED, "threads": OPTIMIZER_THREADS},
        "clock_contract": "network build, path search, RUNTIME_EST, first CUDA-event contraction and warm CUDA-event contractions are separate; scalar zero-bitstring amplitude only",
        "concurrency": "one sequential child process and one GPU timing worker",
        "panel_manifest": {"path": str(args.panel_manifest.resolve().relative_to(ROOT)), "sha256": sha_file(args.panel_manifest.resolve())}, "environment_lock": {"path": str(args.environment_lock.resolve().relative_to(ROOT)), "sha256": sha_file(args.environment_lock.resolve())},
        "selection_manifest": ({"path": str(args.selection_manifest.resolve().relative_to(ROOT)), "sha256": sha_file(args.selection_manifest.resolve()), "selection_id": selection_record.get("selection_id"), "selected_panel_member_ids": selection_record.get("selected_panel_member_ids")} if args.selection_manifest is not None else None),
        "timeouts": {"planner_timeout_seconds": PLANNER_TIMEOUT_SECONDS, "contraction_timeout_seconds": CONTRACTION_TIMEOUT_SECONDS, "parent_cell_timeout_seconds": args.cell_timeout_seconds},
        "runner_sha256": sha_file(Path(__file__)), "raw_status_counts": dict(Counter(str(row["status"]) for row in records)),
        "outputs": {"attempt_records": {"path": attempts_path.name, "sha256": sha_file(attempts_path)}, "raw_csv": {"path": raw_path.name, "sha256": sha_file(raw_path)}, "environment": {"path": environment_path.name, "sha256": sha_file(environment_path)}},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "status_counts": manifest["raw_status_counts"], "checkpoint_only": manifest["checkpoint_only"]}, sort_keys=True))


if __name__ == "__main__":
    main()
