#!/usr/bin/env python3
"""Measure CUDA-Q dense ``sample`` clocks on ``sim_common_q16_v1``.

The parent is deliberately sequential. Each QASM/precision/session runs in a
fresh child process: CUDA-Q target initialization is process-global, so this
is the only defensible interpretation of the recorded first call.  This
runner is not an Aer benchmark and does not emit any QPU-time claim.

The script is intentionally inactive until called from a fresh S38-pinned
environment with Qiskit and CUDA-Q installed.  ``--limit-circuits`` exists for
the predeclared checkpoint only; it never changes the immutable panel list.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
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
PRECISIONS = ("fp32", "fp64")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_recovery_selection(path: Path, panel: list[dict[str, str]], panel_sha256: str) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    """Resolve an R5 selection without weakening the canonical 204-row panel pin."""
    selection = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict) or selection.get("artifact_id") != "cudaq-dense-recovery-preflight-v1" or selection.get("status") != "PASS":
        raise ValueError("recovery manifest is not a passing frozen R5 preflight")
    unsigned = {key: value for key, value in selection.items() if key != "plan_sha256"}
    if sha_json(unsigned) != selection.get("plan_sha256"):
        raise ValueError("R5 preflight self-hash mismatch")
    if selection.get("canonical_panel", {}).get("sha256") != panel_sha256 or selection.get("canonical_panel", {}).get("rows") != 204:
        raise ValueError("R5 preflight does not pin the canonical 204-member panel")
    rows_by_id = {row["panel_member_id"]: row for row in panel}
    selected = selection.get("selection")
    if not isinstance(selected, list) or not selected:
        raise ValueError("R5 selection must contain panel members")
    member_ids = [str(row.get("panel_member_id", "")) for row in selected]
    if len(member_ids) != len(set(member_ids)) or any(member_id not in rows_by_id for member_id in member_ids):
        raise ValueError("R5 selection has duplicate or unknown panel-member IDs")
    for item in selected:
        source = rows_by_id[item["panel_member_id"]]
        for field in ("qasm_sha256", "basename", "family"):
            if str(item.get(field, "")) != str(source.get(field, "")):
                raise ValueError(f"R5 selection {field} mismatch for {item['panel_member_id']}")
    counts = selection.get("selection_counts", {})
    if counts.get("selected_panel_members") != len(selected) or counts.get("expected_cells") != len(selected) * len(selection.get("precisions", [])) * len(selection.get("sessions", [])):
        raise ValueError("R5 selection counts do not match selected identities")
    precisions = selection.get("precisions")
    if not isinstance(precisions, list) or not precisions or len(set(precisions)) != len(precisions) or not set(precisions) <= set(PRECISIONS):
        raise ValueError("R5 precision selection is invalid")
    if selection.get("sessions") != ["session-1", "session-2", "session-3"] or selection.get("shots") != 32 or selection.get("warmups") != 3 or selection.get("warm_repetitions") != 5 or selection.get("cell_timeout_seconds") != 900:
        raise ValueError("R5 sampling/timeout context differs from the frozen contract")
    ordered = sorted((rows_by_id[member_id] for member_id in member_ids), key=lambda row: row["basename"])
    return selection, ordered, list(precisions)


def source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else DOCUMENTS / path


def canonical_ir_hash(circuit: Any) -> str:
    try:
        from cudaq_qiskit_adapter import normalize_qiskit_circuit
    except ModuleNotFoundError:  # imported through a package path in tests
        from benchmark_v1.scripts.cudaq_qiskit_adapter import normalize_qiskit_circuit

    normalized = normalize_qiskit_circuit(circuit)
    return normalized.legacy_ir_sha256("dense_sample_preserve_terminal_measurements")


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"


def capture_package_lock() -> tuple[str, str]:
    """Capture a lock even when the verified historical venv has no pip."""
    pip_output = command_output([sys.executable, "-m", "pip", "freeze", "--all"])
    if not pip_output.startswith("unavailable:"):
        return pip_output, "python -m pip freeze --all"
    uv_output = command_output(["uv", "--quiet", "pip", "freeze", "--python", sys.executable])
    if not uv_output.startswith("unavailable:"):
        return uv_output, "uv pip freeze --python"
    return uv_output, "none"


def environment(args: argparse.Namespace) -> dict[str, Any]:
    package_lock, package_lock_source = capture_package_lock()
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "interpreter": sys.executable,
        "package_lock": package_lock,
        "package_lock_sha256": hashlib.sha256(package_lock.encode()).hexdigest(),
        "package_lock_source": package_lock_source,
        "cuda_path": os.environ.get("CUDA_PATH", ""),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "qiskit_site_packages": str(args.qiskit_site_packages.resolve()) if args.qiskit_site_packages else "in_process_environment",
    }
    try:
        import cudaq
        env["cudaq_version"] = str(cudaq.__version__)
    except Exception as exc:
        env["cudaq_version"] = f"unavailable:{type(exc).__name__}"
    if args.qiskit_site_packages:
        try:
            qiskit_path = args.qiskit_site_packages.resolve()
            if str(qiskit_path) not in sys.path:
                sys.path.insert(0, str(qiskit_path))
            import qiskit
            env["qiskit_version"] = str(qiskit.__version__)
            env["qiskit_file"] = str(Path(qiskit.__file__).resolve())
        except Exception as exc:
            env["qiskit_error"] = f"{type(exc).__name__}:{exc}"
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        env["cupy"] = {
            "runtime_api": int(cp.cuda.runtime.runtimeGetVersion()),
            "driver_api": int(cp.cuda.runtime.driverGetVersion()),
            "device_count": int(cp.cuda.runtime.getDeviceCount()),
            "device_name": props["name"].decode(),
            "compute_capability": f"{props['major']}.{props['minor']}",
            "vram_bytes": int(props["totalGlobalMem"]),
        }
    except Exception as exc:
        env["cupy_error"] = f"{type(exc).__name__}:{exc}"
    return env


def import_qiskit(args: argparse.Namespace) -> Any:
    """Import Qiskit after CUDA-Q, optionally from a sibling site-packages.

    The verified CUDA-Q 0.15.1 environment on this host predates the pinned
    Qiskit package.  Importing CUDA-Q first is important: it preserves the
    verified target discovery, while the parser is loaded from the immutable
    sibling environment without putting a second CUDA-Q package on sys.path.
    """
    if args.qiskit_site_packages:
        path = Path(args.qiskit_site_packages).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Qiskit site-packages not found: {path}")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from qiskit import QuantumCircuit  # type: ignore
    return QuantumCircuit


def child(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "status": "ok", "error": "", "build_seconds": None,
        "first_seconds": None, "first_extract_seconds": None,
        "warm_seconds": [], "warm_extract_seconds": [], "ir_sha256": "",
        "normalized_ir_sha256": "", "normalized_qasm_sha256": "",
        "unitary_qasm_sha256": "", "operation_counts": {},
        "rccx_lowered_count": 0, "rccx_lowering_noop": True,
    }
    try:
        import cudaq
        QuantumCircuit = import_qiskit(args)
        from cudaq.contrib.qiskit_convert import from_qiskit
        try:
            from cudaq_qiskit_adapter import normalize_qiskit_circuit
        except ModuleNotFoundError:  # imported through a package path in tests
            from benchmark_v1.scripts.cudaq_qiskit_adapter import normalize_qiskit_circuit

        qasm = Path(args.qasm_path)
        if sha_file(qasm) != args.qasm_sha256:
            raise ValueError("QASM hash differs from immutable panel manifest")
        circuit = QuantumCircuit.from_qasm_file(str(qasm))
        if circuit.num_qubits != args.width_qubits:
            raise ValueError("QASM width differs from immutable panel manifest")
        build_started = time.perf_counter()
        normalized = normalize_qiskit_circuit(circuit)
        kernel = from_qiskit(normalized.circuit)
        payload["build_seconds"] = time.perf_counter() - build_started
        payload.update({
            "ir_sha256": normalized.legacy_ir_sha256("dense_sample_preserve_terminal_measurements"),
            "normalized_ir_sha256": normalized.normalized_ir_sha256,
            "normalized_qasm_sha256": normalized.normalized_qasm_sha256,
            "unitary_qasm_sha256": normalized.unitary_qasm_sha256,
            "operation_counts": normalized.operation_counts,
            "rccx_lowered_count": normalized.rccx_lowered_count,
            "rccx_lowering_noop": normalized.rccx_lowering_noop,
        })
        cudaq.set_target("nvidia", option=args.precision)
        started = time.perf_counter()
        counts = cudaq.sample(kernel, shots_count=args.shots)
        payload["first_seconds"] = time.perf_counter() - started
        extract_started = time.perf_counter()
        observed = int(sum(counts.values()))
        payload["first_extract_seconds"] = time.perf_counter() - extract_started
        if observed != args.shots:
            raise RuntimeError(f"first sample returned {observed} instead of {args.shots} shots")
        for _ in range(args.warmups):
            counts = cudaq.sample(kernel, shots_count=args.shots)
            if int(sum(counts.values())) != args.shots:
                raise RuntimeError("warmup shot count mismatch")
        for _ in range(args.repetitions):
            started = time.perf_counter()
            counts = cudaq.sample(kernel, shots_count=args.shots)
            payload["warm_seconds"].append(time.perf_counter() - started)
            extract_started = time.perf_counter()
            observed = int(sum(counts.values()))
            payload["warm_extract_seconds"].append(time.perf_counter() - extract_started)
            if observed != args.shots:
                raise RuntimeError(f"warm sample returned {observed} instead of {args.shots} shots")
    except MemoryError as exc:
        payload.update(status="resource_limit", error=f"MemoryError:{exc}")
    except Exception as exc:
        detail = f"{type(exc).__name__}:{exc}".splitlines()[0][:1000]
        lowered = detail.lower()
        status = "resource_limit" if any(token in lowered for token in ("out of memory", "memory allocation", "cuda error 2")) else "unsupported" if "unsupported" in lowered else "adapter_error"
        payload.update(status=status, error=detail)
    Path(args.worker_output).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return 0


def make_rows(item: dict[str, str], precision: str, session_id: str, payload: dict[str, Any], repetitions: int, shots: int) -> list[dict[str, Any]]:
    base = {
        "panel_id": item["panel_id"], "panel_member_id": item["panel_member_id"],
        "basename": item["basename"], "qasm_sha256": item["qasm_sha256"],
        "family": item["family"], "width_qubits": item["width_qubits"],
        "stratum": item["stratum"], "engine_id": "cudaq_dense",
        "method_id": "cudaq_dense_local", "target_id": "cudaq_dense_sample_wall_clock",
        "precision": precision, "shots": shots, "session_id": session_id,
        "status": payload.get("status", "adapter_error"), "error": payload.get("error", ""),
        "ir_sha256": payload.get("ir_sha256", ""),
        "normalized_ir_sha256": payload.get("normalized_ir_sha256", ""),
        "normalized_qasm_sha256": payload.get("normalized_qasm_sha256", ""),
        "unitary_qasm_sha256": payload.get("unitary_qasm_sha256", ""),
        "operation_counts": payload.get("operation_counts", {}),
        "rccx_lowered_count": payload.get("rccx_lowered_count", 0),
        "rccx_lowering_noop": payload.get("rccx_lowering_noop", True),
        "t_build_s": payload.get("build_seconds"),
        "warmups": payload.get("warmups", "frozen_manifest"),
    }
    first = dict(base)
    first.update({
        "observation_kind": "first", "repetition": 0,
        "t_first_execution_s": payload.get("first_seconds"),
        "t_warm_execution_s": "", "t_result_extraction_s": payload.get("first_extract_seconds"),
        "t_end_to_end_s": (float(payload["build_seconds"]) + float(payload["first_seconds"]) + float(payload["first_extract_seconds"])) if payload.get("status") == "ok" else "",
    })
    rows = [first]
    values = payload.get("warm_seconds", [])
    extracts = payload.get("warm_extract_seconds", [])
    for rep in range(repetitions):
        row = dict(base)
        elapsed = values[rep] if rep < len(values) else ""
        extraction = extracts[rep] if rep < len(extracts) else ""
        row.update({
            "observation_kind": "warm", "repetition": rep,
            "t_first_execution_s": "", "t_warm_execution_s": elapsed,
            "t_result_extraction_s": extraction,
            "t_end_to_end_s": (float(elapsed) + float(extraction)) if elapsed != "" and extraction != "" else "",
        })
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--qasm-path", type=Path)
    parser.add_argument("--qasm-sha256")
    parser.add_argument("--width-qubits", type=int)
    parser.add_argument("--precision", choices=PRECISIONS)
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--qiskit-site-packages", type=Path,
                        help="Optional sibling site-packages to import Qiskit from after CUDA-Q initialization")
    parser.add_argument("--bridge-manifest", type=Path,
                        help="Optional immutable environment-decision manifest for a sibling Qiskit bridge")
    parser.add_argument("--panel-manifest", type=Path)
    parser.add_argument("--environment-lock", type=Path, default=ROOT / "benchmark_v1/execution/manifests/simulator_environment_lock_v1.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--cell-timeout-seconds", type=int, default=900)
    parser.add_argument("--limit-circuits", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="resume only from this runner's append-only attempt JSONL")
    args = parser.parse_args()
    if args.worker:
        required = (args.qasm_path, args.qasm_sha256, args.width_qubits, args.precision, args.worker_output)
        if any(value is None for value in required):
            raise SystemExit("worker requires qasm identity, precision and output")
        raise SystemExit(child(args))
    acquire_gpu_lease("run_common_cudaq_dense")
    if args.output_dir is None or args.panel_manifest is None:
        raise SystemExit("parent requires --panel-manifest and --output-dir")
    if args.sessions != 3 or args.warmups != 3 or args.repetitions != 5 or args.shots != 32:
        raise SystemExit("sessions=3, warmups=3, repetitions=5 and shots=32 are frozen by S38")
    out = args.output_dir.resolve()
    if out.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite output directory: {out}")
    with args.panel_manifest.open(newline="", encoding="utf-8") as handle:
        panel = sorted(csv.DictReader(handle), key=lambda row: row["basename"])
    if len(panel) != 204:
        raise SystemExit(f"expected 204 immutable panel rows, got {len(panel)}")
    selected = panel if args.limit_circuits is None else panel[:args.limit_circuits]
    if not selected:
        raise SystemExit("selected panel is empty")
    attempts_path = out / "attempt_records.jsonl"
    if args.resume:
        if not attempts_path.is_file():
            raise SystemExit("--resume requires an existing append-only attempt_records.jsonl")
    else:
        out.mkdir(parents=True)
        attempts_path.touch()
    workers = out / "worker_payloads"; workers.mkdir(exist_ok=True)
    environment_path = out / "environment.json"
    if not args.resume:
        environment_path.write_text(json.dumps(environment(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    existing_cells: set[tuple[str, str, str]] = set()
    for line in attempts_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            previous = json.loads(line)
            if previous.get("observation_kind") == "first":
                existing_cells.add((str(previous["panel_member_id"]), str(previous["precision"]), str(previous["session_id"])))
    for item in selected:
        qasm = source_path(item["source_path"])
        for precision in PRECISIONS:
            for session in range(1, args.sessions + 1):
                session_id = f"session-{session}"
                if (item["panel_member_id"], precision, session_id) in existing_cells:
                    continue
                payload_path = workers / f"{item['basename']}.{precision}.{session_id}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--qasm-path", str(qasm), "--qasm-sha256", item["qasm_sha256"], "--width-qubits", item["width_qubits"], "--precision", precision, "--shots", str(args.shots), "--warmups", str(args.warmups), "--repetitions", str(args.repetitions), "--worker-output", str(payload_path)]
                if args.qiskit_site_packages:
                    command.extend(["--qiskit-site-packages", str(args.qiskit_site_packages.resolve())])
                if args.bridge_manifest:
                    command.extend(["--bridge-manifest", str(args.bridge_manifest.resolve())])
                try:
                    complete = subprocess.run(command, text=True, capture_output=True, timeout=args.cell_timeout_seconds, check=False)
                    payload = json.loads(payload_path.read_text()) if payload_path.exists() else {"status": "adapter_error", "error": f"child exit={complete.returncode}; {complete.stderr[-600:]}"}
                except subprocess.TimeoutExpired:
                    payload = {"status": "timeout", "error": f"cell exceeded {args.cell_timeout_seconds}s"}
                rows = make_rows(item, precision, session_id, payload, args.repetitions, args.shots)
                with attempts_path.open("a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
    records = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit("no completed attempt records")
    fields = list(records[0])
    raw_path = out / "cudaq_dense_common_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    manifest = {
        "artifact_id": "cudaq-dense-common-panel-v1-20260928",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "timing_performed": True,
        "panel_rows_frozen": len(panel), "panel_rows_attempted": len(selected),
        "checkpoint_only": args.limit_circuits is not None,
        "sessions": args.sessions, "warmups": args.warmups, "warm_repetitions": args.repetitions,
        "precisions": list(PRECISIONS), "shots": args.shots,
        "clock_contract": "host perf_counter around CUDA-Q sample and separate result iteration; first/warm/build/extraction/end-to-end are distinct",
        "concurrency": "one sequential child process and one GPU timing worker",
        "panel_manifest": {"path": str(args.panel_manifest.resolve().relative_to(ROOT)), "sha256": sha_file(args.panel_manifest.resolve())},
        "environment_lock": {"path": str(args.environment_lock.resolve().relative_to(ROOT)), "sha256": sha_file(args.environment_lock.resolve())},
        "runner_sha256": sha_file(Path(__file__)),
        "rccx_adapter_sha256": sha_file(Path(__file__).with_name("cudaq_qiskit_adapter.py")),
        "qiskit_site_packages": str(args.qiskit_site_packages.resolve()) if args.qiskit_site_packages else "in_process_environment",
        "bridge_manifest": {"path": str(args.bridge_manifest.resolve().relative_to(ROOT)), "sha256": sha_file(args.bridge_manifest.resolve())} if args.bridge_manifest else None,
        "raw_status_counts": dict(Counter(str(row["status"]) for row in records)),
        "outputs": {"attempt_records": {"path": attempts_path.name, "sha256": sha_file(attempts_path)}, "raw_csv": {"path": raw_path.name, "sha256": sha_file(raw_path)}, "environment": {"path": environment_path.name, "sha256": sha_file(environment_path)}},
    }
    manifest_path = out / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # A manifest cannot contain a stable hash of itself.  Its caller records
    # this file's digest in the release index after the immutable artifact is
    # closed; recording a pre-final self-hash here would be false provenance.
    print(json.dumps({"records": len(records), "status_counts": manifest["raw_status_counts"], "checkpoint_only": manifest["checkpoint_only"]}, sort_keys=True))


if __name__ == "__main__":
    main()
