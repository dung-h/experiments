#!/usr/bin/env python3
"""Measure a quality-constrained CUDA-Q MPS target on ``sim_common_q16_v1``.

The MPS adapter strips terminal measurements only.  It times ``get_state`` in
fresh processes and records an independent CUDA-Q dense-FP64 overlap *after*
the MPS timing.  Fidelity/reference construction is therefore evidence about
configuration quality, never part of the MPS execution clock.
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
except ModuleNotFoundError:  # imported by tests via a file spec
    from benchmark_v1.scripts.gpu_lease import acquire_gpu_lease


ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = ROOT.parent
MPS_BOND = 16
ABS_CUTOFF = 1e-10
SVD_ALGORITHM = "gesvdj"
QUALITY_THRESHOLD = 0.99


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    result: dict[str, Any] = {
        "python": sys.version, "platform": platform.platform(), "interpreter": sys.executable,
        "package_lock": package_lock, "package_lock_sha256": hashlib.sha256(package_lock.encode()).hexdigest(),
        "package_lock_source": package_lock_source,
        "cuda_path": os.environ.get("CUDA_PATH", ""), "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "gpu_concurrency": 1,
        "qiskit_site_packages": str(args.qiskit_site_packages.resolve()) if args.qiskit_site_packages else "in_process_environment",
    }
    try:
        import cudaq
        result["cudaq_version"] = str(cudaq.__version__)
    except Exception as exc:
        result["cudaq_error"] = f"{type(exc).__name__}:{exc}"
    if args.qiskit_site_packages:
        try:
            qiskit_path = args.qiskit_site_packages.resolve()
            if str(qiskit_path) not in sys.path:
                sys.path.insert(0, str(qiskit_path))
            import qiskit
            result["qiskit_version"] = str(qiskit.__version__)
            result["qiskit_file"] = str(Path(qiskit.__file__).resolve())
        except Exception as exc:
            result["qiskit_error"] = f"{type(exc).__name__}:{exc}"
    return result


def import_qiskit(args: argparse.Namespace) -> Any:
    """Import Qiskit only after CUDA-Q has initialized its verified target."""
    if args.qiskit_site_packages:
        path = Path(args.qiskit_site_packages).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Qiskit site-packages not found: {path}")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from qiskit import QuantumCircuit  # type: ignore
    return QuantumCircuit


def state_fidelity(mps_state: Any, dense_state: Any, width: int) -> float:
    import numpy as np

    basis = [format(index, f"0{width}b") for index in range(2**width)]
    mps = np.asarray(mps_state.amplitudes([bits[::-1] for bits in basis]), dtype=np.complex128)
    dense = np.asarray(dense_state.to_numpy(), dtype=np.complex128)
    mps /= np.linalg.norm(mps)
    dense /= np.linalg.norm(dense)
    return float(abs(np.vdot(dense, mps)) ** 2)


def unitary_kernel(qasm: Path, expected_hash: str, expected_width: int) -> tuple[Any, str, int]:
    from qiskit import QuantumCircuit, qasm2
    from cudaq.contrib.qiskit_convert import from_qiskit

    if sha_file(qasm) != expected_hash:
        raise ValueError("QASM hash differs from immutable panel manifest")
    circuit = QuantumCircuit.from_qasm_file(str(qasm))
    if circuit.num_qubits != expected_width:
        raise ValueError("QASM width differs from immutable panel manifest")
    stripped = circuit.remove_final_measurements(inplace=False)
    remaining = int(stripped.count_ops().get("measure", 0))
    if remaining:
        raise ValueError("nonterminal measurement: no unitary MPS-state adapter is declared")
    return from_qiskit(stripped), sha_json({
        "adapter": "cudaq.contrib.qiskit_convert.from_qiskit",
        "mode": "mps_state_strip_terminal_measurements_only",
        "canonical_qasm": qasm2.dumps(stripped),
    }), int(circuit.count_ops().get("measure", 0))


def child(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "status": "ok", "error": "", "build_seconds": None, "first_seconds": None,
        "warm_seconds": [], "ir_sha256": "", "terminal_measurements_stripped": None,
        "fidelity": None, "quality_status": "quality_not_evaluated",
        "quality_reference_seconds": None, "quality_extract_seconds": None,
    }
    try:
        import cudaq

        # Make the Qiskit parser available only after CUDA-Q target discovery.
        import_qiskit(args)

        started = time.perf_counter()
        kernel, payload["ir_sha256"], payload["terminal_measurements_stripped"] = unitary_kernel(
            Path(args.qasm_path), args.qasm_sha256, args.width_qubits
        )
        payload["build_seconds"] = time.perf_counter() - started
        os.environ["CUDAQ_MPS_MAX_BOND"] = str(MPS_BOND)
        os.environ["CUDAQ_MPS_ABS_CUTOFF"] = str(ABS_CUTOFF)
        os.environ["CUDAQ_MPS_SVD_ALGO"] = SVD_ALGORITHM
        cudaq.set_target("tensornet-mps", option="fp64")
        started = time.perf_counter(); mps_state = cudaq.get_state(kernel)
        payload["first_seconds"] = time.perf_counter() - started
        for _ in range(args.warmups):
            cudaq.get_state(kernel)
        for _ in range(args.repetitions):
            started = time.perf_counter(); cudaq.get_state(kernel)
            payload["warm_seconds"].append(time.perf_counter() - started)

        # Deliberately after all MPS timed observations. It may be expensive,
        # but its two clocks cannot pollute MPS first/warm measurements.
        reference_started = time.perf_counter()
        cudaq.set_target("nvidia", option="fp64")
        dense_state = cudaq.get_state(kernel)
        payload["quality_reference_seconds"] = time.perf_counter() - reference_started
        extract_started = time.perf_counter()
        payload["fidelity"] = state_fidelity(mps_state, dense_state, args.width_qubits)
        payload["quality_extract_seconds"] = time.perf_counter() - extract_started
        payload["quality_status"] = "ok" if payload["fidelity"] >= QUALITY_THRESHOLD else "quality_failed"
    except MemoryError as exc:
        payload.update(status="resource_limit", error=f"MemoryError:{exc}")
    except Exception as exc:
        detail = f"{type(exc).__name__}:{exc}".splitlines()[0][:1000]
        lowered = detail.lower()
        status = "resource_limit" if any(token in lowered for token in ("out of memory", "memory allocation", "cuda error 2")) else "unsupported" if "unsupported" in lowered else "adapter_error"
        payload.update(status=status, error=detail)
    Path(args.worker_output).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return 0


def make_rows(item: dict[str, str], session_id: str, payload: dict[str, Any], repetitions: int) -> list[dict[str, Any]]:
    status = payload.get("status", "adapter_error")
    if status == "ok":
        status = payload.get("quality_status", "quality_not_evaluated")
    base = {
        "panel_id": item["panel_id"], "panel_member_id": item["panel_member_id"], "basename": item["basename"],
        "qasm_sha256": item["qasm_sha256"], "family": item["family"], "width_qubits": item["width_qubits"],
        "stratum": item["stratum"], "engine_id": "cudaq_mps", "method_id": "cudaq_mps_local",
        "target_id": "cudaq_mps_state_wall_clock_quality_constrained", "precision": "fp64", "shots": "",
        "mps_max_bond": MPS_BOND, "mps_abs_cutoff": ABS_CUTOFF, "mps_svd_algorithm": SVD_ALGORITHM,
        "quality_threshold": QUALITY_THRESHOLD, "session_id": session_id, "status": status, "error": payload.get("error", ""),
        "ir_sha256": payload.get("ir_sha256", ""), "terminal_measurements_stripped": payload.get("terminal_measurements_stripped"),
        "t_build_s": payload.get("build_seconds"), "fidelity": payload.get("fidelity"),
        "t_quality_reference_s": payload.get("quality_reference_seconds"), "t_quality_extract_s": payload.get("quality_extract_seconds"),
    }
    rows: list[dict[str, Any]] = []
    first = dict(base)
    first.update({"observation_kind": "first", "repetition": 0, "t_first_execution_s": payload.get("first_seconds"), "t_warm_execution_s": "", "t_end_to_end_s": ""})
    rows.append(first)
    values = payload.get("warm_seconds", [])
    for repeat in range(repetitions):
        warm = dict(base)
        warm.update({"observation_kind": "warm", "repetition": repeat, "t_first_execution_s": "", "t_warm_execution_s": values[repeat] if repeat < len(values) else "", "t_end_to_end_s": ""})
        rows.append(warm)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--qasm-path", type=Path); parser.add_argument("--qasm-sha256"); parser.add_argument("--width-qubits", type=int)
    parser.add_argument("--warmups", type=int, default=3); parser.add_argument("--repetitions", type=int, default=5); parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--panel-manifest", type=Path); parser.add_argument("--environment-lock", type=Path, default=ROOT / "benchmark_v1/execution/manifests/simulator_environment_lock_v1.json")
    parser.add_argument("--qiskit-site-packages", type=Path,
                        help="Optional sibling site-packages to import Qiskit from after CUDA-Q initialization")
    parser.add_argument("--bridge-manifest", type=Path,
                        help="Optional immutable environment-decision manifest for a sibling Qiskit bridge")
    parser.add_argument("--output-dir", type=Path); parser.add_argument("--sessions", type=int, default=3); parser.add_argument("--cell-timeout-seconds", type=int, default=900); parser.add_argument("--limit-circuits", type=int); parser.add_argument("--resume", action="store_true", help="resume only from this runner's append-only attempt JSONL")
    args = parser.parse_args()
    if args.worker:
        if any(value is None for value in (args.qasm_path, args.qasm_sha256, args.width_qubits, args.worker_output)):
            raise SystemExit("worker requires QASM identity, width and output")
        raise SystemExit(child(args))
    acquire_gpu_lease("run_common_cudaq_mps")
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
        environment_path.write_text(json.dumps(environment(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    existing_cells: set[tuple[str, str]] = set()
    for line in attempts_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            previous = json.loads(line)
            if previous.get("observation_kind") == "first":
                existing_cells.add((str(previous["panel_member_id"]), str(previous["session_id"])))
    for item in selected:
        for number in range(1, args.sessions + 1):
            session_id = f"session-{number}"
            if (item["panel_member_id"], session_id) in existing_cells:
                continue
            payload_path = workers / f"{item['basename']}.fp64_mps_bond{MPS_BOND}.{session_id}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--qasm-path", str(source_path(item["source_path"])), "--qasm-sha256", item["qasm_sha256"], "--width-qubits", item["width_qubits"], "--warmups", str(args.warmups), "--repetitions", str(args.repetitions), "--worker-output", str(payload_path)]
            if args.qiskit_site_packages:
                command.extend(["--qiskit-site-packages", str(args.qiskit_site_packages.resolve())])
            if args.bridge_manifest:
                command.extend(["--bridge-manifest", str(args.bridge_manifest.resolve())])
            try:
                process = subprocess.run(command, text=True, capture_output=True, timeout=args.cell_timeout_seconds, check=False)
                payload = json.loads(payload_path.read_text()) if payload_path.exists() else {"status": "adapter_error", "error": f"child exit={process.returncode}; {process.stderr[-600:]}"}
            except subprocess.TimeoutExpired:
                payload = {"status": "timeout", "error": f"cell exceeded {args.cell_timeout_seconds}s"}
            rows = make_rows(item, session_id, payload, args.repetitions)
            with attempts_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
    records = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit("no completed attempt records")
    raw_path = output / "cudaq_mps_common_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    manifest = {
        "artifact_id": "cudaq-mps-common-panel-v1-20260928", "created_utc": datetime.now(timezone.utc).isoformat(),
        "timing_performed": True, "panel_rows_frozen": len(panel), "panel_rows_attempted": len(selected), "checkpoint_only": args.limit_circuits is not None,
        "sessions": 3, "warmups": 3, "warm_repetitions": 5, "mps": {"precision": "fp64", "max_bond": MPS_BOND, "abs_cutoff": ABS_CUTOFF, "svd_algorithm": SVD_ALGORITHM},
        "quality": {"reference": "CUDA-Q nvidia fp64 get_state on same adapted unitary", "threshold": QUALITY_THRESHOLD, "excluded_from_timing": True},
        "clock_contract": "host perf_counter around CUDA-Q tensornet-mps get_state; build/first/warm are distinct; dense reference and overlap are separately recorded after timing",
        "concurrency": "one sequential child process and one GPU timing worker",
        "panel_manifest": {"path": str(args.panel_manifest.resolve().relative_to(ROOT)), "sha256": sha_file(args.panel_manifest.resolve())},
        "environment_lock": {"path": str(args.environment_lock.resolve().relative_to(ROOT)), "sha256": sha_file(args.environment_lock.resolve())},
        "runner_sha256": sha_file(Path(__file__)), "raw_status_counts": dict(Counter(str(row["status"]) for row in records)),
        "qiskit_site_packages": str(args.qiskit_site_packages.resolve()) if args.qiskit_site_packages else "in_process_environment",
        "bridge_manifest": {"path": str(args.bridge_manifest.resolve().relative_to(ROOT)), "sha256": sha_file(args.bridge_manifest.resolve())} if args.bridge_manifest else None,
        "outputs": {"attempt_records": {"path": attempts_path.name, "sha256": sha_file(attempts_path)}, "raw_csv": {"path": raw_path.name, "sha256": sha_file(raw_path)}, "environment": {"path": environment_path.name, "sha256": sha_file(environment_path)}},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "status_counts": manifest["raw_status_counts"], "checkpoint_only": manifest["checkpoint_only"]}, sort_keys=True))


if __name__ == "__main__":
    main()
