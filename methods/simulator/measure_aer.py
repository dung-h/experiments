#!/usr/bin/env python3
"""Checkpointed P1 matrix runner with explicit timeout/censoring rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
try:
    from qiskit_ibm_runtime import __version__ as qiskit_ibm_runtime_version
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeWashingtonV2
except ModuleNotFoundError:
    # The local Aer venv can provide the same versioned fake backend classes
    # through Qiskit's provider package without installing the cloud runtime.
    # This fallback does not submit a job and preserves the backend snapshot
    # identity in the environment manifest.
    qiskit_ibm_runtime_version = "uninstalled; local fake-provider fallback"
    from qiskit.providers.fake_provider import FakeSherbrooke, FakeWashingtonV2


class RunTimeout(Exception):
    pass


def timeout_handler(_signum, _frame):
    raise RunTimeout


def timed_call(seconds, fn):
    old = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def metrics(circuit):
    counts = circuit.count_ops()
    two_q_names = {"cx", "ecr", "cz", "rzz", "swap", "iswap"}
    return {
        "physical_width": int(circuit.num_qubits),
        "physical_depth": int(circuit.depth() or 0),
        "physical_ops": int(sum(counts.values())),
        "physical_two_qubit_ops": int(sum(v for n, v in counts.items() if n in two_q_names)),
        "swap_count": int(counts.get("swap", 0)),
        "dag_node_count": int(len(circuit.data)),
    }


def stable_json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def ram_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None


def capture_package_lock(path: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    else:
        # Some uv-created environments intentionally do not install pip. The
        # stdlib metadata view still captures the actual distributions loaded
        # by this interpreter without mutating the environment.
        lines = sorted(
            f"{dist.metadata.get('Name', 'unknown')}=={dist.version}"
            for dist in importlib.metadata.distributions()
        )
    normalized = "\n".join(sorted(lines)) + "\n"
    path.write_text(normalized, encoding="utf-8")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def backend_fingerprint(backend_class_name: str) -> dict[str, object]:
    backend = BACKENDS[backend_class_name]()
    simulator = AerSimulator.from_backend(backend)
    if hasattr(backend, "properties") and backend.properties() is not None:
        properties = backend.properties().to_dict()
    else:
        # Older Qiskit fake-provider snapshots expose Target/qubit properties
        # but no IBM Runtime ``properties()`` method.  Keep a compact,
        # deterministic snapshot rather than treating the local backend as
        # unavailable or serialising the enormous Target repr.
        qubit_properties = getattr(backend, "qubit_properties", [])
        if callable(qubit_properties):
            # BackendV2 in the local Qiskit 0.25 snapshot exposes a
            # per-index ``qubit_properties(qubit)`` method rather than a
            # materialised list.  Preserve the full snapshot deterministically.
            qubit_properties = [
                qubit_properties(index)
                for index in range(int(getattr(backend, "num_qubits", 0)))
            ]
        properties = {
            "backend_version": getattr(backend, "backend_version", None),
            "operation_names": sorted(getattr(backend, "operation_names", [])),
            "qubit_properties": [str(item) for item in (qubit_properties or [])],
        }
    coupling_edges = sorted([list(edge) for edge in backend.coupling_map.get_edges()])
    snapshot = {
        "backend_class": backend_class_name,
        "backend_name": backend.name,
        "backend_version": getattr(backend, "version", None),
        "num_qubits": backend.num_qubits,
        "coupling_edges": coupling_edges,
        "properties": properties,
    }
    noise_model = simulator.options.noise_model
    return {
        "backend_name": backend.name,
        "aer_method": simulator.options.method,
        "fake_backend_snapshot_hash": stable_json_hash(snapshot),
        "noise_model_hash": stable_json_hash(noise_model.to_dict()),
    }


BACKENDS = {
    "FakeWashingtonV2": FakeWashingtonV2,
    "FakeSherbrooke": FakeSherbrooke,
}


FIELDS = [
    "circuit_id", "family", "source_file", "source_sha256", "backend_class", "backend_name",
    "optimization_level", "shots", "seed_transpiler", "seed_simulator", "logical_width",
    "logical_depth", "logical_ops", "logical_two_qubit_ops", "physical_width", "physical_depth",
    "physical_ops", "physical_two_qubit_ops", "swap_count", "dag_node_count", "t_transpile_s",
    "t_first_exec_s", "t_exec_s", "t_total_s", "session_id", "repetition",
    "warmup_count", "observation_kind", "status", "error",
]


def load_existing(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def existing_key(row):
    """Return the raw-attempt identity used by checkpoint/resume.

    Older P0/P1 files did not carry session or repetition columns.  They are
    deliberately treated as session-1/repetition-0 so an old file can still
    be inspected, while a new run refuses to append to an incompatible header
    (see ``validate_output_schema`` below).
    """

    try:
        optimization = int(row.get("optimization_level", -1))
        repetition = int(row.get("repetition", 0) or 0)
    except (TypeError, ValueError):
        return None
    return (
        row.get("circuit_id"),
        row.get("backend_class"),
        optimization,
        row.get("session_id") or "session-1",
        repetition,
    )


def validate_output_schema(path: Path) -> None:
    """Prevent silent mixing of pre-session and raw-repetition CSV formats."""

    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    if header != FIELDS:
        raise SystemExit(
            f"existing output has an incompatible header: {path}; "
            "choose a new output path instead of mixing measurement formats"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mali-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-qubits", type=int, default=16)
    ap.add_argument("--max-logical-ops", type=int, default=0)
    ap.add_argument("--max-logical-depth", type=int, default=0)
    ap.add_argument("--limit-per-family", type=int, default=0)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument(
        "--sessions", type=int, default=1,
        help="independent simulator sessions (core contract: 3)",
    )
    ap.add_argument(
        "--warmups", type=int, default=1,
        help="untimed target-circuit warmups per session (core contract: 3)",
    )
    ap.add_argument(
        "--repetitions", type=int, default=1,
        help="timed warm executions per session (core contract: 5)",
    )
    ap.add_argument("--seed-transpiler", type=int, default=1234)
    ap.add_argument("--seed-simulator", type=int, default=1234)
    ap.add_argument("--max-parallel-threads", type=int, default=0,
                    help="limit Aer CPU threads; 0 keeps Aer default")
    ap.add_argument("--max-parallel-shots", type=int, default=0,
                    help="limit Aer shot parallelism; 0 keeps Aer default")
    ap.add_argument("--max-parallel-experiments", type=int, default=0,
                    help="limit Aer experiment parallelism; 0 keeps Aer default")
    ap.add_argument("--backends", default="FakeWashingtonV2,FakeSherbrooke")
    ap.add_argument("--optimization-levels", default="0,1,2,3")
    ap.add_argument("--exclude-families", default="", help="comma-separated families to omit from a bounded pilot")
    ap.add_argument("--selection-mode", choices=("smallest", "spread"), default="smallest")
    args = ap.parse_args()
    selected_backend_names = [x.strip() for x in args.backends.split(",") if x.strip()]
    opt_levels = [int(x.strip()) for x in args.optimization_levels.split(",") if x.strip()]
    excluded_families = {x.strip() for x in args.exclude_families.split(",") if x.strip()}
    unknown = set(selected_backend_names) - set(BACKENDS)
    if unknown:
        raise SystemExit(f"Unknown backend(s): {sorted(unknown)}")
    if args.sessions < 1 or args.warmups < 1 or args.repetitions < 1:
        raise SystemExit("--sessions, --warmups and --repetitions must all be >= 1")
    with args.manifest.open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    candidates = [
        row for row in manifest
        if row.get("source_status") == "ok"
        and row.get("family") not in excluded_families
        and row.get("logical_width", "").isdigit()
        and int(row["logical_width"]) <= args.max_qubits
        and (not args.max_logical_ops or int(row.get("logical_ops") or 0) <= args.max_logical_ops)
        and (not args.max_logical_depth or int(row.get("logical_depth") or 0) <= args.max_logical_depth)
    ]
    candidates.sort(key=lambda r: (r["family"], int(r["logical_width"]), r["circuit_id"]))
    if args.limit_per_family:
        kept = []
        by_family = {}
        for row in candidates:
            by_family.setdefault(row["family"], []).append(row)
        for family_rows in by_family.values():
            if len(family_rows) <= args.limit_per_family:
                kept.extend(family_rows)
                continue
            if args.selection_mode == "spread":
                # Stratify across the available width/depth range instead of
                # selecting only the smallest circuits in each family.
                if args.limit_per_family == 1:
                    positions = [0]
                else:
                    positions = [round(i * (len(family_rows) - 1) / (args.limit_per_family - 1)) for i in range(args.limit_per_family)]
                kept.extend(family_rows[index] for index in positions)
            else:
                kept.extend(family_rows[:args.limit_per_family])
        candidates = kept
    validate_output_schema(args.output)
    existing = load_existing(args.output)
    done = {key for key in (existing_key(row) for row in existing) if key is not None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.stat().st_size == 0:
        with args.output.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
    env_path = args.output.with_suffix(".environment.json")
    package_lock_path = args.output.with_suffix(".pip-freeze.txt")
    if not env_path.exists():
        fingerprints = {
            name: backend_fingerprint(name) for name in selected_backend_names
        }
        env_path.write_text(json.dumps({
            "status": "captured",
            "host_id": "sha256:" + hashlib.sha256(platform.node().encode("utf-8")).hexdigest()[:16],
            "os": platform.platform(),
            "cpu_model": cpu_model(),
            "ram_bytes": ram_bytes(),
            "python_version": platform.python_version(),
            "package_lock_sha256": capture_package_lock(package_lock_path),
            "package_lock_path": str(package_lock_path),
            "thread_policy": {
                "max_parallel_threads": args.max_parallel_threads or "aer_default",
                "max_parallel_shots": args.max_parallel_shots or "aer_default",
                "max_parallel_experiments": args.max_parallel_experiments or "aer_default",
            },
            "python": sys.version,
            "platform": platform.platform(),
            "qiskit": __import__("qiskit").__version__,
            "qiskit_aer": __import__("qiskit_aer").__version__,
            "qiskit_ibm_runtime": qiskit_ibm_runtime_version,
            "qiskit_version": __import__("qiskit").__version__,
            "aer_version": __import__("qiskit_aer").__version__,
            "aer_method": {name: value["aer_method"] for name, value in fingerprints.items()},
            "fake_backend_class": selected_backend_names,
            "fake_backend_snapshot_hash": {
                name: value["fake_backend_snapshot_hash"] for name, value in fingerprints.items()
            },
            "noise_model_hash": {
                name: value["noise_model_hash"] for name, value in fingerprints.items()
            },
            "shots": args.shots,
            "transpiler_seed": args.seed_transpiler,
            "simulator_seed": args.seed_simulator,
            "timeout_s": args.timeout_s,
            "sessions": args.sessions,
            "untimed_target_warmups_per_session": args.warmups,
            "timed_warm_repetitions_per_session": args.repetitions,
            "max_qubits": args.max_qubits,
            "seed_transpiler": args.seed_transpiler,
            "seed_simulator": args.seed_simulator,
            "max_parallel_threads": args.max_parallel_threads or None,
            "max_parallel_shots": args.max_parallel_shots or None,
            "max_parallel_experiments": args.max_parallel_experiments or None,
            "backends": selected_backend_names,
            "optimization_levels": opt_levels,
            "excluded_families": sorted(excluded_families),
            "selection_mode": args.selection_mode,
            "measurement_protocol": (
                "one new Aer simulator per session; separate transpile, first execution, "
                "untimed target warmups and timed warm execution repetitions"
            ),
        }, indent=2) + "\n")
    else:
        previous = json.loads(env_path.read_text(encoding="utf-8"))
        expected = {
            "shots": args.shots,
            "sessions": args.sessions,
            "untimed_target_warmups_per_session": args.warmups,
            "timed_warm_repetitions_per_session": args.repetitions,
            "seed_transpiler": args.seed_transpiler,
            "seed_simulator": args.seed_simulator,
            "backends": selected_backend_names,
            "optimization_levels": opt_levels,
        }
        mismatches = {
            key: {"recorded": previous.get(key), "requested": value}
            for key, value in expected.items()
            if previous.get(key) != value
        }
        if mismatches:
            raise SystemExit(
                "resume configuration differs from the recorded environment: "
                + json.dumps(mismatches, sort_keys=True)
            )

    with args.output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        for backend_name in selected_backend_names:
            for session_index in range(args.sessions):
                session_id = f"session-{session_index + 1}"
                backend = BACKENDS[backend_name]()
                simulator = AerSimulator.from_backend(backend)
                if args.max_parallel_threads:
                    simulator.set_options(max_parallel_threads=args.max_parallel_threads)
                if args.max_parallel_shots:
                    simulator.set_options(max_parallel_shots=args.max_parallel_shots)
                if args.max_parallel_experiments:
                    simulator.set_options(max_parallel_experiments=args.max_parallel_experiments)

                # Backend construction/lazy initialization is not part of the
                # target clock. This remains a separate, unrecorded setup
                # warm-up; target-circuit warm-ups are recorded in metadata.
                backend_warmup = QuantumCircuit(1)
                backend_warmup.measure_all()
                simulator.run(backend_warmup, shots=1, seed_simulator=args.seed_simulator).result()

                for source in candidates:
                    key_path = args.mali_root / source["source_file"]
                    qc = QuantumCircuit.from_qasm_file(str(key_path))
                    for opt in opt_levels:
                        planned_keys = {
                            (source["circuit_id"], backend_name, opt, session_id, repetition)
                            for repetition in range(args.repetitions)
                        }
                        if planned_keys.issubset(done):
                            continue

                        common = {
                            "circuit_id": source["circuit_id"], "family": source["family"],
                            "source_file": source["source_file"], "source_sha256": source["source_sha256"],
                            "backend_class": backend_name, "backend_name": backend.name,
                            "optimization_level": opt, "shots": args.shots,
                            "seed_transpiler": args.seed_transpiler, "seed_simulator": args.seed_simulator,
                            "logical_width": source.get("logical_width", ""), "logical_depth": source.get("logical_depth", ""),
                            "logical_ops": source.get("logical_ops", ""), "logical_two_qubit_ops": source.get("logical_two_qubit_ops", ""),
                            "session_id": session_id, "warmup_count": args.warmups,
                            "observation_kind": "raw_repetition",
                        }
                        physical = {}
                        t_transpile = ""
                        t_first = ""
                        failure_status = ""
                        failure_error = ""
                        tqc = None
                        try:
                            t0 = time.perf_counter()
                            tqc = timed_call(
                                args.timeout_s,
                                lambda: transpile(qc, backend=backend, optimization_level=opt, seed_transpiler=args.seed_transpiler),
                            )
                            t_transpile = time.perf_counter() - t0
                            physical = metrics(tqc)

                            # Keep first-call behavior separate from warm
                            # execution. It is not used as the primary label.
                            t0 = time.perf_counter()
                            timed_call(
                                args.timeout_s,
                                lambda: simulator.run(tqc, shots=args.shots, seed_simulator=args.seed_simulator).result(),
                            )
                            t_first = time.perf_counter() - t0

                            # These calls are intentionally untimed. A timeout
                            # still becomes a retained failure row below.
                            for _ in range(args.warmups):
                                timed_call(
                                    args.timeout_s,
                                    lambda: simulator.run(tqc, shots=args.shots, seed_simulator=args.seed_simulator).result(),
                                )
                        except RunTimeout:
                            failure_status = "timeout"
                            failure_error = f"timeout>{args.timeout_s}s"
                        except Exception as exc:
                            failure_status = "error"
                            failure_error = f"{type(exc).__name__}: {exc}"

                        for repetition in range(args.repetitions):
                            key = (source["circuit_id"], backend_name, opt, session_id, repetition)
                            if key in done:
                                continue
                            row = {
                                **common,
                                **physical,
                                "t_transpile_s": t_transpile if repetition == 0 else "",
                                "t_first_exec_s": t_first if repetition == 0 else "",
                                "t_exec_s": "",
                                "t_total_s": "",
                                "repetition": repetition,
                                "status": failure_status or "ok",
                                "error": failure_error,
                            }
                            if not failure_status:
                                try:
                                    t0 = time.perf_counter()
                                    timed_call(
                                        args.timeout_s,
                                        lambda: simulator.run(tqc, shots=args.shots, seed_simulator=args.seed_simulator).result(),
                                    )
                                    t_exec = time.perf_counter() - t0
                                    row["t_exec_s"] = t_exec
                                    row["t_total_s"] = (
                                        (float(t_transpile) if t_transpile != "" else 0.0) + t_exec
                                    )
                                except RunTimeout:
                                    row["status"] = "timeout"
                                    row["error"] = f"timeout>{args.timeout_s}s"
                                except Exception as exc:
                                    row["status"] = "error"
                                    row["error"] = f"{type(exc).__name__}: {exc}"
                            writer.writerow(row)
                            handle.flush()
                            done.add(key)
                            print(
                                backend_name, source["circuit_id"],
                                f"opt={opt} {session_id} rep={repetition}",
                                row["status"], row.get("t_exec_s", ""), flush=True,
                            )
    print(f"completed {len(done)} raw matrix keys; output={args.output}")


if __name__ == "__main__":
    main()
