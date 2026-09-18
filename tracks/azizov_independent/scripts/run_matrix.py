#!/usr/bin/env python3
"""Checkpointed P1 matrix runner with explicit timeout/censoring rows."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import signal
import sys
import time
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import __version__ as qiskit_ibm_runtime_version
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeWashingtonV2


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


BACKENDS = {
    "FakeWashingtonV2": FakeWashingtonV2,
    "FakeSherbrooke": FakeSherbrooke,
}


FIELDS = [
    "circuit_id", "family", "source_file", "source_sha256", "backend_class", "backend_name",
    "optimization_level", "shots", "seed_transpiler", "seed_simulator", "logical_width",
    "logical_depth", "logical_ops", "logical_two_qubit_ops", "physical_width", "physical_depth",
    "physical_ops", "physical_two_qubit_ops", "swap_count", "dag_node_count", "t_transpile_s",
    "t_exec_s", "t_total_s", "status", "error",
]


def load_existing(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mali-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-qubits", type=int, default=16)
    ap.add_argument("--limit-per-family", type=int, default=0)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--seed-transpiler", type=int, default=1234)
    ap.add_argument("--seed-simulator", type=int, default=1234)
    ap.add_argument("--backends", default="FakeWashingtonV2,FakeSherbrooke")
    ap.add_argument("--optimization-levels", default="0,1,2,3")
    args = ap.parse_args()
    selected_backend_names = [x.strip() for x in args.backends.split(",") if x.strip()]
    opt_levels = [int(x.strip()) for x in args.optimization_levels.split(",") if x.strip()]
    unknown = set(selected_backend_names) - set(BACKENDS)
    if unknown:
        raise SystemExit(f"Unknown backend(s): {sorted(unknown)}")
    with args.manifest.open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    candidates = [
        row for row in manifest
        if row.get("source_status") == "ok"
        and row.get("logical_width", "").isdigit()
        and int(row["logical_width"]) <= args.max_qubits
    ]
    candidates.sort(key=lambda r: (r["family"], int(r["logical_width"]), r["circuit_id"]))
    if args.limit_per_family:
        kept = []
        counts = {}
        for row in candidates:
            if counts.get(row["family"], 0) < args.limit_per_family:
                kept.append(row)
                counts[row["family"]] = counts.get(row["family"], 0) + 1
        candidates = kept
    existing = load_existing(args.output)
    done = {(r.get("circuit_id"), r.get("backend_class"), int(r.get("optimization_level", -1))) for r in existing}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
    env_path = args.output.with_suffix(".environment.json")
    if not env_path.exists():
        env_path.write_text(json.dumps({
            "python": sys.version,
            "platform": platform.platform(),
            "qiskit": __import__("qiskit").__version__,
            "qiskit_aer": __import__("qiskit_aer").__version__,
            "qiskit_ibm_runtime": qiskit_ibm_runtime_version,
            "shots": args.shots,
            "timeout_s": args.timeout_s,
            "max_qubits": args.max_qubits,
            "seed_transpiler": args.seed_transpiler,
            "seed_simulator": args.seed_simulator,
            "backends": selected_backend_names,
            "optimization_levels": opt_levels,
            "measurement_protocol": "separate transpile and Aer execution clocks; one untimed backend warm-up",
        }, indent=2) + "\n")

    with args.output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        for backend_name in selected_backend_names:
            backend = BACKENDS[backend_name]()
            simulator = AerSimulator.from_backend(backend)
            warmup = QuantumCircuit(1)
            warmup.measure_all()
            simulator.run(warmup, shots=1, seed_simulator=args.seed_simulator).result()
            for source in candidates:
                key_path = args.mali_root / source["source_file"]
                qc = QuantumCircuit.from_qasm_file(str(key_path))
                for opt in opt_levels:
                    key = (source["circuit_id"], backend_name, opt)
                    if key in done:
                        continue
                    row = {
                        "circuit_id": source["circuit_id"], "family": source["family"],
                        "source_file": source["source_file"], "source_sha256": source["source_sha256"],
                        "backend_class": backend_name, "backend_name": backend.name,
                        "optimization_level": opt, "shots": args.shots,
                        "seed_transpiler": args.seed_transpiler, "seed_simulator": args.seed_simulator,
                        "logical_width": source.get("logical_width", ""), "logical_depth": source.get("logical_depth", ""),
                        "logical_ops": source.get("logical_ops", ""), "logical_two_qubit_ops": source.get("logical_two_qubit_ops", ""),
                        "status": "ok", "error": "",
                    }
                    try:
                        t0 = time.perf_counter()
                        tqc = timed_call(args.timeout_s, lambda: transpile(qc, backend=backend, optimization_level=opt, seed_transpiler=args.seed_transpiler))
                        t_transpile = time.perf_counter() - t0
                        row.update(metrics(tqc), t_transpile_s=t_transpile)
                        t0 = time.perf_counter()
                        timed_call(args.timeout_s, lambda: simulator.run(tqc, shots=args.shots, seed_simulator=args.seed_simulator).result())
                        t_exec = time.perf_counter() - t0
                        row.update(t_exec_s=t_exec, t_total_s=t_transpile + t_exec)
                    except RunTimeout:
                        row.update({k: "" for k in ["physical_width", "physical_depth", "physical_ops", "physical_two_qubit_ops", "swap_count", "dag_node_count", "t_transpile_s", "t_exec_s", "t_total_s"]})
                        row["status"] = "timeout"
                        row["error"] = f"timeout>{args.timeout_s}s"
                    except Exception as exc:
                        row.update({k: "" for k in ["physical_width", "physical_depth", "physical_ops", "physical_two_qubit_ops", "swap_count", "dag_node_count", "t_transpile_s", "t_exec_s", "t_total_s"]})
                        row["status"] = "error"
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    writer.writerow(row)
                    handle.flush()
                    done.add(key)
                    print(backend_name, source["circuit_id"], f"opt={opt}", row["status"], row.get("t_exec_s", ""), flush=True)
    print(f"completed {len(done)} unique matrix keys; output={args.output}")


if __name__ == "__main__":
    main()
