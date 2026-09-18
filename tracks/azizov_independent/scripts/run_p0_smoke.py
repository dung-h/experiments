#!/usr/bin/env python3
"""Run the bounded P0 Aer/transpilation smoke experiment."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import __version__ as qiskit_ibm_runtime_version
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeWashingtonV2


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def _select(rows: list[dict[str, str]], max_qubits: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("source_status") == "ok" and row.get("logical_width", "").isdigit():
            if int(row["logical_width"]) <= max_qubits:
                grouped[row["family"]].append(row)
    selected = []
    for family, family_rows in sorted(grouped.items()):
        # Use the largest small circuit in each family to exercise some
        # structure without allowing a smoke run to become a large simulation.
        selected.append(
            max(
                family_rows,
                key=lambda x: (
                    int(x.get("logical_width") or 0),
                    int(x.get("logical_depth") or 0),
                    int(x.get("logical_ops") or 0),
                ),
            )
        )
    return selected


def _backend_objects():
    return [("FakeWashingtonV2", FakeWashingtonV2), ("FakeSherbrooke", FakeSherbrooke)]


def _metrics(circuit: QuantumCircuit) -> dict[str, int]:
    counts = circuit.count_ops()
    two_q = sum(v for name, v in counts.items() if name in {"cx", "ecr", "cz", "rzz", "swap", "iswap"})
    swaps = int(counts.get("swap", 0))
    return {
        "physical_width": int(circuit.num_qubits),
        "physical_depth": int(circuit.depth() or 0),
        "physical_ops": int(sum(counts.values())),
        "physical_two_qubit_ops": int(two_q),
        "swap_count": swaps,
        "dag_node_count": int(len(circuit.data)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mali-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-qubits", type=int, default=8)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--seed-transpiler", type=int, default=1234)
    ap.add_argument("--seed-simulator", type=int, default=1234)
    args = ap.parse_args()

    with args.manifest.open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    selected = _select(manifest, args.max_qubits)
    if not selected:
        raise SystemExit("No smoke circuits selected")
    by_id = {row["circuit_id"]: row for row in manifest}
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "qiskit": _version("qiskit"),
        "qiskit_aer": _version("qiskit_aer"),
        "qiskit_ibm_runtime": qiskit_ibm_runtime_version,
        "measurement_protocol": "one_timed_run_after_one_untimed_backend_warmup",
        "shots": args.shots,
        "seed_transpiler": args.seed_transpiler,
        "seed_simulator": args.seed_simulator,
        "max_qubits": args.max_qubits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env_path = args.output.with_suffix(".environment.json")
    env_path.write_text(json.dumps(env, indent=2) + "\n")

    fields = [
        "circuit_id", "family", "source_file", "source_sha256", "backend_class",
        "backend_name", "optimization_level", "shots", "seed_transpiler",
        "seed_simulator", "logical_width", "logical_depth", "logical_ops",
        "logical_two_qubit_ops", "physical_width", "physical_depth", "physical_ops",
        "physical_two_qubit_ops", "swap_count", "dag_node_count", "t_transpile_s",
        "t_exec_s", "t_total_s", "status", "error",
    ]
    rows: list[dict[str, object]] = []
    for backend_class_name, backend_cls in _backend_objects():
        backend = backend_cls()
        simulator = AerSimulator.from_backend(backend)
        # Backend construction is explicitly outside T_exec. This tiny warm-up
        # removes one-time simulator initialization from the first observation.
        warmup = QuantumCircuit(1)
        warmup.measure_all()
        simulator.run(warmup, shots=1, seed_simulator=args.seed_simulator).result()
        for source in selected:
            source_path = args.mali_root / source["source_file"]
            qc = QuantumCircuit.from_qasm_file(str(source_path))
            for opt in range(4):
                base = {
                    "circuit_id": source["circuit_id"],
                    "family": source["family"],
                    "source_file": source["source_file"],
                    "source_sha256": source["source_sha256"],
                    "backend_class": backend_class_name,
                    "backend_name": backend.name,
                    "optimization_level": opt,
                    "shots": args.shots,
                    "seed_transpiler": args.seed_transpiler,
                    "seed_simulator": args.seed_simulator,
                    "logical_width": source.get("logical_width", ""),
                    "logical_depth": source.get("logical_depth", ""),
                    "logical_ops": source.get("logical_ops", ""),
                    "logical_two_qubit_ops": source.get("logical_two_qubit_ops", ""),
                    "status": "ok",
                    "error": "",
                }
                try:
                    t0 = time.perf_counter()
                    tqc = transpile(qc, backend=backend, optimization_level=opt, seed_transpiler=args.seed_transpiler)
                    t_transpile = time.perf_counter() - t0
                    pmetrics = _metrics(tqc)
                    t0 = time.perf_counter()
                    simulator.run(tqc, shots=args.shots, seed_simulator=args.seed_simulator).result()
                    t_exec = time.perf_counter() - t0
                    base.update(pmetrics, t_transpile_s=t_transpile, t_exec_s=t_exec, t_total_s=t_transpile + t_exec)
                except Exception as exc:
                    base.update({k: "" for k in ["physical_width", "physical_depth", "physical_ops", "physical_two_qubit_ops", "swap_count", "dag_node_count", "t_transpile_s", "t_exec_s", "t_total_s"]})
                    base["status"] = "error"
                    base["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(base)
                print(backend_class_name, source["circuit_id"], f"opt={opt}", base["status"], base.get("t_exec_s", ""), flush=True)

    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
