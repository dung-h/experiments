#!/usr/bin/env python3
"""Bounded local reproduction of SP-3's Qiskit pauli-prop experiment.

The upstream script runs a fixed, large sweep on import. This runner preserves
its 127-qubit heavy-hex TFI circuit, observable, one-time conversion and timed
``propagate_through_rotation_gates`` call, but exposes a safe delta/trial
subset. It does not call BlueQubit and it does not use any upstream credential.
"""

from __future__ import annotations

import argparse
import ast
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from pauli_prop.propagation import circuit_to_rotation_gates, propagate_through_rotation_gates


SOURCE_RELATIVE = Path("pauli_path_tests/experiments/pps_benchmark_qiskit.py")
NUM_QUBITS = 127
NUM_TROTTER_STEPS = 20
RZZ_ANGLE = -np.pi / 2
RX_ANGLE = np.pi / 4
OBS_QUBIT = 62
DEFAULT_MAX_TERMS = {
    1.0e-2: 800_000,
    5.0e-3: 1_200_000,
    1.0e-3: 1_800_000,
    5.0e-4: 2_400_000,
    1.0e-4: 3_000_000,
    5.0e-5: 10_000_000,
    2.5e-5: 40_000_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deltas", default="1e-2")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-terms", type=int, default=0,
                        help="Override source-derived cap for every delta; 0 keeps source caps.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def source_topology(source_file: Path) -> list[tuple[int, int]]:
    """Read the exact fallback edge list from the pinned upstream source.

    Parsing avoids importing the upstream module, whose top-level code launches
    a full benchmark. The fallback list is the source's documented copy of the
    IBM heavy-hex connectivity used by the experiment.
    """
    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    assignments: list[ast.Assign] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            assignments.extend(child for child in node.body if isinstance(child, ast.Assign))
    for assignment in assignments:
        if any(isinstance(target, ast.Name) and target.id == "IBM_127_HEAVY_HEX_MAP"
               for target in assignment.targets):
            value = ast.literal_eval(assignment.value)
            return [(int(left), int(right)) for left, right in value]
    raise RuntimeError(f"could not find fallback IBM_127_HEAVY_HEX_MAP in {source_file}")


def expectation_zero_state(operator: SparsePauliOp) -> float:
    labels = operator.paulis.to_labels()
    coefficients = np.real(operator.coeffs)
    return float(sum(coefficient for label, coefficient in zip(labels, coefficients)
                     if all(character in ("I", "Z") for character in label)))


def existing_keys(path: Path) -> set[tuple[float, int]]:
    completed: set[tuple[float, int]] = set()
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if "error" not in row:
                completed.add((float(row["delta"]), int(row["trial"])))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return completed


def write_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def main() -> int:
    args = parse_args()
    source_file = (args.upstream_root / SOURCE_RELATIVE).resolve()
    if not source_file.is_file():
        raise FileNotFoundError(f"missing pinned upstream source: {source_file}")
    deltas = [float(value.strip()) for value in args.deltas.split(",") if value.strip()]
    if not deltas or args.trials < 1:
        raise ValueError("provide at least one delta and one trial")
    for delta in deltas:
        if args.max_terms <= 0 and delta not in DEFAULT_MAX_TERMS:
            raise ValueError(f"no source max_terms cap registered for delta={delta}; use --max-terms explicitly")

    edges = source_topology(source_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment_path = args.output.with_suffix(".environment.json")
    environment_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "upstream_repository": "https://github.com/arulrhikm/mps-pps-zero-setup-benchmarks.git",
                "upstream_revision": git_revision(args.upstream_root),
                "upstream_source": str(SOURCE_RELATIVE),
                "python": sys.version,
                "platform": platform.platform(),
                "qiskit": distribution_version("qiskit"),
                "pauli_prop": distribution_version("pauli-prop"),
                "numpy": np.__version__,
                "target_semantics": "local_cpu_seconds_of_propagate_through_rotation_gates_only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    completed = existing_keys(args.output) if args.resume else set()

    circuit_start = time.perf_counter()
    circuit = QuantumCircuit(NUM_QUBITS)
    for _ in range(NUM_TROTTER_STEPS):
        for first, second in edges:
            circuit.rzz(RZZ_ANGLE, first, second)
        for qubit in range(NUM_QUBITS):
            circuit.rx(RX_ANGLE, qubit)
    circuit_build_seconds = time.perf_counter() - circuit_start

    conversion_start = time.perf_counter()
    rotation_gates = circuit_to_rotation_gates(circuit)
    conversion_seconds = time.perf_counter() - conversion_start
    pauli_label = "I" * (NUM_QUBITS - 1 - OBS_QUBIT) + "Z" + "I" * OBS_QUBIT
    observable = SparsePauliOp.from_list([(pauli_label, 1.0)])
    common = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "paper_track": "SP-3_zero_setup",
        "target_semantics": "local_cpu_seconds_of_propagate_through_rotation_gates_only",
        "upstream_repository": "https://github.com/arulrhikm/mps-pps-zero-setup-benchmarks.git",
        "upstream_revision": git_revision(args.upstream_root),
        "upstream_source": str(SOURCE_RELATIVE),
        "num_qubits": NUM_QUBITS,
        "num_trotter_steps": NUM_TROTTER_STEPS,
        "rzz_angle": RZZ_ANGLE,
        "rx_angle": RX_ANGLE,
        "observable": f"Z_{OBS_QUBIT}",
        "topology_edges": len(edges),
        "circuit_depth": circuit.depth(),
        "circuit_gate_count": circuit.size(),
        "circuit_build_seconds": circuit_build_seconds,
        "conversion_seconds": conversion_seconds,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    print(f"circuit: {NUM_QUBITS}q, depth={circuit.depth()}, gates={circuit.size()}, edges={len(edges)}")
    print(f"one-time build={circuit_build_seconds:.3f}s conversion={conversion_seconds:.3f}s")
    for delta in deltas:
        max_terms = int(args.max_terms or DEFAULT_MAX_TERMS[delta])
        for trial in range(args.trials):
            if (delta, trial) in completed:
                print(f"skip delta={delta:.2e} trial={trial}")
                continue
            print(f"run delta={delta:.2e} trial={trial} max_terms={max_terms:,}", flush=True)
            try:
                started = time.perf_counter()
                evolved, truncated_norm = propagate_through_rotation_gates(
                    operator=observable,
                    rot_gates=rotation_gates,
                    max_terms=max_terms,
                    atol=delta,
                    frame="h",
                )
                propagation_seconds = time.perf_counter() - started
                row = {
                    **common,
                    "delta": delta,
                    "trial": trial,
                    "max_terms": max_terms,
                    "propagation_seconds": propagation_seconds,
                    "num_paulis": int(evolved.size),
                    "truncated_norm": float(truncated_norm),
                    "expectation_value": expectation_zero_state(evolved),
                }
                write_row(args.output, row)
                print(f"done {propagation_seconds:.3f}s paulis={evolved.size:,}", flush=True)
            except Exception as exc:  # retain failed configuration as a label boundary
                write_row(args.output, {**common, "delta": delta, "trial": trial,
                                        "max_terms": max_terms, "error": repr(exc)})
                print(f"failed: {exc!r}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
