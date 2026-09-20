#!/usr/bin/env python3
"""Generate a controlled local PPS runtime grid from the SP-3 circuit family.

This extension retains the pinned upstream 127-qubit heavy-hex topology and
gate ordering, but varies pre-run controls: Trotter-step count and truncation
threshold. It is separate from the exact 20-step source canary. The timed
target remains only ``propagate_through_rotation_gates``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from pauli_prop.propagation import circuit_to_rotation_gates, propagate_through_rotation_gates

from run_zero_setup_pps_canary import (
    DEFAULT_MAX_TERMS, NUM_QUBITS, OBS_QUBIT, RX_ANGLE, RZZ_ANGLE,
    SOURCE_RELATIVE, distribution_version, expectation_zero_state, git_revision,
    source_topology, write_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trotter-steps", default="4,8,12,20")
    parser.add_argument("--deltas", default="1e-2,5e-3,1e-3")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-terms", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def completed_keys(path: Path) -> set[tuple[int, float, int]]:
    values: set[tuple[int, float, int]] = set()
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            if "error" not in row:
                values.add((int(row["num_trotter_steps"]), float(row["delta"]), int(row["trial"])))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return values


def build_circuit(edges: list[tuple[int, int]], step_count: int) -> QuantumCircuit:
    circuit = QuantumCircuit(NUM_QUBITS)
    for _ in range(step_count):
        for first, second in edges:
            circuit.rzz(RZZ_ANGLE, first, second)
        for qubit in range(NUM_QUBITS):
            circuit.rx(RX_ANGLE, qubit)
    return circuit


def main() -> int:
    args = parse_args()
    steps = [int(value) for value in args.trotter_steps.split(",") if value.strip()]
    deltas = [float(value) for value in args.deltas.split(",") if value.strip()]
    if not steps or any(value < 1 for value in steps) or not deltas or args.trials < 1:
        raise SystemExit("provide positive steps, at least one delta, and positive trials")
    if args.max_terms <= 0 and any(delta not in DEFAULT_MAX_TERMS for delta in deltas):
        raise SystemExit("an unregistered delta requires --max-terms")
    source_file = args.upstream_root / SOURCE_RELATIVE
    if not source_file.is_file():
        raise SystemExit(f"missing upstream source {source_file}")
    edges = source_topology(source_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".environment.json").write_text(json.dumps({
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "upstream_repository": "https://github.com/arulrhikm/mps-pps-zero-setup-benchmarks.git",
        "upstream_revision": git_revision(args.upstream_root), "upstream_source": str(SOURCE_RELATIVE),
        "python": sys.version, "platform": platform.platform(), "qiskit": distribution_version("qiskit"),
        "pauli_prop": distribution_version("pauli-prop"), "numpy": np.__version__,
        "trotter_steps": steps, "deltas": deltas, "trials": args.trials,
        "target_semantics": "local_cpu_seconds_of_propagate_through_rotation_gates_only",
        "extension_contract": "controlled step/delta grid; separate from upstream-equivalent fixed 20-step canary",
    }, indent=2, sort_keys=True) + "\n")
    finished = completed_keys(args.output) if args.resume else set()
    label = "I" * (NUM_QUBITS - 1 - OBS_QUBIT) + "Z" + "I" * OBS_QUBIT
    observable = SparsePauliOp.from_list([(label, 1.0)])
    for step_count in steps:
        build_start = time.perf_counter()
        circuit = build_circuit(edges, step_count)
        build_seconds = time.perf_counter() - build_start
        convert_start = time.perf_counter()
        rotation_gates = circuit_to_rotation_gates(circuit)
        conversion_seconds = time.perf_counter() - convert_start
        common: dict[str, Any] = {
            "timestamp_utc": datetime.now(UTC).isoformat(), "paper_track": "SP-3_zero_setup_extension",
            "target_semantics": "local_cpu_seconds_of_propagate_through_rotation_gates_only",
            "num_qubits": NUM_QUBITS, "num_trotter_steps": step_count,
            "rzz_angle": RZZ_ANGLE, "rx_angle": RX_ANGLE, "observable": f"Z_{OBS_QUBIT}",
            "topology_edges": len(edges), "circuit_depth": circuit.depth(), "circuit_gate_count": circuit.size(),
            "circuit_build_seconds": build_seconds, "conversion_seconds": conversion_seconds,
            "upstream_revision": git_revision(args.upstream_root), "python": sys.version.split()[0],
        }
        print(f"CIRCUIT steps={step_count} depth={circuit.depth()} gates={circuit.size()} build={build_seconds:.3f}s", flush=True)
        for delta in deltas:
            term_cap = int(args.max_terms or DEFAULT_MAX_TERMS[delta])
            for trial in range(args.trials):
                key = (step_count, delta, trial)
                if key in finished:
                    print(f"SKIP steps={step_count} delta={delta:g} trial={trial}", flush=True)
                    continue
                print(f"BEGIN steps={step_count} delta={delta:g} trial={trial} cap={term_cap}", flush=True)
                try:
                    started = time.perf_counter()
                    evolved, truncated_norm = propagate_through_rotation_gates(
                        operator=observable, rot_gates=rotation_gates, max_terms=term_cap, atol=delta, frame="h",
                    )
                    elapsed = time.perf_counter() - started
                    row = {**common, "delta": delta, "trial": trial, "max_terms": term_cap,
                           "propagation_seconds": elapsed, "num_paulis": int(evolved.size),
                           "truncated_norm": float(truncated_norm), "expectation_value": expectation_zero_state(evolved)}
                    write_row(args.output, row)
                    print(f"END seconds={elapsed:.3f} paulis={evolved.size}", flush=True)
                except Exception as exc:
                    write_row(args.output, {**common, "delta": delta, "trial": trial, "max_terms": term_cap, "error": repr(exc)})
                    print(f"FAILED {exc!r}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
