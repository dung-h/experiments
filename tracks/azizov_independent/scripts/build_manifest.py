#!/usr/bin/env python3
"""Build a deterministic manifest from the pinned Ma--Li QASM pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

from qiskit import QuantumCircuit


def family_from_name(path: Path) -> str:
    stem = path.stem
    return stem.split("_indep_", 1)[0]


def source_metrics(qc: QuantumCircuit) -> dict[str, int]:
    counts = qc.count_ops()
    two_qubit = sum(v for name, v in counts.items() if qc.num_qubits and name not in {"measure", "barrier", "reset", "delay"} and _arity(qc, name) == 2)
    return {
        "logical_width": int(qc.num_qubits),
        "logical_depth": int(qc.depth() or 0),
        "logical_ops": int(sum(counts.values())),
        "logical_two_qubit_ops": int(two_qubit),
    }


def _arity(qc: QuantumCircuit, name: str) -> int:
    # QASM inputs in this pool use standard named gates. Looking at the first
    # instruction avoids treating a custom gate name as a two-qubit operation.
    for inst in qc.data:
        if inst.operation.name == name:
            return len(inst.qubits)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mali-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    circuit_dir = args.mali_root / "data" / "quantum_circuits"
    if not circuit_dir.is_dir():
        raise SystemExit(f"QASM directory not found: {circuit_dir}")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in sorted(circuit_dir.glob("*.qasm")):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        try:
            qc = QuantumCircuit.from_qasm_file(str(path))
            metrics = source_metrics(qc)
            status = "ok"
            error = ""
        except Exception as exc:  # retain a row for auditability
            metrics = {"logical_width": "", "logical_depth": "", "logical_ops": "", "logical_two_qubit_ops": ""}
            status = "parse_error"
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "circuit_id": f"{family_from_name(path)}::{path.stem}",
                "family": family_from_name(path),
                # Keep the manifest portable. The runner resolves this path
                # relative to --mali-root at execution time.
                "source_file": str(path.relative_to(args.mali_root)),
                "source_sha256": digest,
                **metrics,
                "source_status": status,
                "source_error": error,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["circuit_id"]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} unique circuits to {args.output}")


if __name__ == "__main__":
    main()
