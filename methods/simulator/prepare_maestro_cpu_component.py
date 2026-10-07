#!/usr/bin/env python3
"""Validate/build the frozen Maestro CPU-SV plan without calling Maestro.

Plan preparation is static. It neither runs QCSim nor authorizes timing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "benchmark_v1/protocol/maestro_cpu_component_benchmark.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_seed_registry_pin(protocol: dict, input_root: Path) -> str:
    path = input_root / "benchmark_v1/registry/seed_registry.json"
    actual = sha(path)
    if actual != protocol["context"]["seed_registry_sha256"]:
        raise ValueError("seed registry hash does not match the frozen protocol pin")
    return actual


def synthetic_qasm(n: int, word: str, repeats: int) -> str:
    if n < 2 or word not in {"one_qubit_noncommuting", "two_qubit_wrapper_control",
                             "two_qubit_cx_interleaved", "zero_gate_sample"}:
        raise ValueError("unsupported synthetic descriptor")
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];", f"creg c[{n}];"]
    if word != "zero_gate_sample":
        lines += ["rx(pi/2) q[0];", f"cx q[0],q[{n-1}];"]
        lines += [f"rx({0.23 + 0.01*q:.8f}) q[{q}];" for q in range(n)]
    elif repeats:
        raise ValueError("sampling descriptor cannot contain operation repeats")
    for i in range(repeats):
        if word == "one_qubit_noncommuting":
            q = i % n
            lines += [f"rz({0.013 + 0.0001*i:.8f}) q[{q}];",
                      f"rx({0.021 + 0.0001*i:.8f}) q[{q}];"]
        else:
            a, b = i % (n-1), i % (n-1) + 1
            lines += [f"rx({0.017 + 0.0001*i:.8f}) q[{a}];"]
            if word == "two_qubit_cx_interleaved":
                lines += [f"cx q[{a}],q[{b}];"]
            lines += [f"rz({0.031 + 0.0001*i:.8f}) q[{b}];"]
    return "\n".join([*lines, "measure q -> c;", ""])


def measurement_profile(raw: str) -> dict:
    from qiskit import qasm2
    circuit = qasm2.loads(raw, strict=True,
                          custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    pairs = [[circuit.find_bit(inst.qubits[0]).index,
              circuit.find_bit(inst.clbits[0]).index]
             for inst in circuit.data if inst.operation.name == "measure"]
    first = next((i for i, inst in enumerate(circuit.data) if inst.operation.name == "measure"), None)
    tail_is_terminal = (first is not None and
                        all(inst.operation.name in {"measure", "barrier"}
                            for inst in circuit.data[first:]))
    injective = (len({q for q, _ in pairs}) == len(pairs) and
                 len({c for _, c in pairs}) == len(pairs))
    return {
        "measurement_map": pairs,
        "measurement_map_sha256": digest(pairs),
        "measured_qubit_count": len({q for q, _ in pairs}),
        "measured_classical_bit_count": len({c for _, c in pairs}),
        "measured_classical_extent": max((c for _, c in pairs), default=-1) + 1,
        "declared_classical_width": int(circuit.num_clbits),
        "measurement_terminal": bool(tail_is_terminal),
        "measurement_map_injective": bool(injective),
    }


def calibration_cells(protocol: dict) -> list[dict]:
    cfg = protocol["calibration"]
    cells = []
    for n in cfg["widths"]:
        for word in cfg["operation_classes"]:
            for repeats in cfg["repeats"]:
                cells.append(dict(kind="operation", width=n, operation=word, repeats=repeats,
                                  shots=cfg["operation_shots"], qasm=synthetic_qasm(n, word, repeats)))
        for shots in cfg["sampling_shots"]:
            cells.append(dict(kind="sampling", width=n, operation="zero_gate_sample", repeats=0,
                              shots=shots, qasm=synthetic_qasm(n, "zero_gate_sample", 0)))
    for cell in cells:
        cell["qasm_sha256"] = hashlib.sha256(cell["qasm"].encode()).hexdigest()
        cell["cell_id"] = digest({k: v for k, v in cell.items() if k != "qasm"})
    if len(cells) != cfg["total_cells"] or len({c["cell_id"] for c in cells}) != len(cells):
        raise ValueError("calibration cell count/identity mismatch")
    return cells


def common_panel(protocol: dict, input_root: Path, source_dir: Path) -> tuple[list[dict], dict]:
    path = input_root / protocol["panel"]["manifest"]
    if sha(path) != protocol["panel"]["manifest_sha256"]:
        raise ValueError("common panel hash changed")
    with path.open(newline="") as handle:
        aliases = [r for r in csv.DictReader(handle) if r["stratum"] == "core_q2_q9"]
    fold_path = input_root / protocol["panel"]["fold_source"]
    if sha(fold_path) != protocol["panel"]["fold_source_sha256"]:
        raise ValueError("frozen fold source hash changed")
    with fold_path.open(newline="") as handle:
        folds = [r for r in csv.DictReader(handle) if r["stratum"] == "core_q2_q9"]
    fold_map = {}
    for row in folds:
        old = fold_map.setdefault(row["source_sha256"], int(row["fold"]))
        if old != int(row["fold"]):
            raise ValueError("source QASM hash crosses folds")
    groups = {}
    for row in aliases:
        group = groups.setdefault(row["qasm_sha256"], [])
        group.append(row)
    if len(groups) != 150 or len(aliases) != 162 or set(groups) != set(fold_map):
        raise ValueError("common core/frozen folds changed")
    if {f: sum(v == f for v in fold_map.values()) for f in range(5)} != {0:37, 1:28, 2:26, 3:25, 4:34}:
        raise ValueError("frozen fold counts changed")
    result = []
    for qhash, members in sorted(groups.items()):
        row = sorted(members, key=lambda r: r["basename"])[0]
        circuit = source_dir / row["basename"]
        if not circuit.is_file() or sha(circuit) != qhash:
            raise ValueError(f"missing or mismatched source QASM: {circuit}")
        raw = circuit.read_text(encoding="utf-8")
        profile = measurement_profile(raw)
        result.append(dict(source_qasm_sha256=qhash, source_file=str(circuit), width=int(row["width_qubits"]),
                           outer_fold=fold_map[qhash], aliases=[r["panel_member_id"] for r in members],
                           shots=protocol["context"]["shots"], **profile, runtime_label=None,
                           status="not_measured_in_new_qcsim_context"))
    return result, dict(panel_sha256=sha(path), fold_source_sha256=sha(fold_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL,
                        help="Full frozen or resolved runtime protocol; defaults to the historical optimizer-on protocol.")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cells = calibration_cells(protocol)
    input_root = args.input_root.resolve()
    panel, pins = common_panel(protocol, input_root, args.source_dir.resolve())
    seed_registry_sha256 = validate_seed_registry_pin(protocol, input_root)
    pins["seed_registry_sha256"] = seed_registry_sha256
    if {c["qasm_sha256"] for c in cells} & {r["source_qasm_sha256"] for r in panel}:
        raise ValueError("synthetic calibration overlaps benchmark QASM")
    pilot = [c["cell_id"] for c in cells if c["width"] == 9 and
             (c["kind"] == "operation" or c["shots"] == 1000)]
    if len(pilot) != 10:
        raise ValueError("pilot must contain exactly ten cells")
    plan = dict(protocol_sha256=sha(protocol_path), protocol_id=protocol["protocol_id"],
                protocol_path=str(protocol_path), preparer_sha256=sha(Path(__file__)), source_pins=pins,
                calibration_cells=cells, panel=panel, pilot_cell_ids=pilot,
                assigned_hashes=150, calibration_cell_count=112,
                measurements_per_cell=15,
                reuse_terminal_pilot=False, training_performed=False, timing_performed=False,
                status="static_plan_valid_timing_not_performed")
    if args.output_dir:
        if args.output_dir.exists():
            raise FileExistsError(f"output exists: {args.output_dir}")
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({k:v for k,v in plan.items() if k not in {"calibration_cells", "panel", "pilot_cell_ids"}}, indent=2))


if __name__ == "__main__":
    main()
