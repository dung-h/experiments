#!/usr/bin/env python3
"""Compare archived recipe candidates with exact archive-matched submitted QASM.

This checks ideal measured-output distributions from the all-zero input. It is
not a proof of full-unitary identity, unique circuit reconstruction, noisy
execution equivalence, or runtime-feature equivalence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
import zipfile
from collections import OrderedDict, defaultdict
from pathlib import Path

from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import CPhaseGate, ECRGate, RZGate, SXGate, SwapGate
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts/real_qpu/logical_recovery"
DEFAULT_SUBMITTED = Path("/home/server/Documents/Qonductor-SC25/data/database/circuits.zip")
TOLERANCE = 1e-8


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_qasm(data: bytes) -> QuantumCircuit:
    custom = [
        qasm2.CustomInstruction("rz", 1, 1, RZGate, builtin=True),
        qasm2.CustomInstruction("sx", 0, 1, SXGate, builtin=True),
        qasm2.CustomInstruction("ecr", 0, 2, ECRGate, builtin=True),
        qasm2.CustomInstruction("cp", 1, 2, CPhaseGate, builtin=True),
        qasm2.CustomInstruction("swap", 0, 2, SwapGate, builtin=True),
    ]
    return qasm2.loads(data.decode("utf-8-sig"), custom_instructions=custom)


def ideal_measured_distribution(circuit: QuantumCircuit) -> tuple[dict[int, float], int]:
    """Marginalize unused/ancilla wires and map measured bits to classical bits."""
    unsupported = {"reset", "if_else", "while_loop", "for_loop", "switch_case"}
    for item in circuit.data:
        if item.operation.name in unsupported:
            raise ValueError(f"unsupported operation for this distribution check: {item.operation.name}")
    measured_pairs = []
    measurement_started = False
    for item in circuit.data:
        if item.operation.name == "measure":
            measurement_started = True
            if len(item.qubits) != 1 or len(item.clbits) != 1:
                raise ValueError("non-single-qubit measurement")
            measured_pairs.append((circuit.find_bit(item.qubits[0]).index,
                                   circuit.find_bit(item.clbits[0]).index))
        elif measurement_started and item.operation.name not in {"barrier", "delay"}:
            raise ValueError("nonterminal measurement")
    qindices = sorted({circuit.find_bit(qubit).index
                       for item in circuit.data if item.operation.name not in {"barrier", "delay"}
                       for qubit in item.qubits})
    if not qindices or not measured_pairs:
        raise ValueError("no active measured circuit")
    if len(qindices) > 9:
        raise ValueError(f"dense-state guard: {len(qindices)} active qubits exceeds 9")
    if len({q for q, _ in measured_pairs}) != len(measured_pairs):
        raise ValueError("repeated measured quantum wire")
    if len({c for _, c in measured_pairs}) != len(measured_pairs):
        raise ValueError("repeated measured classical destination")
    qmap = {old: new for new, old in enumerate(qindices)}
    compact = QuantumCircuit(len(qindices), circuit.num_clbits)
    compact.global_phase = circuit.global_phase
    compact_measurements = []
    for item in circuit.data:
        operation = item.operation
        if operation.name in {"barrier", "delay"}:
            continue
        qubits = [qmap[circuit.find_bit(qubit).index] for qubit in item.qubits]
        clbits = [circuit.find_bit(bit).index for bit in item.clbits]
        if operation.name == "measure":
            compact_measurements.append((qubits[0], clbits[0]))
            compact.measure(qubits[0], clbits[0])
        elif clbits:
            raise ValueError(f"classically conditioned operation unsupported: {operation.name}")
        else:
            compact.append(operation.copy(), qubits)
    unitary = compact.remove_final_measurements(inplace=False)
    probabilities = Statevector.from_instruction(unitary).probabilities()
    output: dict[int, float] = defaultdict(float)
    for basis_index, probability in enumerate(probabilities):
        if probability <= 1e-14:
            continue
        classical_outcome = 0
        for quantum_index, classical_index in compact_measurements:
            classical_outcome |= ((basis_index >> quantum_index) & 1) << classical_index
        output[classical_outcome] += float(probability)
    return dict(output), len(qindices)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--submitted-zip", type=Path, default=DEFAULT_SUBMITTED)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    rows_path = output_dir / "recovery_rows.csv"
    candidate_manifest_path = output_dir / "candidate_recipe_manifest.json"
    recovery_manifest_path = output_dir / "manifest.json"
    if not all(path.is_file() for path in (rows_path, candidate_manifest_path, recovery_manifest_path)):
        raise FileNotFoundError("run the recovery ledger and MQT candidate materializer first")
    recovery = json.loads(recovery_manifest_path.read_text(encoding="utf-8"))
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if recovery["inputs"].get(str(args.submitted_zip.resolve())) != file_sha(args.submitted_zip):
        raise ValueError("submitted-QASM archive hash does not match the frozen input manifest")
    if recovery["candidate_materialization"]["sha256"] != file_sha(candidate_manifest_path):
        raise ValueError("candidate manifest hash mismatch")
    with rows_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != 4482:
        raise ValueError("expected all 4,482 canonical Qonductor rows")

    candidate_by_pair = {
        (item["family"], int(item["width"])): item
        for item in candidate_manifest["records"] if item.get("status") == "pass"
    }
    selected: OrderedDict[str, dict[str, str]] = OrderedDict()
    for row in rows:
        if row["instance_status"] != "archive_resolved_recipe":
            continue
        digest = row["source_qasm_sha256"]
        width = int(row["archive_logical_width"])
        item = candidate_by_pair.get((row["archive_family"], width))
        if not item:
            raise ValueError(f"no candidate for archive-resolved row {row['canonical_observation_id']}")
        if row["generated_candidate_qasm_sha256"] != item["qasm_sha256"]:
            raise ValueError(f"candidate linkage mismatch at {row['canonical_observation_id']}")
        if digest in selected:
            old = selected[digest]
            if (old["archive_family"], old["archive_logical_width"], old["generated_candidate_path"]) != (
                row["archive_family"], row["archive_logical_width"], row["generated_candidate_path"]
            ):
                raise ValueError(f"one physical QASM hash maps to inconsistent recipes: {digest}")
        else:
            selected[digest] = row

    checks = []
    with zipfile.ZipFile(args.submitted_zip) as archive:
        for digest, row in selected.items():
            physical_bytes = archive.read(row["submitted_qasm_member"])
            if hashlib.sha256(physical_bytes).hexdigest() != digest:
                raise ValueError(f"submitted-QASM byte hash mismatch: {row['submitted_qasm_member']}")
            candidate_path = output_dir / row["generated_candidate_path"]
            logical_circuit = parse_qasm(candidate_path.read_bytes())
            physical_circuit = parse_qasm(physical_bytes)
            logical_distribution, logical_active = ideal_measured_distribution(logical_circuit)
            physical_distribution, physical_active = ideal_measured_distribution(physical_circuit)
            keys = logical_distribution.keys() | physical_distribution.keys()
            total_variation = sum(abs(logical_distribution.get(key, 0.0) - physical_distribution.get(key, 0.0))
                                  for key in keys) / 2.0
            max_absolute = max((abs(logical_distribution.get(key, 0.0) - physical_distribution.get(key, 0.0))
                                for key in keys), default=0.0)
            checks.append({
                "submitted_qasm_sha256": digest,
                "submitted_qasm_member": row["submitted_qasm_member"],
                "family": row["archive_family"],
                "archive_logical_width": int(row["archive_logical_width"]),
                "candidate_qasm_sha256": row["generated_candidate_qasm_sha256"],
                "candidate_path": row["generated_candidate_path"],
                "observation_rows": sum(other["source_qasm_sha256"] == digest for other in rows),
                "physical_active_qubits": physical_active,
                "candidate_active_qubits": logical_active,
                "classical_output_bits": logical_circuit.num_clbits,
                "total_variation_distance": total_variation,
                "max_absolute_probability_difference": max_absolute,
                "status": "pass" if total_variation <= TOLERANCE else "mismatch",
            })

    pair_summary = defaultdict(lambda: {"unique_submitted_qasm": 0, "observation_rows": 0, "pass": 0, "mismatch": 0})
    for check in checks:
        summary = pair_summary[(check["family"], check["archive_logical_width"])]
        summary["unique_submitted_qasm"] += 1
        summary["observation_rows"] += check["observation_rows"]
        summary[check["status"]] += 1
    for item in candidate_manifest["records"]:
        if item.get("status") == "pass":
            item["archive_distribution_check"] = {
                "unique_submitted_qasm": 0,
                "observation_rows": 0,
                "pass": 0,
                "mismatch": 0,
            }
    for (family, width), item in candidate_by_pair.items():
        item["archive_distribution_check"] = dict(pair_summary.get((family, width), {"unique_submitted_qasm": 0, "observation_rows": 0, "pass": 0, "mismatch": 0}))
    family_summary = defaultdict(lambda: {"unique_submitted_qasm": 0, "observation_rows": 0, "pass": 0, "mismatch": 0})
    for (family, _width), summary in pair_summary.items():
        for key, value in summary.items():
            family_summary[family][key] += value
    candidate_manifest["archive_distribution_validation"] = {
        "scope": "ideal measured-output distribution for the all-zero input; not full-unitary/channel identity, uniqueness, noisy equivalence, or runtime-feature equivalence",
        "comparison": "MQT indep candidate versus exact-byte archive-matched submitted QASM",
        "tolerance_total_variation_distance": TOLERANCE,
        "distinct_submitted_qasm_checked": len(checks),
        "observation_rows_covered": sum(check["observation_rows"] for check in checks),
        "pass_count": sum(check["status"] == "pass" for check in checks),
        "mismatch_count": sum(check["status"] != "pass" for check in checks),
        "max_total_variation_distance": max((check["total_variation_distance"] for check in checks), default=None),
        "source_archive_sha256": file_sha(args.submitted_zip),
        "execution": "CPU statevector; active-wire compaction; no noise model; all checked circuits at most 9 active qubits",
        "environment": {"python": sys.version, "qiskit": importlib.metadata.version("qiskit")},
        "validator_sha256": file_sha(Path(__file__).resolve()),
        "family_summary": dict(sorted(family_summary.items())),
        "checks": checks,
    }
    for field in ("archive_distribution_match_status", "archive_distribution_total_variation", "archive_distribution_scope"):
        if field not in fieldnames:
            fieldnames.append(field)
    check_by_hash = {item["submitted_qasm_sha256"]: item for item in checks}
    for row in rows:
        item = check_by_hash.get(row["source_qasm_sha256"])
        row["archive_distribution_match_status"] = item["status"] if item else "not_checked_not_archive_resolved"
        row["archive_distribution_total_variation"] = str(item["total_variation_distance"]) if item else ""
        row["archive_distribution_scope"] = "ideal_output_distribution_from_zero_input_only" if item else ""

    candidate_manifest_path.write_text(json.dumps(candidate_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with rows_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    recovery["output"]["sha256"] = file_sha(rows_path)
    recovery["candidate_materialization"]["sha256"] = file_sha(candidate_manifest_path)
    recovery["candidate_materialization"]["archive_distribution_validation"] = {
        key: value for key, value in candidate_manifest["archive_distribution_validation"].items()
        if key != "checks"
    }
    recovery["outputs"]["candidate_recipe_manifest_sha256"] = file_sha(candidate_manifest_path)
    recovery_manifest_path.write_text(json.dumps(recovery, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in candidate_manifest["archive_distribution_validation"].items()
                      if key not in {"checks", "family_summary"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
