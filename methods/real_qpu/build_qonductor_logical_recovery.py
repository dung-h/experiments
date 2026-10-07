#!/usr/bin/env python3
"""Build a row-level, label-blind Qonductor logical-recovery evidence ledger.

This joins frozen canonical rows to their submitted QASM, the two historical
recovery audits and the already materialized structural DAGs. It does not
infer a recovered historical instance from a width candidate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from qiskit import qasm2, qasm3
from qiskit.circuit.library import ECRGate, RZGate, SXGate

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANONICAL = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
DEFAULT_SUBMITTED = Path("/home/server/Documents/Qonductor-SC25/data/database/circuits.zip")
DEFAULT_BENCHMARKS = Path("/home/server/Documents/Qonductor-SC25/data/benchmarks/benchmarks.zip")
DEFAULT_AUDIT = ROOT / "artifacts/benchmark_v1/qonductor_benchmark_archive_logical_identity_20260928/qonductor_archive_logical_identity_audit.csv"
DEFAULT_FAMILY_AUDIT = ROOT / "artifacts/benchmark_v1/qonductor_logical_reconstruction_20260926/audit.json"
DEFAULT_RECIPE_MANIFEST = ROOT / "artifacts/benchmark_v1/mqt103_recipe_qasm_candidates_v1_20260928/recipe_manifest.json"
DEFAULT_DAG_ROWS = ROOT / "artifacts/benchmark_v1/mali_structural_dag_materialized_v3_20260928/row_manifest.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/real_qpu/logical_recovery"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def qasm_circuit(data: bytes):
    text = data.decode("utf-8-sig")
    if "OPENQASM 3" in text[:256]:
        return qasm3.loads(text)
    native = [
        qasm2.CustomInstruction("rz", 1, 1, RZGate, builtin=True),
        qasm2.CustomInstruction("sx", 0, 1, SXGate, builtin=True),
        qasm2.CustomInstruction("ecr", 0, 2, ECRGate, builtin=True),
    ]
    return qasm2.loads(text, custom_instructions=native)


def qasm_facts(circuit) -> dict:
    registers = {register.name: int(register.size) for register in circuit.cregs}
    pairs: list[tuple[int, str, int]] = []
    used: set[int] = set()
    measured: set[int] = set()
    later_quantum_use = False
    control = False
    reset = False
    for instruction in circuit.data:
        operation = instruction.operation
        name = operation.name
        qindices = [int(circuit.find_bit(bit).index) for bit in instruction.qubits]
        used.update(qindices)
        control |= name in {"if_else", "while_loop", "for_loop", "switch_case"}
        reset |= name == "reset"
        if name == "measure":
            if not instruction.qubits or not instruction.clbits:
                continue
            qindex = int(circuit.find_bit(instruction.qubits[0]).index)
            clbit = instruction.clbits[0]
            location = circuit.find_bit(clbit)
            if not location.registers:
                continue
            register, bit_index = location.registers[0]
            pairs.append((qindex, register.name, int(bit_index)))
            measured.add(qindex)
            continue
        if name in {"barrier", "delay"}:
            continue
        if measured.intersection(qindices):
            later_quantum_use = True
    measurements = [list(pair) for pair in pairs]
    counts = Counter(name for name, _ in circuit.count_ops().items() for _ in range(circuit.count_ops()[name]))
    return {
        "qreg_widths": {register.name: int(register.size) for register in circuit.qregs},
        "classical_register_widths": registers,
        "used_quantum_wires": sorted(used),
        "active_width": len(used),
        "measurement_pairs": measurements,
        "measurement_count": len(pairs),
        "control_flow": control,
        "reset_present": reset,
        "terminal_measurement_only": not later_quantum_use,
        "operations": dict(sorted(counts.items())),
        "parse_status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--submitted-zip", type=Path, default=DEFAULT_SUBMITTED)
    parser.add_argument("--benchmark-zip", type=Path, default=DEFAULT_BENCHMARKS)
    parser.add_argument("--archive-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--family-audit", type=Path, default=DEFAULT_FAMILY_AUDIT)
    parser.add_argument("--recipe-manifest", type=Path, default=DEFAULT_RECIPE_MANIFEST)
    parser.add_argument("--dag-rows", type=Path, default=DEFAULT_DAG_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refuse to overwrite {args.output_dir}")

    for path in (args.canonical, args.submitted_zip, args.benchmark_zip, args.archive_audit,
                 args.family_audit, args.recipe_manifest, args.dag_rows):
        if not path.is_file():
            raise FileNotFoundError(path)

    canonical_all = read_csv(args.canonical)
    canonical = [row for row in canonical_all if row["source_id"] == "qonductor_single_circuit_ibm"]
    if len(canonical) != 4482:
        raise ValueError(f"expected 4482 canonical Qonductor rows, got {len(canonical)}")
    if len({row["canonical_row_id"] for row in canonical}) != len(canonical):
        raise ValueError("duplicate canonical row IDs")
    archive_audit = {int(row["source_row_index"]): row for row in read_csv(args.archive_audit)}
    if set(archive_audit) != {int(row["source_row_index"]) for row in canonical}:
        raise ValueError("archive audit and canonical row-index sets differ")
    family_audit = json.loads(args.family_audit.read_text(encoding="utf-8"))
    family_by_name = {row["family"]: row for row in family_audit["families"]}
    recipe_payload = json.loads(args.recipe_manifest.read_text(encoding="utf-8"))
    if recipe_payload.get("source_dist") != "mqt.bench==1.0.3":
        raise ValueError("existing recipe manifest is not pinned to MQT Bench 1.0.3")
    recipes = {(row["family"], int(row["width"])): row for row in recipe_payload["records"]}

    dag_rows = {}
    for row in read_csv(args.dag_rows):
        if row["source_id"] == "qonductor_archive_resolved_logical_recipe":
            dag_rows[int(row["source_row_index"])] = row
    if len(dag_rows) != 230:
        raise ValueError(f"expected 230 existing Qonductor DAG rows, got {len(dag_rows)}")

    benchmark_hashes: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(args.benchmark_zip) as archive:
        for name in archive.namelist():
            if name.endswith(".qasm"):
                benchmark_hashes[sha(archive.read(name))].append(name)

    parsed_by_hash: dict[str, dict] = {}
    parse_errors: Counter[str] = Counter()
    source_qasm: dict[int, bytes] = {}
    with zipfile.ZipFile(args.submitted_zip) as submitted:
        for row in canonical:
            index = int(row["source_row_index"])
            member = row["qasm_path_or_member"]
            data = submitted.read(member)
            actual_hash = sha(data)
            if actual_hash != row["qasm_bytes_sha256"]:
                raise ValueError(f"submitted QASM hash mismatch at source row {index}")
            source_qasm[index] = data
            if actual_hash not in parsed_by_hash:
                try:
                    facts = qasm_facts(qasm_circuit(data))
                    facts["benchmark_zip_exact_members"] = sorted(benchmark_hashes.get(actual_hash, []))
                except Exception as exc:
                    facts = {"parse_status": "error", "parse_error": f"{type(exc).__name__}: {exc}",
                             "benchmark_zip_exact_members": sorted(benchmark_hashes.get(actual_hash, []))}
                    parse_errors[type(exc).__name__] += 1
                parsed_by_hash[actual_hash] = facts

    records: list[dict[str, str]] = []
    for row in canonical:
        index = int(row["source_row_index"])
        audit = archive_audit[index]
        facts = parsed_by_hash[row["qasm_bytes_sha256"]]
        family = row["family"]
        old_family = family_by_name.get(family, {})
        recipe_family = audit.get("logical_family_from_archive") or ""
        recipe_width = int(audit["logical_width_from_archive"]) if audit.get("logical_width_from_archive") else None
        # Candidate width rules are applied only to successfully parsed ASTs.
        if facts.get("parse_status") == "ok":
            from qonductor_recovery_rules import candidate_widths
            width_result = candidate_widths(
                family,
                facts["classical_register_widths"],
                [tuple(pair) for pair in facts["measurement_pairs"]],
                terminal_only=facts["terminal_measurement_only"],
                has_control_flow=facts["control_flow"],
                has_reset=facts["reset_present"],
                archive_width=recipe_width,
            )
        else:
            width_result = {"candidate_widths": [], "instance_status": "unresolved", "reason": "qasm_parse_failure"}

        recipe = recipes.get((recipe_family, recipe_width)) if recipe_family and recipe_width is not None else None
        dag = dag_rows.get(index)
        prior = old_family.get("tier", "unclassified_family")
        if audit["archive_match_status"] == "archive_hash_resolved_recipe_reproducible_candidate":
            instance_status = "archive_resolved_recipe"
            method_eligibility = "existing_recipe_dag_adaptation_only"
            evidence = "exact_submitted_qasm_sha_join_to_archive_name_plus_pinned_mqt_recipe_and_existing_graph"
        elif audit["archive_match_status"] == "archive_hash_resolved_structural_only_candidate":
            instance_status = "structural_only"
            method_eligibility = "structural_sensitivity_requires_method_contract"
            evidence = "exact_submitted_qasm_sha_join_to_graphstate_archive_member"
        elif not width_result.get("candidate_widths"):
            instance_status = "unresolved"
            method_eligibility = "physical_qasm_methods_only_pending_review"
            evidence = width_result.get("reason", "no_width_candidate")
        else:
            instance_status = "candidate_unverified"
            method_eligibility = "candidate_only_not_authorized_for_logical_training"
            evidence = width_result["reason"]

        recipe_path = recipe.get("path", "") if recipe else ""
        graph_ref = f"{args.dag_rows.parent}/graphs.pkl#graph_index={dag['graph_index']}" if dag else ""
        qreg_widths = facts.get("qreg_widths", {})
        canonical_fields = {
            "canonical_observation_id": row["canonical_row_id"],
            "source_row_index": str(index),
            "source_qasm_sha256": row["qasm_bytes_sha256"],
            "submitted_qasm_member": row["qasm_path_or_member"],
            "family": family,
            "backend": row["backend"],
            "backend_capacity_qubits": row["width_qubits"],
            "qreg_widths": json.dumps(qreg_widths, sort_keys=True),
            "classical_register_widths": json.dumps(facts.get("classical_register_widths", {}), sort_keys=True),
            "used_quantum_wires": json.dumps(facts.get("used_quantum_wires", [])),
            "active_wire_count": str(facts.get("active_width", "")),
            "measurement_pairs": json.dumps(facts.get("measurement_pairs", []), separators=(",", ":")),
            "measurement_count": str(facts.get("measurement_count", "")),
            "control_flow": str(facts.get("control_flow", "")),
            "reset_present": str(facts.get("reset_present", "")),
            "terminal_measurement_only": str(facts.get("terminal_measurement_only", "")),
            "parse_status": facts["parse_status"],
            "parse_error": facts.get("parse_error", ""),
            "operation_counts": json.dumps(facts.get("operations", {}), sort_keys=True),
            "archive_match_status": audit["archive_match_status"],
            "archive_members_same_hash": audit["archive_members_same_hash"],
            "archive_family": recipe_family,
            "archive_logical_width": str(recipe_width or ""),
            "family_screen_tier": prior,
            "candidate_logical_widths": json.dumps(width_result.get("candidate_widths", [])),
            "candidate_width_reason": width_result.get("reason", ""),
            "instance_status": instance_status,
            "method_eligibility": method_eligibility,
            "evidence_basis": evidence,
            "existing_recipe_candidate_qasm": recipe_path,
            "existing_graph_artifact": graph_ref,
            "existing_graph_structure_id": dag["structure_id"] if dag else "",
            "existing_graph_index": dag["graph_index"] if dag else "",
            "angles_status": "not_recovered" if family in {"qnn", "qaoa", "vqe", "portfolioqaoa", "portfoliovqe"} else "candidate_depends_on_pinned_generator",
            "frozen_group_hash": row["circuit_group_hash"],
            "frozen_primary_fold": row["source_primary_fold"],
            "failure_reason": "",
        }
        records.append(canonical_fields)

    # Same reconstructed logical recipe across different source QASM groups can
    # expose leakage for a future logical-input split. Preserve existing folds.
    recipe_groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for item in records:
        if item["archive_family"] and item["archive_logical_width"]:
            key = (item["archive_family"], item["archive_logical_width"], item["existing_graph_structure_id"])
            recipe_groups[key].add(item["frozen_group_hash"])
    for item in records:
        if item["archive_family"] and item["archive_logical_width"]:
            key = (item["archive_family"], item["archive_logical_width"], item["existing_graph_structure_id"])
            item["logical_recipe_distinct_physical_groups"] = str(len(recipe_groups[key]))
        else:
            item["logical_recipe_distinct_physical_groups"] = ""

    counts = Counter(item["instance_status"] for item in records)
    parse_counts = Counter(item["parse_status"] for item in records)
    widths = Counter(width for item in records for width in json.loads(item["candidate_logical_widths"]))
    preflight = defaultdict(Counter)
    seen_signatures: dict[str, set[str]] = defaultdict(set)
    for item in records:
        signature = (item["family"], item["parse_status"], item["control_flow"], item["reset_present"], item["terminal_measurement_only"], item["candidate_width_reason"])
        key = str(signature)
        preflight[key]["rows"] += 1
        if item["source_qasm_sha256"] not in seen_signatures[key]:
            preflight[key]["distinct_qasm"] += 1
            seen_signatures[key].add(item["source_qasm_sha256"])

    args.output_dir.mkdir(parents=True)
    output_csv = args.output_dir / "recovery_rows.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        "artifact_id": "qonductor-logical-recovery",
        "scope": "row-level label-blind evidence ledger; no logical training or score replacement",
        "status": "pass_with_candidate_and_unresolved_rows",
        "rows": len(records),
        "distinct_submitted_qasm_sha256": len(parsed_by_hash),
        "instance_status_counts": dict(sorted(counts.items())),
        "parse_status_counts": dict(sorted(parse_counts.items())),
        "candidate_width_histogram": {str(k): v for k, v in sorted(widths.items())},
        "archive_exact_match_counts": dict(sorted(Counter(item["archive_match_status"] for item in records).items())),
        "preflight_signatures": {key: dict(value) for key, value in sorted(preflight.items())},
        "parse_errors_by_type": dict(sorted(parse_errors.items())),
        "existing_materialized_qonductor_recipe_rows_reused": sum(bool(item["existing_graph_artifact"]) for item in records),
        "existing_qpack_rows_reused": 3945,
        "qpack_reference": str(ROOT / "artifacts/benchmark_v1/mali_structural_dag_materialized_v3_20260928/row_manifest.csv"),
        "qpack_encoder_compatibility": "historical angle-insensitive DAG; not automatically compatible with full-feature Ma-Li encoder",
        "inputs": {str(path): file_sha(path) for path in (args.canonical, args.submitted_zip, args.benchmark_zip, args.archive_audit, args.family_audit, args.recipe_manifest, args.dag_rows)},
        "environment": {"python": sys.version, "platform": platform.platform(), "qiskit": __import__("qiskit").__version__, "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "generator": {"version": "1.0.3", "wheel_sha256": "42affe45f6e6c63d0bdc37093399622ce7c53232c9f486fdc43f345d3b08d30c", "new_qasm_generation": "not run; this builder inventories submitted ASTs and reuses the previously materialized 230 archive-resolved DAGs and recipe QASMs"},
        "limits": [
            "family-level reproducibility counts are not per-row recovered-instance counts",
            "candidate width is not logical-instance identity",
            "QASM archive matches are exact-byte matches only",
            "no semantic equivalence proof was attempted for unmatched rows",
            "existing QPack/Qonductor DAGs use a historical angle-insensitive encoder and are not a full-feature Ma-Li fit authorization",
            "no split, label, raw artifact, CURRENT or prior prediction was modified"
        ],
        "output": {"path": output_csv.name, "sha256": file_sha(output_csv)},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
