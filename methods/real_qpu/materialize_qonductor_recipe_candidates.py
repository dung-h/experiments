#!/usr/bin/env python3
"""Materialize pinned MQT Bench independent-level recipe candidates.

These QASMs are candidate inputs, not claims about the historical Qonductor
instances. The script updates only the derived logical_recovery artifact.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts/real_qpu/logical_recovery"
GENERATION_ELIGIBLE = {
    "ae", "dj", "ghz", "grover-noancilla", "qft", "qftentangled",
    "qpeexact", "qpeinexact", "qwalk-noancilla", "random",
    "realamprandom", "su2random", "twolocalrandom", "wstate",
}
UNRESOLVED_INSTANCE_FAMILIES = {
    "graphstate", "qnn", "qaoa", "vqe", "portfolioqaoa", "portfoliovqe",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def recovery_code_hashes() -> dict[str, str]:
    script_dir = Path(__file__).resolve().parent
    return {
        name: file_sha(script_dir / name)
        for name in (
            "build_qonductor_logical_recovery.py",
            "materialize_qonductor_recipe_candidates.py",
            "validate_qonductor_archive_recipes.py",
            "qonductor_recovery_rules.py",
        )
    }


def source_return_literal(source: str, function_name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            returns = [child for child in ast.walk(node) if isinstance(child, ast.Return) and child.value is not None]
            if len(returns) == 1:
                return ast.literal_eval(returns[0].value)
    raise ValueError(f"could not extract one literal return from {function_name}")


def install_mqt_source_shims(mqt_root: Path):
    """Load MQT's actual generator/helper modules, bypassing unrelated optional stacks."""
    bench_root = mqt_root / "bench"
    utils_path = bench_root / "utils.py"
    utils_source = utils_path.read_text(encoding="utf-8")

    mqt = types.ModuleType("mqt")
    mqt.__path__ = [str(mqt_root)]
    bench = types.ModuleType("mqt.bench")
    bench.__path__ = [str(bench_root)]
    benchmarks = types.ModuleType("mqt.bench.benchmarks")
    benchmarks.__path__ = [str(bench_root / "benchmarks")]
    utils = types.ModuleType("mqt.bench.utils")
    for function_name in (
        "get_openqasm_gates", "get_supported_benchmarks", "get_supported_levels",
        "get_supported_compilers", "get_supported_gatesets", "get_supported_devices",
    ):
        setattr(utils, function_name, lambda function_name=function_name: source_return_literal(utils_source, function_name))

    def get_module_for_benchmark(name: str):
        if name == "qnn":
            module_name = "qiskit_application_ml.qnn"
        elif name in {"portfolioqaoa", "portfoliovqe"}:
            module_name = "qiskit_application_finance." + name
        else:
            module_name = name
        return importlib.import_module("mqt.bench.benchmarks." + module_name)

    utils.get_module_for_benchmark = get_module_for_benchmark
    bench.utils = utils
    # The MQT top-level package imports both Qiskit and pytket helpers. This task
    # calls only the Qiskit path; a stub prevents an unrelated pytket dependency.
    tket_helper = types.ModuleType("mqt.bench.tket_helper")
    sys.modules.update({
        "mqt": mqt,
        "mqt.bench": bench,
        "mqt.bench.utils": utils,
        "mqt.bench.benchmarks": benchmarks,
        "mqt.bench.tket_helper": tket_helper,
    })
    generator = importlib.import_module("mqt.bench.benchmark_generator")
    return generator.get_benchmark, {
        "utils.py": file_sha(utils_path),
        "qiskit_helper.py": file_sha(bench_root / "qiskit_helper.py"),
        "benchmark_generator.py": file_sha(bench_root / "benchmark_generator.py"),
        "helper_shim": "AST-extracted literal utility lists and source module resolver; actual MQT benchmark_generator.get_benchmark and qiskit_helper.get_indep_level are executed; unrelated pytket import stubbed",
    }


def qasm2_roundtrip(qasm: str):
    from qiskit import QuantumCircuit
    return QuantumCircuit.from_qasm_str(qasm)


def unitary_roundtrip_equivalent(original, parsed) -> bool:
    """Bounded semantic check of QASM serialization, not historical identity."""
    from qiskit.quantum_info import Operator

    left = original.remove_final_measurements(inplace=False)
    right = parsed.remove_final_measurements(inplace=False)
    return Operator(left).equiv(Operator(right))


def refresh_candidate_analysis(output_dir: Path) -> int:
    """Rebuild row-level frozen-fold diagnostics without regenerating QASMs."""
    import csv

    output_dir = output_dir.resolve()
    rows_path = output_dir / "recovery_rows.csv"
    recovery_manifest_path = output_dir / "manifest.json"
    candidates_path = output_dir / "candidate_recipe_manifest.json"
    if not all(path.is_file() for path in (rows_path, recovery_manifest_path, candidates_path)):
        raise FileNotFoundError("candidate QASMs and both manifests must exist before --refresh-analysis")
    recovery = json.loads(recovery_manifest_path.read_text(encoding="utf-8"))
    candidate_manifest = json.loads(candidates_path.read_text(encoding="utf-8"))
    if recovery.get("candidate_materialization", {}).get("sha256") != file_sha(candidates_path):
        raise ValueError("candidate manifest hash differs from recovery manifest")
    for item in candidate_manifest.get("records", []):
        if item.get("status") == "pass":
            candidate_path = output_dir / item["path"]
            if not candidate_path.is_file() or file_sha(candidate_path) != item["qasm_sha256"]:
                raise ValueError(f"candidate QASM path/hash check failed: {item.get('path')}")
    if recovery.get("rows") != 4482:
        raise ValueError("unexpected canonical denominator")
    candidate_manifest["source_distribution"]["installed_wheel_bytes_independently_hashed"] = False
    candidate_manifest["source_distribution"]["wheel_digest_authority"] = "pin copied from the recovery contract; executed module files were independently hash-checked"
    with rows_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != 4482:
        raise ValueError("recovery ledger row count changed")

    by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    candidate_names: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        digest = row.get("generated_candidate_qasm_sha256", "")
        if digest:
            by_candidate[digest].append(row)
    for item in candidate_manifest["records"]:
        if item.get("status") == "pass":
            candidate_names[item["qasm_sha256"]].add((item["family"], int(item["width"])))
    candidate_by_pair = {
        (item["family"], int(item["width"])): item
        for item in candidate_manifest["records"] if item.get("status") == "pass"
    }
    linked_rows = 0
    for row in rows:
        widths = json.loads(row["candidate_logical_widths"])
        digest = row.get("generated_candidate_qasm_sha256", "")
        path = row.get("generated_candidate_path", "")
        if digest:
            if len(widths) != 1:
                raise ValueError(f"row with generated candidate does not have one width: {row['canonical_observation_id']}")
            item = candidate_by_pair.get((row["family"], int(widths[0])))
            if not item or item["qasm_sha256"] != digest or item["path"] != path:
                raise ValueError(f"row/candidate recipe mapping mismatch: {row['canonical_observation_id']}")
            linked_rows += 1
        elif path:
            raise ValueError(f"candidate path present without hash: {row['canonical_observation_id']}")
    if linked_rows != recovery.get("candidate_materialization", {}).get("row_hypothesis_count"):
        raise ValueError("candidate-to-row denominator differs from recovery manifest")

    diagnostic_records = []
    cross_fold_groups = 0
    cross_fold_rows = 0
    potential_group_collisions = 0
    candidate_input_physical_group_total = 0
    for digest, group in sorted(by_candidate.items()):
        folds = sorted({row["frozen_primary_fold"] for row in group})
        physical_groups = {row["frozen_group_hash"] for row in group if row["frozen_group_hash"]}
        source_qasms = {row["source_qasm_sha256"] for row in group}
        backend_values = sorted({row["backend"] for row in group})
        crosses = len(folds) > 1
        cross_fold_groups += int(crosses)
        cross_fold_rows += len(group) if crosses else 0
        potential_group_collisions += int(len(physical_groups) > 1)
        candidate_input_physical_group_total += len(physical_groups)
        diagnostic_records.append({
            "qasm_sha256": digest,
            "family_width_candidates_with_identical_bytes": [
                {"family": family, "width": width} for family, width in sorted(candidate_names[digest])
            ],
            "observation_rows": len(group),
            "source_qasm_hashes": len(source_qasms),
            "frozen_physical_circuit_groups": len(physical_groups),
            "backends": backend_values,
            "frozen_primary_folds": folds,
            "crosses_frozen_primary_folds": crosses,
            "interpretation": "candidate-reuse warning only; generated recipe is not verified historical logical identity",
        })
    diagnostic = {
        "candidate_qasm_hash_groups": len(by_candidate),
        "distinct_family_width_candidates": sum(1 for item in candidate_manifest["records"] if item.get("status") == "pass"),
        "byte_identical_candidate_groups_across_family_width": sum(len(names) > 1 for names in candidate_names.values()),
        "candidate_groups_representing_multiple_submitted_physical_groups": potential_group_collisions,
        "sum_candidate_to_physical_group_links": candidate_input_physical_group_total,
        "candidate_groups_spanning_frozen_primary_folds": cross_fold_groups,
        "rows_in_candidate_groups_spanning_frozen_primary_folds": cross_fold_rows,
        "current_frozen_splits_modified": False,
        "leakage_claim": "not established for current benchmark; a candidate QASM is a family-width hypothesis, not a verified logical identity. If used as a logical input/group, these cross-fold collisions require a new logical-group split audit before training.",
        "groups": diagnostic_records,
    }
    candidate_manifest["frozen_split_diagnostic"] = diagnostic
    for field in (
        "candidate_recipe_distinct_physical_groups",
        "candidate_recipe_frozen_primary_folds",
        "candidate_recipe_crosses_frozen_fold",
    ):
        if field not in fieldnames:
            fieldnames.append(field)
    for row in rows:
        digest = row.get("generated_candidate_qasm_sha256", "")
        group = by_candidate.get(digest, []) if digest else []
        folds = sorted({entry["frozen_primary_fold"] for entry in group})
        physical_groups = {entry["frozen_group_hash"] for entry in group if entry["frozen_group_hash"]}
        row["candidate_recipe_distinct_physical_groups"] = str(len(physical_groups)) if digest else ""
        row["candidate_recipe_frozen_primary_folds"] = json.dumps(folds) if digest else "[]"
        row["candidate_recipe_crosses_frozen_fold"] = str(len(folds) > 1).lower() if digest else ""
    candidates_path.write_text(json.dumps(candidate_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with rows_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    recovery["output"]["sha256"] = file_sha(rows_path)
    code_hashes = recovery_code_hashes()
    recovery["code_sha256"] = code_hashes
    candidate_manifest["code_sha256"] = {
        name: digest for name, digest in code_hashes.items()
        if name != "build_qonductor_logical_recovery.py"
    }
    candidates_path.write_text(json.dumps(candidate_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    recovery["candidate_materialization"]["sha256"] = file_sha(candidates_path)
    recovery["candidate_materialization"]["frozen_split_candidate_hypothesis_diagnostic"] = {
        key: value for key, value in diagnostic.items() if key != "groups"
    }
    recovery["outputs"]["candidate_recipe_manifest_sha256"] = file_sha(candidates_path)
    recovery_manifest_path.write_text(json.dumps(recovery, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in diagnostic.items() if key != "groups"}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mqt-root", type=Path, default=None,
                        help="installed mqt/ directory; defaults to mqt.bench distribution location")
    parser.add_argument("--refresh-analysis", action="store_true",
                        help="recompute candidate/frozen-fold diagnostics from already generated QASMs")
    args = parser.parse_args()
    if args.refresh_analysis:
        return refresh_candidate_analysis(args.output_dir)
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("set PYTHONHASHSEED=0 before Python starts for reproducible generator ordering")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(key)
        if value and (not value.isdigit() or int(value) > 4):
            raise SystemExit(f"{key} must be unset or <=4; got {value!r}")

    output_dir = args.output_dir.resolve()
    rows_path = output_dir / "recovery_rows.csv"
    manifest_path = output_dir / "manifest.json"
    if not rows_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("run build_qonductor_logical_recovery.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_id") != "qonductor-logical-recovery" or manifest.get("rows") != 4482:
        raise ValueError("unexpected recovery artifact identity or row count")
    if manifest.get("candidate_materialization"):
        raise SystemExit("candidate materialization already recorded; refusing a second write")
    if (output_dir / "candidate_recipes").exists() or (output_dir / "candidate_recipe_manifest.json").exists():
        raise SystemExit("candidate output already exists; refusing overwrite")

    with rows_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != 4482 or len({row["canonical_observation_id"] for row in rows}) != 4482:
        raise ValueError("recovery row ledger identity/denominator check failed")
    if file_sha(rows_path) != manifest["output"]["sha256"]:
        raise ValueError("recovery_rows.csv changed since its manifest was built")

    family_width_rows: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    unsupported_candidate_rows = Counter()
    for row in rows:
        widths = json.loads(row["candidate_logical_widths"])
        family = row["family"]
        if family in GENERATION_ELIGIBLE:
            for width in widths:
                family_width_rows[(family, int(width))].append(row)
        elif widths:
            unsupported_candidate_rows[family] += 1
    if not family_width_rows:
        raise ValueError("no eligible family-width candidates found")

    if args.mqt_root:
        mqt_root = args.mqt_root.resolve()
    else:
        mqt_root = Path(importlib.metadata.distribution("mqt.bench").locate_file("mqt")).resolve()
    if importlib.metadata.version("mqt.bench") != "1.0.3":
        raise ValueError("requires mqt.bench==1.0.3")
    qiskit_version = importlib.metadata.version("qiskit-terra")
    if not qiskit_version.startswith("0.24."):
        raise ValueError(f"requires qiskit-terra 0.24.x, got {qiskit_version}")
    get_benchmark, source_hashes = install_mqt_source_shims(mqt_root)
    family_source_paths = {}
    for family, _ in family_width_rows:
        module_name = family.split("-", 1)[0]
        family_source_paths[family] = mqt_root / "bench" / "benchmarks" / f"{module_name}.py"
    family_source_hashes = {}
    for family, source_path in sorted(family_source_paths.items()):
        if not source_path.is_file():
            raise FileNotFoundError(f"pinned MQT source module missing for {family}: {source_path}")
        family_source_hashes[family] = file_sha(source_path)
    recovery_contract = json.loads((ROOT / "benchmark_v1/protocol/qonductor_logical_recovery.json").read_text(encoding="utf-8"))
    for family, actual_hash in family_source_hashes.items():
        source_key = family.split("-", 1)[0]
        expected_hash = recovery_contract["source_sha256"].get(source_key)
        if actual_hash != expected_hash:
            raise ValueError(f"pinned generator module hash mismatch for {family}")

    from qiskit import __qiskit_version__, qasm2
    from qiskit.quantum_info import Operator
    generated: dict[tuple[str, int], dict] = {}
    payloads: dict[str, bytes] = {}
    for family, width in sorted(family_width_rows):
        record = {
            "family": family,
            "width": width,
            "status": "error",
            "identity_claim": "candidate_recipe_only_not_historical_instance",
        }
        try:
            circuit = get_benchmark(family, level="indep", circuit_size=width, compiler="qiskit")
            if circuit.num_qubits != width:
                raise ValueError(f"generator returned width {circuit.num_qubits}, expected {width}")
            qasm = circuit.qasm()
            parsed = qasm2_roundtrip(qasm)
            if parsed.num_qubits != circuit.num_qubits:
                raise ValueError("QASM round-trip changed qubit width")
            if circuit.num_qubits <= 9:
                equivalent = unitary_roundtrip_equivalent(circuit, parsed)
                if not equivalent:
                    raise ValueError("dense unitary round-trip equivalence failed")
                semantic_check = "qasm_roundtrip_unitary_equivalent_up_to_global_phase"
            else:
                semantic_check = "not_run_width_over_9_parse_and_structure_only"
            relpath = f"candidate_recipes/{family}__q{width}.qasm"
            data = qasm.encode("utf-8")
            payloads[relpath] = data
            record.update({
                "status": "pass",
                "path": relpath,
                "qasm_sha256": sha(data),
                "qasm_bytes": len(data),
                "num_qubits": circuit.num_qubits,
                "depth": circuit.depth(),
                "operation_counts": dict(sorted((str(name), int(count)) for name, count in circuit.count_ops().items())),
                "semantic_check": semantic_check,
                "semantic_scope": "candidate export/import only; no comparison to historical submitted circuit",
                "affected_rows": len(family_width_rows[(family, width)]),
            })
        except Exception as exc:
            record.update({
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "affected_rows": len(family_width_rows[(family, width)]),
            })
        generated[(family, width)] = record

    # Refuse partial bulk outputs: failures are preserved in the manifest but
    # no row is attached to a missing candidate.
    candidate_manifest = {
        "artifact_id": "qonductor-mqt-independent-recipe-candidates",
        "scope": "reproducible MQT independent-level candidate circuits; not recovered historical identities",
        "status": "pass" if all(item["status"] == "pass" for item in generated.values()) else "pass_with_generation_failures",
        "pipeline": "mqt.bench.benchmark_generator.get_benchmark(name, level='indep', circuit_size=width, compiler='qiskit')",
        "source_distribution": {"name": "mqt.bench", "version": importlib.metadata.version("mqt.bench"),
                                "contract_wheel_sha256": json.loads((ROOT / "benchmark_v1/protocol/qonductor_logical_recovery.json").read_text())["generator"]["wheel_sha256"],
                                "installed_wheel_bytes_independently_hashed": False,
                                "wheel_digest_authority": "pin copied from the recovery contract; executed module files are independently hash-checked"},
        "qiskit": {"qiskit_terra": qiskit_version,
                    "qiskit_metadata": {key: None if value is None else str(value)
                                        for key, value in __qiskit_version__.items()}},
        "other_dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "networkx", "joblib")},
        "source_hashes": source_hashes,
        "family_source_hashes": family_source_hashes,
        "environment": {"python": sys.version, "platform": platform.platform(), "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
                        "thread_limits": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "candidate_families": sorted(GENERATION_ELIGIBLE),
        "withheld_families": {family: {"rows_with_width_hypothesis": count, "reason": "historical instance/angles not identifiable from available source evidence" if family in UNRESOLVED_INSTANCE_FAMILIES else "family not in approved generator subset"} for family, count in sorted(unsupported_candidate_rows.items())},
        "candidate_count": len(generated),
        "candidate_success_count": sum(item["status"] == "pass" for item in generated.values()),
        "candidate_failure_count": sum(item["status"] != "pass" for item in generated.values()),
        "row_hypothesis_count": sum(len(value) for value in family_width_rows.values()),
        "identity_claim": "none; candidate generation does not identify the historical optimizer parameters, random graph/seed, backend layout or physical-to-logical mapping",
        "records": [generated[key] for key in sorted(generated)],
    }

    for row in rows:
        widths = json.loads(row["candidate_logical_widths"])
        key = (row["family"], int(widths[0])) if len(widths) == 1 else None
        result = generated.get(key) if key else None
        row["generated_candidate_qasm_sha256"] = result.get("qasm_sha256", "") if result and result["status"] == "pass" else ""
        row["generated_candidate_path"] = result.get("path", "") if result and result["status"] == "pass" else ""
        row["candidate_generation_status"] = (
            "generated_candidate_not_historical_identity" if result and result["status"] == "pass"
            else "candidate_generation_failed" if result
            else "not_generated_instance_or_family_unresolved" if widths
            else "no_unique_width_candidate"
        )
        row["historical_instance_identity_claim"] = "none"
    for field in ("generated_candidate_qasm_sha256", "generated_candidate_path", "candidate_generation_status", "historical_instance_identity_claim"):
        if field not in fieldnames:
            fieldnames.append(field)

    candidate_dir = output_dir / "candidate_recipes"
    candidate_dir.mkdir()
    for relpath, data in payloads.items():
        target = output_dir / relpath
        target.write_bytes(data)
    candidate_manifest_path = output_dir / "candidate_recipe_manifest.json"
    candidate_manifest_path.write_text(json.dumps(candidate_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with rows_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest["candidate_materialization"] = {
        "path": candidate_manifest_path.name,
        "sha256": file_sha(candidate_manifest_path),
        "candidate_count": len(generated),
        "candidate_success_count": candidate_manifest["candidate_success_count"],
        "candidate_failure_count": candidate_manifest["candidate_failure_count"],
        "row_hypothesis_count": candidate_manifest["row_hypothesis_count"],
        "identity_claim": "none; generated recipe QASMs are candidates only",
    }
    manifest["generator"]["new_qasm_generation"] = candidate_manifest["pipeline"]
    manifest["output"]["sha256"] = file_sha(rows_path)
    manifest["outputs"] = {
        "candidate_recipe_manifest_sha256": file_sha(candidate_manifest_path),
        "candidate_qasm_count": len(payloads),
        "candidate_qasm_total_bytes": sum(map(len, payloads.values())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Independent post-write checks ensure every successful record resolves to
    # one extant QASM whose bytes match its manifest and every row remains.
    reread_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    for item in reread_manifest["records"]:
        if item["status"] == "pass":
            candidate_path = output_dir / item["path"]
            if not candidate_path.is_file() or file_sha(candidate_path) != item["qasm_sha256"]:
                raise ValueError(f"candidate output hash/path mismatch: {item['path']}")
    with rows_path.open(newline="", encoding="utf-8") as stream:
        check_rows = list(csv.DictReader(stream))
    if len(check_rows) != 4482 or sum(row["candidate_generation_status"] == "generated_candidate_not_historical_identity" for row in check_rows) != candidate_manifest["row_hypothesis_count"]:
        raise ValueError("row-to-candidate post-write validation failed")
    print(json.dumps({
        "status": candidate_manifest["status"],
        "rows": len(check_rows),
        "candidate_family_width_pairs": len(generated),
        "candidate_success_count": candidate_manifest["candidate_success_count"],
        "candidate_failure_count": candidate_manifest["candidate_failure_count"],
        "rows_with_generated_recipe_candidate": candidate_manifest["row_hypothesis_count"],
        "qasm_files": len(payloads),
        "qasm_total_bytes": sum(map(len, payloads.values())),
        "manifest_sha256": file_sha(candidate_manifest_path),
        "identity_claim": "none",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
