"""Metadata-only preflight for cached simulator refit inputs.

This module verifies packaged inputs; it never imports a method runner or
measurement engine and never trains, predicts, or measures. A successful
input check is not a claim that a family has a portable refit adapter.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping


FAMILY_RUNNERS = {
    "azizov_aer_predictors": "methods/simulator/run_azizov_common_core_adaptation.py",
    "cudaq_dense_statevector_fp32_fp64": "methods/simulator/run_cudaq_dense_classical_predictors.py",
    "cudaq_fixed_mps_runtime_and_residual": "methods/simulator/run_fixed_mps_predictors.py",
    "cudaq_joint_family_aware_mps": "methods/simulator/run_joint_mps_predictors.py",
    "maestro_qcsim_component": "methods/simulator/run_maestro_cpu_component.py",
}
ENGINE_MODULES = {"cudaq", "qiskit", "qiskit_aer", "aer", "cuquantum", "cutensornet", "maestro"}
FILE_INDEX = "provenance/files.json"
AZ_REPRESENTATION = "data/simulator/aer/azizov/representation_manifest.json"


def _load_dispatcher(package_root: Path):
    """Load the stdlib-only shared dispatcher lazily, without importing engines."""
    path = package_root / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("_publication_benchmark_dispatcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared preflight helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_index(package_root: Path, dispatcher) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = dispatcher.safe_join(package_root, FILE_INDEX)
    index = json.loads(path.read_text(encoding="utf-8"))
    entries = index.get("files")
    if not isinstance(entries, list):
        raise ValueError("package file index has no files list")
    by_path: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("malformed package file-index entry")
        if entry["path"] in by_path:
            raise ValueError(f"duplicate package file-index entry: {entry['path']}")
        by_path[entry["path"]] = entry
    return by_path, index


def _indexed_check(relative: str, expected_sha256: str, package_root: Path,
                   dispatcher, index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Require both the package inventory pin and the dispatcher byte check."""
    entry = index.get(relative)
    if entry is None:
        return {"path": relative, "status": "not_in_package_index"}
    if entry.get("sha256") != expected_sha256:
        return {
            "path": relative,
            "status": "package_index_pin_mismatch",
            "declared_sha256": expected_sha256,
            "indexed_sha256": entry.get("sha256"),
        }
    item = {"path": relative, "sha256": expected_sha256, "status": "packaged", "root_kind": "package"}
    result = dispatcher.check_input(item, package_root, {}, [])
    result["path"] = relative
    if result.get("status") == "verified" and entry.get("bytes") is not None:
        actual_bytes = (package_root / relative).stat().st_size
        if actual_bytes != entry["bytes"]:
            result["status"] = "byte_count_mismatch"
            result["indexed_bytes"] = entry["bytes"]
            result["actual_bytes"] = actual_bytes
    return result


def _row_count(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if header is None:
                return {"format": "csv", "status": "empty", "data_rows": 0}
            count = sum(1 for _ in reader)
        return {"format": "csv", "status": "readable", "columns": len(header), "data_rows": count}
    if suffix == ".jsonl":
        count = 0
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {line_number} is not an object")
                count += 1
        return {"format": "jsonl", "status": "readable", "data_rows": count}
    return {"format": suffix.lstrip(".") or "unknown", "status": "not_counted"}


def _safe_row_count(path: Path) -> dict[str, Any]:
    try:
        return _row_count(path)
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError, ValueError) as exc:
        return {"format": path.suffix.lower().lstrip(".") or "unknown",
                "status": "malformed", "detail": f"{type(exc).__name__}: {exc}"}


def _role(path: str, consumer: str = "") -> str:
    value = f"{path} {consumer}".lower()
    if any(word in value for word in ("fold", "split", "assignment")):
        return "folds"
    if any(word in value for word in ("target", "label", "runtime outcome")):
        return "targets"
    if any(word in value for word in ("calibration", "throughput", "clops", "duration map")):
        return "calibration"
    if any(word in value for word in ("feature", "graph", "input", "qasm")):
        return "features"
    return "supporting"


def _static_runner_check(package_root: Path, family_id: str) -> dict[str, Any]:
    relative = FAMILY_RUNNERS.get(family_id)
    if relative is None:
        return {"status": "runner_not_registered", "engine_imports": [], "path": None}
    path = package_root / relative
    if not path.is_file():
        return {"status": "runner_missing", "engine_imports": [], "path": relative}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    except (OSError, SyntaxError) as exc:
        return {"status": "runner_static_scan_failed", "engine_imports": [], "path": relative, "detail": str(exc)}
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    blocked = sorted(imports & ENGINE_MODULES)
    return {
        "path": relative,
        "status": "measurement_engine_dependency" if blocked else "no_direct_engine_import",
        "engine_imports": blocked,
        "scope": "static direct-import scan only; dynamic imports are not executed",
    }


def _azizov_graphs(package_root: Path, dispatcher,
                    index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    manifest_check = _indexed_check(AZ_REPRESENTATION,
                                    index.get(AZ_REPRESENTATION, {}).get("sha256", ""),
                                    package_root, dispatcher, index)
    result: dict[str, Any] = {"representation_manifest_check": manifest_check, "payloads": {}}
    if manifest_check.get("status") != "verified":
        result["status"] = "blocked_unverified_representation_manifest"
        return result
    manifest = json.loads((package_root / AZ_REPRESENTATION).read_text(encoding="utf-8"))
    payloads = manifest.get("package_graph_payloads")
    if not isinstance(payloads, dict) or payloads.get("schema") != "ordered_jsonl_shards_v1":
        result["status"] = "blocked_missing_ordered_shard_contract"
        return result
    all_ok = True
    for name in ("source_graphs", "transpiled_graphs"):
        group = payloads.get(name)
        if not isinstance(group, dict) or not isinstance(group.get("shards"), list) or not group["shards"]:
            result["payloads"][name] = {"status": "blocked_missing_shards"}
            all_ok = False
            continue
        digest = hashlib.sha256()
        total_bytes = 0
        total_rows = 0
        checks = []
        group_ok = True
        for shard in group["shards"]:
            if not isinstance(shard, dict):
                group_ok = False
                checks.append({"status": "malformed_shard_record"})
                continue
            path = shard.get("path")
            sha = shard.get("sha256")
            if not isinstance(path, str) or not isinstance(sha, str):
                group_ok = False
                checks.append({"status": "malformed_shard_pin"})
                continue
            check = _indexed_check(path, sha, package_root, dispatcher, index)
            checks.append(check)
            if check.get("status") != "verified":
                group_ok = False
                continue
            file_path = package_root / path
            shard_rows = _safe_row_count(file_path)
            if shard_rows.get("format") != "jsonl" or shard_rows.get("data_rows") != shard.get("record_count"):
                group_ok = False
                check["row_check"] = {**shard_rows, "declared_rows": shard.get("record_count")}
            else:
                check["row_check"] = shard_rows
            with file_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
                    total_bytes += len(block)
            total_rows += shard_rows.get("data_rows", 0)
        expected_bytes = group.get("original_bytes")
        expected_sha = group.get("original_sha256")
        if total_bytes != expected_bytes or digest.hexdigest() != expected_sha:
            group_ok = False
        result["payloads"][name] = {
            "status": "verified" if group_ok else "blocked_or_mismatch",
            "ordered_shard_checks": checks,
            "concatenated_bytes": total_bytes,
            "declared_original_bytes": expected_bytes,
            "concatenated_sha256": digest.hexdigest(),
            "declared_original_sha256": expected_sha,
            "record_count": total_rows,
        }
        all_ok = all_ok and group_ok
    result["status"] = "verified" if all_ok else "blocked_or_mismatch"
    return result


def inspect_cached_refit_inputs(
    package_root: Path,
    family_id: str,
    capability_id: str,
    manifest_path: Path | None = None,
    external_roots: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Verify cached-refit prerequisites without importing/running a model.

    ``package_root`` is the publication package root. The output distinguishes
    verified saved inputs from a runnable refit adapter; this API never claims
    that a family is runnable and never performs training, inference or timing.
    """
    package_root = Path(package_root).resolve()
    dispatcher = _load_dispatcher(package_root)
    index, _ = _read_index(package_root, dispatcher)
    canonical_manifest = "protocol/reproduction.json"
    if manifest_path is None:
        manifest_relative = canonical_manifest
        selected_manifest = package_root / canonical_manifest
    else:
        supplied = Path(manifest_path)
        selected_manifest = supplied if supplied.is_absolute() else package_root / supplied
        selected_manifest = selected_manifest.resolve(strict=True)
        try:
            manifest_relative = selected_manifest.relative_to(package_root).as_posix()
        except ValueError as exc:
            raise ValueError("manifest_path must be inside the package root") from exc
    manifest_entry = index.get(manifest_relative)
    if manifest_entry is None or not isinstance(manifest_entry.get("sha256"), str):
        raise ValueError(f"manifest is not hash-pinned in {FILE_INDEX}: {manifest_relative}")
    manifest_pin_check = _indexed_check(manifest_relative, manifest_entry["sha256"],
                                         package_root, dispatcher, index)
    if manifest_pin_check.get("status") != "verified":
        raise ValueError(f"manifest failed package hash verification: {manifest_pin_check}")
    manifest = dispatcher.read_manifest(selected_manifest)
    family = next((row for row in manifest["families"] if row.get("family_id") == family_id), None)
    if family is None:
        raise ValueError(f"unknown family_id: {family_id}")
    capability = next((row for row in family["capabilities"] if row.get("capability_id") == capability_id), None)
    if capability is None:
        raise ValueError(f"unknown capability_id {capability_id!r} for {family_id}")

    roots: dict[str, Path] = {}
    for root_id, root_path in (external_roots or {}).items():
        path = Path(root_path)
        if not isinstance(root_id, str) or not root_id or not path.is_absolute() or not path.is_dir():
            raise ValueError("external_roots must map non-empty IDs to existing absolute directories")
        roots[root_id] = path.resolve()
    preflight = dispatcher.preflight(manifest, family_id, capability_id, package_root, roots)
    checks = []
    for prerequisite, checked in zip(capability.get("prerequisites", []), preflight["inputs"]):
        path = prerequisite.get("path", checked.get("input"))
        checks.append({
            "path": path,
            "role": _role(str(path), str(prerequisite.get("consumer", ""))),
            "status": checked.get("status"),
            "detail": checked.get("detail"),
            "expected_sha256": checked.get("expected_sha256", prerequisite.get("sha256")),
            "actual_sha256": checked.get("actual_sha256"),
            "row_check": _safe_row_count(package_root / path) if checked.get("status") == "verified"
            and isinstance(path, str) and (package_root / path).is_file() else None,
        })

    aer_graphs = None
    if family_id == "azizov_aer_predictors" and capability_id == "frozen_split_refit":
        aer_graphs = _azizov_graphs(package_root, dispatcher, index)
        # Replace only the two historical null-SHA placeholders with checks of
        # the byte-pinned, ordered shard representation; do not infer a match
        # from filenames or weaken any other prerequisite.
        for item in checks:
            if item["path"] in {"data/simulator/aer/azizov/source_graphs.jsonl",
                                "data/simulator/aer/azizov/transpiled_graphs.jsonl"}:
                key = "source_graphs" if item["path"].endswith("source_graphs.jsonl") else "transpiled_graphs"
                group = aer_graphs.get("payloads", {}).get(key, {})
                item["status"] = "verified" if group.get("status") == "verified" else group.get("status", "blocked")
                item["detail"] = "resolved only through hash-pinned ordered shards" if item["status"] == "verified" else "see aer_graph_payloads"
                item["row_check"] = {"format": "jsonl", "status": item["status"], "data_rows": group.get("record_count")}

    fixed_target_check = None
    if (family_id == "cudaq_fixed_mps_runtime_and_residual"
            and capability_id in {"fixed_target_preprocessing", "fixed_and_residual_refit"}):
        reader_path = dispatcher.verify_indexed_file(
            package_root, "methods/simulator/fixed_target_reader.py"
        )
        spec = importlib.util.spec_from_file_location("_publication_fixed_target_reader", reader_path)
        if spec is None or spec.loader is None:
            fixed_target_check = {"status": "reader_unavailable"}
        else:
            reader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reader)
            try:
                bundle = reader.load_fixed_targets(package_root)
                fixed_target_check = {
                    "status": "verified",
                    "target_rows": bundle.get("assigned_hashes"),
                    "source_refit_enabled": bundle.get("source_refit_enabled", False),
                    "reader": "methods/simulator/fixed_target_reader.py",
                }
            except Exception as exc:  # surface fail-closed reader diagnostics
                fixed_target_check = {"status": "blocked", "detail": f"{type(exc).__name__}: {exc}"}

    runner = _static_runner_check(package_root, family_id)
    input_failures = [item for item in checks if item["status"] != "verified"]
    if aer_graphs is not None and aer_graphs.get("status") != "verified":
        input_failures.append({"path": "Aer graph shard payloads", "status": aer_graphs.get("status")})
    if fixed_target_check is not None and fixed_target_check.get("status") != "verified":
        input_failures.append({"path": "fixed MPS target reader", "status": fixed_target_check.get("status")})
    if runner.get("status") != "no_direct_engine_import":
        input_failures.append({"path": runner.get("path"), "status": runner.get("status")})

    if any(item["status"] == "scientifically_unavailable" for item in checks):
        input_status = "scientifically_unavailable"
    else:
        input_status = "verified" if not input_failures else "not_ready"
    return {
        "family_id": family_id,
        "capability_id": capability_id,
        "manifest": {
            "path": manifest_relative,
            "sha256": manifest_pin_check.get("actual_sha256"),
            "authority": "authoritative" if manifest_relative == canonical_manifest else "hash_pinned_non_authoritative_override",
        },
        "declared_capability_status": capability.get("status", "not_ready"),
        "input_status": input_status,
        "inputs": checks,
        "aer_graph_payloads": aer_graphs,
        "fixed_mps_target_reader": fixed_target_check,
        "runner_static_check": runner,
        "runner_adapter_status": "not_provided_by_metadata_preflight",
        "failures": input_failures,
        "execution": {"performed": False, "training": False, "inference": False, "measurement": False},
        "scope": "metadata and byte-integrity preflight only; verified inputs do not imply a runnable refit",
    }
