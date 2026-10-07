"""Materialize source-pinned Pasqal analog program and crossed-cell identities.

This preprocessing builds Pulser sequences from the packaged constructors and
records their deterministic descriptors. It does not initialize EMU-MPS, run a
simulation, evaluate fidelity, or fit a runtime formula.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any


PROTOCOL_PATH = "protocol/measurement/pasqal_emu_mps_formula_crossed_pilot.json"
PROGRAMS_PATH = "methods/simulator/analog_programs.py"
REQUIREMENTS_PATH = "requirements-pasqal-preprocess.txt"
FAMILY_NAMES = {"adiabatic_afm", "quench"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _load_program_module(path: Path):
    spec = importlib.util.spec_from_file_location("_publication_pasqal_programs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot_load_packaged_pasqal_program_constructors")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _environment_versions() -> dict[str, Any]:
    versions = {}
    for name in ("numpy", "pulser-core"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python_version": platform.python_version(), "packages": versions}


def _expected_preprocess_environment(manifest: dict[str, Any]) -> dict[str, Any]:
    environments = manifest.get("environments")
    if not isinstance(environments, list):
        raise ValueError("reproduction_manifest_has_no_environments")
    record = next((row for row in environments
                   if isinstance(row, dict)
                   and row.get("environment_id") == "pasqal_source_program_preprocessing"), None)
    if record is None or record.get("status") != "ready":
        raise ValueError("pasqal_preprocessing_environment_not_declared_ready")
    expected = record.get("pins")
    if not isinstance(expected, dict) or not isinstance(expected.get("packages"), dict):
        raise ValueError("pasqal_preprocessing_environment_pins_missing")
    return expected


def _check_preprocess_environment(manifest: dict[str, Any]) -> dict[str, Any]:
    expected = _expected_preprocess_environment(manifest)
    observed = _environment_versions()
    mismatches = []
    if observed["python_version"] != expected.get("python"):
        mismatches.append({"package": "python", "expected": expected.get("python"),
                           "observed": observed["python_version"]})
    for name, version in expected["packages"].items():
        actual = observed["packages"].get(name)
        if actual != version:
            mismatches.append({"package": name, "expected": version, "observed": actual})
    return {"expected": expected, "observed": observed,
            "matches_pins": not mismatches, "mismatches": mismatches}


def materialize(package_root: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Write an immutable program/cell manifest to a new directory outside the package."""
    root = Path(package_root).expanduser().resolve(strict=True)
    index_path = root / "provenance/files.json"
    if index_path.is_symlink() or index_path.parent.is_symlink():
        raise ValueError("package_index_must_not_be_symlink")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = index.get("files")
    if not isinstance(entries, list):
        raise ValueError("package_file_index_has_no_files_list")
    by_path = {row.get("path"): row for row in entries if isinstance(row, dict)}

    def pinned(relative: str) -> Path:
        row = by_path.get(relative)
        if not isinstance(row, dict):
            raise ValueError(f"required_file_not_indexed:{relative}")
        path = root / relative
        if (any(part.is_symlink() for part in [path, *path.parents]
                if part != root and part.is_relative_to(root)) or not path.is_file()):
            raise ValueError(f"required_file_missing_or_symlink:{relative}")
        if path.stat().st_size != row.get("bytes") or _sha256(path) != row.get("sha256"):
            raise ValueError(f"package_file_pin_mismatch:{relative}")
        return path

    protocol_path = pinned(PROTOCOL_PATH)
    reproduction_path = pinned("protocol/reproduction.json")
    programs_path = pinned(PROGRAMS_PATH)
    requirements_path = pinned(REQUIREMENTS_PATH)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("manifest_id") != "pasqal-emu-mps-formula-crossed-pilot-v1":
        raise ValueError("unexpected_pasqal_crossed_grid_protocol")
    source = protocol.get("source_programs", {})
    simulator = protocol.get("simulator", {})
    quality = protocol.get("quality", {})
    environment_check = _check_preprocess_environment(
        json.loads(reproduction_path.read_text(encoding="utf-8"))
    )
    requirements = requirements_path.read_text(encoding="utf-8").splitlines()
    declared_requirements = {
        line.partition("==")[0]: line.partition("==")[2]
        for line in requirements
        if line.strip() and not line.lstrip().startswith("#") and "==" in line
    }
    if declared_requirements != environment_check["expected"]["packages"]:
        raise ValueError("pasqal_preprocessing_requirements_do_not_match_protocol_pins")
    if not environment_check["matches_pins"]:
        return {
            "artifact_id": "pasqal-emu-mps-source-program-preprocessing",
            "status": "not_ready",
            "materialized": False,
            "environment_is_pinned": False,
            "environment_check": environment_check,
            "reason": "Install the optional pinned source-construction requirements in the declared Python environment.",
        }
    families = source.get("families")
    layouts = source.get("layouts")
    bonds = simulator.get("max_bond_dimensions")
    if (not isinstance(families, list) or set(families) != FAMILY_NAMES
            or len(families) != len(set(families))):
        raise ValueError("pasqal_source_family_set_mismatch")
    if (not isinstance(layouts, list) or len(layouts) != 3
            or any(not isinstance(item, list) or len(item) != 2
                   or any(not isinstance(v, int) or v < 1 for v in item) for item in layouts)
            or len({tuple(item) for item in layouts}) != len(layouts)):
        raise ValueError("pasqal_crossed_layout_set_invalid")
    if (not isinstance(bonds, list) or len(bonds) != 3
            or any(not isinstance(v, int) or v < 1 for v in bonds)
            or len(set(bonds)) != len(bonds)):
        raise ValueError("pasqal_crossed_bond_grid_invalid")
    if (not isinstance(source.get("repository"), str)
            or not isinstance(source.get("commit"), str)
            or not isinstance(source.get("document_sha256"), str)
            or len(source["document_sha256"]) != 64):
        raise ValueError("pasqal_upstream_source_identity_incomplete")

    programs = _load_program_module(programs_path)
    imported_numpy = getattr(getattr(programs, "np", None), "__version__", None)
    expected_numpy = environment_check["expected"]["packages"].get("numpy")
    if imported_numpy != expected_numpy:
        environment_check["matches_pins"] = False
        environment_check["mismatches"].append({
            "package": "numpy_imported_module",
            "expected": expected_numpy,
            "observed": imported_numpy,
        })
        return {
            "artifact_id": "pasqal-emu-mps-source-program-preprocessing",
            "status": "not_ready",
            "materialized": False,
            "environment_is_pinned": False,
            "environment_check": environment_check,
            "reason": "The imported NumPy module does not match its installed distribution pin.",
        }
    protocol_hash = _sha256(protocol_path)
    programs_hash = _sha256(programs_path)
    program_records: dict[str, dict[str, Any]] = {}
    cells = []
    for family in families:
        for rows, columns in layouts:
            sequence = programs.source_sequence(family, rows, columns)
            duration_ns = int(sequence.get_duration())
            if duration_ns <= 0:
                raise ValueError(f"invalid_source_sequence_duration:{family}:{rows}x{columns}")
            program_identity = {"protocol_sha256": protocol_hash, "constructor_sha256": programs_hash,
                                "upstream": source, "family": family, "rows": rows,
                                "columns": columns, "sequence_duration_ns": duration_ns}
            program_id = _canonical_sha256(program_identity)
            program_records[program_id] = {"program_id": program_id, **program_identity,
                                           "qubit_count": rows * columns,
                                           "num_steps_estimate": int(round(duration_ns / float(simulator["dt_ns"]))) }
            for bond in bonds:
                cell_identity = {"program_id": program_id, "max_bond_dimension": bond,
                                 "precision": simulator["precision"],
                                 "max_krylov_dim": simulator["max_krylov_dim"],
                                 "interaction_cutoff": simulator["interaction_cutoff"],
                                 "optimize_qubit_ordering": simulator["optimize_qubit_ordering"]}
                cells.append({"cell_id": _canonical_sha256(cell_identity), **cell_identity})

    if len(program_records) != 6 or len(cells) != 18:
        raise ValueError("pasqal_crossed_grid_expected_six_programs_and_eighteen_cells")
    receipt = {
        "artifact_id": "pasqal-emu-mps-source-program-preprocessing",
        "status": "complete",
        "protocol_id": protocol["manifest_id"],
        "protocol_sha256": protocol_hash,
        "constructor_sha256": programs_hash,
        "source_repository": source["repository"],
        "source_commit": source["commit"],
        "source_document_sha256": source["document_sha256"],
        "environment_check": environment_check,
        "environment_is_pinned": True,
        "program_count": len(program_records),
        "cell_count": len(cells),
        "quality_contract": quality,
        "programs": sorted(program_records.values(), key=lambda row: (row["family"], row["rows"], row["columns"])),
        "cells": sorted(cells, key=lambda row: (row["program_id"], row["max_bond_dimension"])),
        "source_programs_constructed": True,
        "materialized": True,
        "timing_performed": False,
        "simulation_performed": False,
        "training_performed": False,
    }

    requested = Path(os.path.abspath(os.fspath(output_dir)))
    if requested.name in {"", ".", ".."}:
        raise ValueError("output_directory_name_invalid")
    parent = requested.parent.resolve(strict=True)
    output = parent / requested.name
    if output == root or output.is_relative_to(root):
        raise ValueError("output_directory_must_be_outside_publication_package")
    if output.exists() or output.is_symlink():
        raise FileExistsError("output_directory_must_not_exist")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        target = staging / "program_manifest.json"
        target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise FileExistsError("output_directory_created_during_preprocessing")
        os.rename(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    receipt["output_dir"] = str(output)
    receipt["output_sha256"] = _sha256(output / "program_manifest.json")
    return receipt
