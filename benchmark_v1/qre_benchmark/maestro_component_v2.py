"""V2 Maestro normalization, calibration reduction, and identity helpers.

This module contains deterministic, untimed contract logic.  It does not run
QCSim.  Timing workers live in the existing calibration/panel entrypoints.
"""
from __future__ import annotations

import enum
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


V2_MANIFEST_ID = "maestro-common-panel-completion-v2"
V2_CLOCK_ID = "maestro_qcsim_process_isolated_reported_execution"
V2_CLOCK_FIELD = "reported_time_seconds"
V2_TIMED_STAGE = "process_isolated_timed_repeat"
V2_UNTIMED_STAGE = "process_isolated_untimed_repeat"

# S5 supersedes v2 for the pending completion experiment.  Keep these
# identifiers distinct so a v2 ledger can never be mistaken for v3 input.
V3_MANIFEST_ID = "maestro-common-panel-completion-v3"
V3_BASE_MANIFEST_ID = V2_MANIFEST_ID
V3_BASE_SHA256 = "60baef053c3583706a767a8bd82e16734aceafbc1327405d13a9913943a22e3d"
V3_TIMED_STAGE = "v3_process_isolated_timed_repeat"
V3_UNTIMED_STAGE = "v3_process_isolated_untimed_call"
V3_INHERIT_SECTIONS = (
    "source", "implementation", "panel", "normalization",
    "panel_candidate_configuration", "quality_policy",
)
V3_OVERLAY_SECTIONS = (
    "schema_version", "manifest_id", "method_id", "reader_label", "variant_id",
    "status", "controlling_decision", "decision_date", "review_role",
    "prior_evidence", "synthetic_corpus", "measurement", "analysis",
    "eligibility", "technical_gates", "outputs", "seed",
)
V3_PANEL_ATTEMPT_STATUS_SCHEMA = {
    "normalization_status": "string",
    "execution_eligibility": ("eligible", "unavailable"),
    "execution_reason": "string",
    "prediction_status": ("available", "unavailable", "out_of_grid"),
    "prediction_reason": "string",
    "timing_status": ("not_started", "ok", "timeout", "error"),
    "timing_reason": "string",
    "quality_status": ("not_required", "not_run", "ok", "quality_failed", "quality_unavailable"),
    "metric_eligibility": "boolean",
    "metric_ineligibility_reason": "string",
}

# S6 is an explicit partial-domain amendment over the immutable S5/v3 base.
# These pins deliberately live beside, rather than replace, the v3 identity.
PARTIAL_MANIFEST_ID = "maestro-common-panel-completion-v3-partial-resume"
PARTIAL_MANIFEST_SHA256 = "6c16608fee8ea664be606d8605da13821be2e07bdcc7874dee44d4700d560db4"
PARTIAL_PARENT_V3_SHA256 = "bc2a3bc150421698ff09e08a2055a40979f96d2965fe2dcb3e20402c1016574d"
PARTIAL_QUARANTINE_REASON = "synthetic_preparation_norm_failure"
PARTIAL_FIT_VERSION = "paired_endpoint_heldout_v3_partial_resume"


class V2Unavailable(ValueError):
    """An input cannot be used under the frozen Maestro v2 contract."""


class V3Unavailable(ValueError):
    """An input cannot be used under the frozen Maestro v3 contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def package_tree_hash(package_root: Path) -> tuple[str, int]:
    """Hash importable Python/native package payloads by relative path."""
    files = sorted(path for path in package_root.rglob("*")
                   if path.is_file() and path.suffix in {".py", ".so", ".pyd", ".dll", ".dylib"}
                   and "__pycache__" not in path.parts)
    material = [(path.relative_to(package_root).as_posix(), sha256_file(path)) for path in files]
    return canonical_hash(material), len(files)


def resource_profile() -> dict[str, Any]:
    """Read-only CPU/RAM/GPU/driver snapshot; CUDA is not used by this runner."""
    cpu_model = platform.processor() or ""
    memory_total_bytes = None
    physical_cores = None
    try:
        cpu_text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        model_match = re.search(r"^model name\s*:\s*(.+)$", cpu_text, re.MULTILINE)
        if model_match:
            cpu_model = model_match.group(1).strip()
        core_pairs = set(re.findall(
            r"^physical id\s*:\s*(\d+)\s*$.*?^cpu cores\s*:\s*(\d+)",
            cpu_text, re.MULTILINE | re.DOTALL,
        ))
        if core_pairs:
            physical_cores = sum(int(cores) for _, cores in core_pairs)
    except OSError:
        pass
    try:
        mem_text = Path("/proc/meminfo").read_text(encoding="ascii", errors="replace")
        match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", mem_text, re.MULTILINE)
        if match:
            memory_total_bytes = int(match.group(1)) * 1024
    except OSError:
        pass
    gpu_inventory: list[dict[str, str]] = []
    driver_version = "unavailable"
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                check=False, capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    columns = [value.strip() for value in line.split(",")]
                    if len(columns) == 3:
                        name, memory_mb, driver = columns
                        gpu_inventory.append({"name": name, "memory_total_mb": memory_mb,
                                              "driver_version": driver})
                        driver_version = driver
            else:
                gpu_inventory.append({"status": "nvidia_smi_query_failed",
                                      "returncode": str(result.returncode)})
        except (OSError, subprocess.TimeoutExpired) as exc:
            gpu_inventory.append({"status": f"nvidia_smi_query_error:{type(exc).__name__}"})
    else:
        gpu_inventory.append({"status": "nvidia_smi_not_installed"})
    try:
        from threadpoolctl import threadpool_info
        native_threadpool_inventory: Any = [
            {key: pool.get(key) for key in (
                "user_api", "internal_api", "num_threads", "prefix", "version",
                "threading_layer", "architecture",
            ) if pool.get(key) is not None}
            for pool in threadpool_info()
        ]
    except ImportError:
        native_threadpool_inventory = "unavailable:threadpoolctl_not_installed"
    except Exception as exc:  # pragma: no cover - platform/library specific
        native_threadpool_inventory = f"unavailable:{type(exc).__name__}"
    return {
        "schema": "maestro_resource_profile_v1",
        "cpu_model": cpu_model or "unreported",
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_core_count": physical_cores,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "memory_total_bytes": memory_total_bytes,
        "gpu_inventory": gpu_inventory,
        "cuda_driver_version": driver_version,
        "cuda_used_for_cpu_timing": False,
        "native_threadpool_inventory": native_threadpool_inventory,
    }


def environment_pins(manifest_path: Path, maestro: Any, code_paths: Iterable[Path]) -> dict[str, Any]:
    """Shared calibration/panel code, package, executable and resource pins."""
    try:
        import qiskit
    except Exception as exc:  # pragma: no cover - environment-specific
        raise V2Unavailable(f"qiskit_unavailable_for_environment_pin:{type(exc).__name__}") from exc
    maestro_root = Path(maestro.__file__).resolve().parent
    qiskit_root = Path(qiskit.__file__).resolve().parent
    py_executable = Path(sys.executable).resolve()
    root = Path(__file__).resolve().parents[2]
    code_hashes = {str(path.resolve().relative_to(root)): sha256_file(path.resolve())
                   for path in sorted(code_paths, key=lambda p: str(p))}
    maestro_hash, maestro_count = package_tree_hash(maestro_root)
    qiskit_hash, qiskit_count = package_tree_hash(qiskit_root)
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "python_executable": str(py_executable),
        "python_executable_sha256": sha256_file(py_executable),
        "python_version": sys.version,
        "platform": platform.platform(),
        "maestro_module": str(Path(maestro.__file__).resolve()),
        "maestro_package_tree_sha256": maestro_hash,
        "maestro_package_file_count": maestro_count,
        "qiskit_module": str(Path(qiskit.__file__).resolve()),
        "qiskit_version": str(qiskit.__version__),
        "qiskit_package_tree_sha256": qiskit_hash,
        "qiskit_package_file_count": qiskit_count,
        "code_hashes": code_hashes,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "native_thread_environment": {name: os.environ.get(name, "") for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
        )},
        "resource_profile": resource_profile(),
    }


def resolve_v3_contract(manifest_path: Path, base_path: Path | None = None) -> dict[str, Any]:
    """Resolve the S5 overlay using only its frozen, allow-listed v2 sections."""
    manifest_path = Path(manifest_path).resolve()
    try:
        overlay = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V3Unavailable(f"v3_manifest_unreadable:{type(exc).__name__}") from exc
    if overlay.get("manifest_id") != V3_MANIFEST_ID or overlay.get("schema_version") != "3.0":
        raise V3Unavailable("v3_manifest_identity_mismatch")

    base_spec = overlay.get("base_contract")
    if not isinstance(base_spec, Mapping):
        raise V3Unavailable("v3_base_contract_missing")
    expected_base_rel = "benchmark_v1/execution/manifests/maestro_common_panel_completion_v2.json"
    if base_spec.get("path") != expected_base_rel:
        raise V3Unavailable("v3_base_contract_path_mismatch")
    if base_spec.get("sha256") != V3_BASE_SHA256:
        raise V3Unavailable("v3_base_contract_declared_hash_mismatch")
    if tuple(base_spec.get("inherit_sections", ())) != V3_INHERIT_SECTIONS:
        raise V3Unavailable("v3_inherit_section_allowlist_mismatch")
    if base_spec.get("unlisted_base_sections_inherited") is not False:
        raise V3Unavailable("v3_unlisted_base_inheritance_not_disabled")
    if base_spec.get("legacy_dispatch_allowed") is not False:
        raise V3Unavailable("v3_legacy_dispatch_not_disabled")

    allowed_top_level = set(V3_OVERLAY_SECTIONS) | {"base_contract"}
    unexpected = sorted(set(overlay) - allowed_top_level)
    missing = sorted(set(V3_OVERLAY_SECTIONS) - set(overlay))
    if unexpected:
        raise V3Unavailable(f"v3_unrecognized_overlay_sections:{unexpected}")
    if missing:
        raise V3Unavailable(f"v3_required_overlay_sections_missing:{missing}")
    overridden = sorted(set(V3_INHERIT_SECTIONS) & set(overlay))
    if overridden:
        raise V3Unavailable(f"v3_inherited_sections_must_not_be_overridden:{overridden}")

    repo_root = Path(__file__).resolve().parents[2]
    selected_base = Path(base_path).resolve() if base_path is not None else repo_root / expected_base_rel
    try:
        base_bytes = selected_base.read_bytes()
        base = json.loads(base_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V3Unavailable(f"v3_base_contract_unreadable:{type(exc).__name__}") from exc
    actual_base_sha = sha256_bytes(base_bytes)
    if actual_base_sha != V3_BASE_SHA256:
        raise V3Unavailable(f"v3_base_contract_hash_mismatch:{actual_base_sha}")
    if base.get("manifest_id") != V3_BASE_MANIFEST_ID:
        raise V3Unavailable("v3_base_manifest_identity_mismatch")
    missing_base = [section for section in V3_INHERIT_SECTIONS if section not in base]
    if missing_base:
        raise V3Unavailable(f"v3_inherited_base_sections_missing:{missing_base}")

    # Copy exactly the named base sections, then the explicitly v3-owned
    # overlay sections. No other base field can leak into the resolved model.
    resolved = {section: copy.deepcopy(base[section]) for section in V3_INHERIT_SECTIONS}
    resolved.update({section: copy.deepcopy(overlay[section]) for section in V3_OVERLAY_SECTIONS})
    resolved["base_contract"] = copy.deepcopy(base_spec)
    resolved["resolution_provenance"] = {
        "v3_manifest_path": "benchmark_v1/execution/manifests/maestro_common_panel_completion_v3.json",
        "v3_manifest_sha256": sha256_file(manifest_path),
        "base_contract_path": expected_base_rel,
        "base_contract_sha256": actual_base_sha,
        "inherited_sections": list(V3_INHERIT_SECTIONS),
        "unlisted_base_sections_inherited": False,
    }
    if resolved["prior_evidence"].get("role") != "diagnostic_only_no_primary_coefficient_reuse":
        raise V3Unavailable("v3_prior_evidence_reuse_policy_mismatch")
    return resolved


def build_v3_run_identity(contract: Mapping[str, Any], pins: Mapping[str, Any],
                          input_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build a v3-only resume identity from resolved contract and execution pins."""
    if contract.get("manifest_id") != V3_MANIFEST_ID:
        raise V3Unavailable("v3_identity_requires_v3_contract")
    if not isinstance(pins, Mapping) or not pins:
        raise V3Unavailable("v3_identity_execution_pins_missing")
    hashes = dict(input_hashes or {})
    for name, digest in hashes.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise V3Unavailable(f"v3_identity_input_hash_invalid:{name}")
    body = {
        "identity_schema": "maestro_v3_calibration_run_identity_v1",
        "manifest_id": V3_MANIFEST_ID,
        "resolved_contract_sha256": canonical_hash(contract),
        "pins": copy.deepcopy(dict(pins)),
        "input_hashes": dict(sorted(hashes.items())),
        "prior_evidence_role": "diagnostic_only_no_primary_coefficient_reuse",
    }
    return {"manifest_id": V3_MANIFEST_ID, "run_identity": body,
            "run_identity_sha256": canonical_hash(body)}


def validate_v3_resume_identity(existing: Mapping[str, Any],
                                expected: Mapping[str, Any]) -> None:
    """Fail closed on v2 ledgers or any changed v3 contract/context/input pin."""
    if existing.get("manifest_id") != V3_MANIFEST_ID:
        raise V3Unavailable("v3_resume_rejects_non_v3_records")
    if expected.get("manifest_id") != V3_MANIFEST_ID:
        raise V3Unavailable("v3_expected_resume_identity_is_not_v3")
    existing_body = existing.get("run_identity")
    expected_body = expected.get("run_identity")
    existing_sha = existing.get("run_identity_sha256")
    expected_sha = expected.get("run_identity_sha256")
    if not isinstance(existing_body, Mapping) or not isinstance(expected_body, Mapping):
        raise V3Unavailable("v3_resume_identity_missing")
    if canonical_hash(existing_body) != existing_sha:
        raise V3Unavailable("v3_existing_resume_identity_hash_mismatch")
    if canonical_hash(expected_body) != expected_sha:
        raise V3Unavailable("v3_expected_resume_identity_hash_mismatch")
    if existing_sha != expected_sha:
        raise V3Unavailable("v3_resume_identity_mismatch")


def _v3_angle(value: float) -> str:
    return f"{float(value):.8f}"


def generate_v3_probe(
    contract: Mapping[str, Any],
    width: int,
    operation_class: str,
    repeats: int = 0,
    *,
    include_measurement: bool = True,
) -> dict[str, Any]:
    """Generate one deterministic S5 background/word probe and provenance."""
    if contract.get("manifest_id") != V3_MANIFEST_ID:
        raise V3Unavailable("v3_probe_requires_v3_resolved_contract")
    n = int(width)
    repeat_count = int(repeats)
    corpus = contract["synthetic_corpus"]
    operation_classes = set(corpus["operation_classes"])
    sampling_classes = {"zero_gate_sample", "entangled_sampling_diagnostic"}
    if n < 2:
        raise V3Unavailable("v3_probe_width_must_be_at_least_two")
    if operation_class not in operation_classes | sampling_classes:
        raise V3Unavailable(f"v3_unknown_operation_class:{operation_class}")
    if operation_class in operation_classes and repeat_count <= 0:
        raise V3Unavailable("v3_operation_repeat_count_must_be_positive")
    if operation_class in sampling_classes and repeat_count != 0:
        raise V3Unavailable("v3_sampling_repeat_count_must_be_zero")

    entangled = operation_class != "zero_gate_sample"
    r = min(5, n // 2) if entangled else 0
    preparation: list[str] = []
    prep_counts = {"rx": 0, "rz": 0, "cx": 0, "h_replaced": 0}
    for i in range(r):
        partner = n - 1 - i
        # This ordered RX/RZ decomposition equals H up to the recorded phase.
        preparation.extend((
            f"rz(pi/2) q[{i}];", f"rx(pi/2) q[{i}];", f"rz(pi/2) q[{i}];",
            f"cx q[{i}],q[{partner}];",
        ))
        prep_counts["rz"] += 2
        prep_counts["rx"] += 1
        prep_counts["cx"] += 1
        prep_counts["h_replaced"] += 1
    if entangled:
        for q in range(n):
            preparation.extend((
                f"rx({_v3_angle(0.41 + 0.007 * q)}) q[{q}];",
                f"rz({_v3_angle(0.23 + 0.011 * q)}) q[{q}];",
            ))
            prep_counts["rx"] += 1
            prep_counts["rz"] += 1
        for layer in range(n):
            for q in range(n):
                preparation.extend((
                    f"rx({_v3_angle(0.19 + 0.003 * layer + 0.007 * q)}) q[{q}];",
                    f"rz({_v3_angle(0.29 + 0.005 * layer + 0.011 * q)}) q[{q}];",
                ))
                prep_counts["rx"] += 1
                prep_counts["rz"] += 1
            for a in range(layer % 2, n - 1, 2):
                preparation.append(f"cx q[{a}],q[{a + 1}];")
                prep_counts["cx"] += 1

    variable: list[str] = []
    variable_counts = {"rx": 0, "rz": 0, "cx": 0}
    for i in range(repeat_count):
        if operation_class == "one_qubit_noncommuting":
            q = i % n
            variable.extend((
                f"rz({_v3_angle(0.013 + 0.0001 * i)}) q[{q}];",
                f"rx({_v3_angle(0.021 + 0.0001 * i)}) q[{q}];",
            ))
            variable_counts["rz"] += 1
            variable_counts["rx"] += 1
        elif operation_class in {"two_qubit_wrapper_control", "two_qubit_cx_interleaved"}:
            a = i % (n - 1)
            b = a + 1
            variable.append(f"rx({_v3_angle(0.017 + 0.0001 * i)}) q[{a}];")
            variable_counts["rx"] += 1
            if operation_class == "two_qubit_cx_interleaved":
                variable.append(f"cx q[{a}],q[{b}];")
                variable_counts["cx"] += 1
            variable.append(f"rz({_v3_angle(0.031 + 0.0001 * i)}) q[{b}];")
            variable_counts["rz"] += 1

    header = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];", f"creg c[{n}];"]
    body = [*preparation, *variable]
    if include_measurement:
        body.append("measure q -> c;")
    program = "\n".join([*header, *body, ""])
    prep_text = "\n".join([*preparation, ""])
    variable_text = "\n".join([*variable, ""])
    whole_counts = {
        gate: prep_counts.get(gate, 0) + variable_counts.get(gate, 0)
        for gate in ("rx", "rz", "cx")
    }
    if include_measurement:
        whole_counts["measure"] = n
    return {
        "program": program,
        "preparation_qasm": prep_text,
        "variable_word_qasm": variable_text,
        "preparation_qasm_sha256": sha256_bytes(prep_text.encode("utf-8")),
        "variable_word_qasm_sha256": sha256_bytes(variable_text.encode("utf-8")),
        "whole_qasm_sha256": sha256_bytes(program.encode("utf-8")),
        "preparation_counts": prep_counts,
        "variable_word_counts": variable_counts,
        "whole_qasm_counts": whole_counts,
        "preparation_h_pairs": r,
        "preparation_global_phase_radians": -math.pi * r / 2.0,
        "width": n,
        "operation_class": operation_class,
        "operation_repeats": repeat_count,
        "entangled_preparation": entangled,
    }


def build_v3_plan(contract: Mapping[str, Any], include_diagnostics: bool = True) -> list[dict[str, Any]]:
    """Materialize the 420 primary and optional 12 diagnostic S5 cells."""
    if contract.get("manifest_id") != V3_MANIFEST_ID:
        raise V3Unavailable("v3_planner_requires_v3_resolved_contract")
    corpus = contract["synthetic_corpus"]
    cells: list[dict[str, Any]] = []
    probe_cache: dict[tuple[int, str, int], dict[str, Any]] = {}

    def add_cell(kind: str, candidate: str, width: int, chi: int | None,
                 operation_class: str, shots: int, repeats: int, allocation: str) -> None:
        probe_key = (int(width), operation_class, int(repeats))
        if probe_key not in probe_cache:
            probe_cache[probe_key] = generate_v3_probe(contract, width, operation_class, repeats)
        probe = probe_cache[probe_key]
        descriptor: dict[str, Any] = {
            "manifest_id": V3_MANIFEST_ID, "kind": kind, "candidate": candidate,
            "width": int(width), "chi": None if chi is None else int(chi),
            "operation_class": operation_class, "shots": int(shots),
            "operation_repeats": int(repeats), "allocation": allocation,
        }
        descriptor.update({key: probe[key] for key in (
            "preparation_qasm_sha256", "variable_word_qasm_sha256", "whole_qasm_sha256",
            "preparation_counts", "variable_word_counts", "whole_qasm_counts",
            "preparation_h_pairs", "preparation_global_phase_radians",
        )})
        descriptor["cell_id"] = canonical_hash({
            "resolved_contract_sha256": canonical_hash(contract),
            "descriptor": descriptor,
        })
        descriptor["scheduled_timed_observations"] = (
            int(contract["measurement"]["sessions"])
            * int(contract["measurement"]["timed_repetitions_per_cell_per_session"])
        )
        cells.append(descriptor)

    for candidate in contract["implementation"]["candidates"]:
        width_key = "statevector_widths" if candidate == "statevector" else "mps_widths"
        widths = [int(value) for value in corpus[width_key]]
        bonds: list[int | None] = ([None] if candidate == "statevector" else
                                   [int(value) for value in corpus["mps_configured_bonds"]])
        for width in widths:
            for chi in bonds:
                for operation_class in corpus["operation_classes"]:
                    for repeats in corpus["operation_repeat_counts"]:
                        add_cell("operation", candidate, width, chi, str(operation_class), 1,
                                 int(repeats), "primary_operation")

    for candidate in corpus["sampling_candidates"]:
        width_key = "statevector_widths" if candidate == "statevector" else "mps_widths"
        widths = [int(value) for value in corpus[width_key]]
        bonds = [None] if candidate == "statevector" else [
            int(value) for value in corpus["mps_configured_bonds"]
        ]
        for width in widths:
            for chi in bonds:
                for shots in corpus["sampling_shots"]:
                    add_cell("sampling", str(candidate), width, chi, "zero_gate_sample",
                             int(shots), 0, "primary_zero_state_sampling")

    if include_diagnostics:
        diagnostic = corpus["entangled_sampling_diagnostic"]
        for width in diagnostic["widths"]:
            for chi in diagnostic["configured_bonds"]:
                for shots in diagnostic["shots"]:
                    add_cell("sampling_diagnostic", str(diagnostic["candidate"]), int(width),
                             int(chi), "entangled_sampling_diagnostic", int(shots), 0,
                             "entangled_sampling_transfer_diagnostic")

    expected_primary = (int(corpus["planned_primary_operation_cells"])
                        + int(corpus["planned_primary_sampling_cells"]))
    expected_total = int(corpus["planned_cells_including_diagnostics"] if include_diagnostics
                         else corpus["planned_primary_cells"])
    if len(cells) != expected_total or (include_diagnostics
            and expected_primary != int(corpus["planned_primary_cells"])):
        raise V3Unavailable(f"v3_plan_count_mismatch:{len(cells)}:{expected_total}")
    return cells


def v3_pilot_cell_ids(cells: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Return the frozen first-ten and pre-expansion 20-cell identity lists."""
    by_key = {
        (str(row["candidate"]), int(row["width"]), row.get("chi"),
         str(row["operation_class"]), int(row["operation_repeats"]),
         str(row["kind"]), int(row["shots"])): str(row["cell_id"])
        for row in cells
    }
    sv_ops = ("one_qubit_noncommuting", "two_qubit_wrapper_control", "two_qubit_cx_interleaved")
    repeat_grid = (256, 512, 1024)
    first_ten = [
        by_key[("statevector", 16, None, op, repeat, "operation", 1)]
        for op in sv_ops for repeat in repeat_grid
    ]
    first_ten.append(by_key[
        ("mps_fixed_chi", 16, 32, "one_qubit_noncommuting", 512, "operation", 1)
    ])
    mps_remaining = [
        by_key[("mps_fixed_chi", 16, 32, op, repeat, "operation", 1)]
        for op in sv_ops for repeat in repeat_grid
        if not (op == "one_qubit_noncommuting" and repeat == 512)
    ]
    sample_cells = [
        by_key[("statevector", 16, None, "zero_gate_sample", 0, "sampling", 1000)],
        by_key[("mps_fixed_chi", 16, 32, "zero_gate_sample", 0, "sampling", 1000)],
    ]
    if len(first_ten) != 10 or len(set(first_ten)) != 10 or len(mps_remaining) != 8:
        raise V3Unavailable("v3_pilot_cell_identity_mismatch")
    return {"first_ten_cell_ids": first_ten,
            "pre_expansion_cell_ids": [*first_ten, *mps_remaining, *sample_cells]}


def _partial_is_quarantined(cell: Mapping[str, Any], quarantine_policy: Mapping[str, Any]) -> bool:
    selector = quarantine_policy.get("selector")
    if not isinstance(selector, Mapping):
        raise V3Unavailable("partial_quarantine_selector_missing")
    candidate = str(selector.get("candidate", ""))
    width = int(selector.get("width", -1))
    chis = {int(value) for value in selector.get("chis", [])}
    kinds = {str(value) for value in selector.get("affected_kinds", [])}
    if (str(cell.get("candidate")) != candidate
            or int(cell.get("width", -1)) != width
            or cell.get("chi") in (None, "")
            or int(cell["chi"]) not in chis):
        return False
    return (str(cell.get("kind")) in kinds
            or str(cell.get("allocation")) in kinds)


def build_partial_v3_plan(
    contract: Mapping[str, Any], quarantine_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply the locked S6 quarantine without changing any S5 cell identity."""
    if contract.get("manifest_id") != V3_MANIFEST_ID:
        raise V3Unavailable("partial_planner_requires_unchanged_v3_parent")
    cells = build_v3_plan(contract)
    if len(cells) != 432:
        raise V3Unavailable(f"partial_assigned_cell_count_mismatch:{len(cells)}")
    result: list[dict[str, Any]] = []
    quarantined_operation = quarantined_diagnostic = 0
    eligible_primary = eligible_diagnostic = 0
    for base_cell in cells:
        cell = dict(base_cell)
        is_quarantined = _partial_is_quarantined(cell, quarantine_policy)
        cell["amendment_manifest_id"] = PARTIAL_MANIFEST_ID
        cell["base_cell_id"] = str(base_cell["cell_id"])
        cell["calibration_eligibility"] = "quarantined" if is_quarantined else "eligible"
        cell["calibration_reason"] = PARTIAL_QUARANTINE_REASON if is_quarantined else ""
        cell["timing_status"] = "not_started"
        cell["timing_reason"] = PARTIAL_QUARANTINE_REASON if is_quarantined else ""
        if is_quarantined:
            if cell["kind"] == "operation":
                quarantined_operation += 1
            elif cell["allocation"] == "entangled_sampling_transfer_diagnostic":
                quarantined_diagnostic += 1
            else:
                raise V3Unavailable(f"partial_unexpected_quarantine_match:{cell['cell_id']}")
        elif cell["allocation"] == "entangled_sampling_transfer_diagnostic":
            eligible_diagnostic += 1
        else:
            eligible_primary += 1
        result.append(cell)
    actual = {
        "assigned_cells": len(result),
        "quarantined_operation_cells": quarantined_operation,
        "quarantined_diagnostic_cells": quarantined_diagnostic,
        "quarantined_total_cells": quarantined_operation + quarantined_diagnostic,
        "eligible_primary_cells": eligible_primary,
        "eligible_diagnostic_cells": eligible_diagnostic,
        "eligible_total_cells": eligible_primary + eligible_diagnostic,
    }
    for key, value in actual.items():
        if quarantine_policy.get(key) != value:
            raise V3Unavailable(f"partial_allocation_accounting_mismatch:{key}:{value}")
    if actual != {
        "assigned_cells": 432, "quarantined_operation_cells": 27,
        "quarantined_diagnostic_cells": 6, "quarantined_total_cells": 33,
        "eligible_primary_cells": 393, "eligible_diagnostic_cells": 6,
        "eligible_total_cells": 399,
    }:
        raise V3Unavailable("partial_frozen_allocation_counts_mismatch")
    return result


def partial_v3_pilot_cell_ids(cells: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Materialize the S6 ten-cell and twenty-cell prefixes exactly."""
    cells = list(cells)
    by_key = {
        (str(row["candidate"]), int(row["width"]), row.get("chi"),
         str(row["operation_class"]), int(row["operation_repeats"]),
         str(row["kind"]), int(row["shots"])): str(row["cell_id"])
        for row in cells
    }
    one = "one_qubit_noncommuting"
    wrapper = "two_qubit_wrapper_control"
    cx = "two_qubit_cx_interleaved"
    repeats = (256, 512, 1024)
    first_ten = [
        by_key[("statevector", 16, None, op, repeat, "operation", 1)]
        for op in (one, wrapper, cx) for repeat in repeats
    ]
    first_ten.append(by_key[("mps_fixed_chi", 12, 32, one, 512, "operation", 1)])
    mps_operations = [
        by_key[("mps_fixed_chi", 12, 32, op, repeat, "operation", 1)]
        for op in (one, wrapper, cx) for repeat in repeats
    ]
    sampling = [
        by_key[("statevector", 16, None, "zero_gate_sample", 0, "sampling", 1000)],
        by_key[("mps_fixed_chi", 12, 32, "zero_gate_sample", 0, "sampling", 1000)],
    ]
    first_twenty = [*first_ten, *(value for value in mps_operations if value not in first_ten), *sampling]
    if len(first_ten) != 10 or len(set(first_ten)) != 10:
        raise V3Unavailable("partial_first_ten_identity_mismatch")
    if len(first_twenty) != 20 or len(set(first_twenty)) != 20:
        raise V3Unavailable("partial_first_twenty_identity_mismatch")
    by_id = {str(cell["cell_id"]): cell for cell in cells}
    if any(by_id[cell_id].get("calibration_eligibility") != "eligible"
           for cell_id in first_twenty):
        raise V3Unavailable("partial_pilot_contains_quarantined_cell")
    return {"first_ten_cell_ids": first_ten, "pre_expansion_cell_ids": first_twenty}


def build_partial_v3_execution_schedule(
    contract: Mapping[str, Any], cells: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain every S5 slot, pin partial identity, and terminally skip quarantine."""
    all_cells = [dict(cell) for cell in cells]
    pilot = partial_v3_pilot_cell_ids(all_cells)
    by_id = {str(cell["cell_id"]): cell for cell in all_cells}
    eligible = [str(cell["cell_id"]) for cell in all_cells
                if cell.get("calibration_eligibility") == "eligible"]
    quarantined = [str(cell["cell_id"]) for cell in all_cells
                   if cell.get("calibration_eligibility") == "quarantined"]
    order_ids = list(pilot["pre_expansion_cell_ids"])
    seen = set(order_ids)
    order_ids.extend(cell_id for cell_id in eligible if cell_id not in seen)
    order_ids.extend(quarantined)
    if len(order_ids) != 432 or len(set(order_ids)) != 432:
        raise V3Unavailable("partial_schedule_cell_order_mismatch")
    ordered_cells = [by_id[cell_id] for cell_id in order_ids]
    schedule = build_v3_execution_schedule(contract, ordered_cells)
    by_cell = {str(cell["cell_id"]): cell for cell in ordered_cells}
    quarantine_slots = 0
    for event in schedule:
        source = by_cell[str(event["cell_id"])]
        event["parent_manifest_id"] = V3_MANIFEST_ID
        event["manifest_id"] = PARTIAL_MANIFEST_ID
        event["amendment_manifest_id"] = PARTIAL_MANIFEST_ID
        event["base_cell_id"] = str(source["base_cell_id"])
        event["calibration_eligibility"] = str(source["calibration_eligibility"])
        event["timing_status"] = "not_started"
        event["timing_reason"] = str(source["calibration_reason"])
        if source["calibration_eligibility"] == "quarantined":
            quarantine_slots += 1
    if len(schedule) != 10_368 or quarantine_slots != 792:
        raise V3Unavailable(
            f"partial_schedule_slot_count_mismatch:{len(schedule)}:{quarantine_slots}"
        )
    if sum(event["calibration_eligibility"] == "eligible" for event in schedule) != 9_576:
        raise V3Unavailable("partial_eligible_schedule_slot_count_mismatch")
    return schedule


def build_partial_v3_run_identity(
    partial_resolution: Mapping[str, Any], pins: Mapping[str, Any],
    input_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bind partial runs to amendment + immutable parent + final execution pins."""
    identity = partial_resolution.get("amendment_identity")
    contract = partial_resolution.get("contract")
    if not isinstance(identity, Mapping) or not isinstance(contract, Mapping):
        raise V3Unavailable("partial_identity_resolution_missing")
    if (identity.get("manifest_id") != PARTIAL_MANIFEST_ID
            or identity.get("sha256") != PARTIAL_MANIFEST_SHA256
            or contract.get("manifest_id") != V3_MANIFEST_ID
            or not re.fullmatch(r"[0-9a-f]{64}", str(partial_resolution.get("parent_contract_sha256", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(partial_resolution.get("resolved_contract_sha256", "")))):
        raise V3Unavailable("partial_identity_parent_or_amendment_mismatch")
    if not isinstance(pins, Mapping) or not pins:
        raise V3Unavailable("partial_identity_execution_pins_missing")
    hashes = dict(input_hashes or {})
    for name, digest in hashes.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise V3Unavailable(f"partial_identity_input_hash_invalid:{name}")
    body = {
        "identity_schema": "maestro_v3_partial_calibration_run_identity_v1",
        "manifest_id": PARTIAL_MANIFEST_ID,
        "resolved_contract_sha256": str(partial_resolution["resolved_contract_sha256"]),
        "amendment_identity": dict(identity),
        "parent_manifest_id": V3_MANIFEST_ID,
        "parent_manifest_sha256": PARTIAL_PARENT_V3_SHA256,
        "parent_contract_sha256": str(partial_resolution["parent_contract_sha256"]),
        "pins": dict(pins),
        "input_hashes": dict(sorted(hashes.items())),
    }
    return {"manifest_id": PARTIAL_MANIFEST_ID, "run_identity": body,
            "run_identity_sha256": canonical_hash(body)}


def validate_partial_v3_resume_identity(
    existing: Mapping[str, Any], expected: Mapping[str, Any],
) -> None:
    """Reject v2/v3 identities and any changed partial amendment or pins."""
    if existing.get("manifest_id") != PARTIAL_MANIFEST_ID:
        raise V3Unavailable("partial_resume_rejects_non_partial_records")
    if expected.get("manifest_id") != PARTIAL_MANIFEST_ID:
        raise V3Unavailable("partial_expected_resume_identity_mismatch")
    existing_body = existing.get("run_identity")
    expected_body = expected.get("run_identity")
    existing_sha = existing.get("run_identity_sha256")
    expected_sha = expected.get("run_identity_sha256")
    if not isinstance(existing_body, Mapping) or not isinstance(expected_body, Mapping):
        raise V3Unavailable("partial_resume_identity_missing")
    if canonical_hash(existing_body) != existing_sha:
        raise V3Unavailable("partial_existing_resume_identity_hash_mismatch")
    if canonical_hash(expected_body) != expected_sha:
        raise V3Unavailable("partial_expected_resume_identity_hash_mismatch")
    if existing_sha != expected_sha:
        raise V3Unavailable("partial_resume_run_identity_mismatch")


def build_v3_execution_schedule(
    contract: Mapping[str, Any],
    cells: Iterable[Mapping[str, Any]] | None = None,
    selected_cell_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Materialize per-session calls, balanced repeat orders and adjacent CX pairs."""
    all_cells = list(cells) if cells is not None else build_v3_plan(contract)
    by_id = {str(cell["cell_id"]): dict(cell) for cell in all_cells}
    if len(by_id) != len(all_cells):
        raise V3Unavailable("v3_schedule_duplicate_cell_id")
    if selected_cell_ids is None:
        selected = list(all_cells)
    else:
        selected_ids = list(selected_cell_ids)
        if len(set(selected_ids)) != len(selected_ids) or any(cell_id not in by_id for cell_id in selected_ids):
            raise V3Unavailable("v3_schedule_selected_cell_ids_invalid")
        selected = [by_id[cell_id] for cell_id in selected_ids]
    measure = contract["measurement"]
    sessions = int(measure["sessions"])
    untimed_per_session = int(measure["untimed_calls_per_cell_per_session"])
    timed_per_session = int(measure["timed_repetitions_per_cell_per_session"])
    repeat_orders = [[int(value) for value in order] for order in measure["session_repeat_orders"]]
    expected_repeats = {int(value) for value in contract["synthetic_corpus"]["operation_repeat_counts"]}
    if len(repeat_orders) != sessions or any(set(order) != expected_repeats for order in repeat_orders):
        raise V3Unavailable("v3_session_repeat_orders_invalid")

    events: list[dict[str, Any]] = []

    def append_event(cell: Mapping[str, Any], session: int, repetition: int, stage: str,
                     *, repeat_order_index: int | None = None,
                     within_repeat_position: int | None = None,
                     pair_group_id: str | None = None) -> None:
        event = {
            "manifest_id": V3_MANIFEST_ID,
            "cell_id": str(cell["cell_id"]),
            "attempt_id": canonical_hash({
                "cell_id": str(cell["cell_id"]), "session": session,
                "repetition": repetition, "stage": stage,
                "repeat_order_index": repeat_order_index,
                "within_repeat_position": within_repeat_position,
            }),
            "stage": stage,
            "candidate": cell["candidate"], "width": int(cell["width"]),
            "chi": cell["chi"], "kind": cell["kind"],
            "operation_class": cell["operation_class"],
            "shots": int(cell["shots"]), "operation_repeats": int(cell["operation_repeats"]),
            "preparation_qasm_sha256": cell["preparation_qasm_sha256"],
            "variable_word_qasm_sha256": cell["variable_word_qasm_sha256"],
            "whole_qasm_sha256": cell["whole_qasm_sha256"],
            "preparation_counts": dict(cell["preparation_counts"]),
            "variable_word_counts": dict(cell["variable_word_counts"]),
            "whole_qasm_counts": dict(cell["whole_qasm_counts"]),
            "session": session, "repetition": repetition,
            "session_repeat_order_index": repeat_order_index,
            "within_repeat_position": within_repeat_position,
            "pair_group_id": pair_group_id,
            "process_start_method": measure["process_start_method"],
            "call_process_isolated": True,
            "timing_status": "not_started",
            "timing_reason": "",
        }
        events.append(event)

    # The untimed allocation is explicitly preconditioning only; every call is
    # process-isolated and does not imply persistent in-process warming.
    for cell in selected:
        for session in range(1, sessions + 1):
            for repetition in range(untimed_per_session):
                append_event(cell, session, repetition, V3_UNTIMED_STAGE)

    operation_blocks: list[tuple[str, int, int | None]] = []
    for cell in selected:
        if cell["kind"] != "operation":
            continue
        block = (str(cell["candidate"]), int(cell["width"]),
                 None if cell["chi"] is None else int(cell["chi"]))
        if block not in operation_blocks:
            operation_blocks.append(block)
    selected_operation_cells = {
        (str(cell["candidate"]), int(cell["width"]),
         None if cell["chi"] is None else int(cell["chi"]),
         str(cell["operation_class"]), int(cell["operation_repeats"])): cell
        for cell in selected if cell["kind"] == "operation"
    }
    for candidate, width, chi in operation_blocks:
        for session in range(1, sessions + 1):
            repeat_order = repeat_orders[session - 1]
            for order_index, repeats in enumerate(repeat_order):
                for repetition in range(timed_per_session):
                    one = selected_operation_cells.get(
                        (candidate, width, chi, "one_qubit_noncommuting", repeats)
                    )
                    wrapper = selected_operation_cells.get(
                        (candidate, width, chi, "two_qubit_wrapper_control", repeats)
                    )
                    cx = selected_operation_cells.get(
                        (candidate, width, chi, "two_qubit_cx_interleaved", repeats)
                    )
                    order: list[tuple[Mapping[str, Any], str | None]] = []
                    if one is not None:
                        order.append((one, None))
                    wrapper_first = ((session - 1 + order_index) % 2 == 0)
                    pair_id = canonical_hash({
                        "candidate": candidate, "width": width, "chi": chi,
                        "session": session, "repeat_count": repeats,
                        "repetition": repetition,
                    })
                    pair = ([(wrapper, pair_id), (cx, pair_id)] if wrapper_first
                            else [(cx, pair_id), (wrapper, pair_id)])
                    order.extend((cell, pair_id_value) for cell, pair_id_value in pair if cell is not None)
                    for position, (cell, pair_group_id) in enumerate(order):
                        append_event(
                            cell, session, repetition, V3_TIMED_STAGE,
                            repeat_order_index=order_index,
                            within_repeat_position=position,
                            pair_group_id=pair_group_id,
                        )

    # Sampling and diagnostic cells have no operation-repeat ordering.
    for cell in selected:
        if cell["kind"] == "operation":
            continue
        for session in range(1, sessions + 1):
            for repetition in range(timed_per_session):
                append_event(cell, session, repetition, V3_TIMED_STAGE)

    expected_event_count = len(selected) * sessions * (untimed_per_session + timed_per_session)
    if len(events) != expected_event_count:
        raise V3Unavailable(f"v3_schedule_count_mismatch:{len(events)}:{expected_event_count}")
    return events


def central_schmidt_rank_v3(amplitudes: Any, width: int,
                            relative_tolerance: float = 1e-6,
                            norm_atol: float = 1e-6) -> dict[str, Any]:
    """Return a logical central-cut rank check, not an internal MPS bond."""
    import numpy as np

    n = int(width)
    if n < 2:
        raise V3Unavailable("v3_rank_width_must_be_at_least_two")
    vector = np.asarray(amplitudes, dtype=np.complex128).reshape(-1)
    if vector.size != 2**n:
        raise V3Unavailable(f"v3_rank_amplitude_dimension_mismatch:{vector.size}:{2**n}")
    if not np.all(np.isfinite(vector)):
        raise V3Unavailable("v3_rank_nonfinite_amplitudes")
    norm = float(np.vdot(vector, vector).real)
    if not math.isfinite(norm) or abs(norm - 1.0) > float(norm_atol):
        raise V3Unavailable(f"v3_rank_amplitude_norm_out_of_tolerance:{norm}")
    left_width = n // 2
    matrix = vector.reshape((2**left_width, 2**(n-left_width)))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if (singular_values.size == 0 or singular_values[0] <= 0
            or not np.all(np.isfinite(singular_values))):
        raise V3Unavailable("v3_rank_singular_values_invalid")
    rank = int(np.count_nonzero(singular_values > float(relative_tolerance) * singular_values[0]))
    return {
        "width": n, "norm": norm, "logical_central_schmidt_rank": rank,
        "rank_tolerance_relative": float(relative_tolerance),
        "internal_bond_dimension_claim": False,
    }


def assess_v3_candidate_preparation(
    amplitudes: Any, width: int, configured_chi: int,
    native_max_bond_dim_reached: Any = None,
    relative_tolerance: float = 1e-6, norm_atol: float = 1e-6,
) -> dict[str, Any]:
    """Validate entanglement while keeping logical rank and native telemetry distinct."""
    rank_info = central_schmidt_rank_v3(amplitudes, width, relative_tolerance, norm_atol)
    chi = int(configured_chi)
    if chi <= 0:
        raise V3Unavailable("v3_configured_chi_must_be_positive")
    feasible = min(chi, 2**(int(width)//2))
    rank = int(rank_info["logical_central_schmidt_rank"])
    if rank <= 1:
        preparation_status, reason = "unavailable", "candidate_preparation_not_entangled"
    else:
        preparation_status, reason = "ok", ""
    if native_max_bond_dim_reached in (None, ""):
        telemetry, telemetry_status = None, "unknown"
    else:
        try:
            telemetry = int(native_max_bond_dim_reached)
        except (TypeError, ValueError) as exc:
            raise V3Unavailable("v3_native_bond_telemetry_non_numeric") from exc
        if telemetry <= 0 or telemetry > chi:
            raise V3Unavailable(f"v3_native_bond_telemetry_out_of_configured_range:{telemetry}:{chi}")
        telemetry_status = "feasible_cap_saturated" if telemetry >= feasible else "below_feasible_cap"
    return {
        **rank_info,
        "candidate_logical_central_schmidt_rank": rank,
        "reference_logical_central_schmidt_rank": None,
        "configured_chi": chi,
        "physically_exercisable_configured_bond": feasible,
        "native_max_bond_dim_reached": telemetry,
        "telemetry_status": telemetry_status,
        "preparation_status": preparation_status,
        "preparation_reason": reason,
        "claim_boundary": "logical_rank_is_not_internal_bond_dimension_or_maximum_throughout_word",
    }


def reference_central_ranks_v3(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Run untimed Qiskit reference-only central-rank checks for the frozen widths."""
    if contract.get("manifest_id") != V3_MANIFEST_ID:
        raise V3Unavailable("v3_rank_check_requires_v3_resolved_contract")
    expected_version = str(contract["normalization"]["qiskit_version"])
    qasm2, _, Statevector, _ = _qiskit_api(expected_version)
    rank_policy = contract["synthetic_corpus"]["background"]
    relative_tolerance = float(rank_policy["relative_singular_value_rank_tolerance"])
    expected_ranks = {int(width): int(rank) for width, rank
                      in rank_policy["reference_central_schmidt_rank_at_tolerance"].items()}
    results = []
    for width in contract["synthetic_corpus"]["statevector_widths"]:
        n = int(width)
        probe = generate_v3_probe(
            contract, n, "entangled_sampling_diagnostic", 0, include_measurement=False
        )
        circuit = _load_strict_legacy_qasm2(qasm2, probe["program"])
        state = Statevector.from_instruction(circuit)
        rank_info = central_schmidt_rank_v3(
            state.data, n, relative_tolerance,
            float(contract["quality_policy"]["vector_norm_atol"]),
        )
        expected = expected_ranks.get(n)
        status = "ok" if rank_info["logical_central_schmidt_rank"] == expected else "rank_mismatch"
        results.append({
            **rank_info, "expected_reference_logical_central_schmidt_rank": expected,
            "preparation_qasm_sha256": probe["preparation_qasm_sha256"],
            "status": status, "source": "qiskit_fp64_statevector_reference",
            "internal_bond_dimension_claim": False,
        })
    return results


def validate_v3_rank_preflight_records(
    contract: Mapping[str, Any], records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate exact-reference and candidate preparation checks for each config.

    A missing native max-bond field is recorded as unknown and limits the
    saturation claim; it is never inferred from logical Schmidt rank.
    """
    corpus = contract["synthetic_corpus"]
    rank_policy = corpus["background"]
    expected_reference = {int(width): int(rank) for width, rank
                          in rank_policy["reference_central_schmidt_rank_at_tolerance"].items()}
    expected: set[tuple[str, int, int | None]] = set()
    for width_value in corpus["statevector_widths"]:
        width = int(width_value)
        expected.add(("statevector", width, None))
        for chi_value in corpus["mps_configured_bonds"]:
            expected.add(("mps_fixed_chi", width, int(chi_value)))
    seen: dict[tuple[str, int, int | None], Mapping[str, Any]] = {}
    for record in records:
        key = (str(record.get("candidate", "")), int(record.get("width", -1)),
               None if record.get("chi") in (None, "") else int(record["chi"]))
        if key not in expected:
            raise V3Unavailable(f"v3_unexpected_rank_preflight_config:{key}")
        if key in seen:
            raise V3Unavailable(f"v3_duplicate_rank_preflight_config:{key}")
        seen[key] = record

    normalized_records: list[dict[str, Any]] = []
    unavailable_reasons: list[str] = []
    telemetry_counts: dict[str, int] = defaultdict(int)
    for candidate, width, chi in sorted(expected, key=lambda item: (item[0], item[1], item[2] or 0)):
        key = (candidate, width, chi)
        record = seen.get(key)
        if record is None:
            unavailable_reasons.append(f"rank_preflight_missing:{candidate}:{width}:{chi}")
            continue
        record_reasons: list[str] = []

        def fail(reason: str) -> None:
            unavailable_reasons.append(reason)
            record_reasons.append(reason)

        expected_rank = expected_reference.get(width)
        ref_rank = record.get("reference_logical_central_schmidt_rank")
        cand_rank = record.get("candidate_logical_central_schmidt_rank")
        try:
            norm = float(record.get("candidate_norm"))
            ref_rank = int(ref_rank)
            cand_rank = int(cand_rank)
        except (TypeError, ValueError):
            fail(f"rank_preflight_invalid_fields:{candidate}:{width}:{chi}")
            continue
        if not math.isfinite(norm) or abs(norm - 1.0) > float(contract["quality_policy"]["vector_norm_atol"]):
            fail(f"rank_preflight_candidate_norm:{candidate}:{width}:{chi}")
        if ref_rank != expected_rank:
            fail(f"rank_preflight_reference_mismatch:{candidate}:{width}:{chi}")
        if cand_rank <= 1:
            fail(f"rank_preflight_candidate_not_entangled:{candidate}:{width}:{chi}")
        if candidate == "statevector" and cand_rank != ref_rank:
            fail(f"rank_preflight_statevector_mismatch:{candidate}:{width}:{chi}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("candidate_amplitudes_sha256", ""))):
            fail(f"rank_preflight_candidate_amplitude_hash_missing:{candidate}:{width}:{chi}")

        telemetry_value = record.get("max_bond_dim_reached")
        if candidate == "statevector":
            telemetry_status = "not_applicable"
            feasible_bond = None
        elif telemetry_value in (None, ""):
            telemetry_status = "unknown"
            feasible_bond = min(int(chi), 2**(width//2))
        else:
            try:
                telemetry = int(telemetry_value)
            except (TypeError, ValueError):
                fail(f"rank_preflight_telemetry_non_numeric:{candidate}:{width}:{chi}")
                continue
            feasible_bond = min(int(chi), 2**(width//2))
            if telemetry <= 0 or telemetry > int(chi):
                fail(f"rank_preflight_telemetry_out_of_range:{candidate}:{width}:{chi}")
                continue
            telemetry_status = "feasible_cap_saturated" if telemetry >= feasible_bond else "below_feasible_cap"
        telemetry_counts[telemetry_status] += 1
        normalized_records.append({
            "candidate": candidate, "width": width, "chi": chi,
            "reference_logical_central_schmidt_rank": ref_rank,
            "candidate_logical_central_schmidt_rank": cand_rank,
            "candidate_norm": norm,
            "candidate_amplitudes_sha256": record["candidate_amplitudes_sha256"],
            "physically_exercisable_configured_bond": feasible_bond,
            "max_bond_dim_reached": telemetry_value if candidate == "mps_fixed_chi" else None,
            "telemetry_status": telemetry_status,
            "preflight_status": "unavailable" if record_reasons else "ok",
            "preflight_reasons": record_reasons,
            "claim_boundary": "logical_rank_is_not_internal_bond_dimension_or_maximum_throughout_word",
        })
    status = "pass" if not unavailable_reasons else "unavailable"
    return {
        "status": status,
        "records": normalized_records,
        "unavailable_reasons": sorted(unavailable_reasons),
        "telemetry_status_counts": dict(sorted(telemetry_counts.items())),
        "claim_boundary": "preparation rank is not maximum bond throughout the operation word",
    }


def validate_partial_v3_rank_preflight_records(
    contract: Mapping[str, Any], records: Iterable[Mapping[str, Any]],
    quarantine_policy: Mapping[str, Any],
    quarantine_evidence: Mapping[tuple[str, int, int | None], Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate enabled controls with the real S5 validator and frozen skips.

    The three quarantined n16 MPS controls are not re-run and never acquire a
    logical-rank claim from rescaled amplitudes. Their pinned failure evidence
    remains explicit unavailable status in this aggregate.
    """
    if contract.get("manifest_id") != V3_MANIFEST_ID:
        raise V3Unavailable("partial_rank_preflight_requires_v3_parent")
    selector = quarantine_policy.get("selector")
    if not isinstance(selector, Mapping):
        raise V3Unavailable("partial_rank_preflight_selector_missing")
    quarantine_keys = {
        (str(selector["candidate"]), int(selector["width"]), int(chi))
        for chi in selector.get("chis", [])
    }
    if quarantine_keys != {
        ("mps_fixed_chi", 16, 4), ("mps_fixed_chi", 16, 8),
        ("mps_fixed_chi", 16, 32),
    }:
        raise V3Unavailable("partial_rank_preflight_quarantine_set_mismatch")
    expected: set[tuple[str, int, int | None]] = set()
    corpus = contract["synthetic_corpus"]
    expected_reference = {
        int(width): int(rank)
        for width, rank in corpus["background"]["reference_central_schmidt_rank_at_tolerance"].items()
    }
    for width in corpus["statevector_widths"]:
        expected.add(("statevector", int(width), None))
        for chi in corpus["mps_configured_bonds"]:
            expected.add(("mps_fixed_chi", int(width), int(chi)))
    expected_quarantine = expected & quarantine_keys
    if expected_quarantine != quarantine_keys:
        raise V3Unavailable("partial_rank_preflight_quarantine_outside_parent_grid")

    seen: dict[tuple[str, int, int | None], Mapping[str, Any]] = {}
    reference_seen: dict[int, Mapping[str, Any]] = {}
    for record in records:
        try:
            candidate = str(record["candidate"])
            width = int(record["width"])
            if candidate == "qiskit_reference":
                if record.get("chi") not in (None, "") or width not in expected_reference or width in reference_seen:
                    raise V3Unavailable(f"partial_reference_rank_record_unexpected_or_duplicate:{width}")
                reference_seen[width] = record
                continue
            key = (candidate, width,
                   None if record.get("chi") in (None, "") else int(record["chi"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise V3Unavailable("partial_rank_preflight_record_key_invalid") from exc
        if key not in expected or key in seen:
            raise V3Unavailable(f"partial_rank_preflight_record_unexpected_or_duplicate:{key}")
        seen[key] = record
    if set(seen) != expected:
        raise V3Unavailable(f"partial_rank_preflight_record_set_mismatch:{sorted(expected-set(seen))}")
    if set(reference_seen) != set(expected_reference):
        raise V3Unavailable(
            f"partial_reference_rank_preflight_coverage_mismatch:{sorted(set(expected_reference)-set(reference_seen))}"
        )
    norm_tolerance = float(contract["quality_policy"]["vector_norm_atol"])
    normalized_references: list[dict[str, Any]] = []
    for width, expected_rank in sorted(expected_reference.items()):
        record = reference_seen[width]
        try:
            rank = int(record["candidate_logical_central_schmidt_rank"])
            reference_rank = int(record["reference_logical_central_schmidt_rank"])
            norm = float(record["candidate_norm"])
            wall = float(record["reference_wall_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise V3Unavailable(f"partial_reference_rank_record_invalid:{width}") from exc
        if (rank != expected_rank or reference_rank != expected_rank
                or not math.isfinite(norm) or abs(norm - 1.0) > norm_tolerance
                or not math.isfinite(wall) or wall < 0
                or record.get("preflight_status") != "ok"):
            raise V3Unavailable(f"partial_reference_rank_control_failed:{width}")
        for field_name in ("candidate_amplitudes_sha256", "qasm_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(record.get(field_name, ""))):
                raise V3Unavailable(f"partial_reference_rank_hash_missing:{width}:{field_name}")
        normalized_references.append(dict(record))

    enabled_records = [record for key, record in seen.items() if key not in quarantine_keys]
    base_result = validate_v3_rank_preflight_records(contract, enabled_records)
    expected_missing = {
        f"rank_preflight_missing:{candidate}:{width}:{chi}"
        for candidate, width, chi in sorted(quarantine_keys)
    }
    if (set(base_result.get("unavailable_reasons", [])) != expected_missing
            or len(base_result.get("records", [])) != len(expected - quarantine_keys)):
        raise V3Unavailable(
            f"partial_enabled_rank_preflight_failed:{base_result.get('unavailable_reasons')}"
        )
    by_enabled_key = {
        (str(record["candidate"]), int(record["width"]),
         None if record.get("chi") is None else int(record["chi"])): record
        for record in base_result["records"]
    }
    if set(by_enabled_key) != expected - quarantine_keys:
        raise V3Unavailable("partial_enabled_rank_preflight_coverage_mismatch")

    unavailable_records: list[dict[str, Any]] = []
    for key in sorted(quarantine_keys):
        source = seen[key]
        evidence = quarantine_evidence.get(key)
        if not isinstance(evidence, Mapping):
            raise V3Unavailable(f"partial_quarantine_evidence_missing:{key}")
        if source.get("preflight_status") != "unavailable":
            raise V3Unavailable(f"partial_quarantine_status_mismatch:{key}")
        if source.get("preflight_reasons") != [PARTIAL_QUARANTINE_REASON]:
            raise V3Unavailable(f"partial_quarantine_reason_mismatch:{key}")
        if source.get("quarantine_evidence") != dict(evidence):
            raise V3Unavailable(f"partial_quarantine_evidence_mismatch:{key}")
        for field_name in ("squared_norm", "qasm_sha256", "amplitude_sha256", "worker_context_sha256"):
            if field_name not in evidence:
                raise V3Unavailable(f"partial_quarantine_evidence_field_missing:{key}:{field_name}")
        if (not math.isfinite(float(evidence["squared_norm"]))
                or abs(float(evidence["squared_norm"]) - 1.0)
                <= float(contract["quality_policy"]["vector_norm_atol"])):
            raise V3Unavailable(f"partial_quarantine_evidence_not_failed_norm:{key}")
        for hash_field in ("qasm_sha256", "amplitude_sha256", "worker_context_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(evidence[hash_field])):
                raise V3Unavailable(f"partial_quarantine_evidence_hash_invalid:{key}:{hash_field}")
        zero = source.get("zero_state_sampling_control")
        expected_qasm = generate_v3_probe(
            contract, key[1], "zero_gate_sample", 0, include_measurement=True
        )["program"]
        if not isinstance(zero, Mapping):
            raise V3Unavailable(f"partial_zero_state_control_missing:{key}")
        if (zero.get("status") != "ok" or zero.get("transport_status") != "ok"
                or zero.get("candidate") != key[0] or zero.get("chi") != key[2]
                or zero.get("declared_width") != key[1] or zero.get("parsed_width") != key[1]
                or zero.get("observed_amplitude_count") != 2**key[1]
                or zero.get("qasm") != expected_qasm
                or zero.get("qasm_sha256") != sha256_bytes(expected_qasm.encode())):
            raise V3Unavailable(f"partial_zero_state_control_identity_or_shape_invalid:{key}")
        for field_name in ("squared_norm", "zero_basis_probability"):
            value = float(zero.get(field_name, math.nan))
            if not math.isfinite(value) or abs(value - 1.0) > norm_tolerance:
                raise V3Unavailable(f"partial_zero_state_control_numerical_failure:{key}:{field_name}")
        for hash_field in ("amplitude_sha256", "worker_context_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(zero.get(hash_field, ""))):
                raise V3Unavailable(f"partial_zero_state_control_hash_missing:{key}:{hash_field}")
        if zero.get("worker_context", {}).get("identity_sha256") != zero["worker_context_sha256"]:
            raise V3Unavailable(f"partial_zero_state_control_context_hash_mismatch:{key}")
        unavailable_records.append({
            "candidate": key[0], "width": key[1], "chi": key[2],
            "preflight_status": "unavailable",
            "preflight_reasons": [PARTIAL_QUARANTINE_REASON],
            "quarantine_evidence": dict(evidence),
            "zero_state_sampling_control": dict(zero),
            "logical_rank_status": "not_computed_from_unnormalized_amplitudes",
            "internal_bond_dimension_claim": False,
        })
    normalized = [*normalized_references, *base_result["records"], *unavailable_records]
    normalized.sort(key=lambda record: (
        str(record["candidate"]), int(record["width"]), int(record.get("chi") or 0)
    ))
    return {
        "status": "pass",
        "records": normalized,
        "enabled_record_count": len(base_result["records"]),
        "quarantined_record_count": len(unavailable_records),
        "reference_record_count": len(normalized_references),
        "reference_records": normalized_references,
        "unavailable_reasons": [
            f"{key[0]}:{key[1]}:{key[2]}:{PARTIAL_QUARANTINE_REASON}"
            for key in sorted(quarantine_keys)
        ],
        "enabled_validator_status": "pass",
        "enabled_telemetry_status_counts": base_result["telemetry_status_counts"],
        "claim_boundary": (
            "enabled controls passed the v3 rank validator; quarantined controls remain "
            "unavailable and make no rank claim from unnormalized amplitudes"
        ),
    }


def _qiskit_api(expected_version: str) -> tuple[Any, Any, Any, str]:
    try:
        import qiskit
        from qiskit import qasm2, transpile
        from qiskit.quantum_info import Statevector
    except Exception as exc:  # pragma: no cover - environment specific
        raise V2Unavailable(f"qiskit_unavailable:{type(exc).__name__}:{exc}") from exc
    version = str(qiskit.__version__)
    if version != expected_version:
        raise V2Unavailable(f"qiskit_version_mismatch:{version}!={expected_version}")
    return qasm2, transpile, Statevector, version


def _load_strict_legacy_qasm2(qasm2: Any, program: str) -> Any:
    """Parse strict OpenQASM 2 with Qiskit's pinned legacy qelib definitions.

    The v2 fixtures originate from the legacy Qiskit exporter, whose qelib1
    includes contain instructions such as rzz, cp, u, and rccx.  Supplying
    Qiskit's version-pinned compatibility tables makes those documented gates
    available without relaxing strict syntax or accepting arbitrary names.
    """
    return qasm2.loads(
        program,
        strict=True,
        include_path=qasm2.LEGACY_INCLUDE_PATH,
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
    )


def _bit_mapping(circuit: Any) -> list[tuple[int, int]]:
    """Return measurement (qubit index, classical-bit index) pairs in order."""
    mapping: list[tuple[int, int]] = []
    for item in circuit.data:
        if item.operation.name == "measure":
            if len(item.qubits) != 1 or len(item.clbits) != 1:
                raise V2Unavailable("non_scalar_measurement_instruction")
            mapping.append((circuit.find_bit(item.qubits[0]).index,
                            circuit.find_bit(item.clbits[0]).index))
    return mapping


def _check_terminal_full_measurement(circuit: Any, expected_width: int) -> list[tuple[int, int]]:
    if circuit.num_qubits != expected_width:
        raise V2Unavailable(f"source_width_mismatch:{circuit.num_qubits}!={expected_width}")
    if circuit.num_clbits < expected_width:
        raise V2Unavailable(f"classical_width_too_small:{circuit.num_clbits}<{expected_width}")
    seen_measurement = False
    control_flow_names = {"if_else", "while_loop", "for_loop", "switch_case"}
    for item in circuit.data:
        name = str(item.operation.name).lower()
        if name == "measure":
            seen_measurement = True
            continue
        if name == "barrier":
            continue
        if seen_measurement:
            raise V2Unavailable(f"nonterminal_measurement_followed_by:{name}")
        if name == "reset":
            raise V2Unavailable("reset_unsupported")
        if name in control_flow_names or getattr(item.operation, "blocks", None):
            raise V2Unavailable(f"control_flow_unsupported:{name}")
        if item.clbits or getattr(item.operation, "condition", None) is not None:
            raise V2Unavailable(f"classical_control_unsupported:{name}")
    mapping = _bit_mapping(circuit)
    expected = set(range(expected_width))
    qubit_indices = [qubit for qubit, _ in mapping]
    classical_indices = [clbit for _, clbit in mapping]
    if (len(mapping) != expected_width or set(qubit_indices) != expected
            or len(set(qubit_indices)) != expected_width
            or len(set(classical_indices)) != expected_width
            or any(classical < 0 or classical >= circuit.num_clbits for classical in classical_indices)):
        raise V2Unavailable("measurement_mapping_not_full_quantum_register_bijection")
    return mapping


def _state_fidelity(left: Any, right: Any, expected_dimension: int, norm_atol: float) -> float:
    import numpy as np

    a = np.asarray(left, dtype=np.complex128).reshape(-1)
    b = np.asarray(right, dtype=np.complex128).reshape(-1)
    if a.size != expected_dimension or b.size != expected_dimension:
        raise V2Unavailable(f"amplitude_dimension_mismatch:{a.size}:{b.size}:{expected_dimension}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise V2Unavailable("nonfinite_amplitudes")
    norm_a = float(np.vdot(a, a).real)
    norm_b = float(np.vdot(b, b).real)
    if not math.isfinite(norm_a) or not math.isfinite(norm_b):
        raise V2Unavailable("nonfinite_amplitude_norm")
    if abs(norm_a - 1.0) > norm_atol or abs(norm_b - 1.0) > norm_atol:
        raise V2Unavailable(f"amplitude_norm_out_of_tolerance:{norm_a}:{norm_b}")
    result = abs(complex(np.vdot(a, b))) ** 2 / (norm_a * norm_b)
    if not math.isfinite(result) or result < 0.0 or result > 1.0 + 1e-12:
        raise V2Unavailable(f"invalid_state_fidelity:{result}")
    return min(1.0, float(result))


def _seeded_product_state(width: int, seed: int) -> Any:
    import numpy as np

    rng = np.random.default_rng(seed)
    factors = []
    for _ in range(width):
        pair = rng.normal(size=2) + 1j * rng.normal(size=2)
        pair = pair / np.linalg.norm(pair)
        factors.append(pair.astype(np.complex128))
    vector = np.asarray([1.0 + 0.0j], dtype=np.complex128)
    # Qiskit uses little-endian qubit indexing: q0 is the least significant bit.
    for pair in reversed(factors):
        vector = np.kron(vector, pair)
    return vector


@dataclass(frozen=True)
class NormalizedQasm:
    status: str
    reason: str
    source_qasm_sha256: str
    normalized_qasm: str
    normalized_qasm_sha256: str
    provenance: Mapping[str, Any]
    features: Mapping[str, int]
    unitary_prefix_qasm: str
    source_unitary_prefix_qasm: str


def normalize_qasm(
    source_qasm: str,
    expected_width: int,
    normalization: Mapping[str, Any],
    maestro: Any,
) -> NormalizedQasm:
    """Normalize QASM2 to rx/rz/cx and run the frozen semantic preflight.

    The function fails closed on partial/mid-circuit measurement, reset,
    classical control, opaque/non-decomposable operations, changed wire or
    measurement mapping, parser rejection, or statevector mismatch.  It does
    not infer equivalence from width or operation counts.
    """
    source_bytes = source_qasm.encode("utf-8")
    source_hash = sha256_bytes(source_bytes)
    qasm2, transpile, Statevector, qiskit_version = _qiskit_api(
        str(normalization["qiskit_version"])
    )
    try:
        source = _load_strict_legacy_qasm2(qasm2, source_qasm)
    except Exception as exc:
        raise V2Unavailable(f"source_qasm_parse_failed:{type(exc).__name__}:{exc}") from exc
    source_mapping = _check_terminal_full_measurement(source, int(expected_width))
    source_global_phase = str(source.global_phase)
    source_gate_counts: dict[str, int] = defaultdict(int)
    for item in source.data:
        if item.operation.name not in {"measure", "barrier"}:
            source_gate_counts[str(item.operation.name)] += 1

    try:
        derived = transpile(
            source,
            basis_gates=list(normalization["basis_gates"]),
            optimization_level=int(normalization["optimization_level"]),
            seed_transpiler=int(normalization["seed_transpiler"]),
            approximation_degree=float(normalization["approximation_degree"]),
            coupling_map=None,
            target=None,
        )
    except Exception as exc:
        raise V2Unavailable(f"exact_basis_transpilation_failed:{type(exc).__name__}:{exc}") from exc
    if derived.num_qubits != expected_width or derived.num_clbits != source.num_clbits:
        raise V2Unavailable("transpilation_changed_declared_width")
    derived_mapping = _check_terminal_full_measurement(derived, int(expected_width))
    if sorted(source_mapping) != sorted(derived_mapping):
        raise V2Unavailable("transpilation_changed_measurement_mapping")
    allowed = {"rx", "rz", "cx", "measure", "barrier"}
    derived_gate_counts: dict[str, int] = defaultdict(int)
    for item in derived.data:
        name = str(item.operation.name).lower()
        if name not in allowed:
            raise V2Unavailable(f"derived_gate_outside_locked_basis:{name}")
        if name not in {"measure", "barrier"}:
            derived_gate_counts[name] += 1

    normalized_qasm = qasm2.dumps(derived)
    try:
        serialized = _load_strict_legacy_qasm2(qasm2, normalized_qasm)
    except Exception as exc:
        raise V2Unavailable(f"derived_qasm_roundtrip_failed:{type(exc).__name__}:{exc}") from exc
    serialized_mapping = _check_terminal_full_measurement(serialized, expected_width)
    if (serialized.num_qubits != expected_width or serialized.num_clbits != source.num_clbits
            or sorted(serialized_mapping) != sorted(source_mapping)):
        raise V2Unavailable("derived_qasm_roundtrip_changed_wires_or_measurements")
    roundtrip_counts: dict[str, int] = defaultdict(int)
    for item in serialized.data:
        name = str(item.operation.name).lower()
        if name not in allowed:
            raise V2Unavailable(f"serialized_gate_outside_locked_basis:{name}")
        if name not in {"measure", "barrier"}:
            roundtrip_counts[name] += 1
    if dict(sorted(roundtrip_counts.items())) != dict(sorted(derived_gate_counts.items())):
        raise V2Unavailable("qasm_serialization_changed_gate_counts")

    source_unitary = source.remove_final_measurements(inplace=False)
    derived_unitary = serialized.remove_final_measurements(inplace=False)
    if source_unitary.num_qubits != derived_unitary.num_qubits:
        raise V2Unavailable("unitary_prefix_width_mismatch")
    width = int(expected_width)
    norm_atol = float(normalization.get("vector_norm_atol", 1e-6))
    fidelity_threshold = float(normalization["equivalence_fidelity_minimum"])
    input_seed = int(normalization["equivalence_input_seed"])
    import numpy as np

    initial_states: list[tuple[str, Any]] = [
        ("zero_state", np.eye(1, 2**width, 0, dtype=np.complex128).reshape(-1)),
        ("seeded_product_state_1", _seeded_product_state(width, input_seed)),
        ("seeded_product_state_2", _seeded_product_state(width, input_seed + 1)),
    ]
    equivalence: list[dict[str, Any]] = []
    for label, initial in initial_states:
        exact = Statevector(initial).evolve(source_unitary).data
        actual = Statevector(initial).evolve(derived_unitary).data
        fidelity = _state_fidelity(exact, actual, 2**width, norm_atol)
        equivalence.append({
            "input": label,
            "input_sha256": sha256_bytes(np.asarray(initial, dtype=np.complex128).tobytes()),
            "fidelity": fidelity,
            "passed": fidelity >= fidelity_threshold,
        })
    failed = [item for item in equivalence if not item["passed"]]
    if failed:
        raise V2Unavailable(f"statevector_equivalence_below_threshold:{failed}")

    try:
        candidate_ir = maestro.QasmToCirc().parse_and_translate(normalized_qasm)
        parser_width = int(candidate_ir.num_qubits)
    except Exception as exc:
        raise V2Unavailable(f"maestro_parser_rejected_normalized_qasm:{type(exc).__name__}:{exc}") from exc
    if parser_width != width:
        raise V2Unavailable(f"maestro_parser_width_mismatch:{parser_width}!={width}")

    unitary_prefix_qasm = qasm2.dumps(derived_unitary)
    global_phase_derived = str(derived.global_phase)
    provenance = {
        "normalization": "qiskit_transpile_exact_basis_v2",
        "qiskit_version": qiskit_version,
        "qasm2_parser": "strict_with_qiskit_legacy_include_and_custom_instruction_tables",
        "basis_gates": list(normalization["basis_gates"]),
        "optimization_level": int(normalization["optimization_level"]),
        "seed_transpiler": int(normalization["seed_transpiler"]),
        "coupling_map": None,
        "backend_target": None,
        "approximation_degree": float(normalization["approximation_degree"]),
        "source_global_phase": source_global_phase,
        "derived_global_phase": global_phase_derived,
        "source_wire_count": source.num_qubits,
        "source_classical_width": source.num_clbits,
        "derived_wire_count": derived.num_qubits,
        "derived_classical_width": derived.num_clbits,
        "measurement_mapping": [[q, c] for q, c in sorted(source_mapping)],
        "source_gate_counts": dict(sorted(source_gate_counts.items())),
        "derived_gate_counts": dict(sorted(derived_gate_counts.items())),
        "equivalence_fidelity_minimum": fidelity_threshold,
        "equivalence_checks": equivalence,
        "maestro_parser": "maestro.QasmToCirc.parse_and_translate",
        "maestro_parser_width": parser_width,
        "trusted_decomposition": "Qiskit 2.5.2 exact basis translation; approximation_degree=1.0",
    }
    return NormalizedQasm(
        status="ok",
        reason="",
        source_qasm_sha256=source_hash,
        normalized_qasm=normalized_qasm,
        normalized_qasm_sha256=sha256_bytes(normalized_qasm.encode("utf-8")),
        provenance=provenance,
        features=dict(sorted(roundtrip_counts.items())),
        unitary_prefix_qasm=unitary_prefix_qasm,
        source_unitary_prefix_qasm=qasm2.dumps(source_unitary),
    )


def statevector_reference_fidelity(
    exact_amplitudes: Any,
    candidate_amplitudes: Any,
    width: int,
    norm_atol: float = 1e-6,
) -> float:
    """Exact normalized state overlap; never invokes mirror fidelity."""
    if width < 1:
        raise V2Unavailable("invalid_quality_width")
    return _state_fidelity(exact_amplitudes, candidate_amplitudes, 2**int(width), norm_atol)


def quality_gate_status(fidelity: float, threshold: float) -> str:
    fidelity = float(fidelity)
    if not math.isfinite(fidelity) or fidelity < 0.0 or fidelity > 1.0:
        return "quality_unavailable"
    return "ok" if fidelity >= float(threshold) else "quality_failed"


def affine_repetition_slope(t256: float, t1024: float) -> float:
    """Affine operation-word slope with repeat-independent intercept removed."""
    t256 = float(t256)
    t1024 = float(t1024)
    if not math.isfinite(t256) or not math.isfinite(t1024) or t256 < 0 or t1024 < 0:
        raise V2Unavailable("invalid_repetition_median")
    return (t1024 - t256) / 768.0


def reduce_session_medians(values: Mapping[int, Mapping[int, float]]) -> float:
    """Median repetitions within each session, then median of session medians."""
    if not values:
        raise V2Unavailable("missing_sessions")
    medians: list[float] = []
    for session in sorted(values):
        reps = [float(v) for _, v in sorted(values[session].items())]
        if not reps or any(not math.isfinite(v) or v < 0 for v in reps):
            raise V2Unavailable(f"invalid_session_repetitions:{session}")
        medians.append(float(statistics.median(reps)))
    return float(statistics.median(medians))


def fit_v2_calibration_rows(
    rows: Iterable[Mapping[str, str]],
    manifest: Mapping[str, Any],
    raw_sha256: str,
) -> Any:
    """Fit v2 while preserving failed/missing cells as unavailable knots.

    Cell-level failures do not discard otherwise usable grid points. Structural
    corruption (unexpected descriptors, duplicate attempt identities, or a
    changed/missing selected clock identity) still fails the fit globally.
    """
    if manifest.get("manifest_id") != V2_MANIFEST_ID:
        raise V2Unavailable("v2_calibration_requires_completion_manifest")
    from benchmark_v1.qre_benchmark.maestro_component_predictor import FittedCalibration

    measurement = manifest["measurement"]
    corpus = manifest["synthetic_corpus"]
    sessions = int(measurement["sessions"])
    reps_per_session = int(measurement["timed_warm_repetitions_per_session"])
    selected_stage = measurement.get("selected_calibration_stage")
    selected_clock_field = measurement.get("selected_clock_field", V2_CLOCK_FIELD)
    expected_attempts = {(session, rep) for session in range(1, sessions + 1)
                         for rep in range(reps_per_session)}
    expected_cells: set[tuple[str, ...]] = set()
    candidates = list(manifest["implementation"]["candidates"])
    widths_by_candidate: dict[str, tuple[int, ...]] = {}
    chi_grid = tuple(int(v) for v in corpus["mps_configured_bonds"])
    for candidate in candidates:
        width_key = "statevector_widths" if candidate == "statevector" else "mps_widths"
        widths_by_candidate[candidate] = tuple(int(v) for v in corpus[width_key])
        bonds: list[int | None] = [None] if candidate == "statevector" else list(chi_grid)
        for width in widths_by_candidate[candidate]:
            for chi in bonds:
                for op in corpus["operation_classes"]:
                    for repeat in corpus["operation_repeat_counts"]:
                        expected_cells.add(("operation", candidate, str(width),
                                            "" if chi is None else str(chi), str(op),
                                            str(corpus["operation_shots"]), str(repeat)))
    for candidate in corpus.get("sampling_candidates", candidates):
        width_key = "statevector_widths" if candidate == "statevector" else "mps_widths"
        widths = tuple(int(v) for v in corpus[width_key])
        bonds = [None] if candidate == "statevector" else list(chi_grid)
        for width in widths:
            for chi in bonds:
                for shots in corpus["sampling_shots"]:
                    expected_cells.add(("sampling", candidate, str(width),
                                        "" if chi is None else str(chi),
                                        "zero_gate_sample", str(shots), "0"))

    grouped: dict[tuple[str, ...], dict[tuple[int, int], tuple[str, float | None, str]]] = defaultdict(dict)
    all_attempt_ids: set[str] = set()
    timed_rows = 0
    for row in rows:
        if row.get("stage") != V2_TIMED_STAGE:
            continue
        timed_rows += 1
        attempt_id = str(row.get("attempt_id", ""))
        if not attempt_id or attempt_id in all_attempt_ids:
            raise V2Unavailable("missing_or_duplicate_timed_attempt_id")
        all_attempt_ids.add(attempt_id)
        clock_id = row.get("clock_id") or row.get("selected_clock_id")
        if clock_id != V2_CLOCK_ID:
            raise V2Unavailable(f"selected_clock_identity_mismatch:{attempt_id}:{clock_id}")
        if selected_stage is not None and row.get("selected_calibration_stage") != selected_stage:
            raise V2Unavailable(
                f"selected_calibration_stage_mismatch:{attempt_id}:"
                f"{row.get('selected_calibration_stage')}!={selected_stage}"
            )
        if row.get("selected_clock_field") != selected_clock_field:
            raise V2Unavailable(
                f"selected_clock_field_mismatch:{attempt_id}:"
                f"{row.get('selected_clock_field')}!={selected_clock_field}"
            )
        key = tuple("" if row.get(name, "") is None else str(row.get(name, "")) for name in (
            "kind", "candidate", "width", "chi", "operation_class", "shots", "operation_repeats"
        ))
        if key not in expected_cells:
            raise V2Unavailable(f"unexpected_calibration_descriptor:{key}")
        session, repetition = int(row["session"]), int(row["repetition"])
        if (session, repetition) not in expected_attempts:
            raise V2Unavailable(f"unexpected_session_or_repetition:{attempt_id}")
        if (session, repetition) in grouped[key]:
            raise V2Unavailable(f"duplicate_cell_repetition:{key}:{session}:{repetition}")
        status = str(row.get("status", "missing_status"))
        raw_value = row.get(V2_CLOCK_FIELD, "")
        value: float | None = None
        reason = ""
        if status == "ok":
            if raw_value in (None, ""):
                reason = "missing_selected_clock"
            else:
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    reason = "non_numeric_selected_clock"
                if value is not None and (not math.isfinite(value) or value < 0):
                    # A negative/non-finite clock violates the global clock
                    # contract and is not a cell-level simulator failure.
                    raise V2Unavailable(f"invalid_selected_clock:{attempt_id}:{value}")
        else:
            reason = f"attempt_status:{status}"
        grouped[key][(session, repetition)] = (status, value, reason)

    cell_medians: dict[tuple[str, ...], float] = {}
    cell_failures: dict[tuple[str, ...], str] = {}
    for key in sorted(expected_cells):
        attempts = grouped.get(key, {})
        reasons: list[str] = []
        missing_attempts = sorted(expected_attempts - set(attempts))
        if missing_attempts:
            reasons.append(f"missing_attempts:{missing_attempts}")
        successful_by_session: dict[int, dict[int, float]] = defaultdict(dict)
        for (session, repetition), (status, value, reason) in attempts.items():
            if status != "ok" or value is None:
                reasons.append(f"session={session},repetition={repetition}:{reason or status}")
            else:
                successful_by_session[session][repetition] = value
        if reasons:
            cell_failures[key] = "incomplete_or_failed_cell:" + ";".join(reasons)
        else:
            try:
                cell_medians[key] = reduce_session_medians(successful_by_session)
            except V2Unavailable as exc:
                cell_failures[key] = f"session_reduction_failed:{exc}"

    def cell(kind: str, candidate: str, width: int, chi: int | None,
             op: str, shots: int, repeat: int) -> tuple[float | None, str | None]:
        key = (kind, candidate, str(width), "" if chi is None else str(chi), op, str(shots), str(repeat))
        if key in cell_medians:
            return cell_medians[key], None
        return None, f"cell={'/'.join(key)}:{cell_failures.get(key, 'missing_cell')}"

    operation_knots: dict[tuple[str, str, int, int | None], float] = {}
    sampling_intercepts: dict[tuple[str, int, int | None], float] = {}
    sampling_slopes: dict[tuple[str, int, int | None], float] = {}
    invalid_knots: dict[tuple[str, str, int, int | None], str] = {}
    repeats = [int(v) for v in corpus["operation_repeat_counts"]]
    if sorted(repeats) != [256, 1024]:
        raise V2Unavailable("v2_repeat_levels_must_be_256_and_1024")
    shot_grid = [int(v) for v in corpus["sampling_shots"]]

    for candidate in candidates:
        widths = widths_by_candidate[candidate]
        bonds: list[int | None] = [None] if candidate == "statevector" else [int(v) for v in corpus["mps_configured_bonds"]]
        for width_value in widths:
            width = int(width_value)
            for chi in bonds:
                sample_pairs = [
                    (shot, *cell("sampling", candidate, width, chi, "zero_gate_sample", shot, 0))
                    for shot in shot_grid
                ]
                sample_failures = [reason for _, _, reason in sample_pairs if reason]
                sample_failure = ";".join(sample_failures) if sample_failures else None
                if sample_failure:
                    invalid_knots[(candidate, "sampling_intercept", width, chi)] = sample_failure
                    invalid_knots[(candidate, "sampling_per_shot", width, chi)] = sample_failure
                elif len(sample_pairs) < 3:
                    failure = "sampling_fit_requires_three_shot_levels"
                    invalid_knots[(candidate, "sampling_intercept", width, chi)] = failure
                    invalid_knots[(candidate, "sampling_per_shot", width, chi)] = failure
                else:
                    xs = [float(shot) for shot, _, _ in sample_pairs]
                    ys = [float(value) for _, value, _ in sample_pairs]
                    x_bar = sum(xs) / len(xs)
                    y_bar = sum(ys) / len(ys)
                    denominator = sum((x - x_bar) ** 2 for x in xs)
                    slope = sum((x - x_bar) * (y - y_bar)
                                for x, y in zip(xs, ys)) / denominator
                    intercept = y_bar - slope * x_bar
                    if not math.isfinite(intercept) or not math.isfinite(slope):
                        reason = "non_finite_sampling_coefficient"
                        invalid_knots[(candidate, "sampling_intercept", width, chi)] = reason
                        invalid_knots[(candidate, "sampling_per_shot", width, chi)] = reason
                    else:
                        if intercept < 0:
                            invalid_knots[(candidate, "sampling_intercept", width, chi)] = "negative_sampling_intercept"
                        else:
                            sampling_intercepts[(candidate, width, chi)] = intercept
                        if slope < 0:
                            invalid_knots[(candidate, "sampling_per_shot", width, chi)] = "negative_sampling_per_shot"
                        else:
                            sampling_slopes[(candidate, width, chi)] = slope

                operation_slopes: dict[str, float] = {}
                operation_failure_parts: dict[str, list[str]] = defaultdict(list)
                for op in corpus["operation_classes"]:
                    if op not in {"one_qubit_noncommuting", "two_qubit_wrapper_control", "two_qubit_cx_interleaved"}:
                        raise V2Unavailable(f"unexpected_operation_class:{op}")
                    by_repeat = {}
                    for repeat in repeats:
                        value, reason = cell("operation", candidate, width, chi, op,
                                             int(corpus["operation_shots"]), repeat)
                        if reason:
                            operation_failure_parts[op].append(reason)
                        else:
                            assert value is not None
                            by_repeat[repeat] = value
                    if op not in operation_failure_parts:
                        try:
                            operation_slopes[op] = affine_repetition_slope(by_repeat[256], by_repeat[1024])
                        except (V2Unavailable, KeyError) as exc:
                            operation_failure_parts[op].append(str(exc))
                operation_failures = {op: ";".join(reasons)
                                      for op, reasons in operation_failure_parts.items()}
                complexity_1q = 2**width if candidate == "statevector" else int(chi) ** 2
                complexity_cx = 2**width if candidate == "statevector" else width * int(chi) ** 3
                one_key = (candidate, "one_qubit", width, chi)
                if "one_qubit_noncommuting" in operation_failures:
                    invalid_knots[one_key] = operation_failures["one_qubit_noncommuting"]
                else:
                    one = operation_slopes["one_qubit_noncommuting"] / (2.0 * complexity_1q)
                    if math.isfinite(one) and one >= 0:
                        operation_knots[one_key] = float(one)
                    else:
                        invalid_knots[one_key] = "negative_or_non_finite_one_qubit_coefficient"
                cx_key = (candidate, "two_qubit_cx", width, chi)
                needed_cx = ("two_qubit_cx_interleaved", "two_qubit_wrapper_control")
                failed_cx_class = next((operation_failures[op] for op in needed_cx if op in operation_failures), None)
                if failed_cx_class:
                    invalid_knots[cx_key] = failed_cx_class
                else:
                    cx = (operation_slopes[needed_cx[0]] - operation_slopes[needed_cx[1]]) / complexity_cx
                    if math.isfinite(cx) and cx >= 0:
                        operation_knots[cx_key] = float(cx)
                    else:
                        invalid_knots[cx_key] = "negative_or_non_finite_cx_differential"

    return FittedCalibration(
        operation_knots=operation_knots,
        sampling_intercept_knots=sampling_intercepts,
        sampling_per_shot_knots=sampling_slopes,
        calibration_raw_sha256=raw_sha256,
        calibration_manifest_id=V2_MANIFEST_ID,
        clock_id=V2_CLOCK_ID,
        calibration_warm_observations=0,
        calibration_timed_observations=timed_rows,
        invalid_knot_states=invalid_knots,
        fit_version="process_isolated_affine_v2",
        interpolation_width_grid=widths_by_candidate,
        interpolation_chi_grid=chi_grid,
    )


def fit_v3_calibration_rows(
    rows: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
    raw_sha256: str,
    *,
    expected_run_identity_sha256: str | None = None,
    rank_preflight_records: Iterable[Mapping[str, Any]] = (),
    partial_resolution: Mapping[str, Any] | None = None,
    partial_quarantine_evidence: Mapping[
        tuple[str, int, int | None], Mapping[str, Any]
    ] | None = None,
) -> Any:
    """Fit S5 paired endpoint coefficients and retain explicit QA failures.

    Only session-reduced 256/1024 values enter operation coefficients. R512
    is a held-out linearity check. Diagnostic entangled sampling rows are
    reported as transfer increments but never fit.
    """
    if contract.get("manifest_id") != V3_MANIFEST_ID:
        raise V3Unavailable("v3_calibration_requires_v3_resolved_contract")
    if "resolution_provenance" not in contract:
        raise V3Unavailable("v3_calibration_requires_resolved_contract")
    if not re.fullmatch(r"[0-9a-f]{64}", str(raw_sha256)):
        raise V3Unavailable("v3_calibration_raw_hash_invalid")
    from benchmark_v1.qre_benchmark.maestro_component_predictor import FittedCalibration

    measurement = contract["measurement"]
    analysis = contract["analysis"]
    corpus = contract["synthetic_corpus"]
    session_count = int(measurement["sessions"])
    repetitions_per_session = int(measurement["timed_repetitions_per_cell_per_session"])
    expected_clock = str(measurement["selected_clock_id"])
    selected_clock_field = str(measurement["selected_clock_field"])
    expected_attempts = {
        (session, repetition)
        for session in range(1, session_count + 1)
        for repetition in range(repetitions_per_session)
    }

    def global_error(reason: str) -> None:
        raise V3Unavailable(reason)

    partial = partial_resolution is not None
    if partial:
        if (partial_resolution.get("contract") is not contract
                and canonical_hash(partial_resolution.get("contract")) != canonical_hash(contract)):
            global_error("partial_fit_parent_contract_mismatch")
        identity = partial_resolution.get("amendment_identity")
        if (not isinstance(identity, Mapping)
                or identity.get("manifest_id") != PARTIAL_MANIFEST_ID
                or identity.get("sha256") != PARTIAL_MANIFEST_SHA256):
            global_error("partial_fit_amendment_identity_mismatch")
        all_cells = build_partial_v3_plan(contract, partial_resolution["quarantine_policy"])
        scheduled_events = build_partial_v3_execution_schedule(contract, all_cells)
        expected_manifest_id = PARTIAL_MANIFEST_ID
    else:
        all_cells = build_v3_plan(contract)
        scheduled_events = build_v3_execution_schedule(contract, all_cells)
        expected_manifest_id = V3_MANIFEST_ID
    scheduled_by_attempt = {
        str(event["attempt_id"]): event for event in scheduled_events
    }
    if len(scheduled_by_attempt) != len(scheduled_events):
        global_error("v3_frozen_schedule_attempt_id_collision")
    cell_by_id = {str(cell["cell_id"]): cell for cell in all_cells}
    primary_by_id = {
        cell_id: cell for cell_id, cell in cell_by_id.items()
        if cell["allocation"] != "entangled_sampling_transfer_diagnostic"
    }
    grouped: dict[str, dict[tuple[int, int], tuple[str, float | None, str]]] = defaultdict(dict)
    diagnostic_groups: dict[str, dict[tuple[int, int], tuple[str, float | None, str]]] = defaultdict(dict)
    attempt_ids: set[str] = set()
    run_ids: set[str] = set()
    worker_environment_ids: set[str] = set()
    worker_context_ids_by_config: dict[tuple[str, str], set[str]] = defaultdict(set)
    resolved_config_ids: dict[tuple[str, int | None], set[str]] = defaultdict(set)
    timed_rows = 0
    untimed_rows = 0
    untimed_status_counts: dict[str, int] = defaultdict(int)

    def validate_scheduled_row(row: Mapping[str, Any], event: Mapping[str, Any],
                               attempt_id: str) -> None:
        for field_name in (
            "manifest_id", "cell_id", "stage", "candidate", "kind", "operation_class",
            "preparation_qasm_sha256", "variable_word_qasm_sha256", "whole_qasm_sha256",
            "pair_group_id", "process_start_method",
        ):
            expected_value = event[field_name]
            actual_value = row.get(field_name)
            if expected_value is None:
                if actual_value not in (None, ""):
                    global_error(f"v3_schedule_provenance_mismatch:{attempt_id}:{field_name}")
            elif str(actual_value) != str(expected_value):
                global_error(f"v3_schedule_provenance_mismatch:{attempt_id}:{field_name}")
        for field_name in (
            "width", "shots", "operation_repeats", "session", "repetition",
            "session_repeat_order_index", "within_repeat_position",
        ):
            expected_value = event[field_name]
            actual_value = row.get(field_name)
            if expected_value is None:
                if actual_value not in (None, ""):
                    global_error(f"v3_schedule_provenance_mismatch:{attempt_id}:{field_name}")
                continue
            try:
                parsed_value = int(actual_value)
            except (TypeError, ValueError):
                global_error(f"v3_schedule_provenance_missing_or_invalid:{attempt_id}:{field_name}")
            if parsed_value != int(expected_value):
                global_error(f"v3_schedule_provenance_mismatch:{attempt_id}:{field_name}")
        expected_chi = event["chi"]
        actual_chi = row.get("chi")
        if expected_chi is None:
            if actual_chi not in (None, ""):
                global_error(f"v3_schedule_provenance_mismatch:{attempt_id}:chi")
        else:
            try:
                parsed_chi = int(actual_chi)
            except (TypeError, ValueError):
                global_error(f"v3_schedule_provenance_missing_or_invalid:{attempt_id}:chi")
            if parsed_chi != int(expected_chi):
                global_error(f"v3_schedule_provenance_mismatch:{attempt_id}:chi")
        isolated = row.get("call_process_isolated")
        if isolated is not True and str(isolated).strip().lower() not in {"true", "1"}:
            global_error(f"v3_schedule_process_isolation_missing:{attempt_id}")
        for count_field in ("preparation_counts", "variable_word_counts", "whole_qasm_counts"):
            expected_counts = event[count_field]
            actual_counts = row.get(count_field)
            if isinstance(actual_counts, str):
                try:
                    actual_counts = json.loads(actual_counts)
                except json.JSONDecodeError:
                    global_error(f"v3_schedule_count_provenance_invalid_json:{attempt_id}:{count_field}")
            if not isinstance(actual_counts, Mapping):
                global_error(f"v3_schedule_count_provenance_missing:{attempt_id}:{count_field}")
            try:
                normalized_counts = {str(key): int(value) for key, value in actual_counts.items()}
            except (TypeError, ValueError):
                global_error(f"v3_schedule_count_provenance_invalid:{attempt_id}:{count_field}")
            if normalized_counts != expected_counts:
                global_error(f"v3_schedule_count_provenance_mismatch:{attempt_id}:{count_field}")

    for source_row in rows:
        row = {str(key): value for key, value in source_row.items()}
        stage = str(row.get("stage", ""))
        row_manifest = str(row.get("manifest_id", ""))
        if row_manifest == V2_MANIFEST_ID or stage == V2_TIMED_STAGE:
            global_error("v3_rejects_v2_calibration_records")
        if row_manifest != expected_manifest_id:
            global_error(f"v3_row_manifest_mismatch:{row.get('attempt_id', '')}")
        if stage not in {V3_TIMED_STAGE, V3_UNTIMED_STAGE}:
            global_error(f"v3_unrecognized_scheduled_stage:{row.get('attempt_id', '')}:{stage}")
        attempt_id = str(row.get("attempt_id", ""))
        if not attempt_id or attempt_id in attempt_ids:
            global_error("v3_missing_or_duplicate_attempt_id")
        attempt_ids.add(attempt_id)
        expected_event = scheduled_by_attempt.get(attempt_id)
        if expected_event is None:
            global_error(f"v3_attempt_id_not_in_frozen_schedule:{attempt_id}")
        validate_scheduled_row(row, expected_event, attempt_id)
        run_identity = str(row.get("run_identity_sha256", ""))
        worker_context = str(row.get("worker_context_sha256", ""))
        worker_environment = str(row.get("worker_environment_sha256", ""))
        config_sha = str(row.get("resolved_config_sha256", ""))
        threading_policy_sha = str(row.get("backend_threading_policy_sha256", ""))
        if not run_identity or not worker_context or not worker_environment or not config_sha or not threading_policy_sha:
            global_error(f"v3_missing_run_or_worker_context_identity:{attempt_id}")
        run_ids.add(run_identity)
        worker_environment_ids.add(worker_environment)
        if expected_run_identity_sha256 is not None and run_identity != expected_run_identity_sha256:
            global_error(f"v3_run_identity_mismatch:{attempt_id}")
        transport_status = str(row.get("transport_status", ""))
        if transport_status != "ok":
            global_error(f"v3_global_transport_error:{attempt_id}:{transport_status}")
        if str(row.get("status", "")) in {
            "transport_error", "hard_exit", "worker_context_drift", "clock_error",
        }:
            global_error(f"v3_global_worker_error:{attempt_id}:{row.get('status')}")

        cell_id = str(row.get("cell_id", ""))
        if cell_id not in cell_by_id:
            global_error(f"v3_unexpected_cell_id:{cell_id}")
        expected_cell = cell_by_id[cell_id]
        config_key = (
            str(expected_cell["candidate"]),
            None if expected_cell["chi"] is None else int(expected_cell["chi"]),
        )
        resolved_config_ids[config_key].add(config_sha)
        worker_context_ids_by_config[(config_sha, threading_policy_sha)].add(worker_context)
        if stage == V3_UNTIMED_STAGE:
            # Untimed process-isolated calls are context/transport checked and
            # retained in protocol coverage, but never enter fitted runtimes.
            untimed_rows += 1
            untimed_status_counts[str(row.get("status", "missing_status"))] += 1
            continue
        timed_rows += 1
        if row.get("clock_id") != expected_clock or row.get("selected_clock_id") != expected_clock:
            global_error(f"v3_selected_clock_identity_mismatch:{attempt_id}")
        if row.get("selected_clock_field") != selected_clock_field:
            global_error(f"v3_selected_clock_field_mismatch:{attempt_id}")
        for field_name in (
            "candidate", "width", "chi", "kind", "operation_class", "shots", "operation_repeats",
            "preparation_qasm_sha256", "variable_word_qasm_sha256", "whole_qasm_sha256",
        ):
            expected_value = expected_cell.get(field_name)
            actual_value = row.get(field_name)
            if actual_value in (None, "") and expected_value is None:
                continue
            if str(actual_value) != str(expected_value):
                global_error(f"v3_cell_provenance_mismatch:{attempt_id}:{field_name}")
        try:
            session = int(row["session"])
            repetition = int(row["repetition"])
        except (KeyError, TypeError, ValueError):
            global_error(f"v3_invalid_session_repetition:{attempt_id}")
        if (session, repetition) not in expected_attempts:
            global_error(f"v3_unexpected_session_repetition:{attempt_id}")
        target = (diagnostic_groups if expected_cell["allocation"]
                  == "entangled_sampling_transfer_diagnostic" else grouped)
        if (session, repetition) in target[cell_id]:
            global_error(f"v3_duplicate_cell_repetition:{cell_id}:{session}:{repetition}")
        status = str(row.get("status", "missing_status"))
        value: float | None = None
        reason = ""
        if status == "ok":
            raw_value = row.get(selected_clock_field)
            if raw_value in (None, ""):
                global_error(f"v3_missing_selected_clock_value:{attempt_id}")
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                global_error(f"v3_non_numeric_selected_clock_value:{attempt_id}")
            if value is None or not math.isfinite(value) or value < 0:
                global_error(f"v3_invalid_selected_clock_value:{attempt_id}:{value}")
        else:
            reason = f"attempt_status:{status}"
        target[cell_id][(session, repetition)] = (status, value, reason)

    if len(run_ids) > 1:
        global_error("v3_run_identity_changed_within_calibration")
    if len(worker_environment_ids) != 1 and timed_rows:
        global_error("v3_worker_environment_missing_or_changed_within_calibration")
    drifting_configs = [key for key, values in resolved_config_ids.items() if len(values) != 1]
    if drifting_configs:
        global_error(f"v3_resolved_config_changed_within_calibration:{drifting_configs}")
    drifting_contexts = [key for key, values in worker_context_ids_by_config.items() if len(values) != 1]
    if drifting_contexts:
        global_error(f"v3_worker_context_changed_within_config:{drifting_contexts}")
    if expected_run_identity_sha256 is not None and run_ids and run_ids != {expected_run_identity_sha256}:
        global_error("v3_calibration_run_identity_mismatch")

    absolute_tolerance = float(analysis["absolute_tolerance_seconds"])
    relative_tolerance = float(analysis["relative_tolerance"])
    cell_medians: dict[str, dict[int, float]] = {}
    cell_failures: dict[str, str] = {}
    cell_qa: dict[str, dict[str, Any]] = {}

    def reduce_cell(cell_id: str, cell: Mapping[str, Any], records: Mapping[tuple[int, int], Any]) -> None:
        reasons: list[str] = []
        missing = sorted(expected_attempts - set(records))
        if missing:
            reasons.append(f"missing_attempts:{missing}")
        medians_by_session: dict[int, float] = {}
        mad_by_session: dict[int, float] = {}
        for session in range(1, session_count + 1):
            values: list[float] = []
            for repetition in range(repetitions_per_session):
                record = records.get((session, repetition))
                if record is None:
                    continue
                status, value, attempt_reason = record
                if status != "ok" or value is None:
                    reasons.append(f"session={session},repetition={repetition}:{attempt_reason or status}")
                else:
                    values.append(float(value))
            if len(values) == repetitions_per_session:
                median = float(statistics.median(values))
                mad = float(statistics.median(abs(value - median) for value in values))
                medians_by_session[session] = median
                mad_by_session[session] = mad
                threshold = max(absolute_tolerance, relative_tolerance * median)
                if mad > threshold:
                    reasons.append(f"unstable_within_session_mad:session={session}:mad={mad}:limit={threshold}")
        between_range: float | None = None
        if len(medians_by_session) == session_count:
            session_medians = list(medians_by_session.values())
            between_range = max(session_medians) - min(session_medians)
            center = float(statistics.median(session_medians))
            threshold = max(absolute_tolerance, relative_tolerance * center)
            if between_range > threshold:
                reasons.append(f"unstable_between_session_medians:range={between_range}:limit={threshold}")
        if reasons:
            cell_failures[cell_id] = ";".join(reasons)
            status = "unavailable"
        else:
            cell_medians[cell_id] = medians_by_session
            status = "ok"
        cell_qa[cell_id] = {
            "status": status, "reasons": reasons,
            "session_medians_seconds": {str(k): v for k, v in medians_by_session.items()},
            "within_session_mad_seconds": {str(k): v for k, v in mad_by_session.items()},
            "between_session_range_seconds": between_range,
            "descriptor": {key: cell.get(key) for key in (
                "candidate", "width", "chi", "kind", "operation_class", "shots", "operation_repeats",
                "allocation",
            )},
        }

    for cell_id, cell in primary_by_id.items():
        reduce_cell(cell_id, cell, grouped.get(cell_id, {}))
    for cell_id, cell in cell_by_id.items():
        if cell["allocation"] == "entangled_sampling_transfer_diagnostic":
            reduce_cell(cell_id, cell, diagnostic_groups.get(cell_id, {}))

    if raw_sha256 == "":
        global_error("v3_calibration_raw_hash_missing")

    if partial:
        if not isinstance(partial_quarantine_evidence, Mapping):
            global_error("partial_fit_quarantine_evidence_missing")
        preflight = validate_partial_v3_rank_preflight_records(
            contract, rank_preflight_records,
            partial_resolution["quarantine_policy"], partial_quarantine_evidence,
        )
    else:
        preflight = validate_v3_rank_preflight_records(contract, rank_preflight_records)
    preflight_by_config = {
        (str(record["candidate"]), int(record["width"]),
         None if record["chi"] is None else int(record["chi"])): record
        for record in preflight["records"]
    }
    operation_knots: dict[tuple[str, str, int, int | None], float] = {}
    sampling_intercepts: dict[tuple[str, int, int | None], float] = {}
    sampling_per_shot: dict[tuple[str, int, int | None], float] = {}
    invalid_knots: dict[tuple[str, str, int, int | None], str] = {}
    operation_diagnostics: dict[str, Any] = {}
    sampling_diagnostics: dict[str, Any] = {}
    entangled_transfer: dict[str, Any] = {}
    by_descriptor: dict[tuple[str, int, int | None, str, int, str, int], str] = {}
    for cell_id, cell in cell_by_id.items():
        key = (
            str(cell["candidate"]), int(cell["width"]),
            None if cell["chi"] is None else int(cell["chi"]),
            str(cell["operation_class"]), int(cell["operation_repeats"]), str(cell["kind"]),
            int(cell["shots"]),
        )
        by_descriptor[key] = cell_id

    def cell_medians_for(candidate: str, width: int, chi: int | None,
                         operation_class: str, repeats: int, kind: str = "operation",
                         shots: int = 1) -> tuple[
                             dict[int, float] | None, str | None, str | None
                         ]:
        key = (candidate, width, chi, operation_class, repeats, kind, shots)
        cell_id = by_descriptor.get(key)
        if cell_id is None:
            return None, None, f"missing_expected_cell:{key}"
        if cell_id in cell_failures:
            return None, cell_id, f"cell_unavailable:{cell_failures[cell_id]}"
        return cell_medians.get(cell_id), cell_id, None

    def increment_gate(values: list[float], label: str) -> str | None:
        if len(values) != session_count:
            return f"unresolved_missing_session_increment:{label}"
        if any(not math.isfinite(value) or value <= 0 for value in values):
            return f"nonpositive_increment:{label}"
        center = float(statistics.median(values))
        if center <= absolute_tolerance:
            return f"unresolved_increment_below_tolerance:{label}:{center}"
        allowed_range = max(absolute_tolerance, relative_tolerance * center)
        observed_range = max(values) - min(values)
        if observed_range > allowed_range:
            return f"unstable_increment:{label}:range={observed_range}:limit={allowed_range}"
        return None

    candidates = [str(value) for value in contract["implementation"]["candidates"]]
    width_by_candidate = {
        "statevector": tuple(int(v) for v in corpus["statevector_widths"]),
        "mps_fixed_chi": tuple(int(v) for v in corpus["mps_widths"]),
    }
    chi_grid = tuple(int(v) for v in corpus["mps_configured_bonds"])
    repeat_grid = tuple(int(v) for v in corpus["operation_repeat_counts"])
    if tuple(sorted(repeat_grid)) != (256, 512, 1024):
        global_error("v3_repeat_grid_must_be_256_512_1024")
    if tuple(sorted(int(v) for v in corpus["fit_repeat_counts"])) != (256, 1024):
        global_error("v3_fit_repeat_grid_must_use_only_endpoints")
    if tuple(int(v) for v in corpus["held_out_repeat_counts"]) != (512,):
        global_error("v3_held_out_repeat_grid_mismatch")
    op_classes = tuple(str(v) for v in corpus["operation_classes"])

    for candidate in candidates:
        widths = width_by_candidate[candidate]
        bonds: tuple[int | None, ...] = (
            (None,) if candidate == "statevector" else chi_grid
        )
        for width in widths:
            for chi in bonds:
                config_label = f"{candidate}:n={width}:chi={chi}"
                preflight_record = preflight_by_config.get((candidate, width, chi))
                rank_reason = None
                if preflight_record is None:
                    rank_reason = f"rank_preflight_missing:{config_label}"
                elif preflight_record.get("preflight_status") != "ok":
                    rank_reason = ";".join(preflight_record.get("preflight_reasons", [])) or (
                        f"rank_preflight_unavailable:{config_label}"
                    )

                op_medians: dict[str, dict[int, dict[int, float]]] = {}
                op_cell_reasons: dict[str, list[str]] = {}
                for op in op_classes:
                    op_medians[op] = {}
                    op_cell_reasons[op] = []
                    for repeats in repeat_grid:
                        session_values, cell_id, failure = cell_medians_for(
                            candidate, width, chi, op, repeats, "operation", 1
                        )
                        if failure:
                            op_cell_reasons[op].append(f"repeat={repeats}:{failure}")
                        elif session_values is not None:
                            op_medians[op][repeats] = session_values

                heldout_reasons: dict[str, list[str]] = {op: [] for op in op_classes}
                heldout_records: dict[str, list[dict[str, Any]]] = {op: [] for op in op_classes}
                endpoint_increments: dict[str, list[float]] = {op: [] for op in op_classes}
                for op in op_classes:
                    values_by_repeat = op_medians[op]
                    if set(values_by_repeat) != set(repeat_grid):
                        continue
                    for session in range(1, session_count + 1):
                        t256 = values_by_repeat[256][session]
                        t512 = values_by_repeat[512][session]
                        t1024 = values_by_repeat[1024][session]
                        predicted_512 = t256 + (256.0 / 768.0) * (t1024 - t256)
                        deviation = abs(t512 - predicted_512)
                        limit = max(absolute_tolerance, relative_tolerance * t512)
                        passed = deviation <= limit
                        heldout_records[op].append({
                            "session": session, "observed_t512_seconds": t512,
                            "endpoint_affine_t512_seconds": predicted_512,
                            "absolute_deviation_seconds": deviation,
                            "tolerance_seconds": limit, "passed": passed,
                        })
                        if not passed:
                            heldout_reasons[op].append(
                                f"nonlinear_held_out_512:session={session}:"
                                f"deviation={deviation}:limit={limit}"
                            )
                        endpoint_increments[op].append(t1024 - t256)

                one_key = (candidate, "one_qubit", width, chi)
                cx_key = (candidate, "two_qubit_cx", width, chi)
                one_reasons: list[str] = []
                cx_reasons: list[str] = []
                if rank_reason:
                    one_reasons.append(rank_reason)
                    cx_reasons.append(rank_reason)
                one_op = "one_qubit_noncommuting"
                if op_cell_reasons.get(one_op):
                    one_reasons.extend(op_cell_reasons[one_op])
                if heldout_reasons.get(one_op):
                    one_reasons.extend(heldout_reasons[one_op])
                one_gate = increment_gate(endpoint_increments.get(one_op, []), "one_qubit")
                if one_gate:
                    one_reasons.append(one_gate)
                one_coeff: float | None = None
                if not one_reasons:
                    one_coeff = (
                        float(statistics.median(endpoint_increments[one_op])) / 768.0
                        / (2.0 * (2**width if candidate == "statevector" else int(chi)**2))
                    )
                    if not math.isfinite(one_coeff) or one_coeff < 0:
                        one_reasons.append("nonpositive_or_nonfinite_one_qubit_coefficient")
                if one_reasons:
                    invalid_knots[one_key] = ";".join(one_reasons)
                else:
                    operation_knots[one_key] = float(one_coeff)

                wrapper_op = "two_qubit_wrapper_control"
                cx_op = "two_qubit_cx_interleaved"
                for op in (wrapper_op, cx_op):
                    if op_cell_reasons.get(op):
                        cx_reasons.extend(op_cell_reasons[op])
                    if heldout_reasons.get(op):
                        cx_reasons.extend(heldout_reasons[op])
                cx_increments: list[float] = []
                if (set(op_medians.get(wrapper_op, {})) == set(repeat_grid)
                        and set(op_medians.get(cx_op, {})) == set(repeat_grid)):
                    for session in range(1, session_count + 1):
                        wrapper_delta = (op_medians[wrapper_op][1024][session]
                                         - op_medians[wrapper_op][256][session])
                        cx_delta = (op_medians[cx_op][1024][session]
                                    - op_medians[cx_op][256][session])
                        cx_increments.append(cx_delta - wrapper_delta)
                else:
                    cx_reasons.append("unresolved_missing_endpoint_pair:cx_minus_wrapper")
                cx_gate = increment_gate(cx_increments, "cx_minus_wrapper")
                if cx_gate:
                    cx_reasons.append(cx_gate)
                cx_coeff: float | None = None
                if not cx_reasons:
                    complexity = (2**width if candidate == "statevector"
                                  else width * int(chi)**3)
                    cx_coeff = float(statistics.median(cx_increments)) / 768.0 / complexity
                    if not math.isfinite(cx_coeff) or cx_coeff < 0:
                        cx_reasons.append("nonpositive_or_nonfinite_cx_coefficient")
                if cx_reasons:
                    invalid_knots[cx_key] = ";".join(cx_reasons)
                else:
                    operation_knots[cx_key] = float(cx_coeff)

                operation_diagnostics[config_label] = {
                    "rank_preflight": preflight_record,
                    "one_qubit_endpoint_increments_seconds": endpoint_increments.get(one_op, []),
                    "cx_minus_wrapper_endpoint_increments_seconds": cx_increments,
                    "one_qubit_identification_reasons": one_reasons,
                    "cx_identification_reasons": cx_reasons,
                    "held_out_512_checks": heldout_records,
                    "native_max_bond_dim_reached": (
                        preflight_record.get("max_bond_dim_reached") if preflight_record else None
                    ),
                    "native_telemetry_status": (
                        preflight_record.get("telemetry_status", "unknown")
                        if candidate == "mps_fixed_chi" else "not_applicable"
                    ),
                    "claim_boundary": "empirical_probe_ensemble_not_a_general_complexity_law",
                }

    sampling_shot_grid = tuple(int(v) for v in corpus["sampling_shots"])
    for candidate in candidates:
        widths = width_by_candidate[candidate]
        bonds = (None,) if candidate == "statevector" else chi_grid
        for width in widths:
            for chi in bonds:
                label = f"{candidate}:n={width}:chi={chi}"
                fit_cells: list[tuple[int, float]] = []
                failures: list[str] = []
                for shots in sampling_shot_grid:
                    session_values, cell_id, failure = cell_medians_for(
                        candidate, width, chi, "zero_gate_sample", 0, "sampling", shots
                    )
                    if failure:
                        failures.append(f"shots={shots}:{failure}")
                    elif session_values is not None:
                        fit_cells.append((shots, float(statistics.median(session_values.values()))))
                intercept_key = (candidate, width, chi)
                slope_key = (candidate, "sampling_per_shot", width, chi)
                intercept_invalid_key = (candidate, "sampling_intercept", width, chi)
                slope_invalid_key = (candidate, "sampling_per_shot", width, chi)
                diagnostics: dict[str, Any] = {
                    "cells": [{"shots": shots, "median_seconds": value} for shots, value in fit_cells],
                    "residuals_seconds": [], "coefficient_status": "unavailable",
                    "reasons": failures,
                }
                if failures or len(fit_cells) != len(sampling_shot_grid):
                    reason = ";".join(failures) if failures else "sampling_fit_missing_shot_cells"
                    invalid_knots[intercept_invalid_key] = reason
                    invalid_knots[slope_invalid_key] = reason
                else:
                    xs = [float(shot) for shot, _ in fit_cells]
                    ys = [float(value) for _, value in fit_cells]
                    xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
                    denominator = sum((x - xbar) ** 2 for x in xs)
                    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator
                    intercept = ybar - slope * xbar
                    residuals = [value - (intercept + slope * shot) for shot, value in fit_cells]
                    diagnostics.update({
                        "intercept_seconds": intercept, "per_shot_seconds": slope,
                        "residuals_seconds": residuals, "coefficient_status": "ok",
                    })
                    if not math.isfinite(intercept) or not math.isfinite(slope):
                        invalid_knots[intercept_invalid_key] = "non_finite_sampling_coefficient"
                        invalid_knots[slope_invalid_key] = "non_finite_sampling_coefficient"
                        diagnostics["coefficient_status"] = "unavailable"
                    else:
                        if intercept < 0:
                            invalid_knots[intercept_invalid_key] = "negative_sampling_intercept"
                        else:
                            sampling_intercepts[intercept_key] = intercept
                        if slope < 0:
                            invalid_knots[slope_invalid_key] = "negative_sampling_per_shot"
                        else:
                            sampling_per_shot[intercept_key] = slope
                        if (intercept < 0) or (slope < 0):
                            diagnostics["coefficient_status"] = "unavailable"
                sampling_diagnostics[label] = diagnostics

    # Entangled-shot cells are transfer diagnostics only: compare increments
    # so fixed preparation cost cancels; never feed these cells into OLS.
    diagnostic = corpus["entangled_sampling_diagnostic"]
    for width_value in diagnostic["widths"]:
        width = int(width_value)
        for chi_value in diagnostic["configured_bonds"]:
            chi = int(chi_value)
            config_label = f"mps_fixed_chi:n={width}:chi={chi}"
            per_session: list[dict[str, Any]] = []
            failures: list[str] = []
            for session in range(1, session_count + 1):
                values_by_shot: dict[int, float] = {}
                for shots in diagnostic["shots"]:
                    key = ("mps_fixed_chi", width, chi, "entangled_sampling_diagnostic",
                           0, "sampling_diagnostic", int(shots))
                    cell_id = by_descriptor.get(key)
                    if cell_id is None or cell_id not in cell_medians:
                        failures.append(f"diagnostic_cell_unavailable:shots={shots}:session={session}")
                        continue
                    values_by_shot[int(shots)] = cell_medians[cell_id][session]
                if len(values_by_shot) == len(diagnostic["shots"]):
                    shot1, shot_mid, shot_high = [int(value) for value in diagnostic["shots"]]
                    delta_low = values_by_shot[shot_mid] - values_by_shot[shot1]
                    delta_high = values_by_shot[shot_high] - values_by_shot[shot_mid]
                    reference_rate = sampling_per_shot.get(("mps_fixed_chi", width, chi))
                    per_session.append({
                        "session": session,
                        "shots": values_by_shot,
                        "increment_1_to_1000_seconds": delta_low,
                        "increment_1000_to_10000_seconds": delta_high,
                        "reference_zero_state_per_shot_seconds": reference_rate,
                        "zero_state_expected_increment_1_to_1000_seconds": (
                            None if reference_rate is None else reference_rate * (shot_mid-shot1)
                        ),
                        "zero_state_expected_increment_1000_to_10000_seconds": (
                            None if reference_rate is None else reference_rate * (shot_high-shot_mid)
                        ),
                    })
            entangled_transfer[config_label] = {
                "status": "ok" if not failures and len(per_session) == session_count else "unavailable",
                "reasons": failures,
                "session_increments": per_session,
                "fit_use": "diagnostic_only_never_refits_zero_state_model",
            }

    if partial:
        for chi in (4, 8, 32):
            for component in ("one_qubit", "two_qubit_cx"):
                key = ("mps_fixed_chi", component, 16, chi)
                operation_knots.pop(key, None)
                invalid_knots[key] = PARTIAL_QUARANTINE_REASON

    expected_operation_knots = sum(
        len(width_by_candidate[candidate]) * (1 if candidate == "statevector" else len(chi_grid)) * 2
        for candidate in candidates
    )
    expected_sampling_knots = sum(
        len(width_by_candidate[candidate]) * (1 if candidate == "statevector" else len(chi_grid)) * 2
        for candidate in candidates
    )
    eligible_cell_count = (sum(cell.get("calibration_eligibility") == "eligible" for cell in all_cells)
                           if partial else len(all_cells))
    quarantined_cell_count = len(all_cells) - eligible_cell_count
    expected_timed_calls = eligible_cell_count * session_count * repetitions_per_session
    expected_untimed_calls = (
        eligible_cell_count * session_count
        * int(measurement["untimed_calls_per_cell_per_session"])
    )
    fit_version = PARTIAL_FIT_VERSION if partial else "paired_endpoint_heldout_512_v3"
    fit_manifest_id = PARTIAL_MANIFEST_ID if partial else V3_MANIFEST_ID
    fit_diagnostics = {
        "fit_version": fit_version,
        "calibration_manifest_id": fit_manifest_id,
        "raw_sha256": raw_sha256,
        "run_identity_sha256": next(iter(run_ids), ""),
        "worker_environment_sha256": next(iter(worker_environment_ids), ""),
        "worker_context_sha256_by_config_policy": {
            f"{config_sha}:{policy_sha}": next(iter(values))
            for (config_sha, policy_sha), values in sorted(worker_context_ids_by_config.items())
        },
        "cell_qa": cell_qa,
        "operation_configs": operation_diagnostics,
        "sampling_configs": sampling_diagnostics,
        "entangled_sampling_transfer": entangled_transfer,
        "rank_preflight": preflight,
        "coverage": {
            "planned_primary_cells": len(primary_by_id),
            "planned_diagnostic_cells": len(cell_by_id) - len(primary_by_id),
            "eligible_cells": eligible_cell_count,
            "quarantined_cells": quarantined_cell_count,
            "quarantine_reason": PARTIAL_QUARANTINE_REASON if partial else "",
            "complete_stable_primary_cells": sum(
                cell_qa[cell_id]["status"] == "ok" for cell_id in primary_by_id
            ),
            "timed_attempt_rows": timed_rows,
            "valid_operation_prediction_knots": len(operation_knots),
            "unavailable_operation_prediction_knots": len(
                [key for key in invalid_knots if key[1] in {"one_qubit", "two_qubit_cx"}]
            ),
            "expected_operation_prediction_knots": expected_operation_knots,
            "valid_sampling_prediction_knots": len(sampling_intercepts) + len(sampling_per_shot),
            "unavailable_sampling_prediction_knots": len(
                [key for key in invalid_knots if key[1] in {"sampling_intercept", "sampling_per_shot"}]
            ),
            "expected_sampling_prediction_knots": expected_sampling_knots,
            "synthetic_quality_status": "not_required",
            "protocol_calls": {
                "expected_timed_calls": expected_timed_calls,
                "observed_timed_calls": timed_rows,
                "expected_untimed_calls": expected_untimed_calls,
                "observed_untimed_calls": untimed_rows,
                "untimed_status_counts": dict(sorted(untimed_status_counts.items())),
                "untimed_protocol_status": (
                    "complete"
                    if (untimed_rows == expected_untimed_calls
                        and set(untimed_status_counts) == {"ok"})
                    else "incomplete_or_failed"
                ),
            },
        },
        "panel_attempt_status_schema": V3_PANEL_ATTEMPT_STATUS_SCHEMA,
        "claim_boundary": "probe_ensemble_coefficients_not_general_cost_law",
    }
    return FittedCalibration(
        operation_knots=operation_knots,
        sampling_intercept_knots=sampling_intercepts,
        sampling_per_shot_knots=sampling_per_shot,
        calibration_raw_sha256=raw_sha256,
        calibration_manifest_id=fit_manifest_id,
        clock_id=expected_clock,
        calibration_warm_observations=0,
        calibration_timed_observations=timed_rows,
        invalid_knot_states=invalid_knots,
        fit_version=fit_version,
        interpolation_width_grid=width_by_candidate,
        interpolation_chi_grid=chi_grid,
        fit_diagnostics=fit_diagnostics,
        resolved_contract_sha256=(
            str(partial_resolution["resolved_contract_sha256"])
            if partial else canonical_hash(contract)
        ),
        run_identity_sha256=next(iter(run_ids), ""),
    )


def resolve_v2_config(maestro: Any, candidate: str, chi: int | None, seed: int,
                      singular_value_threshold: float) -> dict[str, Any]:
    """Create a QCSim config and serialize every exposed public field."""
    simulation = (maestro.SimulationType.Statevector if candidate == "statevector"
                  else maestro.SimulationType.MatrixProductState)
    kwargs: dict[str, Any] = {
        "simulator_type": maestro.SimulatorType.QCSim,
        "simulation_type": simulation,
        "seed": int(seed),
    }
    if candidate != "statevector":
        kwargs["max_bond_dimension"] = int(chi) if chi is not None else None
        kwargs["singular_value_threshold"] = float(singular_value_threshold)
    config = maestro.SimulatorConfig(**kwargs)

    def stable_value(value: Any, field_name: str) -> Any:
        if isinstance(value, enum.Enum):
            enum_value = value.value
            if enum_value is None or isinstance(enum_value, (bool, int, float, str)):
                canonical_enum_value = enum_value
            else:
                raise V2Unavailable(f"unsupported_config_enum_value:{field_name}:{type(enum_value).__name__}")
            return {
                "enum_type": f"{type(value).__module__}.{type(value).__qualname__}",
                "name": value.name,
                "value": canonical_enum_value,
            }
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise V2Unavailable(f"non_finite_config_value:{field_name}")
            return value
        if isinstance(value, (tuple, list)):
            return [stable_value(item, field_name) for item in value]
        if isinstance(value, Mapping):
            return {str(key): stable_value(item, field_name) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        raise V2Unavailable(f"unsupported_config_field_type:{field_name}:{type(value).__module__}.{type(value).__qualname__}")

    fields: dict[str, Any] = {}
    public_names = [name for name in dir(config) if not name.startswith("_")]
    for name in sorted(public_names):
        try:
            value = getattr(config, name)
        except Exception as exc:
            raise V2Unavailable(f"config_field_read_failed:{name}:{type(exc).__name__}") from exc
        if callable(value):
            continue
        fields[name] = stable_value(value, name)
    fields["_candidate"] = candidate
    fields["_configured_chi"] = chi
    fields["_singular_value_threshold"] = float(singular_value_threshold) if candidate != "statevector" else None
    fields["_seed"] = int(seed)
    return {"object": config, "resolved": fields, "sha256": canonical_hash(fields)}


def validate_resume_identity(existing: Mapping[str, Any], expected_identity: Mapping[str, Any]) -> None:
    """Fail closed unless a checkpoint is pinned to exactly the current run."""
    if canonical_hash(existing) != canonical_hash(expected_identity):
        raise V2Unavailable("checkpoint_identity_mismatch")
