#!/usr/bin/env python3
"""Materialize the frozen Maestro optimizer-off overlay as a pinned runtime protocol.

The parent protocol remains immutable.  The output is a derived run input that
contains the parent's unchanged panel/calibration/fold contract plus only the
explicit optimizer-off overrides and the actual isolated native-library pins.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = ROOT / "benchmark_v1/protocol/maestro_cpu_component_benchmark.json"
DEFAULT_OVERLAY = ROOT / "benchmark_v1/protocol/maestro_cpu_component_optimizer_off.json"
EXPECTED_PATCH_SHA256 = "386a945dde3a503488d9abd68b1235e2785327175963f888c1296c20fb014c13"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def resolve_protocol(
    parent_path: Path,
    overlay_path: Path,
    native_module_path: Path,
    boost_library_path: Path,
    openmp_library_path: Path,
) -> dict[str, Any]:
    parent_path, overlay_path = parent_path.resolve(), overlay_path.resolve()
    native_module_path = native_module_path.resolve()
    boost_library_path = boost_library_path.resolve()
    openmp_library_path = openmp_library_path.resolve()
    parent, overlay = read_json(parent_path), read_json(overlay_path)

    parent_pin = overlay["parent_protocol"]
    if parent_pin["path"] != str(parent_path.relative_to(ROOT)):
        raise ValueError("overlay parent path differs from the requested parent protocol")
    parent_sha = sha256_file(parent_path)
    if parent_sha != parent_pin["sha256"]:
        raise ValueError("parent protocol hash differs from the frozen overlay pin")
    if overlay.get("status") != "design_locked_implementation_and_execution_pending":
        raise ValueError("optimizer-off overlay is not in its frozen pending state")
    override = overlay["context_overrides"]
    if override.get("optimize_circuit") is not False:
        raise ValueError("optimizer-off contract must explicitly request optimize_circuit=false")
    if override.get("patch_sha256") != EXPECTED_PATCH_SHA256:
        raise ValueError("optimizer-off contract patch SHA differs from the reviewed patch")
    patch = ROOT / override["patch_path"]
    if not patch.is_file():
        raise FileNotFoundError(
            "required optimizer-off patch is unavailable: "
            f"{override['patch_path']} (GPL-derived dependency withheld pending rights review; "
            f"expected SHA-256 {EXPECTED_PATCH_SHA256})"
        )
    if sha256_file(patch) != EXPECTED_PATCH_SHA256:
        raise ValueError("checked-in optimizer diagnostic patch hash changed")
    revision_match = re.search(r"\b[0-9a-f]{40}\b", override["native_build"])
    if not revision_match or parent["local_execution_environment"]["upstream_source_revision"] != revision_match.group(0):
        raise ValueError("optimizer-off upstream revision differs from the pinned parent source revision")
    for path in (native_module_path, boost_library_path, openmp_library_path):
        if not path.is_file():
            raise FileNotFoundError(f"required pinned native input is missing: {path}")

    resolved = copy.deepcopy(parent)
    resolved["protocol_id"] = overlay["protocol_id"]
    resolved["status"] = "resolved_runtime_protocol"
    resolved["method_id"] = overlay["method_id"]
    resolved["reader_label"] = overlay["reader_label"]
    resolved["fidelity_class"] = overlay["fidelity_class"]
    resolved["context"].update({
        "engine": override["simulator"],
        "precision": override["precision"],
        "optimize_circuit": False,
        "seed_context_id": override["seed_context_id"],
        "native_threads_primary": 1,
        "primary_thread_policy_selected_before_timing": True,
        "resolved_optimizer_flag_required": override["resolved_flag_required"],
    })
    resolved["measurement"]["clock"] = overlay["evaluation"]["evaluation_target_clock"]
    resolved["evaluation"] = copy.deepcopy(overlay["evaluation"])
    resolved["execution_context_id"] = overlay["evaluation"]["execution_context_id"]
    resolved["local_execution_environment"]["native_module_path"] = str(native_module_path)
    resolved["local_execution_environment"]["native_module_sha256"] = sha256_file(native_module_path)
    resolved["local_execution_environment"]["optimizer_off_patch_sha256"] = EXPECTED_PATCH_SHA256
    resolved["local_execution_environment"]["optimizer_off_source_revision"] = revision_match.group(0)
    resolved["local_execution_environment"]["runtime_link_dependency"]["library_path"] = str(boost_library_path)
    resolved["local_execution_environment"]["runtime_link_dependency"]["library_sha256"] = sha256_file(boost_library_path)
    resolved["local_execution_environment"]["openmp_runtime"]["library_path"] = str(openmp_library_path)
    resolved["local_execution_environment"]["openmp_runtime"]["library_sha256"] = sha256_file(openmp_library_path)
    resolved["measurement_artifact_contract"]["root"] = overlay["execution"]["artifact_root"]
    resolved["measurement_artifact_contract"]["optimizer_flag_columns"] = [
        "optimizer_enabled_requested", "optimizer_enabled_resolved"
    ]
    resolved["population"] = copy.deepcopy(overlay["population"])
    resolved["resolved_from"] = {
        "parent_protocol_path": str(parent_path.relative_to(ROOT)),
        "parent_protocol_sha256": parent_sha,
        "overlay_protocol_path": str(overlay_path.relative_to(ROOT)),
        "overlay_protocol_sha256": sha256_file(overlay_path),
        "resolver_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--native-module", type=Path, required=True)
    parser.add_argument("--boost-library", type=Path, required=True)
    parser.add_argument("--openmp-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite resolved protocol: {args.output}")
    resolved = resolve_protocol(args.parent, args.overlay, args.native_module,
                                args.boost_library, args.openmp_library)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(resolved, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "resolved", "protocol_id": resolved["protocol_id"],
                      "output": str(args.output.resolve()),
                      "resolved_protocol_sha256": sha256_file(args.output),
                      "parent_protocol_sha256": resolved["resolved_from"]["parent_protocol_sha256"],
                      "overlay_protocol_sha256": resolved["resolved_from"]["overlay_protocol_sha256"],
                      "native_module_sha256": resolved["local_execution_environment"]["native_module_sha256"]},
                     sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
