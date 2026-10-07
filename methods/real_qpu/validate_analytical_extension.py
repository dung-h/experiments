"""Read-only design/input checks, not the analytical benchmark runner."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path("benchmark_v1/protocol/analytical_extension.json")
METHOD_IDS = {
    "hyb_nominal_r2_effective_cost_v1", "hyb_nominal_r2_shot_effective_cost_v1",
    "hyb_kyoto_composite_r2_effective_cost_v1", "hyb_kyoto_composite_r2_shot_effective_cost_v1",
    "hyb_nominal_r2_log_cost_ridge_v1", "hyb_kyoto_composite_r2_log_cost_ridge_v1",
    "hyb_kyoto_composite_r2_gate_time_ridge_v1", "qcre_nominal_duration_ratio_rank_v1",
    "scholten_reported_nominal_throughput_v1", "qpu_depth_shots_ridge_v1",
    "qpu_seven_global_ridge_v1",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative_path(value):
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def validate_contract(protocol):
    """Check high-risk locked decisions; input validation is separate."""
    errors = []

    def expect(path, value):
        current = protocol
        for key in path.split("."):
            current = current.get(key) if isinstance(current, dict) else None
        # bool is an int subclass; require exact scalar type as well as equality.
        if current != value or (isinstance(value, (bool, int, str)) and type(current) is not type(value)):
            errors.append(f"locked decision mismatch: {path}")

    locked = {
        "protocol_id": "archived-qpu-analytical-extension-v1",
        "status": "design_frozen_implementation_required",
        "execution_started": False,
        "retrospective_extension": True,
        "assigned_rows_per_method": 8767,
        "outer_folds": list(range(5)),
        "inner_folds": list(range(4)),
        "sources": ["mali_real_qpu", "qonductor_single_circuit_ibm", "qpack_mcp"],
        "evaluation_target_clock": "archived_observed_service_execution_time",
        "representation.retranspile": False,
        "representation.hash_graph_before_loading_pickle": True,
        "representation.unavailable_row": "qonductor_single_circuit_ibm|row4477",
        "representation.row4477_method_policy.metadata_only_controls": "predict when their saved global features and shots are valid; keep this as a rare-control-flow stress case, not branch-aware inference",
        "calibration.selection_uses_runtime_labels": False,
        "calibration.kyoto_composite.tier": "tier_3_backend_matched_composite_nominal",
        "calibration.kyoto_composite.expected_directed_ecr_edges": 144,
        "calibration.kyoto_composite.error_one_edges": [[94, 90], [106, 107], [107, 108], [112, 108], [117, 116], [118, 117]],
        "calibration.kyoto_composite.drop_error_one_edges": False,
        "calibration.kyoto_composite.error_clipping": False,
        "calibration.kyoto_composite.fallback_for_missing_csv_required_field": False,
        "hyb.mean_T2_sensitivity_in_queue": False,
        "ridge.alphas": [0.1, 1.0, 10.0, 100.0],
        "ridge.hyb_features": ["asinh(log_effective_cost_seconds)", "log1p(shots)"],
        "ridge.source_or_backend_as_feature": False,
        "qcre_rank.score_as_seconds": False,
        "qcre_rank.calibrate_in_this_queue": False,
        "scholten.definition": "reported_CLOPS_definition_unspecified",
        "scholten.CLOPS_h_to_v_conversion": False,
        "scholten.active_entries": {
            "ibm_brisbane": {"CLOPS": 5000, "backend_version": "1.1.33"},
            "ibm_kyoto": {"CLOPS": 5000, "backend_version": "1.2.38"},
            "ibm_osaka": {"CLOPS": 5000, "backend_version": "1.1.8"},
        },
        "execution.neural_training": False,
        "execution.gpu_required": False,
        "execution.native_threads_per_worker": 1,
        "execution.performance_is_not_acceptance_gate": True,
        "execution.terminal_states": ["predicted", "unavailable", "overflow", "underflow", "failed"],
        "metrics.bootstrap.replicates": 10000,
        "metrics.bootstrap.primary_pairs": [["hyb_nominal_r2_log_cost_ridge_v1", "hyb_kyoto_composite_r2_log_cost_ridge_v1"], ["hyb_kyoto_composite_r2_log_cost_ridge_v1", "hyb_kyoto_composite_r2_gate_time_ridge_v1"], ["qpu_depth_shots_ridge_v1", "qpu_seven_global_ridge_v1"]],
    }
    for path, value in locked.items():
        expect(path, value)
    cards = protocol.get("methods", [])
    ids = [card.get("method_id") for card in cards]
    if len(ids) != len(METHOD_IDS) or set(ids) != METHOD_IDS:
        errors.append("eleven distinct frozen method IDs required")
    for card in cards:
        if card.get("fidelity_class") not in {"analytical_proxy", "adaptation", "baseline"}:
            errors.append("new method fidelity class required")
        if not card.get("reader_label") or not card.get("method_output_clock"):
            errors.append("reader label and output clock required")
    groups = protocol.get("method_input_groups", {})
    graph_methods = set(groups.get("requires_graph_and_backend_duration_or_calibration", []))
    metadata_methods = set(groups.get("uses_saved_global_metadata_and_shots_only", []))
    if graph_methods | metadata_methods != METHOD_IDS or graph_methods & metadata_methods:
        errors.append("method input groups must partition the eleven method IDs")
    pins = protocol.get("input_pins", {})
    expected_pins = {"canonical", "outer", "inner", "representation_rows", "graph_hashes",
                     "nominal_snapshots", "kyoto_candidate", "seeds", "fidelity_guard", "current_guard"}
    if set(pins) != expected_pins:
        errors.append("complete input/immutability pin set required")
    for name, pin in pins.items():
        if not safe_relative_path(pin.get("path", "")) or not re.fullmatch(r"[0-9a-f]{64}", pin.get("sha256", "")):
            errors.append(f"invalid path/hash pin: {name}")
    code_pins = protocol.get("reference_code_pins", {})
    if set(code_pins) != {"compiled_materializer", "snapshot_reader", "historical_hyb", "historical_analytical", "compiled_polynomial", "compiled_graph_mlp"}:
        errors.append("complete reference-code pin set required")
    for name, pin in code_pins.items():
        if not safe_relative_path(pin.get("path", "")) or not re.fullmatch(r"[0-9a-f]{64}", pin.get("sha256", "")):
            errors.append(f"invalid reference-code path/hash: {name}")
    artifact_pins = protocol.get("preflight_artifact_pins", {})
    if set(artifact_pins) != {"kyoto_composite_calibration", "preprocessing_dependency_inventory"}:
        errors.append("complete preflight artifact pin set required")
    for name, pin in artifact_pins.items():
        if not safe_relative_path(pin.get("path", "")) or not re.fullmatch(r"[0-9a-f]{64}", pin.get("sha256", "")):
            errors.append(f"invalid preflight-artifact path/hash: {name}")
    forbidden = {"rewrite_raw", "change_canonical_targets", "change_splits", "write_CURRENT", "rewrite_fidelity_registry",
                 "write_selected_export", "V4_training", "neural_training", "live_QPU", "simulator_timing",
                 "healthy_edge_recompilation", "held_out_label_selection", "git_mutation", "delete_artifacts"}
    if not forbidden <= set(protocol.get("prohibited", [])):
        errors.append("scope/immutability prohibitions missing")
    fields = protocol.get("calibration", {}).get("kyoto_composite", {}).get("csv_fields", {})
    for name, scale in (("T2 (us)", 1e-6), ("Readout length (ns)", 1e-9), ("Gate time (ns)", 1e-9)):
        if fields.get(name, {}).get("scale") != scale:
            errors.append(f"SI unit conversion mismatch: {name}")
    execution = protocol.get("execution", {})
    if not 1 <= execution.get("workers_default", 0) <= execution.get("workers_max", 0) <= 8:
        errors.append("bounded CPU workers required")
    if not safe_relative_path(execution.get("output_root", "")):
        errors.append("relative semantic output root required")
    return errors


def validate_inputs(protocol, root, include_derived=True):
    errors = []
    pins = {**protocol["input_pins"], **protocol.get("reference_code_pins", {})}
    if include_derived:
        pins.update(protocol.get("preflight_artifact_pins", {}))
    for name, pin in pins.items():
        path = root / pin["path"]
        if not path.is_file():
            errors.append(f"missing input: {name}: {pin['path']}")
        elif sha256_file(path) != pin["sha256"]:
            errors.append(f"input hash mismatch: {name}: {pin['path']}")
    if not errors:
        registry = json.loads((root / protocol["input_pins"]["fidelity_guard"]["path"]).read_text())
        collisions = {card["method_id"] for card in registry["methods"]} & METHOD_IDS
        if collisions:
            errors.append(f"historical method ID collision: {sorted(collisions)}")
        if include_derived:
            inventory = json.loads((root / protocol["preflight_artifact_pins"]["preprocessing_dependency_inventory"]["path"]).read_text())
            sidecar_pin = protocol["preflight_artifact_pins"]["kyoto_composite_calibration"]
            sidecar = inventory["frozen_feature_sidecars"]["kyoto_composite_calibration"]
            if sidecar["path"] != sidecar_pin["path"] or sidecar["sha256"] != sidecar_pin["sha256"]:
                errors.append("dependency inventory does not pin the Kyoto calibration sidecar")
            calibration = json.loads((root / sidecar_pin["path"]).read_text())
            if calibration.get("schema") != "kyoto-composite-calibration-sidecar-v1" or calibration.get("job_day_historical_match") is not False:
                errors.append("Kyoto sidecar identity or provenance tier is invalid")
            if len(calibration.get("qubits", {})) != 127 or len(calibration.get("directed_ecr", {})) != 144:
                errors.append("Kyoto sidecar qubit/ECR coverage mismatch")
            if calibration.get("candidate_error_one_directed_edges") != ["94_90", "106_107", "107_108", "112_108", "117_116", "118_117"]:
                errors.append("Kyoto sidecar changed the six error-one edges")
            source_rows = inventory.get("raw_archived_observation_sources", {})
            if {key: item.get("row_count") for key, item in source_rows.items()} != {
                "mali_observations": 340, "qonductor_observations": 4482, "qpack_observations": 3945
            }:
                errors.append("source row-count inventory mismatch")
            if inventory.get("source_circuit_inputs", {}).get("mali_logical_qasm", {}).get("available_circuits") != 1510:
                errors.append("Ma-Li raw QASM inventory incomplete")
    return errors


def reference_log_cost(gate_seconds, t2_seconds, buckets):
    """Independent small acceptance oracle: bucket -> (count, mean error).

    None denotes a nonempty zero-survival bucket, not exponential overflow.
    This loads no circuits, fits no models and generates no saved predictions.
    """
    if not all(math.isfinite(value) and value > 0 for value in (gate_seconds, t2_seconds)):
        raise ValueError("positive finite SI gate duration and T2 required")
    terms = [math.log(gate_seconds), gate_seconds / t2_seconds]
    zero_survival = False
    for count, error in buckets.values():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("nonnegative integer bucket count required")
        if not math.isfinite(error) or not 0 <= error <= 1:
            raise ValueError("finite error in [0, 1] required")
        if count == 0:
            continue
        if error == 1:
            zero_survival = True
        else:
            terms.append(-count * math.log1p(-error))
    if zero_survival:
        return None
    result = math.fsum(terms)
    if not math.isfinite(result):
        raise OverflowError("analytical log arithmetic is nonfinite")
    return result


def bootstrap_seed(registry, protocol_id, method_a, method_b):
    pair_id = "::".join(sorted((method_a, method_b)))
    material = f"{registry['root_seed']}|{registry['streams']['bootstrap']}|{protocol_id}|{pair_id}|0"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    root = args.repository_root.resolve()
    path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    try:
        protocol = json.loads(path.read_text())
        errors = validate_contract(protocol)
        if args.check_inputs and not errors:
            errors.extend(validate_inputs(protocol, root, include_derived=True))
        print(json.dumps({"status": "FAIL" if errors else "PASS", "scope": "design_and_pins_only",
                          "execution_started": False, "checked_input_hashes": args.check_inputs,
                          "protocol_sha256": sha256_file(path), "errors": errors}, sort_keys=True))
        return int(bool(errors))
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"status": "FAIL", "scope": "design_and_pins_only", "errors": [str(error)]}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
