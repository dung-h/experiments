"""CPU runner for the frozen archived-QPU analytical extension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import pickle
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark_v1/scripts"))
import validate_analytical_extension as contract

PROTOCOL_PATH = ROOT / contract.PROTOCOL
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
OUT = ROOT / PROTOCOL["execution"]["output_root"]
REP_DIR = ROOT / "artifacts/benchmark_v3/real_qpu/compiled_unified"
CANON_PATH = ROOT / PROTOCOL["input_pins"]["canonical"]["path"]
OUTER_PATH = ROOT / PROTOCOL["input_pins"]["outer"]["path"]
INNER_PATH = ROOT / PROTOCOL["input_pins"]["inner"]["path"]
REP_PATH = ROOT / PROTOCOL["input_pins"]["representation_rows"]["path"]
GRAPH_HASH_PATH = ROOT / PROTOCOL["input_pins"]["graph_hashes"]["path"]
SNAPSHOT_REGISTRY_PATH = ROOT / PROTOCOL["input_pins"]["nominal_snapshots"]["path"]
CALIBRATION_PATH = ROOT / PROTOCOL["preflight_artifact_pins"]["kyoto_composite_calibration"]["path"]
COMPONENTS_PATH = OUT / "components.csv"
RUN_MANIFEST_PATH = OUT / "run_manifest.json"
GRAPH_METHODS = set(PROTOCOL["method_input_groups"]["requires_graph_and_backend_duration_or_calibration"])
META_METHODS = set(PROTOCOL["method_input_groups"]["uses_saved_global_metadata_and_shots_only"])
ALL_METHODS = [method["method_id"] for method in PROTOCOL["methods"]]
RIDGE_METHODS = {method["method_id"] for method in PROTOCOL["methods"] if method["implementation"] in {"hyb", "control"} and method["transform"].endswith("ridge")}
GATE_TIME_SKIP = {"barrier", "snapshot", "delay", "reset"}
HYB_COST_SKIP = {"delay", "reset"}
QCRE_COST_SKIP = {"measure", "reset", "delay", "barrier", "snapshot"}
FLOAT_MAX_LOG = math.log(float.fromhex("0x1.fffffffffffffp+1023"))
FLOAT_MIN_SUBNORMAL_LOG = math.log(float.fromhex("0x0.0000000000001p-1022"))
COMPONENT_FIELDS = [
    "canonical_row_id", "source_id", "backend", "outer_fold", "leakage_group",
    "graph_file", "graph_sha256", "graph_digest", "graph_status", "graph_reason",
    "shots", "active_width", "structural_depth", "one_qubit_count", "two_qubit_count",
    "swap_like_count", "measurement_count", "operation_count",
    "nominal_gate_seconds", "nominal_log_cost", "nominal_gate_reason", "nominal_log_reason",
    "composite_gate_seconds", "composite_log_cost", "composite_gate_reason", "composite_log_reason",
    "nominal_component_details_json", "composite_component_details_json",
    "qcre_rank", "qcre_reason", "scholten_depth", "scholten_reason",
]
ATTEMPT_FIELDS = ["canonical_row_id", "source_id", "backend", "outer_fold", "leakage_group", "method_id",
                  "evaluation_target_clock", "method_output_clock", "target_seconds", "prediction", "status",
                  "terminal_reason", "selected_alpha", "model_feature_digest", "graph_digest"]


def sha256(path: Path) -> str:
    return contract.sha256_file(path)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def record_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_graph(row: dict[str, str], graph_hashes: dict[str, str]) -> dict | None:
    if row["graph_status"] != "available":
        return None
    name = row["graph_file"]
    expected = graph_hashes.get(name)
    path = REP_DIR / "graphs" / name
    if not expected or not path.is_file():
        raise ValueError(f"missing graph hash/file: {name}")
    data = path.read_bytes()
    if record_sha(data) != expected:
        raise ValueError(f"graph hash mismatch before unpickling: {name}")
    value = pickle.loads(data)
    if value.get("format") != "s71-operation-dag-v1":
        raise ValueError(f"unexpected graph schema: {name}")
    return value


def graph_ops(graph: dict):
    names, arity = graph["names"], graph["arity"]
    arrays = [graph[f"q{i}"] for i in range(4)]
    if len(names) != graph["operation_count"] or len(arity) != len(names):
        raise ValueError("graph operation arrays have inconsistent lengths")
    for index, name in enumerate(names):
        count = int(arity[index])
        qargs = tuple(int(arrays[j][index]) for j in range(count))
        if any(q < 0 or q >= graph["num_qubits"] for q in qargs):
            raise ValueError(f"invalid quantum wire in graph node {index}")
        yield str(name), qargs


def operation_stats(graph: dict) -> dict[str, int]:
    one = two = swap = measures = active_count = 0
    active = set()
    for name, qargs in graph_ops(graph):
        active.update(qargs)
        if name == "measure":
            measures += 1
        elif len(qargs) == 1:
            one += 1
        elif len(qargs) == 2:
            two += 1
        if "swap" in name:
            swap += 1
    return {"active_width": len(active), "one_qubit_count": one, "two_qubit_count": two,
            "swap_like_count": swap, "measurement_count": measures, "operation_count": graph["operation_count"]}


def get_duration(backend, name, qargs):
    value = float(backend.target.durations().get(name, qargs, unit="s"))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid_duration:{name}:{qargs}:{value}")
    return value


def qiskit_backend_maps():
    path = ROOT / "benchmark_v1/scripts/run_recovery_c133_c135_v3.py"
    spec = importlib.util.spec_from_file_location("analytical_extension_recovery", path)
    recovery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recovery)
    from qiskit_ibm_runtime import fake_provider

    pinned_rows = {row["backend_canonical"]: row for row in read_csv(SNAPSHOT_REGISTRY_PATH)}
    if set(pinned_rows) != set(recovery.BACKENDS):
        raise ValueError("snapshot registry backend set differs from the pinned C133 source")
    if importlib.metadata.version("qiskit-ibm-runtime") != "0.49.0":
        raise ValueError("runtime version differs from the pinned C133 snapshot environment")
    if importlib.metadata.version("qiskit") != "2.5.2":
        raise ValueError("Qiskit version differs from the pinned C133 snapshot environment")
    backends = {}
    for backend_name, class_name in recovery.BACKENDS.items():
        row = pinned_rows[backend_name]
        cls = getattr(fake_provider, class_name, None)
        if cls is None or row["fake_backend_class"] != class_name:
            raise ValueError(f"pinned FakeBackend class mismatch: {backend_name}/{class_name}")
        backend = cls()
        module_path = Path(__import__(cls.__module__, fromlist=["__name__"]).__file__ or "")
        if sha256(module_path) != row["class_module_sha256"]:
            raise ValueError(f"FakeBackend class module hash mismatch: {backend_name}")
        module_dir = module_path.parent
        conf_path = module_dir / str(getattr(cls, "conf_filename", ""))
        props_path = module_dir / str(getattr(cls, "props_filename", ""))
        if not conf_path.is_file() or sha256(conf_path) != row["configuration_sha256"]:
            raise ValueError(f"FakeBackend configuration hash mismatch: {backend_name}")
        if not props_path.is_file() or sha256(props_path) != row["properties_sha256"]:
            raise ValueError(f"FakeBackend properties hash mismatch: {backend_name}")
        backends[backend_name] = backend
    return recovery, backends


def qcre_weight_maps(backends):
    result = {}
    for backend_name, backend in backends.items():
        means = {}
        target = backend.target
        for name in target.operation_names:
            if name in QCRE_COST_SKIP or "loop" in name or name == "if_else":
                continue
            properties = target[name]
            values = [float(item.duration) for item in properties.values()
                      if item is not None and item.duration is not None and math.isfinite(float(item.duration))]
            if values:
                means[name] = statistics.fmean(values)
        scale = max((value for value in means.values() if value > 0), default=0.0)
        if scale <= 0:
            result[backend_name] = {}
        else:
            result[backend_name] = {name: value / scale for name, value in means.items()}
    return result


def property_error(properties, name, qargs, composite):
    if composite:
        if name == "measure":
            return float(composite["qubits"][str(qargs[0])]["candidate"]["readout_error"])
        gate_to_field = {"id": "id_error", "rz": "rz_error", "sx": "sx_error", "x": "x_error"}
        if len(qargs) == 1 and name in gate_to_field:
            return float(composite["qubits"][str(qargs[0])]["candidate"][gate_to_field[name]])
        if name == "ecr" and len(qargs) == 2:
            key = f"{qargs[0]}_{qargs[1]}"
            return float(composite["directed_ecr"][key]["error"])
        raise ValueError(f"composite_error_unavailable:{name}:{qargs}")
    if name == "measure":
        return float(properties.readout_error(qargs[0]))
    return float(properties.gate_error(name, qargs))


def property_t2(properties, qubit, composite):
    if composite:
        return float(composite["qubits"][str(qubit)]["candidate"]["T2_seconds"])
    return float(properties.t2(qubit))


def gate_duration(backend, name, qargs, composite):
    if composite and name == "measure":
        return float(composite["qubits"][str(qargs[0])]["candidate"]["readout_seconds"])
    if composite and name == "ecr" and len(qargs) == 2:
        key = f"{qargs[0]}_{qargs[1]}"
        return float(composite["directed_ecr"][key]["duration_seconds"])
    if composite and len(qargs) == 1 and name in {"id", "rz", "sx", "x"}:
        mapping = composite["older_package_single_qubit_gate_durations_seconds"][name]
        return float(mapping["_".join(map(str, qargs))])
    return get_duration(backend, name, qargs)


def gate_and_log_cost(graph, backend, properties, composite):
    gate_time = 0.0
    active = set()
    errors = {"one_qubit": [], "two_qubit": [], "readout": []}
    gate_reason = ""
    noise_reason = ""
    error_one = Counter()
    for name, qargs in graph_ops(graph):
        if name not in {"barrier", "snapshot"}:
            active.update(qargs)
        if name in GATE_TIME_SKIP:
            continue
        try:
            duration = gate_duration(backend, name, qargs, composite)
            gate_time += duration
        except Exception as exc:
            gate_reason = f"{type(exc).__name__}:{exc}"
            break
        if name in HYB_COST_SKIP:
            continue
        bucket = "readout" if name == "measure" else ("one_qubit" if len(qargs) == 1 else "two_qubit" if len(qargs) == 2 else "")
        if not bucket:
            noise_reason = f"unsupported_error_arity:{name}:{len(qargs)}"
            continue
        try:
            err = property_error(properties, name, qargs, composite)
            if not math.isfinite(err) or not 0 <= err <= 1:
                raise ValueError(f"invalid_error:{err}")
            errors[bucket].append(err)
            if err == 1:
                error_one[bucket] += 1
        except Exception as exc:
            noise_reason = f"{type(exc).__name__}:{exc}"
    if not gate_reason and (not math.isfinite(gate_time) or gate_time <= 0):
        gate_reason = "nonpositive_or_nonfinite_total_gate_time"
    if gate_reason:
        return None, None, gate_reason, gate_reason, {"error_one_operation_count": dict(error_one)}
    if not active:
        return gate_time, None, "", "no_active_qubits", {"error_one_operation_count": dict(error_one)}
    t2s = []
    try:
        for qubit in sorted(active):
            t2 = property_t2(properties, qubit, composite)
            if not math.isfinite(t2) or t2 <= 0:
                raise ValueError(f"invalid_T2:{qubit}:{t2}")
            t2s.append(t2)
    except Exception as exc:
        noise_reason = noise_reason or f"{type(exc).__name__}:{exc}"
    details = {"error_one_operation_count": dict(error_one),
               "bucket_operation_count": {name: len(values) for name, values in errors.items()},
               "bucket_mean_error": {name: statistics.fmean(values) for name, values in errors.items() if values},
               "active_qubit_count": len(active), "minimum_active_T2_seconds": min(t2s) if t2s else None}
    if noise_reason:
        return gate_time, None, "", noise_reason, details
    mean_buckets = {}
    for name, values in errors.items():
        if values:
            mean_buckets[name] = (len(values), statistics.fmean(values))
    try:
        log_cost = contract.reference_log_cost(gate_time, min(t2s), mean_buckets)
    except Exception as exc:
        return gate_time, None, "", f"{type(exc).__name__}:{exc}", details
    if log_cost is None:
        return gate_time, None, "", "true_zero_bucket_survival", details
    details["log_effective_cost_seconds"] = log_cost
    return gate_time, log_cost, "", "", details


def rank_and_depth(graph, weights):
    wire_rank = defaultdict(float)
    wire_depth = defaultdict(int)
    max_rank = 0.0
    for name, qargs in graph_ops(graph):
        start_rank = max((wire_rank[q] for q in qargs), default=0.0)
        start_depth = max((wire_depth[q] for q in qargs), default=0)
        if name in {"barrier", "snapshot"}:
            weight, depth_increment = 0.0, 0
        elif name in QCRE_COST_SKIP:
            weight, depth_increment = 0.0, 0
        else:
            if name not in weights:
                return None, None, f"missing_duration_weight:{name}"
            weight, depth_increment = weights[name], 1
        finish_rank = start_rank + weight
        finish_depth = start_depth + depth_increment
        for qubit in qargs:
            wire_rank[qubit] = finish_rank
            wire_depth[qubit] = finish_depth
        max_rank = max(max_rank, finish_rank)
    return max_rank, max(wire_depth.values(), default=0), ""


def validate_split(canonical, outer, inner):
    ids = {row["canonical_row_id"] for row in canonical}
    outer_by_id = {row["canonical_observation_id"]: row for row in outer}
    if len(ids) != 8767 or set(outer_by_id) != ids or len(outer) != 8767:
        raise ValueError("frozen outer split does not cover the canonical IDs exactly once")
    if len(inner) != 8767 * 5:
        raise ValueError("frozen inner split must contain 43,835 row/fold assignments")
    inner_by_outer = defaultdict(dict)
    for row in inner:
        fold = int(row["outer_fold"])
        identity = row["canonical_observation_id"]
        if identity in inner_by_outer[fold]:
            raise ValueError(f"duplicate inner split row: {fold}/{identity}")
        inner_by_outer[fold][identity] = row
    for fold in range(5):
        assignments = inner_by_outer.get(fold, {})
        if set(assignments) != ids:
            raise ValueError(f"inner split fold {fold} fails canonical coverage")
        train = {identity for identity in ids if int(outer_by_id[identity]["outer_fold"]) != fold}
        test = ids - train
        if any(assignments[identity]["inner_fold"] for identity in test):
            raise ValueError(f"outer test rows carry inner assignment in fold {fold}")
        if any(assignments[identity]["inner_fold"] not in {"0", "1", "2", "3"} for identity in train):
            raise ValueError(f"outer train rows lack inner assignment in fold {fold}")
        group_outer = defaultdict(set)
        for identity, row in outer_by_id.items():
            group_outer[row["unified_leakage_group_id"]].add(int(row["outer_fold"]))
        if any(len(value) > 1 for value in group_outer.values()):
            raise ValueError("unified leakage group crosses frozen outer folds")
        group_inner = defaultdict(set)
        for identity in train:
            row = assignments[identity]
            group_inner[row["unified_leakage_group_id"]].add(row["inner_fold"])
        if any(len(value) > 1 for value in group_inner.values()):
            raise ValueError(f"unified leakage group crosses inner folds in outer fold {fold}")
    return outer_by_id, inner_by_outer


def load_inputs():
    errors = contract.validate_contract(PROTOCOL) + contract.validate_inputs(PROTOCOL, ROOT, include_derived=True)
    if errors:
        raise ValueError("contract/input validation failed: " + "; ".join(errors))
    canonical = read_csv(CANON_PATH)
    outer = read_csv(OUTER_PATH)
    inner = read_csv(INNER_PATH)
    representation = read_csv(REP_PATH)
    graph_hashes = json.loads(GRAPH_HASH_PATH.read_text())
    outer_by_id, inner_by_outer = validate_split(canonical, outer, inner)
    rep_by_id = {row["canonical_row_id"]: row for row in representation}
    ids = {row["canonical_row_id"] for row in canonical}
    if len(rep_by_id) != 8767 or set(rep_by_id) != ids:
        raise ValueError("R2 representation rows do not match the frozen canonical identities")
    for identity, split_row in outer_by_id.items():
        rep = rep_by_id[identity]
        if int(rep["outer_fold"]) != int(split_row["outer_fold"]):
            raise ValueError(f"R2 representation outer fold mismatch: {identity}")
    return canonical, outer_by_id, inner_by_outer, rep_by_id, graph_hashes


def source_feature(row, name):
    value = float(row[name])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid nonnegative feature {name} for {row['canonical_row_id']}")
    return value


def component_row(row, split, representation, graph, graph_hash, backend, properties, composite, weight_map):
    identity = row["canonical_row_id"]
    base = {
        "canonical_row_id": identity,
        "source_id": row["source_id"],
        "backend": representation["backend_canonical"],
        "outer_fold": int(split["outer_fold"]),
        "leakage_group": split["unified_leakage_group_id"],
        "graph_file": representation["graph_file"],
        "graph_sha256": graph_hash,
        "graph_digest": representation["graph_input_digest"],
        "graph_status": representation["graph_status"],
        "graph_reason": representation["terminal_reason"],
        "shots": int(row["shots"]),
    }
    for feature in ("active_width", "structural_depth", "one_qubit_count", "two_qubit_count", "swap_like_count", "measurement_count", "operation_count"):
        base[feature] = int(source_feature(representation, feature))
    for name in ("nominal_gate_seconds", "nominal_log_cost", "composite_gate_seconds", "composite_log_cost",
                 "qcre_rank", "scholten_depth"):
        base[name] = ""
    for name in ("nominal_gate_reason", "nominal_log_reason", "composite_gate_reason", "composite_log_reason",
                 "qcre_reason", "scholten_reason"):
        base[name] = representation["terminal_reason"] if graph is None else ""
    base["nominal_component_details_json"] = ""
    base["composite_component_details_json"] = ""
    if graph is None:
        return base
    stats = operation_stats(graph)
    if graph["operation_count"] != int(representation["operation_count"]):
        raise ValueError(f"graph/feature operation count mismatch: {identity}")
    for name in ("active_width", "one_qubit_count", "two_qubit_count", "swap_like_count", "measurement_count", "operation_count"):
        if stats[name] != int(representation[name]):
            raise ValueError(f"graph/feature parity mismatch for {name}: {identity}")
    ng, nl, ngr, nlr, nominal_details = gate_and_log_cost(graph, backend, properties, None)
    # The composite-calibration panel changes Kyoto fields only. All other
    # backends intentionally reuse their frozen nominal properties.
    cg, cl, cgr, clr, composite_details = gate_and_log_cost(graph, backend, properties, composite)
    if ng is not None:
        base["nominal_gate_seconds"] = ng
        base["nominal_log_cost"] = "" if nl is None else nl
    base["nominal_gate_reason"], base["nominal_log_reason"] = ngr, nlr
    base["nominal_component_details_json"] = json.dumps(nominal_details, sort_keys=True, separators=(",", ":"))
    if cg is not None:
        base["composite_gate_seconds"] = cg
        base["composite_log_cost"] = "" if cl is None else cl
    base["composite_gate_reason"], base["composite_log_reason"] = cgr, clr
    base["composite_component_details_json"] = json.dumps(composite_details, sort_keys=True, separators=(",", ":"))
    rank, depth, reason = rank_and_depth(graph, weight_map)
    if rank is not None:
        base["qcre_rank"], base["scholten_depth"] = rank, depth
    base["qcre_reason"] = reason
    base["scholten_reason"] = reason
    return base


def load_graph_constants():
    recovery, backends = qiskit_backend_maps()
    registry = {row["backend_canonical"]: row for row in read_csv(SNAPSHOT_REGISTRY_PATH)}
    if set(backends) != set(registry):
        raise ValueError("FakeBackend set differs from the frozen snapshot registry")
    properties = {name: backend.properties() for name, backend in backends.items()}
    composite = json.loads(CALIBRATION_PATH.read_text())
    weights = qcre_weight_maps(backends)
    return recovery, backends, properties, composite, weights


def build_components(inputs):
    canonical, outer_by_id, _, rep_by_id, graph_hashes = inputs
    _, backends, props, composite, weights = load_graph_constants()
    composite_by_backend = {name: composite if name == "ibm_kyoto" else None for name in backends}
    cache = {}
    rows = []
    for index, row in enumerate(canonical, 1):
        identity = row["canonical_row_id"]
        rep = rep_by_id[identity]
        backend_name = rep["backend_canonical"]
        key = (rep["graph_input_digest"], backend_name)
        if rep["graph_status"] == "available":
            if key not in cache:
                graph = load_graph(rep, graph_hashes)
                cache[key] = component_row(row, outer_by_id[identity], rep, graph,
                                           graph_hashes[rep["graph_file"]], backends[backend_name],
                                           props[backend_name], composite_by_backend[backend_name],
                                           weights[backend_name])
            component = dict(cache[key])
            component.update({"canonical_row_id": identity, "source_id": row["source_id"],
                              "outer_fold": int(outer_by_id[identity]["outer_fold"]),
                              "leakage_group": outer_by_id[identity]["unified_leakage_group_id"],
                              "shots": int(row["shots"])})
        else:
            component = component_row(row, outer_by_id[identity], rep, None, "", None, None, None, {})
        rows.append(component)
        if index % 1000 == 0 or index == len(canonical):
            print(f"components {index}/{len(canonical)}; unique graph/backend contexts={len(cache)}", flush=True)
    atomic_csv(COMPONENTS_PATH, rows, COMPONENT_FIELDS)
    return rows


def preflight(inputs, components):
    _, outer_by_id, _, rep_by_id, _ = inputs
    if len(components) != 8767 or len({row["canonical_row_id"] for row in components}) != 8767:
        raise ValueError("preflight requires the unique, complete 8,767-row ledger")
    by_cell = defaultdict(list)
    for row in components:
        by_cell[(row["source_id"], row["backend"])].append(row)
    selected = set()
    for rows in by_cell.values():
        rows = sorted(rows, key=lambda row: hashlib.sha256(row["canonical_row_id"].encode()).hexdigest())
        selected.update(row["canonical_row_id"] for row in rows[:10])
        largest = sorted(rows, key=lambda row: (-int(row["operation_count"]), row["canonical_row_id"]))
        selected.update(row["canonical_row_id"] for row in largest[:3])
    mandatory = PROTOCOL["execution"]["mandatory_preflight_ids"]
    selected.update(mandatory)
    ids = {row["canonical_row_id"] for row in components}
    if not selected <= ids:
        raise ValueError(f"mandatory preflight row absent: {sorted(selected-ids)}")
    checks = []
    feature_coverage = Counter()
    terminal_reasons = Counter()
    for identity in sorted(selected):
        row = next(value for value in components if value["canonical_row_id"] == identity)
        if row["canonical_row_id"] not in outer_by_id or row["outer_fold"] != int(outer_by_id[identity]["outer_fold"]):
            raise ValueError(f"component ledger split assignment mismatch: {identity}")
        for name in ("shots", "active_width", "structural_depth", "one_qubit_count", "two_qubit_count",
                     "swap_like_count", "measurement_count", "operation_count"):
            if int(row[name]) < 0:
                raise ValueError(f"negative saved feature {name}: {identity}")
        for name in ("nominal_gate_seconds", "nominal_log_cost", "composite_gate_seconds", "composite_log_cost",
                     "qcre_rank", "scholten_depth"):
            value = row[name]
            if value != "" and not math.isfinite(float(value)):
                raise ValueError(f"nonfinite component {name}: {identity}")
        for method_id in RIDGE_METHODS:
            values = feature_for(method_id, row)
            if values is None:
                feature_coverage[f"{method_id}:unavailable"] += 1
            elif not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"nonfinite feature vector for {method_id}: {identity}")
            else:
                feature_coverage[f"{method_id}:available"] += 1
        for reason_field in ("nominal_gate_reason", "nominal_log_reason", "composite_gate_reason", "composite_log_reason",
                             "qcre_reason", "scholten_reason"):
            if row[reason_field]:
                terminal_reasons[f"{reason_field}:{row[reason_field]}"] += 1
        if identity.endswith("row4477"):
            if row["graph_status"] == "available":
                raise ValueError("row4477 unexpectedly has a flat R2 graph")
            if not row["structural_depth"] or not row["shots"]:
                raise ValueError("row4477 metadata-only control features are missing")
        if row["graph_status"] == "available" and (not row["graph_sha256"] or not row["graph_digest"]):
            raise ValueError(f"graph identity/hash absent for {identity}")
        checks.append({"canonical_row_id": identity, "source_id": row["source_id"], "backend": row["backend"],
                       "graph_status": row["graph_status"], "graph_sha256": row["graph_sha256"],
                       "graph_digest": row["graph_digest"]})
    audit = {"status": "PASS", "selected_rows": len(checks), "selection_ids_sha256":
             hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest(), "checks": checks,
             "feature_coverage": dict(feature_coverage), "selected_terminal_reasons": dict(terminal_reasons),
             "components_sha256": sha256(COMPONENTS_PATH),
             "row4477_policy": "graph-dependent methods unavailable; metadata-only Ridge controls eligible"}
    atomic_json(OUT / "preflight.json", audit)
    return audit


def feature_for(method_id, component):
    if method_id in {"hyb_nominal_r2_log_cost_ridge_v1"}:
        raw = component["nominal_log_cost"]
        return None if raw == "" else (math.asinh(float(raw)), math.log1p(component["shots"]))
    if method_id == "hyb_kyoto_composite_r2_log_cost_ridge_v1":
        raw = component["composite_log_cost"]
        return None if raw == "" else (math.asinh(float(raw)), math.log1p(component["shots"]))
    if method_id == "hyb_kyoto_composite_r2_gate_time_ridge_v1":
        raw = component["composite_gate_seconds"]
        if raw == "":
            return None
        return (math.asinh(math.log(float(raw))), math.log1p(component["shots"]))
    if method_id == "qpu_depth_shots_ridge_v1":
        return (math.log1p(component["structural_depth"]), math.log1p(component["shots"]))
    if method_id == "qpu_seven_global_ridge_v1":
        names = ("active_width", "structural_depth", "one_qubit_count", "two_qubit_count",
                 "swap_like_count", "measurement_count", "shots")
        return tuple(math.log1p(component[name]) for name in names)
    raise KeyError(method_id)


def score_source_balanced(predictions, actuals, sources):
    errors = defaultdict(list)
    for identity, pred in predictions.items():
        errors[sources[identity]].append(abs(actuals[identity] - pred))
    if not errors:
        return None
    return statistics.fmean(statistics.fmean(values) for values in errors.values())


def fit_predict_fold(fold, inputs, components):
    canonical, outer_by_id, inner_by_outer, _, _ = inputs
    comp = {row["canonical_row_id"]: row for row in components}
    data = {row["canonical_row_id"]: row for row in canonical}
    train_ids = sorted(identity for identity in data if int(outer_by_id[identity]["outer_fold"]) != fold)
    test_ids = sorted(identity for identity in data if int(outer_by_id[identity]["outer_fold"]) == fold)
    assignments = inner_by_outer[fold]
    y = {identity: float(data[identity]["target_seconds"]) for identity in train_ids + test_ids}
    if any(not math.isfinite(value) or value < 0 for value in y.values()):
        raise ValueError("target labels must be finite nonnegative seconds")
    source = {identity: data[identity]["source_id"] for identity in train_ids + test_ids}
    attempts = []
    fit_receipts = []

    def emit(identity, method_id, clock, pred, status, reason="", alpha="", feat=None):
        c = comp[identity]
        attempts.append({"canonical_row_id": identity, "source_id": data[identity]["source_id"],
                         "backend": data[identity]["backend"], "outer_fold": fold,
                         "leakage_group": outer_by_id[identity]["unified_leakage_group_id"],
                         "method_id": method_id, "evaluation_target_clock": PROTOCOL["evaluation_target_clock"],
                         "method_output_clock": clock, "target_seconds": y[identity],
                         "prediction": "" if pred is None else pred, "status": status,
                         "terminal_reason": reason, "selected_alpha": alpha,
                         "model_feature_digest": "" if feat is None else hashlib.sha256(repr(feat).encode()).hexdigest(),
                         "graph_digest": c["graph_digest"]})

    # Direct outputs retain their distinct clocks and never inspect held-out labels.
    for identity in test_ids:
        c = comp[identity]
        for method in PROTOCOL["methods"]:
            method_id, impl, transform = method["method_id"], method["implementation"], method["transform"]
            if impl == "hyb":
                kind = "nominal" if method["calibration"] == "nominal" else "composite"
                value_key, gate_key, reason_key, log_reason_key = f"{kind}_log_cost", f"{kind}_gate_seconds", f"{kind}_gate_reason", f"{kind}_log_reason"
                if transform == "raw_exp":
                    logvalue = c[value_key]
                elif transform == "shot_exp":
                    logvalue = "" if c[value_key] == "" else float(c[value_key]) + math.log(c["shots"])
                else:
                    continue
                if c[value_key] == "":
                    emit(identity, method_id, method["method_output_clock"], None, "unavailable", c[log_reason_key] or c[reason_key])
                elif logvalue > FLOAT_MAX_LOG:
                    emit(identity, method_id, method["method_output_clock"], None, "overflow", "float64_exponential_overflow")
                elif logvalue < FLOAT_MIN_SUBNORMAL_LOG:
                    emit(identity, method_id, method["method_output_clock"], None, "underflow", "float64_exponential_underflow")
                else:
                    emit(identity, method_id, method["method_output_clock"], math.exp(logvalue), "predicted")
            elif impl == "qcre_rank":
                if c["qcre_rank"] == "":
                    emit(identity, method_id, method["method_output_clock"], None, "unavailable", c["qcre_reason"])
                else:
                    emit(identity, method_id, method["method_output_clock"], float(c["qcre_rank"]), "predicted")
            elif impl == "scholten":
                throughput = PROTOCOL["scholten"]["active_entries"].get(c["backend"])
                if throughput is None or c["scholten_depth"] == "":
                    emit(identity, method_id, method["method_output_clock"], None, "unavailable",
                         "no_verified_nominal_throughput" if throughput is None else c["scholten_reason"])
                else:
                    pred = c["shots"] * int(c["scholten_depth"]) / throughput["CLOPS"]
                    emit(identity, method_id, method["method_output_clock"], pred, "predicted")

    for method_id in sorted(RIDGE_METHODS):
        candidates = []
        inner_scores = {}
        valid_inner = True
        for inner_fold in range(4):
            fit_ids = [identity for identity in train_ids if assignments[identity]["inner_fold"] != str(inner_fold)]
            val_ids = [identity for identity in train_ids if assignments[identity]["inner_fold"] == str(inner_fold)]
            fit_ids = [identity for identity in fit_ids if feature_for(method_id, comp[identity]) is not None]
            val_ids = [identity for identity in val_ids if feature_for(method_id, comp[identity]) is not None]
            fit_groups = {assignments[identity]["unified_leakage_group_id"] for identity in fit_ids}
            val_groups = {assignments[identity]["unified_leakage_group_id"] for identity in val_ids}
            if not fit_ids or not val_ids or fit_groups & val_groups:
                valid_inner = False
                break
            x_fit = np.asarray([feature_for(method_id, comp[identity]) for identity in fit_ids], dtype=float)
            y_fit = np.log1p([y[identity] for identity in fit_ids])
            x_val = np.asarray([feature_for(method_id, comp[identity]) for identity in val_ids], dtype=float)
            y_val = [y[identity] for identity in val_ids]
            source_val = [source[identity] for identity in val_ids]
            for alpha in PROTOCOL["ridge"]["alphas"]:
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(x_fit, y_fit)
                predicted = np.expm1(model.predict(x_val))
                predicted = np.maximum(0.0, predicted)
                by_source = defaultdict(list)
                for actual, estimate, src in zip(y_val, predicted, source_val):
                    if not math.isfinite(float(estimate)):
                        raise ValueError("Ridge inner prediction is nonfinite")
                    by_source[src].append(abs(actual - float(estimate)))
                score = statistics.fmean(statistics.fmean(values) for values in by_source.values())
                inner_scores.setdefault(alpha, []).append(score)
        if not valid_inner or any(len(values) != 4 for values in inner_scores.values()):
            for identity in test_ids:
                emit(identity, method_id, "calibrated_observed_service_seconds", None, "unavailable", "inner_fold_partition_unusable")
            fit_receipts.append({"outer_fold": fold, "method_id": method_id, "status": "unavailable", "selected_alpha": None})
            continue
        alpha = min(PROTOCOL["ridge"]["alphas"], key=lambda value: (statistics.fmean(inner_scores[value]), value))
        eligible_fit = [identity for identity in train_ids if feature_for(method_id, comp[identity]) is not None]
        x_fit = np.asarray([feature_for(method_id, comp[identity]) for identity in eligible_fit], dtype=float)
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x_fit, np.log1p([y[identity] for identity in eligible_fit]))
        feature_fits = []
        for identity in test_ids:
            feat = feature_for(method_id, comp[identity])
            if feat is None:
                c = comp[identity]
                reason = c.get("nominal_log_reason") or c.get("composite_log_reason") or c.get("composite_gate_reason") or "feature_unavailable"
                emit(identity, method_id, "calibrated_observed_service_seconds", None, "unavailable", reason, alpha, None)
                continue
            logpred = float(model.predict(np.asarray(feat, dtype=float).reshape(1, -1))[0])
            if logpred > FLOAT_MAX_LOG:
                emit(identity, method_id, "calibrated_observed_service_seconds", None, "overflow", "inverse_log1p_overflow", alpha, feat)
            else:
                pred = max(0.0, math.expm1(logpred))
                emit(identity, method_id, "calibrated_observed_service_seconds", pred, "predicted", "", alpha, feat)
            feature_fits.append(identity)
        fit_receipts.append({"outer_fold": fold, "method_id": method_id, "status": "fit",
                             "selected_alpha": alpha, "eligible_fit_rows": len(eligible_fit),
                             "fit_ids_sha256": hashlib.sha256("\n".join(eligible_fit).encode()).hexdigest(),
                             "test_prediction_rows": len(feature_fits),
                             "inner_source_balanced_mae_seconds": {str(a): statistics.fmean(inner_scores[a]) for a in sorted(inner_scores)}})
    expected = len(test_ids) * len(ALL_METHODS)
    if len(attempts) != expected:
        counts = Counter(row["method_id"] for row in attempts)
        raise ValueError(f"fold {fold} attempt coverage {len(attempts)} != {expected}: {counts}")
    return attempts, fit_receipts


def read_components():
    rows = read_csv(COMPONENTS_PATH)
    int_fields = {"outer_fold", "shots", "active_width", "structural_depth", "one_qubit_count",
                  "two_qubit_count", "swap_like_count", "measurement_count", "operation_count"}
    float_fields = {"nominal_gate_seconds", "nominal_log_cost", "composite_gate_seconds", "composite_log_cost",
                    "qcre_rank", "scholten_depth"}
    for row in rows:
        for name in int_fields:
            row[name] = int(row[name])
        for name in float_fields:
            row[name] = "" if row[name] == "" else float(row[name])
    return rows


def runtime_environment():
    packages = {}
    for name in ("python", "numpy", "pandas", "scikit-learn", "scipy", "qiskit", "qiskit-ibm-runtime"):
        if name == "python":
            packages[name] = sys.version.split()[0]
        else:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "not_installed"
    return packages


def run_identity():
    return {"protocol_sha256": sha256(PROTOCOL_PATH),
            "runner_sha256": sha256(Path(__file__)),
            "validator_sha256": sha256(ROOT / "benchmark_v1/scripts/validate_analytical_extension.py"),
            "component_builder_sha256": sha256(ROOT / "benchmark_v1/scripts/build_analytical_extension_inputs.py"),
            "dependency_builder_sha256": sha256(ROOT / "benchmark_v1/scripts/build_preprocessing_dependency_inventory.py"),
            "input_pins_sha256": {name: sha256(ROOT / item["path"])
                                  for name, item in {**PROTOCOL["input_pins"], **PROTOCOL["reference_code_pins"],
                                                     **PROTOCOL["preflight_artifact_pins"]}.items()},
            "environment": runtime_environment(), "worker_count": 1, "native_threads_per_worker": 1}


def ensure_run_manifest(output):
    path = output / "run_manifest.json"
    identity = run_identity()
    if path.exists():
        existing = json.loads(path.read_text())
        for key, value in identity.items():
            if key == "runner_sha256":
                continue
            if existing.get(key) != value:
                raise ValueError(f"resume identity mismatch: {key}")
        if existing.get("runner_sha256") != identity["runner_sha256"]:
            allowed_stages = {"components", "preflight"}
            if existing.get("completed_folds") or not set(existing.get("completed_stages", [])) <= allowed_stages:
                raise ValueError("runner changed after preflight/fold work; refusing resume")
            old_hash = existing["runner_sha256"]
            old_preflight = output / "preflight.json"
            if old_preflight.is_file():
                existing["superseded_preflight_sha256"] = sha256(old_preflight)
            existing["preflight_sha256"] = None
            existing["preflight_runner_sha256"] = None
            existing["completed_stages"] = [stage for stage in existing.get("completed_stages", []) if stage != "preflight"]
            existing["component_materializer_runner_sha256"] = old_hash
            existing["runner_sha256"] = identity["runner_sha256"]
            existing["runner_upgrade_before_preflight"] = True
            existing["components_sha256"] = sha256(COMPONENTS_PATH) if COMPONENTS_PATH.is_file() else None
            atomic_json(path, existing)
        return existing
    manifest = {"artifact_id": "archived-qpu-analytical-extension-v1", "status": "running",
                "execution_authorized_by": "user_request_2026-10-06", "execution_started": True,
                "assigned_observations_per_method": 8767, "methods": ALL_METHODS,
                "completed_stages": [], "completed_folds": [], **identity}
    atomic_json(path, manifest)
    return manifest


def update_run_manifest(manifest, output, *, stage=None, fold=None, status=None):
    if stage and stage not in manifest["completed_stages"]:
        manifest["completed_stages"].append(stage)
        manifest["completed_stages"].sort()
    if fold is not None and fold not in manifest["completed_folds"]:
        manifest["completed_folds"].append(fold)
        manifest["completed_folds"].sort()
    if status:
        manifest["status"] = status
    atomic_json(output / "run_manifest.json", manifest)


def write_components_if_needed(inputs, output, manifest):
    if COMPONENTS_PATH.exists():
        rows = read_components()
        if len(rows) != 8767:
            raise ValueError("existing components ledger is incomplete")
        digest = sha256(COMPONENTS_PATH)
        if manifest.get("components_sha256") not in (None, digest):
            raise ValueError("existing components ledger hash differs from the run manifest")
        manifest["components_sha256"] = digest
        atomic_json(output / "run_manifest.json", manifest)
        return rows
    rows = build_components(inputs)
    if len(rows) != 8767:
        raise ValueError("component builder failed the 8,767-row denominator")
    manifest["components_sha256"] = sha256(COMPONENTS_PATH)
    atomic_json(output / "run_manifest.json", manifest)
    return rows


def write_preflight_if_needed(inputs, components, output, manifest):
    path = output / "preflight.json"
    if path.exists() and manifest.get("preflight_sha256") == sha256(path):
        report = json.loads(path.read_text())
        if report.get("status") != "PASS":
            raise ValueError("saved preflight did not pass")
        return report
    report = preflight(inputs, components)
    manifest["preflight_sha256"] = sha256(path)
    manifest["preflight_runner_sha256"] = manifest["runner_sha256"]
    atomic_json(output / "run_manifest.json", manifest)
    return report


def qa_fold(fold, attempts, fit_receipts, inputs):
    canonical, outer_by_id, _, _, _ = inputs
    assigned = {identity for identity in outer_by_id if int(outer_by_id[identity]["outer_fold"]) == fold}
    if len(attempts) != len(assigned) * len(ALL_METHODS):
        raise ValueError(f"fold {fold} attempt count mismatch")
    expected = {(identity, method) for identity in assigned for method in ALL_METHODS}
    observed = {(row["canonical_row_id"], row["method_id"]) for row in attempts}
    if observed != expected:
        raise ValueError(f"fold {fold} attempt identity/method mismatch")
    allowed = set(PROTOCOL["execution"]["terminal_states"])
    if any(row["status"] not in allowed or not row["terminal_reason"] and row["status"] != "predicted" for row in attempts):
        raise ValueError(f"fold {fold} invalid terminal state or missing failure reason")
    if any(row["status"] == "predicted" and not math.isfinite(float(row["prediction"])) for row in attempts):
        raise ValueError(f"fold {fold} nonfinite value marked as prediction")
    expected_fit = RIDGE_METHODS
    if {row["method_id"] for row in fit_receipts} != expected_fit:
        raise ValueError(f"fold {fold} model fit receipt missing")
    if fold == 0:
        meta_rows = [row for row in attempts if row["canonical_row_id"] == "qonductor_single_circuit_ibm|row4477"]
        state = {row["method_id"]: row["status"] for row in meta_rows}
        if any(state.get(method) != "predicted" for method in META_METHODS):
            raise ValueError("row4477 metadata controls did not emit valid OOF predictions")
        if any(state.get(method) != "unavailable" for method in GRAPH_METHODS):
            raise ValueError("row4477 graph-dependent methods were not marked unavailable")
    return {"status": "PASS", "outer_fold": fold, "assigned_rows": len(assigned),
            "method_attempts": len(attempts), "methods": len(ALL_METHODS),
            "terminal_counts": dict(Counter(row["status"] for row in attempts)),
            "method_terminal_counts": {method: dict(Counter(row["status"] for row in attempts if row["method_id"] == method))
                                       for method in ALL_METHODS},
            "fit_receipts": fit_receipts}


def load_or_run_fold(fold, inputs, components, output, manifest):
    attempt_path = output / f"fold_{fold}_attempts.csv"
    receipt_path = output / f"fold_{fold}_receipt.json"
    if attempt_path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("attempts_sha256") != sha256(attempt_path)
                or receipt.get("runner_sha256") != manifest["runner_sha256"]
                or receipt.get("components_sha256") != manifest.get("components_sha256")):
            raise ValueError(f"fold {fold} checkpoint identity/hash mismatch")
        if receipt.get("qa", {}).get("status") != "PASS":
            raise ValueError(f"fold {fold} checkpoint QA is not PASS")
        return receipt
    attempts, fits = fit_predict_fold(fold, inputs, components)
    qa = qa_fold(fold, attempts, fits, inputs)
    atomic_csv(attempt_path, attempts, ATTEMPT_FIELDS)
    receipt = {"outer_fold": fold, "qa": qa, "attempts_sha256": sha256(attempt_path),
               "runner_sha256": manifest["runner_sha256"], "protocol_sha256": manifest["protocol_sha256"],
               "components_sha256": manifest.get("components_sha256")}
    atomic_json(receipt_path, receipt)
    return receipt


def grouped_bootstrap(method_a, method_b, predictions_a, predictions_b, protocol_id, registry, source_by_id, group_by_id):
    common = sorted(set(predictions_a) & set(predictions_b))
    if not common:
        return {"status": "unavailable", "reason": "empty_common_successful_intersection"}
    group_source = {}
    for identity in common:
        group, source = group_by_id[identity], source_by_id[identity]
        if group in group_source and group_source[group] != source:
            raise ValueError(f"bootstrap group crosses sources: {group}")
        group_source[group] = source
    sources = sorted({source_by_id[identity] for identity in common})
    observed_deltas = {identity: abs(float(predictions_a[identity]["target_seconds"]) - float(predictions_a[identity]["prediction"]))
                       - abs(float(predictions_b[identity]["target_seconds"]) - float(predictions_b[identity]["prediction"]))
                       for identity in common}
    source_delta = {source: statistics.fmean(observed_deltas[i] for i in common if source_by_id[i] == source) for source in sources}
    direct_pooled = statistics.fmean(observed_deltas.values())
    direct_macro = statistics.fmean(source_delta.values())
    samples = PROTOCOL["metrics"]["bootstrap"]["replicates"]
    seed = contract.bootstrap_seed(registry, PROTOCOL["protocol_id"], method_a, method_b)
    rng = np.random.default_rng(seed)
    per_source_replicates = {}
    pooled_sum = np.zeros(samples, dtype=np.float64)
    pooled_count = np.zeros(samples, dtype=np.float64)
    for source in sources:
        group_values = defaultdict(list)
        for identity in common:
            if source_by_id[identity] == source:
                group_values[group_by_id[identity]].append(observed_deltas[identity])
        groups = sorted(group_values)
        values = [np.asarray(group_values[group], dtype=np.float64) for group in groups]
        indices = rng.integers(0, len(groups), size=(samples, len(groups)))
        sums = np.asarray([float(np.sum(value)) for value in values])
        counts = np.asarray([len(value) for value in values], dtype=np.float64)
        sampled_sum = sums[indices].sum(axis=1)
        sampled_count = counts[indices].sum(axis=1)
        per_source_replicates[source] = sampled_sum / sampled_count
        pooled_sum += sampled_sum
        pooled_count += sampled_count
    pooled = pooled_sum / pooled_count
    macro = np.mean(np.column_stack([per_source_replicates[source] for source in sources]), axis=1)
    quantile = lambda values, q: float(np.quantile(values, q, method="linear"))
    return {"status": "PASS", "common_rows": len(common),
            "common_ids_sha256": hashlib.sha256("\n".join(common).encode()).hexdigest(),
            "method_A_minus_method_B": {"mae_seconds": {
                "direct_observed_delta": direct_pooled, "bootstrap_mean_delta": float(np.mean(pooled)),
                "ci_95_low": quantile(pooled, .025), "ci_95_high": quantile(pooled, .975)},
                "source_balanced_macro_mae_seconds": {
                "direct_observed_delta": direct_macro, "bootstrap_mean_delta": float(np.mean(macro)),
                "ci_95_low": quantile(macro, .025), "ci_95_high": quantile(macro, .975)}},
            "bootstrap_replicates": samples, "seed_uint32": seed,
            "groups_by_source": {source: len({group_by_id[i] for i in common if source_by_id[i] == source}) for source in sources}}


def aggregate(output, inputs):
    all_attempts = []
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    components_digest = sha256(COMPONENTS_PATH)
    if run_manifest.get("components_sha256") != components_digest:
        raise ValueError("cannot aggregate; components ledger hash differs from run manifest")
    for fold in range(5):
        attempt_path = output / f"fold_{fold}_attempts.csv"
        receipt_path = output / f"fold_{fold}_receipt.json"
        if not attempt_path.is_file() or not receipt_path.is_file():
            raise ValueError(f"cannot aggregate; fold {fold} is incomplete")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("attempts_sha256") != sha256(attempt_path)
                or receipt.get("qa", {}).get("status") != "PASS"
                or receipt.get("components_sha256") != components_digest):
            raise ValueError(f"cannot aggregate; fold {fold} receipt/hash failed")
        all_attempts.extend(read_csv(attempt_path))
    expected = 8767 * len(ALL_METHODS)
    if len(all_attempts) != expected:
        raise ValueError(f"aggregate envelope {len(all_attempts)} != {expected}")
    method_rows = defaultdict(list)
    for row in all_attempts:
        method_rows[row["method_id"]].append(row)
    method_metrics, source_metrics = [], []
    for card in PROTOCOL["methods"]:
        rows = method_rows[card["method_id"]]
        result = {"method_id": card["method_id"], "reader_label": card["reader_label"],
                  "fidelity_class": card["fidelity_class"], "evaluation_target_clock": PROTOCOL["evaluation_target_clock"],
                  "method_output_clock": card["method_output_clock"],
                  "metric_type": "rank" if card["implementation"] == "qcre_rank" else "seconds",
                  "terminal_counts": json.dumps(dict(Counter(row["status"] for row in rows)), sort_keys=True)}
        result.update(metrics(rows, rank=card["implementation"] == "qcre_rank"))
        if card["implementation"] != "qcre_rank":
            result["source_balanced_macro_mae_seconds"] = score_source_balanced(
                {row["canonical_row_id"]: float(row["prediction"]) for row in rows if row["status"] == "predicted"},
                {row["canonical_row_id"]: float(row["target_seconds"]) for row in rows if row["status"] == "predicted"},
                {row["canonical_row_id"]: row["source_id"] for row in rows if row["status"] == "predicted"})
        else:
            result["source_balanced_macro_mae_seconds"] = None
        method_metrics.append(result)
        for source in sorted({row["source_id"] for row in rows}):
            subset = [row for row in rows if row["source_id"] == source]
            submetric = metrics(subset, rank=card["implementation"] == "qcre_rank")
            source_metrics.append({"method_id": card["method_id"], "source_id": source,
                                   "evaluation_target_clock": PROTOCOL["evaluation_target_clock"],
                                   "method_output_clock": card["method_output_clock"], **submetric})
    clock_methods = [card["method_id"] for card in PROTOCOL["methods"] if card["implementation"] != "qcre_rank"]
    by_method_id = {card["method_id"]: card for card in PROTOCOL["methods"]}
    common_pairs = []
    prediction_by_method = {method: {row["canonical_row_id"]: row for row in method_rows[method]} for method in ALL_METHODS}
    for index, method_a in enumerate(clock_methods):
        for method_b in clock_methods[index + 1:]:
            a, b = prediction_by_method[method_a], prediction_by_method[method_b]
            common = sorted(identity for identity in a.keys() & b.keys() if a[identity]["status"] == b[identity]["status"] == "predicted")
            rows_a = [a[identity] for identity in common]
            rows_b = [b[identity] for identity in common]
            ma, mb = metrics(rows_a), metrics(rows_b)
            common_pairs.append({"method_A": method_a, "method_B": method_b,
                                 "method_A_output_clock": by_method_id[method_a]["method_output_clock"],
                                 "method_B_output_clock": by_method_id[method_b]["method_output_clock"],
                                 "common_rows": len(common), "common_ids_sha256": hashlib.sha256("\n".join(common).encode()).hexdigest(),
                                 "method_A_mae_seconds": ma.get("mae_seconds"), "method_B_mae_seconds": mb.get("mae_seconds"),
                                 "observed_mae_delta_A_minus_B_seconds": None if not common else ma["mae_seconds"]-mb["mae_seconds"]})
    canonical, outer_by_id, _, _, _ = inputs
    source_by_id = {row["canonical_row_id"]: row["source_id"] for row in canonical}
    group_by_id = {identity: row["unified_leakage_group_id"] for identity, row in outer_by_id.items()}
    seed_registry = json.loads((ROOT / PROTOCOL["input_pins"]["seeds"]["path"]).read_text())
    bootstrap = []
    for method_a, method_b in PROTOCOL["metrics"]["bootstrap"]["primary_pairs"]:
        bootstrap.append({"method_A": method_a, "method_B": method_b,
                          **grouped_bootstrap(method_a, method_b, prediction_by_method[method_a], prediction_by_method[method_b],
                                              PROTOCOL["protocol_id"], seed_registry, source_by_id, group_by_id)})
    atomic_csv(output / "unified_attempt_envelope.csv", all_attempts, ATTEMPT_FIELDS)
    atomic_csv(output / "method_metrics.csv", method_metrics, list(method_metrics[0]))
    atomic_csv(output / "source_stratified_metrics.csv", source_metrics, list(source_metrics[0]))
    atomic_csv(output / "common_successful_pairs.csv", common_pairs, list(common_pairs[0]))
    atomic_csv(output / "paired_bootstrap.csv", bootstrap, list(bootstrap[0]))
    summary = {"artifact_id": "archived-qpu-analytical-extension-results-v1", "status": "PASS",
               "assigned_observations_per_method": 8767, "method_count": len(ALL_METHODS),
               "attempt_count": expected, "metrics": method_metrics, "paired_bootstrap": bootstrap,
               "evaluation_target_clock": PROTOCOL["evaluation_target_clock"],
               "components_sha256": components_digest,
               "selected_split_hashes": {name: PROTOCOL["input_pins"][name]["sha256"] for name in ("canonical", "outer", "inner")},
               "raw_to_compiled_rebuild": "not required for this score replay; see pinned preprocessing_dependency_inventory.json for raw reconstruction requirements"}
    atomic_json(output / "summary.json", summary)
    return summary


def metrics(rows, rank=False):
    valid = [row for row in rows if row["status"] == "predicted"]
    if not valid:
        return {"n": 0, "coverage": 0.0, "reason": "no_successful_predictions"}
    actual = np.asarray([float(row["target_seconds"]) for row in valid])
    pred = np.asarray([float(row["prediction"]) for row in valid])
    result = {"n": len(valid), "coverage": len(valid) / len(rows), "assigned": len(rows)}
    if rank:
        from scipy.stats import spearmanr
        value = spearmanr(actual, pred).statistic if len(valid) > 1 else float("nan")
        result["spearman"] = None if not math.isfinite(float(value)) else float(value)
        return result
    error = np.abs(actual - pred)
    sst = float(np.sum((actual - np.mean(actual)) ** 2))
    result.update({"mae_seconds": float(np.mean(error)), "medae_seconds": float(np.median(error)),
                   "rmse_seconds": float(np.sqrt(np.mean(error ** 2))),
                   "r2": None if len(valid) < 2 or sst == 0 else float(1 - np.sum((actual-pred)**2)/sst),
                   "r2_null_reason": "n_lt_2_or_zero_target_variance" if len(valid) < 2 or sst == 0 else "",
                   "log1p_mae": float(np.mean(np.abs(np.log1p(actual)-np.log1p(pred)))),
                   "p90_absolute_error_seconds": float(np.quantile(error, .90)),
                   "p99_absolute_error_seconds": float(np.quantile(error, .99)),
                   "max_absolute_error_seconds": float(np.max(error))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["components", "preflight", "fold0", "fold", "all", "aggregate"], required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--workers", type=int, default=PROTOCOL["execution"]["workers_default"])
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if not 1 <= args.workers <= PROTOCOL["execution"]["workers_max"]:
        raise SystemExit("worker count outside the protocol limit")
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output.resolve() != OUT.resolve():
        raise SystemExit("output-dir must be the frozen semantic output root")
    inputs = load_inputs()
    output.mkdir(parents=True, exist_ok=True)
    manifest = ensure_run_manifest(output)
    needs_components = args.stage in {"components", "preflight", "fold0", "fold", "all"}
    components = None
    if needs_components:
        components = write_components_if_needed(inputs, output, manifest)
    if args.stage in {"components", "all"}:
        update_run_manifest(manifest, output, stage="components")
        if args.stage == "components":
            print(json.dumps({"status": "PASS", "stage": "components", "rows": len(components), "sha256": sha256(COMPONENTS_PATH)}))
            return 0
    if args.stage in {"preflight", "all"}:
        report = write_preflight_if_needed(inputs, components, output, manifest)
        update_run_manifest(manifest, output, stage="preflight")
        if args.stage == "preflight":
            print(json.dumps({"status": report["status"], "stage": "preflight", "rows": report["selected_rows"], "sha256": sha256(output/"preflight.json")}))
            return 0
    if args.stage == "fold0":
        if not (output / "preflight.json").is_file():
            raise SystemExit("fold0 requires a passing preflight")
        if manifest.get("preflight_sha256") != sha256(output / "preflight.json"):
            raise SystemExit("fold0 requires a preflight pinned to this runner and component ledger")
        receipt = load_or_run_fold(0, inputs, components, output, manifest)
        update_run_manifest(manifest, output, fold=0)
        qa = receipt["qa"]
        atomic_json(output / "fold0_qa.json", qa)
        print(json.dumps({"status": qa["status"], "stage": "fold0", "method_attempts": qa["method_attempts"],
                          "terminal_counts": qa["terminal_counts"], "receipt_sha256": sha256(output/"fold_0_receipt.json")}))
        return 0
    if args.stage == "fold":
        if args.fold is None:
            raise SystemExit("--stage fold requires --fold N")
        if args.fold == 0:
            raise SystemExit("use --stage fold0 for the fold-0 checkpoint")
        if args.fold and not (output / "fold0_qa.json").is_file():
            raise SystemExit("folds 1-4 require passing fold0 QA")
        receipt = load_or_run_fold(args.fold, inputs, components, output, manifest)
        update_run_manifest(manifest, output, fold=args.fold)
        print(json.dumps({"status": receipt["qa"]["status"], "stage": "fold", "fold": args.fold,
                          "method_attempts": receipt["qa"]["method_attempts"],
                          "receipt_sha256": sha256(output / f"fold_{args.fold}_receipt.json")}))
        return 0
    if args.stage == "all":
        report = write_preflight_if_needed(inputs, components, output, manifest)
        if report["status"] != "PASS":
            raise SystemExit("analytical extension preflight did not pass")
        update_run_manifest(manifest, output, stage="preflight")
        receipt = load_or_run_fold(0, inputs, components, output, manifest)
        update_run_manifest(manifest, output, fold=0)
        atomic_json(output / "fold0_qa.json", receipt["qa"])
        for fold in range(1, 5):
            load_or_run_fold(fold, inputs, components, output, manifest)
            update_run_manifest(manifest, output, fold=fold)
        summary = aggregate(output, inputs)
        update_run_manifest(manifest, output, stage="aggregate", status="complete")
        print(json.dumps({"status": "PASS", "stage": "all", "attempt_count": summary["attempt_count"],
                          "method_count": summary["method_count"], "summary_sha256": sha256(output / "summary.json")}))
        return 0
    if args.stage == "aggregate":
        summary = aggregate(output, inputs)
        update_run_manifest(manifest, output, stage="aggregate", status="complete")
        print(json.dumps({"status": "PASS", "stage": "aggregate", "attempt_count": summary["attempt_count"],
                          "method_count": summary["method_count"], "summary_sha256": sha256(output / "summary.json")}))
        return 0
    raise SystemExit("unhandled stage")


if __name__ == "__main__":
    main()
