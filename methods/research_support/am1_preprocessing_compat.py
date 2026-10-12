"""Raw-input preprocessing in the preserved historical Qiskit environment.

No model, checkpoint, expected graph, observation ID or runtime label is read.
The serving interface transports typed arrays to the separately pinned CUDA
interpreter. It does not infer, time, cache circuit features or install packages.
Historical producers are deliberately not edited.
"""
from __future__ import annotations

import base64
import importlib.metadata
import json
import sys
from functools import lru_cache

import numpy as np


REQUIRED_PACKAGES = {"qiskit": "2.5.2", "numpy": "2.2.6"}
GLOBAL_NAMES = ("active_width", "structural_depth", "one_qubit_count",
                "two_qubit_count", "swap_like_count", "measurement_count", "shots")


def require_environment():
    versions = {k: importlib.metadata.version(k) for k in REQUIRED_PACKAGES}
    if versions != REQUIRED_PACKAGES:
        raise RuntimeError(f"historical_preprocessing_environment_required:{versions}")
    return versions


def parse_qpu(raw: bytes, source: str):
    require_environment()
    import mali_full_features as mf
    from qiskit import qasm2
    if source == "mali_real_qpu":
        return mf.parse_qasm(raw)[0]
    if source == "qonductor_single_circuit_ibm":
        return qasm2.loads(raw.decode("utf-8"), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    raise ValueError("no_serialized_raw_QASM_contract_for_source:" + source)


def calibration(properties: bytes):
    props = json.loads(properties)
    scales = {"s": 1e6, "ms": 1e3, "us": 1.0, "ns": 1e-3}
    result = {}
    for index, entries in enumerate(props["qubits"]):
        values = {p["name"]: p for p in entries if p["name"] in {"T1", "T2"}}
        if set(values) != {"T1", "T2"}:
            continue
        pair = tuple(float(values[k]["value"]) * scales[
            values[k]["unit"].strip().lower().replace("μ", "u").replace("µ", "u")]
                     for k in ("T1", "T2"))
        if not all(np.isfinite(x) and x > 0 for x in pair):
            raise ValueError("invalid_nominal_T1_T2")
        result[index] = pair
    return result


def encode_qpu(globals_raw, record, transform):
    import mali_full_features as mf
    from run_mali_full_features import transform_global
    arrays = {"global_features": transform_global(globals_raw, transform["global"]).reshape(1, -1)}
    if record is not None:
        x = mf.expand_dense_x(record)
        mean, std = (np.asarray(transform["node"][k], dtype=np.float32) for k in ("mean", "std"))
        variable = np.asarray(transform["node"]["variable_mask"], dtype=bool)
        if x.shape[1] != 178 or mean.shape != (178,) or std.shape != (178,):
            raise ValueError("frozen_node_schema_mismatch")
        x[:, variable] = (x[:, variable] - mean[variable]) / std[variable]
        x[:, ~variable] = 0
        arrays.update(x=x, edge_index=record["edge_index"].astype(np.int64))
    return arrays


@lru_cache(maxsize=1)
def pinned_extractor():
    """Resident source functions only, never circuit features or graph arrays."""
    require_environment()
    import mali_full_features as mf
    return mf.load_pinned_global_extractor()[0]


def preprocess_qpu(raw, source, properties, transform, *, graph=True):
    import mali_full_features as mf
    circuit = parse_qpu(raw, source)
    globals_raw = mf.extract_global_features(circuit, pinned_extractor())
    record = mf.compact_graph_record(circuit, calibration(properties))[0] if graph else None
    return encode_qpu(globals_raw, record, transform)


def parse_dense(raw):
    require_environment()
    from qiskit import QuantumCircuit
    # This is the string equivalent of the historical from_qasm_file dispatch.
    return QuantumCircuit.from_qasm_str(raw.decode("utf-8"))


def dense_features(raw):
    from cudaq_qiskit_adapter import _lower_circuit
    from materialize_cudaq_dense_ir_features_v1 import global_features, graph_record
    circuit = parse_dense(raw)
    normalized, count = _lower_circuit(circuit)
    # Canonical QASM identity bookkeeping is not a consumed neural input.
    return {"global_features": global_features(normalized, shots=32),
            "dag": graph_record(normalized)}, count


def encode_dense(feature, contract, *, graph=True):
    values = np.asarray([float(feature["global_features"][n]) for n in GLOBAL_NAMES], dtype=np.float64)
    tfm = contract["global_transform_values"]
    mean, std = (np.asarray(tfm[k], dtype=np.float64) for k in ("mean", "std"))
    if mean.shape != (7,) or std.shape != (7,) or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("invalid_frozen_dense_transform")
    arrays = {"global_features": ((np.log1p(values) - mean) / std).astype(np.float32).reshape(1, 7)}
    if graph:
        dag, vocab = feature["dag"], contract["gate_vocabulary"]
        g = len(vocab)
        n = len(dag["names"])
        x = np.zeros((n, g + 9), dtype=np.float32)
        denominator = max(1, int(dag["num_qubits"]) - 1)
        for index, name in enumerate(dag["names"]):
            x[index, vocab.get(str(name), vocab["__UNK_GATE__"])] = 1
            x[index, g + min(max(int(dag["arity"][index]), 0), 3)] = 1
            x[index, g + 4] = float(bool(dag["parameter"][index]))
            positions = [int(dag[k][index]) for k in ("q0", "q1", "q2", "q3") if int(dag[k][index]) >= 0]
            if positions:
                normalized = np.asarray(positions, dtype=np.float32) / float(denominator)
                x[index, g + 5:g + 8] = [normalized.min(), normalized.mean(), normalized.max()]
                x[index, g + 8] = 1
        arrays.update(x=x, edge_index=np.asarray([dag["edge_src"], dag["edge_dst"]], dtype=np.int64).reshape(2, -1))
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("nonfinite_model_ready_input")
    return arrays


def preprocess_dense(raw, contract, *, graph=True):
    feature, _ = dense_features(raw)
    return encode_dense(feature, contract, graph=graph)


def pack(arrays):
    return {k: {"dtype": v.dtype.str, "shape": list(v.shape),
                "base64": base64.b64encode(np.ascontiguousarray(v).tobytes()).decode("ascii")}
            for k, v in arrays.items()}


def unpack(value):
    return {k: np.frombuffer(base64.b64decode(v["base64"], validate=True), dtype=np.dtype(v["dtype"]))
            .reshape(v["shape"]).copy() for k, v in value.items()}


def handle(request):
    raw = base64.b64decode(request["raw_qasm_base64"], validate=True)
    graph = bool(request.get("graph", True))
    if request["kind"] == "qpu":
        props = base64.b64decode(request.get("properties_base64", ""), validate=True)
        result = preprocess_qpu(raw, request["source"], props, request["transform"], graph=graph)
    elif request["kind"] == "dense":
        result = preprocess_dense(raw, request["contract"], graph=graph)
    else:
        raise ValueError("unsupported_preprocessing_kind")
    return {"status": "PASS", "arrays": pack(result)}


def main():
    if sys.argv[1:] != ["--serve"]:
        raise SystemExit("usage: historical-python -B am1_preprocessing_compat.py --serve")
    require_environment()
    pinned_extractor()  # Static code/import setup belongs outside warm B2.
    for line in sys.stdin:
        try:
            result = handle(json.loads(line))
        except Exception as exc:
            result = {"status": "BLOCKED", "reason": f"{type(exc).__name__}:{exc}"}
        print(json.dumps(result, allow_nan=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
