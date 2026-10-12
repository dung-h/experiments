#!/usr/bin/env python3
"""AM1 Phase 3: deterministic preprocessing, independent frozen-input checks.

Run prepare, diagnose in both environments, bridge in the CUDA interpreter
(CPU tensors only), then report. No predictor is instantiated or called.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "benchmark_v1/scripts"
OUT = ROOT / "artifacts/am1_preprocessing_recovery/v1"
PHASE2 = ROOT / "artifacts/am1_phase2/v1"
PANEL = ROOT / "artifacts/real_qpu/qualified_logical_panel"
DENSE = ROOT / "work/benchmark_recovery/r8_dense_predictors/attempt_002"
HISTORICAL_PYTHON = ROOT.parent / ".venv-qonductor/bin/python"
CUDA_PYTHON = ROOT.parent / "Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python"
sys.path.insert(0, str(SCRIPTS))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_hash(arrays):
    h = hashlib.sha256()
    for k, v in sorted(arrays.items()):
        h.update(json.dumps([k, v.dtype.str, list(v.shape)]).encode())
        h.update(v.tobytes(order="C"))
    return h.hexdigest()


def safety():
    if shutil.disk_usage(ROOT).free < 3 * 2**30:
        raise RuntimeError("disk_floor")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 8 * 2**30:
        raise RuntimeError("RAM_cap")
    if OUT.exists() and sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) > 32 * 2**20:
        raise RuntimeError("derived_output_cap")


def write_json(name, value):
    safety()
    with (OUT / name).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def write_csv(name, data):
    with (OUT / name).open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def verify_pins():
    pins = read(OUT / "acceptance_contract.json")["frozen_pins"]
    bad = [p for p, digest in pins.items() if not Path(p).is_file() or sha(p) != digest]
    if bad:
        raise RuntimeError("frozen_integrity_failure:" + str(bad))
    return len(pins)


def prepare():
    if OUT.exists():
        raise RuntimeError("refuse_existing_recovery_namespace")
    p2 = read(PHASE2 / "provenance_manifest.json")
    pins = dict(p2["source_pins"])
    for p, d in pins.items():
        if sha(p) != d:
            raise RuntimeError("phase2_source_pin_drift:" + p)
    selected = rows(PHASE2 / "selected_workloads.csv")
    panel = {r["canonical_observation_id"]: r for r in rows(PANEL / "panel.csv")}
    for root in [PHASE2, ROOT / "artifacts/am1_overhead_pilot/v1",
                 ROOT / "artifacts/am1_postpilot_audit/v1", ROOT / "artifacts/am1_feasibility/v1",
                 ROOT / "artifacts/final_research"]:
        for p in root.rglob("*"):
            if p.is_file():
                pins[str(p)] = sha(p)
    for r in selected:
        if r["context"] == "qualified_qpu_4515" and r["raw_path"]:
            p = ROOT / panel[r["id"]]["graph_file"]
            if sha(p) != panel[r["id"]]["graph_sha256"]:
                raise RuntimeError("independent_graph_pin")
            pins[str(p)] = sha(p)
    for p in [PANEL / "panel.csv", PANEL / "outer_splits.csv",
              ROOT / "artifacts/real_qpu/logical_inputs/feature_manifest.json",
              SCRIPTS / "prepare_logical_benchmark.py", SCRIPTS / "run_am1_phase2.py",
              SCRIPTS / "analyze_am1_phase2.py",
              ROOT / "work/benchmark_recovery/r3_dense/ir_features_v1/manifest.json",
              ROOT / "work/benchmark_recovery/r3_dense/ir_features_v1/dense_features_by_hash.jsonl"]:
        pins[str(p)] = sha(p)
    for fold in range(5):
        paths = [PANEL / f"neural/fold_{fold}/transform.json"]
        paths += [DENSE / precision / f"neural/fold_{fold}/fold_inputs.json" for precision in ("fp32", "fp64")]
        for p in paths:
            pins[str(p)] = sha(p)
    OUT.mkdir(parents=True)
    write_json("acceptance_contract.json", dict(
        authorization="user_attachment_24ca7545-4aeb-4de8-8b59-ee5a87479234",
        phase="preprocessing_only_no_inference_or_timing", frozen_pins=pins,
        cohort="Phase2 selected IDs unchanged: QPU45 raw-available/50 attempts; dense30 per precision",
        sparse_structure_dtype_shape="exact", model_ready_arrays="exact including float32 dtype and node_index",
        raw_51_globals="absolute tolerance1e-12; rtol0; historical decimal CSV serialization",
        RCCX_static_matrix_atol=1e-14, RCCX_static_matrix_rtol=0,
        canonical_graph="secondary diagnostic only, never sufficient model-ready acceptance",
        snapshot_mapping="logical wire i to nominal physical snapshot i, microseconds; no compilation",
        resources=dict(RAM_bytes=8*2**30, disk_floor_bytes=3*2**30, output_bytes=32*2**20,
                       worker_count=1, worker_timeout_seconds=180),
        historical_python=str(HISTORICAL_PYTHON), cuda_python=str(CUDA_PYTHON),
        code_pins={str(p): sha(p) for p in (Path(__file__), SCRIPTS / "am1_preprocessing_compat.py")},
        already_completed_read_only_root_cause_probe="45 inputs in both existing environments before adapter authoring"))


def sequence(c):
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits],
             [c.find_bit(q).index for q in i.clbits], [str(x) for x in i.operation.params]] for i in c.data]


def causal_keys(c):
    counts = Counter()
    keys = []
    for name, qubits, clbits, params in sequence(c):
        keys.append(json.dumps([name, qubits, clbits, params, [counts[q] for q in qubits]], separators=(",", ":")))
        counts.update(qubits)
    if len(set(keys)) != len(keys):
        raise RuntimeError("ambiguous_causal_key")
    return keys


def graph_canonical_hash(record, clean):
    """Prove a relabeling of named-operation DAG/metadata, not quantum equivalence."""
    import numpy as np
    active = sorted({clean.find_bit(q).index for i in clean.data for q in i.qubits})
    keys = [f"{kind}:{wire}" for wire in active for kind in ("in", "out")] + causal_keys(clean)
    if len(keys) != len(record["node_type"]):
        raise RuntimeError("canonical_node_alignment")
    order = np.argsort(np.asarray(keys))
    inverse = np.empty(len(keys), dtype=np.int64)
    inverse[order] = np.arange(len(keys))
    arr = {k: v[order] for k, v in record.items() if k not in {"edge_index", "node_index"}}
    edges = inverse[record["edge_index"]]
    arr["edge_index"] = edges[:, np.lexsort((edges[1], edges[0]))].astype(np.int32)
    return stable(sorted(keys)) + ":" + array_hash(arr)


def circuit_summary(c):
    defs = {}
    for i in c.data:
        if i.operation.name not in defs:
            d = i.operation.definition
            defs[i.operation.name] = dict(type=type(i.operation).__name__,
                definition=None if d is None else sequence(d))
    return dict(q=c.num_qubits, c=c.num_clbits,
        registers=dict(q=[[x.name, len(x)] for x in c.qregs], c=[[x.name, len(x)] for x in c.cregs]),
        sequence_hash=stable(sequence(c)), definitions=defs, global_phase=str(c.global_phase))


def environment():
    return dict(executable=sys.executable, python=platform.python_version(),
                packages={k: importlib.metadata.version(k) for k in ("qiskit", "numpy", "rustworkx")})


def diagnose(flavor):
    verify_pins()
    import numpy as np
    import mali_full_features as mf
    import am1_preprocessing_compat as compat
    import cudaq_qiskit_adapter as adapter
    import materialize_cudaq_dense_ir_features_v1 as feat
    from qiskit import QuantumCircuit, qasm2
    from qiskit.transpiler.passes import RemoveBarriers, RemoveFinalMeasurements
    from qiskit.converters import circuit_to_dag
    from qiskit.circuit.library import RCCXGate
    env = environment()
    required = "2.5.2" if flavor == "historical" else "0.44.2"
    if env["packages"]["qiskit"] != required:
        raise RuntimeError("diagnostic_interpreter_mismatch")
    selected = rows(PHASE2 / "selected_workloads.csv")
    panel = {r["canonical_observation_id"]: r for r in rows(PANEL / "panel.csv")}
    frozen_dense = {r["qasm_sha256"]: r for r in
                    map(json.loads, (ROOT / "work/benchmark_recovery/r3_dense/ir_features_v1/dense_features_by_hash.jsonl").read_text().splitlines())}
    extractor, _ = mf.load_pinned_global_extractor()
    qpu, dense = [], []
    for r in selected:
        safety()
        if not r["raw_path"]:
            continue
        raw = Path(r["raw_path"]).read_bytes()
        if sha(r["raw_path"]) != r["raw_sha256"]:
            raise RuntimeError("raw_hash")
        if r["context"] == "qualified_qpu_4515":
            c = mf.parse_qasm(raw)[0] if r["source"] == "mali_real_qpu" else qasm2.loads(raw.decode(), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
            after_barrier = RemoveBarriers()(c)
            clean = RemoveFinalMeasurements()(after_barrier)
            rec, meta = mf.compact_graph_record(c, compat.calibration(Path(r["properties_path"]).read_bytes()))
            with np.load(ROOT / panel[r["id"]]["graph_file"], allow_pickle=False) as file:
                saved = {k: file[k] for k in file.files}
            mismatches = [k for k in rec if rec[k].dtype != saved[k].dtype or not np.array_equal(rec[k], saved[k])]
            globals_raw = mf.extract_global_features(c, extractor)
            globals_frozen = [float(panel[r["id"]][f"g{i:02d}"]) for i in range(51)]
            tfm = read(PANEL / f"neural/fold_{r['fold']}/transform.json")
            actual = compat.encode_qpu(globals_raw, rec, tfm)
            expected = compat.encode_qpu(globals_frozen, saved, tfm)
            qpu.append(dict(id=r["id"], source=r["source"], family=r["family"], backend=r["backend"], fold=r["fold"],
                raw_path=r["raw_path"], raw_sha256=r["raw_sha256"], properties_path=r["properties_path"],
                properties_sha256=r["properties_sha256"], snapshot_id=panel[r["id"]]["snapshot_id"],
                logical_tier=panel[r["id"]]["logical_input_tier"], frozen_graph_path=panel[r["id"]]["graph_file"],
                frozen_graph_sha256=panel[r["id"]]["graph_sha256"], raw_summary=circuit_summary(c),
                barrier_sequence_hash=stable(sequence(after_barrier)), clean_sequence_hash=stable(sequence(clean)),
                causal_sequence_hash=stable(sorted(causal_keys(clean))),
                dag_order_hash=stable([[n.name, [clean.find_bit(q).index for q in n.qargs]] for n in circuit_to_dag(clean).op_nodes()]),
                sparse_fresh_hash=array_hash(rec), sparse_frozen_hash=array_hash(saved),
                canonical_graph_hash=graph_canonical_hash(rec, clean), mismatched_fields=mismatches,
                shape_dtype={k: [list(v.shape), v.dtype.str] for k, v in rec.items()}, metadata=meta,
                global51_match=bool(np.allclose(globals_raw, globals_frozen, atol=1e-12, rtol=0)),
                model_ready_hash=array_hash(actual), frozen_model_ready_hash=array_hash(expected),
                model_ready_exact=all(np.array_equal(actual[k], expected[k]) for k in actual),
                model_ready_shapes={k: list(v.shape) for k,v in actual.items()}))
        else:
            c = QuantumCircuit.from_qasm_str(raw.decode()) if flavor == "historical" else qasm2.loads(raw.decode(), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
            item = dict(id=r["id"], context=r["context"], fold=r["fold"], family=r["family"], raw_path=r["raw_path"],
                        raw_sha256=r["raw_sha256"], raw_summary=circuit_summary(c),
                        raw_rccx_declaration="qelib1.inc implicit library; no explicit gate rccx" if "gate rccx" not in raw.decode() else "explicit declaration")
            try:
                n, count = adapter._lower_circuit(c)
                f = dict(global_features=feat.global_features(n), dag=feat.graph_record(n))
                original = frozen_dense[r["id"]]
                item.update(status="EXACT MATCH" if all(f[k] == original[k] for k in f) else "MISMATCH",
                            rccx_lowered_count=count, consumed_hash=stable(f), frozen_consumed_hash=stable({k: original[k] for k in f}))
            except Exception as exc:
                item.update(status="MISMATCH", reason=f"{type(exc).__name__}:{exc}")
            dense.append(item)
    gate = RCCXGate()
    definition = gate.definition
    # Static gate matrices only: no state, simulator or circuit execution.
    static = [dict(name=i.operation.name, qubits=[definition.find_bit(q).index for q in i.qubits],
                   params=[float(x) for x in i.operation.params],
                   real=i.operation.to_matrix().real.tolist(), imag=i.operation.to_matrix().imag.tolist())
              for i in definition.data]
    write_json(f"diagnostic_{flavor}.json", dict(environment=env, qpu=qpu, dense=dense,
        static_rccx_definition=static, static_rccx_matrix=dict(real=gate.to_matrix().real.tolist(), imag=gate.to_matrix().imag.tolist()),
        relevant_dependency_paths={k: str(v) for k,v in dict(qiskit=__import__('qiskit').__file__).items()},
        frozen_pins_verified=verify_pins(), no_inference=True, no_timing=True))
    print(json.dumps(dict(stage="diagnose", flavor=flavor, qpu_exact=sum(not r["mismatched_fields"] for r in qpu),
                         qpu_model_ready=sum(r["model_ready_exact"] for r in qpu), dense_exact=sum(r["status"] == "EXACT MATCH" for r in dense))))


def bridge():
    """CUDA interpreter consumes regenerated CPU arrays; no CUDA/inference calls."""
    verify_pins()
    import numpy as np
    import torch
    import am1_preprocessing_compat as compat
    import run_cudaq_dense_neural_predictors_v1 as dense_source
    if importlib.metadata.version("qiskit") != "0.44.2" or torch.__version__ != "2.7.1+cu128":
        raise RuntimeError("consumer_environment_mismatch")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    child = subprocess.Popen([str(HISTORICAL_PYTHON), "-B", str(SCRIPTS / "am1_preprocessing_compat.py"), "--serve"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    panel = {r["canonical_observation_id"]: r for r in rows(PANEL / "panel.csv")}
    frozen_dense = {r["qasm_sha256"]: r for r in
                    map(json.loads, (ROOT / "work/benchmark_recovery/r3_dense/ir_features_v1/dense_features_by_hash.jsonl").read_text().splitlines())}
    receipts = []
    try:
        for r in rows(PHASE2 / "selected_workloads.csv"):
            if not r["raw_path"]:
                continue
            for graph in (True, False):
                raw = Path(r["raw_path"]).read_bytes()
                req = dict(kind="qpu" if r["context"] == "qualified_qpu_4515" else "dense",
                           raw_qasm_base64=base64.b64encode(raw).decode(), graph=graph)
                if req["kind"] == "qpu":
                    tfm = read(PANEL / f"neural/fold_{r['fold']}/transform.json")
                    req.update(source=r["source"], transform=tfm,
                               properties_base64=base64.b64encode(Path(r["properties_path"]).read_bytes()).decode())
                    from run_mali_full_features import _make_example
                    row = panel[r["id"]]
                    ledger_row = {f"global_{i:02d}": row[f"g{i:02d}"] for i in range(51)}
                    ledger_row.update(graph_status="available", graph_artifact=row["graph_file"],
                                      graph_artifact_sha256=row["graph_sha256"])
                    frozen = _make_example(r["id"], "mali_full_features_graph" if graph else "mali_full_features_mlp",
                        {r["id"]:ledger_row}, None, tfm, ROOT, None, False)
                    expected = dict(global_features=frozen.global_features.numpy())
                    if graph:
                        expected.update(x=frozen.x.numpy(), edge_index=frozen.edge_index.numpy())
                else:
                    folder = DENSE / r["context"].split("_")[-1] / f"neural/fold_{r['fold']}"
                    method = "mali_style_dense_graph" if graph else "matched_metadata_mlp"
                    contract = read(folder / "fold_inputs.json")["feature_contracts"][method]
                    req["contract"] = contract
                    frozen = frozen_dense[r["id"]]
                    # Independently call original frozen tensor encoder, not new expected arrays.
                    gf = dense_source.normalized_globals(frozen, contract["global_transform_values"])
                    expected = dict(global_features=gf.reshape(1, 7))
                    if graph:
                        x, edges = dense_source.encode_dag(frozen["dag"], contract["gate_vocabulary"])
                        expected.update(x=x.numpy(), edge_index=edges.numpy())
                child.stdin.write(json.dumps(req) + "\n")
                child.stdin.flush()
                response = json.loads(child.stdout.readline())
                if response["status"] != "PASS":
                    raise RuntimeError("preprocessing_bridge:" + str(response))
                actual = compat.unpack(response["arrays"])
                same = set(actual) == set(expected) and all(actual[k].shape == expected[k].shape and
                    actual[k].dtype == expected[k].dtype and np.array_equal(actual[k], expected[k]) for k in actual)
                tensors = {k: torch.as_tensor(v.copy()) for k,v in actual.items()}
                receipts.append(dict(id=r["id"], context=r["context"], fold=r["fold"], model="Graph" if graph else "MLP",
                    exact_model_ready_match=bool(same), actual_hash=array_hash(actual), expected_hash=array_hash(expected),
                    cpu_tensor_devices={k: str(v.device) for k,v in tensors.items()}, shape_dtype={k:[list(v.shape),str(v.dtype)] for k,v in tensors.items()}))
                if not same:
                    raise RuntimeError("independent_frozen_input_mismatch:" + r["id"])
                safety()
        child.stdin.close()
        if child.wait(timeout=30):
            raise RuntimeError("preprocessing_server_failed:" + child.stderr.read())
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
    write_json("bridge_validation.json", dict(status="PASS", receipts=receipts,
        consumer_packages={k:importlib.metadata.version(k) for k in ("qiskit", "numpy", "torch", "torch-geometric")},
        CPU_tensor_construction_only=True, model_forward_calls=0, CUDA_calls=0,
        predictions_or_targets_used_for_preprocessing=False, frozen_pins_verified=verify_pins()))
    print(json.dumps(dict(stage="bridge", status="PASS", input_comparisons=len(receipts))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("prepare", "diagnose", "bridge", "report", "check"), required=True)
    p.add_argument("--flavor", choices=("historical", "phase2"))
    args = p.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "diagnose":
        diagnose(args.flavor)
    elif args.stage == "bridge":
        bridge()
    else:
        from report_am1_preprocessing_recovery import report
        report(check=args.stage == "check")


if __name__ == "__main__":
    # Scheduling timeouts are safety controls, never latency observations.
    import signal
    signal.alarm(180)
    resource.setrlimit(resource.RLIMIT_AS, (8*2**30, 8*2**30))
    main()
