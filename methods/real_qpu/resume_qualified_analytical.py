#!/usr/bin/env python3
"""Complete qualified-panel analytical results using verified raw QCRE reuse.

The frozen runner, neural fits and regression fits remain byte-identical.
Only its deterministic QCRE provider is supplied by this executor; calibration
and aggregation still run through the frozen implementation on the new splits.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import run_qualified_logical_benchmark as frozen

STRICT = frozen.ROOT / "artifacts/real_qpu/strict_logical_panel/analytical"
REGISTRY = frozen.ROOT / "artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c133_snapshot_registry_v3.csv"
RAW_METHOD = "strict_qcre_source_estimator_raw"
RAW_CLOCK = "scheduled_unitary_critical_path_seconds"


def now():
    return datetime.now(timezone.utc).isoformat()


def emit(**values):
    line = json.dumps(values, sort_keys=True)
    print(line, flush=True)
    log = frozen.OUT / "execution/analytical.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def canonical_backend(name):
    return "ibm_" + name[5:] if name.startswith("ibmq_") else name


def context_key(row, rep):
    digest = rep["source_qasm_sha256"] if rep["lifecycle_stage"] == "submitted_physical" else rep["compiled_qasm3_sha256"]
    return row["backend"] + "|" + digest


def verify_reuse_row(prior, row, rep, snapshot, assets):
    """Reject stale provenance before taking a raw deterministic prediction."""
    if prior["method_id"] != RAW_METHOD or prior["method_output_clock"] != RAW_CLOCK:
        raise RuntimeError("qcre_reuse_method_or_clock_mismatch")
    expected = {
        "canonical_observation_id": row["canonical_observation_id"],
        "backend": row["backend"],
        "compiled_qasm_sha256": rep["compiled_qasm3_sha256"],
        "source_logical_qasm_sha256": rep["source_qasm_sha256"],
        "snapshot_id": rep["target_snapshot_id"],
    }
    if any(prior[k] != v for k, v in expected.items()):
        raise RuntimeError("qcre_reuse_identity_mismatch")
    if rep["target_snapshot_id"] != snapshot["snapshot_id"] or assets["snapshot_id"] != snapshot["snapshot_id"]:
        raise RuntimeError("qcre_reuse_snapshot_id_mismatch")
    for suffix, rep_key in (("configuration_sha256", "target_configuration_sha256"),
                            ("properties_sha256", "target_properties_sha256")):
        if assets[suffix] != snapshot[suffix] or rep[rep_key] != snapshot[suffix]:
            raise RuntimeError("qcre_reuse_snapshot_bytes_mismatch")
    if float(prior["target_seconds"]) != float(row["target_seconds"]):
        raise RuntimeError("qcre_reuse_target_join_mismatch")
    if int(prior["omitted_measurement_count"]) != int(rep["measurement_count"]):
        raise RuntimeError("qcre_reuse_measurement_semantics_mismatch")
    value = float(prior["prediction_seconds"])
    if prior["status"] != "predicted" or not math.isfinite(value) or value < 0:
        raise RuntimeError("qcre_reuse_nonfinite_or_unsuccessful")
    return value


def prepare(context):
    from run_strict_logical_analytical import QCRE_ROOT, QCRE_REVISION, QCRE_FILE_SHA256
    source = json.loads((STRICT / "run_manifest.json").read_text())
    preflight = json.loads((STRICT / "adapter_preflight.json").read_text())
    if source["status"] != "complete" or source["qcre_revision"] != QCRE_REVISION or source["qcre_estimate_sha256"] != QCRE_FILE_SHA256:
        raise RuntimeError("qcre_reuse_source_pin_mismatch")
    if frozen.sha(STRICT / "raw_attempts.csv") != source["outputs"]["raw_attempts.csv"]:
        raise RuntimeError("qcre_reuse_raw_file_hash_mismatch")
    if frozen.sha(STRICT / "adapter_preflight.json") != source["adapter_preflight_sha256"] or preflight["status"] != "PASS":
        raise RuntimeError("qcre_reuse_preflight_hash_mismatch")
    if frozen.sha(QCRE_ROOT / "estimate.py") != QCRE_FILE_SHA256 or subprocess.check_output(
        ["git", "-C", str(QCRE_ROOT), "rev-parse", "HEAD"], text=True).strip() != QCRE_REVISION:
        raise RuntimeError("qcre_source_pin_mismatch")
    reps = frozen.indexed(frozen.rows(frozen.REP / "representation_rows.csv"), "canonical_row_id")
    registry = frozen.indexed(frozen.rows(REGISTRY), "backend_canonical")
    for name in {canonical_backend(r["backend"]) for r in context["ledger"].values()}:
        entry = registry[name]
        if frozen.sha(entry["class_module_path"]) != entry["class_module_sha256"]:
            raise RuntimeError("qcre_snapshot_class_hash_mismatch")
        for pathkey, hashkey in (("configuration_path", "configuration_sha256"), ("properties_path", "properties_sha256")):
            if frozen.sha(entry[pathkey]) != entry[hashkey]:
                raise RuntimeError("qcre_snapshot_hash_mismatch")
    prior = frozen.indexed([r for r in frozen.rows(STRICT / "raw_attempts.csv") if r["method_id"] == RAW_METHOD])
    cache, reused, checked = {}, [], set()
    for identity, row in context["ledger"].items():
        if row["source_id"] != "mali_real_qpu":
            continue
        rep = reps[identity]
        snapshot = registry[canonical_backend(row["backend"])]
        value = verify_reuse_row(prior[identity], row, rep, snapshot,
                                 preflight["snapshot_assets"][canonical_backend(row["backend"])])
        path = frozen.REP / rep["compiled_qasm3_file"]
        if str(path) not in checked:
            if frozen.sha(path) != rep["compiled_qasm3_sha256"]:
                raise RuntimeError("qcre_reuse_compiled_file_hash_mismatch")
            checked.add(str(path))
        key = context_key(row, rep)
        if key in cache and not math.isclose(cache[key]["value"], value, rel_tol=1e-10, abs_tol=1e-12):
            raise RuntimeError("qcre_reuse_duplicate_context_disagrees")
        cache[key] = {"value": value, "reason": "", "origin": "verified_strict_raw"}
        reused.append(identity)
    if len(reused) != 340:
        raise RuntimeError("qcre_reuse_requires_all_340_verified")
    pins = {"base": context["pins"], "executor_sha256": frozen.sha(Path(__file__)),
            "source_manifest_sha256": frozen.sha(STRICT / "run_manifest.json"),
            "source_raw_sha256": frozen.sha(STRICT / "raw_attempts.csv"),
            "source_preflight_sha256": frozen.sha(STRICT / "adapter_preflight.json"),
            "qcre_revision": QCRE_REVISION, "qcre_sha256": QCRE_FILE_SHA256}
    return reps, registry, cache, reused, pins


def qcre_reusing_raw(context):
    from qiskit import QuantumCircuit, qasm3
    from qiskit_ibm_runtime import fake_provider
    from run_strict_logical_analytical import QCRE_ROOT, qcre_custom_path
    from qasm3_hardware_wire_adapter import restore_saved_qasm3_hardware_wires
    from qonductor_native_adapter import load_pinned_qonductor_methods
    from zipfile import ZipFile
    reps, registry, cache, reused, pins = prepare(context)
    sys.path.append(os.environ.get("STRICT_BQSKIT_SITE", str(frozen.BQSKIT_SITE)))
    sys.path.insert(0, str(QCRE_ROOT))
    from estimate import estimate_runtime
    from bqskit.ext.qiskit import qiskit_to_bqskit
    canonical = frozen.indexed(frozen.rows(frozen.CANONICAL), "canonical_row_id")
    contract = json.loads((frozen.ROOT / "benchmark_v1/protocol/qonductor_native_features.json").read_text())
    _, _, parse_submitted, _ = load_pinned_qonductor_methods(frozen.ROOT.parent / "Qonductor-SC25", contract["source_pin"])
    maps = {}
    for backend in sorted({r["backend"] for r in context["ledger"].values()}):
        entry = registry[canonical_backend(backend)]
        device = getattr(fake_provider, entry["fake_backend_class"].split(".")[-1])()
        duration_map = {}
        for opname in device.target.operation_names:
            op = device.target.operation_from_name(opname)
            if opname in {"measure", "reset", "delay"}:
                continue
            for loc, props in device.target[opname].items():
                if loc is not None and props is not None and props.duration is not None:
                    duration = float(props.duration)
                    if not math.isfinite(duration) or duration < 0:
                        raise RuntimeError("bad_duration")
                    duration_map.setdefault(op.num_qubits, {}).setdefault(tuple(loc), {})[opname] = duration
        maps[backend] = duration_map
    checkpoint = frozen.OUT / "analytical/qcre_context_cache.json"
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved["pins"] != pins:
            raise RuntimeError("qcre_checkpoint_pin_mismatch")
        for key, value in saved["contexts"].items():
            if key in cache and value != cache[key]:
                raise RuntimeError("qcre_checkpoint_reuse_mismatch")
        cache.update(saved["contexts"])
    unique = {context_key(r, reps[i]) for i, r in context["ledger"].items()}
    if not set(cache) <= unique:
        raise RuntimeError("qcre_checkpoint_foreign_context")
    emit(stage="qcre", reused_observations=len(reused), reused_contexts=len({context_key(context["ledger"][i], reps[i]) for i in reused}),
         remaining_contexts=len(unique - set(cache)), total_contexts=len(unique))
    frozen.write_json(checkpoint, {"pins": pins, "contexts": cache})
    output = defaultdict(dict)
    with ZipFile(frozen.QONDUCTOR_ARCHIVE) as archive:
        for identity, row in context["ledger"].items():
            rep = reps[identity]
            key = context_key(row, rep)
            if key not in cache:
                if rep["lifecycle_stage"] == "submitted_physical":
                    source = archive.read(canonical[identity]["qasm_path_or_member"])
                    if hashlib.sha256(source).hexdigest() != rep["source_qasm_sha256"]:
                        raise RuntimeError("submitted_QASM_hash_mismatch")
                    circuit = parse_submitted(source.decode())
                else:
                    path = frozen.REP / rep["compiled_qasm3_file"]
                    if frozen.sha(path) != rep["compiled_qasm3_sha256"]:
                        raise RuntimeError("compiled_hash_mismatch")
                    source, _ = restore_saved_qasm3_hardware_wires(path.read_text(), int(rep["register_width"]))
                    circuit = qasm3.loads(source)
                try:
                    unitary = QuantumCircuit(circuit.num_qubits)
                    measured = False
                    for item in circuit.data:
                        name = item.operation.name
                        if name in {"barrier", "snapshot"}:
                            continue
                        if name == "measure":
                            measured = True
                            continue
                        if measured or item.clbits or getattr(item.operation, "blocks", None) is not None or name in {"reset", "delay"}:
                            raise ValueError("unsupported_nonunitary_or_nonterminal_measurement")
                        unitary.append(item.operation, [unitary.qubits[circuit.find_bit(q).index] for q in item.qubits])
                    value = float(estimate_runtime(qiskit_to_bqskit(unitary), maps[row["backend"]]))
                    reference = qcre_custom_path(unitary, maps[row["backend"]])
                    if not math.isfinite(value) or not math.isclose(value, reference, rel_tol=1e-10, abs_tol=1e-12):
                        raise RuntimeError("qcre_source_adapter_parity_failure")
                    cache[key] = {"value": value, "reason": "", "origin": "computed_source_with_parity"}
                except (ValueError, KeyError, NotImplementedError) as exc:
                    cache[key] = {"value": None, "reason": type(exc).__name__ + ":" + str(exc)[:200], "origin": "computed_source_with_parity"}
                frozen.write_json(checkpoint, {"pins": pins, "contexts": cache})
                emit(stage="qcre", completed_contexts=len(cache), total_contexts=len(unique), context=key,
                     status="predicted" if cache[key]["value"] is not None else "unavailable")
            result = cache[key]
            for method, multiplier in (("qcre_source_estimator", 1), ("qcre_source_estimator_shot_scaled", float(row["shots"]))):
                output[method][identity] = dict(status="predicted" if result["value"] is not None else "unavailable",
                    predicted_seconds="" if result["value"] is None else result["value"] * multiplier,
                    terminal_reason=result["reason"])
    frozen.write_json(frozen.OUT / "analytical/qcre_source_receipt.json", dict(pins=pins,
        source_revision=pins["qcre_revision"], source_sha256=pins["qcre_sha256"],
        unique_compiled_contexts=len(cache), reused_observations=len(reused), reused_observation_ids_sha256=frozen.idhash(reused),
        reused_contexts=sum(v["origin"] == "verified_strict_raw" for v in cache.values()),
        computed_contexts=sum(v["origin"] == "computed_source_with_parity" for v in cache.values()),
        cache_sha256=frozen.sha(checkpoint), terminal_measurements_omitted=True,
        parity="new contexts checked individually; reused outputs bound to strict source/preflight hashes",
        converter_sha256=frozen.sha(Path(sys.modules["bqskit.ext.qiskit.translate"].__file__)),
        environment={p: importlib.metadata.version(p) for p in ("numpy", "qiskit", "qiskit-ibm-runtime", "bqskit", "scikit-learn")}))
    return output, {"qcre_source_estimator": RAW_CLOCK, "qcre_source_estimator_shot_scaled": "shot_scaled_unitary_seconds"}


def complete():
    context = frozen.load_context()
    receipt_path = frozen.OUT / "analytical/execution_receipt.json"
    status_path = frozen.OUT / "execution/status.json"
    previous = json.loads(status_path.read_text())
    for pid in [previous.get("supervisor_pid"), *previous.get("workers", {}).values()]:
        if pid and pid != os.getpid() and Path(f"/proc/{pid}/cmdline").exists():
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
            if b"run_qualified_logical_benchmark.py" in cmd:
                raise RuntimeError("qualified_worker_still_alive")
    _, _, _, reused, pins = prepare(context)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["pins"] != pins:
            raise RuntimeError("analytical_resume_execution_pin_mismatch")
        if receipt["status"] == "complete":
            for name, expected in receipt["output_hashes"].items():
                path = {"analytical_manifest": frozen.OUT / "analytical/manifest.json",
                        "summary_manifest": frozen.OUT / "summary/manifest.json"}.get(name, frozen.OUT / "analytical" / name)
                if frozen.sha(path) != expected:
                    raise RuntimeError("completed_analytical_output_hash_mismatch")
            emit(stage="qualified_benchmark", status="already_complete", rows=4515)
            return
        receipt.update(status="running", resumed_utc=now(), previous_receipt_sha256=frozen.sha(receipt_path))
    else:
        receipt = dict(status="running", started_utc=now(), pins=pins, prior_execution_status=previous,
                       reused_raw_observations=len(reused), previous_cohort_fitted_models_reused=False,
                       fresh_outer_train_calibration=True, frozen_neural_and_regression_results_retained=True)
    frozen.write_json(receipt_path, receipt)
    frozen.write_json(status_path, dict(status="running", pins=context["pins"], executor_sha256=pins["executor_sha256"],
        supervisor_pid=os.getpid(), workers={"analytical": os.getpid()}, started_utc=receipt["started_utc"]))
    original = frozen.qcre_raw
    try:
        # Explicit provider injection: the frozen calibration implementation is
        # unchanged; the receipt binds both executor and cached source evidence.
        frozen.qcre_raw = qcre_reusing_raw
        frozen.cpu_analytical()
        emit(stage="analytical", status="complete", action="aggregate_current_4515")
        frozen.summarize()
        receipt.update(status="complete", completed_utc=now(), output_hashes={
            "qcre_source_receipt.json": frozen.sha(frozen.OUT / "analytical/qcre_source_receipt.json"),
            "analytical_manifest": frozen.sha(frozen.OUT / "analytical/manifest.json"),
            "summary_manifest": frozen.sha(frozen.OUT / "summary/manifest.json")})
        frozen.write_json(receipt_path, receipt)
        frozen.write_json(status_path, dict(status="complete", pins=context["pins"],
            executor_sha256=pins["executor_sha256"], supervisor_pid=os.getpid(), completed_utc=receipt["completed_utc"],
            execution_receipt_sha256=frozen.sha(receipt_path)))
        emit(stage="qualified_benchmark", status="complete", rows=4515)
    except BaseException as exc:
        receipt.update(status="failed", error=str(exc), stopped_utc=now())
        frozen.write_json(receipt_path, receipt)
        frozen.write_json(status_path, dict(status="failed", pins=context["pins"], error=str(exc)))
        raise
    finally:
        frozen.qcre_raw = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "complete"), default="complete")
    args = parser.parse_args()
    if args.stage == "preflight":
        context = frozen.load_context()
        _, _, cache, reused, pins = prepare(context)
        print(json.dumps(dict(status="PASS", reused_observations=len(reused), reused_contexts=len(cache), pins=pins)))
    else:
        complete()


if __name__ == "__main__":
    main()
