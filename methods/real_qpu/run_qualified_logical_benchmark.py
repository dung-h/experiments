#!/usr/bin/env python3
"""Fresh grouped refits on the 4515 reconstruction-qualified QPU observations.

Stages share one explicit context; old runners' global paths are never patched.
Raw observations/features are read-only. See qualified_logical_benchmark.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import itertools
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "benchmark_v1/scripts"
sys.path.insert(0, str(SCRIPTS))
CONTRACT = ROOT / "benchmark_v1/protocol/qualified_logical_benchmark.json"
OUT = ROOT / "artifacts/real_qpu/qualified_logical_panel"
OLD = ROOT / "artifacts/real_qpu/common_panel"
LOGICAL = ROOT / "artifacts/real_qpu/logical_inputs"
NATIVE = ROOT / "artifacts/real_qpu/qonductor_native_features/feature_rows.csv"
REP = ROOT / "artifacts/benchmark_v3/real_qpu/compiled_unified"
COMPONENTS = ROOT / "artifacts/benchmark_v3/real_qpu/analytical_extension/components.csv"
CANONICAL = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
QONDUCTOR_ARCHIVE = ROOT.parent / "Qonductor-SC25/data/database/circuits.zip"
CPU_PYTHON = ROOT.parent / ".venv-qonductor/bin/python"
GPU_PYTHON = ROOT.parent / "Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python"
BQSKIT_SITE = ROOT.parent / ".venv-cdaa-new/lib/python3.10/site-packages"
SEEDS = (42, 1234, 31415)
NEURAL = ("mali_logical_graph", "mali_global_mlp")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    csv.field_size_limit(100_000_000)
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def indexed(values, key="canonical_observation_id"):
    result = {r[key]: r for r in values}
    if len(result) != len(values):
        raise RuntimeError(f"duplicate_ids:{key}")
    return result


def idhash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def write_json(path, value):
    from run_mali_full_features import atomic_json
    atomic_json(Path(path), value)


def write_csv(path, values):
    from run_mali_full_features import write_csv_atomic
    if not values:
        raise ValueError(f"empty_table:{path}")
    write_csv_atomic(Path(path), values, list(values[0]))


def split_groups(selected):
    from freeze_strict_logical_panel import assign_groups, validate_splits
    components = defaultdict(list)
    source = {}
    for r in selected:
        rid = r["canonical_observation_id"]
        components[r["group_id"]].append(rid)
        source[rid] = r["source_id"]
    outer_assignment, support = assign_groups(components, source, 5, 42)
    outer, inner, inner_support = [], [], {}
    for group, ids in sorted(components.items()):
        outer.extend(dict(canonical_observation_id=i, group_id=group,
                          outer_fold=outer_assignment[group]) for i in sorted(ids))
    for fold in range(5):
        train = {g: ids for g, ids in components.items() if outer_assignment[g] != fold}
        allocation, counts = assign_groups(train, source, 4, 43 + fold)
        inner_support[str(fold)] = counts
        for group, ids in sorted(train.items()):
            inner.extend(dict(canonical_observation_id=i, group_id=group,
                              outer_fold=fold, inner_fold=allocation[group]) for i in sorted(ids))
    validate_splits(outer, inner)
    return outer, inner, dict(groups=len(components), outer_source_counts=support,
                             inner_source_counts=inner_support)


def freeze():
    contract = json.loads(CONTRACT.read_text())
    prior = OUT / "manifest.json"
    if prior.exists():
        load_context()
        return json.loads(prior.read_text())
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("refuse_unreceipted_nonempty_panel")
    selected = [r for r in rows(OLD / "panel.csv") if r["logical_input_tier"] in contract["included_tiers"]]
    if len(selected) != 4515 or dict(Counter(r["source_id"] for r in selected)) != contract["source_counts"]:
        raise RuntimeError("qualified_cohort_count_or_source_mismatch")
    target = indexed(rows(OLD / "targets.csv"))
    features = indexed(rows(LOGICAL / "feature_attempts.csv"))
    native = indexed(rows(NATIVE), "canonical_row_id")
    reps = indexed(rows(REP / "representation_rows.csv"), "canonical_row_id")
    graphs = {}
    data = []
    for r in selected:
        rid = r["canonical_observation_id"]
        f, t, n, rep = features[rid], target[rid], native[rid], reps[rid]
        if f["status"] != "available" or f["logical_input_tier"] != r["logical_input_tier"]:
            raise RuntimeError(f"invalid_logical_input:{rid}")
        graph = LOGICAL / f["graph_file"]
        if graph.resolve().parent != (LOGICAL / "graphs").resolve():
            raise RuntimeError("graph_path_escape")
        relative = str(graph.relative_to(ROOT))
        if relative not in graphs:
            if sha(graph) != f["graph_sha256"]:
                raise RuntimeError(f"graph_hash_mismatch:{rid}")
            with np.load(graph, allow_pickle=False) as payload:
                if len(payload["node_type"]) != int(f["node_count"]):
                    raise RuntimeError(f"node_count_mismatch:{rid}")
            graphs[relative] = f["graph_sha256"]
        if graphs[relative] != r["graph_sha256"] or graphs[relative] != f["graph_sha256"]:
            raise RuntimeError(f"row_graph_join_mismatch:{rid}")
        native_hash = rep["source_qasm_sha256"] if rep["lifecycle_stage"] == "submitted_physical" else rep["compiled_qasm3_sha256"]
        if n["availability_status"] != "available" or n["input_qasm_sha256"] != native_hash:
            raise RuntimeError(f"native_compiled_join_mismatch:{rid}")
        canonical_backend = t["backend"].replace("ibmq_", "ibm_")
        if canonical_backend != rep["backend_canonical"]:
            raise RuntimeError(f"native_backend_mismatch:{rid}")
        if int(float(n["shots"])) != int(float(t["shots"])) or float(n["circuit_count"]) != 1:
            raise RuntimeError(f"shots_or_circuit_count_mismatch:{rid}")
        counts = json.loads(rep["gate_counts_json"])
        if float(n["swap"]) != sum(counts.get(k,0) for k in ("cx","cz","ecr")):
            raise RuntimeError(f"upstream_counter_mismatch:{rid}")
        if float(n["depth"]) != float(rep["structural_depth"]) or float(n["num_qubits"]) != float(rep["active_width"]):
            raise RuntimeError(f"compiled_features_mismatch:{rid}")
        if not np.isfinite([float(f[f"g{i:02d}"]) for i in range(51)]).all():
            raise RuntimeError(f"nonfinite_globals:{rid}")
        if not math.isfinite(float(t["target_seconds"])) or float(t["target_seconds"]) <= 0:
            raise RuntimeError(f"bad_target:{rid}")
        data.append({**r, **t, **{f"g{i:02d}": f[f"g{i:02d}"] for i in range(51)},
                     "graph_file": relative, "graph_sha256": f["graph_sha256"],
                     "node_count": f["node_count"], "edge_count": f["edge_count"],
                     "source_qasm_sha256": f["source_qasm_sha256"],
                     "compiled_qasm_sha256": native_hash})
    outer, inner, support = split_groups(data)
    candidates_path = ROOT / "artifacts/real_qpu/strict_logical_panel/qonductor/frozen_candidates.json"
    candidate_source = json.loads(candidates_path.read_text())
    if [candidate_source["candidate_counts"][f] for f in contract["regression"]["families"]] != contract["regression"]["candidate_budget"]:
        raise RuntimeError("candidate_budget_mismatch")
    OUT.mkdir(parents=True)
    write_csv(OUT / "panel.csv", data)
    write_csv(OUT / "outer_splits.csv", outer)
    write_csv(OUT / "inner_splits.csv", inner)
    write_json(OUT / "candidates.json", {"candidates": candidate_source["candidates"],
        "source_revision": candidate_source["source_revision"],
        "source_file_sha256": candidate_source["source_file_sha256"],
        "source_candidate_manifest_sha256": sha(candidates_path)})
    inputs = [CONTRACT, Path(__file__), OLD / "panel.csv", OLD / "targets.csv",
              LOGICAL / "feature_attempts.csv", NATIVE, REP / "representation_rows.csv", COMPONENTS,
              OLD / "analytical/attempts.csv", OLD / "hyb/attempts.csv", candidates_path,
              SCRIPTS / "run_mali_full_features.py", SCRIPTS / "run_mali_batched.py",
              SCRIPTS / "freeze_strict_logical_panel.py", SCRIPTS / "run_strict_logical_qonductor.py",
              SCRIPTS / "run_qonductor_grouped_reproduction.py", SCRIPTS / "run_common_panel_hyb.py"]
    inputs.extend([SCRIPTS / "run_strict_logical_analytical.py",
        ROOT / "benchmark_v1/protocol/analytical_extension.json",
        ROOT / "benchmark_v1/protocol/mali_full_features.json",
        ROOT / "benchmark_v1/protocol/strict_method_completion.json",
        ROOT / "artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c133_snapshot_registry_v3.csv",
        CANONICAL, ROOT / "benchmark_v1/protocol/qonductor_native_features.json",
        SCRIPTS / "qonductor_native_adapter.py", SCRIPTS / "qasm3_hardware_wire_adapter.py"])
    manifest = {"status": "prepared", "rows": 4515, "source_counts": contract["source_counts"],
        "tiers": dict(Counter(r["logical_input_tier"] for r in data)), "support": support,
        "selected_id_sha256": idhash(r["canonical_observation_id"] for r in data),
        "backend_counts": dict(Counter(r["backend"] for r in data)),
        "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in inputs}, "graph_hashes": graphs,
        "outputs": {name: sha(OUT / name) for name in
                    ("panel.csv", "outer_splits.csv", "inner_splits.csv", "candidates.json")},
        "external_hashes": {str(QONDUCTOR_ARCHIVE):sha(QONDUCTOR_ARCHIVE),
                            str(ROOT.parent / "qcre/estimate.py"):sha(ROOT.parent / "qcre/estimate.py")},
        "fresh_fit": True, "old_oof_predictions_reused": False,
        "representation": "reconstruction_qualified_not_all_exact_historical_instances"}
    write_json(prior, manifest)
    return manifest


def load_context():
    manifest = json.loads((OUT / "manifest.json").read_text())
    for relative, expected in manifest["input_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"input_drift:{relative}")
    for path, expected in manifest["external_hashes"].items():
        if sha(path) != expected:
            raise RuntimeError(f"external_input_drift:{path}")
    for relative, expected in manifest["outputs"].items():
        if sha(OUT / relative) != expected:
            raise RuntimeError(f"panel_drift:{relative}")
    for relative, expected in manifest["graph_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"graph_drift:{relative}")
    panel = rows(OUT / "panel.csv")
    outer, inner = rows(OUT / "outer_splits.csv"), rows(OUT / "inner_splits.csv")
    from freeze_strict_logical_panel import validate_splits
    validate_splits(outer, inner)
    if len(panel) != 4515 or set(indexed(panel)) != set(indexed(outer)):
        raise RuntimeError("panel_split_identity_mismatch")
    ledger = {}
    for r in panel:
        ledger[r["canonical_observation_id"]] = {**r, "graph_artifact": r["graph_file"],
            "graph_artifact_sha256": r["graph_sha256"], "graph_node_count": r["node_count"],
            "graph_edge_count": r["edge_count"], "graph_status": "available", "global_status": "available",
            **{f"global_{i:02d}": r[f"g{i:02d}"] for i in range(51)}}
    return {"panel": panel, "ledger": ledger, "targets": {i: float(r["target_seconds"]) for i,r in ledger.items()},
        "feature_root": ROOT, "outer": indexed(outer), "inner": inner, "manifest": manifest,
        "pins": {"manifest": sha(OUT / "manifest.json"), "runner": sha(Path(__file__)), "contract": sha(CONTRACT)}}


def partitions(context, fold):
    test = sorted(i for i,r in context["outer"].items() if int(r["outer_fold"]) == fold)
    train = sorted(set(context["ledger"]) - set(test))
    assignments = indexed([r for r in context["inner"] if int(r["outer_fold"]) == fold])
    if set(assignments) != set(train):
        raise RuntimeError("inner_train_membership_mismatch")
    valid = sorted(i for i,r in assignments.items() if int(r["inner_fold"]) == 0)
    fit = sorted(set(train) - set(valid))
    return dict(train=train, fit=fit, validation=valid, test=test, assignments=assignments)


def cpu_regression():
    from sklearn.base import clone
    from threadpoolctl import threadpool_limits
    from run_strict_logical_qonductor import source_grids_and_models, fit_one_candidate
    context = load_context()
    native = indexed(rows(NATIVE), "canonical_row_id")
    ids = sorted(context["ledger"])
    spec = json.loads(CONTRACT.read_text())["regression"]
    matrix = np.array([[float(native[i][k]) for k in spec["features"]] for i in ids])
    y = np.array([context["targets"][i] for i in ids])
    if not np.isfinite(matrix).all():
        raise RuntimeError("regression_matrix_nonfinite")
    x = dict(matrix=matrix, id_to_index={i:j for j,i in enumerate(ids)})
    templates, _, _ = source_grids_and_models()
    candidates = json.loads((OUT / "candidates.json").read_text())["candidates"]
    folder = OUT / "regression"
    folder.mkdir(exist_ok=True)
    for fold in range(5):
        cell = folder / f"fold_{fold}.json"
        predpath = folder / f"fold_{fold}.csv"
        if cell.exists():
            receipt = json.loads(cell.read_text())
            if receipt["pins"] != context["pins"] or receipt["predictions_sha256"] != sha(predpath):
                raise RuntimeError("regression_resume_mismatch")
            continue
        parts = partitions(context, fold)
        splits = {v: (sorted(i for i in parts["train"] if int(parts["assignments"][i]["inner_fold"]) != v),
                      sorted(i for i in parts["train"] if int(parts["assignments"][i]["inner_fold"]) == v)) for v in range(4)}
        selections, predictions = {}, []
        for family in spec["families"]:
            score_path = folder / f"fold_{fold}_{family.lower().replace(' ', '_')}.json"
            if score_path.exists():
                saved = json.loads(score_path.read_text())
                if saved["pins"] != context["pins"]:
                    raise RuntimeError("candidate_resume_mismatch")
                scores = saved["scores"]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    scores = list(pool.map(lambda candidate: fit_one_candidate(
                        family, candidate, x, y, splits, templates), candidates[family]))
                write_json(score_path, dict(pins=context["pins"], scores=scores))
            good = [r for r in scores if r["status"] == "scored" and math.isfinite(float(r["mean_inner_r2"]))]
            if not good:
                raise RuntimeError(f"no_valid_regression_candidate:{family}:{fold}")
            best = max(good, key=lambda r: (float(r["mean_inner_r2"]), -int(r["candidate_index"])))
            estimator = clone(templates[family]).set_params(**json.loads(best["params_json"]))
            fitidx = [x["id_to_index"][i] for i in parts["train"]]
            testidx = [x["id_to_index"][i] for i in parts["test"]]
            with threadpool_limits(limits=1):
                estimator.fit(matrix[fitidx], y[fitidx])
                estimates = estimator.predict(matrix[testidx])
            if not np.isfinite(estimates).all():
                raise RuntimeError("outer_prediction_nonfinite")
            selections[family] = best
            method = "qonductor_" + family.lower().replace(" ", "_")
            predictions.extend(attempt(context, i, fold, method, float(value)) for i,value in zip(parts["test"], estimates))
            print(json.dumps(dict(stage="regression", fold=fold, family=family, status="complete")), flush=True)
        winner = max(spec["families"], key=lambda f: (float(selections[f]["mean_inner_r2"]), -spec["families"].index(f)))
        winner_id = "qonductor_" + winner.lower().replace(" ", "_")
        predictions.extend({**r, "method_id": "qonductor_budgeted_selector"}
                           for r in list(predictions) if r["method_id"] == winner_id)
        write_csv(predpath, predictions)
        write_json(cell, dict(status="complete", pins=context["pins"], selections=selections,
                             selected_family=winner, predictions_sha256=sha(predpath)))


def attempt(context, identity, fold, method, value=None, status="predicted", reason="", clock="predicted_observed_service_seconds"):
    r = context["ledger"][identity]
    return dict(canonical_observation_id=identity, outer_fold=fold, method_id=method,
        source_id=r["source_id"], backend=r["backend"], logical_input_tier=r["logical_input_tier"],
        group_id=r["group_id"], actual_seconds=r["target_seconds"], predicted_seconds="" if value is None else value,
        status=status, terminal_reason=reason, evaluation_target_clock="archived_observed_service_execution_time",
        method_output_clock=clock)


def cpu_analytical():
    from sklearn.linear_model import LinearRegression
    from threadpoolctl import threadpool_limits
    from run_common_panel_hyb import ridge_fold
    context = load_context()
    raw = defaultdict(dict)
    clocks = {}
    wanted = {"qiskit_estimate_duration_snapshot", "qiskit_shot_scaled_schedule_seconds",
              "scholten_unified_nominal_throughput"}
    for r in rows(OLD / "analytical/attempts.csv"):
        if r["variant"] == "raw" and r["base_method_id"] in wanted and r["canonical_observation_id"] in context["ledger"]:
            method, rid = r["base_method_id"], r["canonical_observation_id"]
            if float(r["actual_seconds"]) != context["targets"][rid]:
                raise RuntimeError("raw_analytical_target_join_mismatch")
            raw[method][rid] = r
            clocks[method] = r["method_output_clock"]
    for r in rows(OLD / "hyb/attempts.csv"):
        if r["method_id"] in {"hyb_nominal_r2_effective_cost_v1", "hyb_nominal_r2_shot_effective_cost_v1"} and r["canonical_observation_id"] in context["ledger"]:
            raw[r["method_id"]][r["canonical_observation_id"]] = r
            clocks[r["method_id"]] = r["method_output_clock"]
    # Run actual pinned QCRE, not the old custom scheduled-path proxy.
    raw_qcre, qcre_clock = qcre_raw(context)
    raw.update(raw_qcre); clocks.update(qcre_clock)
    results, fits = [], []
    for method, by_id in raw.items():
        if set(by_id) != set(context["ledger"]):
            raise RuntimeError(f"analytical_raw_denominator:{method}")
        for fold in range(5):
            parts = partitions(context, fold)
            eligible = [i for i in parts["train"] if by_id[i]["status"] == "predicted"]
            x = np.array([float(by_id[i]["predicted_seconds"]) for i in eligible])
            y = np.array([context["targets"][i] for i in eligible])
            models = {}
            for variant in ("raw", "affine", "log_affine"):
                model = None
                if variant != "raw" and len(x) >= 2:
                    with threadpool_limits(limits=1):
                        model = LinearRegression().fit((np.log1p(x) if variant == "log_affine" else x).reshape(-1,1),
                                                       np.log1p(y) if variant == "log_affine" else y)
                models[variant] = model
                fits.append(dict(method_id=method, variant=variant, outer_fold=fold,
                    fit_id_sha256=idhash(eligible), fit_rows=len(eligible),
                    slope="" if model is None else float(model.coef_[0]),
                    intercept="" if model is None else float(model.intercept_)))
            for identity in parts["test"]:
                r = by_id[identity]
                for variant, model in models.items():
                    value, status, reason = None, r["status"], r["terminal_reason"]
                    if status == "predicted":
                        value = float(r["predicted_seconds"])
                        if variant != "raw":
                            if model is None:
                                value, status, reason = None, "unavailable", "insufficient_outer_train_support"
                            else:
                                z = float(model.predict([[math.log1p(value) if variant == "log_affine" else value]])[0])
                                try:
                                    value = max(0., math.expm1(z) if variant == "log_affine" else z)
                                except OverflowError:
                                    value, status, reason = None, "overflow", "calibration_inverse_overflow"
                    if value is not None and not math.isfinite(value):
                        value, status, reason = None, "failed", "nonfinite_prediction"
                    results.append(attempt(context, identity, fold, method + "__" + variant, value, status, reason,
                        clocks[method] if variant == "raw" else "calibrated_observed_service_seconds"))
    components = indexed(rows(COMPONENTS), "canonical_row_id")
    inputs = dict(ids=set(context["ledger"]), target=context["ledger"], panel=context["ledger"],
        components=components, folds={i:int(r["outer_fold"]) for i,r in context["outer"].items()},
        groups={i:r["group_id"] for i,r in context["ledger"].items()}, inner_rows=context["inner"],
        parent_contract=json.loads((ROOT / "benchmark_v1/protocol/analytical_extension.json").read_text()))
    for fold in range(5):
        method = dict(method_id="hyb_nominal_r2_log_cost_ridge_v1", method_output_clock="calibrated_observed_service_seconds")
        with threadpool_limits(limits=1):
            values, receipts = ridge_fold(method, fold, inputs, components)
        results.extend({k:r[k] for k in results[0]} for r in values)
        # Ridge receipts use a different schema; retain their own file.
        write_json(OUT / "analytical" / f"hyb_ridge_fold_{fold}.json", dict(pins=context["pins"], fits=receipts))
    for identity in context["ledger"]:
        results.append(attempt(context, identity, int(context["outer"][identity]["outer_fold"]),
            "scholten_original_paper", status="unavailable", reason="kernel_n_and_template_D_not_established", clock="original_throughput_seconds"))
    write_csv(OUT / "analytical/attempts.csv", results)
    write_csv(OUT / "analytical/calibration_fits.csv", fits)
    write_json(OUT / "analytical/manifest.json", dict(status="complete", pins=context["pins"],
        output_hashes={name:sha(OUT / "analytical" / name) for name in ("attempts.csv", "calibration_fits.csv")}))


def qcre_raw(context):
    from qiskit import QuantumCircuit, qasm3
    from qiskit_ibm_runtime import fake_provider
    from run_strict_logical_analytical import QCRE_ROOT, QCRE_REVISION, QCRE_FILE_SHA256, qcre_custom_path
    if subprocess.check_output(["git", "-C", str(QCRE_ROOT), "rev-parse", "HEAD"], text=True).strip() != QCRE_REVISION or sha(QCRE_ROOT / "estimate.py") != QCRE_FILE_SHA256:
        raise RuntimeError("qcre_source_hash_mismatch")
    site = os.environ.get("STRICT_BQSKIT_SITE", str(BQSKIT_SITE))
    if site:
        sys.path.append(site)
    sys.path.insert(0, str(QCRE_ROOT))
    from estimate import estimate_runtime
    from bqskit.ext.qiskit import qiskit_to_bqskit
    from qonductor_native_adapter import load_pinned_qonductor_methods
    from qasm3_hardware_wire_adapter import restore_saved_qasm3_hardware_wires
    import zipfile
    native_contract = json.loads((ROOT / "benchmark_v1/protocol/qonductor_native_features.json").read_text())
    _,_,parse_submitted,_ = load_pinned_qonductor_methods(ROOT.parent / "Qonductor-SC25",native_contract["source_pin"])
    canonical = indexed(rows(CANONICAL),"canonical_row_id")
    registry = indexed(rows(ROOT / "artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c133_snapshot_registry_v3.csv"), "backend_canonical")
    reps = indexed(rows(REP / "representation_rows.csv"), "canonical_row_id")
    maps = {}
    for backend in sorted({r["backend"] for r in context["ledger"].values()}):
        name = backend.replace("ibmq_", "ibm_")
        entry = registry[name]
        for pathkey, hashkey in (("configuration_path","configuration_sha256"),("properties_path","properties_sha256")):
            if sha(entry[pathkey]) != entry[hashkey]:
                raise RuntimeError("qcre_snapshot_hash_mismatch")
        # Registry stores the installed class, avoiding guessed backend aliases.
        clsname = entry.get("fake_backend_class") or entry.get("backend_class") or "Fake" + name.removeprefix("ibm_").capitalize()
        clsname = clsname.split(".")[-1]
        device = getattr(fake_provider, clsname)()
        dmap = {}
        for opname in device.target.operation_names:
            op = device.target.operation_from_name(opname)
            if opname in {"measure", "reset", "delay"}:
                continue
            for loc, props in device.target[opname].items():
                if loc is not None and props is not None and props.duration is not None:
                    duration = float(props.duration)
                    if not math.isfinite(duration) or duration < 0:
                        raise RuntimeError("bad_duration")
                    dmap.setdefault(op.num_qubits, {}).setdefault(tuple(loc), {})[opname] = duration
        maps[backend] = dmap
    output, cache = defaultdict(dict), {}
    for identity, row in context["ledger"].items():
        rep = reps[identity]
        submitted = rep["lifecycle_stage"] == "submitted_physical"
        expected_hash = rep["source_qasm_sha256"] if submitted else rep["compiled_qasm3_sha256"]
        key = (row["backend"], expected_hash)
        if key not in cache:
            if submitted:
                with zipfile.ZipFile(QONDUCTOR_ARCHIVE) as archive:
                    source = archive.read(canonical[identity]["qasm_path_or_member"])
                if hashlib.sha256(source).hexdigest() != expected_hash:
                    raise RuntimeError("submitted_QASM_hash_mismatch")
                circuit = parse_submitted(source.decode())
            else:
                path = REP / rep["compiled_qasm3_file"]
                if sha(path) != expected_hash:
                    raise RuntimeError("compiled_hash_mismatch")
                source, _ = restore_saved_qasm3_hardware_wires(path.read_text(),int(rep["register_width"]))
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
                    if measured or item.clbits or getattr(item.operation, "blocks", None) is not None or name in {"reset","delay"}:
                        raise ValueError("unsupported_nonunitary_or_nonterminal_measurement")
                    unitary.append(item.operation, [unitary.qubits[circuit.find_bit(q).index] for q in item.qubits])
                bq = qiskit_to_bqskit(unitary)
                value = float(estimate_runtime(bq, maps[row["backend"]]))
                reference = qcre_custom_path(unitary, maps[row["backend"]])
                if not math.isfinite(value) or not math.isclose(value, reference, rel_tol=1e-10, abs_tol=1e-12):
                    raise RuntimeError("qcre_source_adapter_parity_failure")
                cache[key] = (value, "")
            except (ValueError, KeyError, NotImplementedError) as exc:
                cache[key] = (None, type(exc).__name__ + ":" + str(exc)[:200])
        value, reason = cache[key]
        for method, multiplier in (("qcre_source_estimator",1), ("qcre_source_estimator_shot_scaled",float(row["shots"]))):
            output[method][identity] = dict(status="predicted" if value is not None else "unavailable",
                predicted_seconds="" if value is None else value * multiplier, terminal_reason=reason)
    write_json(OUT / "analytical/qcre_source_receipt.json",dict(source_revision=QCRE_REVISION,
        source_sha256=sha(QCRE_ROOT / "estimate.py"),converter_path=__import__("bqskit.ext.qiskit",fromlist=["__file__"]).__file__,
        unique_compiled_contexts=len(cache),pins=context["pins"],same_unitary_custom_path_parity="PASS",
        terminal_measurements_omitted=True))
    return output, {"qcre_source_estimator":"scheduled_unitary_critical_path_seconds",
                    "qcre_source_estimator_shot_scaled":"shot_scaled_unitary_seconds"}


def neural_transform(context, parts):
    import run_mali_full_features as base
    return dict(global_transform=base.fit_transform_global(
        {i:[float(context["ledger"][i][f"g{j:02d}"]) for j in range(51)] for i in parts["fit"]}, parts["fit"]),
        node_transform=base.fit_node_transform([ROOT / context["ledger"][i]["graph_file"] for i in parts["fit"]], parts["fit"]))


def make_model(method, tfm, torch):
    import run_mali_full_features as base
    modelclass, modelsha = base.load_pinned_model()
    internal = base.METHODS[NEURAL.index(method)]
    model = base.make_model(modelclass, internal, len(tfm["global"]["retained_indices"]), checkpoint_layers=True).to("cuda:0")
    if method == NEURAL[0]:
        model.mask = model.mask.to("cuda:0")
    return model, internal, modelsha


def neural_cell(context, parts, tfm, method, fold, seed, torch):
    import run_mali_full_features as base
    import run_mali_batched as batching
    spec = json.loads(CONTRACT.read_text())["neural"]
    folder = OUT / "neural" / f"fold_{fold}" / method / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    ident = {**context["pins"], "method":method, "fold":fold, "seed":seed,
        "fit_ids":idhash(parts["fit"]), "validation_ids":idhash(parts["validation"]), "test_ids":idhash(parts["test"]),
        "transform":base.stable_hash(tfm), "source_model":base.load_pinned_model()[1],
        "torch":torch.__version__, "cuda":torch.version.cuda, "gpu":torch.cuda.get_device_name(0)}
    receipt = folder / "manifest.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["identity"] != ident or any(sha(folder / k) != h for k,h in saved["outputs"].items()):
            raise RuntimeError("neural_resume_completed_identity_mismatch")
        return saved
    base.seed_all(seed)
    model, internal, _ = make_model(method, tfm, torch)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=spec["optimizer"]["lr"], weight_decay=spec["optimizer"]["weight_decay"])
    cache = batching.ExampleCache(context, tfm, spec["cache_bytes"])
    start, best, best_epoch, state, elapsed_before = 0, float("inf"), -1, None, 0.
    checkpoint = folder / "checkpoint.pt"
    if checkpoint.exists():
        saved = base.load_saved_checkpoint(checkpoint, ident, torch)
        model.load_state_dict(saved["model_state"]); optimizer.load_state_dict(saved["optimizer_state"])
        base.restore_rng_state(saved["rng_state"])
        start, best, best_epoch, state, elapsed_before = saved["next_epoch"], saved["best_validation_mse"], saved["best_epoch"], saved["best_model_state"], saved["elapsed_seconds"]
    begin = time.monotonic()
    graph = method == NEURAL[0]
    fit_targets = {i:context["targets"][i] for i in parts["fit"]}
    for epoch in range(start, spec["epochs"]):
        tick = time.monotonic(); model.train()
        order = np.random.default_rng(int(base.stable_hash([fold,seed,epoch])[:16],16)).permutation(len(parts["fit"]))
        ids = [parts["fit"][int(j)] for j in order]
        batch = spec["effective_batch"]
        train_losses = [batching.loss_step(model, optimizer, ids[j:j+batch], internal,
            spec["node_budget"] if graph else 0, cache, fit_targets, torch) for j in range(0,len(ids),batch)]
        model.eval(); val_sum = 0.
        with torch.no_grad():
            for group in batching.physical_groups(parts["validation"], context["ledger"], spec["node_budget"] if graph else 0, graph):
                examples = [cache.get(i, internal) for i in group]
                payload = base._batch_to_device(examples,"cuda:0",graph)
                pred = base._prediction(model,payload)
                actual = torch.tensor([context["targets"][i] for i in group],dtype=torch.float32,device="cuda:0")
                val_sum += float(torch.nn.functional.mse_loss(pred,actual,reduction="sum").cpu())
                del examples,payload,pred,actual
        val = val_sum / len(parts["validation"])
        if not math.isfinite(val) or not np.isfinite(train_losses).all():
            raise RuntimeError("nonfinite_neural_training")
        if val < best:
            best,best_epoch,state = val,epoch,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.cuda.synchronize()
        elapsed = elapsed_before + time.monotonic()-begin
        base.save_checkpoint(checkpoint, dict(identity=ident,model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer_state=optimizer.state_dict(), rng_state=base.capture_rng_state(), next_epoch=epoch+1,
            best_validation_mse=best,best_epoch=best_epoch,best_model_state=state,elapsed_seconds=elapsed))
        write_json(folder / "progress.json", dict(status="training",epoch=epoch+1,epochs=spec["epochs"],
            epoch_seconds=time.monotonic()-tick,elapsed_seconds=elapsed,validation_mse=val,identity=ident,
            peak_cuda_bytes=torch.cuda.max_memory_allocated()))
        print(json.dumps(dict(stage="neural",method=method,fold=fold,seed=seed,epoch=epoch+1,epoch_seconds=time.monotonic()-tick)),flush=True)
    model.load_state_dict(state); model.eval()
    predictions = []
    with torch.no_grad():
        for identity in parts["test"]:
            payload = base._batch_to_device([cache.get(identity,internal)],"cuda:0",graph)
            value = float(base._prediction(model,payload).cpu().reshape(-1)[0])
            if not math.isfinite(value):
                raise RuntimeError("nonfinite_neural_prediction")
            predictions.append({**attempt(context,identity,fold,method,value), "seed":seed})
            del payload
    write_csv(folder / "predictions.csv",predictions)
    result = dict(status="complete",identity=ident,epochs=spec["epochs"],best_epoch=best_epoch,
        best_validation_mse=best,elapsed_seconds=elapsed,fit_rows=len(parts["fit"]),validation_rows=len(parts["validation"]),
        test_rows=len(parts["test"]),outputs={n:sha(folder / n) for n in ("checkpoint.pt","predictions.csv")})
    write_json(receipt,result)
    del model,optimizer,cache,state
    torch.cuda.empty_cache()
    return result


def neural_run():
    import torch
    import run_mali_full_features as base
    import run_mali_batched as batching
    from gpu_lease import acquire_gpu_lease
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_required_no_CPU_fallback")
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    context = load_context()
    lease = acquire_gpu_lease("qualified-logical-training")
    try:
        gpu = base.validate_gpu_budget(torch,budget_gib=12,reserve_gib=4)
        for fold in range(5):
            parts = partitions(context,fold)
            raw_tfm = neural_transform(context,parts)
            tfm = {"global": raw_tfm["global_transform"], "node": raw_tfm["node_transform"]}
            path = OUT / "neural" / f"fold_{fold}" / "transform.json"
            if path.exists() and json.loads(path.read_text()) != tfm:
                raise RuntimeError("transform_resume_mismatch")
            if not path.exists():
                write_json(path,tfm)
            if fold == 0:
                profile = OUT / "neural/cuda_profile.json"
                if not profile.exists():
                    small = sorted(parts["fit"],key=lambda i:int(context["ledger"][i]["node_count"]))[:2]
                    cache = batching.ExampleCache(context,tfm,8*1024**3)
                    parity = batching.gradient_gate(context,tfm,cache,small,torch)
                    if parity["status"] != "PASS":
                        raise RuntimeError("CUDA_gradient_Adam_parity_failure")
                    largest = max(parts["fit"],key=lambda i:int(context["ledger"][i]["node_count"]))
                    model,internal,_ = make_model(NEURAL[0],tfm,torch)
                    opt = torch.optim.Adam(model.parameters(),lr=.0005,weight_decay=.0001)
                    loss = batching.loss_step(model,opt,[largest],internal,250000,cache,{largest:0.},torch)
                    if not math.isfinite(loss):
                        raise RuntimeError("largest_graph_canary_nonfinite")
                    write_json(profile,dict(status="PASS",pins=context["pins"],gpu=gpu,torch=torch.__version__,
                        cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),parity=parity,largest_graph=largest,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated()))
                    del cache,model,opt; torch.cuda.empty_cache()
                elif json.loads(profile.read_text())["pins"] != context["pins"]:
                    raise RuntimeError("CUDA_profile_resume_mismatch")
            elif validate_neural_fold(context,0)["status"] != "PASS":
                raise RuntimeError("fold0_gate_failed")
            for method in NEURAL:
                for seed in SEEDS:
                    neural_cell(context,parts,tfm,method,fold,seed,torch)
            validate_neural_fold(context,fold)
    finally:
        lease.release()


def validate_neural_fold(context,fold):
    parts = partitions(context,fold)
    receipts = []
    for method in NEURAL:
        for seed in SEEDS:
            folder = OUT / "neural" / f"fold_{fold}" / method / f"seed_{seed}"
            manifest = json.loads((folder / "manifest.json").read_text())
            identity = manifest["identity"]
            if manifest["status"] != "complete" or manifest["epochs"] != 500 or not 0 <= manifest["best_epoch"] < 500:
                raise RuntimeError("neural_cell_not_complete")
            for key,value in context["pins"].items():
                if identity[key] != value:
                    raise RuntimeError("neural_cell_pin_mismatch")
            for key in ("fit","validation","test"):
                if identity[key+"_ids"] != idhash(parts[key]):
                    raise RuntimeError("neural_cell_partition_mismatch")
            if identity["seed"] != seed or identity["method"] != method or identity["fold"] != fold:
                raise RuntimeError("neural_cell_id_mismatch")
            for name,expected in manifest["outputs"].items():
                if sha(folder / name) != expected:
                    raise RuntimeError("neural_output_hash_mismatch")
            predictions = indexed(rows(folder / "predictions.csv"))
            if set(predictions) != set(parts["test"]):
                raise RuntimeError("neural_test_rows_mismatch")
            for i,r in predictions.items():
                if int(r["seed"]) != seed or not math.isfinite(float(r["predicted_seconds"])) or float(r["actual_seconds"]) != context["targets"][i]:
                    raise RuntimeError("neural_prediction_invalid")
            receipts.append(sha(folder / "manifest.json"))
    result = dict(status="PASS",fold=fold,cell_count=6,pins=context["pins"],cell_hashes=receipts)
    write_json(OUT / "neural" / f"fold_{fold}" / "validation.json", result)
    return result


def summarize():
    context = load_context()
    predictions = []
    for fold in range(5):
        predictions.extend(rows(OUT / "regression" / f"fold_{fold}.csv"))
        validate_neural_fold(context,fold)
        for method in NEURAL:
            seed_rows = [indexed(rows(OUT / "neural" / f"fold_{fold}" / method / f"seed_{seed}/predictions.csv")) for seed in SEEDS]
            for identity in partitions(context,fold)["test"]:
                predictions.append(attempt(context,identity,fold,method,float(np.median([float(r[identity]["predicted_seconds"]) for r in seed_rows]))))
    predictions.extend(rows(OUT / "analytical/attempts.csv"))
    methods = sorted({r["method_id"] for r in predictions})
    metric_values, strata = [], []
    for method in methods:
        values = [r for r in predictions if r["method_id"] == method]
        if len(values) != 4515 or len(indexed(values)) != 4515:
            raise RuntimeError("summary_method_denominator_mismatch")
        pooled = metrics(values)
        sources = [metrics([r for r in values if r["source_id"]==s]) for s in sorted({r["source_id"] for r in values})]
        pooled["source_balanced_mae_seconds"] = float(np.mean([r["mae_seconds"] for r in sources])) if all(r["mae_seconds"] != "" for r in sources) else ""
        metric_values.append({"method_id":method,**pooled})
        for field in ("source_id","backend","logical_input_tier"):
            for value in sorted({r[field] for r in values}):
                strata.append(dict(method_id=method,slice_type=field,slice_value=value,
                                   **metrics([r for r in values if r[field]==value])))
    paired = paired_metrics(predictions, context)
    write_csv(OUT / "summary/oof_predictions.csv",predictions)
    write_csv(OUT / "summary/method_metrics.csv",metric_values)
    write_csv(OUT / "summary/stratified_metrics.csv",strata)
    write_csv(OUT / "summary/paired_comparisons.csv",paired)
    write_json(OUT / "summary/manifest.json",dict(status="complete",pins=context["pins"],method_variants=len(methods),
        assigned_rows=4515, fresh_fit=True, output_hashes={name:sha(OUT / "summary" / name) for name in
            ("oof_predictions.csv","method_metrics.csv","stratified_metrics.csv","paired_comparisons.csv")}))


def metrics(values):
    good = [r for r in values if r["status"] == "predicted"]
    y = np.asarray([float(r["actual_seconds"]) for r in good])
    p = np.asarray([float(r["predicted_seconds"]) for r in good])
    error = np.abs(y-p)
    with np.errstate(over="ignore",invalid="ignore"):
        r2 = 1-np.square(y-p).sum()/np.square(y-y.mean()).sum() if len(y)>1 and np.ptp(y)>0 else float("nan")
        rmse = np.sqrt(np.square(y-p).mean()) if len(y) else float("nan")
    finite = lambda x: float(x) if math.isfinite(float(x)) else ""
    return dict(assigned_rows=len(values),predicted_rows=len(good),coverage=len(good)/len(values),
        mae_seconds=finite(error.mean()) if len(y) else "",medae_seconds=finite(np.median(error)) if len(y) else "",
        rmse_seconds=finite(rmse),r2_seconds=finite(r2),log1p_mae=finite(np.abs(np.log1p(y)-np.log1p(np.maximum(p,0))).mean()) if len(y) else "",
        p90_absolute_error_seconds=finite(np.quantile(error,.9)) if len(y) else "",
        p99_absolute_error_seconds=finite(np.quantile(error,.99)) if len(y) else "",
        max_absolute_error_seconds=finite(error.max()) if len(y) else "",negative_prediction_count=int((p<0).sum()),
        scored_ids_sha256=idhash(r["canonical_observation_id"] for r in good),
        squared_error_status="finite" if math.isfinite(float(rmse)) and math.isfinite(float(r2)) else "undefined_or_overflow")


def paired_metrics(predictions,context):
    by_method = defaultdict(dict)
    for row in predictions:
        if row["status"] == "predicted":
            by_method[row["method_id"]][row["canonical_observation_id"]] = row
    result = []
    for left,right in itertools.combinations(sorted(by_method),2):
        ids = sorted(set(by_method[left]) & set(by_method[right]))
        if not ids:
            continue
        grouped = defaultdict(list)
        for identity in ids:
            a,b = by_method[left][identity],by_method[right][identity]
            grouped[context["ledger"][identity]["group_id"]].append(
                abs(float(a["actual_seconds"])-float(a["predicted_seconds"]))-
                abs(float(b["actual_seconds"])-float(b["predicted_seconds"])))
        sums = np.array([sum(v) for _,v in sorted(grouped.items())])
        counts = np.array([len(v) for _,v in sorted(grouped.items())])
        rng = np.random.default_rng(42)
        samples = rng.integers(len(sums),size=(10000,len(sums)))
        estimates = sums[samples].sum(axis=1)/counts[samples].sum(axis=1)
        result.append(dict(left_method=left,right_method=right,n=len(ids),groups=len(sums),ids_sha256=idhash(ids),
            observed_mae_delta_seconds=float(sums.sum()/counts.sum()),bootstrap_mean_delta_seconds=float(estimates.mean()),
            bootstrap_ci_low_seconds=float(np.quantile(estimates,.025)),bootstrap_ci_high_seconds=float(np.quantile(estimates,.975)),
            bootstrap_seed=42,bootstrap_replicates=10000))
    return result


def supervise():
    context = load_context()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    logroot = OUT / "execution"
    logroot.mkdir(exist_ok=True)
    jobs = []
    for stage,python in (("regression",CPU_PYTHON),("analytical",CPU_PYTHON),("neural",GPU_PYTHON)):
        log = (logroot / f"{stage}.log").open("a",encoding="utf-8")
        child = subprocess.Popen([str(python),str(Path(__file__)),"--stage",stage],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((stage,child,log))
    write_json(logroot / "status.json",dict(status="running",pins=context["pins"],supervisor_pid=os.getpid(),
        workers={stage:child.pid for stage,child,_ in jobs},started_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()))
    try:
        while any(child.poll() is None for _,child,_ in jobs):
            for stage,child,_ in jobs:
                if child.poll() not in (None,0):
                    raise RuntimeError(f"worker_failed:{stage}:{child.returncode}")
            time.sleep(5)
        for stage,child,log in jobs:
            log.close()
            if child.returncode:
                raise RuntimeError(f"worker_failed:{stage}:{child.returncode}")
        summarize()
        write_json(logroot / "status.json",dict(status="complete",pins=context["pins"],supervisor_pid=os.getpid()))
    except BaseException as exc:
        for _,child,log in jobs:
            if child.poll() is None:
                child.terminate()
            log.close()
        write_json(logroot / "status.json",dict(status="failed",error=str(exc),pins=context["pins"]))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",required=True,choices=("freeze","preflight","regression","analytical","neural","aggregate","all"))
    args = parser.parse_args()
    if args.stage == "freeze":
        print(json.dumps(freeze(),indent=2))
    elif args.stage == "preflight":
        context=load_context(); print(json.dumps(dict(status="PASS",rows=len(context["ledger"]),pins=context["pins"])))
    else:
        {"regression":cpu_regression,"analytical":cpu_analytical,"neural":neural_run,"aggregate":summarize,"all":supervise}[args.stage]()


if __name__ == "__main__":
    main()
