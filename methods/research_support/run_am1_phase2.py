#!/usr/bin/env python3
"""AM1 Phase 2: frozen replay and serial, raw-QASM estimator timing only.

Never calls training, simulator, transpilation, or producing runner mains.
Prepare precedes replay; context-wide input/prediction parity precedes timing.
All generated data stay in the new Phase 2 namespace. No historical writes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/am1_phase2/v1"
sys.path.insert(0, str(ROOT / "benchmark_v1/scripts"))
import run_am1_overhead_pilot as old

SEEDS = (42, 1234, 31415)
CONTEXTS = ("qualified_qpu_4515", "dense_fp32", "dense_fp64")
REPS, WARMUPS = 30, 3
METHODS = {
    "qualified_qpu_4515": ("mali_logical_graph", "mali_global_mlp"),
    "dense_fp32": ("mali_style_dense_graph", "matched_metadata_mlp"),
    "dense_fp64": ("mali_style_dense_graph", "matched_metadata_mlp"),
}


def write_json(name, value):
    old.check_storage = storage
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    storage(len(data))
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def write_csv(name, data, fields=None):
    storage(2**20)
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def write_md(name, value):
    storage(len(value.encode()))
    with (OUT / name).open("x") as handle:
        handle.write(value.rstrip() + "\n")


def storage(extra=0):
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) if OUT.exists() else 0
    if size + extra > old.STORAGE or shutil.disk_usage(ROOT).free < old.FLOOR + extra:
        raise RuntimeError("storage_cap_or_disk_floor")


def guard(torch=None):
    storage()
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > old.LIMIT:
        raise RuntimeError("RAM_cap")
    if torch is not None and torch.cuda.memory_reserved() > old.LIMIT:
        raise RuntimeError("VRAM_cap")


def prepare():
    if OUT.exists():
        raise RuntimeError("refuse_existing_phase2_namespace")
    plan = old.effective_plan()
    old.verify(plan["source_pins"])
    pins = dict(plan["source_pins"])
    paths = [Path(__file__), old.OUT / "selected_workloads.csv",
             ROOT / "artifacts/am1_postpilot_audit/v1/provenance_manifest.json"]
    panel = {r["canonical_observation_id"]: r for r in old.rows(old.P / "panel.csv")}
    canonical_path = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
    canonical = {r["canonical_row_id"]: r for r in old.rows(canonical_path)}
    candidate_path = ROOT / "artifacts/real_qpu/logical_recovery/candidate_recipe_manifest.json"
    candidates = {r["qasm_sha256"]: r for r in old.read(candidate_path)["records"]}
    registry_path = ROOT / "artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c133_snapshot_registry_v3.csv"
    snapshots = {r["snapshot_id"]: r for r in old.rows(registry_path)}
    sim_path = ROOT / "artifacts/benchmark_v1/sim_common_q16_manifest_20260927/sim_common_q16_manifest.csv"
    sim = {}
    for r in sorted(old.rows(sim_path), key=lambda r: r["basename"]):
        sim.setdefault(r["qasm_sha256"], r)
    paths.extend([canonical_path, candidate_path, registry_path, sim_path])
    selected = []
    for r in old.rows(old.OUT / "selected_workloads.csv"):
        if r["context"] not in CONTEXTS:
            continue
        item = dict(r, raw_path="", raw_sha256="", properties_path="", properties_sha256="",
                    b2_eligibility="candidate_requires_replay", exclusion_reason="")
        if r["context"] == CONTEXTS[0]:
            p = panel[r["id"]]
            if r["source"] == "mali_real_qpu":
                path = ROOT.parent / "Quantum-Execution-Time-Prediction/data/quantum_circuits" / canonical[r["id"]]["qasm_path_or_member"]
            elif r["source"] == "qonductor_single_circuit_ibm":
                path = candidate_path.parent / candidates[p["source_qasm_sha256"]]["path"]
            else:
                # The producing code generated a representative QAOA circuit;
                # no serialized logical QASM at the declared source locations.
                path = None
                item.update(b2_eligibility="unavailable", exclusion_reason="no_saved_raw_QPack_recipe_QASM; no_circuit_generation_authorized")
            if path is not None:
                if old.sha(path) != p["source_qasm_sha256"]:
                    raise RuntimeError("logical_QASM_hash:" + r["id"])
                item.update(raw_path=str(path), raw_sha256=p["source_qasm_sha256"])
                paths.append(path)
            snap = snapshots[p["snapshot_id"]]
            prop = Path(snap["properties_path"])
            if old.sha(prop) != p["properties_sha256"]:
                raise RuntimeError("snapshot_properties_hash")
            item.update(properties_path=str(prop), properties_sha256=p["properties_sha256"])
            paths.append(prop)
        else:
            path = Path(sim[r["id"]]["source_path"])
            if not path.is_absolute():
                path = ROOT.parent / path
            if old.sha(path) != r["id"]:
                raise RuntimeError("dense_QASM_hash")
            item.update(raw_path=str(path), raw_sha256=r["id"])
            paths.append(path)
        selected.append(item)
    if len(selected) != 110 or Counter(r["context"] for r in selected) != Counter({CONTEXTS[0]:50, CONTEXTS[1]:30, CONTEXTS[2]:30}):
        raise RuntimeError("cohort_identity")
    producer_paths = [ROOT / "benchmark_v1/scripts" / n for n in (
        "run_qualified_logical_benchmark.py", "run_cudaq_dense_classical_predictors_v1.py",
        "mali_full_features.py", "run_mali_full_features.py", "run_cudaq_dense_neural_predictors_v1.py",
        "materialize_cudaq_dense_ir_features_v1.py", "cudaq_qiskit_adapter.py")]
    paths.extend(producer_paths)
    scan_roots = [old.P / "regression"]
    scan_roots += [old.D / p / "classical" for p in ("fp32", "fp64")]
    scan_roots += [ROOT / "work/benchmark_recovery/r9_mps/fixed_chi16_recovered_v1",
                   ROOT / "work/benchmark_recovery/r9_mps/family_residual_recovered_v1",
                   old.M / "evaluation_001",
                   ROOT / "work/repo_finalization/publish_candidate/artifacts/benchmark_v3/simulator/azizov_common_core_adaptation"]
    scan = []
    extensions = {".pt", ".pth", ".pkl", ".pickle", ".joblib", ".onnx", ".safetensors", ".npz", ".bin"}
    for root in scan_roots:
        files = sorted(p for p in root.rglob("*") if p.is_file())
        candidates_found = [str(p) for p in files if p.suffix in extensions]
        scan.append(dict(root=str(root), exists=root.exists(), file_count=len(files), serialized_candidates=candidates_found))
        # Pin each producing manifest and receipt inspected, not raw training caches.
        paths.extend(p for p in files if p.suffix == ".json")
    inventory = old.rows(old.FEAS.parent / "estimator_inventory.csv")
    for row in inventory:
        row["phase2_recovery"] = ("existing_checkpoint_or_sufficient_parameters" if row["executability"] == "READY" else "no_new_recovery; see prior inventory and scoped_scan.json") if "executability" in row else "see_original_status_and_scoped_scan"
        row["phase2_b2_scope"] = "selected_context_requires_raw_input_replay" if row.get("method_id", row.get("method", "")) in {m for pair in METHODS.values() for m in pair} else "NOT_EXECUTED; retained parameters do not establish complete_B2"
    for path in paths:
        pins[str(path)] = old.sha(path)
    OUT.mkdir(parents=True)
    write_csv("selected_workloads.csv", selected)
    variants = [dict(context=c, method=m, seeds="42;1234;31415", aggregation="median_of_three_sequential_predictions",
                     boundary="in_memory_raw_QASM_and_metadata_to_host_prediction", gate="context_wide_input_and_seed_replay")
                for c in CONTEXTS for m in METHODS[c]]
    variants += [dict(context=c, method="outer_train_median", seeds="native", aggregation="none",
                      boundary="resident_frozen_fold_statistic_to_host_scalar; QASM_not_required", gate="saved_scalar_identity") for c in CONTEXTS]
    write_csv("selected_variants.csv", variants)
    write_csv("fitted_state_recovery_inventory.csv", inventory)
    write_json("scoped_scan.json", scan)
    frozen = dict(phase2a_verdict="CONDITIONAL_GO", authorization="user_attachment_0a543903-8d42-4eae-9c79-b3be0c0045e8",
                  scope="raw_QASM_Graph_MLP_native_median; no_complete_classical_frontier",
                  created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), source_pins=pins,
                  selected_workloads_sha256=old.sha(OUT / "selected_workloads.csv"),
                  selected_variants_sha256=old.sha(OUT / "selected_variants.csv"),
                  repetitions=REPS, full_pipeline_warmups_per_query=WARMUPS, batch=1, seeds=list(SEEDS),
                  environment=plan["env"], affinity=plan["affinity"], limits=plan["limits"],
                  constants=plan["constants"], atol=1e-6, rtol=1e-5,
                  raw_input_gate="global_51_1e-12; sparse_node_arrays_exact; dense_globals_and_DAG_exact; seed_predictions_atol_rtol",
                  failure_rule="failed_raw_parity_blocks_entire_context_method; no_selection_by_accuracy_or_latency; preserve_unavailable_rows",
                  omitted_stage="dense canonical_QASM_serialization/digest is identity bookkeeping, not model-required input; exact frozen RCCX lowering function reused; compare consumed globals+DAG",
                  blocked_competition="selector_RF_XGB_and_Qonductor_regressors_have_no_verified_fitted_states; no_refit")
    write_json("measurement_plan.json", frozen)
    write_md("phase2a_feasibility.md", """# Phase 2A — CONDITIONAL GO

Một phạm vi hữu ích được chọn: chi phí complete raw-input B2 của frozen Graph
so với frozen MLP trên cùng request IDs và context, cùng reference median
native. Không trả lời complete competitive Pareto frontier. Sáu Qonductor
regressors/selector, dense RF/XGB/Ridge và các routes thiếu fitted states không
được dựng lại từ seeds/hyperparameters. Scoped scan và producing serialization
code xác nhận chỉ có predictions, fit settings và selections tại những roots
đã chỉ định. Existing constants, affine coefficients, Maestro knots và neural
checkpoints không phải fitted states mới tìm lại được.

Giữ selection AM1 cũ: QPU 50, FP32 30, FP64 30; selection không dựa labels hay
timing. QPU QPack rows thiếu serialized raw recipe QASM tại source locations;
không generate circuit để lấp chỗ trống. Giữ chúng trong attempt denominator.
Các eligible requests phải qua raw feature/graph và per-seed replay gates trước
timing. Không lấy cached feature preparation thay complete B2.

Graph–MLP so sánh pipeline có thông tin/representation riêng, không equal-input
architecture contest. Missing strong classical comparator chặn kết luận
Graph/MLP là estimator tối ưu; không làm vô hiệu câu hỏi hẹp về preprocessing.
""")
    write_md("pipeline_input_contracts.md", """# Native input contracts

QPU: in-memory saved logical QASM bytes và raw snapshot properties bytes.
Parse QASM → original 51 globals; Graph thêm nominal T1/T2 decoding, pruning,
compact DAG và 178-wide nodes → frozen outer-fold transforms → H2D → 3 seeds
→ host median. MLP không cần DAG/backend properties: không bắt trả Graph cost.
Không transpile: compiled circuits không là native input của frozen models.
Qonductor input vẫn qualified recipe, không exact historical logical recovery.

Dense: in-memory source QASM bytes và fixed shots=32 → QASM parse → exact
RCCX lowering bằng original `_lower_circuit` → seven globals; Graph thêm DAG
→ original train-only transforms/tensor encoding → H2D → 3 seeds → expm1,
zero clipping, host median. Qiskit 0.44.2 không có qasm2.dumps; pipeline không
gọi serialization-only phần normalize wrapper (unused by model). So exact
consumed globals/DAG against original frozen features; raw source hash pinned.
Không compiler, CUDA-Q, simulator hoặc fresh transpilation được gọi.

Median: frozen fold scalar is sufficient native input; circuit parsing không
cần thiết. Native CPU dispatch đo riêng, không CUDA synchronization wrapper.

Offline/excluded B2: source retrieval/disk I/O, model selection/fold routing,
training/calibration, imports/checkpoint load, metadata acquisition. Setup
separate process/bundle observation; không standalone model cold-start claim.
DAG/feature caches and prebuilt device tensors never enter timed raw pipeline.
Maestro/native analytical B2 and joint-MPS remain unmeasured in this bounded
comparison; saved coefficients alone are not a full raw-pipeline qualification.
""")
    write_md("predeclared_measurement_plan.md", """# Frozen measurement plan

Machine-readable authority: measurement_plan.json, selected_variants.csv,
selected_workloads.csv, written before replay/timing. Source pins checked before
and after. One serial GPU worker, Torch threads=2/interop=1, native threads=1,
fixed prior pilot affinity. Batch one, frozen seeds 42/1234/31415; no fitting.

All candidate raw requests replay before ANY timing; gate per context/method
requires every eligible request to match consumed input and each seed. Three
full-pipeline warmups per request/method; 30 repeats. Rotate Graph/MLP order
by repeat. CUDA sync before/after full B2 and GPU stages; host perf_counter_ns.
Direct total independent of stage sum; instrumented stage sums include only
completed stages, residual dispatcher time not silently allocated to a stage.
Report per-query median/p95, across-query summaries and paired query/group
bootstrap (2000 replicates, fixed analysis seed=20261011, no repeat pseudo-N).
Memory is process/bundle high-water plus stage allocated/reserved peaks, not
isolated incremental model memory. All outliers retained.

8GiB RAM/VRAM; 256MiB output; disk floor3GiB; 30s request; 120s setup;
7200s aggregate Phase2B wall. No budget extension. Per-request timeout logged
with ID/stage; gate failure blocks timing rather than retry or change inputs.
No simulator timings, planning, QPU, training, queues, historical rewrites,
commit/push. Classical states missing: request new scoped amendment if needed,
not automatic reconstruction/refit. No standalone B3 distribution is promised.
""")
    write_md("environment_manifest.md", "# Environment before execution\n\n" + json.dumps(dict(
        executable=str(old.GPU_PY), platform=platform.platform(), affinity=plan["affinity"],
        cpu_info=plan["cpu_info"], inventory=subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv"], text=True),
        disk_free_bytes=shutil.disk_usage(ROOT).free, threads=2, native_threads=1,
        precursor_software="GPU: Python3.10/Torch2.7.1+cu128/Qiskit0.44.2/NumPy1.26.4; worker records actual versions",
        background="desktop GPU/RustDesk activity not stopped; no exclusive hardware or operational break-even claim"), indent=2))
    print(json.dumps(dict(stage="prepare", verdict="CONDITIONAL_GO", observations=len(selected), eligible=sum(bool(r["raw_path"]) for r in selected))), flush=True)


def worker(context, fold, phase):
    setup_start = time.perf_counter_ns()
    plan = old.read(OUT / "measurement_plan.json")
    os.sched_setaffinity(0, plan["affinity"])
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("request_30s_cap")))
    signal.alarm(120)
    import numpy as np
    import torch
    from torch_geometric.data import Data, Batch
    from qiskit import QuantumCircuit, qasm2
    import mali_full_features as mf
    import run_mali_full_features as base
    import materialize_cudaq_dense_ir_features_v1 as features
    import cudaq_qiskit_adapter as adapter
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_required")
    torch.cuda.set_per_process_memory_fraction(.48)
    torch.cuda.init()
    subset = [r for r in old.rows(OUT / "selected_workloads.csv") if r["context"] == context and int(r["fold"]) == fold and r["raw_path"]]
    models, expected, raw, properties, histories = {}, {}, {}, {}, []
    methods = METHODS[context]
    if context == CONTEXTS[0]:
        import run_qualified_logical_benchmark as source
        panel = {r["canonical_observation_id"]: r for r in old.rows(old.P / "panel.csv")}
        tfm = old.read(old.P / "neural" / f"fold_{fold}/transform.json")
        extractor, extractor_pins = mf.load_pinned_global_extractor()
        for method in methods:
            for seed in SEEDS:
                folder = old.P / "neural" / f"fold_{fold}" / method / f"seed_{seed}"
                state = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
                receipt = old.read(folder / "manifest.json")
                if state["identity"] != receipt["identity"] or receipt["identity"]["transform"] != old.stable(tfm):
                    raise RuntimeError("QPU_state_identity")
                model, _, _ = source.make_model(method, tfm, torch)
                model.load_state_dict(state["best_model_state"])
                models[(method, seed)] = model.eval()
                expected[(method, seed)] = {r["canonical_observation_id"]: float(r["predicted_seconds"]) for r in old.rows(folder / "predictions.csv")}
        constant = plan["constants"]["A/outer_train_median"][str(fold)]
    else:
        import run_cudaq_dense_neural_predictors_v1 as source
        precision = context.split("_")[-1]
        folder = old.D / precision / "neural" / f"fold_{fold}"
        fin = old.read(folder / "fold_inputs.json")
        contracts = fin["feature_contracts"]
        frozen_features = source.load_feature_records(ROOT / "work/benchmark_recovery/r3_dense/ir_features_v1/dense_features_by_hash.jsonl")
        seed_rows = old.rows(folder / "seed_predictions.csv")
        for method in methods:
            for seed in SEEDS:
                state = torch.load(folder / method / "checkpoints" / f"seed_{seed}.pt", map_location="cpu", weights_only=False)
                receipt = old.read(folder / method / f"seed_{seed}/complete.json")
                if state["identity"] != receipt["identity"]:
                    raise RuntimeError("dense_state_identity")
                model = source._make_model(method, len(contracts[methods[0]]["gate_vocabulary"]) + 9).to("cuda:0")
                model.load_state_dict(state["model_state"])
                models[(method, seed)] = model.eval()
                expected[(method, seed)] = {r["qasm_sha256"]: float(r["prediction_seconds"]) for r in seed_rows if r["method_id"] == method and int(r["seed"]) == seed}
        constant = plan["constants"][f"{precision}/outer_train_median"][str(fold)]
    for r in subset:
        raw[r["id"]] = Path(r["raw_path"]).read_bytes()
        if hashlib.sha256(raw[r["id"]]).hexdigest() != r["raw_sha256"]:
            raise RuntimeError("raw_input_hash")
        if r["properties_path"]:
            properties[r["id"]] = Path(r["properties_path"]).read_bytes()
    torch.cuda.synchronize()
    signal.alarm(0)
    setup_ns = time.perf_counter_ns() - setup_start
    environment = dict(python=platform.python_version(), executable=sys.executable,
        packages={p: importlib.metadata.version(p) for p in ("torch", "torch-geometric", "numpy", "qiskit")},
        cuda_build=torch.version.cuda, device=torch.cuda.get_device_name(0), affinity=sorted(os.sched_getaffinity(0)), pid=os.getpid(), setup_ns=setup_ns)
    torch.cuda.reset_peak_memory_stats()
    replay, timings = [], []
    latest_stage = "request_dispatch"

    def pipeline(method, rid, measured=False, parity_check=False):
        nonlocal latest_stage
        stages = []
        def stage(name, fn, gpu=False):
            nonlocal latest_stage
            latest_stage = name
            if gpu:
                torch.cuda.synchronize()
            start = time.perf_counter_ns()
            value = fn()
            if gpu:
                torch.cuda.synchronize()
            end = time.perf_counter_ns()
            if measured:
                stages.append(dict(stage=name, start_ns=start, end_ns=end, elapsed_ns=end-start,
                    cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved()))
            return value
        circuit = stage("parse_QASM", lambda: mf.parse_qasm(raw[rid])[0] if context == CONTEXTS[0] and rid.startswith("mali_") else qasm2.loads(raw[rid].decode(), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS))
        graph = method == methods[0]
        if context == CONTEXTS[0]:
            globals_raw = stage("extract_51_globals", lambda: mf.extract_global_features(circuit, extractor))
            if parity_check and not np.allclose(globals_raw, [float(panel[rid][f"g{i:02d}"]) for i in range(51)], rtol=0, atol=1e-12):
                raise RuntimeError("QPU_global_input_mismatch")
            record = None
            if graph:
                def calibration():
                    props = json.loads(properties[rid])
                    result = {}
                    scales = {"s":1e6, "ms":1e3, "us":1., "ns":1e-3}
                    for index, entries in enumerate(props["qubits"]):
                        v = {p["name"]: p for p in entries if p["name"] in {"T1", "T2"}}
                        result[index] = tuple(float(v[k]["value"]) * scales[v[k]["unit"].lower().replace("μ", "u").replace("µ", "u")] for k in ("T1", "T2"))
                    return result
                cal = stage("decode_nominal_T1T2", calibration)
                record = stage("prune_and_compact_DAG", lambda: mf.compact_graph_record(circuit, cal)[0])
                if parity_check:
                    with np.load(ROOT / panel[rid]["graph_file"], allow_pickle=False) as saved:
                        if set(record) != set(saved.files) or any(not np.array_equal(record[k], saved[k]) for k in record):
                            raise RuntimeError("QPU_sparse_graph_input_mismatch")
            def encode():
                gf = torch.as_tensor(base.transform_global(globals_raw, tfm["global"]), dtype=torch.float32).reshape(1, -1)
                if not graph:
                    return Data(global_features=gf)
                x = mf.expand_dense_x(record)
                mean, std = (np.asarray(tfm["node"][k], dtype=np.float32) for k in ("mean", "std"))
                variable = np.asarray(tfm["node"]["variable_mask"], dtype=bool)
                x[:, variable] = (x[:, variable] - mean[variable]) / std[variable]
                x[:, ~variable] = 0
                return Data(x=torch.as_tensor(x), edge_index=torch.as_tensor(record["edge_index"].astype(np.int64), dtype=torch.long), global_features=gf)
            data = stage("frozen_transform_and_tensor_encode", encode)
            payload = stage("batch_and_H2D", lambda: base._batch_to_device([data], "cuda:0", graph), True)
        else:
            # Original mathematical lowering reused, no version-dependent serialization.
            normalized = stage("exact_RCCX_normalization", lambda: adapter._lower_circuit(circuit)[0])
            feat = dict(global_features=stage("extract_seven_globals", lambda: features.global_features(normalized)))
            if graph:
                feat["dag"] = stage("build_operation_DAG", lambda: features.graph_record(normalized))
            if parity_check:
                frozen = frozen_features[rid]
                for key in feat:
                    if feat[key] != frozen[key]:
                        raise RuntimeError("dense_consumed_input_mismatch:" + key)
            contract = contracts[method]
            def encode():
                if graph:
                    return source.record_to_graph({"feature":feat}, contract["gate_vocabulary"], contract["global_transform_values"], include_target=False)
                return Data(global_features=torch.as_tensor(source.normalized_globals(feat, contract["global_transform_values"])).reshape(1,7))
            data = stage("frozen_transform_and_tensor_encode", encode)
            payload = stage("batch_and_H2D", lambda: Batch.from_data_list([data]).to("cuda:0"), True)
        def inference():
            values = []
            for seed in SEEDS:
                output = base._prediction(models[(method, seed)], payload) if context == CONTEXTS[0] else torch.expm1(models[(method, seed)](payload)).clamp_min(0)
                values.append(float(output.detach().cpu().reshape(-1)[0]))
            return values
        values = stage("three_seed_inference_inverse_and_D2H", inference, True)
        estimate = stage("host_median_aggregation", lambda: statistics.median(values))
        if parity_check:
            for seed, actual in zip(SEEDS, values):
                item = old.parity(method, rid, seed, actual, expected[(method, seed)][rid])
                replay.append(item)
                if item["status"] != "PASS":
                    raise RuntimeError("prediction_replay_mismatch")
            replay.append(old.parity(method, rid, "median3", estimate, statistics.median(expected[(method, s)][rid] for s in SEEDS)))
        return estimate, stages

    with torch.no_grad():
        if phase == "replay":
            for r in subset:
                for method in methods:
                    guard(torch)
                    signal.alarm(30)
                    try:
                        pipeline(method, r["id"], parity_check=True)
                        histories.append(dict(id=r["id"], method=method, status="PASS", raw_input_parity=True, reason=""))
                    except Exception as exc:
                        histories.append(dict(id=r["id"], method=method, status="FAIL", raw_input_parity=False, stage=latest_stage, reason=f"{type(exc).__name__}:{exc}"))
                    finally:
                        signal.alarm(0)
            scalar_expected = {}
            if context == CONTEXTS[0]:
                scalar_expected = {r["canonical_observation_id"]:float(r["predicted_seconds"]) for r in old.rows(ROOT / "artifacts/benchmark_deep_dive/tables/qpu_baseline_predictions_v1.csv") if r["method_id"] == "outer_train_median" and int(r["outer_fold"]) == fold}
            else:
                scalar_expected = {r["qasm_sha256"]:float(r["prediction_seconds"]) for r in old.rows(old.D / precision / "classical" / f"fold_{fold}/predictions.csv") if r["method_id"] == "outer_train_median"}
            for r in subset:
                replay.append(old.parity("outer_train_median", r["id"], "native", float(constant), scalar_expected[r["id"]]))
        else:
            gate = old.read(OUT / "replay_gate.json")
            active = [m for m in methods if gate[context][m]]
            for r in subset:
                for m in active:
                    for _ in range(WARMUPS):
                        signal.alarm(30)
                        pipeline(m, r["id"])
                        signal.alarm(0)
                for repeat in range(REPS):
                    for m in active[repeat % len(active):] + active[:repeat % len(active)] if active else []:
                        guard(torch)
                        signal.alarm(30)
                        torch.cuda.synchronize()
                        start = time.perf_counter_ns()
                        try:
                            prediction, stages = pipeline(m, r["id"], measured=True)
                            torch.cuda.synchronize()
                            end = time.perf_counter_ns()
                            for s in stages:
                                timings.append(dict(context=context, fold=fold, method=m, id=r["id"], repeat=repeat, boundary="B2", status="ok", reason="", prediction_seconds=prediction, **s))
                            timings.append(dict(context=context, fold=fold, method=m, id=r["id"], repeat=repeat, boundary="B2", status="ok", reason="", stage="complete_B2", prediction_seconds=prediction, start_ns=start, end_ns=end, elapsed_ns=end-start,
                                                cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved()))
                        except Exception as exc:
                            end = time.perf_counter_ns()
                            timings.append(dict(context=context, fold=fold, method=m, id=r["id"], repeat=repeat, boundary="B2", status="failed", reason=f"{type(exc).__name__}:{exc};stage={latest_stage}", stage="complete_B2", prediction_seconds=None, start_ns=start, end_ns=end, elapsed_ns=end-start,
                                                cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved()))
                            write_json(f"workers/{context}_fold{fold}_timing_partial.json", dict(environment=environment, timings=timings,
                                reason=f"{type(exc).__name__}:{exc}", failed_id=r["id"], failed_method=m, stage=latest_stage))
                            raise
                        finally:
                            signal.alarm(0)
                    if gate[context]["outer_train_median"]:
                        # No GPU synchronization or parsing: scalar native input needs neither.
                        start = time.perf_counter_ns()
                        value = float(constant)
                        end = time.perf_counter_ns()
                        timings.append(dict(context=context, fold=fold, method="outer_train_median", id=r["id"], repeat=repeat, boundary="native_scalar", status="ok", reason="", stage="complete_native_scalar", prediction_seconds=value, start_ns=start, end_ns=end, elapsed_ns=end-start,
                                            cuda_allocated_bytes=0, cuda_reserved_bytes=0))
    guard(torch)
    write_json(f"workers/{context}_fold{fold}_{phase}.json", dict(environment=environment, histories=histories, replay=replay, timings=timings,
        memory=dict(rss_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(), cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved(), attribution="shared six-model bundle; absolute, not marginal per-boundary")))
    print(json.dumps(dict(context=context, fold=fold, phase=phase, raw_input_failures=sum(r["status"] != "PASS" for r in histories), samples=len(timings))), flush=True)


def execute():
    plan = old.read(OUT / "measurement_plan.json")
    old.verify(plan["source_pins"])
    for n in ("selected_workloads", "selected_variants"):
        if old.sha(OUT / f"{n}.csv") != plan[f"{n}_sha256"]:
            raise RuntimeError("frozen_selection_drift")
    begin = time.monotonic()
    results = []
    def launch(c, f, phase):
        if time.monotonic() - begin >= 7080:
            raise RuntimeError("wall_budget_stop")
        command = [str(old.GPU_PY), "-B", str(Path(__file__)), "--stage", "worker", "--context", c, "--fold", str(f), "--phase", phase]
        env = {**os.environ, **plan["environment"], "PYTHONDONTWRITEBYTECODE":"1"}
        start = time.monotonic()
        child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        maxrss, reason = 0, ""
        while child.poll() is None:
            try:
                status = Path(f"/proc/{child.pid}/status").read_text()
                rss = int(next((x.split()[1] for x in status.splitlines() if x.startswith("VmRSS:")), 0))*1024
                maxrss = max(maxrss, rss)
                storage()
                if rss > old.LIMIT:
                    reason = "RAM_cap"
                if time.monotonic() - begin >= 7200:
                    reason = "wall_cap"
                if reason:
                    child.terminate()
                    break
            except FileNotFoundError:
                pass
            time.sleep(.2)
        try:
            stdout, stderr = child.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, stderr = child.communicate()
        result = dict(context=c, fold=f, phase=phase, command=command, pid=child.pid, exit=child.returncode, status="complete" if child.returncode == 0 else "failed", reason=reason,
                      elapsed_seconds=time.monotonic()-start, max_sampled_rss_bytes=maxrss, stdout=stdout, stderr=stderr)
        results.append(result)
        write_json(f"executions/{c}_fold{f}_{phase}.json", result)
        print(stdout.strip() or json.dumps(result), flush=True)
        return child.returncode == 0
    for c in CONTEXTS:
        for f in range(5):
            launch(c, f, "replay")
    gate = {}
    selected = old.rows(OUT / "selected_workloads.csv")
    for c in CONTEXTS:
        gate[c] = {}
        for m in (*METHODS[c], "outer_train_median"):
            expected_n = sum(r["context"] == c and bool(r["raw_path"]) for r in selected)
            evidence = []
            for f in range(5):
                path = OUT / f"workers/{c}_fold{f}_replay.json"
                if not path.exists():
                    continue
                report = old.read(path)
                evidence.extend([r for r in (report["replay"] if m == "outer_train_median" else report["histories"]) if r["method"] == m])
            gate[c][m] = len(evidence) == expected_n and expected_n > 0 and all(r["status"] == "PASS" for r in evidence)
    write_json("replay_gate.json", gate)
    for c in CONTEXTS:
        if any(gate[c].values()):
            for f in range(5):
                if not launch(c, f, "timing"):
                    write_json("execution_stop.json", dict(reason="timing_worker_failure", context=c, fold=f, elapsed_seconds=time.monotonic()-begin))
                    return
    old.verify(plan["source_pins"])
    write_json("execution_complete.json", dict(status="complete", phase2b_wall_seconds=time.monotonic()-begin, source_pins_match=True, executions=len(results), gate=gate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "execute", "worker"), required=True)
    parser.add_argument("--context", choices=CONTEXTS)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--phase", choices=("replay", "timing"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "execute":
        execute()
    else:
        worker(args.context, args.fold, args.phase)


if __name__ == "__main__":
    main()
