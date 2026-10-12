#!/usr/bin/env python3
"""Matched frozen QPU estimator replay/timing. No fitting or quantum execution.

The new CPU server instruments the unmodified Phase 3 compatibility functions.
All writes are exclusive and confined to Phase 4. No automatic retry/resume.
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
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "benchmark_v1/scripts"
OUT = ROOT / "artifacts/am1_phase4/v1"
PHASE2 = ROOT / "artifacts/am1_phase2/v1"
PHASE3 = ROOT / "artifacts/am1_preprocessing_recovery/v1"
PANEL = ROOT / "artifacts/real_qpu/qualified_logical_panel"
CPU = Path("/home/server/Documents/.venv-qonductor/bin/python")
GPU = Path("/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python")
METHODS = ("mali_logical_graph", "mali_global_mlp")
SEEDS = (42, 1234, 31415)
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


def write_json(name, value):
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with (OUT / name).open("x") as f:
        f.write(data)


def write_md(name, text):
    with (OUT / name).open("x") as f:
        f.write(text.rstrip() + "\n")


def write_csv(name, data):
    if not data:
        raise ValueError("empty_required_table:" + name)
    with (OUT / name).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def verify(pins):
    for p, digest in pins.items():
        if sha(p) != digest:
            raise RuntimeError("integrity_mismatch:" + p)
    return len(pins)


def model_dir(fold, method, seed):
    return PANEL / f"neural/fold_{fold}/{method}/seed_{seed}"


def order(index, repetition):
    return METHODS if (index + repetition) % 2 == 0 else METHODS[::-1]


def passing(actual, expected):
    return abs(actual - expected) <= 1e-6 + 1e-5 * abs(expected)


def prepare():
    if OUT.exists():
        raise RuntimeError("refuse_existing_phase4_namespace")
    original = read(PHASE2 / "measurement_plan.json")
    recovery = read(PHASE3 / "acceptance_contract.json")
    final_recovery = read(PHASE3 / "provenance_manifest.json")
    # The Phase3 acceptance snapshot precedes the validated static-function
    # setup cache. Final code authority is its completed provenance receipt.
    # Never repin a mismatching source opportunistically to current bytes.
    verify(final_recovery["payload"])
    verify(final_recovery["code_pins"])
    pins = {**recovery["frozen_pins"], **final_recovery["code_pins"]}
    for p in PHASE3.iterdir():
        if p.is_file():
            pins[str(p)] = sha(p)
    for p in (Path(__file__), SCRIPTS / "analyze_am1_phase4.py", ROOT / "benchmark_v1/tests/test_am1_phase4.py", PHASE2 / "measurement_plan.json", PHASE2 / "selected_workloads.csv"):
        pins[str(p)] = sha(p)
    verify(pins)
    selected = [r for r in rows(PHASE2 / "selected_workloads.csv") if r["context"] == "qualified_qpu_4515" and r["raw_path"]]
    if len(selected) != 45 or len({r["id"] for r in selected}) != 45:
        raise RuntimeError("cohort_identity")
    if Counter(r["source"] for r in selected) != {"mali_real_qpu": 38, "qonductor_single_circuit_ibm": 7}:
        raise RuntimeError("source_composition")
    bridge = read(PHASE3 / "bridge_validation.json")["receipts"]
    selected_ids = {r["id"] for r in selected}
    for model in ("Graph", "MLP"):
        b = [r for r in bridge if r["context"] == "qualified_qpu_4515" and r["model"] == model]
        if {r["id"] for r in b} != selected_ids or not all(r["exact_model_ready_match"] for r in b):
            raise RuntimeError("phase3_input_gate")
    for r in selected:
        if sha(r["raw_path"]) != r["raw_sha256"] or sha(r["properties_path"]) != r["properties_sha256"]:
            raise RuntimeError("raw_context_integrity")
    env = dict(original["environment"], CUDA_VISIBLE_DEVICES="")
    package_probe = "import json,platform,importlib.metadata as m; print(json.dumps(dict(python=platform.python_version(),packages={k:m.version(k) for k in ('qiskit','numpy')})))"
    cpu_environment = json.loads(subprocess.check_output([str(CPU), "-B", "-c", package_probe], env={**os.environ, **env}, text=True, timeout=120))
    if cpu_environment["packages"] != {"qiskit": "2.5.2", "numpy": "2.2.6"}:
        raise RuntimeError("CPU_environment")
    gpu_probe = "import json,platform,importlib.metadata as m; print(json.dumps(dict(python=platform.python_version(),packages={k:m.version(k) for k in ('torch','torch-geometric','qiskit','numpy')})))"
    gpu_environment = json.loads(subprocess.check_output([str(GPU), "-B", "-c", gpu_probe], env={**os.environ, **env}, text=True, timeout=120))
    if gpu_environment["packages"] != {"torch": "2.7.1+cu128", "torch-geometric": "2.6.1", "qiskit": "0.44.2", "numpy": "1.26.4"}:
        raise RuntimeError("GPU_environment")
    limits = original["limits"]
    if shutil.disk_usage(ROOT).free < limits["disk_floor_bytes"] + limits["output_bytes"]:
        raise RuntimeError("disk_headroom")
    checkpoints = []
    for fold in range(5):
        for method in METHODS:
            for seed in SEEDS:
                folder = model_dir(fold, method, seed)
                receipt = read(folder / "manifest.json")
                checkpoints.append(dict(fold=fold, method=method, seed=seed, checkpoint=str(folder / "checkpoint.pt"), sha256=sha(folder / "checkpoint.pt"), identity=receipt["identity"]))
    plan = dict(authorization="user_attachment_7c20ec85-cab5-4ed0-9830-c7e7c8c1f57d", phase="AM1_Phase4", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                selected=selected, methods=list(METHODS), seeds=list(SEEDS), checkpoints=checkpoints, source_pins=pins,
                affinity=original["affinity"], environment=original["environment"], cpu_environment=cpu_environment, gpu_environment=gpu_environment,
                limits=limits, batch=1, repetitions=30, warmups=3, atol_seconds=1e-6, rtol=1e-5,
                aggregation="host median of three sequential original seed predictions, each original output already seconds",
                boundary="resident raw QASM bytes plus nominal properties and frozen transform to host scalar ensemble prediction",
                architecture="one resident CUDA coordinator/inference process holds 30 models (5 folds x 2 methods x 3 seeds); one resident CPU stdio preprocessing child; no feature cache",
                stages_cpu=["request_decode", "parse_QASM", "extract_51_globals", "decode_nominal_T1T2", "prune_and_compact_DAG", "frozen_transform", "array_serialization", "payload_JSON_encode"],
                stages_host=["request_serialization", "request_write_flush", "response_wait_inclusive", "response_JSON_decode", "typed_array_decode", "tensor_construction", "batch_and_H2D", "three_seed_inference_inverse_and_D2H", "host_median_aggregation"],
                nested_spans="CPU spans are nested inside host response_wait_inclusive; never add both; IPC/unattributed residual = write+wait minus sum child durations, using duration differences only, not cross-process timestamps",
                ordering="Phase2 frozen row order; per repetition Graph/MLP order alternates by (row index+rep)%2; 15 of each first-position per row",
                bootstrap=dict(replicates=2000, seed=20261011, unit="paired structural-group cluster, pooled row-weighted statistic; separate source strata"),
                constants=original["constants"]["A/outer_train_median"],
                precision="original float32 parameters/features, no autocast, no altered deterministic/TF32 settings",
                setup_excluded="imports, code extraction, model load, disk input acquisition/fold routing, process startup; separately recorded once, not B3",
                stop="any replay/parity/hash/placement mismatch; any request>30s, setup>120s, aggregate pipeline RSS/allocated or reserved VRAM>8GiB, disk<3GiB, new outputs>256MiB, wall>7200s, foreign compute workload",
                no_training_quantum_execution_refit_dependency_change=True, optional="1000 timer-pair control and 30 empty IPC echo controls; no additional B1 or shared pipeline measurement")
    OUT.mkdir(parents=True)
    write_json("predeclaration.json", plan)
    write_md("predeclaration.md", "# Matched QPU full B2 predeclaration\n\nFrozen before inference or timing. Machine-readable authority: predeclaration.json (SHA256 " + sha(OUT / "predeclaration.json") + ").\n\n" + json.dumps({k:v for k,v in plan.items() if k not in {"source_pins", "checkpoints", "selected"}}, indent=2) + "\n\nExact 45 IDs/raw hashes and 30 checkpoint identities are in the JSON. No feature caches, QPack replacements or test-result-based selection. Warm loaded estimators are standalone per-method requests. CPU transforms execute exactly once before transport.\n")
    write_md("checkpoint_integrity.md", f"# Frozen integrity before execution\n\nPASS: {len(pins)} pins verified. Thirty checkpoints, five transforms, predictions/labels/splits and original producers are pinned. The 45 IDs match Phase2 and Phase3 (38 Ma–Li,7 Qonductor). Nominal properties map logical wire i to snapshot wire i as in the frozen encoder. This is not exact historical logical recovery.\n")
    write_md("gate0_authority_resolution.md", "# Phase 3 source authority\n\nThe first Gate 0 preparation stopped before inference, timing or predeclaration creation. It incorrectly compared the current compatibility module against the initial Phase 3 acceptance snapshot (158efea6...), which predates the static-function setup cache. The completed Phase 3 provenance manifest pins af5575ca..., matching the present file. Its final bridge revalidation records 210 exact comparisons after that change.\n\nThe Phase 4 verifier now checks the completed Phase 3 payload and all four completed code pins, while retaining all 1,458 frozen input/model/producer pins from acceptance_contract.json. No historical pin, source or artifact was edited. This corrects a verifier authority lookup, not a preprocessing or scientific protocol change.\n")
    write_json("gate0.json", dict(status="PASS", pins=len(pins), selected=45, source_counts=dict(Counter(r["source"] for r in selected)), cpu_environment=cpu_environment, gpu_environment=gpu_environment,
                                  gpu_inventory=subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv"], text=True),
                                  compute_inventory=subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True),
                                  cpu_info=Path("/proc/cpuinfo").read_text().split("model name",1)[-1].split("\n",1)[0], disk_free_bytes=shutil.disk_usage(ROOT).free,
                                  meminfo=Path("/proc/meminfo").read_text(), CUDA_placement_smoke="pending before Gate1; no model forward before predeclaration"))
    print(json.dumps(dict(stage="prepare", status="PASS", ids=45, pins=len(pins))), flush=True)


def cpu_serve():
    import am1_preprocessing_compat as compat
    import mali_full_features as mf
    compat.require_environment()
    compat.pinned_extractor()
    print(json.dumps(dict(status="READY", versions=compat.require_environment(), pid=os.getpid())), flush=True)
    for line in sys.stdin:
        if line.strip() == '"echo"':
            print('{"status":"ECHO"}', flush=True)
            continue
        signal.alarm(30)
        try:
            durations = {}
            def stage(name, fn):
                start = time.perf_counter_ns()
                value = fn()
                durations[name] = time.perf_counter_ns() - start
                return value
            req = stage("request_decode", lambda: json.loads(line))
            raw, props = stage("raw_base64_decode", lambda: (base64.b64decode(req["raw_qasm_base64"], validate=True), base64.b64decode(req["properties_base64"], validate=True)))
            circuit = stage("parse_QASM", lambda: compat.parse_qpu(raw, req["source"]))
            globals_raw = stage("extract_51_globals", lambda: mf.extract_global_features(circuit, compat.pinned_extractor()))
            record = None
            if req["graph"]:
                cal = stage("decode_nominal_T1T2", lambda: compat.calibration(props))
                record = stage("prune_and_compact_DAG", lambda: mf.compact_graph_record(circuit, cal)[0])
            arrays = stage("frozen_transform", lambda: compat.encode_qpu(globals_raw, record, req["transform"]))
            packed = stage("array_serialization", lambda: compat.pack(arrays))
            payload = stage("payload_JSON_encode", lambda: json.dumps(packed, allow_nan=False, separators=(",", ":")))
            result = dict(status="PASS", payload=payload, stages_ns=durations, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            print(json.dumps(result, allow_nan=False, separators=(",", ":")), flush=True)
        except Exception as exc:
            print(json.dumps(dict(status="BLOCKED", reason=f"{type(exc).__name__}:{exc}")), flush=True)
        finally:
            signal.alarm(0)


def rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        return 0
    return 0


def worker(phase):
    execution_start = time.monotonic()
    plan = read(OUT / "predeclaration.json")
    verify(plan["source_pins"])
    if phase == "timing" and read(OUT / "gate1.json")["status"] != "PASS":
        raise RuntimeError("timing_requires_full_gate1")
    os.sched_setaffinity(0, plan["affinity"])
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("request_or_setup_deadline")))
    signal.alarm(120)
    setup_start = time.perf_counter_ns()
    import numpy as np
    import torch
    from torch_geometric.data import Data
    import run_mali_full_features as base
    import run_qualified_logical_benchmark as source
    import am1_preprocessing_compat as compat
    from recover_am1_preprocessing import array_hash
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_required_no_fallback")
    torch.cuda.set_per_process_memory_fraction(.48)
    torch.cuda.init()
    smoke = torch.ones(1, device="cuda:0") + 1
    torch.cuda.synchronize()
    assert float(smoke.cpu()[0]) == 2
    models, expected, transforms, identities = {}, {}, {}, []
    for fold in range(5):
        transforms[fold] = read(PANEL / f"neural/fold_{fold}/transform.json")
        for method in METHODS:
            for seed in SEEDS:
                folder = model_dir(fold, method, seed)
                state = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
                receipt = read(folder / "manifest.json")
                if state["identity"] != receipt["identity"] or receipt["identity"]["transform"] != stable(transforms[fold]):
                    raise RuntimeError("checkpoint_refit_transform_identity")
                model, _, _ = source.make_model(method, transforms[fold], torch)
                model.load_state_dict(state["best_model_state"])
                if {str(p.device) for p in model.parameters()} != {"cuda:0"}:
                    raise RuntimeError("model_placement")
                models[(fold, method, seed)] = model.eval()
                expected[(fold, method, seed)] = {r["canonical_observation_id"]: float(r["predicted_seconds"]) for r in rows(folder / "predictions.csv")}
                identities.append(dict(fold=fold, method=method, seed=seed, identity=receipt["identity"], checkpoint_sha256=sha(folder / "checkpoint.pt")))
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > plan["limits"]["ram_bytes"]:
                    raise RuntimeError("setup_RAM_cap")
                if torch.cuda.memory_reserved() > plan["limits"]["vram_bytes"]:
                    raise RuntimeError("setup_VRAM_cap")
    raw = {r["id"]: Path(r["raw_path"]).read_bytes() for r in plan["selected"]}
    props = {r["id"]: Path(r["properties_path"]).read_bytes() for r in plan["selected"]}
    bridge_hashes = {(r["id"], r["model"]):r for r in read(PHASE3 / "bridge_validation.json")["receipts"] if r["context"] == "qualified_qpu_4515"}
    cpu_env = {**os.environ, **plan["environment"], "CUDA_VISIBLE_DEVICES":""}
    child = subprocess.Popen([str(CPU), "-B", str(Path(__file__).resolve()), "--stage", "cpu-serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=cpu_env)
    stop_monitor = threading.Event()
    monitor = dict(peak_aggregate_rss_bytes=0, peak_host_rss_bytes=0, peak_child_rss_bytes=0, violation="")
    def sample():
        while not stop_monitor.is_set():
            host, cpu = rss(os.getpid()), rss(child.pid)
            monitor["peak_host_rss_bytes"] = max(monitor["peak_host_rss_bytes"], host)
            monitor["peak_child_rss_bytes"] = max(monitor["peak_child_rss_bytes"], cpu)
            monitor["peak_aggregate_rss_bytes"] = max(monitor["peak_aggregate_rss_bytes"], host + cpu)
            if host + cpu > plan["limits"]["ram_bytes"]:
                monitor["violation"] = "aggregate_RAM_cap"
                os.kill(os.getpid(), signal.SIGALRM)
                return
            stop_monitor.wait(.05)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    replay, timing_records, failures = [], [], []
    start = execution_start
    preceding_seconds = read(OUT / "replay_receipt.json")["resources"]["wall_seconds"] if phase == "timing" else 0
    environment = {}
    controls = {}
    try:
        ready = json.loads(child.stdout.readline())
        if ready["status"] != "READY":
            raise RuntimeError("CPU_server_startup")
        torch.cuda.synchronize()
        signal.alarm(0)
        environment = dict(python=platform.python_version(), packages={p:importlib.metadata.version(p) for p in ("torch", "torch-geometric", "qiskit", "numpy")}, cpu_environment=ready, cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0), host_pid=os.getpid(), child_pid=child.pid, affinity=sorted(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
            cuda_smoke="PASS", setup_ns=time.perf_counter_ns()-setup_start, model_count=len(models), precision="float32", checkpoint_identities=identities)
        # Preserve CUDA high-water marks from setup as well as measured requests.
        def guard():
            if monitor["violation"]:
                raise RuntimeError(monitor["violation"])
            if torch.cuda.memory_reserved() > plan["limits"]["vram_bytes"] or torch.cuda.memory_allocated() > plan["limits"]["vram_bytes"]:
                raise RuntimeError("VRAM_cap")
            if shutil.disk_usage(ROOT).free < plan["limits"]["disk_floor_bytes"]:
                raise RuntimeError("disk_floor")
            if time.monotonic()-start+preceding_seconds > plan["limits"]["wall_seconds"]:
                raise RuntimeError("wall_cap")
            if sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) > plan["limits"]["output_bytes"]:
                raise RuntimeError("artifact_cap")
        def pipeline(r, method, check=False):
            signal.alarm(30)
            host_stages = {}
            def stage(name, fn, gpu=False):
                if gpu:
                    torch.cuda.synchronize()
                begin = time.perf_counter_ns()
                result = fn()
                if gpu:
                    torch.cuda.synchronize()
                host_stages[name] = time.perf_counter_ns()-begin
                return result
            graph = method == METHODS[0]
            torch.cuda.synchronize()
            begin = time.perf_counter_ns()
            request = stage("request_serialization", lambda: json.dumps(dict(source=r["source"], graph=graph, transform=transforms[int(r["fold"])],
                raw_qasm_base64=base64.b64encode(raw[r["id"]]).decode(), properties_base64=base64.b64encode(props[r["id"]]).decode()), allow_nan=False, separators=(",", ":")) + "\n")
            def send():
                child.stdin.write(request)
                child.stdin.flush()
            stage("request_write_flush", send)
            response_line = stage("response_wait_inclusive", child.stdout.readline)
            response = stage("response_JSON_decode", lambda: json.loads(response_line))
            if response["status"] != "PASS":
                raise RuntimeError("CPU_preprocessing:" + str(response))
            arrays = stage("typed_array_decode", lambda: compat.unpack(json.loads(response["payload"])))
            if check:
                expected_input = bridge_hashes[(r["id"], "Graph" if graph else "MLP")]
                if array_hash(arrays) != expected_input["expected_hash"]:
                    raise RuntimeError("model_ready_array_digest")
            data = stage("tensor_construction", lambda: Data(**{k:torch.as_tensor(v) for k,v in arrays.items()}))
            payload = stage("batch_and_H2D", lambda: base._batch_to_device([data], "cuda:0", graph), True)
            def inference():
                return [float(base._prediction(models[(int(r["fold"]), method, s)], payload).detach().cpu().reshape(-1)[0]) for s in SEEDS]
            values = stage("three_seed_inference_inverse_and_D2H", inference, True)
            estimate = stage("host_median_aggregation", lambda: statistics.median(values))
            total_ns = time.perf_counter_ns()-begin
            signal.alarm(0)
            if not np.isfinite(values).all():
                raise RuntimeError("nonfinite_prediction")
            guard()
            item = dict(id=r["id"], method=method, total_B2_ns=total_ns, predicted_seconds=estimate,
                request_bytes=len(request.encode()), response_bytes=len(response_line.encode()), child_peak_rss_bytes=response["peak_rss_bytes"],
                **{f"host_{k}_ns":v for k,v in host_stages.items()}, **{f"cpu_{k}_ns":v for k,v in response["stages_ns"].items()})
            for name in plan["stages_cpu"] + ["raw_base64_decode"]:
                item.setdefault(f"cpu_{name}_ns", 0)
            item["IPC_unattributed_residual_ns"] = host_stages["request_write_flush"] + host_stages["response_wait_inclusive"] - sum(response["stages_ns"].values())
            item["coordinator_unattributed_ns"] = total_ns - sum(host_stages.values())
            if min(item["IPC_unattributed_residual_ns"], item["coordinator_unattributed_ns"]) < 0:
                raise RuntimeError("invalid_nested_timing_partition")
            return values, estimate, item, array_hash(arrays) if check else None
        with torch.no_grad():
            if phase == "replay":
                for i,r in enumerate(plan["selected"]):
                    for method in order(i, 0):
                        values, estimate, _, digest = pipeline(r, method, True)
                        saved_values = [expected[(int(r["fold"]), method, s)][r["id"]] for s in SEEDS]
                        for seed, actual, saved in zip((*SEEDS, "median3"), (*values, estimate), (*saved_values, statistics.median(saved_values))):
                            replay.append(dict(id=r["id"], source=r["source"], fold=r["fold"], method=method, seed=seed, status="PASS" if passing(actual, saved) else "FAIL", replayed_seconds=actual, saved_seconds=saved,
                                abs_diff_seconds=abs(actual-saved), atol_seconds=1e-6, rtol=1e-5, model_ready_digest=digest, input_dtype_shape_exact=True, units="seconds"))
                        if any(x["status"] != "PASS" for x in replay[-4:]):
                            raise RuntimeError("frozen_prediction_replay:" + r["id"] + ":" + method)
                    print(json.dumps(dict(stage="replay", completed_ids=i+1, total=45)), flush=True)
            else:
                # Instrumentation controls are explicitly not quantum workloads.
                timer=[]
                for _ in range(1000):
                    b=time.perf_counter_ns(); timer.append(time.perf_counter_ns()-b)
                echo=[]
                for _ in range(30):
                    b=time.perf_counter_ns(); child.stdin.write('"echo"\n'); child.stdin.flush()
                    assert json.loads(child.stdout.readline())["status"] == "ECHO"
                    echo.append(time.perf_counter_ns()-b)
                controls=dict(timer_pair_ns=timer, empty_stdio_echo_ns=echo, no_overhead_subtraction=True)
                for i,r in enumerate(plan["selected"]):
                    for rep in range(-plan["warmups"], plan["repetitions"]):
                        for position,method in enumerate(order(i, rep)):
                            values, estimate, item, _ = pipeline(r, method)
                            saved = statistics.median(expected[(int(r["fold"]), method, s)][r["id"]] for s in SEEDS)
                            if not passing(estimate,saved):
                                raise RuntimeError("timed_output_drift:" + r["id"])
                            if rep >= 0:
                                item.update(repetition=rep, order_position=position, source=r["source"], fold=r["fold"], group=r["group"], status="PASS")
                                timing_records.append(item)
                    print(json.dumps(dict(stage="timing", completed_ids=i+1, requests=len(timing_records), total=2700)), flush=True)
    except Exception as exc:
        failures.append(dict(reason=f"{type(exc).__name__}:{exc}", stage=phase, completed_replay=len(replay), completed_timings=len(timing_records)))
    finally:
        signal.alarm(0)
        child.stdin.close()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.terminate(); child.wait(timeout=10)
        stderr=child.stderr.read()
        stop_monitor.set(); thread.join(timeout=1)
        resource_receipt=dict(**monitor, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(), gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(), host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            wall_seconds=time.monotonic()-start, child_exit_code=child.returncode, child_stderr=stderr, subprocess_reaped=child.poll() is not None)
    verified=verify(plan["source_pins"])
    write_json(phase + "_receipt.json", dict(status="PASS" if not failures else "BLOCKED", failures=failures, environment=environment, resources=resource_receipt, controls=controls, frozen_pins_verified=verified, predeclaration_sha256=sha(OUT / "predeclaration.json")))
    if phase == "replay":
        if replay:
            write_csv("replay_validation.csv", replay)
        accepted = not failures and len(replay)==360 and all(r["status"]=="PASS" for r in replay)
        write_json("gate1.json", dict(status="PASS" if accepted else "BLOCKED", Graph_ids=45 if accepted else len({r["id"] for r in replay if r["method"]==METHODS[0] and r["status"]=="PASS"}), MLP_ids=45 if accepted else len({r["id"] for r in replay if r["method"]==METHODS[1] and r["status"]=="PASS"}), rows=len(replay), failures=failures))
    elif timing_records:
        write_csv("raw_b2_timings.csv", timing_records)
    if failures:
        raise RuntimeError(str(failures))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--stage", choices=["prepare","cpu-serve","replay","timing"], required=True); args=p.parse_args()
    if args.stage == "prepare": prepare()
    elif args.stage == "cpu-serve": cpu_serve()
    else: worker(args.stage)


if __name__ == "__main__":
    main()
