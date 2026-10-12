#!/usr/bin/env python3
"""Authorized frozen dense estimator replay and warm raw-input B2. No fitting.

Historical producers are imported for pure preprocessing/model definitions only.
All new writes are exclusive and confined to this amendment. No auto-resume.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

from run_am1_phase4 import ROOT, SCRIPTS, CPU, GPU, PHASE2, PHASE3, sha, read, rows, verify, rss, passing

OUT = ROOT / "artifacts/am1_phase5_dense_b2/v1"
DENSE = ROOT / "work/benchmark_recovery/r8_dense_predictors/attempt_002"
METHODS = ("mali_style_dense_graph", "matched_metadata_mlp")
SEEDS = (42, 1234, 31415)
CONTEXTS = ("fp32", "fp64")
CPU_STAGES = ("request_decode", "raw_base64_decode", "parse_QASM", "normalize_circuit",
              "extract_7_globals", "construct_DAG", "frozen_transform",
              "array_serialization", "payload_JSON_encode")


def write_json(name, value):
    with (OUT / name).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def write_md(name, text):
    with (OUT / name).open("x") as f:
        f.write(text.rstrip() + "\n")


def write_csv(name, data):
    import csv
    if not data:
        raise ValueError("empty_required_table:" + name)
    with (OUT / name).open("x", newline="") as f:
        w = csv.DictWriter(f, list(data[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(data)


def folder(precision, fold):
    return DENSE / precision / f"neural/fold_{fold}"


def order(index, repetition):
    return METHODS if (index + repetition) % 2 == 0 else METHODS[::-1]


def count(value):
    from decimal import Decimal
    x = Decimal(str(value))
    if not x.is_finite() or x < 0 or x != x.to_integral_value():
        raise ValueError("invalid_integral_count:" + str(value))
    return int(x)


def compute_inventory(allowed_pid=None):
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True, timeout=10)
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, name, _ = [v.strip() for v in line.split(",", 2)]
        if pid != str(allowed_pid) and name != "/usr/share/rustdesk/rustdesk":
            raise RuntimeError("foreign_GPU_compute:" + line)
    return output


def prepare():
    if OUT.exists():
        raise RuntimeError("refuse_existing_phase5_namespace")
    recovered = read(PHASE3 / "provenance_manifest.json")
    verify(recovered["payload"])
    verify(recovered["code_pins"])
    p4 = ROOT / "artifacts/am1_phase4/v1/provenance_manifest.json"
    p4_receipt = read(p4)
    verify(p4_receipt["payload"])
    pins = {**read(PHASE3 / "acceptance_contract.json")["frozen_pins"], **recovered["code_pins"], **recovered["payload"], **p4_receipt["payload"]}
    for p in (p4, PHASE3 / "provenance_manifest.json", ROOT / "AGENTS.md",
              SCRIPTS / "run_am1_phase4.py", SCRIPTS / "analyze_am1_phase4.py",
              Path(__file__), SCRIPTS / "analyze_am1_phase5_dense_b2.py",
              ROOT / "benchmark_v1/tests/test_am1_phase5_dense_b2.py",
              PHASE2 / "measurement_plan.json", PHASE2 / "selected_workloads.csv"):
        pins[str(p)] = sha(p)
    for p in DENSE.rglob("*"):
        if p.is_file() and p.suffix in {".json", ".csv", ".pt"}:
            pins[str(p)] = sha(p)
    selected = [r for r in rows(PHASE2 / "selected_workloads.csv") if r["context"] in {"dense_fp32", "dense_fp64"}]
    bridge = read(PHASE3 / "bridge_validation.json")["receipts"]
    for precision in CONTEXTS:
        cohort = [r for r in selected if r["context"] == "dense_" + precision]
        ids = {r["id"] for r in cohort}
        if len(cohort) != 30 or len(ids) != 30 or Counter(r["fold"] for r in cohort) != {str(f): 6 for f in range(5)}:
            raise RuntimeError("cohort_or_fold_routing:" + precision)
        for model in ("Graph", "MLP"):
            evidence = [r for r in bridge if r["context"] == "dense_" + precision and r["model"] == model]
            if {r["id"] for r in evidence} != ids or not all(r["exact_model_ready_match"] for r in evidence):
                raise RuntimeError("Phase3_model_ready_gate")
        for r in cohort:
            if not r["raw_path"] or sha(r["raw_path"]) != r["id"] or r["raw_sha256"] != r["id"]:
                raise RuntimeError("raw_identity")
            for k in ("width", "depth", "gates", "nodes", "edges"):
                count(r[k])
    original = read(PHASE2 / "measurement_plan.json")
    checkpoints = []
    for precision in CONTEXTS:
        for fold in range(5):
            inputs = read(folder(precision, fold) / "fold_inputs.json")
            for method in METHODS:
                for seed in SEEDS:
                    checkpoint = folder(precision, fold) / method / "checkpoints" / f"seed_{seed}.pt"
                    receipt = read(folder(precision, fold) / method / f"seed_{seed}/complete.json")
                    if receipt["checkpoint_sha256"] != sha(checkpoint) or receipt["identity"] != inputs["identity"] or receipt["seed"] != seed or receipt["method_id"] != method:
                        raise RuntimeError("checkpoint_identity")
                    pred = folder(precision, fold) / method / f"seed_{seed}/predictions.csv"
                    if sha(pred) != receipt["predictions_sha256"]:
                        raise RuntimeError("seed_prediction_identity")
                    checkpoints.append(dict(precision=precision, fold=fold, method=method, seed=seed, path=str(checkpoint), sha256=sha(checkpoint), identity=receipt["identity"]))
    env = {**os.environ, **original["environment"], "CUDA_VISIBLE_DEVICES": ""}
    def packages(exe, names):
        code = f"import json,platform,importlib.metadata as m; print(json.dumps(dict(python=platform.python_version(),packages={{k:m.version(k) for k in {names!r}}})))"
        return json.loads(subprocess.check_output([str(exe), "-B", "-c", code], env=env, text=True, timeout=120))
    cpu = packages(CPU, ("qiskit", "numpy"))
    gpu = packages(GPU, ("torch", "torch-geometric", "qiskit", "numpy"))
    if cpu["packages"] != {"qiskit": "2.5.2", "numpy": "2.2.6"} or gpu["packages"] != {"torch": "2.7.1+cu128", "torch-geometric": "2.6.1", "qiskit": "0.44.2", "numpy": "1.26.4"}:
        raise RuntimeError("environment_identity")
    inventory = compute_inventory()
    limits = original["limits"]
    if shutil.disk_usage(ROOT).free < limits["disk_floor_bytes"] + limits["output_bytes"]:
        raise RuntimeError("disk_headroom")
    verify(pins)
    plan = dict(authorization="user_attachment_485bfa96-70bc-41fa-b711-f1113cee4c8c", phase="AM1_Phase5_Dense_B2",
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), selected=selected,
                methods=list(METHODS), seeds=list(SEEDS), checkpoints=checkpoints, source_pins=pins,
                cpu_environment=cpu, gpu_environment=gpu, environment=original["environment"], affinity=original["affinity"], limits=limits,
                batch=1, repetitions=30, warmups=3, primary_timed_requests=3600, atol_seconds=1e-6, rtol=1e-5,
                aggregation="host median3; each CUDA seed predicts log1p seconds, original expm1 then clamp_min(0), D2H",
                boundary="resident raw QASM bytes plus frozen context/transform to host scalar ensemble prediction at coordinator",
                architecture="one resident CUDA coordinator and one historical CPU typed-array stdio child; 30 models/precision resident; no feature cache",
                actual_model_precision="float32 in BOTH contexts; FP32/FP64 denotes historical simulator precision, not estimator arithmetic",
                stages_cpu=list(CPU_STAGES),
                stages_host=["request_serialization", "request_write_flush", "response_wait_inclusive", "response_JSON_decode", "typed_array_decode", "tensor_construction", "batch_and_H2D", "three_seed_inference_inverse_and_D2H", "host_median_aggregation"],
                MLP_work="parse, original RCCX normalization, seven globals, frozen transform; omit DAG construction only, not normalization",
                nested_spans="CPU durations nested in host wait; residual=host write+wait minus child durations, never cross-process timestamps; not pure IPC",
                ordering="FP32 then FP64; frozen circuit order; alternating Graph/MLP by (row_index+rep)%2; 15/15 first positions",
                bootstrap=dict(replicates=2000, seed=20261011, unit="paired existing QASM-hash structural groups within each precision; no repetition resampling"),
                accuracy="frozen median3 predictions, separate exact30 timed subset and full150 OOF/context; saved train-only median reference; no model selection",
                constants={p: original["constants"][p + "/outer_train_median"] for p in CONTEXTS},
                scaling="descriptive Spearman and Pearson versus existing width/depth/gates/nodes/edges; report range/family confounding; no asymptotic law",
                historical_execution="ID/context/seconds-matched R5 targets only, labeled historical cross-environment simulator runtime; no operational break-even",
                setup_excluded="imports, process start, resident model load, input disk retrieval and fold routing; recorded separately, not cold B3",
                stop="any digest/parity/hash/placement drift, 30s request,120s setup,8GiB aggregate RSS/VRAM,7200s all replay+timing,256MiB outputs,3GiB diskfloor or foreign GPU workload; no retry",
                optional_controls="1000 timer-pair and30 empty stdio echo samples, no subtraction",
                no_training_refit_simulator_execution_new_circuits_dependency_change=True)
    OUT.mkdir(parents=True)
    write_json("predeclaration.json", plan)
    write_md("predeclaration.md", "# Dense matched full-B2 predeclaration\n\nFrozen before CUDA inference or timing. SHA256 `" + sha(OUT / "predeclaration.json") + "`.\n\n" + json.dumps({k:v for k,v in plan.items() if k not in {"selected", "checkpoints", "source_pins"}}, indent=2) + "\n\nExact60 context requests and60 checkpoints are in the JSON. No circuit or seed may be selected by outcomes. Stage medians are not additive.\n")
    write_json("gate0.json", dict(status="PASS", contexts={p:dict(status="PASS", ids=30) for p in CONTEXTS}, pins=len(pins), cpu_environment=cpu, gpu_environment=gpu, compute_inventory=inventory,
        gpu_inventory=subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv"], text=True),
        cpu_info=Path("/proc/cpuinfo").read_text().split("model name",1)[-1].split("\n",1)[0], meminfo=Path("/proc/meminfo").read_text(), disk_free_bytes=shutil.disk_usage(ROOT).free))
    print(json.dumps(dict(stage="prepare", status="PASS", pins=len(pins), contexts=2, ids_per_context=30)), flush=True)


def preprocess_request(req, stage):
    import am1_preprocessing_compat as compat
    from cudaq_qiskit_adapter import _lower_circuit
    from materialize_cudaq_dense_ir_features_v1 import global_features, graph_record
    raw = stage("raw_base64_decode", lambda: base64.b64decode(req["raw_qasm_base64"], validate=True))
    circuit = stage("parse_QASM", lambda: compat.parse_dense(raw))
    normalized, _ = stage("normalize_circuit", lambda: _lower_circuit(circuit))
    feature = {"global_features": stage("extract_7_globals", lambda: global_features(normalized, shots=32))}
    if req["graph"]:
        feature["dag"] = stage("construct_DAG", lambda: graph_record(normalized))
    return stage("frozen_transform", lambda: compat.encode_dense(feature, req["contract"], graph=req["graph"]))


def cpu_serve():
    import am1_preprocessing_compat as compat
    from cudaq_qiskit_adapter import _lower_circuit
    from materialize_cudaq_dense_ir_features_v1 import global_features, graph_record
    versions = compat.require_environment()
    print(json.dumps(dict(status="READY", versions=versions, pid=os.getpid())), flush=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("CPU_request_deadline")))
    for line in sys.stdin:
        if line.strip() == '"echo"':
            print('{"status":"ECHO"}', flush=True)
            continue
        signal.alarm(30)
        try:
            durations = {}
            def stage(name, fn):
                begin = time.perf_counter_ns(); result = fn()
                durations[name] = time.perf_counter_ns() - begin
                return result
            req = stage("request_decode", lambda: json.loads(line))
            arrays = preprocess_request(req, stage)
            packed = stage("array_serialization", lambda: compat.pack(arrays))
            payload = stage("payload_JSON_encode", lambda: json.dumps(packed, separators=(",", ":"), allow_nan=False))
            print(json.dumps(dict(status="PASS", payload=payload, stages_ns=durations,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024), separators=(",", ":"), allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps(dict(status="BLOCKED", reason=f"{type(exc).__name__}:{exc}")), flush=True)
        finally:
            signal.alarm(0)


def worker(phase, precision):
    start = time.monotonic()
    plan = read(OUT / "predeclaration.json")
    verify(plan["source_pins"])
    prefix = phase + "_" + precision
    if (OUT / (prefix + "_receipt.json")).exists():
        raise RuntimeError("refuse_reexecution")
    if phase == "timing" and read(OUT / f"gate1_{precision}.json")["status"] != "PASS":
        raise RuntimeError("timing_requires_context_wide_replay")
    prior = sum(read(p)["resources"]["wall_seconds"] for p in OUT.glob("*_receipt.json") if p.name.startswith(("replay_", "timing_")))
    os.sched_setaffinity(0, plan["affinity"])
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("setup_or_request_deadline")))
    signal.alarm(120)
    setup_start = time.perf_counter_ns()
    import numpy as np
    import torch
    from torch_geometric.data import Data, Batch
    import run_cudaq_dense_neural_predictors_v1 as dense
    import am1_preprocessing_compat as compat
    from recover_am1_preprocessing import array_hash
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_required_no_CPU_fallback")
    compute_inventory()
    torch.cuda.set_per_process_memory_fraction(.48)
    smoke = torch.ones(1, device="cuda:0") + 1
    torch.cuda.synchronize()
    assert float(smoke.cpu()[0]) == 2
    models, contracts, expected, identities = {}, {}, {}, []
    for fold in range(5):
        inputs = read(folder(precision, fold) / "fold_inputs.json")
        contracts[fold] = inputs["feature_contracts"]
        seed_rows = rows(folder(precision, fold) / "seed_predictions.csv")
        for method in METHODS:
            for seed in SEEDS:
                path = folder(precision, fold) / method / "checkpoints" / f"seed_{seed}.pt"
                state = torch.load(path, map_location="cpu", weights_only=False)
                receipt = read(folder(precision, fold) / method / f"seed_{seed}/complete.json")
                if state["identity"] != inputs["identity"] or state["identity"] != receipt["identity"] or state["method_id"] != method or state["seed"] != seed or state["epoch_completed"] != 500:
                    raise RuntimeError("frozen_checkpoint_identity")
                model = dense._make_model(method, inputs["node_width"]).to("cuda:0")
                model.load_state_dict(state["model_state"])
                if {str(p.device) for p in model.parameters()} != {"cuda:0"} or {str(p.dtype) for p in model.parameters()} != {"torch.float32"}:
                    raise RuntimeError("model_placement_or_precision")
                models[(fold, method, seed)] = model.eval()
                expected[(fold, method, seed)] = {r["qasm_sha256"]:float(r["prediction_seconds"]) for r in seed_rows if r["method_id"] == method and int(r["seed"]) == seed}
                identities.append(dict(method=method, fold=fold, seed=seed, checkpoint_sha256=sha(path), identity=state["identity"]))
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > plan["limits"]["ram_bytes"] or torch.cuda.memory_reserved() > plan["limits"]["vram_bytes"]:
                    raise RuntimeError("setup_resource_cap")
    selected = [r for r in plan["selected"] if r["context"] == "dense_" + precision]
    raw = {r["id"]:Path(r["raw_path"]).read_bytes() for r in selected}
    digests = {(r["id"],r["model"]):r["expected_hash"] for r in read(PHASE3 / "bridge_validation.json")["receipts"] if r["context"] == "dense_" + precision}
    child = subprocess.Popen([str(CPU), "-B", str(Path(__file__)), "--stage", "cpu-serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, **plan["environment"], "CUDA_VISIBLE_DEVICES":""})
    stop = threading.Event()
    monitor = dict(peak_aggregate_rss_bytes=0, peak_host_rss_bytes=0, peak_child_rss_bytes=0, violation="")
    def sample():
        while not stop.is_set():
            host, cpu = rss(os.getpid()), rss(child.pid)
            for k,v in (("peak_aggregate_rss_bytes",host+cpu),("peak_host_rss_bytes",host),("peak_child_rss_bytes",cpu)):
                monitor[k] = max(monitor[k], v)
            if host+cpu > plan["limits"]["ram_bytes"]:
                monitor["violation"] = "aggregate_RAM_cap"; os.kill(os.getpid(), signal.SIGALRM); return
            stop.wait(.05)
    thread = threading.Thread(target=sample, daemon=True); thread.start()
    replay, timings, failures, controls, environment = [], [], [], {}, {}
    calls = 0
    try:
        ready = json.loads(child.stdout.readline())
        if ready["status"] != "READY":
            raise RuntimeError("CPU_startup")
        signal.alarm(0)
        environment = dict(python=platform.python_version(), packages={k:importlib.metadata.version(k) for k in ("torch","torch-geometric","qiskit","numpy")},
            CUDA_smoke="PASS", cuda=torch.version.cuda, device=torch.cuda.get_device_name(0), host_pid=os.getpid(), child_pid=child.pid,
            cpu_environment=ready, model_count=len(models), checkpoint_identities=identities, affinity=sorted(os.sched_getaffinity(0)),
            torch_threads=2, interop_threads=1, estimator_precision="float32", simulator_target_precision=precision, setup_ns=time.perf_counter_ns()-setup_start)
        def guard():
            if monitor["violation"]: raise RuntimeError(monitor["violation"])
            if max(torch.cuda.memory_allocated(),torch.cuda.memory_reserved()) > plan["limits"]["vram_bytes"]: raise RuntimeError("VRAM_cap")
            if shutil.disk_usage(ROOT).free < plan["limits"]["disk_floor_bytes"]: raise RuntimeError("disk_floor")
            if time.monotonic()-start+prior > plan["limits"]["wall_seconds"]: raise RuntimeError("aggregate_wall_cap")
            if sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) > plan["limits"]["output_bytes"]: raise RuntimeError("artifact_cap")
        def pipeline(r, method, check=False):
            nonlocal calls
            guard(); signal.alarm(30)
            stages = {}
            def stage(name, fn, gpu=False):
                if gpu: torch.cuda.synchronize()
                begin = time.perf_counter_ns(); result = fn()
                if gpu: torch.cuda.synchronize()
                stages[name] = time.perf_counter_ns()-begin
                return result
            graph = method == METHODS[0]
            torch.cuda.synchronize(); begin = time.perf_counter_ns()
            request = stage("request_serialization", lambda: json.dumps(dict(graph=graph, contract=contracts[int(r["fold"])][method], precision=precision,
                raw_qasm_base64=base64.b64encode(raw[r["id"]]).decode()), separators=(",", ":"), allow_nan=False)+"\n")
            def send(): child.stdin.write(request); child.stdin.flush()
            stage("request_write_flush", send)
            line = stage("response_wait_inclusive", child.stdout.readline)
            response = stage("response_JSON_decode", lambda: json.loads(line))
            if response["status"] != "PASS": raise RuntimeError("CPU_preprocessing:"+str(response))
            arrays = stage("typed_array_decode", lambda: compat.unpack(json.loads(response["payload"])))
            if check and array_hash(arrays) != digests[(r["id"], "Graph" if graph else "MLP")]:
                raise RuntimeError("model_ready_digest")
            data = stage("tensor_construction", lambda: Data(**{k:torch.as_tensor(v) for k,v in arrays.items()}))
            payload = stage("batch_and_H2D", lambda: Batch.from_data_list([data]).to("cuda:0"), True)
            def inference():
                return [float(torch.expm1(models[(int(r["fold"]),method,s)](payload)).clamp_min(0).detach().cpu().reshape(-1)[0]) for s in SEEDS]
            values = stage("three_seed_inference_inverse_and_D2H", inference, True)
            estimate = stage("host_median_aggregation", lambda: statistics.median(values))
            total = time.perf_counter_ns()-begin; signal.alarm(0)
            if not np.isfinite(values).all(): raise RuntimeError("nonfinite_prediction")
            item = dict(context="dense_"+precision, id=r["id"], method=method, total_B2_ns=total, predicted_seconds=estimate, request_bytes=len(request.encode()),
                response_bytes=len(line.encode()), child_peak_rss_bytes=response["peak_rss_bytes"], **{f"host_{k}_ns":v for k,v in stages.items()}, **{f"cpu_{k}_ns":v for k,v in response["stages_ns"].items()})
            for k in CPU_STAGES: item.setdefault(f"cpu_{k}_ns",0)
            item["IPC_unattributed_residual_ns"] = stages["request_write_flush"]+stages["response_wait_inclusive"]-sum(response["stages_ns"].values())
            item["coordinator_unattributed_ns"] = total-sum(stages.values())
            if min(item["IPC_unattributed_residual_ns"],item["coordinator_unattributed_ns"]) < 0: raise RuntimeError("nested_partition")
            calls += 1
            if calls % 60 == 0: compute_inventory(os.getpid())
            guard()
            return values, estimate, item, array_hash(arrays) if check else None
        with torch.no_grad():
            if phase == "timing":
                timer=[]; echo=[]
                for _ in range(1000):
                    t=time.perf_counter_ns(); timer.append(time.perf_counter_ns()-t)
                for _ in range(30):
                    t=time.perf_counter_ns(); child.stdin.write('"echo"\n'); child.stdin.flush()
                    assert json.loads(child.stdout.readline())["status"]=="ECHO"; echo.append(time.perf_counter_ns()-t)
                controls=dict(timer_pair_ns=timer,empty_stdio_echo_ns=echo,no_subtraction=True)
            for i,r in enumerate(selected):
                if phase == "replay":
                    for method in order(i,0):
                        values,estimate,_,digest=pipeline(r,method,True)
                        saved=[expected[(int(r["fold"]),method,s)][r["id"]] for s in SEEDS]
                        for seed,actual,wanted in zip((*SEEDS,"median3"),(*values,estimate),(*saved,statistics.median(saved))):
                            replay.append(dict(context="dense_"+precision,id=r["id"],fold=r["fold"],method=method,seed=seed,status="PASS" if passing(actual,wanted) else "FAIL",
                                replayed_seconds=actual,saved_seconds=wanted,abs_diff_seconds=abs(actual-wanted),model_ready_digest=digest,input_dtype_shape_exact=True,units="seconds"))
                        if any(x["status"]!="PASS" for x in replay[-4:]): raise RuntimeError("prediction_replay:"+r["id"]+":"+method)
                else:
                    for rep in range(-plan["warmups"],plan["repetitions"]):
                        for position,method in enumerate(order(i,rep)):
                            values,estimate,item,_=pipeline(r,method)
                            if not all(passing(v,expected[(int(r["fold"]),method,s)][r["id"]]) for v,s in zip(values,SEEDS)):
                                raise RuntimeError("timed_seed_drift:"+r["id"])
                            if rep >= 0:
                                item.update(repetition=rep,order_position=position,fold=r["fold"],group=r["group"],status="PASS"); timings.append(item)
                print(json.dumps(dict(stage=phase,precision=precision,completed_ids=i+1,total_ids=30,timed_requests=len(timings))),flush=True)
    except Exception as exc:
        failures.append(dict(stage=phase,precision=precision,reason=f"{type(exc).__name__}:{exc}",replay_rows=len(replay),timed_requests=len(timings)))
    finally:
        signal.alarm(0); child.stdin.close()
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired: child.terminate(); child.wait(timeout=10)
        stderr=child.stderr.read(); stop.set(); thread.join(timeout=1)
        res=dict(**monitor,host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 wall_seconds=time.monotonic()-start,child_exit_code=child.returncode,child_stderr=stderr,subprocess_reaped=child.poll() is not None)
    n=verify(plan["source_pins"])
    write_json(prefix+"_receipt.json",dict(status="PASS" if not failures else "BLOCKED",failures=failures,environment=environment,resources=res,controls=controls,
        source_pins_verified=n,predeclaration_sha256=sha(OUT/"predeclaration.json")))
    if replay: write_csv(prefix+".csv",replay)
    if timings: write_csv(prefix+".csv",timings)
    expected_n=240 if phase=="replay" else 1800
    passed=not failures and len(replay if phase=="replay" else timings)==expected_n
    write_json(f"gate{1 if phase=='replay' else 2}_{precision}.json",dict(status="PASS" if passed else "BLOCKED",Graph_ids=30 if passed else None,MLP_ids=30 if passed else None,rows=expected_n if passed else len(replay if phase=="replay" else timings),failures=failures))
    if not passed: raise RuntimeError("context_blocked:"+str(failures))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=["prepare","cpu-serve","replay","timing"],required=True); parser.add_argument("--context",choices=CONTEXTS); args=parser.parse_args()
    if args.stage=="prepare": prepare()
    elif args.stage=="cpu-serve": cpu_serve()
    elif args.context: worker(args.stage,args.context)
    else: parser.error("context required for execution")


if __name__=="__main__": main()
