#!/usr/bin/env python3
"""Bounded, inference-only AM1 amendment. Never invokes producing runner mains.

Prepare freezes selection before execution. Workers replay first; failed
variants are not timed. Fresh serial processes record setup separately.
All writes are confined to the new amendment directory, never source artifacts.
"""
from __future__ import annotations

import argparse
import calendar
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
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/am1_overhead_pilot/v1"
FEAS = ROOT / "artifacts/am1_feasibility/v1/provenance_manifest.json"
P = ROOT / "artifacts/real_qpu/qualified_logical_panel"
D = ROOT / "work/benchmark_recovery/r8_dense_predictors/attempt_002"
M = ROOT / "work/repo_finalization/publish_candidate/artifacts/benchmark_v3/simulator/maestro_cpu_component_optimizer_off"
GPU_PY = ROOT.parent / "Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python"
CPU_PY = ROOT.parent / ".venv-qonductor/bin/python"
SEEDS = (42, 1234, 31415)
LIMIT = 8 * 1024**3
STORAGE = 256 * 1024**2
FLOOR = 3 * 1024**3
WARMUPS, REPS, COLD = 10, 30, 3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    csv.field_size_limit(100_000_000)
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def check_storage(extra=0):
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) if OUT.exists() else 0
    if size + extra > STORAGE or shutil.disk_usage(ROOT).free < FLOOR + extra:
        raise RuntimeError("output_storage_or_disk_floor")


def save(path, value):
    path = Path(path)
    if OUT not in path.resolve().parents:
        raise RuntimeError("write_outside_amendment")
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    check_storage(len(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(data)


def csv_save(name, data, fields=None):
    check_storage(4 * 1024**2)
    with (OUT / name).open("x", newline="") as f:
        writer = csv.DictWriter(f, fields or list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def md(name, text):
    check_storage(len(text.encode()))
    with (OUT / name).open("x") as f:
        f.write(text.rstrip() + "\n")


def verify(pins):
    bad = [p for p, s in pins.items() if not Path(p).is_file() or sha(p) != s]
    if bad:
        raise RuntimeError("source_drift:" + str(bad[:3]))


def qkey(track, context, identity):
    return hashlib.sha256(f"AM1-ESTIMATOR-OVERHEAD|{track}|{context}|{identity}".encode()).hexdigest()


def width_bin(value):
    n = float(value)
    return "2-4" if n <= 4 else "5-7" if n <= 7 else "8-9" if n <= 9 else "10-16" if n <= 16 else ">16"


def select(data, cap, axes):
    """Greedy non-Cartesian strata coverage, with explicit uncovered strata."""
    ordered = sorted(data, key=lambda r: qkey(r["track"], r["context"], r["id"]))
    chosen = []
    covered = {axis: set() for axis in axes}
    for axis in axes:
        for value in sorted({r[axis] for r in ordered}):
            if value in covered[axis] or len(chosen) == cap:
                continue
            r = next(r for r in ordered if r[axis] == value and r not in chosen)
            chosen.append(r)
            for a in axes:
                covered[a].add(r[a])
    chosen.extend(r for r in ordered if r not in chosen and len(chosen) < cap)
    # The generator above evaluates the growing list lazily; cap remains exact.
    uncovered = {a: sorted({r[a] for r in ordered} - {r[a] for r in chosen}) for a in axes}
    return chosen, uncovered


def tertiles(data, field):
    values = sorted(float(r[field]) for r in data)
    lo, hi = values[len(values)//3], values[2*len(values)//3]
    for r in data:
        v = float(r[field])
        r[field + "_bin"] = "low" if v <= lo else "middle" if v <= hi else "high"


def prepare():
    if OUT.exists():
        raise RuntimeError("refuse_existing_pilot_namespace")
    feasibility = read(FEAS)
    pins = dict(feasibility["source_pins"])
    for p in [FEAS, Path(__file__), ROOT/"AGENTS.md",
              ROOT/"artifacts/research_question_closure/v1/provenance_manifest.json",
              ROOT/"artifacts/thesis_interpretation/v1/provenance_manifest.json"]:
        pins[str(p)] = sha(p)
    verify(pins)
    inventory = rows(FEAS.parent/"estimator_inventory.csv")
    assert len(inventory) == 86
    canonical = {r["canonical_row_id"]: r for r in rows(ROOT/"artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv")}
    fold_map = {r["canonical_observation_id"]: int(r["outer_fold"]) for r in rows(P/"outer_splits.csv")}
    selected, uncovered = [], {}
    qpu = []
    for r in rows(P/"panel.csv"):
        c = canonical[r["canonical_observation_id"]]
        qpu.append(dict(track="A", context="qualified_qpu_4515", id=r["canonical_observation_id"],
            fold=fold_map[r["canonical_observation_id"]], source=r["source_id"], family=c["family"],
            backend=r["backend"], width=c["width_qubits"] or r["g44"], depth=r["g45"],
            gates=r["node_count"], nodes=r["node_count"], edges=r["edge_count"], group=r["group_id"],
            input_digest=r["graph_sha256"], width_bin=width_bin(c["width_qubits"] or r["g44"])))
    for f in range(5):
        candidates = [r for r in qpu if r["fold"] == f]
        for k in ("depth", "gates", "nodes"):
            tertiles(candidates,k)
        choice, missing = select(candidates,10,("source","family","backend","width_bin","depth_bin","gates_bin","nodes_bin"))
        selected.extend(choice); uncovered[f"A/fold{f}"] = missing
    features = {r["qasm_sha256"]: r for r in map(json.loads, (ROOT/"work/benchmark_recovery/r3_dense/ir_features_v1/dense_features_by_hash.jsonl").read_text().splitlines())}
    targets = rows(ROOT/"work/benchmark_recovery/r5_dense/targets_recovered_v2/dense_hash_targets.csv")
    panel = rows(ROOT/"artifacts/benchmark_v1/sim_common_q16_manifest_20260927/sim_common_q16_manifest.csv")
    family = {r.get("qasm_sha256",r.get("source_qasm_sha256")): r.get("family",r.get("circuit_family","unknown")) for r in panel}
    for precision in ("fp32","fp64"):
        data = []
        for r in targets:
            if r["precision"] != precision or r["stratum"] != "core_q2_q9":
                continue
            h=r["qasm_sha256"]; feat=features[h]; g=feat["global_features"]; dag=feat["dag"]
            data.append(dict(track="B",context=f"dense_{precision}",id=h,fold=int(r["outer_fold"]),source="local_dense",
                family=family.get(h,"unknown"),backend=f"CUDA-Q SV {precision} sample32",width=g["active_width"],
                depth=g["structural_depth"],gates=len(dag["names"]),nodes=len(dag["names"]),edges=len(dag["edge_src"]),
                group=h,input_digest=feat["normalized_ir_sha256"],width_bin=width_bin(g["active_width"])))
        # Six per fold gives 30/context, preserving all five frozen routes.
        for f in range(5):
            subset=[r for r in data if r["fold"]==f]
            for k in ("depth","gates","nodes"): tertiles(subset,k)
            choice, missing=select(subset,6,("family","width_bin","depth_bin","gates_bin","nodes_bin"))
            selected.extend(choice); uncovered[f"{precision}/fold{f}"]=missing
    maestro=[]
    for r in rows(M/"evaluation_001/oof_predictions.csv"):
        h=r["source_qasm_sha256"]
        # No eligibility/target-based selection. All assigned hashes are candidates.
        maestro.append(dict(track="B",context="maestro",id=h,fold=int(r["fold"]),source="local_qcsim",
            family=family.get(h,"unknown"),backend="QCSim CPU SV FP64 optimizer-off shots1000",width=features[h]["global_features"]["active_width"],
            depth=features[h]["global_features"]["structural_depth"],gates=len(features[h]["dag"]["names"]),
            nodes=len(features[h]["dag"]["names"]),edges=len(features[h]["dag"]["edge_src"]),group=h,
            input_digest=r["normalized_qasm_sha256"] or "UNAVAILABLE",width_bin=width_bin(features[h]["global_features"]["active_width"])))
    for k in ("depth","gates","nodes"): tertiles(maestro,k)
    choice,missing=select(maestro,30,("family","width_bin","depth_bin","gates_bin","nodes_bin"))
    selected.extend(choice); uncovered["maestro"]=missing
    contexts=[("qpu",f) for f in range(5)]+[(f"dense_{p}",f) for p in ("fp32","fp64") for f in range(5)]+[("analytical",-1),("maestro",-1)]
    methods={"qpu":["mali_logical_graph","mali_global_mlp","outer_train_median"],
             "dense_fp32":["mali_style_dense_graph","matched_metadata_mlp","outer_train_median"],
             "dense_fp64":["mali_style_dense_graph","matched_metadata_mlp","outer_train_median"],
             "analytical":["qiskit_estimate_duration_snapshot__raw","qiskit_estimate_duration_snapshot__affine"],
             "maestro":["maestro_style_cpu_sv_component_optimizer_off_adaptation","outer_train_median"]}
    variant_rows=[dict(context=c,fold=f,method=m,seeds="42;1234;31415" if m in methods[c][:2] and c not in ("analytical","maestro") else "NA",
                       aggregation="median3" if c not in ("analytical","maestro") and m!="outer_train_median" else "native",
                       B1="selected",B2="NOT EXECUTED: full logical/QASM feature path not qualified in this narrow pilot",
                       partial_pipeline="selected frozen-feature encode/H2D" if c not in ("analytical","maestro") else "selected compiled-QASM parse" if c=="analytical" else "NOT EXECUTED",
                       B3="3 fresh processes per context/fold bundle") for c,f in contexts for m in methods[c]]
    check_storage(); OUT.mkdir(parents=True)
    # Predeclared manifests contain no timing outputs and precede every worker.
    csv_save("selected_variants.csv",variant_rows)
    csv_save("selected_workloads.csv",selected)
    env={k:"1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")}
    env.update(PYTHONDONTWRITEBYTECODE="1",QISKIT_IN_PARALLEL="FALSE")
    cpu_info=subprocess.check_output(["lscpu","-J"],text=True)
    affin=sorted(os.sched_getaffinity(0))
    freqs={i:int(Path(f"/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq").read_text()) for i in affin if Path(f"/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq").exists()}
    best=max(freqs.values()) if freqs else None
    pcores=[i for i in affin if freqs.get(i)==best] if freqs else affin
    cpus=pcores[:2]
    plan=dict(amendment="AM1-ESTIMATOR-OVERHEAD",authorized_by_attachment="6ab9573a-6f87-4818-9855-90b8cc53af0f",created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        source_pins=pins,script_sha256=sha(__file__),contexts=contexts,methods=methods,uncovered_strata=uncovered,
        sample_policy="SHA256 greedy label-free coverage; cap 10/fold QPU, 6/fold/precision dense, 30 Maestro; uncovered strata retained",
        repetitions=REPS,warmups=WARMUPS,cold_restarts=COLD,batch=1,seeds=list(SEEDS),atol_neural=1e-6,rtol_neural=1e-5,atol_scalar=1e-12,rtol_scalar=1e-10,
        selected_workloads_sha256=sha(OUT/"selected_workloads.csv"),selected_variants_sha256=sha(OUT/"selected_variants.csv"),
        env=env,affinity=cpus,cpu_max_frequency_khz=best,cpu_info=json.loads(cpu_info),
        hardware_inventory=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm,power.draw","--format=csv"],text=True),
        memory_inventory=Path("/proc/meminfo").read_text(),disk_free_bytes=shutil.disk_usage(ROOT).free,
        constants=feasibility["saved_constant_parameters"],limits=dict(wall_seconds=7200,request_seconds=30,cold_seconds=120,ram_bytes=LIMIT,vram_bytes=LIMIT,output_bytes=STORAGE,disk_floor_bytes=FLOOR),
        B2_full="NOT EXECUTED",exclusions=["joint-MPS and fixed-MPS are outside narrowed pilot, not replay failures","QCRE/Hyb/Scholten not selected for native timing; no substitute lookup","missing Qonductor selectors/RF/XGB/Azizov states: no refit","cuTensorNet planning/contraction and Pasqal simulation excluded"])
    save(OUT/"selection_manifest.json",plan)
    print(json.dumps(dict(stage="prepare",selected=len(selected),contexts=len(contexts),source_pins=len(pins))))


def qualify_indirect_pins():
    """Read exact producing receipts, without changing selected queries/config."""
    plan=read(OUT/"selection_manifest.json")
    pins=dict(plan["source_pins"])
    def add(path,expected):
        path=Path(path)
        if sha(path)!=expected: raise RuntimeError("indirect_source_drift:"+str(path))
        pins[str(path)]=expected
    for precision in ("fp32","fp64"):
        for f in range(5):
            folder=D/precision/"neural"/f"fold_{f}"
            receipt=read(folder/"fold_manifest.json")
            add(folder/"seed_predictions.csv",receipt["seed_prediction_sha256"])
            add(ROOT/"work/benchmark_recovery/r3_dense/ir_features_v1/dense_features_by_hash.jsonl",receipt["identity"]["input_hashes"]["r3_features"])
    representations={r["canonical_row_id"]:r for r in rows(ROOT/"artifacts/benchmark_v3/real_qpu/compiled_unified/representation_rows.csv")}
    for r in rows(OUT/"selected_workloads.csv"):
        if r["track"]=="A":
            rep=representations[r["id"]]
            if rep["lifecycle_stage"]!="submitted_physical":
                add(ROOT/"artifacts/benchmark_v3/real_qpu/compiled_unified"/rep["compiled_qasm3_file"],rep["compiled_qasm3_sha256"])
    for path,expected in read(P/"manifest.json")["external_hashes"].items(): add(path,expected)
    save(OUT/"adapter_correction.json",dict(reason="chain indirect producing-receipt pins; tolerate process exit during RSS observation",original_script_sha256=plan["script_sha256"],script_sha256=sha(__file__),
        original_selection_manifest_sha256=sha(OUT/"selection_manifest.json"),source_pins=pins,selection_unchanged=True,models_seeds_tolerances_budgets_unchanged=True,
        initiated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())))
    print(json.dumps(dict(stage="qualify_indirect_pins",total_pins=len(pins))))


def effective_plan():
    plan=read(OUT/"selection_manifest.json")
    correction=OUT/"adapter_correction.json"
    if correction.exists():
        addendum=read(correction)
        if sha(OUT/"selection_manifest.json")!=addendum["original_selection_manifest_sha256"]: raise RuntimeError("selection_manifest_drift")
        plan["source_pins"]=addendum["source_pins"]
        plan["script_sha256"]=addendum["script_sha256"]
    return plan


def alarm_handler(*_):
    raise TimeoutError("request_30s_cap")


def guard(torch=None):
    check_storage()
    rss=int(next(s.split()[1] for s in Path("/proc/self/status").read_text().splitlines() if s.startswith("VmRSS:")))*1024
    if rss>LIMIT: raise RuntimeError("RAM_8GiB_cap")
    if torch is not None and torch.cuda.memory_reserved()>LIMIT: raise RuntimeError("VRAM_8GiB_cap")


def parity(method, identity, seed, actual, expected):
    scalar=seed=="native"
    atol,rtol=(1e-12,1e-10) if scalar else (1e-6,1e-5)
    delta=abs(actual-expected)
    return dict(method=method,id=identity,seed=seed,replayed_seconds=actual,saved_seconds=expected,abs_diff_seconds=delta,
        relative_diff=delta/abs(expected) if expected else "NA",atol_seconds=atol,rtol=rtol,
        status="PASS" if math.isfinite(actual) and delta<=atol+rtol*abs(expected) else "REPLAY FAILED")


def worker(context,fold,phase,cold):
    start=time.perf_counter_ns()
    plan=effective_plan()
    if sha(__file__)!=plan["script_sha256"]: raise RuntimeError("pilot_script_changed")
    os.sched_setaffinity(0,plan["affinity"])
    signal.signal(signal.SIGALRM,alarm_handler)
    subset=[r for r in rows(OUT/"selected_workloads.csv") if r["context"]==("qualified_qpu_4515" if context in ("qpu","analytical") else context) and (fold<0 or int(r["fold"])==fold)]
    bindings=[]; stage_times=[]; methods=plan["methods"][context]; torch=None
    timings=[]; replay=[]; payloads={}; expected={}; models={}; preprocess={}
    def stage(name, fn):
        signal.alarm(30)
        s=time.perf_counter_ns(); result=fn(); e=time.perf_counter_ns(); signal.alarm(0)
        stage_times.append(dict(stage=name,start_ns=s,end_ns=e,elapsed_ns=e-s))
        return result
    def bind(path):
        path=Path(path); s=sha(path)
        if plan["source_pins"].get(str(path)) != s: raise RuntimeError("unqualified_asset:"+str(path))
        bindings.append(dict(path=str(path),sha256=s))
        return path
    native_functions={}
    if context=="qpu" or context.startswith("dense_"):
        def imports():
            import torch as t
            import torch_geometric
            t.set_num_threads(2); t.set_num_interop_threads(1)
            if not t.cuda.is_available(): raise RuntimeError("CUDA_REQUIRED")
            return t
        torch=stage("framework_import",imports)
        def init():
            torch.cuda.init(); a=torch.ones((8,8),device="cuda"); b=a@a; torch.cuda.synchronize()
            return torch.cuda.get_device_properties(0).name
        stage("cuda_initialization",init)
        torch.cuda.set_per_process_memory_fraction(0.48,0)
        sys.path.insert(0,str(ROOT/"benchmark_v1/scripts"))
        if context=="qpu":
            import run_qualified_logical_benchmark as source
            import run_mali_full_features as base
            import run_mali_batched as batching
            panel=rows(bind(P/"panel.csv")); byid={r["canonical_observation_id"]:r for r in panel}
            tfm=read(bind(P/"neural"/f"fold_{fold}"/"transform.json"))
            ledger={i:{**r,"graph_artifact":r["graph_file"],"graph_artifact_sha256":r["graph_sha256"],"graph_node_count":r["node_count"],"graph_status":"available","global_status":"available",**{f"global_{j:02d}":r[f"g{j:02d}"] for j in range(51)}} for i,r in byid.items()}
            cx=dict(ledger=ledger,feature_root=ROOT)
            for method in methods[:2]:
                internal=base.METHODS[source.NEURAL.index(method)]
                for seed in SEEDS:
                    directory=P/"neural"/f"fold_{fold}"/method/f"seed_{seed}"
                    receipt=read(bind(directory/"manifest.json"))
                    checkpoint=bind(directory/"checkpoint.pt")
                    if receipt["identity"]["transform"]!=stable(tfm): raise RuntimeError("transform_identity")
                    def load(method=method,checkpoint=checkpoint,receipt=receipt):
                        model,_,_=source.make_model(method,tfm,torch)
                        state=torch.load(checkpoint,map_location="cpu",weights_only=False)
                        if state["identity"]!=receipt["identity"]: raise RuntimeError("checkpoint_identity")
                        model.load_state_dict(state["best_model_state"]); model.eval()
                        torch.cuda.synchronize(); return model
                    models[(method,seed)]=stage(f"load/{method}/{seed}",load)
                    expected[(method,seed)]={r["canonical_observation_id"]:float(r["predicted_seconds"]) for r in rows(bind(directory/"predictions.csv"))}
                cache=batching.ExampleCache(cx,tfm,1024**3)
                for r in subset:
                    rid=r["id"]
                    bind(ROOT/byid[rid]["graph_file"])
                    preprocess[(method,rid)]=lambda rid=rid,internal=internal,graph=method==methods[0]: base._batch_to_device([base._make_example(rid,internal,ledger,None,tfm,ROOT,"cpu",False,set())],"cuda:0",graph)
                    payloads[(method,rid)]=stage(f"prepare/{method}/{rid}",lambda rid=rid,internal=internal,graph=method==methods[0]:base._batch_to_device([cache.get(rid,internal)],"cuda:0",graph))
            def predict_seed(method,rid,seed,payload=None):
                return float(base._prediction(models[(method,seed)],payload if payload is not None else payloads[(method,rid)]).detach().cpu().reshape(-1)[0])
            constants=plan["constants"]["A/outer_train_median"]
            reference=rows(bind(ROOT/"artifacts/benchmark_deep_dive/tables/qpu_baseline_predictions_v1.csv"))
            expected[(methods[2],"native")]={r["canonical_observation_id"]:float(r["predicted_seconds"]) for r in reference if r["method_id"]=="outer_train_median" and int(r["outer_fold"])==fold}
        else:
            import run_cudaq_dense_neural_predictors_v1 as source
            from torch_geometric.data import Batch,Data
            precision=context.split("_")[-1]; folder=D/precision/"neural"/f"fold_{fold}"
            fin=read(bind(folder/"fold_inputs.json"))
            feature=source.load_feature_records(bind(ROOT/"work/benchmark_recovery/r3_dense/ir_features_v1/dense_features_by_hash.jsonl"))
            seed_rows=rows(bind(folder/"seed_predictions.csv"))
            for method in methods[:2]:
                contract=fin["feature_contracts"][method]; tfm=contract["global_transform_values"]
                for seed in SEEDS:
                    checkpoint=bind(folder/method/"checkpoints"/f"seed_{seed}.pt")
                    receipt=read(bind(folder/method/f"seed_{seed}/complete.json"))
                    def load(method=method,checkpoint=checkpoint,receipt=receipt):
                        width=len(fin["feature_contracts"][methods[0]]["gate_vocabulary"])+9
                        model=source._make_model(method,width).to("cuda:0")
                        state=torch.load(checkpoint,map_location="cpu",weights_only=False)
                        if state["identity"]!=receipt["identity"]: raise RuntimeError("dense_checkpoint_identity")
                        model.load_state_dict(state["model_state"]); model.eval(); torch.cuda.synchronize(); return model
                    models[(method,seed)]=stage(f"load/{method}/{seed}",load)
                    expected[(method,seed)]={r["qasm_sha256"]:float(r["prediction_seconds"]) for r in seed_rows if r["method_id"]==method and int(r["seed"])==seed}
                def prep(rid,method=method,contract=contract,tfm=tfm):
                    if method==methods[0]:
                        data=source.record_to_graph({"feature":feature[rid]},contract["gate_vocabulary"],tfm,include_target=False)
                    else:
                        # The frozen MLP consumes globals only; no gratuitous DAG work.
                        data=Data(global_features=torch.as_tensor(source.normalized_globals(feature[rid],tfm)).reshape(1,7))
                    return Batch.from_data_list([data]).to("cuda:0")
                for r in subset:
                    rid=r["id"]; preprocess[(method,rid)]=lambda rid=rid,prep=prep: prep(rid)
                    payloads[(method,rid)]=stage(f"prepare/{method}/{rid}",preprocess[(method,rid)])
            def predict_seed(method,rid,seed,payload=None):
                data=payload if payload is not None else payloads[(method,rid)]
                return float(torch.expm1(models[(method,seed)](data)).clamp_min(0).detach().cpu().reshape(-1)[0])
            constants=plan["constants"][f"{precision}/outer_train_median"]
            classical=D/precision/"classical"/f"fold_{fold}"
            expected[(methods[2],"native")]={r["qasm_sha256"]:float(r["prediction_seconds"]) for r in rows(bind(classical/"predictions.csv")) if r["method_id"]=="outer_train_median"}
        native_functions[methods[2]]=lambda rid:float(constants[str(fold)])
        def predict(method,rid,partial=False):
            if method==methods[2]: return native_functions[method](rid)
            with torch.no_grad():
                payload=preprocess[(method,rid)]() if partial else payloads[(method,rid)]
                return statistics.median(predict_seed(method,rid,s,payload) for s in SEEDS)
    elif context=="maestro":
        calibration=read(bind(M/"full_001/calibration_model.json"))
        knots={int(r["width"]):r for r in calibration["width_coefficients"]}
        data={r["source_qasm_sha256"]:r for r in rows(bind(M/"evaluation_001/oof_predictions.csv"))}
        for method in methods:
            expected[(method,"native")]={h:float(r[f"pred_{method}_seconds"]) for h,r in data.items() if r.get(f"pred_{method}_seconds")}
        def predict(method,rid,partial=False):
            r=data[rid]
            if method=="outer_train_median": return float(plan["constants"]["maestro/outer_train_median"][r["fold"]])
            n=int(r["normalized_width"]); k=knots[n]
            return 2**n*(float(k["C1q"])*int(r["normalized_1q_gate_count"])+float(k["Ccx"])*int(r["normalized_cx_gate_count"]))+float(k["b"])+float(k["m"])*1000
    else:
        def imports():
            from qiskit import qasm3
            from qiskit_ibm_runtime import fake_provider
            return qasm3,fake_provider
        qasm3,fake=stage("qiskit_import",imports)
        sys.path.insert(0,str(ROOT/"benchmark_v1/scripts"))
        from qasm3_hardware_wire_adapter import restore_saved_qasm3_hardware_wires
        from qonductor_native_adapter import load_pinned_qonductor_methods
        import zipfile
        contract=read(bind(ROOT/"benchmark_v1/protocol/qonductor_native_features.json"))
        _,_,parse_submitted,_=load_pinned_qonductor_methods(ROOT.parent/"Qonductor-SC25",contract["source_pin"])
        canonical={r["canonical_row_id"]:r for r in rows(bind(ROOT/"artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"))}
        reps={r["canonical_row_id"]:r for r in rows(bind(ROOT/"artifacts/benchmark_v3/real_qpu/compiled_unified/representation_rows.csv"))}
        registry={r["backend_canonical"]:r for r in rows(bind(ROOT/"artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c133_snapshot_registry_v3.csv"))}
        devices={}
        for b in sorted({r["backend"] for r in subset}):
            entry=registry[b.replace("ibmq_","ibm_")]
            for key in ("configuration_path","properties_path"): bind(entry[key])
            cls=entry["fake_backend_class"].split(".")[-1]
            devices[b]=stage(f"target_initialization/{b}",lambda cls=cls:getattr(fake,cls)())
        texts={}; circuits={}; byid={r["id"]:r for r in subset}
        for r in subset:
            rid=r["id"]; rep=reps[rid]
            if rep["lifecycle_stage"]=="submitted_physical":
                archive=ROOT.parent/"Qonductor-SC25/data/database/circuits.zip"
                # Entire archive is already pinned by the frozen panel external hashes.
                with zipfile.ZipFile(archive) as z: text=z.read(canonical[rid]["qasm_path_or_member"]).decode()
                if hashlib.sha256(text.encode()).hexdigest()!=rep["source_qasm_sha256"]: raise RuntimeError("physical_qasm_hash")
                parser=parse_submitted
            else:
                path=bind(ROOT/"artifacts/benchmark_v3/real_qpu/compiled_unified"/rep["compiled_qasm3_file"])
                text=path.read_text()
                def parser(text, width=int(rep["register_width"])):
                    restored,_=restore_saved_qasm3_hardware_wires(text,width)
                    return qasm3.loads(restored)
            texts[rid]=(text,parser)
            circuits[rid]=stage(f"parse_frozen_compiled/{rid}",lambda text=text,parser=parser:parser(text))
        fits={(r["method_id"],int(r["outer_fold"])):r for r in rows(bind(P/"analytical/calibration_fits.csv")) if r["variant"]=="affine"}
        attempts=rows(bind(P/"analytical/attempts.csv"))
        for method in methods:
            expected[(method,"native")]={r["canonical_observation_id"]:float(r["predicted_seconds"]) for r in attempts if r["method_id"]==method and r["status"]=="predicted"}
        def predict(method,rid,partial=False):
            text,parser=texts[rid]
            circuit=parser(text) if partial else circuits[rid]
            r=byid[rid]
            value=float(circuit.estimate_duration(target=devices[r["backend"]].target,unit="s"))
            if method.endswith("__affine"):
                fit=fits[("qiskit_estimate_duration_snapshot",int(r["fold"]))]
                value=max(0.,float(fit["slope"])*value+float(fit["intercept"]))
            return value
    ready=time.perf_counter_ns()
    environment=dict(python=platform.python_version(),executable=sys.executable,platform=platform.platform(),
        packages={p:importlib.metadata.version(p) for p in ("torch","numpy","torch-geometric","qiskit") if importlib.util.find_spec(p.replace("-","_")) is not None},
        cuda_build=torch.version.cuda if torch is not None else "NA",device="cuda:0" if torch is not None else "CPU",
        affinity=sorted(os.sched_getaffinity(0)),pid=os.getpid(),threads=2 if torch is not None else 1)
    if phase=="replay":
        for method in methods:
            for r in subset:
                rid=r["id"]; guard(torch)
                if torch is not None and method!=methods[2]:
                    for seed in SEEDS:
                        with torch.no_grad(): value=predict_seed(method,rid,seed)
                        replay.append(parity(method,rid,seed,value,expected[(method,seed)][rid]))
                    saved=statistics.median(expected[(method,s)][rid] for s in SEEDS)
                    replay.append(parity(method,rid,"median3",predict(method,rid),saved))
                elif rid in expected[(method,"native")]:
                    replay.append(parity(method,rid,"native",predict(method,rid),expected[(method,"native")][rid]))
                else:
                    replay.append(dict(method=method,id=rid,seed="native",replayed_seconds="NA",saved_seconds="NA",abs_diff_seconds="NA",relative_diff="NA",atol_seconds=1e-12,rtol=1e-10,status="UNAVAILABLE_SAVED_OUTPUT"))
        save(OUT/f"workers/{context}_fold{fold}_replay.json",dict(environment=environment,bindings=bindings,stages=stage_times,replay=replay))
        print(json.dumps(dict(context=context,fold=fold,stage="replay",rows=len(replay),failures=sum(r["status"]!="PASS" for r in replay))),flush=True)
        return
    allowed=read(OUT/"replay_gate.json")[context]
    active=[m for m in methods if allowed[m]]
    admissible={m:{r["id"] for r in read(OUT/f"workers/{context}_fold{fold}_replay.json")["replay"] if r["method"]==m and r["status"]=="PASS"} for m in active}
    def sync():
        if torch is not None: torch.cuda.synchronize()
    def timed(method,rid,boundary,repeat,fn):
        guard(torch); signal.alarm(30)
        sync(); s=time.perf_counter_ns(); result=fn(); sync(); e=time.perf_counter_ns(); signal.alarm(0)
        timings.append(dict(context=context,fold=fold,method=method,id=rid,boundary=boundary,cold_process=cold,repeat=repeat,
            start_ns=s,end_ns=e,elapsed_ns=e-s,prediction_seconds=result if result is not None else "NA",batch=1,
            cache="resident" if boundary=="B1" else "no feature-result cache" if boundary=="P1" else "NA",timer="host_perf_counter_ns",
            synchronization="CUDA before/after" if torch is not None else "NA",status="ok",reason=""))
    if torch is not None: torch.cuda.reset_peak_memory_stats()
    with torch.no_grad() if torch is not None else __import__("contextlib").nullcontext():
        for method in active:
            warm_id=next(r["id"] for r in subset if r["id"] in admissible[method])
            for _ in range(WARMUPS): predict(method,warm_id)
        sync(); warm_ready=time.perf_counter_ns()
        stage_times.append(dict(stage="warmup",start_ns=ready,end_ns=warm_ready,elapsed_ns=warm_ready-ready))
        stage_times.append(dict(stage="child_start_to_ready",start_ns=start,end_ns=warm_ready,elapsed_ns=warm_ready-start))
        for method in active:
            first_id=next(r["id"] for r in subset if r["id"] in admissible[method])
            timed(method,first_id,"first_after_ready",0,lambda method=method,rid=first_id:predict(method,rid))
        if cold==0:
            for repeat in range(REPS):
                timed("noop_dispatch","control","CONTROL",repeat,lambda:None)
                for r in subset:
                    rid=r["id"]
                    order=active[repeat%len(active):]+active[:repeat%len(active)] if active else []
                    for method in order:
                        if rid not in admissible[method]: continue
                        timed(method,rid,"B1",repeat,lambda method=method,rid=rid:predict(method,rid))
                        if context in ("qpu","analytical") or context.startswith("dense_"):
                            if method!="outer_train_median":
                                timed(method,rid,"P1",repeat,lambda method=method,rid=rid:predict(method,rid,True))
    guard(torch)
    memory=dict(rss_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated() if torch is not None else "NA",
        cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved() if torch is not None else "NA",
        attribution="process/bundle absolute peaks; not per-boundary incremental memory")
    save(OUT/f"workers/{context}_fold{fold}_cold{cold}.json",dict(environment=environment,bindings=bindings,stages=stage_times,timings=timings,memory=memory))
    print(json.dumps(dict(context=context,fold=fold,stage="timing",cold=cold,samples=len(timings),memory=memory)),flush=True)


def execute():
    plan=effective_plan()
    verify(plan["source_pins"])
    if sha(OUT/"selected_workloads.csv")!=plan["selected_workloads_sha256"]: raise RuntimeError("selection_changed")
    original_start=calendar.timegm(time.strptime(plan["created_utc"],"%Y-%m-%dT%H:%M:%SZ"))
    begun=time.monotonic()-(time.time()-original_start); executions=[]
    def run(c,f,phase,cold=0):
        guard(); elapsed=time.monotonic()-begun
        if elapsed>=7080: raise RuntimeError("wall_budget_stop")
        python=GPU_PY if c=="qpu" or c.startswith("dense_") else CPU_PY
        command=[str(python),"-B",str(Path(__file__)),"--stage","worker","--context",c,"--fold",str(f),"--phase",phase,"--cold",str(cold)]
        env={**os.environ,**plan["env"]}
        if python==CPU_PY: env["CUDA_VISIBLE_DEVICES"]=""
        s=time.perf_counter_ns(); child=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        maxrss=0; reason=""; start_child=time.monotonic()
        while child.poll() is None:
            try:
                status=Path(f"/proc/{child.pid}/status").read_text()
                rss=int(next((x.split()[1] for x in status.splitlines() if x.startswith("VmRSS:")),0))*1024
                maxrss=max(maxrss,rss)
                check_storage()
                if rss>LIMIT: reason="RAM_cap"
                if time.monotonic()-begun>=7200: reason="wall_cap"
                if phase=="replay" and time.monotonic()-start_child>600: reason="replay_cell_600s_cap"
                if phase=="timing" and cold>0 and time.monotonic()-start_child>120: reason="cold_120s_cap"
                if reason: child.terminate(); break
            except FileNotFoundError: pass
            time.sleep(.2)
        try: stdout,stderr=child.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill(); stdout,stderr=child.communicate()
        e=time.perf_counter_ns()
        result=dict(context=c,fold=f,phase=phase,cold=cold,command=command,pid=child.pid,exit=child.returncode,
            start_ns=s,end_ns=e,elapsed_ns=e-s,max_sampled_rss_bytes=maxrss,status="complete" if child.returncode==0 else "failed",reason=reason,stdout=stdout,stderr=stderr)
        executions.append(result)
        path=OUT/f"executions/{c}_fold{f}_{phase}_{cold}.json"
        if path.exists():
            # Manual harness-only correction retains the original failure receipt.
            result["preceding_execution_sha256"]=sha(path)
            path=path.with_name(path.stem+"_adapter_corrected.json")
        save(path,result)
        print(stdout.strip() or json.dumps({k:result[k] for k in ("context","fold","phase","status","reason")}),flush=True)
        if reason: raise RuntimeError(reason)
    # Entire chosen context/method must pass replay before latency/accuracy linkage.
    for c,f in plan["contexts"]:
        receipt=OUT/f"workers/{c}_fold{f}_replay.json"
        if receipt.exists():
            # Preserve existing valid Gate-1 receipts. No automatic failed-model retry.
            verify({r["path"]:r["sha256"] for r in read(receipt)["bindings"]})
            continue
        run(c,f,"replay")
    gate={c:{m:True for m in plan["methods"][c]} for c,_ in plan["contexts"]}
    for c,f in plan["contexts"]:
        path=OUT/f"workers/{c}_fold{f}_replay.json"
        if not path.exists(): gate[c]={m:False for m in gate[c]}; continue
        receipt=read(path)
        for r in receipt["replay"]:
            if r["status"]=="REPLAY FAILED": gate[c][r["method"]]=False
        for method in gate[c]:
            if not any(r["method"]==method and r["status"]=="PASS" for r in receipt["replay"]): gate[c][method]=False
    save(OUT/"replay_gate.json",gate)
    for c,f in plan["contexts"]:
        if not any(gate[c].values()): continue
        for cold in range(COLD): run(c,f,"timing",cold)
    verify(plan["source_pins"])
    save(OUT/"execution_manifest.json",dict(status="complete",wall_seconds=time.monotonic()-begun,processes=len(executions),source_pins_unchanged=True,
        max_sampled_rss_bytes=max(r["max_sampled_rss_bytes"] for r in executions),output_bytes=sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["prepare","qualify-indirect-pins","execute","worker"],required=True)
    p.add_argument("--context"); p.add_argument("--fold",type=int,default=0)
    p.add_argument("--phase",choices=["replay","timing"],default="replay"); p.add_argument("--cold",type=int,default=0)
    args=p.parse_args()
    if args.stage=="prepare": prepare()
    elif args.stage=="qualify-indirect-pins": qualify_indirect_pins()
    elif args.stage=="execute": execute()
    else: worker(args.context,args.fold,args.phase,args.cold)


if __name__=="__main__":
    try: main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
