#!/usr/bin/env python3
"""Independent saved-evidence audit and optional final derived receipt seal.

No model imports, inference, fitting or simulator execution. Historical sources
are checked against the pre-run declaration. Only new receipt files are written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
import shutil
import statistics
import subprocess
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
from run_am1_phase5_dense_b2 import OUT, ROOT, SCRIPTS, PHASE3, CONTEXTS, METHODS, SEEDS, folder, read, rows, sha, verify, write_json

REQUIRED = ["README.md", "predeclaration.md", "environment_manifest.md", "artifact_integrity.md", "replay_validation.csv", "replay_validation.md",
            "raw_b2_timings.csv", "stage_timing_summary.csv", "fp32_matched_b2.md", "fp64_matched_b2.md", "matched_accuracy_analysis.md",
            "accuracy_cost_tradeoff.md", "within_range_scaling.md", "historical_execution_context.md", "memory_and_resources.md",
            "failures_and_exclusions.md", "rq4_trackb_decision.md", "validation_report.md"]


def near(actual, wanted):
    if not math.isclose(float(actual), float(wanted), rel_tol=0, abs_tol=1e-12):
        raise AssertionError(f"numerical_mismatch:{actual}:{wanted}")


def audit():
    plan=read(OUT/"predeclaration.json"); verify(plan["source_pins"])
    for name in REQUIRED:
        if not (OUT/name).is_file():raise AssertionError("missing_required:"+name)
    analysis=read(OUT/"analysis.json"); paired=rows(OUT/"paired_circuit_results.csv");timings=rows(OUT/"raw_b2_timings.csv");replay=rows(OUT/"replay_validation.csv")
    if len(paired)!=60 or len(timings)!=3600 or len(replay)!=480:raise AssertionError("denominator")
    digest={(r["context"],r["id"],r["model"]):r["expected_hash"] for r in read(PHASE3/"bridge_validation.json")["receipts"]}
    for r in replay:
        model="Graph" if r["method"]==METHODS[0] else "MLP"
        if r["model_ready_digest"]!=digest[(r["context"],r["id"],model)] or r["status"]!="PASS":raise AssertionError("input_digest")
    by=defaultdict(list)
    for r in timings:
        by[(r["context"],r["id"],r["method"])].append(r)
        cpu=sum(int(v) for k,v in r.items() if k.startswith("cpu_") and k.endswith("_ns"))
        host=sum(int(v) for k,v in r.items() if k.startswith("host_") and k.endswith("_ns") and k not in {"host_response_wait_inclusive_ns","host_request_write_flush_ns"})
        if cpu+host+int(r["IPC_unattributed_residual_ns"])+int(r["coordinator_unattributed_ns"])!=int(r["total_B2_ns"]):raise AssertionError("exclusive_duration_total")
    for p in CONTEXTS:
        context="dense_"+p;selected=[r for r in paired if r["context"]==context]
        wanted_ids={r["id"] for r in plan["selected"] if r["context"]==context}
        if len(selected)!=30 or {r["id"] for r in selected}!=wanted_ids:raise AssertionError("paired_ids")
        saved={}; seed_saved={}
        for fold in range(5):
            for r in rows(folder(p,fold)/"predictions.csv"):saved[(r["qasm_sha256"],r["method_id"])]=r
            for r in rows(folder(p,fold)/"seed_predictions.csv"):seed_saved[(r["qasm_sha256"],r["method_id"],r["seed"])]=float(r["prediction_seconds"])
        for r in selected:
            for name,method in zip(("graph","mlp"),METHODS):
                frozen=saved[(r["id"],method)]
                near(r[name+"_prediction_seconds"],frozen["prediction_seconds"]);near(r["target_seconds"],frozen["target_seconds"])
                near(frozen["prediction_seconds"],statistics.median(seed_saved[(r["id"],method,str(s))] for s in SEEDS))
                requests=by[(context,r["id"],method)]
                if len(requests)!=30 or {int(t["repetition"]) for t in requests}!=set(range(30)) or Counter(t["order_position"] for t in requests)!={"0":15,"1":15}:raise AssertionError("raw_request_keys")
                near(r[name+"_B2_median_ms"],statistics.median(int(t["total_B2_ns"])/1e6 for t in requests))
                near(r[name+"_B2_p95_ms"],np.quantile([int(t["total_B2_ns"])/1e6 for t in requests],.95))
        for scope in ("exact_timed_subset","full_frozen_core_OOF"):
            for name,method in zip(("graph","mlp"),METHODS):
                if scope=="exact_timed_subset":
                    y=[float(r["target_seconds"]) for r in selected];pred=[float(r[name+"_prediction_seconds"]) for r in selected]
                    summary=next(r for r in analysis["contexts"][p]["summary"] if r["method"]==name)
                else:
                    source=[r for (rid,m),r in saved.items() if m==method]
                    y=[float(r["target_seconds"]) for r in source];pred=[float(r["prediction_seconds"]) for r in source]
                    summary=next(r for r in analysis["full_context_accuracy"] if r["context"]==context and r["method"]==name)
                for k,v in dict(mae_seconds=mean_absolute_error(y,pred),medae_seconds=median_absolute_error(y,pred),rmse_seconds=math.sqrt(mean_squared_error(y,pred)),r2=r2_score(y,pred)).items():near(summary[k],v)
        for r in replay:
            if r["context"]!=context:continue
            expected=seed_saved[(r["id"],r["method"],r["seed"])] if r["seed"]!="median3" else float(saved[(r["id"],r["method"])]["prediction_seconds"])
            near(r["saved_seconds"],expected)
    for receipt in OUT.glob("*_receipt.json"):
        if not receipt.name.startswith(("timing_","replay_")):continue
        r=read(receipt)
        if r["status"]!="PASS" or r["predeclaration_sha256"]!=sha(OUT/"predeclaration.json"):raise AssertionError("execution_receipt")
        for key in ("host_pid","child_pid"):
            pid=r["environment"][key]
            cmdline=Path(f"/proc/{pid}/cmdline")
            if cmdline.exists() and b"run_am1_phase5_dense_b2" in cmdline.read_bytes():raise AssertionError("worker_remaining")
    size=sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    if size>plan["limits"]["output_bytes"] or shutil.disk_usage(ROOT).free<plan["limits"]["disk_floor_bytes"]:raise AssertionError("disk_guard")
    tests=read(OUT/"targeted_tests.json")
    if tests["exit_code"]!=0:raise AssertionError("focused_tests")
    if subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True).returncode:raise AssertionError("git_diff_check")
    return dict(status="PASS",source_pins=len(plan["source_pins"]),required_files=len(REQUIRED),replay_rows=480,timed_rows=3600,paired_context_requests=60,
                independent_metrics="sklearn MAE/MedAE/RMSE/R2 versus frozen predictions, exact30 and full150 per context; atol1e-12 rtol0",
                exact_input_digests=True,seed_ensemble_identity=True,stage_partition_nanosecond_exact=True,workers_remaining=False,
                artifact_bytes_before_seal=size,no_training_or_quantum_workloads=True,no_commit_push=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--seal",action="store_true");args=parser.parse_args()
    result=audit()
    if args.seal:
        write_json("independent_validation.json",result)
        payload={str(p):sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
        codes=[SCRIPTS/"run_am1_phase5_dense_b2.py",SCRIPTS/"analyze_am1_phase5_dense_b2.py",Path(__file__),ROOT/"benchmark_v1/tests/test_am1_phase5_dense_b2.py"]
        write_json("provenance_manifest.json",dict(status="SUCCESS",required_files=REQUIRED,predeclaration_sha256=sha(OUT/"predeclaration.json"),
                   payload=payload,code_pins={str(p):sha(p) for p in codes},source_pin_count=result["source_pins"],
                   post_measurement_code="Independent saved-evidence validator added after predeclaration; runner/analyzer/test pins unchanged",no_training_quantum_workloads_commit_push=True))
    elif (OUT/"provenance_manifest.json").exists():
        p=read(OUT/"provenance_manifest.json");verify(p["payload"]);verify(p["code_pins"])
    print(result)
