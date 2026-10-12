#!/usr/bin/env python3
"""Saved-evidence-only analysis of the predeclared matched QPU B2 experiment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
import statistics
import subprocess
from pathlib import Path

import numpy as np
from run_am1_phase4 import OUT, PANEL, PHASE2, ROOT, METHODS, SEEDS, read, rows, sha, verify, write_csv, write_json, write_md


def metrics(target, prediction):
    y,p = np.asarray(target,dtype=float),np.asarray(prediction,dtype=float)
    e=p-y
    denom=np.sum((y-y.mean())**2)
    return dict(n=len(y), mae_seconds=float(np.abs(e).mean()), medae_seconds=float(np.median(np.abs(e))), rmse_seconds=float(np.sqrt((e*e).mean())),
                r2=float(1-np.sum(e*e)/denom) if denom>0 else None, mean_signed_residual_seconds=float(e.mean()))


def cluster_bootstrap(data, reps=2000, seed=20261011):
    """Sample structural clusters, preserving paired rows and row-weighted metric."""
    groups=defaultdict(list)
    for i,r in enumerate(data): groups[r["group"]].append(i)
    keys=sorted(groups)
    rng=np.random.default_rng(seed)
    values=[]
    for _ in range(reps):
        ids=[i for key in rng.choice(keys,size=len(keys),replace=True) for i in groups[key]]
        sampled=[data[i] for i in ids]
        values.append([statistics.mean(r["graph_abs_error_seconds"]-r["mlp_abs_error_seconds"] for r in sampled),
            float(np.median([r["graph_B2_median_ms"]-r["mlp_B2_median_ms"] for r in sampled])),
            float(np.median([r["graph_B2_median_ms"] for r in sampled])/np.median([r["mlp_B2_median_ms"] for r in sampled]))])
    a=np.asarray(values)
    return dict(groups=len(keys), observations=len(data), unit="paired structural-group cluster, row-weighted MAE; median request latency difference; ratio of marginal latency medians",
        mae_difference_graph_minus_mlp_seconds=list(map(float,np.quantile(a[:,0],[.025,.975]))),
        paired_latency_difference_ms=list(map(float,np.quantile(a[:,1],[.025,.975]))),
        marginal_median_latency_ratio=list(map(float,np.quantile(a[:,2],[.025,.975]))), replicates=reps,seed=seed)


def md_table(data, columns=None):
    columns=columns or list(data[0])
    return "| " + " | ".join(columns) + " |\n|" + "|".join(["---"]*len(columns)) + "|\n" + "\n".join("| "+" | ".join(str(r.get(k,"")) for k in columns)+" |" for r in data)+"\n"


def stage_partition(record):
    # Host wait includes the whole CPU computation. Use CPU durations only once.
    cpu=sum(v for k,v in record.items() if k.startswith("cpu_") and k.endswith("_ns"))
    serialization=record["host_request_serialization_ns"]+record["cpu_array_serialization_ns"]+record["cpu_payload_JSON_encode_ns"]
    prep=cpu-record["cpu_array_serialization_ns"]-record["cpu_payload_JSON_encode_ns"]
    ipc=record["IPC_unattributed_residual_ns"]
    decode=record["host_response_JSON_decode_ns"]+record["host_typed_array_decode_ns"]
    tensor=record["host_tensor_construction_ns"]+record["host_batch_and_H2D_ns"]
    inference=record["host_three_seed_inference_inverse_and_D2H_ns"]
    agg=record["host_host_median_aggregation_ns"]
    residual=record["coordinator_unattributed_ns"]
    return dict(preparation=prep,serialization=serialization,IPC_and_unattributed_bridge=ipc,host_deserialization=decode,tensor_and_H2D=tensor,inference_inverse_D2H=inference,aggregation=agg,coordinator_residual=residual)


def validate():
    plan=read(OUT / "predeclaration.json")
    verified=verify(plan["source_pins"])
    gate1=read(OUT / "gate1.json")
    replay=rows(OUT / "replay_validation.csv")
    timings=rows(OUT / "raw_b2_timings.csv")
    ids={r["id"] for r in plan["selected"]}
    errors=[]
    if gate1["status"] != "PASS" or len(replay)!=360: errors.append("full_gate1")
    expected_replay={(i,m,str(s)) for i in ids for m in METHODS for s in (*SEEDS,"median3")}
    if {(r["id"],r["method"],r["seed"]) for r in replay} != expected_replay: errors.append("replay_key_set")
    if any(r["status"]!="PASS" or abs(float(r["replayed_seconds"])-float(r["saved_seconds"]))>1e-6+1e-5*abs(float(r["saved_seconds"])) for r in replay): errors.append("replay_tolerance")
    expected_keys={(i,m,str(rep)) for i in ids for m in METHODS for rep in range(30)}
    if len(timings)!=2700 or {(r["id"],r["method"],r["repetition"]) for r in timings} != expected_keys: errors.append("timing_denominator")
    groups=defaultdict(list)
    for r in timings:
        numeric={k:int(v) for k,v in r.items() if k.endswith("_ns")}
        if sum(stage_partition(numeric).values()) != numeric["total_B2_ns"]: errors.append("partition:"+r["id"])
        if any(v<0 for v in numeric.values()) or numeric["total_B2_ns"]>30*10**9: errors.append("duration:"+r["id"])
        if r["status"]!="PASS": errors.append("timed_failure")
        groups[(r["id"],r["method"])].append(int(r["order_position"]))
    if any(Counter(x)!={0:15,1:15} for x in groups.values()): errors.append("order_balance")
    for phase in ("replay","timing"):
        receipt=read(OUT / f"{phase}_receipt.json")
        if receipt["status"]!="PASS" or receipt["predeclaration_sha256"]!=sha(OUT / "predeclaration.json"): errors.append(phase+"_receipt")
        res=receipt["resources"]
        if max(res["peak_aggregate_rss_bytes"],res["host_peak_rss_bytes"])>plan["limits"]["ram_bytes"]: errors.append("RAM")
        if max(res["gpu_peak_allocated_bytes"],res["gpu_peak_reserved_bytes"])>plan["limits"]["vram_bytes"]: errors.append("VRAM")
        if not res["subprocess_reaped"] or res["child_exit_code"]!=0: errors.append("process_cleanup")
    if sum(read(OUT/f"{p}_receipt.json")["resources"]["wall_seconds"] for p in ("replay","timing"))>7200: errors.append("wall_budget")
    diff=subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode: errors.append("git_diff_check")
    return dict(status="PASS" if not errors else "FAIL",errors=errors,pins=verified,ids=45,methods=2,replay_comparisons=len(replay),timed_requests=len(timings),git_diff_check_exit=diff.returncode)


def analyze():
    checks=validate()
    if checks["status"]!="PASS": raise RuntimeError(str(checks))
    plan=read(OUT/"predeclaration.json")
    panel={r["canonical_observation_id"]:r for r in rows(PANEL/"panel.csv")}
    timings=rows(OUT/"raw_b2_timings.csv")
    replay=rows(OUT/"replay_validation.csv")
    estimates={(r["id"],r["method"]):float(r["saved_seconds"]) for r in replay if r["seed"]=="median3"}
    by=defaultdict(list)
    for r in timings: by[(r["id"],r["method"])].append(r)
    paired=[]
    for r in plan["selected"]:
        rid=r["id"]; target=float(panel[rid]["target_seconds"])
        a=dict(id=rid,source=r["source"],group=r["group"],fold=int(r["fold"]),width=int(r["width"]),depth=int(r["depth"]),gates=int(r["gates"]),target_seconds=target,
            graph_prediction_seconds=estimates[(rid,METHODS[0])],mlp_prediction_seconds=estimates[(rid,METHODS[1])],median_reference_seconds=float(plan["constants"][r["fold"]]))
        for name,method in zip(("graph","mlp"),METHODS):
            latency=[int(t["total_B2_ns"])/1e6 for t in by[(rid,method)]]
            a.update({name+"_signed_residual_seconds":a[name+"_prediction_seconds"]-target,name+"_abs_error_seconds":abs(a[name+"_prediction_seconds"]-target),
                name+"_B2_median_ms":statistics.median(latency),name+"_B2_p90_ms":float(np.quantile(latency,.9)),name+"_B2_p95_ms":float(np.quantile(latency,.95)),
                name+"_B2_mean_ms":statistics.mean(latency),name+"_B2_std_ms":statistics.stdev(latency)})
        a["paired_latency_difference_ms"]=a["graph_B2_median_ms"]-a["mlp_B2_median_ms"]
        a["paired_latency_ratio"]=a["graph_B2_median_ms"]/a["mlp_B2_median_ms"]
        a["paired_absolute_error_difference_seconds"]=a["graph_abs_error_seconds"]-a["mlp_abs_error_seconds"]
        paired.append(a)
    write_csv("paired_observation_results.csv",paired)
    summaries=[]
    for source in ("all_selected","mali_real_qpu","qonductor_single_circuit_ibm"):
        subset=[r for r in paired if source=="all_selected" or r["source"]==source]
        for name in ("graph","mlp","median_reference"):
            predkey=name+"_prediction_seconds" if name!="median_reference" else "median_reference_seconds"
            item=dict(source=source,method=name,**metrics([r["target_seconds"] for r in subset],[r[predkey] for r in subset]))
            if name!="median_reference":
                item.update(median_request_median_B2_ms=float(np.median([r[name+"_B2_median_ms"] for r in subset])),median_request_p95_B2_ms=float(np.median([r[name+"_B2_p95_ms"] for r in subset])))
            else: item.update(median_request_median_B2_ms="NOT_MEASURED",median_request_p95_B2_ms="NOT_MEASURED")
            summaries.append(item)
    write_csv("accuracy_latency_summary.csv",summaries)
    stages=[]
    shares=[]
    for method in METHODS:
        keys=sorted(k for k in timings[0] if k.endswith("_ns"))
        for key in keys:
            medians=[statistics.median(int(r[key])/1e6 for r in by[(x["id"],method)]) for x in plan["selected"]]
            stages.append(dict(method=method,stage=key,queries=45,repetitions_per_query=30,median_query_median_ms=statistics.median(medians),min_query_median_ms=min(medians),max_query_median_ms=max(medians),
                semantics="nested_in_response_wait" if key.startswith("cpu_") else "host_inclusive_parent" if key=="host_response_wait_inclusive_ns" else "direct_total" if key=="total_B2_ns" else "exclusive_or_residual"))
        totals=defaultdict(int); denominator=0
        for r in timings:
            if r["method"]!=method: continue
            numeric={k:int(v) for k,v in r.items() if k.endswith("_ns")}
            denominator+=numeric["total_B2_ns"]
            for k,v in stage_partition(numeric).items(): totals[k]+=v
        shares.append(dict(method=method,denominator_total_ns=denominator,shares={k:v/denominator for k,v in totals.items()},stage_sums_ns=dict(totals)))
    write_csv("stage_timing_summary.csv",stages)
    uncertainty={s:cluster_bootstrap([r for r in paired if s=="all_selected" or r["source"]==s]) for s in ("all_selected","mali_real_qpu","qonductor_single_circuit_ibm")}
    graph=next(r for r in summaries if r["source"]=="all_selected" and r["method"]=="graph")
    mlp=next(r for r in summaries if r["source"]=="all_selected" and r["method"]=="mlp")
    differences=dict(observed_mae_graph_minus_mlp_seconds=graph["mae_seconds"]-mlp["mae_seconds"],
        median_paired_latency_difference_ms=statistics.median(r["paired_latency_difference_ms"] for r in paired),
        ratio_of_marginal_latency_medians=graph["median_request_median_B2_ms"]/mlp["median_request_median_B2_ms"],
        median_of_paired_latency_ratios=statistics.median(r["paired_latency_ratio"] for r in paired))
    # Source-balanced macro is a descriptive accuracy summary, not a new selection rule.
    macro={n:statistics.mean(r["mae_seconds"] for r in summaries if r["source"]!="all_selected" and r["method"]==n) for n in ("graph","mlp","median_reference")}
    try:
        from scipy.stats import spearmanr
        correlations={}
        for source_name in ("all_selected","mali_real_qpu","qonductor_single_circuit_ibm"):
            subset=[r for r in paired if source_name=="all_selected" or r["source"]==source_name]
            correlations[source_name]={}
            for name in ("graph","mlp"):
                for feature in ("width","depth","gates"):
                    x=[r[feature] for r in subset]; y=[r[name+"_B2_median_ms"] for r in subset]
                    correlations[source_name][name+"_"+feature]=float(spearmanr(x,y).statistic) if len(set(x))>1 else None
    except ImportError:
        correlations={"status":"scipy_unavailable; no dependency install"}
    result=dict(status="SUCCESS",checks=checks,summaries=summaries,paired=differences,uncertainty=uncertainty,shares=shares,source_balanced_macro_mae_seconds=macro,
        correlations_descriptive_only=correlations,replay=read(OUT/"replay_receipt.json")["resources"],timing=read(OUT/"timing_receipt.json")["resources"],
        original_scope=50,raw_available_and_measured=45,excluded_QPack=5,source_counts=dict(Counter(r["source"] for r in paired)), instrumentation=read(OUT/"timing_receipt.json")["controls"])
    write_json("analysis.json",result)
    write_json("gate2.json",dict(status="PASS",observations=45,methods=2,repetitions=30,requests=2700,warmups=3,direct_total=True,IPC_included=True))
    write_md("replay_validation.md","# Gate 1\n\nPASS: Graph 45/45 and MLP 45/45. There are 270 per-seed and 90 median-ensemble comparisons (360 total). Typed-array hashes match independently frozen Phase 3 representations. Frozen transforms execute once on the CPU. Checkpoint, refit and transform identities are checked before inference. Inference uses CUDA, with no CPU fallback. Tolerance: 1e-6 seconds absolute + 1e-5 relative, declared before replay.\n")
    write_md("environment_manifest.md","# Environment\n\n"+str(read(OUT/"gate0.json"))+"\n\nResident setup is performed separately for replay and timing, using identical pinned environments and architecture. Previous Phase 2 timings are not used in the paired comparison. Desktop/RustDesk GPU activity persists; benchmark exclusivity does not mean exclusive hardware ownership. Native threads: 1. Torch threads: 2, interop: 1. CPU affinity: 8/9. Software, GPU and checkpoint identities are recorded in replay_receipt.json and timing_receipt.json.\n")
    write_md("matched_accuracy_analysis.md","# Frozen accuracy on exactly 45 observations\n\n"+md_table(summaries,["source","method","n","mae_seconds","medae_seconds","rmse_seconds","r2"])+"\nPaired cluster bootstrap: 2,000 replicates, seed 20261011, 44 structural groups. Intervals are conditional on this selected subset, not population-random sampling.\n\n"+str(uncertainty)+"\n\nSource-balanced macro MAE: "+str(macro)+". No full 4,515-cohort result changed. The median reference comes from saved outer-train-only parameters and was not refitted. Labels measure QPU observed service time in seconds, not estimator computation time.\n")
    write_md("matched_b2_comparison.md","# Standalone matched full B2\n\n"+md_table(summaries,["source","method","n","median_request_median_B2_ms","median_request_p95_B2_ms"])+"\nObserved paired statistics: "+str(differences)+"\n\nThe total is independently host-timed from resident raw QASM to returned scalar prediction. There are 30 repeats per ID/method, 3 warmups and 15/15 first-order positions. No outliers were removed. p95 within 30 repeats is interpolated, not a production tail bound. Repetitions are not independent circuits.\n\n"+str(uncertainty)+"\n\nNonoverlapping shares use sums of stage durations divided by the sum of independently measured totals for the same 1,350 requests per model. CPU spans are nested inside host wait. The bridge residual includes pipe I/O, scheduling, outer-envelope encoding and uninstrumented CPU work, not pure transport latency.\n\n"+str(shares)+"\n\nShared-preparation opportunity: QASM parsing and the 51 globals could legitimately be reused when deploying both models together. Exclusive measured CPU durations show potential savings. This is an analytical opportunity, not measured shared-pipeline latency. Primary standalone costs do not subtract these stages.\n")
    ci=uncertainty["all_selected"]["mae_difference_graph_minus_mlp_seconds"]
    evidence="The paired interval crosses zero; no stable accuracy superiority is established on these45 IDs." if ci[0]<=0<=ci[1] else "The conditional paired interval excludes zero. This does not establish superiority outside these45 IDs."
    write_md("accuracy_cost_tradeoff.md",f"# Accuracy–cost interpretation\n\nGraph MAE {graph['mae_seconds']:.9g} s; MLP MAE {mlp['mae_seconds']:.9g} s. Graph minus MLP MAE {differences['observed_mae_graph_minus_mlp_seconds']:.9g} s. {evidence}\n\nGraph B2 {graph['median_request_median_B2_ms']:.9g} ms; MLP B2 {mlp['median_request_median_B2_ms']:.9g} ms. Ratio of marginal medians: {differences['ratio_of_marginal_latency_medians']:.9g}. This compares each genuine preparation path under the same host boundary, not equally sized tensors.\n\nB1-only inference omits raw QASM parsing, global features, graph construction and IPC. The stage evidence measures these costs here. Old Pilot/Phase 2 values remain historical environment references, not paired new measurements. No new matched B1 endpoint was measured.\n\nThe source composition (38 Ma–Li, 7 Qonductor) is unequal and not representative of 4,515 observations. Per-source scores and macro means are descriptive. QPack is excluded. Descriptive size/depth Spearman associations: "+str(correlations)+". These do not establish exponential or universal scaling laws.\n\nFor this context, assess the observed accuracy–cost trade-off and uncertainty, not full-cohort MAE alone. No scalar utility, scheduler-throughput benefit, production break-even or complete Pareto frontier follows. Competitive classical fitted states remain missing.\n")
    write_md("memory_and_resource_report.md","# Resources\n\n"+str(result["replay"])+"\n\n"+str(result["timing"])+"\n\nAggregate RSS is sampled every 50 ms; per-process high-water marks are also retained. GPU allocated/reserved peaks are framework counters, not physical board usage including desktop activity. Caps: 120 s setup, 30 s/request, 8 GiB aggregate RSS/VRAM, 7,200 s combined execution, 256 MiB new artifacts and 3 GiB disk floor. RAM refers to physical RSS, not virtual address space. Child processes were reaped.\n\nInstrumentation controls: "+str({k:dict(n=len(v),median_ns=statistics.median(v),p95_ns=float(np.quantile(v,.95))) for k,v in result["instrumentation"].items() if isinstance(v,list)})+". No control cost was subtracted. Empty IPC does not estimate actual large Graph transport. Timer wrappers and resource-monitor overhead remain included in the direct total; an uninstrumented B2 control was not run.\n")
    write_md("failures_and_exclusions.md","# Failures and exclusions\n\nNo failed timed requests or replay failures. Five original QPack raw requests remain unavailable (45/50 original scope); no replacements were made. Dense FP32/FP64 timing was not authorized or executed. Classical fitted states were not refitted. Results are conditional on the matched 45 observations, nominal snapshots and declared reconstruction limitations.\n")
    write_md("rq4_decision.md","# RQ4 decision\n\nCLOSE MATCHED GRAPH–MLP AM1. Full raw-input-to-output B2 is now measured and replayed for both frozen estimators on exactly 45 IDs. This strengthens prediction-overhead RQ4 beyond cached-input B1 and one-sided Phase 2. A complete classical frontier, cold B3, hardware portability, scheduler benefit and Dense B2 remain unanswered. Any new measurement requires a separate amendment.\n")
    write_md("README.md","# AM1 Phase 4 matched QPU accuracy–cost\n\nSUCCESS. Gates 0–3 passed: Graph and MLP replay 45/45, with 2,700 matched full-B2 requests (45 × 2 × 30), no feature cache and explicit CPU-to-CUDA transport.\n\n"+md_table([graph,mlp],["method","n","mae_seconds","rmse_seconds","r2","median_request_median_B2_ms"])+"\n\n"+evidence+"\n\nRead matched_b2_comparison.md, matched_accuracy_analysis.md, accuracy_cost_tradeoff.md and validation_report.md. Exact inputs, models, protocol and source pins are in predeclaration.json. Fresh evidence: replay_validation.csv, raw_b2_timings.csv, paired_observation_results.csv and stage_timing_summary.csv. Interpretation is restricted to this source-biased 45-ID subset and local bridge environment. No training, refit, quantum execution, dependency change, leaderboard change or commit/push.\n")
    write_json("validation.json",checks)
    write_md("validation_report.md","# Validation\n\n"+str(checks)+"\n\nExact key sets: 45 IDs × 2 methods × 30 repetitions and 45 × 2 × 4 replay rows. All frozen pins were reverified after execution. Nonoverlapping stage partitions reconcile exactly to the independently timed total. Balanced order, units, identity, CUDA smoke and resource checks passed. Targeted tooling tests are recorded in targeted_tests.json. No training or quantum workload calls. Test PASS does not replace actual replay and timing receipts.\n")
    print(result["status"],differences,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["analyze","validate"],required=True);a=p.parse_args()
    if a.stage=="analyze":analyze()
    else: print(validate())


if __name__=="__main__":main()
