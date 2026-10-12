#!/usr/bin/env python3
"""Saved-evidence-only dense accuracy/cost analysis. Never calls estimator."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import statistics
import subprocess

import numpy as np
from scipy.stats import spearmanr, pearsonr
from analyze_am1_phase4 import metrics, cluster_bootstrap, stage_partition, md_table
from run_am1_phase5_dense_b2 import OUT, ROOT, DENSE, CONTEXTS, METHODS, SEEDS, PHASE2, folder, count, passing, read, rows, sha, verify, write_csv, write_json, write_md


def saved_context(precision):
    all_rows=[]
    for fold in range(5):
        all_rows.extend(rows(folder(precision,fold)/"predictions.csv"))
    result={}
    for method in METHODS:
        selected=[r for r in all_rows if r["method_id"]==method]
        if len(selected)!=150 or len({r["qasm_sha256"] for r in selected})!=150 or any(r["status"]!="ok" or r["precision"]!=precision or r["prediction_aggregation"]!="rowwise_median" for r in selected):
            raise RuntimeError("full_context_prediction_contract")
        result[method]={r["qasm_sha256"]:r for r in selected}
    if set(result[METHODS[0]])!=set(result[METHODS[1]]): raise RuntimeError("unpaired_full_context")
    targets={r["qasm_sha256"]:r for r in rows(ROOT/"work/benchmark_recovery/r5_dense/targets_recovered_v2/dense_hash_targets.csv") if r["precision"]==precision and r["stratum"]=="core_q2_q9"}
    for fold in range(5):
        seed_rows=rows(folder(precision,fold)/"seed_predictions.csv")
        for method in METHODS:
            grouped=defaultdict(list)
            for r in seed_rows:
                if r["method_id"]==method: grouped[r["qasm_sha256"]].append(r)
            for rid, values in grouped.items():
                if {int(r["seed"]) for r in values}!=set(SEEDS) or len(values)!=3:
                    raise RuntimeError("saved_seed_denominator")
                wanted=statistics.median(float(r["prediction_seconds"]) for r in values)
                if abs(wanted-float(result[method][rid]["prediction_seconds"]))>1e-12:
                    raise RuntimeError("frozen_median3_mismatch")
    for rid in result[METHODS[0]]:
        a,b=result[METHODS[0]][rid],result[METHODS[1]][rid]
        if a["fold"]!=b["fold"] or float(a["target_seconds"])!=float(b["target_seconds"]) or float(a["target_seconds"])!=float(targets[rid]["target_seconds"]):
            raise RuntimeError("target_or_fold_identity")
    return result


def validate():
    plan=read(OUT/"predeclaration.json"); n=verify(plan["source_pins"]); errors=[]; wall=0
    all_replay=[]; all_timings=[]
    for precision in CONTEXTS:
        ids={r["id"] for r in plan["selected"] if r["context"]=="dense_"+precision}
        if len(ids)!=30: errors.append("cohort:"+precision)
        replay=rows(OUT/f"replay_{precision}.csv"); timings=rows(OUT/f"timing_{precision}.csv")
        if len(replay)!=240 or {(r["id"],r["method"],r["seed"]) for r in replay}!={(rid,m,str(s)) for rid in ids for m in METHODS for s in (*SEEDS,"median3")}:
            errors.append("replay_denominator:"+precision)
        if any(r["status"]!="PASS" or not passing(float(r["replayed_seconds"]),float(r["saved_seconds"])) for r in replay): errors.append("replay_parity")
        if len(timings)!=1800 or {(r["id"],r["method"],r["repetition"]) for r in timings}!={(rid,m,str(rep)) for rid in ids for m in METHODS for rep in range(30)}:
            errors.append("timing_denominator:"+precision)
        orders=defaultdict(list)
        for r in timings:
            numeric={k:int(v) for k,v in r.items() if k.endswith("_ns")}
            if sum(stage_partition(numeric).values())!=numeric["total_B2_ns"]: errors.append("stage_partition")
            if any(v<0 for v in numeric.values()) or numeric["total_B2_ns"]>30*10**9: errors.append("timing_bounds")
            if r["context"]!="dense_"+precision or r["status"]!="PASS": errors.append("context_status")
            if r["method"]==METHODS[1] and int(r["cpu_construct_DAG_ns"])!=0: errors.append("MLP_unnecessary_DAG")
            orders[(r["id"],r["method"])].append(int(r["order_position"]))
        if any(Counter(v)!={0:15,1:15} for v in orders.values()): errors.append("order_balance")
        for phase in ("replay","timing"):
            receipt=read(OUT/f"{phase}_{precision}_receipt.json"); res=receipt["resources"]
            if receipt["status"]!="PASS" or receipt["predeclaration_sha256"]!=sha(OUT/"predeclaration.json"): errors.append("receipt_pin")
            if receipt["environment"]["CUDA_smoke"]!="PASS" or receipt["environment"]["model_count"]!=30: errors.append("CUDA_model_setup")
            if max(res["host_peak_rss_bytes"],res["peak_aggregate_rss_bytes"])>plan["limits"]["ram_bytes"]: errors.append("RAM")
            conservative=res["host_peak_rss_bytes"]+max([int(r["child_peak_rss_bytes"]) for r in timings] or [res["peak_child_rss_bytes"]])
            if conservative>plan["limits"]["ram_bytes"]: errors.append("conservative_RAM_bound")
            if max(res["gpu_peak_reserved_bytes"],res["gpu_peak_allocated_bytes"])>plan["limits"]["vram_bytes"]: errors.append("VRAM")
            if res["child_exit_code"]!=0 or not res["subprocess_reaped"]: errors.append("child_not_reaped")
            wall+=res["wall_seconds"]
        all_replay.extend(replay); all_timings.extend(timings); saved_context(precision)
    if wall>plan["limits"]["wall_seconds"]: errors.append("aggregate_wall")
    check=subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True)
    if check.returncode: errors.append("git_diff_check")
    value=dict(status="PASS" if not errors else "FAIL",errors=errors,source_pins_verified=n,replay_comparisons=len(all_replay),timed_requests=len(all_timings),ids_per_context=30,methods=2,seeds=3,repetitions=30,aggregate_execution_seconds=wall,git_diff_check_exit=check.returncode)
    if errors: raise RuntimeError(str(value))
    return value


def analyze():
    checks=validate(); plan=read(OUT/"predeclaration.json")
    replay=sum((rows(OUT/f"replay_{p}.csv") for p in CONTEXTS),[])
    timings=sum((rows(OUT/f"timing_{p}.csv") for p in CONTEXTS),[])
    write_csv("replay_validation.csv",replay); write_csv("raw_b2_timings.csv",timings)
    all_paired=[]; summaries=[]; stages=[]; shares=[]; results={}; full=[]
    for precision in CONTEXTS:
        selected=[r for r in plan["selected"] if r["context"]=="dense_"+precision]
        saved=saved_context(precision); by=defaultdict(list)
        for t in timings:
            if t["context"]=="dense_"+precision: by[(t["id"],t["method"])].append(t)
        paired=[]
        for r in selected:
            rid=r["id"]; target=float(saved[METHODS[0]][rid]["target_seconds"])
            a=dict(context="dense_"+precision,id=rid,group=r["group"],family=r["family"],fold=int(r["fold"]),target_seconds=target,
                   **{k:count(r[k]) for k in ("width","depth","gates","nodes","edges")},median_reference_seconds=plan["constants"][precision][r["fold"]])
            for name,method in zip(("graph","mlp"),METHODS):
                p=float(saved[method][rid]["prediction_seconds"]); latency=[int(t["total_B2_ns"])/1e6 for t in by[(rid,method)]]
                a.update({name+"_prediction_seconds":p,name+"_signed_residual_seconds":p-target,name+"_abs_error_seconds":abs(p-target),
                          name+"_B2_median_ms":statistics.median(latency),name+"_B2_p95_ms":float(np.quantile(latency,.95)),name+"_B2_min_ms":min(latency),name+"_B2_max_ms":max(latency)})
            a["paired_latency_ratio"]=a["graph_B2_median_ms"]/a["mlp_B2_median_ms"]
            a["paired_latency_difference_ms"]=a["graph_B2_median_ms"]-a["mlp_B2_median_ms"]
            a["paired_abs_error_difference_seconds"]=a["graph_abs_error_seconds"]-a["mlp_abs_error_seconds"]
            paired.append(a)
        all_paired.extend(paired)
        context_summary=[]
        for name in ("graph","mlp","median_reference"):
            key=name+"_prediction_seconds" if name!="median_reference" else "median_reference_seconds"
            value=dict(context="dense_"+precision,population="exact_timed_subset",method=name,**metrics([r["target_seconds"] for r in paired],[r[key] for r in paired]))
            value.update(median_request_median_B2_ms=statistics.median(r[name+"_B2_median_ms"] for r in paired) if name!="median_reference" else "NOT_MEASURED",
                         median_request_p95_B2_ms=statistics.median(r[name+"_B2_p95_ms"] for r in paired) if name!="median_reference" else "NOT_MEASURED")
            context_summary.append(value); summaries.append(value)
        for name,method in zip(("graph","mlp"),METHODS):
            records=list(saved[method].values())
            full.append(dict(context="dense_"+precision,population="full_frozen_core_OOF",method=name,**metrics([float(r["target_seconds"]) for r in records],[float(r["prediction_seconds"]) for r in records])))
            trows=[t for t in timings if t["context"]=="dense_"+precision and t["method"]==method]
            denominator=sum(int(t["total_B2_ns"]) for t in trows); category=defaultdict(int)
            for t in trows:
                for k,v in stage_partition({k:int(v) for k,v in t.items() if k.endswith("_ns")}).items(): category[k]+=v
            shares.append(dict(context="dense_"+precision,method=method,denominator_total_ns=denominator,shares={k:v/denominator for k,v in category.items()},stage_sums_ns=dict(category)))
            for key in sorted(k for k in trows[0] if k.endswith("_ns")):
                medians=[statistics.median(int(t[key])/1e6 for t in by[(r["id"],method)]) for r in selected]
                stages.append(dict(context="dense_"+precision,method=method,stage=key,queries=30,repetitions_per_query=30,median_query_median_ms=statistics.median(medians),min_query_median_ms=min(medians),max_query_median_ms=max(medians),
                    semantics="CPU_nested_inside_host_wait" if key.startswith("cpu_") else "inclusive_parent" if key=="host_response_wait_inclusive_ns" else "direct_total" if key=="total_B2_ns" else "exclusive_or_residual"))
        graph,mlp=context_summary[:2]; ci=cluster_bootstrap(paired)
        differences=dict(observed_mae_graph_minus_mlp_seconds=graph["mae_seconds"]-mlp["mae_seconds"],median_paired_latency_difference_ms=statistics.median(r["paired_latency_difference_ms"] for r in paired),
            ratio_of_marginal_latency_medians=graph["median_request_median_B2_ms"]/mlp["median_request_median_B2_ms"],median_of_paired_latency_ratios=statistics.median(r["paired_latency_ratio"] for r in paired))
        correlations=[]
        for name in ("graph","mlp"):
            for feature in ("width","depth","gates","nodes","edges"):
                x=[r[feature] for r in paired]; y=[r[name+"_B2_median_ms"] for r in paired]
                correlations.append(dict(method=name,feature=feature,min=min(x),max=max(x),spearman=float(spearmanr(x,y).statistic) if len(set(x))>1 else None,pearson=float(pearsonr(x,y).statistic) if len(set(x))>1 else None))
        historical={name:dict(label="historical cross-environment simulator runtime",n=30,median_B2_over_historical_execution=statistics.median(r[name+"_B2_median_ms"]/1000/r["target_seconds"] for r in paired),
            B2_above_historical_execution_n=sum(r[name+"_B2_median_ms"]/1000>r["target_seconds"] for r in paired)) for name in ("graph","mlp")}
        result=dict(summary=context_summary,paired=differences,uncertainty=ci,scaling=correlations,historical_execution=historical,families=dict(Counter(r["family"] for r in paired)))
        results[precision]=result
        write_md(precision+"_matched_b2.md","# "+precision.upper()+" matched full B2\n\n"+md_table(context_summary,["method","n","mae_seconds","rmse_seconds","r2","median_request_median_B2_ms","median_request_p95_B2_ms"])+
            "\nDirect total from resident raw QASM to returned scalar at CUDA coordinator. Each circuit/model has3 warmups and30 repetitions, batch1. p95 is median of30 per-circuit interpolated p95 values, not pooled production tail. No outlier deletion.\n\nPaired statistics:\n"+json_text(differences)+"\n\nConditional paired cluster bootstrap:\n"+json_text(ci))
    write_csv("paired_circuit_results.csv",all_paired); write_csv("accuracy_latency_summary.csv",summaries); write_csv("full_context_accuracy.csv",full); write_csv("stage_timing_summary.csv",stages)
    resources={phase+"_"+p:read(OUT/f"{phase}_{p}_receipt.json")["resources"] for p in CONTEXTS for phase in ("replay","timing")}
    analysis=dict(status="SUCCESS",validation=checks,contexts=results,full_context_accuracy=full,shares=shares,resources=resources)
    write_json("analysis.json",analysis)
    write_md("README.md","# Dense Graph–MLP estimator accuracy–cost\n\nSUCCESS. Both precisions pass raw-input replay and matched B2:60 context requests,480 replay comparisons,3,600 timed predictions. No training or quantum simulation.\n\n"+md_table([r for r in summaries if r["method"]!="median_reference"],["context","method","mae_seconds","rmse_seconds","r2","median_request_median_B2_ms","median_request_p95_B2_ms"])+"\nRead accuracy_cost_tradeoff.md, matched_accuracy_analysis.md and rq4_trackb_decision.md. Exact inputs/checkpoints and source hashes:predeclaration.json. Each precision is a separate target. Original models are float32 for both simulator precision contexts. Full context is150 frozen core OOF circuits; timing subset is30.\n")
    write_md("replay_validation.md","# Full frozen replay\n\nFP32 Graph30/30, MLP30/30; FP64 Graph30/30, MLP30/30. 360 seed comparisons plus120 median3 comparisons. Model-ready digests match independent Phase3 frozen arrays, including shapes/dtypes/edge indices. Predictions pass atol1e-6 seconds+rtol1e-5. Transforms applied once. Original output expm1/clamp preserved. CUDA smoke and60 checkpoint identities verified. No CPU inference fallback.\n")
    write_md("environment_manifest.md","# Environment\n\n"+json_text(read(OUT/"gate0.json"))+"\n\nOne CUDA coordinator+one CPU preprocessing child. Affinity8/9, native threads1,Torch threads2/interop1. Qiskit2.5.2/NumPy2.2.6 provide verified compatibility, not proof of unknown original producer package versions. Torch2.7.1+cu128/PyG2.6.1/NumPy1.26.4 consumer. RustDesk desktop activity remains disclosed; no other benchmark workload. JSON/base64 stdio implementation-specific cost. No feature cache. Imports/startup/model loading/input disk reads excluded; setup recorded separately.\n")
    write_md("artifact_integrity.md","# Frozen source integrity\n\n"+str(checks["source_pins_verified"])+" source pins verified before and after execution. Includes completed Phase3 compatibility code/payload, Phase4 evidence, raw QASM, R3 features/IR,R5 targets,60 checkpoints,fold transforms,seed and median predictions,producer/protocol/environment and new tooling. Phase3 final provenance is code authority; its earlier acceptance snapshot is not repinned. No historical artifacts modified. Exact source mapping:predeclaration.json source_pins.\n")
    write_md("matched_accuracy_analysis.md","# Frozen predictive accuracy\n\nTimed subset:\n\n"+md_table(summaries,["context","method","n","mae_seconds","medae_seconds","rmse_seconds","r2","mean_signed_residual_seconds"])+"\nFull frozen core context (150 OOF/context, not204 envelope or30 subset):\n\n"+md_table(full)+"\n\nAccuracy uses frozen predictions and R5 warm sample(32) targets. Timing estimates are replay checks, not newly selected predictions. Reference uses already-frozen outer-train medians, no test-label fitting. Paired errors, signed residuals and target seconds:paired_circuit_results.csv. Cluster CIs use30 existing QASM-hash groups/context and2,000 replicates seed20261011. Repetitions are not independent circuits. Subset chosen for structural coverage, not population-random inference. Full-context scores unchanged.\n")
    paragraphs=[]
    for precision,result in results.items():
        ci=result["uncertainty"]["mae_difference_graph_minus_mlp_seconds"]
        text="Conditional accuracy interval contains zero; no stable accuracy advantage." if ci[0]<=0<=ci[1] else "Conditional accuracy interval excludes zero on this subset only."
        paragraphs.append(precision.upper()+": "+json_text(result["paired"])+"\n"+text)
    write_md("accuracy_cost_tradeoff.md","# Accuracy and full online estimator cost\n\n"+"\n\n".join(paragraphs)+"\n\nCost is standalone genuine per-method work at the same external boundary. Graph pays DAG construction/encoding/large-payload bridge; MLP omits DAG only. Both pay raw parsing,RCCX normalization,seven globals and frozen transforms. Stage evidence distinguishes preparation and inference; no stage medians are summed. Source/family mixture and small target variance limit accuracy interpretation. No utility score,Pareto-frontier,universal winner or scheduler claim. Cached Pilot B1(approximately3.23/0.49ms FP32 and3.21/0.49ms FP64) omits preparation and bridge. Warmup/instrumentation/environment differ, so no exact matched B2/B1 speedup is asserted.\n")
    write_md("within_range_scaling.md","# Within-range empirical associations\n\n"+"\n\n".join(p.upper()+"\n\n"+md_table(r["scaling"])+"\nFamilies:"+str(r["families"]) for p,r in results.items())+"\n\nThirty circuits/context. Spearman and Pearson are descriptive, not causal tests. Gate and node counts coincide by representation, so not independent evidence. Width/depth/family/gate type are confounded. No asymptotic or universal law. No new circuit generation or elaborate fit.\n")
    write_md("historical_execution_context.md","# Historical cross-environment simulator runtime\n\n"+json_text({p:r["historical_execution"] for p,r in results.items()})+"\n\nExact QASM-hash/precision/seconds joins to frozen R5 warm sample(32) labels. Simulator historical context differs in software/time from Phase5 estimator measurement. Ratios are descriptive only, not matched execution runtime, operational break-even or a reason to run simulator workloads. No FP32/FP64 collapsed comparison; selected circuit sets also differ.\n")
    write_md("memory_and_resources.md","# Resources\n\n"+json_text(resources)+"\n\nAll four replay/timing processes sequential. Aggregate RSS sampled50ms, per-process high-water retained; peaks may not coincide. CUDA allocated/reserved include setup but exclude desktop board memory. RAM refers to physical RSS, not virtual address space. Caps:8GiB aggregate RSS/VRAM,7,200s combined execution,256MiB outputs,3GiB diskfloor,30s request,120s setup. Startup not in B2. Timer-pair/empty-echo controls retained without subtraction; instrumentation and monitor overhead remain included. Empty echo is not real-payload IPC estimate. Children reaped.\n")
    write_md("failures_and_exclusions.md","# Failures and exclusions\n\nNo replay/timing failures. No circuit dropped. Both exact30/context cohorts unchanged. 3,600 retained timed requests plus360 full-pipeline warmups and120 replay requests. No simulator/training/QPU/QCEC,dependency change or new circuits. No outlier deletion. Unavailable classical fitted states not replaced. Historical phase evidence preserved.\n")
    write_md("rq4_trackb_decision.md","# RQ4 Track B decision\n\nANSWERED WITHIN DENSE CONTEXT. Full raw-to-output costs and frozen accuracy are paired for Graph/MLP in FP32/FP64 on exactly30 circuits each. This closes this bounded amendment, not all simulator contexts,competitive classical frontier,hardware portability,cold-start,B3,production benefit or universal scaling. Accuracy uncertainty and150/30 denominators remain explicit.\n\nNext action:CLOSE DENSE GRAPH–MLP AM1. No automatic additional experiment.\n")
    write_json("validation.json",checks)
    write_md("validation_report.md","# Validation\n\n"+json_text(checks)+"\n\nExact replay/timing keys,context precision,balanced order,unit/shape/dtype digests,seed outputs,checkpoint identities,unchanged pins,nested stage reconciliation and caps checked on saved results. Fresh focused tests and independent numerical audit recorded separately. No previous-phase PASS substitutes for this execution.\n")
    print(json_text({p:dict(paired=r["paired"],summary=r["summary"]) for p,r in results.items()}))


def json_text(value):
    import json
    return json.dumps(value,sort_keys=True,indent=2,allow_nan=False)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["analyze","validate"],required=True);a=p.parse_args()
    if a.stage=="analyze":analyze()
    else:print(json_text(validate()))
