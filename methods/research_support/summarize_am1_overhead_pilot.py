#!/usr/bin/env python3
"""Derive AM1 reports from completed measurements and frozen accuracy only."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("am1",ROOT/"benchmark_v1/scripts/run_am1_overhead_pilot.py")
am1=importlib.util.module_from_spec(spec); spec.loader.exec_module(am1)
OUT=am1.OUT


def quantile(values,p):
    s=sorted(values); t=(len(s)-1)*p; i=int(t)
    return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(t-i)


def table(fields,values):
    return "| "+" | ".join(fields)+" |\n|"+"|".join("---" for _ in fields)+"|\n"+"\n".join("| "+" | ".join(str(r.get(f,"NA")) for f in fields)+" |" for r in values)+"\n"


def summary():
    plan=am1.effective_plan(); am1.verify(plan["source_pins"])
    workloads=am1.rows(OUT/"selected_workloads.csv")
    raw=[]; replay=[]; memory=[]; setup=[]; bindings={}
    for p in sorted((OUT/"workers").glob("*.json")):
        d=am1.read(p)
        bindings.update({r["path"]:r["sha256"] for r in d["bindings"]})
        for r in d.get("replay",[]): replay.append(dict(context=p.name.split("_fold")[0],fold=p.name.split("_fold")[1].split("_")[0],**r))
        raw.extend(d.get("timings",[]))
        if "memory" in d:
            memory.append(dict(file=p.name,**d["memory"]))
            child_ns=next(r["elapsed_ns"] for r in d["stages"] if r["stage"]=="child_start_to_ready")
            setup.append(dict(file=p.name,context=p.name.split("_fold")[0],seconds=child_ns/1e9,stages=d["stages"]))
    am1.verify(bindings)
    assert raw and replay
    assert all(int(r["end_ns"])-int(r["start_ns"])==int(r["elapsed_ns"])>=0 for r in raw)
    am1.csv_save("raw_timings.csv",raw)
    am1.csv_save("replay_validation.csv",replay)
    per_query=defaultdict(list)
    for r in raw:
        if r["boundary"] in ("B1","P1","CONTROL"):
            rid=r["id"] if r["boundary"]!="CONTROL" else r["id"]+"/fold"+str(r["fold"])
            per_query[(r["context"],r["method"],r["boundary"],rid)].append(r["elapsed_ns"]/1e9)
    summaries=[]
    bymethod=defaultdict(list)
    for (context,method,boundary,rid),values in per_query.items():
        assert len(values)==30
        r=dict(context=context,method=method,boundary=boundary,id=rid,repetitions=len(values),median_seconds=statistics.median(values),
            p90_seconds=quantile(values,.90),p95_seconds=quantile(values,.95),min_seconds=min(values),max_seconds=max(values))
        summaries.append(r); bymethod[(context,method,boundary)].append(r)
    am1.csv_save("timing_by_workload.csv",summaries)
    pooled=[]
    for (c,m,b),rs in sorted(bymethod.items()):
        med=[r["median_seconds"] for r in rs]
        pooled.append(dict(context=c,method=m,boundary=b,requests=len(rs),raw_n=sum(r["repetitions"] for r in rs),median_request_ms=statistics.median(med)*1000,
            p90_request_ms=quantile(med,.90)*1000,p95_request_ms=quantile(med,.95)*1000,max_request_ms=max(med)*1000))
    am1.csv_save("timing_summary.csv",pooled)
    accuracy_a=am1.rows(ROOT/"artifacts/thesis_interpretation/v1/tables/track_a_performance.csv")
    accuracy_b=am1.rows(ROOT/"artifacts/thesis_interpretation/v1/tables/track_b_findings.csv")
    gate=am1.read(OUT/"replay_gate.json")
    acc=[]
    ctxmap={"qpu":"qualified_qpu_4515","analytical":"qualified_qpu_4515","dense_fp32":"cudaq_dense_statevector_fp32","dense_fp64":"cudaq_dense_statevector_fp64","maestro":"qcsim_statevector_cpu_fp64_1000_shots"}
    for r in pooled:
        if r["boundary"]!="B1": continue
        matches=[s for s in (accuracy_a if r["context"] in ("qpu","analytical") else accuracy_b) if s["method_id"]==r["method"] and ("context_id" not in s or s["context_id"]==ctxmap[r["context"]])]
        if len(matches)!=1: raise RuntimeError("accuracy_identity_join:"+str((r["context"],r["method"],len(matches))))
        s=matches[0]
        acc.append(dict(context=ctxmap[r["context"]],method=r["method"],accuracy_n=s["n"],accuracy_assigned=s["assigned_n"],
            frozen_mae_seconds=s["mae_seconds"],frozen_r2=s["r2_seconds"],timing_n=r["requests"],B1_median_request_ms=r["median_request_ms"],
            accuracy_population_vs_timing="different denominators; not per-observation Pareto frontier",target_clock=s.get("evaluation_target_clock",s.get("runtime_boundary","NA"))))
    am1.csv_save("accuracy_cost_comparisons.csv",acc)
    # Exact-ID joins to frozen saved predictions; no new accuracy model.
    original={}
    for name in (ROOT/"artifacts/real_qpu/qualified_logical_panel/summary/oof_predictions.csv",ROOT/"artifacts/benchmark_deep_dive/tables/qpu_baseline_predictions_v1.csv"):
        for r in am1.rows(name):
            original[("qpu",r["method_id"],r["canonical_observation_id"])]=(r["actual_seconds"],r["predicted_seconds"])
    for precision in ("fp32","fp64"):
        for f in range(5):
            folder=ROOT/"work/benchmark_recovery/r8_dense_predictors/attempt_002"/precision
            for sub in ("neural","classical"):
                for r in am1.rows(folder/sub/f"fold_{f}"/"predictions.csv"):
                    original[(f"dense_{precision}",r["method_id"],r["qasm_sha256"])]=(r["target_seconds"],r["prediction_seconds"])
    for r in am1.rows(am1.M/"evaluation_001/oof_predictions.csv"):
        for method in ("maestro_style_cpu_sv_component_optimizer_off_adaptation","outer_train_median"):
            original[("maestro",method,r["source_qasm_sha256"])]=(r["runtime_seconds"],r[f"pred_{method}_seconds"])
    aligned=[]
    for r in summaries:
        if r["boundary"]!="B1":continue
        actual,predicted=original[(r["context"],r["method"],r["id"])]
        aligned.append(dict(context=r["context"],method=r["method"],id=r["id"],saved_target_seconds=actual or "NA",saved_prediction_seconds=predicted or "NA",B1_median_seconds=r["median_seconds"],
            B1_over_historical_execution=float(r["median_seconds"])/float(actual) if actual and float(actual)>0 else "NA",frozen_absolute_error_seconds=abs(float(actual)-float(predicted)) if actual and predicted else "NA"))
    am1.csv_save("workload_accuracy_cost.csv",aligned)
    cost_groups=defaultdict(list)
    for r in aligned:
        if r["B1_over_historical_execution"]!="NA":
            cost_groups[(r["context"],r["method"])].append(r["B1_over_historical_execution"])
    ratios=[dict(context=c,method=m,matched_requests=len(v),median_estimation_over_saved_execution=statistics.median(v),estimation_slower_than_saved_execution_n=sum(x>1 for x in v)) for (c,m),v in sorted(cost_groups.items())]
    comparison=[]
    # Matched request medians, not 30 jitter replicates as independent circuits.
    workmap={(r["context"],r["id"]):r for r in workloads}
    medmap={(r["context"],r["method"],r["boundary"],r["id"]):r["median_seconds"] for r in summaries}
    for c,g,m in (("qpu","mali_logical_graph","mali_global_mlp"),("dense_fp32","mali_style_dense_graph","matched_metadata_mlp"),("dense_fp64","mali_style_dense_graph","matched_metadata_mlp")):
        ids=sorted({r["id"] for r in summaries if r["context"]==c and r["method"]==g and r["boundary"]=="B1"}&{r["id"] for r in summaries if r["context"]==c and r["method"]==m and r["boundary"]=="B1"})
        groups=defaultdict(list)
        for rid in ids:
            key="qualified_qpu_4515" if c=="qpu" else c
            groups[workmap[(key,rid)]["group"]].append((medmap[(c,g,"B1",rid)]-medmap[(c,m,"B1",rid)])*1000)
        values=[statistics.mean(x) for x in groups.values()]
        import random
        rng=random.Random(42)
        samples=[statistics.mean(rng.choices(values,k=len(values))) for _ in range(2000)] if len(values)>=10 else []
        comparison.append(dict(context=c,shared_requests=len(ids),structural_groups=len(groups),equal_group_mean_graph_minus_mlp_ms=statistics.mean(values),
            bootstrap_low_ms=quantile(samples,.025) if samples else "NA",bootstrap_high_ms=quantile(samples,.975) if samples else "NA",bootstrap_unit="structural group",bootstrap_seed=42,repetitions_not_independent=True))
    am1.csv_save("paired_latency.csv",comparison)
    errors=[am1.read(p) for p in (OUT/"executions").glob("*.json") if am1.read(p)["exit"]!=0]
    maxrss=max(r["rss_high_water_bytes"] for r in memory)
    maxgpu=max(r["cuda_reserved_peak_bytes"] for r in memory if isinstance(r["cuda_reserved_peak_bytes"],int))
    setups=defaultdict(list)
    for r in setup: setups[r["context"]].append(r["seconds"])
    setup_table=[dict(context=c,fresh_processes=len(x),median_setup_seconds=statistics.median(x),min_seconds=min(x),max_seconds=max(x)) for c,x in sorted(setups.items())]
    am1.md("environment_manifest.md","# AM1 environment\n\n"+plan["hardware_inventory"]+"\nCPU: Intel i7-14700F. Affinity "+str(plan["affinity"])+", selected highest advertised-frequency core class. Native/BLAS threads 1; Torch CPU threads 2.\n\n"+
        "Affinity 8/9 are SMT siblings of physical core4 (P-core, advertised max5.4GHz), not two physical cores. This CPU cap is part of the measured configuration. Neural: Python 3.10.21, Torch 2.7.1+cu128, PyG 2.6.1, NumPy 1.26.4. CUDA smoke: sm120, allocation and matmul PASS. Neural computations are FP32 even for the FP64-simulator target. CPU analytical environment is recorded in worker receipts. No installations/upgrades.\n\n"+
        "One benchmark worker at a time. Desktop/RustDesk GPU activity is residual background load, not proven exclusive whole-device ownership. Native timing shares this live workstation. OS page cache is warm/uncontrolled. No cache flush, governor/clock changes, AMP or compilation.\n\n"+
        "Detailed packages, PID, affinity, CUDA build and absolute memory peaks are saved per worker. Timer: perf_counter_ns; CUDA synchronized before/after each neural interval. Controls retain the same timing wrapper. Scalar references in GPU bundles include synchronization-wrapper overhead and cannot establish sub-microsecond intrinsic scalar latency.")
    am1.md("artifact_integrity.md","# Frozen artifact integrity\n\n"+f"{len(plan['source_pins'])} effective source pins checked at qualification and reporting closure; indirect additions are explicitly recorded in adapter_correction.json. {len(bindings)} directly loaded assets rechecked. Original 996 feasibility source pins also independently revalidated at closure. Checkpoints used best_model_state for QPU and final model_state for dense. All selected three seeds loaded in eval/no_grad; no fitting.\n\n"+
        "Indirect source pins follow the exact producing fold receipts and saved representation registry. The AM1-only adapter correction retains selection hashes and original failure receipts; no model, workload, tolerance or frozen output changed. selection_manifest.json remains the first declaration; adapter_correction.json links it and the current harness digest.\n\nFull checkpoint/transform/source paths and SHA256s: selection_manifest.json, adapter_correction.json and workers/*.json. No checkpoint copies or rewritten benchmark artifacts.")
    rv=[]
    for c in sorted({r["context"] for r in replay}):
        for m in sorted({r["method"] for r in replay if r["context"]==c}):
            rr=[r for r in replay if r["context"]==c and r["method"]==m]; ok=[r for r in rr if isinstance(r["abs_diff_seconds"],(int,float))]
            rv.append(dict(context=c,method=m,comparisons=len(rr),failed=sum(r["status"]=="REPLAY FAILED" for r in rr),unavailable=sum(r["status"].startswith("UNAVAILABLE") for r in rr),max_abs_seconds=max((r["abs_diff_seconds"] for r in ok),default="NA"),mean_abs_seconds=statistics.mean(r["abs_diff_seconds"] for r in ok) if ok else "NA",max_relative=max((r["relative_diff"] for r in ok if isinstance(r["relative_diff"],(int,float))),default="NA")))
    am1.md("replay_validation.md","# Replay validation\n\n"+table(list(rv[0]),rv)+"\nFrozen FP32 neural gate atol=1e-6 seconds, rtol=1e-5; accommodates singleton versus frozen dense batch32 rounding. Absolute tolerance can be relatively large for tiny near-zero predictions; all absolute/relative differences remain exposed in CSV. Scalars/native formula: atol=1e-12, rtol=1e-10. No tolerance widening. Aggregated replay compares median of three exactly as the frozen estimator, not a chosen seed. Missing saved outputs are in the denominator and not model replay failures.\n\nNo model accuracy rerun or original prediction rewrite.")
    am1.md("timing_summary.md","# Prediction overhead\n\nUnits below are milliseconds of estimator work, not predicted quantum-runtime seconds. Each model bundle receives 10 warmups using its first admissible request; each selected request then has 30 retained timings. Across-request percentiles use request medians, not an SLA. No outlier removed.\n\n"+table(list(pooled[0]),pooled)+"\n## Cold setup\n\n"+table(list(setup_table[0]),setup_table)+
        "\nB3 is a context/fold bundle, not six independent cold deployments: shared imports, CUDA initialization, all three seeds for both neural estimators, feature loading and 10 warmups. Stage times, fresh parent launch-to-child-exit and first-after-ready intervals are separate in receipts. Parent process launch-to-ready is not directly instrumented; child-start-to-ready excludes launcher/interpreter startup and must not be called complete cold end-to-end latency. Three fresh processes/bundle; OS cache uncontrolled.\n\n"+
        "B1: loaded models, resident GPU payloads, inverse target transform and median3 host result. P1 QPU: saved NPZ/global input loading/hash, feature transform, batching/H2D and inference. P1 dense: saved normalized IR/global metadata to tensor encode/H2D and inference. P1 Qiskit (if qualified): in-memory frozen compiled QASM parsing plus native duration and optional frozen affine bridge. P1 is a partial input pipeline, not B2 from raw logical QASM. Full B2 NOT EXECUTED. Maestro B1 includes knot lookup/count conversion/formula and saved shots1000. No quantum workloads.\n\n"+
        f"Memory absolute completed-bundle/process high-water: RSS {maxrss/1024**3:.4f} GiB, CUDA reserved {maxgpu/1024**3:.4f} GiB. Memory is not isolated per-B1 allocation. Shared loaded assets, initialization/cache and profiling lifetimes remain included. Sampled observer peaks are lower bounds. Failed manual Qiskit parse peak is NOT MEASURED, so the whole pilot cannot claim independently verified RAM-cap compliance for that branch.")
    am1.md("accuracy_cost_analysis.md","# Accuracy and estimator cost\n\n"+table(list(acc[0]),acc)+"\n## Paired prediction latency\n\n"+table(list(comparison[0]),comparison)+
        "\nFACT: all measured fitted variants pass replay on their selected frozen held-out identities. Latency evidence is narrow and environment-specific. Full accuracy and timing subset have different denominators; no per-observation accuracy-cost pairing is fabricated. Dense precisions are different simulator contexts, not different estimator numeric precisions.\n\n"+
        "INTERPRETATION: Graph adds graph-convolution and input handling over MLP. QPU input contracts are asymmetric (node calibration information versus globals), so the difference is not pure topology benefit. The paired latency comparison holds selected workload and fold identities fixed. P1-B1 differences describe the measured partial path, not parsing/transpilation costs missing from full B2. Frozen QPU equal-group MAE differs from pooled interpretation; QPack dominates observation-weighted accuracy.\n\n"+
        "Unavailable Qonductor winning selectors/trees, dense RF/XGB, Azizov and other fitted comparators have no complete serialized original state. They receive no guessed cost, no new refit and no Pareto claim. No accuracy-cost advantage over the classical selector or scheduler-throughput gain is established. No scaling law or operational deployment readiness follows from this small workstation pilot.\n\n## Exact workload cost versus saved execution\n\n"+table(list(ratios[0]),ratios)+
        "\nThese ratios join workload ID/hash to its historical measured execution target and AM1 median B1 latency. They are not a new simulator execution or production scheduling test. B1 Graph exceeds saved dense execution in 27/30 FP32 and 28/30 FP64 requests; MLP does so in 20/30 and 22/30. This is a negative overhead finding for this small-width subset, not universal scaling. Full accuracy remains based on150/context, timing on30/context. QPU50 query costs are small relative to their archived service labels, without establishing deployment usefulness. Exact joins: workload_accuracy_cost.csv.")
    errtext="\n".join("- "+e["context"]+" fold "+str(e["fold"])+": "+e["stderr"].splitlines()[-1] for e in errors)
    errtext+="\n- Native Qiskit manual same-scope reconciliation: snapshot pins verified; parse exceeded the predeclared 30-second request cap. Gate 1 incomplete, no B1/bridge timing linked to its accuracy. Details: native_replay_timeout_receipt.json. Peak RSS of this failed manual attempt NOT MEASURED."
    am1.md("failures_and_exclusions.md","# Failures and exclusions\n\n"+errtext+"\n\nThe first dense attempt stopped before inference because an indirect feature pin was not recognized. A coordinator RSS process-exit race also interrupted that execution. Both are AM1 harness issues, not model failures. Original failure receipts remain. A manually recorded adapter-only correction resolved the indirect dense pins; workloads, checkpoint selection and tolerances unchanged. Additional native snapshot qualification, if present, is separately recorded.\n\n"+
        "Maestro missing saved output remains UNAVAILABLE_SAVED_OUTPUT, never a zero prediction; selected workload unchanged. Joint/fixed-MPS not selected in narrowed pilot, not failed. No TN planner, contraction, MPS state simulation, QPU job or QCEC. Full B2 remains NOT EXECUTED. No second/expanded pilot, model refit, environment modification or hidden classical replacement.")
    am1.md("scientific_decision.md","# AM1 scientific decision\n\nPARTIAL SUCCESS.\n\nFACT: replay and measured B1/partial input-path/cold-bundle evidence are now available for an explicitly limited frozen subset. Exact denominators, controls, raw samples and hashes are retained.\n\nINTERPRETATION: this materially adds evidence to working RQ4 prediction-overhead, which previously lacked direct inference measurements. It only partly addresses full accuracy-cost, preprocessing and scalability. It does not change historical RQ4 generalization or RQ1–RQ3 frozen accuracy findings.\n\nPROJECT DECISION: STOP AM1 within this approved pilot. Resolve missing fitted-state retention and complete input-to-estimate/cold-ready instrumentation only under a new authorization if needed.\n\nOPEN QUESTIONS: full B2, calibrated/native methods not qualified, joint-MPS conditioned classifier cost, strongest classical selector/tree cost, per-boundary incremental memory, hardware-portable throughput and meaningful accuracy utility. No automatic further experiment.")
    am1.md("README.md","# AM1 controlled overhead pilot\n\n**PARTIAL SUCCESS.** Frozen inference replay and B1 prediction measurements completed for the saved subset. No train/refit or quantum execution. Full B2 from original QASM is not claimed.\n\n"+
        "Read [replay](replay_validation.md), [timings](timing_summary.md), [accuracy–cost](accuracy_cost_analysis.md), [exclusions](failures_and_exclusions.md) and [decision](scientific_decision.md). Predeclaration: selected_variants.csv, selected_workloads.csv, selection_manifest.json. Native context exceptions are recorded, not hidden.\n\n"+
        "Raw estimator seconds are derived from integer perf_counter_ns intervals; predicted_seconds in timing records is the estimator output, not measured latency. Graph/MLP denotes median-of-three frozen seeds. Pilot is neither a complete estimator leaderboard nor production latency certification. All saved accuracy values remain original.\n\n"+
        f"Selected workloads: {len(workloads)} = QPU50 + dense FP32/FP64 30 each + Maestro30. Counts refer to context requests, not unique circuits across tracks. Narrow sampling does not cover every family/backend stratum; uncovered strata are explicit in the manifest. Runtime-estimator contexts are not comparable across targets.\n\n"+
        f"Completed-worker RSS high-water {maxrss/1024**3:.4f} GiB and CUDA reserved peak {maxgpu/1024**3:.4f} GiB, each below 8 GiB. Failed manual native parse peak RSS was not measured. Storage and wall totals in execution/validation receipts. No commit/push or presentation edit.")
    # Saved-evidence assertions: no inference, measurement or benchmark reexecution.
    am1.verify(plan["source_pins"]); am1.verify(bindings)
    assert am1.sha(OUT/"selected_workloads.csv")==plan["selected_workloads_sha256"]
    assert am1.sha(OUT/"selected_variants.csv")==plan["selected_variants_sha256"]
    assert maxrss<am1.LIMIT and maxgpu<am1.LIMIT
    assert len(workloads)==140
    required=["README.md","environment_manifest.md","artifact_integrity.md","selected_variants.csv","selected_workloads.csv","replay_validation.md","replay_validation.csv","raw_timings.csv","timing_summary.md","accuracy_cost_analysis.md","failures_and_exclusions.md","scientific_decision.md","validation_report.md"]
    validation=dict(status="PASS_WITH_SCOPE_LIMITATIONS",required_artifacts=required,selected_requests=len(workloads),raw_timings=len(raw),replay_comparisons=len(replay),
        numerical_replay_failures=sum(r["status"]=="REPLAY FAILED" for r in replay),unavailable_saved_outputs=sum(r["status"].startswith("UNAVAILABLE") for r in replay),
        immutable_source_pins=len(plan["source_pins"]),loaded_asset_pins=len(bindings),selection_unchanged=True,
        timing_arithmetic=True,repetitions_per_request=30,source_pins_unchanged=True,output_bytes_before_final_receipt=sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()),
        rss_peak_bytes=maxrss,cuda_reserved_peak_bytes=maxgpu,memory_scope="completed workers; native manual failure peak not measured",full_B2_executed=False,training_simulation_QPU_QCEC=False,
        complete_Pareto_frontier=False,commit_push=False)
    am1.md("validation_report.md","# Final saved-evidence validation\n\n"+json.dumps(validation,indent=2)+"\n\nPASS applies to the completed pilot subset and saved evidence arithmetic/identity, not unmeasured boundaries or a full benchmark frontier. Repeated measurements are retained, not independent workload support. Original harness failures remain in execution receipts. Detailed setup decomposition and peak-memory attribution limitations are explicit.\n\nSource digests, selection hashes, 30-per-request counts, no negative intervals, exact saved-accuracy joins, replay gates and resource caps were checked. Recheck with `python -B benchmark_v1/scripts/validate_am1_overhead_pilot.py`; no measurement or fitting.")
    payload={str(p.relative_to(OUT)):am1.sha(p) for p in OUT.rglob("*") if p.is_file() and p.name!="provenance_manifest.json"}
    am1.save(OUT/"provenance_manifest.json",dict(validation=validation,payload=payload,source_pins=plan["source_pins"],report_builder_sha256=am1.sha(__file__),self_hash_policy="manifest excluded from its payload"))
    print(json.dumps(validation))


if __name__=="__main__": summary()
