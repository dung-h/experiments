#!/usr/bin/env python3
"""Derived AM1 Phase 2 analysis; default performs saved-evidence checks only.

--emit-patch emits authored reports for apply_patch; no shell file writes.
--diagnose-graph is a bounded CPU-only raw-array reconciliation, no model calls.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark_v1/scripts"))
import run_am1_phase2 as run
import run_am1_overhead_pilot as old
from build_am1_postpilot_audit import metrics, md_table

OUT = run.OUT


def csv_text(records, fields=None):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields or list(records[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue()


def quantile(values, probability):
    ordered = sorted(values)
    index = (len(ordered)-1)*probability
    lower = int(index)
    return ordered[lower] + (ordered[min(lower+1, len(ordered)-1)]-ordered[lower])*(index-lower)


def rank_values(values):
    """Average tied ranks; descriptive correlation only, no fitted predictor."""
    ordered = sorted(set(values))
    counts = Counter(values)
    ranks, offset = {}, 0
    for value in ordered:
        ranks[value] = offset + (counts[value]+1)/2
        offset += counts[value]
    return [ranks[v] for v in values]


def spearman(a, b):
    ra, rb = rank_values(a), rank_values(b)
    ma, mb = st.mean(ra), st.mean(rb)
    numerator = math.fsum((x-ma)*(y-mb) for x,y in zip(ra,rb))
    denominator = math.sqrt(math.fsum((x-ma)**2 for x in ra)*math.fsum((y-mb)**2 for y in rb))
    return numerator/denominator if denominator else "NA"


def diagnose_graph():
    """Single original selected request: exact fields, not an alternative mapping."""
    import os
    import signal
    import resource
    import numpy as np
    import mali_full_features as mf
    plan = old.read(OUT / "measurement_plan.json")
    os.sched_setaffinity(0, plan["affinity"])
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("diagnostic_30s_cap")))
    signal.alarm(30)
    row = next(r for r in old.rows(OUT / "selected_workloads.csv") if r["id"] == "mali_real_qpu|row91")
    panel = next(r for r in old.rows(old.P / "panel.csv") if r["canonical_observation_id"] == row["id"])
    circuit = mf.parse_qasm(Path(row["raw_path"]).read_bytes())[0]
    properties = old.read(row["properties_path"])
    scales = {"s":1e6, "ms":1e3, "us":1., "ns":1e-3}
    cal = {}
    for i, entries in enumerate(properties["qubits"]):
        v = {p["name"]:p for p in entries if p["name"] in {"T1", "T2"}}
        cal[i] = tuple(float(v[k]["value"])*scales[v[k]["unit"].lower().replace("μ", "u").replace("µ", "u")] for k in ("T1", "T2"))
    fresh, meta = mf.compact_graph_record(circuit, cal)
    differences = []
    with np.load(ROOT / panel["graph_file"], allow_pickle=False) as saved:
        for name in sorted(set(saved.files) | set(fresh)):
            if name not in fresh or name not in saved.files:
                differences.append(dict(field=name, reason="key_absent", equal=False))
                continue
            a, b = fresh[name], saved[name]
            if a.shape != b.shape:
                differences.append(dict(field=name, fresh_shape=list(a.shape), saved_shape=list(b.shape), equal=False))
                continue
            idx = np.argwhere(a != b)
            examples = [dict(index=key.tolist(), fresh=float(a[tuple(key)]), saved=float(b[tuple(key)])) for key in idx[:3]]
            differences.append(dict(field=name, dtype=str(a.dtype), shape=list(a.shape), unequal_elements=len(idx), equal=not len(idx), examples=examples))
    signal.alarm(0)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if rss > old.LIMIT:
        raise RuntimeError("diagnostic_RAM_cap")
    run.write_json("qpu_raw_graph_diagnostic.json", dict(id=row["id"], raw_sha256=row["raw_sha256"], graph_sha256=panel["graph_sha256"],
        frozen_mapping_unchanged=True, raw_parsing_only=True, new_inference=False, meta=meta, differences=differences,
        rss_high_water_bytes=rss, conclusion="input gate failed; do not repair or reorder frozen representation automatically"))
    print(json.dumps(dict(stage="diagnostic", unequal_fields=[d["field"] for d in differences if not d["equal"]], rss=rss)))


def build():
    plan = old.read(OUT / "measurement_plan.json")
    old.verify(plan["source_pins"])
    for name in ("selected_workloads", "selected_variants"):
        assert old.sha(OUT / f"{name}.csv") == plan[f"{name}_sha256"]
    completed = old.read(OUT / "execution_complete.json")
    gate = old.read(OUT / "replay_gate.json")
    assert completed["status"] == "complete" and completed["source_pins_match"]
    assert completed["phase2b_wall_seconds"] <= 7200
    selection = old.rows(OUT / "selected_workloads.csv")
    selected = {(r["context"], r["id"]):r for r in selection}
    assert len(selected) == 110
    original = {}
    canonical_targets = {r["canonical_observation_id"]:float(r["target_seconds"]) for r in old.rows(old.P / "panel.csv")}
    target_serialization_deltas = []
    def add(context, method, rid, y, pred, fold, group):
        key = context, method, rid
        target = float(y)
        if context == run.CONTEXTS[0]:
            target_serialization_deltas.append(abs(target-canonical_targets[rid]))
            assert math.isclose(target, canonical_targets[rid], rel_tol=0, abs_tol=1e-10)
            target = canonical_targets[rid]
        value = target, float(pred), int(fold), group
        if key in original:
            assert original[key] == value
        original[key] = value
    for r in old.rows(old.P / "summary/oof_predictions.csv"):
        if r["method_id"] in run.METHODS[run.CONTEXTS[0]]:
            add(run.CONTEXTS[0], r["method_id"], r["canonical_observation_id"], r["actual_seconds"], r["predicted_seconds"], r["outer_fold"], r["group_id"])
    for r in old.rows(ROOT / "artifacts/benchmark_deep_dive/tables/qpu_baseline_predictions_v1.csv"):
        if r["method_id"] == "outer_train_median":
            add(run.CONTEXTS[0], r["method_id"], r["canonical_observation_id"], r["actual_seconds"], r["predicted_seconds"], r["outer_fold"], r["group_id"])
    for precision in ("fp32", "fp64"):
        for fold in range(5):
            for stage in ("neural", "classical"):
                for r in old.rows(old.D / precision / stage / f"fold_{fold}/predictions.csv"):
                    if r["method_id"] in (*run.METHODS[f"dense_{precision}"], "outer_train_median"):
                        add(f"dense_{precision}", r["method_id"], r["qasm_sha256"], r["target_seconds"], r["prediction_seconds"], fold, r["qasm_sha256"])
    timings, replay, histories, memories, environments = [], [], [], [], []
    receipts = sorted((OUT / "workers").glob("*.json"))
    assert len(receipts) == 30
    for path in receipts:
        data = old.read(path)
        assert data["environment"]["packages"]["torch"] == "2.7.1+cu128"
        assert data["environment"]["packages"]["qiskit"] == "0.44.2"
        assert data["environment"]["setup_ns"] <= 120*10**9
        memories.append(data["memory"])
        environments.append(data["environment"])
        timings.extend(data["timings"])
        replay.extend(data["replay"])
        histories.extend([dict(receipt=str(path.relative_to(ROOT)), **r) for r in data["histories"]])
    by_key = defaultdict(list)
    samples = set()
    for t in timings:
        assert t["status"] == "ok"
        assert int(t["end_ns"]) - int(t["start_ns"]) == int(t["elapsed_ns"]) >= 0
        sample = t["context"], t["method"], t["id"], int(t["repeat"]), t["stage"]
        assert sample not in samples
        samples.add(sample)
        request = selected[(t["context"], t["id"])]
        assert int(request["fold"]) == t["fold"]
        y, p, fold, group = original[(t["context"], t["method"], t["id"])]
        assert fold == t["fold"] and group == request["group"]
        assert old.parity(t["method"], t["id"], "native" if t["method"] == "outer_train_median" else "median3", float(t["prediction_seconds"]), p)["status"] == "PASS"
        assert t["elapsed_ns"] <= 30*10**9
        by_key[(t["context"], t["method"], t["id"], t["stage"])].append(t)
    assert all(len(v) == 30 and {t["repeat"] for t in v} == set(range(30)) for v in by_key.values())
    total_keys = {k:v for k,v in by_key.items() if k[3] in {"complete_B2", "complete_native_scalar"}}
    stage_summary, metric_rows, comparison_rows = [], [], []
    per_query = {}
    for k, v in sorted(by_key.items()):
        values = [t["elapsed_ns"] / 1e6 for t in v]
        per_query[k] = st.median(values)
    for context, method, stage in sorted({(k[0], k[1], k[3]) for k in by_key}):
        keys = [k for k in by_key if k[0] == context and k[1] == method and k[3] == stage]
        stage_summary.append(dict(context=context, method=method, stage=stage, queries=len(keys), repetitions_per_query=30,
            median_query_median_ms=st.median(per_query[k] for k in keys), median_query_p95_ms=st.median(quantile([t["elapsed_ns"] / 1e6 for t in by_key[k]], .95) for k in keys),
            min_query_median_ms=min(per_query[k] for k in keys), max_query_median_ms=max(per_query[k] for k in keys)))
    for context, method in sorted({k[:2] for k in total_keys}):
        keys = sorted(k for k in total_keys if k[:2] == (context, method))
        ids = [k[2] for k in keys]
        vals = [original[(context, method, i)][:2] for i in ids]
        metric_rows.append(dict(context=context, method=method, population="exact_measured_IDs", **metrics(vals),
            groups=len({selected[(context, i)]["group"] for i in ids}),
            median_full_pipeline_ms=st.median(per_query[k] for k in keys), boundary=keys[0][3],
            source_counts=json.dumps(Counter(selected[(context,i)]["source"] for i in ids), sort_keys=True)))
    for context in run.CONTEXTS:
        graph, mlp = run.METHODS[context]
        for a, b in ((graph, mlp), (mlp, "outer_train_median")):
            akeys = {k[2]:k for k in total_keys if k[:2] == (context,a)}
            bkeys = {k[2]:k for k in total_keys if k[:2] == (context,b)}
            ids = sorted(set(akeys) & set(bkeys))
            if not ids:
                comparison_rows.append(dict(context=context, comparison=f"{a} vs {b}", shared_queries=0, groups=0, status="NOT_EXECUTED: raw_input_gate_failed", paired_mean_latency_delta_ms="NA", latency_ci95="NA", observed_MAE_delta_seconds="NA", MAE_delta_ci95="NA"))
                continue
            grouped = defaultdict(list)
            for rid in ids:
                y, pa, _, group = original[(context,a,rid)]
                yb, pb, _, gb = original[(context,b,rid)]
                assert y == yb and group == gb
                grouped[group].append((per_query[akeys[rid]]-per_query[bkeys[rid]], abs(y-pa)-abs(y-pb)))
            # Equal-group latency and cluster-resampled observation-weighted MAE.
            groups = sorted(grouped)
            rng = random.Random(20261011)
            latency, errors = [], []
            for _ in range(2000):
                drawn = rng.choices(groups, k=len(groups))
                latency.append(st.mean(st.mean(v[0] for v in grouped[g]) for g in drawn))
                errors.append(st.mean(v[1] for g in drawn for v in grouped[g]))
            comparison_rows.append(dict(context=context, comparison=f"{a} vs {b}", shared_queries=len(ids), groups=len(groups), status="measured native contracts; unequal information allowed",
                paired_mean_latency_delta_ms=st.mean(st.mean(v[0] for v in grouped[g]) for g in groups), latency_ci95=[quantile(latency,.025),quantile(latency,.975)],
                observed_MAE_delta_seconds=st.mean(abs(original[(context,a,i)][0]-original[(context,a,i)][1])-abs(original[(context,b,i)][0]-original[(context,b,i)][1]) for i in ids),
                MAE_delta_ci95=[quantile(errors,.025),quantile(errors,.975)]))
    stages = {r["stage"]:r["median_query_median_ms"] for r in stage_summary if r["context"] == run.CONTEXTS[0] and r["method"] == "mali_global_mlp"}
    qpu_panel = {r["canonical_observation_id"]:r for r in old.rows(old.P / "panel.csv")}
    qpu_measured_ids = sorted(k[2] for k in total_keys if k[:2] == (run.CONTEXTS[0], "mali_global_mlp"))
    qpu_costs = [per_query[(run.CONTEXTS[0], "mali_global_mlp", i, "complete_B2")] for i in qpu_measured_ids]
    scale_axes = {
        "logical_width_g44": [float(qpu_panel[i]["g44"]) for i in qpu_measured_ids],
        "logical_depth_g45": [float(qpu_panel[i]["g45"]) for i in qpu_measured_ids],
        "saved_graph_node_count_covariate": [int(qpu_panel[i]["node_count"]) for i in qpu_measured_ids],
        "raw_QASM_bytes": [Path(selected[(run.CONTEXTS[0],i)]["raw_path"]).stat().st_size for i in qpu_measured_ids],
    }
    scaling = [dict(covariate=name, n=len(qpu_costs), spearman_with_B2_ms=spearman(v, qpu_costs), interpretation="exploratory selected_sample; source/family/size confounded") for name,v in scale_axes.items()]
    source_latencies = []
    for source_id in sorted({selected[(run.CONTEXTS[0],i)]["source"] for i in qpu_measured_ids}):
        costs = [per_query[(run.CONTEXTS[0], "mali_global_mlp", i, "complete_B2")] for i in qpu_measured_ids if selected[(run.CONTEXTS[0],i)]["source"] == source_id]
        source_latencies.append(dict(source=source_id, n=len(costs), median_B2_ms=st.median(costs), min_B2_ms=min(costs), max_B2_ms=max(costs)))
    assert gate[run.CONTEXTS[0]]["mali_global_mlp"]
    assert not any(gate[c][run.METHODS[c][0]] for c in run.CONTEXTS)
    assert not any(gate[c][run.METHODS[c][1]] for c in run.CONTEXTS[1:])
    assert len(total_keys) == 150  # QPU 45 MLP+45 median; dense 30+30 median.
    failures = [r for r in histories if r["status"] != "PASS"]
    assert len(histories) == 210 and len(failures) == 46
    assert len([r for r in failures if r["reason"] == "RuntimeError:QPU_sparse_graph_input_mismatch"]) == 38
    assert all(r["status"] == "PASS" for r in replay)
    maxrss = max(m["rss_high_water_bytes"] for m in memories)
    maxvram = max(m["cuda_reserved_peak_bytes"] for m in memories)
    assert maxrss <= old.LIMIT and maxvram <= old.LIMIT
    plan_pins = dict(plan["source_pins"])
    for p in (OUT / "measurement_plan.json", OUT / "replay_gate.json", OUT / "scoped_scan.json",
              ROOT / "benchmark_v1/scripts/analyze_am1_phase2.py", OUT / "qpu_raw_graph_diagnostic.json"):
        plan_pins[str(p)] = old.sha(p)
    for p in sorted((OUT / "executions").glob("*.json")) + receipts + [OUT / "execution_complete.json"]:
        plan_pins[str(p)] = old.sha(p)
    old_inventory = old.rows(old.FEAS.parent / "estimator_inventory.csv")
    new_inventory = []
    for row in old_inventory:
        c, m = row["context_id"], row["estimator_id"]
        c = {"cudaq_dense_statevector_fp32":"dense_fp32", "cudaq_dense_statevector_fp64":"dense_fp64"}.get(c, c)
        value = dict(row, newly_recovered_competitive_fitted_state="false",
            phase2_state_class="existing_frozen_checkpoint_or_sufficient_parameter_state" if row["status"] == "READY" else "predictions_or_parameters_insufficient" if row["status"] == "BLOCKED" else "native_pipeline_or_not_a_predictor",
            phase2_b2_status="NOT_EXECUTED_not_selected")
        if c in gate and m in gate[c]:
            value["phase2_b2_status"] = "MEASURED" if gate[c][m] else "NOT_EXECUTED_raw_input_parity_failed"
        new_inventory.append(value)
    files = {"raw_stage_timings.csv":csv_text(timings), "fitted_state_recovery_inventory.csv":csv_text(new_inventory)}
    files["README.md"] = """# AM1 Phase 2 — pipeline overhead

CONDITIONAL GO, bounded execution complete. QPU global MLP full B2 measured
from saved raw logical QASM on 45/50 requests. Graph QPU and both dense
neural pipelines failed context-wide raw-input qualification; no timing was
silently made on a passing subset. Native frozen median controls measured
on 105 requests. No new competitive classical fitted state recovered.

Read [feasibility](phase2a_feasibility.md), [contracts](pipeline_input_contracts.md),
[frozen plan](predeclared_measurement_plan.md), [B2](b2_pipeline_summary.md),
[trade-offs](accuracy_cost_tradeoffs.md), [blockers](limitations_and_blockers.md),
[RQ4 decision](rq4_decision.md), [validation](validation_report.md).
Authority and exact hashes: measurement_plan.json and provenance_manifest.json.
Historical AM1 pilot/post-pilot audit are unchanged. This is not a new full
benchmark, a complete Pareto frontier, or a release approval.
"""
    files["b2_pipeline_summary.md"] = "# Complete B2 results\n\n" + md_table(metric_rows) + "\n## Stage costs\n\n" + md_table(stage_summary) + """

Complete B2 = instrumented, independently timed raw in-memory QASM-to-host
prediction. No disk retrieval, route selection, checkpoint load or training.
Global extraction itself uses upstream structural/Supermarq calculations,
including its required DAG traversal; MLP does not use the Graph node encoder.
Latency summaries are median over per-query medians, NOT pooled repeat-N.
p95 is linearly interpolated within 30 repeats, then summarized across queries;
not a reliable worst-case service bound. Native scalar timer is close to
perf_counter/float-access overhead: no claim of nanosecond deployment endpoint.

No fresh Graph complete B2 or dense neural complete B2 data exist here.
Their gates failed before timing. Preprocessing/inference breakdown comes
only from the qualified QPU MLP, not a substitute Graph path.
"""
    # A ratio of stage totals from the SAME instrumented requests, not ratio of medians.
    qpu_totals = [t for t in timings if t["stage"] == "complete_B2"]
    qpu_prep = [t for t in timings if t["stage"] in {"parse_QASM", "extract_51_globals", "frozen_transform_and_tensor_encode", "batch_and_H2D"}]
    prep_fraction = math.fsum(t["elapsed_ns"] for t in qpu_prep)/math.fsum(t["elapsed_ns"] for t in qpu_totals)
    files["accuracy_cost_tradeoffs.md"] = "# Accuracy–cost interpretation\n\n" + md_table(comparison_rows) + f"""

QPU MLP measured population =45 IDs (38 Ma–Li,7 Qonductor), not the full
4515 cohort and not the previous 50-ID AM1 subset. QPack5 remain unavailable
for this declared raw-input boundary; no accuracy/timing-based selection.
Mean pooled preparation fraction over the SAME 1350 B2 requests:
{prep_fraction:.6%}. Stage median parsing={stages['parse_QASM']:.6f}ms,
global extraction={stages['extract_51_globals']:.6f}ms,
three-seed inference/inverse/D2H={stages['three_seed_inference_inverse_and_D2H']:.6f}ms.
Parsing and global features materially change MLP deployment cost relative to
prediction-only; original B1 is historical corroboration, not same-run subtraction.

Comparison to native train-fold median is useful but weak: it needs no circuit
features. It is not the missing strong selector/RF/XGB. Accuracy uses original
frozen OOF predictions; replay outputs are ONLY integrity evidence. No refit.
Paired intervals use2000 fixed-seed cluster bootstrap replicates. Equal-group
mean latency and observation-weighted MAE have distinct declared weighting.
Generalization to unmeasured QPack-dominant parent population is not justified.
QPU paired reanalysis uses identical panel.csv target_seconds for all methods;
saved target text has float serialization differences of at most
{max(target_serialization_deltas):.3g}s. Original labels/predictions untouched.

Graph full-overhead comparison has NOT been answered: its raw-input gate
failed. No result supports choosing Graph by full B2 efficiency. Original
B1 Graph–MLP conclusions and all full-cohort accuracy remain unchanged.
Dense context median reference alone is not a competitive frontier or a
measurement of dense neural B2. Saved FP32/FP64 execution timings still have
different software/time provenance; no operational break-even claim.

Full-cohort accuracy authority remains
artifacts/am1_postpilot_audit/v1/accuracy_population_metrics.csv and original
producing OOF files. In particular QPU Graph MAE0.834133s versus MLP0.893423s
on4515; dense Graph MAE0.001715940s/0.001730218s versus MLP0.001577722s/
0.001590082s on150 each. These are not latency-population scores.
"""
    files["accuracy_cost_tradeoffs.md"] += "\n## Exploratory size/complexity description\n\n" + md_table(scaling) + "\n" + md_table(source_latencies) + """

Đây là mô tả sau đo, không confirmatory scaling law. Dùng logical width từ
g44, không dùng width physical/allocated của cột selection cũ. Source và family
cùng thay đổi với size; không suy diễn luật2^n, quan hệ nhân quả hay ngoại suy
width. Graph node count chỉ là covariate cấu trúc, không feature của MLP.
Model dispatch/fold routing, imports và model load đã làm offline. B2 có
instrumentation và CUDA stage synchronizations; không trừ timer overhead để
biến scalar access thành nanosecond deployment claim. Shared bundle setup
recorded ở worker environments, không phải standalone B3 benchmark.
"""
    diagnostic = old.read(OUT / "qpu_raw_graph_diagnostic.json")
    diff = [d for d in diagnostic["differences"] if not d["equal"]]
    files["limitations_and_blockers.md"] = """# Limitations and blockers

- Strong classical comparator remains BLOCKED. Scoped scan: QPU regression
  40 files; dense classical11/precision; fixed-MPS28; family-MPS40;
  Maestro evaluation4; Azizov322. No candidate fitted serialization files.
  Producing code discards fitted estimators after predicting; settings/seeds,
  scaler summaries and predictions do not recover trees, SVs or coefficients.
  Existing median scalars/affine coefficients/Maestro knots/neural states
  were already recoverable: zero newly recovered competitive models.
- QPU Graph:38/45 raw reconstructions fail exact sparse-array equality. All
  45 globals/MLP predictions pass. Do not change node ordering/calibration
  or cache saved graph to relabel as full raw B2. Single-request diagnostic
  below documents discrepant fields; it does not diagnose all38 cases.
- Dense: FP32 three circuits and FP64 one circuit fail both model paths at
  RCCX lowering. Parser/environment definition includesu1/u2 where frozen
  normalizer requirescx/h/t/tdg. All30/context attempted; no cherry-picking
  27/29 passing subsets. Context-wide gate blocks both neural methods.
- QPack5 selected raw QASM unavailable at declared locations: no generator
  execution or invented QASM to bypass limitation. This does not change
  qualified dataset or reconstruction labels.
- Historical compilation/backend acquisition, exact historical recovery,
  standalone cold starts, batch amortization, cross-hardware scaling,
  planner cost, and strongest classical full-pipeline frontier remain open.
- Shared model-bundle RAM/VRAM peaks and stage-end allocation snapshots are
  not marginal per-model memory costs. Desktop background activity retained;
  hardware is not exclusive. No target simulation was run for break-even.
- Native Qiskit compiled-QASM timeout root cause is still unresolved. Logical
  QPU MLP parsing is a different input path and cannot clear that historical
  timeout. Maestro/MPS/native analytical/TN B2 NOT EXECUTED in Phase2.

## Single-request graph reconciliation

""" + json.dumps(dict(id=diagnostic["id"], discrepant_fields=diff), indent=2) + "\n"
    files["rq4_decision.md"] = """# Working RQ4 — PARTIALLY ANSWERED

This working RQ4 concerns estimator overhead, not historical Contractv1.1
RQ4 generalization (completed21/21). AM1 Pilotv1 remains closed and its audit
corrections stand. Phase2 adds complete native raw-input MLP B2 costs on45
qualified selected QPU requests and avoids GPU-wrapped scalar timing bias.
Preparation dominates those MLP requests; warm inference alone does not
describe full request cost. Graph–MLP B2 and strongest classical comparison
remain BLOCKED by raw pipeline/fitted-state gates, not bad test accuracy.

STOP Phase2 after this bounded execution and validation. Recommend preserving
existing B1 accuracy–cost claims, reporting new partial B2 evidence separately,
and leaving full Pareto/frontier/scalability claims unmade. If further work is
essential, a separate user-approved amendment should freeze a compatible
Qiskit/preprocessing+CUDA environment and audit retained fitted states; no
automatic environment repair, refit, third phase or resumed queue.
"""
    executions = [old.read(p) for p in sorted((OUT / "executions").glob("*.json"))]
    assert len(executions) == 30 and all(e["exit"] == 0 for e in executions)
    assert len([r for r in timings if r["stage"] == "complete_B2"]) == 1350
    assert len([r for r in timings if r["stage"] == "complete_native_scalar"]) == 3150
    assert len(timings) == 12600
    assert sum(t["elapsed_ns"] for t in timings if t["stage"] != "complete_B2" and t["boundary"] == "B2") <= sum(t["elapsed_ns"] for t in qpu_totals)
    validation = dict(source_pins_match=True, original_artifacts_unchanged=True, phase2b_authorized_by_conditional_gate=True,
        raw_input_replay_requests=len(histories), raw_input_replay_pass=sum(r["status"] == "PASS" for r in histories),
        raw_input_replay_fail=len(failures), numerical_replay_comparisons=len(replay), numerical_replay_pass=sum(r["status"] == "PASS" for r in replay),
        gate_false_methods_not_timed=True, requests_with_complete_B2=45, complete_B2_requests_including_repeats=1350,
        native_scalar_queries=105, raw_stage_rows=len(timings), timed_query_units=len(total_keys), measurement_repetitions=30,
        phase2b_wall_seconds=completed["phase2b_wall_seconds"], ram_peak_bytes=maxrss, vram_reserved_peak_bytes=maxvram,
        highest_gpu_stage_end_allocation_bytes=max(t["cuda_allocated_bytes"] for t in timings),
        train_refit_simulator_QPU_planning=False, standalone_cold_B3=False, complete_graph_MLP_B2_comparison=False,
        full_Pareto_frontier=False, full_RQ4_closed=False, commit_push=False, positive_clock_arithmetic=True,
        quantile="linear interpolation within30 requests, median acrossqueries", no_outlier_deletion=True,
        exact_parent_ID_fold_and_target_alignment=True, frozen_med3_predictions_unchanged=True,
        qpu_common_target_authority="panel.csv target_seconds; saved actuals within1e-10s",
        maximum_saved_target_serialization_delta_seconds=max(target_serialization_deltas),
        preparation_fraction_same_requests=prep_fraction, unmatched_target_execution_environment=True)
    files["validation_report.md"] = "# Saved-evidence validation\n\nPASS for scoped measurements; failed qualification gates are preserved, not called measurement PASS.\n\n" + json.dumps(validation, indent=2) + """

Commands: python3 -B benchmark_v1/scripts/analyze_am1_phase2.py --check
and focused Phase2/postpilot/pilot saved-evidence pytest. Evidence reproducibility
recalculates metrics, repetitions, ID joins, interval weighting, caps and pins.
Tests never launch estimators, training, QASM parsing or target execution.
Do not substitute a prior release/test PASS for Phase2 validation.
"""
    # Keep prepare's seven authored files; inventory corrected into clear original field schema.
    new_outputs = set(files)
    payload = {p.name:old.sha(p) for p in OUT.glob("*") if p.is_file() and p.name not in new_outputs | {"provenance_manifest.json"}}
    payload.update({name:hashlib_sha(content) for name,content in files.items()})
    manifest = dict(phase2a_verdict="CONDITIONAL_GO", phase2b_execution="SCOPED_COMPLETE", source_pins=plan_pins,
        payload=payload, validation=validation, summary=dict(stage_summary=stage_summary, measured_accuracy=metric_rows, paired_comparisons=comparison_rows, exploratory_scaling=scaling, source_latency=source_latencies),
        scientific_interpretation="MLP_raw_B2_only; Graph_and_dense_neural_gates_failed; not_full_frontier")
    files["provenance_manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+"\n"
    return files


def hashlib_sha(content):
    import hashlib
    return hashlib.sha256(content.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnose-graph", action="store_true")
    ap.add_argument("--emit-patch", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--materialize-data", action="store_true", help="mechanical CSV reduction only; authored reports use --emit-patch")
    args = ap.parse_args()
    if args.diagnose_graph:
        diagnose_graph()
        return
    files = build()
    if args.materialize_data:
        for name, content in files.items():
            if name.endswith((".csv", ".json")):
                # Computational data reduction, not a historical-artifact edit.
                run.storage(len(content.encode()))
                (OUT/name).write_text(content)
        print(json.dumps(dict(status="DATA_REDUCTION_COMPLETE", files=[n for n in files if n.endswith((".csv", ".json"))])))
    elif args.emit_patch:
        print("*** Begin Patch")
        for name, content in files.items():
            if name.endswith((".csv", ".json")):
                continue
            p = OUT / name
            if p.exists():
                print("*** Update File: " + str(p))
                print("@@")
                for line in p.read_text().splitlines():
                    print("-"+line)
                for line in content.splitlines():
                    print("+"+line)
            else:
                print("*** Add File: " + str(p))
                for line in content.splitlines():
                    print("+"+line)
        print("*** End Patch")
    else:
        assert args.check
        for name, content in files.items():
            assert (OUT/name).read_text() == content, name
        old.verify(old.read(OUT/"provenance_manifest.json")["source_pins"])
        print(json.dumps(dict(status="PASS", report_files=len(files), validation=old.read(OUT/"provenance_manifest.json")["validation"])))


if __name__ == "__main__":
    main()
