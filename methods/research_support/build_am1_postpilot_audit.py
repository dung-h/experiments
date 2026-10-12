#!/usr/bin/env python3
"""Independent saved-evidence AM1 audit. No estimator, parser or framework imports.

Default emits derived file contents as JSON for controlled apply_patch creation.
--check recalculates the audit and verifies saved payload and source pins.
Never writes inputs or launches workers. No timing API is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import random
import statistics as st
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "artifacts/am1_overhead_pilot/v1"
OUT = ROOT / "artifacts/am1_postpilot_audit/v1"
PANEL = ROOT / "artifacts/real_qpu/qualified_logical_panel"
DENSE = ROOT / "work/benchmark_recovery/r8_dense_predictors/attempt_002"
TARGETS = ROOT / "work/benchmark_recovery/r5_dense/targets_recovered_v2"
MAESTRO = ROOT / "work/repo_finalization/publish_candidate/artifacts/benchmark_v3/simulator/maestro_cpu_component_optimizer_off"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def csv_text(records):
    stream = io.StringIO(newline="")
    if records:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return stream.getvalue()


def md_table(records):
    fields = list(records[0])
    return "| " + " | ".join(fields) + " |\n|" + "|".join("---" for _ in fields) + "|\n" + "\n".join(
        "| " + " | ".join(str(r[k]) for k in fields) + " |" for r in records) + "\n"


def metrics(values):
    valid = [(float(y), float(p)) for y, p in values if y not in ("", "NA", None) and p not in ("", "NA", None)]
    if not valid:
        return dict(n=0, mae_seconds="NA", medae_seconds="NA", rmse_seconds="NA", r2="NA")
    assert all(math.isfinite(v) for pair in valid for v in pair)
    actual, prediction = zip(*valid)
    errors = [abs(y-p) for y, p in valid]
    # Python 3.12 changed float sum(); use explicit compensated summation so
    # this saved-evidence report reproduces byte-for-byte under 3.10 and 3.13.
    sse = math.fsum((y-p)**2 for y, p in valid)
    mean_y = st.mean(actual)
    sst = math.fsum((y-mean_y)**2 for y in actual)
    return dict(n=len(valid), mae_seconds=st.mean(errors), medae_seconds=st.median(errors),
                rmse_seconds=math.sqrt(sse/len(valid)), r2=1-sse/sst if sst > 0 else "NA")


def audit():
    pins = {}
    def track(path):
        path = Path(path)
        pins[str(path)] = sha(path)
        return path
    def js(path):
        return json.loads(track(path).read_text())
    def rows(path):
        csv.field_size_limit(100_000_000)
        with track(path).open(newline="") as f:
            return list(csv.DictReader(f))
    manifest = js(PILOT / "provenance_manifest.json")
    for name, digest in manifest["payload"].items():
        assert sha(track(PILOT/name)) == digest, name
    for name, digest in manifest["source_pins"].items():
        assert sha(Path(name)) == digest, name
        pins[name] = digest
    for name, digest in manifest["implementation_pins"].items():
        assert sha(track(ROOT/name)) == digest, name
    for root in (ROOT/"artifacts/am1_feasibility/v1", ROOT/"artifacts/research_question_closure/v1"):
        for path in sorted(root.glob("*.md")): track(path)
        js(root/"provenance_manifest.json")
    selected = rows(PILOT/"selected_workloads.csv")
    plan = js(PILOT/"selection_manifest.json")
    assert sha(PILOT/"selected_workloads.csv") == plan["selected_workloads_sha256"]
    raw = rows(PILOT/"raw_timings.csv")
    by_latency = defaultdict(list)
    by_repetition = defaultdict(set)
    boundaries = Counter()
    for r in raw:
        assert int(r["end_ns"])-int(r["start_ns"]) == int(r["elapsed_ns"]) >= 0
        boundaries[r["boundary"]] += 1
        if r["boundary"] in ("B1", "P1"):
            key = (r["context"], r["method"], r["id"])
            if r["boundary"] == "P1": key += ("P1",)
            by_latency[key].append(int(r["elapsed_ns"])/1e9)
            by_repetition[key].add(int(r["repeat"]))
    assert all(len(v) == 30 for v in by_latency.values())
    assert all(v == set(range(30)) for v in by_repetition.values())
    latency = {k: st.median(v) for k, v in by_latency.items() if len(k) == 3}
    published_latency = rows(PILOT/"timing_by_workload.csv")
    for r in published_latency:
        if r["boundary"] == "B1": assert float(r["median_seconds"]) == latency[(r["context"], r["method"], r["id"])]

    # Rebuild execution labels from exact source member, precision, IR and sessions.
    target_manifest = js(TARGETS/"manifest.json")
    targets = rows(TARGETS/"dense_hash_targets.csv")
    assert sha(TARGETS/"dense_hash_targets.csv") == target_manifest["outputs"]["dense_hash_targets.csv"]
    historical_dirs = [ROOT/"artifacts/benchmark_v1/cudaq_dense_common_q16_bridge_full_v1_20260928", ROOT/"work/family_dense_recovery_v1/r5_attempt_004"]
    runs = {}
    for directory in historical_dirs:
        rm = js(directory/"run_manifest.json"); env = js(directory/"environment.json")
        observations = rows(directory/"cudaq_dense_common_raw.csv")
        entry = next(r for r in target_manifest["inputs"]["runs"] if r["run_id"] in (directory.name, rm["artifact_id"].replace("cudaq-dense-common-panel-v1-20260928", "cudaq_dense_common_q16_bridge_full_v1_20260928")))
        assert sha(directory/"run_manifest.json") == entry["run_manifest_sha256"]
        assert sha(directory/"environment.json") == entry["environment_sha256"]
        assert sha(directory/"cudaq_dense_common_raw.csv") == entry["raw_sha256"]
        cells = defaultdict(list)
        for r in observations:
            if r["observation_kind"] == "warm" and r["status"] == "ok":
                cells[(r["qasm_sha256"], r["precision"], r["panel_member_id"])].append(r)
        runs[entry["run_id"]] = (cells, env, rm, directory)
    dense_targets = {(r["precision"], r["qasm_sha256"]): r for r in targets}
    simulation = []
    for request in selected:
        if not request["context"].startswith("dense_"): continue
        precision = request["context"].split("_")[-1]; h = request["id"]
        target = dense_targets[(precision, h)]
        cells, env, rm, directory = runs[target["source_run_id"]]
        originals = cells[(h, precision, target["source_panel_member_id"])]
        assert len(originals) == 15
        sessions = defaultdict(list)
        for r in originals:
            assert r["ir_sha256"] == target["legacy_dense_ir_sha256"]
            assert r["engine_id"] == "cudaq_dense" and r["target_id"] == "cudaq_dense_sample_wall_clock" and int(r["shots"]) == 32
            assert int(r["width_qubits"]) == int(target["width_qubits"])
            sessions[r["session_id"]].append(float(r["t_warm_execution_s"]))
        assert len(sessions) == 3 and all(len(v) == 5 for v in sessions.values())
        actual = st.median(st.median(v) for v in sessions.values())
        assert actual == float(target["target_seconds"])
        graph = latency[(request["context"], "mali_style_dense_graph", h)]
        mlp = latency[(request["context"], "matched_metadata_mlp", h)]
        simulation.append(dict(context=request["context"], qasm_sha256=h, outer_fold=request["fold"], family=target["family"], width_qubits=target["width_qubits"],
            source_member_id=target["source_panel_member_id"], source_run_id=target["source_run_id"], normalized_ir_sha256=target["normalized_ir_sha256"], legacy_ir_sha256=target["legacy_dense_ir_sha256"],
            simulation_engine="CUDA-Q 0.15.1 nvidia", simulation_precision=precision, shots=32, runtime_boundary="warm host sample return excluding build/result iteration", execution_seconds=actual,
            execution_repetitions=15, execution_sessions=3, graph_B1_seconds=graph, mlp_B1_seconds=mlp, estimator_repetitions_per_method=30,
            prediction_input_contract="resident frozen graph/global features; held-out fold; three seed models; FP32 estimator arithmetic; batch 1",
            graph_over_execution=graph/actual, mlp_over_execution=mlp/actual, graph_slower=graph>actual, mlp_slower=mlp>actual,
            original_environment_path=str(directory/"environment.json"), original_environment_sha256=sha(directory/"environment.json"),
            estimator_environment_receipt=str(PILOT/f"workers/{request['context']}_fold{request['fold']}_cold0.json"),
            comparison_class="B historical cross-software/time deployment comparison; model/driver matched", historical_gpu_uuid="NOT RECORDED"))
    assert len(simulation) == 60
    sim_summary = []
    for c in ("dense_fp32", "dense_fp64"):
        subset = [r for r in simulation if r["context"] == c]
        assert len({r["qasm_sha256"] for r in subset}) == 30
        sim_summary.append(dict(context=c, circuit_hashes=30, graph_slower=sum(r["graph_slower"] for r in subset), mlp_slower=sum(r["mlp_slower"] for r in subset),
            median_graph_ratio=st.median(r["graph_over_execution"] for r in subset), median_mlp_ratio=st.median(r["mlp_over_execution"] for r in subset)))

    # Independent exact-ID populations from producing predictions, never replay outputs.
    original = {}; groups = {}
    def add(context, method, identity, y, pred, fold, group, source):
        key = (context, method, identity)
        value = (y, pred, str(fold), group, source)
        if key in original: assert original[key] == value
        original[key] = value; groups[(context, identity)] = group
    neural_ids = {"mali_logical_graph", "mali_global_mlp"}
    for r in rows(PANEL/"summary/oof_predictions.csv"):
        if r["method_id"] in neural_ids:
            add("qpu", r["method_id"], r["canonical_observation_id"], r["actual_seconds"], r["predicted_seconds"], r["outer_fold"], r["group_id"], r["source_id"])
    for r in rows(ROOT/"artifacts/benchmark_deep_dive/tables/qpu_baseline_predictions_v1.csv"):
        if r["method_id"] == "outer_train_median":
            add("qpu", r["method_id"], r["canonical_observation_id"], r["actual_seconds"], r["predicted_seconds"], r["outer_fold"], r["group_id"], r["source_id"])
    for precision in ("fp32", "fp64"):
        for fold in range(5):
            for stage in ("neural", "classical"):
                for r in rows(DENSE/precision/stage/f"fold_{fold}"/"predictions.csv"):
                    if r["method_id"] in ("mali_style_dense_graph", "matched_metadata_mlp", "outer_train_median"):
                        add(f"dense_{precision}", r["method_id"], r["qasm_sha256"], r["target_seconds"], r["prediction_seconds"], fold, r["qasm_sha256"], "local_dense")
    maestro_rows = rows(MAESTRO/"evaluation_001/oof_predictions.csv")
    for r in maestro_rows:
        for method in ("maestro_style_cpu_sv_component_optimizer_off_adaptation", "outer_train_median"):
            add("maestro", method, r["source_qasm_sha256"], r["runtime_seconds"], r[f"pred_{method}_seconds"], r["fold"], r["source_qasm_sha256"], "local_maestro")
    selection = {("qpu" if r["context"] == "qualified_qpu_4515" else r["context"], r["id"]): r for r in selected}
    aligned_old = rows(PILOT/"workload_accuracy_cost.csv")
    for r in aligned_old:
        key = (r["context"], r["method"], r["id"]); y, pred, fold, _, _ = original[key]
        assert fold == selection[(r["context"], r["id"])]["fold"]
        if y: assert float(y) == float(r["saved_target_seconds"])
        if pred: assert float(pred) == float(r["saved_prediction_seconds"])
        assert float(r["B1_median_seconds"]) == latency[key]
        if r["context"].startswith("dense_"): assert float(y) == float(dense_targets[(r["context"].split("_")[-1], r["id"])]["target_seconds"])
    population = []
    for context, method in sorted({k[:2] for k in original}):
        full = [(k, v) for k, v in original.items() if k[:2] == (context, method)]
        for label, data in (("full_frozen", full), ("exact_timing_subset", [(k, v) for k, v in full if k in latency])):
            valid = [(k, v) for k, v in data if v[0] and v[1]]
            metric = metrics([(v[0], v[1]) for _, v in data])
            costs = [latency[k] for k, _ in valid if k in latency]
            source_counts = Counter(v[4] for _, v in valid)
            population.append(dict(context=context, method=method, population=label, assigned=len(data), **metric,
                structural_groups=len({v[3] for _, v in valid}), source_counts=json.dumps(source_counts, sort_keys=True),
                B1_requests=len(costs), B1_median_ms=1000*st.median(costs) if costs else "NA", latency_boundary="B1 measured subset only"))
    frozen_full = rows(PILOT/"accuracy_cost_comparisons.csv")
    context_map = {"qualified_qpu_4515": "qpu", "cudaq_dense_statevector_fp32": "dense_fp32", "cudaq_dense_statevector_fp64": "dense_fp64", "qcsim_statevector_cpu_fp64_1000_shots": "maestro"}
    for r in frozen_full:
        match = next(v for v in population if v["population"] == "full_frozen" and v["context"] == context_map[r["context"]] and v["method"] == r["method"])
        assert match["n"] == int(r["accuracy_n"])
        assert math.isclose(match["mae_seconds"], float(r["frozen_mae_seconds"]), rel_tol=0, abs_tol=1e-10)
        assert math.isclose(match["r2"], float(r["frozen_r2"]), rel_tol=0, abs_tol=1e-10)

    paired = []
    def quantile(values, p):
        values = sorted(values); index = (len(values)-1)*p; lower = int(index)
        return values[lower]+(values[min(lower+1,len(values)-1)]-values[lower])*(index-lower)
    for context, graph, mlp in (("qpu", "mali_logical_graph", "mali_global_mlp"),
                               ("dense_fp32", "mali_style_dense_graph", "matched_metadata_mlp"),
                               ("dense_fp64", "mali_style_dense_graph", "matched_metadata_mlp")):
        ids = sorted({k[2] for k in latency if k[:2] == (context,graph)} & {k[2] for k in latency if k[:2] == (context,mlp)})
        differences = defaultdict(list)
        for identity in ids:
            group = selection[(context,identity)]["group"]
            assert group == original[(context,graph,identity)][3]
            differences[group].append(1000*(latency[(context,graph,identity)]-latency[(context,mlp,identity)]))
        values = [st.mean(v) for v in differences.values()]
        rng = random.Random(42)
        resamples = [st.mean(rng.choices(values,k=len(values))) for _ in range(2000)]
        paired.append(dict(context=context, shared_requests=len(ids), groups=len(values), graph_minus_mlp_ms=st.mean(values),
                           low_ms=quantile(resamples,.025), high_ms=quantile(resamples,.975)))
    for r in rows(PILOT/"paired_latency.csv"):
        calculated = next(v for v in paired if v["context"] == r["context"])
        for key, old in (("graph_minus_mlp_ms","equal_group_mean_graph_minus_mlp_ms"), ("low_ms","bootstrap_low_ms"), ("high_ms","bootstrap_high_ms")):
            assert math.isclose(calculated[key],float(r[old]),rel_tol=0,abs_tol=1e-12)

    replay = rows(PILOT/"replay_validation.csv")
    denominator = []
    for context, method in sorted({(r["context"], r["method"]) for r in replay}):
        rr = [r for r in replay if (r["context"], r["method"]) == (context, method)]
        for r in rr:
            if r["status"] == "PASS":
                difference = abs(float(r["replayed_seconds"])-float(r["saved_seconds"]))
                assert difference <= float(r["atol_seconds"])+float(r["rtol"])*abs(float(r["saved_seconds"]))
                assert difference == float(r["abs_diff_seconds"])
        denominator.append(dict(context=context, method=method, selected_ids=len({r["id"] for r in rr}), comparisons=len(rr),
            per_seed_comparisons=sum(r["seed"] in ("42", "1234", "31415") for r in rr), ensemble_comparisons=sum(r["seed"] == "median3" for r in rr),
            native_comparisons=sum(r["seed"] == "native" for r in rr), passed=sum(r["status"] == "PASS" for r in rr), unavailable=sum(r["status"].startswith("UNAVAILABLE") for r in rr)))
    assert Counter(r["status"] for r in replay) == {"PASS": 1049, "UNAVAILABLE_SAVED_OUTPUT": 1}
    seed_groups = defaultdict(dict)
    for r in replay:
        if r["seed"] != "native": seed_groups[(r["context"], r["method"], r["id"])][r["seed"]] = r
    for (context, method, identity), seeds in seed_groups.items():
        assert set(seeds) == {"42", "1234", "31415", "median3"}
        assert float(seeds["median3"]["saved_seconds"]) == st.median(float(seeds[s]["saved_seconds"]) for s in ("42", "1234", "31415"))
        assert float(seeds["median3"]["saved_seconds"]) == float(original[(context, method, identity)][1])
    assert len(seed_groups) == 220
    missing = next(r for r in replay if r["status"] != "PASS")
    missing_maestro = next(r for r in maestro_rows if r["source_qasm_sha256"] == missing["id"])
    assert missing_maestro["target_status"] == "unavailable_unsupported" and missing_maestro["component_prediction_status"] == "normalized_execution_metadata_unavailable"

    # Native timeout diagnostics: text/metadata inspection only, never a parser call.
    representations = {r["canonical_row_id"]: r for r in rows(ROOT/"artifacts/benchmark_v3/real_qpu/compiled_unified/representation_rows.csv")}
    canonical = {r["canonical_row_id"]: r for r in rows(ROOT/"artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv")}
    inputs = []
    qpu_requests = [r for r in selected if r["track"] == "A"]
    archive = ROOT.parent/"Qonductor-SC25/data/database/circuits.zip"
    track(archive)
    for order, r in enumerate(qpu_requests):
        rep = representations[r["id"]]
        if rep["lifecycle_stage"] == "submitted_physical":
            path = str(archive)+"::"+canonical[r["id"]]["qasm_path_or_member"]
            with zipfile.ZipFile(archive) as z: data = z.read(canonical[r["id"]]["qasm_path_or_member"])
            digest = rep["source_qasm_sha256"]; parser = "pinned Qonductor submitted parser"
        else:
            path = ROOT/"artifacts/benchmark_v3/real_qpu/compiled_unified"/rep["compiled_qasm3_file"]
            data = track(path).read_bytes(); digest = rep["compiled_qasm3_sha256"]
            parser = "openqasm3.parse -> hardware-wire AST restoration -> qiskit.qasm3.loads"
        assert hashlib.sha256(data).hexdigest() == digest
        text = data.decode()
        prefix = re.search(r"OPENQASM\s+([^;]+);", text)
        inputs.append(dict(scheduled_order=order, observation_id=r["id"], family=r["family"], backend=r["backend"], path=str(path), sha256=digest,
            source_lifecycle_stage=rep["lifecycle_stage"], parser_input_kind="exact submitted physical QASM" if rep["lifecycle_stage"] == "submitted_physical" else "saved nominal compiled physical QASM3", version=prefix.group(1) if prefix else "UNVERIFIED", file_bytes=len(data), line_count=len(text.splitlines()),
            allocated_width=rep["register_width"], active_width=rep["active_width"], operations=rep["operation_count"], depth=rep["structural_depth"], parser_path=parser,
            observed_failing_id="UNRESOLVED; not retained in failed receipt", per_stage_cap_seconds=30))
    timeout = js(PILOT/"native_replay_timeout_receipt.json")
    track(ROOT/"benchmark_v1/scripts/qasm3_hardware_wire_adapter.py")
    assert timeout["prediction_parity"] == "NOT COMPLETED" and timeout["request_cap_seconds"] == 30
    assert not any(r["context"] == "analytical" for r in raw)

    worker_envs = {}; setups = defaultdict(list); controls = defaultdict(list)
    for path in sorted((PILOT/"workers").glob("*.json")):
        receipt = js(path)
        worker_envs[path.name] = receipt["environment"]
        for binding in receipt["bindings"]: assert sha(track(binding["path"])) == binding["sha256"]
        for stage in receipt["stages"]:
            assert stage["end_ns"]-stage["start_ns"] == stage["elapsed_ns"]
            if stage["stage"] == "child_start_to_ready": setups[path.name.split("_fold")[0]].append(stage["elapsed_ns"]/1e9)
    for r in raw:
        if r["boundary"] == "CONTROL": controls[r["context"]].append(int(r["elapsed_ns"])/1e9)
    assert {k: len(v) for k, v in setups.items()} == {"qpu":15, "dense_fp32":15, "dense_fp64":15, "maestro":3}
    assert all(r["batch"] == "1" and r["synchronization"] == "CUDA before/after" for r in raw if r["context"] != "maestro")
    code = (ROOT/"benchmark_v1/scripts/run_am1_overhead_pilot.py").read_text()
    assert "sync(); s=time.perf_counter_ns(); result=fn(); sync(); e=time.perf_counter_ns()" in code
    assert "statistics.median(predict_seed(method,rid,s,payload) for s in SEEDS)" in code
    assert "if cold==0:" in code
    issues = [
        dict(priority="P0", issue="Dense overhead/execution alignment", status="SUPPORTED WITH QUALIFICATION", resolution="27/30,28/30 Graph and 20/30,22/30 MLP numerically verified from original 15 execution timings and 30 B1 repetitions each. Category B historical cross-software/time comparison, not contemporaneous matched-environment operational break-even."),
        dict(priority="P0", issue="Accuracy population", status="RESOLVED", resolution="Exact timing-subset MAE/MedAE/RMSE/R2 computed from frozen OOF predictions. Full cohort unchanged. Subset 30/50 label-free stratified, not population-representative or full frontier."),
        dict(priority="P1", issue="Qiskit parse timeout", status="UNRESOLVED ROOT CAUSE", resolution="AST parse cap 30 s confirmed. All 50 candidate IDs/sizes recorded, culprit unknown. Parsing started; no transpilation or estimate_duration result. No method speed/accuracy conclusion."),
        dict(priority="P1", issue="Replay denominators", status="RESOLVED", resolution="1,049 PASS/1,050 comparisons includes 660 seed, 220 ensemble, 170 scalar/formula comparisons. 140 context requests. Analytical 50 not part of completed replay denominator."),
        dict(priority="P1", issue="Maestro missing request", status="CORRECTED EXPLANATION", resolution="Missing hash is frozen unavailable_unsupported target and normalized_execution_metadata_unavailable, not lost prediction file. Component 29/30, baseline 30/30 timing but 29/30 labeled accuracy."),
        dict(priority="P1", issue="Instrumentation and scalar timing", status="QUALIFIED", resolution="GPU reference/noop includes synchronization wrapper. Not intrinsic scalar compute; no control subtraction. Graph-MLP whole estimator latency defensible, not device-only architecture causal effect."),
        dict(priority="P1", issue="B2 preprocessing", status="INCONCLUSIVE FULL PIPELINE", resolution="Saved NPZ/hash or normalized IR preparation only. P1 includes I/O/validation and cannot establish complete parse/transpile/DAG or per-stage deployment cost."),
        dict(priority="P1", issue="B3 startup and memory", status="QUALIFIED", resolution="48 fresh timing processes exist but primary warm measurements use cold0 only. Shared six-model bundle includes validation/prediction CSV loading, not standalone deployment. Peak memory process/bundle, not B1 incremental."),
        dict(priority="P1", issue="Validation receipt storage count", status="REPORTING CORRECTION", resolution="Historical validation_report and provenance have different intermediate output_bytes_before_final_receipt. They are not final storage size; audit records actual immutable payload size independently."),
    ]
    files = {
        "simulator_comparisons.csv": csv_text(simulation),
        "accuracy_population_metrics.csv": csv_text(population),
        "replay_denominators.csv": csv_text(denominator),
        "qiskit_selected_inputs.csv": csv_text(inputs),
    }
    env0 = next(iter(runs.values()))[1]
    estimator_env = worker_envs["dense_fp32_fold0_cold0.json"]
    env_table = [dict(item="GPU model / driver", historical=env0["nvidia_smi"], estimator=plan["hardware_inventory"].splitlines()[1]),
                 dict(item="Python / framework", historical=env0["python"]+" / "+env0["cudaq_version"], estimator=estimator_env["python"]+" / Torch "+estimator_env["packages"]["torch"]),
                 dict(item="CUDA runtime", historical="13.0 recorded CuPy runtime API13000; CUDA-Q cu13", estimator=estimator_env["cuda_build"]),
                 dict(item="Kernel", historical=env0["platform"], estimator=estimator_env["platform"]),
                 dict(item="CPU/affinity/load/clocks", historical="not fully recorded for historical sessions", estimator="i7-14700F, SMT8/9 one P-core; residual desktop load; not exclusive whole GPU"),
                 dict(item="Historical GPU UUID", historical="not recorded", estimator="GPU-7c51e3b6-b160-fc40-c14f-292641e5c95e")]
    population_view = [dict(context=r["context"], method=r["method"], population=r["population"], assigned=r["assigned"], n=r["n"], MAE_s=r["mae_seconds"], R2=r["r2"], B1_median_ms=r["B1_median_ms"]) for r in population]
    files["README.md"] = "# AM1 post-pilot scientific audit\n\nVALID WITH CORRECTIONS. Numerical replay and warm Graph/MLP cost remain valid. Historical execution ratios need an explicit cross-environment qualifier. Full B2, standalone cold-start and the missing strongest-classical comparisons remain unanswered. STOP AM1 within its approved scope. This is an audit of existing evidence, not new inference, simulation, timing or model evaluation.\n\nRead [issues](issue_register.md), [simulation alignment](simulator_overhead_alignment.md), [population audit](accuracy_cost_population_audit.md), [fidelity](replay_and_timing_fidelity.md), [corrected findings](corrected_scientific_findings.md), [closure](rq4_closure_decision.md) and [validation](validation_report.md). Historical AM1 files remain immutable. Exact paths and source hashes are in provenance_manifest.json.\n"
    files["issue_register.md"] = "# Issues and resolutions\n\n"+md_table(issues)+"\nNo negative method result is inferred from a missing asset, parse timeout, unsupported target or unmeasured boundary. Corrections refer to the [historical AM1 report](../../am1_overhead_pilot/v1/README.md) and absolute pinned source paths. They do not rewrite raw evidence.\n"
    files["simulator_overhead_alignment.md"] = "# Dense simulator alignment\n\n"+md_table(sim_summary)+"\nEach row in simulator_comparisons.csv joins an exact QASM SHA256, original member alias, precision, fold, normalized and legacy IR digest. Execution is CUDA-Q nvidia sample(32) warm host-return time, excluding build and explicit result iteration. Three session medians of five successful repetitions supply each saved target. B1 is the median of 30 repeated three-seed host-return estimator calls for that hash, with inputs resident and CUDA sync. Repetitions are not independent circuits. The targets are not first-call, end-to-end, state retrieval or pure device kernel times.\n\n"+md_table(env_table)+"\nClassification B: explicit historical cross-software/time comparison on matched GPU model and driver, rather than A fully matched contemporaneous environment. Historical GPU UUID, CPU affinity, clock/load and complete hardware reservation are unavailable. Sources point to the same project workstation but do not prove all operating conditions match. Estimator uses FP32 arithmetic for both simulation precision targets.\n\nCorrection to historical accuracy_cost_analysis.md and the completion message: retain the four numeric counts as descriptive ratios against saved CUDA-Q execution labels. Do not present them as a measured operational speedup/break-even, universal reason to skip prediction, or contemporaneous same-environment comparison. Size support is q2–9, no large-width law. Prediction-only excludes raw-input preparation and setup. These omissions generally add estimator cost but do not quantify total deployment latency.\n"
    files["accuracy_cost_population_audit.md"] = "# Accuracy population alignment\n\n"+md_table(population_view)+"\nFull frozen scores are independently recomputed from authoritative saved predictions and match the historical AM1 accuracy table within 1e-10. No leaderboard value or prediction is replaced. Full assigned populations: QPU 4,515, each dense precision 150, Maestro150 with 143 observed targets. Exact timing subset: QPU 50, dense 30/context, Maestro component 29/30 and median30/30 with 29 evaluable labels. Detailed MedAE/RMSE, group support, source counts and denominators: accuracy_population_metrics.csv.\n\nQuery routing uses its frozen held-out outer fold, with the original three-seed median for neural outputs. The timing-subset metrics are descriptive post-pilot calculations of already-saved predictions, not new inference, reselection, calibration or tuning. Selection used metadata but is deliberately stratified and does not estimate the full-population utility without a sampling model. QPU timing50 differs from full source proportions. Pooled and source/group-balanced accuracy must not be conflated.\n\nA descriptive accuracy-latency comparison is now possible for exactly aligned IDs and context. This does not supply a complete Pareto frontier, an exchange rate between MAE and compute time, error-tolerance guarantees or a scheduler benefit. Missing original Qonductor selector/tree and dense RF/XGB fitted states prohibit corresponding cost claims. Maestro median's extra 30th timing has no execution label and is excluded from paired accuracy.\n"
    files["replay_and_timing_fidelity.md"] = "# Replay and measurement fidelity\n\n"+md_table(denominator)+"\nCounts: 660 per-seed + 220 ensemble + 170 scalar/formula =1,050 comparisons, 1,049 PASS and 1 unavailable. Unique context requests: 140: QPU 50, FP32/FP64 each 30, Maestro 30. Dense circuits can repeat across precision/Maestro; do not sum these as unique physical circuits. No analytical 50 replay completed; they are excluded from 1,050, not silently successful. Tolerance: 1e-6 s+1e-5*abs(saved) for FP32 neural, 1e-12+1e-10 for scalar. Maximum small-prediction relative discrepancy can exceed 1e-5 because absolute tolerance applies. Values remain within gate. All 220 ensemble saved medians independently match three saved seed outputs and full OOF medians.\n\nMaestro unavailable ID `"+missing["id"]+"`: target_status=unavailable_unsupported, runtime empty, normalized metadata missing, component_prediction_status=normalized_execution_metadata_unavailable. Coverage is 29/30 component timing, not a model output lost during AM1; reference timing 30/30 but accuracy 29/30.\n\nB1 host timer starts after initial CUDA sync, includes three actual sequential model calls, each output transfer/inverse transform, host median and final sync. All batch 1. No measurement is seed-latency multiplied by 3. Graph/MLP input contracts differ as frozen, so comparisons concern implemented estimators, not a controlled pure-topology causal effect. Scalar references in GPU workers pay final sync overhead; controls ~6–7 microseconds versus reference ~8–10 microseconds dominate intrinsic scalar compute. Maestro CPU no-op ~0.28 microseconds is smaller but nonzero relative to component ~1.76 microseconds. No controls are subtracted. Control is no-op with the timing wrapper, not separate device-kernel time.\n\nTen warmups per method/bundle use its first eligible request, not ten for every workload. 30 primary repetitions come from cold process 0 only; cold1/2 contribute startup and first-after-ready, not three independent warm campaigns. Matched Graph/MLP methods rotate order across repetitions. Bootstrap of request medians by structural group addresses workload selection, not repeated-process or operating-system temporal uncertainty.\n\nB1 and partial P1 are timed sequentially; P1 also changes caches/allocator state, so subtraction is not isolated stage profiling. P1 QPU includes disk read/hash; dense starts at saved IR/global metadata. Neither is complete B2. B3 has48 fresh process bundles, shared six models for neural contexts, all inputs, expected prediction CSV/hash validation and warmups. It starts after Python interpreter startup; child-start-to-ready and parent-launch-to-exit are different intervals. There is no parent-launch-to-ready timer or standalone model cold deployment measurement. Memory is bundle/process absolute peak. Native manual failure lacks per-input identity/progress and peak RAM trace.\n"
    first = inputs[0]
    files["corrected_scientific_findings.md"] = "# Corrected findings\n\n| Claim | Assessment | Correct wording |\n|---|---|---|\n| Frozen replay | SUPPORTED | All completed neural/formula/statistical replay comparisons match within declared tolerance. This is subset identity, not full replay coverage. |\n| Graph versus MLP B1 | SUPPORTED WITH QUALIFICATION | Graph requires more warm host-return time on shared selected requests with frozen ensemble and input contracts. Not pure graph-architecture causal cost or production latency. |\n| Dense overhead versus execution | SUPPORTED WITH QUALIFICATION | 27/30 and 28/30 Graph, 20/30 and22/30 MLP B1 medians exceed historical CUDA-Q warm sample32 labels, in different software/time environments with matched GPU model/driver. |\n| Full preprocessing cost | INCONCLUSIVE | P1 shows saved-input preparation adds cost and tail variability; complete raw QASM B2 and isolated per-stage costs remain unmeasured. |\n| Practical estimator-selection benefit | INCONCLUSIVE | Lower MLP compute cost is measured; accuracy requirements, strongest classical costs, hardware portability and scheduler gains are not established. |\n| Native Qiskit slow/inaccurate | UNSUPPORTED | A setup AST parse hit cap before native inference. No native timing or accuracy qualification resulted. |\n| Standalone cold start / incremental B1 memory | UNSUPPORTED | Shared-bundle setup/absolute process peaks do not identify these quantities. |\n\n## Native parse evidence\n\nManual receipt: INCOMPLETE_REQUEST_TIMEOUT, 30 s signal alarm per setup stage, 120 s process bound. Failure path is `parse_frozen_compiled -> restore_saved_qasm3_hardware_wires -> openqasm3.parse`; no parse/transpile/timing rerun in this audit. The current path loads already compiled nominal physical circuits, so fresh transpilation was never intended. estimate_duration and frozen affine bridge are after all 50 parses and were not reached.\n\nAll 50 scheduled input IDs, sizes, widths, operations and parser routes appear in qiskit_selected_inputs.csv. First scheduled input is `"+first["observation_id"]+"`, family "+first["family"]+", "+str(first["file_bytes"])+" bytes, "+str(first["operations"])+" operations. This is ordering evidence, NOT proof it was the failing input. The failure receipt saves neither ID nor successful preceding parse stages. Root cause and culprit remain UNRESOLVED. Header/layout style is consistent with the intended OpenQASM3 hardware-wire AST adapter, but text/header inspection cannot validate AST compatibility. Input size is a plausible contributor, not a verified cause.\n\n## Historical statements requiring clarification\n\n1. accuracy_cost_analysis.md: add the category B software/time qualifier to execution ratios.\n2. failures_and_exclusions.md: explain Maestro's missing output is frozen unsupported target/normalized metadata, not a file-retention loss.\n3. validation_report.md/provenance_manifest.json intermediate storage counts differ. Both are historical non-final observations; final immutable payload size is audited here, not inferred from those fields.\n4. Correct the summary interpretation of total duration: execution_manifest.wall_seconds covers the corrected coordinated run and excludes earlier attempts/manual native reconciliation. resource_closure is later full predeclaration elapsed. Neither is a pure estimator compute sum.\n\nNo historical source or raw evidence overwritten.\n"
    files["rq4_closure_decision.md"] = "# RQ4 closure decision\n\nSTOP AM1. VALID WITH CORRECTIONS for the narrow controlled pilot.\n\nThe working RQ4 asks prediction overhead and accuracy-cost. Replay-qualified warm ensemble latency, partial saved-input preparation, shared-bundle setup and context-specific limitations now have direct evidence. Thus RQ4 is partly answered and the approved B1 pilot can close. The full accuracy-cost/online scalability question cannot be marked fully closed. Historical RQ4 generalization grid remains a separate completed experiment.\n\nA complete B2 study could materially change a deployment-selection conclusion; retained strongest-classical fitted-state comparison could change an accuracy-cost ranking. Neither conclusion was established by this pilot. These gaps do not invalidate bounded Graph-MLP B1 evidence or require new measurement to correct it. New measurement would need new authorization and cannot recover missing original fitted state by refitting and relabeling. Qiskit timeout diagnostics remain unresolved without a new traced parse attempt, which is not necessary to preserve existing B1 conclusions. No follow-up proposed merely to add models/rows/charts.\n\nRecommendation: accept qualified historical execution ratios and aligned subset accuracy, retain explicit full-B2/classical/cold-start gaps, keep AM1 stopped. No automatic retry or expanded pilot.\n"
    files["accuracy_cost_population_audit.md"] += "\n## Sampling and interpretation\n\nThe QPU timing subset contains 38 Ma–Li, 7 Qonductor and 5 QPack observations. The full cohort contains 340, 230 and 3,945 respectively. The subset gives metadata/size coverage and heavily enriches Ma–Li, rather than approximating the pooled deployment mix. Its MAE and R² must not replace or explain full-cohort performance as a representative sample. Dense FP64 Graph R² is about 0.925 on the selected 30 but about 0.136 on the full 150. This is a population effect, not improved model fitting. Maestro's baseline latency on 29 label-matched requests is 0.7345 microseconds, slightly different from 0.7355 over all 30 timed requests.\n"
    files["replay_and_timing_fidelity.md"] += "\n## Independently reconciled paired latency\n\n"+md_table(paired)+"\nThe fixed seed 42, 2,000 structural-group bootstrap reproduces the historical intervals. It resamples selected groups, not timing repetitions. These positive intervals support extra Graph cost within the selected contexts. They do not include independent day/process variability, changed background load, unobserved full B2 or hardware portability.\n"
    track(__file__)
    track(ROOT/"benchmark_v1/tests/test_am1_postpilot_audit.py")
    validation = dict(status="PASS_WITH_REPORTING_CORRECTIONS", historical_pilot_payload_files=len(manifest["payload"]), historical_pilot_bytes=sum(p.stat().st_size for p in PILOT.rglob("*") if p.is_file()),
        immutable_original_source_pins=len(manifest["source_pins"]), raw_timing_rows=len(raw), raw_boundaries=dict(boundaries), replay_statuses=dict(Counter(r["status"] for r in replay)),
        seed_comparisons=660, ensemble_comparisons=220, scalar_formula_comparisons=170, neural_query_method_groups=len(seed_groups),
        dense_execution_identity_rows=len(simulation), original_execution_repetitions_checked=15*len(simulation), simulator_ratios=sim_summary,
        accuracy_population_rows=len(population), paired_latency_contexts_verified=len(paired), selected_requests=len(selected), total_sources_audited=len(pins), qiskit_candidate_inputs=len(inputs), qiskit_failed_input="UNRESOLVED",
        new_inference=False, new_parser_call=False, new_measurement=False, training=False, original_data_modified=False, complete_Pareto_frontier=False, full_RQ4_closed=False, stop_am1=True)
    files["validation_report.md"] = "# Saved-evidence audit validation\n\n```json\n"+json.dumps(validation, indent=2)+"\n```\n\nFresh checks recompute timings from nanosecond intervals, exact 30-repetition membership, selected fold identities, dense target reduction from three sessions of five original measurements, QASM/IR/precision/shots, full and subset prediction metrics, seed medians, declared tolerances, missing-target semantics and 48 setup bundles. Final pilot payload bytes are measured read-only; historical interim byte fields are not final totals. Sources and all old AM1 artifacts are hash-pinned before and after.\n\nReproduce this audit without any model/engine call: `python -B benchmark_v1/scripts/build_am1_postpilot_audit.py --check`. No imports of Torch, CUDA-Q, Qiskit parser or producing runners. Source/payload hashes are in provenance_manifest.json. Unknown historical GPU UUID and timeout culprit remain unknown; arithmetic validation is not operating-condition equivalence.\n"
    track(__file__)
    for path, digest in pins.items(): assert sha(path) == digest, path
    inherited_paths = set(manifest["source_pins"]) | {str(PILOT/name) for name in manifest["payload"]}
    direct_pins = {path:digest for path,digest in pins.items() if path not in inherited_paths}
    files["provenance_manifest.json"] = json.dumps(dict(source_pins=direct_pins,
        inherited_source_manifest=dict(path=str(PILOT/"provenance_manifest.json"),sha256=sha(PILOT/"provenance_manifest.json"),source_pins=len(manifest["source_pins"]),payload_files=len(manifest["payload"]),validation="all inherited digests rechecked on every audit call"),
        payload={name:hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()},
        validation=validation, authority="derived saved-evidence audit; preserves Final Scientific Freeze v2 and AM1 originals", self_excluded=True), indent=2, sort_keys=True)+"\n"
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--emit", nargs="+", help="emit only named derived files as JSON")
    args = parser.parse_args()
    files = audit()
    if args.check:
        for name, content in files.items(): assert (OUT/name).read_text() == content, name
        print(json.dumps(dict(status="PASS", reports=8, reconciliation_tables=4, saved_evidence_only=True, scientific_verdict="VALID WITH CORRECTIONS")))
    else:
        print(json.dumps({name:files[name] for name in args.emit} if args.emit else files))


if __name__ == "__main__": main()
