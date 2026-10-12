"""Static AM1 feasibility reports, not an estimator timing harness.

Stdlib only. Reads JSON/CSV/source and hashes bytes; never imports model code,
deserializes checkpoints, fits, predicts, simulates or measures latency.
Default: validate existing reports. --emit-patch: print an explicit new-file
patch; no filesystem writes. Reports deliberately retain frozen paths.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/am1_feasibility/v1"
P = "artifacts/real_qpu/qualified_logical_panel/"
D = "work/benchmark_recovery/r8_dense_predictors/attempt_002/"
W = "work/repo_finalization/publish_candidate/"
S = W + "artifacts/benchmark_v3/simulator/"
M = S + "maestro_cpu_component_optimizer_off/"
J = S + "family_aware_joint_mps_ladder_v1/training/"
F = "work/benchmark_recovery/r9_mps/fixed_chi16_recovered_v1/"
R = "work/benchmark_recovery/r9_mps/family_residual_recovered_v1/"
A = S + "azizov_common_core_adaptation/"
SEEDS = (42, 1234, 31415)
STATUSES = ("READY", "CONDITIONAL", "BLOCKED", "NOT APPLICABLE")
FIELDS = ("estimator_id", "context_id", "track", "category", "frozen_experiment_id",
          "saved_asset_locations", "integrity_evidence", "preprocessing",
          "feature_availability", "backend_metadata", "calibration_artifacts",
          "dependency_environment", "deterministic_replay", "inference_without_refit",
          "initialization", "status", "blockers", "eligible_accuracy_rows",
          "evaluation_target_clock", "native_output_semantics", "producing_source",
          "fidelity_card", "fidelity_class")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    csv.field_size_limit(100_000_000)
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(Path(path).read_text())


def link(path, label=None):
    p = Path(path)
    return f"[{label or p.name}]({p if p.is_absolute() else ROOT / p})"


def closure():
    path = ROOT / "benchmark_v1/scripts/build_research_question_closure.py"
    spec = importlib.util.spec_from_file_location("am1_saved_closure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_assets():
    """Verify saved asset bytes against producing receipts; no torch.load."""
    pins, checks, constants = {}, collections.Counter(), {}

    def pin(path, expected=None):
        path = Path(path)
        path = path if path.is_absolute() else ROOT / path
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f"asset_hash_mismatch:{path}")
        pins[str(path)] = digest
        return digest

    c = closure()
    c.validate()
    pins.update(c.load_sources())
    for path in sorted((ROOT / "artifacts/research_question_closure/v1").glob("*")):
        if path.is_file():
            pin(path)
    pin("artifacts/thesis_interpretation/v1/evidence_to_claims.md")
    pin("artifacts/thesis_interpretation/v1/tables/method_context_index.csv")
    pin("artifacts/final_research/manifest.json")
    pin("artifacts/final_research/evidence.json")
    # The freeze manifest is historical reporting authority. Do not reinterpret
    # later approved reporting edits as corrupted predictions or rewrite pins.
    for path in ("benchmark_v1/scripts/run_mali_full_features.py",
                 "benchmark_v1/scripts/run_mali_batched.py",
                 "benchmark_v1/scripts/run_cudaq_dense_neural_predictors_v1.py",
                 "benchmark_v1/scripts/run_cudaq_dense_classical_predictors_v1.py",
                 "benchmark_v1/scripts/train_family_aware_joint_mps_v1.py",
                 "benchmark_v1/scripts/run_common_cutensornet_scalar.py",
                 "benchmark_v1/scripts/run_strict_logical_analytical.py",
                 "benchmark_v1/scripts/resume_qualified_analytical.py",
                 "benchmark_v1/scripts/run_mps_fixed_chi16_runtime_adaptation_v1.py",
                 W + "benchmark_v1/scripts/run_family_residual_fixed_mps_runtime_v1.py",
                 W + "benchmark_v1/scripts/run_azizov_common_core_adaptation_v1.py",
                 W + "benchmark_v1/scripts/evaluate_maestro_cpu_component_v1.py"):
        pin(path)
    pin(ROOT.parent / "qcre/estimate.py", "e85873a810519c2741df281de5f9a8c6a89cedf80c17da65b74c7df39fa497c0")
    pin(ROOT.parent / ".venv-cdaa-new/lib/python3.10/site-packages/bqskit/ext/qiskit/translate.py",
        "112c85936588ab7849d1cd6c40324b4344376c9cba176d0b27718796fa2e5dd5")

    for path in sorted((ROOT / (P + "neural")).glob("fold_*/*/seed_*/manifest.json")):
        value = read_json(path)
        assert value["status"] == "complete"
        pin(path)
        for name, expected in value["outputs"].items():
            pin(path.parent / name, expected)
            checks["qpu_checkpoint_files" if name.endswith(".pt") else "qpu_prediction_files"] += 1
    for path in sorted((ROOT / (P + "neural")).glob("fold_*/transform.json")):
        pin(path)
        transform = read_json(path)
        digest = hashlib.sha256(json.dumps(transform, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        assert all(read_json(p)["identity"]["transform"] == digest
                   for p in path.parent.glob("*/seed_*/manifest.json"))
        checks["qpu_transform_files"] += 1
    for name in ("manifest.json", "summary/oof_predictions.csv",
                 "analytical/calibration_fits.csv", "analytical/attempts.csv",
                 "analytical/qcre_source_receipt.json", "analytical/qcre_context_cache.json",
                 "analytical/execution_receipt.json"):
        pin(P + name)
    parent = read_json(ROOT / (P + "manifest.json"))
    for name, expected in parent.get("outputs", {}).items():
        pin(P + name, expected)
    for name, expected in parent["graph_hashes"].items():
        pin(name, expected)
        checks["qpu_prepared_graph_files"] += 1
    for name, expected in parent["input_hashes"].items():
        pin(name, expected)
    # Constants are fully specified by one saved scalar per declared context.
    # Extracting that scalar is not reconstructing a missing fitted regressor.
    pin("artifacts/benchmark_deep_dive/scripts/qpu.py")
    pin("artifacts/benchmark_deep_dive/tables/qpu_baseline_predictions_v1.csv")
    pin("artifacts/benchmark_deep_dive/diagnostics/qpu_baseline_fits_v1.json")
    qpu = read_csv(ROOT / "artifacts/benchmark_deep_dive/tables/qpu_baseline_predictions_v1.csv")
    for method in ("outer_train_mean", "outer_train_median", "outer_train_source_median"):
        values = collections.defaultdict(set)
        for row in qpu:
            if row["method_id"] == method and row["predicted_seconds"] != "":
                key = row["outer_fold"] + ("|" + row["source_id"] if method.endswith("source_median") else "")
                values[key].add(float(row["predicted_seconds"]))
        assert values and all(len(v) == 1 for v in values.values())
        constants["A/" + method] = {k: next(iter(v)) for k, v in sorted(values.items())}

    for precision in ("fp32", "fp64"):
        for path in sorted((ROOT / (D + precision + "/neural")).glob("fold_*/*/seed_*/complete.json")):
            value = read_json(path)
            pin(path)
            model = path.parent.parent / "checkpoints" / f"seed_{value['seed']}.pt"
            pin(model, value["checkpoint_sha256"])
            pin(path.parent / "predictions.csv", value["predictions_sha256"])
            checks["dense_checkpoint_files"] += 1
        for path in sorted((ROOT / (D + precision + "/neural")).glob("fold_*/fold_inputs.json")):
            pin(path)
            checks["dense_transform_files"] += 1
        scalars = {}
        for fold in range(5):
            base = ROOT / (D + precision + f"/classical/fold_{fold}")
            value = read_json(base / "fold_manifest.json")
            pin(base / "fold_manifest.json")
            pin(base / "predictions.csv", value["predictions_csv_sha256"])
            vals = {float(r["prediction_seconds"]) for r in read_csv(base / "predictions.csv")
                    if r["method_id"] == "outer_train_median" and r["prediction_seconds"] != ""}
            assert len(vals) == 1
            scalars[str(fold)] = next(iter(vals))
        constants[precision + "/outer_train_median"] = scalars

    for split in ("c44", "family_holdout"):
        for fold in range(5):
            base = ROOT / (J + split + f"/fold_{fold}")
            value = read_json(base / "run_manifest.json")
            assert value["status"] == "completed_fold_technical_qa_pass"
            pin(base / "run_manifest.json")
            pin(base / "fold_transform_and_support.json", value["transform_sha256"])
            pin(base / "oof_predictions.csv", value["prediction_sha256"])
            checks["joint_transform_files"] += 1
            for directory in ("checkpoints", "family_classifier_checkpoints"):
                for name, expected in value[directory].items():
                    pin(base / directory / name, expected)
                    checks["joint_runtime_checkpoint_files" if directory == "checkpoints" else "joint_classifier_checkpoint_files"] += 1

    for base in (P + "regression", F, R, A, M + "evaluation_001"):
        folder = ROOT / base
        files = [p for p in folder.rglob("*") if p.is_file()]
        assert folder.is_dir() and not [p for p in files if p.suffix in (".pt", ".pkl", ".pickle", ".joblib", ".onnx", ".safetensors")]
        checks["unsaved_model_roots_inspected"] += 1
        for path in files:
            if path.suffix == ".json":
                pin(path)
    for path in sorted((ROOT / (P + "analytical")).glob("hyb_ridge_fold_*.json")):
        pin(path)
        assert all("coef" not in fit and "coef_" not in fit and "intercept" not in fit
                   for fit in read_json(path)["fits"])
    model = ROOT / (M + "full_001/calibration_model.json")
    pin(model)
    pin(M + "evaluation_001/oof_predictions.csv")
    maestro = read_json(model)
    assert maestro["artifact_status"] == "PASS" and maestro["assigned_calibration_cells"] == 112
    assert len(maestro["width_coefficients"]) == 8
    assert all(r["status"] == "valid" and all(math.isfinite(float(r[k])) for k in ("C1q", "Ccx", "b", "m"))
               for r in maestro["width_coefficients"])
    checks["maestro_saved_width_knots"] = 8
    vals = collections.defaultdict(set)
    for row in read_csv(ROOT / (M + "evaluation_001/oof_predictions.csv")):
        if row["pred_outer_train_median_seconds"]:
            vals[row["fold"]].add(float(row["pred_outer_train_median_seconds"]))
    assert len(vals) == 5 and all(len(v) == 1 for v in vals.values())
    constants["maestro/outer_train_median"] = {k: next(iter(v)) for k,v in sorted(vals.items())}
    fixed_scalars = {}
    for fold in range(5):
        path = ROOT / (F + f"fold_{fold}/fold{fold}_predictions.csv")
        pin(path)
        rows = read_csv(path)
        # This producer uses wide prediction columns; record the exact one.
        cols = [k for k in rows[0] if "median" in k and "three" not in k and "graph" not in k]
        assert cols, rows[0].keys()
        selected = [k for k in cols if all(r[k] not in ("", None) for r in rows)]
        constant_cols = {k: {float(r[k]) for r in rows} for k in selected}
        constant_cols = {k:v for k,v in constant_cols.items() if len(v) == 1}
        assert len(constant_cols) == 1, constant_cols
        name, value = next(iter(constant_cols.items()))
        fixed_scalars[str(fold)] = {"column": name, "value": next(iter(value))}
    constants["fixed_mps/train_fold_median"] = fixed_scalars
    for path in sorted((ROOT / (P + "regression")).glob("fold_?.json")):
        value = read_json(path)
        pin(path.parent / (path.stem + ".csv"), value["predictions_sha256"])
    checks["constant_estimators_verified"] = len(constants)
    assert checks["qpu_checkpoint_files"] == 30
    assert checks["dense_checkpoint_files"] == 60
    assert checks["joint_runtime_checkpoint_files"] == 60
    assert checks["joint_classifier_checkpoint_files"] == 30
    assert checks["qpu_transform_files"] == 5 and checks["dense_transform_files"] == 10
    assert checks["joint_transform_files"] == 10
    return pins, dict(checks), constants, c.facts()


def inventory():
    results = []
    qpu = read_csv(ROOT / "artifacts/benchmark_deep_dive/tables/qpu_metrics_v1.csv")
    sim = read_csv(ROOT / "artifacts/benchmark_deep_dive/tables/simulator_metrics_v1.csv")
    fidelity = read_csv(ROOT / "artifacts/thesis_interpretation/v1/tables/method_context_index.csv")
    native_clocks = collections.defaultdict(set)
    for attempt in read_csv(ROOT / (P + "analytical/attempts.csv")):
        native_clocks[attempt["method_id"]].add(attempt["method_output_clock"])
    def simulator_clock(context):
        if context.startswith("cudaq_dense"):
            return "warm CUDA-Q sample(32) execution seconds; precision in context_id"
        if context.startswith("cudaq_mps"):
            return "warm CUDA-Q get_state execution seconds; quality labels retained separately"
        if context.startswith("qiskit_aer"):
            return "warm noisy Aer execution seconds; FakeSherbrooke opt1 shots1024"
        if context.startswith("qcsim"):
            return "QCSim engine-reported execution seconds, not whole API wall time"
        if context == "native_same_plan_scalar_amplitude":
            return "same-plan CUDA-event warm scalar contraction seconds"
        return "host MPSBackend.run seconds; analog companion only"

    def row(method, context, track, category, root, source, preprocessing, features,
            backend="none", calibration="none", environment="stdlib", status="READY",
            blocker="none identified by static inspection; future parity gate required", n="",
            output="predicted observed execution/service seconds", init="small JSON/scalar setup"):
        value = dict(zip(FIELDS, [method, context, track, category, root.rstrip("/"), root,
            "source_pins and asset_checks in provenance_manifest.json", preprocessing, features,
            backend, calibration, environment,
            "conditional on saved-input parity; bitwise cross-version replay not assumed" if status == "READY" else "not established for this variant",
            "yes; existing states/scalars/formula only" if status == "READY" else "no under current assets" if status == "BLOCKED" else "separate native-pipeline eligibility" if status == "CONDITIONAL" else "not a validated predictor",
            init, status, blocker, str(n),
            "archived_observed_service_execution_time" if track == "A" else simulator_clock(context),
            output, source]))
        card_method = {"scholten_original_paper":"scholten_original",
                       "cutensornet_runtime_est":"tn",
                       "pasqal_emu_mps_resource_diagnostic":"pasqal"}.get(method, method)
        cards = [r for r in fidelity if r["track"] == track and r["method_id"] == card_method
                 and (track == "A" or r["context_id"] == context or card_method in ("tn", "pasqal"))]
        assert len(cards) == 1, (method, context, cards)
        value.update(fidelity_card=cards[0]["card_id"], fidelity_class=cards[0]["category"])
        assert set(value) == set(FIELDS)
        results.append(value)

    for record in qpu:
        method = record["method_id"]
        kw = dict(n=record["n"])
        if method.startswith("mali_"):
            row(method, "qualified_qpu_4515", "A", "graph+global" if method.endswith("graph") else "MLP",
                P + "neural/fold_{0..4}/" + method + "/seed_{42,1234,31415}/checkpoint.pt",
                "benchmark_v1/scripts/run_qualified_logical_benchmark.py",
                ("logical recipe/QASM to DAG with 178-dimensional node features and 51 globals" if method.endswith("graph") else "logical recipe/QASM to 51 global columns (no node/edge tensors required)") + "; saved fit-only transform selects 40; three predictions then median",
                P + "manifest.json and logical_inputs/graphs; fold transforms saved",
                backend="nominal wire-index T1/T2 in graph nodes; no backend identity/shots encoding added" if method.endswith("graph") else "no calibration/backend identity/shots input",
                calibration=P + "neural/fold_{0..4}/transform.json",
                environment="GPU Python: Torch 2.7.1+cu128; PyG 2.6.1; NumPy 1.26.4; CUDA only",
                init="process/import/CUDA/model load; select best_model_state, not last model_state", **kw)
        elif method.startswith("outer_train_"):
            row(method, "qualified_qpu_4515", "A", "statistical reference", "artifacts/benchmark_deep_dive/diagnostics/qpu_baseline_fits_v1.json",
                "artifacts/benchmark_deep_dive/scripts/qpu.py",
                "fold routing; source lookup only for source median", "unique saved scalar per fold or fold/source; values in provenance",
                backend="none; source-median explicitly requires source identity", **kw)
        elif method.startswith("qonductor_") or method == "hyb_nominal_r2_log_cost_ridge_v1":
            is_hyb = method.startswith("hyb_")
            row(method, "qualified_qpu_4515", "A", "Ridge" if is_hyb else "regression/tree/selector",
                P + ("analytical/hyb_ridge_fold_{0..4}.json" if is_hyb else "regression/fold_{0..4}.json"),
                "benchmark_v1/scripts/run_qualified_logical_benchmark.py",
                "compiled duration/error/T2 log-cost transform" if is_hyb else "native counter semantics; compiled 2Q/depth/width + shots + circuit_count=1; outer/inner selection",
                "features/predictions/selection receipts available; fitted coefficients/tree states absent",
                environment="CPU scikit-learn 1.7.2; fitting is forbidden in AM1",
                status="BLOCKED", blocker="no serialized fitted model or complete coefficient/scaler state; selected hyperparameters are not a checkpoint", **kw)
        else:
            base, variant = method.split("__")
            qcre = base.startswith("qcre")
            backend = "pinned nominal Target/durations" + ("; error/T2/readout" if base.startswith("hyb_") else "; nominal throughput table" if base.startswith("scholten") else "")
            source = "benchmark_v1/scripts/run_strict_logical_analytical.py" if qcre else "benchmark_v1/scripts/run_qualified_logical_benchmark.py"
            row(method, "qualified_qpu_4515", "A", "analytical/timing/proxy" + (" + frozen calibration bridge" if variant != "raw" else ""),
                P + "analytical/calibration_fits.csv; " + P + "analytical/attempts.csv", source,
                "logical recipe to frozen target-transpiled representation; native calculation; optional shot scaling; saved affine/log-affine bridge",
                "eligible compiled features/duration components and input identities saved; no simulation or fresh fitting",
                backend=backend, calibration="5 saved slope/intercept pairs per bridge; raw requires none",
                environment="CPU Qiskit 2.5.2/runtime 0.49.0; QCRE pinned estimate.py and external BQSKIT 1.2.0 site/adapter when required",
                output="; ".join(sorted(native_clocks[method])),
                init="backend snapshot load/import; QCRE conversion setup; no optimization/calibration fitting", **kw)
    row("scholten_original_paper", "qualified_qpu_4515", "A", "original analytical unavailable",
        P + "analytical/attempts.csv", "benchmark_v1/scripts/run_qualified_logical_benchmark.py",
        "original effective depth/kernel contract", "kernel_n and template_D not established",
        status="BLOCKED", blocker="original kernel/effective-depth eligibility absent; nominal proxy is a different estimator", n=0,
        output="original throughput seconds (unavailable)")

    for record in sim:
        method, context = record["method_id"], record["context_id"]
        kw = dict(n=record["n"], output="prediction in seconds for this frozen simulator target, not estimator latency")
        if context.startswith("cudaq_dense"):
            precision = context.rsplit("_",1)[1]
            root = D + precision + "/"
            neural = method in ("mali_style_dense_graph", "matched_metadata_mlp")
            constant = method == "outer_train_median"
            row(method, context, "B", "graph" if method.endswith("graph") else "MLP" if neural else "statistical reference" if constant else "regression/tree",
                root + ("neural/fold_{0..4}/" + method + "/checkpoints/seed_{42,1234,31415}.pt" if neural else "classical/fold_{0..4}/fold_manifest.json"),
                "benchmark_v1/scripts/run_cudaq_dense_" + ("neural" if neural else "classical") + "_predictors_v1.py",
                "source DAG + seven globals, saved gate vocabulary/log1p scaler; median of three CUDA predictions" if neural else "saved fold scalar" if constant else "seven globals or work features; saved transform but fitted weights absent",
                root + "neural/fold_{0..4}/fold_inputs.json and frozen source topology",
                backend="simulator FP32/FP64 context fixed; no CUDA-Q workload is needed for estimator inference",
                calibration="saved fold transforms; never refit",
                environment="Torch 2.7.1+cu128/PyG 2.6.1 CUDA" if neural else "stdlib" if constant else "CPU sklearn/XGBoost dependencies insufficient without fitted objects",
                status="READY" if neural or constant else "BLOCKED",
                blocker="future saved-prediction parity gate" if neural or constant else "only fit receipts/predictions/params saved; no fitted regressor state",
                init="CUDA/model load for neural; scalar lookup for median; estimator precision is not simulator precision", **kw)
        elif context.startswith("cudaq_mps_ladder"):
            split = "c44" if context.endswith("c44") else "family_holdout"
            row(method, context, "B", "joint runtime-quality neural model", J + split + "/fold_{0..4}/",
                "benchmark_v1/scripts/train_family_aware_joint_mps_v1.py",
                "static features + chi, saved transform; conditioned variant includes three saved family classifiers; runtime median and quality-probability mean",
                "all runtime heads, classifier states, vocabularies and transform JSON saved",
                backend="CUDA-Q MPS FP64 fixed ladder context; do not run MPS workload",
                calibration="fold_transform_and_support.json; no fitting",
                environment="Torch 2.7.1+cu128/PyG 2.6.1; CUDA inference",
                init="load three runtime heads; conditioned variant also loads three family classifiers", **kw)
        elif context.startswith("cudaq_mps_fp64_bond16"):
            constant = method == "train_fold_median"
            root = R if "family_" in method else F
            row(method, context, "B", "statistical reference" if constant else "Ridge/graph/residual neural",
                root, "benchmark_v1/scripts/run_mps_fixed_chi16_runtime_adaptation_v1.py" if root == F else W + "benchmark_v1/scripts/run_family_residual_fixed_mps_runtime_v1.py",
                "fold scalar" if constant else "source features/DAG/family classification and fold transforms",
                "saved targets/predictions only; no learned checkpoint" if not constant else "saved unique fold scalar",
                backend="chi16 FP64 warm get_state; 150 runtime rows, 147 quality passes audited separately; do not run CUDA-Q",
                status="READY" if constant else "BLOCKED", blocker="future scalar parity gate" if constant else "runner did not serialize fitted runtime model; other chi/target checkpoints cannot substitute", **kw)
        elif context.startswith("qiskit_aer"):
            view = next(v for v in ("source", "hybrid", "transpiled") if v in method)
            row(method, context, "B", "GNN" if method.startswith("azizov_gnn") else "regression/tree", A,
                W + "benchmark_v1/scripts/run_azizov_common_core_adaptation_v1.py",
                view + " features/DAG; 41/54/41 view contracts; FakeSherbrooke opt1; three-seed GNN median where applicable",
                "QPY/QASM, materialized graphs/features and predictions exist; trained estimator state absent",
                backend="FakeSherbrooke snapshot; Aer noisy opt1 shots1024 context, not measured QPU",
                environment="historical sklearn/XGBoost or Torch CUDA/PyG; assets, not package installation, are blocker",
                status="BLOCKED", blocker="no saved GNN weights or fitted classical models in authoritative output; do not refit", **kw)
        elif context.startswith("qcsim"):
            component = method.startswith("maestro_style")
            constant = method == "outer_train_median"
            row(method, context, "B", "component" if component else "statistical reference" if constant else "Ridge/graph",
                M + ("full_001/calibration_model.json" if component else "evaluation_001/oof_predictions.csv"),
                W + "benchmark_v1/scripts/evaluate_maestro_cpu_component_v1.py",
                "normalized width/1Q/CX counts + shots=1000; evaluate saved C1q/Ccx/b/m width knots" if component else "fold scalar" if constant else "source DAG/six globals; fold-specific learned fits",
                "8 valid saved width knots (q2-9), normalized metadata and 112 synthetic calibration identities" if component else "unique saved fold constants" if constant else "predictions/selected alpha saved but model weights not saved",
                backend="QCSim CPU SV FP64 optimizer off/threads1 target; not full API wall-clock",
                calibration="independent synthetic calibration JSON; no fit to panel targets" if component else "none for scalar; learned scaler state unavailable",
                environment="stdlib scalar knot formula; Qiskit normalization for B2" if component else "stdlib" if constant else "historical sklearn or Torch CUDA",
                status="READY" if component or constant else "BLOCKED",
                blocker="future formula/scalar parity gate; never invoke evaluator's fit entrypoint" if component or constant else "no nested-Ridge coefficients or source-graph checkpoints", **kw)
        else:
            raise ValueError(context)
    row("cutensornet_runtime_est", "native_same_plan_scalar_amplitude", "B", "native architecture-aware analytical diagnostic",
        "work/repo_finalization/publication/data/simulator/tensor_network/measurement/cutensornet_scalar_common_raw.csv",
        "benchmark_v1/scripts/run_common_cutensornet_scalar.py",
        "logical unitary to einsum/operands to Network to TIME_TUNED path; then RUNTIME_EST attribute",
        "QASM/input hashes and telemetry saved; serialized selected historical contraction paths not retained by producer",
        backend="GPU architecture and 14,947,909,632-byte workspace; seed137/samples32/threads1 frozen",
        environment="dedicated Python3.14.4/cuquantum-python-cu13 26.6.0/cupy-cuda13x 14.2.0/cutensornet-cu13 2.14.0; metadata only, imports/ABI not tested",
        status="CONDITIONAL", blocker="isolate native attribute access versus full planning; identical historical plan replay unavailable; any fresh planning requires explicit AM1 approval and no contraction",
        output="selected-plan predicted contraction seconds; not sampling or whole API", init="CUDA/cuQuantum + network/plan setup; potentially dominant")
    row("pasqal_emu_mps_resource_diagnostic", "analog_tdvp_companion", "B", "diagnostic; no promoted predictor",
        "artifacts/benchmark_deep_dive/tables/pasqal_stage_quality_v1.csv",
        "tracks/pasqal_emu_mps/scripts/fit_runtime_formula.py",
        "analog/TDVP stage and quality analysis", "18 diagnostic cells; coefficients not identified/promoted",
        status="NOT APPLICABLE", blocker="no validated estimator inference to time; simulator run cost is not estimator overhead",
        n=18, output="diagnostic simulation/stage costs only")
    assert len(qpu) == 34 and len(sim) == 49 and len(results) == 86
    assert len({(r["track"],r["context_id"],r["estimator_id"]) for r in results}) == 86
    return results


def render():
    pins, checks, constants, facts = inspect_assets()
    entries = inventory()
    counts = {track: {status: sum(r["track"] == track and r["status"] == status for r in entries)
                      for status in STATUSES} for track in ("A", "B")}
    assert counts == {"A": {"READY":26,"CONDITIONAL":0,"BLOCKED":9,"NOT APPLICABLE":0},
                      "B": {"READY":13,"CONDITIONAL":1,"BLOCKED":36,"NOT APPLICABLE":1}}
    files = {}
    def doc(name, text):
        files[name] = text.strip() + "\n"
    table = "| Track | READY | CONDITIONAL | BLOCKED | NOT APPLICABLE | Total |\n|---|---:|---:|---:|---:|---:|\n"
    for track in ("A", "B"):
        table += f"| {track} | " + " | ".join(str(counts[track][s]) for s in STATUSES) + f" | {sum(counts[track].values())} |\n"
    doc("README.md", f"""
# AM1 feasibility: estimator overhead

**CONDITIONAL GO — planning only, 2026-10-11.** Có thể đo một subset của
estimators đã đóng băng mà không train/refit. Không đủ assets để đo toàn bộ
leaderboard. AM1 chưa được chạy; đây không phải authorization hoặc timing harness.

{table}
Đơn vị đếm là **method × execution context/variant**, không phải số papers,
checkpoints hay independent methods: 34 QPU metric rows + original Scholten;
49 simulator metric rows + native cuTensorNet + analog companion. Raw/affine/
log-affine và FP32/FP64 được đếm riêng. READY nghĩa là assets và code có vẻ đủ
qua static inspection, không phải model đã load hay replay/inference đã PASS.
READY của analytical chỉ áp dụng trên eligibility đã lưu, không biến missing
rows thành available. 39 READY rows không tạo thành 39 independent methods.

**Phát hiện quyết định:** QPU Graph/MLP, dense neural và joint-MPS có 180
checkpoint files với hashes khớp receipts (150 runtime heads + 30 family
classifiers). Qonductor six regressors/selector, Hyb log-Ridge, dense classical,
fixed-MPS learned, Azizov và Maestro learned comparators không có fitted states
đủ trong output authoritative. Hyperparameters/predictions không thay cho model.
Không được refit để đo rồi ghép latency mới với accuracy cũ.

AM1 đáng làm để bổ sung **một phần** working RQ4 Accuracy–Cost Trade-off;
không đóng toàn bộ RQ4/scalability/production scheduling. Tên RQ4 trong freeze
cũ là generalization; ở task này dùng working RQ1–RQ5 của closure review.
Không đổi tên hay nghĩa của historical protocol.

## Đọc theo thứ tự

1. [Inventory](estimator_inventory.md) và [CSV](estimator_inventory.csv).
2. [Measurement boundaries](measurement_boundaries.md).
3. [Fair comparison](fair_comparison_protocol.md).
4. [Existing timing](existing_timing_evidence.md).
5. [Accuracy–cost analysis](accuracy_cost_analysis_plan.md).
6. [Implementation plan](implementation_plan.md), [decision](decision_memo.md).
7. [Validation](validation_report.md), [pins](provenance_manifest.json).

Nguồn: {link('artifacts/research_question_closure/v1/README.md', 'closure')};
{link('artifacts/thesis_interpretation/v1/estimator_fidelity_registry.md', 'fidelity')};
{link('artifacts/final_research/manifest.json', 'freeze manifest')}.
Không cập nhật manuscript/presentation; không commit/push.
""")
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader(); writer.writerows(entries)
    files["estimator_inventory.csv"] = out.getvalue()
    doc("estimator_inventory.md", f"""
# Executability inventory

{table}
Exact row-level paths, required fields và eligible accuracy denominators nằm
trong [CSV](estimator_inventory.csv). Fidelity card/class được chép từ saved
method-context index, không nâng adaptation/proxy thành reproduction. Các path
có `{{0..4}}`/seed braces là
**notation**, không phải file literal; expand đúng năm folds/ba seeds rồi kiểm
source pins. Full absolute file/hash list ở [manifest](provenance_manifest.json).

## READY: assets sufficient by static inspection

- QPU: `mali_logical_graph`, `mali_global_mlp`; ba `outer_train_*` references;
  raw/affine/log-affine của bảy bases: `qcre_source_estimator`,
  `qcre_source_estimator_shot_scaled`, `qiskit_estimate_duration_snapshot`,
  `qiskit_shot_scaled_schedule_seconds`, `hyb_nominal_r2_effective_cost_v1`,
  `hyb_nominal_r2_shot_effective_cost_v1`, `scholten_unified_nominal_throughput`.
- Dense FP32 và FP64: `mali_style_dense_graph`, `matched_metadata_mlp`,
  `outer_train_median` trong từng context.
- Fixed MPS chi16: chỉ `train_fold_median`.
- Joint MPS: `family_conditioned`, `family_agnostic_ablation` trong `c44` và
  `family_holdout`; conditioned includes the saved family classifiers.
- QCSim: `maestro_style_cpu_sv_component_optimizer_off_adaptation` và
  `outer_train_median`. Đây là saved independent synthetic calibration knots,
  không fit thêm trên evaluation targets. QCSim không cần chạy để predict.

Parent QPU cần **best_model_state**, không final `model_state`. Dense cần final
500-epoch `model_state`; joint cần saved `model` sau 160 epochs. Không load
optimizer hoặc resume epoch. Native precision của simulator target không xác
định precision của neural estimator (frozen heads vẫn FP32).

## BLOCKED: không có fitted model

Track A: sáu Qonductor families + budgeted selector (7); Hyb log-cost Ridge (1);
original Scholten (1, thiếu kernel/effective-depth eligibility). QPU regressors
chỉ ghi selections/scores/predictions; Hyb Ridge receipts không có coef/scaler.

Track B: 6 dense learned classical routes × 2 precision = 12; fixed-MPS Ridge,
Graph, family residual/agnostic = 4; Azizov 3 GNN + 15 regression-view routes =
18; Maestro nested Ridge/source-DAG Graph = 2. Tổng 36. Authoritative roots đã
kiểm file extensions và producing serialization code. Không tìm thấy model
khác đủ identity để thay vào. Không kết luận rằng upstream method không thể
deploy; đây là **artifact retention limitation của benchmark hiện tại**.

Không đủ coefficients để “tái tạo” fitted Ridge từ alpha; không thể khôi phục
trees/support vectors từ predictions. Không dùng estimator cùng tên ở cohort,
split, engine, chi hay historical run khác. Nếu owner chỉ ra serialization
đúng frozen identities, được audit read-only rồi xin duyệt mở coverage; không
âm thầm refit. Các missing serialization không làm mất giá trị saved accuracy.

## CONDITIONAL và NOT APPLICABLE

cuTensorNet có native pipeline, không learned inference. Read attribute sau
planning khác toàn bộ chi phí tạo estimate. Historical selected paths không
được serialize trong producer; seed/optimizer settings không chứng minh path
identity. Không ghép fresh-plan latency với historical same-plan accuracy như
một paired point. Native extension chỉ sau explicit approval, đúng libraries,
workspace và plan-trace policy; tuyệt đối không execute contraction.

Pasqal có 18 analog diagnostic cells, không promoted predictor hay identified
resource coefficients; không time MPSBackend.run rồi gọi là estimator overhead.

## Additional frozen evidence, không nhân đôi primary count

21 generalization cases/126 Graph–MLP seed cells là case-specific refits của
hai architecture families, không 126 methods. Chúng không được gộp với parent
OOF models hoặc dùng làm parent timing. Source routing/identity authority là
{link('artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/reader_tables/cell_lineage.csv', 'cell lineage')};
{link('artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/manifest.json', 'full-grid manifest')}.
AM1 initial proposal không time các deployment instances này; nếu bổ sung cần
inventory riêng theo case, hash/parity gates và cùng paired support. Historical
PyTorch width/RX launch studies đo simulator work, không là current estimators.

## Integrity scope

Verified: 30 QPU checkpoints + 60 dense + 60 joint runtime heads + 30 joint
family classifiers; 25 transform files; associated saved predictions/receipts.
Source pins preserve measured and adapted method semantics. File hashes chỉ
chứng minh byte identity, không prediction parity; parity là gate của amendment
tương lai. Model internals chưa deserialize. Scalars được đối soát single-valued
theo fold hoặc fold/source từ saved predictions và ghi nguyên giá trị trong
manifest, không tính lại train statistic/fit model ở bước này.
""")
    doc("measurement_boundaries.md", """
# Measurement boundaries

Predicted execution seconds và time spent estimating là hai đại lượng khác.
Mọi timing dùng seconds, raw integer nanoseconds; tuyệt đối không lấy
`predicted_seconds`/`RUNTIME_EST` làm inference time.

| Boundary | Start | Stop | Accounting |
|---|---|---|---|
| B1 prediction-only | prepared features/tensors resident; model loaded, eval mode | required native outputs materialized, inverse target transform and frozen aggregation complete | no parse/graph/fit/setup; report single-seed component and actual three-seed estimator separately |
| B2 online | declared in-memory QASM/recipe/circuit plus already-available target metadata | final runtime/quality result on host | parse, extraction, required transpilation, normalization, H2D, prediction, aggregation/D2H included |
| B3 cold setup | parent launches a fresh isolated child | ready for first request after imports/load/device init/declared warmup | process startup, snapshot/model/calibration artifact I/O, CUDA and warmup; first-request time is a separate interval |

Cold total = setup + first B2 request only when recorded as disjoint intervals
of the same run. Do not sum medians. Host request latency is primary, synchronized
CUDA-event device time supplemental. For B1 resident GPU inputs, report host
return/aggregation honestly; a device-resident timing is a distinct boundary.
No neural CPU fallback, mixed precision, quantization, torch.compile, layer
checkpoint toggles or architecture changes for faster timings.

## Per-class contracts

- Graph and MLP: same frozen logical input qualification, global feature semantics,
  own saved fold transform. Graph additionally builds node/edge tensors. Query
  routing uses its original held-out fold; **not all five OOF models**. Three
  seeds for that fold plus median are estimator cost; seed runtimes not summed
  from independently measured medians.
- Statistical reference: a saved scalar is the complete estimator; no test
  label/source leakage. Source median needs source identity; report that privilege.
- Raw analytical: B1 receives required compiled circuit/metadata, actually calls
  native computation. A cached raw scalar lookup is not the native estimator.
  QCRE includes conversion to BQSKIT in B2; omit terminal measurements only as
  in frozen code. Qiskit includes its distinct measurement/schedule semantics.
  Hyb and Scholten proxy retain own eligibility/output clock; don't replace
  formula, infer original CLOPS, or silently fix missing snapshots.
- Calibrated analytical: B1/B2 includes raw calculation plus frozen scalar
  affine/log-affine transform with saved fold coefficients and original clamp.
  Timing only slope multiplication is labeled **bridge-only**, not whole method.
- Maestro: independent saved C1q/Ccx/b/m knots, normalization and shot count.
  No synthetic circuits are regenerated, measured or calibrated. All panel
  widths are stored knots; no new interpolation/extrapolation rule.
- Joint MPS: conditioned version includes predicted-family classifier cost,
  never true family labels as a free input. Both runtime and quality head
  outputs and frozen aggregation are included. Multi-rung query batch distinct
  from a single (hash, chi) request. Reuse classifier once per circuit only if
  cache key and amortization are reported; compare uncached route too.
- cuTensorNet: network build, path optimizer plus internal cost calculation,
  and attribute access are three boundaries. RUNTIME_EST availability generally
  requires a plan. Attribute-read latency alone does not isolate internal model
  computation; no public-stage attribution is invented. No contraction in AM1.

## Offline, shared, repeated

Training T_train, synthetic calibration T_cal, one-time setup S, shared-preparation
C_shared and per-request C_i are reported separately. For explicit N requests,
amortized online cost = (S + C_shared + sum C_i)/N. Optional lifecycle amortization
adds measured T_train/T_cal separately; absent offline timers are missing, not 0.
Historical training-loop timers cannot supply pure training compute.

Standalone B2 charges each method's own full required path. For a combined
multi-estimator service, represent preprocessing as dependency stages and charge
each shared stage exactly once; report shared total and marginal costs, not
arbitrary per-method attribution. Cache hits/misses carry keys including circuit,
fold/transform, snapshot/context and representation hashes. No cross-fold
transform reuse. Local artifact I/O is B3 or an explicitly labeled disk-request
boundary; B2 primary starts with an in-memory circuit. Network/QPU submission is
out of scope.
""")
    doc("fair_comparison_protocol.md", """
# Proposed fair-comparison protocol — approval required

## Populations and deterministic sampling

Track A retains the 4,515 qualified observations/165 conservative groups and
frozen outer-test assignment. Track B retains each context's exact hash/rung
population and target/quality eligibility. Timing selection never uses target
runtime, residual, winning model, best seed or availability-conditioned accuracy.

Proposed screening panel: at most **50 QPU observations (10 per frozen outer
fold)** and **30 circuit hashes per executable simulator context**. Freeze a
selection manifest before timing; use the same label-free selection within
each QPU fold, then union without exceeding the per-fold cap.
Using frozen metadata only, form strata by source/context, family, backend,
width bins (q2–4/5–7/8–9/10–16/>16 where present), within-context depth/gate/graph
node tertiles. These are not full Cartesian quotas. Deterministic greedy
coverage selects the SHA256-smallest candidate for each uncovered nonempty
stratum, cycling axes in this order; deduplicate IDs. If mandatory coverage
exceeds cap, require approval or list uncovered strata, never claim full
representativeness. Fill to cap by SHA256 of `AM1-ESTIMATOR-OVERHEAD|track|context|ID`.
Exact repeated model-ready inputs retain observation IDs; record both unique
structural/feature-input timing and observation-weighted service interpretation.

Previously identified QNN/Random or large/collision groups can have a separately
declared diagnostic panel selected by metadata/family, not extreme error.
Do not merge these enriched strata into unbiased population estimates. No
source-held-out retraining or new model generation. Joint MPS retains all six
assigned chi rungs for a selected hash and original quality policy. Do not
evaluate quality by new state simulation or invent labels for missing rungs.

All methods attempt the same selected requests where contracts permit; keep
unavailable/failure rows in denominator. Report full eligibility and exact
shared-success intersections, with input requirements disclosed. No single
cross-context Track B ranking. Preserve raw versus calibrated output semantics.
Classical models missing fitted states remain blocked, not presumed cheap.

## Hardware/software record and isolation

Observed inventory on 2026-10-11 (read-only, not a reservation): Intel i7-14700F,
20 cores/28 logical CPUs; RAM 49,239,826,432 bytes (~45.86 GiB); RTX 5070 Ti,
16,303 MiB VRAM, driver 595.91.07; Linux 7.0.0-38-generic x86_64/glibc2.43.
At inspection ~40.1 GiB RAM available, ~49.05 GiB disk available, GPU ~996 MiB
used/4% utilization. These volatile readings do not prove an idle/exclusive host.

CUDA environment exists at
`/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python`:
Torch 2.7.1+cu128, PyG 2.6.1, NumPy 1.26.4, sklearn 1.3.0, Qiskit 0.44.2.
CPU metadata environment `/home/server/Documents/.venv-qonductor/bin/python`:
Qiskit 2.5.2, runtime 0.49.0, NumPy 2.2.6, sklearn 1.7.2, Torch 2.14.0+cpu.
Use the first for neural, never the CPU Torch installation. QCRE's pinned
BQSKIT 1.2.0/adapter lives in `.venv-cdaa-new/lib/python3.10/site-packages`,
appended as in the producing route. B2 may require a clearly timed CPU-feature
worker to CUDA-worker IPC: include serialization/transfer, never conceal it.
No single merged environment is assumed. Metadata availability is not an import
or ABI test; future smoke/parity gate must establish compatibility without refit.

cuTensorNet's current dedicated environment is Python3.14.4/Qiskit2.5.2/
NumPy2.5.3, cuquantum-python-cu13 26.6.0, cupy-cuda13x 14.2.0,
cutensornet-cu13 2.14.0 and cutensor-cu13 2.8.1. Metadata proves installed
distributions, not import/ABI viability or historical environment parity.
The missing historical selected-plan state remains a separate blocker.
No dependency installation/update in this task.

At approved execution: record OS/kernel, CPU affinity/P–E core class, SMT,
CPU governor/frequency, thread settings, host/GPU load, GPU UUID, clocks/power,
driver/runtime, package versions and checkpoint placement/precision. Primary
CPU native/BLAS threads=1; neural Torch CPU threads=2, CUDA exclusive one worker.
Pin to a documented physical core class, not arbitrary CPU0 on this hybrid CPU.
No heavy CPU/GPU workload in parallel; no changing governor/clocks silently.
One measurement worker, one independent resource observer. Report residual
background load. Compare complete deployable CPU/GPU routes without claiming
the difference is caused only by architecture.

## Measurements (not executed)

- For each warmed fold/model bundle: 10 unscored warmups, then 30 repetitions
  of each selected request. Fresh identical input objects per uncached B2
  request; existing valid cache for explicitly cached B1. Eval/no_grad and
  frozen inverse transform/aggregation; no compile/autotuning added.
- Three isolated fresh processes per bundle for B3; OS page cache remains
  uncontrolled/warm and is disclosed. No sudo/drop_caches. Capture launch,
  import, load, CUDA init, warmup and first request separately.
- Primary batch=1; optional exact batch32 if 32 distinct eligible requests
  exist within the same context/fold. Never mix folds to fill a batch. Joint
  MPS six-rung request is its own batch. If <32, omit, don't pad silently.
- Paired rotating method order fixed by deterministic hash order and repetition
  index, no best-latency run. Monotonic perf_counter_ns; synchronize CUDA before
  start and after completion including host result, then stop. GPU events are
  supplemental device-only timers, not host pipeline latency.
- Record median, p90, all raw samples and failure counts. p99 with only30
  repeats is descriptive/weak, not an SLA. Across-request quantiles are distinct
  from within-request jitter. Structural-group bootstrap paired median/MAE
  differences; repeats do not increase independent workload support. <10 groups:
  descriptive uncertainty only, following existing support rule.
- Empty timer pair, no-op dispatch and sync-only controls measured identically.
  Retain unadjusted primary latency, report control distribution; do not subtract
  noisy medians to manufacture negative/near-zero cost. If timer/control floor
  dominates, report unresolved; any loop amplification is separately approved
  and not batch latency. No selective outlier removal.
- Isolated-process RSS high-water + sampled child memory; Torch allocated/
  reserved peaks and GPU process memory separately. Observer interval/baseline
  recorded; sampled peaks are lower bounds. Profiled-memory run separated from
  primary low-overhead latency if instrumentation perturbs timings.
- Timeout/OOM/nonfinite output/parity failure retained. Environment interruption
  receives reason, not deletion or automatic retry. Require approval before
  changing repetition count, workload, device, budget or method implementation.
""")
    phases = facts["tn_phases"]
    timing = "| Saved phase | Independent plan sessions | Median s | Range s | Meaning |\n|---|---:|---:|---|---|\n"
    for key, meaning in (("t_network_build_s","host converter + operands + Network"),
                         ("t_path_search_s","optimizer/path including hidden costs; not isolated EST"),
                         ("t_first_contract_s","CUDA-event contraction, not estimator latency"),
                         ("t_warm_contract_s","median of five warm samples per session, then across sessions")):
        v=phases[key]
        timing += f"| {key} | {v['n']} | {v['median_seconds']:.15g} | {v['min_seconds']:.15g}–{v['max_seconds']:.15g} | {meaning} |\n"
    cost_table = "| Method | Saved cells | Min / median / max s | Sum cell s |\n|---|---:|---|---:|\n"
    for method,v in facts["training_receipts"].items():
        cost_table += f"| {method} | {v['cells']} | {v['min_seconds']:.6f} / {v['median_seconds']:.6f} / {v['max_seconds']:.6f} | {v['sum_cell_seconds']:.6f} |\n"
    doc("existing_timing_evidence.md", f"""
# Reusable saved timing evidence

No new latency measurements. Values recalculated deterministically from saved
receipts/telemetry by the read-only closure calculator, with source hashes pinned.

## Thirty QPU neural training-loop receipts

{cost_table}
Producer: {link('benchmark_v1/scripts/run_qualified_logical_benchmark.py', 'neural_cell')}.
Timer starts after initial model/optimizer/cache setup, includes training,
validation/cache access and earlier checkpoint writes, then is frozen **before**
final checkpoint write and outer-test inference. It excludes some initialization
and is not pure optimizer kernel time. Sum is sum of cell timers, not wall time
of an entire campaign. Different best epochs do not change the saved500-epoch
loop limit. Descriptive historical cost only; cannot use it as B1/B2/B3 or
causal graph-versus-MLP training-efficiency evidence.

## cuTensorNet component telemetry

Source: {link(closure().SOURCES['tn_telemetry'], 'raw telemetry')};
producer {link('benchmark_v1/scripts/run_common_cutensornet_scalar.py', 'native runner')}.
3,672 rows = 612 first + 3,060 warm; 603 successful sessions × (1 first + 5 warm)
= 3,618 successful rows; nine failed/timeout sessions yield 54 timeout rows.
Successful sessions cover 201 panel members and 188 unique hashes, each with
three session IDs. Session key is `(panel_member_id,session_id)`, not session_id
alone. Session observations of the same circuit are not independent circuits.

{timing}
Build ~294.770 ms; path ~12.511 ms; warm contraction ~0.137600 ms.
`t_end_to_end_s` has **0** nonempty rows. Do not sum these medians into E2E,
derive API latency from CUDA-event contraction, or call `runtime_est_s` a timer.
Native code reads RUNTIME_EST after contract_path without a separate timer;
internal cost-model work may be embedded in optimizer stages. Attribute-get
cost and full planning cost are not independently measured here.

## Other receipts and classifications

Dense classical `fit_wall_seconds` surrounds the multi-method fit/predict loop,
attempt assembly and prediction CSV write. It is not per-estimator fit or
inference latency. Reuse only as labeled historical bundle elapsed, not AM1.
Simulator first/warm execution tables and analog stage times measure simulated
work, not prediction overhead. Generalization refit receipts are execution
provenance and must not be pooled into matched parent inference timings.

Directly measured: the stage/loop intervals above. Derived: sums of loop
intervals and per-session-then-across-session medians (hierarchy specified).
Inferred: likely initialization/planning dominance is a hypothesis, not a new
measurement. Missing: B1 inference, full B2 online pipeline, B3 cold setup,
matched peak memory, RUNTIME_EST-internal computation latency and amortization
under a real request stream. No existing evidence closes those AM1 questions.
""")
    doc("accuracy_cost_analysis_plan.md", """
# Accuracy–cost analysis plan

Primary accuracy remains the existing frozen OOF/quality-conditioned results;
do not recalculate predictions, improve hyperparameters or choose a seed.
Track A uses the 4,515-row qualified cohort and route-specific eligibility.
Track B uses each exact engine/task/precision/shots/quality boundary. Do not
rank QPU service predictors against simulator predictors, or contraction EST
against sample/get_state/API timings.

For every future table, retain context/method/fold/model-state/seed-aggregation,
accuracy population N and groups, timing population N and groups, coverage,
B1/B2/B3 scope, cache state, device and units. Full-cohort accuracy next to
subset timing is allowed only with separate denominators and explicit warning:
it is **not** a per-workload paired cost frontier. Workload-aligned plots join
exact ID/hash/rung and outer-fold to frozen predictions; unmatched rows excluded
from paired effects but retained in coverage. Source-ID privileged references
remain distinct. QPack dominates pooled QPU rows; show pooled, equal-source and
equal-structural-group views without treating three sources as three large iid
samples or case assignments as unique rows.

Planned panels, only after execution approval:

1. Frozen MAE/MedAE/RMSE/R² and paired uncertainty versus B1, B2 and B3 separately.
   Negative R² retained. Never compare method-specific available-row MAEs as
   if they had the same test population; show shared-row scores from existing
   predictions alongside the 4,515-row attempt envelope.
2. Accuracy versus RSS/GPU peak, with host/device memory separate. A Pareto
   statement only within one context, target, eligibility, population,
   aggregation and measurement boundary. No strict dominance when paired
   uncertainty overlaps; fewer executable models limits frontier claims.
3. B2 stage stacks/latency versus width, depth, gates, nodes/edges, source/family
   and unique feature count. Size effects are descriptive associations; do not
   fit a universal scaling law from q2–9 or confounded source/width bins.
4. Cache hit/miss, batch1 versus batch32 and per-query six-rung amortization.
   Each uses measured bundle total, not sum of medians. Missing/blocked models
   never receive zero or guessed cost. Timeout/OOM rates are part of feasibility.
5. Optional break-even: for measured setup S and mean request c under specified
   request mix, T(N)=S+N*c. For two methods with distinct setups/c, solve the
   crossing N only when denominator/signs admit a positive crossing; report
   uncertainty and stable-mix assumption. This is computational cost crossing,
   not scheduling benefit or “latency repaid by MAE”. Do not add unmeasured
   training/calibration costs as if zero.

Accuracy gain can justify extra cost **conditional on** an externally specified
request budget and error tolerance; report both axes and leave utility to the
user. No universal MAE-seconds/latency-seconds exchange rate. Fractional MAE gain
is not fractional scheduler throughput gain. If Graph accuracy increment is
unresolved while extra cost is clear, say so; lower estimator latency does not
establish deployment safety, reliable queues or improved quantum utilization.

Important scope loss: absent QPU classical states and dense RF/XGB weights mean
AM1 cannot resolve graph-versus-strong-tree cost on the original winning fits.
Do not substitute an arbitrary cheap regression or claim neural Pareto advantage
because a competitor's artifact was not retained. Report this limitation beside
any selected-subset frontier, not in an obscure appendix.
""")
    doc("implementation_plan.md", """
# AM1 implementation plan — not a harness

Amendment identifier: **AM1-ESTIMATOR-OVERHEAD**. Proposed future output:
`artifacts/amendments/am1_estimator_overhead/run_001/`. This directory is NOT
initialized by this task. No timing code or adapters have been implemented here.

## Approval and phases

0. Owner approves narrowed method list, B1/B2/B3, workload cap, precision/device,
   budgets, and whether native cuTensorNet planning is included. BLOCKED models
   stay blocked; locate existing exact states only. No training/refitting.
1. Implement a separate inference-only adapter layer, never call runner `main`,
   `--stage all`, evaluators that fit Ridge, or resume methods. Reuse pure
   feature/constructor/formula functions after static side-effect review.
   Parent: `run_qualified_logical_benchmark.make_model`,
   `run_mali_full_features` global/node transform application and pinned model
   constructor, `run_mali_batched.ExampleCache` feature materialization. Separate
   constructor from all fit/cache-to-disk effects; no frozen file changes.
   Dense: `_make_model`, saved `fold_inputs.json`, `_predict` semantics from
   `run_cudaq_dense_neural_predictors_v1.py`. Joint:
   `train_family_aware_joint_mps_v1.static_features` and saved head/classifier
   architecture; do not invoke `prepare_fold` if it fits a classifier.
   Analytical: pinned QCRE `estimate_runtime` + BQSKIT converter; native duration,
   Hyb/proxy functions in analytical producers, saved bridge coefficients.
   Maestro: `load_component_model`/`component_predictions` in pinned evaluator,
   but not nested_ridge_fold/whole evaluate routine. Native TN optional:
   strict_unitary/converter/Network/contract_path/optimizer_value only, never
   child/parent runner (they execute contraction). No historical source edit.
2. Materialize **new** query/adapter manifests from existing inputs only. Preserve
   row IDs, source/physical/recipe/feature/snapshot digests and frozen transforms.
   Hash inputs/outputs before/after. Formal read-only source checks precede
   any approved inference. Fail if source pin or eligible output semantics differs.
3. After separate execution approval, load trusted exact checkpoint states
   under compatible environment, eval/no_grad. Run parity on the selected frozen
   held-out rows before latency: parent best state, dense final state, joint
   runtime+quality+family classification. Expected predictions from pinned
   saved files, not a newly fitted model. For frozen FP32 neural output use
   atol=1e-6 seconds + rtol=1e-5; quality probability atol=1e-6/rtol=1e-5 and
   exact quality decisions. Native scalar/analytical atol=1e-12 seconds +
   rtol=1e-10. These are proposed numerical replay gates, not a license to alter
   model thresholds; any failure requires review, no tolerance widening. Report
   small discrepancies, never overwrite saved predictions. Verify recovered
   reference scalar equals all saved fold/source reference predictions exactly.
4. Run serial isolated B1/B2/B3 using the [protocol](fair_comparison_protocol.md).
   Warm repetitions=30, warmups=10, cold restarts=3, batch1 primary. No automated
   retry, budget expansion or stage fallback. Keep raw timing and system records.
5. Summarize after per-adapter gates. Join frozen accuracy only by identity and
   compatible context, report blocked/failure coverage. Recompute tables from
   raw timing, cross-check stage composition, denominator and uncertainty.
   New timings/calculations never become replacements for frozen predictions.

## Raw and derived schema

Future raw JSONL/CSV fields: amendment/run ID; request ID/canonical observation
ID/QASM hash/rung; track/context/representation/snapshot/feature digest;
structural_group/source/family/width/depth/gates/nodes/edges; fold; method;
model/checkpoint/transform hash; seed list and aggregation; boundary/stage;
cache state; cold process/repetition ID; batch size/member IDs; start_ns,
end_ns, elapsed_ns; host/device timer type; synchronization; setup included;
status/reason; prediction parity max difference; process/child RSS high-water;
Torch allocated/reserved GPU bytes; sampled process GPU bytes; thread/affinity;
software/environment pin; instrumentation-control ID; load/temperature/clocks.
Labels/predictions referenced separately, never written over original artifacts.

Summaries: latency seconds=(end_ns-start_ns)/1e9; per-request median/p90 over
retained valid repetitions; report censoring/failure/timeouts separately. Group
uncertainty resamples independent structural groups, not timing repetitions;
within-query jitter is separate. Batch throughput=batch_n/measured_total_s and
amortized request time=measured_total_s/batch_n, not reciprocal median if
aggregation differs. Peak memory=max observed minus recorded baseline only
when same metric/process lifecycle supports it; also retain absolute peak.
Formula details and comparability rules in the two companion protocol docs.

## Proposed resource budget (unmeasured planning bounds)

- One measurement worker, no parallel training/simulator job. CPU native threads1,
  Torch CPU threads2; CUDA required for neural. Budget host RAM8 GiB, neural
  GPU allocation8 GiB, extra disk<=256 MiB; disk floor3 GiB plus pending write
  reserve. Samples stay small; no checkpoints copied unnecessarily.
- Two-hour hard wall ceiling **for a pilot**, not an ETA or promise of all
  repeats/contexts. At the 50/30 caps and 30 reps, query cost is unknown until
  approved pilot. Request timeout30s, B3 startup120s; stop and request review
  rather than automatically extend. Report any unfinished cell as budget-stop,
  not method failure. Optional fresh TN planning is excluded from this initial
  budget: frozen workspace14,947,909,632 bytes exceeds the8GiB neural budget;
  a separate native-plan budget and authorization are needed, not a silently
  reduced workspace/new estimator. No contraction is ever part of AM1 overhead.
- Resource caps are maximums, not measured required memory; if largest eligible
  graph cannot fit, report censored feasibility, no batch/model change. Use no
  destructive cleanup or data regeneration. Warm asset caching and cold
  filesystem state disclosed; avoid flushing system cache.

Stop on changed scientific input/hash, unknown checkpoint identity, prediction
parity failure, hidden fitting/calibration/simulation, incompatible output clock,
resource floor/OOM, unexpected concurrent GPU owner, or unsupported state-load.
Retain logs/partial results in amendment namespace; do not repair protocols,
drop difficult rows or install a different framework to force success.
""")
    doc("decision_memo.md", """
# Decision: CONDITIONAL GO

AM1 has clear scientific value, but only for the **executable frozen subset**.
The important missing evidence is cost of producing an estimate, not another
accuracy model. Hardware and retained states make controlled Graph–MLP,
analytical/bridged/reference and some simulator component/joint comparisons
plausible. Execution must still be approved and pass model-load/prediction-parity
gates; static READY is not an observed success or production endorsement.

The classical retention gap is material: original winning QPU selector and
dense RF/XGB do not retain fitted states. Full across-class accuracy–cost
frontier is therefore **NO-GO under current artifacts/no-refit restriction**.
A narrowed comparison is worth doing if published as such. No missing model
may be replaced by retraining or a surrogate based on saved predictions.

| Question | Feasible contribution | Remaining limitation |
|---|---|---|
| AM1-Q1 preprocessing | parse/feature/DAG/transpile stage costs on fixed subset | recipe qualification, pipeline/environment asymmetry, small width support |
| AM1-Q2 cold/warm | B1/B2/B3 for retained models/coefficient routes | OS page cache not truly cold, no latency claim yet |
| AM1-Q3 scaling | observed complexity-cost associations within sample | no large-width or cross-hardware law |
| AM1-Q4 accuracy-cost | paired conditional comparisons retaining old accuracy | strong classical contenders missing, no universal utility/scheduler gain |
| AM1-Q5 deployability | execution feasibility, memory/budget/failure reports | not production validation, prospective reliability or scheduler experiment |

AM1 would **partially strengthen working RQ4**, not fully close accuracy–cost
and scalability for all methods/contexts. Even successful execution does not
validate simulation runtime extrapolation, universal backend generalization,
exact historical logical recovery or scheduling throughput. RQ1/3/5 frozen
findings do not change; full21 generalization grid remains completed with its
original limits and historical authorization disclosure.

Recommended minimum: owner approve a two-hour bounded AM1 pilot for parent
QPU Graph/MLP, saved references and eligible native/bridged analytical routes;
then approved dense/joint-MPS and Maestro subsets on the same host if the pilot
parity/resource gates pass within the approved scope. Keep all missing states
explicitly BLOCKED. Exclude native fresh TN planning initially; reuse saved
build/path telemetry descriptively. Do not ask for refitting just to produce a
more complete cost chart.

**RECOMMENDED NEXT ACTION: Approve or reject this narrowed AM1 protocol and
resource cap explicitly; keep Scientific Freeze v2 and do not start AM1 now.**
""")
    doc("validation_report.md", f"""
# Static validation report

Status: **PASS — documentation/asset feasibility only**. This is not a timing,
inference, scientific result or public-release PASS.

- All ten requested artifacts present; one provenance manifest; one stdlib
  documentation builder/validator, not an AM1 timing harness.
- Inventory: 86 distinct track/context/method rows, 23 CSV columns including
  fidelity card/class; exact 34 QPU and 49 simulator result rows covered, plus original
  Scholten/nativeTN/analog companion. Status totals reproduce README/table.
- {checks['qpu_checkpoint_files']} parent QPU + {checks['dense_checkpoint_files']}
  dense + {checks['joint_runtime_checkpoint_files']} joint runtime +
  {checks['joint_classifier_checkpoint_files']} family classifier checkpoint
  SHA256s match producing receipts; no state deserialization or model imports.
- 25 saved transform files pinned; seven complete scalar references verified
  single-valued per fold/fold-source and preserved in provenance. Maestro 8 width
  knots contain saved finite coefficients. Five unsaved-model roots audited;
  dense classical states also absent per producing code/output inventory;
  Hyb Ridge receipts contain selection info but no coefficients/scalers.
- Saved closure validation rerun; 30 loop timer boundaries and 603-session TN
  telemetry hierarchy reproduced, E2E nonempty0. No source/history rewrite.
- Every direct source/asset digest is recorded in manifest. Default validator
  checks source digests, exact payload bytes, inventory coverage/uniqueness,
  local Markdown links and status counts. Experimental-directory absence and
  planned-vs-measured wording are creation-time scope checks. Source digests rerun
  after report creation; assertion failures stop validation.
- Environment versions were read via distribution metadata, never Torch CUDA
  import or availability calls. GPU/host inventory was read-only nvidia-smi,
  lscpu/free/df; volatile inventory is not an estimator measurement.

Revalidate (reads only; no model/checkpoint loading):

```bash
PYTHONDONTWRITEBYTECODE=1 /home/server/Documents/.venv-qonductor/bin/python -B benchmark_v1/scripts/build_am1_feasibility.py
```

Pins: [provenance_manifest.json](provenance_manifest.json). The manifest excludes
itself from payload hashes; no self-hash cycle. It pins this static builder.
No training/refit/inference/QPU/simulator/QCEC/dependency install/latency timing,
frozen data/protocol/prediction/checkpoint changes, commit/push or presentation
updates. Scientific Freeze v2 preserved. READY remains a static judgment pending
future separately authorized execution/parity tests.
""")
    pins[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    manifest = dict(scope="STATIC_FEASIBILITY_AND_PROTOCOL_ONLY", verdict="CONDITIONAL GO",
        scientific_freeze="Final Scientific Freeze v2", amendment_id="AM1-ESTIMATOR-OVERHEAD",
        execution_authorized=False, new_model_inference=False, model_deserialization=False,
        new_training_refit=False, new_workloads=False, new_estimator_timing=False,
        source_pins=dict(sorted(pins.items())), asset_checks=checks,
        saved_constant_parameters=constants, status_counts=counts,
        payload={name: hashlib.sha256(text.encode()).hexdigest() for name,text in sorted(files.items())},
        self_hash_policy="manifest not included in its own payload; source builder pinned",
        planned_experiment_output="artifacts/amendments/am1_estimator_overhead/run_001/",
        original_freeze_results_modified=False)
    files["provenance_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)+"\n"
    return files, manifest


def validate():
    files, manifest = render()
    for name, text in files.items():
        if (OUT / name).read_text() != text:
            raise ValueError(f"payload_drift:{name}")
    for path, expected in manifest["source_pins"].items():
        if sha(path) != expected:
            raise ValueError(f"source_drift:{path}")
    inventory_rows = list(csv.DictReader(io.StringIO(files["estimator_inventory.csv"])))
    assert len(inventory_rows) == 86
    assert all(set(r) == set(FIELDS) and all(v != "" for k,v in r.items() if k != "eligible_accuracy_rows") for r in inventory_rows)
    for name,text in files.items():
        if name.endswith(".md"):
            for target in re.findall(r"\]\(([^)]+)\)", text):
                path = Path(target)
                path = path if path.is_absolute() else OUT / path
                if not path.exists():
                    raise ValueError(f"broken_local_link:{name}:{target}")
    return dict(status="PASS", scope=manifest["scope"], requested_documents=10,
                inventory_rows=86, status_counts=manifest["status_counts"],
                checkpoint_files_verified=180, transform_files_verified=25,
                source_pins_checked=len(manifest["source_pins"]), model_loads=0,
                new_inference_or_latency_measurements=0, source_files_unchanged=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-patch", action="store_true")
    args = parser.parse_args()
    if args.emit_patch:
        files, _ = render()
        if OUT.exists():
            raise SystemExit("refusing new-file patch: AM1 output directory already exists")
        print("*** Begin Patch")
        for name,text in files.items():
            print("*** Add File: " + str(OUT / name))
            for line in text.splitlines():
                print("+" + line)
        print("*** End Patch")
    else:
        print(json.dumps(validate(), ensure_ascii=False, sort_keys=True))
