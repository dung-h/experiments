"""Render Vietnamese thesis reporting from frozen evidence; never fit or infer.

--emit-patch prints additions for apply_patch. Default checks saved reporting.
No file-writing API, model import, prediction call or experiment dispatcher.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/thesis_interpretation/v1"
ADD = "artifacts/benchmark_interpretation/v1/"
DEEP = "artifacts/benchmark_deep_dive/tables/"
GRID = "artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/"
PARENT = "artifacts/real_qpu/qualified_logical_panel/"
GRAPH, MLP, SELECTOR = "mali_logical_graph", "mali_global_mlp", "qonductor_budgeted_selector"
STATUS = {"SUPPORTED", "SUPPORTED WITH QUALIFICATION", "INCONCLUSIVE", "NOT EVALUATED"}
CATEGORIES = {"Original/native estimator", "Faithful reproduction", "Architecture/core adaptation",
              "Generalized proxy", "Calibrated adaptation", "Benchmark-defined diagnostic", "UNVERIFIED / not evaluated"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Sources:
    def __init__(self):
        self.pins = {}

    def read(self, path):
        p = ROOT / path
        self.pins[str(p)] = sha(p)
        return p.read_text()

    def rows(self, path):
        return list(csv.DictReader(io.StringIO(self.read(path))))

    def js(self, path):
        return json.loads(self.read(path))


def unique(rows, **keys):
    found = [r for r in rows if all(r[k] == str(v) for k, v in keys.items())]
    assert len(found) == 1, (keys, len(found))
    return found[0]


def num(value):
    return float(value)


def close(a, b):
    assert math.isclose(num(a), num(b), rel_tol=1e-10, abs_tol=1e-11), (a, b)


def csv_text(rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def md_table(rows, columns):
    def cell(v):
        # CSV preserves original precision; reader Markdown uses bounded precision.
        try:
            if str(v).strip() and math.isfinite(float(v)):
                x = float(v)
                return str(int(x)) if x.is_integer() else format(x, ".6g")
        except (ValueError, TypeError):
            pass
        return str(v).replace("|", "\\|").replace("\n", " ")
    head = "| " + " | ".join(label for _, label in columns) + " |\n"
    head += "| " + " | ".join("---" for _ in columns) + " |\n"
    return head + "".join("| " + " | ".join(cell(r.get(k, "")) for k, _ in columns) + " |\n" for r in rows)


def links(paths):
    return "; ".join(f"[{Path(p).name}]({ROOT / p})" for p in paths.split(";") if p)


def render():
    s = Sources()
    authority = s.js(ADD + "manifest.json")
    assert sha(ROOT / (ADD + "manifest.json")) == "68f7aa406417531fe5856634867498d2dd9521adf395d3ed2856d63a12850fa1"
    # Consume the existing package, not regenerate its 16 evidence tables.
    for entry in authority["payload"]:
        assert sha(ROOT / entry["path"]) == entry["sha256"]
        s.read(entry["path"])
    for p in ["docs/research_contract.md", "docs/research_conclusions.md",
              "artifacts/final_research/completion_report.md", "artifacts/final_research/rq4_report.md",
              "artifacts/final_research/slides.md", "artifacts/final_research/manifest.json"]:
        s.read(p)
    old_validation = s.js(ADD + "validation.json")
    assert old_validation["status"] == "PASS" and old_validation["derived_tables_replayed"] == 16
    question_matrix = s.rows(ADD + "benchmark_question_matrix.csv")
    assert len(question_matrix) == 120
    for r in question_matrix:
        for p in r["evidence_paths"].split(";"):
            s.read(p)
    qpu = s.rows(DEEP + "qpu_metrics_v1.csv")
    parent_metadata = s.rows(PARENT + "summary/method_comparison.csv")
    sim = s.rows(DEEP + "simulator_metrics_v1.csv")
    fidelity = s.rows(DEEP + "method_fidelity_v1.csv")
    pair_refs = s.rows(DEEP + "qpu_shared_reference_comparisons_v1.csv")
    parent_pairs = s.rows(PARENT + "summary/shared_row_comparisons.csv")
    sim_pairs = s.rows(DEEP + "simulator_paired_v1.csv")
    analytical = s.rows(DEEP + "analytical_controls_v1.csv")
    decomp = s.rows(ADD + "generalization_decomposition.csv")
    skills = s.rows(ADD + "generalization_predictive_skill.csv")
    macro = s.rows(ADD + "qpu_source_balanced_skill.csv")
    ledger = s.rows(ADD + "global_feature_ledger.csv")
    widths = s.rows(ADD + "effective_input_widths.csv")
    qcec = s.js("artifacts/real_qpu/provenance_equivalence/results.json")
    safety = s.rows(DEEP + "mps_selection_safety_v1.csv")
    tn = s.rows(DEEP + "tn_same_plan_slices_v1.csv")
    tn_core = unique(tn, slice_type="stratum", slice="core_q2_q9")
    tn_frontier = unique(tn, slice_type="stratum", slice="frontier_q10_q16")
    topology = s.js("artifacts/validation/mali_physical_dag_v1/topology_controls_validation/summary.json")
    s.read("experiments/evaluate_mali_topology_controls.py")
    for p in ["artifacts/mali/graph_intervention_audit_20260921/REPORT.md",
              "artifacts/validation/mali_physical_dag_v1/topology_controls_validation/REPORT.md",
              "artifacts/validation/mali_physical_dag_v1/feature_ablation/REPORT.md",
              "artifacts/validation/mali_physical_dag_v1/critical_subgraph_evaluation/REPORT.md",
              "artifacts/validation/mali_strict_transfer_logical_v1/evaluation/REPORT.md"]:
        s.read(p)
    full_pairs = s.rows(GRID + "reader_tables/paired_comparisons.csv")
    coverage = s.rows(GRID + "reader_tables/case_coverage.csv")
    s.read(GRID + "reader_tables/seed_metrics.csv")
    s.read(GRID + "reader_tables/cell_lineage.csv")
    assert len(qpu) == 34 and len(sim) == 49 and len(decomp) == 42 and len(widths) == 257

    g, m, c = [unique(qpu, method_id=x) for x in [GRAPH, MLP, SELECTOR]]
    median = unique(qpu, method_id="outer_train_median")
    skill_global = 1 - num(g["mae_seconds"]) / num(median["mae_seconds"])
    skill_selector = 1 - num(g["mae_seconds"]) / num(c["mae_seconds"])
    saved_global = unique(s.rows(ADD + "qpu_predictive_skill.csv"), scope="qpu_pooled", method_id=GRAPH, reference_id="outer_train_median")
    saved_selector = unique(s.rows(ADD + "qpu_predictive_skill.csv"), scope="qpu_pooled", method_id=GRAPH, reference_id="inner_selected_classical_regression")
    close(skill_global, saved_global["skill"])
    close(skill_selector, saved_selector["skill"])
    assert round(100 * skill_global, 1) == 26.7 and round(100 * skill_selector, 1) == 5.9

    # Cards classify the local implementation, not the original publication.
    cards = []

    def card(key, label, source_method, category, unit, native_target, target, inputs, mechanism, differences, allowed, forbidden, evidence):
        original = unique(fidelity, method=source_method) if source_method else {}
        cards.append(dict(card_id=key, reader_label=label, category=category,
            original_source=original.get("primary_source", "Frozen internal protocol"),
            original_task=original.get("paper_objective", "Local comparator"), original_observation_unit=unit,
            original_inputs=original.get("original_inputs", inputs),
            original_mechanism=original.get("original_formulation", mechanism), original_target=native_target,
            benchmark_target=target, implemented_inputs=inputs, implemented_component=mechanism,
            adaptation=differences, allowed_claim=allowed, prohibited_claim=forbidden,
            evidence_paths=evidence, classification_scope="implementation/component; not the entire paper"))

    learned_target = "archived observed service/execution seconds; source boundaries retained"
    fa = ADD + "mali_feature_schema_audit.md"
    qa = DEEP + "qpu_metrics_v1.csv"
    sm = DEEP + "simulator_metrics_v1.csv"
    card("mali_graph", "Ma–Li-style Graph + globals", "Ma–Li QPU", "Architecture/core adaptation", "circuit execution", "observed runtime", learned_target,
         "DAG, raw178 node/calibration; raw51 globals, parent40; no shots/backend ID", "TransformerConv + global MLP, three-seed median",
         "Fit-only per-feature moments, nominal wire-index T1/T2, qualified recipes, grouped splits; no active acquisition; parent retains inner1–3 fit", "So sánh pipeline trên cùng test IDs", "Exact reproduction; topology-only gain; recovered job-day placement", fa + ";" + qa)
    card("mali_mlp", "Matched global-feature MLP", "Ma–Li QPU", "Architecture/core adaptation", "circuit execution", "observed runtime", learned_target,
         "raw51 / parent40; no graph/calibration/shots/backend", "Global-only network, matched neural splits/seeds", "No author processed real tensor; input support varies by fit", "Global-only pipeline contrast", "Identical-information topology ablation; hardware adaptation from collisions", fa + ";" + qa)
    card("qonductor", "Qonductor-style six regressors + inner selector", "Qonductor-style regression", "Architecture/core adaptation", "job; restricted here to one circuit", "observed job time", learned_target,
         "swap counts CX/CZ/ECR; operand-stack depth; active width; shots; circuit_count=1", "ET/RF/GB/Ada/HGB/polynomial; 307 bounded candidates; four inner folds then full outer refit",
         "Pinned extraction/constructors; bounded rather than full upstream search; qualified nominal compiled QPack view; mitigation not reproduced", "Six-family and inner-only selector comparison", "Literal SWAP; full search/mitigation reproduction; source-local fits masquerading as unified", ADD + "track_a_fidelity.csv;" + qa)
    card("qcre", "QCRE/Tremba pinned timing component", "QCRE/Tremba", "Original/native estimator", "compiled unitary circuit", "unitary critical-path seconds", learned_target,
         "BQSKit unitary view and nominal native duration map", "Pinned estimate_runtime, max per-wire arrival time", "Native component only; terminal measurement/barriers omitted; historical routing unknown",
         "Original component invoked under declared nominal context", "Full QCRE compilation/paper reproduction; raw service-time predictor", PARENT + "analytical/qcre_source_receipt.json;" + qa)
    card("qiskit", "Qiskit scheduled-duration reference", "Qiskit duration", "Original/native estimator", "scheduled circuit", "scheduled single-shot seconds", learned_target,
         "Frozen circuit and nominal Target", "QuantumCircuit.estimate_duration", "Backend-matched snapshot not verified job-day; possible measurement/reset semantics differ from QCRE",
         "Native API timing reference", "Scheduled duration equals observed service or QCRE unitary clock", ADD + "track_a_fidelity.csv;" + qa)
    card("shot_scale", "Deterministic shot multiplication", None, "Generalized proxy", "one circuit at shots S", "S × scheduled one-shot seconds", learned_target,
         "Native timing and shots", "Multiply duration by S", "No measured inter-shot/readout/host/batch cost decomposition", "Declared repeat-cost proxy", "Verified shot extrapolation or provider runtime identity", ADD + "runtime_estimator_interpretation.md;" + qa)
    card("scholten_proxy", "Scholten-inspired generalized proxy", "Scholten nominal", "Generalized proxy", "generic one-circuit observation (original: kernel job)", "M*S*D_eff/C original; generic-depth nominal seconds locally", learned_target,
         "Compiled wire depth + admitted nominal throughput", "Generic depth/shots/throughput, 3399/4515 eligible", "Generic wire depth is not ensemble/QV-normalized D_eff; 1116 unavailable; no CLOPS_h/v conversion",
         "Evaluate explicitly generalized proxy on its eligible rows", "Scholten original accuracy; manufactured CLOPS; full-panel ranking", ADD + "track_a_fidelity.csv;" + qa)
    card("scholten_original", "Scholten original (not evaluable)", "Scholten nominal", "UNVERIFIED / not evaluated", "kernel job", "M*S*D_eff/C", learned_target,
         "Missing kernel/template and QV ensemble expectations with compatible throughput", "D_eff=E[depth(U†(x)U(y))]/E[depth(QV(v,v))]*v; v=ceil(sqrt(2Dn))",
         "0/4515 verified eligibility; not executed as original", "Explain input eligibility and method definition", "Use proxy scores as original-paper reproduction", ADD + "runtime_estimator_interpretation.md")
    card("hyb", "Hyb-HANAS identifiable cost component", "Hyb-HANAS component", "Architecture/core adaptation", "reliability-adjusted circuit/training work", "useful-work/resource effective seconds", learned_target,
         "Native timing/errors/T2/routing and survival/log-cost", "Gate cost divided by survival; identifiable one-circuit terms", "No full NAS/gradient/step model; finite4175; log-cost4367; undefined values explicit",
         "Cost-component and missingness study", "Literal execution-time identity; full hybrid NAS reproduction", ADD + "track_a_fidelity.csv;" + qa)
    card("service_bridge", "Outer-train analytical-to-service bridge", None, "Calibrated adaptation", "eligible circuit observation", "native timing/cost/throughput", learned_target,
         "Raw or shot-scaled predictor plus eligible outer-train labels", "Affine/log-affine; separately named log-cost Ridge", "New fitted predictor; coverage depends on route", "Train-only service calibration and matched reference comparison", "Physical launch/host overhead identified; native estimator fidelity restored", DEEP + "analytical_controls_v1.csv;" + qa)
    card("references", "Train-only constant references", None, "Benchmark-defined diagnostic", "held-out observation", "observed service/runtime reference", learned_target,
         "Outer-train target labels; source identity only for source median", "Mean/global median/source median; fallback rules explicit", "Retrospective secondary controls; source-provenance information advantage", "Measure skill beyond simple constants", "Preregistered primary models or equal-information source reference", ADD + "qpu_predictive_skill.csv;" + qa)
    card("sim_mali", "Context-specific Graph/global controls", "Ma–Li-style simulator controls", "Architecture/core adaptation", "circuit/configuration", "simulator observed execution time", "context-specific warm sample/get_state or engine-reported time",
         "Local DAG/node/globals per frozen context", "Graph/global networks", "Different engines, schemas and fit policies; not exact upstream model run", "Within-context pipeline comparison", "Universal simulator winner; isolated topology benefit", sm)
    card("azizov", "Azizov-described source/hybrid/transpiled GNN", "Azizov local Aer", "Architecture/core adaptation", "circuit × noise/transpilation context", "noisy Aer after-transpile execution", "warm Aer execution seconds after transpilation",
         "Source/compiled graph and globals; FakeSherbrooke Opt1 shots1024", "Three two-branch GNN representations; three-seed medians", "Paper full encoding/code unavailable; one backend/level and local hardware/software", "Local source-to-transpiled pipeline contrast", "Exact authors encoding or full two-backend/four-level reproduction", sm + ";" + DEEP + "simulator_paired_v1.csv")
    card("sim_classical", "Context-specific classical/complexity controls", "Dense empirical controls", "Benchmark-defined diagnostic", "circuit/configuration", "context-specific measured runtime", "same context as each neural comparator",
         "Local source/compiled counters or 2^n×gate work features", "Linear/Ridge/SVR/RF/XGBoost; frozen context-specific fits", "Different feature views, budgets and clocks; not a single pooled method", "Matched-context empirical/complexity baseline", "Physical launch coefficient or universal scaling law", sm)
    card("family_joint", "Family-aware joint MPS adaptation", "Family-Aware MPS", "Architecture/core adaptation", "circuit × approximation rung", "joint runtime/approximation target", "warm get_state seconds plus quality probability",
         "Family context + local six-chi ladder, fidelity>=0.99", "Family classifier, residual/FiLM and joint heads; agnostic control", "Quantum Rings threshold replaced by local bond/fidelity policy; 840rungs/140hashes",
         "Runtime/quality/safe-selection evaluated separately", "Original engine reproduction; Brier improvement proves safe selection", sm + ";" + DEEP + "mps_selection_safety_v1.csv")
    card("family_fixed", "Fixed-MPS family residual control", "Family-Aware MPS", "Architecture/core adaptation", "circuit at fixed chi16", "family-conditioned runtime", "warm get_state at FP64 chi16",
         "Fixed-state context family/global inputs", "Family residual versus family-agnostic network", "Runtime-only fixed-bond control, not full joint approximation model",
         "Fixed-context family-conditioning contrast", "Joint model reproduction or quality guarantee", sm)
    card("maestro", "Maestro-style local component", "Maestro component", "Architecture/core adaptation", "circuit/engine/hardware", "calibrated runtime supporting engine choice", "QCSim CPU FP64 engine-reported seconds; fresh processes",
         "Width/gate/sampling calibration, shots1000", "Local operation-complexity sum + sampling terms", "Optimizer off, one engine; closed Composer selection not reproduced; 143/150 hashes",
         "Component skill versus same-target Ridge/median", "Persistent warm wall-clock or full automatic selection reproduction", sm)
    card("tn", "cuTensorNet native RUNTIME_EST", "cuTensorNet", "Original/native estimator", "selected contraction plan", "all-slice selected-plan contraction seconds", "same-plan warm scalar contraction",
         "Network/path/slices/workspace/GPU/complex64", "Native architecture-aware cost estimate", "Scalar not full state/sample; internals opaque; build/search/first excluded from comparison",
         "Native same-plan calibration diagnostic", "Internal formula recovered; predicts whole simulator API", DEEP + "tn_same_plan_slices_v1.csv")
    card("pasqal", "Pasqal analog MPS/TDVP companion", "Pasqal analog", "Benchmark-defined diagnostic", "analog program/configuration", "analog MPS/TDVP runtime resource model", "host MPSBackend.run, excluding quality capture",
         "Atoms, pulse program, chi and TDVP settings", "Documented N²chi³/N³chi² work terms; measured quality/timing grid", "18cells; coefficients not identifiable/promoted; no validated predictor", "Analog timing/quality observations", "Digital leaderboard or identified extrapolation coefficients", DEEP + "pasqal_stage_quality_v1.csv")
    card("historical_controls", "Historical structural/transfer/width controls", None, "Benchmark-defined diagnostic", "historical circuit or kernel population", "study-specific service/log-runtime/kernel time", "historical clocks kept separate from parent4515",
         "Coarse graphs, static blocks, interventions or PyTorch work features", "Fixed-model intervention or retrained controls, explicitly distinguished", "340/170/coarse representation or separate kernel; topology scores mean predictions, not mean seed MAEs or current median ensemble",
         "Bounded reliance/incremental evidence", "Full4515 topology causality or universal pretraining/scaling result", ADD + "feature_contribution.md")
    card("holdout_families", "QNN / Random held-out families (not estimators)", None, "Benchmark-defined diagnostic", "family holdout", "observed service seconds", learned_target,
         "Frozen family test IDs", "Evaluate existing Graph/MLP/CPU models", "These labels identify workloads, not QNN or random-prediction baseline algorithms",
         "Systematic generalization failures", "Invent a quantum-neural predictor or random estimator score", ADD + "generalization.md")
    cardmap = {r["card_id"]: r for r in cards}

    def qpu_card(method):
        if method.startswith("outer_train_"): return "references"
        if method == GRAPH: return "mali_graph"
        if method == MLP: return "mali_mlp"
        if method.startswith("qonductor_"): return "qonductor"
        if method.endswith(("__affine", "__log_affine")) or "log_cost_ridge" in method: return "service_bridge"
        if method.startswith("scholten_"): return "scholten_proxy"
        if "shot" in method: return "shot_scale" if not method.startswith("hyb_") else "hyb"
        if method.startswith("qcre_"): return "qcre"
        if method.startswith("qiskit_"): return "qiskit"
        if method.startswith("hyb_"): return "hyb"
        raise AssertionError(method)

    def sim_card(context, method):
        if method in {"outer_train_median", "train_fold_median"}: return "references"
        if "family_aware" in method or "family_conditioned" in method or "ablation" in method: return "family_joint"
        if method.startswith(("family_residual", "family_agnostic_median")): return "family_fixed"
        if "azizov_gnn" in method: return "azizov"
        if "maestro_style" in method: return "maestro"
        if "graph" in method or "mlp" in method: return "sim_mali"
        return "sim_classical"

    mappings = []
    for r in qpu:
        cid = qpu_card(r["method_id"])
        meta = [x for x in parent_metadata if x["method_id"] == r["method_id"]]
        output_clock = meta[0]["method_output_clock"] if meta else "observed_service_reference_seconds"
        mappings.append(dict(track="A", context_id="qualified4515_grouped_OOF", method_id=r["method_id"], card_id=cid,
            category=cardmap[cid]["category"], successful_n=r["n"], assigned_n=r["assigned_n"], evidence_path=qa))
        mappings[-1].update(evaluation_target_clock="archived_observed_service_execution_time", method_output_clock=output_clock,
                            coverage_unit="held-out observations", native_or_bridge="label-fitted bridge" if cid=="service_bridge" else "see method card")
    for r in sim:
        cid = sim_card(r["context_id"], r["method_id"])
        mappings.append(dict(track="B", context_id=r["context_id"], method_id=r["method_id"], card_id=cid,
            category=cardmap[cid]["category"], successful_n=r["n"], assigned_n=r["assigned_n"], evidence_path=sm))
    for cid, context, n, assigned in [("tn", "same_plan_scalar_contraction", 603, 612), ("pasqal", "analog_TDVP", 18, 18), ("scholten_original", "qualified4515_not_evaluable", 0, 4515)]:
        mappings.append(dict(track="A" if cid.startswith("scholten") else "B", context_id=context, method_id=cid, card_id=cid,
            category=cardmap[cid]["category"], successful_n=n, assigned_n=assigned, evidence_path=cardmap[cid]["evidence_paths"]))

    files = {}

    def table(name, title, rows, cols, note):
        files["tables/" + name + ".csv"] = csv_text(rows)
        files["tables/" + name + ".md"] = "# " + title + "\n\n" + note + "\n\n" + md_table(rows, cols)

    table("estimator_identity", "Bảng 1. Định danh và fidelity", cards,
          [("reader_label", "Estimator/thành phần"), ("original_target", "Target gốc"), ("benchmark_target", "Target benchmark"),
           ("category", "Phân loại implementation"), ("adaptation", "Thay đổi chính"), ("prohibited_claim", "Giới hạn")],
          "Phân loại áp dụng cho implementation/thành phần đã dùng. Không có cơ sở gọi toàn bộ pipeline ở đây là Faithful reproduction. QNN/Random là family, không phải estimator. Chi tiết và nguồn ở registry.")
    files["tables/method_context_index.csv"] = csv_text(mappings)
    registry = "# Estimator identity và fidelity\n\nFinal Scientific Freeze v2. Registry diễn giải dẫn xuất, không thay registry/protocol lịch sử. Phạm vi bao phủ: mọi variant trong 34 dòng Track A, 49 dòng Track B, native TN/analog và original Scholten không evaluable. Không coi số variant là số paper. Các controls lịch sử được nhóm riêng, không tráo vào cohort hiện tại.\n\n"
    registry += "Các category mô tả implementation: native component có thể được gọi nguyên bản trong một pipeline vẫn có nominal-context adaptation. Calibrated adaptation là predictor mới có học nhãn. Faithful reproduction chỉ dùng khi toàn bộ method/procedure được chứng minh; không gán nhãn đó chỉ vì sử dụng cùng data hoặc có số gần paper. `UNVERIFIED / not evaluated` giữ bất định thay vì ép category.\n\n"
    registry += "[Bảng định danh](tables/estimator_identity.md) và [index đầy đủ method × context](tables/method_context_index.csv). Các constant references dùng clock của context tương ứng; không mang clock QPU sang simulator.\n\n"
    for r in cards:
        registry += f"## {r['card_id']}: {r['reader_label']}\n\n"
        for key, label in [("original_source", "Source/paper"), ("original_task", "Task gốc"), ("original_observation_unit", "Observation unit gốc"),
                           ("original_target", "Target gốc"), ("original_inputs", "Input gốc"), ("original_mechanism", "Mechanism gốc"),
                           ("implemented_inputs", "Input benchmark"), ("implemented_component", "Implementation"), ("adaptation", "Khác biệt"),
                           ("category", "Fidelity của implementation"), ("benchmark_target", "Target đánh giá"), ("allowed_claim", "Được kết luận"), ("prohibited_claim", "Không được kết luận")]:
            registry += f"- {label}: {r[key]}.\n"
        registry += "\nEvidence: " + links(r["evidence_paths"]) + ".\n\n"
    registry += "## Quy tắc runtime boundary\n\nQCRE unitary path, Qiskit single-shot schedule, shot multiplication, Hyb reliability cost và Scholten proxy đều khác observed service. Chỉ các bridges có học outer-train labels mới xuất dự đoán service. Dense sample, MPS get_state, Aer sau transpile, QCSim engine time, scalar contraction và analog TDVP phải đọc ở context riêng. Native `RUNTIME_EST` không phải learned service bridge.\n"
    files["estimator_fidelity_registry.md"] = registry

    performance = []
    for r in qpu:
        out = dict(r)
        meta = [x for x in parent_metadata if x["method_id"] == r["method_id"]]
        out.update(card_id=qpu_card(r["method_id"]), evaluation_target_clock="archived_observed_service_execution_time",
                   method_output_clock=meta[0]["method_output_clock"] if meta else "observed_service_reference_seconds",
                   comparison_rule="identical eligible held-out rows only; not rank across coverage sets")
        if int(r["n"]) == 4515:
            out["skill_vs_global_median"] = 1 - num(r["mae_seconds"]) / num(median["mae_seconds"])
            out["skill_vs_classical_selector"] = 1 - num(r["mae_seconds"]) / num(c["mae_seconds"])
        performance.append(out)
    table("track_a_performance", "Bảng 2. Track A trên cohort 4.515", performance,
          [("method_id", "Variant"), ("method_output_clock", "Output clock"), ("n", "n thành công"), ("coverage", "Coverage /4515"), ("mae_seconds", "MAE (s)"),
           ("medae_seconds", "MedAE (s)"), ("rmse_seconds", "RMSE (s)"), ("r2_seconds", "R²"), ("source_balanced_mae", "Equal-source MAE (s)")],
          "Giữ toàn bộ 34 dòng đã lưu, kể cả cost/shot-scaled variants có sai số lớn. Đây không phải leaderboard ngang hàng giữa native outputs và calibrated service estimators. Coverage=n_success/4515. MAE=(Σ|prediction−target|)/n_success; RMSE=sqrt(mean residual²); R²=1−SSE/SST trên chính rows đó. Skill=1−MAE_method/MAE_reference chỉ trên cùng rows. Graph 26.7195604% so với global median và 5.9237157% so với inner-selected regressor đã đối chiếu độc lập. Source-balanced là mean của ba source MAE, không mean R². Các controls eligible-specific ở bảng analytical riêng.")
    table("analytical_controls", "Analytical raw, affine và matched controls", analytical,
          [("raw_method", "Route"), ("variant", "Variant/control"), ("n", "n"), ("mae_seconds", "MAE (s)"), ("r2_seconds", "R²")],
          "35 dòng giữ nguyên. Eligible controls dùng chính training eligibility của route; không thay bằng global reference của full cohort. Calibration là predictor bổ sung. Scholten affine .74835 so eligible median .82007 và source median .69207, không phải hai median đều .69207.")

    general = []
    for r in decomp:
        case = r["case_id"]
        ref = unique(skills, scope=case, method_id=r["method_id"], reference_id="train_global_median")
        selector = unique(skills, scope=case, method_id=SELECTOR, reference_id="train_global_median")
        p = unique(full_pairs, case_id=case, slice_axis="support_scope", scope="all_test", method_a=MLP, method_b=GRAPH)
        general.append(dict(r, train_global_median_mae_seconds=ref["reference_mae_seconds"],
            skill_vs_train_median=ref["skill"], selector_mae_seconds=selector["mae_seconds"],
            graph_minus_mlp_observed_mae_delta_seconds=-num(p["observed_mae_delta_a_minus_b_seconds"]) if p["observed_mae_delta_a_minus_b_seconds"] else "",
            graph_minus_mlp_ci_low_seconds=-num(p["bootstrap_ci_high_seconds"]) if p["bootstrap_ci_high_seconds"] else "",
            graph_minus_mlp_ci_high_seconds=-num(p["bootstrap_ci_low_seconds"]) if p["bootstrap_ci_low_seconds"] else "",
            uncertainty_status=p["bootstrap_status"], estimator="protocol-defined per-row median-of-three, seeds42/1234/31415"))
    table("generalization_robustness", "Bảng 3. Generalization theo từng case", general,
          [("experiment", "Loại"), ("case_id", "Case"), ("method_id", "Method"), ("n", "n"), ("structural_groups", "Groups"),
           ("mae_seconds", "MAE (s)"), ("r2_seconds", "R²"), ("train_global_median_mae_seconds", "Train median MAE (s)"),
           ("exact_consumed_input_collision_rows", "Fit-input collisions"), ("uncertainty_status", "Paired CI status")],
          "42 case×method rows, 21 cases; không pool 6684 assignments thành IID dataset. Có 4515 unique IDs. Family embargo, backend-context structural overlap và size probes là các estimand khác nhau. CSV giữ residual/tail/support, CPU selector, observed Graph−MLP delta và CI đúng chiều. Bootstrap 10000/seed42 conditional theo structural group; <10groups không có CI. Parent/refit ratio chỉ descriptive vì population và training procedure khác nhau.")

    features = [
        ("schema", "Static extraction +257 saved transforms", "51raw; original sim41; parent40; RQ4 final11/22/40", "Exact support mechanism; tensor width khác số varying fields", "Không chứng minh author real tensor; không hard-code41", fa),
        ("fixed_model_edges", "Single-seed340 test-time intervention", "scratchMAE .4783; no-edge5.9652; reverse.5984 s", "Fixed-model reliance", "Không phải retrained gain hoặc current4515 ablation", "artifacts/mali/graph_intervention_audit_20260921/REPORT.md"),
        ("fixed_model_nodes", "Single-seed340 test-time intervention", "qubit-zero4.6862; type-zero.8964; index-zero.5043; T1T2-zero.4756 s", "Input sensitivity", "Abnormal inputs; không individual causal attribution", "artifacts/mali/graph_intervention_audit_20260921/REPORT.md"),
        ("coarse_topology", "Retrained; arithmetic mean of per-row predictions across three seeds;340/170;256bin", "node-onlyMAE .81454; trueDAG .84483; deltaCI[-.0165,.0762]s", "Không resolve DAG gain trong coarse representation", "Không phải mean of seed MAEs hoặc current median estimator; không generalize thành mọi graph vô dụng", "artifacts/validation/mali_physical_dag_v1/topology_controls_validation/REPORT.md"),
        ("static_blocks", "Retrained grouped physical descriptors", "QCRE.8274→compiled.6889→timing.5109→rich.4937 s", "Timing/static descriptor sets có signal", "Không complete T0–T3 ladder trên4515; không GNN superiority", "artifacts/validation/mali_physical_dag_v1/feature_ablation/REPORT.md"),
        ("critical_subgraph", "Retrained descriptor increment", "rich.4937→.4987 grouped; .5230→.5306 strict; .5753→.5934 family s", "Không stable MAE increment", "Một số log-R² gain không thay MAE finding", "artifacts/validation/mali_physical_dag_v1/critical_subgraph_evaluation/REPORT.md"),
        ("strict_transfer", "Compact logical retrained3seed mean", "scratchMAE.6831; fine-tune.6956; frozen1.9152 s", "Fine-tune gain unresolved; frozen harmful", "Không full-model/paper-wide transfer theorem", "artifacts/validation/mali_strict_transfer_logical_v1/evaluation/REPORT.md"),
        ("paper_components", "External paper component tables", "fullsimR².943; graph-only.928; global-only.573", "Authors' evidence", "Không locally reproduced component scores", ADD + "feature_contribution.md"),
        ("aer_views", "Locally fitted representation variants", "source.44966; hybrid.40636; transpiled.30076 s", "Compiled view improves vs source in local matched test", "Hybrid increment unresolved; not individual feature causality", sm),
        ("family_conditioning", "Matched retrained conditioned/agnostic", "MAE.31163/.30752 s; CI includes0", "Không resolved local runtime gain", "Không equivalence proof; quality safety là endpoint khác", sm),
    ]
    fr = [dict(control_id=a, control_type=b, finding=c0, interpretation=d, limitation=e, evidence_paths=f) for a,b,c0,d,e,f in features]
    table("feature_structural_contribution", "Bảng 4. Feature và structural evidence thực sự đã đánh giá", fr,
          [("control_id", "Đối chiếu"), ("control_type", "Loại evidence"), ("finding", "Finding"), ("interpretation", "Diễn giải"), ("limitation", "Giới hạn")],
          "Không dựng complete T0–T3 ladder. Intervention sensitivity, retrained increment và paper-reported scores có cấp evidence khác nhau. Các số controls lịch sử là số đã lưu/rounded, không tính lại trên cohort mới. Mỗi CSV row giữ đúng nguồn.")

    clocks = {
        "cudaq_dense_statevector_fp32": "FP32 warm sample wall-clock",
        "cudaq_dense_statevector_fp64": "FP64 warm sample wall-clock",
        "cudaq_mps_fp64_bond16_warm_state_execution_seconds": "FP64 chi16 warm get_state wall-clock",
        "qiskit_aer_fake_sherbrooke_noisy_opt1": "warm Aer execution after transpilation; Opt1/1024 shots",
        "qcsim_statevector_cpu_fp64_1000_shots": "fresh-process engine-reported execution; optimizer off/1000shots",
        "cudaq_mps_ladder_c44": "warm get_state; six chi rungs; grouped hash folds",
        "cudaq_mps_ladder_family_holdout": "warm get_state; six chi rungs; family holdout",
    }
    sim_rows = [dict(r, runtime_boundary=clocks[r["context_id"]], card_id=sim_card(r["context_id"], r["method_id"]),
                     comparison_rule="same context and shared hashes/rungs only") for r in sim]
    sim_rows.extend([
        dict(context_id="cuTensorNet same-plan", method_id="native_RUNTIME_EST", assigned_n=612, n=603,
             runtime_boundary="all-slice complex64 scalar contraction vs same-plan warm", diagnostic="EST/warm median: core3.734012; frontier4.132416;201members/188hashes", card_id="tn", comparison_rule="plan rows not independent circuit hashes"),
        dict(context_id="Pasqal analog companion", method_id="unpromoted_resource_formula", assigned_n=18, n=18,
             runtime_boundary="host MPSBackend.run excluding quality capture", diagnostic="17qualitypass/18; no identified predictive coefficients", card_id="pasqal", comparison_rule="measured cells, not18 validated runtime predictions"),
    ])
    table("track_b_findings", "Bảng 5. Track B theo execution context", sim_rows,
          [("context_id", "Context"), ("runtime_boundary", "Runtime boundary"), ("method_id", "Estimator"), ("n", "n"),
           ("mae_seconds", "MAE (s)"), ("r2_seconds", "R²"), ("diagnostic", "Diagnostic nếu không có MAE")],
          "49 predictive metric rows trong7contexts +2native/analog diagnostics. Mỗi context có frozen grouped evaluation riêng. Ladder n=840rungs từ140hashes, không840circuits độc lập;150hashes được assign. FixedMPS150 có147qualitypass/3fail. Maestro143/150 supported. Không dùng runtime của different APIs để tạo universal rank. CSV giữ đầy đủ tails/coverage; missing metric là unavailable, không0.")

    opens = [
        ("Full4515 topology-only increment", "Historical coarse controls; whole Graph–MLP contrast", "Input/calibration/capacity differ", "Matched full-model retrained topology control", "Giới hạn attribution, không invalidates pipeline comparison"),
        ("Individual feature/calibration benefit", "Zeroing + static blocks +support", "Reliance không equal retrained importance", "Matched fit-only individual/block controls", "Không claim causal feature value"),
        ("Author executed real tensor", "Notebook12; default41; released340 support40", "No processed author tensor/run receipt", "Verified author tensor/config/provenance", "Không exact reproduction claim"),
        ("Systematic unseen-shot transfer", "Observational six Qonductor shot values", "Confounded source/workload; neural no shots input", "Frozen shot-held-out crossed protocol", "Không verified shot extrapolation"),
        ("Isolated hardware-generation effect", "10backend contexts", "Source/structures/clock/shots not crossed independently", "Matched circuits/contexts across devices", "Backend-context only"),
        ("Prospective prediction intervals", "Conditional paired error bootstrap", "Not predictive coverage or training variability", "Predeclared calibrated coverage evaluation", "Không deployment error guarantee"),
        ("Original Scholten accuracy", "Original definition and generic proxy", "Kernel/QV ensemble input semantics missing", "Qualified kernel experiment with compatible throughput", "Original NOT EVALUATED"),
        ("General multi-engine/hardware simulator transfer", "Local per-context comparisons", "Different clocks/tasks, one local resource profile", "Matched target transfer protocol", "Không universal simulator predictor"),
        ("Identified Pasqal coefficients/TN internals", "18analog cells/native plan estimates", "Work terms confounded; closed internal estimator", "Identifiability grid/public native model specification", "Diagnostic not mechanism discovery"),
        ("Physical launch-overhead decomposition", "Saved zero-work/width microbench", "API/state/sampling costs intermixed", "Isolated timing boundaries/controls", "Intercept not physical launch latency"),
    ]
    open_rows = [dict(question=a, existing_evidence=b, why_insufficient=c0, required_evidence=d, thesis_impact=e,
                      status="NOT EVALUATED", recommendation="Future amendment only if stronger claim is pursued; no current experiment necessary") for a,b,c0,d,e in opens]
    table("open_questions", "Bảng 6. Câu hỏi khoa học còn mở", open_rows,
          [("question", "Câu hỏi"), ("existing_evidence", "Evidence hiện có"), ("why_insufficient", "Chưa đủ vì"),
           ("required_evidence", "Evidence cần nếu nghiên cứu tiếp"), ("thesis_impact", "Giới hạn thesis")],
          "Các câu hỏi này không phải authorized backlog. NOT EVALUATED áp dụng cho stronger question ghi trong cột đầu, không phủ nhận completed negative results. Không cần experiment bổ sung để giữ các bounded claims hiện tại.")

    claims = []

    def claim(cid, rq, scope, target, split, methods, paths, metrics, kind, strength, limitation, wording, stronger):
        assert strength in STATUS
        for p in paths.split(";"):
            s.read(p)
        claims.append(dict(claim_id=cid, reporting_question=rq, dataset_unit=scope, target_semantics=target,
            split_context=split, estimators=methods, evidence_paths=paths,
            exact_metrics_json=json.dumps(metrics, ensure_ascii=False, sort_keys=True), evidence_type=kind,
            evidence_strength=strength, limitations_confounders=limitation, thesis_wording=wording,
            unsupported_stronger_claim=stronger))

    pg = unique(pair_refs, method_id=GRAPH, reference_id="outer_train_median")
    ps = unique(parent_pairs, left_method=GRAPH, right_method=SELECTOR)
    pm = unique(parent_pairs, left_method=MLP, right_method=GRAPH)
    macro_g = unique(macro, method_id=GRAPH, reference_id="inner_selected_classical_regression")
    claim("C01", "RQ1", "4515observations/165groups", learned_target, "5outer grouped OOF", GRAPH + ";outer_train_median", qa + ";" + DEEP + "qpu_shared_reference_comparisons_v1.csv",
          dict(graph_mae=num(g["mae_seconds"]), median_mae=num(median["mae_seconds"]), skill=skill_global,
               observed_delta=num(pg["observed_mae_delta_seconds"]), ci_low=num(pg["ci_low_seconds"]), ci_high=num(pg["ci_high_seconds"])),
          "saved OOF paired observational comparison", "SUPPORTED WITH QUALIFICATION", "Conditional group bootstrap; heterogeneous source boundaries",
          "Graph giảm pooled MAE 26,7% so với global train median trên cùng 4.515 rows; paired interval âm trong phép so sánh này.", "Graph universally superior to all baselines")
    claim("C02", "RQ1", "4515observations/165groups", learned_target, "same outer OOF", GRAPH + ";" + SELECTOR, qa + ";" + PARENT + "summary/shared_row_comparisons.csv",
          dict(skill=skill_selector, observed_delta=num(ps["observed_mae_delta_seconds"]), ci_low=num(ps["bootstrap_ci_low_seconds"]), ci_high=num(ps["bootstrap_ci_high_seconds"])),
          "saved paired comparison", "INCONCLUSIVE", "CI includes0; pipeline inputs/budgets/refits differ",
          "Graph có point MAE thấp hơn selector 5,9%; evidence chưa resolve superiority theo paired uncertainty đã lưu.", "Graph decisively beats strongest classical model")
    claim("C03", "RQ1/RQ4", "3sources; same4515rows", learned_target, "equal-source versus pooled", GRAPH + ";" + SELECTOR + ";" + MLP, qa + ";" + ADD + "qpu_source_balanced_skill.csv",
          dict(graph_macro=num(g["source_balanced_mae"]), selector_macro=num(c["source_balanced_mae"]), graph_macro_skill=num(macro_g["source_balanced_skill"]), graph_equal_group=num(g["equal_group_mae"]), mlp_equal_group=num(m["equal_group_mae"])),
          "deterministic reweighting of frozen errors", "SUPPORTED", "Weighting answers different estimands; source-aware reference uses provenance",
          "Selector có equal-source MAE thấp hơn Graph; equal-group weighting đảo thứ tự Graph/MLP. Thứ hạng phụ thuộc distribution.", "One universal winner independent of source/group weights")
    claim("C04", "RQ1/RQ3", "4515observations/165groups", learned_target, "same OOF rows", GRAPH + ";" + MLP, qa + ";" + PARENT + "summary/shared_row_comparisons.csv",
          dict(graph_mae=num(g["mae_seconds"]), mlp_mae=num(m["mae_seconds"]), graph_minus_mlp_delta=-num(pm["observed_mae_delta_seconds"]), ci_low=-num(pm["bootstrap_ci_high_seconds"]), ci_high=-num(pm["bootstrap_ci_low_seconds"])),
          "whole-pipeline comparison", "INCONCLUSIVE", "Graph gets nodes/T1T2; capacity differs; interval includes0",
          "Point MAE nghiêng về Graph, nhưng chưa có resolved advantage so với global MLP; không xác định topology-only gain.", "DAG topology causally improves accuracy")
    tc = topology["paired_group_bootstrap"]["true_dag_minus_node_only"]
    claim("C05", "RQ3", "340observations/170QASM historical coarse256bin", "historical log/runtime labels", "retrained; per-row arithmetic-mean predictions, then score", "node-only;true/reverse/shuffledDAG", fr[3]["evidence_paths"]+";artifacts/validation/mali_physical_dag_v1/topology_controls_validation/summary.json;experiments/evaluate_mali_topology_controls.py",
          dict(node_only_mae=topology["ensemble_metrics"]["node_only"]["mae_seconds"], true_dag_mae=topology["ensemble_metrics"]["true_dag"]["mae_seconds"], delta_ci_low=tc["bootstrap_mae_seconds_95pct"][0], delta_ci_high=tc["bootstrap_mae_seconds_95pct"][1], seeds=topology["seeds"],bootstrap_replicates=topology["bootstrap_replicates"],bootstrap_seed=topology["bootstrap_seed"]),
          "retrained topology control", "INCONCLUSIVE", "Coarsened representation; not full4515model",
          "Saved controlled topology study không resolve positive DAG increment trong coarse representation đó. Scoring dùng mean predictions của ba seeds, không mean của seed MAEs.", "Topology irrelevant for every graph model")
    claim("C06", "RQ3", "340historical observations", "historical service/runtime", "fixed-model test-time perturbation", "scratchGraph", fr[1]["evidence_paths"],
          dict(full_mae=.4783, no_edge_mae=5.9652, reverse_mae=.5984, qubit_zero_mae=4.6862),
          "intervention without retraining", "SUPPORTED WITH QUALIFICATION", "Unnatural inputs; historical preprocessing/single seed",
          "Model đã fit phụ thuộc edges/wire identities dưới perturbation; không suy ra gain sau retrain.", "Edge removal proves causal practical topology advantage")
    claim("C07", "RQ2/RQ3", "1510namedQASM×2sim contexts; parent4515;257transforms", "feature schema, not target metric", "static support and fit-only replay", "Ma–Li Graph/MLP", fa + ";" + ADD + "effective_input_widths.csv;" + ADD + "global_feature_ledger.csv",
          dict(raw=51, original_sim_support=41, current_parent=40, transforms=257, rq4_final_widths=[11,22,40]),
          "source-feature reconstruction and exact transform replay", "SUPPORTED", "Tensor positions may include zeroed constants; support varies by train population",
          "51 raw positions support-refine thành 41 trong simulator corpus và 40 ở mọi parent fit; RQ4 width theo fit population.", "Hard-coded41 or51actual current inputs")
    claim("C08", "RQ2", "published340real rows/upstream notebook", "author input provenance", "source/notebook comparison", "author Ma–Li real model", fa + ";" + ADD + "upstream_pipeline_evidence.json",
          dict(recomputed_released_support=40, notebook_example=12, constructor_default=41), "source provenance audit", "INCONCLUSIVE", "Author processed real tensor/executed mask absent",
          "Không xác minh được tác giả thực thi real tensor 12/40/41; current width 40 được verified độc lập.", "Authors definitely used40, or current model is paper-exact")
    claim("C09", "RQ2", "qualified4515 and per-context simulator", "native versus learned output clocks", "method-source audit", "all pipeline families", ADD + "track_a_fidelity.csv;" + DEEP + "method_fidelity_v1.csv",
          {}, "implementation/source comparison", "SUPPORTED WITH QUALIFICATION", "Dataset change alone not fidelity failure; implementation/procedure differences matter",
          "Graph/MLP và nhiều methods là adaptations; pinned native components không làm toàn pipeline trở thành exact reproduction.", "Every evaluated row is an original-paper reproduction")
    claim("C10", "RQ2", "4515timing rows", learned_target, "raw/shot-scaled versus outer-train bridge", "QCRE;Qiskit", qa + ";" + DEEP + "analytical_controls_v1.csv",
          dict(qcre_one_shot_mae=num(unique(qpu,method_id="qcre_source_estimator__raw")["mae_seconds"]),
               qiskit_one_shot_mae=num(unique(qpu,method_id="qiskit_estimate_duration_snapshot__raw")["mae_seconds"]),
               qcre_shot_affine_mae=num(unique(qpu,method_id="qcre_source_estimator_shot_scaled__affine")["mae_seconds"]),
               qiskit_shot_affine_mae=num(unique(qpu,method_id="qiskit_shot_scaled_schedule_seconds__affine")["mae_seconds"])),
          "analytical definition and train-only calibrated comparison", "SUPPORTED WITH QUALIFICATION", "Nominal context; omitted terms; affine is new model",
          "Scheduled one-shot timing khác observed service. Calibration giảm error nhưng không identify physical overhead.", "Raw duration and service time are interchangeable")
    schol = {r["variant"]: num(r["mae_seconds"]) for r in analytical if r["raw_method"] == "scholten_unified_nominal_throughput__raw"}
    claim("C11", "RQ2", "3399successful/4515assigned", learned_target, "eligible grouped OOF", "Scholten-inspired proxy/bridges", qa + ";" + DEEP + "analytical_controls_v1.csv;" + ADD + "runtime_estimator_interpretation.md",
          schol, "generalized proxy plus saved secondary controls", "SUPPORTED WITH QUALIFICATION", "Original eligibility0; nominal throughput/generic depth;1116unavailable",
          "Calibrated proxy tốt hơn eligible global median, không tốt hơn source-median point estimate. Không đánh giá accuracy original Scholten.", "Original D_eff verified or calibrated proxy full-panel winner")
    claim("C12", "RQ2", "0verified eligible/4515", "original kernel/QV runtime", "no qualified original evaluation", "Scholten original", ADD + "runtime_estimator_interpretation.md", {},
          "eligibility determination", "NOT EVALUATED", "Missing original kernel/QV ensemble semantics",
          "Original Scholten predictive accuracy không được evaluable trong cohort hiện tại.", "Proxy failure refutes original paper")
    claim("C13", "RQ2/RQ5", "4175finite raw;4367log-cost/4515", learned_target, "route-specific grouped OOF", "Hyb identifiable component/bridges", qa + ";" + DEEP + "analytical_controls_v1.csv",
          dict(raw_n=4175, log_n=4367, raw_mae=num(unique(qpu,method_id="hyb_nominal_r2_effective_cost_v1__raw")["mae_seconds"]), affine_mae=num(unique(qpu,method_id="hyb_nominal_r2_effective_cost_v1__affine")["mae_seconds"])),
          "cost semantics and matched reference comparison", "SUPPORTED WITH QUALIFICATION", "Zero survival/overflow explicit; log route different subset",
          "Reliability-adjusted resource cost không phải literal service time; vài bridges còn kém matched simple references.", "Reliability-adjusted seconds directly predict observed execution")
    for cid, family in [("C14","qnn"),("C15","random")]:
        rr = [r for r in decomp if r["case_id"] == "family:" + family]
        claim(cid, "RQ4/RQ5", f"{rr[0]['n']}rows/{rr[0]['structural_groups']}groups", learned_target, "family holdout with structural embargo", GRAPH + ";" + MLP + ";CPUselector", ADD + "generalization_decomposition.csv;" + ADD + "generalization_predictive_skill.csv",
              {r["method_id"]: {k:r[k] for k in ["mae_seconds","medae_seconds","rmse_seconds","r2_seconds","overprediction_rows","p99_absolute_error_seconds","max_absolute_error_seconds"]} for r in rr},
              "frozen family-OOD prediction/residual comparison", "SUPPORTED", "Support diagnostics descriptive, no mechanism-isolating intervention",
              f"Family {family}: cả Graph/MLP overpredict mọi row, MAE/MedAE lớn và R² âm nghiêm trọng. Relative Graph win không practical generalization.", "Discard rare family/outliers or claim useful model because Graph wins")
    claim("C16", "RQ4", "148Kyoto rows/117groups", learned_target, "backend-context holdout", GRAPH + ";" + MLP, ADD + "generalization_decomposition.csv;artifacts/final_research/rq4_report.md",
          dict(mlp_raw_and_transformed_collisions=148, observation_overlap=0, graph_complete_collisions=0),
          "raw/post-transform equality and saved split audit", "SUPPORTED WITH QUALIFICATION", "Repeated known structures; no MLP backend/calibration inputs",
          "Kyoto là known-circuit backend-context transfer với representation collision, không observation-ID leakage hoặc learned hardware adaptation.", "All148unseen inputs or isolated generation transfer")
    claim("C17", "RQ4", "21cases:9family/10backend/2size;126cells", learned_target, "frozen case-specific partitions", "Graph/MLP/CPU", ADD + "generalization.md;artifacts/final_research/rq4_report.md",
          dict(cpu_cases=21, neural_cases=21, cells=126, assignments=6684, unique_ids=4515, mlp_lower_mae_backend_cases=9),
          "completed-grid saved evaluation", "SUPPORTED WITH QUALIFICATION", "Cases overlap; topology/source/shots/clock confounding; historical Twolocal incident preserved",
          "Full 21-case grid complete, không đồng nghĩa universal generalization hay causal hardware-generation effect.", "Pool6684asindependent; universal hardware transfer")
    claim("C18", "RQ4", "1321rows/3groups and498rows/2groups", learned_target, "QPack size probes", GRAPH + ";" + MLP, ADD + "generalization_decomposition.csv", {},
          "frozen low-support size holdout", "SUPPORTED WITH QUALIFICATION", "No CI below10groups; second-case negativeR²/all underprediction",
          "Size results là descriptive narrow probes, không general scaling law.", "Broad unseen-size extrapolation established")
    claim("C19", "RQ4/RQ5", "observational shot diversity in4515", "observed service", "no systematic shot holdout", "all estimators", ADD + "shot_diversity.csv;" + ADD + "generalization.md", {},
          "inventory of completed protocols", "NOT EVALUATED", "Neural lacks shots; source/workload confounding; simulator shot probe separate",
          "Không có systematic real-QPU unseen-shot test; deterministic multiplication không thay thế evidence.", "Verified real-QPU shot scaling")
    claim("C20", "RQ2/RQ4", "32pairs/11recipes/230linked observations", "functional zero-input equivalence", "frozen declared layout mappings", "DD/ZX;measured-output checker", "artifacts/real_qpu/provenance_equivalence/results.json",
          dict(prepared_proved=qcec["dd_prepared_state_proofs"], measured_match=qcec["zero_input_measured_output_matches"], unitary_proved=qcec["dd_full_unitary_fixed_layout_proofs"], unitary_counterexample=qcec["dd_full_unitary_fixed_layout_counterexamples"]),
          "formal DD prepared-state plus numerical output checks", "SUPPORTED WITH QUALIFICATION", "Assumed mapping, zero input;ZX inconclusive not non-equivalence",
          "32/32 prepared-state PROVED và output match hỗ trợ recipe qualification; 8/32 full-unitary PROVED, 24 counterexamples. Không exact historical recovery.", "230independentproofs; full algorithmic/historical identity")
    for cid, context, method, reference, text, limitation in [
        ("C21","cudaq_dense_statevector_fp32","random_forest","outer_train_median","Dense FP32: RF có positive skill và lower MAE trong small-width local context.","FP32 sample only; no width26–30 extrapolation"),
        ("C22","cudaq_dense_statevector_fp64","random_forest","outer_train_median","Dense FP64: RF có positive skill; không cần graph để đạt lowest listed MAE.","MAE winner không R² winner; local panel only"),
        ("C23","cudaq_mps_fp64_bond16_warm_state_execution_seconds","graph_median_three_seeds","ridge_alpha_1","Fixed-chi MPS: Graph lower MAE hơn Ridge với saved CI âm.","150runtime rows include3quality failures; get_state not sample"),
        ("C24","qcsim_statevector_cpu_fp64_1000_shots","maestro_style_cpu_sv_component_optimizer_off_adaptation","nested_grouped_ridge","Maestro-style local component improves on matched Ridge.","Engine-reported fresh-process time; no closed selection reproduction"),
    ]:
        r=unique(sim,context_id=context,method_id=method); pr=unique(sim,context_id=context,method_id=reference)
        claim(cid,"Track B",f"{r['n']}shared hashes",clocks[context],"frozen grouped OOF",method+";"+reference,sm+";"+DEEP+"simulator_paired_v1.csv",
              dict(method_mae=num(r["mae_seconds"]),reference_mae=num(pr["mae_seconds"]),r2=num(r["r2_seconds"]),skill=1-num(r["mae_seconds"])/num(pr["mae_seconds"]),
                   paired_evidence=unique(sim_pairs,context_id=context,model=method,reference=reference)),
              "saved matched-context comparison","SUPPORTED WITH QUALIFICATION",limitation,text,"Universal simulator superiority")
    claim("C25","RQ3/Track B","150shared hashes","warm noisyAer after transpile","FakeSherbrooke Opt1 shots1024","source/hybrid/transpiledGNN;XGBoost",sm+";"+DEEP+"simulator_paired_v1.csv",
          dict(metric_rows={r["method_id"]:num(r["mae_seconds"]) for r in sim if "azizov_gnn" in r["method_id"] or r["method_id"]=="classical_xgboost_transpiled"},
               paired_evidence=[r for r in sim_pairs if r["context_id"]=="qiskit_aer_fake_sherbrooke_noisy_opt1"]),
          "locally retrained representation variants","SUPPORTED WITH QUALIFICATION","Source→transpiled CI resolves; hybrid→transpiled/GNN→XGB do not",
          "Compiled representation improves source-GNN locally; transpiled GNN superiority over XGBoost chưa resolve.","Full paper encoding reproduced or all GNNs beat trees")
    claim("C26","RQ3/Track B","840rungs/140hashes","warm get_state seconds","grouped hash and family holdout","family_conditioned;family_agnostic_ablation",sm+";"+DEEP+"simulator_paired_v1.csv",
          dict(paired_evidence=[r for r in sim_pairs if r["context_id"] in {"cudaq_mps_ladder_c44","cudaq_mps_ladder_family_holdout"}]),
          "matched retrained family-conditioning control","INCONCLUSIVE","Saved runtime delta intervals include0; this does not prove equality",
          "Không resolved runtime increment từ family conditioning trên local ladder.","Family context never helps or methods proven equivalent")
    claim("C27","RQ5/Track B","140assessed/150assignedhashes","selected-state fidelity>=.99","frozen threshold and selection policy","both jointMPSselectors",DEEP+"mps_selection_safety_v1.csv;"+DEEP+"mps_hash_tradeoff_v1.csv",
          {r["method_id"]:{k:r[k] for k in ["selected_observed","quality_violations","quality_violation_fraction_observed","unassessed_selections"]} for r in safety},
          "frozen quality-safety evaluation","SUPPORTED","10unassessed separate; runtime/Brier/operational endpoints distinct",
          "Cả hai selectors fail fidelity ở 64/140 choices dù 138/140 hashes có passing rung. Runtime accuracy không bảo đảm quality safety.","Hide failed choices or infer safe selection from Brier")
    claim("C28","RQ2/RQ5/Track B","603/612planrows;188hashes","same-plan scalar contraction","frozen path/slices/workspace/GPU","nativeRUNTIME_EST",DEEP+"tn_same_plan_slices_v1.csv",dict(success=603,assigned=612,core_ratio=num(tn_core["ratio_median"]),frontier_ratio=num(tn_frontier["ratio_median"])),
          "native plan-level calibration diagnostic","SUPPORTED WITH QUALIFICATION","No matched simple learned reference; opaqueinternals; excludesbuild/search",
          "Native EST overpredicts median warm contraction trong panel này; không whole-API runtime estimate.","Internal cost model known or general CUDA-Q timing predictor")
    claim("C29","RQ5/Track B","18analogcells","MPSBackend.run +separatequality","analogTDVPgrid","Pasqal companion",DEEP+"pasqal_stage_quality_v1.csv",dict(cells=18,quality_pass=17),
          "measured timing/quality companion","SUPPORTED WITH QUALIFICATION","No identified promoted cost coefficients/predictive evaluation",
          "Analog companion ghi quality/time tradeoff, chưa establish resource-formula predictive skill.","Identified asymptotic predictor or digital comparison")
    claim("C30","RQ5/Track B","historical208kernelrows +separateRXprobe","kernel/API clocks kept separate","saved grouped/width holdout","PyTorchworkRidge/HGB;CUDA-Qmicrobench",ADD+"runtime_estimator_interpretation.md;experiments/simulator_runtime_v2/README.md;artifacts/cudaq_runtime/launch_scaling_v1_20260922/analysis/REPORT.md",
          dict(width_holdout_ridge_log_r2=.8754,hgb_log_r2=.7295),"saved historical scaling diagnostics","SUPPORTED WITH QUALIFICATION","Different kernels; zero-work includes state/API/sampling costs",
          "Width holdout có thể đảo lựa chọn model; fitted intercept không physical launch latency.","General T=launch+k2^ngates law verified across simulators")
    claim("C31","RQ3","historicalphysical descriptor population","historicalservice/logruntime","retrained grouped/strict/family controls","compiled/static/timingblocks",fr[4]["evidence_paths"]+";"+fr[5]["evidence_paths"],{},
          "retrained feature-block controls","SUPPORTED WITH QUALIFICATION","Different population and representation than full4515",
          "Compiled/timing/static blocks có predictive signal; critical-subgraph additions không stable MAE benefit.","Complete native4515feature ladder or per-gate causal values")
    claim("C32","RQ5","currentbenchmark","prospective deployment accuracy","no prospective interval test","allestimators",ADD+"benchmark_question_matrix.csv",{},
          "scope/evidence inventory","NOT EVALUATED","Conditional bootstrap estimates comparison error, not deployment coverage",
          "Chưa có calibrated prospective prediction guarantee; report coverage/tails và uncertainty đúng mục đích.","95%pairedCI means95%futurejobs covered")

    files["evidence_to_claims.csv"] = csv_text(claims)
    claims_md = "# Evidence-to-claim registry\n\nCác claim bên dưới tái sử dụng matrix120rows, không chạy thêm experiments. Negative result đã đánh giá có thể SUPPORTED. INCONCLUSIVE chỉ dùng khi stronger comparison/provenance chưa resolve, không phải vì model thất bại. Exact metrics và source selectors ở CSV; path/hash ở manifest.\n\n"
    claims_md += md_table(claims,[("claim_id","ID"),("reporting_question","Mục"),("evidence_strength","Evidence strength"),("thesis_wording","Câu thesis có thể bảo vệ")]) + "\n"
    for r in claims:
        claims_md += f"## {r['claim_id']}\n\nUnit: {r['dataset_unit']}. Target: {r['target_semantics']}. Context: {r['split_context']}. Estimators: {r['estimators']}.\n\nEvidence type: {r['evidence_type']}. Strength: {r['evidence_strength']}.\n\n{r['thesis_wording']}\n\nGiới hạn: {r['limitations_confounders']}.\n\nKhông được suy ra: {r['unsupported_stronger_claim']}.\n\nNguồn: {links(r['evidence_paths'])}.\n\nExact metrics: `{r['exact_metrics_json']}`.\n\n"
    files["evidence_to_claims.md"] = claims_md.rstrip() + "\n"

    # Preserve 120 source questions explicitly; do not pretend statuses are equivalent.
    dimension_map = {
        "Estimator fidelity": "C07;C08;C09;C12", "Target validity/alignment": "C09;C10;C11;C13;C20",
        "In-distribution accuracy": "C01;C02;C04;C21;C22;C23;C24;C25;C26;C28;C29",
        "Predictive value beyond simple baselines": "C01;C02;C03;C11;C13;C21;C22;C23;C24;C25;C26",
        "Feature/representation contribution": "C04;C05;C06;C07;C25;C26;C31",
        "Generalization": "C14;C15;C16;C17;C18;C19",
        "Scaling/extrapolation": "C18;C19;C30", "Reliability/uncertainty/applicability/coverage": "C13;C14;C15;C27;C28;C29;C32",
    }
    files["evidence_matrix_crosswalk.csv"] = csv_text([dict(source_matrix_row=i, source_track=r["track"], source_scope=r["method_or_context"],
        source_dimension=r["dimension"], source_status_unchanged=r["status"], source_conclusion_unchanged=r["conclusion"],
        related_claim_ids=dimension_map[r["dimension"]], relation="dimension-level index; intersect claim dataset/context before applying") for i,r in enumerate(question_matrix)])

    files["final_scientific_interpretation.md"] = narrative(g,m,c,skill_global,skill_selector)
    files["executive_summary.md"] = executive()
    files["thesis_claims_and_limitations.md"] = discipline()
    files["presentation_brief.md"] = presentation()
    files["README.md"] = """# Diễn giải khoa học cho thesis

READY WITH LIMITATIONS. Đây là reporting package của Final Scientific Freeze v2,
không phải benchmark version mới. Thư mục `v1` chỉ đánh số tài liệu dẫn xuất.

Đọc [executive summary](executive_summary.md), rồi
[diễn giải theo câu hỏi](final_scientific_interpretation.md).
[Registry](estimator_fidelity_registry.md) giải thích đúng từng implementation.
[Claim matrix](evidence_to_claims.md) và CSV giữ evidence/giới hạn cho từng câu.
[Claim discipline](thesis_claims_and_limitations.md) dùng khi viết Results/Discussion.
[Presentation brief](presentation_brief.md) chỉ là nội dung để cập nhật deck sau.

Sáu bảng chính nằm ở `tables/`, mỗi bảng có Markdown và CSV. Bảng analytical
controls và method-context index bổ sung việc truy ngược, không tạo thêm experiment.
`evidence_matrix_crosswalk.csv` giữ nguyên 120 source questions; related IDs chỉ
là dimension index, không gán claim của context khác cho mọi row.

Contract v1.1 có bốn RQ. Báo cáo này dùng năm reporting questions theo yêu cầu:

| Mục reporting | Frozen contract |
| --- | --- |
| RQ1 accuracy/skill | RQ1 pipelines |
| RQ2 fidelity/clock | RQ1 implementation + RQ2 analytical |
| RQ3 feature contribution | RQ3 representation/context |
| RQ4 generalization | RQ4 |
| RQ5 reliability/scaling | Cross-cutting RQ2–RQ4, không fifth contractual RQ |
| Track B | Frozen RQ3 per-context + RQ1 comparison principles |

Không sửa Research Contract lịch sử. Các câu 10/21 trong contract mô tả snapshot
cũ. Current completion authority là full-21 report và addendum, không dùng
contract làm live status ledger. Main cohort gồm 4.515 observations, 165 groups.

`provenance_manifest.json` pin exact sources và payload. Validation report
phân biệt fresh reporting replay với prior experimental validation.
Reporting replay không phải clean-clone/QPU-retraining reproduction.
Private row-level QPU inputs không tự được phép đưa vào public payload.

Từ repo root, kiểm tra tài liệu đã lưu (không ghi file):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\
 /home/server/Documents/.venv-qonductor/bin/python -B \\
 benchmark_v1/scripts/build_thesis_interpretation.py
```

Không training, inference, runtime measurement, QCEC, queue resume,
commit/push hoặc Canva edit. Release ownership/rights và exact public-payload
verification là gates riêng, không phải yêu cầu chạy thêm benchmark.
"""

    assert len(files["executive_summary.md"].split()) <= 1000
    assert len({r["claim_id"] for r in claims}) == len(claims)
    assert all(r["category"] in CATEGORIES for r in cards)
    assert {r["case_id"] for r in decomp} == {r["case_id"] for r in general}
    assert all(int(r["overprediction_rows"]) == int(r["n"]) for r in decomp if r["case_id"] in {"family:qnn","family:random"})
    assert qcec["dd_prepared_state_proofs"] == 32 and qcec["zero_input_measured_output_matches"] == 32
    assert qcec["dd_full_unitary_fixed_layout_proofs"] == 8 and qcec["dd_full_unitary_fixed_layout_counterexamples"] == 24
    assert len([r for r in widths if r["case_id"].startswith("outer_fold:") and r["effective_tensor_width"] == "40"]) == 5
    close(schol["eligible_median"], .8200668937428495)
    close(schol["eligible_source_median"], .692070418476177)
    for r in sim_pairs:
        if r["n"]:
            close(num(r["model_mae_seconds"])-num(r["reference_mae_seconds"]),r["observed_mae_delta_seconds"])
    counts = dict(claims=len(claims), estimator_cards=len(cards), method_context_entries=len(mappings),
                  question_matrix_rows=120, reused_derived_tables=16, saved_transforms=257,
                  track_a_metric_rows=34, track_b_predictive_rows=49, track_b_contexts=7,
                  generalization_case_model_rows=42, generalization_cases=21, neural_cells=126,
                  analytical_control_rows=35, executive_summary_words=len(files["executive_summary.md"].split()))
    # Pin this new reporting implementation, never historical producing code.
    s.read(str(Path(__file__).relative_to(ROOT)))
    test = "benchmark_v1/tests/test_thesis_interpretation.py"
    if (ROOT/test).exists(): s.read(test)
    manifest = dict(report_class="derived thesis interpretation of Final Scientific Freeze v2", experimental_scope_unchanged=True,
        benchmark_authority="Final Scientific Freeze v2 + completed interpretation addendum", frozen_contract_rqs=4,
        reporting_questions=5, source_pins=s.pins, counts=counts,
        no_training_inference_measurement_QCEC=True, no_commit_push_external_edits=True,
        payload=[dict(path=str((OUT/name).relative_to(ROOT)),sha256=hashlib.sha256(content.encode()).hexdigest(),bytes=len(content.encode())) for name,content in sorted(files.items())],
        excluded_from_payload_hashes=["provenance_manifest.json","validation_report.md","validation_receipt.json"],
        source_of_truth="Frozen predictions/results/protocols > addendum exact replay tables > this derived narrative",
        public_scope="Local reporting; rights/private evidence reviewed separately before export")
    files["provenance_manifest.json"] = json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n"
    return files, counts


def narrative(g,m,c,sg,sc):
    return f"""# Diễn giải khoa học cuối cùng

READY WITH LIMITATIONS. Final Scientific Freeze v2 giữ nguyên. Cohort gồm
340 Ma–Li, 230 Qonductor và 3.945 QPack observations, trong 165 conservative
structural groups. Target là archived observed execution/service time theo
boundary của từng nguồn, không phải một pulse clock đồng nhất. QPack chiếm
87,4% số hàng nhưng chỉ có sáu cấu trúc. Input có các mức exact logical và
reconstruction-qualified; không gọi toàn bộ là 4.515 historical logical
instances đã phục hồi chính xác.

Các mục dưới là reporting questions. [Crosswalk](README.md) nối chúng với
bốn RQ của Research Contract v1.1. [Registry](estimator_fidelity_registry.md)
và [claim matrix](evidence_to_claims.md) quy định cách gọi method và giới hạn.

## RQ1 — Predictive accuracy và baseline skill

**Question.** Pipeline nào dự đoán observed service time hiệu quả, hơn reference nào?

**Evidence.** Grouped five-outer/four-inner evaluation dùng chung test IDs.
Graph/MLP dùng protocol-defined median-of-three predictions, không chọn seed
theo test. Bảng 2 giữ toàn bộ 34 metric rows cùng tail errors. Graph có MAE
{num(g['mae_seconds']):.8f} s và R² {num(g['r2_seconds']):.8f}; MLP
{num(m['mae_seconds']):.8f} s/{num(m['r2_seconds']):.8f}; selector
{num(c['mae_seconds']):.8f} s/{num(c['r2_seconds']):.8f} [C01–C04].

**Finding.** Skill = 1 − MAE_method/MAE_reference. Graph đạt
{100*sg:.7f}% so với global train median và {100*sc:.7f}% so với inner-selected
classical regressor trên cùng 4.515 hàng. Graph−median có paired 95% CI
[−0.72887, −0.04188] s. Graph−selector có CI [−0.31786, +0.45791] s;
Graph−MLP có CI [−0.32907, +0.48457] s. Hai đối chiếu sau chưa resolve superiority.

**Interpretation.** Graph khai thác được signal vượt global constant trong study
này. Lợi thế giảm trước comparator mạnh hơn. Equal-source MAE là Graph 0.89803,
MLP 0.97602 và selector 0.87898 s. Equal-group MAE đổi thứ tự Graph/MLP:
0.63324/0.58070 s. Per-source scores là slices của unified fits, không phải
các model được train riêng từng nguồn.

**Limitation.** Source median biết provenance, nên không phải equal-information
reference. Graph nhận node/calibration và có khoảng 842k parameters; MLP
khoảng 369k, chỉ dùng globals. Parent neural fit inner folds 1–3, chọn epoch
trên inner 0 và không full-outer refit. Regression dùng bốn inner validations
rồi full-outer refit. Same test rows không đồng nghĩa same information/budget.
Bootstrap conditional theo groups và fitted predictions, không đo training
variability hoặc future prediction coverage.

**Thesis takeaway.** Trình bày point advantage kèm comparator, weighting và CI.
Không biến pooled ranking thành kết luận neural superiority phổ quát.

## RQ2 — Estimator fidelity và target alignment

**Question.** Những thành phần nào được dùng nguyên bản, adapted hay proxied?
Outputs có cùng ý nghĩa với target đánh giá không?

**Evidence.** Registry và Bảng 1/index bao phủ 34 Track A metric rows, 49 Track B
rows và native/analog companions [C09–C13, C20]. QCRE gọi pinned original
timing component. Qiskit gọi native API. Original Scholten đòi hỏi kernel/QV
ensemble effective depth; verified eligibility là 0/4.515.

**Finding.** Graph/MLP là architecture/core adaptations. Qonductor giữ counter
gốc gọi là "swap", thực tế đếm CX/CZ/ECR, năm statistics và sáu regressor
families; search được giới hạn ở 307 candidates. Nominal snapshot khớp tên
backend không chứng minh calibration đúng ngày đo. QCRE unitary critical path
bỏ terminal measurement; Qiskit schedule có semantics riêng. Hyb chia cost
cho survival probability, nên biểu diễn useful-work/resource cost chứ không
tự động là thời gian thực thi một lần.

**Interpretation.** Accuracy tốt không xóa semantic mismatch. Raw, shot-scaled,
affine và log-affine service bridges là các estimators khác nhau. QCRE/Qiskit
shot-affine MAE 1.05127/1.04970 s không thắng source-median point reference
0.97436 s trên matched rows. Scholten generalized proxy có raw MAE 610.288,
affine 0.74835 và log-affine 0.72660 s trên 3.399 rows. Eligible global median
là 0.82007 s, source median 0.69207 s. Proxy cải thiện reference thứ nhất,
không reference thứ hai. Hyb finite-raw coverage 4.175 và log-cost coverage
4.367 phải được trình bày riêng; undefined/overflow không thay bằng 0.

**Limitation.** Original Scholten accuracy chưa được đánh giá. Native component
không đồng nghĩa full-paper workflow reproduction. Calibration học statistical
bridge từ labels, không đo được physical overhead. QCEC chứng minh zero-input
prepared-state agreement và measured-output agreement cho 32 pairs theo
declared mappings. Full-unitary chỉ có tám proofs và 24 counterexamples.
Đây không phải exact historical logical recovery.

**Thesis takeaway.** Nêu method identity, fidelity và clock trước khi so error.
Đổi dataset không tự làm method kém faithful; thay representation/procedure
phải được khai báo cụ thể.

## RQ3 — Feature và structural contribution

**Question.** Evidence hỗ trợ signal, reliance và incremental value của input nào?

**Evidence.** Ledger 51 raw fields, 257 saved transforms và historical controls
trong Bảng 4 [C05–C08, C31]. Raw schema gồm 44 gate counters, hai size và
năm structural summaries. Refinement dựa trên positive-sum support, không lấy
41 vị trí đầu tiên.

**Finding.** Original simulator corpus có 41 supported positions. Released
340 real rows được recompute có 40; mọi current parent fit cũng giữ 40.
Ngoài mười zero simulator fields, rccx không có support trong cohort hiện tại.
RQ4 final tensor widths là 11/22/40 tùy fit population. Một positive constant
có thể được giữ trong tensor nhưng bị variable mask zero. Author executed real
tensor còn chưa xác minh: notebook example 12, constructor default 41 và
released-corpus support 40 không phải cùng một bằng chứng.

**Interpretation.** Historical fixed-model no-edge perturbation làm MAE tăng
0.4783→5.9652 s, cho thấy reliance. Retrained coarse node-only có MAE 0.81454 s
so với true DAG 0.84483 s; delta CI [−0.0165, +0.0762] chưa resolve topology
gain. Retrained static/timing blocks có predictive signal; critical-subgraph
additions không cho stable MAE benefit. Không có complete T0–T3 ladder trên
4.515 rows. Paper component scores là external evidence, không local replication.

**Limitation.** Graph–MLP khác information và capacity. Intervention sensitivity
không phải retrained benefit. Historical 340-row/coarse controls không thay
thế native full-4.515 topology ablation. Random test-only fields và QNN range
shifts là descriptive diagnostics, chưa cô lập nguyên nhân thất bại.

**Thesis takeaway.** Bảo vệ useful compiled/timing information và verified
schema. Không claim isolated DAG topology là nguyên nhân thắng hoặc ép width 41.

## RQ4 — Generalization và robustness

**Question.** New family, backend-context và size được kiểm chứng đến đâu?

**Evidence.** Full grid gồm 21 CPU cases, 21 neural cases và 126 cells:
chín family, mười backend-context, hai QPack size holdouts [C14–C19].
Bảng 3 giữ per-case MAE, MedAE, RMSE, R², residual/tail errors, reference,
paired CI và support. Có 6.684 test assignments phủ 4.515 unique IDs.
Train/test observation IDs disjoint trong từng case.

**Finding.** QNN có 26 rows/14 groups: Graph/MLP MAE 36.272/50.764 s,
train-only median 2.658 s. Random có 30/16: MAE 145.708/188.815 s,
median 3.627 s. Cả hai models overpredict mọi test row, R² âm rất lớn.
Đây không phải một outlier. Kyoto có 148/148 raw và transformed MLP inputs
trùng fit vectors nhưng không trùng observation IDs. Osaka có 149/192 collisions
và 43 new-group rows. Quito có 1.051/1.051 collisions, gồm 1.009 QPack
và 42 Qonductor observations trong bảy groups. Complete Graph inputs không
exactly collide do nominal node context khác.

**Interpretation.** Family holdouts có structural embargo và 14–27 groups,
khác backend transfer cố ý cho repeated structures. MLP có MAE thấp hơn
Graph ở 9/10 backend cases không chứng minh đã học hardware adaptation.
QPack size probes có 1.321 rows/3 groups và 498/2. Case sau có Graph/MLP
R² −4.685/−13.060; cả hai underpredict mọi row. QNN giữ 40 globals;
Random giữ 22 và masks 18 test-only fields. Extreme scaled counts mô tả
support shift, không causal attribution.

**Limitation.** Chỉ Kyoto/Osaka có ít nhất mười groups trong backend cases để
tính CI theo frozen rule. Source, workload, clocks và shots confound generation.
Không pool cross-case errors. Không có systematic real-QPU shot-held-out hoặc
source-held-out neural study. Source-stratified scores/reweighting không phải
source-OOD evaluation. Twolocalrandom vẫn được ghi unauthorized-at-execution,
later independently admitted; không viết lại authorization history.

**Thesis takeaway.** Full grid trả lời bounded transfer questions và phát hiện
catastrophic failures. Không bảo đảm universal generalization hoặc isolated
hardware-generation effect.

## RQ5 — Reliability, scaling và operational limitations

Mục này là reporting dimension bổ sung, không phải RQ thứ năm của frozen contract.

**Question.** Runtime predictions đã đủ an toàn cho scheduler, quality và scale chưa?

**Evidence.** Saved coverage/tails, conditional CI, MPS quality gates,
TN/Pasqal và width probes [C18–C20, C27–C30, C32]. Chưa có prospective
calibrated prediction-interval study.

**Finding.** Runtime completion không bảo đảm required state quality. Fixed-MPS
có 147/150 quality passes và ba failures. Joint selectors fail fidelity ở
64/140 assessed choices; mười assigned hashes chưa assessed. Trong 140 hashes
đã đo, 138 có passing rung. Hyb finite-cost và nominal-throughput missingness
có coverage riêng. Width-dependent API zero-work cost không phải isolated
physical launch latency.

**Interpretation.** Operational reporting phải giữ riêng runtime, quality,
failure và coverage. Negative safety finding là kết quả đã được evaluated,
không INCONCLUSIVE chỉ vì model thất bại. Chưa có universal entanglement,
shot hoặc hardware scaling law.

**Limitation.** Conditional error bootstrap không phải 95% future-job guarantee.
TN internals opaque; Pasqal coefficients chưa identifiable. Limited width,
bond và structural support giới hạn extrapolation.

**Thesis takeaway.** Benchmark chỉ ra nơi estimator có thể fail dù average
error tốt. Chưa thiết kế universal production estimator hoặc chứng minh safe scheduling.

## Track B — Simulator runtime prediction theo context

**Question.** Predictor nào có giá trị trong cùng engine/task/context?

**Evidence.** Bảng 5 giữ 49 predictive metric rows trong bảy contexts và native
TN/analog diagnostics, frozen shared-hash/rung protocols cùng paired uncertainty
[C21–C30].

**Finding.** Dense FP32/FP64 warm sample: RF MAE 0.821/0.813 ms so với
median 1.749/1.753 ms và Graph 1.716/1.730 ms. Fixed-chi16 FP64 warm
get_state: Graph 0.29609 s so với Ridge 1.27567 s; paired CI
[−1.88582, −0.33726] s. Aer FakeSherbrooke Opt1: source/hybrid/transpiled
GNN MAE 0.44966/0.40636/0.30076 s; compiled XGBoost 0.29704 s.
Source→transpiled gain được resolve; hybrid→transpiled và GNN→XGBoost
chưa resolve. Family-conditioned/agnostic MAE 0.31163/0.30752 s, family
holdout 0.31101/0.30879 s không có resolved runtime gain.
Maestro-style QCSim component có MAE 0.10370 ms so với Ridge 0.13361 ms
trên 143 supported hashes, saved paired CI âm. Native TN có 603/612 plan
rows với median EST/warm 3.734 cho core và 4.132 cho frontier. Pasqal có
18 cells/17 quality passes nhưng
chưa có identified/promoted predictor.

**Interpretation.** Graph có useful signal cho fixed MPS nhưng không thắng
dense RF theo MAE. Compiled views giúp Aer mà chưa chứng minh cần GNN.
Component calibration có local value trong QCSim. Joint quality selection
là endpoint riêng với runtime accuracy.

**Limitation.** Sample, state, Aer execution, engine-reported time, scalar
contraction và analog TDVP khác clocks; cùng đơn vị seconds không đủ để pool.
Ladder có 840 rungs từ 140 hashes, không 840 independent circuits.
Aer chỉ q2–9, một backend/level; không full Azizov replication. Closed
Composer/engine selection chưa reproduced. Không có universal paper winner.

**Thesis takeaway.** Hai domain dùng chung nguyên tắc benchmark nhưng không
một leaderboard. Report per-context errors, matched controls, quality,
coverage và fidelity.

## Kết luận thesis

Benchmark cung cấp evidence có thể truy vết về năng lực và giới hạn của
runtime-estimation pipelines trên qualified multi-source QPU data và local
simulator contexts. Lợi thế dự đoán phụ thuộc comparator, representation,
nguồn và clock. Strong simple baselines, negative OOD results và quality-safety
tests làm rõ giới hạn ứng dụng. Không cần experiment mới để bảo vệ các kết
luận có điều kiện này. Các open questions ở Bảng 6 cần amendment riêng nếu
muốn đưa ra stronger claims.
"""


def executive():
    return """# Tóm tắt khoa học

**Verdict: READY WITH LIMITATIONS.** Benchmark hoàn tất trong Final Scientific
Freeze v2. Không thêm thí nghiệm, thay dữ liệu hoặc sửa kết quả. Các findings
dưới đây được xếp theo ý nghĩa khoa học, không theo độ phức tạp của model.

1. **FACT [C01–C04].** Trên 4.515 observations, Graph/MLP/selector có MAE
0.83413/0.89342/0.88666 s. Graph cải thiện 26,7% so với global train median,
nhưng 5,9% so với selector. Paired CI của Graph−selector và Graph−MLP chứa 0.
**INTERPRETATION:** có signal vượt constant reference, chưa resolve superiority
trước stronger comparators.

2. **FACT [C03].** Equal-source MAE selector 0.87898 s thấp hơn Graph
0.89803 s. Equal-group weighting đảo thứ tự Graph/MLP. QPack chiếm 87,4%
rows nhưng chỉ sáu structures. **INTERPRETATION:** ranking phụ thuộc estimand
và source distribution.

3. **FACT [C14–C18].** Full RQ4 có 21 CPU/21 neural cases, 126 cells.
QNN/Random là held-out families, không baseline algorithms. Graph/MLP MAE
36.272/50.764 s và 145.708/188.815 s; cả hai overpredict mọi row, kém
train-only median nhiều lần. Kyoto 148/148 MLP inputs trùng fit vectors,
không trùng IDs. **INTERPRETATION:** catastrophic OOD failures và context-blind
collisions giới hạn practical/hardware-transfer claims.

4. **FACT [C07,C08].** Ma–Li có 51 raw positions, support-refined thành 41
trong original simulator corpus và 40 ở mọi parent fit. Có 257 transforms
đã kiểm tra; RQ4 widths tùy fit population. **OPEN QUESTION:** notebook 12,
default 41 và released support 40 chưa xác định author executed real tensor.
**PROJECT DECISION:** giữ adaptation, không hard-code 41 hoặc retrain chỉ để
khớp dimension label.

5. **FACT [C05,C06,C31].** Fixed-model perturbations cho thấy reliance lên
edges/wire identities. Retrained coarse controls chưa resolve topology gain;
static/timing blocks có signal. **INTERPRETATION:** reliance khác retrained
increment. Graph−MLP không isolated topology ablation.

6. **FACT [C09–C13].** QCRE/Qiskit duration khác observed service. Hyb cost
điều chỉnh survival, không literal execution time. Original Scholten thiếu
verified kernel/QV inputs; current route là generalized proxy. Calibration
là predictor mới. **INTERPRETATION:** low error sau calibration không chứng
minh native clock phù hợp hoặc physical overhead đã được đo.

7. **FACT [C21–C29].** Simulator findings theo context: dense RF có MAE
thấp hơn Graph; fixed-MPS Graph tốt hơn Ridge; Aer transpiled GNN cải thiện
source GNN nhưng chưa resolve hơn XGBoost. Maestro component có local gain,
không full engine-selector reproduction. **INTERPRETATION:** không có universal
Track B winner hoặc clock chung với Track A.

8. **FACT [C27].** Joint MPS selectors fail fidelity 64/140 assessed choices,
mười unassessed, dù 138/140 hashes có passing rung. **INTERPRETATION:** runtime
accuracy hoặc quality probability score không đồng nghĩa safe configuration
selection.

9. **FACT [C20].** QCEC có 32/32 zero-input prepared-state proofs và
measured-output matches. Full-unitary có 8/32 proofs, 24 counterexamples
theo declared maps. Có 230 observations liên kết 32 pairs.
**INTERPRETATION:** hỗ trợ functional recipe qualification, không exact historical
recovery hoặc 230 independent proofs.

**Đóng góp có thể bảo vệ:** qualified multi-source data với provenance;
shared-test pipeline comparisons cùng strong simple controls; clock/fidelity/
coverage discipline; negative OOD và quality-safety findings. Không claim
thuật toán estimator mới hoặc paper-exact replication. Bootstrap hiện có
conditional theo groups/fits, không prospective prediction guarantee.

**PROJECT DECISION:** khóa experimental scope. **OPEN QUESTIONS:** individual
feature/topology attribution, unseen shots, isolated hardware-generation effects,
prospective intervals và broader engine/width transfer. Không cần experiment
bổ sung để bảo vệ bounded claims hiện tại. Ownership, rights, metadata và
exact-public-payload verification vẫn là release gates riêng.
"""


def discipline():
    return """# Claims, đóng góp và giới hạn

## Phân loại phát biểu

FACT là số liệu hoặc sự kiện được source-linked. INTERPRETATION là suy luận
có phạm vi. PROJECT DECISION là lựa chọn đã duyệt, không phải finding thực nghiệm.
OPEN QUESTION là điều evidence chưa trả lời. Claim registry dùng bốn strength
labels; negative outcome không mặc định INCONCLUSIVE.

| Loại | Ví dụ hợp lệ |
| --- | --- |
| FACT | C14/C15: mọi QNN/Random test row bị overpredict; không lọc outlier |
| INTERPRETATION | Absolute errors không hữu ích dù Graph có MAE thấp hơn MLP |
| PROJECT DECISION | Giữ frozen results và nhãn adaptations; không thêm experiment |
| OPEN QUESTION | Topology riêng lẻ có retrained gain trên full 4.515 không? |

## Đóng góp thesis có thể bảo vệ

- Qualified multi-source panel gồm 340 Ma–Li, 230 Qonductor và 3.945 QPack
  observations trong 165 groups, có filter/reconstruction/provenance và
  missingness policies truy ngược được. Không gọi toàn bộ là exact historical
  logical circuits.
- Shared-test pipeline benchmark cho learned, analytical, calibrated và simple
  controls, cùng pooled/source/group weighting và conditional paired uncertainty.
- Local simulator comparisons theo clock/configuration, có quality gates và
  failures, không ghép các APIs thành leaderboard.
- Full frozen family/backend/size grid, phát hiện systematic overprediction và
  representation collisions. Negative findings là đóng góp, không che giấu.
- Feature-fidelity trace từ raw 51 đến supported 41/40 và fit-dependent RQ4
  widths; tách source audit khỏi author executed tensor chưa được xác minh.

Đây là empirical/methodological benchmark contributions. Không tuyên bố
thuật toán runtime estimator mới, phát hiện internal NVIDIA formula hoặc giải
quyết universal hardware generalization. QCRE/Tremba là một source family,
không hai kết quả độc lập.

## Threats to validity

**Construct validity.** Provider/source clocks có service components khác nhau.
Raw schedule, cost và throughput proxy không phải service identity. Nominal
snapshot không chứng minh historical calibration/routing. Shot multiplication
có assumptions chưa được kiểm chứng bằng shot-held-out study.

**Input validity.** Qonductor recipe qualification dựa trên 32 pairs và
zero-input mappings, không historical recovery. QPack representative angles
không phải historical optimizer angles. Missing features/costs/failures không
được thay bằng 0 để tăng coverage. Graph nominal wire-index calibration không
phục hồi physical placement.

**Internal validity.** Graph/MLP khác node/context/capacity. Regression dùng
compiled statistics, search và refit policy khác. Same outer test không bảo đảm
causal attribution. Historical controls không thay full-4.515 controls; paper
scores không phải local replication. Upstream global normalization và local
fit-only per-feature moments khác nhau.

**Statistical validity.** QPack có 3.945 rows nhưng sáu structures. RQ4 cases
overlap; không pool 6.684 assignments thành IID data hoặc average R².
Bootstrap conditional theo fits, không training variability, simultaneous
multiple-comparison guarantee hoặc prospective coverage. Không CI dưới mười
groups. Median-of-three tính per row trước scoring; không average seed metrics
hay chọn seed tốt nhất theo test.

**External validity.** Local hardware/software/engine contexts, small-width
panels, Aer một backend/level, low-support size probes, unseen shots và causal
hardware-generation effects hạn chế extrapolation. Backend-context cases có thể
chủ yếu là known structures. Source slices không phải source holdout.

**Provenance.** Twolocalrandom unauthorized-at-execution, later audited/admitted
vẫn giữ lịch sử. Saved-evidence replay không phải clean-checkout full-QPU refit.
Reduced public payload thiếu private QPU row materials; local hash/path không
đồng nghĩa distribution permission.

## Các câu không được dùng

- “Graph/GNN thắng mọi baseline hoặc mọi backend.” C02/C04 uncertainty và C03
  weighting reversal không hỗ trợ.
- “Topology là nguyên nhân thắng.” Whole-pipeline contrast không isolated control.
- “Tác giả dùng 40 global features cho real model.” Executed real tensor chưa verified.
- “51 actual current inputs” hoặc “fixed 41.” Raw/support/tensor/varying widths khác nhau.
- “Original Scholten đã được benchmark faithful.” Current route là generalized proxy.
- “QCRE/Qiskit raw seconds là observed execution seconds.” Clock mismatch.
- “Affine intercept đo được launch overhead.” Statistical bridge không physical decomposition.
- “QNN/Random là baseline predictors.” Đó là held-out circuit families.
- “Kyoto MLP học hardware adaptation trên unseen circuits.” Inputs 148/148 collide.
- “Có thể bỏ quality failures vì runtime predictor đúng.” Safety là endpoint bắt buộc giữ.
- “230 independent QCEC proofs” hoặc “exact historical recovery.” Chỉ 32 pairs/zero-input scope.
- “21-case completion chứng minh universal generalization.” Support/confounding vẫn còn.

## Kết luận có thể đưa vào thesis

Trong benchmark đã khóa, runtime-estimation pipelines khai thác được signal
hữu ích ở một số contexts nhưng hiệu quả phụ thuộc baseline, nguồn,
representation và timing boundary. Graph có pooled advantage so với global
median, chưa có resolved superiority trước stronger classical/global controls.
Compiled information hỗ trợ local noisy-Aer prediction, còn quality-constrained
MPS selection và family-OOD neural prediction bộc lộ failures đáng kể.
Việc giữ reconstruction qualification, source/group support, coverage và
negative results cho phép đưa ra các kết luận giới hạn nhưng có thể bảo vệ.

## Việc tiếp theo

Viết Results/Discussion từ C01–C32 và sáu bảng. Mở đầu bằng data qualification
và timing semantics, sau đó accuracy/controls, features, transfer và safety.
Đưa open questions vào Future Work, không mở experiment queue. Metadata,
ownership và exact release payload là gates riêng. Không đổi frozen contract
để khớp reporting headings.
"""


def presentation():
    return """# Presentation brief

Nội dung cho agent làm slide sau khi người dùng duyệt. Không sửa Canva.
Exact source hashes ở provenance manifest. Frozen source figures và pins ở
[slide index](/home/server/Documents/quantum-runtime-estimator-replications/artifacts/final_research/slides.md)
và [evidence index](/home/server/Documents/quantum-runtime-estimator-replications/artifacts/final_research/evidence.json).
Brief không tạo figures mới hoặc khẳng định đã inspect live Canva.

| Slide/ý chính | Nội dung | Table/evidence | Claim |
| --- | --- | --- | --- |
| Mục tiêu và data | 4.515 observations/165 groups; 340/230/3.945; QPack 87,4% rows/sáu structures | Generalization table, frozen QPU report | C03,C20 |
| Qualification | Exact/reconstructed tiers; 32 pairs không 230 proofs; không historical recovery | QCEC results/report | C20 |
| Method identity | Original target, output clock, local adaptation; QCRE/Tremba cùng family | Estimator identity | C09–C13 |
| Main QPU comparison | Graph/MLP/selector MAE 0.83413/0.89342/0.88666 s, R²; same rows | Track A table | C01,C02,C04 |
| Strong controls | 26,7% vs median, 5,9% vs selector kèm CI; equal-source reversal | Track A, paired tables | C01–C04 |
| Analytical | Raw/shot/affine riêng; eligible controls; proxy 3.399, original 0 eligible | Analytical controls | C10–C13 |
| Features | Raw 51/sim 41/parent 40; real tensor unresolved; no isolated topology gain | Feature table | C05–C08,C31 |
| Simulator | Per-context panels: dense RF, fixed-MPS Graph, Aer compiled, Maestro component | Track B table | C21–C26,C28,C29 |
| Safety | 64/140 quality failures, 10 unassessed; 138/140 attainable quality | MPS safety table/figure | C27 |
| Full RQ4 | Nine family, ten backend, two size cases; 126 cells; n/groups riêng | Generalization table | C17,C18 |
| Failures | QNN/Random all-row overprediction, absolute errors và CPU reference | Decomposition table | C14,C15 |
| Backend meaning | Kyoto 148/148 MLP collisions/no ID leak; Osaka 149/192; Quito 1.051/1.051 | Frozen RQ4/support tables | C16,C17 |
| Kết luận | Bounded findings, clock/source/support limitations | Summary/open questions | C01–C32 |

## Những thông điệp cũ cần thay nếu còn trong deck

- “Current RQ4 10/21 hoặc 12/21”: thay bằng 21/21; giữ old reports là snapshots.
- “51 actual inputs” hoặc “fixed 41”: dùng raw 51/post-mask 40 parent và fit-dependent RQ4.
- “Qonductor original/full search”: nêu pinned counters cùng bounded-search adaptation.
- “Original Scholten/effective depth verified”: generalized proxy, original NOT EVALUATED.
- “Graph winner/topology gain/hardware-generation transfer”: thêm comparator, CI, weighting/support.
- “QNN/Random rare outliers”: systematic all-row failures và negative R².
- “MPS runtime accurate nên safe”: giữ 64 failures/140 và mười unassessed.
- “QCEC exact recovery/230 proofs”: 32 pairs, zero input, tám/24 full-unitary outcomes.
- “Completed experiments nên public QPU refit self-contained”: giữ private-material/release limitations.

Có thể dùng năm reporting questions để tổ chức phần thảo luận nhưng giữ
crosswalk đến contract RQ1–RQ4; không gọi RQ5 là frozen-contract amendment.
"""


def emit_patch(files):
    print("*** Begin Patch")
    for name, content in sorted(files.items()):
        path = OUT / name
        if path.exists():
            assert path.read_text() == content, "refuse overwrite divergent reporting: " + str(path)
            continue
        print("*** Add File: " + str(path))
        for line in content.splitlines(): print("+" + line)
    print("*** End Patch")


def validate():
    files, counts = render()
    for name, content in files.items():
        assert (OUT/name).read_bytes() == content.encode(), "report_replay:" + name
    manifest = json.loads(files["provenance_manifest.json"])
    for path, digest in manifest["source_pins"].items():
        assert sha(path) == digest, "source_hash:" + path
    receipt_path = OUT / "validation_receipt.json"
    receipt_checked = False
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt["reporting_validation"]["manifest_sha256"] == sha(OUT / "provenance_manifest.json")
        assert receipt["validation_report_sha256"] == sha(OUT / "validation_report.md")
        assert receipt["reporting_validation"]["counts"] == counts
        assert receipt["frozen_authority_validation"]["status"] == "PASS"
        assert receipt["focused_tests"]["passed"] == 33
        receipt_checked = True
    return dict(status="PASS", scientific_verdict="READY WITH LIMITATIONS", counts=counts,
        reporting_files_replayed=len(files), source_pins_checked=len(manifest["source_pins"]),
        final_reporting_receipt_checked=receipt_checked,
        manifest_sha256=sha(OUT/"provenance_manifest.json"), scientific_freeze_v2_preserved=True,
        no_experiments_or_inference=True, no_commit_push_Canva=True,
        scope="derived reporting replay, not clean-checkout or independent QPU retraining")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-patch",action="store_true")
    parser.add_argument("--file",help="Bound patch emission to one reporting file")
    args=parser.parse_args()
    if args.emit_patch:
        files,_=render()
        emit_patch({args.file:files[args.file]} if args.file else files)
    else:
        print(json.dumps(validate(),ensure_ascii=False,indent=2,sort_keys=True))
