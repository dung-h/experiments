#!/usr/bin/env python3
"""Resolve saved provenance declarations and render the approved five-RQ closure.

Only stdlib file reads, hashing and read-only Git queries are used. No model,
circuit parser, simulator or producing runner is imported or executed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/final_rq_closure_v2"
PRIOR = ROOT / "artifacts/final_exhaustive_validation/v1"
PINS: dict[str, str] = {}
VERDICT = "SCIENTIFICALLY SUFFICIENT WITH MATERIAL LIMITATIONS"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def path_for(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def pin(value):
    p = path_for(value).resolve()
    value = digest(p)
    if str(p) in PINS:
        assert PINS[str(p)] == value, f"source changed during read: {p}"
    PINS[str(p)] = value
    return p


def text(value):
    return pin(value).read_text(encoding="utf-8")


def rows(value):
    with pin(value).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def document(value):
    return json.loads(text(value))


def rel(path):
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def links(values):
    return "; ".join(f"[{Path(v).name}]({v if Path(v).is_absolute() else '../../'+v})" for v in values)


def table(values, columns):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "| " + " | ".join(columns) + " |\n|" + "|".join("---" for _ in columns) + "|\n" + "\n".join(
        "| " + " | ".join(cell(r.get(k, "")) for k in columns) + " |" for r in values) + "\n"


def csv_content(values):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(values)
    return stream.getvalue()


def code_dict(source, name):
    tree = ast.parse(text(source))
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            candidates.append((ast.literal_eval(node.value), node.lineno))
    assert len(candidates) == 1, (source, name, len(candidates))
    return candidates[0]


def provenance():
    inventory = rows("artifacts/final_exhaustive_validation/v1/final_artifact_manifest.csv")
    mismatches = [r for r in inventory if r["validation_status"] == "HASH MISMATCH"]
    pathless = [r for r in inventory if r["validation_status"] == "UNVERIFIED"]
    assert len(mismatches) == 6 and len(pathless) == 182
    full_path_refs = defaultdict(list)
    for r in inventory:
        if r["validation_status"] == "PASS" and r["artifact_type"] != "named_hash_without_path":
            full_path_refs[(str(path_for(r["artifact_path"]).resolve()), r["hash"])].append(r["provenance_source"])

    mapping_source = "benchmark_v1/scripts/aggregate_real_qpu_generalization.py"
    mapping, mapping_line = code_dict(mapping_source, "neural_pin_paths")
    mapping_refs = {k: f"{mapping_source}:{mapping_line}; benchmark_v1/scripts/run_generalization_neural.py:263" for k in mapping}
    qualified = {
        "contract": "benchmark_v1/protocol/qualified_logical_benchmark.json",
        "manifest": "artifacts/real_qpu/qualified_logical_panel/manifest.json",
        "runner": "benchmark_v1/scripts/run_qualified_logical_benchmark.py",
    }
    mapping.update(qualified)
    mapping_refs.update({k: "benchmark_v1/scripts/run_qualified_logical_benchmark.py:232 (load_context pins; CONTRACT/OUT constants)" for k in qualified})
    resume = {
        "executor_sha256": "benchmark_v1/scripts/resume_qualified_analytical.py",
        "source_manifest_sha256": "artifacts/real_qpu/strict_logical_panel/analytical/run_manifest.json",
        "source_raw_sha256": "artifacts/real_qpu/strict_logical_panel/analytical/raw_attempts.csv",
        "source_preflight_sha256": "artifacts/real_qpu/strict_logical_panel/analytical/adapter_preflight.json",
        "qcre_sha256": str(ROOT.parent / "qcre/estimate.py"),
        "analytical_manifest": "artifacts/real_qpu/qualified_logical_panel/analytical/manifest.json",
        "summary_manifest": "artifacts/real_qpu/qualified_logical_panel/summary/manifest.json",
    }
    mapping.update(resume)
    mapping_refs.update({k: "benchmark_v1/scripts/resume_qualified_analytical.py:125-128,252-253,276-277; benchmark_v1/scripts/run_strict_logical_analytical.py:36-38" for k in resume})
    supporting = [mapping_source, "benchmark_v1/scripts/run_generalization_neural.py",
                  "benchmark_v1/scripts/run_qualified_logical_benchmark.py", "benchmark_v1/scripts/resume_qualified_analytical.py",
                  "benchmark_v1/scripts/run_strict_logical_analytical.py"]
    for p in supporting:
        pin(p)
    git_root = ROOT.parent / "qcre"
    git_revision = subprocess.check_output(["git", "-C", str(git_root), "rev-parse", "HEAD"], text=True).strip()
    git_blob = subprocess.check_output(["git", "-C", str(git_root), "show", f"{git_revision}:estimate.py"])
    assert hashlib.sha256(git_blob).hexdigest() == digest(pin(git_root / "estimate.py"))

    resolved = []
    for n, r in enumerate(pathless, 1):
        symbol = r["artifact_path"]
        declaration = r["provenance_source"].split("::")[0]
        pin(declaration)
        if symbol == "qcre_revision":
            candidate = git_root
            actual = git_revision
            hash_type = "GIT_COMMIT_SHA1"
            proof = "benchmark_v1/scripts/resume_qualified_analytical.py:85-93,128; benchmark_v1/scripts/run_strict_logical_analytical.py:36-38"
            chain = "QCRE_ROOT is ROOT.parent/qcre; source run_manifest.qcre_revision; Git HEAD; estimate.py blob SHA256 matches qcre_sha256"
            classification = "RESOLVED TO AUTHORITATIVE FILE"
        else:
            assert symbol in mapping, f"no contextual mapping for {symbol}"
            candidate = pin(mapping[symbol])
            actual = digest(candidate)
            hash_type = "SHA256"
            proof = mapping_refs[symbol]
            references = full_path_refs.get((str(candidate), r["hash"]), [])
            chain = f"{declaration}::{r['provenance_source'].split('::')[-1]}[{symbol}] -> explicit producer/consumer mapping -> {rel(candidate)}"
            if references:
                chain += " -> independently path-pinned by " + "; ".join(references[:3])
            classification = "RESOLVED TO IMMUTABLE GENERATED PAYLOAD" if "artifacts/" in rel(candidate) else "RESOLVED TO AUTHORITATIVE FILE"
        matches = actual == r["hash"]
        assert matches, f"resolved candidate digest differs: {symbol}: {declaration}"
        impacts = "code/input lineage"
        if "split" in symbol or "support" in symbol:
            impacts = "train/test identity and structural support"
        elif "manifest" in symbol or symbol == "contract":
            impacts = "dataset/protocol/metric provenance"
        elif "feature" in symbol or "mali_helpers" in symbol or "batch" in symbol:
            impacts = "feature transformations and frozen fitted pipeline"
        elif "raw" in symbol:
            impacts = "saved analytical predictions and target join"
        resolved.append(dict(declaration_id=f"P{n:03d}", historical_declaration=declaration,
                             symbolic_field=symbol, recorded_hash=r["hash"], hash_type=hash_type,
                             candidate_path=rel(candidate), current_hash=actual, hash_matches=True,
                             context_mapping=proof, candidate_count_after_context=1,
                             provenance_chain=chain, classification=classification,
                             scientific_materiality=impacts, unresolved_primary_claim_risk="NONE_AFTER_VERIFICATION"))

    readme_producers = {
        "artifacts/am1_feasibility/v1/provenance_manifest.json": "benchmark_v1/scripts/build_am1_feasibility.py:1052,1063",
        "artifacts/am1_overhead_pilot/v1/provenance_manifest.json": "benchmark_v1/scripts/summarize_am1_overhead_pilot.py:175-176; benchmark_v1/scripts/validate_am1_overhead_pilot.py:29",
        "artifacts/am1_phase2/v1/provenance_manifest.json": "benchmark_v1/scripts/analyze_am1_phase2.py:427-430,464",
        "artifacts/am1_postpilot_audit/v1/provenance_manifest.json": "benchmark_v1/scripts/build_am1_postpilot_audit.py:367",
        "artifacts/research_question_closure/v1/provenance_manifest.json": "benchmark_v1/scripts/build_research_question_closure.py:835,845",
        "artifacts/real_qpu/qualified_logical_panel/summary/final_validation.json": "benchmark_v1/scripts/finalize_qualified_benchmark.py:484,499-505",
    }
    historical = []
    root_readme = pin("README.md")
    root_hash = digest(root_readme)
    pin("artifacts/final_exhaustive_validation/v1/build_final_exhaustive_validation.py")
    for n, r in enumerate(mismatches, 1):
        declaration = r["provenance_source"].split("::")[0]
        mp = pin(declaration)
        intended = pin(mp.parent / "README.md")
        expected = r["hash"]
        assert digest(intended) == expected
        evidence = readme_producers[declaration]
        for part in evidence.split(";"):
            pin(part.strip().split(":")[0])
        historical.append(dict(mismatch_id=f"H{n:02d}", historical_manifest=declaration,
                               recorded_hash=expected, current_file_path=rel(intended),
                               current_hash=digest(intended), current_matches_recorded=True,
                               previous_checker_current_path="README.md", previous_checker_current_hash=root_hash,
                               expected_mutability="frozen generated package payload; root live README is a different file",
                               pin_intent="historical package content, resolved relative to manifest directory",
                               immutable_historical_content=rel(intended), provenance_chain=evidence,
                               classification="MANIFEST DESIGN ISSUE",
                               classification_detail="validator path-resolution false positive; original package-local convention is demonstrated by producing code",
                               documentation_drift_found=False, scientific_evidence_affected=False,
                               resolution="PASS_AT_CORRECT_NAMESPACE; no hash or historical artifact changed"))
    return historical, resolved


def scientific_records():
    paths = {
        "qpu": "artifacts/benchmark_deep_dive/tables/qpu_metrics_v1.csv",
        "pairs": "artifacts/real_qpu/qualified_logical_panel/summary/shared_row_comparisons.csv",
        "sources": "artifacts/benchmark_deep_dive/tables/qpu_source_metrics_v1.csv",
        "controls": "artifacts/benchmark_deep_dive/tables/analytical_controls_v1.csv",
        "sim": "artifacts/benchmark_deep_dive/tables/simulator_metrics_v1.csv",
        "sim_pairs": "artifacts/benchmark_deep_dive/tables/simulator_paired_v1.csv",
        "mps": "artifacts/benchmark_deep_dive/tables/mps_selection_safety_v1.csv",
        "tn": "artifacts/benchmark_deep_dive/tables/tn_same_plan_slices_v1.csv",
        "pasqal": "artifacts/benchmark_deep_dive/tables/pasqal_stage_quality_v1.csv",
        "grid": "artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/method_metrics.csv",
        "grid_report": "artifacts/final_research/rq4_report.md",
        "fidelity": "artifacts/thesis_interpretation/v1/estimator_fidelity_registry.md",
        "feature": "artifacts/benchmark_interpretation/v1/mali_feature_schema_audit.md",
        "recovery": "artifacts/am1_preprocessing_recovery/v1/root_cause_analysis.md",
        "p4": "artifacts/am1_phase4/v1/analysis.json",
        "p5": "artifacts/am1_phase5_dense_b2/v1/analysis.json",
        "densefull": "artifacts/am1_phase5_dense_b2/v1/full_context_accuracy.csv",
        "qcec": "artifacts/real_qpu/provenance_equivalence/report.md",
    }
    loaded = {k: rows(p) for k, p in paths.items() if p.endswith(".csv")}
    p4, p5 = document(paths["p4"]), document(paths["p5"])
    for p in paths.values():
        pin(p)
    qpu = {r["method_id"]: r for r in loaded["qpu"]}
    main_methods = ["mali_logical_graph", "mali_global_mlp", "qonductor_budgeted_selector", "qonductor_polynomial_regression", "outer_train_median", "outer_train_source_median"]
    assert all(int(qpu[m]["n"]) == 4515 for m in main_methods)
    claims = []
    def claim(cid, rq, track, importance, support, statement, boundary, sources, values=None):
        refs = [paths.get(s, s) for s in sources]
        for p in refs:
            pin(p)
        claims.append(dict(claim_id=cid, rq=rq, track=track, role=importance,
                           support=support, claim=statement, scope_and_limitations=boundary,
                           source_artifacts_json=json.dumps(refs, ensure_ascii=False),
                           source_hashes_json=json.dumps({p: PINS[str(path_for(p).resolve())] for p in refs}, sort_keys=True),
                           saved_metric_records_json=json.dumps(values or [], ensure_ascii=False, sort_keys=True)))
    claim("C01", "RQ1", "A", "CENTRAL", "SUPPORTED WITH QUALIFICATION",
          "Các pipeline đã khai báo được so sánh trên cùng 4.515 held-out observations; Graph có pooled point MAE thấp hơn MLP/selector.",
          "Point-estimate ordering không xác lập superiority; source/group weighting đổi ordering; inputs/capacity/refit khác nhau.",
          ["qpu", "pairs", "sources"], [qpu[m] for m in main_methods])
    claim("C02", "RQ1/RQ3", "A", "CENTRAL", "STRONGLY SUPPORTED",
          "Pooled, source-balanced và equal-group estimands cho các góc nhìn khác nhau; QPack chiếm phần lớn observations nhưng chỉ sáu structural groups.",
          "Không suy luận 4.515 independent circuit structures hoặc universal ranking.", ["qpu", "sources", "docs/research_conclusions.md"])
    claim("C03", "RQ1/RQ3/RQ5", "A", "CENTRAL", "SUPPORTED WITH QUALIFICATION",
          "Outer-train calibration có thể nối schedule/cost proxies với observed service labels; raw và calibrated predictors có ý nghĩa khác nhau.",
          "Không gọi intercept là physically measured overhead; comparisons chỉ trên eligible shared rows; original Scholten chưa evaluable.", ["controls", "fidelity"] , loaded["controls"])
    claim("C04", "RQ1", "B", "CENTRAL", "SUPPORTED WITH QUALIFICATION",
          "Accuracy có thể benchmark trong từng frozen simulator context; Aer compiled view hữu ích, Dense RF và fixed-MPS Graph có local signal khác nhau.",
          "Không có cross-engine leaderboard; paper-exact reproduction và phổ hardware/width rộng không được chứng minh.", ["sim", "sim_pairs"], loaded["sim"])
    claim("C05", "RQ3/RQ5", "B", "CENTRAL", "STRONGLY SUPPORTED",
          "Joint MPS selectors vi phạm fidelity gate trên 64/140 assessed choices dù phần lớn hashes có passing rung.",
          "Đây là deployed frozen threshold trên assessed panel; 10/150 hashes unassessed; classification accuracy không bảo đảm safe quality selection.", ["mps", "docs/research_conclusions.md"], loaded["mps"])
    claim("C06", "RQ1/RQ5", "B", "SECONDARY", "SUPPORTED WITH QUALIFICATION",
          "Maestro component có accuracy trên CPU QCSim context; cuTensorNet EST được đánh giá cùng plan/clock; Pasqal chỉ là analog quality/time companion.",
          "Không claim closed Composer selection reproduction, công thức nội bộ RUNTIME_EST, hoặc Pasqal predictive coefficients đã identifiable.", ["sim", "tn", "pasqal", "fidelity"])
    selected_grid = [r for r in loaded["grid"] if r.get("scope") == "all_test" and r.get("case_id") in {"family:qnn", "family:random"} and r.get("method_id") in {"mali_logical_graph", "mali_global_mlp", "outer_train_median"}]
    claim("C07", "RQ2/RQ3", "A", "CENTRAL", "STRONGLY SUPPORTED",
          "Full 21-case grid cho thấy severe QNN/Random overprediction ở cả Graph/MLP, đồng thời backend transfer có known-input collisions.",
          "Negative results trả lời evaluated generalization; không chứng minh universal transfer hay cô lập generation/topology.", ["grid", "grid_report"], selected_grid)
    claim("C08", "RQ2/RQ3", "A", "SECONDARY", "STRONGLY SUPPORTED",
          "Kyoto 148/148 MLP fit-input collisions không đi cùng observation-ID overlap; Osaka mixed known/new và Quito repeated-source structures khác nhau.",
          "Representation collision/known-circuit context transfer; không gọi MLP learned hardware adaptation.", ["grid_report", "feature"])
    claim("C09", "RQ2", "B", "CENTRAL", "SUPPORTED WITH QUALIFICATION",
          "Hash OOF và family-held-out MPS/historical width evidence chỉ hỗ trợ các transfer đã đo; cross-engine/hardware portability còn unevaluated.",
          "FP32 và FP64 có fits riêng; đó không phải precision transfer. Historical width corpus khác current CUDA-Q endpoint.", ["sim", "artifacts/research_question_closure/v1/rq_framework.md"])
    claim("C10", "RQ4", "A", "CENTRAL", "SUPPORTED WITH QUALIFICATION",
          "Trên đúng 45 observations, Graph có lower point MAE nhưng paired interval chứa zero; warm full-B2 median khoảng 2,22× MLP.",
          "38 Ma–Li + 7 Qonductor, không QPack; không ngoại suy accuracy/cost ra full 4.515; không complete all-method frontier.", ["p4"], [p4["summaries"][0], p4["summaries"][1], p4["paired"], p4["uncertainty"]["all_selected"]])
    for precision, cid in (("fp32", "C11"), ("fp64", "C12")):
        block = p5["contexts"][precision]
        assert block["paired"]["observed_mae_graph_minus_mlp_seconds"] > 0
        assert block["uncertainty"]["mae_difference_graph_minus_mlp_seconds"][0] > 0
        assert block["uncertainty"]["marginal_median_latency_ratio"][0] > 1
        claim(cid, "RQ4", "B", "CENTRAL", "SUPPORTED WITH QUALIFICATION",
              f"Trong exact30 Dense {precision.upper()}, MLP có lower MAE và lower warm B2; conditional paired intervals hỗ trợ cả hai hướng.",
              "Dominance chỉ cho hai criteria của Graph–MLP trên selected30; full150 R² yếu; historical simulator timing cross-environment.",
              ["p5", "densefull"], block["summary"] + [block["paired"], block["uncertainty"]])
    claim("C13", "RQ4", "A/B", "CENTRAL", "STRONGLY SUPPORTED",
          "Saved full-B2 stage measurements cho thấy parsing/features/graph/IPC là phần đáng kể của warm online estimator cost.",
          "B1 và B2 khác occasions/instrumentation; không dùng tỷ số B2/B1 làm matched slowdown hoặc operational break-even; cold-start/classical frontier thiếu.", ["p4", "p5"], p4["shares"] + p5["shares"])
    claim("C14", "RQ5", "A/B", "CENTRAL", "STRONGLY SUPPORTED",
          "51 raw globals được support-refine: reconstructed original simulator 41, main local QPU 40; executed author real tensor vẫn chưa được xác minh.",
          "Local Graph/MLP là architecture/core adaptations; effective widths/fit moments không phải tác giả đã cung cấp tensors.", ["feature", "fidelity"])
    claim("C15", "RQ5/RQ3", "A/B", "SECONDARY", "STRONGLY SUPPORTED",
          "Compatibility recovery xác minh graph order và RCCX library decomposition làm khác consumed inputs; historical preprocessing phục hồi exact frozen arrays.",
          "Không causal topology ablation, không xác lập mọi environment tương đương; original Dense producing package versions vẫn thiếu metadata.", ["recovery", "artifacts/am1_preprocessing_recovery/v1/historical_environment_provenance.md"])
    claim("C16", "RQ5", "A", "CENTRAL", "SUPPORTED WITH QUALIFICATION",
          "QCEC 32 pairs hỗ trợ zero-input prepared-state/measured-output qualification cho 230 linked observations.",
          "32 DD state proofs/output matches; 8 full-unitary proofs, 24 counterexamples under assumed mapping; không exact historical logical recovery.", ["qcec", "docs/research_conclusions.md"])
    return paths, loaded, qpu, p4, p5, claims


def rq_records(paths):
    data = [
        ("RQ1", "A", "ANSWERED WITH QUALIFICATIONS", "Shared 4.515-row accuracy và competitive/analytical controls đủ bảo vệ pipeline comparison; point wins còn conditional uncertainty.", "qpu", "pairs"),
        ("RQ1", "B", "ANSWERED WITH QUALIFICATIONS", "Matched context-specific accuracy đủ; seven predictive contexts/native companions giữ clock và method applicability riêng.", "sim", "sim_pairs"),
        ("RQ2", "A", "PARTIALLY ANSWERED", "21/21 cases và severe negative family findings đã evaluated. Source/shot và isolated generation effects chưa tested; low-support size/backend cases descriptive.", "grid", "grid_report"),
        ("RQ2", "B", "PARTIALLY ANSWERED", "Family MPS và historical width findings có evidence; systematic current-engine/hardware/precision-transfer chưa measured.", "sim", "artifacts/research_question_closure/v1/rq_framework.md"),
        ("RQ3", "A", "ANSWERED WITH QUALIFICATIONS", "Tails, source-weighted rank changes, QNN/Random và collisions đã established. Causal feature attribution/prospective safety chưa established.", "qpu", "grid_report"),
        ("RQ3", "B", "ANSWERED WITH QUALIFICATIONS", "Quality violations, subset-selection sensitivity, metric dependence/TN bias và resource/support failures đã established trong từng context.", "mps", "sim"),
        ("RQ4", "A", "ANSWERED WITH QUALIFICATIONS", "ANSWERED WITHIN MEASURED SCOPE: matched warm B2 + accuracy cho Graph/MLP exact45; stronger classical/cold/scaling utility không thuộc bounded conclusion.", "p4", "artifacts/am1_phase4/v1/accuracy_cost_tradeoff.md"),
        ("RQ4", "B", "ANSWERED WITH QUALIFICATIONS", "ANSWERED WITHIN MEASURED SCOPE: exact30 per Dense precision cho paired accuracy/cost; MLP ưu thế có conditional CIs. Các engine/cold/complete Pareto frontier khác còn ngoài đo.", "p5", "densefull"),
        ("RQ5", "A", "ANSWERED WITH QUALIFICATIONS", "Fidelity/input/clock applicability được chốt, provenance ambiguities đã resolved; historical author tensor/mappings và original Scholten giữ limits.", "fidelity", "feature"),
        ("RQ5", "B", "ANSWERED WITH QUALIFICATIONS", "Context/API/quality/native-versus-adaptation rules và practical conditional selection đủ; không production-safe automatic selection hoặc universal engine portability.", "fidelity", "mps"),
    ]
    out = []
    for rq, track, status, why, *src in data:
        sources = [paths.get(s, s) for s in src]
        out.append(dict(rq=rq, track=track, status=status, blocking="NON-BLOCKING",
                        explanation=why, bounded_scope_closed="YES" if status != "PARTIALLY ANSWERED" else "EVALUATED TRANSFERS CLOSED; BROAD PORTABILITY PARTIAL",
                        evidence_paths_json=json.dumps(sources, ensure_ascii=False),
                        evidence_hashes_json=json.dumps({p: digest(pin(p)) for p in sources}, sort_keys=True)))
    return out


def validate_saved_sources():
    py = str(ROOT.parent / ".venv-qonductor/bin/python")
    commands = [
        [py, "-B", "artifacts/final_research/verify.py"],
        [py, "-B", "benchmark_v1/scripts/validate_benchmark_interpretation.py"],
        [py, "-B", "benchmark_v1/scripts/report_am1_phase4.py", "--stage", "validate"],
        [py, "-B", "benchmark_v1/scripts/validate_am1_phase5_dense_b2.py"],
    ]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    results = []
    for command in commands:
        pin(command[2])
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
        results.append(dict(command=" ".join(command), exit_code=result.returncode, status="PASS",
                            scope="existing deterministic saved-evidence validator; scientific closure assessed separately"))
    return results


def render():
    # Preserve the entire prior validation package and the current freeze/authority.
    for p in PRIOR.iterdir():
        if p.is_file():
            pin(p)
    pin(Path(__file__))
    for p in ["AGENTS.md", "docs/research_contract.md", "docs/research_conclusions.md",
              "artifacts/final_research/manifest.json", "artifacts/final_research/validation.json",
              "artifacts/release_review/scientific_freeze.json", "artifacts/research_question_closure/v1/provenance_manifest.json",
              "artifacts/real_qpu/qualified_logical_panel/panel.csv", "artifacts/real_qpu/qualified_logical_panel/outer_splits.csv",
              "artifacts/real_qpu/qualified_logical_panel/inner_splits.csv", "artifacts/real_qpu/qualified_logical_panel/summary/oof_predictions.csv",
              "benchmark_v1/protocol/qualified_logical_benchmark.json", "benchmark_v1/protocol/real_qpu_generalization.json",
              "benchmark_v1/registry/method_fidelity_registry_v2.json"]:
        pin(p)
    historical, resolved = provenance()
    paths, loaded, qpu, p4, p5, claims = scientific_records()
    statuses = rq_records(paths)
    assert len(statuses) == 10 and len({(r["rq"], r["track"]) for r in statuses}) == 10
    assert all(r["blocking"] == "NON-BLOCKING" for r in statuses)
    assert len(claims) == 16 and all(r["support"] in {"STRONGLY SUPPORTED", "SUPPORTED WITH QUALIFICATION"} for r in claims)

    files = {"pathless_hash_resolution.csv": csv_content(resolved), "historical_hash_mismatches.csv": csv_content(historical),
             "final_rq_status.csv": csv_content(statuses), "final_claim_matrix.csv": csv_content(claims)}
    files["README.md"] = """# Final research-question closure

Decision: **SCIENTIFICALLY SUFFICIENT WITH MATERIAL LIMITATIONS**. Không còn mandatory experiment cho central thesis claim trong scope đã đo.

Đọc [provenance resolution](provenance_resolution.md), [RQ status](final_rq_status.md), [central claims](central_claims.md), [contributions](thesis_contributions.md), [gaps](final_gap_register.md) và [completion decision](experiment_completion_decision.md). [Claim matrix](final_claim_matrix.md) nối mọi kết luận với source path/hash. Các CSV lưu một declaration, RQ hoặc claim trên mỗi dòng.

Gói này bổ sung erratum cho `final_exhaustive_validation/v1`, giữ nguyên các artifact lịch sử. Six README mismatches là lỗi namespace resolution của checker trước; all 182 symbolic declarations được truy nguyên qua mapping tường minh và xác minh identity. Không thay historical manifest/hash để tạo PASS.

Working RQ1–RQ5 theo yêu cầu closure khác numbering của Research Contract v1.1: historical RQ4 generalization nằm trong working RQ2; AM1 accuracy–cost là working RQ4. Contract, frozen data/protocols/results và các historical reports giữ nguyên.

Tái tạo: `python3 -B benchmark_v1/scripts/build_final_rq_closure_v2.py`; kiểm tra byte/identity: thêm `--check`. Chỉ đọc saved evidence và read-only Git metadata. Không gọi training/inference/checker/simulator/timing workload.
"""
    files["provenance_resolution.md"] = f"""# Provenance resolution

**Không còn unresolved provenance item trong 188 declarations được giao làm suy yếu primary scientific claim.** Cả 6 đúng README payload và 182 symbolic declarations đều được đối chiếu thành công. Phần author Ma–Li real tensor vẫn thiếu historical provenance; giới hạn đó đã được công bố và không được dùng để gọi local adaptation là faithful reproduction.

## Erratum: six README checks

Checker cũ `artifacts/final_exhaustive_validation/v1/build_final_exhaustive_validation.py:resolve_pin` chọn `ROOT / p` trước `manifest.parent / p`. Vì root `README.md` có thật, nó băm nhầm file cho sáu package-local payload/report entries. Producing/validating code dùng `OUT / name`, `P / name` hoặc `SUMMARY / name`; tất cả sáu README đúng namespace hiện vẫn khớp recorded hash. Không phát hiện historical target README đã evolve, mất snapshot hoặc scientific-data drift.

Category theo danh sách yêu cầu: `MANIFEST DESIGN ISSUE`, subtype **validator namespace-resolution false positive**. Convention tương đối của original manifest được xác định bằng producing code; không đề xuất sửa original manifest. Table giữ cả erroneous prior path/hash và correct path/hash, pin intent, content-preservation evidence.

Taxonomy được yêu cầu không có `VALIDATOR_RESOLUTION_ERROR`; category này mô tả vấn đề diễn giải namespace của checker, không kết luận original manifest bị sai thiết kế hoặc mất content.

{table(historical, ['mismatch_id','historical_manifest','current_file_path','current_matches_recorded','classification','scientific_evidence_affected'])}

## Symbolic declarations

182 declaration rows tương ứng 23 tên symbolic. 180 là SHA-256 bytes, hai `qcre_revision` là Git commit SHA-1 (40 ký tự), không phải SHA-256 file. Twelve Osaka/Quito neural cells lặp lại cùng 12 named code/data pins; repetition là receipt lineage, không independent evidence. Base qualified pins, source analytical receipts và generated manifests có mappings khác được ghi riêng.

Mapping được đọc từ consumer/producer tường minh, không suy ra bằng filename similarity hoặc hash search đơn lẻ. Sau khi context xác định path, bytes được băm lại; QCRE commit kiểm bằng Git HEAD và cùng-commit `estimate.py` blob, đồng thời blob SHA-256 khớp code-file pin. Supplementary path-qualified references từ inventory cũ được giữ trong provenance_chain khi có.

Classification counts: `{dict(Counter(r['classification'] for r in resolved))}`. Mỗi row đã giải có `candidate_count_after_context=1`, matching identity và recorded provenance chain. Không còn ambiguous/no-path/stale declaration. Chi tiết tất cả 182 rows ở [CSV](pathless_hash_resolution.csv).

## Scientific impact

Dataset/splits/targets/predictions/checkpoints/transforms/timing/metrics không được sửa để giải provenance. Named pins chủ yếu nhận diện contract, split/support inputs, original helper/runner code và analytical cached results. Việc kiểm tra chính xác chúng bảo vệ lineage đã dùng; không tạo thêm accuracy, generalization hoặc historical-instance proof. Six README issue chỉ ảnh hưởng kết luận của generic documentation checker trước. Các 748.588 saved metric/timing/alignment checks của gói cũ và current component-validator PASS giữ giá trị trong scoped evidence của chúng.
"""
    files["final_rq_status.md"] = "# Final RQ status\n\n" + table(statuses, ["rq", "track", "status", "blocking", "explanation"]) + """
RQ2 partial phản ánh broad source/hardware/engine transfer chưa evaluated. Các evaluated negative findings vẫn là answers có ích và central comparison không cần universal portability. RQ4A/B được đóng **ANSWERED WITHIN MEASURED SCOPE**, biểu diễn bằng `ANSWERED WITH QUALIFICATIONS` trong status vocabulary chung. Missing classical/cold/scheduler frontier hạn chế claim rộng hơn, không buộc chạy thêm để bảo vệ measured Graph–MLP trade-off.

Không average R², không pool 6.684 assignments thành independent observations và không đổi evaluation rules sau khi thấy test scores. Parent OOF/holdout reports, per-seed và protocol-defined median-of-three giữ nguyên. Conditional bootstrap không bao gồm training variability hoặc future-workload guarantee.
"""
    main_methods = ["mali_logical_graph", "mali_global_mlp", "qonductor_budgeted_selector", "qonductor_polynomial_regression", "outer_train_median", "outer_train_source_median"]
    primary_table = [dict(method=m, rows=qpu[m]["n"], mae_s=qpu[m]["mae_seconds"], r2=qpu[m]["r2_seconds"], source_balanced_mae_s=qpu[m]["source_balanced_mae"], equal_group_mae_s=qpu[m]["equal_group_mae"]) for m in main_methods]
    files["central_claims.md"] = """# Central claims and their limits

Central thesis objective: systematic comparison of runtime-estimation pipelines for archived real-QPU execution and recorded simulator configurations, using explicit input, clock, split, coverage and quality semantics.

The essential claims are: (1) qualified shared-row/context evaluation is reproducible and attributable; (2) conditional accuracy and failure patterns are established against simple/classical/analytical controls; (3) measured warm Graph–MLP cost includes preparation, not only cached neural inference; (4) applicability/fidelity boundaries prevent schedule, service, sample, state, contraction and analog costs being silently interchanged. A universal estimator, graph causal benefit or reproduction of every original paper is not required to establish these claims.

## Track A

""" + table(primary_table, ["method", "rows", "mae_s", "r2", "source_balanced_mae_s", "equal_group_mae_s"]) + f"""
Source: {links([paths['qpu'], paths['pairs'], paths['sources']])}. Graph–MLP point difference favors Graph but grouped CI crosses zero. Graph–selector paired delta is −0.052523 s with interval [−0.317858, +0.457908] s. Per-source/group views change rankings. QPack 3.945/4.515 rows from six structures dominates pooled frequency; 165 groups and qualified reconstructions limit unseen-design claims.

Original QCRE critical-path component and native Qiskit schedule remain different clocks from service. Hyb is an effective-cost adaptation. Scholten current scored variant is a generalized proxy (3.399 eligible); original effective-depth model has no verified eligible rows. Outer-train affine/log calibration is a learned bridge. It does not identify physical launch/inter-shot overhead. Partial eligible coverage must remain beside errors.

Ma–Li's 41 là supported width tái dựng cho original simulator corpus; parent real-QPU width là 40 từ 51 raw positions. Current Dense adaptation dùng seven globals, vì vậy không gán width41 cho mọi simulator branch. Historical author real tensor vẫn unverified; no faithful-paper claim follows merely from matching a displayed feature count.

## Track B

Within-context OOF compares 150 core hashes where supported: noisy Aer, Dense FP32/FP64, fixed MPS, joint runtime–quality MPS and CPU QCSim component. Native same-plan cuTensorNet and Pasqal analog companion answer separate timing/quality questions. Compiled Aer GNN improves versus source GNN, but superiority over XGBoost is unresolved. Dense RF has lower local MAE than neural branches; fixed-MPS Graph has useful conditional signal. Family conditioning lacks resolved runtime gain and frozen selectors fail the quality gate on 64/140 assessed hashes. Pasqal coefficients remain unpromoted, so its 18 cells support descriptive quality/time findings rather than validated predictive accuracy.

These are declared implementations and adaptations, with source-independent runtime labels retained. Quantum simulator endpoint/API/noise/precision/shots/quality/workspace/plan form the target context. Same QASM or same seconds unit alone is insufficient for cross-engine ranking.

## Secondary findings and limits

QNN/Random catastrophic overprediction, backend input collisions, subset-selected R², TN estimate bias and preprocessing compatibility are secondary research findings supporting robustness/applicability. They need no successful universal transfer result to be useful. QCEC zero-input checks qualify recipes; they do not recover historical logical instances, mapping or compilation history.

## Blocking gaps

None for the central claims stated above. There are gaps for stronger claims (all-paper-exact reproduction, full competitive deployment frontier, hardware/source portability, scheduler utility), which remain explicitly unclaimed. Publication rights/metadata/clean-release verification are independent delivery gates, not a reason to reopen training.
"""

    p4rows = [dict(context="QPU exact45", model=r["method"], n=r["n"], mae_s=r["mae_seconds"], rmse_s=r["rmse_seconds"], r2=r["r2"], median_B2_ms=r["median_request_median_B2_ms"]) for r in p4["summaries"] if r.get("source") == "all_selected" and r["method"] in {"graph", "mlp"}]
    p5rows = [dict(context=f"Dense {precision.upper()} exact30", model=r["method"], n=r["n"], mae_ms=float(r["mae_seconds"])*1000, rmse_ms=float(r["rmse_seconds"])*1000, r2=r["r2"], median_B2_ms=r["median_request_median_B2_ms"]) for precision,b in p5["contexts"].items() for r in b["summary"] if r["method"] in {"graph", "mlp"}]
    files["experiment_completion_decision.md"] = f"""# Experiment-completion decision

## {VERDICT}

Central benchmark comparison and measured Graph–MLP accuracy–cost contributions are scientifically defensible. Material limits concern input reconstruction, source/group support, procedure asymmetry, nominal hardware context, limited simulator resource/engine support and cost scope. They must be prominent in thesis claims. They do not make the measured comparison unanswerable.

RQ4 **Track A: ANSWERED WITHIN MEASURED SCOPE**. Source {links([paths['p4']])}.

{table(p4rows, ['context','model','n','mae_s','rmse_s','r2','median_B2_ms'])}

Graph−MLP observed MAE = {p4['paired']['observed_mae_graph_minus_mlp_seconds']:.9f} s; conditional 95% interval {p4['uncertainty']['all_selected']['mae_difference_graph_minus_mlp_seconds']} s includes zero. Ratio of marginal B2 medians = {p4['paired']['ratio_of_marginal_latency_medians']:.6f}; interval {p4['uncertainty']['all_selected']['marginal_median_latency_ratio']}. Median paired ratio = {p4['paired']['median_of_paired_latency_ratios']:.6f}, a different statistic. Exact45 includes 38 Ma–Li/7 Qonductor and no QPack. No full4.515 cost generalization follows.

RQ4 **Track B: ANSWERED WITHIN MEASURED SCOPE**. Sources {links([paths['p5'], paths['densefull']])}.

{table(p5rows, ['context','model','n','mae_ms','rmse_ms','r2','median_B2_ms'])}

MLP has lower MAE and lower B2 in both exact30 contexts with conditional paired intervals excluding equality. The conclusion is bounded to the two saved estimators, warm B2, selected circuits and recorded hardware/environment. Full150 neural R² remains about 0.13–0.14. No all-method Pareto frontier, operational break-even, scheduler-throughput gain, cross-hardware scaling or cold-start cost is established. Historical simulator execution timings are cross-environment and cannot provide matched break-even.

The previous closure v1 predates completed AM1 and described a material matched-overhead gap. This v2 supersedes that gap/status only for executed Phase4/Phase5 scope; original decisions and historical authority remain untouched. Missing strong classical fitted-state overhead limits competitive deployment selection but does not erase the bounded Graph–MLP comparison. RQ2 portability questions remain partial/non-blocking.

Mandatory new experiments: **NONE**. Existing frozen scope can stop. Future extensions require a separate research amendment.

**STOP EXPERIMENTS AND MOVE TO THESIS SYNTHESIS**
"""
    contributions = [
        dict(id="C1", support="STRONGLY SUPPORTED", contribution="Semantics-aware benchmark methodology", defensible="Một benchmark ghi rõ population/input tiers, train-only fits, clock/coverage/quality và shared-row comparisons cho QPU/simulator.", boundary="Source-retained QPU service boundaries và simulator contexts vẫn heterogeneous; không unified physical clock.", evidence="C01,C02,C03,C04,C05,C16"),
        dict(id="C2", support="SUPPORTED WITH QUALIFICATION", contribution="Empirical comparative accuracy/robustness", defensible="Accuracy, eligible-control benefit và negative generalization/failure findings của declared pipelines được xác minh trên frozen cohorts/contexts.", boundary="Không universal ranking, statistically proven Graph win hoặc original-paper reproduction toàn bộ.", evidence="C01,C03,C04,C07,C08,C09"),
        dict(id="C3", support="STRONGLY SUPPORTED", contribution="Warm estimator pipeline cost", defensible="Full-B2 stage accounting đo substantial parsing/preparation/encoding/IPC costs bị bỏ qua khi chỉ báo cached inference.", boundary="Warm standalone path, selected IDs; B1/B2 historical occasions không matched scaling, không cold production deployment.", evidence="C10,C11,C12,C13"),
        dict(id="C4", support="SUPPORTED WITH QUALIFICATION", contribution="Model complexity has conditional value", defensible="Graph không luôn có accuracy benefit bù cost: QPU exact45 accuracy gain unresolved với 2.22× B2; MLP lower MAE và B2 trong Dense exact30 FP32/FP64.", boundary="Không claim graph topology vô ích hoặc MLP thống trị mọi workload; conditional subset CIs.", evidence="C01,C10,C11,C12"),
        dict(id="C5", support="STRONGLY SUPPORTED", contribution="Feature and preprocessing fidelity lessons", defensible="Support refinement, node ordering và dependency/library decomposition ảnh hưởng exact frozen serving input; compatibility được kiểm bằng immutable arrays.", boundary="Author real tensor và original Dense producing environment vẫn chưa hoàn toàn documented; không causal feature attribution.", evidence="C14,C15,C16"),
    ]
    files["thesis_contributions.md"] = "# Defensible thesis contributions\n\n" + table(contributions, ["id","support","contribution","defensible","boundary","evidence"]) + "\nCác formulations ở cột defensible thay cho stronger claims. Central contribution là benchmark + interpretation có traceability, không một predictor mới.\n"
    gaps = [
        dict(id="G1", rq="RQ2", track="A", gap="Unseen source/shot và isolated generation/topology transfer", impact="Không claim portability qua nguồn/throughput/hardware generation độc lập.", blocking="NON-BLOCKING", future_value="Tách causal context effects khỏi workload/source confounding.", execution_needed="New data/controlled fits required for broader claim"),
        dict(id="G2", rq="RQ2/RQ4", track="B", gap="Systematic current-engine width/hardware/precision portability", impact="Core small-width/context scores không quy thành universal scaling.", blocking="NON-BLOCKING", future_value="Kiểm chuyển hardware/engine trên labels cùng boundary.", execution_needed="New simulator labels and/or model evaluation required"),
        dict(id="G3", rq="RQ4", track="A/B", gap="Strong classical fitted-state overhead/complete frontier", impact="Không chọn deployment-optimal estimator trên toàn method set.", blocking="NON-BLOCKING", future_value="So competitive accuracy/cost với classical methods.", execution_needed="Recover exact fitted states; any new fitting/measurement needs amendment"),
        dict(id="G4", rq="RQ4", track="A/B", gap="Cold start, deployment, operational break-even/scheduler utility", impact="Warm B2 không include full lifecycle/system gains.", blocking="NON-BLOCKING", future_value="Định lượng lifecycle amortization và scheduling loss.", execution_needed="New deployment/timing or scheduler evaluation required"),
        dict(id="G5", rq="RQ4", track="B", gap="MPS/Maestro/cuTensorNet full-B2 under equivalent selection task", impact="Không full multi-engine cost frontier; plan/build latency không isolated internal EST inference.", blocking="NON-BLOCKING", future_value="Đánh giá plan/quality/task preparation cost trước decision.", execution_needed="New controlled timing if pursued"),
        dict(id="G6", rq="RQ5", track="A", gap="Author Ma–Li real tensor, original logical/layout/history and Scholten eligibility", impact="Chỉ adaptations/native components/proxy claims; không faithful original paper inference.", blocking="NON-BLOCKING", future_value="Faithful original-method provenance khi upstream cung cấp missing assets.", execution_needed="Upstream assets first; no automatic retraining"),
        dict(id="G7", rq="RQ3/RQ5", track="B", gap="Prospective quality-safe MPS selection and Pasqal predictor identification", impact="Observed safety failure là finding; analog companion không claimed predictor.", blocking="NON-BLOCKING", future_value="Thiết kế safe abstention/quality and identifiable analog grid.", execution_needed="New protocol/data/model execution would be needed"),
        dict(id="G8", rq="RQ3", track="A/B", gap="Causal feature/topology attribution and prospective intervals", impact="Descriptive support shifts/tails không causal mechanism/future safety guarantee.", blocking="NON-BLOCKING", future_value="Thử causal intervention/coverage nếu new thesis scope yêu cầu.", execution_needed="New intervention/prospective data likely required"),
        dict(id="G9", rq="Delivery", track="A/B", gap="Public rights/metadata, exact export and clean-release reproduction", impact="Scientific sufficiency không public-release approval; reduced QPU materials không self-contained source-refit.", blocking="NON-BLOCKING FOR SCIENCE; RELEASE GATE", future_value="Transparent owner-qualified publication and scoped reproduction.", execution_needed="Metadata/export validation; no scientific experiment required"),
    ]
    files["final_gap_register.md"] = "# Final gap register\n\n" + table(gaps, ["id","rq","track","gap","impact","blocking","future_value","execution_needed"]) + "\nKhông gap nào ở đây phủ định central claims đang dùng. Nếu một stronger claim được thêm sau này, chính scope đó cần amendment; không resume old queue theo limitation này.\n"
    files["final_claim_matrix.md"] = "# Evidence-to-claim matrix\n\n" + table(claims, ["claim_id","rq","track","role","support","claim","scope_and_limitations"]) + "\nPaths, SHA-256 và exact saved metric records nằm trong [CSV](final_claim_matrix.csv). Giá trị chỉ được copy từ saved evidence, không fit/recompute confidence interval hoặc chọn seed theo test performance.\n"

    component_validators = validate_saved_sources()
    unchanged = all(digest(p) == h for p, h in PINS.items())
    assert unchanged
    validation = dict(status="PASS", task="reporting and scientific closure only", readme_checks=6,
                      readme_actual_namespace_matches=6, pathless_declarations=182,
                      sha256_declarations=180, git_revision_declarations=2, unresolved_declarations=0,
                      primary_claim_provenance_risks=0, rq_track_rows=10, claims=16,
                      blocking_scientific_gaps=0, required_new_experiments=0,
                      source_files_checked=len(PINS), source_hashes_unchanged=unchanged,
                      deterministic_saved_evidence_validators=component_validators,
                      original_exhaustive_package_preserved=True, scientific_freeze_preserved=True,
                      new_training=False, new_inference=False, new_workloads=False, new_timing=False,
                      historical_manifests_modified=False, commit_push=False)
    files["validation_report.md"] = "# Closure validation\n\n```json\n" + json.dumps(validation, indent=2, ensure_ascii=False) + "\n```\n\n" + """All ten RQ/track rows have one allowed status and explicit blocking decision. Claims contain verified local source paths/SHA-256 and saved metric records. P4 exact45, P5 exact30 and Dense full150 remain separate; source-balanced, pooled and equal-group errors retain distinct fields. Seconds-to-milliseconds conversion is used only for the displayed Dense table. Conditional CI signs are checked for Dense pair statements; QPU paired interval includes zero. No source statistic is invented or reselected.

Working RQ mapping is stated without rewriting Research Contract v1.1. Adaptation/native/proxy labels, QNN/Random failures, collisions, QCEC zero-input boundaries, subset-selection and cross-environment timing limits remain visible. Earlier v1 status is preserved and superseded only for current AM1 closure. All source hashes captured by this builder are checked again before writing.

`--check` verifies every derived payload byte and rechecks source identities, mappings and source evidence. `git diff --check` is executed after writing/checking; its output is returned by the command. Existing exhaustive test/validator results are referenced from the immutable prior package; no new full-suite PASS is claimed here.

""" + "## Captured source hashes\n\n" + table([dict(path=rel(Path(p)), sha256=h) for p,h in sorted(PINS.items())], ["path","sha256"])
    assert set(files) == {"README.md","provenance_resolution.md","pathless_hash_resolution.csv","historical_hash_mismatches.csv","final_rq_status.md","final_rq_status.csv","central_claims.md","thesis_contributions.md","final_gap_register.md","experiment_completion_decision.md","final_claim_matrix.md","final_claim_matrix.csv","validation_report.md"}
    return files, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files, validation = render()
    if args.check:
        for name, content in files.items():
            assert (OUT / name).read_bytes() == content.encode(), f"derived payload drift: {name}"
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (OUT / name).write_text(content, encoding="utf-8")
    assert all(digest(p) == h for p,h in PINS.items()), "source modified during reporting"
    whitespace = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert whitespace.returncode == 0, whitespace.stdout + whitespace.stderr
    print(json.dumps({**validation, "requested_outputs": len(files), "byte_replay": args.check, "git_diff_check": "PASS", "scientific_verdict": VERDICT}, indent=2))


if __name__ == "__main__":
    main()
